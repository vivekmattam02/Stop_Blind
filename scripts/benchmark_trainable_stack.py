#!/usr/bin/env python3
"""Benchmark the full-size trainable SpatialVLM visual stack.

This probe uses the paper-scale projector, GATr, and SVA dimensions with
synthetic inputs. It measures synchronized forward/backward/optimizer steps,
so its timings are suitable for comparing micro-batch candidates on one host.
It deliberately excludes the frozen vision encoders and Qwen backbone; the
winning candidate must still pass a real-weight end-to-end memory gate.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` without hard-coding an accelerator requirement."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    """Synchronize accelerator work before reading a wall-clock timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def linux_memory_available_bytes() -> int | None:
    """Read Linux MemAvailable, which is useful on unified-memory systems."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


class TrainableVisualStack(nn.Module):
    """Paper-scale trainable projectors, GATr, SVA, and norm matching."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        from spatialvlm.encoders.projector import MLPProjector
        from spatialvlm.fusion.norm_matching import RMSNormMatching
        from spatialvlm.fusion.sva import SpatialVisionAggregator
        from spatialvlm.geometry.gatr_wrapper import GATrWrapper

        self.siglip_projector = MLPProjector(in_dim=3456, out_dim=4096)
        self.dino_projector = MLPProjector(in_dim=3072, out_dim=4096)
        self.gatr = GATrWrapper(
            num_blocks=8,
            projector_out_dim=4096,
            checkpoint_blocks=True,
            device=device,
        )
        self.sva = SpatialVisionAggregator(
            hidden_dim=4096,
            num_queries=1369,
            num_layers=2,
            num_heads=32,
            use_typed_attention_bias=True,
        )
        self.norm_matching = RMSNormMatching()
        self.to(device=device, dtype=dtype)

    def forward(
        self,
        siglip_features: torch.Tensor,
        dino_features: torch.Tensor,
        positions_3d: torch.Tensor,
    ) -> torch.Tensor:
        siglip_tokens = self.siglip_projector(siglip_features)
        dino_tokens = self.dino_projector(dino_features)
        gatr_tokens = self.gatr(positions_3d)
        fused = self.sva(
            siglip_tokens=siglip_tokens,
            dinov2_tokens=dino_tokens,
            gatr_tokens=gatr_tokens,
        )
        return self.norm_matching(fused)


@dataclass(frozen=True)
class BenchmarkResult:
    status: str
    batch_size: int
    warmup_steps: int
    measured_steps: int
    dtype: str
    device: str
    trainable_parameters: int
    step_times_seconds: list[float]
    mean_step_seconds: float
    samples_per_second: float
    final_loss: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    system_available_before_bytes: int | None
    system_available_after_bytes: int | None


def make_inputs(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create reusable paper-shape synthetic inputs."""
    siglip = torch.randn(batch_size, 576, 3456, device=device, dtype=dtype)
    dino = torch.randn(batch_size, 1369, 3072, device=device, dtype=dtype)
    positions = torch.randn(batch_size, 1369, 3, device=device, dtype=torch.float32)
    return siglip, dino, positions


def optimization_step(
    model: TrainableVisualStack,
    optimizer: torch.optim.Optimizer,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Run one normalized synthetic optimization step."""
    optimizer.zero_grad(set_to_none=True)
    fused = model(*inputs)
    # A subset loss avoids the near-constant full-vector second moment created
    # by the final normalization layer while keeping the backward graph intact.
    loss = fused[..., :32].float().square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    device = resolve_device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision(args.float32_matmul_precision)

    model = TrainableVisualStack(device=device, dtype=dtype)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01)
    inputs = make_inputs(args.batch_size, device, dtype)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for _ in range(args.warmup_steps):
        optimization_step(model, optimizer, inputs)
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    available_before = linux_memory_available_bytes()

    step_times: list[float] = []
    final_loss = float("nan")
    for _ in range(args.measure_steps):
        synchronize(device)
        started = time.perf_counter()
        loss = optimization_step(model, optimizer, inputs)
        synchronize(device)
        step_times.append(time.perf_counter() - started)
        final_loss = float(loss.cpu().item())

    available_after = linux_memory_available_bytes()
    total_seconds = sum(step_times)
    mean_step_seconds = total_seconds / len(step_times)
    samples_per_second = args.batch_size * len(step_times) / total_seconds

    peak_allocated: int | None = None
    peak_reserved: int | None = None
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

    return BenchmarkResult(
        status="passed",
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measure_steps,
        dtype=args.dtype,
        device=str(device),
        trainable_parameters=trainable_parameters,
        step_times_seconds=step_times,
        mean_step_seconds=mean_step_seconds,
        samples_per_second=samples_per_second,
        final_loss=final_loss,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        system_available_before_bytes=available_before,
        system_available_after_bytes=available_after,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
        help="Keep 'highest' for the initial A100-scope comparison.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.warmup_steps < 1:
        parser.error("--warmup-steps must be positive")
    if args.measure_steps < 1:
        parser.error("--measure-steps must be positive")
    return args


def render_payload(result: BenchmarkResult, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "float32_matmul_precision": args.float32_matmul_precision,
        "result": asdict(result),
    }
    if torch.cuda.is_available():
        device = resolve_device(args.device)
        payload["accelerator"] = {
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    return payload


def main() -> int:
    args = parse_args()
    result = run_benchmark(args)
    payload = render_payload(result, args)
    rendered = json.dumps(payload, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
