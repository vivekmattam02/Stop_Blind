#!/usr/bin/env python3
"""Probe PyTorch and key SpatialVLM dependencies inside a DGX Spark container."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def module_version(name: str) -> dict[str, Any]:
    """Import a module and report its version without failing the whole probe."""
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - depends on container contents
        return {"available": False, "version": None, "error": repr(exc)}
    return {
        "available": True,
        "version": getattr(module, "__version__", "unknown"),
        "error": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. The report is always printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    payload: dict[str, Any] = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": {
            "version": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        },
        "modules": {
            name: module_version(name)
            for name in ("xformers", "transformers", "peft", "trl", "accelerate", "numpy")
        },
    }

    matrix_test: dict[str, Any] = {"passed": False}
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        payload["torch"].update(
            {
                "device_name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(device)),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            payload["torch"]["cuda_memory"] = {
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
            }
        except Exception as exc:  # pragma: no cover - unified-memory platform behavior
            payload["torch"]["cuda_memory_error"] = repr(exc)

        try:
            torch.manual_seed(0)
            left = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
            right = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
            result = left @ right
            torch.cuda.synchronize(device)
            finite = bool(torch.isfinite(result).all().item())
            matrix_test = {
                "passed": finite,
                "dtype": str(result.dtype),
                "shape": list(result.shape),
                "mean": float(result.float().mean().item()),
            }
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            matrix_test = {"passed": False, "error": repr(exc)}

    payload["bf16_matrix_test"] = matrix_test
    payload["ready_for_spatialvlm_bootstrap"] = bool(
        payload["torch"]["cuda_available"]
        and payload["torch"].get("bf16_supported")
        and matrix_test["passed"]
    )

    rendered = json.dumps(payload, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    if not payload["ready_for_spatialvlm_bootstrap"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
