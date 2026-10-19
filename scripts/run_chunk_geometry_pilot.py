#!/usr/bin/env python3
"""NATIVE-CHUNK-GEOMETRY — first trained run of the geometry branch. Exploratory, single seed.

Protocol: docs/GEOMETRY_ARM_PROTOCOL.md. One variable against the completed no-geometry baseline
`native_chunk_scale_134k_v1`: the policy additionally receives, per H4 frame, 64 geometry tokens
(backprojected metric depth -> GATr v2 -> projector) injected at `target_start` with RoPE3D-v2
phases and stateless norm matching. Backbone, revision, LoRA contract, objective, cache, splits,
sampler seed and update budget are held at their frozen values.

Every calibration point scores the frozen selection surface twice: geometry ON and geometry
MASKED (zero embeddings, origin coordinates) at the same weights, so "the model ignores the
branch" is a measurable outcome rather than a hidden one.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor

from spatialvlm.backbone.rope_patch_v2 import (
    ROPE_PATCH_V2_VERSION,
    apply_rope_patch_v2,
    stash_geometry,
)
from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.data.history import HISTORY_CONTRACTS, HistoryMode, build_history_window
from spatialvlm.data.native_chunk_dataset import (
    SELECTION_SPLIT,
    TRAIN_SPLIT,
    chunk_sample_summary,
    load_chunk_samples,
)
from spatialvlm.data.native_qwen_chunk import NativeQwenChunkConfig, build_native_qwen_chunk_batch
from spatialvlm.data.native_qwen_chunk_geometry import (
    GEOMETRY_SPLICE_VERSION,
    filler_token_id,
    splice_geometry_block,
)
from spatialvlm.eval.action_chunk_decision import build_action_chunk_codebook
from spatialvlm.geometry.geometry_branch import GEOMETRY_BRANCH_VERSION, GeometryBranch
from spatialvlm.training.native_c0_gate import (
    NATIVE_C0_MODEL_ID,
    NATIVE_C0_MODEL_REVISION,
    gradient_report,
    validate_lora_only_trainables,
)
from spatialvlm.training.native_chunk_loss import CHUNK_LOSS_METHOD, restricted_chunk_loss

_STOP_REQUESTED = False


def _request_stop(signum: int, frame: Any) -> None:  # noqa: ARG001
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_object(sample: Any) -> dict[str, Any]:
    if sample.object_path is None:
        raise RuntimeError(f"Sample {sample.cache_key} carries no object path; use lazy loading.")
    return torch.load(sample.object_path, map_location="cpu", weights_only=False)


def _build_inputs(
    processor: Any,
    sample: Any,
    contract: Any,
    config: NativeQwenChunkConfig,
    codebook: Any,
    filler_id: int,
    n_geo_tokens: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], tuple[int, ...], torch.Tensor, torch.Tensor]:
    """Render, splice, and return (inputs, supervised_positions, spatial_mask, depth_frames)."""
    obj = _load_object(sample)  # one read serves both RGB and depth
    rgb = obj["rgb_frames"]
    window = build_history_window(
        [rgb[i].numpy() for i in range(rgb.shape[0])],
        [R2RCEAction(int(a)) for a in sample.history_action_ids],
        contract=contract,
    )
    batch = build_native_qwen_chunk_batch(
        processor,
        window,
        instruction=sample.instruction,
        target_actions=sample.target_actions,
        config=config,
        codebook=codebook,
    )
    inputs, positions, mask, _ = splice_geometry_block(
        batch, n_tokens=n_geo_tokens, filler_id=filler_id
    )
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "labels"}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    return inputs, positions, mask.to(device), obj["depth_frames_m"]


def _geometry_grad_report(branch: GeometryBranch) -> dict[str, Any]:
    total = l2sq = 0.0
    finite = True
    nonzero_params = param_count = 0
    for p in branch.parameters():
        param_count += 1
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not bool(torch.isfinite(g).all()):
            finite = False
        norm = float(torch.linalg.vector_norm(g.float()))
        l2sq += norm * norm
        total += float(g.abs().sum())
        if norm > 0:
            nonzero_params += 1
    return {
        "all_gradients_finite": finite,
        "gradient_l2_norm": math.sqrt(l2sq),
        "params_with_nonzero_grad": nonzero_params,
        "param_tensors": param_count,
    }


@torch.no_grad()
def _score_surface(
    model: Any,
    branch: GeometryBranch,
    rotary: Any,
    embed: Any,
    processor: Any,
    samples: list[Any],
    contract: Any,
    config: NativeQwenChunkConfig,
    codebook: Any,
    filler_id: int,
    device: torch.device,
    geometry_on: bool,
) -> dict[str, Any]:
    losses: list[float] = []
    predicted: list[list[int]] = []
    targets: list[list[int]] = []
    ids = [codebook.action_token_id(a) for a in R2RCEAction]
    for sample in samples:
        inputs, positions, mask, depth = _build_inputs(
            processor, sample, contract, config, codebook, filler_id, branch.n_tokens, device
        )
        if geometry_on:
            coords, validity = branch.points_from_depth(depth)
            coords, validity = coords.to(device), validity.to(device)
            embeds = branch(coords, validity)
        else:  # masked: zero content, origin phases, same sequence layout
            coords = torch.zeros(1, branch.n_tokens, 3, device=device)
            embeds = torch.zeros(1, branch.n_tokens, branch.hidden_size, device=device)
        stash_geometry(
            rotary, embed, coords_m=coords, spatial_mask=mask, geometry_embeds=embeds
        )
        out = model(**inputs)
        loss, _ = restricted_chunk_loss(
            out.logits,
            action_token_positions=positions,
            codebook=codebook,
            target_actions=sample.target_actions,
        )
        losses.append(float(loss))
        predicted.append(
            [int(torch.stack([out.logits[0, p - 1, i] for i in ids]).argmax()) for p in positions]
        )
        targets.append([int(a) for a in sample.target_actions])
        del out

    flat_pred = [p for row in predicted for p in row]
    flat_tgt = [t for row in targets for t in row]
    per_action: dict[str, dict[str, int]] = {}
    for action in R2RCEAction:
        per_action[action.name] = {
            "predicted_count": sum(1 for p in flat_pred if p == int(action)),
            "recall_numerator": sum(
                1 for p, t in zip(flat_pred, flat_tgt) if t == int(action) and p == t
            ),
            "recall_denominator": sum(1 for t in flat_tgt if t == int(action)),
        }
    return {
        "selection_chunk_ce_mean": sum(losses) / len(losses),
        "position_accuracy": sum(1 for p, t in zip(flat_pred, flat_tgt) if p == t)
        / len(flat_tgt),
        "constant_predicted_chunks": sum(1 for row in predicted if len(set(row)) == 1),
        "exact_chunk_match": sum(1 for p, t in zip(predicted, targets) if p == t),
        "per_action": per_action,
        "scored_decisions": len(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--cache-object-root", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--sampler-seed", type=int, default=20260825)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--geometry-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--max-updates", type=int, default=3000)
    parser.add_argument("--calibrate-every", type=int, default=150)
    parser.add_argument("--calibration-decisions", type=int, default=256)
    parser.add_argument("--patch-size-px", type=int, default=32)
    parser.add_argument("--gatr-blocks", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=16.0)
    parser.add_argument("--limit-train", type=int, default=None, help="smoke tests only")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + args.max_hours * 3600.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed_everything(args.seed)

    processor = AutoProcessor.from_pretrained(NATIVE_C0_MODEL_ID, revision=NATIVE_C0_MODEL_REVISION)
    codebook = build_action_chunk_codebook(processor.tokenizer, chunk_size=args.chunk_size)
    config = NativeQwenChunkConfig(chunk_size=args.chunk_size)
    contract = HISTORY_CONTRACTS[HistoryMode.H4_SHORT_WINDOW]
    filler_id = filler_token_id(processor.tokenizer)

    dataset = R2RCEDataset.from_release(args.episodes, args.ground_truth, split="train")
    common = dict(
        cache_index_path=args.cache_index,
        object_root=args.cache_object_root,
        dataset=dataset,
        sampler_seed=args.sampler_seed,
        chunk_size=args.chunk_size,
    )
    train_samples = load_chunk_samples(**common, split=TRAIN_SPLIT, lazy_frames=True)
    if args.limit_train:
        train_samples = train_samples[: args.limit_train]
    selection_samples = load_chunk_samples(
        **common, split=SELECTION_SPLIT, limit=args.calibration_decisions, lazy_frames=True
    )

    from spatialvlm.backbone.qwen3_vl import Qwen3VLBackbone

    backbone = Qwen3VLBackbone(
        model_id=NATIVE_C0_MODEL_ID,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        enable_lora=True,
        freeze_base_model=True,
        apply_peft_2880_workaround=False,
        apply_spatial_rope=False,  # v1 patch stays off; v2 is applied below
        device=device,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        revision=NATIVE_C0_MODEL_REVISION,
    )
    model = backbone.model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = validate_lora_only_trainables(backbone)
    rotary, embed = apply_rope_patch_v2(model)  # after the require-grads hook: order matters

    # Probe one sample for the depth geometry, then build the branch to match it.
    probe_depth = _load_object(train_samples[0])["depth_frames_m"]
    branch = GeometryBranch(
        hidden_size=backbone.hidden_size,  # introspected, never hardcoded
        n_frames=int(probe_depth.shape[0]),
        patch_size_px=args.patch_size_px,
        depth_hw=(int(probe_depth.shape[1]), int(probe_depth.shape[2])),
        gatr_blocks=args.gatr_blocks,
        device=device,
    )
    branch = branch.float()
    geometry_params = list(branch.parameters())
    for p in geometry_params:
        if p.dtype != torch.float32:
            raise RuntimeError("Geometry branch must train in fp32 (writability question 4).")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": args.learning_rate,
            },
            {"params": geometry_params, "lr": args.geometry_lr},
        ],
        weight_decay=args.weight_decay,
    )

    manifest = {
        "run_id": out.name,
        "status_boundary": "exploratory_geometry_arm_not_a_paper_artifact",
        "protocol": "docs/GEOMETRY_ARM_PROTOCOL.md",
        "objective": CHUNK_LOSS_METHOD,
        "model_id": NATIVE_C0_MODEL_ID,
        "model_revision": NATIVE_C0_MODEL_REVISION,
        "trainable_contract": trainable,
        "geometry": {
            "branch_version": GEOMETRY_BRANCH_VERSION,
            "rope_patch_version": ROPE_PATCH_V2_VERSION,
            "splice_version": GEOMETRY_SPLICE_VERSION,
            "n_geometry_tokens": branch.n_tokens,
            "tokens_per_frame": branch.tokens_per_frame,
            "patch_size_px": args.patch_size_px,
            "gatr_blocks": args.gatr_blocks,
            "geometry_param_count": sum(p.numel() for p in geometry_params),
            "geometry_lr": args.geometry_lr,
            "filler_token_id": filler_id,
            "rope_v2_kernel_diagnostics": rotary._v2_encoder.kernel_diagnostics(),
        },
        "hyperparameters": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
        "train_summary": chunk_sample_summary(train_samples),
        "selection_summary": chunk_sample_summary(selection_samples),
        "sealed_splits_never_opened": ["primary_evaluation", "balanced_diagnostic", "val_unseen"],
        "started_unix": started,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ckpt_path, best_path = out / "state.pt", out / "best.pt"
    best_ce = float("inf")
    start_update = 0
    trace: list[dict[str, Any]] = []
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model, state["adapter"])
        branch.load_state_dict(state["geometry"])
        optimizer.load_state_dict(state["optimizer"])
        start_update = int(state["update"])
        trace = state.get("trace", [])
        prior = [e["calibration"]["selection_chunk_ce_mean"] for e in trace if "calibration" in e]
        best_ce = min(prior) if prior else float("inf")
        print(f"[resume] continuing from update {start_update}", flush=True)

    order = list(range(len(train_samples)))
    rng = random.Random(args.seed)
    model.train()
    branch.train()
    cursor = start_update * args.accumulation
    status = "completed"

    def calibrate() -> dict[str, Any]:
        model.eval()
        branch.eval()
        on = _score_surface(
            model, branch, rotary, embed, processor, selection_samples, contract, config,
            codebook, filler_id, device, geometry_on=True,
        )
        masked = _score_surface(
            model, branch, rotary, embed, processor, selection_samples, contract, config,
            codebook, filler_id, device, geometry_on=False,
        )
        model.train()
        branch.train()
        return {**on, "geometry_masked": masked}

    for update in range(start_update + 1, args.max_updates + 1):
        if _STOP_REQUESTED or time.time() > deadline:
            status = "timed_out_partial" if time.time() > deadline else "stopped_on_signal"
            break
        optimizer.zero_grad(set_to_none=True)
        step_losses: list[float] = []
        for _ in range(args.accumulation):
            if cursor % len(order) == 0:
                rng.shuffle(order)
            sample = train_samples[order[cursor % len(order)]]
            cursor += 1
            inputs, positions, mask, depth = _build_inputs(
                processor, sample, contract, config, codebook, filler_id, branch.n_tokens, device
            )
            coords, validity = branch.points_from_depth(depth)
            coords, validity = coords.to(device), validity.to(device)
            embeds = branch(coords, validity)
            stash_geometry(
                rotary, embed, coords_m=coords, spatial_mask=mask, geometry_embeds=embeds
            )
            outputs = model(**inputs)
            loss, _ = restricted_chunk_loss(
                outputs.logits,
                action_token_positions=positions,
                codebook=codebook,
                target_actions=sample.target_actions,
            )
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite chunk loss at update {update}.")
            step_losses.append(value)
            (loss / args.accumulation).backward()
            del outputs, loss, inputs, embeds

        grads = gradient_report(backbone)
        geo_grads = _geometry_grad_report(branch)
        if not grads["all_gradients_finite"] or not geo_grads["all_gradients_finite"]:
            raise RuntimeError(f"Non-finite gradients at update {update}.")
        if geo_grads["params_with_nonzero_grad"] == 0:
            raise RuntimeError(
                f"Geometry branch received zero gradient at update {update}; the injection is "
                "disconnected and training it would be theatre. Refusing to continue."
            )
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] + geometry_params, 1.0
        )
        optimizer.step()

        entry = {
            "update": update,
            "train_chunk_ce_mean": sum(step_losses) / len(step_losses),
            "gradient_l2_norm": grads["gradient_l2_norm"],
            "geometry_gradient_l2_norm": geo_grads["gradient_l2_norm"],
            "geometry_params_with_grad": geo_grads["params_with_nonzero_grad"],
            "elapsed_seconds": time.time() - started,
            "epoch_fraction": cursor / len(order),
        }
        if update % args.calibrate_every == 0 or update == args.max_updates:
            entry["calibration"] = calibrate()
            from peft import get_peft_model_state_dict

            adapter = get_peft_model_state_dict(model)
            geometry_state = branch.state_dict()
            torch.save(
                {
                    "update": update,
                    "adapter": adapter,
                    "geometry": geometry_state,
                    "optimizer": optimizer.state_dict(),
                    "trace": [*trace, entry],
                },
                ckpt_path,
            )
            torch.save(
                {
                    "update": update,
                    "adapter": adapter,
                    "geometry": geometry_state,
                    "calibration": entry["calibration"],
                },
                out / f"adapter_u{update:05d}.pt",
            )
            current_ce = entry["calibration"]["selection_chunk_ce_mean"]
            if current_ce < best_ce:
                best_ce = current_ce
                torch.save(
                    {
                        "update": update,
                        "adapter": adapter,
                        "geometry": geometry_state,
                        "selection_chunk_ce_mean": current_ce,
                        "calibration": entry["calibration"],
                    },
                    best_path,
                )
                print(f"[best] selection CE {current_ce:.4f} at update {update}", flush=True)
        trace.append(entry)
        with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        cal = entry.get("calibration")
        msg = (
            f"[{update:5d}] train_ce={entry['train_chunk_ce_mean']:.4f} "
            f"geo_gl2={entry['geometry_gradient_l2_norm']:.2e} "
            f"ep={entry['epoch_fraction']:.2f} t={entry['elapsed_seconds'] / 3600:.2f}h"
        )
        if cal:
            msg += (
                f" | ON ce={cal['selection_chunk_ce_mean']:.4f} "
                f"acc={cal['position_accuracy']:.3f} "
                f"STOPpred={cal['per_action']['STOP']['predicted_count']} "
                f"| MASKED ce={cal['geometry_masked']['selection_chunk_ce_mean']:.4f}"
            )
        print(msg, flush=True)

    result = {
        "run_id": out.name,
        "status": status,
        "completed_updates": trace[-1]["update"] if trace else start_update,
        "elapsed_seconds": time.time() - started,
        "final": trace[-1] if trace else None,
        "calibration_points": [e for e in trace if "calibration" in e],
        "best_selection_chunk_ce_mean": best_ce if best_ce < float("inf") else None,
        "best_checkpoint_path": str(best_path) if best_path.exists() else None,
        "retained_adapter_checkpoints": sorted(p.name for p in out.glob("adapter_u*.pt")),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[done] status={status} updates={result['completed_updates']}", flush=True)


if __name__ == "__main__":
    main()
