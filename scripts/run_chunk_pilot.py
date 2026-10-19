#!/usr/bin/env python3
"""NATIVE-CHUNK-PILOT-v1 — chunk-objective training on the frozen C0 observation cache.

Exploratory capability pilot. **Not a paper artifact and not a frozen gate.** It changes exactly one
thing against `native_c0_development_v1`: the supervision objective becomes a k-action chunk instead
of a single action token. Backbone, revision, LoRA rank and target modules, optimizer, learning
rate, weight decay, accumulation, history contract, observation cache, partitions, and sampler seed
are all held at their frozen values so the contrast is attributable.

Sealed surfaces are enforced by `native_chunk_dataset`: only `train` and `selection` may be opened.
The natural-outcome and balanced-diagnostic partitions and `val_unseen` are never touched.

Resumable: state is checkpointed every calibration point, and `--resume` restarts from the last
completed update rather than from zero.
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

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.data.history import HISTORY_CONTRACTS, HistoryMode
from spatialvlm.data.native_chunk_dataset import (
    SELECTION_SPLIT,
    TRAIN_SPLIT,
    chunk_sample_summary,
    load_chunk_samples,
)
from spatialvlm.data.native_qwen_chunk import NativeQwenChunkConfig, build_native_qwen_chunk_batch
from spatialvlm.eval.action_chunk_decision import build_action_chunk_codebook
from spatialvlm.training.native_c0_gate import (
    NATIVE_C0_MODEL_ID,
    NATIVE_C0_MODEL_REVISION,
    gradient_report,
    validate_lora_only_trainables,
)
from spatialvlm.training.native_chunk_loss import CHUNK_LOSS_METHOD, restricted_chunk_loss

RUN_ID = "native_chunk_pilot_v1"
_STOP_REQUESTED = False


def _request_stop(signum: int, frame: Any) -> None:  # noqa: ARG001
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_inputs(processor, sample, contract, config, codebook, device):
    batch = build_native_qwen_chunk_batch(
        processor,
        sample.window(contract),
        instruction=sample.instruction,
        target_actions=sample.target_actions,
        config=config,
        codebook=codebook,
    )
    inputs = {k: v.to(device) for k, v in batch.model_inputs.items() if k != "labels"}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    return inputs, batch.record["supervised_token_positions"]


@torch.no_grad()
def _calibrate(model, processor, samples, contract, config, codebook, device) -> dict[str, Any]:
    """Score the frozen 256-decision selection surface under the chunk objective."""
    model.eval()
    losses: list[float] = []
    position_losses: list[list[float]] = []
    predicted: list[list[int]] = []
    targets: list[list[int]] = []
    for sample in samples:
        inputs, positions = _build_inputs(
            processor, sample, contract, config, codebook, device
        )
        out = model(**inputs)
        loss, per = restricted_chunk_loss(
            out.logits,
            action_token_positions=positions,
            codebook=codebook,
            target_actions=sample.target_actions,
        )
        losses.append(float(loss))
        position_losses.append(per)
        ids = [codebook.action_token_id(a) for a in R2RCEAction]
        predicted.append(
            [
                int(torch.stack([out.logits[0, p - 1, i] for i in ids]).argmax())
                for p in positions
            ]
        )
        targets.append([int(a) for a in sample.target_actions])
        del out
    model.train()

    flat_pred = [p for row in predicted for p in row]
    flat_tgt = [t for row in targets for t in row]
    correct = sum(1 for p, t in zip(flat_pred, flat_tgt) if p == t)
    per_action: dict[str, dict[str, int]] = {}
    for action in R2RCEAction:
        denom = sum(1 for t in flat_tgt if t == int(action))
        num = sum(1 for p, t in zip(flat_pred, flat_tgt) if t == int(action) and p == t)
        per_action[action.name] = {
            "predicted_count": sum(1 for p in flat_pred if p == int(action)),
            "recall_numerator": num,
            "recall_denominator": denom,
        }
    exact_chunk = sum(1 for p, t in zip(predicted, targets) if p == t)
    return {
        "selection_chunk_ce_mean": sum(losses) / len(losses),
        "selection_position_ce_mean": [
            sum(row[i] for row in position_losses) / len(position_losses)
            for i in range(len(position_losses[0]))
        ],
        "position_accuracy": correct / len(flat_tgt),
        "exact_chunk_match": exact_chunk,
        "exact_chunk_match_total": len(predicted),
        "constant_predicted_chunks": sum(1 for row in predicted if len(set(row)) == 1),
        "per_action": per_action,
        "scored_decisions": len(samples),
        "supervised_positions": len(flat_tgt),
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
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--max-updates", type=int, default=4000)
    parser.add_argument("--calibrate-every", type=int, default=60)
    parser.add_argument("--calibration-decisions", type=int, default=256)
    parser.add_argument("--max-hours", type=float, default=12.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--lazy-frames",
        action="store_true",
        help=(
            "Do not retain cached RGB in memory for the training split; re-read each object "
            "when the sample is used. Every object is still opened and validated at load. "
            "Required above roughly 60k decisions, where the eager path exceeds host memory."
        ),
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + args.max_hours * 3600.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed_everything(args.seed)

    processor = AutoProcessor.from_pretrained(
        NATIVE_C0_MODEL_ID, revision=NATIVE_C0_MODEL_REVISION
    )
    codebook = build_action_chunk_codebook(processor.tokenizer, chunk_size=args.chunk_size)
    config = NativeQwenChunkConfig(chunk_size=args.chunk_size)
    contract = HISTORY_CONTRACTS[HistoryMode.H4_SHORT_WINDOW]

    dataset = R2RCEDataset.from_release(args.episodes, args.ground_truth, split="train")
    train_samples = load_chunk_samples(
        cache_index_path=args.cache_index,
        object_root=args.cache_object_root,
        dataset=dataset,
        sampler_seed=args.sampler_seed,
        chunk_size=args.chunk_size,
        split=TRAIN_SPLIT,
        lazy_frames=args.lazy_frames,
    )
    selection_samples = load_chunk_samples(
        cache_index_path=args.cache_index,
        object_root=args.cache_object_root,
        dataset=dataset,
        sampler_seed=args.sampler_seed,
        chunk_size=args.chunk_size,
        split=SELECTION_SPLIT,
        limit=args.calibration_decisions,
    )

    from spatialvlm.backbone.qwen3_vl import Qwen3VLBackbone

    model = Qwen3VLBackbone(
        model_id=NATIVE_C0_MODEL_ID,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        enable_lora=True,
        freeze_base_model=True,
        apply_peft_2880_workaround=False,
        apply_spatial_rope=False,
        device=device,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        revision=NATIVE_C0_MODEL_REVISION,
    )
    base = model.model
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    base.config.use_cache = False
    trainable = validate_lora_only_trainables(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    manifest = {
        "run_id": args.out.name or RUN_ID,
        "status_boundary": "exploratory_capability_pilot_not_a_paper_artifact",
        "objective": CHUNK_LOSS_METHOD,
        "chunk_codebook": codebook.to_record(),
        "model_id": NATIVE_C0_MODEL_ID,
        "model_revision": NATIVE_C0_MODEL_REVISION,
        "trainable_contract": trainable,
        "hyperparameters": {
            k: v for k, v in vars(args).items() if not isinstance(v, Path)
        },
        "train_summary": chunk_sample_summary(train_samples),
        "selection_summary": chunk_sample_summary(selection_samples),
        "sealed_splits_never_opened": ["primary_evaluation", "balanced_diagnostic", "val_unseen"],
        "started_unix": started,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ckpt_path = out / "state.pt"
    best_path = out / "best.pt"
    best_ce = float("inf")
    start_update = 0
    trace: list[dict[str, Any]] = []
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(base, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        start_update = int(state["update"])
        trace = state.get("trace", [])
        prior = [e["calibration"]["selection_chunk_ce_mean"] for e in trace if "calibration" in e]
        best_ce = min(prior) if prior else float("inf")
        print(f"[resume] continuing from update {start_update}", flush=True)

    order = list(range(len(train_samples)))
    rng = random.Random(args.seed)
    model.train()
    cursor = start_update * args.accumulation
    status = "completed"

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
            inputs, positions = _build_inputs(
                processor, sample, contract, config, codebook, device
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
            del outputs, loss, inputs

        grads = gradient_report(model)
        if not grads["all_gradients_finite"]:
            raise RuntimeError(f"Non-finite gradients at update {update}.")
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        entry = {
            "update": update,
            "train_chunk_ce_mean": sum(step_losses) / len(step_losses),
            "gradient_l2_norm": grads["gradient_l2_norm"],
            "elapsed_seconds": time.time() - started,
            "epoch_fraction": cursor / len(order),
        }
        if update % args.calibrate_every == 0 or update == args.max_updates:
            entry["calibration"] = _calibrate(
                model, processor, selection_samples, contract, config, codebook, device
            )
            from peft import get_peft_model_state_dict

            adapter = get_peft_model_state_dict(base)
            payload = {
                "update": update,
                "adapter": adapter,
                "optimizer": optimizer.state_dict(),
                "trace": [*trace, entry],
            }
            # Latest, for --resume.
            torch.save(payload, ckpt_path)
            # Every calibration point, so no checkpoint is ever lost to overwriting. This is the
            # defect that cost native_chunk_pilot_v1 its minimum-CE checkpoint at update 450.
            torch.save(
                {"update": update, "adapter": adapter, "calibration": entry["calibration"]},
                out / f"adapter_u{update:05d}.pt",
            )
            # Best-so-far by the selection criterion, named so it cannot be confused with latest.
            current_ce = entry["calibration"]["selection_chunk_ce_mean"]
            if current_ce < best_ce:
                best_ce = current_ce
                torch.save(
                    {
                        "update": update,
                        "adapter": adapter,
                        "selection_chunk_ce_mean": current_ce,
                        "calibration": entry["calibration"],
                    },
                    best_path,
                )
                print(f"[best] new minimum selection CE {current_ce:.4f} at update {update}",
                      flush=True)
        trace.append(entry)
        with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        cal = entry.get("calibration")
        msg = (
            f"[{update:5d}] train_ce={entry['train_chunk_ce_mean']:.4f} "
            f"gl2={entry['gradient_l2_norm']:.2e} ep={entry['epoch_fraction']:.2f} "
            f"t={entry['elapsed_seconds'] / 3600:.2f}h"
        )
        if cal:
            msg += (
                f" | SEL ce={cal['selection_chunk_ce_mean']:.4f} "
                f"acc={cal['position_accuracy']:.3f} "
                f"exact={cal['exact_chunk_match']}/{cal['exact_chunk_match_total']} "
                f"const={cal['constant_predicted_chunks']} "
                f"STOPpred={cal['per_action']['STOP']['predicted_count']} "
                f"STOPrec={cal['per_action']['STOP']['recall_numerator']}/"
                f"{cal['per_action']['STOP']['recall_denominator']}"
            )
        print(msg, flush=True)

    result = {
        "run_id": args.out.name or RUN_ID,
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
