#!/usr/bin/env python3
"""NATIVE-CHUNK-CLOSED-LOOP-v1 — execute a chunk-objective checkpoint in Habitat.

Exploratory capability pilot. **Not a paper artifact and not a frozen gate.** It answers the one
question the chunk track has never answered: what does a chunk-trained policy *do* when it drives,
rather than what it scores on cached observations.

The policy emits `chunk_size` actions per visual context through the autoregressive scorer in
`spatialvlm.eval.native_chunk_scorer` — one multimodal prefill, then cached single-token decodes,
with the STOP-tail mask discarding anything proposed after a predicted STOP.

The episode suite, step cap, success radius and evaluator are the frozen ones used by
`native_c0_closed_loop_v1`, so behaviour is directly comparable to the manuscript's numbers. This
run does not re-verify those inputs against their hashes the way the frozen runner does; it is a
pilot, and its outputs must not be promoted into the manuscript without that verification.

`val_unseen` is never referenced.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from spatialvlm.data.datasets import R2RCEAction, R2RCEDataset
from spatialvlm.data.habitat_sim_r2r import HabitatSimR2RConfig, HabitatSimR2REnvironment
from spatialvlm.data.history import HistoryMode
from spatialvlm.eval.action_chunk_decision import build_action_chunk_codebook
from spatialvlm.eval.native_chunk_scorer import AutoregressiveChunkScorer, ChunkScorerConfig
from spatialvlm.eval.native_closed_loop import run_instrumented_r2r_ce_episode
from spatialvlm.eval.native_policy import NativePolicyConfig, NativeQwenPolicy
from spatialvlm.training.native_c0_gate import NATIVE_C0_MODEL_ID, NATIVE_C0_MODEL_REVISION

RUN_ID = "native_chunk_closed_loop_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--step-cap", type=int, default=500)
    parser.add_argument("--episode-limit", type=int, default=61)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--split", default="val_seen",
                        help="Split label stamped on the artifacts. It must name the release "
                             "--episodes actually came from; it selects nothing.")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="Last resort for a checkpoint whose arm manifest records no "
                             "lora_alpha. Alpha leaves no trace in a state dict.")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes").mkdir(exist_ok=True)
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        NATIVE_C0_MODEL_ID, revision=NATIVE_C0_MODEL_REVISION
    )
    codebook = build_action_chunk_codebook(processor.tokenizer, chunk_size=args.chunk_size)

    from spatialvlm.backbone.qwen3_vl import Qwen3VLBackbone
    from spatialvlm.eval.adapter_contract import resolve_adapter_contract, verify_adapter_keys

    # The adapter is built to the checkpoint's own contract, never to a default. A runner that
    # hardcodes rank/alpha/targets silently evaluates a different model than the one that trained
    # (question 5 of the five): only a rank difference raises, alpha and target-module differences
    # do not. See spatialvlm.eval.adapter_contract.
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    adapter_contract = resolve_adapter_contract(
        args.checkpoint, state["adapter"], alpha_override=args.lora_alpha
    )
    print(f"[adapter] {adapter_contract.describe()}", flush=True)

    model = Qwen3VLBackbone(
        model_id=NATIVE_C0_MODEL_ID,
        lora_rank=adapter_contract.rank,
        lora_alpha=adapter_contract.alpha,
        lora_dropout=0.0,
        lora_target_modules=adapter_contract.target_modules,
        enable_lora=True,
        freeze_base_model=True,
        apply_peft_2880_workaround=False,
        apply_spatial_rope=False,
        device=device,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        revision=NATIVE_C0_MODEL_REVISION,
    )
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    set_peft_model_state_dict(model.model, state["adapter"])
    verify_adapter_keys(state["adapter"], get_peft_model_state_dict(model.model))
    model.model.config.use_cache = True
    model.eval()
    print(
        f"[load] adapter from update {state.get('update')} "
        f"selection_ce={state.get('selection_chunk_ce_mean')}",
        flush=True,
    )

    scorer = AutoregressiveChunkScorer(
        model,
        processor,
        codebook,
        config=ChunkScorerConfig(chunk_size=args.chunk_size),
        device=device,
    )

    dataset = R2RCEDataset.from_release(args.episodes, args.ground_truth, split=args.split)
    entries = json.loads(args.suite_manifest.read_text(encoding="utf-8"))["episodes"]
    entries = entries[: args.episode_limit]
    environment = HabitatSimR2REnvironment(
        HabitatSimR2RConfig(scene_root=args.scene_root, gpu_device_id=args.gpu_device_id)
    )

    manifest = {
        "run_id": RUN_ID,
        "status_boundary": "exploratory_capability_pilot_not_a_paper_artifact",
        "checkpoint": str(args.checkpoint),
        "checkpoint_update": state.get("update"),
        "checkpoint_selection_ce": state.get("selection_chunk_ce_mean"),
        "chunk_size": args.chunk_size,
        "step_cap": args.step_cap,
        "episode_count": len(entries),
        "model_id": NATIVE_C0_MODEL_ID,
        "model_revision": NATIVE_C0_MODEL_REVISION,
        "scorer_method": "autoregressive_kv_cached_restricted_chunk_decode_v1",
        "split": args.split,
        "adapter_contract": adapter_contract.as_dict(),
        "val_unseen_referenced": False,
        "started_unix": started,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    metrics: list[dict[str, Any]] = []
    action_totals: Counter[str] = Counter()
    constant_episodes = 0
    stop_events: list[dict[str, Any]] = []

    for index, entry in enumerate(entries):
        if (time.time() - started) / 3600.0 >= args.max_hours:
            print("[stop] wall-clock cap reached", flush=True)
            break
        episode_id = str(entry["episode_id"])
        episode = dataset.episode_record(episode_id)
        ground_truth = dataset.ground_truth_record(episode_id)
        policy = NativeQwenPolicy(
            scorer=scorer,
            config=NativePolicyConfig.for_mode(
                HistoryMode.H4_SHORT_WINDOW,
                chunk_length=args.chunk_size,
                step_cap=args.step_cap,
            ),
        )
        t0 = time.time()
        result = run_instrumented_r2r_ce_episode(
            episode,
            ground_truth,
            environment,
            policy,
            episode_index=int(entry.get("index", index)),
            max_steps=args.step_cap,
        )
        elapsed = time.time() - t0
        record = result.evaluation.to_dict()
        actions = [R2RCEAction(a).name for a in record["trace"]["actions"]]
        counts = Counter(actions)
        action_totals.update(counts)
        is_constant = len(counts) == 1
        constant_episodes += int(is_constant)
        m = record["metrics"]
        if m["stopped"]:
            stop_events.append(
                {
                    "episode_id": episode_id,
                    "step_count": m["step_count"],
                    "navigation_error": m["navigation_error"],
                    "success": m["success"],
                }
            )
        summary = {
            **m,
            "episode_seconds": elapsed,
            "action_histogram": dict(counts),
            "constant_action_episode": is_constant,
            "distinct_actions": len(counts),
        }
        metrics.append(summary)
        (out / "episodes" / f"{episode_id}.json").write_text(
            json.dumps({**record, "pilot_summary": summary}, indent=2), encoding="utf-8"
        )
        print(
            f"[{index + 1:3d}/{len(entries)}] ep {episode_id:>6} "
            f"SR={m['success']:.0f} NE={m['navigation_error']:.2f} steps={m['step_count']:3d} "
            f"stopped={m['stopped']} actions={dict(counts)} {elapsed:.1f}s",
            flush=True,
        )

    n = len(metrics)
    agg = {
        "episodes_evaluated": n,
        "success_rate": sum(m["success"] for m in metrics) / n if n else None,
        "oracle_success_rate": sum(m["oracle_success"] for m in metrics) / n if n else None,
        "spl": sum(m["spl"] for m in metrics) / n if n else None,
        "ndtw": sum(m["ndtw"] for m in metrics) / n if n else None,
        "sdtw": sum(m["sdtw"] for m in metrics) / n if n else None,
        "mean_navigation_error": statistics.mean(m["navigation_error"] for m in metrics)
        if n
        else None,
        "median_navigation_error": statistics.median(m["navigation_error"] for m in metrics)
        if n
        else None,
        "episodes_that_stopped": sum(1 for m in metrics if m["stopped"]),
        "constant_action_episodes": constant_episodes,
        "constant_action_fraction": constant_episodes / n if n else None,
        "total_actions": sum(action_totals.values()),
        "action_histogram": dict(action_totals),
        "stop_events": stop_events,
        "mean_episode_seconds": statistics.mean(m["episode_seconds"] for m in metrics)
        if n
        else None,
    }
    result_doc = {
        "run_id": RUN_ID,
        "status": "completed" if n == len(entries) else "partial",
        "elapsed_seconds": time.time() - started,
        "checkpoint_update": state.get("update"),
        "aggregate": agg,
        "per_episode": metrics,
    }
    (out / "result.json").write_text(json.dumps(result_doc, indent=2), encoding="utf-8")
    print(
        f"\n[done] {n} episodes | SR={agg['success_rate']} OSR={agg['oracle_success_rate']} "
        f"SPL={agg['spl']} SDTW={agg['sdtw']} | stopped={agg['episodes_that_stopped']} "
        f"| constant-action episodes={constant_episodes}/{n} | actions={agg['action_histogram']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
