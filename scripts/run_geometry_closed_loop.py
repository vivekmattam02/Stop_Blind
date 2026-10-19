#!/usr/bin/env python3
"""GEOMETRY closed-loop runner. Additive; run_chunk_closed_loop.py is left untouched.

Derived from the chunk closed-loop runner, with the one change that makes a geometry checkpoint
evaluable at all: the geometry branch is loaded from the checkpoint, the v2 rotary patch and
norm-matched embedding hook are applied, and scoring goes through GeometryChunkScorer so the
rollout sequence carries the same 256 spliced geometry tokens the adapter was trained with.

Running the generic runner on this checkpoint would load state["adapter"] and silently drop the
409 geometry tensors, evaluating an adapter on a sequence layout it never saw. That produces
well-formed actions and meaningless metrics, so the two runners are kept separate rather than
switched by a flag.
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

RUN_ID = "native_chunk_geometry_closed_loop_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--geometry-frames", type=int, default=4)
    parser.add_argument("--patch-size-px", type=int, default=32)
    parser.add_argument("--depth-size", type=int, default=256)
    parser.add_argument("--gatr-blocks", type=int, default=8)
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
    parser.add_argument("--legacy-contract", action="store_true",
                        help="Reproduce a pre-stamp checkpoint using the CLI geometry args. "
                             "Never use this for rescue-campaign checkpoints.")
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
    if "geometry" not in state:
        raise SystemExit(
            f"{args.checkpoint} carries no 'geometry' state. This runner is only for geometry-arm "
            "checkpoints; use scripts/run_chunk_closed_loop.py for adapter-only ones."
        )

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

    # Geometry branch + v2 injection, reconstructed to match the training configuration.
    from spatialvlm.backbone.rope_patch_v2 import apply_rope_patch_v2
    from spatialvlm.geometry.geometry_branch import GeometryBranch

    rotary, embed = apply_rope_patch_v2(model.model)
    # Deploy must match train (question 5 of the five). NOTHING about the branch's token count or
    # output contract lives in state_dict: patch_size_px, n_frames and output_rms_norm change no
    # parameter shape, so strict=True loads cleanly even when they disagree with training and the
    # policy then sees a different number of geometry tokens, un-normalised. This script's CLI
    # defaults (patch 32) do NOT match the rescue campaign's training (patch 64), so guessing here
    # would silently corrupt every closed-loop number. Read the stamped contract; fail closed.
    contract = state.get("geometry_contract")
    if contract is None:
        if not args.legacy_contract:
            raise SystemExit(
                f"{args.checkpoint} carries no geometry_contract stamp. Refusing to guess the "
                "branch shape from CLI defaults -- that is a silent train/deploy mismatch. Pass "
                "--legacy-contract only to reproduce a pre-stamp run with its original arguments."
            )
        contract = {"patch_size_px": args.patch_size_px, "n_frames": args.geometry_frames,
                    "depth_hw": [args.depth_size, args.depth_size],
                    "gatr_blocks": args.gatr_blocks, "output_rms_norm": False, "n_tokens": None}
        print("[load] LEGACY contract from CLI args (pre-stamp checkpoint)", flush=True)
    branch = GeometryBranch(
        hidden_size=model.hidden_size,
        n_frames=int(contract["n_frames"]),
        patch_size_px=int(contract["patch_size_px"]),
        depth_hw=(int(contract["depth_hw"][0]), int(contract["depth_hw"][1])),
        gatr_blocks=int(contract["gatr_blocks"]),
        output_rms_norm=bool(contract["output_rms_norm"]),
        device=device,
    ).float()
    if contract.get("n_tokens") is not None and branch.n_tokens != int(contract["n_tokens"]):
        raise SystemExit(
            f"Rebuilt branch emits {branch.n_tokens} tokens but training used "
            f"{contract['n_tokens']}. Refusing to run a mismatched policy."
        )
    print(f"[load] geometry contract: patch={contract['patch_size_px']} "
          f"frames={contract['n_frames']} tokens={branch.n_tokens} "
          f"output_rms_norm={branch.output_rms_norm}", flush=True)
    missing = branch.load_state_dict(state["geometry"], strict=True)
    branch.eval()
    print(
        f"[load] geometry branch: {sum(p.numel() for p in branch.parameters()):,} params, "
        f"{branch.n_tokens} tokens/decision",
        flush=True,
    )
    model.model.config.use_cache = True
    model.eval()
    print(
        f"[load] adapter from update {state.get('update')} "
        f"selection_ce={state.get('selection_chunk_ce_mean')}",
        flush=True,
    )

    from spatialvlm.eval.geometry_chunk_scorer import (
        GEOMETRY_SCORER_METHOD,
        GeometryChunkScorer,
    )

    scorer = GeometryChunkScorer(
        model,
        processor,
        codebook,
        branch=branch,
        rotary=rotary,
        embed=embed,
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
        "run_id": args.out.name or RUN_ID,
        "status_boundary": "exploratory_capability_pilot_not_a_paper_artifact",
        "checkpoint": str(args.checkpoint),
        "checkpoint_update": state.get("update"),
        "checkpoint_selection_ce": state.get("selection_chunk_ce_mean"),
        "chunk_size": args.chunk_size,
        "step_cap": args.step_cap,
        "episode_count": len(entries),
        "model_id": NATIVE_C0_MODEL_ID,
        "model_revision": NATIVE_C0_MODEL_REVISION,
        "scorer_method": GEOMETRY_SCORER_METHOD,
        "geometry_tokens_per_decision": branch.n_tokens,
        "geometry_param_count": sum(p.numel() for p in branch.parameters()),
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
