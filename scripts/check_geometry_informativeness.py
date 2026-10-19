"""Does the geometry branch carry information, or has it collapsed to a constant?

ON-vs-MASKED cannot distinguish these. A branch that emits the same vector for every input still
makes MASKED (zeros) look catastrophic, because zeroing is a large change from that constant --
yet such a branch carries no geometry at all. This measures the thing that matters directly:

1. between-sample variation: do different scenes produce different embeddings?
2. within-sample variation: do the 256 tokens of one scene differ from each other?
3. how much of the output is the learned per-frame embedding rather than the depth content?

Read-only: loads a saved adapter checkpoint, touches no running artifact.
"""
import glob, json, sys, torch
sys.path.insert(0, "src")
from spatialvlm.geometry.geometry_branch import GeometryBranch

ck = sorted(glob.glob("/w/artifacts/spark/native_chunk_geometry_134k_v1/adapter_u*.pt"))[-1]
state = torch.load(ck, map_location="cpu", weights_only=False)
print(f"checkpoint: {ck.split('/')[-1]} (update {state['update']})")

probe = torch.load(sorted(glob.glob("/objects/*/*.pt"))[0], map_location="cpu", weights_only=False)
d = probe["depth_frames_m"]
branch = GeometryBranch(hidden_size=4096, n_frames=int(d.shape[0]),
                        depth_hw=(int(d.shape[1]), int(d.shape[2])), gatr_blocks=8).float()
branch.load_state_dict(state["geometry"]); branch.eval()

files = sorted(glob.glob("/objects/*/*.pt"))[:24]
embs, coords_all = [], []
with torch.no_grad():
    for f in files:
        o = torch.load(f, map_location="cpu", weights_only=False)
        c, v = branch.points_from_depth(o["depth_frames_m"])
        embs.append(branch(c, v)[0]); coords_all.append(c[0])
E = torch.stack(embs)                      # [S, 256, 4096]
C = torch.stack(coords_all)                # [S, 256, 3]

per_token_mean = E.mean(0)                                  # [256, 4096]
between = (E - per_token_mean).pow(2).mean().sqrt()         # variation across scenes
within  = (per_token_mean - per_token_mean.mean(0)).pow(2).mean().sqrt()
overall = E.std()
print(f"\ninput depth actually differs across the {len(files)} samples: "
      f"coord std {C.std():.4f} m, mean |coord| {C.abs().mean():.4f} m")
print(f"\n  BETWEEN-scene variation (same token slot, different scene): {between:.6f}")
print(f"  WITHIN-scene  variation (token slots differ from each other): {within:.6f}")
print(f"  overall embedding std:                                       {overall:.6f}")
ratio = float(between / (within + 1e-9))
print(f"\n  between/within ratio: {ratio:.4f}")

# how much survives if we replace real depth with a constant plane?
with torch.no_grad():
    flat = torch.full_like(probe["depth_frames_m"], 3.0)
    cf, vf = branch.points_from_depth(flat)
    Ef = branch(cf, vf)[0]
    c0, v0 = branch.points_from_depth(probe["depth_frames_m"])
    E0 = branch(c0, v0)[0]
delta = (E0 - Ef).pow(2).mean().sqrt()
print(f"  real depth vs a flat 3 m wall, same branch: RMS difference {delta:.6f}")
print(f"    (relative to overall std: {float(delta/overall):.3f})")

import json as _json
_out = {
    "checkpoint": ck.split("/")[-1], "update": int(state["update"]),
    "between_scene_variation": float(between), "within_scene_variation": float(within),
    "overall_embedding_std": float(overall), "between_within_ratio": float(ratio),
    "flat_wall_rms_shift": float(delta), "flat_wall_shift_over_std": float(delta/overall),
    "probe_samples": len(files),
}
_p = pathlib.Path("artifacts/context/geometry_informativeness.json") if (pathlib := __import__("pathlib")) else None
_p.write_text(_json.dumps(_out, indent=2))
print(f"[artifact] {_p}")
verdict = ("INFORMATIVE - output tracks the depth input"
           if ratio > 0.05 and float(delta/overall) > 0.05
           else "COLLAPSED - output barely depends on the depth input")
print(f"\n  VERDICT: {verdict}")
