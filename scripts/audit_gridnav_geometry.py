"""Adversarial audit of the GridNav geometry: does the code do what the paper says?

Checks the claims a reader would have to take on trust otherwise -- wavelength range, metric
units end to end, validity handling, the fixed GATr unit constant, direction sensitivity, and the
output normalisation. Run alongside pytest; the tests check behaviour, this checks the PAPER.
"""
import torch, math
from spatialvlm.geometry.gridcell_rope3d_v2 import IcosahedralRoPE3Dv2
from spatialvlm.geometry.geometry_branch import GeometryBranch, UNIT_SCALE_M
from spatialvlm.utils.camera import CameraIntrinsics

fails = []
def chk(ok, name, detail):
    print(("  OK   " if ok else "  FAIL ") + f"{name:<44} {detail}")
    if not ok: fails.append(name)

# 1. Wavelengths: paper says 0.60-12.0 m
enc = IcosahedralRoPE3Dv2()
lo, hi = enc.wavelengths_m
chk(abs(hi-12.0) < 0.01 and 0.59 < lo < 0.60, "RoPE3D wavelengths match paper (0.60-12.0 m)",
    f"{lo:.4f} .. {hi:.4f} m")

# 2. Is the longest wavelength >= scene extent? (question 3: dynamic range)
chk(hi >= 10.0, "longest wavelength >= 10 m depth clip", f"{hi:.2f} m vs 10 m clip")
# and shortest > one forward step (0.25 m) so consecutive steps are distinguishable, not aliased
chk(lo > 0.25, "shortest wavelength > one 0.25 m step", f"{lo:.4f} m = {lo/0.25:.1f} steps")

# 3. Aliasing. A MULTI-SCALE code aliases only where every frequency coincides, not at the
#    longest wavelength -- an earlier version of this audit asserted the opposite and "failed"
#    against correct code. Check the property that actually matters: separations spanning the
#    scene must all be distinguishable.
prev = enc(torch.zeros(1,1,3))
worst = min((float((enc(torch.tensor([[[s,0.,0.]]])) - prev).abs().max()), s)
            for s in (0.25, 0.6, 1.0, 3.0, 6.0, 10.0, 12.0))
chk(worst[0] > 1e-2, "no aliasing at any separation across the scene",
    f"weakest separation {worst[1]} m -> phase diff {worst[0]:.4f}")
b = torch.zeros(1,2,3); b[0,1,0] = 6.0         # half the extent: must differ
p2 = enc(b)
chk(not torch.allclose(p2[0,0], p2[0,1], atol=1e-3), "points 6 m apart are distinguishable",
    f"max phase diff {(p2[0,0]-p2[0,1]).abs().max():.4f}")

# 4. Units: backprojection must return METRES matching the input depth
intr = CameraIntrinsics.from_hfov(width=256, height=256, hfov_degrees=90.0)
d = torch.full((1,256,256), 3.0)               # a wall at exactly 3 m
br = GeometryBranch(hidden_size=64, n_frames=1, patch_size_px=64, depth_hw=(256,256),
                    gatr_blocks=1, output_rms_norm=True).float()
pts, val = br.points_from_depth(d)
zc = pts[0,:,2]
chk(abs(float(zc.mean())-3.0) < 1e-3, "backprojection preserves metric depth",
    f"wall at 3.0 m -> mean z {float(zc.mean()):.4f} m")
chk(float(val.min()) == 1.0, "fully valid depth -> validity 1", f"min validity {float(val.min())}")

# 5. Validity: an all-zero (invalid) depth frame must zero coords AND validity, not sit at origin
d0 = torch.zeros((1,256,256))
p0, v0 = br.points_from_depth(d0)
chk(float(v0.max()) == 0.0 and float(p0.abs().max()) == 0.0,
    "all-invalid depth -> zero coords and zero validity", f"validity max {float(v0.max())}")

# 6. UNIT_SCALE_M is a fixed constant, not a per-sample statistic (the original defect)
chk(UNIT_SCALE_M == 10.0, "GATr unit constant is fixed at the depth clip", f"{UNIT_SCALE_M} m")

# 7. Equivariance sanity: rotating the point cloud must rotate the extract_point readout,
#    i.e. the branch is not silently direction-blind.
torch.manual_seed(0)
c = torch.randn(1, br.n_tokens, 3); v = torch.ones(1, br.n_tokens)
th = math.pi/2
R = torch.tensor([[math.cos(th),-math.sin(th),0.],[math.sin(th),math.cos(th),0.],[0.,0.,1.]])
o1 = br(c, v); o2 = br(c @ R.T, v)
chk(not torch.allclose(o1, o2, atol=1e-4), "rotating the scene changes the branch output",
    f"max |diff| {float((o1-o2).abs().max()):.4f}  (direction-blind would be ~0)")

# 8. RMS norm: output magnitude must be scale-free
big = GeometryBranch(hidden_size=64, n_frames=1, patch_size_px=64, depth_hw=(256,256),
                     gatr_blocks=1, output_rms_norm=True).float()
with torch.no_grad():
    for prm in big.projector.parameters(): prm.mul_(1e5)
ob = big(c, v)
chk(float(ob.abs().max()) < 1e2, "RMS norm keeps output bounded under 1e5 weight inflation",
    f"absmax {float(ob.abs().max()):.3f}")

print()
print("AUDIT PASS" if not fails else f"*** {len(fails)} FAILED: {fails} ***")
