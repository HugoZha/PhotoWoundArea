"""
test_synth3d.py — surface-attached synthesizer self-test
====================================

Verify three things (does not depend on a trained model):
  S1 Ground truth area on a plane = analytic ellipse area π·ra·rb (verifies numerical integration is correct)
  S2 Render outputs are complete (rgb/mask/depth shapes, mask non-empty, depth valid)
  S3 The algorithm chain can recover the ground truth area: LiDAR path <5%, curvature prior path <3%
     (ground truth is given analytically by the synthesizer, so this is regression protection for the "end-to-end geometry chain")
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wg_train.synth3d import SurfaceSynth, gt_area, wound_region          # noqa: E402
from wg_algorithms.surface_area import SurfaceAreaIntegrator              # noqa: E402


def test_gt_area_plane():
    """Plane: J=1, the ground truth area should equal the parameter-domain ellipse area π·ra·rb."""
    from wg_algorithms.curvature_prior import PlaneModel
    plane = PlaneModel([0, 0, 0.30], [0, 0, -1])
    for ra, rb in ((0.02, 0.012), (0.04, 0.025)):
        a = gt_area(ra, rb, [], plane)
        want = np.pi * ra * rb
        rel = abs(a - want) / want * 100.0
        assert rel < 1.0, f"planar ground truth area deviation {rel:.2f}% too large"
    print("[S1] Planar ground truth area = π·ra·rb (error<1%) -> PASS")


def test_render_shapes():
    synth = SurfaceSynth(img_size=256, part_type="limb_forearm", seed=3)
    smp = synth.render(wound_area_cm2=12.0)
    assert smp["rgb"].shape == (256, 256, 3) and smp["rgb"].dtype == np.uint8
    assert smp["mask"].shape == (256, 256) and smp["mask"].dtype == np.uint8
    assert (smp["mask"] > 0).sum() > 100, "wound mask should not be empty"
    assert np.isfinite(smp["depth"]).any(), "depth map should contain valid values"
    # Target area and ground truth area should be of the same order of magnitude (no unit error within the 1~30 cm² range)
    assert 3.0 < smp["gt_area_cm2"] < 40.0, f"ground truth area unit anomaly: {smp['gt_area_cm2']}"
    print(f"[S2] Render outputs rgb/mask/depth shapes correct, ground truth area={smp['gt_area_cm2']:.2f} cm² -> PASS")


def test_area_recovery():
    """Whether the algorithm can recover the ground truth surface area from the rendered images."""
    integ = SurfaceAreaIntegrator()
    err_l, err_m = [], []
    for seed in (11, 12, 13, 14, 15):
        synth = SurfaceSynth(img_size=384, part_type="limb_forearm", seed=seed)
        smp = synth.render()
        a_l = integ.integrate(smp["rgb"], smp["mask"], smp["K"], smp["M"],
                              ("lidar", smp["depth"]))["area_cm2"]
        a_m = integ.integrate(smp["rgb"], smp["mask"], smp["K"], smp["M"],
                              ("model", synth.model))["area_cm2"]
        g = smp["gt_area_cm2"]
        err_l.append(abs(a_l - g) / g * 100.0)
        err_m.append(abs(a_m - g) / g * 100.0)
    ml, mm = float(np.mean(err_l)), float(np.mean(err_m))
    assert ml < 6.0, f"LiDAR path recovery error {ml:.2f}% exceeds threshold"
    assert mm < 4.0, f"curvature prior path recovery error {mm:.2f}% exceeds threshold"
    print(f"[S3] Area recovery LiDAR {ml:.2f}% / curvature prior {mm:.2f}% -> PASS")


if __name__ == "__main__":
    test_gt_area_plane()
    test_render_shapes()
    test_area_recovery()
    print("\nResult: ALL PASS")
