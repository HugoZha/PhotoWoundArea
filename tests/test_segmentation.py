"""
test_segmentation.py — real segmentation model interface / training-evaluation scaffold self-test
============================================================

Coverage: metrics (known masks), synthetic data generation, dataset shapes, model forward pass, and
the end-to-end interface smoke test of "training checkpoint -> load -> inference -> segmentation interface".
When torch is missing, model/dataset/training-related cases are automatically skipped, keeping only pure numpy cases.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wg_train import synthesize_data, metrics  # noqa: E402
from wg_algorithms.segmentation import (  # noqa: E402
    WoundSegModel, _keep_largest_cc,
)

try:
    import torch  # noqa: E402
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False

# Filled by test_synthesize for reuse by subsequent cases (avoid regenerating)
RGB = None


def test_metrics_known():
    a = np.zeros((10, 10), np.uint8)
    a[2:8, 2:8] = 255
    b = a.copy()
    assert abs(metrics.dice(a, b) - 1.0) < 1e-6, "identical should give dice=1"
    assert abs(metrics.iou(a, b) - 1.0) < 1e-6, "identical should give iou=1"
    c = np.zeros((10, 10), np.uint8)
    c[0:2, 0:2] = 255          # completely disjoint from a[2:8,2:8]
    assert metrics.dice(a, c) == 0.0, "disjoint should give dice=0"
    # Sanity of boundary metric values
    assert metrics.hd95(a, b) == 0.0
    assert 0.0 <= metrics.boundary_f1(a, b, tol=2) <= 1.0
    print("[S1] Metrics (known mask) dice=1/iou=1/disjoint=0 -> PASS")


def test_synthesize():
    global RGB
    rng = np.random.default_rng(7)
    rgb, mask = synthesize_data.generate_pair(img_size=128, rng=rng)
    RGB = rgb
    assert rgb.shape == (128, 128, 3) and mask.shape == (128, 128)
    assert rgb.dtype == np.uint8 and mask.dtype == np.uint8
    assert mask.sum() > 0, "synthetic wound mask should not be empty"
    assert rgb.max() <= 255 and rgb.min() >= 0
    print(f"[S2] Synthetic data  rgb{rgb.shape} mask{mask.shape} wound pixels={int((mask>0).sum())} -> PASS")


def test_dataset_and_model():
    if not HAS_TORCH:
        print("[S3/S4] Skipped: torch not installed")
        return
    from wg_train.dataset import SyntheticSegDataset
    from wg_algorithms.seg_model import WoundUNet, load_wound_model, predict_proba

    ds = SyntheticSegDataset(n=16, img_size=64, augment=False, seed=1)
    assert len(ds) == 16
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["mask"].shape == (1, 64, 64)
    print("[S3] Dataset shapes (3,64,64)/(1,64,64) -> PASS")

    model = WoundUNet(in_ch=3, base=16, depth=3)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 64, 64) and float(out.min()) >= 0 and float(out.max()) <= 1
    print("[S4] Model forward (2,1,64,64) value range [0,1] -> PASS")

    # Training checkpoint -> save -> load -> inference interface chain
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_ckpt.pt")
    try:
        torch.save({"state_dict": model.state_dict(),
                    "config": {"in_ch": 3, "base": 16, "depth": 3, "img_size": 64}},
                   ckpt)
        m2 = load_wound_model(ckpt, device="cpu")
        proba = predict_proba(m2, RGB, "cpu", img_size=64)
        assert proba.shape == (128, 128) and proba.dtype == np.float32
        assert proba.min() >= 0 and proba.max() <= 1
    finally:
        if os.path.exists(ckpt):
            os.remove(ckpt)
    print("[S5] Checkpoint save/load/inference interface chain -> PASS")


def test_segmenter_interface():
    if not HAS_TORCH:
        print("[S6] Skipped: torch not installed")
        return
    from wg_algorithms.seg_model import WoundUNet

    model = WoundUNet(in_ch=3, base=16, depth=3)
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_ckpt2.pt")
    try:
        torch.save({"state_dict": model.state_dict(),
                    "config": {"in_ch": 3, "base": 16, "depth": 3, "img_size": 64}},
                   ckpt)
        seg = WoundSegModel(weights_path=ckpt, device="cpu", img_size=64)
        mask, info = seg.segment(RGB)
        assert mask.shape == (128, 128) and mask.dtype == np.uint8
        assert info["synthetic"] is False, "real model info must have synthetic=False"
        assert "model_path" in info and "prob_mean" in info
    finally:
        if os.path.exists(ckpt):
            os.remove(ckpt)
    print("[S6] WoundSegModel.segment interface (synthetic=False) -> PASS")


def test_keep_largest_cc():
    mask = np.zeros((20, 20), np.uint8)
    mask[2:5, 2:5] = 255       # small block
    mask[10:16, 10:16] = 255   # large block
    out = _keep_largest_cc(mask)
    assert out.sum() == (6 * 6) * 255, "only the large block should be kept"
    print("[S7] Keep largest connected component -> PASS")


if __name__ == "__main__":
    test_metrics_known()
    test_synthesize()
    test_dataset_and_model()
    test_segmenter_interface()
    test_keep_largest_cc()
    print("\nResult: ALL PASS")
