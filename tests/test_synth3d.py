"""
test_synth3d.py — 曲面贴附合成器 自检
====================================

验证三件事（不依赖训练好的模型）：
  S1 平面上的真值面积 = 解析椭圆面积 π·ra·rb（验证数值积分正确）
  S2 渲染输出完整（rgb/mask/depth 形状、掩膜非空、深度有效）
  S3 算法链路能复原真值面积：LiDAR 路径 <5%、曲率先验路径 <3%
     （真值由合成器解析给出，故这是"端到端几何链路"的回归保护）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wg_train.synth3d import SurfaceSynth, gt_area, wound_region          # noqa: E402
from wg_algorithms.surface_area import SurfaceAreaIntegrator              # noqa: E402


def test_gt_area_plane():
    """平面：J=1，真值面积应等于参数域椭圆面积 π·ra·rb。"""
    from wg_algorithms.curvature_prior import PlaneModel
    plane = PlaneModel([0, 0, 0.30], [0, 0, -1])
    for ra, rb in ((0.02, 0.012), (0.04, 0.025)):
        a = gt_area(ra, rb, [], plane)
        want = np.pi * ra * rb
        rel = abs(a - want) / want * 100.0
        assert rel < 1.0, f"平面真值面积偏差 {rel:.2f}% 过大"
    print("[S1] 平面真值面积 = π·ra·rb（误差<1%） -> PASS")


def test_render_shapes():
    synth = SurfaceSynth(img_size=256, part_type="limb_forearm", seed=3)
    smp = synth.render(wound_area_cm2=12.0)
    assert smp["rgb"].shape == (256, 256, 3) and smp["rgb"].dtype == np.uint8
    assert smp["mask"].shape == (256, 256) and smp["mask"].dtype == np.uint8
    assert (smp["mask"] > 0).sum() > 100, "伤口掩膜不应为空"
    assert np.isfinite(smp["depth"]).any(), "深度图应含有效值"
    # 目标面积与真值面积应同量级（1~30 cm² 区间内不应量纲错误）
    assert 3.0 < smp["gt_area_cm2"] < 40.0, f"真值面积量纲异常: {smp['gt_area_cm2']}"
    print(f"[S2] 渲染输出 rgb/mask/depth 形状正确，真值面积={smp['gt_area_cm2']:.2f} cm² -> PASS")


def test_area_recovery():
    """算法能否从渲染的图里复原真值曲面面积。"""
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
    assert ml < 6.0, f"LiDAR 路径复原误差 {ml:.2f}% 超阈值"
    assert mm < 4.0, f"曲率先验路径复原误差 {mm:.2f}% 超阈值"
    print(f"[S3] 面积复原 LiDAR {ml:.2f}% / 曲率先验 {mm:.2f}% -> PASS")


if __name__ == "__main__":
    test_gt_area_plane()
    test_render_shapes()
    test_area_recovery()
    print("\n结果: ALL PASS")
