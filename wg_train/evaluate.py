#!/usr/bin/env python3
"""
evaluate.py — segmentation evaluation CLI with an optional area-chain check
===================================================

用法（合成验证集，与 train 同分布不同种子）:
    python -m wg_train.evaluate --model runs/exp1/best.pt --synthetic 60 --out runs/exp1/eval

用法（真实验证集）:
    python -m wg_train.evaluate --model runs/exp1/best.pt \
        --image_dir data/val_images --mask_dir data/val_masks

Optional area-chain check (predicted mask -> curved-surface area):
    ... --area_demo

输出:
    <out>/metrics.json        逐项指标均值
    <out>/samples_*.png       预测/真值叠加可视化（最多 8 张）
    <out>/area_demo.json      面积链路 demo 结果（若启用）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from wg_algorithms.seg_model import load_wound_model, predict_proba, HAS_TORCH


def _json_safe(obj):
    """把 inf/nan 转成 null。

    为什么要单独处理：Python 的 json.dump 默认会写出 `Infinity` / `NaN` 这种
    **非标准 JSON** 字面量，其他语言的解析器读到会直接报错。
    HD95 在出现完全漏检时会是 inf，所以这一步是必需的。
    """
    import math
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _auto_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if HAS_TORCH:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _save_overlay(path, rgb, pred, gt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(rgb); ax[0].set_title("RGB"); ax[0].axis("off")
    ax[1].imshow(rgb); ax[1].imshow(pred > 0, alpha=0.4, cmap="Reds")
    ax[1].set_title("Pred"); ax[1].axis("off")
    ax[2].imshow(rgb); ax[2].imshow(gt > 0, alpha=0.4, cmap="Greens")
    ax[2].set_title("GT"); ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def area_demo(model, device, img_size: int = 256, distance: float = 0.30, fx: float = 1500.0):
    """验证「真实模型掩膜 -> 三维曲面面积」端到端：合成一枚伤口，模型预测掩膜，
    在已知平面（距离 D）上用 SurfaceAreaIntegrator 计算真实面积，与 GT 掩膜面积对照。"""
    from wg_algorithms.segmentation import _keep_largest_cc
    from wg_algorithms.geometry import intrinsics_to_K
    from wg_algorithms.curvature_prior import PlaneModel
    from wg_algorithms.surface_area import SurfaceAreaIntegrator
    from wg_train import synthesize_data

    rgb, gt = synthesize_data.generate_pair(
        img_size=img_size,
        rng=np.random.default_rng(12345), wound_frac=0.20)
    proba = predict_proba(model, rgb, device, img_size)
    pred = (proba >= 0.5).astype(np.uint8) * 255
    pred = _keep_largest_cc(pred)

    K = intrinsics_to_K(fx, fx, img_size / 2, img_size / 2)
    M = np.eye(4)
    plane = PlaneModel([0, 0, distance], [0, 0, 1])
    integ = SurfaceAreaIntegrator()
    zeros = np.zeros((img_size, img_size, 3), np.uint8)
    a_pred = integ.integrate(zeros, pred, K, M, ("model", plane))
    a_gt = integ.integrate(zeros, gt, K, M, ("model", plane))
    return {
        "distance_m": distance,
        "pred_area_cm2": a_pred["area_cm2"],
        "gt_area_cm2": a_gt["area_cm2"],
        "pred_px": int((pred > 0).sum()),
        "gt_px": int((gt > 0).sum()),
        "rel_err_pct": abs(a_pred["area_cm2"] - a_gt["area_cm2"]) /
                       max(a_gt["area_cm2"], 1e-6) * 100.0,
    }


def _forward_proba(model, x, tta: bool = False):
    """前向得到概率图 [H,W]。

    tta=True 时做「水平+垂直翻转」三次前向取均值。
    为什么值得做：这不需要重新训练，代价只有 3× 推理时间，
    对边界抖动敏感的小目标（本项目创面中位只占画面 6.8%）通常能稳定涨一点 dice，
    而且面积是从掩膜像素积分来的，边界稳一点直接等于面积误差小一点。
    """
    import torch
    with torch.no_grad():
        p = model(x)[0, 0]
        if not tta:
            return p
        ps = [p]
        for dim in (3, 2):                       # 3=W 翻转, 2=H 翻转
            pf = model(torch.flip(x, [dim]))[0, 0]
            ps.append(torch.flip(pf, [dim - 2]))  # 翻回原朝向再平均
        return torch.stack(ps).mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="best.pt / last.pt 路径")
    ap.add_argument("--synthetic", type=int, default=0, help="合成验证集样本数（与 train 不同种子）")
    ap.add_argument("--image_dir", default=None)
    ap.add_argument("--mask_dir", default=None)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1000, help="合成验证集随机种子（避开训练 seed）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/exp1/eval")
    ap.add_argument("--area_demo", action="store_true", help="额外运行面积链路 demo")
    ap.add_argument("--no_largest_cc", action="store_true",
                    help="关闭「只保留最大连通域」后处理（多发病灶的验证集请开启）")
    ap.add_argument("--tta", action="store_true", help="翻转测试时增强（水平+垂直，概率图平均）")
    ap.add_argument("--thr", type=float, default=0.5, help="二值化阈值（默认 0.5）")
    ap.add_argument("--tune_thr", action="store_true",
                    help="在本集上搜索最优阈值 ⚠️ 只能在【验证集】上做，"
                         "在测试集上寻优等于把测试集信息泄漏进评估")
    args = ap.parse_args()

    if not HAS_TORCH:
        print("❌ 评估需要 torch；请先 `pip install torch`。")
        sys.exit(1)

    device = _auto_device(args.device)
    model = load_wound_model(args.model, device=device)
    print(f"▶ 评估模型: {args.model} | 设备: {device}")

    from torch.utils.data import DataLoader
    from wg_train.dataset import ImageMaskDataset, SyntheticSegDataset
    from wg_train.metrics import evaluate_pairs

    if args.synthetic > 0:
        ds = SyntheticSegDataset(n=args.synthetic, img_size=args.img_size,
                                 augment=False, seed=args.seed, wound_frac=0.20)
    else:
        ds = ImageMaskDataset(args.image_dir, args.mask_dir, img_size=args.img_size,
                              augment=False, seed=args.seed)
    ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    preds, gts = [], []
    os.makedirs(args.out, exist_ok=True)
    n_save = 0
    keep_cc = None
    if not args.no_largest_cc:
        from wg_algorithms.segmentation import _keep_largest_cc as keep_cc

    # 先收集概率图：阈值寻优需要同一批预测在不同阈值下重算，不能一边推理一边定阈值
    probas, gts_raw, rgbs = [], [], []
    for i, batch in enumerate(ld):
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        probas.append(_forward_proba(model, x, tta=args.tta).cpu().numpy())
        gts_raw.append((y[0, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255)
        if n_save < 8:
            rgbs.append((x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        n_save += 1

    def _thr_metrics(thr: float, cheap: bool = False):
        """cheap=True 只算 dice —— 阈值寻优要扫 19 个阈值，
        全套指标里的 HD95/boundary-F1 太慢，没必要在这里付这份代价。"""
        ps, gs = [], []
        for pb, gt in zip(probas, gts_raw):
            pred = (pb >= thr).astype(np.uint8) * 255
            if keep_cc is not None:
                pred = keep_cc(pred)
            ps.append(pred); gs.append(gt)
        if cheap:
            from wg_train.metrics import dice
            return ps, gs, {"dice": float(np.mean([dice(p, g) for p, g in zip(ps, gs)]))}
        return ps, gs, evaluate_pairs(ps, gs)

    thr = args.thr
    if args.tune_thr:
        best = (-1.0, args.thr, None)
        sweep = [round(0.05 * k, 2) for k in range(1, 20)]
        print("  阈值寻优中（本集，仅 dice 准则）…")
        for t in sweep:
            d = _thr_metrics(t, cheap=True)[2]["dice"]
            if d > best[0]:
                best = (d, t, None)
        thr = best[1]
        print(f"  最优阈值 {thr:.2f}（dice {best[0]:.4f}），"
              f"默认 0.5 对照：{_thr_metrics(0.5, cheap=True)[2]['dice']:.4f}")
        print("  ⚠️ 该阈值只在验证集上取，评估测试集时请用 --thr 显式传入")

    preds, gts, metrics = _thr_metrics(thr)
    metrics["threshold"] = thr
    metrics["tta"] = bool(args.tta)

    for i in range(min(8, len(rgbs))):
        _save_overlay(os.path.join(args.out, f"samples_{i:02d}.png"), rgbs[i], preds[i], gts[i])

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    print("\n" + "=" * 56)
    print("评估指标（验证集均值）")
    print("=" * 56)
    for k, v in metrics.items():
        print(f"  {k:>12}: {v}")
    print(f"\n✅ 指标已写入 {os.path.join(args.out, 'metrics.json')}")

    if args.area_demo:
        demo = area_demo(model, device, img_size=args.img_size)
        with open(os.path.join(args.out, "area_demo.json"), "w") as f:
            json.dump(demo, f, indent=2)
        print("\n" + "=" * 56)
        print("面积链路 demo（合成伤口 @ 平面距离 %.2f m）" % demo["distance_m"])
        print("=" * 56)
        print(f"  GT 掩膜面积   : {demo['gt_area_cm2']:.2f} cm²  (像素 {demo['gt_px']})")
        print(f"  模型预测面积 : {demo['pred_area_cm2']:.2f} cm²  (像素 {demo['pred_px']})")
        print(f"  相对误差     : {demo['rel_err_pct']:.2f}%")
        print(f"  → 真实模型掩膜已驱动三维面积计算；误差主要来自分割边界与假设平面。")
        print(f"✅ 面积 demo 已写入 {os.path.join(args.out, 'area_demo.json')}")


if __name__ == "__main__":
    main()
