#!/usr/bin/env python3
"""
train.py — 伤口分割模型训练 CLI
================================

Usage (synthetic data; runs without external datasets):
    python -m wg_train.train --synthetic 300 --epochs 25 --img_size 128 \
        --batch 8 --out runs/exp1

用法（真实数据，标注到位后）:
    python -m wg_train.train --image_dir data/images --mask_dir data/masks \
        --epochs 50 --out runs/exp_real

输出:
    <out>/best.pt        验证集 Dice 最优权重（含 config 元数据）
    <out>/last.pt        最后一轮权重
    <out>/train_log.json 逐轮 train/val 损失与 val-Dice
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from wg_algorithms.seg_model import WoundUNet, HAS_TORCH


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


def _dice_loss(proba, target, eps: float = 1e-6):
    p = proba.reshape(-1)
    t = target.reshape(-1)
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + eps) / (p.sum() + t.sum() + eps)


def _tversky_loss(proba, target, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-6):
    """Tversky 损失：alpha>beta 时偏向召回（减少小目标欠分割）。
    alpha=beta=0.5 即退化为 Dice。"""
    p = proba.reshape(-1)
    t = target.reshape(-1)
    tp = (p * t).sum()
    fp = (p * (1.0 - t)).sum()
    fn = ((1.0 - p) * t).sum()
    return 1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def _boundary_loss(proba, target, sigma: float = 5.0, k: int = 3, lam: float = 8.0):
    """边界感知损失（设备原生，纯张量运算，无 scipy CPU↔GPU 往返）：
    用形态学梯度（膨胀−腐蚀）得到 GT 掩膜轮廓，作为 BCE 的逐像素权重，
    使梯度集中在创面边界像素上，直接压低边界误差——
    对小创面（边界误差占面积比重大）增益最大。
    proba/target: (B,1,H,W) 张量。返回标量。
    """
    import torch
    import torch.nn.functional as F
    # max-pooling 实现膨胀/腐蚀（padding 保持尺寸）
    pad = k // 2
    dil = -F.max_pool2d(-target, k, stride=1, padding=pad)   # 膨胀
    ero = F.max_pool2d(target, k, stride=1, padding=pad)      # 腐蚀
    boundary = (dil - ero).clamp(min=0.0)                      # 轮廓指示 (B,1,H,W)
    w = 1.0 + lam * boundary                                    # 边界像素权重放大
    bce = F.binary_cross_entropy(proba, target, reduction="none")
    return (bce * w).mean() / w.mean().clamp(min=1.0)


def _make_loss(name: str, sigma: float, alpha: float, beta: float):
    if name == "bcedice":
        def f(p, t):
            import torch.nn as nn
            return nn.BCELoss()(p, t) + _dice_loss(p, t)
        return f
    if name == "boundary":
        def f(p, t):
            return _dice_loss(p, t) + _boundary_loss(p, t, sigma=sigma)
        return f
    if name == "tversky":
        def f(p, t):
            return _tversky_loss(p, t, alpha=alpha, beta=beta)
        return f
    raise ValueError(f"未知损失: {name}")


def build_loaders(args, device):
    import torch
    from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
    from wg_train.dataset import ImageMaskDataset, SyntheticSegDataset

    # ---- split.json 按「组」过滤 + 小创面加权采样（小创面专项）----
    if getattr(args, "split_json", None):
        import json
        sp = json.load(open(args.split_json))
        by_group = {}
        for s in sp["samples"]:
            by_group.setdefault(s["group"], []).append(s)
        tr_stems = [s["stem"] for s in by_group.get(args.train_group, [])]
        va_stems = [s["stem"] for s in by_group.get(args.val_group, [])]
        tr_ds = ImageMaskDataset(args.image_dir, args.mask_dir,
                                 img_size=args.img_size, augment=True,
                                 seed=args.seed, stems=tr_stems)
        va_ds = ImageMaskDataset(args.image_dir, args.mask_dir,
                                 img_size=args.img_size, augment=False,
                                 seed=args.seed, stems=va_stems)
        # 小创面过采样：fg_frac 低于阈值的样本权重放大 small_boost 倍
        sampler = None
        if args.small_boost and args.small_boost > 0:
            fg = {s["stem"]: s.get("fg_frac", 1.0) for s in by_group.get(args.train_group, [])}
            w = [args.small_boost if fg.get(st, 1.0) < args.small_frac else 1.0
                 for st in tr_stems]
            sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
            n_small = sum(1 for x in w if x > 1.0)
            print(f"   小创面过采样: {n_small}/{len(w)} 样本权重×{args.small_boost}")
        tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=(sampler is None),
                          sampler=sampler, num_workers=0)
        va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=0)
        print(f"   训练集 {len(tr_ds)} / 验证集 {len(va_ds)}（split.json 组划分，组内一致）")
        return tr_ld, va_ld

    # 显式 train/val 目录优先：由 build_public_splits 按「组」划分而来，
    # 可防止同一患者同时出现在训练集与验证集（patient-level leakage）。
    if getattr(args, "train_image_dir", None):
        tr_ds = ImageMaskDataset(args.train_image_dir, args.train_mask_dir,
                                 img_size=args.img_size, augment=True, seed=args.seed)
        va_ds = ImageMaskDataset(args.val_image_dir, args.val_mask_dir,
                                 img_size=args.img_size, augment=False, seed=args.seed)
        print(f"   训练集 {len(tr_ds)} / 验证集 {len(va_ds)}（外部划分，组内一致）")
        return (DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=0),
                DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=0))

    if args.synthetic and args.synthetic > 0:
        ds = SyntheticSegDataset(n=args.synthetic, img_size=args.img_size,
                                 augment=True, seed=args.seed, wound_frac=args.wound_frac)
    else:
        ds = ImageMaskDataset(args.image_dir, args.mask_dir, img_size=args.img_size,
                              augment=True, seed=args.seed)
    n_val = max(1, int(len(ds) * args.val_ratio))
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(args.seed) if HAS_TORCH else None
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    return train_ld, val_ld


def eval_val(model, val_ld, device):
    import torch
    from wg_train.metrics import dice
    model.eval()
    dices = []
    with torch.no_grad():
        for batch in val_ld:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            p = model(x)
            pred = (p >= 0.5).float()
            for i in range(x.size(0)):
                dices.append(dice(pred[i, 0].cpu().numpy(), y[i, 0].cpu().numpy()))
    return float(np.mean(dices)) if dices else 0.0


def main():
    ap = argparse.ArgumentParser()
    # 数据源（二选一）
    ap.add_argument("--synthetic", type=int, default=0,
                    help="合成样本数（>0 时启用合成数据集，无需真实标注）")
    ap.add_argument("--image_dir", default=None, help="真实 RGB 图目录")
    ap.add_argument("--mask_dir", default=None, help="真实掩膜目录")
    # 显式划分（推荐）：由 wg_train.build_public_splits 产出
    ap.add_argument("--train_image_dir", default=None, help="训练集图像目录")
    ap.add_argument("--train_mask_dir", default=None, help="训练集掩膜目录")
    ap.add_argument("--val_image_dir", default=None, help="验证集图像目录")
    ap.add_argument("--val_mask_dir", default=None, help="验证集掩膜目录")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    # 按 split.json 组划分（小创面专项：支持小创面过采样）
    ap.add_argument("--split_json", default=None,
                    help="split.json（含 samples[{stem,group,fg_frac}]），按 group 过滤训练/验证")
    ap.add_argument("--train_group", default="train")
    ap.add_argument("--val_group", default="val")
    # 损失函数（小创面专项）
    ap.add_argument("--loss", default="bcedice", choices=["bcedice", "boundary", "tversky"],
                    help="bcedice=基线; boundary=Dice+边界加权; tversky=偏向召回")
    ap.add_argument("--sigma", type=float, default=5.0, help="boundary 损失的距离图尺度")
    ap.add_argument("--alpha", type=float, default=0.7, help="tversky 假阴性权重")
    ap.add_argument("--beta", type=float, default=0.3, help="tversky 假阳性权重")
    # 小创面过采样
    ap.add_argument("--small_boost", type=float, default=0.0,
                    help="fg_frac<small_frac 的样本权重放大倍数（0=不过采样）")
    ap.add_argument("--small_frac", type=float, default=0.05,
                    help="判定为「小创面」的前景占比阈值")
    ap.add_argument("--wound_frac", type=float, default=0.20,
                    help="合成样本中创面占比（仅合成数据生效）")
    # 模型
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=128)
    # 训练
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init", default=None,
                    help="从已有 checkpoint 初始化权重（继续训练/微调），如 runs/exp1/best.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=0,
                    help="每 N 轮额外存一份 epN.pt（便于中途评估/挑选，0=不存）")
    # 运行
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/exp1")
    args = ap.parse_args()

    if not HAS_TORCH:
        print("❌ 训练需要 torch；请先 `pip install torch`。")
        sys.exit(1)
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    device = _auto_device(args.device)
    print(f"▶ 设备: {device} | 数据源: {'合成 x%d' % args.synthetic if args.synthetic else '真实'} "
          f"| img_size={args.img_size} | base={args.base} depth={args.depth}")

    os.makedirs(args.out, exist_ok=True)
    train_ld, val_ld = build_loaders(args, device)

    model = WoundUNet(in_ch=3, base=args.base, depth=args.depth).to(device)
    if args.init:
        # 从既有权重出发继续训练。用途：欠拟合时加轮次不必从头再来（省一半时间）；
        # 注意 out 换新目录，避免覆盖原 best.pt。
        ck = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print(f"  从 {args.init} 初始化 (该权重 val_dice={ck.get('val_dice', '?')})")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = _make_loss(args.loss, args.sigma, args.alpha, args.beta)

    log = []
    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        n_b = 0
        for batch in train_ld:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            p = model(x)
            loss = loss_fn(p, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += float(loss.item()) * x.size(0)
            n_b += x.size(0)
        train_loss = run_loss / max(n_b, 1)
        val_dice = eval_val(model, val_ld, device)
        log.append({"epoch": epoch, "train_loss": train_loss, "val_dice": val_dice})
        print(f"  epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                "state_dict": model.state_dict(),
                "config": {"in_ch": 3, "base": args.base, "depth": args.depth,
                           "img_size": args.img_size, "loss": args.loss,
                           "sigma": args.sigma, "alpha": args.alpha, "beta": args.beta,
                           "small_boost": args.small_boost, "small_frac": args.small_frac},
                "val_dice": val_dice, "epoch": epoch,
            }, os.path.join(args.out, "best.pt"))
        if args.save_every and epoch % args.save_every == 0:
            torch.save({
                "state_dict": model.state_dict(),
                "config": {"in_ch": 3, "base": args.base, "depth": args.depth,
                           "img_size": args.img_size, "loss": args.loss,
                           "sigma": args.sigma, "alpha": args.alpha, "beta": args.beta,
                           "small_boost": args.small_boost, "small_frac": args.small_frac},
                "val_dice": val_dice, "epoch": epoch,
            }, os.path.join(args.out, f"ep{epoch}.pt"))
    # 末轮也存
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"in_ch": 3, "base": args.base, "depth": args.depth,
                   "img_size": args.img_size},
        "val_dice": val_dice, "epoch": args.epochs,
    }, os.path.join(args.out, "last.pt"))
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n✅ 训练完成。最优 val_dice={best_dice:.4f}，权重: {os.path.join(args.out, 'best.pt')}")


if __name__ == "__main__":
    main()
