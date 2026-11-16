#!/usr/bin/env python3
"""
train.py — wound segmentation model training CLI
================================

Usage (synthetic data; runs without external datasets):
    python -m wg_train.train --synthetic 300 --epochs 25 --img_size 128 \
        --batch 8 --out runs/exp1

Usage (real data, once annotations are ready):
    python -m wg_train.train --image_dir data/images --mask_dir data/masks \
        --epochs 50 --out runs/exp_real

Output:
    <out>/best.pt        best validation-Dice weights (includes config metadata)
    <out>/last.pt        last-epoch weights
    <out>/train_log.json per-epoch train/val loss and val-Dice
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
    """Tversky loss: when alpha>beta it favors recall (reduces under-segmentation of small targets).
    alpha=beta=0.5 degenerates to Dice."""
    p = proba.reshape(-1)
    t = target.reshape(-1)
    tp = (p * t).sum()
    fp = (p * (1.0 - t)).sum()
    fn = ((1.0 - p) * t).sum()
    return 1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def _boundary_loss(proba, target, sigma: float = 5.0, k: int = 3, lam: float = 8.0):
    """Boundary-aware loss (device-native, pure tensor ops, no scipy CPU<->GPU round trip):
    obtain the GT mask contour via a morphological gradient (dilation - erosion) and use it as
    per-pixel weights for BCE, concentrating gradients on wound-boundary pixels and directly
    reducing boundary error —
    the largest gain is for small wounds (where boundary error accounts for a large share of the area).
    proba/target: (B,1,H,W) tensors. Returns a scalar.
    """
    import torch
    import torch.nn.functional as F
    # max-pooling implements dilation/erosion (padding preserves size)
    pad = k // 2
    dil = -F.max_pool2d(-target, k, stride=1, padding=pad)   # dilation
    ero = F.max_pool2d(target, k, stride=1, padding=pad)      # erosion
    boundary = (dil - ero).clamp(min=0.0)                      # contour indicator (B,1,H,W)
    w = 1.0 + lam * boundary                                    # boundary-pixel weights amplified
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
    raise ValueError(f"Unknown loss: {name}")


def build_loaders(args, device):
    import torch
    from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
    from wg_train.dataset import ImageMaskDataset, SyntheticSegDataset

    # ---- split.json group filtering + small-wound weighted sampling (small-wound track) ----
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
        # Small-wound oversampling: samples with fg_frac below the threshold get their weight multiplied by small_boost
        sampler = None
        if args.small_boost and args.small_boost > 0:
            fg = {s["stem"]: s.get("fg_frac", 1.0) for s in by_group.get(args.train_group, [])}
            w = [args.small_boost if fg.get(st, 1.0) < args.small_frac else 1.0
                 for st in tr_stems]
            sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
            n_small = sum(1 for x in w if x > 1.0)
            print(f"   small-wound oversampling: {n_small}/{len(w)} samples, weight ×{args.small_boost}")
        tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=(sampler is None),
                          sampler=sampler, num_workers=0)
        va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=0)
        print(f"   train set {len(tr_ds)} / val set {len(va_ds)} (split.json group split, consistent within groups)")
        return tr_ld, va_ld

    # Explicit train/val directories take priority: produced by build_public_splits with group-aware splitting,
    # which prevents the same patient from appearing in both training and validation (patient-level leakage).
    if getattr(args, "train_image_dir", None):
        tr_ds = ImageMaskDataset(args.train_image_dir, args.train_mask_dir,
                                 img_size=args.img_size, augment=True, seed=args.seed)
        va_ds = ImageMaskDataset(args.val_image_dir, args.val_mask_dir,
                                 img_size=args.img_size, augment=False, seed=args.seed)
        print(f"   train set {len(tr_ds)} / val set {len(va_ds)} (external split, consistent within groups)")
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
    # Data source (choose one)
    ap.add_argument("--synthetic", type=int, default=0,
                    help="Number of synthetic samples (>0 enables the synthetic dataset, no real annotations needed)")
    ap.add_argument("--image_dir", default=None, help="Directory of real RGB images")
    ap.add_argument("--mask_dir", default=None, help="Directory of real masks")
    # Explicit split (recommended): produced by wg_train.build_public_splits
    ap.add_argument("--train_image_dir", default=None, help="Training-set image directory")
    ap.add_argument("--train_mask_dir", default=None, help="Training-set mask directory")
    ap.add_argument("--val_image_dir", default=None, help="Validation-set image directory")
    ap.add_argument("--val_mask_dir", default=None, help="Validation-set mask directory")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    # Group split via split.json (small-wound track: supports small-wound oversampling)
    ap.add_argument("--split_json", default=None,
                    help="split.json (contains samples[{stem,group,fg_frac}]); filter train/val by group")
    ap.add_argument("--train_group", default="train")
    ap.add_argument("--val_group", default="val")
    # Loss function (small-wound track)
    ap.add_argument("--loss", default="bcedice", choices=["bcedice", "boundary", "tversky"],
                    help="bcedice=baseline; boundary=Dice+boundary weighting; tversky=recall-oriented")
    ap.add_argument("--sigma", type=float, default=5.0, help="Distance-map scale for the boundary loss")
    ap.add_argument("--alpha", type=float, default=0.7, help="tversky false-negative weight")
    ap.add_argument("--beta", type=float, default=0.3, help="tversky false-positive weight")
    # Small-wound oversampling
    ap.add_argument("--small_boost", type=float, default=0.0,
                    help="Weight multiplier for samples with fg_frac<small_frac (0=no oversampling)")
    ap.add_argument("--small_frac", type=float, default=0.05,
                    help="Foreground-fraction threshold for classifying a sample as a 'small wound'")
    ap.add_argument("--wound_frac", type=float, default=0.20,
                    help="Wound fraction in synthetic samples (synthetic data only)")
    # Model
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=128)
    # Training
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init", default=None,
                    help="Initialize weights from an existing checkpoint (continue training/fine-tune), e.g. runs/exp1/best.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=0,
                    help="Also save an epN.pt every N epochs (for intermediate evaluation/selection; 0=do not save)")
    # Runtime
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/exp1")
    args = ap.parse_args()

    if not HAS_TORCH:
        print("❌ Training requires torch; please run `pip install torch` first.")
        sys.exit(1)
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    device = _auto_device(args.device)
    print(f"▶ device: {device} | data source: {'synthetic x%d' % args.synthetic if args.synthetic else 'real'} "
          f"| img_size={args.img_size} | base={args.base} depth={args.depth}")

    os.makedirs(args.out, exist_ok=True)
    train_ld, val_ld = build_loaders(args, device)

    model = WoundUNet(in_ch=3, base=args.base, depth=args.depth).to(device)
    if args.init:
        # Continue training from existing weights. Use case: when underfitting, add epochs without starting over (saves about half the time);
        # note: use a new out directory to avoid overwriting the original best.pt.
        ck = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print(f"  initialized from {args.init} (these weights val_dice={ck.get('val_dice', '?')})")
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
    # Also save the final epoch
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"in_ch": 3, "base": args.base, "depth": args.depth,
                   "img_size": args.img_size},
        "val_dice": val_dice, "epoch": args.epochs,
    }, os.path.join(args.out, "last.pt"))
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n✅ Training complete. Best val_dice={best_dice:.4f}, weights: {os.path.join(args.out, 'best.pt')}")


if __name__ == "__main__":
    main()
