"""
synth3d.py — analytic-ground-truth wound benchmark generator
================================================================

思路：不去编造"看起来像伤口"的 2D 图，而是**模拟一次真实拍摄**：

    已知几何的体表曲面（平面/圆柱/球面）
      + 伤口按弧长参数贴在曲面上（形状、大小已知）
      + 相机以已知内参/位姿拍照（透视投影 + 曲面光照/遮挡）
    => RGB 图 + 像素级掩膜 + 逐像素深度 + **解析真值曲面面积（cm²）**

关键点：真值面积是「伤口在三维曲面上的面积」，不是 2D 像素数。
因此可以端到端评测「分割 -> 三维面积」的真实误差，而不只是 dice。

三条用法：
  1) 纯程序化：伤口/皮肤纹理由代码生成（零外部依赖）；
  2) 真实外观 + 合成几何：把公开数据集（FUSC/DFUC/Medetec 等）抠出的
     伤口 patch 贴到曲面上（--wound_patches DIR），皮肤仍程序化；
  3) 混合：两者按比例混合，提升外观多样性。

Quick run:
    python -m wg_train.synth3d --n 12 --img_size 512
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wg_algorithms.geometry import intrinsics_to_K            # noqa: E402
from wg_algorithms.curvature_prior import (                    # noqa: E402
    PlaneModel, CylinderModel, SphereModel, PART_PRESETS,
)

# 若干档肤色（Fitzpatrick 近似），提升多样性与公平性
SKIN_TONES = np.array([
    [255, 224, 196], [234, 192, 156], [206, 160, 122], [176, 128, 92],
    [140, 96, 66], [96, 62, 40],
], dtype=np.float32)

# 创面床基色：肉芽(红)/腐肉(黄)/坏死(黑)/渗液(亮粉)
WOUND_TONES = np.array([
    [214, 74, 62], [196, 104, 70], [232, 150, 90], [96, 62, 60],
    [168, 44, 52], [226, 118, 108],
], dtype=np.float32)


# ---------------------------------------------------------------- 射线
def camera_rays(K, W, H):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def intersect(dirs, model, origin=np.zeros(3)):
    """向量化射线-曲面求交，返回 (t (H,W), hit (H,W,3), ok (H,W))。"""
    H, W = dirs.shape[:2]
    t = np.full((H, W), np.nan)
    ok = np.zeros((H, W), bool)

    if model.surface_kind == "plane":
        n = model.normal
        denom = dirs @ n
        good = np.abs(denom) > 1e-9
        tt = np.where(good, np.dot(model.center - origin, n) / np.where(good, denom, 1), np.nan)
        ok = good & (tt > 1e-6)
        t = np.where(ok, tt, np.nan)

    elif model.surface_kind == "sphere":
        oc = origin - model.center
        b = dirs @ oc
        c = float(oc @ oc - model.radius ** 2)
        disc = b * b - c
        good = disc >= 0
        sq = np.sqrt(np.where(good, disc, 0))
        tt = -b - sq
        ok = good & (tt > 1e-6)
        t = np.where(ok, tt, np.nan)

    else:  # cylinder
        a = dirs[..., 0]
        e = np.array([1.0, 0.0, 0.0]) if abs(model.axis_dir[0]) > 0.9 else None
        # 通用：去掉轴向分量
        ax = model.axis_dir / (np.linalg.norm(model.axis_dir) + 1e-12)
        W0 = (origin - model.axis_point)
        W0p = W0 - np.dot(W0, ax) * ax
        Dp = dirs - (dirs @ ax)[..., None] * ax
        aa = np.sum(Dp * Dp, axis=-1)
        bb = Dp @ W0p
        cc = float(W0p @ W0p - model.radius ** 2)
        disc = bb * bb - aa * cc
        good = (aa > 1e-12) & (disc >= 0)
        sq = np.sqrt(np.where(good, disc, 0))
        tt = np.where(good, (-bb - sq) / np.where(good, aa, 1), np.nan)
        ok = good & (tt > 1e-6)
        t = np.where(ok, tt, np.nan)

    hit = origin + t[..., None] * dirs
    return t, hit, ok


def surface_params(hit, model):
    """曲面点 -> (s, x) 弧长参数（米）：s 沿圆周方向，x 沿轴向/子午方向。"""
    if model.surface_kind == "plane":
        # 平面上取两个正交切向
        n = model.normal
        e1 = np.cross(n, [1.0, 0, 0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(n, [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        rel = hit - model.center
        return rel @ e1, rel @ e2

    if model.surface_kind == "sphere":
        rel = hit - model.center
        n = rel / (np.linalg.norm(rel, axis=-1, keepdims=True) + 1e-12)
        # 以伤口中心方向为极点：s = 沿纬向弧长，x = 沿经向弧长
        pole = np.array([0.0, 0.0, -1.0])          # 朝相机一侧为极点
        e1 = np.cross(pole, [1.0, 0, 0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(pole, [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(pole, e1)
        lat = np.arcsin(np.clip(-(n @ pole), -1, 1))        # 到极点角距
        lon = np.arctan2(n @ e2, n @ e1)
        R = model.radius
        return R * lat * np.cos(lon), R * lat * np.sin(lon)

    # cylinder：x 沿轴，s = R * θ
    ax = model.axis_dir / (np.linalg.norm(model.axis_dir) + 1e-12)
    rel = hit - model.axis_point
    x = rel @ ax
    rad = rel - x[..., None] * ax
    # 以「朝相机的最近母线」为 s=0
    ref = np.array([0.0, 0.0, -1.0])
    ref = ref - np.dot(ref, ax) * ax
    ref /= np.linalg.norm(ref) + 1e-12
    e2 = np.cross(ax, ref)
    theta = np.arctan2(rad @ e2, rad @ ref)
    return model.radius * theta, x


def jacobian(s, x, model):
    """面积元 dA = J(s,x) ds dx。"""
    if model.surface_kind == "plane":
        return np.ones_like(s)
    if model.surface_kind == "cylinder":
        return np.ones_like(s)
    # 球面（测地极坐标近似）：J = sinc 形式
    R = model.radius
    r = np.sqrt(s ** 2 + x ** 2)
    return np.where(r < 1e-9, 1.0, np.sin(r / R) / (r / R + 1e-12))


# ---------------------------------------------------------------- 伤口形状
def wound_region(s, x, ra, rb, harmonics):
    """不规则伤口区域：扰动椭圆，返回 rho（<=1 为伤口内）。"""
    ang = np.arctan2(x / max(rb, 1e-9), s / max(ra, 1e-9))
    rr = np.sqrt((s / ra) ** 2 + (x / rb) ** 2)
    pert = np.zeros_like(rr)
    for k, amp, ph in harmonics:
        pert = pert + amp * np.sin(k * ang + ph)
    return rr * (1.0 + pert)


def gt_area(ra, rb, harmonics, model, c_s=0.0, c_x=0.0, n_grid=1400):
    """在参数域数值积分求真值曲面面积（m²）。"""
    ext_s = ra * 1.35
    ext_x = rb * 1.35
    ss = np.linspace(c_s - ext_s, c_s + ext_s, n_grid)
    xx = np.linspace(c_x - ext_x, c_x + ext_x, n_grid)
    S, X = np.meshgrid(ss, xx, indexing="ij")
    rho = wound_region(S - c_s, X - c_x, ra, rb, harmonics)
    inside = rho <= 1.0
    J = jacobian(S, X, model)
    ds = ss[1] - ss[0]
    dx = xx[1] - xx[0]
    return float((J * inside).sum() * ds * dx)


# ---------------------------------------------------------------- 纹理
def _low_freq_noise(H, W, rng, scale=4):
    n = rng.standard_normal((max(2, H // scale), max(2, W // scale))).astype(np.float32)
    from PIL import Image
    img = Image.fromarray(((n - n.min()) / (np.ptp(n) + 1e-6) * 255).astype(np.uint8))
    img = img.resize((W, H), Image.BICUBIC)
    return np.asarray(img, np.float32) / 255.0


def _shade(base, normals, light, ambient=0.55):
    lam = np.clip(normals @ light, 0, 1)[..., None]
    return np.clip(base * (ambient + (1 - ambient) * lam), 0, 255)


def surface_normals(hit, model):
    if model.surface_kind == "plane":
        return np.broadcast_to(model.normal, hit.shape).copy()
    if model.surface_kind == "sphere":
        rel = hit - model.center
        return rel / (np.linalg.norm(rel, axis=-1, keepdims=True) + 1e-12)
    ax = model.axis_dir / (np.linalg.norm(model.axis_dir) + 1e-12)
    rel = hit - model.axis_point
    x = (rel @ ax)[..., None]
    rad = rel - x * ax
    return rad / (np.linalg.norm(rad, axis=-1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------- 主类
class SurfaceSynth:
    """把伤口贴到已知几何曲面上渲染，产出 (RGB, 掩膜, 深度, 真值面积)。"""

    def __init__(self, img_size=512, fx=None, distance_m=0.30,
                 part_type="limb_forearm", seed=0, patch_dir=None):
        self.img_size = int(img_size)
        self.distance = float(distance_m)
        self.rng = np.random.default_rng(seed)
        self.patch_dir = patch_dir
        self._patches = None
        # Optional independent RNG for appearance-patch selection.  When None
        # (default) patch choice consumes the main RNG, preserving the exact
        # behaviour of all previously exported datasets.  Setting it makes the
        # geometry/camera random stream identical with and without patches,
        # which is required for a single-variable appearance ablation.
        self.patch_rng = None

        self.model_type, self.R = PART_PRESETS.get(part_type, ("cylinder", 0.04))
        self.part_type = part_type

        # 相机：位于原点看向 +Z，曲面最近点距相机 distance
        if self.model_type == "plane":
            self.model = PlaneModel([0, 0, self.distance], [0, 0, -1])
        elif self.model_type == "sphere":
            R = self.R
            self.model = SphereModel([0, 0, self.distance + R], R)
        else:
            R = self.R
            self.model = CylinderModel([0, 0, self.distance + R], [1, 0, 0], R)

        # 内参：按常见手机 FOV 反推 fx（60° 水平 FOV）
        self.fx = float(fx) if fx else (self.img_size * 0.5) / np.tan(np.deg2rad(30))
        self.K = intrinsics_to_K(self.fx, self.fx, self.img_size / 2, self.img_size / 2)

    # ---------- 真实伤口 patch（可选） ----------
    def _load_patches(self):
        if self._patches is not None or not self.patch_dir:
            return self._patches
        from PIL import Image
        out = []
        for fn in sorted(os.listdir(self.patch_dir)):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem = os.path.splitext(fn)[0]
            mpath = None
            for cand in (os.path.join(self.patch_dir, stem + "_mask.png"),
                         os.path.join(self.patch_dir, stem + ".mask.png")):
                if os.path.exists(cand):
                    mpath = cand
                    break
            if mpath is None:
                continue
            out.append((Image.open(os.path.join(self.patch_dir, fn)).convert("RGB"),
                        Image.open(mpath).convert("L")))
        self._patches = out
        return out

    def render(self, wound_area_cm2=None, irregularity=0.13, tone=None):
        H = W = self.img_size
        rays = camera_rays(self.K, W, H)
        t, hit, ok = intersect(rays, self.model)

        s, x = surface_params(hit, self.model)
        s = np.where(ok, s, np.nan)
        x = np.where(ok, x, np.nan)

        # 伤口尺寸：给定面积（cm²）反推半轴，长宽比随机
        if wound_area_cm2 is None:
            wound_area_cm2 = float(self.rng.uniform(1.0, 30.0))
        A = wound_area_cm2 * 1e-4                       # cm² -> m²
        ar = float(self.rng.uniform(1.15, 2.0))         # 长宽比
        ra = np.sqrt(A * ar / np.pi)                    # 半轴（米）
        rb = ra / ar
        harmonics = [(2, irregularity * 0.7, self.rng.uniform(0, 2 * np.pi)),
                     (3, irregularity * 0.45, self.rng.uniform(0, 2 * np.pi)),
                     (5, irregularity * 0.22, self.rng.uniform(0, 2 * np.pi))]

        rho = wound_region(s, x, ra, rb, harmonics)
        wmask = (rho <= 1.0) & ok

        # ---- 纹理 ----
        skin = SKIN_TONES[self.rng.integers(0, len(SKIN_TONES))]
        wtone = WOUND_TONES[self.rng.integers(0, len(WOUND_TONES))] if tone is None else tone
        n1 = _low_freq_noise(H, W, self.rng, scale=6)
        n2 = _low_freq_noise(H, W, self.rng, scale=2)

        base = np.broadcast_to(skin, (H, W, 3)).copy()
        base *= (0.92 + 0.16 * n1[..., None])
        base += (n2[..., None] - 0.5) * 8.0

        wbase = np.broadcast_to(wtone, (H, W, 3)).copy()
        wbase *= (0.80 + 0.40 * n2[..., None])
        wbase += (n1[..., None] - 0.5) * 14.0

        # 真实伤口 patch（若提供）：在伤口区域内按参数归一化坐标采样
        patches = self._load_patches()
        if patches:
            pim, pmk = patches[(self.patch_rng or self.rng).integers(0, len(patches))]
            pw = self.img_size
            pim_r = np.asarray(pim.resize((pw, pw)), np.float32)
            pmk_r = np.asarray(pmk.resize((pw, pw)), np.float32) > 127
            ys, xs = np.where(pmk_r)
            if len(xs) > 50 and wmask.any():
                # 把伤口 patch 拉伸贴合到当前伤口的参数域包围盒
                us = np.clip(((s - (-ra * 1.2)) / (2.4 * ra) * pw).astype(int), 0, pw - 1)
                ux = np.clip(((x - (-rb * 1.2)) / (2.4 * rb) * pw).astype(int), 0, pw - 1)
                sel = pmk_r[ux, us]
                wbase = np.where(sel[..., None], pim_r[ux, us], wbase)

        # 边界软过渡：以 rho=1 为中心（alpha=0.5），使"可见边缘"与真值区域一致
        edge_m = 0.0015
        eps = edge_m / max(ra, rb)
        alpha = np.clip((1.0 + eps - rho) / (2 * eps), 0, 1)
        alpha = np.where(ok, alpha, 0.0)[..., None]

        col = base * (1 - alpha) + wbase * alpha
        normals = surface_normals(hit, self.model)
        light = np.array([-0.35, 0.45, 0.82])
        light /= np.linalg.norm(light)
        col = _shade(col, np.nan_to_num(normals), light)
        # 伤口湿润高光
        spec = np.clip(np.nan_to_num(normals) @ light, 0, 1) ** 8
        col = col + (spec[..., None] * 40.0) * alpha

        rgb = np.clip(col, 0, 255).astype(np.uint8)
        rgb = np.where(ok[..., None], rgb, 0)

        # 掩膜严格取 rho<=1，与 gt_area 的数值积分区域一致（避免口径偏差）
        mask = wmask.astype(np.uint8) * 255

        depth = np.where(ok, np.linalg.norm(hit, axis=-1), np.nan)

        area_m2 = gt_area(ra, rb, harmonics, self.model)
        return {
            "rgb": rgb,
            "mask": mask,
            "depth": depth.astype(np.float32),
            "K": self.K,
            "M": np.eye(4),
            "gt_area_cm2": area_m2 * 1e4,
            "target_area_cm2": wound_area_cm2,
            "surface_kind": self.model.surface_kind,
            "part_type": self.part_type,
            "wound_semiaxis_m": (float(ra), float(rb)),
            "ok_ratio": float(ok.mean()),
        }


# ------------------------------------------------------- example entry point
def run_demo(n=12, img_size=512, seed=0, part_type="limb_forearm",
             patch_dir=None, out_dir=None, model_path=None, device="auto"):
    """生成 n 个样本，用真实算法链路估算面积，与真值对比。

    model_path: 若给出，则端到端使用训练好的 WoundSegModel；否则用色彩基线。
    """
    from wg_algorithms.surface_area import SurfaceAreaIntegrator
    from wg_algorithms.segmentation import ColorThresholdSegmenter
    from wg_train.metrics import dice

    synth = SurfaceSynth(img_size=img_size, part_type=part_type,
                         seed=seed, patch_dir=patch_dir)
    integ = SurfaceAreaIntegrator()
    if model_path:
        from wg_algorithms.segmentation import WoundSegModel
        seg = WoundSegModel(weights_path=model_path, device=device, img_size=img_size)
        seg_name = "WoundSegModel(训练)"
    else:
        seg = ColorThresholdSegmenter()
        seg_name = "ColorThreshold(基线)"

    rows = []
    for i in range(n):
        smp = synth.render()
        rgb, gt = smp["rgb"], smp["mask"]
        pred, _ = seg.segment(rgb)
        # 路径 A：LiDAR（用渲染出的深度，模拟有 LiDAR 设备）
        a_lidar = integ.integrate(rgb, gt, smp["K"], smp["M"],
                                  ("lidar", smp["depth"]))
        # 路径 B：曲率先验（无 LiDAR，射线与参数曲面求交）
        a_model = integ.integrate(rgb, gt, smp["K"], smp["M"],
                                  ("model", synth.model))
        # 路径 C：端到端（色彩基线分割出的掩膜 + 曲率先验）——最贴近真实使用
        a_pred = integ.integrate(rgb, pred, smp["K"], smp["M"],
                                 ("model", synth.model))
        g = smp["gt_area_cm2"]
        rows.append({
            "gt": g,
            "lidar": a_lidar["area_cm2"],
            "model": a_model["area_cm2"],
            "e2e": a_pred["area_cm2"],
            "dice": dice(pred, gt),
        })

    def _err(a, b):
        return abs(a - b) / max(b, 1e-6) * 100.0

    err_l = [_err(r["lidar"], r["gt"]) for r in rows]
    err_m = [_err(r["model"], r["gt"]) for r in rows]
    err_e = [_err(r["e2e"], r["gt"]) for r in rows]
    dc = [r["dice"] for r in rows]

    print(f"\nSynthWound sample | 部位={part_type} | 曲面={synth.model.surface_kind} "
          f"| R={synth.R*100:.1f}cm | 样本={n}")
    print("-" * 74)
    print(f"{'GT(cm²)':>9} {'LiDAR':>9} {'曲率先验':>9} {'端到端':>9} {'dice':>7}")
    for r in rows:
        print(f"{r['gt']:9.2f} {r['lidar']:9.2f} {r['model']:9.2f} "
              f"{r['e2e']:9.2f} {r['dice']:7.3f}")
    print("-" * 74)
    print(f"LiDAR 路径   相对误差 均值 {np.mean(err_l):6.2f}%  中位 {np.median(err_l):6.2f}%")
    print(f"曲率先验路径 相对误差 均值 {np.mean(err_m):6.2f}%  中位 {np.median(err_m):6.2f}%")
    print(f"端到端(含分割)相对误差 均值 {np.mean(err_e):6.2f}%  中位 {np.median(err_e):6.2f}%")
    print(f"{seg_name} 分割 dice 均值 {np.mean(dc):.3f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        from PIL import Image
        for i in range(min(6, n)):
            smp = synth.render()
            Image.fromarray(smp["rgb"]).save(os.path.join(out_dir, f"synth_{i:02d}.png"))
            Image.fromarray(smp["mask"]).save(os.path.join(out_dir, f"synth_{i:02d}_mask.png"))
        print(f"示例已保存: {out_dir}")
    return rows


def dump_dataset(out_dir, n=300, img_size=384, seed=0, patch_dir=None):
    """导出可直接喂给 wg_train.train 的数据集：images/ + masks/ + manifest.csv。

    部位按真实分布混合（前臂/大腿/骶尾平面/足跟球面），面积 1~30 cm²。
    manifest.csv 额外记录每例的**真值曲面面积**，供后续面积误差评测。
    """
    import csv
    from PIL import Image

    img_dir = os.path.join(out_dir, "images")
    msk_dir = os.path.join(out_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    parts = ["limb_forearm", "limb_forearm", "limb_thigh", "flat", "heel"]
    rows = []
    for i in range(n):
        pt = parts[i % len(parts)]
        synth = SurfaceSynth(img_size=img_size, part_type=pt,
                             seed=seed + i, patch_dir=patch_dir)
        smp = synth.render()
        name = f"s{i:05d}.png"
        Image.fromarray(smp["rgb"]).save(os.path.join(img_dir, name))
        Image.fromarray(smp["mask"]).save(os.path.join(msk_dir, name))
        rows.append({"file": name, "gt_area_cm2": round(smp["gt_area_cm2"], 3),
                     "part": pt, "surface": smp["surface_kind"]})

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "gt_area_cm2", "part", "surface"])
        w.writeheader()
        w.writerows(rows)
    print(f"数据集已导出: {out_dir}  (n={n}, img_size={img_size})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--part_type", default="limb_forearm")
    ap.add_argument("--patch_dir", default=None,
                    help="真实伤口 patch 目录（xxx.png + xxx_mask.png），可选")
    ap.add_argument("--out", default=None, help="示例图输出目录")
    ap.add_argument("--model", default=None, help="训练好的模型权重（端到端评测用）")
    ap.add_argument("--dump", default=None, help="导出数据集到该目录")
    ap.add_argument("--dump_n", type=int, default=300)
    args = ap.parse_args()

    if args.dump:
        dump_dataset(args.dump, n=args.dump_n, img_size=args.img_size,
                     patch_dir=args.patch_dir)
        return 0

    run_demo(n=args.n, img_size=args.img_size, part_type=args.part_type,
             patch_dir=args.patch_dir, out_dir=args.out, model_path=args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
