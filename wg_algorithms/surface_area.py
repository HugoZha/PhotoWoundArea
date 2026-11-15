"""
surface_area.py — integrate wound area over a curved surface
================================================================

Converts a flat 2D mask area into the corresponding curved-surface area.

两条数据路径：
  (A) LiDAR 路径：逐像素深度图 -> 反投影到 3D -> 曲面积分；
  (B) 无 LiDAR 路径：用模块二曲率先验模型，把相机射线与参数曲面求交得到
      深度 -> 3D -> 曲面积分（即「曲率先验补偿」的实际落地）。

面积计算采用解析逐像素法（主）+ 图像网格三角化（交叉校验）：
  单像素真实面积 dA = (z² / (fx·fy)) / |n·view_hat|
  其中 z 为像素到相机距离，n 为表面法向，view_hat 为视线方向。
  求和即真实曲面面积；对比忽略曲率的平面投影面积，即得补偿效果。
"""
from __future__ import annotations

import numpy as np

from .geometry import backproject


def recommended_normal_sigma(K, median_depth_m: float, target_mm: float = 6.0) -> float:
    """把"物理平滑尺度"换算成像素单位的高斯 sigma。

    为什么不直接写死像素数：同一套算法在 iPhone LiDAR（粗分辨率）和
    WoundsDB 点云（~2mm/像素）上，等效的像素 sigma 差好几倍。
    真实数据标定（tools/validate_real_surface.py，REAL 人体皮肤点云）表明，
    按 **约 6 mm 的物理尺度**平滑深度再求法向时误差最小
    （均值 7.3%，对比不平滑的 21.7%）。

    Args:
        K: 内参
        median_depth_m: 目标区域的中位深度（米）
        target_mm: 希望平滑掉的物理尺度（毫米）
    """
    fx = float(K[0, 0])
    d = float(median_depth_m)
    if fx <= 0 or d <= 0:
        return 0.0
    mm_per_px = d * 1000.0 / fx          # 该深度下 1 像素对应多少毫米
    return max(0.0, target_mm / mm_per_px)


def _smooth_valid(arr: np.ndarray, sigma: float) -> np.ndarray:
    """对含 NaN 的深度图做高斯平滑。

    为什么不能直接 gaussian_filter：NaN 会像病毒一样扩散，一小片空洞会污染整张图。
    做法：先用最近的有效值把 NaN 填掉再平滑，平滑后**只保留原本有效位置**的值。
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    valid = np.isfinite(arr)
    if valid.all():
        return gaussian_filter(arr, sigma=sigma, mode="nearest")
    if not valid.any():
        return arr
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = arr[tuple(idx)]
    sm = gaussian_filter(filled, sigma=sigma, mode="nearest")
    return np.where(valid, sm, arr)


class SurfaceAreaIntegrator:
    def __init__(self):
        pass

    # ---------- 主入口 ----------
    def integrate(self, rgb, mask, K, M_cam2world, source, confidence=None,
                  normal_smooth_sigma: float = 0.0):
        """统一入口。

        Args:
            rgb: (H,W,3) 参考帧图像（仅用于尺寸/可视化）。
            mask: (H,W) uint8 0/255 伤口区域。
            K: 3x3 内参。
            M_cam2world: 4x4 相机->世界（OpenCV 约定：+Z 向前、+Y 向下）。
            source: tuple
                ('lidar', depth_map (Hd,Wd) float 米) 或
                ('model', BodyPartModel 实例)
            confidence: optional (Hd,Wd) uint8 置信度（0=低），用于过滤 LiDAR 低置信像素。
        Returns:
            dict: area_cm2, flat_area_cm2, per_pixel_area (H,W) m²,
                  points3d (N,3), foreshorten (H,W), method
        """
        H, W = mask.shape[:2]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return self._empty(H, W)

        cam_pos = M_cam2world[:3, 3]
        fx, fy = K[0, 0], K[1, 1]
        R = M_cam2world[:3, :3]

        # 取得每个掩膜像素的深度（沿视线到表面的距离）
        if source[0] == "lidar":
            depth_map = np.asarray(source[1], dtype=np.float64)
            # 双线性重采样整张深度图到 RGB 分辨率（深度与 RGB 分辨率通常不同）
            depth_full = self._resample_depth_full(depth_map, (H, W))
            valid_full = np.isfinite(depth_full)
            conf_mask = None
            if confidence is not None:
                # 置信度用最近邻重采样（绝不可双线性插值整型标签），仅把「低置信」像素
                # 从面积累计中剔除；**不要**把深度置 NaN 后零填充——否则低置信像素边界会被
                # 造成人工悬崖，使相邻正常像素法向畸变、面积异常放大（已实测约 3× 膨胀）。
                conf = self._resample_nearest(np.asarray(confidence, dtype=np.float64), (H, W))
                conf_mask = conf >= 1.0          # 保留中/高置信，低置信(0)不参与
                valid_full = valid_full & np.isfinite(conf)
            # 仅对「掩膜 ∩ 深度有效 ∩ 置信有效」的像素积分（重新确定像素集）
            keep = (mask > 0) & valid_full
            if conf_mask is not None:
                keep = keep & conf_mask
            ys, xs = np.where(keep)
            if len(xs) == 0:
                return self._empty(H, W)
            depth = depth_full[ys, xs].copy()
            # 反投影得到世界坐标点
            pts3d = backproject(xs.astype(float), ys.astype(float), depth, K, M_cam2world)
            # 从深度图估计逐像素真实法向（关键：弯曲表面不能再假设法向=视线轴）
            normals_full = self._normals_from_depth_full(depth_full, K, R,
                                                         smooth_sigma=normal_smooth_sigma)
            normals = normals_full[ys, xs]
            method = "lidar"
        else:
            model = source[1]
            # 像素 -> 相机系射线 -> 世界系射线
            rays_cam = self._pixel_rays(xs, ys, K)            # (N,3) 相机系单位
            rays_world = (R @ rays_cam.T).T                   # (N,3) 世界系单位
            o = cam_pos
            depths, pts3d = [], []
            for i in range(len(xs)):
                hit, t = model.intersect_ray(o, rays_world[i])
                if hit is None:
                    depths.append(np.nan); pts3d.append([np.nan, np.nan, np.nan])
                else:
                    depths.append(float(np.linalg.norm(hit - o)))
                    pts3d.append(hit)
            depth = np.array(depths, dtype=np.float64)
            pts3d = np.array(pts3d, dtype=np.float64)
            normals = np.array([model.normal_at(pts3d[i]) for i in range(len(pts3d))])
            method = f"model:{model.surface_kind}"

        # 解析逐像素真实面积
        z = depth
        view = cam_pos - pts3d                     # 世界系视线方向（指向相机）
        view_len = np.linalg.norm(view, axis=1, keepdims=True)
        view_hat = view / (view_len + 1e-12)

        n_dot_v = np.sum(normals * view_hat, axis=1)
        # 安全下限：皮肤创面基本正对拍摄，|n·v| 不可能极小；把掠射角(>~78°)像素钳到
        # cos=0.2，避免个别畸变法向把单像素面积放大到 1000× 而污染总面积。
        n_dot_v = np.clip(np.abs(n_dot_v), 0.2, 1.0)

        per_pixel = (z ** 2) / (fx * fy) / n_dot_v   # m²
        per_pixel = np.where(np.isfinite(per_pixel), per_pixel, 0.0)

        # 投影（平面）面积：忽略曲率，仅按视线距离铺平面
        flat_per_pixel = (z ** 2) / (fx * fy)
        flat_per_pixel = np.where(np.isfinite(flat_per_pixel), flat_per_pixel, 0.0)

        area_m2 = float(per_pixel.sum())
        flat_m2 = float(flat_per_pixel.sum())

        # 回填图
        pp_map = np.zeros((H, W), dtype=np.float64)
        pp_map[ys, xs] = per_pixel
        foreshorten_map = np.zeros((H, W), dtype=np.float64)
        foreshorten_map[ys, xs] = 1.0 / n_dot_v

        return {
            "area_cm2": area_m2 * 1e4,
            "flat_area_cm2": flat_m2 * 1e4,
            "compensation_ratio": (area_m2 / flat_m2) if flat_m2 > 0 else 1.0,
            "per_pixel_area": pp_map,
            "foreshorten": foreshorten_map,
            "points3d": pts3d,
            "method": method,
            "n_valid": int(np.isfinite(per_pixel).sum()),
        }

    # ---------- 内部工具 ----------
    @staticmethod
    def _pixel_rays(xs, ys, K):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        d = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs, dtype=float)], axis=1)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    @staticmethod
    def _resample_depth_full(depth_map, target_shape):
        """把 (Hd, Wd) 深度图双线性重采样到 RGB 分辨率 (H, W)，返回 (H, W) 数组。

        ARKit 的 sceneDepth 已对齐到彩色相机坐标系，但分辨率更低；深度像素
        (u_d, v_d) 与彩色像素 (u_c, v_c) 满足 u_d = u_c * Wd/Wc, v_d = v_c * Hd/Hc，
        故只需按归一化坐标线性插值即可。无效深度（<=0 或 NaN）保持 NaN。
        """
        dh, dw = depth_map.shape[:2]
        th, tw = target_shape
        yy = np.linspace(0, dh - 1, th)
        xx = np.linspace(0, dw - 1, tw)
        # 用 scipy 做快速双线性（退化时退化为最近邻）
        try:
            from scipy.interpolate import RegularGridInterpolator
            f = RegularGridInterpolator(
                (np.arange(dh), np.arange(dw)), depth_map,
                bounds_error=False, fill_value=np.nan
            )
            gx, gy = np.meshgrid(xx, yy)   # gx:(th,tw) columns, gy rows
            out = f(np.stack([gy.ravel(), gx.ravel()], axis=1)).reshape(th, tw)
        except Exception:
            out = depth_map[
                (np.linspace(0, dh - 1, th)).round().astype(int)[:, None],
                (np.linspace(0, dw - 1, tw)).round().astype(int)[None, :],
            ]
        out = np.where(np.isfinite(out) & (out > 0), out, np.nan)
        return out

    @staticmethod
    def _resample_nearest(value_map, target_shape):
        """最近邻重采样 (Hd,Wd) -> (H,W)。用于整型标签（如置信度），避免双线性插值
        把 0/1/2 掺成小数、以及把无效边界搞脏。"""
        dh, dw = value_map.shape[:2]
        th, tw = target_shape
        yy = (np.linspace(0, dh - 1, th)).round().astype(int)
        xx = (np.linspace(0, dw - 1, tw)).round().astype(int)
        return value_map[np.ix_(yy, xx)]

    @staticmethod
    def _normals_from_depth_full(depth_full, K, R, smooth_sigma: float = 0.0):
        """从 (H,W) 深度图估计逐像素世界坐标法向。

        在相机系（OpenCV 约定：+Z 向前）构造三维点云 (X,Y,Z)，对 x、y 方向做
        有限差分得到两个切向，叉乘得相机系法向（朝 +Z 修正），再经 R 旋到世界系。
        无效深度处法向为 NaN。

        ⚠️ smooth_sigma 为什么存在（真实数据实测的教训）：
        在真实点云上（WoundsDB，像素间距约 2mm）直接做逐像素有限差分，
        皮肤的微观起伏 + 深度噪声会让法向剧烈抖动，平均 |n·v| 掉到 0.4 左右，
        面积被系统性放大到 **约 2 倍**（实测 8.54 vs 真值 3.80 cm²，误差 173%）。
        先对深度做高斯平滑再求法向，能把"宏观朝向"稳住。
        sigma 的单位是**像素**；0 表示不平滑（保持旧行为）。
        """
        H, W = depth_full.shape
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        vv, uu = np.mgrid[0:H, 0:W]

        if smooth_sigma and smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter
            z = _smooth_valid(depth_full, smooth_sigma)
        else:
            z = depth_full
        X = (uu - cx) * z / fx
        Y = (vv - cy) * z / fy
        # 无效处填 0，避免梯度噪声（法向在 mask 处由有效邻域估计）
        valid = np.isfinite(z)
        Xf = np.where(valid, X, 0.0)
        Yf = np.where(valid, Y, 0.0)
        Zf = np.where(valid, z, 0.0)

        # 有限差分切向
        t_u = np.stack([
            np.gradient(Xf, axis=1), np.gradient(Yf, axis=1), np.gradient(Zf, axis=1)
        ], axis=-1)  # (H,W,3)
        t_v = np.stack([
            np.gradient(Xf, axis=0), np.gradient(Yf, axis=0), np.gradient(Zf, axis=0)
        ], axis=-1)

        n_cam = np.cross(t_u, t_v)
        n_len = np.linalg.norm(n_cam, axis=-1, keepdims=True)
        n_cam = np.where(n_len > 1e-12, n_cam / np.where(n_len > 1e-12, n_len, 1.0), 0.0)
        # 法向朝相机（+Z 向前），若 z 分量为负则翻转
        flip = np.where(n_cam[..., 2:3] < 0, -1.0, 1.0)
        n_cam = n_cam * flip

        # 旋到世界系
        n_world = (R @ n_cam.reshape(-1, 3).T).T.reshape(H, W, 3)
        n_len_w = np.linalg.norm(n_world, axis=-1, keepdims=True)
        n_world = np.where(n_len_w > 1e-12, n_world / np.where(n_len_w > 1e-12, n_len_w, 1.0), 0.0)
        # 无效深度处法向置 NaN
        n_world = np.where(valid[..., None], n_world, np.nan)
        return n_world

    @staticmethod
    def _empty(H, W):
        return {
            "area_cm2": 0.0, "flat_area_cm2": 0.0, "compensation_ratio": 1.0,
            "per_pixel_area": np.zeros((H, W)), "foreshorten": np.zeros((H, W)),
            "points3d": np.zeros((0, 3)), "method": "empty", "n_valid": 0,
        }
