#!/usr/bin/env python
"""Sensitivity of area error to the expert outline convention.

WoundsDB provides a single expert outline per image. Where that outline is
drawn shifts the signed area bias, so this script reports a sensitivity band
rather than changing the reference masks:

  (1) dilate the expert mask boundary in mesh space by radius r, recompute
      the curved-surface area, and track the signed error as a function of r;
  (2) find the reference radius r* that brings the median signed bias near
      zero, then re-evaluate the baseline at that one fixed convention;
  (3) inspect individual cases: a case needing a much larger r* than the
      cohort suggests an annotation outlier rather than a model error.

Outputs: caliber_sensitivity.json, caliber_curve.png and a before/after table.
The primary results always use the unmodified expert masks.
"""
import os, sys, json, glob, argparse
import numpy as np
from PIL import Image
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from e2e_woundsdb import (
    predict_mask, mask_photo_to_mesh, smooth_xyz, photo_to_stereo,
    read_ply_grid_any, align_pinhole_fast, recommended_normal_sigma,
    intrinsics_to_K, nominal_spacing, triangle_area_over_region,
    PAT_ROOT, RES_ROOT, MESH_H, MESH_W,
)
from wg_algorithms.seg_model import load_wound_model

THR_SMALL = 18.7  # cm² stratification threshold


def dilate_mesh(mask_bool, r):
    if r <= 0:
        return mask_bool
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask_bool.astype(np.uint8), k).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/wdb_ms60/best.pt")
    ap.add_argument("--only_group", default="test")
    ap.add_argument("--split_file", default="data/woundsdb_seg_ms/split.json")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--radii", default="0,2,4,6,8,10,12,16,20")
    ap.add_argument("--out", default="runs/caliber_scan")
    args = ap.parse_args()
    radii = [int(x) for x in args.radii.split(",")]
    model = load_wound_model(os.path.join(ROOT, args.model), device=args.device)

    cases = []
    for mp in sorted(glob.glob(os.path.join(PAT_ROOT, "case_*", "day_*",
                                            "data", "scene_*", "mask.png"))):
        d = os.path.dirname(mp)
        if not os.path.exists(os.path.join(d, "stereo_l.png")):
            continue
        rel = os.path.relpath(d, PAT_ROOT)
        res = os.path.join(RES_ROOT, rel.replace("data", "results", 1), "stereo-mesh.ply")
        if os.path.exists(res):
            cases.append((d, res, rel))
    if args.only_group:
        sp = json.load(open(os.path.join(ROOT, args.split_file)))
        c2g = {s["case"]: s["group"] for s in sp["samples"]}
        cases = [c for c in cases if c2g.get(c[2].split(os.sep)[0]) == args.only_group]
    print(f"▶ {len(cases)} pairs | model {args.model} | group {args.only_group}")
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)

    import time as _t
    rec = {}
    for idx, (pdir, ply, rel) in enumerate(cases):
        t0 = _t.time()
        try:
            photo = np.asarray(Image.open(os.path.join(pdir, "photo.jpg")).convert("RGB"))
            mask_gt = (np.asarray(Image.open(os.path.join(pdir, "mask.png"))
                                  .convert("L")) > 127).astype(np.uint8) * 255
            stereo = np.asarray(Image.open(os.path.join(pdir, "stereo_l.png")).convert("RGB"))
            left = stereo[:, : stereo.shape[1] // 2]
            M, inl, _ = photo_to_stereo(photo, left)
            if M is None or inl < 15:
                print(f"  SKIP {rel} sift inl={inl}"); continue
            keep_gt = mask_photo_to_mesh(mask_gt, M)
            if keep_gt.sum() < 150:
                print(f"  SKIP {rel} gt<150"); continue
            pred = predict_mask(model, args.device, photo, mode="roi", thr=0.5,
                                mask_roi_hint=mask_gt)
            keep_pred = mask_photo_to_mesh(pred, M)
            xyz_mm, _ = read_ply_grid_any(ply, (MESH_H, MESH_W))
            _, perm, Kc = align_pinhole_fast(xyz_mm)
            xyz_mm = xyz_mm[:, :, list(perm)]
            fx, fy, cx, cy = Kc
            depth_m = xyz_mm[:, :, 2] / 1000.0
            sigma_px = recommended_normal_sigma(
                intrinsics_to_K(fx, fy, cx, cy), float(np.nanmedian(depth_m)), target_mm=6.0)
            xyz_s = smooth_xyz(xyz_mm, sigma_px)
            spacing_mm = nominal_spacing(xyz_mm)   # real physical spacing corresponding to 1 mesh px (mm)
            gap = spacing_mm * 4.0
            pred_mm2, _ = triangle_area_over_region(xyz_s, keep_pred, max_gap_mm=gap)
            gt0_mm2, _ = triangle_area_over_region(xyz_s, keep_gt, max_gap_mm=gap)
            per_r = {}
            for r in radii:
                kg = dilate_mesh(keep_gt, r)
                gmm2, covg = triangle_area_over_region(xyz_s, kg, max_gap_mm=gap)
                cov_thr = 0.8 if r == 0 else 0.5   # r=0 baseline strictly matches e2e (cov>=0.8)
                if gmm2 <= 0 or covg < cov_thr:
                    continue
                signed = (pred_mm2 - gmm2) / gmm2 * 100.0
                per_r[r] = {"signed": float(signed), "abs": float(abs(signed)),
                            "ar": float(pred_mm2 / gmm2), "gt_cm2": float(gmm2 / 100.0)}
            rec[rel] = {"pred_cm2": float(pred_mm2 / 100.0),
                        "gt0_cm2": float(gt0_mm2 / 100.0),
                        "spacing_mm": float(spacing_mm),
                        "per_r": per_r}
            print(f"  [{idx+1}/{len(cases)}] {rel} pred={pred_mm2/100:.2f} "
                  f"gt0={gt0_mm2/100:.2f} r scanned {len(per_r)} bins ({_t.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  ERR {rel}: {e}")

    # ---- fixed cohort: keep only samples with a valid r=0 baseline (cov>=0.8, exactly
    # matching e2e). Otherwise relaxing the coverage threshold for r>0 would admit 3-4 more
    # cases, so the points on the curve would not be the same set of samples — not publishable.
    cohort = [rel for rel in rec if 0 in rec[rel]["per_r"]]
    print(f"▶ fixed cohort = {len(cohort)} scenes (r=0 valid; cov>=0.8, matches e2e)")

    # ---- aggregate by r, and stratify into small/large/case_8 (all restricted to the same cohort)
    def collect(subset):
        out = {}
        for r in radii:
            sg, ab = [], []
            for rel in cohort:
                d = rec[rel]
                case = rel.split(os.sep)[0]
                if subset == "small" and d["gt0_cm2"] >= THR_SMALL:
                    continue
                if subset == "large" and d["gt0_cm2"] < THR_SMALL:
                    continue
                if subset == "case_8" and case != "case_8":
                    continue
                if r in d["per_r"]:
                    sg.append(d["per_r"][r]["signed"])
                    ab.append(d["per_r"][r]["abs"])
            if sg:
                out[r] = {"signed_med": float(np.median(sg)),
                          "abs_med": float(np.median(ab)),
                          "lt10": float(np.mean([a < 10 for a in ab]) * 100),
                          "n": len(sg)}
        return out

    summ = {k: collect(k) for k in ["overall", "small", "large", "case_8"]}

    # ---- find r*: the interpolated radius where overall signed_med crosses zero
    rstar = None
    ov = summ["overall"]
    rs = sorted(ov.keys())
    for i in range(len(rs) - 1):
        a, b = ov[rs[i]]["signed_med"], ov[rs[i + 1]]["signed_med"]
        if a <= 0 <= b or b <= 0 <= a:
            rstar = rs[i] + (0 - a) * (rs[i + 1] - rs[i]) / (b - a)
            break
    if rstar is None:  # if it does not cross zero, take the bin closest to zero
        rstar = min(rs, key=lambda r: abs(ov[r]["signed_med"]))

    # ---- re-evaluate at a unified r* (linearly interpolate signed/abs at r* for each rel)
    def interp_at(rstar):
        res = {}
        for rel, d in rec.items():
            pr = d["per_r"]
            ks = sorted(pr.keys())
            if not ks:
                continue
            if rstar <= ks[0]:
                v = pr[ks[0]]
            elif rstar >= ks[-1]:
                v = pr[ks[-1]]
            else:
                for i in range(len(ks) - 1):
                    if ks[i] <= rstar <= ks[i + 1]:
                        t = (rstar - ks[i]) / (ks[i + 1] - ks[i])
                        s = pr[ks[i]]["signed"] + t * (pr[ks[i + 1]]["signed"] - pr[ks[i]]["signed"])
                        break
                v = {"signed": s, "abs": abs(s)}
            res[rel] = v
        return res

    calib = interp_at(rstar)
    def stats(subset):
        sg, ab = [], []
        for rel in cohort:
            d = rec[rel]
            case = rel.split(os.sep)[0]
            if subset == "small" and d["gt0_cm2"] >= THR_SMALL:
                continue
            if subset == "large" and d["gt0_cm2"] < THR_SMALL:
                continue
            if subset == "case_8" and case != "case_8":
                continue
            if rel in calib:
                sg.append(calib[rel]["signed"])
                ab.append(calib[rel]["abs"])
        if not ab:
            return None
        return {"signed_med": float(np.median(sg)), "abs_med": float(np.median(ab)),
                "lt10": float(np.mean([a < 10 for a in ab]) * 100), "n": len(ab)}

    before = {k: summ[k].get(0) for k in summ}          # r=0 before calibration
    after = {k: stats(k) for k in ["overall", "small", "large", "case_8"]}

    # ---- physical scale conversion: real mm spacing of 1 mesh px (from the point cloud, not assumed)
    sp = np.array([rec[r]["spacing_mm"] for r in cohort], float)

    def zero_cross(rd):
        ks = sorted(rd)
        for i in range(len(ks) - 1):
            a, b = rd[ks[i]]["signed_med"], rd[ks[i + 1]]["signed_med"]
            if a <= 0 <= b or b <= 0 <= a:
                return ks[i] + (0 - a) * (ks[i + 1] - ks[i]) / (b - a)
        return None

    r8 = zero_cross(summ["case_8"])
    mm_px = float(np.median(sp))
    physical = {
        "mm_per_mesh_px_median": mm_px,
        "mm_per_mesh_px_p25": float(np.percentile(sp, 25)),
        "mm_per_mesh_px_p75": float(np.percentile(sp, 75)),
        "mm_per_mesh_px_range": [float(sp.min()), float(sp.max())],
        "rstar_px": float(rstar),
        "rstar_mm": float(rstar * mm_px),
        "case8_zero_px": (float(r8) if r8 is not None else None),
        "case8_zero_mm": (float(r8 * mm_px) if r8 is not None else None),
        "per_scene_spacing_mm": {r: rec[r]["spacing_mm"] for r in cohort},
    }

    out = {
        "model": args.model, "only_group": args.only_group,
        "radii": radii, "rstar": float(rstar),
        "fixed_cohort_n": len(cohort), "cohort": cohort,
        "summary_by_r": summ,
        "calibrated_before_r0": before,
        "calibrated_after_rstar": after,
        "physical_scale": physical,
        "per_case8": {r: rec[r]["per_r"] for r in rec if r.split(os.sep)[0] == "case_8"},
    }
    json.dump(out, open(os.path.join(ROOT, args.out, "caliber_sensitivity.json"), "w"),
              indent=2, ensure_ascii=False)

    # ---- plot the curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"overall": "#1d9e75", "small": "#378add", "large": "#ba7517", "case_8": "#e24b4a"}
    for k in ["overall", "small", "case_8"]:
        rsx = sorted(summ[k].keys())
        ax1.plot(rsx, [summ[k][r]["signed_med"] for r in rsx], "o-", color=colors[k], label=k)
        ax2.plot(rsx, [summ[k][r]["abs_med"] for r in rsx], "o-", color=colors[k], label=k)
    ax1.axhline(0, color="#888", lw=1)
    ax1.axvline(rstar, color="#444", ls="--", lw=1)
    ax1.set_title(f"Signed area bias vs GT boundary dilation (r*={rstar:.2f}px)")
    ax1.set_xlabel("GT boundary dilation (mesh px)"); ax1.set_ylabel("Signed area bias (%)")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_title("Absolute area error vs GT boundary dilation")
    ax2.set_xlabel("GT boundary dilation (mesh px)"); ax2.set_ylabel("Absolute area error (%)")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, args.out, "caliber_curve.png"), dpi=120)
    plt.close(fig)

    print("\n=== Ground-truth caliber calibration results (fixed cohort) ===")
    print(f"r* (convention that zeroes the overall bias) = {rstar:.2f} px = {rstar*mm_px:.2f} mm "
          f"(1 mesh px = {mm_px:.2f} mm, IQR {physical['mm_per_mesh_px_p25']:.2f}-"
          f"{physical['mm_per_mesh_px_p75']:.2f})")
    if r8 is not None:
        print(f"case_8 zero-bias convention = {r8:.2f} px = {r8*mm_px:.2f} mm")
    print(f"{'subset':8s} {'abs% before':>10s} {'abs% after':>10s} {'<10% before':>10s} {'<10% after':>10s}")
    for k in ["overall", "small", "large", "case_8"]:
        b, a = before[k], after[k]
        print(f"{k:8s} {b['abs_med']:10.1f} {a['abs_med']:10.1f} "
              f"{b['lt10']:10.0f} {a['lt10']:10.0f}")
    print(f"\ndetails -> {os.path.join(ROOT, args.out, 'caliber_sensitivity.json')}")
    print(f"curve -> {os.path.join(ROOT, args.out, 'caliber_curve.png')}")


if __name__ == "__main__":
    main()
