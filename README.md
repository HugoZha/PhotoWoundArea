# PhotoWoundArea

**An analytic-ground-truth benchmark and an error-attribution framework for measuring wound area from a single photograph.**

---

## What this project is (and is not)

Chronic-wound care records **wound surface area in cm²** at every visit. When that area is read from a phone photograph, there is usually no independent physical reference: the "truth" is another estimate made from another image. This repository provides the missing reference and the tools to find out *where* an area error actually comes from.

* **SynthWound** attaches a wound to a surface whose geometry is known, renders it with a known camera, and computes the **true curved-surface area analytically** —
  `A_GT^analytic = ∬_Ω sqrt(det(JᵀJ)) du dv`. Each rendered scene carries the colour image, the mask, the depth map, the camera intrinsics and that ground-truth area; the exported dataset stores images, masks and a manifest of true areas.
* An **attribution protocol** relates segmentation Dice to area error, measures the *geometric floor* (the error left when the predicted mask is replaced by the expert mask), splits results by wound size, and sweeps how the expert drew the boundary.
* A compact mobile measurement pipeline (WoundUNet + multi-scale crop search + physically scaled normal smoothing + quality score + frame rejection) is included as an **end-to-end validation carrier**, not as the main contribution.

**Scope.** This is **two-dimensional wound surface area** measured from one photograph with a scale reference. A depth map is used only to convert a flat mask area into the corresponding *curved* surface area. This is **not** 3D reconstruction, volume estimation, or tissue classification. Relative error, absolute cm² and a size split are always reported together.

---

## Main results

| Setting | Cohort | Median relative error | Median absolute error | Dice | Geometric floor |
|---|---|---:|---:|---:|---:|
| SynthWound self-consistency | 60 synthetic scenes | 1.50% | — | 0.978 | 0.62% (integration only) |
| Controlled appearance ablation (appearance only changed) | same 60 scenes | 3.25% → 12.27% | — | 0.975 → 0.906 | 1.73% (both arms) |
| Real WoundsDB, patient-level hold-out | 12 scenes / 6 unseen patients | 12.62% | 4.30 cm² | 0.859 | 1.69–1.73% |
| └ small wounds (< 18.7 cm²) | 6 | 26.86% | 4.07 cm² | 0.807 | — |
| └ large wounds (≥ 18.7 cm²) | 6 | 6.32% | 4.75 cm² | 0.924 | — |

The central finding: **area accuracy is set by the quality of the two-dimensional outline, not by the surface geometry.** With a perfect mask the geometry-and-integration chain still leaves about 1.7%; above that floor, Dice determines the error. A 1.45 mm difference in where an expert draws the boundary moves small-wound error by tens of percentage points, so the paper publishes that sensitivity as a band and never modifies the expert masks for a primary metric.

**Note.** The 1.50% synthetic figure is a **methodological self-consistency check, not clinical accuracy**. The 12.62% figure on patient-held-out real data is the primary endpoint.

---

## Repository layout

```
wg_algorithms/     Core library: geometry, scale recovery, curvature prior,
                   segmentation, surface-area integration, end-to-end pipeline
wg_train/          SynthWound renderer (synth3d.py), dataset, training,
                   evaluation, metrics, patient-level split builder
tests/             Unit tests for the plane / curvature / LiDAR-depth paths
tools/             Reproducible experiment drivers (see tools/README.md)
docs/protocol.md   Evaluation protocol and the definition of every metric
docs/results.md    Numbers behind Tables 1-5, with their source artefacts
requirements.txt   Python dependencies
DATA_LICENSES.md   Why no dataset images are distributed here
CITATION.cff       How to cite this work
LICENSE            MIT licence for the code
```

---

## Quick start

```bash
git clone https://github.com/HugoZha/PhotoWoundArea.git
cd PhotoWoundArea
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # core geometry / surface-area unit tests
```

Generate a small synthetic benchmark with an analytic ground-truth area:

```bash
python -m wg_train.synth3d --dump data/synth3d --dump_n 320 --img_size 384
```

`data/synth3d/` then contains `images/`, `masks/` and `manifest.csv`; each manifest row gives the file name, the analytic ground-truth curved-surface area (`gt_area_cm2`), the body part and the surface type. `--patch_dir` composites real wound patches onto the synthetic surface; geometry and true area are unchanged, only appearance changes. **That option requires public wound images that you obtain yourself** — see `DATA_LICENSES.md`.

Run the end-to-end measurement and the attribution analyses after preparing data:

```bash
python tools/e2e_woundsdb.py --only_group test
python tools/analyze_area_vs_dice.py
python tools/appearance_ablation_v08.py
python tools/ablation_summary.py
```

> **Data are not included.** WoundsDB, FUSeg, AZH, Medetec and the wound-tissue atlas are third-party resources with their own terms and access procedures. Trained checkpoints and per-scene result JSONs are likewise not committed; the scripts regenerate them. Some drivers expect the App-side capture format (`frame_NNNN_rgb.png`, `frame_NNNN_depth.raw`, `frame_NNNN.json`) and a `wg_ar_loader` adapter supplied by the mobile project.

---

## Reproducibility rules encoded in the code

1. **Never mix training cases into evaluation.** WoundsDB cases are split 32 / 7 / 8 (train / validation / test) at patient level; 12 scenes from 6 test patients form the primary cohort. Every reported real-data number uses `--only_group test`.
2. **Training and inference framing must match.** The model is trained on 320 × 320 wound-centred crops where the wound fills about 7% of the image. Feeding a whole photograph resized to 256 × 256 drops Dice to 0.201 with 6 of 14 frames missed; a multi-scale crop search restores 0.691 with no complete misses.
3. **Smoothing width is a physical length, not a pixel count.** `σ = fΔ/z` with `Δ = 6 mm`; a fixed pixel width breaks down at other working distances. On real skin point clouds this reduces area error from 21.7% to 7.3%.
4. **Depth must actually reach the integrator.** A loaded-but-unbound depth map silently degrades to a 0.35 m plane estimate and inflates area by `(0.35/d_real)²`.
5. **Mask and analytic truth share one boundary** (`ρ ≤ 1`). Different thresholds underestimate area by about 5%.
6. **Annotation convention is reported, never learned away.** Widening the expert mask is a sensitivity band, not a correction; the primary metrics always use the unmodified expert outlines.

---

## Relation to the paper

This repository accompanies:

> Lu, G., Zhang, C., Zha, Y. (2026). *SynthWound: An Analytic Benchmark and Attribution Framework for Photographic Wound Area Measurement.* Submitted to *Medical Image Analysis*.

The manuscript, figures and submission documents are **not** part of this repository. The code here is the reference implementation of the benchmark, the attribution protocol and the measurement pipeline.

---

## Licence and acknowledgements

Code: MIT (see `LICENSE`). Third-party datasets keep their own licences (see `DATA_LICENSES.md`).

We thank Wenjie Han for valuable advice on the theoretical idea behind this study and on the design of its technical route.
