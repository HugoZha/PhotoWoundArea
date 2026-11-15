# Evaluation protocol

This file defines every metric used in the accompanying manuscript. The rules
below are the ones that turned out to matter in practice; breaking any of them
silently changes the reported numbers.

## 1. Scope of the measurement

The task is **two-dimensional wound surface area (cm²) from one photograph with
a scale reference**. A depth map, when present, converts a flat mask area into
the corresponding *curved* surface area. There is no 3D reconstruction, no
volume and no tissue classification. Every reported result gives the relative
error (%), the absolute error (cm²) and a split by wound size together. The
1 mm geometric error of a 3D reconstruction method and the 10% area error of
this method are different quantities and are never compared directly.

## 2. Ground truth

SynthWound attaches a wound to a surface whose geometry is known analytically
and renders it with a known camera. The true curved-surface area is computed
from the surface parameterisation,

```
A_GT = integral over Omega of sqrt(det(J^T J)) du dv
```

where `J` is the surface Jacobian. The exported mask uses the same boundary
convention as the analytic truth (`rho <= 1`). Mixing conventions (for example
using a soft edge centred on `rho = 1` for truth and a hard threshold for the
mask) biases area low by about 5%. Always check this before reporting.

## 3. Metrics

| Metric | Definition |
|---|---|
| Relative area error | `abs(pred_cm2 - gt_cm2) / gt_cm2 x 100%`, per scene |
| **Median error** | Median of the per-scene relative errors over the cohort. Not a mean, not an average over configurations. |
| **Geometric floor** | The error that remains when the predicted mask is replaced by the expert mask. It isolates the geometry, depth sampling and numerical-integration part of the chain from the segmentation part. |
| Two floors | On an analytically known surface, numerical integration alone gives **0.62%**. Running the complete pipeline (mask-to-mesh alignment included) gives a conservative **1.69-1.73%**. These are different quantities; keep them separate. |
| Dice | 2D overlap between predicted and expert wound mask in the image plane. |
| Size split | Wounds below 18.7 cm² versus at or above it, on the held-out cohort. |
| Confidence proxy | `score = sqrt(P x IoU)`, with `P = clamp((p_bar - 0.5)/0.45, 0, 1)` and IoU computed between masks at thresholds 0.4 and 0.6. Used only for frame ranking/rejection, never as an accuracy claim. |
| Three reporting conventions | (i) flat planar estimate, (ii) depth-based curved-surface integration, (iii) whole-image control. Report all three when comparing a real session. |

## 4. Non-negotiable splitting rule

**Any case that took part in training must be excluded from evaluation.**
WoundsDB cases are grouped at patient level and split 32 / 7 / 8 for training /
validation / test in the segmentation crop set. Only the 12 test scenes from six
unseen patients with a valid registered point cloud, at least 15 matched image
features and a usable depth map enter the primary endpoint. Every real-data
number is produced with:

```bash
python tools/e2e_woundsdb.py --only_group test
```

Mixing training cases into evaluation inflates the headline result. The
baseline and the final system are therefore always evaluated on the same
unseen test cases, and all real-data numbers are produced with
`--only_group test`.

## 5. Training / inference framing

The deployed checkpoint was trained on 320 x 320 wound-centred crops, where the
wound fills roughly 7% of the image. Sending a whole photograph resized to
256 x 256 makes the wound occupy 0.03%-0.21% of the frame and the network
returns almost no response (mean Dice 0.201, six of 14 frames completely
missed). Inference therefore performs a **multi-scale crop search**: a coarse
whole-image peak, the image centre and five offsets, each at four box sizes,
scored by maximum predicted probability. This restores mean Dice to 0.691 with
no complete misses. If you retrain with a different crop protocol, change the
inference search to match it.

Two related traps:

* A confidence score based on whole-image pixel statistics saturates on white
  dressings (median probability pinned at 0.500, mask flooding the crop box).
  Frames with `medianProb >= 0.5` must be rejected, not measured.
* In the offline path, the loaded depth map must be bound to
  `inputs.depth / depthW / depthH`. Loading it but not passing it on silently
  falls back to a 0.35 m plane estimate and inflates area by
  `(0.35 / d_real)^2` (about 4.5x at a 16.5 cm working distance).

## 6. Surface normal smoothing

Normals estimated by per-pixel finite differences on a real skin point cloud
(~2 mm between neighbouring points) double the apparent area. Smooth the depth
map before differencing at a **physical** width of 6 mm:

```
sigma_pixels = f * Delta / z,   Delta = 6 mm
```

where `f` is the focal length in pixels and `z` the working distance. A fixed
pixel count breaks down on other devices and at other distances. On real point
clouds this step cuts area error from 21.7% to 7.3% (R = 0.978). See
`wg_algorithms/surface_area.py` (`recommended_normal_sigma`) and
`tools/validate_real_surface.py`.

## 7. Annotation convention is a sensitivity band, not a correction

Experts differ in where they draw the wound edge. Widening the expert mask by
one mesh pixel (2.16 mm) moves the signed bias by 1.45% and changes
small-wound error by 1.95 percentage points. Publish this as a band
(`tools/calibrate_gt_caliber.py`); **never** modify expert masks to improve a
primary metric.

## 8. Reproducing the pipeline

```bash
python -m wg_train.synth3d --dump data/synth3d --dump_n 320 --img_size 384
python -m wg_train.build_public_splits          # hash dedup + patient-level grouping + leakage check
python tools/e2e_woundsdb.py --only_group test
python tools/analyze_area_vs_dice.py
python tools/appearance_ablation_v08.py
python tools/ablation_summary.py
python tools/calibrate_gt_caliber.py
pytest -q
```

`tools/e2e_woundsdb.py` and some other drivers expect the App capture format
(`frame_NNNN_rgb.png`, `frame_NNNN_depth.raw`, `frame_NNNN.json`) and a
`wg_ar_loader` adapter from the mobile project. The underscore in the filenames
is part of the format; `frame_NNNN.rgb.png` is a different, unreadable name.
