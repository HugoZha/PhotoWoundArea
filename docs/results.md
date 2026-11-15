# Where the numbers come from

The tables in the manuscript are built from local experiment artefacts that are
**not committed** to this repository (they contain third-party data, model
weights or large per-scene outputs). This file maps each published number to the
script that regenerates it and the local artefact it reads.

| Manuscript item | Headline numbers | Regenerating script | Local artefact (git-ignored) |
|---|---|---|---|
| Table 1, SynthWound self-consistency (n = 60) | Dice 0.978; median 1.50%, mean 2.58%; colour-threshold baseline 0.101 / 3055.6%; numerical-integration floor 0.62%; depth-map path 2.03% | `wg_train/synth3d.py`, `wg_algorithms/surface_area.py` | `data/synth3d/`, `runs/synth3d/best.pt` |
| Table 2 / Section 4.2, attribution | SynthWound r = -0.82, floor 1.69%; real appearance r = -0.947, floor 1.73%, median 12.27%; WoundsDB r = -0.861, 5.76 / 26.86 / 42.39% by Dice band | `tools/analyze_area_vs_dice.py`, `tools/appearance_ablation_v08.py` | `runs/area_attribution.json`, `runs/appearance_ablation_v08.json` |
| Section 4.2, controlled appearance ablation | same checkpoint, same 60 scenes; Dice 0.975 -> 0.906; median error 3.25% -> 12.27% (+9.03 pp); floor 1.73% in both arms (difference 0.00 pp) | `tools/appearance_ablation_v08.py` | `runs/appearance_ablation_v08.json` |
| Table 3, component ablation (N = 11) | A 0.166 / 73.5%; B 0.166 / 77.6%; C 0.868 / 9.2%; D 0.868 / 7.6% / R 0.994; E 0.863 / 8.1%; F 0.848 / 10.3% | `tools/ablation_summary.py` | `runs/ablation_summary.json`, `runs/abl_A_*` ... `runs/abl_F_*` |
| Table 4, real end-to-end (N = 12) | median 12.62%, mean 18.8%, 4.30 cm², Dice 0.859, R = 0.992, 50% within 10%, signed bias +6.33% (9/12 high); small 26.86% / 4.07 cm² / Dice 0.807; large 6.32% / 4.75 cm² / Dice 0.924 | `tools/e2e_woundsdb.py --only_group test` | `runs/e2e_woundsdb_ms60/results.json`, `data/woundsdb_seg_ms/split.json` |
| Table 5, drawing-rule sensitivity | 1 mesh pixel = 2.16 mm; 1.45% signed-bias shift; 1.95 pp small-wound error shift | `tools/calibrate_gt_caliber.py` | `runs/caliber_scan/` |
| Supplementary Table S1, public 2D segmentation | 0.782 with black borders / HD95 12.5 px; 0.658 cropped on 302 held-out images; 0.435 cross-source atlas; flip averaging +0.007 | `wg_train/evaluate.py`, `wg_train/build_public_splits.py` | `runs/publicA/`, `runs/publicA_cropped/`, `data/public_splits*/split_report.json` |
| Real-surface normal smoothing | 21.7% -> 7.3% at 6 mm, R = 0.978 | `tools/validate_real_surface.py` | WoundsDB `results/` point clouds |
| Mobile framing search | mean Dice 0.201 -> 0.691; complete misses 6/14 -> 0/14 | App-side `predictBestROI`, offline driver `tools/e2e_woundsdb.py` | captured frame sets |

## Cohort bookkeeping

* WoundsDB cases: 32 / 7 / 8 (train / validation / test), patient-level.
* Primary cohort: 12 test scenes from six unseen patients, each with a valid
  registered point cloud, >= 15 matched image features and a usable depth map.
* Component ablation: the same 11 scenes in all six configurations, so the
  comparison is paired.
* Size split: < 18.7 cm² (n = 6) versus >= 18.7 cm² (n = 6).

## What is a claim and what is not

* **1.50%** is a methodological self-consistency check on synthetic data
  rendered by the same generator used for training. It is **not** clinical
  accuracy.
* **12.62%** on patient-held-out real data is the primary endpoint.
* The synthetic-to-real progression **1.50% -> 12.27% -> 12.62%** is the
  benchmark's prediction, not three interchangeable accuracy figures: the
  12.27% arm comes from a separate 60-scene attribution cohort.
* The expert-mask floor (about 1.7%) is the part of the error that a better
  segmenter cannot remove. Above it, the two-dimensional outline quality sets
  the area accuracy.
