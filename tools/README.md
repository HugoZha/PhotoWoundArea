# Experiment drivers

All paths below are relative to the repository root. Data directories, model
weights and result JSONs are git-ignored; see `DATA_LICENSES.md` and
`docs/protocol.md`.

| Script | Purpose |
|---|---|
| `fetch_public_datasets.py` | Download and normalise public wound segmentation sets into `data/`. Handles codeload quirks, Medetec's (1,1,1) foreground and coloured-label black regions. |
| `wg_train/build_public_splits.py` | Content-hash deduplication, patient/case-level grouping and a leakage check before writing split lists. |
| `build_woundsdb_seg.py` | Build WoundsDB segmentation crops (mask + photo + point cloud) with the patient-level split. |
| `build_mix.py` | Mix public and WoundsDB crops with the domain weighting used in the final checkpoint. |
| `repair_split_zip.py` | Recover the WoundsDB multi-volume zip64 archive whose central-directory offsets do not match the real layout: sequential local-header scan (`--force-scan`). |
| `e2e_woundsdb.py` | End-to-end area measurement on WoundsDB. **Always use `--only_group test`.** |
| `analyze_area_vs_dice.py` | Dice-to-area-error attribution: correlation, Dice bands and the expert-mask geometric floor. |
| `appearance_ablation_v08.py` | Controlled single-variable ablation: same checkpoint, same 60 scenes, only the wound's colour/texture changed. |
| `ablation_summary.py` | Aggregate configurations A-F of the component ablation on the same 11 scenes. |
| `calibrate_gt_caliber.py` | Sensitivity band for how the expert drew the wound boundary. Never used to modify a primary metric. |
| `validate_real_surface.py` | Calibrate the 6 mm physical normal-smoothing width against real skin point clouds. |

## App-side integration

`e2e_woundsdb.py` and `validate_real_surface.py` may import `wg_ar_loader`, an
adapter from the mobile capture project that reads frames named
`frame_NNNN_rgb.png`, `frame_NNNN_depth.raw` and `frame_NNNN.json`. If you only
want the synthetic benchmark and the unit tests, that adapter is not needed:

```bash
pytest -q tests/test_core.py tests/test_synth3d.py
```
