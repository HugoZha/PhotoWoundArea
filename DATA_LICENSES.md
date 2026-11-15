# Data and licence notes

**No third-party wound images, masks, point clouds, trained weights or per-scene
result files are distributed in this repository.** Everything here is code,
documentation and small configuration files. The datasets must be obtained
independently from their owners, subject to their own terms.

| Resource | Used for | Status / licence note |
|---|---|---|
| WoundsDB (Kręcichwost et al., 2021) | Real photographs, expert masks, registered point clouds; patient-level hold-out evaluation | Requires registration at <https://chronicwounddatabase.eu>. Distribution terms must be checked before any data release. |
| FUSeg / FUSeg2021 (Wang et al., 2024) | Public wound masks for segmentation training | CC BY-ND 4.0. The ND term prohibits redistribution of derived/adapted datasets, so this repository ships **generation scripts and split lists only**, never re-cropped or re-formatted images. |
| AZH (Anisuzzaman et al., 2022) | Public wound masks for segmentation training | Terms must be confirmed by the user before redistribution; not bundled here. |
| Medetec (Thomas, 2010) | Public wound images | Terms must be confirmed by the user before redistribution; not bundled here. |
| Wound tissue atlas / DFUC2022 / EBIS | Explored as additional domains or as manually licensed resources | Several require a signed data-use agreement; none are bundled here. |

## What the scripts regenerate

`wg_train/build_public_splits.py` performs content-hash deduplication and
patient/case-level grouping (so no case appears in two splits), writes split
lists, and runs a leakage check. `tools/fetch_public_datasets.py` downloads and
normalises the public sets into a local `data/` directory that is ignored by
Git. `tools/build_woundsdb_seg.py` and `tools/build_mix.py` build the training
crops from those local copies.

## Known preprocessing traps that affect reproducibility

* Several public sets contain letterbox-style black padding (one set has a
  71-pixel black strip on the right). Train and inference must use **identical
  cropping**, otherwise Dice collapses from ~0.78 to ~0.58.
* One Medetec release stores foreground as RGB (1,1,1), not white; a
  `>127` threshold silently produces empty masks.
* Coloured-label sets contain unlabelled black regions that need a
  multi-background-colour distance rule rather than a luminance threshold.
* WoundsDB PLY point clouds store coordinates in **(y, x, z)** order, and the
  stereo mesh is 1024 rows x 768 columns while the depth mesh is 176 x 144.

## If you publish a derivative

Check each licence yourself before sharing any derived dataset. For citation of
this code, see `CITATION.cff`. Code in this repository is MIT licensed; the
third-party data are not. The manuscript text and figures are not distributed
with this repository.
