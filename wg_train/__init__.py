"""
Training and evaluation code for the wound segmentation model (WoundUNet),
together with the SynthWound analytic-ground-truth benchmark generator.

  synth3d.py                benchmark renderer with analytically known area
  synthesize_data.py        synthetic RGB/mask pairs for smoke tests
  dataset.py                file-based and in-memory datasets
  metrics.py                Dice / IoU / precision / recall / HD95 / boundary F1
  train.py                  training CLI
  evaluate.py               evaluation CLI
  build_public_splits.py    hash deduplication, patient-level grouping, leakage check

Real-data experiments use third-party wound images that are not distributed
here; see DATA_LICENSES.md and docs/protocol.md for access and split rules.
"""
