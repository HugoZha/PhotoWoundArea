"""
Core library for measuring two-dimensional wound surface area (cm2) from a
single photograph with a scale reference.

  scale_recovery.py    metric scale from AR camera poses
  curvature_prior.py   curvature prior for sparse depth
  segmentation.py      pluggable wound-mask interface
  seg_model.py         lightweight U-Net implementation
  surface_area.py      flat-mask to curved-surface area integration
  geometry.py          camera geometry helpers
  pipeline.py          end-to-end orchestration

Pure Python / NumPy reference implementation. A depth map, when available,
is used only to convert a flat mask area into the corresponding curved
surface area; this library does not reconstruct a 3D model or estimate volume.
"""
__version__ = "0.1.0"

from .geometry import (
    intrinsics_to_K,
    camera_transform_to_cam2world,
    cam2world_to_proj,
    pixel_to_ray,
    backproject,
    triangulate_pair,
)
from .scale_recovery import MetricReconstructor
from .curvature_prior import (
    BodyPartModel,
    PlaneModel,
    CylinderModel,
    SphereModel,
    fit_model_to_cloud,
)
from .segmentation import (
    BaseSegmenter,
    ColorThresholdSegmenter,
    SyntheticWoundSegmenter,
    WoundSegModel,
)
from .surface_area import SurfaceAreaIntegrator
from .pipeline import run_pipeline

__all__ = [
    "intrinsics_to_K",
    "camera_transform_to_cam2world",
    "cam2world_to_proj",
    "pixel_to_ray",
    "backproject",
    "triangulate_pair",
    "MetricReconstructor",
    "BodyPartModel",
    "PlaneModel",
    "CylinderModel",
    "SphereModel",
    "fit_model_to_cloud",
    "BaseSegmenter",
    "ColorThresholdSegmenter",
    "SyntheticWoundSegmenter",
    "WoundSegModel",
    "SurfaceAreaIntegrator",
    "run_pipeline",
]
