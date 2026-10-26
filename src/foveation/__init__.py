"""RoadSentinel Active Foveation Engine Package."""

from src.foveation.types import (
    Detection, CropProposal, CropResult, FoveationTiming, FoveationTrace,
    FoveationConfig, CLASS_NAMES, VRU_CLASS_NAMES
)
from src.foveation.uncertainty_signals import (
    UncertaintySignalEvaluator, propose_fixed_grid_crops, propose_random_crops
)
from src.foveation.cropper import ActiveCropper
from src.foveation.box_fusion import BoxFusionEngine
from src.foveation.engine import ActiveFoveationEngine
from src.foveation.reference_pipeline import ReferenceFullFramePipeline

__all__ = [
    "Detection",
    "CropProposal",
    "CropResult",
    "FoveationTiming",
    "FoveationTrace",
    "FoveationConfig",
    "CLASS_NAMES",
    "VRU_CLASS_NAMES",
    "UncertaintySignalEvaluator",
    "propose_fixed_grid_crops",
    "propose_random_crops",
    "ActiveCropper",
    "BoxFusionEngine",
    "ActiveFoveationEngine",
    "ReferenceFullFramePipeline",
]
