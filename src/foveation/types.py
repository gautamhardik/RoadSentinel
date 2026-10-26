"""
RoadSentinel Phase 3 — Active Foveation Engine: Data Structures & Interfaces
Defines immutable, fully-typed dataclasses for detections, crop proposals,
crop results, execution traces, and engine configurations.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


# Verified BDD100K class mappings
CLASS_NAMES: List[str] = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]
VRU_CLASS_NAMES: List[str] = ["person", "rider", "bike", "motor"]


@dataclass
class Detection:
    """Represents a single bounding box prediction in original image coordinates."""
    box: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    class_id: int                           # 0..9
    class_name: str                         # 'person', 'rider', etc.
    confidence: float                       # [0.0, 1.0]
    source: str = "full_frame"              # 'full_frame', 'crop', 'fused'
    crop_id: Optional[int] = None           # Index of crop that produced this detection
    trigger_reason: Optional[str] = None    # Uncertainty trigger if applicable
    uncertainty_score: Optional[float] = None

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_vru(self) -> bool:
        return self.class_name in VRU_CLASS_NAMES

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": [round(float(v), 2) for v in self.box],
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "confidence": round(float(self.confidence), 4),
            "source": str(self.source),
            "crop_id": self.crop_id,
            "trigger_reason": self.trigger_reason,
            "uncertainty_score": round(float(self.uncertainty_score), 4) if self.uncertainty_score is not None else None,
            "area": round(float(self.area), 1),
            "is_vru": bool(self.is_vru),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Detection":
        return cls(
            box=(float(d["box"][0]), float(d["box"][1]), float(d["box"][2]), float(d["box"][3])),
            class_id=int(d["class_id"]),
            class_name=str(d["class_name"]),
            confidence=float(d["confidence"]),
            source=str(d.get("source", "full_frame")),
            crop_id=d.get("crop_id"),
            trigger_reason=d.get("trigger_reason"),
            uncertainty_score=d.get("uncertainty_score"),
        )


@dataclass
class CropProposal:
    """Defines a proposed crop window on the original frame."""
    crop_id: int
    crop_box: Tuple[float, float, float, float]  # [x1, y1, x2, y2] on original image
    trigger_reason: str                          # 'low_confidence', 'small_vru', 'cluster', 'boundary', 'fixed', 'random'
    trigger_score: float                         # Normalized urgency score [0.0, 1.0]
    source_detection_idx: Optional[int] = None
    context_margin: float = 0.25

    @property
    def width(self) -> float:
        return max(0.0, self.crop_box[2] - self.crop_box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.crop_box[3] - self.crop_box[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crop_id": int(self.crop_id),
            "crop_box": [round(float(v), 2) for v in self.crop_box],
            "trigger_reason": str(self.trigger_reason),
            "trigger_score": round(float(self.trigger_score), 4),
            "source_detection_idx": self.source_detection_idx,
            "context_margin": round(float(self.context_margin), 3),
            "area": round(float(self.area), 1),
        }


@dataclass
class CropResult:
    """Contains detections resulting from inference on a single crop."""
    proposal: CropProposal
    crop_shape: Tuple[int, int]                  # (height, width)
    raw_crop_detections: List[Detection]         # Coordinates relative to crop [0, W_crop, 0, H_crop]
    remapped_detections: List[Detection]         # Coordinates mapped back to original image
    inference_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "crop_shape": [int(self.crop_shape[0]), int(self.crop_shape[1])],
            "num_detections": len(self.remapped_detections),
            "remapped_detections": [d.to_dict() for d in self.remapped_detections],
            "inference_time_ms": round(float(self.inference_time_ms), 2),
        }


@dataclass
class FoveationTiming:
    """Detailed timing profiling breakdown in milliseconds."""
    read_ms: float = 0.0
    full_frame_infer_ms: float = 0.0
    uncertainty_proposal_ms: float = 0.0
    crop_extraction_ms: float = 0.0
    crop_infer_ms: float = 0.0
    fusion_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: round(float(v), 2) for k, v in asdict(self).items()}


@dataclass
class FoveationTrace:
    """Complete, auditable execution trace for a single image."""
    image_id: str
    image_shape: Tuple[int, int]                 # (height, width)
    policy: str                                  # 'uncertainty_active', 'fixed_grid', 'random', 'full_frame_only'
    full_frame_detections: List[Detection] = field(default_factory=list)
    crop_proposals: List[CropProposal] = field(default_factory=list)
    crop_results: List[CropResult] = field(default_factory=list)
    fused_detections: List[Detection] = field(default_factory=list)
    timing: FoveationTiming = field(default_factory=FoveationTiming)
    num_crops_executed: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": str(self.image_id),
            "image_shape": [int(self.image_shape[0]), int(self.image_shape[1])],
            "policy": str(self.policy),
            "num_full_frame_dets": len(self.full_frame_detections),
            "num_crops_executed": int(self.num_crops_executed),
            "num_fused_dets": len(self.fused_detections),
            "crop_proposals": [p.to_dict() for p in self.crop_proposals],
            "crop_results": [r.to_dict() for r in self.crop_results],
            "fused_detections": [d.to_dict() for d in self.fused_detections],
            "timing_ms": self.timing.to_dict(),
            "notes": self.notes,
        }


@dataclass
class FoveationConfig:
    """Engine configuration settings."""
    full_frame_imgsz: int = 960
    crop_imgsz: int = 640
    max_crops_per_image: int = 3
    policy: str = "uncertainty_active"          # 'full_frame_only', 'fixed_grid', 'random', 'uncertainty_active'
    
    # Uncertainty Signal Thresholds
    low_conf_min: float = 0.10
    low_conf_max: float = 0.35
    small_box_area_threshold: float = 1024.0    # 32^2 px
    cluster_iou_threshold: float = 0.35
    boundary_margin_ratio: float = 0.05
    
    # Crop Extraction Settings
    context_margin_ratio: float = 0.25
    min_crop_size: int = 128
    aspect_ratio: Optional[float] = None        # If None, variable aspect ratio matching context
    
    # Fusion Settings
    fusion_iou_threshold: float = 0.50
    boundary_suppress_margin: int = 8           # Suppress crop detections within N px of crop edge
    operational_conf_threshold: float = 0.25
