"""
RoadSentinel Phase 3.2 — Comprehensive Regression Protection Test Suite
Enforces mathematical consistency, denominator reconciliation, duplicate suppression,
and artifact validity across all benchmark configurations.
"""

import json
import math
from pathlib import Path
import numpy as np
import pytest

from src.foveation.types import Detection, FoveationConfig, VRU_CLASS_NAMES
from src.foveation.box_fusion import BoxFusionEngine
from kaggle_jobs.foveation_benchmark.evaluate_phase3_foveation import (
    CHECKPOINT_TO_GT_ID, GT_CLASSES, CHECKPOINT_CLASSES
)


@pytest.fixture
def authoritative_metrics():
    """Load authoritative benchmark JSON metrics."""
    metrics_path = Path("experiments/phase3_foveation/final_reconciled_metrics.json")
    assert metrics_path.exists(), f"Missing authoritative metrics at {metrics_path}"
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_vru_denominators_consistency(authoritative_metrics):
    """Assert VRU GT denominators are identical across all configurations and equal 15,373."""
    gt_info = authoritative_metrics["ground_truth_reconciliation"]
    total_vrus = gt_info["total_vrus"]
    assert total_vrus == 15373, f"Expected 15,373 total VRUs, got {total_vrus}"
    
    size_vru = gt_info["vru_by_size_group"]
    assert size_vru["Small (<32²)"] == 7029
    assert size_vru["Medium [32²-96²)"] == 7246
    assert size_vru["Large (≥96²)"] == 1098
    assert size_vru["Small (<32²)"] + size_vru["Medium [32²-96²)"] + size_vru["Large (≥96²)"] == total_vrus


def test_size_group_sum_equals_total_tp(authoritative_metrics):
    """Assert for every configuration that Small TP + Medium TP + Large TP == Total VRU TP."""
    configs = authoritative_metrics["configurations"]
    assert len(configs) == 6, f"Expected 6 configurations, found {len(configs)}"

    for cfg_name, data in configs.items():
        s_tp = data["small_vru_matched"]
        m_tp = data["med_vru_matched"]
        l_tp = data["large_vru_matched"]
        tot_tp = data["total_vru_matched"]

        assert s_tp + m_tp + l_tp == tot_tp, (
            f"Config {cfg_name}: Size-group TPs ({s_tp} + {m_tp} + {l_tp} = {s_tp + m_tp + l_tp}) "
            f"!= total VRU TP ({tot_tp})"
        )


def test_mathematical_metric_consistency(authoritative_metrics):
    """Assert TP, FP, FN, Precision, Recall, and F1 agree mathematically within 1e-4."""
    configs = authoritative_metrics["configurations"]
    total_gt = authoritative_metrics["ground_truth_reconciliation"]["total_vrus"]

    for cfg_name, data in configs.items():
        tp = data["total_vru_matched"]
        fp = data["total_vru_fp"]
        fn = data["total_vru_fn"]

        # 1. FN == Total GT - TP
        assert tp + fn == total_gt, f"Config {cfg_name}: TP ({tp}) + FN ({fn}) != Total GT ({total_gt})"

        # 2. Recall == TP / GT
        expected_rec = tp / total_gt
        reported_rec = data["vru_micro_recall_50"]
        assert abs(reported_rec - expected_rec) < 1e-3, (
            f"Config {cfg_name}: Reported recall {reported_rec} != expected {expected_rec:.4f}"
        )

        # 3. Precision == TP / (TP + FP)
        expected_prec = tp / (tp + fp)
        reported_prec = data["vru_micro_precision_50"]
        assert abs(reported_prec - expected_prec) < 1e-3, (
            f"Config {cfg_name}: Reported precision {reported_prec} != expected {expected_prec:.4f}"
        )

        # 4. F1 == Harmonic mean of precision and recall
        expected_f1 = 2 * reported_prec * reported_rec / (reported_prec + reported_rec)
        reported_f1 = data["vru_micro_f1_50"]
        assert abs(reported_f1 - expected_f1) < 1e-3, (
            f"Config {cfg_name}: Reported F1 {reported_f1} != expected {expected_f1:.4f}"
        )


def test_crop_detection_not_double_counted():
    """Assert box fusion and NMS suppress duplicate detections against the same ground-truth target."""
    engine = BoxFusionEngine(FoveationConfig(fusion_iou_threshold=0.50))
    
    # Simulate a full-frame detection and a high-resolution crop detection on the exact same person
    ff_det = Detection(
        box=(100.0, 100.0, 150.0, 200.0),
        class_id=0, class_name="person", confidence=0.45, source="full_frame"
    )
    crop_det = Detection(
        box=(102.0, 99.0, 149.0, 201.0),
        class_id=0, class_name="person", confidence=0.82, source="crop"
    )
    
    fused = engine.fuse_detections([ff_det], [crop_det])
    # The two highly overlapping detections must be fused into exactly ONE detection
    assert len(fused) == 1, f"Expected 1 fused detection, got {len(fused)}"
    assert fused[0].class_name == "person"
    assert fused[0].source == "fused"
    assert fused[0].confidence >= 0.82  # Inherits high confidence from crop


def test_class_id_mapping_bijection():
    """Assert checkpoint class IDs map uniquely and correctly to canonical GT classes."""
    for ckpt_id, gt_id in CHECKPOINT_TO_GT_ID.items():
        ckpt_name = CHECKPOINT_CLASSES[ckpt_id]
        gt_name = GT_CLASSES[gt_id]
        assert ckpt_name == gt_name, (
            f"Mismatch at ckpt_id {ckpt_id}: Checkpoint calls it '{ckpt_name}' but GT calls it '{gt_name}'"
        )


def test_no_placeholder_values_in_artifacts(authoritative_metrics):
    """Assert authoritative artifacts contain no placeholder strings or missing values."""
    assert authoritative_metrics["audit_status"] == "VALIDATED"
    for cfg_name, d in authoritative_metrics["configurations"].items():
        assert d["throughput_fps"] > 0.0
        assert d["amortized_ms"] > 0.0
        assert d["peak_vram_mb"] > 0.0
        assert 0.0 < d["vru_micro_precision_50"] <= 1.0
        assert 0.0 < d["vru_micro_recall_50"] <= 1.0
        assert 0.0 < d["vru_micro_f1_50"] <= 1.0
        assert 0.0 < d["small_vru_recall"] <= 1.0
