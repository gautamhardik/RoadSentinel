"""
RoadSentinel Phase 3.1 — Audit & Integrity Test Suite
Verifies:
1. Ground-truth reconciliation across BDD100K validation labels.
2. Size-group mutually exclusive and exhaustive partitioning.
3. Serialization integrity for NumPy scalars, tensors, arrays, non-finite values, and nested structures.
"""

import json
import math
from pathlib import Path
import numpy as np
import pytest

from kaggle_jobs.foveation_benchmark.evaluate_phase3_foveation import (
    CHECKPOINT_TO_GT_ID, GT_CLASSES, VRU_CLASS_NAMES
)

def test_ground_truth_reconciliation_exact_counts():
    """Verify ground-truth parsing against the actual BDD100K validation set."""
    label_dir = Path("bdd100k_dataset/val/labels")
    assert label_dir.exists(), f"Label dir does not exist: {label_dir}"
    
    import os
    label_files = [e.path for e in os.scandir(label_dir) if e.name.endswith(".txt")]
    label_files.sort()
    assert len(label_files) == 10000, f"Expected 10,000 label files, got {len(label_files)}"
    
    total_objects = 0
    class_counts = {cname: 0 for cname in GT_CLASSES.values()}
    size_counts = {"Small (<32²)": 0, "Medium [32²-96²)": 0, "Large (≥96²)": 0}
    vru_size_counts = {"Small (<32²)": 0, "Medium [32²-96²)": 0, "Large (≥96²)": 0}
    zero_dim_boxes = 0

    img_w, img_h = 1280.0, 720.0

    for lf in label_files:
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                xc, yc, w, h = [float(x) for x in parts[1:5]]
                if w <= 0 or h <= 0:
                    zero_dim_boxes += 1
                
                bw = w * img_w
                bh = h * img_h
                area = max(0.0, bw) * max(0.0, bh)

                total_objects += 1
                cname = GT_CLASSES[cid]
                class_counts[cname] += 1
                is_vru = cname in VRU_CLASS_NAMES

                if area < 1024.0:
                    sz = "Small (<32²)"
                elif area < 9216.0:
                    sz = "Medium [32²-96²)"
                else:
                    sz = "Large (≥96²)"

                size_counts[sz] += 1
                if is_vru:
                    vru_size_counts[sz] += 1

    # Exact reconciliations
    assert zero_dim_boxes == 52, f"Expected exactly 52 zero-dimension boxes in BDD100K val, got {zero_dim_boxes}"
    assert total_objects == 185578, f"Expected 185,578 objects, got {total_objects}"
    
    # Class-level verification
    assert class_counts["person"] == 13265
    assert class_counts["rider"] == 649
    assert class_counts["bike"] == 1007
    assert class_counts["motor"] == 452
    
    total_vru = class_counts["person"] + class_counts["rider"] + class_counts["bike"] + class_counts["motor"]
    assert total_vru == 15373, f"Expected 15,373 VRUs, got {total_vru}"

    # Size-group verification
    assert vru_size_counts["Small (<32²)"] == 7029
    assert vru_size_counts["Medium [32²-96²)"] == 7246
    assert vru_size_counts["Large (≥96²)"] == 1098
    
    vru_size_sum = sum(vru_size_counts.values())
    assert vru_size_sum == total_vru, f"Sum of VRU size tiers {vru_size_sum} != total VRUs {total_vru}"

    # Overall size tiers
    assert size_counts["Small (<32²)"] == 102781
    assert size_counts["Medium [32²-96²)"] == 59669
    assert size_counts["Large (≥96²)"] == 23128
    assert sum(size_counts.values()) == total_objects


def test_serialization_integrity():
    """Verify that recursive serialization converts numpy types and edge-case values safely."""
    def to_serializable(val):
        if isinstance(val, (np.floating, np.float32, np.float64)):
            if math.isnan(val) or math.isinf(val):
                return None
            return float(val)
        if isinstance(val, (np.integer, np.int32, np.int64)):
            return int(val)
        if isinstance(val, np.ndarray):
            return [to_serializable(x) for x in val.tolist()]
        if isinstance(val, (list, tuple)):
            return [to_serializable(x) for x in val]
        if isinstance(val, dict):
            return {k: to_serializable(v) for k, v in val.items()}
        if isinstance(val, set):
            return [to_serializable(x) for x in sorted(list(val))]
        if isinstance(val, float):
            if math.isnan(val) or math.isinf(val):
                return None
        return val

    # Test payload containing all troublesome types
    complex_payload = {
        "float32_scalar": np.float32(0.5403),
        "float64_scalar": np.float64(0.5120),
        "int32_scalar": np.int32(10000),
        "int64_scalar": np.int64(185578),
        "numpy_array_1d": np.array([10.5, 20.2, 30.8], dtype=np.float32),
        "numpy_array_2d": np.zeros((2, 2), dtype=np.int32),
        "set_of_ints": {3, 1, 2},
        "tuple_of_floats": (np.float32(1.1), np.float32(2.2)),
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "nested_dict": {
            "inner_array": np.array([np.float32(42.0)]),
            "inner_set": {"item1", "item2"},
        }
    }

    # Should serialize without error
    serialized = to_serializable(complex_payload)
    json_str = json.dumps(serialized, indent=2)
    assert isinstance(json_str, str)

    # Should deserialize back into pure Python native types
    parsed = json.loads(json_str)
    assert parsed["float32_scalar"] == pytest.approx(0.5403, 1e-4)
    assert parsed["int32_scalar"] == 10000
    assert parsed["set_of_ints"] == [1, 2, 3]
    assert parsed["nan_val"] is None
    assert parsed["inf_val"] is None
    assert parsed["nested_dict"]["inner_array"] == [42.0]


def test_phase7_controlled_latency_consistency():
    """Verify internal arithmetic and consistency of Phase 7 controlled latency report."""
    from scripts.audit_latency_consistency import run_consistency_audit
    summary = run_consistency_audit()
    assert summary["status"] == "ARITHMETIC_VERIFIED_CONSISTENT"
    assert len(summary["discrepancies_found"]) == 0
    assert summary["sample_design"]["unique_images_evaluated"] == 15
    assert summary["sample_design"]["total_observations_per_pipeline"] == 30
    assert summary["recomputed_statistics"]["pooled"]["raw_full_frame"]["n_observations"] == 30
    paired = summary["recomputed_statistics"]["paired_analysis_fov_minus_horizon"]
    assert paired["total_pairs"] == 30
    assert paired["percentage_horizon_faster"] == 66.7

    # Stage accounting and causal status check
    stage = summary["stage_accounting"]
    assert stage["causal_status_of_358_1_ms_gap"]["epistemic_status"] == "MEASURED_AGGREGATE_RESIDUAL_CAUSALLY_UNRESOLVED"
    assert stage["causal_status_of_358_1_ms_gap"]["gap_value_ms"] == pytest.approx(358.1, abs=0.1)


def test_phase7_standard_deviation_convention_regression():
    """Regression test: verify sample (ddof=1) vs population (ddof=0) std calculations on raw timings."""
    from scripts.audit_latency_consistency import run_consistency_audit
    summary = run_consistency_audit()
    std_audit = summary["standard_deviation_convention_audit"]

    # Horizon Slicing Pooled
    hz = std_audit["horizon_slicing_pooled"]
    assert hz["stored_std_ms"] == 268.5
    assert hz["calculated_population_std_ddof0_ms"] == pytest.approx(268.46, abs=0.05)
    assert hz["calculated_sample_std_ddof1_ms"] == pytest.approx(273.05, abs=0.05)
    assert hz["delta_sample_minus_stored_ms"] == pytest.approx(4.55, abs=0.05)
    assert hz["status"] == "STORED_USES_POPULATION_DDOF0"

    # Raw Full-Frame Pooled
    raw = std_audit["raw_full_frame_pooled"]
    assert raw["stored_std_ms"] == 239.6
    assert raw["calculated_population_std_ddof0_ms"] == pytest.approx(239.59, abs=0.05)
    assert raw["calculated_sample_std_ddof1_ms"] == pytest.approx(243.68, abs=0.05)
    assert raw["status"] == "STORED_USES_POPULATION_DDOF0"

    # Active Foveation Pooled
    fov = std_audit["active_foveation_pooled"]
    assert fov["stored_std_ms"] == 651.1
    assert fov["calculated_population_std_ddof0_ms"] == pytest.approx(651.09, abs=0.05)
    assert fov["calculated_sample_std_ddof1_ms"] == pytest.approx(662.22, abs=0.05)
    assert fov["status"] == "STORED_USES_POPULATION_DDOF0"



