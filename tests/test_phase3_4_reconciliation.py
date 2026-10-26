"""
Unit tests for Phase 3.4 assertion and reconciliation logic.
"""

import json
import pytest
from pathlib import Path
from scripts.reconcile_phase3_4_records import reconcile_and_assert

def test_reconcile_and_assert_synthetic(tmp_path: Path):
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "out"
    run_dir.mkdir()

    # Create synthetic raw_metrics.json matching authoritative denominators
    synthetic_metrics = {
        "benchmark_name": "Test Benchmark",
        "dataset": {"images_evaluated": 10000, "total_gt": 185578},
        "configurations": {}
    }

    configs = [
        "FullFrame-640", "FullFrame-960", "FullFrame-1280",
        "FixedGrid-960", "RandomCrop-960", "ActiveFoveation-960"
    ]

    for c in configs:
        synthetic_metrics["configurations"][c] = {
            "name": c, "type": "test", "imgsz": 960, "crops_per_image": 2,
            "total_crops_executed": 100, "fraction_images_cropped": 0.5,
            "timing_and_compute": {
                "throughput_fps": 30.0, "amortized_ms": 33.3, "peak_vram_mb": 1500.0,
                "batch1_stats": {
                    "in_memory": {"mean_ms": 15.0, "median_p50_ms": 14.5, "p95_ms": 18.0},
                    "pipeline": {"mean_ms": 25.0, "median_p50_ms": 24.0, "p95_ms": 30.0},
                }
            },
            "overall_aggregates": {"overall_mAP_50": 0.55, "overall_mAP_50_95": 0.32},
            "vru_aggregates": {
                "vru_macro_mAP_50": 0.52, "vru_macro_mAP_50_95": 0.28,
                "vru_micro_precision_50": 0.75, "vru_micro_recall_50": 0.57, "vru_micro_f1_50": 0.65
            },
            "size_group_results": {
                "Small (<32²)": {"total_gt": 102781, "matched_gt": 50000, "vru_gt": 7029, "vru_matched": 2558, "vru_recall": 0.3639},
                "Medium [32²-96²)": {"total_gt": 59669, "matched_gt": 45000, "vru_gt": 7246, "vru_matched": 5265, "vru_recall": 0.7266},
                "Large (≥96²)": {"total_gt": 23128, "matched_gt": 20000, "vru_gt": 1098, "vru_matched": 927, "vru_recall": 0.8443},
            },
            "per_class_results": [
                {"class_name": "person", "is_vru": True, "gt_count": 13265, "tp_50": 7790, "fp_50": 2000, "fn_50": 5475, "precision_50": 0.79, "recall_50": 0.58, "f1_50": 0.67, "ap_50": 0.60, "ap_50_95": 0.35},
                {"class_name": "rider", "is_vru": True, "gt_count": 649, "tp_50": 337, "fp_50": 200, "fn_50": 312, "precision_50": 0.62, "recall_50": 0.52, "f1_50": 0.56, "ap_50": 0.50, "ap_50_95": 0.25},
                {"class_name": "bike", "is_vru": True, "gt_count": 1007, "tp_50": 379, "fp_50": 300, "fn_50": 628, "precision_50": 0.55, "recall_50": 0.37, "f1_50": 0.44, "ap_50": 0.45, "ap_50_95": 0.22},
                {"class_name": "motor", "is_vru": True, "gt_count": 452, "tp_50": 244, "fp_50": 150, "fn_50": 208, "precision_50": 0.61, "recall_50": 0.54, "f1_50": 0.57, "ap_50": 0.52, "ap_50_95": 0.27},
            ]
        }

    raw_path = run_dir / "raw_metrics.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(synthetic_metrics, f)

    # Total VRU TP = 7790 + 337 + 379 + 244 = 8750
    # Tier TP = 2558 + 5265 + 927 = 8750
    # TP + FN = 8750 + (5475 + 312 + 628 + 208) = 8750 + 6623 = 15373.

    # Create dummy per_image_records.jsonl
    jsonl_path = run_dir / "per_image_records.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in configs:
            f.write(json.dumps({"config": c, "image": "img1.jpg", "vru_tp_count": 8750}) + "\n")

    # Run assertion and export
    reconcile_and_assert(run_dir, out_dir)

    assert (out_dir / "phase3_4_final_ablation.csv").exists()
    assert (out_dir / "phase3_4_final_size_groups.csv").exists()
    assert (out_dir / "phase3_4_final_latency.csv").exists()
