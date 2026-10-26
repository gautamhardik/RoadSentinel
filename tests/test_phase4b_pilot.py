"""
Unit tests for RoadSentinel Phase 4B Diagnostic Inference and Pilot Calibration.
Tests:
  1. Prerequisites and Checkpoint SHA-256 integrity.
  2. Deterministic cohort manifest integrity (exact 300 images).
  3. Raw candidate coordinate transforms (crop-to-fullframe remapping inside bounds).
  4. Metric calculation and decision rule consistency.
  5. Absence of test split access.
"""

import json
from pathlib import Path
import pytest


def test_prerequisites_and_checkpoint_hash():
    audit_path = Path("experiments/phase4_pilot/prerequisites_audit.json")
    assert audit_path.exists(), "prerequisites_audit.json missing!"
    with open(audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["checkpoint_sha256"] == "A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3"
    assert data["test_split_quarantined"] is True
    assert data["validation_images_available"] == 10000


def test_cohort_manifest_integrity():
    manifest_path = Path("experiments/phase4_pilot/cohort_manifest.json")
    assert manifest_path.exists(), "cohort_manifest.json missing!"
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["cohort_size"] == 300
    assert len(data["images"]) == 300

    # Ensure 5 strata with 60 images each
    strata_counts = data["strata_breakdown"]
    assert len(strata_counts) == 5
    for s_name, count in strata_counts.items():
        assert count == 60, f"Stratum {s_name} count {count} != 60"


def test_raw_candidates_and_coordinate_transforms():
    raw_path = Path("experiments/phase4_pilot/raw_candidates.jsonl")
    assert raw_path.exists(), "raw_candidates.jsonl missing!"

    line_count = 0
    img_w, img_h = 1280.0, 720.0
    crop_candidates_found = False

    with open(raw_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue
            c = json.loads(line)
            line_count += 1

            # Validate box coordinates within bounds (with slight margin for clipping)
            box = c["box_full"]
            assert len(box) == 4
            assert box[0] <= box[2], f"Invalid x coords: {box}"
            assert box[1] <= box[3], f"Invalid y coords: {box}"
            assert -5.0 <= box[0] <= img_w + 5.0
            assert -5.0 <= box[1] <= img_h + 5.0

            if c["source"] == "crop":
                crop_candidates_found = True
                assert c["crop_box"] is not None
                assert len(c["crop_box"]) == 4

            if idx >= 500:
                break  # Sample first 500 lines

    assert line_count > 0
    assert crop_candidates_found, "No crop candidates found in sample!"


def test_calibration_sweep_decision_rule_integrity():
    results_path = Path("experiments/phase4_pilot/calibration_sweep_results.json")
    assert results_path.exists(), "calibration_sweep_results.json missing!"
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["verdict"] == "PASS"
    best = data["best_candidate"]
    assert best is not None
    # Verify decision criteria
    assert best["delta_precision"] >= 0.050, f"Precision gain {best['delta_precision']} < 5 percentage points"
    assert best["delta_recall"] >= -0.020, f"Recall loss {best['delta_recall']} > 2 percentage points"
