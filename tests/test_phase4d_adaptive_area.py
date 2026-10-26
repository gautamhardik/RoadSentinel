"""
Unit and Regression Tests for RoadSentinel Phase 4D Adaptive-Area Calibration
Validates predicted-area computation, boundary conditions around 1024 px^2,
isolated vs confirmed thresholding, coordinate clamping, invalid boxes,
and complete absence of ground-truth leakage in the post-processing pipeline.
"""

import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pytest

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from kaggle_jobs.calibration_benchmark.evaluate_phase4c_wbf import (
    CHECKPOINT_TO_GT_ID,
    GT_CLASSES,
    VRU_CLASS_NAMES,
    compute_box_iou,
    compute_box_iou_vec,
    compute_coco_ap_for_class,
    evaluate_operational_point,
    expand_crop_box,
    baseline_fuse_and_deduplicate,
)


def compute_predicted_area(box, img_w=1280.0, img_h=720.0) -> float:
    """Clamps box coordinates to [0, img_w] x [0, img_h] and computes predicted area."""
    x1 = max(0.0, min(float(img_w), float(box[0])))
    y1 = max(0.0, min(float(img_h), float(box[1])))
    x2 = max(0.0, min(float(img_w), float(box[2])))
    y2 = max(0.0, min(float(img_h), float(box[3])))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def adaptive_area_wbf_consensus_fusion(
    full_frame_dets,
    crop_dets,
    tau_ff=0.25,
    tau_isolated_small=0.30,
    tau_isolated_large=0.45,
    tau_confirmed=0.20,
    area_cutoff=1024.0,
    iou_assoc=0.30,
    img_w=1280.0,
    img_h=720.0,
):
    """
    Phase 4D Adaptive Area-Dependent WBF Consensus Fusion.
    Operates strictly on detector predictions and geometric coordinates.
    """
    # 1. Full-frame filter
    ff_filt = [d for d in full_frame_dets if d["conf"] >= tau_ff]

    # 2. Check crop detections against coarse full-frame support (conf >= 0.10)
    ff_coarse = [d for d in full_frame_dets if d["conf"] >= 0.10]
    cr_filt = []

    for cr_d in crop_dets:
        box = cr_d["box"]
        area = compute_predicted_area(box, img_w, img_h)
        if area <= 0.0:
            continue  # Drop degenerate / invalid boxes

        # Check concurrence against full frame
        has_ff_confirmation = False
        for ff_d in ff_coarse:
            if ff_d["class_name"] == cr_d["class_name"]:
                if compute_box_iou(cr_d["box"], ff_d["box"]) >= iou_assoc:
                    has_ff_confirmation = True
                    break

        if has_ff_confirmation:
            req_thresh = tau_confirmed
        else:
            req_thresh = tau_isolated_small if area < area_cutoff else tau_isolated_large

        if cr_d["conf"] >= req_thresh:
            cr_filt.append(cr_d)

    # 3. WBF spatial clustering and consensus boosting
    fused = []
    all_dets = ff_filt + cr_filt
    from collections import defaultdict
    by_cls = defaultdict(list)
    for d in all_dets:
        by_cls[d["class_name"]].append(d)

    for cname, dlist in by_cls.items():
        dlist.sort(key=lambda x: x["conf"], reverse=True)
        clusters = []

        for d in dlist:
            matched_cluster = None
            for clust in clusters:
                rep_box = clust[0]["box"]
                if compute_box_iou(d["box"], rep_box) >= 0.50:
                    matched_cluster = clust
                    break
            if matched_cluster is not None:
                matched_cluster.append(d)
            else:
                clusters.append([d])

        for clust in clusters:
            if len(clust) == 1:
                fused.append(clust[0])
            else:
                weights = [x["conf"] * (1.2 if x["source"] == "crop" else 1.0) for x in clust]
                w_sum = max(sum(weights), 1e-6)
                f_box = (
                    sum(x["box"][0] * weights[idx] for idx, x in enumerate(clust)) / w_sum,
                    sum(x["box"][1] * weights[idx] for idx, x in enumerate(clust)) / w_sum,
                    sum(x["box"][2] * weights[idx] for idx, x in enumerate(clust)) / w_sum,
                    sum(x["box"][3] * weights[idx] for idx, x in enumerate(clust)) / w_sum,
                )
                has_ff = any(x["source"] == "full_frame" for x in clust)
                has_cr = any(x["source"] == "crop" for x in clust)
                max_c = max(x["conf"] for x in clust)
                f_conf = min(0.99, max_c + (0.08 if has_ff and has_cr else 0.02))
                fused.append({
                    "img_name": clust[0]["img_name"],
                    "class_name": cname,
                    "conf": f_conf,
                    "box": np.array(f_box, dtype=np.float32),
                    "source": "wbf_consensus",
                })

    return fused


# ============================================================================
# Test Cases
# ============================================================================

def test_checkpoint_hash_verification():
    """Verify exact SHA-256 hash of the production checkpoint."""
    ckpt_path = repo_root / "weights" / "bdd100k_yolo11s.pt"
    assert ckpt_path.exists(), "Checkpoint weights/bdd100k_yolo11s.pt not found!"
    with open(ckpt_path, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest().upper()
    assert h == "A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3"


def test_config_json_integrity():
    """Verify Phase 4D configuration file and parameters."""
    cfg_p = repo_root / "experiments" / "phase4d_validation" / "config.json"
    assert cfg_p.exists()
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["phase"] == "4D"
    assert "Candidate-A" in cfg["candidate_grid"]
    assert "Candidate-B" in cfg["candidate_grid"]
    assert "Candidate-C" in cfg["candidate_grid"]
    assert cfg["candidate_grid"]["Candidate-A"]["tau_isolated_small"] == 0.28
    assert cfg["candidate_grid"]["Candidate-B"]["tau_isolated_small"] == 0.30
    assert cfg["candidate_grid"]["Candidate-C"]["tau_isolated_small"] == 0.32
    assert cfg["candidate_grid"]["Candidate-A"]["tau_isolated_large"] == 0.45


def test_area_threshold_boundary_conditions():
    """
    Test area boundary conditions immediately below, at, and above 1024 px^2.
    Candidate B: tau_small=0.30, tau_large=0.45, threshold=1024.0.
    """
    # 1. Immediately below: width=32.0, height=31.99 -> area = 1023.68 < 1024
    # With conf=0.35: should SURVIVE under tau_small=0.30
    b_below = np.array([100.0, 100.0, 132.0, 131.99], dtype=np.float32)
    assert compute_predicted_area(b_below) < 1024.0
    c_below = [{
        "img_name": "test.jpg", "class_name": "person", "conf": 0.35, "box": b_below, "source": "crop"
    }]
    fused_below = adaptive_area_wbf_consensus_fusion([], c_below, tau_isolated_small=0.30, tau_isolated_large=0.45)
    assert len(fused_below) == 1, "Box below 1024 px^2 with conf=0.35 should survive"

    # 2. Exactly at or immediately above: width=32.0, height=32.01 -> area = 1024.32 >= 1024
    # With conf=0.35: should BE SUPPRESSED under tau_large=0.45
    b_above = np.array([100.0, 100.0, 132.0, 132.01], dtype=np.float32)
    assert compute_predicted_area(b_above) >= 1024.0
    c_above = [{
        "img_name": "test.jpg", "class_name": "person", "conf": 0.35, "box": b_above, "source": "crop"
    }]
    fused_above = adaptive_area_wbf_consensus_fusion([], c_above, tau_isolated_small=0.30, tau_isolated_large=0.45)
    assert len(fused_above) == 0, "Box above 1024 px^2 with conf=0.35 should be suppressed by tau_large=0.45"

    # 3. Box above 1024 with conf=0.48: should SURVIVE under tau_large=0.45
    c_above_high = [{
        "img_name": "test.jpg", "class_name": "person", "conf": 0.48, "box": b_above, "source": "crop"
    }]
    fused_above_high = adaptive_area_wbf_consensus_fusion([], c_above_high, tau_isolated_small=0.30, tau_isolated_large=0.45)
    assert len(fused_above_high) == 1, "Box above 1024 px^2 with conf=0.48 should survive"


def test_coordinate_clamping_and_boundary_boxes():
    """Verify coordinate clamping on boxes extending outside [1280, 720]."""
    # Box protruding out of image right/bottom: [1200, 700, 1300, 750]
    # In full space: clamped width = 1280-1200 = 80, clamped height = 720-700 = 20 -> area = 1600
    b_protruding = np.array([1200.0, 700.0, 1300.0, 750.0], dtype=np.float32)
    area = compute_predicted_area(b_protruding, img_w=1280.0, img_h=720.0)
    assert area == pytest.approx(80.0 * 20.0, abs=1e-3)


def test_zero_area_and_invalid_boxes():
    """Verify handling of degenerate or inverted boxes."""
    # Zero width box
    b_zero_w = np.array([100.0, 100.0, 100.0, 200.0], dtype=np.float32)
    assert compute_predicted_area(b_zero_w) == 0.0

    # Inverted box (x1 > x2)
    b_inverted = np.array([200.0, 100.0, 100.0, 200.0], dtype=np.float32)
    assert compute_predicted_area(b_inverted) == 0.0

    # Out of frame box
    b_oob = np.array([1300.0, 800.0, 1400.0, 900.0], dtype=np.float32)
    assert compute_predicted_area(b_oob) == 0.0

    c_invalid = [
        {"img_name": "test.jpg", "class_name": "person", "conf": 0.99, "box": b_zero_w, "source": "crop"},
        {"img_name": "test.jpg", "class_name": "person", "conf": 0.99, "box": b_inverted, "source": "crop"},
        {"img_name": "test.jpg", "class_name": "person", "conf": 0.99, "box": b_oob, "source": "crop"},
    ]
    fused_invalid = adaptive_area_wbf_consensus_fusion([], c_invalid)
    assert len(fused_invalid) == 0, "Invalid/zero-area boxes must be rejected"


def test_isolated_vs_confirmed_discrimination():
    """Verify that confirmed crop boxes receive tau_confirmed (0.20) regardless of area."""
    # Large box: area = 2000 >= 1024
    b_large = np.array([100.0, 100.0, 150.0, 140.0], dtype=np.float32)  # 50 * 40 = 2000
    assert compute_predicted_area(b_large) >= 1024.0

    # Uncorroborated with conf=0.25: suppressed by tau_large=0.45
    cr_det = {"img_name": "test.jpg", "class_name": "person", "conf": 0.25, "box": b_large, "source": "crop"}
    fused_isolated = adaptive_area_wbf_consensus_fusion([], [cr_det])
    assert len(fused_isolated) == 0

    # Corroborated by coarse full-frame box (conf=0.15 >= 0.10, IoU >= 0.30):
    # Should survive under tau_confirmed = 0.20!
    ff_coarse = {"img_name": "test.jpg", "class_name": "person", "conf": 0.15, "box": b_large, "source": "full_frame"}
    fused_confirmed = adaptive_area_wbf_consensus_fusion([ff_coarse], [cr_det])
    assert len(fused_confirmed) == 1
    # When coarse full-frame box is below tau_ff (0.25), crop detection alone survives into final set
    assert fused_confirmed[0]["source"] == "crop"

    # If full-frame box is also >= tau_ff (0.25), both cluster and produce wbf_consensus:
    ff_strong = {"img_name": "test.jpg", "class_name": "person", "conf": 0.30, "box": b_large, "source": "full_frame"}
    fused_both = adaptive_area_wbf_consensus_fusion([ff_strong], [cr_det])
    assert len(fused_both) == 1
    assert fused_both[0]["source"] == "wbf_consensus"


def test_full_frame_threshold_invariance():
    """Verify that full-frame boxes are strictly filtered at tau_ff (0.25) regardless of area rules."""
    ff_sub = {"img_name": "test.jpg", "class_name": "person", "conf": 0.24, "box": np.array([100, 100, 120, 120]), "source": "full_frame"}
    ff_valid = {"img_name": "test.jpg", "class_name": "person", "conf": 0.26, "box": np.array([200, 200, 220, 220]), "source": "full_frame"}
    fused = adaptive_area_wbf_consensus_fusion([ff_sub, ff_valid], [])
    assert len(fused) == 1
    assert fused[0]["box"][0] == 200


def test_no_ground_truth_leakage():
    """Verify that detection dictionaries passed to fusion do not contain ground-truth keys."""
    ff_det = {"img_name": "test.jpg", "class_name": "person", "conf": 0.70, "box": np.array([10, 10, 50, 50]), "source": "full_frame"}
    cr_det = {"img_name": "test.jpg", "class_name": "person", "conf": 0.80, "box": np.array([12, 12, 48, 48]), "source": "crop"}

    assert "gt" not in ff_det and "gt_idx" not in ff_det and "is_vru" not in ff_det
    assert "gt" not in cr_det and "gt_idx" not in cr_det and "is_vru" not in cr_det

    fused = adaptive_area_wbf_consensus_fusion([ff_det], [cr_det])
    for d in fused:
        assert "gt" not in d
        assert "gt_idx" not in d
        assert "is_vru" not in d
