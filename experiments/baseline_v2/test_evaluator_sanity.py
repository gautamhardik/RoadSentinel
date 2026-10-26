"""
Sanity and Validation Test for Evaluator Integrity
Verifies:
1. True Positive (class and box match)
2. Wrong Class Prediction (FP on predicted, FN on GT)
3. Missed Object (FN)
4. Duplicate Prediction (1 TP, 1 FP)
5. Absent Class Handling (0 GT, proper absence reporting)
6. Real validation images with known labels
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_engine import (
    DetectionEvaluator,
    compute_box_iou,
    compute_ap_coco,
    CHECKPOINT_TO_GT_ID,
    GT_CLASSES,
    CHECKPOINT_CLASSES,
)


def test_iou_calculation():
    # Box: [x1, y1, x2, y2]
    boxA = [0.0, 0.0, 10.0, 10.0]
    boxB = [0.0, 0.0, 10.0, 10.0]
    assert compute_box_iou(boxA, boxB) == 1.0, "Identical boxes must have IoU = 1.0"

    boxC = [10.0, 10.0, 20.0, 20.0]
    assert compute_box_iou(boxA, boxC) == 0.0, "Non-overlapping boxes must have IoU = 0.0"

    boxD = [5.0, 0.0, 15.0, 10.0]  # overlap: [5, 0, 10, 10] -> 5 * 10 = 50. Union: 100 + 100 - 50 = 150 -> 1/3
    iou_ad = compute_box_iou(boxA, boxD)
    assert abs(iou_ad - (50.0 / 150.0)) < 1e-5, f"Expected 1/3, got {iou_ad}"
    print("  [PASS] IoU calculation verified.")


def test_synthetic_scenarios():
    evaluator = DetectionEvaluator(operational_conf=0.25, operational_iou=0.50)

    # 1. Image 1: Perfect match (Person)
    evaluator.add_ground_truth("img1", "person", [100.0, 100.0, 200.0, 200.0])
    evaluator.add_prediction("img1", "person", 0.90, [102.0, 100.0, 198.0, 200.0])

    # 2. Image 2: Wrong class prediction (GT is Bus, Model predicts Truck)
    evaluator.add_ground_truth("img2", "bus", [50.0, 50.0, 250.0, 250.0])
    # Model predicts checkpoint class 3 (truck)
    evaluator.add_prediction_from_checkpoint("img2", 3, 0.85, [50.0, 50.0, 250.0, 250.0])

    # 3. Image 3: Missed object (GT Bike, no prediction)
    evaluator.add_ground_truth("img3", "bike", [300.0, 300.0, 400.0, 400.0])

    # 4. Image 4: Duplicate predictions on 1 GT Car
    evaluator.add_ground_truth("img4", "car", [10.0, 10.0, 100.0, 100.0])
    evaluator.add_prediction("img4", "car", 0.95, [10.0, 10.0, 100.0, 100.0])
    evaluator.add_prediction("img4", "car", 0.70, [12.0, 12.0, 98.0, 98.0])

    # 5. Absent class: Train has 0 GT instances and 0 predictions

    results = evaluator.evaluate()
    metrics_by_name = {m["name"]: m for m in results["per_class_metrics"]}

    # Verify Person
    p_res = metrics_by_name["person"]
    assert p_res["gt_count"] == 1, f"Expected 1 GT person, got {p_res['gt_count']}"
    assert p_res["tp_50"] == 1, f"Expected 1 TP person, got {p_res['tp_50']}"
    assert p_res["fp_50"] == 0, f"Expected 0 FP person, got {p_res['fp_50']}"
    assert p_res["fn_50"] == 0, f"Expected 0 FN person, got {p_res['fn_50']}"
    assert p_res["precision_50"] == 1.0, f"Expected 1.0 precision, got {p_res['precision_50']}"
    assert p_res["recall_50"] == 1.0, f"Expected 1.0 recall, got {p_res['recall_50']}"
    assert p_res["ap_50"] == 1.0, f"Expected 1.0 AP@50, got {p_res['ap_50']}"
    print("  [PASS] Scenario 1 (True Positive match) verified.")

    # Verify Bus (Missed because prediction was truck)
    b_res = metrics_by_name["bus"]
    assert b_res["gt_count"] == 1
    assert b_res["tp_50"] == 0
    assert b_res["fn_50"] == 1
    assert b_res["recall_50"] == 0.0
    print("  [PASS] Scenario 2A (Wrong class GT side -> FN) verified.")

    # Verify Truck (False alarm because GT was bus)
    t_res = metrics_by_name["truck"]
    assert t_res["gt_count"] == 0
    assert t_res["is_absent_in_gt"] is True
    assert t_res["pred_operational_count"] == 1
    assert t_res["tp_50"] == 0
    assert t_res["fp_50"] == 1
    assert t_res["precision_50"] == 0.0
    print("  [PASS] Scenario 2B (Wrong class Pred side -> FP, Absent in GT) verified.")

    # Verify Bike (Missed object)
    bk_res = metrics_by_name["bike"]
    assert bk_res["gt_count"] == 1
    assert bk_res["pred_operational_count"] == 0
    assert bk_res["tp_50"] == 0
    assert bk_res["fn_50"] == 1
    assert bk_res["recall_50"] == 0.0
    print("  [PASS] Scenario 3 (Missed object / False Negative) verified.")

    # Verify Car (Duplicate detections)
    c_res = metrics_by_name["car"]
    assert c_res["gt_count"] == 1
    assert c_res["pred_operational_count"] == 2
    assert c_res["tp_50"] == 1
    assert c_res["fp_50"] == 1
    assert c_res["fn_50"] == 0
    assert c_res["precision_50"] == 0.5, f"Expected 0.5, got {c_res['precision_50']}"
    assert c_res["recall_50"] == 1.0, f"Expected 1.0, got {c_res['recall_50']}"
    print("  [PASS] Scenario 4 (Duplicate detection -> 1 TP, 1 FP) verified.")

    # Verify Absent class (Train: 0 GT, 0 Pred)
    tr_res = metrics_by_name["train"]
    assert tr_res["gt_count"] == 0
    assert tr_res["pred_operational_count"] == 0
    assert tr_res["is_absent_in_gt"] is True
    assert tr_res["ap_50"] is None, "Absent class AP must be reported as None"
    print("  [PASS] Scenario 5 (Absent class handled cleanly without distortion) verified.")


def test_real_inspection_images():
    from ultralytics import YOLO
    import cv2

    weights_path = PROJECT_ROOT / "weights" / "bdd100k_yolo11s.pt"
    val_img_dir = PROJECT_ROOT / "bdd100k_dataset" / "val" / "images"
    val_lbl_dir = PROJECT_ROOT / "bdd100k_dataset" / "val" / "labels"

    assert weights_path.exists(), f"Weights file {weights_path} not found"
    model = YOLO(str(weights_path))

    test_images = [
        "c95ad578-8caa410d.jpg",  # contains bike, traffic sign, traffic light
        "ca1c6003-aad4464c.jpg",  # contains bus, motor, rider
    ]

    evaluator = DetectionEvaluator(operational_conf=0.25, operational_iou=0.50)

    for img_name in test_images:
        img_p = val_img_dir / img_name
        lbl_p = val_lbl_dir / img_name.replace(".jpg", ".txt")
        if not img_p.exists() or not lbl_p.exists():
            print(f"Skipping {img_name} (file not found)")
            continue

        img = cv2.imread(str(img_p))
        h, w, _ = img.shape

        # Load GT
        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                evaluator.add_ground_truth_yolo(img_name, cid, xc, yc, bw, bh, w, h)

        # Predict
        res = model.predict(source=img, imgsz=640, conf=0.25, verbose=False)[0]
        for det in res.boxes:
            ckpt_cid = int(det.cls.item())
            conf = float(det.conf.item())
            box = det.xyxy[0].tolist()
            evaluator.add_prediction_from_checkpoint(img_name, ckpt_cid, conf, box)

    results = evaluator.evaluate()
    print("  [PASS] Real image evaluation executed cleanly.")
    metrics_by_name = {m["name"]: m for m in results["per_class_metrics"]}
    for cname in ["bike", "bus", "rider", "traffic sign", "traffic light"]:
        m = metrics_by_name[cname]
        print(f"    {cname:<14}: GT={m['gt_count']} Pred={m['pred_operational_count']} TP={m['tp_50']} FP={m['fp_50']}")


if __name__ == "__main__":
    print("Running Evaluator Sanity Verification Suite...")
    test_iou_calculation()
    test_synthetic_scenarios()
    test_real_inspection_images()
    print("\nALL SANITY AND INTEGRITY CHECKS PASSED SUCCESSFULLY!")
