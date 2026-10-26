"""
RoadSentinel Phase 7 — Matched Readiness Ablation Evaluation
Evaluates 4 matched configurations across identical 300 pilot development images:
1. Pipeline 1: Frozen Full-Frame Baseline (tau=0.25)
2. Pipeline 2: Lower-Threshold Full-Frame (tau=0.15)
3. Pipeline 3: Full-Frame (tau=0.25) + Autonomous Horizon Slice + Calibrated WBF (EXP-7B)
4. Pipeline 4: Existing Active Foveation Baseline (Uncertainty Crops + Calibrated WBF)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, os.getcwd())

import time
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
torch.set_num_threads(16)
from ultralytics import YOLO

from kaggle_jobs.calibration_benchmark.evaluate_phase4c_wbf import (
    PILOT_COHORT_IMAGE_SET,
    ALL_CLASS_NAMES,
    VRU_CLASS_NAMES,
    GT_CLASSES,
    compute_box_iou_vec,
    calibrated_wbf_consensus_fusion,
)
from src.foveation.horizon_slicer import HorizonSlicer

def run_readiness_evaluation():
    ckpt_path = Path("experiments/phase6_session2_error/runs/phase6_yolo11m_960_session1/weights/best.pt")
    val_img_dir = Path("bdd100k_dataset/val/images")
    val_lbl_dir = Path("bdd100k_dataset/val/labels")

    pilot_images = sorted(list(PILOT_COHORT_IMAGE_SET))
    img_w, img_h = 1280.0, 720.0

    # Parse GT
    gt_boxes = {c: defaultdict(list) for c in ALL_CLASS_NAMES}
    gt_objects = []

    for img_name in pilot_images:
        lbl_p = val_lbl_dir / (Path(img_name).stem + ".txt")
        if not lbl_p.exists():
            continue
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                cname = GT_CLASSES[cid]
                xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                x1 = (xc - bw / 2.0) * img_w
                y1 = (yc - bh / 2.0) * img_h
                x2 = (xc + bw / 2.0) * img_w
                y2 = (yc + bh / 2.0) * img_h
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                sz_grp = "Small (<32²)" if area < 1024.0 else ("Medium [32²-96²)" if area < 9216.0 else "Large (≥96²)")

                gt_idx = len(gt_boxes[cname][img_name])
                gt_boxes[cname][img_name].append([x1, y1, x2, y2])
                gt_objects.append({
                    "img_name": img_name,
                    "class_name": cname,
                    "gt_idx": gt_idx,
                    "size_group": sz_grp,
                    "is_vru": cname in VRU_CLASS_NAMES,
                })

    gt_by_class_img = {
        c: {img: np.array(boxes, dtype=np.float32) for img, boxes in gt_boxes[c].items()}
        for c in ALL_CLASS_NAMES
    }

    total_vru_gt = sum(1 for o in gt_objects if o["is_vru"])
    total_small_vru_gt = sum(1 for o in gt_objects if o["is_vru"] and o["size_group"] == "Small (<32²)")

    model = YOLO(str(ckpt_path))

    # 1. Full Frame Predictions at tau=0.01 (batched)
    raw_preds_by_img = defaultdict(list)
    b_size = 16
    for i in range(0, len(pilot_images), b_size):
        chunk = pilot_images[i:i+b_size]
        b_paths = [str(val_img_dir / f) for f in chunk]
        results = model.predict(b_paths, imgsz=960, conf=0.01, device="cpu", verbose=False)
        for img_name, r in zip(chunk, results):
            for b in r.boxes:
                cname = model.names[int(b.cls.item())]
                conf = float(b.conf.item())
                coords = b.xyxy[0].cpu().numpy()
                raw_preds_by_img[img_name].append({
                    "img_name": img_name,
                    "class_name": cname,
                    "conf": conf,
                    "box": coords,
                    "source": "full_frame",
                })

    # 2. Horizon Slice Extraction & Inference (EXP-7B)
    slicer = HorizonSlicer()
    horizon_slices = []
    slice_coords_list = []

    for img_name in pilot_images:
        p = str(val_img_dir / img_name)
        frame = cv2.imread(p)
        sub_img, s_coords = slicer.extract_slice(frame)
        horizon_slices.append(sub_img)
        slice_coords_list.append(s_coords)

    horizon_preds_by_img = defaultdict(list)
    for i in range(0, len(horizon_slices), b_size):
        sub_chunk = horizon_slices[i:i+b_size]
        chunk_names = pilot_images[i:i+b_size]
        chunk_coords = slice_coords_list[i:i+b_size]
        results = model.predict(sub_chunk, imgsz=960, conf=0.05, device="cpu", verbose=False)
        for img_name, s_coords, r in zip(chunk_names, chunk_coords, results):
            for b in r.boxes:
                cname = model.names[int(b.cls.item())]
                conf = float(b.conf.item())
                coords = b.xyxy[0].cpu().numpy()
                remapped_box = slicer.remap_detection(tuple(coords), s_coords, (720, 1280), conf=conf)
                if remapped_box is not None:
                    horizon_preds_by_img[img_name].append({
                        "img_name": img_name,
                        "class_name": cname,
                        "conf": conf,
                        "box": remapped_box,
                        "source": "crop",  # tag as crop for WBF
                    })

    # Evaluator helper
    def evaluate_detection_set(dets_by_img, tau_cut=0.25):
        total_tp, total_fp = 0, 0
        matched_gt_by_img = defaultdict(set)

        for cname in VRU_CLASS_NAMES:
            c_preds = []
            for img_name in pilot_images:
                for d in dets_by_img[img_name]:
                    if d["class_name"] == cname and d["conf"] >= tau_cut:
                        c_preds.append(d)
            c_preds.sort(key=lambda x: x["conf"], reverse=True)
            c_gt = gt_by_class_img[cname]

            for p in c_preds:
                img_name = p["img_name"]
                gts = c_gt.get(img_name)
                if gts is None or len(gts) == 0:
                    total_fp += 1
                    continue
                ious = compute_box_iou_vec(p["box"], gts)
                best_idx = int(np.argmax(ious))
                if float(ious[best_idx]) >= 0.50 and best_idx not in matched_gt_by_img[(cname, img_name)]:
                    total_tp += 1
                    matched_gt_by_img[(cname, img_name)].add(best_idx)
                else:
                    total_fp += 1

        total_fn = total_vru_gt - total_tp
        prec = total_tp / max(total_tp + total_fp, 1)
        rec = total_tp / max(total_vru_gt, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        small_tp = sum(
            1 for o in gt_objects
            if o["is_vru"] and o["size_group"] == "Small (<32²)"
            and o["gt_idx"] in matched_gt_by_img.get((o["class_name"], o["img_name"]), set())
        )
        small_rec = small_tp / max(total_small_vru_gt, 1)
        return {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "small_tp": small_tp,
            "small_recall": round(small_rec, 4),
        }

    # Pipeline 1: Frozen Full-Frame Baseline (tau=0.25)
    res_p1 = evaluate_detection_set(raw_preds_by_img, tau_cut=0.25)

    # Pipeline 2: Lower-Threshold Full-Frame (tau=0.15)
    res_p2 = evaluate_detection_set(raw_preds_by_img, tau_cut=0.15)

    # Pipeline 3: Full-Frame + Horizon Slice + Calibrated WBF (EXP-7B)
    fused_horizon_by_img = {}
    for img_name in pilot_images:
        ff = raw_preds_by_img[img_name]
        hz = horizon_preds_by_img[img_name]
        fused = calibrated_wbf_consensus_fusion(
            full_frame_dets=ff,
            crop_dets=hz,
            tau_ff=0.25,
            tau_isolated=0.35,
            tau_confirmed=0.15,
            iou_assoc=0.30,
        )
        fused_horizon_by_img[img_name] = fused

    res_p3 = evaluate_detection_set(fused_horizon_by_img, tau_cut=0.25)

    # Pipeline 4: Existing Phase 6 Active Foveation Baseline (from json)
    with open("experiments/phase6_system_eval/system_evaluation_report.json") as f:
        p6_data = json.load(f)
    fov_res = p6_data["foveation_wbf"]
    res_p4 = {
        "tp": fov_res["vru_tp"],
        "fp": fov_res["vru_fp"],
        "fn": fov_res["vru_fn"],
        "precision": fov_res["vru_operational_precision"],
        "recall": fov_res["vru_operational_recall"],
        "f1": fov_res["vru_operational_f1"],
        "small_tp": fov_res["size_groups"]["Small (<32²)"]["matched_vru_gt"],
        "small_recall": fov_res["size_groups"]["Small (<32²)"]["vru_recall"],
    }

    out = {
        "pipeline_1_frozen_raw_tau025": res_p1,
        "pipeline_2_raw_tau015": res_p2,
        "pipeline_3_horizon_slice_wbf": res_p3,
        "pipeline_4_active_foveation_baseline": res_p4,
    }

    out_p = Path("experiments/phase7_ablations/phase7_readiness_metrics.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    run_readiness_evaluation()
