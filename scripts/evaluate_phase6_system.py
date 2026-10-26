"""
RoadSentinel Phase 6 — System-Level Matched Evaluation Benchmark
Evaluates:
  1. Pipeline A: Raw YOLO11m (FullFrame-960, tau=0.25)
  2. Pipeline B: Active Foveation + Calibrated WBF (tau_ff=0.25, tau_iso=0.40, tau_conf=0.20, consensus boost +0.08)
  3. Pipeline C: Temporal Tracking (ByteTrack) with Sequence Boundary Resets

Evaluated on the matched 300-image pilot cohort from Phase 4B/4C.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
torch.set_num_threads(16)
from ultralytics import YOLO

from kaggle_jobs.calibration_benchmark.evaluate_phase4c_wbf import (
    PILOT_COHORT_IMAGE_SET,
    ALL_CLASS_NAMES,
    VRU_CLASS_NAMES,
    GT_CLASSES,
    IOU_THRESHOLDS,
    compute_box_iou,
    compute_box_iou_vec,
    compute_coco_ap_for_class,
    evaluate_operational_point,
    generate_uncertainty_crops,
    calibrated_wbf_consensus_fusion,
)
from src.foveation.cropper import ActiveCropper
from src.foveation.types import FoveationConfig, CropProposal, Detection


def run_benchmark():
    t_start = time.time()
    ckpt_path = Path("experiments/phase6_session2_error/runs/phase6_yolo11m_960_session1/weights/best.pt")
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

    with open(ckpt_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    val_img_dir = Path("bdd100k_dataset/val/images")
    val_lbl_dir = Path("bdd100k_dataset/val/labels")
    assert val_img_dir.exists(), f"Image dir not found: {val_img_dir}"
    assert val_lbl_dir.exists(), f"Label dir not found: {val_lbl_dir}"

    pilot_images = sorted(list(PILOT_COHORT_IMAGE_SET))
    assert len(pilot_images) == 300, f"Expected 300 pilot images, got {len(pilot_images)}"

    print(f"Loaded {len(pilot_images)} pilot cohort images.")
    print(f"Loading YOLO11m model from {ckpt_path}...")
    model = YOLO(str(ckpt_path))

    # Build dynamic class mapping based on model.names
    gt_name_to_id = {v: k for k, v in GT_CLASSES.items()}
    ckpt_to_gt_id = {int(k): gt_name_to_id[v] for k, v in model.names.items()}
    print(f"Model class mapping verified: {ckpt_to_gt_id}")

    # Load Ground Truth for Pilot Cohort
    print("Parsing pilot cohort ground truth...")
    img_w, img_h = 1280.0, 720.0
    gt_boxes_raw = {c: defaultdict(list) for c in ALL_CLASS_NAMES}
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
                box = [x1, y1, x2, y2]
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if area < 1024.0:
                    sz_grp = "Small (<32²)"
                elif area < 9216.0:
                    sz_grp = "Medium [32²-96²)"
                else:
                    sz_grp = "Large (≥96²)"

                gt_idx = len(gt_boxes_raw[cname][img_name])
                gt_boxes_raw[cname][img_name].append(box)
                gt_objects.append({
                    "img_name": img_name,
                    "class_name": cname,
                    "gt_idx": gt_idx,
                    "box": box,
                    "size_group": sz_grp,
                    "is_vru": cname in VRU_CLASS_NAMES,
                })

    # Convert GT boxes to numpy arrays
    gt_by_class_img = {
        c: {img: np.array(boxes, dtype=np.float32) for img, boxes in gt_boxes_raw[c].items()}
        for c in ALL_CLASS_NAMES
    }

    total_gt = len(gt_objects)
    total_vru_gt = sum(1 for o in gt_objects if o["is_vru"])
    print(f"Pilot Ground Truth: {total_gt} total objects, {total_vru_gt} VRUs.")

    # ------------------------------------------------------------------------
    # Pipeline 1: Raw YOLO11m Detector (FullFrame-960)
    # ------------------------------------------------------------------------
    print("\n--- Running Pipeline A: Raw YOLO11m (FullFrame-960) ---")
    raw_preds_by_class = {c: [] for c in ALL_CLASS_NAMES}
    raw_preds_by_img = defaultdict(list)
    raw_times = []

    # Process in batches for high throughput
    batch_size = 16
    for i in range(0, len(pilot_images), batch_size):
        b_files = pilot_images[i:i+batch_size]
        b_paths = [str(val_img_dir / f) for f in b_files]
        t0 = time.perf_counter()
        results = model.predict(b_paths, imgsz=960, conf=0.01, device="cpu", verbose=False)
        dt = (time.perf_counter() - t0) * 1000.0 / len(b_files)
        raw_times.extend([dt] * len(b_files))

        for img_name, r in zip(b_files, results):
            for b in r.boxes:
                ckpt_cid = int(b.cls.item())
                cname = model.names[ckpt_cid]
                conf = float(b.conf.item())
                coords = b.xyxy[0].cpu().numpy()
                pred_item = {
                    "img_name": img_name,
                    "class_name": cname,
                    "conf": conf,
                    "box": coords,
                    "source": "full_frame",
                }
                raw_preds_by_class[cname].append(pred_item)
                raw_preds_by_img[img_name].append(pred_item)

    # ------------------------------------------------------------------------
    # Pipeline 2: Active Foveation + Calibrated WBF Consensus Fusion
    # ------------------------------------------------------------------------
    print("\n--- Running Pipeline B: YOLO11m + Active Foveation + Calibrated WBF ---", flush=True)
    foveation_config = FoveationConfig()
    cropper = ActiveCropper(foveation_config)
    fused_preds_by_class = {c: [] for c in ALL_CLASS_NAMES}
    foveation_times = []
    total_crops_extracted = 0

    for idx, img_name in enumerate(pilot_images):
        t0 = time.perf_counter()
        img_path = str(val_img_dir / img_name)
        frame = cv2.imread(img_path)

        # Full frame detections for this image
        ff_dets = raw_preds_by_img[img_name]

        # Uncertainty crop generation
        crop_proposals = generate_uncertainty_crops(ff_dets, int(img_w), int(img_h), max_crops=2)
        total_crops_extracted += len(crop_proposals)

        crop_dets_remapped = []
        for c_idx, prop in enumerate(crop_proposals):
            crop_b = prop["crop_box"]
            sub_img, int_coords = cropper.extract_crop_image(frame, crop_b)
            if sub_img.shape[0] < 8 or sub_img.shape[1] < 8:
                continue

            crop_res = model.predict(sub_img, imgsz=640, conf=0.01, device="cpu", verbose=False)[0]
            cx1, cy1, cx2, cy2 = int_coords

            for b in crop_res.boxes:
                ckpt_cid = int(b.cls.item())
                cname = model.names[ckpt_cid]
                conf = float(b.conf.item())
                coords = b.xyxy[0].cpu().numpy()
                rx1, ry1, rx2, ry2 = coords

                # Discard edge artifacts
                margin = foveation_config.boundary_suppress_margin
                sub_w = cx2 - cx1
                sub_h = cy2 - cy1
                is_boundary = (rx1 <= margin and cx1 > margin) or \
                              (rx2 >= sub_w - margin and cx2 < img_w - margin) or \
                              (ry1 <= margin and cy1 > margin) or \
                              (ry2 >= sub_h - margin and cy2 < img_h - margin)
                if is_boundary and conf < 0.35:
                    continue

                ox1 = max(0.0, min(img_w, cx1 + rx1))
                oy1 = max(0.0, min(img_h, cy1 + ry1))
                ox2 = max(0.0, min(img_w, cx1 + rx2))
                oy2 = max(0.0, min(img_h, cy1 + ry2))

                if (ox2 - ox1) >= 2.0 and (oy2 - oy1) >= 2.0:
                    crop_dets_remapped.append({
                        "img_name": img_name,
                        "class_name": cname,
                        "conf": conf,
                        "box": np.array([ox1, oy1, ox2, oy2], dtype=np.float32),
                        "source": "crop",
                    })

        # WBF Consensus Fusion
        fused = calibrated_wbf_consensus_fusion(
            full_frame_dets=ff_dets,
            crop_dets=crop_dets_remapped,
            tau_ff=0.25,
            tau_isolated=0.40,
            tau_confirmed=0.20,
            iou_assoc=0.30,
        )

        for d in fused:
            fused_preds_by_class[d["class_name"]].append(d)

        dt = (time.perf_counter() - t0) * 1000.0
        foveation_times.append(dt)

    # ------------------------------------------------------------------------
    # Pipeline 3: Temporal Tracking (ByteTrack) with Sequence Boundary Resets
    # ------------------------------------------------------------------------
    print("\n--- Running Pipeline C: YOLO11m + ByteTrack (with Sequence Resets) ---")
    tracked_preds_by_class = {c: [] for c in ALL_CLASS_NAMES}
    tracking_times = []
    current_video_id = None

    for img_name in pilot_images:
        t0 = time.perf_counter()
        img_path = str(val_img_dir / img_name)
        video_id = img_name.split("-")[0]

        # Reset tracker state if video changes
        is_new_seq = (video_id != current_video_id)
        current_video_id = video_id

        track_res = model.track(
            img_path,
            tracker="bytetrack.yaml",
            imgsz=960,
            conf=0.01,
            device="cpu",
            persist=not is_new_seq,
            verbose=False,
        )[0]

        for b in track_res.boxes:
            ckpt_cid = int(b.cls.item())
            cname = model.names[ckpt_cid]
            conf = float(b.conf.item())
            coords = b.xyxy[0].cpu().numpy()
            track_id = int(b.id.item()) if b.id is not None else -1

            tracked_preds_by_class[cname].append({
                "img_name": img_name,
                "class_name": cname,
                "conf": conf,
                "box": coords,
                "track_id": track_id,
                "source": "bytetrack",
            })

        dt = (time.perf_counter() - t0) * 1000.0
        tracking_times.append(dt)

    # ------------------------------------------------------------------------
    # Metric Evaluation Function
    # ------------------------------------------------------------------------
    def evaluate_predictions(preds_dict: Dict[str, List[Dict[str, Any]]], name: str) -> Dict[str, Any]:
        target_imgs = set(pilot_images)
        per_class_results = {}
        all_aps_50 = []
        all_aps_50_95 = []
        vru_aps_50 = []
        vru_aps_50_95 = []

        total_vru_tp = 0
        total_vru_fp = 0
        total_vru_fn = 0
        vru_matched_gt_by_img = defaultdict(set)

        for cname in ALL_CLASS_NAMES:
            c_preds = [p for p in preds_dict[cname] if p["img_name"] in target_imgs]
            c_gt = gt_by_class_img[cname]
            n_gt = sum(len(boxes) for img, boxes in c_gt.items() if img in target_imgs)

            ap50, ap50_95 = compute_coco_ap_for_class(c_preds, c_gt, n_gt)
            all_aps_50.append(ap50)
            all_aps_50_95.append(ap50_95)

            tp, fp, fn, prec, rec, f1, matched_gt = evaluate_operational_point(
                c_preds, c_gt, n_gt, conf_thresh=0.25, iou_thresh=0.50
            )

            if cname in VRU_CLASS_NAMES:
                vru_aps_50.append(ap50)
                vru_aps_50_95.append(ap50_95)
                total_vru_tp += tp
                total_vru_fp += fp
                total_vru_fn += fn
                for img_k, s in matched_gt.items():
                    vru_matched_gt_by_img[(cname, img_k)].update(s)

            per_class_results[cname] = {
                "gt_count": n_gt,
                "ap50": round(ap50, 4),
                "ap50_95": round(ap50_95, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

        # Size Tier Recall Analysis
        size_tier_counts = {
            "Small (<32²)": {"total_gt": 0, "matched_gt": 0},
            "Medium [32²-96²)": {"total_gt": 0, "matched_gt": 0},
            "Large (≥96²)": {"total_gt": 0, "matched_gt": 0},
        }

        for gt_obj in gt_objects:
            if not gt_obj["is_vru"]:
                continue
            grp = gt_obj["size_group"]
            c_name = gt_obj["class_name"]
            i_name = gt_obj["img_name"]
            g_idx = gt_obj["gt_idx"]

            size_tier_counts[grp]["total_gt"] += 1
            if g_idx in vru_matched_gt_by_img.get((c_name, i_name), set()):
                size_tier_counts[grp]["matched_gt"] += 1

        size_group_results = {}
        for grp, cnts in size_tier_counts.items():
            r = cnts["matched_gt"] / max(cnts["total_gt"], 1)
            size_group_results[grp] = {
                "total_vru_gt": cnts["total_gt"],
                "matched_vru_gt": cnts["matched_gt"],
                "vru_recall": round(r, 4),
            }

        vru_prec = total_vru_tp / max(total_vru_tp + total_vru_fp, 1)
        vru_rec = total_vru_tp / max(total_vru_gt, 1)
        vru_f1 = 2 * vru_prec * vru_rec / max(vru_prec + vru_rec, 1e-8)

        return {
            "name": name,
            "overall_mAP_50": round(float(np.mean(all_aps_50)), 4),
            "overall_mAP_50_95": round(float(np.mean(all_aps_50_95)), 4),
            "vru_macro_mAP_50": round(float(np.mean(vru_aps_50)), 4),
            "vru_macro_mAP_50_95": round(float(np.mean(vru_aps_50_95)), 4),
            "vru_operational_precision": round(vru_prec, 4),
            "vru_operational_recall": round(vru_rec, 4),
            "vru_operational_f1": round(vru_f1, 4),
            "vru_tp": total_vru_tp,
            "vru_fp": total_vru_fp,
            "vru_fn": total_vru_fn,
            "size_groups": size_group_results,
            "per_class": per_class_results,
        }

    res_raw = evaluate_predictions(raw_preds_by_class, "Raw YOLO11m (FullFrame-960)")
    res_raw["latency_p50_ms"] = round(float(np.median(raw_times)), 1)
    res_raw["latency_mean_ms"] = round(float(np.mean(raw_times)), 1)

    res_fused = evaluate_predictions(fused_preds_by_class, "YOLO11m + Active Foveation + Calibrated WBF")
    res_fused["latency_p50_ms"] = round(float(np.median(foveation_times)), 1)
    res_fused["latency_mean_ms"] = round(float(np.mean(foveation_times)), 1)
    res_fused["total_crops"] = total_crops_extracted
    res_fused["crops_per_image"] = round(total_crops_extracted / len(pilot_images), 2)

    res_track = evaluate_predictions(tracked_preds_by_class, "YOLO11m + ByteTrack (Sequence Resets)")
    res_track["latency_p50_ms"] = round(float(np.median(tracking_times)), 1)
    res_track["latency_mean_ms"] = round(float(np.mean(tracking_times)), 1)

    out_payload = {
        "benchmark_timestamp": time.time(),
        "checkpoint_sha256": ckpt_sha256,
        "pilot_images_count": len(pilot_images),
        "total_vru_ground_truth": total_vru_gt,
        "raw_yolo11m": res_raw,
        "foveation_wbf": res_fused,
        "bytetrack": res_track,
    }

    out_file = Path("experiments/phase6_system_eval/system_evaluation_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("MATCHED PILOT BENCHMARK COMPLETE (300 IMAGES)")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Raw YOLO11m':<15} | {'Active Fov+WBF':<15} | {'ByteTrack':<15}")
    print("-" * 80)
    print(f"{'VRU Precision (tau=0.25)':<30} | {res_raw['vru_operational_precision']:<15.4f} | {res_fused['vru_operational_precision']:<15.4f} | {res_track['vru_operational_precision']:<15.4f}")
    print(f"{'VRU Recall (tau=0.25)':<30} | {res_raw['vru_operational_recall']:<15.4f} | {res_fused['vru_operational_recall']:<15.4f} | {res_track['vru_operational_recall']:<15.4f}")
    print(f"{'VRU F1-Score':<30} | {res_raw['vru_operational_f1']:<15.4f} | {res_fused['vru_operational_f1']:<15.4f} | {res_track['vru_operational_f1']:<15.4f}")
    print(f"{'VRU False Positives (FP)':<30} | {res_raw['vru_fp']:<15} | {res_fused['vru_fp']:<15} | {res_track['vru_fp']:<15}")
    print(f"{'Small VRU Recall (<32²)':<30} | {res_raw['size_groups']['Small (<32²)']['vru_recall']:<15.4f} | {res_fused['size_groups']['Small (<32²)']['vru_recall']:<15.4f} | {res_track['size_groups']['Small (<32²)']['vru_recall']:<15.4f}")
    print(f"{'VRU Macro mAP@50':<30} | {res_raw['vru_macro_mAP_50']:<15.4f} | {res_fused['vru_macro_mAP_50']:<15.4f} | {res_track['vru_macro_mAP_50']:<15.4f}")
    print(f"{'VRU Macro mAP@50-95':<30} | {res_raw['vru_macro_mAP_50_95']:<15.4f} | {res_fused['vru_macro_mAP_50_95']:<15.4f} | {res_track['vru_macro_mAP_50_95']:<15.4f}")
    print(f"{'Latency P50 (ms/img)':<30} | {res_raw['latency_p50_ms']:<15.1f} | {res_fused['latency_p50_ms']:<15.1f} | {res_track['latency_p50_ms']:<15.1f}")
    print("=" * 80)
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    run_benchmark()
