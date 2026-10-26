"""
Empirical Threshold Sweep Audit on 300-image Development Cohort.
Evaluates Raw YOLO11m and Foveation across tau in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30].
Quantifies:
- Raw proposal recall
- Precision, FP, FN
- Small-VRU recall (<32^2)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, os.getcwd())

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
torch.set_num_threads(16)
from ultralytics import YOLO

from kaggle_jobs.calibration_benchmark.evaluate_phase4c_wbf import (
    PILOT_COHORT_IMAGE_SET,
    ALL_CLASS_NAMES,
    VRU_CLASS_NAMES,
    GT_CLASSES,
    compute_box_iou_vec,
)

def run_threshold_sweep():
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

    # Run Raw model inference down to conf=0.01 (batched)
    model = YOLO(str(ckpt_path))
    raw_preds = defaultdict(list)
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
                raw_preds[cname].append({
                    "img_name": img_name,
                    "conf": conf,
                    "box": coords,
                })

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    sweep_results = []

    for tau in thresholds:
        total_tp, total_fp = 0, 0
        matched_gt_by_img = defaultdict(set)

        for cname in VRU_CLASS_NAMES:
            c_preds = [p for p in raw_preds[cname] if p["conf"] >= tau]
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

        # Small VRU recall
        small_tp = sum(
            1 for o in gt_objects
            if o["is_vru"] and o["size_group"] == "Small (<32²)"
            and o["gt_idx"] in matched_gt_by_img.get((o["class_name"], o["img_name"]), set())
        )
        small_rec = small_tp / max(total_small_vru_gt, 1)

        sweep_results.append({
            "tau": tau,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "small_vru_tp": small_tp,
            "small_vru_recall": round(small_rec, 4),
        })

    print(json.dumps(sweep_results, indent=2))
    out_p = Path("scratch/threshold_sweep_results.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

if __name__ == "__main__":
    run_threshold_sweep()
