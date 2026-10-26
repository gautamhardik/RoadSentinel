"""
RoadSentinel Phase 4B — Step 2 & Step 3: Targeted Diagnostic Inference & Failure Atlas Generation
Runs full-frame (960px) and active foveation crop (640px) inference on the 300-image diagnostic cohort.
Saves immutable raw candidates (conf >= 0.01), executes baseline fusion, matches ground truth,
and renders the visual failure atlas.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CHECKPOINT_TO_GT_ID: Dict[int, int] = {
    0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 9, 6: 6, 7: 5, 8: 7, 9: 8
}
GT_CLASSES: Dict[int, str] = {
    0: "person", 1: "rider", 2: "car", 3: "bus", 4: "truck",
    5: "bike", 6: "motor", 7: "traffic light", 8: "traffic sign", 9: "train"
}
VRU_CLASS_NAMES = {"person", "rider", "bike", "motor"}


def compute_box_iou(boxA, boxB) -> float:
    xi1 = max(boxA[0], boxB[0])
    yi1 = max(boxA[1], boxB[1])
    xi2 = min(boxA[2], boxB[2])
    yi2 = min(boxA[3], boxB[3])
    iw = max(0.0, xi2 - xi1)
    ih = max(0.0, yi2 - yi1)
    inter = iw * ih
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / max(union, 1e-8)


def expand_crop_box(box, img_w, img_h, margin=0.30, min_size=160):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_w = max(bw * margin, (min_size - bw) / 2.0 if bw < min_size else 0.0)
    pad_h = max(bh * margin, (min_size - bh) / 2.0 if bh < min_size else 0.0)
    cx1 = max(0.0, x1 - pad_w)
    cy1 = max(0.0, y1 - pad_h)
    cx2 = min(float(img_w), x2 + pad_w)
    cy2 = min(float(img_h), y2 + pad_h)
    return (cx1, cy1, cx2, cy2)


def generate_uncertainty_crops(detections, img_w, img_h, max_crops=2):
    candidates = []
    for d in detections:
        cname = d["class_name"]
        conf = d["conf"]
        box = d["box_full"]
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        area = bw * bh
        is_vru = cname in VRU_CLASS_NAMES

        # Signal 1: Low Confidence VRU (0.10 <= conf < 0.40)
        if is_vru and (0.10 <= conf < 0.40):
            urgency = (0.40 - conf) * 1.5
            crop_b = expand_crop_box(box, img_w, img_h, margin=0.30)
            candidates.append({"crop_box": crop_b, "reason": "low_conf_vru", "score": urgency})
        # Signal 2: Small VRU (area < 1024 px^2)
        elif is_vru and (area < 1024.0):
            urgency = (1024.0 - area) / 1024.0
            crop_b = expand_crop_box(box, img_w, img_h, margin=0.35)
            candidates.append({"crop_box": crop_b, "reason": "small_vru", "score": urgency})
        # Signal 3: Boundary Truncation
        elif is_vru and (box[0] < 40 or box[2] > img_w - 40 or box[1] < 30 or box[3] > img_h - 30):
            crop_b = expand_crop_box(box, img_w, img_h, margin=0.30)
            candidates.append({"crop_box": crop_b, "reason": "boundary_vru", "score": 0.65})

    if not candidates:
        return []

    candidates.sort(key=lambda x: x["score"], reverse=True)
    merged = []
    for c in candidates:
        matched = False
        for m in merged:
            if compute_box_iou(c["crop_box"], m["crop_box"]) > 0.30:
                ux1 = min(c["crop_box"][0], m["crop_box"][0])
                uy1 = min(c["crop_box"][1], m["crop_box"][1])
                ux2 = max(c["crop_box"][2], m["crop_box"][2])
                uy2 = max(c["crop_box"][3], m["crop_box"][3])
                m["crop_box"] = (ux1, uy1, ux2, uy2)
                m["score"] = max(c["score"], m["score"])
                m["reason"] = f"{m['reason']}+{c['reason']}"
                matched = True
                break
        if not matched:
            merged.append(c)

    return merged[:max_crops]


def baseline_fuse_and_deduplicate(ff_dets, cr_dets, iou_thresh=0.50):
    fused = []
    ff_by_class = defaultdict(list)
    cr_by_class = defaultdict(list)

    for d in ff_dets:
        ff_by_class[d["class_name"]].append(d)
    for d in cr_dets:
        cr_by_class[d["class_name"]].append(d)

    all_cnames = set(ff_by_class.keys()).union(cr_by_class.keys())

    for cname in all_cnames:
        ff_list = ff_by_class[cname]
        cr_list = cr_by_class[cname]
        matched_crop_indices = set()

        for ff_d in ff_list:
            best_idx = -1
            best_iou = 0.0
            for idx, cr_d in enumerate(cr_list):
                if idx in matched_crop_indices:
                    continue
                iou = compute_box_iou(ff_d["box_full"], cr_d["box_full"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_thresh and best_idx >= 0:
                matched_crop_indices.add(best_idx)
                cr_d = cr_list[best_idx]
                w_ff = ff_d["conf"]
                w_cr = cr_d["conf"] * 1.2
                w_sum = max(w_ff + w_cr, 1e-6)
                b_ff = ff_d["box_full"]
                b_cr = cr_d["box_full"]
                fused_box = [
                    float((b_ff[0] * w_ff + b_cr[0] * w_cr) / w_sum),
                    float((b_ff[1] * w_ff + b_cr[1] * w_cr) / w_sum),
                    float((b_ff[2] * w_ff + b_cr[2] * w_cr) / w_sum),
                    float((b_ff[3] * w_ff + b_cr[3] * w_cr) / w_sum),
                ]
                fused_conf = float(min(0.99, max(ff_d["conf"], cr_d["conf"]) + 0.05 * min(ff_d["conf"], cr_d["conf"])))
                fused.append({
                    "class_name": cname,
                    "conf": fused_conf,
                    "box_full": fused_box,
                    "source": "fused",
                    "parent_ff": ff_d,
                    "parent_crop": cr_d,
                })
            else:
                fused.append(ff_d)

        for idx, cr_d in enumerate(cr_list):
            if idx not in matched_crop_indices:
                fused.append(cr_d)

    return fused


def main():
    repo_root = Path(".")
    ckpt_path = repo_root / "weights" / "bdd100k_yolo11s.pt"
    val_img_dir = repo_root / "bdd100k_dataset" / "val" / "images"
    val_lbl_dir = repo_root / "bdd100k_dataset" / "val" / "labels"
    out_dir = repo_root / "experiments" / "phase4_pilot"
    atlas_dir = out_dir / "visual_atlas"
    manifest_path = out_dir / "cohort_manifest.json"

    assert manifest_path.exists(), f"Cohort manifest {manifest_path} missing! Run build_phase4b_cohort.py first."
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cohort_images = manifest["images"]
    print(f"Loaded cohort of {len(cohort_images)} images from manifest.")

    device = "cpu"
    print(f"Loading YOLO model from {ckpt_path} on {device}...")
    model = YOLO(str(ckpt_path))

    # Output storage
    raw_candidates_path = out_dir / "raw_candidates.jsonl"
    f_raw = open(raw_candidates_path, "w", encoding="utf-8")

    all_eval_records = []
    atlas_index = []

    img_w, img_h = 1280.0, 720.0
    CONF_FLOOR = 0.01  # Documented floor to retain full spectrum for sweeps

    t0_all = time.perf_counter()
    print("Beginning diagnostic inference across cohort...")

    for i, item in enumerate(cohort_images):
        fname = item["image"]
        stratum = item["stratum"]
        img_p = val_img_dir / fname
        lbl_p = val_lbl_dir / (Path(fname).stem + ".txt")

        img = cv2.imread(str(img_p))
        if img is None:
            print(f"Warning: Failed to load {fname}")
            continue

        h_orig, w_orig = img.shape[:2]

        # 1. Parse Ground Truth
        gts = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    parts = line.strip().split()
                    if not parts: continue
                    cid = int(parts[0])
                    cname = GT_CLASSES.get(cid)
                    xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                    x1 = (xc - bw / 2.0) * img_w
                    y1 = (yc - bh / 2.0) * img_h
                    x2 = (xc + bw / 2.0) * img_w
                    y2 = (yc + bh / 2.0) * img_h
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    sz = "Small" if area < 1024.0 else ("Medium" if area < 9216.0 else "Large")
                    gts.append({
                        "gt_idx": idx,
                        "class_id": cid,
                        "class_name": cname,
                        "is_vru": (cname in VRU_CLASS_NAMES),
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "area": float(area),
                        "size_group": sz,
                    })

        # 2. Full-Frame Inference (imgsz=960, conf=0.01)
        res_ff = model.predict(source=img, imgsz=960, conf=CONF_FLOOR, device=device, verbose=False)
        ff_candidates = []
        for b in res_ff[0].boxes:
            cid = CHECKPOINT_TO_GT_ID[int(b.cls.item())]
            cname = GT_CLASSES[cid]
            coords = [float(x) for x in b.xyxy[0].cpu().numpy()]
            conf = float(b.conf.item())
            cand = {
                "image_id": fname,
                "stratum": stratum,
                "source": "full_frame",
                "class_id": cid,
                "class_name": cname,
                "conf": conf,
                "box_full": coords,
                "crop_id": None,
                "crop_box": None,
                "survives_baseline_thresh": (conf >= 0.25),
            }
            ff_candidates.append(cand)
            f_raw.write(json.dumps(cand) + "\n")

        # 3. Active Foveation Crop Generation
        proposals = generate_uncertainty_crops(ff_candidates, int(img_w), int(img_h), max_crops=2)
        crop_candidates = []

        if proposals:
            for c_idx, prop in enumerate(proposals):
                cx1, cy1, cx2, cy2 = [int(round(v)) for v in prop["crop_box"]]
                cx1, cy1 = max(0, cx1), max(0, cy1)
                cx2, cy2 = min(int(img_w), cx2), min(int(img_h), cy2)
                if (cx2 - cx1) < 10 or (cy2 - cy1) < 10:
                    continue

                crop_patch = img[cy1:cy2, cx1:cx2]
                res_crop = model.predict(source=crop_patch, imgsz=640, conf=CONF_FLOOR, device=device, verbose=False)

                for sb in res_crop[0].boxes:
                    scid = CHECKPOINT_TO_GT_ID[int(sb.cls.item())]
                    scname = GT_CLASSES[scid]
                    scoords = [float(x) for x in sb.xyxy[0].cpu().numpy()]
                    sconf = float(sb.conf.item())
                    # Remap to full frame
                    ox1 = float(cx1 + scoords[0])
                    oy1 = float(cy1 + scoords[1])
                    ox2 = float(cx1 + scoords[2])
                    oy2 = float(cy1 + scoords[3])
                    cr_cand = {
                        "image_id": fname,
                        "stratum": stratum,
                        "source": "crop",
                        "class_id": scid,
                        "class_name": scname,
                        "conf": sconf,
                        "box_full": [ox1, oy1, ox2, oy2],
                        "crop_id": c_idx,
                        "crop_box": [cx1, cy1, cx2, cy2],
                        "crop_reason": prop["reason"],
                        "crop_local_coords": scoords,
                        "survives_baseline_thresh": (sconf >= 0.25),
                    }
                    crop_candidates.append(cr_cand)
                    f_raw.write(json.dumps(cr_cand) + "\n")

        # 4. Production Baseline Fusion (at tau = 0.25)
        ff_prod = [d for d in ff_candidates if d["conf"] >= 0.25]
        cr_prod = [d for d in crop_candidates if d["conf"] >= 0.25]
        prod_fused = baseline_fuse_and_deduplicate(ff_prod, cr_prod, iou_thresh=0.50)

        # 5. Diagnostic Matching against Ground Truth
        # Separate VRU matching
        vru_gts = [g for g in gts if g["is_vru"]]
        matched_gt_indices = set()
        vru_dets = [d for d in prod_fused if d["class_name"] in VRU_CLASS_NAMES]
        # Sort by confidence
        vru_dets.sort(key=lambda x: x["conf"], reverse=True)

        tp_dets = []
        fp_dets = []

        for d in vru_dets:
            best_idx = -1
            best_iou = 0.0
            for g_idx, g in enumerate(vru_gts):
                if g_idx in matched_gt_indices:
                    continue
                if g["class_name"] != d["class_name"]:
                    continue
                iou = compute_box_iou(d["box_full"], g["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = g_idx
            if best_iou >= 0.50 and best_idx >= 0:
                matched_gt_indices.add(best_idx)
                tp_dets.append({"det": d, "gt": vru_gts[best_idx], "iou": best_iou})
            else:
                fp_dets.append({"det": d, "best_iou": best_iou})

        fn_gts = [g for idx, g in enumerate(vru_gts) if idx not in matched_gt_indices]

        # Record evaluation metadata
        img_rec = {
            "image": fname,
            "stratum": stratum,
            "ground_truth": gts,
            "vru_gt_count": len(vru_gts),
            "num_ff_candidates": len(ff_candidates),
            "num_crop_candidates": len(crop_candidates),
            "proposals": proposals,
            "baseline_tp_count": len(tp_dets),
            "baseline_fp_count": len(fp_dets),
            "baseline_fn_count": len(fn_gts),
            "tp_records": tp_dets,
            "fp_records": fp_dets,
            "fn_records": fn_gts,
        }
        all_eval_records.append(img_rec)

        # 6. Render Visual Failure Atlas (Sample every image)
        vis_img = img.copy()

        # A. Draw Proposals in Magenta
        for p in proposals:
            px1, py1, px2, py2 = [int(round(v)) for v in p["crop_box"]]
            cv2.rectangle(vis_img, (px1, py1), (px2, py2), (255, 0, 255), 2)
            cv2.putText(vis_img, f"Crop: {p['reason']}", (px1 + 5, py1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # B. Draw Ground Truth in Green
        for g in vru_gts:
            gx1, gy1, gx2, gy2 = [int(round(v)) for v in g["box"]]
            is_fn = g in fn_gts
            color = (0, 255, 255) if is_fn else (0, 255, 0)  # Cyan if missed, Green if caught
            label = f"GT: {g['class_name']} ({g['size_group']})"
            if is_fn: label += " [MISSED]"
            cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), color, 2)
            cv2.putText(vis_img, label, (gx1, max(15, gy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # C. Draw Predictions
        for td in tp_dets:
            d = td["det"]
            dx1, dy1, dx2, dy2 = [int(round(v)) for v in d["box_full"]]
            src_lbl = "CROP" if d["source"] == "crop" else ("FUSED" if d["source"] == "fused" else "FF")
            lbl = f"{d['class_name']} {d['conf']:.2f} ({src_lbl}) [TP]"
            cv2.rectangle(vis_img, (dx1, dy1), (dx2, dy2), (0, 255, 0), 1)
            cv2.putText(vis_img, lbl, (dx1, min(int(img_h) - 5, dy2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        for fd in fp_dets:
            d = fd["det"]
            dx1, dy1, dx2, dy2 = [int(round(v)) for v in d["box_full"]]
            src_lbl = "CROP" if d["source"] == "crop" else ("FUSED" if d["source"] == "fused" else "FF")
            lbl = f"{d['class_name']} {d['conf']:.2f} ({src_lbl}) [FP]"
            cv2.rectangle(vis_img, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)  # Red for FP
            cv2.putText(vis_img, lbl, (dx1, min(int(img_h) - 5, dy2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # Overlay Stratum & Image Name
        header = f"{fname} | {stratum} | VRU GT: {len(vru_gts)} | TP: {len(tp_dets)} | FP: {len(fp_dets)} | FN: {len(fn_gts)}"
        cv2.putText(vis_img, header, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(vis_img, header, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

        # Save atlas visualization
        atlas_save_path = atlas_dir / f"atlas_{i:03d}_{Path(fname).stem}.jpg"
        cv2.imwrite(str(atlas_save_path), vis_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Categorize primary failure
        primary_diag = "True Negative / Control"
        if len(tp_dets) > 0 and any(t["det"]["source"] in ("crop", "fused") and t["gt"]["size_group"] == "Small" for t in tp_dets):
            primary_diag = "Recovered Small VRU via Crop"
        elif len(fp_dets) > 0 and any(f["det"]["source"] == "crop" for f in fp_dets):
            primary_diag = "Crop False Alarm (Hallucination)"
        elif len(fn_gts) > 0:
            primary_diag = "Persistent Miss / Low Activation"
        elif len(fp_dets) > 0:
            primary_diag = "Full-Frame False Positive"

        atlas_index.append({
            "index": i,
            "image": fname,
            "stratum": stratum,
            "atlas_image_path": str(atlas_save_path.relative_to(repo_root)),
            "vru_gt_count": len(vru_gts),
            "tp": len(tp_dets),
            "fp": len(fp_dets),
            "fn": len(fn_gts),
            "proposals_count": len(proposals),
            "primary_diagnosis": primary_diag,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(cohort_images):
            elapsed = time.perf_counter() - t0_all
            print(f"  Processed {i + 1}/{len(cohort_images)} images ({elapsed:.1f}s elapsed)...")

    f_raw.close()
    print(f"Raw immutable candidate predictions written to: {raw_candidates_path}")

    with open(out_dir / "cohort_evaluation_records.json", "w", encoding="utf-8") as f:
        json.dump(all_eval_records, f, indent=2)

    with open(out_dir / "atlas_index.json", "w", encoding="utf-8") as f:
        json.dump(atlas_index, f, indent=2)

    total_time = time.perf_counter() - t0_all
    print(f"\nCompleted diagnostic inference on {len(cohort_images)} images in {total_time:.1f}s ({total_time / len(cohort_images):.2f}s/image).")
    print(f"Visual Failure Atlas rendered to: {atlas_dir}")


if __name__ == "__main__":
    main()
