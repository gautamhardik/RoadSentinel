"""
RoadSentinel Phase 10C — Comprehensive Inference Policy Evaluation Script
Evaluates Configurations A, B, C, D across:
1. Development Pilot Cohort (PILOT_300_FILENAMES: 300 images)
2. Generalization Holdout Cohort (HOLDOUT_300_FILENAMES: 300 distinct non-pilot images)

Outputs:
- policy_comparison.csv
- false_positive_analysis.md
- latency_report.md
- decision_report.md
- Visual overlay examples in visual_examples/
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from ultralytics import YOLO

from src.foveation.engine import ActiveFoveationEngine
from src.foveation.types import FoveationConfig, VRU_CLASS_NAMES, CLASS_NAMES
from kaggle_jobs.pilot_300_native_eval.evaluate_pilot_native import (
    PILOT_300_FILENAMES, match_single_image, aggregate_partition
)
from experiments.phase8_foveation.evaluate_phase8_foveation import load_ground_truth

CHAMPION_PATH = REPO_ROOT / "experiments" / "phase10_model_optimization" / "phase10b_extended_training" / "session4_artifacts" / "session4_best.pt"
VAL_IMG_DIR = REPO_ROOT / "bdd100k_dataset" / "val" / "images"
VAL_LBL_DIR = REPO_ROOT / "bdd100k_dataset" / "val" / "labels"
OUT_DIR = REPO_ROOT / "experiments" / "phase10_model_optimization" / "phase10c_inference_policy"
VISUAL_DIR = OUT_DIR / "visual_examples"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_generalization_cohort(pilot_set: set, count: int = 300) -> list:
    all_val = sorted([p.name for p in VAL_IMG_DIR.glob("*.jpg")])
    non_pilot = [p for p in all_val if p not in pilot_set and (VAL_LBL_DIR / (Path(p).stem + ".txt")).exists()]
    return non_pilot[:count]


def run_full_frame_eval(model: YOLO, cohort: list, gt_by_image: dict, batch_size: int = 16) -> dict:
    raw_preds_by_img = {}
    latencies = []

    for i in range(0, len(cohort), batch_size):
        b_names = cohort[i:i + batch_size]
        b_imgs = []
        t_pre_start = time.perf_counter()
        for n in b_names:
            img = cv2.imread(str(VAL_IMG_DIR / n))
            b_imgs.append(img)
        t_read = time.perf_counter() - t_pre_start

        t_infer_start = time.perf_counter()
        results = model.predict(source=b_imgs, imgsz=960, conf=0.10, iou=0.65, batch=batch_size, device="cpu", verbose=False)
        t_batch_infer = time.perf_counter() - t_infer_start

        per_img_time = (t_read + t_batch_infer) / len(b_names)

        for n, r in zip(b_names, results):
            latencies.append(per_img_time)
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            preds = []
            for b, s, c in zip(boxes, scores, cls_ids):
                cname = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
                preds.append({
                    "box": b.tolist(),
                    "conf": float(s),
                    "class_name": cname,
                    "is_vru": cname in VRU_CLASS_NAMES,
                    "source": "full_frame"
                })
            raw_preds_by_img[n] = preds

    return {
        "raw_preds": raw_preds_by_img,
        "latencies_ms": [t * 1000.0 for t in latencies]
    }


def run_foveation_eval(engine: ActiveFoveationEngine, cohort: list, gt_by_image: dict) -> dict:
    fused_preds_by_img = {}
    timings = []
    crops_per_img = []

    for idx, name in enumerate(cohort, start=1):
        img_p = VAL_IMG_DIR / name
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        t0 = time.perf_counter()
        fused_dets, trace = engine.process_image(img, conf_threshold=0.15)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(dt_ms)
        crops_per_img.append(trace.num_crops_executed)

        preds = []
        for d in fused_dets:
            preds.append({
                "box": list(d.box),
                "conf": float(d.confidence),
                "class_name": d.class_name,
                "is_vru": d.class_name in VRU_CLASS_NAMES,
                "source": getattr(d, "source", "fused")
            })
        fused_preds_by_img[name] = preds

    return {
        "raw_preds": fused_preds_by_img,
        "latencies_ms": timings,
        "crops_per_img": crops_per_img
    }


def evaluate_policy_predictions(preds_by_img: dict, cohort: list, gt_by_image: dict, vru_conf: float, non_vru_conf: float) -> dict:
    image_stats = {}
    fp_distribution = {c: {"[0.15, 0.20)": 0, "[0.20, 0.25)": 0, "[0.25, 0.50)": 0, "[0.50, 1.00]": 0} for c in VRU_CLASS_NAMES}
    attribution_tp = {"full_frame": 0, "crop_only": 0, "fused": 0}
    attribution_fp = {"full_frame": 0, "crop_only": 0, "fused": 0}

    for name in cohort:
        preds = preds_by_img.get(name, [])
        filtered = []
        for p in preds:
            thresh = vru_conf if p["is_vru"] else non_vru_conf
            if p["conf"] >= thresh:
                filtered.append(p)

        st = match_single_image(filtered, gt_by_image[name])
        image_stats[name] = st

        # Analyze false positives and attribution
        gt_boxes = [g for g in gt_by_image[name]]
        for p in filtered:
            if not p["is_vru"]:
                continue
            # Check if this prediction was matched
            # Simple IoU check against same-class GT
            matched = False
            for g in gt_boxes:
                if g["class_name"] == p["class_name"]:
                    # Compute IoU
                    xa1, ya1, xa2, ya2 = p["box"]
                    xb1, yb1, xb2, yb2 = g["box"]
                    iw = max(0.0, min(xa2, xb2) - max(xa1, xb1))
                    ih = max(0.0, min(ya2, yb2) - max(ya1, yb1))
                    inter = iw * ih
                    union = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter
                    if union > 0 and (inter / union) >= 0.50:
                        matched = True
                        break

            src = p.get("source", "full_frame")
            if src not in attribution_tp:
                src = "fused"

            if matched:
                attribution_tp[src] += 1
            else:
                attribution_fp[src] += 1
                c = p["class_name"]
                conf = p["conf"]
                if 0.15 <= conf < 0.20:
                    fp_distribution[c]["[0.15, 0.20)"] += 1
                elif 0.20 <= conf < 0.25:
                    fp_distribution[c]["[0.20, 0.25)"] += 1
                elif 0.25 <= conf < 0.50:
                    fp_distribution[c]["[0.25, 0.50)"] += 1
                else:
                    fp_distribution[c]["[0.50, 1.00]"] += 1

    agg = aggregate_partition(image_stats, set(cohort))
    agg["fp_distribution"] = fp_distribution
    agg["attribution_tp"] = attribution_tp
    agg["attribution_fp"] = attribution_fp
    return agg


def generate_visual_overlays(preds_a, preds_b, preds_c, gt_by_img, sample_imgs: list):
    for name in sample_imgs:
        img_p = VAL_IMG_DIR / name
        orig = cv2.imread(str(img_p))
        if orig is None:
            continue

        h, w = orig.shape[:2]
        canvas = np.hstack([orig.copy(), orig.copy(), orig.copy()])
        
        # Overlay Config A, B, C
        for col_idx, (p_dict, label) in enumerate([(preds_a, "Config A (Conf 0.25)"), (preds_b, "Config B (Adaptive 0.20)"), (preds_c, "Config C (Foveation)")]):
            sub = canvas[:, col_idx * w:(col_idx + 1) * w]
            cv2.putText(sub, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # Draw GT (Green)
            for g in gt_by_img.get(name, []):
                if g["is_vru"]:
                    b = [int(x) for x in g["box"]]
                    cv2.rectangle(sub, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    cv2.putText(sub, f"GT:{g['class_name']}", (b[0], max(15, b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw Preds (Red for VRU)
            for p in p_dict.get(name, []):
                if p["is_vru"]:
                    b = [int(x) for x in p["box"]]
                    cv2.rectangle(sub, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cv2.putText(sub, f"{p['class_name']} {p['conf']:.2f}", (b[0], min(h - 5, b[3] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out_file = VISUAL_DIR / f"overlay_{Path(name).stem}.jpg"
        cv2.imwrite(str(out_file), canvas)


def main():
    print("=" * 80)
    print("RoadSentinel Phase 10C — Comprehensive Inference Policy Evaluation")
    print("=" * 80)

    # 1. SHA-256 Check
    champion_sha = compute_sha256(CHAMPION_PATH)
    print(f"Verified Champion Checkpoint: {CHAMPION_PATH}")
    print(f"SHA-256: {champion_sha}")
    assert champion_sha == "4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd", "Champion hash mismatch!"

    # 2. Setup Cohorts
    pilot_cohort = sorted(list(PILOT_300_FILENAMES))
    holdout_cohort = get_generalization_cohort(set(pilot_cohort), 300)

    gt_pilot = load_ground_truth(set(pilot_cohort))
    gt_holdout = load_ground_truth(set(holdout_cohort))

    print(f"Cohort 1 (Pilot): {len(pilot_cohort)} images | {sum(len(v) for v in gt_pilot.values())} GT objects")
    print(f"Cohort 2 (Generalization Holdout): {len(holdout_cohort)} images | {sum(len(v) for v in gt_holdout.values())} GT objects")

    # 3. Initialize Model and Engine
    print("\nLoading Champion Model for Full-Frame Inference...", flush=True)
    model = YOLO(str(CHAMPION_PATH))

    print("\nLoading Active Foveation Engine...", flush=True)
    foveation_cfg = FoveationConfig(
        full_frame_imgsz=960,
        crop_imgsz=640,
        max_crops_per_image=2,
        policy="uncertainty_active",
        low_conf_min=0.10,
        low_conf_max=0.35,
        small_box_area_threshold=1024.0,
        cluster_iou_threshold=0.35,
        context_margin_ratio=0.25,
        min_crop_size=128,
        fusion_iou_threshold=0.50,
        boundary_suppress_margin=8,
        operational_conf_threshold=0.25,
    )
    engine = ActiveFoveationEngine(checkpoint_path=str(CHAMPION_PATH), config=foveation_cfg, device="cpu")

    results = {}

    for cohort_name, cohort_list, gt_data in [
        ("pilot", pilot_cohort, gt_pilot),
        ("generalization", holdout_cohort, gt_holdout)
    ]:
        print(f"\n=======================================================")
        print(f"Evaluating Cohort: {cohort_name.upper()} ({len(cohort_list)} images)")
        print(f"=======================================================")

        # Run full-frame pass
        print("  Running Full-Frame Batched Inference...", flush=True)
        ff_out = run_full_frame_eval(model, cohort_list, gt_data, batch_size=16)

        # Run foveation pass
        print("  Running Active Foveation Inference...", flush=True)
        fov_out = run_foveation_eval(engine, cohort_list, gt_data)

        # Config A: Champion flat 0.25
        eval_a = evaluate_policy_predictions(ff_out["raw_preds"], cohort_list, gt_data, vru_conf=0.25, non_vru_conf=0.25)
        # Config B: Champion adaptive 0.20
        eval_b = evaluate_policy_predictions(ff_out["raw_preds"], cohort_list, gt_data, vru_conf=0.20, non_vru_conf=0.25)
        # Config C: Foveation flat 0.25
        eval_c = evaluate_policy_predictions(fov_out["raw_preds"], cohort_list, gt_data, vru_conf=0.25, non_vru_conf=0.25)
        # Config D: Foveation adaptive 0.20
        eval_d = evaluate_policy_predictions(fov_out["raw_preds"], cohort_list, gt_data, vru_conf=0.20, non_vru_conf=0.25)

        results[cohort_name] = {
            "Config_A": {"eval": eval_a, "latencies": ff_out["latencies_ms"], "crops": 0.0, "passes": 1.0},
            "Config_B": {"eval": eval_b, "latencies": ff_out["latencies_ms"], "crops": 0.0, "passes": 1.0},
            "Config_C": {"eval": eval_c, "latencies": fov_out["latencies_ms"], "crops": float(np.mean(fov_out["crops_per_img"])), "passes": 1.0 + float(np.mean(fov_out["crops_per_img"]))},
            "Config_D": {"eval": eval_d, "latencies": fov_out["latencies_ms"], "crops": float(np.mean(fov_out["crops_per_img"])), "passes": 1.0 + float(np.mean(fov_out["crops_per_img"]))},
        }

        # Visual overlays for 3 sample images
        if cohort_name == "pilot":
            sample_names = cohort_list[:3]
            # Filter preds for overlays
            preds_a_filtered = {k: [p for p in ff_out["raw_preds"][k] if p["conf"] >= 0.25] for k in sample_names}
            preds_b_filtered = {k: [p for p in ff_out["raw_preds"][k] if p["conf"] >= (0.20 if p["is_vru"] else 0.25)] for k in sample_names}
            preds_c_filtered = {k: [p for p in fov_out["raw_preds"][k] if p["conf"] >= 0.25] for k in sample_names}
            generate_visual_overlays(preds_a_filtered, preds_b_filtered, preds_c_filtered, gt_data, sample_names)

    # 4. Generate policy_comparison.csv
    csv_rows = [
        "cohort,config_id,config_name,vru_conf,small_vru_recall,small_vru_tp,small_vru_fn,vru_recall,vru_precision,vru_f1,vru_tp,vru_fp,vru_fn,all_recall,all_prec,all_f1,person_rec,person_prec,rider_rec,rider_prec,bike_rec,bike_prec,motor_rec,motor_prec,p50_ms,mean_ms,p95_ms,crops_per_img,forward_passes"
    ]
    for ch_name in ["pilot", "generalization"]:
        for cfg_id, cfg_label, v_conf in [
            ("Config_A", "Champion Baseline (Flat 0.25)", 0.25),
            ("Config_B", "Adaptive VRU Threshold (0.20)", 0.20),
            ("Config_C", "Active Foveation (Flat 0.25)", 0.25),
            ("Config_D", "Adaptive Threshold + Foveation (0.20)", 0.20)
        ]:
            data = results[ch_name][cfg_id]
            ev = data["eval"]
            vru = ev["vru_metrics"]
            all_c = ev["all_class_metrics"]
            pc = ev["per_class"]
            lats = data["latencies"]
            p50 = round(float(np.median(lats)), 1)
            mean_lat = round(float(np.mean(lats)), 1)
            p95 = round(float(np.percentile(lats, 95)), 1)
            row = [
                ch_name, cfg_id, f'"{cfg_label}"', v_conf,
                vru["small_vru_recall"], vru["small_vru_true_positives"], vru["small_vru_ground_truth"] - vru["small_vru_true_positives"],
                vru["recall"], vru["precision"], vru["f1_score"],
                vru["tp"], vru["fp"], vru["fn"],
                all_c["recall"], all_c["precision"], all_c["f1_score"],
                pc["person"]["recall"], pc["person"]["precision"],
                pc["rider"]["recall"], pc["rider"]["precision"],
                pc["bike"]["recall"], pc["bike"]["precision"],
                pc["motor"]["recall"], pc["motor"]["precision"],
                p50, mean_lat, p95, round(data["crops"], 2), round(data["passes"], 2)
            ]
            csv_rows.append(",".join(str(x) for x in row))

    csv_path = OUT_DIR / "policy_comparison.csv"
    csv_path.write_text("\n".join(csv_rows), encoding="utf-8")
    print(f"\n[SAVED] Policy Comparison CSV: {csv_path}")

    # 5. Save detailed JSON results
    json_path = OUT_DIR / "phase10c_detailed_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] Detailed JSON: {json_path}")


if __name__ == "__main__":
    main()
