"""
RoadSentinel Phase 10F — Paired Image-Level Statistical Audit Script
Runs on Cohort 5 (Untouched Confirmation Cohort, 1,000 images) to compute:
1. Checkpoint integrity verification
2. Per-image matched statistics under Baseline (Flat 0.25, IoU 0.65) and Config E (Adaptive 0.22, VRU IoU 0.55)
3. Paired, image-level bootstrap distribution (B = 2,000 replicates, fixed seed = 42)
4. Empirical 95% Confidence Intervals for paired differences:
   - Δ Small-VRU Recall
   - Δ Overall VRU Recall
   - Δ VRU Precision
   - Δ VRU F1
   - Δ False Positives per Image
5. Precision floor audit against >= 72.0% requirement
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

from kaggle_jobs.pilot_300_native_eval.evaluate_pilot_native import (
    match_single_image, CLASS_NAMES, VRU_NAMES
)
from experiments.phase8_foveation.evaluate_phase8_foveation import load_ground_truth

CHAMPION_PATH = REPO_ROOT / "experiments" / "phase10_model_optimization" / "phase10b_extended_training" / "session4_artifacts" / "session4_best.pt"
BASELINE_PATH = REPO_ROOT / "scratch" / "kaggle_upload_weights" / "best.pt"
VAL_IMG_DIR = REPO_ROOT / "bdd100k_dataset" / "val" / "images"
VAL_LBL_DIR = REPO_ROOT / "bdd100k_dataset" / "val" / "labels"
OUT_DIR = REPO_ROOT / "experiments" / "phase10_model_optimization" / "phase10f_final_eval_readiness"

EXPECTED_CHAMPION_SHA = "4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd"
EXPECTED_BASELINE_SHA = "42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def apply_tighter_nms(preds: list, iou_thresh: float) -> list:
    """Applies greedy NMS suppression per-class for VRU candidates."""
    if not preds or iou_thresh >= 0.65:
        return preds

    by_class = defaultdict(list)
    for p in preds:
        by_class[p["class_name"]].append(p)

    survived = []
    for cname, c_preds in by_class.items():
        if cname not in VRU_NAMES or len(c_preds) <= 1:
            survived.extend(c_preds)
            continue

        c_preds.sort(key=lambda x: x["conf"], reverse=True)
        boxes = np.array([p["box"] for p in c_preds], dtype=np.float32)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        keep = []
        suppressed = np.zeros(len(c_preds), dtype=bool)
        for i in range(len(c_preds)):
            if suppressed[i]:
                continue
            keep.append(c_preds[i])
            xx1 = np.maximum(x1[i], x1[i+1:])
            yy1 = np.maximum(y1[i], y1[i+1:])
            xx2 = np.minimum(x2[i], x2[i+1:])
            yy2 = np.minimum(y2[i], y2[i+1:])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[i+1:] - inter
            iou = np.where(union > 0, inter / union, 0.0)
            suppressed[i+1:][iou >= iou_thresh] = True
        survived.extend(keep)

    return survived


def main():
    print("=" * 80)
    print("RoadSentinel Phase 10F — Paired Image-Level Statistical Audit")
    print("=" * 80)

    # 1. Integrity Audit
    champ_sha = compute_sha256(CHAMPION_PATH)
    base_sha = compute_sha256(BASELINE_PATH)
    print(f"Verified Champion Checkpoint: {CHAMPION_PATH}")
    print(f"  SHA-256: {champ_sha}")
    assert champ_sha == EXPECTED_CHAMPION_SHA, "Champion hash mismatch!"

    print(f"Verified Baseline Checkpoint: {BASELINE_PATH}")
    print(f"  SHA-256: {base_sha}")
    assert base_sha == EXPECTED_BASELINE_SHA, "Baseline hash mismatch!"
    print("[PASSED] Model checkpoints verified.")

    # 2. Load Cohort 5 Manifest (1,000 images)
    manifest_p = REPO_ROOT / "experiments" / "phase10_model_optimization" / "phase10e_policy_confirmation" / "cohort_manifest.csv"
    import pandas as pd
    m_df = pd.read_csv(manifest_p)
    c5_df = m_df[m_df["cohort_name"] == "cohort5_untouched_1000"]
    cohort5_filenames = c5_df["filename"].tolist()
    print(f"Loaded Cohort 5: {len(cohort5_filenames)} images from manifest.")

    gt_c5 = load_ground_truth(set(cohort5_filenames))
    n_gt = sum(len(objs) for objs in gt_c5.values())
    n_vru = sum(sum(1 for o in objs if o["is_vru"]) for objs in gt_c5.values())
    n_sm = sum(sum(1 for o in objs if o["is_vru"] and o["size_group"] == "Small (<32²)") for objs in gt_c5.values())
    print(f"Cohort 5 GT: {n_gt} objects | {n_vru} VRUs ({n_sm} Small VRUs)")

    # 3. Model Inference on Cohort 5
    raw_cache_path = OUT_DIR / "cohort5_raw_preds.json"
    if raw_cache_path.exists():
        print(f"Loading cached raw predictions from {raw_cache_path}...")
        with open(raw_cache_path, "r", encoding="utf-8") as f:
            raw_preds_by_img = json.load(f)
    else:
        print("\nRunning Batched Inference on Cohort 5 (Batch=16, 960px, conf=0.10, iou=0.65)...", flush=True)
        model = YOLO(str(CHAMPION_PATH))
        raw_preds_by_img = {}
        batch_size = 16
        t0 = time.perf_counter()
        for i in range(0, len(cohort5_filenames), batch_size):
            b_names = cohort5_filenames[i:i + batch_size]
            b_imgs = [cv2.imread(str(VAL_IMG_DIR / n)) for n in b_names]
            results = model.predict(source=b_imgs, imgsz=960, conf=0.10, iou=0.65, batch=batch_size, device="cpu", verbose=False)
            for n, r in zip(b_names, results):
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
                        "is_vru": cname in VRU_NAMES
                    })
                raw_preds_by_img[n] = preds

        t_infer = time.perf_counter() - t0
        print(f"Inference completed in {t_infer:.1f}s ({t_infer/len(cohort5_filenames)*1000.0:.1f} ms/img).")
        with open(raw_cache_path, "w", encoding="utf-8") as f:
            json.dump(raw_preds_by_img, f)
        print(f"Cached raw predictions to {raw_cache_path}")

    # 4. Extract Per-Image Statistics for Both Policies
    # Policy Baseline: Flat 0.25, VRU IoU 0.65
    # Policy Config E: Adaptive 0.22, VRU IoU 0.55
    base_tp, base_fp, base_fn = [], [], []
    base_sm_tp, base_sm_fn = [], []

    ce_tp, ce_fp, ce_fn = [], [], []
    ce_sm_tp, ce_sm_fn = [], []

    per_image_audit = []

    for name in cohort5_filenames:
        all_p = raw_preds_by_img[name]
        gt_list = gt_c5[name]

        # Baseline
        preds_base = [p for p in all_p if p["conf"] >= 0.25]
        st_base = match_single_image(preds_base, gt_list)

        btp = sum(st_base["class_stats"][c]["tp"] for c in VRU_NAMES)
        bfp = sum(st_base["class_stats"][c]["fp"] for c in VRU_NAMES)
        bfn = sum(st_base["class_stats"][c]["fn"] for c in VRU_NAMES)
        bsm_tp = st_base["vru_size_stats"]["Small (<32²)"]["tp"]
        bsm_fn = st_base["vru_size_stats"]["Small (<32²)"]["fn"]

        base_tp.append(btp)
        base_fp.append(bfp)
        base_fn.append(bfn)
        base_sm_tp.append(bsm_tp)
        base_sm_fn.append(bsm_fn)

        # Config E
        preds_ce = [p for p in all_p if p["conf"] >= (0.22 if p["is_vru"] else 0.25)]
        preds_ce = apply_tighter_nms(preds_ce, 0.55)
        st_ce = match_single_image(preds_ce, gt_list)

        etp = sum(st_ce["class_stats"][c]["tp"] for c in VRU_NAMES)
        efp = sum(st_ce["class_stats"][c]["fp"] for c in VRU_NAMES)
        efn = sum(st_ce["class_stats"][c]["fn"] for c in VRU_NAMES)
        esm_tp = st_ce["vru_size_stats"]["Small (<32²)"]["tp"]
        esm_fn = st_ce["vru_size_stats"]["Small (<32²)"]["fn"]

        ce_tp.append(etp)
        ce_fp.append(efp)
        ce_fn.append(efn)
        ce_sm_tp.append(esm_tp)
        ce_sm_fn.append(esm_fn)

        per_image_audit.append({
            "image": name,
            "baseline": {"tp": btp, "fp": bfp, "fn": bfn, "sm_tp": bsm_tp, "sm_fn": bsm_fn},
            "config_e": {"tp": etp, "fp": efp, "fn": efn, "sm_tp": esm_tp, "sm_fn": esm_fn},
            "delta_vru_tp": etp - btp,
            "delta_vru_fp": efp - bfp,
            "delta_sm_tp": esm_tp - bsm_tp
        })

    # Convert to NumPy arrays
    arr_b_tp = np.array(base_tp)
    arr_b_fp = np.array(base_fp)
    arr_b_fn = np.array(base_fn)
    arr_b_sm_tp = np.array(base_sm_tp)
    arr_b_sm_fn = np.array(base_sm_fn)

    arr_e_tp = np.array(ce_tp)
    arr_e_fp = np.array(ce_fp)
    arr_e_fn = np.array(ce_fn)
    arr_e_sm_tp = np.array(ce_sm_tp)
    arr_e_sm_fn = np.array(ce_sm_fn)

    # Point estimates
    pt_b_sm_rec = arr_b_sm_tp.sum() / (arr_b_sm_tp.sum() + arr_b_sm_fn.sum()) * 100.0
    pt_e_sm_rec = arr_e_sm_tp.sum() / (arr_e_sm_tp.sum() + arr_e_sm_fn.sum()) * 100.0
    pt_d_sm_rec = pt_e_sm_rec - pt_b_sm_rec

    pt_b_vru_rec = arr_b_tp.sum() / (arr_b_tp.sum() + arr_b_fn.sum()) * 100.0
    pt_e_vru_rec = arr_e_tp.sum() / (arr_e_tp.sum() + arr_e_fn.sum()) * 100.0
    pt_d_vru_rec = pt_e_vru_rec - pt_b_vru_rec

    pt_b_vru_prec = arr_b_tp.sum() / (arr_b_tp.sum() + arr_b_fp.sum()) * 100.0
    pt_e_vru_prec = arr_e_tp.sum() / (arr_e_tp.sum() + arr_e_fp.sum()) * 100.0
    pt_d_vru_prec = pt_e_vru_prec - pt_b_vru_prec

    pt_b_f1 = 2 * pt_b_vru_prec * pt_b_vru_rec / (pt_b_vru_prec + pt_b_vru_rec) / 100.0
    pt_e_f1 = 2 * pt_e_vru_prec * pt_e_vru_rec / (pt_e_vru_prec + pt_e_vru_rec) / 100.0
    pt_d_f1 = pt_e_f1 - pt_b_f1

    pt_b_fp_img = arr_b_fp.mean()
    pt_e_fp_img = arr_e_fp.mean()
    pt_d_fp_img = pt_e_fp_img - pt_b_fp_img

    print(f"\n--- Point Estimates (Cohort 5: 1,000 images) ---")
    print(f"Small-VRU Recall: Baseline={pt_b_sm_rec:.2f}%, Config E={pt_e_sm_rec:.2f}%, Δ = {pt_d_sm_rec:+.2f} pp")
    print(f"Overall VRU Recall: Baseline={pt_b_vru_rec:.2f}%, Config E={pt_e_vru_rec:.2f}%, Δ = {pt_d_vru_rec:+.2f} pp")
    print(f"VRU Precision: Baseline={pt_b_vru_prec:.2f}%, Config E={pt_e_vru_prec:.2f}%, Δ = {pt_d_vru_prec:+.2f} pp")
    print(f"VRU F1: Baseline={pt_b_f1:.4f}, Config E={pt_e_f1:.4f}, Δ = {pt_d_f1:+.4f}")
    print(f"Mean FP / Img: Baseline={pt_b_fp_img:.3f}, Config E={pt_e_fp_img:.3f}, Δ = {pt_d_fp_img:+.3f}")

    # 5. Paired Image-Level Bootstrap (B = 2,000 resamples)
    B = 2000
    N = len(cohort5_filenames)
    seed = 42
    rng = np.random.RandomState(seed)
    boot_indices = rng.randint(0, N, size=(B, N))

    # Resample both policies on identical images
    b_b_tp = np.take(arr_b_tp, boot_indices).sum(axis=1)
    b_b_fp = np.take(arr_b_fp, boot_indices).sum(axis=1)
    b_b_fn = np.take(arr_b_fn, boot_indices).sum(axis=1)
    b_b_sm_tp = np.take(arr_b_sm_tp, boot_indices).sum(axis=1)
    b_b_sm_fn = np.take(arr_b_sm_fn, boot_indices).sum(axis=1)

    b_e_tp = np.take(arr_e_tp, boot_indices).sum(axis=1)
    b_e_fp = np.take(arr_e_fp, boot_indices).sum(axis=1)
    b_e_fn = np.take(arr_e_fn, boot_indices).sum(axis=1)
    b_e_sm_tp = np.take(arr_e_sm_tp, boot_indices).sum(axis=1)
    b_e_sm_fn = np.take(arr_e_sm_fn, boot_indices).sum(axis=1)

    # Compute metrics per replicate
    b_b_rec = b_b_tp / (b_b_tp + b_b_fn) * 100.0
    b_e_rec = b_e_tp / (b_e_tp + b_e_fn) * 100.0

    b_b_prec = b_b_tp / (b_b_tp + b_b_fp) * 100.0
    b_e_prec = b_e_tp / (b_e_tp + b_e_fp) * 100.0

    b_b_f1 = 2 * (b_b_prec/100.0) * (b_b_rec/100.0) / ((b_b_prec/100.0) + (b_b_rec/100.0))
    b_e_f1 = 2 * (b_e_prec/100.0) * (b_e_rec/100.0) / ((b_e_prec/100.0) + (b_e_rec/100.0))

    b_b_sm_rec = b_b_sm_tp / (b_b_sm_tp + b_b_sm_fn) * 100.0
    b_e_sm_rec = b_e_sm_tp / (b_e_sm_tp + b_e_sm_fn) * 100.0

    b_b_fp_img = np.take(arr_b_fp, boot_indices).mean(axis=1)
    b_e_fp_img = np.take(arr_e_fp, boot_indices).mean(axis=1)

    # Paired differences per replicate
    delta_sm_rec = b_e_sm_rec - b_b_sm_rec
    delta_vru_rec = b_e_rec - b_b_rec
    delta_vru_prec = b_e_prec - b_b_prec
    delta_vru_f1 = b_e_f1 - b_b_f1
    delta_fp_img = b_e_fp_img - b_b_fp_img

    # 95% Percentile Confidence Intervals
    ci_delta_sm_rec = [round(float(np.percentile(delta_sm_rec, 2.5)), 2), round(float(np.percentile(delta_sm_rec, 97.5)), 2)]
    ci_delta_vru_rec = [round(float(np.percentile(delta_vru_rec, 2.5)), 2), round(float(np.percentile(delta_vru_rec, 97.5)), 2)]
    ci_delta_vru_prec = [round(float(np.percentile(delta_vru_prec, 2.5)), 2), round(float(np.percentile(delta_vru_prec, 97.5)), 2)]
    ci_delta_vru_f1 = [round(float(np.percentile(delta_vru_f1, 2.5)), 4), round(float(np.percentile(delta_vru_f1, 97.5)), 4)]
    ci_delta_fp_img = [round(float(np.percentile(delta_fp_img, 2.5)), 3), round(float(np.percentile(delta_fp_img, 97.5)), 3)]

    # Absolute CIs for Config E
    ci_e_prec = [round(float(np.percentile(b_e_prec, 2.5)), 2), round(float(np.percentile(b_e_prec, 97.5)), 2)]
    ci_e_sm_rec = [round(float(np.percentile(b_e_sm_rec, 2.5)), 2), round(float(np.percentile(b_e_sm_rec, 97.5)), 2)]

    # Precision floor audit
    pct_prec_above_72 = round(float(np.mean(b_e_prec >= 72.0) * 100.0), 2)
    min_prec_observed = round(float(np.min(b_e_prec)), 2)

    print(f"\n--- Paired Bootstrap 95% Confidence Intervals (B = 2,000, seed = {seed}) ---")
    print(f"Δ Small-VRU Recall: Point = {pt_d_sm_rec:+.2f} pp, 95% CI = [{ci_delta_sm_rec[0]:+.2f}, {ci_delta_sm_rec[1]:+.2f}] pp (P(Δ > 0) = {np.mean(delta_sm_rec > 0)*100:.1f}%)")
    print(f"Δ Overall VRU Recall: Point = {pt_d_vru_rec:+.2f} pp, 95% CI = [{ci_delta_vru_rec[0]:+.2f}, {ci_delta_vru_rec[1]:+.2f}] pp (P(Δ > 0) = {np.mean(delta_vru_rec > 0)*100:.1f}%)")
    print(f"Δ VRU Precision: Point = {pt_d_vru_prec:+.2f} pp, 95% CI = [{ci_delta_vru_prec[0]:+.2f}, {ci_delta_vru_prec[1]:+.2f}] pp")
    print(f"Δ VRU F1: Point = {pt_d_f1:+.4f}, 95% CI = [{ci_delta_vru_f1[0]:+.4f}, {ci_delta_vru_f1[1]:+.4f}] (P(Δ > 0) = {np.mean(delta_vru_f1 > 0)*100:.1f}%)")
    print(f"Δ Mean FP / Img: Point = {pt_d_fp_img:+.3f}, 95% CI = [{ci_delta_fp_img[0]:+.3f}, {ci_delta_fp_img[1]:+.3f}]")
    print(f"\nConfig E Absolute Precision: Point = {pt_e_vru_prec:.2f}%, 95% CI = [{ci_e_prec[0]}, {ci_e_prec[1]}%]")
    print(f"Fraction of bootstrap iterations with Precision >= 72.0%: {pct_prec_above_72}% (Min observed = {min_prec_observed}%)")

    # 6. Save JSON results
    out_results = {
        "integrity": {
            "champion_sha256": champ_sha,
            "baseline_sha256": base_sha,
            "champion_verified": champ_sha == EXPECTED_CHAMPION_SHA,
            "baseline_verified": base_sha == EXPECTED_BASELINE_SHA
        },
        "cohort": {
            "name": "cohort5_untouched_1000",
            "image_count": len(cohort5_filenames),
            "total_gt_objects": n_gt,
            "vru_gt_objects": n_vru,
            "small_vru_gt_objects": n_sm
        },
        "point_estimates": {
            "baseline": {
                "small_vru_recall": round(pt_b_sm_rec, 2),
                "vru_recall": round(pt_b_vru_rec, 2),
                "vru_precision": round(pt_b_vru_prec, 2),
                "vru_f1": round(pt_b_f1, 4),
                "mean_fp_per_img": round(pt_b_fp_img, 3),
                "tp": int(arr_b_tp.sum()),
                "fp": int(arr_b_fp.sum()),
                "fn": int(arr_b_fn.sum()),
                "small_tp": int(arr_b_sm_tp.sum()),
                "small_fn": int(arr_b_sm_fn.sum())
            },
            "config_e": {
                "small_vru_recall": round(pt_e_sm_rec, 2),
                "vru_recall": round(pt_e_vru_rec, 2),
                "vru_precision": round(pt_e_vru_prec, 2),
                "vru_f1": round(pt_e_f1, 4),
                "mean_fp_per_img": round(pt_e_fp_img, 3),
                "tp": int(arr_e_tp.sum()),
                "fp": int(arr_e_fp.sum()),
                "fn": int(arr_e_fn.sum()),
                "small_tp": int(arr_e_sm_tp.sum()),
                "small_fn": int(arr_e_sm_fn.sum())
            },
            "paired_deltas": {
                "delta_small_vru_recall": round(pt_d_sm_rec, 2),
                "delta_vru_recall": round(pt_d_vru_rec, 2),
                "delta_vru_precision": round(pt_d_vru_prec, 2),
                "delta_vru_f1": round(pt_d_f1, 4),
                "delta_fp_per_img": round(pt_d_fp_img, 3)
            }
        },
        "paired_bootstrap": {
            "replicates": B,
            "seed": seed,
            "sampling_unit": "image_level",
            "interval_method": "percentile_95",
            "ci_95_paired_deltas": {
                "delta_small_vru_recall": ci_delta_sm_rec,
                "delta_vru_recall": ci_delta_vru_rec,
                "delta_vru_precision": ci_delta_vru_prec,
                "delta_vru_f1": ci_delta_vru_f1,
                "delta_fp_per_img": ci_delta_fp_img
            },
            "config_e_absolute_ci_95": {
                "vru_precision": ci_e_prec,
                "small_vru_recall": ci_e_sm_rec
            },
            "precision_floor_audit_72": {
                "pct_replicates_ge_72": pct_prec_above_72,
                "min_replicate_precision": min_prec_observed,
                "observed_point_margin_pp": round(pt_e_vru_prec - 72.0, 2),
                "observed_ci_lower_margin_pp": round(ci_e_prec[0] - 72.0, 2)
            }
        }
    }

    out_json_path = OUT_DIR / "paired_statistical_results.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_results, f, indent=2)
    print(f"\n[SAVED] Paired Statistical Audit JSON: {out_json_path}")


if __name__ == "__main__":
    main()
