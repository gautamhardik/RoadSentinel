"""
RoadSentinel Phase 10E — Policy Confirmation and Controlled Ablation Script

Executes:
1. Model Checksum & Quarantine Verification
2. Cohort Partitioning & Manifest Generation (Cohort 4: 1,000 images, Cohort 5: 1,000 images)
3. Stage A — Threshold Effect Confirmation on Cohort 4 (Flat 0.25, Adaptive 0.22, Adaptive 0.20 with clustered bootstrap 95% CIs)
4. Stage B — Duplicate Suppression NMS Ablation on Cohort 4 (IoU 0.65, 0.60, 0.55)
5. Stage D — Final Policy Confirmation on Untouched Cohort 5 (1,000 images with bootstrap 95% CIs)
6. Error Audit & Side-by-Side Visual Overlays in visual_examples/
7. Generation of threshold_comparison.csv, nms_ablation.csv, and phase10e_detailed_results.json
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
    PILOT_300_FILENAMES, match_single_image, aggregate_partition, CLASS_NAMES, VRU_NAMES
)
from experiments.phase8_foveation.evaluate_phase8_foveation import load_ground_truth

CHAMPION_PATH = REPO_ROOT / "experiments" / "phase10_model_optimization" / "phase10b_extended_training" / "session4_artifacts" / "session4_best.pt"
BASELINE_PATH = REPO_ROOT / "scratch" / "kaggle_upload_weights" / "best.pt"
VAL_IMG_DIR = REPO_ROOT / "bdd100k_dataset" / "val" / "images"
VAL_LBL_DIR = REPO_ROOT / "bdd100k_dataset" / "val" / "labels"
OUT_DIR = REPO_ROOT / "experiments" / "phase10_model_optimization" / "phase10e_policy_confirmation"
VISUAL_DIR = OUT_DIR / "visual_examples"

EXPECTED_CHAMPION_SHA = "4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd"
EXPECTED_BASELINE_SHA = "42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def setup_phase10e_cohorts():
    """Builds Cohort 4 (Confirmation) and Cohort 5 (Untouched Confirmation)."""
    all_val = sorted([p.name for p in VAL_IMG_DIR.glob("*.jpg") if (VAL_LBL_DIR / (p.stem + ".txt")).exists()])
    pilot_set = set(PILOT_300_FILENAMES)
    non_pilot = [p for p in all_val if p not in pilot_set]

    # Quarantined previous slices
    gen_300 = non_pilot[:300]
    val_1000 = non_pilot[300:1300]
    prev_used = pilot_set | set(gen_300) | set(val_1000)

    # Phase 10E fresh cohorts
    cohort4_confirm_1000 = non_pilot[1300:2300]
    cohort5_untouched_1000 = non_pilot[2300:3300]

    c4_set = set(cohort4_confirm_1000)
    c5_set = set(cohort5_untouched_1000)

    # Verify disjointness
    assert len(prev_used & c4_set) == 0, "Cohort 4 overlaps with previous phases!"
    assert len(prev_used & c5_set) == 0, "Cohort 5 overlaps with previous phases!"
    assert len(c4_set & c5_set) == 0, "Cohort 4 and Cohort 5 overlap!"

    return {
        "cohort4_confirm_1000": cohort4_confirm_1000,
        "cohort5_untouched_1000": cohort5_untouched_1000
    }


def build_cohort_manifest(cohorts: dict, gt_by_cohort: dict) -> Path:
    """Writes cohort_manifest.csv."""
    manifest_path = OUT_DIR / "cohort_manifest.csv"
    rows = [
        "image_id,filename,cohort_name,total_gt_objects,vru_gt_objects,small_vru_gt_objects,medium_vru_gt_objects,large_vru_gt_objects"
    ]
    for ch_name, img_list in cohorts.items():
        gt_data = gt_by_cohort[ch_name]
        for fname in img_list:
            objs = gt_data.get(fname, [])
            n_tot = len(objs)
            vru_objs = [o for o in objs if o["is_vru"]]
            n_vru = len(vru_objs)
            n_sm = sum(1 for o in vru_objs if o["size_group"] == "Small (<32²)")
            n_med = sum(1 for o in vru_objs if o["size_group"] == "Medium [32²-96²)")
            n_lg = sum(1 for o in vru_objs if o["size_group"] == "Large (≥96²)")
            rows.append(f"{Path(fname).stem},{fname},{ch_name},{n_tot},{n_vru},{n_sm},{n_med},{n_lg}")

    manifest_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"[SAVED] Cohort Manifest: {manifest_path} ({len(rows)-1} images)")
    return manifest_path


def run_batched_inference(model: YOLO, cohort: list, batch_size: int = 16) -> dict:
    """Runs batched inference with conf=0.10 and iou=0.65."""
    raw_preds_by_img = {}
    latencies = []

    for i in range(0, len(cohort), batch_size):
        b_names = cohort[i:i + batch_size]
        b_imgs = [cv2.imread(str(VAL_IMG_DIR / n)) for n in b_names]

        t_start = time.perf_counter()
        results = model.predict(source=b_imgs, imgsz=960, conf=0.10, iou=0.65, batch=batch_size, device="cpu", verbose=False)
        t_batch = (time.perf_counter() - t_start) * 1000.0
        per_img_lat = t_batch / len(b_names)

        for n, r in zip(b_names, results):
            latencies.append(per_img_lat)
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

    return {
        "raw_preds": raw_preds_by_img,
        "latencies_ms": latencies
    }


def apply_tighter_nms(preds: list, iou_thresh: float) -> list:
    """Applies greedy NMS suppression per-class for VRU candidates."""
    if not preds or iou_thresh >= 0.65:
        return preds

    by_class = defaultdict(list)
    for p in preds:
        by_class[p["class_name"]].append(p)

    survived = []
    for cname, c_preds in by_class.items():
        # Only apply tighter NMS to VRU classes
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


def evaluate_cohort_policy(raw_preds_by_img: dict, cohort: list, gt_by_img: dict, vru_conf: float, non_vru_conf: float = 0.25, vru_nms_iou: float = 0.65) -> dict:
    """Evaluates policy and collects image-level statistics."""
    image_stats = {}
    img_vru_tp = []
    img_vru_fp = []
    img_vru_fn = []
    img_sm_vru_tp = []
    img_sm_vru_fn = []

    for name in cohort:
        preds = raw_preds_by_img.get(name, [])
        filtered = [
            p for p in preds if p["conf"] >= (vru_conf if p["is_vru"] else non_vru_conf)
        ]
        if vru_nms_iou < 0.65:
            filtered = apply_tighter_nms(filtered, vru_nms_iou)

        st = match_single_image(filtered, gt_by_img[name])
        image_stats[name] = st

        vru_tp_i = sum(st["class_stats"][c]["tp"] for c in VRU_NAMES)
        vru_fp_i = sum(st["class_stats"][c]["fp"] for c in VRU_NAMES)
        vru_fn_i = sum(st["class_stats"][c]["fn"] for c in VRU_NAMES)
        sm_tp_i = st["vru_size_stats"]["Small (<32²)"]["tp"]
        sm_fn_i = st["vru_size_stats"]["Small (<32²)"]["fn"]

        img_vru_tp.append(vru_tp_i)
        img_vru_fp.append(vru_fp_i)
        img_vru_fn.append(vru_fn_i)
        img_sm_vru_tp.append(sm_tp_i)
        img_sm_vru_fn.append(sm_fn_i)

    agg = aggregate_partition(image_stats, set(cohort))

    # Clustered Bootstrap (1,000 resamples)
    B = 1000
    N = len(cohort)
    rng = np.random.RandomState(42)
    boot_idx = rng.randint(0, N, size=(B, N))

    arr_tp = np.array(img_vru_tp)
    arr_fp = np.array(img_vru_fp)
    arr_fn = np.array(img_vru_fn)
    arr_sm_tp = np.array(img_sm_vru_tp)
    arr_sm_fn = np.array(img_sm_vru_fn)

    b_tp = np.take(arr_tp, boot_idx).sum(axis=1)
    b_fp = np.take(arr_fp, boot_idx).sum(axis=1)
    b_fn = np.take(arr_fn, boot_idx).sum(axis=1)
    b_sm_tp = np.take(arr_sm_tp, boot_idx).sum(axis=1)
    b_sm_fn = np.take(arr_sm_fn, boot_idx).sum(axis=1)

    b_prec = np.where((b_tp + b_fp) > 0, b_tp / (b_tp + b_fp), 0.0)
    b_rec = np.where((b_tp + b_fn) > 0, b_tp / (b_tp + b_fn), 0.0)
    b_f1 = np.where((b_prec + b_rec) > 0, 2 * b_prec * b_rec / (b_prec + b_rec), 0.0)
    b_sm_rec = np.where((b_sm_tp + b_sm_fn) > 0, b_sm_tp / (b_sm_tp + b_sm_fn), 0.0)

    agg["ci_95"] = {
        "vru_precision": [round(float(np.percentile(b_prec, 2.5) * 100.0), 2), round(float(np.percentile(b_prec, 97.5) * 100.0), 2)],
        "vru_recall": [round(float(np.percentile(b_rec, 2.5) * 100.0), 2), round(float(np.percentile(b_rec, 97.5) * 100.0), 2)],
        "vru_f1": [round(float(np.percentile(b_f1, 2.5)), 4), round(float(np.percentile(b_f1, 97.5)), 4)],
        "small_vru_recall": [round(float(np.percentile(b_sm_rec, 2.5) * 100.0), 2), round(float(np.percentile(b_sm_rec, 97.5) * 100.0), 2)],
    }
    agg["per_image_fp"] = {n: fp for n, fp in zip(cohort, img_vru_fp)}
    return agg


def render_visual_overlays(preds_dict: dict, gt_by_img: dict, selected_images: list, v_conf: float, prefix: str):
    """Renders side-by-side overlays."""
    for name in selected_images:
        img_p = VAL_IMG_DIR / name
        orig = cv2.imread(str(img_p))
        if orig is None:
            continue
        h, w = orig.shape[:2]
        canvas = np.hstack([orig.copy(), orig.copy()])

        for col_idx, (c_thresh, label) in enumerate([(0.25, "Baseline: Flat 0.25"), (v_conf, f"Adaptive: Conf {v_conf:.2f}")]):
            sub = canvas[:, col_idx * w:(col_idx + 1) * w]
            cv2.rectangle(sub, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.putText(sub, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Ground Truth
            for g in gt_by_img.get(name, []):
                if g["is_vru"]:
                    b = [int(x) for x in g["box"]]
                    cv2.rectangle(sub, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    cv2.putText(sub, f"GT:{g['class_name']}", (b[0], max(15, b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # Preds
            for p in preds_dict.get(name, []):
                thresh = c_thresh if p["is_vru"] else 0.25
                if p["conf"] >= thresh and p["is_vru"]:
                    b = [int(x) for x in p["box"]]
                    color = (255, 0, 255) if p["conf"] < 0.25 else (0, 0, 255)
                    tag = f"{p['class_name']} {p['conf']:.2f}"
                    cv2.rectangle(sub, (b[0], b[1]), (b[2], b[3]), color, 2)
                    cv2.putText(sub, tag, (b[0], min(h - 5, b[3] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imwrite(str(VISUAL_DIR / f"{prefix}_{Path(name).stem}.jpg"), canvas)


def main():
    print("=" * 80)
    print("RoadSentinel Phase 10E — Policy Confirmation and Controlled Ablation")
    print("=" * 80)

    # 1. SHA-256 Checkpoint Verification
    champ_sha = compute_sha256(CHAMPION_PATH)
    base_sha = compute_sha256(BASELINE_PATH)
    print(f"Verified Champion Checkpoint: {CHAMPION_PATH}")
    print(f"  SHA-256: {champ_sha}")
    assert champ_sha == EXPECTED_CHAMPION_SHA, "Champion hash mismatch!"
    print(f"Verified Frozen Baseline Checkpoint: {BASELINE_PATH}")
    print(f"  SHA-256: {base_sha}")
    assert base_sha == EXPECTED_BASELINE_SHA, "Baseline hash mismatch!"
    print("[PASSED] Model checkpoints intact.")

    # 2. Setup Cohorts
    cohorts = setup_phase10e_cohorts()
    gt_by_cohort = {k: load_ground_truth(set(v)) for k, v in cohorts.items()}
    for k, v in cohorts.items():
        n_gt = sum(len(objs) for objs in gt_by_cohort[k].values())
        n_vru = sum(sum(1 for o in objs if o["is_vru"]) for objs in gt_by_cohort[k].values())
        n_sm = sum(sum(1 for o in objs if o["is_vru"] and o["size_group"] == "Small (<32²)") for objs in gt_by_cohort[k].values())
        print(f"Cohort '{k}': {len(v)} images | {n_gt} GT objects | {n_vru} VRUs ({n_sm} Small VRUs)")

    build_cohort_manifest(cohorts, gt_by_cohort)

    # 3. Model Initialization
    print("\nLoading Champion Model for Batched Inference...", flush=True)
    model = YOLO(str(CHAMPION_PATH))

    # =========================================================================
    # STAGE A: Threshold Confirmation on Cohort 4 (1,000 images)
    # =========================================================================
    print("\n=======================================================")
    print("STAGE A: Threshold Effect Confirmation on Cohort 4 (1,000 images)")
    print("=======================================================", flush=True)

    c4_imgs = cohorts["cohort4_confirm_1000"]
    gt_c4 = gt_by_cohort["cohort4_confirm_1000"]

    t0_c4 = time.perf_counter()
    raw_c4 = run_batched_inference(model, c4_imgs, batch_size=16)
    c4_inference_time = time.perf_counter() - t0_c4
    print(f"Cohort 4 inference complete in {c4_inference_time:.1f}s ({c4_inference_time/1000.0*1000:.1f} ms/img).")

    stage_a_evals = {}
    threshold_policies = [
        ("Flat_0.25", "Baseline (Flat 0.25)", 0.25),
        ("Adaptive_0.22", "Adaptive VRU 0.22", 0.22),
        ("Adaptive_0.20", "Adaptive VRU 0.20", 0.20),
    ]

    thresh_rows = [
        "policy_id,policy_name,vru_conf,small_vru_recall,small_ci_low,small_ci_high,vru_recall,rec_ci_low,rec_ci_high,vru_precision,prec_ci_low,prec_ci_high,vru_f1,vru_tp,vru_fp,vru_fn,mean_fp_per_img,pct_zero_fp,precision_gate_72"
    ]

    for pid, pname, conf in threshold_policies:
        ev = evaluate_cohort_policy(raw_c4["raw_preds"], c4_imgs, gt_c4, vru_conf=conf, non_vru_conf=0.25, vru_nms_iou=0.65)
        stage_a_evals[pid] = ev
        vm = ev["vru_metrics"]
        ci = ev["ci_95"]
        fp_vals = list(ev["per_image_fp"].values())
        mean_fp = round(float(np.mean(fp_vals)), 3)
        pct_zero = round(float(np.mean([1.0 if x == 0 else 0.0 for x in fp_vals]) * 100.0), 1)
        prec_pass = "PASS" if vm["precision"] >= 72.0 else "FAIL"

        row = [
            pid, f'"{pname}"', conf,
            vm["small_vru_recall"], ci["small_vru_recall"][0], ci["small_vru_recall"][1],
            vm["recall"], ci["vru_recall"][0], ci["vru_recall"][1],
            vm["precision"], ci["vru_precision"][0], ci["vru_precision"][1],
            vm["f1_score"],
            vm["tp"], vm["fp"], vm["fn"],
            mean_fp, pct_zero, prec_pass
        ]
        thresh_rows.append(",".join(str(x) for x in row))
        print(f"  {pname}: Small-VRU Rec={vm['small_vru_recall']}% [{ci['small_vru_recall'][0]}, {ci['small_vru_recall'][1]}%], VRU Prec={vm['precision']}% [{ci['vru_precision'][0]}, {ci['vru_precision'][1]}%], Gate={prec_pass}")

    thresh_csv_path = OUT_DIR / "threshold_comparison.csv"
    thresh_csv_path.write_text("\n".join(thresh_rows), encoding="utf-8")
    print(f"[SAVED] Threshold Comparison CSV: {thresh_csv_path}")

    # Determine Winning Candidate from Stage A under >= 72.0% Precision Floor
    eligible = [(pid, pname, conf, stage_a_evals[pid]) for pid, pname, conf in threshold_policies if stage_a_evals[pid]["vru_metrics"]["precision"] >= 72.0]
    if eligible:
        # Pick the eligible candidate with highest small-vru recall
        winning_candidate = max(eligible, key=lambda x: x[3]["vru_metrics"]["small_vru_recall"])
        selected_thresh = winning_candidate[2]
        selected_pid = winning_candidate[0]
        print(f"\nStage A Selected Policy: {winning_candidate[1]} (conf={selected_thresh}) with {winning_candidate[3]['vru_metrics']['small_vru_recall']}% small-VRU recall and {winning_candidate[3]['vru_metrics']['precision']}% precision.")
    else:
        print("\nNeither adaptive threshold met >= 72.0% precision! Falling back to Flat 0.25.")
        selected_thresh = 0.25
        selected_pid = "Flat_0.25"

    # =========================================================================
    # STAGE B: Duplicate Suppression NMS Ablation on Cohort 4
    # =========================================================================
    print("\n=======================================================")
    print("STAGE B: Duplicate Suppression NMS Ablation on Cohort 4")
    print("=======================================================", flush=True)

    nms_rows = [
        "policy_id,vru_conf,vru_nms_iou,vru_tp,vru_fp,vru_fn,small_vru_recall,vru_recall,vru_precision,vru_f1,tp_change_vs_65,fp_eliminated_vs_65"
    ]
    nms_evals = {}

    # Test NMS IoU at 0.65, 0.60, 0.55 on the candidate threshold (and 0.20 if different)
    thresholds_to_test = list(set([selected_thresh, 0.20, 0.22]))

    for c_th in sorted(thresholds_to_test, reverse=True):
        baseline_eval_65 = None
        for iou_th in [0.65, 0.60, 0.55]:
            ev_nms = evaluate_cohort_policy(raw_c4["raw_preds"], c4_imgs, gt_c4, vru_conf=c_th, non_vru_conf=0.25, vru_nms_iou=iou_th)
            vm = ev_nms["vru_metrics"]
            nms_evals[f"conf_{c_th}_iou_{iou_th}"] = ev_nms

            if iou_th == 0.65:
                baseline_eval_65 = ev_nms
                tp_change = 0
                fp_elim = 0
            else:
                tp_change = vm["tp"] - baseline_eval_65["vru_metrics"]["tp"]
                fp_elim = baseline_eval_65["vru_metrics"]["fp"] - vm["fp"]

            row = [
                f"conf_{c_th}_iou_{iou_th}", c_th, iou_th,
                vm["tp"], vm["fp"], vm["fn"],
                vm["small_vru_recall"], vm["recall"], vm["precision"], vm["f1_score"],
                tp_change, fp_elim
            ]
            nms_rows.append(",".join(str(x) for x in row))
            print(f"  Conf {c_th} | NMS IoU {iou_th}: TP={vm['tp']} (Δ={tp_change}), FP={vm['fp']} (Elim={fp_elim}), Prec={vm['precision']}%, SmallRec={vm['small_vru_recall']}%")

    nms_csv_path = OUT_DIR / "nms_ablation.csv"
    nms_csv_path.write_text("\n".join(nms_rows), encoding="utf-8")
    print(f"[SAVED] NMS Ablation CSV: {nms_csv_path}")

    # Determine optimal NMS setting for selected policy
    # Check if tighter NMS improves precision while keeping precision >= 72.0% and minimal TP loss
    best_iou = 0.65
    candidate_65 = nms_evals[f"conf_{selected_thresh}_iou_0.65"]["vru_metrics"]
    for test_iou in [0.60, 0.55]:
        c_metrics = nms_evals[f"conf_{selected_thresh}_iou_{test_iou}"]["vru_metrics"]
        # Check if precision improves and recall is not degraded by > 0.5 pp
        if c_metrics["precision"] > candidate_65["precision"] and c_metrics["small_vru_recall"] >= candidate_65["small_vru_recall"] - 0.5:
            if c_metrics["f1_score"] >= candidate_65["f1_score"]:
                best_iou = test_iou

    print(f"\nFinalized Stage A/B Policy: VRU Conf={selected_thresh}, VRU NMS IoU={best_iou}")

    # =========================================================================
    # STAGE D: Confirmation on Untouched Cohort 5 (1,000 images)
    # =========================================================================
    print("\n=======================================================")
    print("STAGE D: Policy Confirmation on Untouched Cohort 5 (1,000 images)")
    print("=======================================================", flush=True)

    c5_imgs = cohorts["cohort5_untouched_1000"]
    gt_c5 = gt_by_cohort["cohort5_untouched_1000"]

    t0_c5 = time.perf_counter()
    raw_c5 = run_batched_inference(model, c5_imgs, batch_size=16)
    c5_inference_time = time.perf_counter() - t0_c5
    print(f"Cohort 5 inference complete in {c5_inference_time:.1f}s ({c5_inference_time/1000.0*1000:.1f} ms/img).")

    # Evaluate Baseline (Flat 0.25) vs Final Selected Policy on Cohort 5
    eval_c5_baseline = evaluate_cohort_policy(raw_c5["raw_preds"], c5_imgs, gt_c5, vru_conf=0.25, non_vru_conf=0.25, vru_nms_iou=0.65)
    eval_c5_selected = evaluate_cohort_policy(raw_c5["raw_preds"], c5_imgs, gt_c5, vru_conf=selected_thresh, non_vru_conf=0.25, vru_nms_iou=best_iou)

    # Also evaluate Adaptive 0.20 on Cohort 5 for full cross-cohort transparency
    eval_c5_020 = evaluate_cohort_policy(raw_c5["raw_preds"], c5_imgs, gt_c5, vru_conf=0.20, non_vru_conf=0.25, vru_nms_iou=0.65)

    stage_d_results = {
        "Baseline_0.25": eval_c5_baseline,
        "Selected_Policy": eval_c5_selected,
        "Adaptive_0.20_Reference": eval_c5_020
    }

    for name, ev in stage_d_results.items():
        vm = ev["vru_metrics"]
        ci = ev["ci_95"]
        print(f"  Cohort 5 - {name}: SmallRec={vm['small_vru_recall']}% [{ci['small_vru_recall'][0]}, {ci['small_vru_recall'][1]}%], VRU Prec={vm['precision']}% [{ci['vru_precision'][0]}, {ci['vru_precision'][1]}%], F1={vm['f1_score']}")

    # Render Visual Overlays for Cohort 4 and Cohort 5
    # Find sample images with recoveries
    render_visual_overlays(raw_c4["raw_preds"], gt_c4, c4_imgs[:3], v_conf=selected_thresh, prefix="cohort4_confirm")
    render_visual_overlays(raw_c5["raw_preds"], gt_c5, c5_imgs[:3], v_conf=selected_thresh, prefix="cohort5_untouched")

    # Save Detailed JSON Results
    detailed_json = {
        "stage_a": stage_a_evals,
        "stage_b": nms_evals,
        "stage_d": stage_d_results,
        "selected_policy": {
            "vru_conf": selected_thresh,
            "vru_nms_iou": best_iou,
            "non_vru_conf": 0.25
        },
        "latencies": {
            "cohort4_ms": raw_c4["latencies_ms"],
            "cohort5_ms": raw_c5["latencies_ms"]
        }
    }
    json_path = OUT_DIR / "phase10e_detailed_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_json, f, indent=2, default=str)
    print(f"\n[SAVED] Phase 10E Detailed JSON: {json_path}")


if __name__ == "__main__":
    main()
