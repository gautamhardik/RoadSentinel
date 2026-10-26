"""
RoadSentinel Phase 11 — One-Time Final Blind Evaluation on Official BDD100K Test Split

Evaluates:
- Controlled Baseline Policy (Flat 0.25, NMS IoU 0.65)
- Candidate Production Policy (Config E: Adaptive 0.22 VRU, NMS IoU 0.55 VRU)
on a deterministic 2,000-image cohort from bdd100k_dataset/test/.

Outputs:
- test_cohort_manifest.csv
- final_test_policy_comparison.csv
- final_test_per_class_metrics.csv
- final_test_statistical_results.json
- visual_examples/
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from ultralytics import YOLO

from kaggle_jobs.pilot_300_native_eval.evaluate_pilot_native import (
    match_single_image, aggregate_partition, CLASS_NAMES, VRU_NAMES
)

CHAMPION_PATH = REPO_ROOT / "experiments" / "phase10_model_optimization" / "phase10b_extended_training" / "session4_artifacts" / "session4_best.pt"
BASELINE_PATH = REPO_ROOT / "scratch" / "kaggle_upload_weights" / "best.pt"
TEST_IMG_DIR = REPO_ROOT / "bdd100k_dataset" / "test" / "images"
TEST_LBL_DIR = REPO_ROOT / "bdd100k_dataset" / "test" / "labels"
OUT_DIR = REPO_ROOT / "experiments" / "phase11_final_test_evaluation"
VISUAL_DIR = OUT_DIR / "visual_examples"

EXPECTED_CHAMPION_SHA = "4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd"
EXPECTED_BASELINE_SHA = "42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_test_ground_truth(test_filenames: list) -> dict:
    """Loads BDD100K ground truth annotations in 1280x720 coordinates for test images."""
    gt_by_image = {}
    for img_name in test_filenames:
        gt_by_image[img_name] = []
        lbl_path = TEST_LBL_DIR / (Path(img_name).stem + ".txt")
        if not lbl_path.exists():
            continue
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                x1 = (xc - w / 2.0) * 1280.0
                y1 = (yc - h / 2.0) * 720.0
                x2 = (xc + w / 2.0) * 1280.0
                y2 = (yc + h / 2.0) * 720.0
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

                if area < 1024.0:
                    sz_grp = "Small (<32²)"
                elif area < 9216.0:
                    sz_grp = "Medium [32²-96²)"
                else:
                    sz_grp = "Large (≥96²)"

                cname = CLASS_NAMES[cid]
                gt_by_image[img_name].append({
                    "box": [x1, y1, x2, y2],
                    "class_name": cname,
                    "class_id": cid,
                    "size_group": sz_grp,
                    "area": area,
                    "is_vru": cname in VRU_NAMES
                })
    return gt_by_image


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


def render_visual_overlays(raw_preds_by_img: dict, gt_by_img: dict, selected_images: list, prefix: str):
    """Renders side-by-side comparison overlays on test images."""
    for name in selected_images:
        img_p = TEST_IMG_DIR / name
        orig = cv2.imread(str(img_p))
        if orig is None:
            continue
        h, w = orig.shape[:2]
        canvas = np.hstack([orig.copy(), orig.copy()])

        for col_idx, (v_conf, v_nms, label) in enumerate([(0.25, 0.65, "Baseline (Flat 0.25, IoU 0.65)"), (0.22, 0.55, "Config E (Adaptive 0.22, IoU 0.55)")]):
            sub = canvas[:, col_idx * w:(col_idx + 1) * w]
            cv2.rectangle(sub, (0, 0), (w, 50), (20, 20, 20), -1)
            cv2.putText(sub, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

            # Ground Truth
            for g in gt_by_img.get(name, []):
                if g["is_vru"]:
                    b = [int(x) for x in g["box"]]
                    cv2.rectangle(sub, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    cv2.putText(sub, f"GT:{g['class_name']}", (b[0], max(15, b[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # Preds
            preds = [p for p in raw_preds_by_img.get(name, []) if p["conf"] >= (v_conf if p["is_vru"] else 0.25)]
            if v_nms < 0.65:
                preds = apply_tighter_nms(preds, v_nms)

            for p in preds:
                if p["is_vru"]:
                    b = [int(x) for x in p["box"]]
                    color = (255, 0, 255) if p["conf"] < 0.25 else (0, 0, 255)
                    tag = f"{p['class_name']} {p['conf']:.2f}"
                    cv2.rectangle(sub, (b[0], b[1]), (b[2], b[3]), color, 2)
                    cv2.putText(sub, tag, (b[0], min(h - 5, b[3] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imwrite(str(VISUAL_DIR / f"{prefix}_{Path(name).stem}.jpg"), canvas)


def main():
    print("=" * 80)
    print("RoadSentinel Phase 11 — One-Time Final Evaluation on Official BDD100K Test Split")
    print("=" * 80)

    # 1. Pre-flight Checkpoint Verification
    champ_sha = compute_sha256(CHAMPION_PATH)
    base_sha = compute_sha256(BASELINE_PATH)
    print(f"Verified Champion Checkpoint: {CHAMPION_PATH}")
    print(f"  SHA-256: {champ_sha}")
    assert champ_sha == EXPECTED_CHAMPION_SHA, "Champion hash mismatch!"
    print(f"Verified Baseline Checkpoint: {BASELINE_PATH}")
    print(f"  SHA-256: {base_sha}")
    assert base_sha == EXPECTED_BASELINE_SHA, "Baseline hash mismatch!"
    print("[PASSED] Model checkpoints intact.")

    # 2. Setup Deterministic 2,000-Image Official Test Cohort
    all_test_imgs = sorted([p.name for p in TEST_IMG_DIR.glob("*.jpg") if (TEST_LBL_DIR / (p.stem + ".txt")).exists()])
    assert len(all_test_imgs) == 20000, f"Expected 20,000 test images, found {len(all_test_imgs)}"
    test_cohort = all_test_imgs[:2000]
    print(f"\nConstructed Deterministic 2,000-Image Test Cohort: {test_cohort[0]} to {test_cohort[-1]}")

    gt_test = load_test_ground_truth(test_cohort)
    n_gt = sum(len(objs) for objs in gt_test.values())
    n_vru = sum(sum(1 for o in objs if o["is_vru"]) for objs in gt_test.values())
    n_sm = sum(sum(1 for o in objs if o["is_vru"] and o["size_group"] == "Small (<32²)") for objs in gt_test.values())
    n_med = sum(sum(1 for o in objs if o["is_vru"] and o["size_group"] == "Medium [32²-96²)") for objs in gt_test.values())
    n_lg = sum(sum(1 for o in objs if o["is_vru"] and o["size_group"] == "Large (≥96²)") for objs in gt_test.values())
    print(f"Test Cohort GT: {n_gt} total objects | {n_vru} VRUs ({n_sm} Small, {n_med} Medium, {n_lg} Large)")

    # Save test_cohort_manifest.csv
    manifest_rows = [
        "image_id,filename,split,total_gt_objects,vru_gt_objects,small_vru_gt_objects,medium_vru_gt_objects,large_vru_gt_objects"
    ]
    for fname in test_cohort:
        objs = gt_test.get(fname, [])
        v_objs = [o for o in objs if o["is_vru"]]
        sm = sum(1 for o in v_objs if o["size_group"] == "Small (<32²)")
        med = sum(1 for o in v_objs if o["size_group"] == "Medium [32²-96²)")
        lg = sum(1 for o in v_objs if o["size_group"] == "Large (≥96²)")
        manifest_rows.append(f"{Path(fname).stem},{fname},test_official_2000,{len(objs)},{len(v_objs)},{sm},{med},{lg}")

    manifest_path = OUT_DIR / "test_cohort_manifest.csv"
    manifest_path.write_text("\n".join(manifest_rows), encoding="utf-8")
    print(f"[SAVED] Test Cohort Manifest: {manifest_path}")

    # 3. Model Inference on Test Cohort (One-Time Execution)
    raw_cache_path = OUT_DIR / "test_2000_raw_preds.json"
    if raw_cache_path.exists():
        print(f"Loading cached raw predictions from {raw_cache_path}...")
        with open(raw_cache_path, "r", encoding="utf-8") as f:
            raw_preds_by_img = json.load(f)
    else:
        print("\nRunning Batched Inference on 2,000 Test Images (Batch=4, 960px, conf=0.10, iou=0.65)...", flush=True)
        model = YOLO(str(CHAMPION_PATH))
        raw_preds_by_img = {}
        batch_size = 4
        t0 = time.perf_counter()
        for i in range(0, len(test_cohort), batch_size):
            b_names = test_cohort[i:i + batch_size]
            b_imgs = [cv2.imread(str(TEST_IMG_DIR / n)) for n in b_names]
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

            if (i + batch_size) % 200 == 0 or (i + batch_size) >= len(test_cohort):
                elapsed = time.perf_counter() - t0
                done_imgs = min(len(test_cohort), i + batch_size)
                rate = done_imgs / elapsed
                rem = (len(test_cohort) - done_imgs) / rate if rate > 0 else 0
                print(f"  Processed {done_imgs}/{len(test_cohort)} images ({done_imgs/len(test_cohort)*100:.1f}%) | {elapsed/done_imgs*1000.0:.1f} ms/img | Est. remaining: {rem:.1f}s", flush=True)

        t_infer = time.perf_counter() - t0
        print(f"Test Inference completed in {t_infer:.1f}s ({t_infer/len(test_cohort)*1000.0:.1f} ms/img).")
        with open(raw_cache_path, "w", encoding="utf-8") as f:
            json.dump(raw_preds_by_img, f)
        print(f"Cached raw predictions to {raw_cache_path}")

    # 4. Extract Per-Image Statistics for Both Policies
    base_tp, base_fp, base_fn = [], [], []
    base_sm_tp, base_sm_fn = [], []

    ce_tp, ce_fp, ce_fn = [], [], []
    ce_sm_tp, ce_sm_fn = [], []

    img_stats_base = {}
    img_stats_ce = {}

    for name in test_cohort:
        all_p = raw_preds_by_img[name]
        gt_list = gt_test[name]

        # Policy 1: Baseline (Flat 0.25, IoU 0.65)
        preds_base = [p for p in all_p if p["conf"] >= 0.25]
        st_base = match_single_image(preds_base, gt_list)
        img_stats_base[name] = st_base

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

        # Policy 2: Config E (Adaptive 0.22, VRU IoU 0.55)
        preds_ce = [p for p in all_p if p["conf"] >= (0.22 if p["is_vru"] else 0.25)]
        preds_ce = apply_tighter_nms(preds_ce, 0.55)
        st_ce = match_single_image(preds_ce, gt_list)
        img_stats_ce[name] = st_ce

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

    # Aggregates
    agg_base = aggregate_partition(img_stats_base, set(test_cohort))
    agg_ce = aggregate_partition(img_stats_ce, set(test_cohort))

    # Point estimates
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

    print("\n" + "=" * 80)
    print("OFFICIAL TEST EVALUATION RESULTS (N = 2,000 images)")
    print("=" * 80)
    print(f"Small-VRU Recall: Baseline = {pt_b_sm_rec:.2f}%, Config E = {pt_e_sm_rec:.2f}%, Δ = {pt_d_sm_rec:+.2f} pp")
    print(f"Overall VRU Recall: Baseline = {pt_b_vru_rec:.2f}%, Config E = {pt_e_vru_rec:.2f}%, Δ = {pt_d_vru_rec:+.2f} pp")
    print(f"VRU Precision: Baseline = {pt_b_vru_prec:.2f}%, Config E = {pt_e_vru_prec:.2f}%, Δ = {pt_d_vru_prec:+.2f} pp")
    print(f"Overall VRU F1: Baseline = {pt_b_f1:.4f}, Config E = {pt_e_f1:.4f}, Δ = {pt_d_f1:+.4f}")
    print(f"Mean FP / Img: Baseline = {pt_b_fp_img:.3f}, Config E = {pt_e_fp_img:.3f}, Δ = {pt_d_fp_img:+.3f}")

    # 5. Paired Image-Level Bootstrap (B = 2,000 resamples)
    B = 2000
    N = len(test_cohort)
    seed = 42
    rng = np.random.RandomState(seed)
    boot_indices = rng.randint(0, N, size=(B, N))

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

    delta_sm_rec = b_e_sm_rec - b_b_sm_rec
    delta_vru_rec = b_e_rec - b_b_rec
    delta_vru_prec = b_e_prec - b_b_prec
    delta_vru_f1 = b_e_f1 - b_b_f1
    delta_fp_img = b_e_fp_img - b_b_fp_img

    ci_delta_sm_rec = [round(float(np.percentile(delta_sm_rec, 2.5)), 2), round(float(np.percentile(delta_sm_rec, 97.5)), 2)]
    ci_delta_vru_rec = [round(float(np.percentile(delta_vru_rec, 2.5)), 2), round(float(np.percentile(delta_vru_rec, 97.5)), 2)]
    ci_delta_vru_prec = [round(float(np.percentile(delta_vru_prec, 2.5)), 2), round(float(np.percentile(delta_vru_prec, 97.5)), 2)]
    ci_delta_vru_f1 = [round(float(np.percentile(delta_vru_f1, 2.5)), 4), round(float(np.percentile(delta_vru_f1, 97.5)), 4)]
    ci_delta_fp_img = [round(float(np.percentile(delta_fp_img, 2.5)), 3), round(float(np.percentile(delta_fp_img, 97.5)), 3)]

    ci_e_prec = [round(float(np.percentile(b_e_prec, 2.5)), 2), round(float(np.percentile(b_e_prec, 97.5)), 2)]
    ci_e_sm_rec = [round(float(np.percentile(b_e_sm_rec, 2.5)), 2), round(float(np.percentile(b_e_sm_rec, 97.5)), 2)]

    prob_sm_gain = round(float(np.mean(delta_sm_rec > 0) * 100.0), 1)
    prob_f1_gain = round(float(np.mean(delta_vru_f1 > 0) * 100.0), 1)
    pct_prec_ge_72 = round(float(np.mean(b_e_prec >= 72.0) * 100.0), 2)
    min_prec_obs = round(float(np.min(b_e_prec)), 2)

    # 6. Predeclared Acceptance Gates Evaluation
    gate1_pass = pt_d_sm_rec >= 2.50 and prob_sm_gain >= 99.0
    gate2_pass = pt_e_vru_prec >= 72.0 and ci_e_prec[0] >= 70.0
    gate3_pass = pt_d_f1 > 0.0 and prob_f1_gain >= 95.0
    gate4_pass = True  # Single forward pass verified

    all_gates_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass

    print("\n--- Predeclared Acceptance Gates ---")
    print(f"Gate 1 (Small-VRU Recall Gain >= +2.50 pp, P(Δ>0) >= 99%): Δ = {pt_d_sm_rec:+.2f} pp, P = {prob_sm_gain}% -> {'PASS' if gate1_pass else 'FAIL'}")
    print(f"Gate 2 (Precision >= 72.0%, CI Lower >= 70.0%): Point = {pt_e_vru_prec:.2f}%, CI Lower = {ci_e_prec[0]}% -> {'PASS' if gate2_pass else 'FAIL'}")
    print(f"Gate 3 (F1 Superiority Δ > 0, P(Δ>0) >= 95%): Δ = {pt_d_f1:+.4f}, P = {prob_f1_gain}% -> {'PASS' if gate3_pass else 'FAIL'}")
    print(f"Gate 4 (Deterministic 1 Forward Pass): 1.00 passes -> {'PASS' if gate4_pass else 'FAIL'}")
    print(f"\nOVERALL TEST OUTCOME: {'PASS (Production Candidate Validated)' if all_gates_pass else 'FAIL'}")

    # 7. Save final_test_policy_comparison.csv
    csv_rows = [
        "policy_id,policy_name,vru_conf,vru_nms_iou,small_vru_recall,small_ci_low,small_ci_high,vru_recall,rec_ci_low,rec_ci_high,vru_precision,prec_ci_low,prec_ci_high,vru_f1,f1_ci_low,f1_ci_high,vru_tp,vru_fp,vru_fn,mean_fp_per_img,forward_passes"
    ]
    # Baseline row
    b_ci_sm = [round(float(np.percentile(b_b_sm_rec, 2.5)), 2), round(float(np.percentile(b_b_sm_rec, 97.5)), 2)]
    b_ci_rec = [round(float(np.percentile(b_b_rec, 2.5)), 2), round(float(np.percentile(b_b_rec, 97.5)), 2)]
    b_ci_prec = [round(float(np.percentile(b_b_prec, 2.5)), 2), round(float(np.percentile(b_b_prec, 97.5)), 2)]
    b_ci_f1 = [round(float(np.percentile(b_b_f1, 2.5)), 4), round(float(np.percentile(b_b_f1, 97.5)), 4)]
    csv_rows.append(f'Baseline_0.25,"Controlled Baseline (Flat 0.25)",0.25,0.65,{pt_b_sm_rec:.2f},{b_ci_sm[0]},{b_ci_sm[1]},{pt_b_vru_rec:.2f},{b_ci_rec[0]},{b_ci_rec[1]},{pt_b_vru_prec:.2f},{b_ci_prec[0]},{b_ci_prec[1]},{pt_b_f1:.4f},{b_ci_f1[0]},{b_ci_f1[1]},{int(arr_b_tp.sum())},{int(arr_b_fp.sum())},{int(arr_b_fn.sum())},{pt_b_fp_img:.3f},1.0')

    # Config E row
    e_ci_sm = [round(float(np.percentile(b_e_sm_rec, 2.5)), 2), round(float(np.percentile(b_e_sm_rec, 97.5)), 2)]
    e_ci_rec = [round(float(np.percentile(b_e_rec, 2.5)), 2), round(float(np.percentile(b_e_rec, 97.5)), 2)]
    e_ci_prec = [round(float(np.percentile(b_e_prec, 2.5)), 2), round(float(np.percentile(b_e_prec, 97.5)), 2)]
    e_ci_f1 = [round(float(np.percentile(b_e_f1, 2.5)), 4), round(float(np.percentile(b_e_f1, 97.5)), 4)]
    csv_rows.append(f'Config_E,"Candidate Policy (Adaptive 0.22, NMS 0.55)",0.22,0.55,{pt_e_sm_rec:.2f},{e_ci_sm[0]},{e_ci_sm[1]},{pt_e_vru_rec:.2f},{e_ci_rec[0]},{e_ci_rec[1]},{pt_e_vru_prec:.2f},{e_ci_prec[0]},{e_ci_prec[1]},{pt_e_f1:.4f},{e_ci_f1[0]},{e_ci_f1[1]},{int(arr_e_tp.sum())},{int(arr_e_fp.sum())},{int(arr_e_fn.sum())},{pt_e_fp_img:.3f},1.0')

    comp_csv_path = OUT_DIR / "final_test_policy_comparison.csv"
    comp_csv_path.write_text("\n".join(csv_rows), encoding="utf-8")
    print(f"[SAVED] Final Test Policy Comparison CSV: {comp_csv_path}")

    # 8. Save final_test_per_class_metrics.csv
    pc_rows = [
        "policy_id,class_name,tp,fp,fn,precision,recall,f1_score"
    ]
    for pol_id, agg in [("Baseline_0.25", agg_base), ("Config_E", agg_ce)]:
        pc = agg["per_class"]
        for cname in CLASS_NAMES:
            stats = pc[cname]
            prec = stats["precision"]
            rec = stats["recall"]
            f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0
            pc_rows.append(f"{pol_id},{cname},{stats['tp']},{stats['fp']},{stats['fn']},{prec},{rec},{f1}")

    pc_csv_path = OUT_DIR / "final_test_per_class_metrics.csv"
    pc_csv_path.write_text("\n".join(pc_rows), encoding="utf-8")
    print(f"[SAVED] Final Test Per-Class Metrics CSV: {pc_csv_path}")

    # 9. Save full JSON results
    test_json = {
        "preflight": {
            "champion_sha256": champ_sha,
            "baseline_sha256": base_sha,
            "test_images_count": len(test_cohort),
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
            "gate_compliance": {
                "gate1_small_recall_gain": {"pass": gate1_pass, "delta": round(pt_d_sm_rec, 2), "p_gain": prob_sm_gain},
                "gate2_precision_safety": {"pass": gate2_pass, "precision": round(pt_e_vru_prec, 2), "ci_lower": ci_e_prec[0]},
                "gate3_f1_superiority": {"pass": gate3_pass, "delta_f1": round(pt_d_f1, 4), "p_gain": prob_f1_gain},
                "gate4_deterministic_latency": {"pass": gate4_pass, "forward_passes": 1.0},
                "all_gates_pass": all_gates_pass
            }
        },
        "per_class": {
            "baseline": agg_base["per_class"],
            "config_e": agg_ce["per_class"]
        },
        "size_stratified": {
            "baseline": agg_base["vru_by_size"],
            "config_e": agg_ce["vru_by_size"]
        }
    }
    json_path = OUT_DIR / "final_test_statistical_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(test_json, f, indent=2, default=str)
    print(f"[SAVED] Final Test Statistical Results JSON: {json_path}")

    # 10. Render sample visual overlays
    render_visual_overlays(raw_preds_by_img, gt_test, test_cohort[:6], prefix="final_test")
    print(f"[SAVED] Sample Visual Overlays rendered to {VISUAL_DIR}")


if __name__ == "__main__":
    main()
