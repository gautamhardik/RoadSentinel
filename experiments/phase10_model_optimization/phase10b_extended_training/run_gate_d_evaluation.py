"""
RoadSentinel Phase 10B — Gate D: Matched Champion Selection Evaluation
Evaluates the Frozen Phase 6 Baseline against the Extended 50-Epoch Phase 10B Champion
on the exact 300-image development pilot cohort at 960px operational resolution.

Invariants Verified:
  - Exact 300-image development pilot cohort (PILOT_300_FILENAMES)
  - Quarantined test split (bdd100k_dataset/test) is NEVER accessed
  - Identical inference configuration: conf=0.25, iou=0.65, imgsz=960
  - Identical matching rules: class-aware confidence-ordered greedy matching at IoU 0.50
  - Small-VRU threshold: area < 1024 px² in original 1280x720 coordinates
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
import numpy as np
import cv2

# Ensure OpenMP stability on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from ultralytics import YOLO

from kaggle_jobs.pilot_300_native_eval.evaluate_pilot_native import (
    PILOT_300_FILENAMES, CLASS_NAMES, VRU_NAMES, VRU_IDS,
    match_single_image, aggregate_partition
)
from experiments.phase8_foveation.evaluate_phase8_foveation import load_ground_truth

BASELINE_PATH = REPO_ROOT / "scratch" / "kaggle_upload_weights" / "best.pt"
EXTENDED_BEST_PATH = REPO_ROOT / "experiments" / "phase10_model_optimization" / "phase10b_extended_training" / "session4_artifacts" / "session4_best.pt"
EXTENDED_FINAL_PATH = REPO_ROOT / "experiments" / "phase10_model_optimization" / "phase10b_extended_training" / "session4_artifacts" / "session4_final_checkpoint_ep50.pt"

VAL_IMG_DIR = REPO_ROOT / "bdd100k_dataset" / "val" / "images"
VAL_LBL_DIR = REPO_ROOT / "bdd100k_dataset" / "val" / "labels"
OUT_DIR = REPO_ROOT / "experiments" / "phase10_model_optimization" / "phase10b_extended_training"

EXPECTED_BASELINE_HASH = "42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e"
EXPECTED_EXTENDED_BEST_HASH = "4a245c1f81d1240f28fbd4d35bd175cdff765fcadc4c3137f52fb5ca912666cd"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def evaluate_model_on_pilot(model_path: Path, imgsz: int = 960, conf: float = 0.25, iou: float = 0.65) -> dict:
    print(f"\n--- Evaluating: {model_path.name} ---")
    model_hash = sha256(model_path)
    print(f"Path: {model_path}\nSHA-256: {model_hash}")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device} | imgsz: {imgsz} | conf: {conf} | iou: {conf}")
    model = YOLO(str(model_path))

    # Pre-load ground truth for 300 pilot images
    pilot_subset = sorted(list(PILOT_300_FILENAMES))
    gt_by_image = load_ground_truth(PILOT_300_FILENAMES)

    image_stats = {}
    total_infer_time = 0.0

    batch_size = 16
    for i in range(0, len(pilot_subset), batch_size):
        batch_names = pilot_subset[i:i + batch_size]
        batch_imgs = [cv2.imread(str(VAL_IMG_DIR / name)) for name in batch_names]
        
        t0 = time.perf_counter()
        results = model.predict(
            source=batch_imgs,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            batch=batch_size,
            device=device,
            verbose=False
        )
        total_infer_time += (time.perf_counter() - t0)

        for name, r in zip(batch_names, results):
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

            st = match_single_image(preds, gt_by_image[name])
            image_stats[name] = st

        processed = min(i + batch_size, len(pilot_subset))
        print(f"  Processed {processed} / {len(pilot_subset)} images (batch {i//batch_size + 1}/{(len(pilot_subset)+batch_size-1)//batch_size})...", flush=True)

    avg_ms = (total_infer_time / len(pilot_subset)) * 1000.0
    agg = aggregate_partition(image_stats, set(pilot_subset))
    agg["model_path"] = str(model_path)
    agg["model_sha256"] = model_hash
    agg["avg_infer_latency_ms"] = round(avg_ms, 2)
    return agg


def main():
    print("=" * 75)
    print("RoadSentinel Phase 10B — Gate D: Champion Selection Evaluation")
    print("=" * 75)

    # 1. Verify frozen baseline SHA-256
    assert BASELINE_PATH.exists(), f"Missing baseline: {BASELINE_PATH}"
    base_hash = sha256(BASELINE_PATH)
    print(f"Frozen Baseline: {BASELINE_PATH} (SHA-256: {base_hash})")
    assert base_hash == EXPECTED_BASELINE_HASH, f"Baseline hash mismatch: {base_hash}"

    # 2. Verify extended model SHA-256
    assert EXTENDED_BEST_PATH.exists(), f"Missing extended best model: {EXTENDED_BEST_PATH}"
    ext_best_hash = sha256(EXTENDED_BEST_PATH)
    print(f"Extended Champion: {EXTENDED_BEST_PATH} (SHA-256: {ext_best_hash})")
    assert ext_best_hash == EXPECTED_EXTENDED_BEST_HASH, f"Extended best hash mismatch: {ext_best_hash}"

    # 3. Evaluate Baseline Model
    baseline_eval = evaluate_model_on_pilot(BASELINE_PATH)

    # 4. Evaluate Extended Champion Model
    extended_eval = evaluate_model_on_pilot(EXTENDED_BEST_PATH)

    # 5. Comparative Delta Analysis
    b_vru = baseline_eval["vru_metrics"]
    e_vru = extended_eval["vru_metrics"]

    b_all = baseline_eval["all_class_metrics"]
    e_all = extended_eval["all_class_metrics"]

    small_vru_delta_pp = e_vru["small_vru_recall"] - b_vru["small_vru_recall"]
    vru_recall_delta_pp = e_vru["recall"] - b_vru["recall"]
    vru_prec_delta_pp = e_vru["precision"] - b_vru["precision"]
    all_recall_delta_pp = e_all["recall"] - b_all["recall"]
    all_prec_delta_pp = e_all["precision"] - b_all["precision"]

    # Gate Acceptance Verification
    pass_small_vru = bool(small_vru_delta_pp >= 2.00)
    pass_vru_prec = bool(e_vru["precision"] >= 75.00)
    pass_zero_forgetting = bool(all_recall_delta_pp >= 0.0)

    comparison = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort": "300-Image Development Pilot (PILOT_300_FILENAMES)",
        "resolution": 960,
        "operational_conf": 0.25,
        "baseline_model": {
            "name": "Phase 6 Baseline (Epoch 26)",
            "sha256": base_hash,
            "small_vru_tp": b_vru["small_vru_true_positives"],
            "small_vru_total": b_vru["small_vru_ground_truth"],
            "small_vru_recall": b_vru["small_vru_recall"],
            "vru_precision": b_vru["precision"],
            "vru_recall": b_vru["recall"],
            "vru_f1": b_vru["f1_score"],
            "all_class_precision": b_all["precision"],
            "all_class_recall": b_all["recall"],
            "all_class_f1": b_all["f1_score"],
            "avg_latency_ms": baseline_eval["avg_infer_latency_ms"],
            "per_class": baseline_eval["per_class"]
        },
        "extended_champion": {
            "name": "Phase 10B Extended Champion (Epoch 50)",
            "sha256": ext_best_hash,
            "small_vru_tp": e_vru["small_vru_true_positives"],
            "small_vru_total": e_vru["small_vru_ground_truth"],
            "small_vru_recall": e_vru["small_vru_recall"],
            "vru_precision": e_vru["precision"],
            "vru_recall": e_vru["recall"],
            "vru_f1": e_vru["f1_score"],
            "all_class_precision": e_all["precision"],
            "all_class_recall": e_all["recall"],
            "all_class_f1": e_all["f1_score"],
            "avg_latency_ms": extended_eval["avg_infer_latency_ms"],
            "per_class": extended_eval["per_class"]
        },
        "deltas": {
            "small_vru_recall_pp": round(small_vru_delta_pp, 2),
            "small_vru_tp_gain": e_vru["small_vru_true_positives"] - b_vru["small_vru_true_positives"],
            "vru_recall_pp": round(vru_recall_delta_pp, 2),
            "vru_precision_pp": round(vru_prec_delta_pp, 2),
            "all_class_recall_pp": round(all_recall_delta_pp, 2),
            "all_class_precision_pp": round(all_prec_delta_pp, 2)
        },
        "gates": {
            "small_vru_recall_gate": {
                "threshold": "+2.00 pp",
                "achieved": f"{small_vru_delta_pp:+.2f} pp",
                "passed": pass_small_vru
            },
            "vru_precision_gate": {
                "threshold": ">= 75.00%",
                "achieved": f"{e_vru['precision']:.2f}%",
                "passed": pass_vru_prec
            },
            "zero_forgetting_gate": {
                "threshold": ">= 0.0 pp on all-class recall",
                "achieved": f"{all_recall_delta_pp:+.2f} pp",
                "passed": pass_zero_forgetting
            }
        }
    }

    out_file = OUT_DIR / "gate_d_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[SAVED] Gate D Comparison saved to: {out_file}")

    print("\n" + "=" * 75)
    print("GATE D COMPARISON SUMMARY")
    print("=" * 75)
    print(f"Metric                       | Baseline (26ep) | Extended (50ep) | Delta")
    print("-" * 75)
    print(f"Small-VRU Recall (<1024 px²) | {b_vru['small_vru_recall']:6.2f}%        | {e_vru['small_vru_recall']:6.2f}%        | {small_vru_delta_pp:+6.2f} pp ({'+' if pass_small_vru else ''}{'PASS' if pass_small_vru else 'FAIL'})")
    print(f"VRU Overall Recall           | {b_vru['recall']:6.2f}%        | {e_vru['recall']:6.2f}%        | {vru_recall_delta_pp:+6.2f} pp")
    print(f"VRU Overall Precision        | {b_vru['precision']:6.2f}%        | {e_vru['precision']:6.2f}%        | {vru_prec_delta_pp:+6.2f} pp ({'PASS' if pass_vru_prec else 'FAIL'})")
    print(f"All-Class Recall             | {b_all['recall']:6.2f}%        | {e_all['recall']:6.2f}%        | {all_recall_delta_pp:+6.2f} pp ({'PASS' if pass_zero_forgetting else 'FAIL'})")
    print(f"All-Class Precision          | {b_all['precision']:6.2f}%        | {e_all['precision']:6.2f}%        | {all_prec_delta_pp:+6.2f} pp")
    print("=" * 75)


if __name__ == "__main__":
    main()
