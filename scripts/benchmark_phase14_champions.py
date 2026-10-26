"""
RoadSentinel — Benchmark Phase 14 Precision Lock vs Previous Champions
File: scripts/benchmark_phase14_champions.py

Evaluates:
  1. session4_best.pt (Phase 10F 50-epoch Champion Baseline)
  2. phase13_best.pt (Phase 13 Standalone: 61.01% mAP50 / 56.61% Recall)
  3. phase13_surgical_champion.pt (Phase 13 Decoupled-Head Graft)
  4. phase14_best.pt (Phase 14 Hard-Negative Precision Lock)
  5. roadsentinel_ultimate_champion.pt (Ultimate Decoupled-Head Graft: Phase 13 + Phase 14)

Outputs:
  - experiments/phase14_precision_lock/holdout_benchmark_results.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_SPLIT_FILE = REPO_ROOT / "experiments" / "phase_13_output" / "holdout_split.txt"
TRAIN_IMG_DIR = REPO_ROOT / "bdd100k_dataset" / "train" / "images"
TRAIN_LBL_DIR = REPO_ROOT / "bdd100k_dataset" / "train" / "labels"
OUT_DIR = REPO_ROOT / "experiments" / "phase14_precision_lock"
OUT_JSON = OUT_DIR / "holdout_benchmark_results.json"

COHORT_SIZE = 50  # 50 dense holdout images (942 ground-truth objects)

MODELS = [
    {
        "name": "Session 4 Baseline (50 Ep)",
        "id": "session4_best",
        "path": REPO_ROOT / "weights" / "session4_best.pt",
        "role": "Phase 10F Baseline"
    },
    {
        "name": "Phase 13 Standalone (16 Ep)",
        "id": "phase13_best",
        "path": REPO_ROOT / "weights" / "phase13_best.pt",
        "role": "Phase 13 Recall Champion"
    },
    {
        "name": "Phase 13 Surgical Champion",
        "id": "phase13_surgical",
        "path": REPO_ROOT / "weights" / "phase13_surgical_champion.pt",
        "role": "Decoupled Head Graft (S4+P13)"
    },
    {
        "name": "Phase 14 Precision Lock (8 Ep)",
        "id": "phase14_best",
        "path": REPO_ROOT / "weights" / "phase14_best.pt",
        "role": "Hard-Negative Precision Lock"
    },
    {
        "name": "RoadSentinel Ultimate Champion",
        "id": "ultimate_champion",
        "path": REPO_ROOT / "weights" / "roadsentinel_ultimate_champion.pt",
        "role": "Decoupled Graft (P13 Recall + P14 Precision)"
    }
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def prepare_validation_data_yaml(cohort_size: int = COHORT_SIZE) -> Path:
    with open(HOLDOUT_SPLIT_FILE, "r") as f:
        holdout_raw = [line.strip() for line in f if line.strip()]

    cohort_raw = holdout_raw[:cohort_size]
    cohort_paths = []
    for raw in cohort_raw:
        img_name = Path(raw).name
        local_path = (TRAIN_IMG_DIR / img_name).resolve()
        if local_path.exists():
            cohort_paths.append(str(local_path).replace("\\", "/"))

    cohort_txt = OUT_DIR / f"holdout_cohort_{cohort_size}.txt"
    cohort_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(cohort_txt, "w") as f:
        for p in cohort_paths:
            f.write(p + "\n")

    yaml_path = OUT_DIR / f"benchmark_val_{cohort_size}.yaml"
    yaml_content = f"""path: {str(REPO_ROOT).replace("\\\\", "/")}
train: {str(cohort_txt).replace("\\\\", "/")}
val: {str(cohort_txt).replace("\\\\", "/")}
nc: 10
names: ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 'traffic light', 'traffic sign', 'train']
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"[dataset] Prepared {len(cohort_paths)} holdout benchmark cohort images.")
    return yaml_path

def evaluate_model(model_cfg: Dict[str, Any], data_yaml: Path) -> Dict[str, Any]:
    name = model_cfg["name"]
    path = model_cfg["path"]
    print(f"\n--- Evaluating: {name} ---")
    print(f"Path: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    sha_hash = sha256_file(path)

    t0 = time.time()
    model = YOLO(str(path))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=960,
        batch=8,
        conf=0.15,
        iou=0.60,
        device=0 if torch.cuda.is_available() else "cpu",
        plots=False,
        save_json=False,
        verbose=False
    )
    eval_time = time.time() - t0

    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    result = {
        "name": name,
        "id": model_cfg["id"],
        "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "role": model_cfg["role"],
        "file_size_mb": round(file_size_mb, 2),
        "sha256": sha_hash,
        "mAP50": round(map50 * 100, 2),
        "mAP50_95": round(map5095 * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1, 4),
        "eval_time_sec": round(eval_time, 2)
    }

    print(f"  mAP50:    {result['mAP50']}%")
    print(f"  mAP50-95: {result['mAP50_95']}%")
    print(f"  Precision:{result['precision']}%")
    print(f"  Recall:   {result['recall']}%")
    print(f"  F1 Score: {result['f1_score']}")
    print(f"  SHA-256:  {result['sha256'][:16]}...")
    return result

def main():
    print("=" * 80)
    print("ROADSENTINEL PHASE 14 COMPREHENSIVE HOLDOUT BENCHMARK")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_yaml = prepare_validation_data_yaml(COHORT_SIZE)
    print(f"Validation dataset configured: {data_yaml} ({COHORT_SIZE} holdout frames)")

    results = []
    for m in MODELS:
        res = evaluate_model(m, data_yaml)
        results.append(res)

    results.sort(key=lambda x: (x["f1_score"], x["mAP50"]), reverse=True)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort_size": COHORT_SIZE,
        "resolution": 960,
        "champion_id": results[0]["id"],
        "champion_name": results[0]["name"],
        "results": results
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 95)
    print(f"{'Model Candidate':<34} | {'mAP50':<8} | {'mAP50-95':<9} | {'Precision':<10} | {'Recall':<8} | {'F1':<7}")
    print("-" * 95)
    for r in results:
        marker = " [CHAMPION]" if r["id"] == results[0]["id"] else ""
        print(f"{r['name'] + marker:<34} | {r['mAP50']:>6.2f}% | {r['mAP50_95']:>7.2f}% | {r['precision']:>8.2f}% | {r['recall']:>6.2f}% | {r['f1_score']:>6.4f}")
    print("=" * 95)
    print(f"\nEmpirical results successfully saved to: {OUT_JSON}")

if __name__ == "__main__":
    main()
