"""
RoadSentinel — Model Comparison Benchmark: Decoupled Surgical Champion vs Baselines
File: scripts/benchmark_surgical_champion.py

Evaluates:
  1. session4_best.pt (Phase 10B 50-epoch Champion Baseline)
  2. phase13_best.pt (Phase 13 VRU Polish standalone model)
  3. phase13_soup_5050.pt (50/50 linear average soup)
  4. phase13_surgical_champion.pt (Decoupled-Head surgical graft: 0.92 backbone/box + 0.68 classifier)

Validation Protocol:
  - Dataset: Exact holdout validation benchmark cohort from holdout_split.txt
  - Resolution: imgsz=960 (operational standard)
  - Zero tilde estimation: Real Ultralytics validation pass computing exact mAP50, mAP50-95,
    Precision, Recall, and F1.
  - Output: experiments/phase13_vru_polish/surgical_comparison_results.json
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
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_SPLIT_FILE = REPO_ROOT / "experiments" / "phase_13_output" / "holdout_split.txt"
TRAIN_IMG_DIR = REPO_ROOT / "bdd100k_dataset" / "train" / "images"
TRAIN_LBL_DIR = REPO_ROOT / "bdd100k_dataset" / "train" / "labels"
OUT_DIR = REPO_ROOT / "experiments" / "phase13_vru_polish"
OUT_JSON = OUT_DIR / "surgical_comparison_results.json"

COHORT_SIZE = 50  # 50 dense holdout images (~940 ground-truth objects)

MODELS = [
    {
        "name": "Session 4 Baseline (50 Ep)",
        "id": "session4_best",
        "path": REPO_ROOT / "weights" / "session4_best.pt",
        "role": "Prior Champion Baseline"
    },
    {
        "name": "Phase 13 Standalone (16 Ep)",
        "id": "phase13_best",
        "path": REPO_ROOT / "weights" / "phase13_best.pt",
        "role": "Native-Frame VRU Polish"
    },
    {
        "name": "Phase 13 50/50 Model Soup",
        "id": "phase13_soup_5050",
        "path": REPO_ROOT / "weights" / "phase13_soup_5050.pt",
        "role": "Linear 50/50 Weight Average"
    },
    {
        "name": "Phase 13 Surgical Champion",
        "id": "phase13_surgical_champion",
        "path": REPO_ROOT / "weights" / "phase13_surgical_champion.pt",
        "role": "Decoupled-Head Graft (0.92 Box / 0.68 Cls)"
    }
]


def sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare_benchmark_dataset() -> Path:
    """Prepares the cohort txt and yaml files from the holdout split."""
    with open(HOLDOUT_SPLIT_FILE, "r") as f:
        holdout_raw = [line.strip() for line in f if line.strip()]

    cohort_raw = holdout_raw[:COHORT_SIZE]
    cohort_paths = []
    for raw in cohort_raw:
        img_name = Path(raw).name
        local_path = (TRAIN_IMG_DIR / img_name).resolve()
        assert local_path.exists(), f"Image missing: {local_path}"
        cohort_paths.append(str(local_path).replace("\\", "/"))

    cohort_txt = OUT_DIR / f"holdout_benchmark_cohort_{COHORT_SIZE}.txt"
    cohort_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(cohort_txt, "w") as f:
        for p in cohort_paths:
            f.write(p + "\n")

    yaml_path = OUT_DIR / f"holdout_benchmark_cohort_{COHORT_SIZE}.yaml"
    yaml_content = f"""path: {str(REPO_ROOT).replace("\\\\", "/")}
train: {str(cohort_txt).replace("\\\\", "/")}
val: {str(cohort_txt).replace("\\\\", "/")}
nc: 10
names: ['person', 'rider', 'car', 'bus', 'truck', 'bike', 'motor', 'traffic light', 'traffic sign', 'train']
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"[dataset] Prepared {len(cohort_paths)} holdout benchmark cohort images.")
    print(f"[dataset] Manifest: {cohort_txt}")
    print(f"[dataset] Config  : {yaml_path}")
    return yaml_path


def evaluate_checkpoint(model_info: Dict[str, Any], data_yaml: Path, imgsz: int = 960) -> Dict[str, Any]:
    model_path = model_info["path"]
    assert model_path.exists(), f"Model file not found: {model_path}"
    model_hash = sha256_file(model_path)
    file_size_mb = model_path.stat().st_size / (1024 * 1024)

    print("\n" + "-" * 80)
    print(f"BENCHMARKING: {model_info['name']}")
    print(f"Path   : {model_path}")
    print(f"SHA-256: {model_hash}")
    print(f"Size   : {file_size_mb:.2f} MB")
    print("-" * 80)

    model = YOLO(str(model_path))
    t0 = time.time()
    val_res = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=4,
        device="cpu",
        plots=False,
        save=False,
        verbose=False
    )
    elapsed = time.time() - t0

    res_dict = val_res.results_dict
    p = float(res_dict.get("metrics/precision(B)", 0.0))
    r = float(res_dict.get("metrics/recall(B)", 0.0))
    map50 = float(res_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(res_dict.get("metrics/mAP50-95(B)", 0.0))
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    print(f"Finished in {elapsed:.2f}s ({elapsed / COHORT_SIZE:.2f}s/img)")
    print(f"  Precision: {p * 100:.2f}% | Recall: {r * 100:.2f}% | F1: {f1:.4f}")
    print(f"  mAP50    : {map50 * 100:.2f}% | mAP50-95: {map50_95 * 100:.2f}%")

    return {
        "id": model_info["id"],
        "name": model_info["name"],
        "role": model_info["role"],
        "path": str(model_path),
        "sha256": model_hash,
        "size_mb": round(file_size_mb, 2),
        "imgsz": imgsz,
        "cohort_size": COHORT_SIZE,
        "elapsed_seconds": round(elapsed, 2),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "precision_pct": round(p * 100.0, 2),
        "recall_pct": round(r * 100.0, 2),
        "map50_pct": round(map50 * 100.0, 2),
        "map50_95_pct": round(map50_95 * 100.0, 2),
    }


def main():
    print("=" * 80)
    print("ROADSENTINEL: SURGICAL CHAMPION COMPARATIVE BENCHMARK")
    print("=" * 80)

    data_yaml = prepare_benchmark_dataset()

    results = []
    for model_info in MODELS:
        m_res = evaluate_checkpoint(model_info, data_yaml, imgsz=960)
        results.append(m_res)

    # Compile Summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "benchmark_cohort": {
            "source": "holdout_split.txt",
            "size": COHORT_SIZE,
            "imgsz": 960,
        },
        "models": results,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 92)
    print("EMPIRICAL BENCHMARK COMPARISON TABLE (Measured Real Results — NO Estimates)")
    print("=" * 92)
    print(f"{'Model Name':<34} | {'mAP50':<8} | {'mAP50-95':<10} | {'Precision':<10} | {'Recall':<8} | {'F1':<6}")
    print("-" * 92)
    for m in results:
        print(
            f"{m['name']:<34} | {m['map50_pct']:>6.2f}% | {m['map50_95_pct']:>8.2f}% | "
            f"{m['precision_pct']:>8.2f}% | {m['recall_pct']:>6.2f}% | {m['f1_score']:>6.4f}"
        )
    print("=" * 92)
    print(f"\n[output] Saved exact benchmark metrics to: {OUT_JSON}")


if __name__ == "__main__":
    main()
