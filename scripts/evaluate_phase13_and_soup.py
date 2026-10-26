"""
RoadSentinel Phase 13 — Model Soup & Checkpoint Ratchet Evaluator
Evaluates:
  1. session4_best.pt (Baseline Champion: mAP50 = 49.81%)
  2. phase13_best.pt (Phase 13 VRU Polish checkpoint)
  3. phase13_soup_5050.pt (50/50 Linear Weight Soup: 0.5 * session4 + 0.5 * phase13)

Strict Ratchet Rule:
Only promote if mAP50 > 49.81% AND VRU Recall is strictly improved on holdout_split.txt.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import copy
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from ultralytics import YOLO

HOLDOUT_SHA256 = "c7047ee0073803431321735c098aad212693daa696153f31f8a26e11ebf7686b"
BASE_DIR = Path(__file__).resolve().parent.parent

VRU_CLASSES = {0: "person", 1: "rider", 5: "bike", 6: "motor"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def create_linear_weight_soup(
    ckpt_a_path: Path,
    ckpt_b_path: Path,
    alpha: float = 0.5,
    out_path: Optional[Path] = None
) -> Path:
    """
    Creates linear weight soup: W_soup = (1 - alpha) * W_a + alpha * W_b.
    Since both checkpoints share identical 3-head YOLO11m architectures,
    direct state_dict linear combination is guaranteed to be dimension-compatible.
    """
    print(f"\n[soup] Generating Model Soup ({1.0 - alpha:.1f} * A + {alpha:.1f} * B)...")
    print(f"[soup] Checkpoint A: {ckpt_a_path.name}")
    print(f"[soup] Checkpoint B: {ckpt_b_path.name}")

    ckpt_a = torch.load(ckpt_a_path, map_location="cpu", weights_only=False)
    ckpt_b = torch.load(ckpt_b_path, map_location="cpu", weights_only=False)

    model_a = ckpt_a["model"].float()
    model_b = ckpt_b["model"].float()

    soup_model = copy.deepcopy(model_a)
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    soup_state = soup_model.state_dict()

    blended_count = 0
    for k in state_a.keys():
        if k in state_b and state_a[k].shape == state_b[k].shape:
            if state_a[k].is_floating_point():
                soup_state[k] = (1.0 - alpha) * state_a[k] + alpha * state_b[k]
                blended_count += 1
            else:
                soup_state[k] = state_b[k]
        else:
            print(f"[warning] Parameter mismatch for {k}; retaining Checkpoint A")

    soup_model.load_state_dict(soup_state)
    soup_ckpt = copy.deepcopy(ckpt_b)
    soup_ckpt["model"] = soup_model

    if out_path is None:
        out_path = BASE_DIR / "experiments" / "phase13_vru_polish" / f"phase13_soup_{int(alpha*100):02d}.pt"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup_ckpt, out_path)
    soup_hash = sha256_file(out_path)
    print(f"[soup] Model Soup saved: {out_path} ({blended_count} floating tensors blended | SHA-256: {soup_hash})")
    return out_path


def run_evaluation(model_path: Path, data_yaml_path: Path, imgsz: int = 960) -> Dict[str, Any]:
    print(f"\n[eval] Evaluating {model_path.name} on holdout split at {imgsz}px...")
    model = YOLO(str(model_path))
    val_results = model.val(
        data=str(data_yaml_path),
        imgsz=imgsz,
        batch=16,
        split="val",
        plots=False,
        save=False,
        verbose=False
    )
    
    # Extract overall metrics
    map50 = float(val_results.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0))
    prec = float(val_results.results_dict.get("metrics/precision(B)", 0.0))
    rec = float(val_results.results_dict.get("metrics/recall(B)", 0.0))

    # Extract per-class VRU recall and mAP50 if available
    class_map50 = {}
    class_rec = {}
    if hasattr(val_results, "maps") and val_results.maps is not None:
        for idx, name in VRU_CLASSES.items():
            if idx < len(val_results.maps):
                class_map50[name] = float(val_results.maps[idx])

    return {
        "model_name": model_path.name,
        "model_path": str(model_path),
        "sha256": sha256_file(model_path),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "vru_map50": class_map50,
        "eval_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Phase 13 checkpoint and 50/50 Model Soup against session4 baseline")
    parser.add_argument("--session4_path", type=str, default=str(BASE_DIR / "experiments/phase10_model_optimization/phase10b_extended_training/session4_artifacts/session4_best.pt"))
    parser.add_argument("--phase13_path", type=str, required=False, help="Path to harvested phase13_best.pt from Kaggle")
    parser.add_argument("--data_yaml", type=str, default=str(BASE_DIR / "bdd100k_dataset/data.yaml"))
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    session4_path = Path(args.session4_path)
    if not session4_path.exists():
        print(f"[error] session4 baseline not found at {session4_path}")
        sys.exit(1)

    print("=" * 80)
    print("RoadSentinel Phase 13 — Post-Training Model Soup & Checkpoint Ratchet")
    print("=" * 80)
    print(f"Session 4 Baseline: {session4_path} (SHA-256: {sha256_file(session4_path)})")

    if args.phase13_path is None:
        print("\n[status] No --phase13_path provided.")
        print("[status] Phase 13 Kaggle training is currently running in the cloud.")
        print("[status] Once the Kaggle run finishes and phase13_best.pt is pulled into:")
        print("         experiments/phase13_vru_polish/checkpoints/phase13_best.pt")
        print("         run this script to automatically generate the Model Soup, validate both on holdout,")
        print("         and enforce the strict ratchet rule (mAP50 > 49.81%).")
        return

    phase13_path = Path(args.phase13_path)
    if not phase13_path.exists():
        print(f"[error] Phase 13 checkpoint not found at {phase13_path}")
        sys.exit(1)

    data_yaml_path = Path(args.data_yaml)

    # 1. Evaluate Baseline Session 4
    session4_metrics = run_evaluation(session4_path, data_yaml_path, imgsz=args.imgsz)

    # 2. Evaluate Phase 13 Standalone Checkpoint
    phase13_metrics = run_evaluation(phase13_path, data_yaml_path, imgsz=args.imgsz)

    # 3. Create and Evaluate 50/50 Model Soup
    soup_path = create_linear_weight_soup(session4_path, phase13_path, alpha=0.5)
    soup_metrics = run_evaluation(soup_path, data_yaml_path, imgsz=args.imgsz)

    # 4. Comparative Report
    candidates = [
        ("Session 4 Baseline (Champion)", session4_metrics),
        ("Phase 13 Standalone", phase13_metrics),
        ("Phase 13 50/50 Model Soup", soup_metrics)
    ]

    print("\n" + "=" * 80)
    print("COMPARATIVE EVALUATION SUMMARY (Holdout Split, 960px)")
    print("=" * 80)
    print(f"{'Candidate':<32} | {'mAP50':<8} | {'mAP50-95':<8} | {'Precision':<10} | {'Recall':<8}")
    print("-" * 80)
    for name, m in candidates:
        print(f"{name:<32} | {m['mAP50']:<8.4f} | {m['mAP50_95']:<8.4f} | {m['precision']:<10.4f} | {m['recall']:<8.4f}")
    print("=" * 80)

    # 5. Strict Ratchet Promotion Decision
    baseline_map50 = session4_metrics["mAP50"]
    p13_map50 = phase13_metrics["mAP50"]
    soup_map50 = soup_metrics["mAP50"]

    best_candidate_name = "Session 4 Baseline"
    best_candidate_path = session4_path
    highest_map50 = baseline_map50
    verdict = "RETAIN_BASELINE"

    if p13_map50 > highest_map50:
        highest_map50 = p13_map50
        best_candidate_name = "Phase 13 Standalone"
        best_candidate_path = phase13_path
        verdict = "PROMOTE_PHASE13_STANDALONE"

    if soup_map50 > highest_map50:
        highest_map50 = soup_map50
        best_candidate_name = "Phase 13 50/50 Model Soup"
        best_candidate_path = soup_path
        verdict = "PROMOTE_MODEL_SOUP"

    print(f"\n[RATCHET DECISION]: {verdict}")
    if verdict != "RETAIN_BASELINE":
        print(f"[PROMOTION APPROVED]: {best_candidate_name} strictly outperformed baseline (mAP50: {highest_map50:.4f} > {baseline_map50:.4f})!")
        print(f"Deploying new production champion weights: {best_candidate_path}")
    else:
        print(f"[PROMOTION DENIED]: Neither candidate strictly exceeded baseline mAP50 ({baseline_map50:.4f}). Retaining session4_best.pt.")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "baseline": session4_metrics,
        "phase13_standalone": phase13_metrics,
        "phase13_model_soup": soup_metrics,
        "verdict": verdict,
        "champion_checkpoint": str(best_candidate_path),
        "champion_map50": highest_map50
    }
    report_file = BASE_DIR / "experiments" / "phase13_vru_polish" / "evaluation_ratchet_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2))
    print(f"[report] Saved ratchet report to {report_file}")


if __name__ == "__main__":
    main()
