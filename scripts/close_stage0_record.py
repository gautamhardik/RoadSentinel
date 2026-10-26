"""
RoadSentinel Stage 0 — close the evaluation record (CPU only).

  A1  Archive manifest: sha256 of every Stage 0 artifact, the generated kernel source and the reports.
  A2  Reconciliation: every reported VRU metric re-derived from TP/FP/FN and ground-truth counts, for every
      model x policy x partition; per-class, size-group and pilot+non-pilot additivity checks.
  A3  Confidence-floor audit: raw-policy AP is computed from predictions with conf >= 0.01 (YOLO predict(conf=0.01),
      max_det=300 per call). For each model and VRU class, recompute AP50 from saved raw detections and an
      optimistic upper bound that assumes every GT unmatched at the floor is recovered by pure true positives
      ranked immediately after the floor. The true untruncated AP50 lies in [AP_floor, AP_upper].
  A5  Checkpoint preservation: sha256 + size of the original and Phase 5A checkpoints; files set read-only.

Usage: python scripts/close_stage0_record.py
Output: experiments/stage0_matched_eval/stage0_record.json
"""

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0_operating_points as op  # noqa: E402  (loads the embedded evaluator)

ROOT = Path(__file__).resolve().parent.parent
S0 = ROOT / "experiments" / "stage0_matched_eval"
RUN = S0 / "stage0"
GT_TOTAL = {"full_validation_metrics": 15373, "pilot_cohort_metrics": 1740, "non_pilot_confirmation_metrics": 13633}
CHECKPOINTS = {
    "original_yolo11s": ROOT / "weights" / "bdd100k_yolo11s.pt",
    "phase5a_u_best": ROOT / "experiments/phase5a_modeling/runs/phase5a_u_ablation/phase5a_training/yolo11s_vru_finetune/weights/best.pt",
    "phase5a_u_last": ROOT / "experiments/phase5a_modeling/runs/phase5a_u_ablation/phase5a_training/yolo11s_vru_finetune/weights/last.pt",
    "phase5a_r_best": ROOT / "experiments/phase5a_modeling/runs/phase5a_r_training/phase5a_r_training/yolo11s_vru_weighted_finetune/weights/best.pt",
    "phase5a_r_last": ROOT / "experiments/phase5a_modeling/runs/phase5a_r_training/phase5a_r_training/yolo11s_vru_weighted_finetune/weights/last.pt",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def archive_manifest():
    files = sorted(p for p in S0.rglob("*") if p.is_file() and p.name != "stage0_record.json")
    files += [ROOT / "kaggle_jobs/stage0_matched_eval/kernel/stage0_matched_eval.py",
              ROOT / "kaggle_jobs/stage0_matched_eval/kernel/kernel-metadata.json",
              ROOT / "kaggle_jobs/stage0_matched_eval/build_kernel.py",
              ROOT / "docs/phase6/stage0_matched_evaluation_report.md",
              ROOT / "docs/phase6/stage1_split_audit.md"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files}


def reconcile():
    checks, failures = 0, []

    def check(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            failures.append(label)

    for model in op.MODELS:
        cfgs = json.loads((RUN / model / "phase4d_metrics.json").read_text())["configurations"]
        for cfg_name, cfg in cfgs.items():
            parts = {k: cfg[k] for k in GT_TOTAL}
            for part, m in parts.items():
                tag = f"{model}/{cfg_name}/{part}"
                tp, fp, fn = m["vru_tp"], m["vru_fp"], m["vru_fn"]
                check(tp + fn == GT_TOTAL[part], f"{tag}: TP+FN={tp + fn} != GT {GT_TOTAL[part]}")
                p, r = tp / max(tp + fp, 1), tp / GT_TOTAL[part]
                f1 = 2 * p * r / max(p + r, 1e-9)
                check(abs(round(p, 4) - m["vru_micro_precision_50"]) < 1e-4, f"{tag}: precision")
                check(abs(round(r, 4) - m["vru_micro_recall_50"]) < 1e-4, f"{tag}: recall")
                check(abs(round(f1, 4) - m["vru_micro_f1_50"]) < 1e-4, f"{tag}: f1")
                vru = [c for c in m["per_class_results"] if c["is_vru"]]
                check(sum(c["gt_count"] for c in vru) == GT_TOTAL[part], f"{tag}: per-class GT sum")
                check(sum(c["tp_50"] for c in vru) == tp and sum(c["fp_50"] for c in vru) == fp, f"{tag}: per-class TP/FP sum")
                check(all(c["tp_50"] + c["fn_50"] == c["gt_count"] for c in m["per_class_results"]), f"{tag}: per-class TP+FN")
                sg = m["size_group_results"].values()
                check(sum(g["vru_gt"] for g in sg) == GT_TOTAL[part], f"{tag}: size-group GT sum")
                check(sum(g["vru_matched"] for g in sg) == tp, f"{tag}: size-group TP sum")
                for g in sg:
                    check(abs(round(g["vru_matched"] / max(g["vru_gt"], 1), 4) - g["vru_recall"]) < 1e-4, f"{tag}: size-group recall")
            full, pil, non = (parts[k] for k in GT_TOTAL)
            for key in ("vru_tp", "vru_fp", "vru_fn"):
                check(pil[key] + non[key] == full[key], f"{model}/{cfg_name}: pilot+non-pilot {key} additivity")
    return {"checks": checks, "failed": len(failures), "failures": failures[:50]}


def ap50_interpolated(cum_tp, cum_fp, n_gt):
    rec = cum_tp / n_gt
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
    levels = np.linspace(0.0, 1.0, 101)
    return float(np.mean([prec[rec >= r].max() if (rec >= r).any() else 0.0 for r in levels]))


def floor_audit(images, gt):
    out = {}
    for model in op.MODELS:
        preds = op.fused_vru_preds(RUN / model / "raw_detections.jsonl.gz", images)
        k = json.loads((RUN / model / "phase4d_metrics.json").read_text())
        reported = {c["class_name"]: c["ap_50"] for c in
                    k["configurations"]["ActiveFoveation-960"]["non_pilot_confirmation_metrics"]["per_class_results"]}
        per = {}
        for c in op.VRU:
            n_gt = sum(len(v) for v in gt[c].values())
            ps = sorted(preds[c], key=lambda d: d["conf"], reverse=True)
            tp = np.zeros(len(ps))
            matched = {}
            for i, d in enumerate(ps):
                g = gt[c].get(d["img_name"])
                if g is None or len(g) == 0:
                    continue
                ious = op.ev.compute_box_iou_vec(d["box"], g)
                j = int(np.argmax(ious))
                if ious[j] >= 0.5 and j not in matched.setdefault(d["img_name"], set()):
                    tp[i] = 1
                    matched[d["img_name"]].add(j)
            cum_tp, cum_fp = np.cumsum(tp), np.cumsum(1 - tp)
            missing = int(n_gt - cum_tp[-1])
            up_tp = np.concatenate([cum_tp, cum_tp[-1] + np.arange(1, missing + 1)])
            up_fp = np.concatenate([cum_fp, np.full(missing, cum_fp[-1])])
            per[c] = {
                "n_pred_at_floor": len(ps), "recall_at_floor": round(float(cum_tp[-1] / n_gt), 4),
                "precision_at_floor": round(float(cum_tp[-1] / len(ps)), 4),
                "ap50_floor": round(ap50_interpolated(cum_tp, cum_fp, n_gt), 4),
                "ap50_upper_bound": round(ap50_interpolated(up_tp, up_fp, n_gt), 4),
                "ap50_reported_by_kernel": reported[c],
            }
        lo = float(np.mean([v["ap50_floor"] for v in per.values()]))
        hi = float(np.mean([v["ap50_upper_bound"] for v in per.values()]))
        out[model] = {"per_class": per, "vru_mAP50_floor": round(lo, 4), "vru_mAP50_upper_bound": round(hi, 4),
                      "max_possible_truncation_loss": round(hi - lo, 4)}
    return out


def preserve_checkpoints():
    rec = {}
    for name, p in CHECKPOINTS.items():
        if not p.exists():
            rec[name] = {"path": str(p), "status": "MISSING"}
            continue
        os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        rec[name] = {"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(p),
                     "bytes": p.stat().st_size, "read_only": not os.access(p, os.W_OK)}
    return rec


def main():
    val_images = sorted(p.name for p in (ROOT / "bdd100k_dataset" / "val" / "images").glob("*.jpg"))
    images = set(val_images) - set(op.ev.PILOT_COHORT_IMAGE_SET)
    gt, _ = op.load_gt(images)
    record = {
        "A2_reconciliation": reconcile(),
        "A3_confidence_floor_audit": floor_audit(images, gt),
        "A5_checkpoints": preserve_checkpoints(),
    }
    record["A1_archive_manifest"] = archive_manifest()
    (S0 / "stage0_record.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "A1_archive_manifest"}, indent=1))
    print(f"A1 archived files: {len(record['A1_archive_manifest'])}")


if __name__ == "__main__":
    main()
