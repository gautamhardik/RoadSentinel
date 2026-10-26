"""
RoadSentinel Stage 0b — untruncated raw-policy AP recompute (CPU only).

Reads raw detections (any confidence floor), rebuilds canonical ActiveFoveation-960 fusion with the embedded
evaluator's own `baseline_fuse_and_deduplicate`, and scores VRU AP50 / AP50-95 with its own
`compute_coco_ap_for_class` on the 9,700-image non-pilot partition. Also reports the residual truncation bound at
the new floor (same optimistic-tail method as scripts/close_stage0_record.py).

Usage:
  python scripts/stage0b_ap_recompute.py <raw_dir> <suffix>
    e.g. experiments/stage0b_floor_dump/stage0b _raw_floor.jsonl.gz
         experiments/stage0_matched_eval/stage0 /raw_detections.jsonl.gz   (reproduces Stage 0 AP: self-check)
Output: <raw_dir>/ap_recompute.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0_operating_points as op  # noqa: E402
from close_stage0_record import ap50_interpolated  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def tail_bound(preds, gt):
    """AP50 upper bound if every GT unmatched at the floor were recovered by pure TPs right after it."""
    n_gt = sum(len(v) for v in gt.values())
    ps = sorted(preds, key=lambda d: d["conf"], reverse=True)
    tp = np.zeros(len(ps))
    matched = {}
    for i, d in enumerate(ps):
        g = gt.get(d["img_name"])
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
    return float(cum_tp[-1] / n_gt), ap50_interpolated(up_tp, up_fp, n_gt)


def main():
    raw_dir, suffix = Path(sys.argv[1]), sys.argv[2]
    val_images = sorted(p.name for p in (ROOT / "bdd100k_dataset" / "val" / "images").glob("*.jpg"))
    images = set(val_images) - set(op.ev.PILOT_COHORT_IMAGE_SET)
    gt, _ = op.load_gt(images)
    out = {}
    for m in op.MODELS:
        f = raw_dir / f"{m}{suffix}"
        if not f.exists():
            print(f"missing {f}")
            continue
        print(f"[{m}] fusing {f.name} ...", flush=True)
        preds = op.fused_vru_preds(f, images)
        per = {}
        for c in op.VRU:
            n_gt = sum(len(v) for v in gt[c].values())
            ap50, ap5095 = op.ev.compute_coco_ap_for_class(preds[c], gt[c], n_gt)
            rec_floor, ub = tail_bound(preds[c], gt[c])
            per[c] = {"n_pred": len(preds[c]), "ap50": round(ap50, 4), "ap50_95": round(ap5095, 4),
                      "recall_at_floor": round(rec_floor, 4), "ap50_upper_bound": round(ub, 4)}
            print(f"   {c:<7} n={len(preds[c]):>7} AP50={ap50:.4f} AP50-95={ap5095:.4f} R@floor={rec_floor:.4f} UB={ub:.4f}", flush=True)
        out[m] = {"per_class": per,
                  "vru_mAP50": round(float(np.mean([v["ap50"] for v in per.values()])), 4),
                  "vru_mAP50_95": round(float(np.mean([v["ap50_95"] for v in per.values()])), 4),
                  "vru_mAP50_upper_bound": round(float(np.mean([v["ap50_upper_bound"] for v in per.values()])), 4)}
        print(f"[{m}] VRU mAP50={out[m]['vru_mAP50']} mAP50-95={out[m]['vru_mAP50_95']} UB={out[m]['vru_mAP50_upper_bound']}", flush=True)
    (raw_dir / "ap_recompute.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
