"""
RoadSentinel Stage 0 — operating-point sweep from saved raw detections (CPU only, no GPU).

For each model, rebuilds the canonical ActiveFoveation-960 fusion output from raw_detections.jsonl.gz using the
embedded evaluator's own `baseline_fuse_and_deduplicate` and `evaluate_operational_point`, then sweeps the
confidence threshold on the 9,700-image non-pilot partition. This separates "the model detects better" from
"the model is more confident": compare recall at matched precision instead of recall at a fixed tau.

Usage: python scripts/stage0_operating_points.py [stage0_dir]
Output: <stage0_dir>/operating_points.json + printed tables.
"""

import gzip
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STAGE0 = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments" / "stage0_matched_eval" / "stage0"
EVALUATOR = ROOT / "kaggle_jobs" / "calibration_benchmark" / "evaluate_phase4d_adaptive_area.py"
VAL_LABELS = ROOT / "bdd100k_dataset" / "val" / "labels"
MODELS = ["base_yolo11s", "phase5a_u", "phase5a_r"]
TAUS = [round(t, 3) for t in np.arange(0.05, 0.6001, 0.01)]

spec = importlib.util.spec_from_file_location("ev", EVALUATOR)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
VRU = ev.VRU_CLASS_NAMES


def load_gt(images):
    """VRU ground truth exactly as the evaluator builds it (1280x720, small = area < 1024 px^2)."""
    gt, small = {c: {} for c in VRU}, {c: {} for c in VRU}
    for img in images:
        p = VAL_LABELS / (Path(img).stem + ".txt")
        boxes, flags = defaultdict(list), defaultdict(list)
        for line in p.read_text().split("\n"):
            parts = line.split()
            if not parts:
                continue
            cname = ev.GT_CLASSES[int(parts[0])]
            if cname not in VRU:
                continue
            xc, yc, bw, bh = map(float, parts[1:5])
            x1, y1, x2, y2 = (xc - bw / 2) * 1280, (yc - bh / 2) * 720, (xc + bw / 2) * 1280, (yc + bh / 2) * 720
            boxes[cname].append([x1, y1, x2, y2])
            flags[cname].append(max(0.0, x2 - x1) * max(0.0, y2 - y1) < 1024.0)
        for c in boxes:
            gt[c][img] = np.array(boxes[c], dtype=np.float32)
            small[c][img] = flags[c]
    return gt, small


def fused_vru_preds(raw_path, images):
    """Re-runs the evaluator's canonical fusion per image from raw detections."""
    preds = {c: [] for c in VRU}
    cur, ff, cr = None, [], []

    def flush():
        if cur in images:
            for d in ev.baseline_fuse_and_deduplicate(ff, cr, iou_thresh=0.50):
                if d["class_name"] in preds:
                    preds[d["class_name"]].append(d)

    with gzip.open(raw_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["img"] != cur:
                flush()
                cur, ff, cr = r["img"], [], []
            d = {"img_name": r["img"], "class_name": r["cls"], "conf": r["conf"],
                 "box": np.array(r["box"], dtype=np.float32), "source": r["src"]}
            (ff if r["src"] == "full_frame" else cr).append(d)
    flush()
    return preds


def sweep(preds, gt, small):
    n_gt = {c: sum(len(v) for v in gt[c].values()) for c in VRU}
    n_small = sum(sum(v) for c in VRU for v in small[c].values())
    rows = []
    for tau in TAUS:
        tp = fp = small_tp = 0
        for c in VRU:
            t, f, _, _, _, _, matched = ev.evaluate_operational_point(preds[c], gt[c], n_gt[c], tau, 0.50)
            tp, fp = tp + t, fp + f
            small_tp += sum(small[c][img][i] for img, idx in matched.items() for i in idx)
        p, r = tp / max(tp + fp, 1), tp / max(sum(n_gt.values()), 1)
        rows.append({"tau": tau, "P": p, "R": r, "F1": 2 * p * r / max(p + r, 1e-9),
                     "SmallR": small_tp / max(n_small, 1), "TP": tp, "FP": fp})
    return rows


def at_precision(rows, target):
    """Highest recall among thresholds whose precision >= target (no interpolation)."""
    ok = [r for r in rows if r["P"] >= target]
    return max(ok, key=lambda r: r["R"]) if ok else None


def main():
    val_images = sorted(p.name for p in (ROOT / "bdd100k_dataset" / "val" / "images").glob("*.jpg"))
    images = set(val_images) - set(ev.PILOT_COHORT_IMAGE_SET)
    assert len(images) == 9700, len(images)
    gt, small = load_gt(images)

    curves = {}
    for m in MODELS:
        print(f"[{m}] fusing raw detections...", flush=True)
        curves[m] = sweep(fused_vru_preds(STAGE0 / m / "raw_detections.jsonl.gz", images), gt, small)

    # sanity: tau=0.25 must reproduce the kernel's own numbers
    for m in MODELS:
        k = json.loads((STAGE0 / m / "phase4d_metrics.json").read_text())
        km = k["configurations"]["ActiveFoveation-960"]["non_pilot_confirmation_metrics"]
        r25 = next(r for r in curves[m] if r["tau"] == 0.25)
        print(f"[check] {m}: sweep TP/FP@0.25 = {r25['TP']}/{r25['FP']} | kernel = {km['vru_tp']}/{km['vru_fp']}")

    print("\n=== ActiveFoveation-960 curves | NonPilot_9700 ===")
    for m in MODELS:
        print(f"\n{m}\n   tau      P      R     F1  SmallR")
        for r in curves[m]:
            print(f"{r['tau']:>6.3f} {r['P']:.4f} {r['R']:.4f} {r['F1']:.4f} {r['SmallR']:.4f}")

    print("\n=== Recall at matched precision (best tau with P >= target) ===")
    targets = [0.55, 0.60, 0.62, 0.65, 0.70, 0.75]
    print("target  " + "".join(f"{m:>28}" for m in MODELS))
    for t in targets:
        cells = []
        for m in MODELS:
            r = at_precision(curves[m], t)
            cells.append(f"R={r['R']:.4f} S={r['SmallR']:.4f} @{r['tau']:.3f}" if r else "n/a")
        print(f"{t:>6.2f}  " + "".join(f"{c:>28}" for c in cells))

    print("\n=== Best-F1 threshold per model (selected ON VAL: exploratory, not a clean estimate) ===")
    for m in MODELS:
        b = max(curves[m], key=lambda r: r["F1"])
        print(f"{m:<14} tau={b['tau']:.3f} P={b['P']:.4f} R={b['R']:.4f} F1={b['F1']:.4f} SmallR={b['SmallR']:.4f}")

    (STAGE0 / "operating_points.json").write_text(json.dumps(curves, indent=1))


if __name__ == "__main__":
    main()
