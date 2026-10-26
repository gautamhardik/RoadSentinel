"""
RoadSentinel Stage 0 — matched-policy comparison table (CPU only).

Reads experiments/stage0_matched_eval/<model>/phase4d_metrics.json (downloaded kernel output) and prints
every model x policy on the 9,700-image non-pilot confirmation partition, plus deltas vs the original model.

Usage: python scripts/compare_stage0.py [stage0_dir]
"""

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "experiments" / "stage0_matched_eval"
MODELS = ["base_yolo11s", "phase5a_u", "phase5a_r"]
POLICIES = ["ActiveFoveation-960", "Calibrated-ActiveFoveation-960"]
SMALL = "Small (<32²)"


def row(m):
    sg = m["size_group_results"]
    per = {r["class_name"]: r for r in m["per_class_results"]}
    return {
        "P": m["vru_micro_precision_50"], "R": m["vru_micro_recall_50"], "F1": m["vru_micro_f1_50"],
        "SmallR": sg[SMALL]["vru_recall"], "TP": m["vru_tp"], "FP": m["vru_fp"], "FN": m["vru_fn"],
        "vru_mAP50": m["vru_macro_mAP_50"],
        **{f"R_{c}": per[c]["recall_50"] for c in ("person", "rider", "bike", "motor")},
    }


def main():
    results, timing = {}, {}
    for name in MODELS:
        f = ROOT / name / "phase4d_metrics.json"
        if not f.exists():
            print(f"missing: {f}")
            continue
        d = json.loads(f.read_text())
        for pol in POLICIES:
            cfg = d["configurations"][pol]
            results[(name, pol)] = row(cfg["non_pilot_confirmation_metrics"])
            timing[name] = cfg["timing_and_compute"]
        results[(name, "_sha")] = d["checkpoint"]["sha256"]

    cols = ["P", "R", "F1", "SmallR", "TP", "FP", "FN", "vru_mAP50", "R_person", "R_rider", "R_bike", "R_motor"]
    for pol in POLICIES:
        print(f"\n=== {pol} | NonPilot_9700 ===")
        print(f"{'model':<14}" + "".join(f"{c:>10}" for c in cols))
        base = results.get(("base_yolo11s", pol))
        for name in MODELS:
            r = results.get((name, pol))
            if r is None:
                continue
            print(f"{name:<14}" + "".join(f"{r[c]:>10.4f}" if isinstance(r[c], float) else f"{r[c]:>10}" for c in cols))
            if base and name != "base_yolo11s":
                delta = {c: r[c] - base[c] for c in cols}
                print(f"{'  delta':<14}" + "".join(
                    f"{delta[c] * 100:>+9.2f}p" if isinstance(r[c], float) else f"{delta[c]:>+10d}" for c in cols))

    print("\n=== timing (T4, batch-1) ===")
    for name, t in timing.items():
        print(f"{name:<14} pipeline P50 {t['batch1_pipeline_p50_ms']} ms | P95 {t['batch1_pipeline_p95_ms']} ms | "
              f"{t['throughput_fps']} FPS | crop rate {t['crop_trigger_rate']}")
    for name in MODELS:
        if (name, "_sha") in results:
            print(f"{name:<14} sha256 {results[(name, '_sha')]}")


if __name__ == "__main__":
    main()
