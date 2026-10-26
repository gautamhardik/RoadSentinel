"""
RoadSentinel Phase 6 — full-run schedule estimate from measured smoke + probe throughput (CPU only).

MEASURED inputs (no assumptions):
  probe_report.json  mosaic-on epochs: steady-state global img/s (p10/p50/p90, CUDA-synchronized batch timings,
                     first 20 steps per epoch excluded), epoch-level img/s, holdout validation time, launch overhead
  smoke_report.json  mosaic-off epochs: epoch-level img/s (used for the final close_mosaic epochs)

EXTRAPOLATED (labelled as such): linear scaling from 6,500 to 65,000 training images per epoch.

Session model: each session = launch overhead + epochs x (train + holdout val incl. checkpoint save)
               + final holdout eval of best.pt; session 2 also runs the official-val report (~2x holdout val).
Budget per session = SESSION_H x (1 - MARGIN). Epochs are split at a fixed epoch boundary between sessions.

Usage: python scripts/phase6_schedule_estimate.py experiments/phase6_probe/probe_report.json experiments/phase6_smoke/smoke_report.json
"""

import json
import math
import sys
from pathlib import Path

FULL_TRAIN_IMAGES = 65000
SESSION_H, MARGIN = 11.0, 0.15
CLOSE_MOSAIC = 5


def main():
    probe = json.loads(Path(sys.argv[1]).read_text())
    smoke = json.loads(Path(sys.argv[2]).read_text())
    pm = probe["measured"]
    ss = pm["steady_state_all_ranks_epochs"]
    epoch_level = pm["epoch_level_img_per_s"]
    smoke_rates = [smoke["split"]["train_images"] / e["train_s"] for e in smoke["epochs"]]

    measured = {
        "mosaic_steady_img_per_s": {k: ss[f"global_img_per_s_{k}"] for k in ("p10", "p50", "p90")},
        "mosaic_epoch_level_img_per_s": epoch_level,
        "mosaic_data_wait_fraction": ss["data_wait_fraction"],
        "no_mosaic_epoch_level_img_per_s": [round(r, 2) for r in smoke_rates],
        "holdout_val_s": pm["holdout_val_s"] + [e["val_s"] for e in smoke["epochs"]],
        "launch_overhead_s": [pm["launch_overhead_s"], smoke["estimate"]["startup_and_final_eval_overhead_s"]],
    }

    val_s = max(measured["holdout_val_s"])            # per-epoch holdout validation + checkpoint save
    overhead_s = max(measured["launch_overhead_s"])   # startup, dataset scan, final eval of one launch
    official_val_s = 2 * val_s                        # 10,000 val images vs 5,000 holdout
    budget = SESSION_H * 3600 * (1 - MARGIN)

    scenarios = {
        # optimistic / typical use steady-state percentiles; conservative also caps at the slowest epoch-level rate
        "optimistic": (ss["global_img_per_s_p90"], max(smoke_rates)),
        "typical": (ss["global_img_per_s_p50"], sorted(smoke_rates)[len(smoke_rates) // 2]),
        # conservative: slowest rate measured in ANY session for every epoch type; mosaic-on probe epochs ran faster
        # than mosaic-off smoke epochs, which points to session-to-session hardware variance, so the floor is shared
        "conservative": (min(ss["global_img_per_s_p10"], *epoch_level, *smoke_rates), min(smoke_rates)),
    }
    out = {"measured": measured, "extrapolated": {}}
    for name, (r_mosaic, r_plain) in scenarios.items():
        t_m = FULL_TRAIN_IMAGES / r_mosaic + val_s     # one full mosaic epoch incl. validation
        t_p = FULL_TRAIN_IMAGES / r_plain + val_s      # one full close_mosaic epoch incl. validation
        n1 = math.floor((budget - overhead_s) / t_m)
        b2 = budget - overhead_s - official_val_s - CLOSE_MOSAIC * t_p
        n2 = CLOSE_MOSAIC + max(0, math.floor(b2 / t_m))
        out["extrapolated"][name] = {
            "mosaic_img_per_s": round(r_mosaic, 2), "no_mosaic_img_per_s": round(r_plain, 2),
            "full_mosaic_epoch_h": round(t_m / 3600, 3), "full_no_mosaic_epoch_h": round(t_p / 3600, 3),
            "session1_epochs": n1, "session2_epochs": n2, "two_session_total_epochs": n1 + n2,
            "session1_used_h": round((overhead_s + n1 * t_m) / 3600, 2),
            "session2_used_h": round((overhead_s + official_val_s + (n2 - CLOSE_MOSAIC) * t_m + CLOSE_MOSAIC * t_p) / 3600, 2),
        }
    c = out["extrapolated"]["conservative"]
    out["recommended_fixed_total_epochs"] = c["two_session_total_epochs"]
    out["recommended_split"] = {"session1": c["session1_epochs"], "session2": c["session2_epochs"]}
    out["assumptions"] = {"session_h": SESSION_H, "safety_margin": MARGIN, "close_mosaic_epochs": CLOSE_MOSAIC,
                          "full_train_images": FULL_TRAIN_IMAGES, "scaling": "linear in training images per epoch"}
    print(json.dumps(out, indent=2))
    Path(sys.argv[1]).with_name("schedule_estimate.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
