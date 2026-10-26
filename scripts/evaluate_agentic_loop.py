"""
RoadSentinel Phase 6 — Agentic Deliberation Loop Evaluation
Evaluates the selective VRU investigation gate on ambiguous pilot cohort detections (conf in [0.15, 0.45]).
Measures:
- Trigger rate / query budget compliance
- Classification stability & safety escalation
- Night/day illumination adaptation
- Latency overhead & projected token cost
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, os.getcwd())

import time
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
torch.set_num_threads(16)
from ultralytics import YOLO

from kaggle_jobs.calibration_benchmark.evaluate_phase4c_wbf import (
    PILOT_COHORT_IMAGE_SET,
    ALL_CLASS_NAMES,
    VRU_CLASS_NAMES,
    GT_CLASSES,
)
from src.agentic.investigator import AgenticInvestigator


def run_agentic_evaluation(num_images: int = 50):
    ckpt_path = Path("experiments/phase6_session2_error/runs/phase6_yolo11m_960_session1/weights/best.pt")
    val_img_dir = Path("bdd100k_dataset/val/images")
    assert ckpt_path.exists()
    assert val_img_dir.exists()

    model = YOLO(str(ckpt_path))
    pilot_images = sorted(list(PILOT_COHORT_IMAGE_SET))[:num_images]

    investigator = AgenticInvestigator(
        api_key=os.environ.get("NEBIUS_API_KEY"),
        model_name="nvidia/nemotron-nano-12b-v2-vl",
        base_url="https://api.tokenfactory.nebius.com/v1",
        max_cost_queries_per_frame=1,
    )

    total_detections = 0
    total_vru_detections = 0
    ambiguous_vru_candidates = 0
    investigated_candidates = 0
    safety_escalations = 0
    night_scenes = 0
    deliberation_times_ms = []
    verdicts = []

    print(f"Running Agentic Deliberation Evaluation on {len(pilot_images)} development frames...")

    for img_name in pilot_images:
        p = str(val_img_dir / img_name)
        frame = cv2.imread(p)
        frame_shape = frame.shape[:2]

        investigator.reset_frame_budget()

        # Run detector
        res = model.predict(p, imgsz=960, conf=0.01, device="cpu", verbose=False)[0]

        for b in res.boxes:
            cname = model.names[int(b.cls.item())]
            conf = float(b.conf.item())
            box = b.xyxy[0].cpu().numpy().tolist()
            is_vru = cname in VRU_CLASS_NAMES

            total_detections += 1
            if is_vru:
                total_vru_detections += 1
                if 0.15 <= conf <= 0.45:
                    ambiguous_vru_candidates += 1

                det = {"class_name": cname, "conf": conf, "box": box}
                if investigator.should_investigate(det, frame_shape):
                    t0 = time.perf_counter()
                    verdict = investigator.investigate(frame, det, image_id=img_name)
                    dt = (time.perf_counter() - t0) * 1000.0

                    investigated_candidates += 1
                    deliberation_times_ms.append(dt)
                    if verdict.safety_escalation:
                        safety_escalations += 1
                    if "night_scene_low_luminance" in verdict.visual_evidence:
                        night_scenes += 1

                    verdicts.append({
                        "img_name": img_name,
                        "initial_class": cname,
                        "initial_conf": round(conf, 3),
                        "verified_class": verdict.verified_class,
                        "epistemic_conf": round(verdict.epistemic_confidence, 3),
                        "visual_evidence": verdict.visual_evidence,
                        "safety_escalation": verdict.safety_escalation,
                        "source": verdict.source,
                        "latency_ms": round(dt, 2),
                    })

    # Nebius Token Factory cost model: ~$0.0005 per VLM call
    cost_per_query_usd = 0.0005
    projected_cost_usd = investigated_candidates * cost_per_query_usd

    results = {
        "evaluation_cohort_size": len(pilot_images),
        "total_detections_raw": total_detections,
        "total_vru_detections": total_vru_detections,
        "ambiguous_vru_in_conf_window": ambiguous_vru_candidates,
        "gated_for_investigation": investigated_candidates,
        "gating_rate_percent": round(investigated_candidates / max(len(pilot_images), 1) * 100.0, 1),
        "safety_escalation_triggers": safety_escalations,
        "night_low_light_detections": night_scenes,
        "mean_investigation_latency_ms": round(float(np.mean(deliberation_times_ms)), 2) if deliberation_times_ms else 0.0,
        "projected_cost_usd": round(projected_cost_usd, 5),
        "cost_per_frame_usd": round(projected_cost_usd / max(len(pilot_images), 1), 6),
        "live_credentials_available": investigator.has_valid_key,
        "execution_mode": "live_nebius_vlm" if investigator.has_valid_key else "deterministic_fail_safe",
        "sample_verdicts": verdicts[:10],
    }

    out_p = Path("experiments/phase6_system_eval/agentic_evaluation_report.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("AGENTIC EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Frames Evaluated:           {results['evaluation_cohort_size']}")
    print(f"Total Detections:           {results['total_detections_raw']}")
    print(f"Ambiguous VRU Candidates:   {results['ambiguous_vru_in_conf_window']}")
    print(f"Selectively Investigated:   {results['gated_for_investigation']} ({results['gating_rate_percent']}% of frames)")
    print(f"Safety Escalations Triggered:{results['safety_escalation_triggers']}")
    print(f"Night/Low-Light Detections: {results['night_low_light_detections']}")
    print(f"Deliberation Latency (mean):{results['mean_investigation_latency_ms']} ms")
    print(f"Projected Cloud Cost:       ${results['projected_cost_usd']:.5f} (${results['cost_per_frame_usd']:.6f}/frame)")
    print(f"Live API Integration:       {results['execution_mode']}")
    print(f"Results saved to: {out_p}")


if __name__ == "__main__":
    run_agentic_evaluation(num_images=50)
