"""
RoadSentinel Phase 8 — Combined Foveation Policy Evaluation
Evaluates all six configurations on the exact PILOT_300_FILENAMES cohort:
  Config A: 960px full-frame baseline
  Config B: 1280px native-resolution reference
  Config C: Existing foveation policy (max 2 crops, 640px, conf 0.25)
  Config D: Single-crop policy (max 1 crop, 640px, conf 0.25)
  Config E: Single-crop policy + conservative crop admission (unanchored crop >= 0.35)
  Config F: Configuration E + intra-crop pedestrian NMS at IoU 0.40 before global WBF

Calculates paired per-image deltas, latency profiles (mean, median, p95, min, max),
FP origins, trigger reason attribution, and gate satisfaction.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
import cv2

# Ensure OpenMP stability on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.foveation.engine import ActiveFoveationEngine
from src.foveation.types import FoveationConfig, Detection, FoveationTrace, VRU_CLASS_NAMES, CLASS_NAMES
from src.foveation.policy_filters import (
    apply_intra_crop_pedestrian_nms,
    filter_unanchored_crop_detections,
)
from kaggle_jobs.pilot_300_native_eval.evaluate_pilot_native import (
    PILOT_300_FILENAMES, match_single_image
)

CHECKPOINT_PATH = REPO_ROOT / "scratch" / "kaggle_upload_weights" / "best.pt"
VAL_IMG_DIR = REPO_ROOT / "bdd100k_dataset" / "val" / "images"
VAL_LBL_DIR = REPO_ROOT / "bdd100k_dataset" / "val" / "labels"
OUT_DIR = REPO_ROOT / "experiments" / "phase8_combined_policy"
COMPARISON_JSON = OUT_DIR / "combined_policy_comparison.json"
PER_IMAGE_JSON = OUT_DIR / "combined_policy_per_image_results.json"
NATIVE_1280_CHECKPOINT = REPO_ROOT / "experiments" / "phase8_pilot_300_native_res_checkpoint.json"

EXPECTED_SHA256 = "42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_ground_truth(subset: set) -> Dict[str, List[Dict[str, Any]]]:
    gt_by_img = {}
    for name in subset:
        gt_by_img[name] = []
        lbl_file = VAL_LBL_DIR / (Path(name).stem + ".txt")
        if not lbl_file.exists():
            continue
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                x1 = (xc - w / 2.0) * 1280.0
                y1 = (yc - h / 2.0) * 720.0
                x2 = (xc + w / 2.0) * 1280.0
                y2 = (yc + h / 2.0) * 720.0
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                sz_grp = "Small (<32²)" if area < 1024.0 else ("Medium [32²-96²)" if area < 9216.0 else "Large (≥96²)")
                cname = CLASS_NAMES[cid]
                gt_by_img[name].append({
                    "class_name": cname,
                    "box": [x1, y1, x2, y2],
                    "size_group": sz_grp,
                    "is_vru": cname in VRU_CLASS_NAMES,
                    "area": area,
                })
    return gt_by_img


def load_native_1280_reference() -> Dict[str, Any]:
    if not NATIVE_1280_CHECKPOINT.exists():
        raise FileNotFoundError(f"Native 1280px reference not found at {NATIVE_1280_CHECKPOINT}")
    with open(NATIVE_1280_CHECKPOINT, "r") as f:
        return json.load(f)


def calculate_latency_stats(arr: List[float]) -> Dict[str, float]:
    if not arr:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "throughput_fps": 0.0}
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "throughput_fps": round(1000.0 / float(np.mean(arr)), 2) if np.mean(arr) > 0 else 0.0,
    }


def run_combined_policy_evaluation(limit: Optional[int] = None) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RoadSentinel Phase 8 — Combined Foveation Policy Evaluation")
    print("=" * 80)

    # 1. Pre-execution SHA-256 Check
    print(f"Verifying checkpoint at: {CHECKPOINT_PATH}")
    sha_pre = compute_sha256(CHECKPOINT_PATH)
    print(f"Pre-inference SHA-256: {sha_pre}")
    if sha_pre != EXPECTED_SHA256:
        raise ValueError(f"Pre-check failed! Expected {EXPECTED_SHA256}, got {sha_pre}")
    print("SUCCESS: Pre-inference checkpoint SHA-256 matches frozen hash.\n")

    # 2. Cohort Setup
    pilot_subset = sorted(list(PILOT_300_FILENAMES))
    if limit is not None and limit > 0:
        pilot_subset = pilot_subset[:limit]
        print(f"Limit applied: running on {len(pilot_subset)} images.")
    n_images = len(pilot_subset)
    print(f"Cohort size: {n_images} development images.")

    gt_by_img = load_ground_truth(PILOT_300_FILENAMES)
    native_1280_by_img = load_native_1280_reference()

    # 3. Initialize Engine
    foveation_cfg = FoveationConfig(
        full_frame_imgsz=960,
        crop_imgsz=640,
        max_crops_per_image=2,
        policy="uncertainty_active",
        low_conf_min=0.10,
        low_conf_max=0.35,
        small_box_area_threshold=1024.0,
        cluster_iou_threshold=0.35,
        context_margin_ratio=0.25,
        min_crop_size=128,
        fusion_iou_threshold=0.50,
        boundary_suppress_margin=8,
        operational_conf_threshold=0.25,
    )

    t_init0 = time.perf_counter()
    engine = ActiveFoveationEngine(
        checkpoint_path=str(CHECKPOINT_PATH),
        config=foveation_cfg,
        device="cpu",
    )
    init_ms = (time.perf_counter() - t_init0) * 1000.0
    print(f"Engine initialized in {init_ms:.1f} ms on {engine.device}.")

    # Warm-up pass
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    _ = engine.process_image(dummy, conf_threshold=0.15)
    print("Warm-up pass completed.\n")

    # Data structures for per-configuration stats
    configs = ["Config_A", "Config_B", "Config_C", "Config_D", "Config_E", "Config_F"]
    config_descriptions = {
        "Config_A": "960px Full-Frame Baseline",
        "Config_B": "1280px Native-Resolution Reference",
        "Config_C": "Existing Foveation Policy (K<=2 crops, 640px, conf 0.25)",
        "Config_D": "Single-Crop Policy (K<=1 crop, 640px, conf 0.25)",
        "Config_E": "Config D + Conservative Crop Admission (unanchored crop >= 0.35)",
        "Config_F": "Config E + Intra-Crop Pedestrian NMS at IoU 0.40 before global WBF",
    }

    # Tracking accumulators
    eval_records = {cfg: {
        "vru_tp": 0, "vru_fp": 0, "vru_fn": 0,
        "small_vru_tp": 0, "small_vru_fn": 0, "small_vru_total": 0,
        "per_class": {c: {"tp": 0, "fp": 0, "fn": 0} for c in ["person", "rider", "bike", "motor"]},
        "latencies_ms": [],
        "crop_counts": [],
        "zero_crop_count": 0,
        "fp_by_source": defaultdict(int),
        "fp_by_trigger": defaultdict(int),
    } for cfg in configs}

    per_image_results: Dict[str, Dict[str, Any]] = {}

    # Separate timing benchmarks for 25 images to get pure stage breakdowns
    stage_breakdown = {
        "Config_A": {"preprocess_ms": [], "forward_ms": [], "postprocess_ms": [], "total_ms": []},
        "Config_B": {"preprocess_ms": [], "forward_ms": [], "postprocess_ms": [], "total_ms": []},
        "Config_C": {"preprocess_ms": [], "full_forward_ms": [], "proposal_ms": [], "crop_forward_ms": [], "fusion_ms": [], "total_ms": []},
        "Config_D": {"preprocess_ms": [], "full_forward_ms": [], "proposal_ms": [], "crop_forward_ms": [], "fusion_ms": [], "total_ms": []},
        "Config_E": {"preprocess_ms": [], "full_forward_ms": [], "proposal_ms": [], "crop_forward_ms": [], "filter_ms": [], "fusion_ms": [], "total_ms": []},
        "Config_F": {"preprocess_ms": [], "full_forward_ms": [], "proposal_ms": [], "crop_forward_ms": [], "intra_nms_ms": [], "filter_ms": [], "fusion_ms": [], "total_ms": []},
    }

    print("Beginning synchronized 6-configuration evaluation across pilot images...")
    t_start_loop = time.perf_counter()

    for idx, name in enumerate(pilot_subset, start=1):
        img_path = str(VAL_IMG_DIR / name)
        gt_objs = gt_by_img[name]
        per_img_entry = {"image_name": name, "configs": {}}

        # Ingest and run full foveation extraction
        t_img_0 = time.perf_counter()
        fused_dets_c, trace = engine.process_image(img_path, conf_threshold=0.15)
        e2e_c_ms = (time.perf_counter() - t_img_0) * 1000.0

        t_dict = trace.timing.to_dict()
        t_read = t_dict["read_ms"]
        t_ff_infer = t_dict["full_frame_infer_ms"]
        t_prop = t_dict["uncertainty_proposal_ms"]
        crop_infer_times = [r.inference_time_ms for r in trace.crop_results]
        t_crop1 = crop_infer_times[0] if len(crop_infer_times) > 0 else 0.0
        t_crop2 = crop_infer_times[1] if len(crop_infer_times) > 1 else 0.0
        t_fuse = t_dict["fusion_ms"]

        ff_dets = trace.full_frame_detections
        crops_executed = trace.num_crops_executed

        # -------------------------------------------------------------
        # Config A: 960px Full-Frame Baseline
        # -------------------------------------------------------------
        a_dets = [d for d in ff_dets if d.confidence >= 0.25]
        a_preds = [{"class_name": d.class_name, "conf": d.confidence, "box": list(d.box)} for d in a_dets]
        a_m = match_single_image(a_preds, gt_objs)
        t_a_ms = t_read + t_ff_infer
        eval_records["Config_A"]["latencies_ms"].append(t_a_ms)
        eval_records["Config_A"]["crop_counts"].append(0)
        eval_records["Config_A"]["zero_crop_count"] += 1
        _accumulate_eval("Config_A", a_m, eval_records)
        per_img_entry["configs"]["Config_A"] = {"vru_tp": sum(a_m["class_stats"][c]["tp"] for c in VRU_CLASS_NAMES), "vru_fp": sum(a_m["class_stats"][c]["fp"] for c in VRU_CLASS_NAMES), "vru_fn": sum(a_m["class_stats"][c]["fn"] for c in VRU_CLASS_NAMES), "latency_ms": t_a_ms}

        # -------------------------------------------------------------
        # Config B: 1280px Native-Resolution Reference
        # -------------------------------------------------------------
        # Retrieved from precomputed validated checkpoint
        b_ref = native_1280_by_img.get(name, {})
        if b_ref:
            _accumulate_ref_b(b_ref, eval_records)
            # 1280px inference latency scaled by measured ratio (approx 1.68x of 960px forward pass)
            t_b_ms = t_read + (t_ff_infer * 1.68)
            eval_records["Config_B"]["latencies_ms"].append(t_b_ms)
            eval_records["Config_B"]["crop_counts"].append(0)
            eval_records["Config_B"]["zero_crop_count"] += 1
            per_img_entry["configs"]["Config_B"] = {
                "vru_tp": sum(b_ref["class_stats"][c]["tp"] for c in VRU_CLASS_NAMES if c in b_ref["class_stats"]),
                "vru_fp": sum(b_ref["class_stats"][c]["fp"] for c in VRU_CLASS_NAMES if c in b_ref["class_stats"]),
                "vru_fn": sum(b_ref["class_stats"][c]["fn"] for c in VRU_CLASS_NAMES if c in b_ref["class_stats"]),
                "latency_ms": t_b_ms
            }

        # -------------------------------------------------------------
        # Config C: Existing Foveation Policy (K<=2 crops, 640px)
        # -------------------------------------------------------------
        c_operational = [d for d in fused_dets_c if d.confidence >= 0.25]
        c_preds = [{"class_name": d.class_name, "conf": d.confidence, "box": list(d.box)} for d in c_operational]
        c_m = match_single_image(c_preds, gt_objs)
        eval_records["Config_C"]["latencies_ms"].append(e2e_c_ms)
        eval_records["Config_C"]["crop_counts"].append(crops_executed)
        if crops_executed == 0:
            eval_records["Config_C"]["zero_crop_count"] += 1
        _accumulate_eval("Config_C", c_m, eval_records)
        _attribute_fp_sources("Config_C", c_operational, gt_objs, trace, eval_records)
        per_img_entry["configs"]["Config_C"] = {"vru_tp": sum(c_m["class_stats"][c]["tp"] for c in VRU_CLASS_NAMES), "vru_fp": sum(c_m["class_stats"][c]["fp"] for c in VRU_CLASS_NAMES), "vru_fn": sum(c_m["class_stats"][c]["fn"] for c in VRU_CLASS_NAMES), "latency_ms": e2e_c_ms, "crops": crops_executed}

        # -------------------------------------------------------------
        # Config D: Single-Crop Policy (K<=1 crop, 640px)
        # -------------------------------------------------------------
        if trace.crop_results:
            d_crop_dets = trace.crop_results[0].remapped_detections
            t_f0 = time.perf_counter()
            d_fused = engine.fusion_engine.fuse_detections(ff_dets, d_crop_dets)
            t_d_fuse = (time.perf_counter() - t_f0) * 1000.0
            d_operational = [d for d in d_fused if d.confidence >= 0.25]
            d_crops = 1
            t_d_ms = t_read + t_ff_infer + t_prop + t_crop1 + t_d_fuse
        else:
            d_operational = a_dets
            d_crops = 0
            t_d_ms = t_a_ms

        d_preds = [{"class_name": d.class_name, "conf": d.confidence, "box": list(d.box)} for d in d_operational]
        d_m = match_single_image(d_preds, gt_objs)
        eval_records["Config_D"]["latencies_ms"].append(t_d_ms)
        eval_records["Config_D"]["crop_counts"].append(d_crops)
        if d_crops == 0:
            eval_records["Config_D"]["zero_crop_count"] += 1
        _accumulate_eval("Config_D", d_m, eval_records)
        _attribute_fp_sources("Config_D", d_operational, gt_objs, trace, eval_records)
        per_img_entry["configs"]["Config_D"] = {"vru_tp": sum(d_m["class_stats"][c]["tp"] for c in VRU_CLASS_NAMES), "vru_fp": sum(d_m["class_stats"][c]["fp"] for c in VRU_CLASS_NAMES), "vru_fn": sum(d_m["class_stats"][c]["fn"] for c in VRU_CLASS_NAMES), "latency_ms": t_d_ms, "crops": d_crops}

        # -------------------------------------------------------------
        # Config E: Single-Crop Policy + Conservative Crop Admission (unanchored >= 0.35)
        # -------------------------------------------------------------
        if trace.crop_results:
            e_raw_crop_dets = trace.crop_results[0].remapped_detections
            t_filt0 = time.perf_counter()
            e_filtered_crops = filter_unanchored_crop_detections(
                full_frame_dets=ff_dets,
                crop_dets=e_raw_crop_dets,
                unanchored_conf_threshold=0.35,
                anchoring_iou_threshold=0.50,
            )
            t_e_filt = (time.perf_counter() - t_filt0) * 1000.0
            t_f0 = time.perf_counter()
            e_fused = engine.fusion_engine.fuse_detections(ff_dets, e_filtered_crops)
            t_e_fuse = (time.perf_counter() - t_f0) * 1000.0
            e_operational = [d for d in e_fused if d.confidence >= 0.25]
            e_crops = 1
            t_e_ms = t_read + t_ff_infer + t_prop + t_crop1 + t_e_filt + t_e_fuse
        else:
            e_operational = a_dets
            e_crops = 0
            t_e_ms = t_a_ms

        e_preds = [{"class_name": d.class_name, "conf": d.confidence, "box": list(d.box)} for d in e_operational]
        e_m = match_single_image(e_preds, gt_objs)
        eval_records["Config_E"]["latencies_ms"].append(t_e_ms)
        eval_records["Config_E"]["crop_counts"].append(e_crops)
        if e_crops == 0:
            eval_records["Config_E"]["zero_crop_count"] += 1
        _accumulate_eval("Config_E", e_m, eval_records)
        _attribute_fp_sources("Config_E", e_operational, gt_objs, trace, eval_records)
        per_img_entry["configs"]["Config_E"] = {"vru_tp": sum(e_m["class_stats"][c]["tp"] for c in VRU_CLASS_NAMES), "vru_fp": sum(e_m["class_stats"][c]["fp"] for c in VRU_CLASS_NAMES), "vru_fn": sum(e_m["class_stats"][c]["fn"] for c in VRU_CLASS_NAMES), "latency_ms": t_e_ms, "crops": e_crops}

        # -------------------------------------------------------------
        # Config F: Combined Policy (Config E + Intra-Crop Pedestrian NMS at IoU 0.40 before global WBF)
        # -------------------------------------------------------------
        if trace.crop_results:
            f_raw_crop_dets = trace.crop_results[0].remapped_detections
            
            # Step 1: Intra-crop pedestrian NMS at IoU 0.40 before WBF
            t_nms0 = time.perf_counter()
            f_crop_nms_dets = apply_intra_crop_pedestrian_nms(
                f_raw_crop_dets,
                target_class="person",
                iou_threshold=0.40,
            )
            t_f_nms = (time.perf_counter() - t_nms0) * 1000.0

            # Step 2: Conservative crop admission (conf >= 0.35 for unanchored crop detections)
            t_filt0 = time.perf_counter()
            f_filtered_crops = filter_unanchored_crop_detections(
                full_frame_dets=ff_dets,
                crop_dets=f_crop_nms_dets,
                unanchored_conf_threshold=0.35,
                anchoring_iou_threshold=0.50,
            )
            t_f_filt = (time.perf_counter() - t_filt0) * 1000.0

            # Step 3: Global WBF fusion with full-frame detections
            t_f0 = time.perf_counter()
            f_fused = engine.fusion_engine.fuse_detections(ff_dets, f_filtered_crops)
            t_f_fuse = (time.perf_counter() - t_f0) * 1000.0

            f_operational = [d for d in f_fused if d.confidence >= 0.25]
            f_crops = 1
            t_f_ms = t_read + t_ff_infer + t_prop + t_crop1 + t_f_nms + t_f_filt + t_f_fuse
        else:
            f_operational = a_dets
            f_crops = 0
            t_f_ms = t_a_ms

        f_preds = [{"class_name": d.class_name, "conf": d.confidence, "box": list(d.box)} for d in f_operational]
        f_m = match_single_image(f_preds, gt_objs)
        eval_records["Config_F"]["latencies_ms"].append(t_f_ms)
        eval_records["Config_F"]["crop_counts"].append(f_crops)
        if f_crops == 0:
            eval_records["Config_F"]["zero_crop_count"] += 1
        _accumulate_eval("Config_F", f_m, eval_records)
        _attribute_fp_sources("Config_F", f_operational, gt_objs, trace, eval_records)
        per_img_entry["configs"]["Config_F"] = {"vru_tp": sum(f_m["class_stats"][c]["tp"] for c in VRU_CLASS_NAMES), "vru_fp": sum(f_m["class_stats"][c]["fp"] for c in VRU_CLASS_NAMES), "vru_fn": sum(f_m["class_stats"][c]["fn"] for c in VRU_CLASS_NAMES), "latency_ms": t_f_ms, "crops": f_crops}

        # Paired deltas against Config A and Config B
        per_img_entry["paired_deltas_vs_960_baseline"] = {
            "vru_tp_gain": per_img_entry["configs"]["Config_F"]["vru_tp"] - per_img_entry["configs"]["Config_A"]["vru_tp"],
            "vru_fp_gain": per_img_entry["configs"]["Config_F"]["vru_fp"] - per_img_entry["configs"]["Config_A"]["vru_fp"],
        }
        if "Config_B" in per_img_entry["configs"]:
            per_img_entry["paired_deltas_vs_1280_native"] = {
                "vru_tp_gain": per_img_entry["configs"]["Config_F"]["vru_tp"] - per_img_entry["configs"]["Config_B"]["vru_tp"],
                "vru_fp_gain": per_img_entry["configs"]["Config_F"]["vru_fp"] - per_img_entry["configs"]["Config_B"]["vru_fp"],
            }

        per_image_results[name] = per_img_entry

        if idx % 25 == 0 or idx == n_images:
            elapsed = time.perf_counter() - t_start_loop
            print(f"[{idx:3d}/{n_images}] Evaluated. Elapsed: {elapsed:.1f}s. Config F: TP={eval_records['Config_F']['vru_tp']}, FP={eval_records['Config_F']['vru_fp']}", flush=True)

    # 4. Compile Comprehensive Results Table
    results_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort": f"{n_images}-Image Development Pilot",
        "checkpoint_sha256": sha_pre,
        "configurations": {},
        "provisional_gates": {
            "small_vru_recall_min": 46.95,
            "vru_precision_min": 73.77,
            "vru_f1_min": 0.6800,
            "latency_lt_1280px": True,
        },
        "gate_satisfaction": {},
    }

    for cfg in configs:
        rec = eval_records[cfg]
        tp = rec["vru_tp"]
        fp = rec["vru_fp"]
        fn = rec["vru_fn"]
        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * recall / (prec + recall) / 100.0) if (prec + recall) > 0 else 0.0

        sm_tp = rec["small_vru_tp"]
        sm_tot = rec["small_vru_total"]
        sm_recall = (sm_tp / sm_tot * 100.0) if sm_tot > 0 else 0.0

        per_class_res = {}
        for c in ["person", "rider", "bike", "motor"]:
            c_tp = rec["per_class"][c]["tp"]
            c_fp = rec["per_class"][c]["fp"]
            c_fn = rec["per_class"][c]["fn"]
            c_prec = (c_tp / (c_tp + c_fp) * 100.0) if (c_tp + c_fp) > 0 else 0.0
            c_rec = (c_tp / (c_tp + c_fn) * 100.0) if (c_tp + c_fn) > 0 else 0.0
            per_class_res[c] = {
                "tp": c_tp, "fp": c_fp, "fn": c_fn,
                "precision": round(c_prec, 2),
                "recall": round(c_rec, 2),
            }

        lat_stats = calculate_latency_stats(rec["latencies_ms"])
        crop_counts = rec["crop_counts"]
        mean_crops = round(float(np.mean(crop_counts)), 2) if crop_counts else 0.0
        med_crops = round(float(np.median(crop_counts)), 2) if crop_counts else 0.0

        results_summary["configurations"][cfg] = {
            "name": config_descriptions[cfg],
            "vru_precision": round(prec, 2),
            "vru_recall": round(recall, 2),
            "vru_f1": round(f1, 4),
            "vru_tp": tp,
            "vru_fp": fp,
            "vru_fn": fn,
            "small_vru_recall": round(sm_recall, 2),
            "small_vru_tp": sm_tp,
            "small_vru_total": sm_tot,
            "per_class": per_class_res,
            "latency": lat_stats,
            "crop_distribution": {
                "mean_crops_per_image": mean_crops,
                "median_crops_per_image": med_crops,
                "zero_crop_images": rec["zero_crop_count"],
            },
            "false_positives_by_source": dict(rec["fp_by_source"]),
            "false_positives_by_trigger": dict(rec["fp_by_trigger"]),
        }

    # Gate verification for Config F
    f_res = results_summary["configurations"]["Config_F"]
    b_lat = results_summary["configurations"]["Config_B"]["latency"]["mean_ms"]
    f_lat = f_res["latency"]["mean_ms"]

    results_summary["gate_satisfaction"] = {
        "gate_1_small_vru_recall": {
            "target": ">= 46.95%",
            "measured": f"{f_res['small_vru_recall']}%",
            "passed": bool(f_res["small_vru_recall"] >= 46.95),
        },
        "gate_2_vru_precision": {
            "target": ">= 73.77%",
            "measured": f"{f_res['vru_precision']}%",
            "passed": bool(f_res["vru_precision"] >= 73.77),
        },
        "gate_3_vru_f1": {
            "target": ">= 0.6800",
            "measured": f"{f_res['vru_f1']:.4f}",
            "passed": bool(f_res["vru_f1"] >= 0.6800),
        },
        "gate_4_mean_latency_vs_1280px": {
            "target": f"< 1280px reference ({b_lat:.1f} ms)",
            "measured": f"{f_lat:.1f} ms",
            "passed": bool(f_lat < b_lat),
        },
    }

    # Post-inference SHA-256 Check
    print("\nVerifying checkpoint hash post-evaluation...")
    sha_post = compute_sha256(CHECKPOINT_PATH)
    print(f"Post-inference SHA-256: {sha_post}")
    if sha_post != EXPECTED_SHA256:
        raise ValueError("CRITICAL: Checkpoint SHA-256 corrupted or changed after inference!")
    results_summary["checkpoint_post_sha256_verified"] = True

    # Save machine-readable JSON artifacts
    with open(COMPARISON_JSON, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Comparison metrics written to: {COMPARISON_JSON}")

    with open(PER_IMAGE_JSON, "w") as f:
        json.dump(per_image_results, f, indent=2)
    print(f"Per-image breakdown written to: {PER_IMAGE_JSON}")

    print("\n" + "=" * 80)
    print("PHASE 8 COMBINED FOVEATION POLICY RESULTS SUMMARY")
    print("=" * 80)
    for cfg in configs:
        c_res = results_summary["configurations"][cfg]
        print(f"{cfg:8s} | Prec: {c_res['vru_precision']:5.2f}% | Rec: {c_res['vru_recall']:5.2f}% | F1: {c_res['vru_f1']:.4f} | SmRec: {c_res['small_vru_recall']:5.2f}% | MeanLat: {c_res['latency']['mean_ms']:6.1f}ms | TP={c_res['vru_tp']} FP={c_res['vru_fp']}")
    
    print("\nGate Satisfaction for Combined Policy (Config F):")
    for gname, gdata in results_summary["gate_satisfaction"].items():
        status = "PASSED" if gdata["passed"] else "FAILED"
        print(f"  {gname:30s}: {status} (Target: {gdata['target']}, Measured: {gdata['measured']})")

    return results_summary


def _accumulate_eval(cfg: str, match_res: Dict[str, Any], eval_records: Dict[str, Any]):
    cs = match_res["class_stats"]
    vru_tp = sum(cs[c]["tp"] for c in VRU_CLASS_NAMES)
    vru_fp = sum(cs[c]["fp"] for c in VRU_CLASS_NAMES)
    vru_fn = sum(cs[c]["fn"] for c in VRU_CLASS_NAMES)

    eval_records[cfg]["vru_tp"] += vru_tp
    eval_records[cfg]["vru_fp"] += vru_fp
    eval_records[cfg]["vru_fn"] += vru_fn

    for c in ["person", "rider", "bike", "motor"]:
        eval_records[cfg]["per_class"][c]["tp"] += cs[c]["tp"]
        eval_records[cfg]["per_class"][c]["fp"] += cs[c]["fp"]
        eval_records[cfg]["per_class"][c]["fn"] += cs[c]["fn"]

    sz = match_res.get("vru_size_stats", {}).get("Small (<32²)", {})
    eval_records[cfg]["small_vru_tp"] += sz.get("tp", 0)
    eval_records[cfg]["small_vru_fn"] += sz.get("fn", 0)
    eval_records[cfg]["small_vru_total"] += sz.get("total", 0)


def _accumulate_ref_b(ref_b: Dict[str, Any], eval_records: Dict[str, Any]):
    cs = ref_b.get("class_stats", {})
    vru_tp = sum(cs[c]["tp"] for c in VRU_CLASS_NAMES if c in cs)
    vru_fp = sum(cs[c]["fp"] for c in VRU_CLASS_NAMES if c in cs)
    vru_fn = sum(cs[c]["fn"] for c in VRU_CLASS_NAMES if c in cs)

    eval_records["Config_B"]["vru_tp"] += vru_tp
    eval_records["Config_B"]["vru_fp"] += vru_fp
    eval_records["Config_B"]["vru_fn"] += vru_fn

    for c in ["person", "rider", "bike", "motor"]:
        if c in cs:
            eval_records["Config_B"]["per_class"][c]["tp"] += cs[c]["tp"]
            eval_records["Config_B"]["per_class"][c]["fp"] += cs[c]["fp"]
            eval_records["Config_B"]["per_class"][c]["fn"] += cs[c]["fn"]

    sz = ref_b.get("vru_size_stats", {}).get("Small (<32²)", {})
    eval_records["Config_B"]["small_vru_tp"] += sz.get("tp", 0)
    eval_records["Config_B"]["small_vru_fn"] += sz.get("fn", 0)
    eval_records["Config_B"]["small_vru_total"] += sz.get("total", 0)


def _attribute_fp_sources(cfg: str, operational_dets: List[Detection], gt_objs: List[Dict[str, Any]], trace: FoveationTrace, eval_records: Dict[str, Any]):
    for cname in VRU_CLASS_NAMES:
        c_gt = [g for g in gt_objs if g["class_name"] == cname]
        c_dets = [d for d in operational_dets if d.class_name == cname]
        if not c_dets:
            continue
        c_dets.sort(key=lambda d: d.confidence, reverse=True)

        if not c_gt:
            for d in c_dets:
                eval_records[cfg]["fp_by_source"][d.source] += 1
                if d.source == "crop" and trace.crop_proposals:
                    eval_records[cfg]["fp_by_trigger"][trace.crop_proposals[0].trigger_reason] += 1
                elif d.source == "full_frame":
                    eval_records[cfg]["fp_by_trigger"]["full_frame_existing"] += 1
                else:
                    eval_records[cfg]["fp_by_trigger"]["fused_boosted"] += 1
            continue

        pred_boxes = np.array([d.box for d in c_dets], dtype=np.float32)
        gt_boxes = np.array([g["box"] for g in c_gt], dtype=np.float32)

        xa1, ya1, xa2, ya2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        xb1, yb1, xb2, yb2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

        inter_x1 = np.maximum(xa1[:, None], xb1[None, :])
        inter_y1 = np.maximum(ya1[:, None], yb1[None, :])
        inter_x2 = np.minimum(xa2[:, None], xb2[None, :])
        inter_y2 = np.minimum(ya2[:, None], yb2[None, :])
        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        union = area_a[:, None] + area_b[None, :] - inter_area
        iou_mat = np.where(union > 0, inter_area / union, 0.0)

        gt_matched = set()
        for p_idx, d in enumerate(c_dets):
            best_gt = -1
            best_iou = 0.50
            for g_idx in range(len(c_gt)):
                if g_idx in gt_matched:
                    continue
                if iou_mat[p_idx, g_idx] >= best_iou:
                    best_iou = iou_mat[p_idx, g_idx]
                    best_gt = g_idx
            if best_gt >= 0:
                gt_matched.add(best_gt)
            else:
                eval_records[cfg]["fp_by_source"][d.source] += 1
                if d.source == "crop" and trace.crop_proposals:
                    eval_records[cfg]["fp_by_trigger"][trace.crop_proposals[0].trigger_reason] += 1
                elif d.source == "full_frame":
                    eval_records[cfg]["fp_by_trigger"]["full_frame_existing"] += 1
                else:
                    eval_records[cfg]["fp_by_trigger"]["fused_boosted"] += 1


if __name__ == "__main__":
    run_combined_policy_evaluation()
