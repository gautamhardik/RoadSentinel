"""
Script to build the phase2a_evaluation_audit.ipynb reproducible notebook.
"""

import json
from pathlib import Path

nb = {
    "cells": [],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.13.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()]
    })

def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()]
    })

add_md("""# RoadSentinel — Phase 2A Evaluation Integrity Audit & Corrected Baseline
**Competition:** OpenCV AI Competition 2026, powered by AWS  
**Author:** RoadSentinel Engineering Team (DeepMind Antigravity)  
**Dataset:** BDD100K Object Detection (VRU-Enriched Validation Subset)  
**Model:** Pretrained YOLO11s (`weights/bdd100k_yolo11s.pt`)  
**Status:** Audit Complete & Scientifically Validated Baseline Established

---
## Executive Summary
This notebook documents and executes the forensic audit and mathematical repair of the RoadSentinel Phase 2A evaluation pipeline:
1. **Root-Cause Forensic Analysis:** Identify why previous metrics showed 0 traffic signs and ~0 traffic light precision.
2. **Explicit Class Mapping:** Bridge the BDD100K 2020 Scalabel checkpoint class order with the Kaggle YOLO label convention.
3. **Rigorous Evaluator Implementation:** 101-point COCO interpolated AP@50 and AP@50-95, documented IoU matching (0.50), explicit absent class handling.
4. **Reproducible Baseline:** Re-evaluate on the 100-image VRU-enriched validation sample.
""")

add_code("""import os
import sys
import time
import json
import platform
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Project Root Setup
PROJECT_ROOT = Path(r'C:\\Users\\hiten\\Downloads\\cv')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Working Directory : {os.getcwd()}")
print(f"Python Version    : {platform.python_version()}")
print(f"PyTorch Version   : {torch.__version__} (CUDA: {torch.cuda.is_available()})")
print(f"OpenCV Version    : {cv2.__version__}")
""")

add_md("""## 1. Forensic Audit: Class Order Mismatch
The dataset `bdd100k_dataset/data.yaml` originated from the Kaggle YOLO conversion, which uses the historical 2018 BDD100K CVPR class order:
`[person, rider, car, bus, truck, bike, motor, traffic light, traffic sign, train]`

Conversely, the pretrained checkpoint `weights/bdd100k_yolo11s.pt` was trained using the official BDD100K 2020 Scalabel `det.toml` hierarchy:
`[person, rider, car, truck, bus, train, motor, bike, traffic light, traffic sign]`

The previous evaluation compared integer class IDs directly (`pred_cid == gt_cid`), causing traffic lights to be evaluated as traffic signs, bikes as traffic lights, and trucks as buses.
""")

add_code("""from src.evaluation_engine import (
    GT_CLASSES,
    CHECKPOINT_CLASSES,
    CHECKPOINT_TO_GT_ID,
    ALL_CLASS_NAMES,
    VRU_CLASS_NAMES,
    DetectionEvaluator
)

print(f"{'Index':<6} {'Checkpoint Name':<18} {'GT Label Name':<18} {'Alignment':<12} {'Remapping'}")
print("-" * 75)
for ckpt_id in range(10):
    ckpt_name = CHECKPOINT_CLASSES[ckpt_id]
    gt_id = CHECKPOINT_TO_GT_ID[ckpt_id]
    gt_name_direct = GT_CLASSES[ckpt_id]
    gt_name_mapped = GT_CLASSES[gt_id]
    
    match_status = 'MATCH' if ckpt_name == gt_name_direct else 'MISMATCH'
    remap_str = f'Ckpt ID {ckpt_id} ({ckpt_name}) -> GT ID {gt_id}'
    print(f"{ckpt_id:<6} {ckpt_name:<18} {gt_name_direct:<18} {match_status:<12} {remap_str}")
""")

add_md("""## 2. Unit Testing & Sanity Verification Suite
Before running inference, we verify our evaluator against synthetic edge cases (exact matches, wrong-class false alarms, missed objects, duplicate boxes, absent classes) and real crop inspections.
""")

add_code("""from experiments.baseline_v2.test_evaluator_sanity import (
    test_iou_calculation,
    test_synthetic_scenarios,
    test_real_inspection_images
)

print("Running Evaluator Sanity Verification Suite...")
test_iou_calculation()
test_synthetic_scenarios()
test_real_inspection_images()
print("\\nAll Sanity Checks Passed!")
""")

add_md("""## 3. Hardware Profiling & Isolated Model Warm-Up
We isolate model loading latency and run 5 untimed warm-up cycles to eliminate initialization bias from inference latency benchmarks.
""")

add_code("""WEIGHTS_PATH = PROJECT_ROOT / 'weights' / 'bdd100k_yolo11s.pt'
VAL_IMG_DIR = PROJECT_ROOT / 'bdd100k_dataset' / 'val' / 'images'
VAL_LBL_DIR = PROJECT_ROOT / 'bdd100k_dataset' / 'val' / 'labels'

# Measure Model Load Time
t_load_0 = time.perf_counter()
model = YOLO(str(WEIGHTS_PATH))
t_load_1 = time.perf_counter()
load_time_ms = (t_load_1 - t_load_0) * 1000
print(f"Model Load Latency: {load_time_ms:.2f} ms")

# Warm-up (5 cycles)
dummy_img = cv2.imread(str(VAL_IMG_DIR / 'b1c66a42-6f7d68ca.jpg'))
print("Running 5 warm-up iterations...")
for _ in range(5):
    _ = model.predict(source=dummy_img, imgsz=640, conf=0.25, verbose=False)
print("Warm-up complete.")
""")

add_md("""## 4. VRU-Enriched Validation Subset Selection
We select 100 validation images consisting of:
- **70 VRU-rich images:** Images containing pedestrian, rider, bicycle, or motorcycle annotations.
- **30 Standard traffic images:** Diverse highway, urban, and night scenes.
*Note: This is a targeted VRU stress test, NOT a representative sample of the full 10,000-image BDD100K validation split.*
""")

add_code("""from experiments.baseline_v2.run_baseline_v2 import build_evaluation_dataset

eval_images = build_evaluation_dataset()
print(f"Total evaluation images: {len(eval_images)}")
print(f"Sample images (first 5): {eval_images[:5]}")
""")

add_md("""## 5. Corrected Baseline Evaluation
Running timed inference and feeding detections into `DetectionEvaluator` with explicit class alignment and 101-point COCO interpolated AP calculation.
""")

add_code("""from experiments.baseline_v2.run_baseline_v2 import run_evaluation

# Executes evaluation, computes COCO AP@50/50-95, saves baseline_metrics.json and failure cases
results = run_evaluation()
""")

add_md("""## 6. Comparison: Buggy vs Corrected Baseline
Side-by-side comparison illustrating how class alignment repaired the metrics.
""")

add_code("""# Load old baseline metrics
with open('experiments/baseline/baseline_metrics.json', 'r') as f:
    old_data = json.load(f)

# Load corrected baseline metrics
with open('experiments/baseline_v2/baseline_metrics.json', 'r') as f:
    new_data = json.load(f)

old_by_name = {m['name']: m for m in old_data['per_class_metrics']}
new_by_name = {m['name']: m for m in new_data['per_class_metrics']}

header = f"{'Class':<14} | {'Old GT':<7} {'New GT':<7} | {'Old P@50':<9} {'New P@50':<9} | {'Old R@50':<9} {'New R@50':<9} | {'Old AP50':<9} {'New AP50':<9}"
print(header)
print("-" * len(header))
for cname in ALL_CLASS_NAMES:
    o = old_by_name.get(cname, {})
    n = new_by_name.get(cname, {})
    o_gt = o.get('gt_count', '-')
    n_gt = n.get('gt_count', '-')
    o_p = f"{o.get('precision_50', 0):.3f}" if 'precision_50' in o else '-'
    n_p = f"{n.get('precision_50', 0):.3f}" if 'precision_50' in n else '-'
    o_r = f"{o.get('recall_50', 0):.3f}" if 'recall_50' in o else '-'
    n_r = f"{n.get('recall_50', 0):.3f}" if 'recall_50' in n else '-'
    o_ap = f"{o.get('map_50', 0):.3f}" if 'map_50' in o else '-'
    n_ap = f"{n.get('ap_50', 0):.3f}" if n.get('ap_50') is not None else 'N/A'
    print(f"{cname:<14} | {str(o_gt):<7} {str(n_gt):<7} | {o_p:<9} {n_p:<9} | {o_r:<9} {n_r:<9} | {o_ap:<9} {n_ap:<9}")
""")

add_md("""## 7. Vulnerable Road User (VRU) Failure Cases
Visual evidence of missed pedestrians, cyclists, and motorcyclists saved in `experiments/baseline_v2/failure_cases/`.
""")

add_code("""failures_dir = PROJECT_ROOT / 'experiments' / 'baseline_v2' / 'failure_cases'
fail_files = sorted(list(failures_dir.glob('*.jpg')))
print(f"Found {len(fail_files)} annotated failure case images.")
for f in fail_files[:5]:
    print(f"  Artifact: {f.name} ({f.stat().st_size / 1024:.1f} KB)")
""")

target_path = Path(r"C:\Users\hiten\Downloads\cv\notebooks\phase2a_evaluation_audit.ipynb")
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Successfully created: {target_path}")
