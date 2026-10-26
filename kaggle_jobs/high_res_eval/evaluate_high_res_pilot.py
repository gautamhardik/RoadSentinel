"""
RoadSentinel Phase 2B.1 — High-Resolution Inference Pilot Benchmark
Evaluates YOLO11s on 100 VRU-enriched BDD100K validation images across 640, 960, and 1280.
Measures COCO-style 101-point AP, operational VRU metrics, size-group recall, GPU latency & VRAM.
"""

import os
import sys
import subprocess

try:
    import ultralytics
except ImportError:
    print("Installing ultralytics package in Kaggle environment...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    import ultralytics

import time
import json
import hashlib
import platform
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================================
# 1. Deterministic 100-Image Evaluation Set
# ============================================================================
PILOT_IMAGE_NAMES: List[str] = [
    "b1c66a42-6f7d68ca.jpg", "b1ca2e5d-84cf9134.jpg", "b1ca8418-84a133a0.jpg",
    "b1cd1e94-549d0bfe.jpg", "b1ceb32e-3f481b43.jpg", "b1ceb32e-51852abe.jpg",
    "b1ceb32e-a106591d.jpg", "b1d0a191-03dcecc2.jpg", "b1d0a191-06deb55d.jpg",
    "b1d0a191-28f0e779.jpg", "b1d0a191-2ed2269e.jpg", "b1d0a191-5490450b.jpg",
    "b1d0a191-65deaeef.jpg", "b1d0a191-de8948f6.jpg", "b1d10d08-da110fcb.jpg",
    "b1d22ed6-f1cac061.jpg", "b1d4b62c-d9805029.jpg", "b1d7b3ac-0bdb47dc.jpg",
    "b1d7b3ac-995f9d8a.jpg", "b1d968b9-563405f4.jpg", "b1d968b9-ce42734f.jpg",
    "b1d9e136-6c94ea3f.jpg", "b1d9e136-9ab25cb3.jpg", "b1dce572-c6a8cb5e.jpg",
    "b1dd58c1-8b546ba7.jpg", "b1df722f-5bcc3db7.jpg", "b1e0c01d-dd9e6e2f.jpg",
    "b1e1a7b8-65ec7612.jpg", "b1e3e9f5-92377424.jpg", "b1e9ee0e-67e26f2e.jpg",
    "b1ebfc3c-740ec84a.jpg", "b1ebfc3c-cc9c2bb8.jpg", "b1f022d3-3774de8b.jpg",
    "b1f0efd9-e900c6e5.jpg", "b1f20aa0-3401c3bf.jpg", "b1f20aa0-50213047.jpg",
    "b1f25ff6-1ddb7e43.jpg", "b1f4491b-16256d7c.jpg", "b1f4491b-33824f31.jpg",
    "b1f4491b-97465266.jpg", "b1f4491b-9958bd99.jpg", "b1f4491b-cf446195.jpg",
    "b1f4491b-d8d1459c.jpg", "b1f62c41-ed0c6521.jpg", "b1fbf878-b31a8293.jpg",
    "b1fc95c9-644e3c3f.jpg", "b1ff4656-ebcfeb35.jpg", "b20234fd-822029be.jpg",
    "b2046bcf-8bc6a044.jpg", "b204a5c1-05981158.jpg", "b204a5c1-064b0040.jpg",
    "b204a5c1-fa3d5b88.jpg", "b206a78b-99f405ab.jpg", "b2080dc7-f9b98a5f.jpg",
    "b20b69d2-bd242bf0.jpg", "b20e291a-32ac11c1.jpg", "b20e291a-6012d836.jpg",
    "b20ff95c-b9444127.jpg", "b213e4eb-09c01a17.jpg", "b214d1e1-f248c616.jpg",
    "b215943a-10e44587.jpg", "b216243d-ad4306b9.jpg", "b21742c2-18d3463a.jpg",
    "b2194b15-1825056a.jpg", "b21bfb83-ea32f716.jpg", "b21c68e6-65674a17.jpg",
    "b2208b0f-2796a692.jpg", "b222c329-5dc8dbf7.jpg", "b22a4d9f-48b2e986.jpg",
    "b22f174d-a19c875e.jpg",
    # 30 standard traffic images
    "b1c81faa-3df17267.jpg", "b1c81faa-c80764c5.jpg", "b1c9c847-3bda4659.jpg",
    "b1cac6a7-04e33135.jpg", "b1cd1e94-26dd524f.jpg", "b1ceb32e-813f84b2.jpg",
    "b1cebfb7-284f5117.jpg", "b1d0091f-75824d0d.jpg", "b1d0091f-f2c2d2ae.jpg",
    "b1d10d08-5b108225.jpg", "b1d10d08-743fd86c.jpg", "b1d10d08-c35503b8.jpg",
    "b1d10d08-ec660956.jpg", "b1d22449-117aa773.jpg", "b1d22449-15fb948f.jpg",
    "b1d3907b-2278601b.jpg", "b1d4b62c-60aab822.jpg", "b1d4b62c-89eeb5d3.jpg",
    "b1d59b1f-a38aec79.jpg", "b1d7b3ac-2a92e19f.jpg", "b1d7b3ac-36f2d3b7.jpg",
    "b1d7b3ac-5744370e.jpg", "b1d7b3ac-5af8623b.jpg", "b1d7b3ac-9e14f05f.jpg",
    "b1d7b3ac-afa57f22.jpg", "b1d8735d-eee9f184.jpg", "b1d971b4-ac67ca0d.jpg",
    "b1da9c53-0f3d4c5d.jpg", "b1dac7f7-6b2e0382.jpg", "b1db7e22-cfa74dc3.jpg"
]


# ============================================================================
# 2. Class Definitions and Explicit Semantic Remapping
# ============================================================================
GT_CLASSES: Dict[int, str] = {
    0: "person",
    1: "rider",
    2: "car",
    3: "bus",
    4: "truck",
    5: "bike",
    6: "motor",
    7: "traffic light",
    8: "traffic sign",
    9: "train",
}

CHECKPOINT_CLASSES: Dict[int, str] = {
    0: "person",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motor",
    7: "bike",
    8: "traffic light",
    9: "traffic sign",
}

CHECKPOINT_TO_GT_ID: Dict[int, int] = {
    0: 0,  # person       -> person
    1: 1,  # rider        -> rider
    2: 2,  # car          -> car
    3: 4,  # truck (ckpt) -> truck (gt 4)
    4: 3,  # bus (ckpt)   -> bus (gt 3)
    5: 9,  # train (ckpt) -> train (gt 9)
    6: 6,  # motor        -> motor
    7: 5,  # bike (ckpt)  -> bike (gt 5)
    8: 7,  # t-light(ckpt)-> t-light (gt 7)
    9: 8,  # t-sign (ckpt)-> t-sign (gt 8)
}

ALL_CLASS_NAMES: List[str] = [
    "person", "rider", "car", "bus", "truck", "bike", "motor",
    "traffic light", "traffic sign", "train"
]
VRU_CLASS_NAMES: List[str] = ["person", "rider", "bike", "motor"]

IOU_THRESHOLDS = [round(x, 2) for x in np.linspace(0.50, 0.95, 10)]


# ============================================================================
# 3. Vectorized Math & Evaluation Helpers
# ============================================================================
def compute_box_iou_vec(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Computes 1-to-N IoU between box [4] and boxes [N, 4]. [x1, y1, x2, y2]."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)

    xi1 = np.maximum(box[0], boxes[:, 0])
    yi1 = np.maximum(box[1], boxes[:, 1])
    xi2 = np.minimum(box[2], boxes[:, 2])
    yi2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, xi2 - xi1) * np.maximum(0.0, yi2 - yi1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter

    return np.where(union > 0, inter / union, 0.0)


def compute_ap_coco(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """101-point interpolated COCO-style AP: p_interp(r) = max_{r' >= r} p(r')."""
    if len(recalls) == 0:
        return 0.0

    recall_thresholds = np.linspace(0.0, 1.0, 101)
    interpolated_precisions = np.zeros(101)

    for i, r in enumerate(recall_thresholds):
        matching = precisions[recalls >= r]
        if len(matching) > 0:
            interpolated_precisions[i] = np.max(matching)
        else:
            interpolated_precisions[i] = 0.0

    return float(np.mean(interpolated_precisions))


def get_file_sha256(filepath: Path) -> str:
    """Calculates SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


# ============================================================================
# 4. Fast Path Discovery
# ============================================================================
def discover_paths() -> Tuple[Path, Path, Path]:
    print("Checking known dataset and checkpoint candidate paths...")
    CANDIDATES = [
        # Kaggle paths
        (
            Path("/kaggle/input/datasets/hardikgautam25/roadsentinel-weights/bdd100k_yolo11s.pt"),
            Path("/kaggle/input/datasets/a7madmostafa/bdd100k-yolo/val/images"),
            Path("/kaggle/input/datasets/a7madmostafa/bdd100k-yolo/val/labels"),
        ),
        (
            Path("/kaggle/input/roadsentinel-weights/bdd100k_yolo11s.pt"),
            Path("/kaggle/input/bdd100k-yolo/val/images"),
            Path("/kaggle/input/bdd100k-yolo/val/labels"),
        ),
        # Local paths
        (
            Path("weights/bdd100k_yolo11s.pt"),
            Path("bdd100k_dataset/val/images"),
            Path("bdd100k_dataset/val/labels"),
        ),
    ]

    for ckpt, img_dir, lbl_dir in CANDIDATES:
        if ckpt.exists() and img_dir.exists() and lbl_dir.exists():
            print(f"  Checkpoint: {ckpt} ({ckpt.stat().st_size / 1024 / 1024:.1f} MB)")
            print(f"  Val Images: {img_dir}")
            print(f"  Val Labels: {lbl_dir}")
            return ckpt, img_dir, lbl_dir

    # Fallback search if exact candidates fail
    input_root = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path(".")
    print(f"Falling back to scanning {input_root}...")
    ckpt_path = None
    for p in input_root.rglob("bdd100k_yolo11s.pt"):
        if p.is_file():
            ckpt_path = p
            break
    val_img_dir = None
    val_lbl_dir = None
    for p in input_root.rglob("images"):
        if p.parent.name == "val" and p.is_dir():
            val_img_dir = p
            candidate_lbl = p.parent / "labels"
            if candidate_lbl.exists():
                val_lbl_dir = candidate_lbl
            break

    if ckpt_path is None or val_img_dir is None or val_lbl_dir is None:
        raise FileNotFoundError("Could not find validation images/labels or weights checkpoint.")

    print(f"  Found Checkpoint: {ckpt_path}")
    print(f"  Found Val Images: {val_img_dir}")
    print(f"  Found Val Labels: {val_lbl_dir}")
    return ckpt_path, val_img_dir, val_lbl_dir


# ============================================================================
# 5. Main Multi-Resolution Benchmark Runner
# ============================================================================
def main():
    print("=" * 80)
    print("ROADSENTINEL PHASE 2B.1 — HIGH-RESOLUTION INFERENCE PILOT BENCHMARK")
    print("=" * 80)

    # Output directories
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("experiments/high_res_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir = out_dir / "comparison_cases"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    # Discover paths
    ckpt_path, val_img_dir, val_lbl_dir = discover_paths()
    ckpt_hash = get_file_sha256(ckpt_path)
    print(f"Checkpoint SHA-256: {ckpt_hash}")

    # Hardware detection
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)"
    print(f"\nHardware Environment:")
    print(f"  Device        : {device} ({gpu_name})")
    print(f"  PyTorch       : {torch.__version__} | CUDA: {torch.version.cuda}")
    print(f"  Platform      : {platform.platform()} ({platform.processor()})")

    # Load 100 validation images
    available_imgs = [fname for fname in PILOT_IMAGE_NAMES if (val_img_dir / fname).exists()]
    print(f"\nVerified {len(available_imgs)}/100 pilot validation images exist on disk.")
    if len(available_imgs) != 100:
        raise RuntimeError(f"Expected exactly 100 validation images, but found {len(available_imgs)}.")

    # Load and cache ground-truth annotations and size categorization
    print("\nParsing ground-truth annotations...")
    gt_by_class_img = defaultdict(lambda: defaultdict(list))
    gt_counts = Counter()
    all_gt_boxes_meta = []  # For size-group categorization: {img_name, class_name, box, area, size_group}

    for img_name in available_imgs:
        lbl_file = val_lbl_dir / (Path(img_name).stem + ".txt")
        if not lbl_file.exists():
            continue
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                gt_cid = int(parts[0])
                cname = GT_CLASSES.get(gt_cid, "unknown")
                xc, yc, w, h = map(float, parts[1:])
                # Convert normalized coordinates to 1280x720 pixel space
                x1 = (xc - w / 2.0) * 1280.0
                y1 = (yc - h / 2.0) * 720.0
                x2 = (xc + w / 2.0) * 1280.0
                y2 = (yc + h / 2.0) * 720.0
                box = [x1, y1, x2, y2]
                area = (x2 - x1) * (y2 - y1)

                # Size group standard: Small (< 32^2 = 1024), Medium [1024, 9216), Large (>= 96^2 = 9216)
                if area < 1024.0:
                    sg = "Small (<32²)"
                elif area < 9216.0:
                    sg = "Medium [32²-96²)"
                else:
                    sg = "Large (≥96²)"

                gt_by_class_img[cname][img_name].append(box)
                gt_counts[cname] += 1
                all_gt_boxes_meta.append({
                    "img_name": img_name,
                    "class_name": cname,
                    "is_vru": cname in VRU_CLASS_NAMES,
                    "box": box,
                    "area": area,
                    "size_group": sg,
                })

    for cname in ALL_CLASS_NAMES:
        for img_name in available_imgs:
            arr = np.array(gt_by_class_img[cname][img_name], dtype=np.float32) if img_name in gt_by_class_img[cname] else np.zeros((0, 4), dtype=np.float32)
            gt_by_class_img[cname][img_name] = arr

    total_gt_boxes = sum(gt_counts.values())
    total_vru_gt = sum(gt_counts[c] for c in VRU_CLASS_NAMES)
    print(f"Total Ground-Truth Boxes in Pilot : {total_gt_boxes}")
    print(f"Total VRU Ground-Truth Boxes       : {total_vru_gt} (person={gt_counts['person']}, rider={gt_counts['rider']}, bike={gt_counts['bike']}, motor={gt_counts['motor']})")

    # Define Configurations
    CONFIGS = [
        {"name": "Reference-640", "imgsz": 640, "batch_size": 16},
        {"name": "HighRes-960", "imgsz": 960, "batch_size": 16},
        {"name": "MaxRes-1280", "imgsz": 1280, "batch_size": 8},
    ]

    # Model initialization
    print(f"\nLoading model {ckpt_path.name}...")
    model = YOLO(str(ckpt_path))
    model.to(device)

    all_config_results = {}
    preds_per_config_by_img = {}

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        imgsz = cfg["imgsz"]
        batch_size = cfg["batch_size"]

        print("\n" + "-" * 75)
        print(f"EXECUTING CONFIGURATION: {cfg_name} (imgsz={imgsz}, initial batch_size={batch_size})")
        print("-" * 75)

        dummy_sample_path = str(val_img_dir / available_imgs[0])
        dummy_bgr = cv2.imread(dummy_sample_path)

        # 5 untimed warm-up cycles
        print(f"Executing 5 warm-up cycles at imgsz={imgsz}...")
        for _ in range(5):
            _ = model.predict(source=dummy_bgr, imgsz=imgsz, conf=0.25, device=device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Batch Inference Loop with OOM Protection
        batch_paths = [str(val_img_dir / fname) for fname in available_imgs]
        predictions_by_class = defaultdict(list)
        preds_by_img = defaultdict(list)

        t_infer_start = time.perf_counter()
        actual_batch_size = batch_size
        oom_occurred = False

        try:
            for i in range(0, len(available_imgs), actual_batch_size):
                chunk_paths = batch_paths[i : i + actual_batch_size]
                chunk_names = available_imgs[i : i + actual_batch_size]

                results = model.predict(
                    source=chunk_paths,
                    imgsz=imgsz,
                    conf=0.01,  # Full curve capture
                    device=device,
                    verbose=False,
                )

                for img_name, res in zip(chunk_names, results):
                    for det in res.boxes:
                        ckpt_cid = int(det.cls.item())
                        conf = float(det.conf.item())
                        box = det.xyxy[0].cpu().numpy()
                        cname = CHECKPOINT_CLASSES[ckpt_cid]

                        pred_item = {
                            "img_name": img_name,
                            "class_name": cname,
                            "conf": conf,
                            "box": box,
                        }
                        predictions_by_class[cname].append(pred_item)
                        preds_by_img[img_name].append(pred_item)

        except torch.cuda.OutOfMemoryError as e:
            print(f"WARNING: CUDA OutOfMemory at batch_size={actual_batch_size}! Reducing batch size...")
            torch.cuda.empty_cache()
            actual_batch_size = max(1, actual_batch_size // 2)
            oom_occurred = True
            predictions_by_class.clear()
            preds_by_img.clear()
            t_infer_start = time.perf_counter()

            for i in range(0, len(available_imgs), actual_batch_size):
                chunk_paths = batch_paths[i : i + actual_batch_size]
                chunk_names = available_imgs[i : i + actual_batch_size]
                results = model.predict(source=chunk_paths, imgsz=imgsz, conf=0.01, device=device, verbose=False)
                for img_name, res in zip(chunk_names, results):
                    for det in res.boxes:
                        ckpt_cid = int(det.cls.item())
                        conf = float(det.conf.item())
                        box = det.xyxy[0].cpu().numpy()
                        cname = CHECKPOINT_CLASSES[ckpt_cid]
                        pred_item = {"img_name": img_name, "class_name": cname, "conf": conf, "box": box}
                        predictions_by_class[cname].append(pred_item)
                        preds_by_img[img_name].append(pred_item)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_vram_mb = None

        t_infer_end = time.perf_counter()
        infer_duration_s = t_infer_end - t_infer_start
        mean_ms_per_img = (infer_duration_s / len(available_imgs)) * 1000
        throughput_fps = len(available_imgs) / infer_duration_s

        print(f"Inference Completed in {infer_duration_s:.2f} s ({throughput_fps:.2f} FPS, {mean_ms_per_img:.2f} ms/img)")
        print(f"Peak GPU VRAM Allocated: {peak_vram_mb:.1f} MB" if peak_vram_mb is not None else "Peak GPU VRAM: N/A (CPU)")

        preds_per_config_by_img[cfg_name] = preds_by_img

        # --------------------------------------------------------------------
        # Quantitative Metric Calculation
        # --------------------------------------------------------------------
        per_class_results = []
        op_matched_gt_flags = {cname: {img_name: np.zeros(len(gts), dtype=bool) for img_name, gts in gt_by_class_img[cname].items()} for cname in ALL_CLASS_NAMES}

        for cname in ALL_CLASS_NAMES:
            n_gt = gt_counts[cname]
            preds = predictions_by_class[cname]
            n_preds = len(preds)

            # Operational metrics (tau=0.25, IoU=0.50)
            op_preds = [p for p in preds if p["conf"] >= 0.25]
            n_op = len(op_preds)

            op_preds_by_img = defaultdict(list)
            for p in op_preds:
                op_preds_by_img[p["img_name"]].append(p)

            op_tp = 0
            op_fp = 0

            for img_name, img_preds in op_preds_by_img.items():
                gts = gt_by_class_img[cname].get(img_name, np.zeros((0, 4), dtype=np.float32))
                if len(gts) == 0:
                    op_fp += len(img_preds)
                    continue

                matched = op_matched_gt_flags[cname][img_name]
                img_preds.sort(key=lambda x: x["conf"], reverse=True)

                for p in img_preds:
                    ious = compute_box_iou_vec(p["box"], gts)
                    best_idx = np.argmax(ious)
                    best_iou = ious[best_idx]
                    if best_iou >= 0.50 and not matched[best_idx]:
                        op_tp += 1
                        matched[best_idx] = True
                    else:
                        op_fp += 1

            op_fn = n_gt - op_tp
            op_prec = (op_tp / n_op) if n_op > 0 else 0.0
            op_rec = (op_tp / n_gt) if n_gt > 0 else 0.0
            op_f1 = (2 * op_prec * op_rec / (op_prec + op_rec)) if (op_prec + op_rec) > 0 else 0.0

            # COCO-style 101-point AP across [0.50 : 0.05 : 0.95]
            if n_gt == 0:
                ap_50 = 0.0
                ap_50_95 = 0.0
            else:
                preds.sort(key=lambda x: x["conf"], reverse=True)
                ap_by_iou = {}

                for iou_thr in IOU_THRESHOLDS:
                    matched_map = {
                        img_name: np.zeros(len(gts), dtype=bool)
                        for img_name, gts in gt_by_class_img[cname].items()
                    }
                    tp_vec = np.zeros(n_preds, dtype=np.float32)
                    fp_vec = np.zeros(n_preds, dtype=np.float32)

                    for idx, p in enumerate(preds):
                        img_name = p["img_name"]
                        gts = gt_by_class_img[cname].get(img_name, np.zeros((0, 4), dtype=np.float32))
                        if len(gts) == 0:
                            fp_vec[idx] = 1.0
                            continue

                        ious = compute_box_iou_vec(p["box"], gts)
                        best_idx = np.argmax(ious)
                        best_iou = ious[best_idx]

                        if best_iou >= iou_thr and not matched_map[img_name][best_idx]:
                            tp_vec[idx] = 1.0
                            matched_map[img_name][best_idx] = True
                        else:
                            fp_vec[idx] = 1.0

                    cum_tp = np.cumsum(tp_vec)
                    cum_fp = np.cumsum(fp_vec)
                    rec = cum_tp / n_gt
                    prec = cum_tp / (cum_tp + cum_fp)
                    ap_by_iou[iou_thr] = compute_ap_coco(rec, prec)

                ap_50 = ap_by_iou.get(0.50, 0.0)
                ap_50_95 = float(np.mean([ap_by_iou[t] for t in IOU_THRESHOLDS]))

            per_class_results.append({
                "class_name": cname,
                "is_vru": cname in VRU_CLASS_NAMES,
                "gt_count": n_gt,
                "total_preds": n_preds,
                "operational_preds": n_op,
                "tp_50": op_tp,
                "fp_50": op_fp,
                "fn_50": op_fn,
                "precision_50": round(op_prec, 4),
                "recall_50": round(op_rec, 4),
                "f1_50": round(op_f1, 4),
                "ap_50": round(ap_50, 4),
                "ap_50_95": round(ap_50_95, 4),
            })

        # Summary Aggregates
        present_classes = [c for c in per_class_results if c["gt_count"] > 0]
        overall_map50 = float(np.mean([c["ap_50"] for c in present_classes]))
        overall_map50_95 = float(np.mean([c["ap_50_95"] for c in present_classes]))
        overall_macro_prec = float(np.mean([c["precision_50"] for c in present_classes]))
        overall_macro_rec = float(np.mean([c["recall_50"] for c in present_classes]))

        # VRU Specific Aggregates
        vru_classes = [c for c in per_class_results if c["is_vru"]]
        vru_macro_map50 = float(np.mean([c["ap_50"] for c in vru_classes]))
        vru_macro_map50_95 = float(np.mean([c["ap_50_95"] for c in vru_classes]))
        vru_tp_total = sum(c["tp_50"] for c in vru_classes)
        vru_fp_total = sum(c["fp_50"] for c in vru_classes)
        vru_fn_total = sum(c["fn_50"] for c in vru_classes)
        vru_preds_total = sum(c["operational_preds"] for c in vru_classes)

        vru_micro_prec = (vru_tp_total / vru_preds_total) if vru_preds_total > 0 else 0.0
        vru_micro_rec = (vru_tp_total / total_vru_gt) if total_vru_gt > 0 else 0.0
        vru_micro_f1 = (2 * vru_micro_prec * vru_micro_rec / (vru_micro_prec + vru_micro_rec)) if (vru_micro_prec + vru_micro_rec) > 0 else 0.0

        # Size Group Recall Breakdown
        size_group_stats = defaultdict(lambda: {"total_gt": 0, "matched_gt": 0, "vru_gt": 0, "vru_matched": 0})
        for item in all_gt_boxes_meta:
            sg = item["size_group"]
            cname = item["class_name"]
            img_name = item["img_name"]
            is_vru = item["is_vru"]

            size_group_stats[sg]["total_gt"] += 1
            if is_vru:
                size_group_stats[sg]["vru_gt"] += 1

            gts = gt_by_class_img[cname][img_name]
            box_np = np.array(item["box"], dtype=np.float32)
            dists = np.sum(np.abs(gts - box_np), axis=1) if len(gts) > 0 else np.array([])
            if len(dists) > 0 and np.min(dists) < 1e-3:
                matched_idx = np.argmin(dists)
                if op_matched_gt_flags[cname][img_name][matched_idx]:
                    size_group_stats[sg]["matched_gt"] += 1
                    if is_vru:
                        size_group_stats[sg]["vru_matched"] += 1

        size_group_results = {}
        for sg, st in sorted(size_group_stats.items()):
            rec_all = (st["matched_gt"] / st["total_gt"]) if st["total_gt"] > 0 else 0.0
            rec_vru = (st["vru_matched"] / st["vru_gt"]) if st["vru_gt"] > 0 else 0.0
            size_group_results[sg] = {
                "total_gt": st["total_gt"],
                "matched_gt": st["matched_gt"],
                "overall_recall": round(rec_all, 4),
                "vru_gt": st["vru_gt"],
                "vru_matched": st["vru_matched"],
                "vru_recall": round(rec_vru, 4),
            }

        config_report = {
            "name": cfg_name,
            "imgsz": imgsz,
            "initial_batch_size": batch_size,
            "actual_batch_size": actual_batch_size,
            "oom_occurred": oom_occurred,
            "latency": {
                "num_images": len(available_imgs),
                "total_infer_time_s": round(infer_duration_s, 3),
                "amortized_ms_per_image": round(mean_ms_per_img, 2),
                "throughput_fps": round(throughput_fps, 2),
                "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb is not None else None,
            },
            "summary": {
                "overall_mAP_50": round(overall_map50, 4),
                "overall_mAP_50_95": round(overall_map50_95, 4),
                "overall_macro_precision_50": round(overall_macro_prec, 4),
                "overall_macro_recall_50": round(overall_macro_rec, 4),
            },
            "vru_aggregates": {
                "vru_macro_mAP_50": round(vru_macro_map50, 4),
                "vru_macro_mAP_50_95": round(vru_macro_map50_95, 4),
                "vru_micro_precision_50": round(vru_micro_prec, 4),
                "vru_micro_recall_50": round(vru_micro_rec, 4),
                "vru_micro_f1_50": round(vru_micro_f1, 4),
            },
            "per_class_results": per_class_results,
            "size_group_results": size_group_results,
        }

        all_config_results[cfg_name] = config_report

        print(f"Results for {cfg_name}:")
        print(f"  Overall mAP@50     : {overall_map50 * 100:.2f}% | mAP@50-95: {overall_map50_95 * 100:.2f}%")
        print(f"  VRU Macro mAP@50   : {vru_macro_map50 * 100:.2f}% | Recall@50: {vru_micro_rec * 100:.2f}%")
        print(f"  Small VRU Recall   : {size_group_results.get('Small (<32²)', {}).get('vru_recall', 0.0) * 100:.2f}%")

    # ========================================================================
    # 6. Failure & Comparison Case Analysis
    # ========================================================================
    print("\nExtracting and rendering representative comparison cases across resolutions...")
    ref_preds = preds_per_config_by_img["Reference-640"]
    high_preds = preds_per_config_by_img["HighRes-960"]
    max_preds = preds_per_config_by_img["MaxRes-1280"]

    saved_comparison_cases = []

    for img_name in available_imgs:
        img_path = val_img_dir / img_name
        bgr = cv2.imread(str(img_path))
        h, w = bgr.shape[:2]

        vru_gts = []
        for vc in VRU_CLASS_NAMES:
            for b in gt_by_class_img[vc].get(img_name, []):
                vru_gts.append({"class": vc, "box": b})

        p640 = [p for p in ref_preds[img_name] if p["conf"] >= 0.25 and p["class_name"] in VRU_CLASS_NAMES]
        p960 = [p for p in high_preds[img_name] if p["conf"] >= 0.25 and p["class_name"] in VRU_CLASS_NAMES]
        p1280 = [p for p in max_preds[img_name] if p["conf"] >= 0.25 and p["class_name"] in VRU_CLASS_NAMES]

        # Case A: Recovered VRU
        if len(vru_gts) > 0 and len([c for c in saved_comparison_cases if c["case_type"] == "Recovered VRU"]) < 4:
            for gt in vru_gts:
                iou_640 = float(np.max(compute_box_iou_vec(np.array(gt["box"], dtype=np.float32), np.array([p["box"] for p in p640], dtype=np.float32)))) if len(p640) > 0 else 0.0
                iou_960 = float(np.max(compute_box_iou_vec(np.array(gt["box"], dtype=np.float32), np.array([p["box"] for p in p960], dtype=np.float32)))) if len(p960) > 0 else 0.0
                iou_1280 = float(np.max(compute_box_iou_vec(np.array(gt["box"], dtype=np.float32), np.array([p["box"] for p in p1280], dtype=np.float32)))) if len(p1280) > 0 else 0.0

                if iou_640 < 0.50 and (iou_960 >= 0.50 or iou_1280 >= 0.50):
                    case_title = f"CaseA_Recovered_{gt['class']}_{img_name}"
                    vis = bgr.copy()
                    gb = [int(x) for x in gt["box"]]
                    cv2.rectangle(vis, (gb[0], gb[1]), (gb[2], gb[3]), (0, 255, 0), 2)
                    cv2.putText(vis, f"GT: {gt['class']}", (gb[0], max(15, gb[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    for hp in p960 + p1280:
                        hb = [int(x) for x in hp["box"]]
                        cv2.rectangle(vis, (hb[0], hb[1]), (hb[2], hb[3]), (0, 255, 255), 2)
                        cv2.putText(vis, f"HighRes {hp['conf']:.2f}", (hb[0], min(h - 5, hb[3] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                    cv2.imwrite(str(comparisons_dir / f"{case_title}.jpg"), vis)
                    saved_comparison_cases.append({
                        "case_type": "Recovered VRU",
                        "image": img_name,
                        "class": gt["class"],
                        "file": f"{case_title}.jpg",
                        "notes": "Missed at 640px; successfully recovered at higher resolution.",
                    })
                    break

        # Case C: Persistent Miss
        if len(vru_gts) > 0 and len([c for c in saved_comparison_cases if c["case_type"] == "Persistent Miss"]) < 4:
            for gt in vru_gts:
                iou_640 = float(np.max(compute_box_iou_vec(np.array(gt["box"], dtype=np.float32), np.array([p["box"] for p in p640], dtype=np.float32)))) if len(p640) > 0 else 0.0
                iou_960 = float(np.max(compute_box_iou_vec(np.array(gt["box"], dtype=np.float32), np.array([p["box"] for p in p960], dtype=np.float32)))) if len(p960) > 0 else 0.0
                iou_1280 = float(np.max(compute_box_iou_vec(np.array(gt["box"], dtype=np.float32), np.array([p["box"] for p in p1280], dtype=np.float32)))) if len(p1280) > 0 else 0.0

                if iou_640 < 0.50 and iou_960 < 0.50 and iou_1280 < 0.50:
                    case_title = f"CaseC_PersistentMiss_{gt['class']}_{img_name}"
                    vis = bgr.copy()
                    gb = [int(x) for x in gt["box"]]
                    cv2.rectangle(vis, (gb[0], gb[1]), (gb[2], gb[3]), (0, 0, 255), 2)
                    cv2.putText(vis, f"MISSED ALL: {gt['class']}", (gb[0], max(15, gb[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imwrite(str(comparisons_dir / f"{case_title}.jpg"), vis)
                    saved_comparison_cases.append({
                        "case_type": "Persistent Miss",
                        "image": img_name,
                        "class": gt["class"],
                        "file": f"{case_title}.jpg",
                        "notes": "Missed across all resolutions; requires active foveation.",
                    })
                    break

    # ========================================================================
    # 7. Write Machine-Readable Outputs
    # ========================================================================
    final_output = {
        "benchmark_name": "RoadSentinel Phase 2B.1 — High-Resolution Inference Pilot Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": {
            "file": ckpt_path.name,
            "sha256": ckpt_hash,
        },
        "hardware": {
            "device": device,
            "gpu_name": gpu_name,
            "pytorch_version": torch.__version__,
            "cuda_version": str(torch.version.cuda) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
        "pilot_dataset": {
            "num_images": len(available_imgs),
            "total_gt_boxes": total_gt_boxes,
            "vru_gt_boxes": total_vru_gt,
        },
        "configurations": all_config_results,
        "visual_comparison_cases": saved_comparison_cases,
    }

    metrics_json_path = out_dir / "high_res_pilot_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved machine-readable metrics to: {metrics_json_path}")

    csv_class_path = out_dir / "per_class_comparison.csv"
    with open(csv_class_path, "w", encoding="utf-8") as f:
        f.write("Class,Target,GT_Count,Ref640_AP50,HighRes960_AP50,MaxRes1280_AP50,Ref640_Rec50,HighRes960_Rec50,MaxRes1280_Rec50\n")
        for cname in ALL_CLASS_NAMES:
            r640 = next(c for c in all_config_results["Reference-640"]["per_class_results"] if c["class_name"] == cname)
            r960 = next(c for c in all_config_results["HighRes-960"]["per_class_results"] if c["class_name"] == cname)
            r1280 = next(c for c in all_config_results["MaxRes-1280"]["per_class_results"] if c["class_name"] == cname)
            target = "VRU" if cname in VRU_CLASS_NAMES else "Standard"
            f.write(f"{cname},{target},{r640['gt_count']},{r640['ap_50']},{r960['ap_50']},{r1280['ap_50']},{r640['recall_50']},{r960['recall_50']},{r1280['recall_50']}\n")
    print(f"Saved per-class comparison CSV to: {csv_class_path}")

    csv_sg_path = out_dir / "size_group_analysis.csv"
    with open(csv_sg_path, "w", encoding="utf-8") as f:
        f.write("SizeGroup,Total_GT,VRU_GT,Ref640_VRU_Recall,HighRes960_VRU_Recall,MaxRes1280_VRU_Recall,Ref640_All_Recall,HighRes960_All_Recall,MaxRes1280_All_Recall\n")
        sgs = sorted(all_config_results["Reference-640"]["size_group_results"].keys())
        for sg in sgs:
            sg640 = all_config_results["Reference-640"]["size_group_results"][sg]
            sg960 = all_config_results["HighRes-960"]["size_group_results"][sg]
            sg1280 = all_config_results["MaxRes-1280"]["size_group_results"][sg]
            f.write(f'"{sg}",{sg640["total_gt"]},{sg640["vru_gt"]},{sg640["vru_recall"]},{sg960["vru_recall"]},{sg1280["vru_recall"]},{sg640["overall_recall"]},{sg960["overall_recall"]},{sg1280["overall_recall"]}\n')
    print(f"Saved size-group comparison CSV to: {csv_sg_path}")

    print("\n" + "=" * 80)
    print("PILOT BENCHMARK EXECUTION COMPLETE SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
