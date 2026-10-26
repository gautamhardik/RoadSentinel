"""
RoadSentinel Phase 4D — Pre-Registered Adaptive-Area Calibration Experiment
Evaluates inference-time, predicted-area-dependent thresholds for isolated crop detections:
  - Candidate A: tau_small = 0.28, tau_large = 0.45
  - Candidate B: tau_small = 0.30, tau_large = 0.45
  - Candidate C: tau_small = 0.32, tau_large = 0.45
against:
  - Baseline: ActiveFoveation-960 (tau=0.25, greedy fusion)
  - Phase 4C: Calibrated-ActiveFoveation-960 (static tau_isolated=0.40)

All 5 configurations evaluated in a single forward pass across:
  1. Complete 10,000 validation images
  2. 300-image pilot cohort
  3. 9,700-image non-pilot confirmation subset
"""

import os
import sys
import subprocess

try:
    import ultralytics
except ImportError:
    print("Installing ultralytics package in environment...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
    import ultralytics

import time
import json
import math
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================================
# Embedded 300-Image Pilot Cohort Set (Phase 4B Manifest)
# ============================================================================

PILOT_COHORT_IMAGE_SET = set(['b1c66a42-6f7d68ca.jpg', 'b1c81faa-3df17267.jpg', 'b1ca2e5d-84cf9134.jpg', 'b1cd1e94-549d0bfe.jpg', 'b1d7b3ac-995f9d8a.jpg', 'b1d9e136-9ab25cb3.jpg', 'b1f20aa0-50213047.jpg', 'b1f4491b-33824f31.jpg', 'b1f62c41-ed0c6521.jpg', 'b20ff95c-b9444127.jpg', 'b216243d-ad4306b9.jpg', 'b23d2079-0c019d8e.jpg', 'b24071b8-b3ee1196.jpg', 'b266429a-ac0c4ed3.jpg', 'b27127df-eac9b95e.jpg', 'b27d5dcb-5bdec286.jpg', 'b284c0e8-85fbc42a.jpg', 'b2860d72-0b43e851.jpg', 'b294386c-24ddd0c3.jpg', 'b2daf29d-6198754f.jpg', 'b2e54795-601d2d78.jpg', 'b2e54795-db1f3bad.jpg', 'b3060c96-d2391b61.jpg', 'b3159650-0d7fbee8.jpg', 'b321c6b6-ab603d42.jpg', 'b3432539-8f9fc466.jpg', 'b35dac08-a80be8c5.jpg', 'b36a5c73-64eacfbf.jpg', 'b37b9cf2-9d52b009.jpg', 'b37c5be2-b9164b91.jpg', 'b3853f81-272f6c37.jpg', 'b396946b-d29f75f9.jpg', 'b39ffdda-b9a4ad33.jpg', 'b3b44b49-7336e3ee.jpg', 'b3c0357a-10fef986.jpg', 'b3e0345b-ae566cbd.jpg', 'b3e08585-3f5e6565.jpg', 'b3e32b36-0ca66c00.jpg', 'b4065fc4-ede06556.jpg', 'b41d35f8-6cf85033.jpg', 'b4301c7b-735da0d0.jpg', 'b43bcc83-c97d569e.jpg', 'b44a71bf-712b4538.jpg', 'b44b6eff-dda9780d.jpg', 'b44ee0ec-a0786e4c.jpg', 'b4536117-bc85ba28.jpg', 'b4542860-0b880bb4.jpg', 'b4654b11-c935938c.jpg', 'b487a2fa-53094db8.jpg', 'b4a9b6ca-b621ed11.jpg', 'b4aabd48-38c15aba.jpg', 'b4aabd48-89db54e4.jpg', 'b4c86653-3594c3a4.jpg', 'b4ca8154-fd2ddd24.jpg', 'b4eb79bd-288f1507.jpg', 'b4ef573f-199846ab.jpg', 'b4ef573f-5122732a.jpg', 'b5042634-edf42094.jpg', 'b53a1501-f21dfab8.jpg', 'b555790d-cfb88988.jpg', 'b555790d-d715196b.jpg', 'b563b270-0fe8e937.jpg', 'b581255d-1c6658b4.jpg', 'b58aff80-bea21173.jpg', 'b59413ff-9094612f.jpg', 'b5a09604-69dd5d9f.jpg', 'b5a50c2d-a18ba669.jpg', 'b5b17442-48053f00.jpg', 'b5b17442-ad1dc2ea.jpg', 'b5cb11ce-46ac0ae7.jpg', 'b5ecfd7e-fcdaceb0.jpg', 'b5ed8eaa-3fb2802d.jpg', 'b5fc0f1e-882a7cce.jpg', 'b60db8b3-7cb88369.jpg', 'b61aec41-121a3d3a.jpg', 'b620fed9-6d88be73.jpg', 'b6241b17-b82f1b25.jpg', 'b643ed1d-e4a05157.jpg', 'b67206f2-c8c9f269.jpg', 'b68e35b5-865cf5ee.jpg', 'b69b7d13-a9c704af.jpg', 'b6d855a5-c1d5a996.jpg', 'b6d9b0ba-a0a7525e.jpg', 'b6df605f-51c158b8.jpg', 'b6f4b139-d7311df5.jpg', 'b6ffe7bd-7c91eac9.jpg', 'b740e2ac-c01ee78f.jpg', 'b7480124-540c4b41.jpg', 'b74bedfc-6187ffd8.jpg', 'b769f2fa-026548da.jpg', 'b775af1c-b71a7f85.jpg', 'b77df2a1-b2c5d1bb.jpg', 'b77f3729-fcbf9b93.jpg', 'b7831880-0327e785.jpg', 'b78cb53c-65159143.jpg', 'b7a8e795-ff80fddf.jpg', 'b7abfbea-3067ba7b.jpg', 'b7bb0d6e-1922b34d.jpg', 'b7e1237b-f08b7c08.jpg', 'b7f136f3-1a77da25.jpg', 'b7fcda2b-2e4d36ef.jpg', 'b817df43-48a371e7.jpg', 'b825dfb7-442f3c75.jpg', 'b832508c-8f2be0ee.jpg', 'b83a044e-9b529b91.jpg', 'b842dbf5-4b5b09b7.jpg', 'b873b095-4530040c.jpg', 'b87ca2c5-c3c18e68.jpg', 'b8ab4367-297944fb.jpg', 'b8b06f77-5b618032.jpg', 'b8c1f257-22c2b2cc.jpg', 'b8cca913-a59e1a1b.jpg', 'b8f0315c-0fb7602c.jpg', 'b8f2d63d-6aac4195.jpg', 'b8f65e48-10600b5e.jpg', 'b92a7fe6-714c37c2.jpg', 'b94cd5b4-26c295e5.jpg', 'b9586e5c-33b83562.jpg', 'b967b2a7-2003bfd2.jpg', 'b97b9ee8-612c36ef.jpg', 'b99c5931-834bb88c.jpg', 'b9cabddd-e49dded9.jpg', 'b9d34460-2c3b1ab3.jpg', 'b9df54a4-66f58812.jpg', 'b9df54a4-8203d67b.jpg', 'b9fc97df-397f8a5d.jpg', 'ba1612fa-c9ac53d1.jpg', 'ba2610cc-76fb5c2d.jpg', 'ba3dc656-c88512cc.jpg', 'ba4bcdc5-31c36605.jpg', 'ba5c759c-623fcd9d.jpg', 'ba65c274-e0f6a546.jpg', 'ba85dccd-ad3b457d.jpg', 'baa50786-856eefa3.jpg', 'bab48a4e-43fa0dfe.jpg', 'bab90acb-5e9f3a31.jpg', 'bac25f5a-af4fd9fd.jpg', 'bae552dc-36e0fe4e.jpg', 'baeb920c-a950cb02.jpg', 'baef72b1-809d1abe.jpg', 'baf567e1-5ffa6b77.jpg', 'bb087b1c-6c9e904f.jpg', 'bb198c24-1f5691ed.jpg', 'bb1b7e42-67806924.jpg', 'bb2c3ff8-a1444b2b.jpg', 'bb2c5719-38a69465.jpg', 'bb46f3cd-b72d2152.jpg', 'bb541ef2-1c0906c0.jpg', 'bb71ba86-3204e97e.jpg', 'bb8dba97-c7ec77a3.jpg', 'bb9071ac-6b111ce9.jpg', 'bb97c366-2adf3cdc.jpg', 'bb99a578-c060f56f.jpg', 'bbc0f7ea-77bee0c4.jpg', 'bbc755d2-e35a6b06.jpg', 'bbe88e6d-c0e7b375.jpg', 'bbebc8cb-5ff7981f.jpg', 'bbebc8cb-c0c59383.jpg', 'bc24e2f5-286f9781.jpg', 'bc3edef9-98053500.jpg', 'bc5bcfab-23b275db.jpg', 'bc5bcfab-a639a82e.jpg', 'bc5fadac-790bcdec.jpg', 'bc7cbf85-85f4301c.jpg', 'bcae6e67-059fedea.jpg', 'bcaf73c1-e0c7165a.jpg', 'bcafa205-40fde1da.jpg', 'bce15cb0-2d6aec27.jpg', 'bcea7303-e7f7d067.jpg', 'bd14a97d-cb209ed9.jpg', 'bd1d310c-f5fb429f.jpg', 'bd338426-4b27de63.jpg', 'bd5fe5fb-dfcd1cd2.jpg', 'bd8aa347-4a6405a9.jpg', 'bd8e0b69-64d450fe.jpg', 'bd989210-c3df0371.jpg', 'bda1cdfa-5da26736.jpg', 'bdda4e2e-6d2bc480.jpg', 'bde7a73a-ee84f1a1.jpg', 'bdf33dc4-ae128055.jpg', 'bdfc6bf1-523e96ae.jpg', 'be042f25-579f9c94.jpg', 'be177534-4a736a9b.jpg', 'be1a0aee-7127f0d8.jpg', 'be26ac57-35dabb9c.jpg', 'be2d3f1c-2644c9e3.jpg', 'be37b92a-4873e7e0.jpg', 'be49ccee-b084e427.jpg', 'be52b29f-3c297726.jpg', 'be56ab92-96e20e29.jpg', 'be5c5aae-d31145ff.jpg', 'be6fd6ac-a0cc8633.jpg', 'be7783b0-b44ff154.jpg', 'be85292c-a3fbba40.jpg', 'be97ef40-3cdcb388.jpg', 'bea02fed-e64fced9.jpg', 'beb2d175-69dae2f2.jpg', 'bec7b5c0-f928ccf5.jpg', 'bf0038bc-5512f257.jpg', 'bf0b285b-afa81125.jpg', 'bf35cb3d-af58fa54.jpg', 'bf5b7ee3-18a07bcb.jpg', 'bf76fa8e-889c317d.jpg', 'bf8220b9-086e9e29.jpg', 'bf9e3b67-7c3ff8f9.jpg', 'bfa5fc6e-d9573dba.jpg', 'bfb9cf70-29fd574e.jpg', 'bfd64d5f-172c9bb5.jpg', 'bfe98b68-b1125889.jpg', 'bffe8c14-48544350.jpg', 'c007dc97-377cfd8a.jpg', 'c0082590-683d229f.jpg', 'c0625a26-cefa81e9.jpg', 'c0677163-613b4f10.jpg', 'c06aa107-6d6d0938.jpg', 'c08d18c6-6d211ae8.jpg', 'c0bbcd5e-e207560d.jpg', 'c0bbcd5e-f57df892.jpg', 'c0c8efe2-cb74e696.jpg', 'c1025fcf-69509758.jpg', 'c119a2ea-f59c8486.jpg', 'c121b93f-93807706.jpg', 'c126dfe1-4bdd6502.jpg', 'c146d100-cb76d05b.jpg', 'c155d65b-ae8038a6.jpg', 'c174b1bd-e90c56a2.jpg', 'c18c0928-a28a4f94.jpg', 'c1b93aa0-3f3c05a3.jpg', 'c1ba5ee6-d5c419d3.jpg', 'c1e404d7-4a3eedc5.jpg', 'c1f8d9b3-343f4798.jpg', 'c2074a7c-7905ced0.jpg', 'c21ef8aa-cabc5b7a.jpg', 'c24838fd-064f7c3f.jpg', 'c2775fbb-7b8764dc.jpg', 'c27e2a5d-93699ec3.jpg', 'c2853fd5-304c554e.jpg', 'c2a0e52f-013aa867.jpg', 'c2bd70f5-009e856e.jpg', 'c2d33423-8f1d3d05.jpg', 'c2e294a9-07a2a060.jpg', 'c3061560-cb940839.jpg', 'c328b7ab-8bb33358.jpg', 'c33251a6-fa2a8810.jpg', 'c35e0eea-7596f0bd.jpg', 'c37382d8-010bbe82.jpg', 'c3884826-d8789adb.jpg', 'c39dc2ff-4399acc0.jpg', 'c3b940ca-5e220748.jpg', 'c3c28990-04db5664.jpg', 'c3f50821-ff9823a2.jpg', 'c417a291-7ea03054.jpg', 'c41c4557-f9e2022e.jpg', 'c43daa45-612283ce.jpg', 'c44e7068-5dafd017.jpg', 'c45d3e0a-b26d5167.jpg', 'c482a775-8e9c09c8.jpg', 'c48f5464-ebc458bf.jpg', 'c49c8353-f56b7e30.jpg', 'c4a19ba6-872ab53e.jpg', 'c4eb5462-c917fefd.jpg', 'c5121aec-f19f3e8d.jpg', 'c546c109-92e1a97e.jpg', 'c55b474f-f39c51a8.jpg', 'c56915dc-1eb8abe1.jpg', 'c569c251-16b6f263.jpg', 'c59500f0-d3272eb4.jpg', 'c5b7b466-4adbd41c.jpg', 'c5b80ec1-bb9fd172.jpg', 'c5f8a1ae-0d55fc7a.jpg', 'c60d6c5c-689d8921.jpg', 'c6265010-8caa470f.jpg', 'c6716c49-04844b01.jpg', 'c6716c49-d48c3543.jpg', 'c682e776-d20a6f25.jpg', 'c69e5864-1e46bd03.jpg', 'c6dca574-6e641016.jpg', 'c6dfbd67-2c7061bf.jpg', 'c71fcdc8-62a7fa51.jpg', 'c74a122c-24c19251.jpg', 'c74a122c-e7005b78.jpg', 'c7955e20-b48c7a6a.jpg', 'c79e3f00-5879235c.jpg', 'c7c2a783-d98d5909.jpg', 'c8111dda-1a7dded7.jpg', 'c811793c-7a592fd4.jpg', 'c83c10de-85eef163.jpg', 'c877c1e3-b9b26376.jpg', 'c89ee093-65ac0019.jpg', 'c8bd76fd-04a1c915.jpg', 'c8d7cfb3-cd9f103f.jpg', 'c8f90ec2-8b0d49de.jpg', 'c926408d-8de795c6.jpg', 'c933ade8-f63b2db4.jpg', 'c96e0b6a-0acdb8b5.jpg', 'c977423c-43dddd3b.jpg', 'c993615f-ae30e2cb.jpg', 'c9aa50be-9e8cd399.jpg', 'c9df836e-5955e515.jpg', 'ca049e30-cb9982a6.jpg'])


# ============================================================================
# 1. Class Mapping & Data Structures
# ============================================================================

CHECKPOINT_CLASSES: Dict[int, str] = {
    0: "person", 1: "rider", 2: "car", 3: "truck", 4: "bus",
    5: "train", 6: "motor", 7: "bike", 8: "traffic light", 9: "traffic sign"
}
CHECKPOINT_TO_GT_ID: Dict[int, int] = {
    0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 9, 6: 6, 7: 5, 8: 7, 9: 8
}
GT_CLASSES: Dict[int, str] = {
    0: "person", 1: "rider", 2: "car", 3: "bus", 4: "truck",
    5: "bike", 6: "motor", 7: "traffic light", 8: "traffic sign", 9: "train"
}

ALL_CLASS_NAMES: List[str] = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]
VRU_CLASS_NAMES: List[str] = ["person", "rider", "bike", "motor"]
IOU_THRESHOLDS = [round(x, 2) for x in np.linspace(0.50, 0.95, 10)]


# ============================================================================
# 2. Geometric & Predicted Area Utilities
# ============================================================================

def compute_box_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    xi1 = max(boxA[0], boxB[0])
    yi1 = max(boxA[1], boxB[1])
    xi2 = min(boxA[2], boxB[2])
    yi2 = min(boxA[3], boxB[3])
    iw = max(0.0, xi2 - xi1)
    ih = max(0.0, yi2 - yi1)
    inter = iw * ih
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    if union <= 1e-8:
        return 0.0
    return inter / union


def compute_box_iou_vec(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    xi1 = np.maximum(box[0], boxes[:, 0])
    yi1 = np.maximum(box[1], boxes[:, 1])
    xi2 = np.minimum(box[2], boxes[:, 2])
    yi2 = np.minimum(box[3], boxes[:, 3])
    iw = np.maximum(0.0, xi2 - xi1)
    ih = np.maximum(0.0, yi2 - yi1)
    inter = iw * ih
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return inter / np.maximum(union, 1e-8)


def compute_predicted_area(box, img_w=1280.0, img_h=720.0) -> float:
    """Clamps box coordinates to [0, img_w] x [0, img_h] and computes predicted area."""
    x1 = max(0.0, min(float(img_w), float(box[0])))
    y1 = max(0.0, min(float(img_h), float(box[1])))
    x2 = max(0.0, min(float(img_w), float(box[2])))
    y2 = max(0.0, min(float(img_h), float(box[3])))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def expand_crop_box(box: Tuple[float, float, float, float], img_w: int, img_h: int, margin: float = 0.25, min_size: int = 160) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_w = max(bw * margin, (min_size - bw) / 2.0 if bw < min_size else 0.0)
    pad_h = max(bh * margin, (min_size - bh) / 2.0 if bh < min_size else 0.0)
    cx1 = max(0.0, x1 - pad_w)
    cy1 = max(0.0, y1 - pad_h)
    cx2 = min(float(img_w), x2 + pad_w)
    cy2 = min(float(img_h), y2 + pad_h)
    return (cx1, cy1, cx2, cy2)


def generate_uncertainty_crops(
    detections: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    max_crops: int = 2,
) -> List[Dict[str, Any]]:
    candidates = []
    for d in detections:
        cname = d["class_name"]
        conf = d["conf"]
        box = d["box"]
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        area = bw * bh
        is_vru = cname in VRU_CLASS_NAMES

        if is_vru and (0.10 <= conf < 0.40):
            urgency = (0.40 - conf) * 1.5
            crop_b = expand_crop_box(box, img_w, img_h, margin=0.30)
            candidates.append({"crop_box": crop_b, "reason": "low_conf_vru", "score": urgency})
        elif is_vru and (area < 1024.0):
            urgency = (1024.0 - area) / 1024.0
            crop_b = expand_crop_box(box, img_w, img_h, margin=0.35)
            candidates.append({"crop_box": crop_b, "reason": "small_vru", "score": urgency})
        elif is_vru and (box[0] < 40 or box[2] > img_w - 40 or box[1] < 30 or box[3] > img_h - 30):
            crop_b = expand_crop_box(box, img_w, img_h, margin=0.30)
            candidates.append({"crop_box": crop_b, "reason": "boundary_vru", "score": 0.65})

    if not candidates:
        return []

    candidates.sort(key=lambda x: x["score"], reverse=True)
    merged = []
    for c in candidates:
        matched = False
        for m in merged:
            if compute_box_iou(c["crop_box"], m["crop_box"]) > 0.30:
                ux1 = min(c["crop_box"][0], m["crop_box"][0])
                uy1 = min(c["crop_box"][1], m["crop_box"][1])
                ux2 = max(c["crop_box"][2], m["crop_box"][2])
                uy2 = max(c["crop_box"][3], m["crop_box"][3])
                m["crop_box"] = (ux1, uy1, ux2, uy2)
                m["score"] = max(c["score"], m["score"])
                m["reason"] = f"{m['reason']}+{c['reason']}"
                matched = True
                break
        if not matched:
            merged.append(c)

    return merged[:max_crops]


# ============================================================================
# 3. Post-Processing & Fusion Policies
# ============================================================================

def baseline_fuse_and_deduplicate(
    full_frame_dets: List[Dict[str, Any]],
    crop_dets: List[Dict[str, Any]],
    iou_thresh: float = 0.50,
) -> List[Dict[str, Any]]:
    # Canonical Phase 3.4/4C baseline fusion: all detections down to conf=0.01
    # participate in fusion; operational threshold (0.25) is applied post-fusion.
    fused: List[Dict[str, Any]] = []
    ff_by_class = defaultdict(list)
    cr_by_class = defaultdict(list)

    for d in full_frame_dets: ff_by_class[d["class_name"]].append(d)
    for d in crop_dets: cr_by_class[d["class_name"]].append(d)

    all_cnames = set(ff_by_class.keys()).union(cr_by_class.keys())

    for cname in all_cnames:
        ff_list = ff_by_class[cname]
        cr_list = cr_by_class[cname]
        matched_crop_indices = set()

        for ff_d in ff_list:
            best_idx = -1
            best_iou = 0.0
            for idx, cr_d in enumerate(cr_list):
                if idx in matched_crop_indices: continue
                iou = compute_box_iou(ff_d["box"], cr_d["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_thresh and best_idx >= 0:
                matched_crop_indices.add(best_idx)
                cr_d = cr_list[best_idx]
                w_ff = ff_d["conf"]
                w_cr = cr_d["conf"] * 1.2
                w_sum = max(w_ff + w_cr, 1e-6)
                b_ff = ff_d["box"]
                b_cr = cr_d["box"]
                fused_box = (
                    (b_ff[0] * w_ff + b_cr[0] * w_cr) / w_sum,
                    (b_ff[1] * w_ff + b_cr[1] * w_cr) / w_sum,
                    (b_ff[2] * w_ff + b_cr[2] * w_cr) / w_sum,
                    (b_ff[3] * w_ff + b_cr[3] * w_cr) / w_sum,
                )
                fused_conf = min(0.99, max(ff_d["conf"], cr_d["conf"]) + 0.05 * min(ff_d["conf"], cr_d["conf"]))
                fused.append({
                    "img_name": ff_d["img_name"],
                    "class_name": cname,
                    "conf": fused_conf,
                    "box": np.array(fused_box, dtype=np.float32),
                    "source": "fused",
                })
            else:
                fused.append(ff_d)

        for idx, cr_d in enumerate(cr_list):
            if idx not in matched_crop_indices:
                fused.append(cr_d)

    return fused


def adaptive_area_wbf_consensus_fusion(
    full_frame_dets: List[Dict[str, Any]],
    crop_dets: List[Dict[str, Any]],
    tau_ff: float = 0.25,
    tau_isolated_small: float = 0.30,
    tau_isolated_large: float = 0.45,
    tau_confirmed: float = 0.20,
    area_cutoff: float = 1024.0,
    iou_assoc: float = 0.30,
    img_w: float = 1280.0,
    img_h: float = 720.0,
) -> List[Dict[str, Any]]:
    # Step 1: Filter full-frame detections
    ff_filt = [d for d in full_frame_dets if d["conf"] >= tau_ff]

    # Step 2: Check crop detections against coarse full-frame support (conf >= 0.10)
    ff_coarse = [d for d in full_frame_dets if d["conf"] >= 0.10]
    cr_filt = []

    for cr_d in crop_dets:
        box = cr_d["box"]
        area = compute_predicted_area(box, img_w, img_h)
        if area <= 0.0:
            continue

        has_ff_confirmation = False
        for ff_d in ff_coarse:
            if ff_d["class_name"] == cr_d["class_name"]:
                if compute_box_iou(cr_d["box"], ff_d["box"]) >= iou_assoc:
                    has_ff_confirmation = True
                    break

        if has_ff_confirmation:
            req_thresh = tau_confirmed
        else:
            req_thresh = tau_isolated_small if area < area_cutoff else tau_isolated_large

        if cr_d["conf"] >= req_thresh:
            cr_filt.append(cr_d)

    # Step 3: Weighted Box Fusion with Multi-Scale Consensus Boosting
    fused: List[Dict[str, Any]] = []
    all_dets = ff_filt + cr_filt
    by_cls = defaultdict(list)
    for d in all_dets: by_cls[d["class_name"]].append(d)

    for cname, dlist in by_cls.items():
        dlist.sort(key=lambda x: x["conf"], reverse=True)
        clusters = []

        for d in dlist:
            matched_cluster = None
            for clust in clusters:
                rep_box = clust[0]["box"]
                if compute_box_iou(d["box"], rep_box) >= 0.50:
                    matched_cluster = clust
                    break
            if matched_cluster is not None:
                matched_cluster.append(d)
            else:
                clusters.append([d])

        for clust in clusters:
            if len(clust) == 1:
                fused.append(clust[0])
            else:
                weights = [x["conf"] * (1.2 if x["source"] == "crop" else 1.0) for x in clust]
                w_sum = max(sum(weights), 1e-6)
                f_box = (
                    sum(x["box"][0] * weights[idx] for idx, x in enumerate(clust)) / w_sum,
                    sum(x["box"][1] * weights[idx] for idx, x in enumerate(clust)) / w_sum,
                    sum(x["box"][2] * weights[idx] for idx, x in enumerate(clust)) / w_sum,
                    sum(x["box"][3] * weights[idx] for idx, x in enumerate(clust)) / w_sum,
                )
                has_ff = any(x["source"] == "full_frame" for x in clust)
                has_cr = any(x["source"] == "crop" for x in clust)
                max_c = max(x["conf"] for x in clust)
                f_conf = min(0.99, max_c + (0.08 if has_ff and has_cr else 0.02))
                fused.append({
                    "img_name": clust[0]["img_name"],
                    "class_name": cname,
                    "conf": f_conf,
                    "box": np.array(f_box, dtype=np.float32),
                    "source": "wbf_consensus",
                })

    return fused


# ============================================================================
# 4. Evaluation Engine Across Subsets
# ============================================================================

def compute_coco_ap_for_class(
    preds: List[Dict[str, Any]],
    gt_by_img: Dict[str, np.ndarray],
    num_gt: int,
) -> Tuple[float, float]:
    if num_gt == 0 or not preds:
        return 0.0, 0.0
    preds_sorted = sorted(preds, key=lambda x: x["conf"], reverse=True)
    n_preds = len(preds_sorted)
    ap_per_iou = []

    for iou_thresh in IOU_THRESHOLDS:
        matched_gt = defaultdict(set)
        tp = np.zeros(n_preds, dtype=np.float32)
        fp = np.zeros(n_preds, dtype=np.float32)

        for idx, p in enumerate(preds_sorted):
            img_name = p["img_name"]
            gt_boxes = gt_by_img.get(img_name)
            if gt_boxes is None or len(gt_boxes) == 0:
                fp[idx] = 1.0
                continue
            ious = compute_box_iou_vec(p["box"], gt_boxes)
            best_idx = int(np.argmax(ious))
            if float(ious[best_idx]) >= iou_thresh and best_idx not in matched_gt[img_name]:
                tp[idx] = 1.0
                matched_gt[img_name].add(best_idx)
            else:
                fp[idx] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / float(num_gt)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

        rec_levels = np.linspace(0.0, 1.0, 101)
        prec_levels = np.zeros(101, dtype=np.float32)
        for r_idx, r in enumerate(rec_levels):
            cands = precisions[recalls >= r]
            prec_levels[r_idx] = np.max(cands) if len(cands) > 0 else 0.0
        ap_per_iou.append(float(np.mean(prec_levels)))

    return ap_per_iou[0], float(np.mean(ap_per_iou))


def evaluate_operational_point(
    preds: List[Dict[str, Any]],
    gt_by_img: Dict[str, np.ndarray],
    num_gt: int,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
) -> Tuple[int, int, int, float, float, float, Dict[str, set]]:
    filt_preds = [p for p in preds if p["conf"] >= conf_thresh]
    filt_preds.sort(key=lambda x: x["conf"], reverse=True)
    matched_gt = defaultdict(set)
    tp, fp = 0, 0

    for p in filt_preds:
        img_name = p["img_name"]
        gt_boxes = gt_by_img.get(img_name)
        if gt_boxes is None or len(gt_boxes) == 0:
            fp += 1
            continue
        ious = compute_box_iou_vec(p["box"], gt_boxes)
        best_idx = int(np.argmax(ious))
        if float(ious[best_idx]) >= iou_thresh and best_idx not in matched_gt[img_name]:
            tp += 1
            matched_gt[img_name].add(best_idx)
        else:
            fp += 1

    fn = num_gt - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(num_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return tp, fp, fn, prec, rec, f1, matched_gt


def evaluate_subset_metrics(
    preds_by_class: Dict[str, List[Dict[str, Any]]],
    gt_by_class_img: Dict[str, Dict[str, np.ndarray]],
    gt_objects: List[Dict[str, Any]],
    target_images: Set[str],
) -> Dict[str, Any]:
    filt_preds = {
        c: [p for p in preds_by_class[c] if p["img_name"] in target_images]
        for c in ALL_CLASS_NAMES
    }
    filt_gt_by_img = {
        c: {img: boxes for img, boxes in gt_by_class_img[c].items() if img in target_images}
        for c in ALL_CLASS_NAMES
    }
    filt_gt_counts = {
        c: sum(len(boxes) for boxes in filt_gt_by_img[c].values())
        for c in ALL_CLASS_NAMES
    }

    per_class_results = []
    matched_gts = {}

    for cls_name in ALL_CLASS_NAMES:
        c_preds = filt_preds[cls_name]
        c_gt = filt_gt_by_img[cls_name]
        n_gt = filt_gt_counts[cls_name]

        ap_50, ap_50_95 = compute_coco_ap_for_class(c_preds, c_gt, n_gt)
        tp, fp, fn, prec, rec, f1, m_gt = evaluate_operational_point(c_preds, c_gt, n_gt, 0.25, 0.50)
        matched_gts[cls_name] = m_gt

        per_class_results.append({
            "class_name": cls_name,
            "is_vru": (cls_name in VRU_CLASS_NAMES),
            "gt_count": n_gt,
            "tp_50": tp, "fp_50": fp, "fn_50": fn,
            "precision_50": round(prec, 4),
            "recall_50": round(rec, 4),
            "f1_50": round(f1, 4),
            "ap_50": round(ap_50, 4),
            "ap_50_95": round(ap_50_95, 4),
        })

    active_classes = [r for r in per_class_results if r["gt_count"] > 0]
    overall_mAP_50 = float(np.mean([r["ap_50"] for r in active_classes])) if active_classes else 0.0
    overall_mAP_50_95 = float(np.mean([r["ap_50_95"] for r in active_classes])) if active_classes else 0.0

    vru_classes = [r for r in per_class_results if r["is_vru"]]
    vru_macro_mAP_50 = float(np.mean([r["ap_50"] for r in vru_classes])) if vru_classes else 0.0
    vru_macro_mAP_50_95 = float(np.mean([r["ap_50_95"] for r in vru_classes])) if vru_classes else 0.0

    vru_tp = sum(r["tp_50"] for r in vru_classes)
    vru_fp = sum(r["fp_50"] for r in vru_classes)
    vru_fn = sum(r["fn_50"] for r in vru_classes)
    vru_prec = vru_tp / max(vru_tp + vru_fp, 1)
    vru_rec = vru_tp / max(vru_tp + vru_fn, 1)
    vru_f1 = 2 * vru_prec * vru_rec / max(vru_prec + vru_rec, 1e-8)

    size_tier_counts = {
        "Large (≥96²)": {"total_gt": 0, "matched_gt": 0, "vru_gt": 0, "vru_matched": 0},
        "Medium [32²-96²)": {"total_gt": 0, "matched_gt": 0, "vru_gt": 0, "vru_matched": 0},
        "Small (<32²)": {"total_gt": 0, "matched_gt": 0, "vru_gt": 0, "vru_matched": 0},
    }

    for gt_obj in gt_objects:
        if gt_obj["img_name"] not in target_images:
            continue
        grp = gt_obj["size_group"]
        c_name = gt_obj["class_name"]
        i_name = gt_obj["img_name"]
        g_idx = gt_obj["gt_idx"]
        is_vru = gt_obj["is_vru"]

        size_tier_counts[grp]["total_gt"] += 1
        if is_vru:
            size_tier_counts[grp]["vru_gt"] += 1

        if g_idx in matched_gts[c_name].get(i_name, set()):
            size_tier_counts[grp]["matched_gt"] += 1
            if is_vru:
                size_tier_counts[grp]["vru_matched"] += 1

    size_group_results = {}
    for grp, cnts in size_tier_counts.items():
        size_group_results[grp] = {
            "total_gt": cnts["total_gt"],
            "matched_gt": cnts["matched_gt"],
            "overall_recall": round(cnts["matched_gt"] / max(cnts["total_gt"], 1), 4),
            "vru_gt": cnts["vru_gt"],
            "vru_matched": cnts["vru_matched"],
            "vru_recall": round(cnts["vru_matched"] / max(cnts["vru_gt"], 1), 4),
        }

    return {
        "image_count": len(target_images),
        "overall_mAP_50": round(overall_mAP_50, 4),
        "overall_mAP_50_95": round(overall_mAP_50_95, 4),
        "vru_macro_mAP_50": round(vru_macro_mAP_50, 4),
        "vru_macro_mAP_50_95": round(vru_macro_mAP_50_95, 4),
        "vru_micro_precision_50": round(vru_prec, 4),
        "vru_micro_recall_50": round(vru_rec, 4),
        "vru_micro_f1_50": round(vru_f1, 4),
        "vru_tp": vru_tp,
        "vru_fp": vru_fp,
        "vru_fn": vru_fn,
        "size_group_results": size_group_results,
        "per_class_results": per_class_results,
    }


# ============================================================================
# 5. Main Benchmark Runner
# ============================================================================

def main():
    print("=" * 80)
    print("RoadSentinel Phase 4D — Pre-Registered Adaptive-Area Calibration Experiment")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu_name})")

    import argparse
    parser = argparse.ArgumentParser(description="RoadSentinel Phase 4D Benchmark")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working" if Path("/kaggle/working").exists() else "experiments/phase4d_validation")
    parser.add_argument("--weights", type=str, default=None, help="Path to checkpoint weights file")
    parser.add_argument("--limit", type=int, default=10000, help="Maximum number of images to evaluate")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class TeeLogger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = TeeLogger(out_dir / "console.log")

    if args.weights:
        ckpt_path = Path(args.weights)
    else:
        input_root = Path("/kaggle/input")
        best_pt_candidates = list(input_root.rglob("best.pt")) if input_root.exists() else []
        if best_pt_candidates:
            ckpt_path = best_pt_candidates[0]
            print(f"Auto-discovered trained checkpoint in input sources: {ckpt_path}")
        else:
            ckpt_candidates = [
                Path("/kaggle/input/datasets/hardikgautam25/roadsentinel-weights/bdd100k_yolo11s.pt"),
                Path("/kaggle/input/roadsentinel-weights/bdd100k_yolo11s.pt"),
                Path("weights/bdd100k_yolo11s.pt"),
            ]
            ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
    assert ckpt_path is not None and ckpt_path.exists(), f"Checkpoint {ckpt_path} not found!"

    val_img_candidates = [
        Path("/kaggle/input/datasets/a7madmostafa/bdd100k-yolo/val/images"),
        Path("/kaggle/input/bdd100k-yolo/val/images"),
        Path("bdd100k_dataset/val/images"),
    ]
    val_img_dir = next((p for p in val_img_candidates if p.exists()), None)
    assert val_img_dir is not None, "Validation images directory not found!"

    val_lbl_dir = val_img_dir.parent / "labels"
    assert val_lbl_dir.exists(), f"Validation labels not found at {val_lbl_dir}!"

    # Verify Checkpoint Hash
    with open(ckpt_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest().upper()
    EXPECTED_SHA = "A4747292648F37F68AB997BEA65A36391C41356F35E309A9F3D2EB1D3EA8D8E3"
    print(f"Checkpoint Path:   {ckpt_path}")
    print(f"Checkpoint SHA256: {ckpt_sha256}")
    if ckpt_path.name == "bdd100k_yolo11s.pt":
        assert ckpt_sha256 == EXPECTED_SHA, f"Checkpoint hash mismatch! Got {ckpt_sha256}"
    else:
        print(f"Evaluating fine-tuned candidate checkpoint: {ckpt_sha256}")

    pilot_images_set = set(PILOT_COHORT_IMAGE_SET)
    all_image_files = sorted([f.name for f in val_img_dir.glob("*.jpg")])
    total_images_count = len(all_image_files)
    print(f"Discovered {total_images_count} total validation images.")

    # Ground Truth Loading
    print("Loading ground-truth annotations...")
    gt_boxes_raw = {c: defaultdict(list) for c in ALL_CLASS_NAMES}
    gt_objects = []
    img_w, img_h = 1280.0, 720.0

    for img_name in all_image_files:
        lbl_p = val_lbl_dir / (Path(img_name).stem + ".txt")
        if not lbl_p.exists(): continue
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                cid = int(parts[0])
                cname = GT_CLASSES[cid]
                xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                x1 = (xc - bw / 2.0) * img_w
                y1 = (yc - bh / 2.0) * img_h
                x2 = (xc + bw / 2.0) * img_w
                y2 = (yc + bh / 2.0) * img_h
                box = [x1, y1, x2, y2]
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if area < 1024.0: sz_grp = "Small (<32²)"
                elif area < 9216.0: sz_grp = "Medium [32²-96²)"
                else: sz_grp = "Large (≥96²)"

                gt_idx = len(gt_boxes_raw[cname][img_name])
                gt_boxes_raw[cname][img_name].append(box)
                gt_objects.append({
                    "img_name": img_name,
                    "class_name": cname,
                    "gt_idx": gt_idx,
                    "box": box,
                    "area": area,
                    "size_group": sz_grp,
                    "is_vru": (cname in VRU_CLASS_NAMES),
                })

    gt_by_class_img = {c: {} for c in ALL_CLASS_NAMES}
    for cname in ALL_CLASS_NAMES:
        for img_name, boxes in gt_boxes_raw[cname].items():
            gt_by_class_img[cname][img_name] = np.array(boxes, dtype=np.float32)

    EVAL_LIMIT = min(args.limit, total_images_count)
    eval_images = all_image_files[:EVAL_LIMIT]
    eval_images_set = set(eval_images)

    pilot_eval_set = eval_images_set.intersection(pilot_images_set)
    non_pilot_eval_set = eval_images_set - pilot_eval_set

    print(f"\nEvaluation Partition Summary:")
    print(f"  Total Images in Run:       {len(eval_images_set)}")
    print(f"  Pilot Cohort Images:       {len(pilot_eval_set)}")
    print(f"  Non-Pilot Confirmation:    {len(non_pilot_eval_set)} (Previously Inspected; Exploratory)")

    # Candidate Definitions
    CONFIGS = {
        "ActiveFoveation-960": {"type": "reference_baseline"},
        "Calibrated-ActiveFoveation-960": {"type": "reference_phase4c", "tau_iso": 0.40},
        "Candidate-A-AdaptiveArea": {"type": "adaptive_area", "tau_small": 0.28, "tau_large": 0.45},
        "Candidate-B-AdaptiveArea": {"type": "adaptive_area", "tau_small": 0.30, "tau_large": 0.45},
        "Candidate-C-AdaptiveArea": {"type": "adaptive_area", "tau_small": 0.32, "tau_large": 0.45},
    }

    # Measure Batch-1 Latency on 50 images
    print("\nMeasuring isolated Batch-1 latency across candidate configurations...")
    batch1_subset = all_image_files[:50]
    model = YOLO(str(ckpt_path))
    model.to(device)

    w_img = cv2.imread(str(val_img_dir / batch1_subset[0]))
    for _ in range(10):
        _ = model.predict(source=w_img, imgsz=960, conf=0.25, device=device, verbose=False)
    if torch.cuda.is_available() and "cuda" in device: torch.cuda.synchronize()

    preloaded_imgs = [cv2.imread(str(val_img_dir / fname)) for fname in batch1_subset]
    lat_in = []
    lat_pipe = []

    for img in preloaded_imgs:
        h, w = img.shape[:2]
        if torch.cuda.is_available() and "cuda" in device: torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = model.predict(source=img, imgsz=960, conf=0.25, device=device, verbose=False)
        raw_dets = []
        for b in res[0].boxes:
            coords = b.xyxy[0].cpu().numpy().tolist()
            cname = model.names[int(b.cls.item())]
            raw_dets.append({"class_name": cname, "conf": float(b.conf.item()), "box": coords})
        props = generate_uncertainty_crops(raw_dets, w, h, max_crops=2)
        for p in props:
            cx1, cy1, cx2, cy2 = [int(v) for v in p["crop_box"]]
            sub = img[cy1:cy2, cx1:cx2]
            _ = model.predict(source=sub, imgsz=640, conf=0.25, device=device, verbose=False)

        if torch.cuda.is_available() and "cuda" in device: torch.cuda.synchronize()
        t_in = (time.perf_counter() - t0) * 1000.0
        lat_in.append(t_in)
        lat_pipe.append(t_in + 4.10)

    b1_in_p50 = float(np.median(lat_in))
    b1_pipe_p50 = float(np.median(lat_pipe))
    b1_pipe_p95 = float(np.percentile(lat_pipe, 95))
    print(f"  Foveation Pipeline Batch-1: In-Memory P50 = {b1_in_p50:.2f} ms | Pipeline P50 = {b1_pipe_p50:.2f} ms | P95 = {b1_pipe_p95:.2f} ms")

    # Joint Evaluation across all 10,000 images in a single forward pass
    print("\nExecuting joint forward pass across all 10,000 validation images...")
    predictions_by_cfg = {cfg_name: defaultdict(list) for cfg_name in CONFIGS}
    total_crops_executed = 0
    images_with_crops = 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t_start = time.perf_counter()

    for idx, fname in enumerate(eval_images):
        fpath = str(val_img_dir / fname)
        img = cv2.imread(fpath)
        h, w = img.shape[:2]

        # 1. Full-frame pass (imgsz=960, conf=0.01)
        res = model.predict(source=img, imgsz=960, conf=0.01, device=device, verbose=False)
        ff_dets = []
        for b in res[0].boxes:
            coords = b.xyxy[0].cpu().numpy()
            cname = model.names[int(b.cls.item())]
            ff_dets.append({
                "img_name": fname,
                "class_name": cname,
                "conf": float(b.conf.item()),
                "box": coords,
                "source": "full_frame",
            })

        # 2. Uncertainty crop pass (budget K=2, imgsz=640, conf=0.01)
        proposals = generate_uncertainty_crops(ff_dets, w, h, max_crops=2)
        crop_dets = []
        if proposals:
            images_with_crops += 1
            total_crops_executed += len(proposals)
            for p in proposals:
                cx1, cy1, cx2, cy2 = [int(round(v)) for v in p["crop_box"]]
                cx1, cy1 = max(0, cx1), max(0, cy1)
                cx2, cy2 = min(w, cx2), min(h, cy2)
                if (cx2 - cx1) < 10 or (cy2 - cy1) < 10: continue
                sub = img[cy1:cy2, cx1:cx2]
                sub_res = model.predict(source=sub, imgsz=640, conf=0.01, device=device, verbose=False)
                for sb in sub_res[0].boxes:
                    scoords = sb.xyxy[0].cpu().numpy()
                    ox1, oy1 = cx1 + scoords[0], cy1 + scoords[1]
                    ox2, oy2 = cx1 + scoords[2], cy1 + scoords[3]
                    scname = model.names[int(sb.cls.item())]
                    crop_dets.append({
                        "img_name": fname,
                        "class_name": scname,
                        "conf": float(sb.conf.item()),
                        "box": np.array([ox1, oy1, ox2, oy2], dtype=np.float32),
                        "source": "crop",
                    })

        # 3. Post-processing branch for each configuration
        # A. Baseline Reference (tau=0.25, greedy)
        f_base = baseline_fuse_and_deduplicate(ff_dets, crop_dets, iou_thresh=0.50)
        for d in f_base: predictions_by_cfg["ActiveFoveation-960"][d["class_name"]].append(d)

        # B. Phase 4C Reference (static tau_isolated=0.40)
        f_4c = adaptive_area_wbf_consensus_fusion(
            ff_dets, crop_dets, tau_ff=0.25, tau_isolated_small=0.40, tau_isolated_large=0.40, tau_confirmed=0.20
        )
        for d in f_4c: predictions_by_cfg["Calibrated-ActiveFoveation-960"][d["class_name"]].append(d)

        # C. Candidate A (tau_small=0.28, tau_large=0.45)
        f_candA = adaptive_area_wbf_consensus_fusion(
            ff_dets, crop_dets, tau_ff=0.25, tau_isolated_small=0.28, tau_isolated_large=0.45, tau_confirmed=0.20
        )
        for d in f_candA: predictions_by_cfg["Candidate-A-AdaptiveArea"][d["class_name"]].append(d)

        # D. Candidate B (tau_small=0.30, tau_large=0.45)
        f_candB = adaptive_area_wbf_consensus_fusion(
            ff_dets, crop_dets, tau_ff=0.25, tau_isolated_small=0.30, tau_isolated_large=0.45, tau_confirmed=0.20
        )
        for d in f_candB: predictions_by_cfg["Candidate-B-AdaptiveArea"][d["class_name"]].append(d)

        # E. Candidate C (tau_small=0.32, tau_large=0.45)
        f_candC = adaptive_area_wbf_consensus_fusion(
            ff_dets, crop_dets, tau_ff=0.25, tau_isolated_small=0.32, tau_isolated_large=0.45, tau_confirmed=0.20
        )
        for d in f_candC: predictions_by_cfg["Candidate-C-AdaptiveArea"][d["class_name"]].append(d)

        if (idx + 1) % 2500 == 0 or (idx + 1) == EVAL_LIMIT:
            print(f"  Processed {idx + 1}/{EVAL_LIMIT} images...")

    if torch.cuda.is_available(): torch.cuda.synchronize()
    total_time_s = time.perf_counter() - t_start
    fps = EVAL_LIMIT / max(total_time_s, 1e-5)
    amortized_ms = (total_time_s * 1000.0) / EVAL_LIMIT
    peak_vram = (torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)) if torch.cuda.is_available() else 0.0

    print(f"\nInference complete in {total_time_s:.2f} s ({fps:.2f} FPS). Computing partition metrics...")

    # Evaluate Metrics Across All Configurations & Subsets
    config_results = {}
    for cfg_name in CONFIGS:
        preds_dict = predictions_by_cfg[cfg_name]
        full_m = evaluate_subset_metrics(preds_dict, gt_by_class_img, gt_objects, eval_images_set)
        pilot_m = evaluate_subset_metrics(preds_dict, gt_by_class_img, gt_objects, pilot_eval_set)
        non_pilot_m = evaluate_subset_metrics(preds_dict, gt_by_class_img, gt_objects, non_pilot_eval_set)

        config_results[cfg_name] = {
            "name": cfg_name,
            "config_spec": CONFIGS[cfg_name],
            "full_validation_metrics": full_m,
            "pilot_cohort_metrics": pilot_m,
            "non_pilot_confirmation_metrics": non_pilot_m,
            "timing_and_compute": {
                "throughput_fps": round(fps, 2),
                "amortized_ms": round(amortized_ms, 2),
                "batch1_in_memory_p50_ms": round(b1_in_p50, 2),
                "batch1_pipeline_p50_ms": round(b1_pipe_p50, 2),
                "batch1_pipeline_p95_ms": round(b1_pipe_p95, 2),
                "peak_vram_mb": round(peak_vram, 1),
                "total_crops": total_crops_executed,
                "crop_triggered_images": images_with_crops,
                "crop_trigger_rate": round(images_with_crops / max(EVAL_LIMIT, 1), 4),
            }
        }
        print(f"  {cfg_name:<30}: Non-Pilot Prec={non_pilot_m['vru_micro_precision_50']:.4f} | Rec={non_pilot_m['vru_micro_recall_50']:.4f} | F1={non_pilot_m['vru_micro_f1_50']:.4f} | SmallRec={non_pilot_m['size_group_results']['Small (<32²)']['vru_recall']:.4f}")

    # Write Output Artifacts
    print("\nWriting auditable output artifacts...")

    def to_serializable(val):
        if val is None or isinstance(val, (str, bool)): return val
        if isinstance(val, (int, np.integer)): return int(val)
        if isinstance(val, (float, np.floating)):
            if math.isnan(val) or math.isinf(val): return None
            return float(val)
        if isinstance(val, np.ndarray): return [to_serializable(x) for x in val.tolist()]
        if isinstance(val, (list, tuple, set)): return [to_serializable(x) for x in val]
        if isinstance(val, dict): return {str(k): to_serializable(v) for k, v in val.items()}
        if hasattr(val, "item"): return to_serializable(val.item())
        return str(val)

    final_output = {
        "benchmark_name": "RoadSentinel Phase 4D — Pre-Registered Adaptive-Area Calibration Experiment",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": {"file": ckpt_path.name, "sha256": ckpt_sha256},
        "hardware": {"device": device, "gpu_name": gpu_name},
        "dataset": {
            "total_images": EVAL_LIMIT,
            "pilot_images": len(pilot_eval_set),
            "non_pilot_images": len(non_pilot_eval_set),
            "total_gt": len(gt_objects),
        },
        "configurations": config_results,
    }

    serialized = to_serializable(final_output)
    with open(out_dir / "phase4d_metrics.json", "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)
    with open(out_dir / "raw_metrics.json", "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)

    # Candidate Comparison CSV (focusing on 9,700 non-pilot confirmation subset)
    cmp_header = "Configuration,Type,Partition,VRU_Precision,VRU_Recall,VRU_F1,SmallVRU_Recall,MedVRU_Recall,LargeVRU_Recall,VRU_TP,VRU_FP,VRU_FN,Delta_Prec_pp_vs_Baseline,Delta_Rec_pp_vs_Baseline,Gate1_Prec,Gate2_Rec,Gate3_SmallRec,Gate4_F1,Gate_Verdict\n"
    with open(out_dir / "candidate_comparison.csv", "w", encoding="utf-8") as f:
        f.write(cmp_header)
        base_np = config_results["ActiveFoveation-960"]["non_pilot_confirmation_metrics"]
        base_prec = base_np["vru_micro_precision_50"]
        base_rec = base_np["vru_micro_recall_50"]

        for cname, cr in config_results.items():
            np_m = cr["non_pilot_confirmation_metrics"]
            p = np_m["vru_micro_precision_50"]
            r = np_m["vru_micro_recall_50"]
            f1 = np_m["vru_micro_f1_50"]
            sm = np_m["size_group_results"]["Small (<32²)"]["vru_recall"]
            md = np_m["size_group_results"]["Medium [32²-96²)"]["vru_recall"]
            lg = np_m["size_group_results"]["Large (≥96²)"]["vru_recall"]

            d_p = (p - base_prec) * 100.0
            d_r = (r - base_rec) * 100.0

            g1 = "PASS" if d_p >= 5.0 else "FAIL"
            g2 = "PASS" if d_r >= -2.0 else "FAIL"
            g3 = "PASS" if sm >= 0.4500 else "FAIL"
            g4 = "PASS" if f1 >= 0.6500 else "FAIL"
            overall_g = "PASS" if (g1 == "PASS" and g2 == "PASS" and g3 == "PASS" and g4 == "PASS") else "FAIL"

            f.write(f"{cname},{cr['config_spec']['type']},NonPilot_9700,"
                    f"{p:.4f},{r:.4f},{f1:.4f},{sm:.4f},{md:.4f},{lg:.4f},"
                    f"{np_m['vru_tp']},{np_m['vru_fp']},{np_m['vru_fn']},"
                    f"{d_p:+.2f},{d_r:+.2f},{g1},{g2},{g3},{g4},{overall_g}\n")

    # Size Group Analysis CSV across all 3 partitions
    sg_header = "Configuration,Partition,ImageCount,SmallVRU_Recall,MedVRU_Recall,LargeVRU_Recall,SmallVRU_TP,MedVRU_TP,LargeVRU_TP\n"
    with open(out_dir / "size_group_analysis.csv", "w", encoding="utf-8") as f:
        f.write(sg_header)
        for cname, cr in config_results.items():
            for part_name, m_dict in [
                ("Full_10000_Val", cr["full_validation_metrics"]),
                ("Pilot_300_Cohort", cr["pilot_cohort_metrics"]),
                ("NonPilot_9700_Confirm", cr["non_pilot_confirmation_metrics"]),
            ]:
                sg = m_dict["size_group_results"]
                sm_rec = sg["Small (<32²)"]["vru_recall"]
                md_rec = sg["Medium [32²-96²)"]["vru_recall"]
                lg_rec = sg["Large (≥96²)"]["vru_recall"]
                sm_tp = sg["Small (<32²)"]["vru_matched"]
                md_tp = sg["Medium [32²-96²)"]["vru_matched"]
                lg_tp = sg["Large (≥96²)"]["vru_matched"]
                f.write(f"{cname},{part_name},{m_dict['image_count']},"
                        f"{sm_rec:.4f},{md_rec:.4f},{lg_rec:.4f},{sm_tp},{md_tp},{lg_tp}\n")

    # Per-Image Records JSONL
    print("Writing per-image evaluation records...")
    with open(out_dir / "per_image_records.jsonl", "w", encoding="utf-8") as f_recs:
        for fname in eval_images[:500]:  # Save 500 representative per-image candidate records
            rec = {
                "image": fname,
                "is_pilot": (fname in pilot_images_set),
                "candidates": {
                    cn: len([p for p in predictions_by_cfg[cn]["person"] if p["img_name"] == fname])
                    for cn in CONFIGS
                }
            }
            f_recs.write(json.dumps(rec) + "\n")

    print("\nBenchmark completed successfully! Outputs written to:", out_dir)


if __name__ == "__main__":
    main()
