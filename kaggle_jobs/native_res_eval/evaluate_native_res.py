"""RoadSentinel Phase 8 — Milestone 2: Native-Resolution (1280px) Inference Benchmark.

Evaluates the frozen Phase 6 epoch-26 YOLO11m checkpoint at native resolution (1280px)
and compares side-by-side against the 960-pixel baseline across:
  1. Small-VRU recall & precision
  2. Overall VRU recall & F1
  3. Per-class performance (person, rider, bike, motor)
  4. Inference latency & peak GPU memory
  5. Partition consistency (300 pilot vs. 9,700 expanded vs. 10,000 aggregate)

Guarantees:
  - Checkpoint frozen: SHA-256 42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e
  - Identical matching rules: single-pass streamed greedy matching at IoU 0.50
  - Identical confidence threshold: 0.25
  - Identical small-object definition: area < 1024 px in original coordinates
  - 20,000-image test split remains strictly quarantined
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
import numpy as np
import yaml
import torch
import ultralytics
from ultralytics import YOLO

# Frozen 960-pixel baseline metrics for automated delta computation
BASELINE_960 = {
    "aggregate_10000": {
        "vru_recall": 63.31,
        "vru_precision": 76.39,
        "vru_f1": 0.6924,
        "small_vru_recall": 44.47,
        "small_vru_tp": 3126,
        "small_vru_total": 7029,
        "all_class_map50": 61.66,
        "all_class_map50_95": 35.88,
        "vru_macro_map50": 59.83,
        "vru_macro_map50_95": 31.56,
        "person_ap50": 72.58,
        "person_recall": 64.93,
        "rider_ap50": 54.83,
        "rider_recall": 51.31,
        "bike_ap50": 57.35,
        "bike_recall": 55.21,
        "motor_ap50": 54.54,
        "motor_recall": 51.11,
        "fps": 20.47,
    },
    "pilot_300": {
        "vru_recall": 60.46,
        "vru_precision": 78.45,
        "vru_f1": 0.6829,
        "small_vru_recall": 42.73,
    },
    "expanded_9700": {
        "vru_recall": 63.68,
        "vru_precision": 76.14,
        "vru_f1": 0.6935,
        "small_vru_recall": 44.73,
    }
}

EXPECTED_SHA256 = "42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e"
IMGSZ = 1280
BATCH = 16
DEVICE = 0 if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]
VRU_NAMES = {"person", "rider", "bike", "motor"}
VRU_IDS = [0, 1, 5, 6]

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
INPUT = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path(".")

PILOT_COHORT_IMAGE_SET = set([
    'b1c66a42-6f7d68ca.jpg', 'b1c81faa-3df17267.jpg', 'b1ca2e5d-84cf9134.jpg', 'b1cd1e94-549d0bfe.jpg',
    'b1d7b3ac-995f9d8a.jpg', 'b1d9e136-9ab25cb3.jpg', 'b1f20aa0-50213047.jpg', 'b1f4491b-33824f31.jpg',
    'b1f62c41-ed0c6521.jpg', 'b20ff95c-b9444127.jpg', 'b216243d-ad4306b9.jpg', 'b23d2079-0c019d8e.jpg',
    'b24071b8-b3ee1196.jpg', 'b266429a-ac0c4ed3.jpg', 'b27127df-eac9b95e.jpg', 'b27d5dcb-5bdec286.jpg',
    'b284c0e8-85fbc42a.jpg', 'b2860d72-0b43e851.jpg', 'b294386c-24ddd0c3.jpg', 'b2daf29d-6198754f.jpg',
    'b2e54795-601d2d78.jpg', 'b2e54795-db1f3bad.jpg', 'b3060c96-d2391b61.jpg', 'b3159650-0d7fbee8.jpg',
    'b321c6b6-ab603d42.jpg', 'b3432539-8f9fc466.jpg', 'b35dac08-a80be8c5.jpg', 'b36a5c73-64eacfbf.jpg',
    'b37b9cf2-9d52b009.jpg', 'b37c5be2-b9164b91.jpg', 'b3853f81-272f6c37.jpg', 'b396946b-d29f75f9.jpg',
    'b39ffdda-b9a4ad33.jpg', 'b3b44b49-7336e3ee.jpg', 'b3c0357a-10fef986.jpg', 'b3e0345b-ae566cbd.jpg',
    'b3e08585-3f5e6565.jpg', 'b3e32b36-0ca66c00.jpg', 'b4065fc4-ede06556.jpg', 'b41d35f8-6cf85033.jpg',
    'b4301c7b-735da0d0.jpg', 'b43bcc83-c97d569e.jpg', 'b44a71bf-712b4538.jpg', 'b44b6eff-dda9780d.jpg',
    'b44ee0ec-a0786e4c.jpg', 'b4536117-bc85ba28.jpg', 'b4542860-0b880bb4.jpg', 'b4654b11-c935938c.jpg',
    'b487a2fa-53094db8.jpg', 'b4a9b6ca-b621ed11.jpg', 'b4aabd48-38c15aba.jpg', 'b4aabd48-89db54e4.jpg',
    'b4c86653-3594c3a4.jpg', 'b4ca8154-fd2ddd24.jpg', 'b4eb79bd-288f1507.jpg', 'b4ef573f-199846ab.jpg',
    'b4ef573f-5122732a.jpg', 'b5042634-edf42094.jpg', 'b53a1501-f21dfab8.jpg', 'b555790d-cfb88988.jpg',
    'b555790d-d715196b.jpg', 'b563b270-0fe8e937.jpg', 'b581255d-1c6658b4.jpg', 'b58aff80-bea21173.jpg',
    'b59413ff-9094612f.jpg', 'b5a09604-69dd5d9f.jpg', 'b5a50c2d-a18ba669.jpg', 'b5b17442-48053f00.jpg',
    'b5b17442-ad1dc2ea.jpg', 'b5cb11ce-46ac0ae7.jpg', 'b5ecfd7e-fcdaceb0.jpg', 'b5ed8eaa-3fb2802d.jpg',
    'b5fc0f1e-882a7cce.jpg', 'b60db8b3-7cb88369.jpg', 'b61aec41-121a3d3a.jpg', 'b620fed9-6d88be73.jpg',
    'b6241b17-b82f1b25.jpg', 'b643ed1d-e4a05157.jpg', 'b67206f2-c8c9f269.jpg', 'b68e35b5-865cf5ee.jpg',
    'b69b7d13-a9c704af.jpg', 'b6d855a5-c1d5a996.jpg', 'b6d9b0ba-a0a7525e.jpg', 'b6df605f-51c158b8.jpg',
    'b6f4b139-d7311df5.jpg', 'b6ffe7bd-7c91eac9.jpg', 'b740e2ac-c01ee78f.jpg', 'b7480124-540c4b41.jpg',
    'b74bedfc-6187ffd8.jpg', 'b769f2fa-026548da.jpg', 'b775af1c-b71a7f85.jpg', 'b77df2a1-b2c5d1bb.jpg',
    'b77f3729-fcbf9b93.jpg', 'b7831880-0327e785.jpg', 'b78cb53c-65159143.jpg', 'b7a8e795-ff80fddf.jpg',
    'b7abfbea-3067ba7b.jpg', 'b7bb0d6e-1922b34d.jpg', 'b7e1237b-f08b7c08.jpg', 'b7f136f3-1a77da25.jpg',
    'b7fcda2b-2e4d36ef.jpg', 'b817df43-48a371e7.jpg', 'b825dfb7-442f3c75.jpg', 'b832508c-8f2be0ee.jpg',
    'b83a044e-9b529b91.jpg', 'b842dbf5-4b5b09b7.jpg', 'b873b095-4530040c.jpg', 'b87ca2c5-c3c18e68.jpg',
    'b8ab4367-297944fb.jpg', 'b8b06f77-5b618032.jpg', 'b8c1f257-22c2b2cc.jpg', 'b8cca913-a59e1a1b.jpg',
    'b8f0315c-0fb7602c.jpg', 'b8f2d63d-6aac4195.jpg', 'b8f65e48-10600b5e.jpg', 'b92a7fe6-714c37c2.jpg',
    'b94cd5b4-26c295e5.jpg', 'b9586e5c-33b83562.jpg', 'b967b2a7-2003bfd2.jpg', 'b97b9ee8-612c36ef.jpg',
    'b99c5931-834bb88c.jpg', 'b9cabddd-e49dded9.jpg', 'b9d34460-2c3b1ab3.jpg', 'b9df54a4-66f58812.jpg',
    'b9df54a4-8203d67b.jpg', 'b9fc97df-397f8a5d.jpg', 'ba1612fa-c9ac53d1.jpg', 'ba2610cc-76fb5c2d.jpg',
    'ba3dc656-c88512cc.jpg', 'ba4bcdc5-31c36605.jpg', 'ba5c759c-623fcd9d.jpg', 'ba65c274-e0f6a546.jpg',
    'ba85dccd-ad3b457d.jpg', 'baa50786-856eefa3.jpg', 'bab48a4e-43fa0dfe.jpg', 'bab90acb-5e9f3a31.jpg',
    'bac25f5a-af4fd9fd.jpg', 'bae552dc-36e0fe4e.jpg', 'baeb920c-a950cb02.jpg', 'baef72b1-809d1abe.jpg',
    'baf567e1-5ffa6b77.jpg', 'bb087b1c-6c9e904f.jpg', 'bb198c24-1f5691ed.jpg', 'bb1b7e42-67806924.jpg',
    'bb2c3ff8-a1444b2b.jpg', 'bb2c5719-38a69465.jpg', 'bb46f3cd-b72d2152.jpg', 'bb541ef2-1c0906c0.jpg',
    'bb71ba86-3204e97e.jpg', 'bb8dba97-c7ec77a3.jpg', 'bb9071ac-6b111ce9.jpg', 'bb97c366-2adf3cdc.jpg',
    'bb99a578-c060f56f.jpg', 'bbc0f7ea-77bee0c4.jpg', 'bbc755d2-e35a6b06.jpg', 'bbe88e6d-c0e7b375.jpg',
    'bbebc8cb-5ff7981f.jpg', 'bbebc8cb-c0c59383.jpg', 'bc24e2f5-286f9781.jpg', 'bc3edef9-98053500.jpg',
    'bc5bcfab-23b275db.jpg', 'bc5bcfab-a639a82e.jpg', 'bc5fadac-790bcdec.jpg', 'bc7cbf85-85f4301c.jpg',
    'bcae6e67-059fedea.jpg', 'bcaf73c1-e0c7165a.jpg', 'bcafa205-40fde1da.jpg', 'bce15cb0-2d6aec27.jpg',
    'bcea7303-e7f7d067.jpg', 'bd14a97d-cb209ed9.jpg', 'bd1d310c-f5fb429f.jpg', 'bd338426-4b27de63.jpg',
    'bd5fe5fb-dfcd1cd2.jpg', 'bd8aa347-4a6405a9.jpg', 'bd8e0b69-64d450fe.jpg', 'bd989210-c3df0371.jpg',
    'bda1cdfa-5da26736.jpg', 'bdda4e2e-6d2bc480.jpg', 'bde7a73a-ee84f1a1.jpg', 'bdf33dc4-ae128055.jpg',
    'bdfc6bf1-523e96ae.jpg', 'be042f25-579f9c94.jpg', 'be177534-4a736a9b.jpg', 'be1a0aee-7127f0d8.jpg',
    'be26ac57-35dabb9c.jpg', 'be2d3f1c-2644c9e3.jpg', 'be37b92a-4873e7e0.jpg', 'be49ccee-b084e427.jpg',
    'be52b29f-3c297726.jpg', 'be56ab92-96e20e29.jpg', 'be5c5aae-d31145ff.jpg', 'be6fd6ac-a0cc8633.jpg',
    'be7783b0-b44ff154.jpg', 'be85292c-a3fbba40.jpg', 'be97ef40-3cdcb388.jpg', 'bea02fed-e64fced9.jpg',
    'beb2d175-69dae2f2.jpg', 'bec7b5c0-f928ccf5.jpg', 'bf0038bc-5512f257.jpg', 'bf0b285b-afa81125.jpg',
    'bf35cb3d-af58fa54.jpg', 'bf5b7ee3-18a07bcb.jpg', 'bf76fa8e-889c317d.jpg', 'bf8220b9-086e9e29.jpg',
    'bf9e3b67-7c3ff8f9.jpg', 'bfa5fc6e-d9573dba.jpg', 'bfb9cf70-29fd574e.jpg', 'bfd64d5f-172c9bb5.jpg',
    'bfe98b68-b1125889.jpg', 'bffe8c14-48544350.jpg', 'c007dc97-377cfd8a.jpg', 'c0082590-683d229f.jpg',
    'c0625a26-cefa81e9.jpg', 'c0677163-613b4f10.jpg', 'c06aa107-6d6d0938.jpg', 'c08d18c6-6d211ae8.jpg',
    'c0bbcd5e-e207560d.jpg', 'c0bbcd5e-f57df892.jpg', 'c0c8efe2-cb74e696.jpg', 'c1025fcf-69509758.jpg',
    'c119a2ea-f59c8486.jpg', 'c121b93f-93807706.jpg', 'c126dfe1-4bdd6502.jpg', 'c146d100-cb76d05b.jpg',
    'c155d65b-ae8038a6.jpg', 'c174b1bd-e90c56a2.jpg', 'c18c0928-a28a4f94.jpg', 'c1b93aa0-3f3c05a3.jpg',
    'c1ba5ee6-d5c419d3.jpg', 'c1e404d7-4a3eedc5.jpg', 'c1f8d9b3-343f4798.jpg', 'c2074a7c-7905ced0.jpg',
    'c21ef8aa-cabc5b7a.jpg', 'c24838fd-064f7c3f.jpg', 'c2775fbb-7b8764dc.jpg', 'c27e2a5d-93699ec3.jpg',
    'c2853fd5-304c554e.jpg', 'c2a0e52f-013aa867.jpg', 'c2bd70f5-009e856e.jpg', 'c2d33423-8f1d3d05.jpg',
    'c2e294a9-07a2a060.jpg', 'c3061560-cb940839.jpg', 'c328b7ab-8bb33358.jpg', 'c33251a6-fa2a8810.jpg',
    'c35e0eea-7596f0bd.jpg', 'c37382d8-010bbe82.jpg', 'c3884826-d8789adb.jpg', 'c39dc2ff-4399acc0.jpg',
    'c3b940ca-5e220748.jpg', 'c3c28990-04db5664.jpg', 'c3f50821-ff9823a2.jpg', 'c417a291-7ea03054.jpg',
    'c41c4557-f9e2022e.jpg', 'c43daa45-612283ce.jpg', 'c44e7068-5dafd017.jpg', 'c45d3e0a-b26d5167.jpg',
    'c482a775-8e9c09c8.jpg', 'c48f5464-ebc458bf.jpg', 'c49c8353-f56b7e30.jpg', 'c4a19ba6-872ab53e.jpg',
    'c4eb5462-c917fefd.jpg', 'c5121aec-f19f3e8d.jpg', 'c546c109-92e1a97e.jpg', 'c55b474f-f39c51a8.jpg',
    'c56915dc-1eb8abe1.jpg', 'c569c251-16b6f263.jpg', 'c59500f0-d3272eb4.jpg', 'c5b7b466-4adbd41c.jpg',
    'c5b80ec1-bb9fd172.jpg', 'c5f8a1ae-0d55fc7a.jpg', 'c60d6c5c-689d8921.jpg', 'c6265010-8caa470f.jpg',
    'c6716c49-04844b01.jpg', 'c6716c49-d48c3543.jpg', 'c682e776-d20a6f25.jpg', 'c69e5864-1e46bd03.jpg',
    'c6dca574-6e641016.jpg', 'c6dfbd67-2c7061bf.jpg', 'c71fcdc8-62a7fa51.jpg', 'c74a122c-24c19251.jpg',
    'c74a122c-e7005b78.jpg', 'c7955e20-b48c7a6a.jpg', 'c79e3f00-5879235c.jpg', 'c7c2a783-d98d5909.jpg',
    'c8111dda-1a7dded7.jpg', 'c811793c-7a592fd4.jpg', 'c83c10de-85eef163.jpg', 'c877c1e3-b9b26376.jpg',
    'c89ee093-65ac0019.jpg', 'c8bd76fd-04a1c915.jpg', 'c8d7cfb3-cd9f103f.jpg', 'c8f90ec2-8b0d49de.jpg',
    'c926408d-8de795c6.jpg', 'c933ade8-f63b2db4.jpg', 'c96e0b6a-0acdb8b5.jpg', 'c977423c-43dddd3b.jpg',
    'c993615f-ae30e2cb.jpg', 'c9aa50be-9e8cd399.jpg', 'c9df836e-5955e515.jpg', 'ca049e30-cb9982a6.jpg'
])

DATA_ROOTS = [
    INPUT / "bdd100k-yolo",
    INPUT / "datasets" / "a7madmostafa" / "bdd100k-yolo",
    Path("bdd100k_dataset"),
]


def data_root() -> Path:
    for r in DATA_ROOTS:
        if (r / "val" / "images").is_dir():
            return r
    raise FileNotFoundError(f"Could not find val/images in {[str(r) for r in DATA_ROOTS]}")


def find_checkpoint() -> Path:
    for p in INPUT.rglob("*.pt"):
        if "test" in str(p).lower():
            continue
        if "epoch26" in p.name.lower() or "best" in p.name.lower():
            return p
    local = Path("scratch/kaggle_upload_weights/yolo11m_phase6_epoch26.pt")
    if local.exists():
        return local
    raise FileNotFoundError("Could not find Phase 6 Epoch 26 checkpoint.")


def match_single_image(pred_objs: list, gt_objs: list) -> dict:
    """Matches predictions and ground truths for a single image in O(N_pred * N_gt) time."""
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
    vru_size_stats = {
        "Small (<32²)": {"tp": 0, "fn": 0, "total": 0},
        "Medium [32²-96²)": {"tp": 0, "fn": 0, "total": 0},
        "Large (≥96²)": {"tp": 0, "fn": 0, "total": 0},
    }
    n_gt = len(gt_objs)
    n_pred = len(pred_objs)

    for cname in CLASS_NAMES:
        c_gt = [g for g in gt_objs if g["class_name"] == cname]
        c_pred = [p for p in pred_objs if p["class_name"] == cname]

        if not c_gt and not c_pred:
            continue
        if not c_gt and c_pred:
            class_stats[cname]["fp"] += len(c_pred)
            continue
        if c_gt and not c_pred:
            class_stats[cname]["fn"] += len(c_gt)
            for g in c_gt:
                sz = g["size_group"]
                if g["is_vru"]:
                    vru_size_stats[sz]["fn"] += 1
                    vru_size_stats[sz]["total"] += 1
            continue

        c_pred.sort(key=lambda x: x["conf"], reverse=True)
        pred_boxes = np.array([p["box"] for p in c_pred], dtype=np.float32)
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
        union_area = area_a[:, None] + area_b[None, :] - inter_area
        iou_mat = np.where(union_area > 0, inter_area / union_area, 0.0)

        gt_matched = set()
        for p_idx in range(len(c_pred)):
            best_gt_idx = -1
            best_iou = 0.50
            for g_idx in range(len(c_gt)):
                if g_idx in gt_matched:
                    continue
                if iou_mat[p_idx, g_idx] >= best_iou:
                    best_iou = iou_mat[p_idx, g_idx]
                    best_gt_idx = g_idx

            if best_gt_idx >= 0:
                gt_matched.add(best_gt_idx)
                class_stats[cname]["tp"] += 1
                sz = c_gt[best_gt_idx]["size_group"]
                if c_gt[best_gt_idx]["is_vru"]:
                    vru_size_stats[sz]["tp"] += 1
                    vru_size_stats[sz]["total"] += 1
            else:
                class_stats[cname]["fp"] += 1

        for g_idx in range(len(c_gt)):
            if g_idx not in gt_matched:
                class_stats[cname]["fn"] += 1
                sz = c_gt[g_idx]["size_group"]
                if c_gt[g_idx]["is_vru"]:
                    vru_size_stats[sz]["fn"] += 1
                    vru_size_stats[sz]["total"] += 1

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "class_stats": class_stats,
        "vru_size_stats": vru_size_stats,
    }


def aggregate_partition(image_stats: dict, image_subset: set) -> dict:
    """Aggregates pre-computed per-image statistics in O(N_subset) time."""
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
    vru_size_stats = {
        "Small (<32²)": {"tp": 0, "fn": 0, "total": 0},
        "Medium [32²-96²)": {"tp": 0, "fn": 0, "total": 0},
        "Large (≥96²)": {"tp": 0, "fn": 0, "total": 0},
    }
    total_gt = 0
    total_pred = 0

    for name in image_subset:
        st = image_stats.get(name)
        if not st:
            continue
        total_gt += st["n_gt"]
        total_pred += st["n_pred"]
        for c in CLASS_NAMES:
            class_stats[c]["tp"] += st["class_stats"][c]["tp"]
            class_stats[c]["fp"] += st["class_stats"][c]["fp"]
            class_stats[c]["fn"] += st["class_stats"][c]["fn"]
        for sz in ["Small (<32²)", "Medium [32²-96²)", "Large (≥96²)"]:
            vru_size_stats[sz]["tp"] += st["vru_size_stats"][sz]["tp"]
            vru_size_stats[sz]["fn"] += st["vru_size_stats"][sz]["fn"]
            vru_size_stats[sz]["total"] += st["vru_size_stats"][sz]["total"]

    vru_tp = sum(class_stats[c]["tp"] for c in VRU_NAMES)
    vru_fp = sum(class_stats[c]["fp"] for c in VRU_NAMES)
    vru_fn = sum(class_stats[c]["fn"] for c in VRU_NAMES)
    vru_prec = vru_tp / (vru_tp + vru_fp) if (vru_tp + vru_fp) > 0 else 0.0
    vru_rec = vru_tp / (vru_tp + vru_fn) if (vru_tp + vru_fn) > 0 else 0.0
    vru_f1 = 2 * vru_prec * vru_rec / (vru_prec + vru_rec) if (vru_prec + vru_rec) > 0 else 0.0

    all_tp = sum(class_stats[c]["tp"] for c in CLASS_NAMES)
    all_fp = sum(class_stats[c]["fp"] for c in CLASS_NAMES)
    all_fn = sum(class_stats[c]["fn"] for c in CLASS_NAMES)
    all_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    all_rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    all_f1 = 2 * all_prec * all_rec / (all_prec + all_rec) if (all_prec + all_rec) > 0 else 0.0

    small_vru_tot = vru_size_stats["Small (<32²)"]["total"]
    small_vru_tp = vru_size_stats["Small (<32²)"]["tp"]
    small_vru_recall = small_vru_tp / small_vru_tot if small_vru_tot > 0 else 0.0

    return {
        "image_count": len(image_subset),
        "total_gt_objects": total_gt,
        "total_predictions": total_pred,
        "vru_metrics": {
            "tp": vru_tp, "fp": vru_fp, "fn": vru_fn,
            "precision": round(vru_prec * 100.0, 2),
            "recall": round(vru_rec * 100.0, 2),
            "f1_score": round(vru_f1, 4),
            "small_vru_ground_truth": small_vru_tot,
            "small_vru_true_positives": small_vru_tp,
            "small_vru_recall": round(small_vru_recall * 100.0, 2),
        },
        "all_class_metrics": {
            "tp": all_tp, "fp": all_fp, "fn": all_fn,
            "precision": round(all_prec * 100.0, 2),
            "recall": round(all_rec * 100.0, 2),
            "f1_score": round(all_f1, 4),
        },
        "per_class": {
            c: {
                "tp": class_stats[c]["tp"], "fp": class_stats[c]["fp"], "fn": class_stats[c]["fn"],
                "precision": round(class_stats[c]["tp"] / max(class_stats[c]["tp"] + class_stats[c]["fp"], 1) * 100.0, 2),
                "recall": round(class_stats[c]["tp"] / max(class_stats[c]["tp"] + class_stats[c]["fn"], 1) * 100.0, 2),
            } for c in CLASS_NAMES
        },
        "vru_by_size": {
            sz: {
                "tp": vru_size_stats[sz]["tp"], "fn": vru_size_stats[sz]["fn"], "total": vru_size_stats[sz]["total"],
                "recall": round(vru_size_stats[sz]["tp"] / max(vru_size_stats[sz]["total"], 1) * 100.0, 2),
            } for sz in ["Small (<32²)", "Medium [32²-96²)", "Large (≥96²)"]
        }
    }


def main():
    t0_start = time.time()
    root = data_root()
    val_img_dir = root / "val" / "images"
    val_lbl_dir = root / "val" / "labels"

    val_images = sorted([f.name for f in val_img_dir.glob("*.jpg")])
    assert len(val_images) == 10000, f"Expected 10,000 images, got {len(val_images)}"
    print(f"Validation images verified: {len(val_images)} images found.")

    ckpt_path = find_checkpoint()
    print(f"Found checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()
    assert ckpt_hash == EXPECTED_SHA256, f"Checkpoint hash mismatch: {ckpt_hash} vs {EXPECTED_SHA256}"
    print(f"Verified SHA-256: {ckpt_hash}")

    # 1. Parse Ground Truth
    print("Parsing official ground-truth annotations...")
    t0_gt = time.time()
    gt_by_img = {}
    for img_name in val_images:
        gt_by_img[img_name] = []
        lbl_file = val_lbl_dir / (Path(img_name).stem + ".txt")
        if lbl_file.exists():
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    x1 = (xc - w/2.0) * 1280.0
                    y1 = (yc - h/2.0) * 720.0
                    x2 = (xc + w/2.0) * 1280.0
                    y2 = (yc + h/2.0) * 720.0
                    area = (x2 - x1) * (y2 - y1)
                    if area < 1024.0:
                        sz_grp = "Small (<32²)"
                    elif area < 9216.0:
                        sz_grp = "Medium [32²-96²)"
                    else:
                        sz_grp = "Large (≥96²)"
                    cname = CLASS_NAMES[cid]
                    gt_by_img[img_name].append({
                        "class_name": cname,
                        "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                        "size_group": sz_grp,
                        "is_vru": cname in VRU_NAMES,
                    })
    gt_elapsed = time.time() - t0_gt
    print(f"Parsed annotations in {gt_elapsed:.2f}s.")

    # 2. Partition verification
    all_val_set = set(val_images)
    pilot_set = set(PILOT_COHORT_IMAGE_SET) & all_val_set
    expanded_set = all_val_set - pilot_set
    assert len(pilot_set) == 300
    assert len(expanded_set) == 9700
    print(f"Partitions verified: Pilot = {len(pilot_set)}, Expanded = {len(expanded_set)}")

    # 3. Ultralytics Official model.val() Evaluation at Native Resolution (1280px)
    val_data = {
        "path": str(root),
        "train": str(val_img_dir),
        "val": str(val_img_dir),
        "names": dict(enumerate(CLASS_NAMES))
    }
    val_yaml_path = WORK / "val_official_native.yaml"
    val_yaml_path.write_text(yaml.dump(val_data))

    pass1_file = WORK / "pass1_native_res_map.json"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if pass1_file.exists():
        print(f"Found existing Pass 1 native-res checkpoint at {pass1_file}. Loading...")
        with open(pass1_file, "r", encoding="utf-8") as f:
            pass1_summary = json.load(f)
    else:
        print(f"\n--- Running Official Ultralytics model.val() at Native Resolution (imgsz={IMGSZ}, batch={BATCH}) ---")
        t0_val = time.time()
        model = YOLO(str(ckpt_path))
        val_res = model.val(
            data=str(val_yaml_path),
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            plots=True,
            project=str(WORK / "eval_native_run"),
            name="official_val_native",
            verbose=True,
        )
        val_elapsed = time.time() - t0_val
        val_peak_gpu_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 0.0
        print(f"Pass 1 (model.val() @ {IMGSZ}px) completed in {val_elapsed:.2f}s. Peak GPU memory: {val_peak_gpu_mb} MB.")

        ap50 = val_res.box.all_ap[:, 0]
        ap50_95 = val_res.box.all_ap.mean(1)
        classes = list(val_res.box.ap_class_index)

        ultralytics_per_class = {
            CLASS_NAMES[c]: {
                "AP50": round(float(ap50[i]) * 100.0, 2),
                "AP50-95": round(float(ap50_95[i]) * 100.0, 2)
            } for i, c in enumerate(classes)
        }
        vru_indices = [classes.index(c) for c in VRU_IDS if c in classes]
        vru_macro_ap50 = round(float(ap50[vru_indices].mean()) * 100.0, 2)
        vru_macro_ap50_95 = round(float(ap50_95[vru_indices].mean()) * 100.0, 2)

        pass1_summary = {
            "milestone": "Milestone 2 — Native Resolution (1280px) Evaluation",
            "benchmark_metadata": {
                "checkpoint_path": str(ckpt_path),
                "checkpoint_sha256": ckpt_hash,
                "architecture": "YOLO11m",
                "epoch": 26,
                "imgsz": IMGSZ,
                "batch_size": BATCH,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "peak_gpu_memory_mb": val_peak_gpu_mb,
                "pass1_runtime_seconds": round(val_elapsed, 2),
            },
            "ultralytics_official_map": {
                "all_class_mAP50": round(float(val_res.box.map50) * 100.0, 2),
                "all_class_mAP50_95": round(float(val_res.box.map) * 100.0, 2),
                "all_class_precision": round(float(val_res.box.mp) * 100.0, 2),
                "all_class_recall": round(float(val_res.box.mr) * 100.0, 2),
                "vru_macro_mAP50": vru_macro_ap50,
                "vru_macro_mAP50_95": vru_macro_ap50_95,
                "per_class_ap": ultralytics_per_class,
            }
        }
        with open(pass1_file, "w", encoding="utf-8") as f:
            json.dump(pass1_summary, f, indent=2)
        print(f"Saved Pass 1 metrics immediately to {pass1_file}.")

    # 4. Operational Prediction Sweep & Matching at Native Resolution (1280px)
    print(f"\n--- Running Operational Prediction Sweep & Matching at {IMGSZ}px (conf=0.25, iou=0.65) ---")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0_pred = time.time()
    total_imgs = len(val_images)
    ckpt_stats_file = WORK / "checkpoint_native_res_stats.json"
    image_stats = {}

    if ckpt_stats_file.exists():
        print(f"Found recoverable progress checkpoint at {ckpt_stats_file}. Loading...")
        with open(ckpt_stats_file, "r", encoding="utf-8") as f:
            image_stats = json.load(f)
        print(f"Recovered {len(image_stats)}/{total_imgs} images.")

    remaining_images = [img for img in val_images if img not in image_stats]
    print(f"Images to process: {len(remaining_images)}/{total_imgs}")

    if remaining_images:
        model = YOLO(str(ckpt_path))
        for i in range(0, len(remaining_images), BATCH):
            batch_names = remaining_images[i:i+BATCH]
            batch_paths = [str(val_img_dir / name) for name in batch_names]
            results = model.predict(batch_paths, imgsz=IMGSZ, conf=0.25, iou=0.65, device=DEVICE, verbose=False)
            for name, r in zip(batch_names, results):
                img_preds = []
                for b in r.boxes:
                    cid = int(b.cls.item())
                    cname = model.names[cid]
                    conf = float(b.conf.item())
                    xyxy = b.xyxy[0].cpu().numpy()
                    img_preds.append({
                        "class_name": cname,
                        "conf": conf,
                        "box": xyxy,
                    })
                image_stats[name] = match_single_image(img_preds, gt_by_img.get(name, []))

            if (i // BATCH) % 100 == 0 or (i + BATCH >= len(remaining_images)):
                completed = len(image_stats)
                print(f"Processed {completed}/{total_imgs} images ({(completed/total_imgs)*100.0:.1f}%)...")
                with open(ckpt_stats_file, "w", encoding="utf-8") as f:
                    json.dump(image_stats, f)

    pred_elapsed = time.time() - t0_pred
    pred_fps = round(total_imgs / max(pred_elapsed, 1e-6), 2)
    latency_per_img_ms = round((pred_elapsed / total_imgs) * 1000.0, 2)
    pred_peak_gpu_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 0.0
    print(f"Prediction sweep completed in {pred_elapsed:.1f}s ({pred_fps} FPS, {latency_per_img_ms} ms/img). Peak GPU: {pred_peak_gpu_mb} MB.")

    # 5. Partition-wise Aggregation
    print("Aggregating partition metrics...")
    t0_agg = time.time()
    pilot_metrics = aggregate_partition(image_stats, pilot_set)
    expanded_metrics = aggregate_partition(image_stats, expanded_set)
    aggregate_metrics = aggregate_partition(image_stats, all_val_set)
    agg_elapsed = time.time() - t0_agg
    print(f"Partition aggregation completed in {agg_elapsed:.3f}s.")

    # 6. Side-by-Side Delta Comparison Against Frozen 960px Baseline
    b960_agg = BASELINE_960["aggregate_10000"]
    delta_vru_recall = round(aggregate_metrics["vru_metrics"]["recall"] - b960_agg["vru_recall"], 2)
    delta_vru_prec = round(aggregate_metrics["vru_metrics"]["precision"] - b960_agg["vru_precision"], 2)
    delta_vru_f1 = round(aggregate_metrics["vru_metrics"]["f1_score"] - b960_agg["vru_f1"], 4)
    delta_small_vru_rec = round(aggregate_metrics["vru_metrics"]["small_vru_recall"] - b960_agg["small_vru_recall"], 2)
    delta_small_vru_tp = aggregate_metrics["vru_metrics"]["small_vru_true_positives"] - b960_agg["small_vru_tp"]
    delta_map50 = round(pass1_summary["ultralytics_official_map"]["all_class_mAP50"] - b960_agg["all_class_map50"], 2)
    delta_vru_map50 = round(pass1_summary["ultralytics_official_map"]["vru_macro_mAP50"] - b960_agg["vru_macro_map50"], 2)
    delta_fps = round(pred_fps - b960_agg["fps"], 2)

    delta_pilot_vru_rec = round(pilot_metrics["vru_metrics"]["recall"] - BASELINE_960["pilot_300"]["vru_recall"], 2)
    delta_pilot_small_rec = round(pilot_metrics["vru_metrics"]["small_vru_recall"] - BASELINE_960["pilot_300"]["small_vru_recall"], 2)
    delta_exp_vru_rec = round(expanded_metrics["vru_metrics"]["recall"] - BASELINE_960["expanded_9700"]["vru_recall"], 2)
    delta_exp_small_rec = round(expanded_metrics["vru_metrics"]["small_vru_recall"] - BASELINE_960["expanded_9700"]["small_vru_recall"], 2)

    comparison_report = {
        "experiment": "Controlled Native-Resolution (1280px) vs Frozen 960px Baseline",
        "key_deltas": {
            "aggregate_10000": {
                "vru_recall": {"baseline_960": b960_agg["vru_recall"], "native_1280": aggregate_metrics["vru_metrics"]["recall"], "delta": delta_vru_recall},
                "vru_precision": {"baseline_960": b960_agg["vru_precision"], "native_1280": aggregate_metrics["vru_metrics"]["precision"], "delta": delta_vru_prec},
                "vru_f1": {"baseline_960": b960_agg["vru_f1"], "native_1280": aggregate_metrics["vru_metrics"]["f1_score"], "delta": delta_vru_f1},
                "small_vru_recall": {"baseline_960": b960_agg["small_vru_recall"], "native_1280": aggregate_metrics["vru_metrics"]["small_vru_recall"], "delta": delta_small_vru_rec},
                "small_vru_tp_gain": delta_small_vru_tp,
                "all_class_map50": {"baseline_960": b960_agg["all_class_map50"], "native_1280": pass1_summary["ultralytics_official_map"]["all_class_mAP50"], "delta": delta_map50},
                "vru_macro_map50": {"baseline_960": b960_agg["vru_macro_map50"], "native_1280": pass1_summary["ultralytics_official_map"]["vru_macro_mAP50"], "delta": delta_vru_map50},
                "throughput_fps": {"baseline_960": b960_agg["fps"], "native_1280": pred_fps, "delta": delta_fps},
                "latency_per_img_ms": latency_per_img_ms,
                "peak_gpu_memory_mb": pred_peak_gpu_mb,
            },
            "per_class_vru": {
                "person": {
                    "ap50_delta": round(pass1_summary["ultralytics_official_map"]["per_class_ap"]["person"]["AP50"] - b960_agg["person_ap50"], 2),
                    "recall_delta": round(aggregate_metrics["per_class"]["person"]["recall"] - b960_agg["person_recall"], 2),
                },
                "rider": {
                    "ap50_delta": round(pass1_summary["ultralytics_official_map"]["per_class_ap"]["rider"]["AP50"] - b960_agg["rider_ap50"], 2),
                    "recall_delta": round(aggregate_metrics["per_class"]["rider"]["recall"] - b960_agg["rider_recall"], 2),
                },
                "bike": {
                    "ap50_delta": round(pass1_summary["ultralytics_official_map"]["per_class_ap"]["bike"]["AP50"] - b960_agg["bike_ap50"], 2),
                    "recall_delta": round(aggregate_metrics["per_class"]["bike"]["recall"] - b960_agg["bike_recall"], 2),
                },
                "motor": {
                    "ap50_delta": round(pass1_summary["ultralytics_official_map"]["per_class_ap"]["motor"]["AP50"] - b960_agg["motor_ap50"], 2),
                    "recall_delta": round(aggregate_metrics["per_class"]["motor"]["recall"] - b960_agg["motor_recall"], 2),
                }
            },
            "partition_consistency": {
                "pilot_300": {"vru_recall_delta": delta_pilot_vru_rec, "small_vru_recall_delta": delta_pilot_small_rec},
                "expanded_9700": {"vru_recall_delta": delta_exp_vru_rec, "small_vru_recall_delta": delta_exp_small_rec},
                "persists_across_both_partitions": (delta_pilot_vru_rec > 0 and delta_exp_vru_rec > 0),
            }
        }
    }

    # 7. Final Report Save
    total_runtime_s = round(time.time() - t0_start, 1)
    final_report = {
        "milestone": "Milestone 2 — Native Resolution (1280px) Benchmark",
        "benchmark_metadata": {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_hash,
            "architecture": "YOLO11m",
            "epoch": 26,
            "imgsz": IMGSZ,
            "batch_size": BATCH,
            "operational_conf_threshold": 0.25,
            "nms_iou_threshold": 0.65,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "peak_gpu_memory_mb": pred_peak_gpu_mb,
            "timing_breakdown": {
                "ground_truth_parsing_seconds": round(gt_elapsed, 2),
                "pass1_model_val_seconds": pass1_summary["benchmark_metadata"].get("pass1_runtime_seconds", 0.0),
                "pass2_predict_and_match_seconds": round(pred_elapsed, 2),
                "pass3_partition_aggregation_seconds": round(agg_elapsed, 4),
                "total_runtime_seconds": total_runtime_s,
                "prediction_throughput_fps": pred_fps,
                "latency_per_img_ms": latency_per_img_ms,
            }
        },
        "ultralytics_official_map": pass1_summary["ultralytics_official_map"],
        "partitions": {
            "pilot_300_development_cohort": pilot_metrics,
            "expanded_9700_validation_partition": expanded_metrics,
            "aggregate_10000_official_validation": aggregate_metrics,
        },
        "comparison_vs_960_baseline": comparison_report,
    }

    out_file = WORK / "native_res_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    cmp_file = WORK / "comparison_vs_960_baseline.json"
    with open(cmp_file, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    print("\n=======================================================")
    print("MILESTONE 2 (NATIVE 1280px) BENCHMARK COMPLETE")
    print(f"Aggregate 10,000-Image mAP@50:    {pass1_summary['ultralytics_official_map']['all_class_mAP50']}% (Δ {delta_map50:+.2f}%)")
    print(f"Aggregate VRU Macro mAP@50:       {pass1_summary['ultralytics_official_map']['vru_macro_mAP50']}% (Δ {delta_vru_map50:+.2f}%)")
    print(f"Aggregate VRU Recall (conf=0.25): {aggregate_metrics['vru_metrics']['recall']}% (Δ {delta_vru_recall:+.2f}%)")
    print(f"Aggregate Small-VRU Recall:       {aggregate_metrics['vru_metrics']['small_vru_recall']}% (Δ {delta_small_vru_rec:+.2f}%)")
    print(f"Small-VRU True Positives Gain:    {delta_small_vru_tp:+d} detections")
    print(f"Throughput & Latency:             {pred_fps} FPS ({latency_per_img_ms} ms/img) vs 20.47 FPS baseline")
    print(f"Peak GPU Memory:                  {pred_peak_gpu_mb} MB")
    print(f"Consistency Across Partitions:    Pilot Δ {delta_pilot_vru_rec:+.2f}%, Expanded Δ {delta_exp_vru_rec:+.2f}%")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
