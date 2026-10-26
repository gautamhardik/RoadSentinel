import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
from ultralytics import YOLO

val_img_dir = Path("bdd100k_dataset/val/images")
val_lbl_dir = Path("bdd100k_dataset/val/labels")

VRU_CLASS_NAMES = ["person", "rider", "bike", "motor"]
GT_CLASSES = {
    0: "person", 1: "rider", 2: "car", 3: "bus", 4: "truck",
    5: "bike", 6: "motor", 7: "traffic light", 8: "traffic sign", 9: "train"
}

def compute_box_iou(boxA, boxB):
    xi1, yi1 = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xi2, yi2 = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / max(union, 1e-8)

model = YOLO("experiments/phase5a_modeling/runs/phase5a_u_ablation/phase5a_training/yolo11s_vru_finetune/weights/best.pt")
print("Model loaded. names:", model.names)

# Evaluate on first 100 images
images = sorted(list(val_img_dir.glob("*.jpg")))[:100]
print(f"Testing on {len(images)} images...")

tp_by_cls = defaultdict(int)
fp_by_cls = defaultdict(int)
gt_by_cls = defaultdict(int)

for img_path in images:
    lbl_path = val_lbl_dir / (img_path.stem + ".txt")
    gts = defaultdict(list)
    if lbl_path.exists():
        with open(lbl_path) as fp:
            for l in fp:
                p = l.strip().split()
                if p:
                    cid = int(p[0])
                    cname = GT_CLASSES[cid]
                    xc, yc, bw, bh = [float(x) for x in p[1:5]]
                    x1 = (xc - bw / 2.0) * 1280.0
                    y1 = (yc - bh / 2.0) * 720.0
                    x2 = (xc + bw / 2.0) * 1280.0
                    y2 = (yc + bh / 2.0) * 720.0
                    gts[cname].append([x1, y1, x2, y2])
                    gt_by_cls[cname] += 1

    img = cv2.imread(str(img_path))
    res = model.predict(source=img, imgsz=960, conf=0.25, verbose=False)[0]

    preds_by_c = defaultdict(list)
    for b in res.boxes:
        cname = model.names[int(b.cls.item())]
        coords = b.xyxy[0].cpu().numpy().tolist()
        conf = float(b.conf.item())
        preds_by_c[cname].append((conf, coords))

    for cname in model.names.values():
        c_preds = sorted(preds_by_c[cname], key=lambda x: x[0], reverse=True)
        c_gts = gts[cname]
        matched_gt = set()
        for conf, pbox in c_preds:
            best_iou = 0.0
            best_idx = -1
            for gidx, gbox in enumerate(c_gts):
                if gidx in matched_gt: continue
                iou = compute_box_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gidx
            if best_iou >= 0.50:
                tp_by_cls[cname] += 1
                matched_gt.add(best_idx)
            else:
                fp_by_cls[cname] += 1

print("\n--- RESULTS on 100 images with model.names ---")
vru_tp = sum(tp_by_cls[c] for c in VRU_CLASS_NAMES)
vru_fp = sum(fp_by_cls[c] for c in VRU_CLASS_NAMES)
vru_gt = sum(gt_by_cls[c] for c in VRU_CLASS_NAMES)
vru_prec = vru_tp / max(vru_tp + vru_fp, 1)
vru_rec = vru_tp / max(vru_gt, 1)
print(f"VRU (person, rider, bike, motor): TP={vru_tp}, FP={vru_fp}, GT={vru_gt}")
print(f"VRU Precision: {vru_prec:.4f}, Recall: {vru_rec:.4f}")
for c in VRU_CLASS_NAMES:
    c_tp = tp_by_cls[c]
    c_fp = fp_by_cls[c]
    c_gt = gt_by_cls[c]
    cp = c_tp / max(c_tp + c_fp, 1)
    cr = c_tp / max(c_gt, 1)
    print(f"  {c}: TP={c_tp}, FP={c_fp}, GT={c_gt}, Prec={cp:.4f}, Rec={cr:.4f}")
