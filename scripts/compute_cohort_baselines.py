import json
import os
import re
from collections import defaultdict
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
records_path = repo_root / "experiments/phase3_foveation/runs/kaggle_run_v2/per_image_records.jsonl"
labels_dir = repo_root / "bdd100k_dataset/val/labels"
p4c_path = repo_root / "kaggle_jobs/calibration_benchmark/evaluate_phase4c_wbf.py"

# 1. Pilot cohort images (300 images)
p4c_text = open(p4c_path, "r", encoding="utf-8").read()
pilot_images = set(re.findall(r'b[0-9a-f]{7}-[0-9a-f]{8}\.jpg|c[0-9a-f]{7}-[0-9a-f]{8}\.jpg', p4c_text))
print(f"Pilot cohort images: {len(pilot_images)}")

# 2. GT labels
gt_classes = {0: "person", 1: "rider", 2: "car", 3: "bus", 4: "truck", 5: "bike", 6: "motor", 7: "traffic light", 8: "traffic sign", 9: "train"}
vru_classes = {"person", "rider", "bike", "motor"}
gt_by_img_cls = defaultdict(lambda: defaultdict(list))

gt_totals = {"all": defaultdict(int), "pilot": defaultdict(int), "conf": defaultdict(int)}
for f in sorted(os.listdir(labels_dir)):
    if not f.endswith(".txt"):
        continue
    iname = f.replace(".txt", ".jpg")
    cohort = "pilot" if iname in pilot_images else "conf"
    for line in open(labels_dir / f):
        parts = line.strip().split()
        if not parts:
            continue
        cname = gt_classes.get(int(parts[0]))
        bw = float(parts[3]) * 1280.0
        bh = float(parts[4]) * 720.0
        area = bw * bh
        sz = "small" if area < 1024.0 else ("medium" if area < 9216.0 else "large")
        gt_by_img_cls[iname][cname].append(sz)
        if cname in vru_classes:
            gt_totals["all"]["vru"] += 1
            gt_totals["all"][sz] += 1
            gt_totals[cohort]["vru"] += 1
            gt_totals[cohort][sz] += 1

# 3. TP and Size TP from per_image_records
tps = {"all": defaultdict(int), "pilot": defaultdict(int), "conf": defaultdict(int)}
size_tps = {
    "all": {"small": 0, "medium": 0, "large": 0},
    "pilot": {"small": 0, "medium": 0, "large": 0},
    "conf": {"small": 0, "medium": 0, "large": 0},
}

with open(records_path, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("config") != "ActiveFoveation-960":
            continue
        iname = rec["image"]
        cohort = "pilot" if iname in pilot_images else "conf"
        tps["all"]["vru"] += rec["vru_tp_count"]
        tps[cohort]["vru"] += rec["vru_tp_count"]
        for mv in rec["matched_vrus"]:
            cname = mv["class"]
            gidx = mv["gt_idx"]
            sz = gt_by_img_cls[iname][cname][gidx]
            size_tps["all"][sz] += 1
            size_tps[cohort][sz] += 1

print("\n--- EXACT CANONICAL BASELINE GROUND TRUTH & RECALL ---")
for k, label in [("all", "Full Validation (10,000)"), ("pilot", "Pilot Cohort (300)"), ("conf", "Confirmation Cohort (9,700)")]:
    gt_v = gt_totals[k]["vru"]
    tp_v = tps[k]["vru"]
    fn_v = gt_v - tp_v
    rec_v = (tp_v / gt_v) * 100 if gt_v else 0
    sm_gt = gt_totals[k]["small"]
    sm_tp = size_tps[k]["small"]
    sm_rec = (sm_tp / sm_gt) * 100 if sm_gt else 0
    med_gt = gt_totals[k]["medium"]
    med_tp = size_tps[k]["medium"]
    med_rec = (med_tp / med_gt) * 100 if med_gt else 0
    lg_gt = gt_totals[k]["large"]
    lg_tp = size_tps[k]["large"]
    lg_rec = (lg_tp / lg_gt) * 100 if lg_gt else 0
    print(f"[{label}]")
    print(f"  VRU GT: {gt_v:,} | TP: {tp_v:,} | FN: {fn_v:,} | Recall: {rec_v:.2f}%")
    print(f"  Small VRU:  GT={sm_gt:,} | TP={sm_tp:,} | Recall={sm_rec:.2f}%")
    print(f"  Medium VRU: GT={med_gt:,} | TP={med_tp:,} | Recall={med_rec:.2f}%")
    print(f"  Large VRU:  GT={lg_gt:,} | TP={lg_tp:,} | Recall={lg_rec:.2f}%")
