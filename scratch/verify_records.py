import json, os, sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

lbl_dir = Path('bdd100k_dataset/val/labels')
gt_by_img_cls = defaultdict(lambda: defaultdict(list))
size_gt_totals = {'Small (<32²)': 0, 'Medium [32²-96²)': 0, 'Large (≥96²)': 0}
size_gt_pilot = {'Small (<32²)': 0, 'Medium [32²-96²)': 0, 'Large (≥96²)': 0}
size_gt_nonpilot = {'Small (<32²)': 0, 'Medium [32²-96²)': 0, 'Large (≥96²)': 0}

GT_CLASSES = {0: 'person', 1: 'rider', 2: 'car', 3: 'bus', 4: 'truck', 5: 'bike', 6: 'motor', 7: 'traffic light', 8: 'traffic sign', 9: 'train'}
VRU_CLASS_NAMES = {'person', 'rider', 'bike', 'motor'}

# Load pilot cohort set
p4c_text = open('kaggle_jobs/calibration_benchmark/evaluate_phase4c_wbf.py', encoding='utf-8').read()
import re
pilot_images = set(re.findall(r'b[0-9a-f]{7}-[0-9a-f]{8}\.jpg|c[0-9a-f]{7}-[0-9a-f]{8}\.jpg', p4c_text))
print(f"Loaded {len(pilot_images)} pilot cohort images.")

for f in sorted(os.listdir(lbl_dir)):
    if not f.endswith('.txt'): continue
    iname = f.replace('.txt', '.jpg')
    is_pilot = iname in pilot_images
    for line in open(lbl_dir / f):
        parts = line.strip().split()
        if not parts: continue
        cid = int(parts[0])
        cname = GT_CLASSES.get(cid)
        bw = float(parts[3]) * 1280.0
        bh = float(parts[4]) * 720.0
        area = bw * bh
        if area < 1024.0: sz = 'Small (<32²)'
        elif area < 9216.0: sz = 'Medium [32²-96²)'
        else: sz = 'Large (≥96²)'
        
        gt_by_img_cls[iname][cname].append(sz)
        if cname in VRU_CLASS_NAMES:
            size_gt_totals[sz] += 1
            if is_pilot:
                size_gt_pilot[sz] += 1
            else:
                size_gt_nonpilot[sz] += 1

print('Ground truth VRU size totals (Full):', size_gt_totals)
print('Ground truth VRU size totals (Pilot 300):', size_gt_pilot)
print('Ground truth VRU size totals (Non-Pilot 9700):', size_gt_nonpilot)

# Recompute from records
af_tp_full = {'Small (<32²)': 0, 'Medium [32²-96²)': 0, 'Large (≥96²)': 0}
af_tp_pilot = {'Small (<32²)': 0, 'Medium [32²-96²)': 0, 'Large (≥96²)': 0}
af_tp_nonpilot = {'Small (<32²)': 0, 'Medium [32²-96²)': 0, 'Large (≥96²)': 0}

tot_tp_full = 0
tot_tp_pilot = 0
tot_tp_nonpilot = 0

with open('experiments/phase3_foveation/runs/kaggle_run_v2/per_image_records.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        if rec['config'] != 'ActiveFoveation-960': continue
        iname = rec['image']
        is_pilot = iname in pilot_images
        tp_cnt = rec['vru_tp_count']
        tot_tp_full += tp_cnt
        if is_pilot: tot_tp_pilot += tp_cnt
        else: tot_tp_nonpilot += tp_cnt

        for mv in rec['matched_vrus']:
            cname = mv['class']
            gidx = mv['gt_idx']
            sz = gt_by_img_cls[iname][cname][gidx]
            af_tp_full[sz] += 1
            if is_pilot: af_tp_pilot[sz] += 1
            else: af_tp_nonpilot[sz] += 1

print(f'\n--- RECOMPUTED RECALL METRICS ---')
print(f'Full Val: TP={tot_tp_full} / {sum(size_gt_totals.values())} = {tot_tp_full/sum(size_gt_totals.values()):.4f}')
for sz in af_tp_full:
    print(f'  {sz}: TP={af_tp_full[sz]}/{size_gt_totals[sz]} = {af_tp_full[sz]/size_gt_totals[sz]:.4f}')

print(f'\nPilot 300: TP={tot_tp_pilot} / {sum(size_gt_pilot.values())} = {tot_tp_pilot/sum(size_gt_pilot.values()):.4f}')
for sz in af_tp_pilot:
    print(f'  {sz}: TP={af_tp_pilot[sz]}/{size_gt_pilot[sz]} = {af_tp_pilot[sz]/size_gt_pilot[sz]:.4f}')

print(f'\nNon-Pilot 9700: TP={tot_tp_nonpilot} / {sum(size_gt_nonpilot.values())} = {tot_tp_nonpilot/sum(size_gt_nonpilot.values()):.4f}')
for sz in af_tp_nonpilot:
    print(f'  {sz}: TP={af_tp_nonpilot[sz]}/{size_gt_nonpilot[sz]} = {af_tp_nonpilot[sz]/size_gt_nonpilot[sz]:.4f}')
