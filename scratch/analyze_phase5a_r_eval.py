import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

with open("experiments/phase5a_modeling/runs/phase5a_r_training/eval/phase4d_metrics.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for name, cfg in data["configurations"].items():
    m = cfg["non_pilot_confirmation_metrics"]
    print(f"=== {name} ===")
    print(f"VRU TP: {m['vru_tp']}, FP: {m['vru_fp']}, FN: {m['vru_fn']}")
    print(f"Precision: {m['vru_micro_precision_50']:.4f}, Recall: {m['vru_micro_recall_50']:.4f}, F1: {m['vru_micro_f1_50']:.4f}")
    sg = m.get("size_group_results", {})
    for grp_name, grp_data in sg.items():
        ascii_grp = grp_name.replace("≥", ">=").replace("²", "^2")
        print(f"  {ascii_grp}: VRU Recall = {grp_data.get('vru_recall', 0):.4f} (matched {grp_data.get('vru_matched', 0)}/{grp_data.get('vru_gt', 0)})")
    print("VRU Per-Class:")
    for c in m.get("per_class_results", []):
        if c.get("is_vru"):
            print(f"  {c['class_name']}: TP={c['tp_50']}, FP={c['fp_50']}, FN={c['fn_50']}, Prec={c['precision_50']:.4f}, Rec={c['recall_50']:.4f}, AP50={c['ap_50']:.4f}")
    print()
