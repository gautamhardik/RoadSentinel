import json, sys

sys.stdout.reconfigure(encoding='utf-8')

p3 = json.load(open('experiments/phase3_foveation/runs/kaggle_run_v2/phase3_foveation_metrics.json'))
p4c = json.load(open('experiments/phase4c_validation/phase4c_metrics.json', encoding='utf-8'))
p4d = json.load(open('experiments/phase4d_validation/phase4d_metrics.json', encoding='utf-8'))

def check_metrics(name, source, m, expected_gt=None):
    tp = m.get('vru_tp')
    fp = m.get('vru_fp')
    fn = m.get('vru_fn')
    prec = m.get('vru_micro_precision_50')
    rec = m.get('vru_micro_recall_50')
    f1 = m.get('vru_micro_f1_50')

    print(f"\n--- Checking {name} ({source}) ---")
    if tp is None:
        print("  TP/FP/FN not in top dict, checking per-class:")
        tp = sum(r['tp_50'] for r in m.get('per_class_results', []) if r['is_vru'])
        fp = sum(r['fp_50'] for r in m.get('per_class_results', []) if r['is_vru'])
        fn = sum(r['fn_50'] for r in m.get('per_class_results', []) if r['is_vru'])
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    print(f"  TP={tp}, FP={fp}, FN={fn}, GT={tp+fn}")
    calc_prec = tp / max(tp + fp, 1)
    calc_rec = tp / max(tp + fn, 1)
    calc_f1 = 2 * calc_prec * calc_rec / max(calc_prec + calc_rec, 1e-8)
    
    print(f"  Reported Prec: {prec:.4f} | Calculated Prec: {calc_prec:.4f} | Diff: {abs(prec - calc_prec):.6f}")
    print(f"  Reported Rec:  {rec:.4f} | Calculated Rec:  {calc_rec:.4f} | Diff: {abs(rec - calc_rec):.6f}")
    print(f"  Reported F1:   {f1:.4f} | Calculated F1:   {calc_f1:.4f} | Diff: {abs(f1 - calc_f1):.6f}")
    if expected_gt is not None:
        print(f"  GT Check: TP+FN = {tp+fn} == Expected {expected_gt} -> {tp+fn == expected_gt}")

# 1. Phase 3.4 ActiveFoveation-960
check_metrics("ActiveFoveation-960", "Phase 3.4 Full Val", p3['configurations']['ActiveFoveation-960'], 15373)

# 2. Phase 4C ActiveFoveation-960 & Calibrated
for cname in ['ActiveFoveation-960', 'Calibrated-ActiveFoveation-960']:
    cfg = p4c['configurations'][cname]
    check_metrics(cname, "Phase 4C Full Val", cfg['full_validation_metrics'], 15373)
    check_metrics(cname, "Phase 4C Pilot 300", cfg['pilot_cohort_metrics'], 1740)
    check_metrics(cname, "Phase 4C Non-Pilot 9700", cfg['non_pilot_confirmation_metrics'], 13633)

# 3. Phase 4D all 5 configs
for cname in p4d['configurations']:
    cfg = p4d['configurations'][cname]
    check_metrics(cname, "Phase 4D Full Val", cfg['full_validation_metrics'], 15373)
    check_metrics(cname, "Phase 4D Pilot 300", cfg['pilot_cohort_metrics'], 1740)
    check_metrics(cname, "Phase 4D Non-Pilot 9700", cfg['non_pilot_confirmation_metrics'], 13633)
