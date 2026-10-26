import json

with open("experiments/phase6_system_eval/system_evaluation_report.json") as f:
    data = json.load(f)

for pipe in ["raw_yolo11m", "foveation_wbf", "bytetrack"]:
    p_data = data[pipe]
    print(f"=== {pipe} ===")
    vru_classes = ["person", "rider", "bike", "motor"]
    tps = sum(p_data["per_class"][c]["tp"] for c in vru_classes)
    fps = sum(p_data["per_class"][c]["fp"] for c in vru_classes)
    fns = sum(p_data["per_class"][c]["fn"] for c in vru_classes)
    gts = sum(p_data["per_class"][c]["gt_count"] for c in vru_classes)

    prec = tps / max(tps + fps, 1)
    rec = tps / max(gts, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    print(f"  Sum of class TPs: {tps} (recorded vru_tp: {p_data['vru_tp']})")
    print(f"  Sum of class FPs: {fps} (recorded vru_fp: {p_data['vru_fp']})")
    print(f"  Sum of class FNs: {fns} (recorded vru_fn: {p_data['vru_fn']})")
    print(f"  Computed Prec: {prec:.4f} (recorded: {p_data['vru_operational_precision']:.4f})")
    print(f"  Computed Rec:  {rec:.4f} (recorded: {p_data['vru_operational_recall']:.4f})")
    print(f"  Computed F1:   {f1:.4f} (recorded: {p_data['vru_operational_f1']:.4f})")

    for c in vru_classes:
        cd = p_data["per_class"][c]
        c_p = cd["tp"] / max(cd["tp"] + cd["fp"], 1)
        c_r = cd["tp"] / max(cd["gt_count"], 1)
        c_f1 = 2 * c_p * c_r / max(c_p + c_r, 1e-8)
        print(f"    {c:6s}: GT={cd['gt_count']:4d} TP={cd['tp']:4d} FP={cd['fp']:3d} FN={cd['fn']:3d} | P={c_p*100:.2f}% ({cd['precision']*100:.2f}%) R={c_r*100:.2f}% ({cd['recall']*100:.2f}%) F1={c_f1:.4f} ({cd['f1']:.4f})")
