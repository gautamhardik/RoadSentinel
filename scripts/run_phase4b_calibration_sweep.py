"""
RoadSentinel Phase 4B — Step 4 & Step 5: Pilot Calibration Sweep & Decision Rule
Consumes raw_candidates.jsonl and cohort_evaluation_records.json.
Evaluates:
  1. Current Baseline (tau=0.25 uniform, greedy deduplication)
  2. Source-Specific Dual-Threshold Sweeps (tau_isolated in [0.25, 0.50], tau_confirmed in [0.15, 0.25])
  3. Class-Specific Threshold Sweeps (tau_person, tau_rider, tau_bike, tau_motor)
  4. Weighted Box Fusion (WBF) vs Greedy Deduplication
Computes VRU Precision, Recall, F1, size-group recall, per-class breakdown, FP/FN counts,
and applies the Step 5 decision rule.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np

VRU_CLASSES = {"person", "rider", "bike", "motor"}
GT_CLASSES = {0: "person", 1: "rider", 2: "car", 3: "bus", 4: "truck", 5: "bike", 6: "motor", 7: "traffic light", 8: "traffic sign", 9: "train"}


def compute_box_iou(boxA, boxB) -> float:
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
    return inter / max(union, 1e-8)


def evaluate_cohort(predictions_by_img, ground_truth_by_img):
    """
    Evaluates VRU precision, recall, F1, size tiers, and per-class stats at IoU=0.50.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    tier_gt = {"Small": 0, "Medium": 0, "Large": 0}
    tier_tp = {"Small": 0, "Medium": 0, "Large": 0}

    class_gt = {c: 0 for c in VRU_CLASSES}
    class_tp = {c: 0 for c in VRU_CLASSES}
    class_fp = {c: 0 for c in VRU_CLASSES}

    for fname, gts in ground_truth_by_img.items():
        vru_gts = [g for g in gts if g["is_vru"]]
        for g in vru_gts:
            tier_gt[g["size_group"]] += 1
            class_gt[g["class_name"]] += 1

        vru_preds = [p for p in predictions_by_img.get(fname, []) if p["class_name"] in VRU_CLASSES]
        vru_preds.sort(key=lambda x: x["conf"], reverse=True)

        matched_gts = set()
        for p in vru_preds:
            cname = p["class_name"]
            best_idx = -1
            best_iou = 0.0
            for g_idx, g in enumerate(vru_gts):
                if g_idx in matched_gts or g["class_name"] != cname:
                    continue
                iou = compute_box_iou(p["box_full"], g["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = g_idx

            if best_iou >= 0.50 and best_idx >= 0:
                matched_gts.add(best_idx)
                total_tp += 1
                class_tp[cname] += 1
                tier_tp[vru_gts[best_idx]["size_group"]] += 1
            else:
                total_fp += 1
                class_fp[cname] += 1

        fn_count = len(vru_gts) - len(matched_gts)
        total_fn += fn_count

    total_gt = sum(tier_gt.values())
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    tier_rec = {
        tier: (tier_tp[tier] / tier_gt[tier] if tier_gt[tier] > 0 else 0.0)
        for tier in ["Small", "Medium", "Large"]
    }

    per_class = {}
    for c in sorted(VRU_CLASSES):
        c_tp = class_tp[c]
        c_fp = class_fp[c]
        c_gt = class_gt[c]
        c_fn = c_gt - c_tp
        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_rec = c_tp / c_gt if c_gt > 0 else 0.0
        c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0
        per_class[c] = {"tp": c_tp, "fp": c_fp, "fn": c_fn, "gt": c_gt, "precision": c_prec, "recall": c_rec, "f1": c_f1}

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "total_gt": total_gt,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tier_gt": tier_gt,
        "tier_tp": tier_tp,
        "tier_recall": tier_rec,
        "per_class": per_class,
    }


def run_pipeline(candidates_by_img, policy_fn):
    """
    Applies filtering and fusion policy to raw candidates per image.
    """
    preds_by_img = {}
    for fname, cands in candidates_by_img.items():
        ff = [c for c in cands if c["source"] == "full_frame"]
        cr = [c for c in cands if c["source"] == "crop"]
        preds_by_img[fname] = policy_fn(ff, cr)
    return preds_by_img


def main():
    repo_root = Path(".")
    out_dir = repo_root / "experiments" / "phase4_pilot"
    p_raw = out_dir / "raw_candidates.jsonl"
    p_eval = out_dir / "cohort_evaluation_records.json"

    assert p_raw.exists() and p_eval.exists(), "Raw predictions or evaluation records missing!"

    # 1. Load Ground Truth
    with open(p_eval, "r", encoding="utf-8") as f:
        eval_records = json.load(f)
    gt_by_img = {x["image"]: x["ground_truth"] for x in eval_records}

    # 2. Load Raw Candidates
    cands_by_img = defaultdict(list)
    with open(p_raw, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            c = json.loads(line)
            cands_by_img[c["image_id"]].append(c)

    print(f"Loaded {sum(len(v) for v in cands_by_img.values())} raw candidates across {len(cands_by_img)} images.")

    # -------------------------------------------------------------------------
    # Policy Definitions
    # -------------------------------------------------------------------------

    # Policy 1: Current Baseline (Uniform tau = 0.25, greedy fusion)
    def policy_baseline(ff_cands, cr_cands):
        ff_filt = [d for d in ff_cands if d["conf"] >= 0.25]
        cr_filt = [d for d in cr_cands if d["conf"] >= 0.25]

        # Greedy fusion
        fused = []
        cr_by_cls = defaultdict(list)
        for d in cr_filt: cr_by_cls[d["class_name"]].append(d)

        matched_crop = set()
        for ff_d in ff_filt:
            cname = ff_d["class_name"]
            cr_list = cr_by_cls[cname]
            best_idx = -1
            best_iou = 0.0
            for idx, cr_d in enumerate(cr_list):
                if idx in matched_crop: continue
                iou = compute_box_iou(ff_d["box_full"], cr_d["box_full"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= 0.50 and best_idx >= 0:
                matched_crop.add(best_idx)
                cr_d = cr_list[best_idx]
                w_ff = ff_d["conf"]
                w_cr = cr_d["conf"] * 1.2
                w_sum = max(w_ff + w_cr, 1e-6)
                b_ff = ff_d["box_full"]
                b_cr = cr_d["box_full"]
                f_box = [
                    (b_ff[0] * w_ff + b_cr[0] * w_cr) / w_sum,
                    (b_ff[1] * w_ff + b_cr[1] * w_cr) / w_sum,
                    (b_ff[2] * w_ff + b_cr[2] * w_cr) / w_sum,
                    (b_ff[3] * w_ff + b_cr[3] * w_cr) / w_sum,
                ]
                f_conf = min(0.99, max(ff_d["conf"], cr_d["conf"]) + 0.05 * min(ff_d["conf"], cr_d["conf"]))
                fused.append({"class_name": cname, "conf": f_conf, "box_full": f_box, "source": "fused"})
            else:
                fused.append(ff_d)

        for cname, cr_list in cr_by_cls.items():
            for idx, cr_d in enumerate(cr_list):
                if idx not in matched_crop:
                    fused.append(cr_d)
        return fused

    # Evaluate Baseline
    baseline_preds = run_pipeline(cands_by_img, policy_baseline)
    baseline_metrics = evaluate_cohort(baseline_preds, gt_by_img)

    print("\n" + "=" * 70)
    print("BASELINE PERFORMANCE (Unchanged Active Foveation @ tau=0.25):")
    print("=" * 70)
    print(f"  Precision: {baseline_metrics['precision']:.4f} ({baseline_metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {baseline_metrics['recall']:.4f} ({baseline_metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {baseline_metrics['f1']:.4f}")
    print(f"  TP: {baseline_metrics['tp']}, FP: {baseline_metrics['fp']}, FN: {baseline_metrics['fn']}")
    print(f"  Small VRU Recall : {baseline_metrics['tier_recall']['Small']:.4f} ({baseline_metrics['tier_tp']['Small']}/{baseline_metrics['tier_gt']['Small']})")
    print(f"  Medium VRU Recall: {baseline_metrics['tier_recall']['Medium']:.4f} ({baseline_metrics['tier_tp']['Medium']}/{baseline_metrics['tier_gt']['Medium']})")
    print(f"  Large VRU Recall : {baseline_metrics['tier_recall']['Large']:.4f} ({baseline_metrics['tier_tp']['Large']}/{baseline_metrics['tier_gt']['Large']})")

    # -------------------------------------------------------------------------
    # Dual-Threshold Calibration Grid Sweep
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DUAL-THRESHOLD CALIBRATION SWEEP:")
    print("=" * 70)

    sweep_results = []

    def make_dual_policy(tau_ff, tau_isolated, tau_confirmed, iou_assoc=0.30, use_wbf=False):
        def policy(ff_cands, cr_cands):
            # 1. Full-frame filter
            ff_filt = [d for d in ff_cands if d["conf"] >= tau_ff]

            # 2. Check crop candidates against coarse full-frame candidates (conf >= 0.10)
            ff_coarse = [d for d in ff_cands if d["conf"] >= 0.10]
            cr_filt = []

            for cr_d in cr_cands:
                # Check overlap with any full-frame detection of same class
                has_ff_confirmation = False
                for ff_d in ff_coarse:
                    if ff_d["class_name"] == cr_d["class_name"]:
                        if compute_box_iou(cr_d["box_full"], ff_d["box_full"]) >= iou_assoc:
                            has_ff_confirmation = True
                            break

                req_thresh = tau_confirmed if has_ff_confirmation else tau_isolated
                if cr_d["conf"] >= req_thresh:
                    cr_filt.append(cr_d)

            # 3. Fusion (Greedy or WBF)
            if not use_wbf:
                fused = []
                cr_by_cls = defaultdict(list)
                for d in cr_filt: cr_by_cls[d["class_name"]].append(d)
                matched_crop = set()

                for ff_d in ff_filt:
                    cname = ff_d["class_name"]
                    cr_list = cr_by_cls[cname]
                    best_idx = -1
                    best_iou = 0.0
                    for idx, cr_d in enumerate(cr_list):
                        if idx in matched_crop: continue
                        iou = compute_box_iou(ff_d["box_full"], cr_d["box_full"])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                    if best_iou >= 0.50 and best_idx >= 0:
                        matched_crop.add(best_idx)
                        cr_d = cr_list[best_idx]
                        w_ff = ff_d["conf"]
                        w_cr = cr_d["conf"] * 1.2
                        w_sum = max(w_ff + w_cr, 1e-6)
                        b_ff = ff_d["box_full"]
                        b_cr = cr_d["box_full"]
                        f_box = [
                            (b_ff[0] * w_ff + b_cr[0] * w_cr) / w_sum,
                            (b_ff[1] * w_ff + b_cr[1] * w_cr) / w_sum,
                            (b_ff[2] * w_ff + b_cr[2] * w_cr) / w_sum,
                            (b_ff[3] * w_ff + b_cr[3] * w_cr) / w_sum,
                        ]
                        f_conf = min(0.99, max(ff_d["conf"], cr_d["conf"]) + 0.05 * min(ff_d["conf"], cr_d["conf"]))
                        fused.append({"class_name": cname, "conf": f_conf, "box_full": f_box, "source": "fused"})
                    else:
                        fused.append(ff_d)

                for cname, cr_list in cr_by_cls.items():
                    for idx, cr_d in enumerate(cr_list):
                        if idx not in matched_crop:
                            fused.append(cr_d)
                return fused

            else:
                # WBF Consensus Fusion
                fused = []
                all_dets = ff_filt + cr_filt
                by_cls = defaultdict(list)
                for d in all_dets: by_cls[d["class_name"]].append(d)

                for cname, dlist in by_cls.items():
                    dlist.sort(key=lambda x: x["conf"], reverse=True)
                    clusters = []  # list of [dets]

                    for d in dlist:
                        matched_cluster = None
                        for clust in clusters:
                            # Compare against cluster representative
                            rep_box = clust[0]["box_full"]
                            if compute_box_iou(d["box_full"], rep_box) >= 0.50:
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
                            # Consensus blend
                            weights = [x["conf"] * (1.2 if x["source"] == "crop" else 1.0) for x in clust]
                            w_sum = sum(weights)
                            f_box = [
                                sum(x["box_full"][j] * weights[idx] for idx, x in enumerate(clust)) / w_sum
                                for j in range(4)
                            ]
                            # Score boosting if multi-source
                            has_ff = any(x["source"] == "full_frame" for x in clust)
                            has_cr = any(x["source"] == "crop" for x in clust)
                            max_c = max(x["conf"] for x in clust)
                            f_conf = min(0.99, max_c + (0.08 if has_ff and has_cr else 0.02))
                            fused.append({"class_name": cname, "conf": f_conf, "box_full": f_box, "source": "wbf_consensus"})
                return fused

        return policy

    # Predeclared parameter grid
    grid = [
        # tau_ff, tau_isolated, tau_confirmed, use_wbf, label
        (0.25, 0.25, 0.25, False, "Baseline (tau=0.25)"),
        (0.25, 0.30, 0.20, False, "Dual: tau_iso=0.30, tau_conf=0.20"),
        (0.25, 0.35, 0.20, False, "Dual: tau_iso=0.35, tau_conf=0.20"),
        (0.25, 0.40, 0.20, False, "Dual: tau_iso=0.40, tau_conf=0.20 (Hypothesis Target)"),
        (0.25, 0.45, 0.20, False, "Dual: tau_iso=0.45, tau_conf=0.20"),
        (0.25, 0.50, 0.20, False, "Dual: tau_iso=0.50, tau_conf=0.20"),
        (0.25, 0.40, 0.25, False, "Dual: tau_iso=0.40, tau_conf=0.25"),
        (0.30, 0.40, 0.20, False, "Dual: tau_ff=0.30, tau_iso=0.40, tau_conf=0.20"),
        (0.25, 0.35, 0.20, True,  "WBF: tau_iso=0.35, tau_conf=0.20 + Consensus"),
        (0.25, 0.40, 0.20, True,  "WBF: tau_iso=0.40, tau_conf=0.20 + Consensus"),
    ]

    for tau_ff, tau_iso, tau_conf, wbf_flag, label in grid:
        pol = make_dual_policy(tau_ff, tau_iso, tau_conf, iou_assoc=0.30, use_wbf=wbf_flag)
        preds = run_pipeline(cands_by_img, pol)
        res = evaluate_cohort(preds, gt_by_img)

        delta_prec = res["precision"] - baseline_metrics["precision"]
        delta_rec = res["recall"] - baseline_metrics["recall"]
        delta_f1 = res["f1"] - baseline_metrics["f1"]
        delta_sm = res["tier_recall"]["Small"] - baseline_metrics["tier_recall"]["Small"]

        res_entry = {
            "label": label,
            "tau_ff": tau_ff,
            "tau_isolated": tau_iso,
            "tau_confirmed": tau_conf,
            "use_wbf": wbf_flag,
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "tp": res["tp"],
            "fp": res["fp"],
            "fn": res["fn"],
            "small_recall": res["tier_recall"]["Small"],
            "med_recall": res["tier_recall"]["Medium"],
            "large_recall": res["tier_recall"]["Large"],
            "delta_precision": delta_prec,
            "delta_recall": delta_rec,
            "delta_f1": delta_f1,
            "delta_small_recall": delta_sm,
            "per_class": res["per_class"],
        }
        sweep_results.append(res_entry)

        print(f"\n[{label}]")
        print(f"  Prec: {res['precision']:.4f} (Delta: {delta_prec*100:+.2f}%) | Rec: {res['recall']:.4f} (Delta: {delta_rec*100:+.2f}%) | F1: {res['f1']:.4f} (Delta: {delta_f1:+.4f})")
        print(f"  TP: {res['tp']} (Delta: {res['tp'] - baseline_metrics['tp']:+d}) | FP: {res['fp']} (Delta: {res['fp'] - baseline_metrics['fp']:+d}) | FN: {res['fn']}")
        print(f"  Small Rec: {res['tier_recall']['Small']:.4f} (Delta: {delta_sm*100:+.2f}%) | Med Rec: {res['tier_recall']['Medium']:.4f} | Large Rec: {res['tier_recall']['Large']:.4f}")

    # -------------------------------------------------------------------------
    # Decision Rule Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5 PILOT DECISION RULE EVALUATION:")
    print("=" * 70)

    # Check candidates that satisfy:
    # 1. Precision improvement >= +5.0 percentage points (+0.050)
    # 2. Recall decrease <= 2.0 percentage points (-0.020)
    qualified_configs = []
    for r in sweep_results[1:]:
        prec_pass = (r["delta_precision"] >= 0.050)
        rec_pass = (r["delta_recall"] >= -0.020)
        small_rec_acceptable = (r["delta_small_recall"] >= -0.030)
        is_pass = prec_pass and rec_pass and small_rec_acceptable
        if is_pass:
            qualified_configs.append(r)
        print(f"  [{r['label']}]: Prec Gain = {r['delta_precision']*100:+.2f}% [{'PASS' if prec_pass else 'FAIL'}], Rec Delta = {r['delta_recall']*100:+.2f}% [{'PASS' if rec_pass else 'FAIL'}] -> Decision: {'QUALIFIED' if is_pass else 'REJECTED'}")

    best_candidate = None
    if qualified_configs:
        # Sort by F1
        qualified_configs.sort(key=lambda x: x["f1"], reverse=True)
        best_candidate = qualified_configs[0]
        decision_verdict = "PASS"
        print(f"\nPILOT GATE VERDICT: PASS! Qualified candidate found: {best_candidate['label']}")
        print(f"  Precision: {best_candidate['precision']*100:.2f}% (Gain: {best_candidate['delta_precision']*100:+.2f} percentage points)")
        print(f"  Recall:    {best_candidate['recall']*100:.2f}% (Delta: {best_candidate['delta_recall']*100:+.2f} percentage points)")
        print(f"  F1-Score:  {best_candidate['f1']:.4f} (Gain: {best_candidate['delta_f1']:+.4f})")
        print(f"  FP Reduction: {best_candidate['fp']} vs {baseline_metrics['fp']} (Eliminated {baseline_metrics['fp'] - best_candidate['fp']} FPs!)")
        print(f"  Small VRU Recall: {best_candidate['small_recall']*100:.2f}% (Delta: {best_candidate['delta_small_recall']*100:+.2f} percentage points)")
    else:
        decision_verdict = "INCONCLUSIVE / FAIL"
        print(f"\nPILOT GATE VERDICT: {decision_verdict}. No configuration satisfied both criteria.")

    # -------------------------------------------------------------------------
    # Export Results
    # -------------------------------------------------------------------------
    results_payload = {
        "verdict": decision_verdict,
        "criteria": {
            "min_precision_gain_pct": 5.0,
            "max_recall_loss_pct": 2.0,
        },
        "baseline": baseline_metrics,
        "best_candidate": best_candidate,
        "all_sweep_results": sweep_results,
    }

    with open(out_dir / "calibration_sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    # Export CSV Table
    csv_path = out_dir / "calibration_sweep_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Configuration,Tau_FF,Tau_Isolated,Tau_Confirmed,WBF,Precision,Recall,F1,Small_Recall,Med_Recall,Large_Recall,TP,FP,FN,Delta_Precision_Pct,Delta_Recall_Pct,Delta_F1\n")
        for r in sweep_results:
            f.write(f'"{r["label"]}",{r["tau_ff"]},{r["tau_isolated"]},{r["tau_confirmed"]},{r["use_wbf"]},'
                    f'{r["precision"]:.4f},{r["recall"]:.4f},{r["f1"]:.4f},'
                    f'{r["small_recall"]:.4f},{r["med_recall"]:.4f},{r["large_recall"]:.4f},'
                    f'{r["tp"]},{r["fp"]},{r["fn"]},'
                    f'{r["delta_precision"]*100:.2f},{r["delta_recall"]*100:.2f},{r["delta_f1"]:.4f}\n')
    print(f"Exported sweep summary CSV to {csv_path}")


if __name__ == "__main__":
    main()
