# RoadSentinel Phase 10F — False-Positive and NMS Duplicate Suppression Audit

## 1. Executive Summary

This audit examines the empirical behavior of **Class-Specific NMS Duplicate Suppression ($\text{IoU}_{\text{VRU}} = 0.55$)** introduced in Phase 10E. We evaluate whether the observed precision recovery ($+1.72\text{ pp}$ on Cohort 4, $+4.80\text{ pp}$ relative to 0.20 on Cohort 5) is driven by genuine duplicate suppression or if it inflicts unacceptable collateral suppression on legitimate adjacent pedestrians.

No thresholds, NMS parameters, or weights were altered during this audit.

---

## 2. Granular Accounting of NMS Suppression Effects

### A. Stage B Controlled Ablation on Cohort 4 (1,000 Images)
Evaluated with fixed VRU confidence threshold $\tau = 0.22$:

| NMS IoU Setting | VRU TP | VRU FP | Precision | F1 Score | TPs Lost vs 0.65 | FPs Eliminated vs 0.65 | Elimination-to-Loss Ratio |
|---|---|---|---|---|---|---|---|
| **0.65 (Standard Baseline)** | 1,085 | 417 | 72.24% | 0.6962 | 0 | 0 | Baseline |
| **0.60 (Moderate Tightening)** | 1,083 | 393 | 73.37% | 0.7007 | **-2** | **+24** | **12.0 : 1** |
| **0.55 (Selected Setting)** | 1,082 | 381 | 73.96% | 0.7031 | **-3** | **+36** | **12.0 : 1** |

### B. Impact on Small-VRU True Positives
- At $\text{IoU} = 0.65$: Small-VRU Recall was $48.54\%$ ($316 / 651$ TPs).
- At $\text{IoU} = 0.55$: Small-VRU Recall was $48.39\%$ ($315 / 651$ TPs).
- **Collateral Loss:** Only **1 small VRU true positive** was suppressed across 1,000 images, while **36 duplicate false-positive boxes** were eliminated.

---

## 3. Physical Error Mode Classification

Auditing individual detections altered by tightening $\text{IoU} \in [0.55, 0.65)$ reveals:

### 1. Duplicate Detections Suppressed (36 instances on Cohort 4)
- **Mechanism:** In the confidence range $[0.22, 0.35)$, the detector frequently predicts secondary, jittered bounding boxes around the torso or legs of a primary pedestrian already detected with high confidence ($\text{conf} \ge 0.40$).
- Because their overlap with the primary box falls between $0.55$ and $0.65$, standard NMS ($\text{IoU}=0.65$) allowed both boxes to survive.
- Tightening NMS to $0.55$ successfully suppressed the lower-confidence secondary phantom, directly eliminating 36 false alarms.

### 2. Legitimate Nearby VRUs Suppressed (3 instances on Cohort 4)
- **Mechanism:** In dense urban scenes, pedestrians walking shoulder-to-shoulder or carrying children can physically overlap with $\text{IoU} \in [0.55, 0.62]$.
- In these 3 specific instances, a lower-confidence pedestrian ($\text{conf} \approx 0.23$) walking immediately adjacent to a higher-confidence pedestrian ($\text{conf} \approx 0.65$) was suppressed by greedy NMS.
- **Risk Assessment:** While regrettable, losing 3 true positives to eliminate 36 false positives is a **12:1 favorable ratio**, which is essential for ensuring that the detector does not trigger phantom emergency braking events on autonomous vehicle actuation systems.

### 3. Newly Admitted Detections Due to Lower Threshold ($\tau = 0.22$)
- On Cohort 5 ($1,000$ images), lowering threshold from $0.25$ to $0.22$ admitted **$+32$ net True Positives** ($827 \to 859$) while adding only **$+17$ net False Positives** ($239 \to 256$).
- The net false-alarm increase was constrained to just **$+0.017\text{ FP per image}$** (1 additional false alarm every 59 frames).

---

## 4. Visual Evidence & Overlay Review

Inspection of comparative visual overlays in `experiments/phase10_model_optimization/phase10e_policy_confirmation/visual_examples/`:
- **Cohort 4 Overlays (`cohort4_confirm_*.jpg`):** Shows clean suppression of secondary jitter boxes on pedestrians waiting at traffic lights, without clipping bicycles or adjacent riders.
- **Cohort 5 Overlays (`cohort5_untouched_*.jpg`):** Confirms distant sidewalk pedestrians at confidences $0.22 - 0.24$ cleanly recovered and drawn in magenta, with zero secondary ghost boxes around vehicles or traffic signs.
