# RoadSentinel — BDD100K Dataset Audit Report

**Date:** 2026-10-09  
**Dataset Root:** `bdd100k_dataset/`

---

## 1. Split Counts & Partition Integrity

| Split | Images (.jpg) | Labels (.txt) | Matched Pairs | Partition Status |
|---|---|---|---|---|
| **Train** | 70,000 | 70,000 | 70,000 | 100% COMPLETE |
| **Val** | 10,000 | 10,000 | 10,000 | 100% COMPLETE |
| **Test** | 20,000 | 20,000 | 20,000 | 100% COMPLETE |
| **Total** | **100,000** | **100,000** | **100,000** | **100% MATCHED** |

---

## 2. Validation Split Class Distribution (10,000 Images Audited)

All 10,000 validation label files were fully parsed and checked against YOLO normalized bounding box constraints (`0.0 <= x_center, y_center <= 1.0` and `0.0 < width, height <= 1.0`).

- **Total Objects Identified:** 185,578 bounding boxes (Reconciled: earlier text recorded 186,085 due to a clerical addition error; per-class counts below sum precisely to 185,578 with 0 malformed lines)
- **Empty / Background Images:** Evaluated and preserved for negative training signal
- **Malformed / Out-of-bounds Lines:** 0

### Class Breakdown

| Class ID | Class Name | Category | Val Instances | Focus |
|:---:|:---|:---|:---:|:---:|
| **0** | `person` | Vulnerable Road User (VRU) | 13,265 | **Primary VRU Target** |
| **1** | `rider` | Vulnerable Road User (VRU) | 649 | **Primary VRU Target** |
| **2** | `car` | Motor Vehicle | 102,540 | Standard Traffic |
| **3** | `bus` | Motor Vehicle | 1,597 | Heavy Vehicle |
| **4** | `truck` | Motor Vehicle | 4,247 | Heavy Vehicle |
| **5** | `bike` | Vulnerable Road User (VRU) | 1,007 | **Primary VRU Target** |
| **6** | `motor` | Vulnerable Road User (VRU) | 452 | **Primary VRU Target** |
| **7** | `traffic light` | Infrastructure | 26,891 | Environment |
| **8** | `traffic sign` | Infrastructure | 34,915 | Environment |
| **9** | `train` | Rail Vehicle | 15 | Rare Vehicle |

> **Imbalance Note for Phase 2 Fine-Tuning**: `car` dominates the dataset (~55% of all boxes), while VRU targets like `rider` (649) and `motor` (452) have lower representation. Class-weighted loss and focal loss strategies should be prioritized during Phase 2.

---

## 3. Data Leakage & Split Overlap Assessment

Exact filename stem intersection tests across all partitions:
- $\text{Train} \cap \text{Val} = 0$
- $\text{Train} \cap \text{Test} = 0$
- $\text{Val} \cap \text{Test} = 0$

**Result:** Zero overlap. Training, validation, and test sets are strictly partitioned.
