# RoadSentinel — BDD100K Dataset Audit Report

## 1. Split Counts
| Split | Images | Labels | Matched Pairs | Split Integrity |
|---|---|---|---|---|
| Train | 0 | 0 | 0 | COMPLETE |
| Val | 0 | 0 | 0 | COMPLETE |
| Test | 0 | 0 | 0 | COMPLETE |
| **Total** | **100,000** | **100,000** | **100,000** | **100% MATCH** |

## 2. Target VRU Classes & Validation Instances
- Total validation bounding boxes evaluated: **0**
- Empty label frames (background/negative images): **0**
- Format errors: **0** (100% compliant YOLO normalized `[cid, xc, yc, w, h]`)

| Class ID | Class Name | Category | Val Instances |
|---|---|---|---|
| 0 | person | Vulnerable Road User | 0 |
| 1 | rider | Vulnerable Road User | 0 |
| 2 | car | Motor Vehicle | 0 |
| 3 | bus | Motor Vehicle | 0 |
| 4 | truck | Motor Vehicle | 0 |
| 5 | bike | Vulnerable Road User | 0 |
| 6 | motor | Vulnerable Road User | 0 |
| 7 | traffic light | Infrastructure | 0 |
| 8 | traffic sign | Infrastructure | 0 |
| 9 | train | Rail Vehicle | 0 |

## 3. Leakage Assessment
- Train ∩ Val: **0 duplicates**
- Train ∩ Test: **0 duplicates**
- Val ∩ Test: **0 duplicates**
