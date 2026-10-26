# RoadSentinel — Model Card: YOLO11n Baseline Checkpoint

**Date:** 2026-10-09  
**Model Identifier:** `yolo11n.pt`  
**Framework:** Ultralytics (YOLO11)

---

## 1. Architecture & Checkpoint Overview

- **Model Type:** Single-Stage Real-Time Object Detection (YOLO11 Nano)
- **Parameters:** ~2.6M parameters
- **Format:** PyTorch serialized checkpoint (`.pt`)
- **Default Resolution:** 640x640 (dynamic input resolution supported)
- **Weights Source:** Official Ultralytics GitHub release asset (`v8.4.0`)
- **License:** AGPL-3.0

---

## 2. Pretrained Classes & Distribution

- **Training Distribution:** MS COCO 2017 (80 classes)
- **Relevant Road/VRU Classes Supported in COCO:**
  - Class 0: `person`
  - Class 1: `bicycle`
  - Class 2: `car`
  - Class 3: `motorcycle`
  - Class 5: `bus`
  - Class 7: `truck`
  - Class 9: `traffic light`
  - Class 11: `stop sign`

---

## 3. RoadSentinel Role & Class Mapping Decision

In MS COCO:
- BDD100K `rider` (distinct class for person riding bicycle/motorcycle) does **not** exist in standard COCO (COCO detects the human as `person` and the vehicle as `bicycle`/`motorcycle`).
- BDD100K defines 10 traffic-specific classes with dedicated IDs `[0: person, 1: rider, 2: car, 3: bus, 4: truck, 5: bike, 6: motor, 7: traffic light, 8: traffic sign, 9: train]`.

**Phase 2 Decision**:
We will use `yolo11n.pt` backbone weights for transfer learning, replacing the detection head with the 10-class BDD100K schema (`bdd100k_dataset/data.yaml`) to formally train and benchmark RoadSentinel.
