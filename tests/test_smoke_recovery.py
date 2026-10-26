import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np
import yaml
from pathlib import Path

# Ensure OpenMP runtime doesn't error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
from scripts.matching_engine import match_single_image, aggregate_partition_from_image_stats

CLASS_NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
VRU_NAMES = {"person", "rider", "bike", "motor"}
VRU_IDS = [0, 1, 5, 6]

class TestSmokeRecovery(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="smoke_eval_")
        self.work_dir = Path(self.temp_dir)
        self.ckpt_path = Path("scratch/kaggle_upload_weights/yolo11m_phase6_epoch26.pt")
        self.assertTrue(self.ckpt_path.exists(), f"Checkpoint not found at {self.ckpt_path}")

        # Use 4 sample images from val
        self.val_img_dir = Path("bdd100k_dataset/val/images")
        self.val_lbl_dir = Path("bdd100k_dataset/val/labels")
        self.sample_names = sorted(os.listdir(self.val_img_dir))[:4]
        
        # Create a tiny mock val dataset folder for model.val()
        self.mock_dataset_dir = self.work_dir / "mock_dataset"
        self.mock_img_dir = self.mock_dataset_dir / "val" / "images"
        self.mock_lbl_dir = self.mock_dataset_dir / "val" / "labels"
        self.mock_img_dir.mkdir(parents=True, exist_ok=True)
        self.mock_lbl_dir.mkdir(parents=True, exist_ok=True)

        for name in self.sample_names:
            shutil.copy(self.val_img_dir / name, self.mock_img_dir / name)
            lbl_name = Path(name).stem + ".txt"
            if (self.val_lbl_dir / lbl_name).exists():
                shutil.copy(self.val_lbl_dir / lbl_name, self.mock_lbl_dir / lbl_name)

        # Create mock data.yaml
        val_data = {
            "path": str(self.mock_dataset_dir),
            "train": str(self.mock_img_dir),
            "val": str(self.mock_img_dir),
            "names": dict(enumerate(CLASS_NAMES))
        }
        self.yaml_path = self.work_dir / "mock_data.yaml"
        self.yaml_path.write_text(yaml.dump(val_data))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pass1_immediate_persistence_and_checkpoint_recovery(self):
        """Verify Pass 1 immediate persistence and recoverable progress checkpoints."""
        model = YOLO(str(self.ckpt_path))

        # --- Phase 1: Pass 1 execution and immediate persistence ---
        val_res = model.val(
            data=str(self.yaml_path),
            imgsz=320,  # fast imgsz for smoke test
            batch=2,
            device="cpu",
            plots=False,
            project=str(self.work_dir / "runs"),
            name="smoke_val",
            verbose=False,
        )

        ap50 = val_res.box.all_ap[:, 0]
        ap50_95 = val_res.box.all_ap.mean(1)
        classes = list(val_res.box.ap_class_index)
        ultralytics_per_class = {
            CLASS_NAMES[c]: {
                "AP50": round(float(ap50[i]) * 100.0, 2),
                "AP50-95": round(float(ap50_95[i]) * 100.0, 2)
            } for i, c in enumerate(classes)
        }
        vru_indices = [classes.index(c) for c in VRU_IDS if c in classes]
        vru_macro_ap50 = round(float(ap50[vru_indices].mean()) * 100.0, 2) if vru_indices else 0.0

        pass1_file = self.work_dir / "pass1_official_map.json"
        pass1_payload = {
            "milestone": "Milestone 1 — Pass 1 Immediate Recovery Checkpoint",
            "benchmark_metadata": {
                "checkpoint_sha256": "42d4b385ec69d01209e76dcf85f0fb4ffb33857d85f4daa02a4271d20f76541e",
                "imgsz": 320,
            },
            "ultralytics_official_map": {
                "all_class_mAP50": round(float(val_res.box.map50) * 100.0, 2),
                "vru_macro_mAP50": vru_macro_ap50,
                "per_class_ap": ultralytics_per_class,
            }
        }
        # Write Pass 1 immediately
        with open(pass1_file, "w", encoding="utf-8") as f:
            json.dump(pass1_payload, f, indent=2)

        # Assert Pass 1 file exists and is readable immediately
        self.assertTrue(pass1_file.exists())
        with open(pass1_file, "r", encoding="utf-8") as f:
            loaded_p1 = json.load(f)
        self.assertIn("all_class_mAP50", loaded_p1["ultralytics_official_map"])
        print(f"Pass 1 persisted immediately: mAP50 = {loaded_p1['ultralytics_official_map']['all_class_mAP50']}")

        # --- Phase 2: Progress Checkpoint and Simulated Interruption ---
        # Parse ground truth for the 4 images
        gt_by_img = {}
        for name in self.sample_names:
            gt_by_img[name] = []
            lbl_file = self.mock_lbl_dir / (Path(name).stem + ".txt")
            if lbl_file.exists():
                for line in lbl_file.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    # normalized xywh to absolute (assume 1280x720)
                    xc, yc, w, h = map(float, parts[1:5])
                    x1 = (xc - w/2) * 1280
                    y1 = (yc - h/2) * 720
                    x2 = (xc + w/2) * 1280
                    y2 = (yc + h/2) * 720
                    area = (x2 - x1) * (y2 - y1)
                    sz = "Small (<32²)" if area < 1024 else ("Medium [32²-96²)" if area < 9216 else "Large (≥96²)")
                    gt_by_img[name].append({
                        "class_name": CLASS_NAMES[cid],
                        "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                        "size_group": sz,
                        "is_vru": CLASS_NAMES[cid] in VRU_NAMES
                    })

        # Simulate processing first 2 images, then save checkpoint
        ckpt_stats_file = self.work_dir / "checkpoint_image_stats.json"
        first_half = self.sample_names[:2]
        image_stats = {}
        for name in first_half:
            res = model.predict(str(self.mock_img_dir / name), imgsz=320, conf=0.25, iou=0.65, device="cpu", verbose=False)[0]
            preds = []
            for b in res.boxes:
                cid = int(b.cls.item())
                preds.append({
                    "class_name": model.names[cid],
                    "conf": float(b.conf.item()),
                    "box": b.xyxy[0].cpu().numpy()
                })
            image_stats[name] = match_single_image(preds, gt_by_img[name], CLASS_NAMES, VRU_NAMES)

        # Save checkpoint simulating interruption
        with open(ckpt_stats_file, "w", encoding="utf-8") as f:
            json.dump(image_stats, f)
        self.assertTrue(ckpt_stats_file.exists())
        self.assertEqual(len(image_stats), 2)

        # --- Phase 3: Recovery from checkpoint ---
        # A new process/run starts:
        recovered_stats = {}
        with open(ckpt_stats_file, "r", encoding="utf-8") as f:
            recovered_stats = json.load(f)
        self.assertEqual(len(recovered_stats), 2)

        # Remaining images to process
        remaining = [img for img in self.sample_names if img not in recovered_stats]
        self.assertEqual(len(remaining), 2)
        self.assertEqual(remaining, self.sample_names[2:])

        for name in remaining:
            res = model.predict(str(self.mock_img_dir / name), imgsz=320, conf=0.25, iou=0.65, device="cpu", verbose=False)[0]
            preds = []
            for b in res.boxes:
                cid = int(b.cls.item())
                preds.append({
                    "class_name": model.names[cid],
                    "conf": float(b.conf.item()),
                    "box": b.xyxy[0].cpu().numpy()
                })
            recovered_stats[name] = match_single_image(preds, gt_by_img[name], CLASS_NAMES, VRU_NAMES)

        # All 4 images completed
        self.assertEqual(len(recovered_stats), 4)

        # Aggregate across partition
        agg = aggregate_partition_from_image_stats(recovered_stats, set(self.sample_names), CLASS_NAMES, VRU_NAMES)
        self.assertEqual(agg["image_count"], 4)
        self.assertIn("vru_metrics", agg)
        self.assertIn("recall", agg["vru_metrics"])
        print(f"Recovered and aggregated: total_gt = {agg['total_gt_objects']}, VRU Recall = {agg['vru_metrics']['recall']}%")

if __name__ == "__main__":
    unittest.main()
