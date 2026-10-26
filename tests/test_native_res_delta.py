import unittest
import numpy as np
from scripts.matching_engine import match_single_image, aggregate_partition_from_image_stats

CLASS_NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
VRU_NAMES = {"person", "rider", "bike", "motor"}

class TestNativeResDeltaLogic(unittest.TestCase):
    def test_delta_computation_consistency(self):
        # Baseline 960 reference
        b960_vru_recall = 63.31
        b960_small_vru_recall = 44.47

        # Synthetic native resolution result with small object recovery
        native_stats = {
            "img1": {
                "n_gt": 10,
                "n_pred": 12,
                "class_stats": {c: {"tp": 1, "fp": 1, "fn": 0} for c in CLASS_NAMES},
                "vru_size_stats": {
                    "Small (<32²)": {"tp": 3, "fn": 1, "total": 4},
                    "Medium [32²-96²)": {"tp": 4, "fn": 0, "total": 4},
                    "Large (≥96²)": {"tp": 2, "fn": 0, "total": 2},
                }
            }
        }
        res = aggregate_partition_from_image_stats(native_stats, {"img1"}, CLASS_NAMES, VRU_NAMES)
        
        # Verify metric calculation
        self.assertEqual(res["vru_metrics"]["small_vru_ground_truth"], 4)
        self.assertEqual(res["vru_metrics"]["small_vru_true_positives"], 3)
        self.assertEqual(res["vru_metrics"]["small_vru_recall"], 75.0)

        delta_small_recall = round(res["vru_metrics"]["small_vru_recall"] - b960_small_vru_recall, 2)
        self.assertEqual(delta_small_recall, 30.53)

if __name__ == "__main__":
    unittest.main()
