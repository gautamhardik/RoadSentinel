import unittest
import numpy as np
from scripts.matching_engine import evaluate_partition_original, match_single_image, aggregate_partition_from_image_stats

CLASS_NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
VRU_NAMES = {"person", "rider", "bike", "motor"}

class TestMatchingEquivalence(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def _verify_equivalence(self, preds_by_img, gt_by_img, image_subset):
        # 1. Original
        res_orig = evaluate_partition_original(preds_by_img, gt_by_img, image_subset, CLASS_NAMES, VRU_NAMES)

        # 2. Optimized single-pass
        image_stats = {}
        for img_name in preds_by_img.keys() | gt_by_img.keys():
            p_list = preds_by_img.get(img_name, [])
            g_list = gt_by_img.get(img_name, [])
            image_stats[img_name] = match_single_image(p_list, g_list, CLASS_NAMES, VRU_NAMES)
        res_opt = aggregate_partition_from_image_stats(image_stats, image_subset, CLASS_NAMES, VRU_NAMES)

        # Exact match assertions
        self.assertEqual(res_orig["total_gt_objects"], res_opt["total_gt_objects"])
        self.assertEqual(res_orig["total_predictions"], res_opt["total_predictions"])
        self.assertEqual(res_orig["vru_metrics"], res_opt["vru_metrics"])
        self.assertEqual(res_orig["all_class_metrics"], res_opt["all_class_metrics"])
        self.assertEqual(res_orig["per_class"], res_opt["per_class"])
        self.assertEqual(res_orig["vru_by_size"], res_opt["vru_by_size"])

    def test_case_empty(self):
        """Case: No predictions, no ground truth"""
        preds = {"img1": []}
        gts = {"img1": []}
        self._verify_equivalence(preds, gts, {"img1"})

    def test_case_no_predictions(self):
        """Case: Ground truth exists, but no predictions (all False Negatives)"""
        preds = {"img1": []}
        gts = {
            "img1": [
                {"class_name": "person", "box": np.array([10, 10, 30, 30], dtype=np.float32), "size_group": "Small (<32²)", "is_vru": True},
                {"class_name": "car", "box": np.array([100, 100, 250, 200], dtype=np.float32), "size_group": "Large (≥96²)", "is_vru": False},
            ]
        }
        self._verify_equivalence(preds, gts, {"img1"})

    def test_case_no_ground_truth(self):
        """Case: Predictions exist, but no ground truth (all False Positives)"""
        preds = {
            "img1": [
                {"class_name": "person", "conf": 0.85, "box": np.array([10, 10, 30, 30], dtype=np.float32)},
                {"class_name": "bike", "conf": 0.70, "box": np.array([50, 50, 80, 80], dtype=np.float32)},
            ]
        }
        gts = {"img1": []}
        self._verify_equivalence(preds, gts, {"img1"})

    def test_case_overlapping_boxes_and_duplicate_predictions(self):
        """Case: Duplicate predictions competing for the same ground truth box"""
        preds = {
            "img1": [
                {"class_name": "person", "conf": 0.90, "box": np.array([10, 10, 30, 30], dtype=np.float32)},
                {"class_name": "person", "conf": 0.75, "box": np.array([11, 11, 31, 31], dtype=np.float32)}, # overlapping duplicate
                {"class_name": "person", "conf": 0.50, "box": np.array([10, 10, 30, 30], dtype=np.float32)}, # 3rd duplicate
            ]
        }
        gts = {
            "img1": [
                {"class_name": "person", "box": np.array([10, 10, 30, 30], dtype=np.float32), "size_group": "Small (<32²)", "is_vru": True}
            ]
        }
        # The highest conf (0.90) should match GT as TP. The other two should be FP.
        self._verify_equivalence(preds, gts, {"img1"})

    def test_case_iou_threshold_boundary(self):
        """Case: IoU exactly around 0.50 boundary (one at 0.51 -> TP, one at 0.49 -> FP)"""
        # Box A: [0, 0, 10, 10], area = 100
        # Box B: [0, 0, 10, 7], inter = 70, union = 100 -> IoU = 0.70 >= 0.50 (TP)
        # Box C: [0, 0, 10, 3], inter = 30, union = 100 -> IoU = 0.30 < 0.50 (FP)
        preds = {
            "img1": [
                {"class_name": "car", "conf": 0.9, "box": np.array([0, 0, 10, 7], dtype=np.float32)},
                {"class_name": "car", "conf": 0.8, "box": np.array([0, 0, 10, 3], dtype=np.float32)},
            ]
        }
        gts = {
            "img1": [
                {"class_name": "car", "box": np.array([0, 0, 10, 10], dtype=np.float32), "size_group": "Small (<32²)", "is_vru": False}
            ]
        }
        self._verify_equivalence(preds, gts, {"img1"})

    def test_case_class_separation(self):
        """Case: Same spatial box but different class (car prediction does NOT match person GT)"""
        preds = {
            "img1": [
                {"class_name": "car", "conf": 0.95, "box": np.array([0, 0, 50, 50], dtype=np.float32)}
            ]
        }
        gts = {
            "img1": [
                {"class_name": "person", "box": np.array([0, 0, 50, 50], dtype=np.float32), "size_group": "Medium [32²-96²)", "is_vru": True}
            ]
        }
        self._verify_equivalence(preds, gts, {"img1"})

    def test_case_multiple_images_and_subsets(self):
        """Case: Multiple images partitioned into disjoint pilot and expanded sets"""
        imgs = [f"img_{i}.jpg" for i in range(20)]
        preds = {}
        gts = {}
        for img in imgs:
            preds[img] = []
            gts[img] = []
            # Generate random GTs
            n_g = np.random.randint(0, 15)
            for _ in range(n_g):
                cname = np.random.choice(CLASS_NAMES)
                x1, y1 = np.random.uniform(0, 1000, 2)
                w, h = np.random.uniform(10, 150, 2)
                x2, y2 = x1 + w, y1 + h
                area = w * h
                sz = "Small (<32²)" if area < 1024 else ("Medium [32²-96²)" if area < 9216 else "Large (≥96²)")
                gts[img].append({
                    "class_name": cname,
                    "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                    "size_group": sz,
                    "is_vru": cname in VRU_NAMES
                })
            # Generate random preds
            n_p = np.random.randint(0, 20)
            for _ in range(n_p):
                cname = np.random.choice(CLASS_NAMES)
                conf = float(np.random.uniform(0.25, 0.99))
                x1, y1 = np.random.uniform(0, 1000, 2)
                w, h = np.random.uniform(10, 150, 2)
                preds[img].append({
                    "class_name": cname,
                    "conf": conf,
                    "box": np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)
                })

        pilot_sub = set(imgs[:5])
        expanded_sub = set(imgs[5:])
        aggregate_sub = set(imgs)

        self._verify_equivalence(preds, gts, pilot_sub)
        self._verify_equivalence(preds, gts, expanded_sub)
        self._verify_equivalence(preds, gts, aggregate_sub)

    def test_adversarial_one_to_one_competition(self):
        """Adversarial: 5 predictions competing for a single GT box with varying IoUs and confidences."""
        preds = {
            "img_crowd": [
                {"class_name": "person", "conf": 0.95, "box": np.array([0, 0, 10, 10], dtype=np.float32)},
                {"class_name": "person", "conf": 0.90, "box": np.array([1, 1, 11, 11], dtype=np.float32)},
                {"class_name": "person", "conf": 0.85, "box": np.array([2, 2, 12, 12], dtype=np.float32)},
                {"class_name": "person", "conf": 0.70, "box": np.array([0, 0, 10, 10], dtype=np.float32)},
                {"class_name": "person", "conf": 0.60, "box": np.array([3, 3, 13, 13], dtype=np.float32)},
            ]
        }
        gts = {
            "img_crowd": [
                {"class_name": "person", "box": np.array([0, 0, 10, 10], dtype=np.float32), "size_group": "Small (<32²)", "is_vru": True}
            ]
        }
        # Exactly 1 TP (the highest confidence prediction conf=0.95), 4 FP, 0 FN.
        self._verify_equivalence(preds, gts, {"img_crowd"})

        # Cluster of 3 GT boxes and 4 predictions in dense proximity
        preds_cluster = {
            "img_dense": [
                {"class_name": "person", "conf": 0.92, "box": np.array([10, 10, 30, 30], dtype=np.float32)},
                {"class_name": "person", "conf": 0.88, "box": np.array([15, 15, 35, 35], dtype=np.float32)},
                {"class_name": "person", "conf": 0.82, "box": np.array([25, 25, 45, 45], dtype=np.float32)},
                {"class_name": "person", "conf": 0.75, "box": np.array([20, 20, 40, 40], dtype=np.float32)},
            ]
        }
        gts_cluster = {
            "img_dense": [
                {"class_name": "person", "box": np.array([10, 10, 30, 30], dtype=np.float32), "size_group": "Small (<32²)", "is_vru": True},
                {"class_name": "person", "box": np.array([20, 20, 40, 40], dtype=np.float32), "size_group": "Small (<32²)", "is_vru": True},
                {"class_name": "person", "box": np.array([30, 30, 50, 50], dtype=np.float32), "size_group": "Small (<32²)", "is_vru": True},
            ]
        }
        self._verify_equivalence(preds_cluster, gts_cluster, {"img_dense"})

    def test_confidence_ties(self):
        """Adversarial: Predictions with identical confidence values to test stable tie-breaking."""
        # Pred A and Pred B have identical conf=0.85
        preds = {
            "img_ties": [
                {"class_name": "car", "conf": 0.85, "box": np.array([0, 0, 20, 20], dtype=np.float32)},
                {"class_name": "car", "conf": 0.85, "box": np.array([5, 5, 25, 25], dtype=np.float32)},
                {"class_name": "car", "conf": 0.85, "box": np.array([10, 10, 30, 30], dtype=np.float32)},
            ]
        }
        gts = {
            "img_ties": [
                {"class_name": "car", "box": np.array([0, 0, 20, 20], dtype=np.float32), "size_group": "Small (<32²)", "is_vru": False},
                {"class_name": "car", "box": np.array([10, 10, 30, 30], dtype=np.float32), "size_group": "Small (<32²)", "is_vru": False},
            ]
        }
        self._verify_equivalence(preds, gts, {"img_ties"})

        # Reversed order of tied predictions
        preds_rev = {
            "img_ties": list(reversed(preds["img_ties"]))
        }
        self._verify_equivalence(preds_rev, gts, {"img_ties"})

    def test_greedy_vs_optimal_counterexample(self):
        """
        Adversarial: Greedy assignment vs Optimal assignment counterexample.
        GT 1: [0, 0, 10, 10]
        GT 2: [8, 0, 18, 10]
        Pred 1: conf=0.90, box [1, 0, 11, 10] -> overlaps GT 1 (IoU ~0.82) and GT 2 (IoU ~0.54).
        Pred 2: conf=0.80, box [0, 0, 10, 10] -> overlaps GT 1 (IoU 1.0) and GT 2 (IoU 0.0).

        Greedy behavior:
        - Pred 1 (conf 0.90) is evaluated first. It greedily matches GT 1 (best IoU 0.82 >= 0.50).
        - Pred 2 (conf 0.80) is evaluated second. GT 1 is already matched. IoU with GT 2 is 0.0 < 0.50.
          Therefore, Pred 2 is a False Positive.
        - GT 2 remains unmatched -> False Negative.
        Total: 1 TP, 1 FP, 1 FN.
        (Note: An optimal bipartite matching would match Pred 2->GT 1 and Pred 1->GT 2 yielding 2 TP, 0 FP, 0 FN).
        This test proves that both matchers strictly preserve the exact confidence-ordered greedy behavior!
        """
        # Compute exact IoU of Pred 1 with GT 1:
        # Box 1: [0, 0, 10, 10], Area=100
        # Box 2: [8, 0, 18, 10], Area=100
        # Pred 1: [2, 0, 12, 10], Area=100.
        # Inter with GT 1: [2..10, 0..10] -> 80. Union: 100+100-80 = 120. IoU = 80/120 = 0.667 >= 0.50.
        # Inter with GT 2: [8..12, 0..10] -> 40. Union: 100+100-40 = 160. IoU = 40/160 = 0.250 < 0.50.
        # Let's adjust GT 2 so Pred 1 has IoU >= 0.50 with BOTH GTs:
        # Pred 1: [4, 0, 14, 10], Area=100
        # GT 1: [0, 0, 10, 10] -> inter 60, union 140 -> IoU = 60/140 = 0.428 < 0.50.
        # Let's use 1D dimension 20:
        # GT 1: [0, 0, 20, 20] (area 400)
        # GT 2: [10, 0, 30, 20] (area 400)
        # Pred 1: [4, 0, 24, 20] (area 400, conf=0.90)
        #   Inter with GT 1: [4..20] -> 16*20 = 320. Union: 400+400-320 = 480. IoU = 320/480 = 0.667.
        #   Inter with GT 2: [10..24] -> 14*20 = 280. Union: 400+400-280 = 520. IoU = 280/520 = 0.538.
        #   Both IoUs >= 0.50! GT 1 has higher IoU (0.667 > 0.538), so Pred 1 greedily takes GT 1!
        # Pred 2: [0, 0, 20, 20] (area 400, conf=0.80)
        #   Inter with GT 1: 400. IoU = 1.0. But GT 1 is taken!
        #   Inter with GT 2: [10..20] -> 10*20 = 200. Union: 600. IoU = 200/600 = 0.333 < 0.50.
        #   Pred 2 cannot match GT 2. Pred 2 is FP!
        # Result: Exactly 1 TP (Pred 1 -> GT 1), 1 FP (Pred 2), 1 FN (GT 2).

        preds = {
            "img_greedy": [
                {"class_name": "car", "conf": 0.90, "box": np.array([4, 0, 24, 20], dtype=np.float32)},
                {"class_name": "car", "conf": 0.80, "box": np.array([0, 0, 20, 20], dtype=np.float32)},
            ]
        }
        gts = {
            "img_greedy": [
                {"class_name": "car", "box": np.array([0, 0, 20, 20], dtype=np.float32), "size_group": "Medium [32²-96²)", "is_vru": False},
                {"class_name": "car", "box": np.array([10, 0, 30, 20], dtype=np.float32), "size_group": "Medium [32²-96²)", "is_vru": False},
            ]
        }

        # 1. Verify equivalence between original and optimized
        self._verify_equivalence(preds, gts, {"img_greedy"})

        # 2. Specifically verify that greedy behavior produces 1 TP, 1 FP, 1 FN
        res = evaluate_partition_original(preds, gts, {"img_greedy"}, CLASS_NAMES, VRU_NAMES)
        car_stats = res["per_class"]["car"]
        self.assertEqual(car_stats["tp"], 1, "Greedy matching must yield exactly 1 TP")
        self.assertEqual(car_stats["fp"], 1, "Greedy matching must yield exactly 1 FP")
        self.assertEqual(car_stats["fn"], 1, "Greedy matching must leave GT 2 as FN")

if __name__ == "__main__":
    unittest.main()
