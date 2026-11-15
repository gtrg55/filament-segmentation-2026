from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pq_metric import pairwise_iou_dice, pq_score


class PQMetricTest(unittest.TestCase):
    def test_pairwise_perfect_masks(self) -> None:
        masks = np.zeros((2, 6, 6), dtype=bool)
        masks[0, :2, :2] = True
        masks[1, 4:, 4:] = True
        iou, dice = pairwise_iou_dice(masks, masks)
        np.testing.assert_array_equal(iou, np.eye(2))
        np.testing.assert_array_equal(dice, np.eye(2))

    def test_perfect_pq(self) -> None:
        mask = np.zeros((1, 4, 4), dtype=bool)
        mask[0, 1:3, 1:3] = True
        result = pq_score([mask], [mask])
        self.assertEqual(result.pq, 1.0)
        self.assertEqual(result.true_positive_pairs, 1)

    def test_one_missing_instance(self) -> None:
        gt = np.zeros((2, 5, 5), dtype=bool)
        gt[0, :2, :2] = True
        gt[1, 3:, 3:] = True
        pred = gt[:1].copy()
        result = pq_score([gt], [pred])
        self.assertAlmostEqual(result.pq, 2 / 3)
        self.assertEqual(result.false_negative_instances, 1)
        self.assertEqual(result.false_positive_instances, 0)

    def test_threshold_is_strict(self) -> None:
        gt = np.zeros((1, 1, 3), dtype=bool)
        pred = np.zeros((1, 1, 3), dtype=bool)
        gt[0, 0, :2] = True
        pred[0, 0, 1:] = True  # IoU = 1/3, definitely not a hit.
        result = pq_score([gt], [pred])
        self.assertEqual(result.pq, 0.0)
        self.assertEqual(result.false_negative_instances, 1)
        self.assertEqual(result.false_positive_instances, 1)


if __name__ == "__main__":
    unittest.main()
