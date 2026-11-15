from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from masks import mask_to_rle, remove_instance_overlaps, rle_to_mask


class MaskUtilitiesTest(unittest.TestCase):
    def test_overlap_pixels_go_to_highest_confidence(self) -> None:
        masks = np.zeros((2, 4, 4), dtype=bool)
        masks[0, 0:3, 0:3] = True
        masks[1, 1:4, 1:4] = True
        cleaned = remove_instance_overlaps(masks, np.array([0.2, 0.9]), min_area=1)
        self.assertEqual(cleaned.shape, (2, 4, 4))
        self.assertTrue(cleaned[0, 2, 2])
        self.assertFalse(cleaned[1, 2, 2])
        self.assertFalse(np.logical_and(cleaned[0], cleaned[1]).any())

    def test_tiny_remainder_is_removed(self) -> None:
        masks = np.ones((2, 3, 3), dtype=bool)
        cleaned = remove_instance_overlaps(masks, np.array([0.9, 0.1]), min_area=2)
        self.assertEqual(cleaned.shape, (1, 3, 3))

    def test_compressed_rle_round_trip(self) -> None:
        mask = np.zeros((8, 7), dtype=bool)
        mask[1:5, 2:6] = True
        mask[6, 1] = True
        counts = mask_to_rle(mask)
        decoded = rle_to_mask(counts, height=8, width=7)
        np.testing.assert_array_equal(decoded, mask)


if __name__ == "__main__":
    unittest.main()
