from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_yolo_seg import annotation_polygon, polygon_area
from export_coco_ground_truth import decode_annotation


class PrepareYoloSegTest(unittest.TestCase):
    def test_polygon_area(self) -> None:
        square = np.asarray([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        self.assertEqual(polygon_area(square), 4.0)

    def test_largest_coco_polygon_is_normalized(self) -> None:
        annotation = {
            "segmentation": [
                [0, 0, 1, 0, 1, 1, 0, 1],
                [2, 2, 8, 2, 8, 8, 2, 8],
            ],
            "bbox": [0, 0, 10, 10],
        }
        polygon, multipart, used_bbox = annotation_polygon(annotation, width=10, height=10)
        self.assertTrue(multipart)
        self.assertFalse(used_bbox)
        assert polygon is not None
        np.testing.assert_allclose(polygon, [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])

    def test_bbox_fallback(self) -> None:
        polygon, multipart, used_bbox = annotation_polygon(
            {"segmentation": [], "bbox": [2, 1, 4, 3]}, width=10, height=10
        )
        self.assertFalse(multipart)
        self.assertTrue(used_bbox)
        assert polygon is not None
        np.testing.assert_allclose(polygon, [[0.2, 0.1], [0.6, 0.1], [0.6, 0.4], [0.2, 0.4]])

    def test_flat_magfilo_polygon_is_supported(self) -> None:
        polygon, multipart, used_bbox = annotation_polygon(
            {"segmentation": [0, 0, 5, 0, 5, 5, 0, 5, 0, 0]},
            width=10,
            height=10,
        )
        self.assertFalse(multipart)
        self.assertFalse(used_bbox)
        assert polygon is not None
        self.assertEqual(polygon.shape, (4, 2))
        mask = decode_annotation(
            {"segmentation": [0, 0, 5, 0, 5, 5, 0, 5, 0, 0]},
            height=10,
            width=10,
        )
        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(int(mask.sum()), 0)


if __name__ == "__main__":
    unittest.main()
