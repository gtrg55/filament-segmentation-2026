from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filament_seg.d4 import VIEWS, forward, inverse
from filament_seg.fuse import cluster_instances, fuse_cluster, fuse_sources, resolve_overlaps
from filament_seg.metrics import paired_bootstrap, pq_from_parts, score_copy, summarize
from filament_seg.post import components, distance_shrink, erode, fill_holes
from filament_seg.rle import SIZE, decode, encode


def box(y0: int, y1: int, x0: int, x1: int) -> dict:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return encode(mask)


class D4Test(unittest.TestCase):
    def test_roundtrip_all_views(self) -> None:
        image = np.arange(30, dtype=np.uint8).reshape(5, 6)[:, :5]
        for k, flip in VIEWS:
            np.testing.assert_array_equal(inverse(forward(image, k, flip), k, flip), image)


class MetricsTest(unittest.TestCase):
    def test_perfect_prediction(self) -> None:
        gt = [box(0, 10, 0, 10), box(100, 120, 100, 130)]
        score = score_copy(gt, list(gt))
        self.assertEqual((score.tp, score.fp, score.fn), (2, 0, 0))
        self.assertAlmostEqual(pq_from_parts(score.parts), 1.0)
        self.assertAlmostEqual(score.semantic_dice, 1.0)

    def test_threshold_is_strict(self) -> None:
        gt = [box(0, 10, 0, 10)]
        half = [box(0, 10, 0, 5)]  # IoU exactly 0.5 is not a hit
        score = score_copy(gt, half)
        self.assertEqual((score.tp, score.fp, score.fn), (0, 1, 1))

    def test_links_count_any_overlap(self) -> None:
        gt = [box(0, 10, 0, 10), box(0, 10, 12, 22)]
        pred = [box(0, 10, 9, 13)]  # touches both GT filaments by one or two columns
        score = score_copy(gt, pred)
        self.assertEqual(score.pred_links.tolist(), [2])
        self.assertEqual(score.gt_links.tolist(), [1, 1])

    def test_summary_rates(self) -> None:
        gt = [box(0, 10, 0, 10)]
        summary = summarize({"a-1": score_copy(gt, gt), "b-1": score_copy(gt, [box(500, 510, 500, 510)])})
        self.assertAlmostEqual(summary["pq"], 1.0 / (1 + 0.5 + 0.5))
        self.assertAlmostEqual(summary["gt_unmatched"], 0.5)

    def test_paired_bootstrap_identical_is_zero(self) -> None:
        parts = np.array([[5.0, 6, 2, 3], [4.0, 5, 3, 1], [2.0, 3, 1, 2]])
        gain, low, high, _ = paired_bootstrap(parts, parts, n_boot=200)
        self.assertEqual((gain, low, high), (0.0, 0.0, 0.0))


class FuseTest(unittest.TestCase):
    def test_cluster_one_member_per_source(self) -> None:
        same = box(0, 20, 0, 20)
        sources = [[(same, 0.9), (box(0, 20, 1, 21), 0.8)], [(box(0, 20, 0, 19), 0.7)]]
        clusters = cluster_instances(sources, cluster_iou=0.3)
        sizes = sorted(len(cluster) for cluster in clusters)
        self.assertEqual(sizes, [1, 2])  # the second same-source instance cannot join the first cluster

    def test_vote_and_score(self) -> None:
        members = [(0, box(0, 10, 0, 10), 0.9), (1, box(0, 10, 0, 10), 0.6), (2, box(0, 10, 0, 4), 0.3)]
        fused = fuse_cluster(members, n_sources=4, vote=0.5)
        self.assertAlmostEqual(fused.score, (0.9 + 0.6 + 0.3) / 4)
        self.assertEqual(int(decode(fused.rle).sum()), 100)  # pixels in >= 2 of 3 members

    def test_resolve_overlaps_prefers_higher_score(self) -> None:
        fused = fuse_sources([[(box(0, 10, 0, 10), 0.9), (box(5, 15, 0, 10), 0.5)]])
        kept = resolve_overlaps(fused, min_score=0.1)
        areas = sorted(int(mask.sum()) for _, mask in kept)
        self.assertEqual(areas, [50, 100])


class PostTest(unittest.TestCase):
    def test_erode_identity_and_shrink(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        np.testing.assert_array_equal(erode(mask, 1), mask)
        self.assertEqual(int(erode(mask, 3).sum()), 64)

    def test_distance_shrink_radius_one_matches_cross_erosion(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        np.testing.assert_array_equal(distance_shrink(mask, 1.0), erode(mask, 3))

    def test_distance_shrink_keeps_thin_masks(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 2:18] = True  # one pixel wide: a plain erosion would delete it
        out = distance_shrink(mask, 1.0, keep_fraction=0.3)
        np.testing.assert_array_equal(out, mask)

    def test_fill_holes_only_enclosed_and_small(self) -> None:
        mask = np.ones((12, 12), dtype=bool)
        mask[4:6, 4:6] = False
        mask[0, 0] = False  # touches the border: not a hole
        filled = fill_holes(mask, max_hole=10)
        self.assertTrue(filled[4:6, 4:6].all())
        self.assertFalse(filled[0, 0])

    def test_components_largest_first(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[0:2, 0:2] = True
        mask[10:15, 10:15] = True
        pieces = components(mask)
        self.assertEqual([int(p.sum()) for p in pieces], [25, 4])


if __name__ == "__main__":
    unittest.main()
