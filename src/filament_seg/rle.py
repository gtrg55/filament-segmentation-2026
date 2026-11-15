"""COCO run-length helpers shared by the metric, fusion and export code."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

SIZE = 2048

RLE = dict  # {"size": [H, W], "counts": bytes}


def encode(mask: np.ndarray) -> RLE:
    """Encode a boolean mask (H x W) as an uncompressed-size COCO RLE with bytes counts."""
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def decode(rle: RLE) -> np.ndarray:
    return mask_utils.decode(rle).astype(bool)


def from_counts(counts: str | bytes, size: int = SIZE) -> RLE:
    """Rebuild an RLE from the `counts` string stored in submission files."""
    return {"size": [size, size], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}


def to_counts(rle: RLE) -> str:
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def area(rles: list[RLE]) -> np.ndarray:
    return mask_utils.area(rles).astype(np.float64) if rles else np.zeros(0)


def iou_matrix(gt: list[RLE], pred: list[RLE]) -> np.ndarray:
    """IoU between every ground-truth mask (rows) and prediction (columns)."""
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)))
    return np.asarray(mask_utils.iou(pred, gt, [0] * len(gt)), dtype=np.float64).T


def read_instances(path: Path, by: str = "observation") -> dict[str, list[RLE]]:
    """Read a competition CSV (`filament_id`, `segmentation_rle`) grouped by image id.

    `filament_id` is `<image id>_<k>`; for ground-truth files the image id is `<annotator>-<observation>`.
    """
    grouped: dict[str, list[RLE]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id, _ = row["filament_id"].rsplit("_", 1)
            grouped[image_id].append(from_counts(row["segmentation_rle"]))
    return dict(grouped)


def write_submission(path: Path, instances_by_observation: dict[str, list[RLE]]) -> int:
    """Write the competition CSV; returns the number of instance rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for observation in sorted(instances_by_observation):
            for index, rle in enumerate(instances_by_observation[observation], start=1):
                writer.writerow([f"{observation}_{index}", to_counts(rle)])
                rows += 1
    return rows
