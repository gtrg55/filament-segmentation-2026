#!/usr/bin/env python3
"""Validate Filament CSV ids and optionally decode every COCO RLE."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


HEADER = ["filament_id", "segmentation_rle"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()

    test_ids = None
    if args.test_dir:
        test_ids = {path.stem for path in args.test_dir.glob("*.jpeg")}
        if not test_ids:
            raise ValueError(f"No JPEG files found under {args.test_dir}")

    mask_util = None
    if args.decode:
        try:
            from pycocotools import mask as mask_util
        except ImportError as exc:
            raise RuntimeError("Install pycocotools to use --decode") from exc

    seen: set[str] = set()
    predicted_images: set[str] = set()
    with args.submission.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADER:
            raise ValueError(f"Expected header {HEADER}, got {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            filament_id = row["filament_id"].strip()
            rle = row["segmentation_rle"].strip()
            if not filament_id or "_" not in filament_id:
                raise ValueError(f"Line {line_number}: invalid filament_id={filament_id!r}")
            image_id, suffix = filament_id.rsplit("_", 1)
            if not image_id or not suffix:
                raise ValueError(f"Line {line_number}: invalid filament_id={filament_id!r}")
            if filament_id in seen:
                raise ValueError(f"Line {line_number}: duplicate filament_id={filament_id}")
            if not rle:
                raise ValueError(f"Line {line_number}: empty segmentation_rle")
            if test_ids is not None and image_id not in test_ids:
                raise ValueError(f"Line {line_number}: unknown image id={image_id}")
            if mask_util is not None:
                try:
                    decoded = mask_util.decode({"size": [2048, 2048], "counts": rle.encode("utf-8")})
                except Exception as exc:
                    raise ValueError(f"Line {line_number}: invalid COCO RLE") from exc
                if decoded.shape != (2048, 2048) or not decoded.any():
                    raise ValueError(f"Line {line_number}: decoded mask is empty or wrong-sized")
            seen.add(filament_id)
            predicted_images.add(image_id)

    if not seen:
        raise ValueError("Submission contains no predicted filaments")
    if test_ids is not None:
        print(
            f"OK: {len(seen)} instances across {len(predicted_images)}/{len(test_ids)} test images"
        )
    else:
        print(f"OK: {len(seen)} instances")


if __name__ == "__main__":
    main()

