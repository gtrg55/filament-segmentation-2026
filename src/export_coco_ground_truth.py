#!/usr/bin/env python3
"""Export one prepared MAGFiLO split from COCO annotations to competition RLE CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from masks import mask_to_rle


def decode_annotation(annotation: dict, height: int, width: int) -> np.ndarray | None:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list) and segmentation:
        raw_polygons = [segmentation] if isinstance(segmentation[0], (int, float)) else segmentation
        parts = mask_utils.frPyObjects(raw_polygons, height, width)
        encoded = mask_utils.merge(parts) if isinstance(parts, list) else parts
    elif isinstance(segmentation, dict):
        encoded = segmentation
        if isinstance(encoded.get("counts"), list):
            encoded = mask_utils.frPyObjects(encoded, height, width)
    else:
        bbox = annotation.get("bbox")
        if not bbox:
            return None
        x, y, box_width, box_height = map(int, bbox)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[max(0, y) : min(height, y + box_height), max(0, x) : min(width, x + box_width)] = 1
        return mask if mask.any() else None
    mask = np.asarray(mask_utils.decode(encoded), dtype=np.uint8)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2).astype(np.uint8)
    return mask if mask.any() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        selected_ids = {
            row["image_id"] for row in csv.DictReader(handle) if row["split"] == args.split
        }
    payload = json.loads(args.coco_json.read_text(encoding="utf-8"))
    images = {str(image["id"]): image for image in payload["images"]}
    annotations_by_image: dict[str, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        image_id = str(annotation["image_id"])
        if image_id in selected_ids:
            annotations_by_image[image_id].append(annotation)

    rows: list[dict[str, str]] = []
    dropped = 0
    for image_id in sorted(selected_ids):
        image = images[image_id]
        height = int(image.get("height", 2048))
        width = int(image.get("width", 2048))
        instance_index = 0
        for annotation in annotations_by_image.get(image_id, []):
            mask = decode_annotation(annotation, height, width)
            if mask is None:
                dropped += 1
                continue
            instance_index += 1
            rows.append(
                {
                    "filament_id": f"{image_id}_{instance_index}",
                    "segmentation_rle": mask_to_rle(mask),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"annotator_images={len(selected_ids)} annotated_images={len(annotations_by_image)} "
        f"instances={len(rows)} dropped={dropped} output={args.output}"
    )


if __name__ == "__main__":
    main()
