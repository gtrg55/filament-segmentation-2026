#!/usr/bin/env python3
"""Convert the MAGFiLO COCO polygons into a leak-safe YOLO segmentation dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


def polygon_area(points: np.ndarray) -> float:
    """Return the unsigned shoelace area of an Nx2 polygon."""
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2


def annotation_polygon(
    annotation: dict, width: int, height: int
) -> tuple[np.ndarray | None, bool, bool]:
    """Return the largest valid polygon and flags for multi-part/bbox fallback."""
    segmentation = annotation.get("segmentation")
    polygons: list[np.ndarray] = []
    if isinstance(segmentation, list) and segmentation:
        raw_polygons = [segmentation] if isinstance(segmentation[0], (int, float)) else segmentation
        for raw in raw_polygons:
            points = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
            if len(points) > 1 and np.array_equal(points[0], points[-1]):
                points = points[:-1]
            if len(points) >= 3 and polygon_area(points) > 0:
                polygons.append(points)
    used_bbox = False
    if not polygons:
        bbox = annotation.get("bbox")
        if not bbox or len(bbox) != 4:
            return None, False, False
        x, y, box_width, box_height = map(float, bbox)
        if box_width <= 0 or box_height <= 0:
            return None, False, False
        polygons = [
            np.asarray(
                [[x, y], [x + box_width, y], [x + box_width, y + box_height], [x, y + box_height]],
                dtype=np.float64,
            )
        ]
        used_bbox = True
    multipart = len(polygons) > 1
    points = max(polygons, key=polygon_area)
    points[:, 0] = np.clip(points[:, 0], 0, width)
    points[:, 1] = np.clip(points[:, 1], 0, height)
    if polygon_area(points) <= 0:
        return None, multipart, used_bbox
    points /= np.asarray([width, height], dtype=np.float64)
    return points, multipart, used_bbox


def hardlink_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be in (0,1)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")

    payload = json.loads(args.coco_json.read_text(encoding="utf-8"))
    annotations_by_image: dict[str, list[dict]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        if not annotation.get("iscrowd", 0):
            annotations_by_image[str(annotation["image_id"])].append(annotation)

    # Multiple COCO image records reference the same physical observation but
    # different independent annotators. Keep each annotator-image as one
    # training sample, while splitting only by physical filename.
    samples: list[dict] = []
    physical_names: set[str] = set()
    for image in payload.get("images", []):
        name = Path(image["file_name"]).name
        source = args.image_dir / name
        if not source.is_file():
            alternate = source.with_suffix(".jpg" if source.suffix.lower() == ".jpeg" else ".jpeg")
            if alternate.is_file():
                source = alternate
                name = alternate.name
            else:
                continue
        image_id = str(image["id"])
        samples.append(
            {
                "image_id": image_id,
                "file_name": name,
                "source": source,
                "width": int(image.get("width", 2048)),
                "height": int(image.get("height", 2048)),
                "annotations": annotations_by_image.get(image_id, []),
            }
        )
        physical_names.add(name)
    if not samples:
        raise ValueError("no COCO images matched --image-dir")

    # Preserve the six observatory suffixes in both folds while keeping every
    # timestamp/stem in exactly one fold.
    names_by_observatory: dict[str, list[str]] = defaultdict(list)
    for name in physical_names:
        names_by_observatory[Path(name).stem[-2:]].append(name)
    rng = random.Random(args.seed)
    val_names: set[str] = set()
    for names in names_by_observatory.values():
        names.sort()
        rng.shuffle(names)
        val_count = max(1, round(len(names) * args.val_fraction)) if len(names) > 1 else 0
        val_names.update(names[:val_count])
    if not val_names or len(val_names) == len(physical_names):
        raise ValueError("split must contain both train and validation images")

    for split in ("train", "val"):
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    total_instances = 0
    dropped_instances = 0
    multipart_instances = 0
    bbox_fallbacks = 0
    for sample in sorted(samples, key=lambda item: item["image_id"]):
        name = sample["file_name"]
        split = "val" if name in val_names else "train"
        prepared_name = f"{sample['image_id']}{sample['source'].suffix.lower()}"
        hardlink_or_copy(
            sample["source"], args.output_dir / "images" / split / prepared_name
        )
        labels: list[str] = []
        for annotation in sample["annotations"]:
            polygon, multipart, used_bbox = annotation_polygon(
                annotation, sample["width"], sample["height"]
            )
            multipart_instances += int(multipart)
            bbox_fallbacks += int(used_bbox)
            if polygon is None:
                dropped_instances += 1
                continue
            coordinates = " ".join(f"{value:.8f}" for value in polygon.reshape(-1))
            labels.append(f"0 {coordinates}")
        total_instances += len(labels)
        label_path = args.output_dir / "labels" / split / f"{sample['image_id']}.txt"
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        rows.append(
            {
                "image_id": sample["image_id"],
                "observation_id": Path(name).stem,
                "file_name": name,
                "prepared_file_name": prepared_name,
                "split": split,
                "annotator": sample["image_id"].split("-", 1)[0],
                "observatory": Path(name).stem[-2:],
                "instances": len(labels),
            }
        )

    args.output_dir.joinpath("data.yaml").write_text(
        f"path: {args.output_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n",
        encoding="utf-8",
    )
    with args.output_dir.joinpath("manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "observation_id",
                "file_name",
                "prepared_file_name",
                "split",
                "annotator",
                "observatory",
                "instances",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "annotator_images": len(rows),
        "unique_observations": len(physical_names),
        "train_annotator_images": sum(row["split"] == "train" for row in rows),
        "val_annotator_images": sum(row["split"] == "val" for row in rows),
        "train_unique_observations": len(physical_names - val_names),
        "val_unique_observations": len(val_names),
        "instances": total_instances,
        "dropped_instances": dropped_instances,
        "multipart_instances_largest_polygon_used": multipart_instances,
        "bbox_fallbacks": bbox_fallbacks,
        "observatories": {key: len(value) for key, value in sorted(names_by_observatory.items())},
        "seed": args.seed,
        "val_fraction": args.val_fraction,
    }
    args.output_dir.joinpath("summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
