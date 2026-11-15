#!/usr/bin/env python3
"""CLI: cache dihedral-view instances of one checkpoint for a list of observations.

Observations come from a manifest split (`--manifest/--split`, optionally `--fold-csv/--fold` to take one
cross-validation fold of the training rows) or from an image directory (`--image-dir` only, e.g. the test set).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from filament_seg.infer import cache_images


def observation_images(args: argparse.Namespace) -> list[Path]:
    if args.manifest is None:
        return sorted(args.image_dir.glob("*.jp*g"))
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == args.split]
    if args.fold_csv is not None:
        with args.fold_csv.open(newline="", encoding="utf-8") as handle:
            fold_of = {row["observation_id"]: int(row["fold"]) for row in csv.DictReader(handle)}
        rows = [row for row in rows if fold_of[row["observation_id"]] == args.fold]
    names = sorted({row["file_name"] for row in rows})
    return [args.image_dir / name for name in names]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--fold-csv", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--imgsz", type=int, default=2048)
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument("--iou", type=float, default=0.0)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--retina-masks", action="store_true")
    parser.add_argument("--views", default="all", help="'all' or comma list of view indices (0-7)")
    args = parser.parse_args()

    from filament_seg.d4 import VIEWS

    views = VIEWS if args.views == "all" else [VIEWS[int(i)] for i in args.views.split(",")]
    cache_images(
        args.weights, observation_images(args), args.cache_dir, args.imgsz, args.conf, args.iou, args.max_det,
        args.device, views, args.retina_masks, log=lambda line: print(line, flush=True),
    )


if __name__ == "__main__":
    main()
