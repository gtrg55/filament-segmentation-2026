#!/usr/bin/env python3
"""Build a YOLO dataset from every labelled observation (fit set + holdout) for the final refit of the recipe.

Used only after the recipe was validated on the holdout: the refit models see all 707 training observations, so
their own holdout score is not an honest estimate and is never reported as one. Images and labels are symlinked;
`val` holds a few images only because Ultralytics validates once after the last epoch.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="yolo_seg_observation_split_v1 directory")
    parser.add_argument("--output", type=Path, required=True, help="directory for the all-data YOLO dataset")
    parser.add_argument("--val-images", type=int, default=24)
    args = parser.parse_args()

    names = []
    for split in ("train", "val"):
        for image in sorted((args.dataset / "images" / split).iterdir()):
            label = args.dataset / "labels" / split / f"{image.stem}.txt"
            link(image, args.output / "images" / "train" / image.name)
            if label.exists():
                link(label, args.output / "labels" / "train" / label.name)
            names.append((image, label))
    for image, label in names[-args.val_images:]:
        link(image, args.output / "images" / "val" / image.name)
        if label.exists():
            link(label, args.output / "labels" / "val" / label.name)
    (args.output / "data.yaml").write_text(
        f"path: {args.output.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n", encoding="utf-8")
    print(f"{len(names)} images linked into {args.output}")


if __name__ == "__main__":
    main()
