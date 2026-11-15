#!/usr/bin/env python3
"""Build observation-grouped, observatory-stratified K-fold YOLO datasets from the training split.

The 141 holdout observations of `yolo_seg_observation_split_v1` are never touched: folds are cut
from its `train` rows only. Every annotator copy of one observation stays in one fold. Images and
labels are symlinked, so the fold datasets cost no disk space.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def assign_folds(rows: list[dict], n_folds: int, seed: int) -> dict[str, int]:
    """Round-robin observations inside each observatory after a seeded shuffle."""
    by_observatory: dict[str, list[str]] = defaultdict(list)
    for observation in sorted({row["observation_id"] for row in rows}):
        by_observatory[observation[-2:]].append(observation)
    rng = random.Random(seed)
    folds: dict[str, int] = {}
    offset = 0
    for observatory in sorted(by_observatory):
        observations = by_observatory[observatory]
        rng.shuffle(observations)
        for index, observation in enumerate(observations):
            folds[observation] = (index + offset) % n_folds
        offset += len(observations)
    return folds


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="yolo_seg_observation_split_v1 directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()

    with (args.dataset / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    folds = assign_folds(rows, args.folds, args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "folds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["observation_id", "fold"])
        writer.writerows(sorted(folds.items()))

    summary = {"folds": args.folds, "seed": args.seed, "train_observations": len(folds), "per_fold": {}}
    for fold in range(args.folds):
        root = args.output / f"fold{fold}"
        counts = {"train": 0, "val": 0}
        for row in rows:
            part = "val" if folds[row["observation_id"]] == fold else "train"
            name = row["prepared_file_name"]
            label = name.rsplit(".", 1)[0] + ".txt"
            link(args.dataset / "images" / "train" / name, root / "images" / part / name)
            link(args.dataset / "labels" / "train" / label, root / "labels" / part / label)
            counts[part] += 1
        (root / "data.yaml").write_text(
            f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n",
            encoding="utf-8",
        )
        held = sorted(o for o, f in folds.items() if f == fold)
        summary["per_fold"][fold] = {"annotator_images": counts, "held_out_observations": len(held)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
