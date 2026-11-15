#!/usr/bin/env python3
"""Score identity-view caches produced at several inference sizes (single checkpoint, holdout).

For each size the score threshold is swept (overlap resolution + 3x3 erosion as in the baseline) and the
best threshold is compared with the 2048 px control by a paired observation-level bootstrap.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from filament_seg.fuse import fuse_sources, resolve_overlaps
from filament_seg.infer import read_cache
from filament_seg.metrics import observation_parts, paired_bootstrap, score_split, summarize
from filament_seg.post import erode, to_rle
from filament_seg.rle import read_instances


def predictions_for(cache_dir: Path, observations: list[str], threshold: float, duplicate_fraction: float | None) -> dict:
    out = {}
    for observation in observations:
        per_view = read_cache(cache_dir, observation)
        fused = fuse_sources([[(rle, conf) for rle, conf in view] for view in per_view])
        kept = resolve_overlaps(fused, threshold, duplicate_fraction=duplicate_fraction)
        masks = [to_rle(erode(mask, 3)) for _, mask in kept]
        out[observation] = [m for m in masks if m is not None]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--control", default="s2048")
    parser.add_argument("--glob", default="s*")
    parser.add_argument("--duplicate-fraction", type=float)
    parser.add_argument("--thresholds", default="0.10,0.13,0.16,0.19,0.22,0.26,0.30")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gt = read_instances(args.ground_truth)
    observations = sorted({copy.split("-", 1)[1] for copy in gt})
    thresholds = [float(t) for t in args.thresholds.split(",")]
    best_parts: dict[str, np.ndarray] = {}
    rows = []
    for cache_dir in sorted(args.probe_dir.glob(args.glob)):
        if len(list(cache_dir.glob("*.pkl"))) < len(observations):
            print(f"skip {cache_dir.name}: incomplete cache")
            continue
        best = None
        for threshold in thresholds:
            scores = score_split(gt, predictions_for(cache_dir, observations, threshold, args.duplicate_fraction))
            summary = summarize(scores)
            _, parts = observation_parts(scores)
            rows.append({"size": cache_dir.name, "threshold": threshold, "pq": round(summary["pq"], 4),
                         "tp": summary["tp"], "fp": summary["fp"], "fn": summary["fn"],
                         "sem_dice": round(summary["mean_semantic_dice"], 4)})
            if best is None or summary["pq"] > best[0]:
                best = (summary["pq"], threshold, parts)
        best_parts[cache_dir.name] = best[2]
        print(f"{cache_dir.name}: best PQ {best[0]:.4f} at threshold {best[1]}", flush=True)
    if args.control in best_parts:
        for name, parts in best_parts.items():
            gain, low, high, p = paired_bootstrap(parts, best_parts[args.control], seed=3)
            print(f"{name} vs {args.control}: gain {gain:+.4f} CI [{low:+.4f}, {high:+.4f}] P(gain<=0)={p:.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
