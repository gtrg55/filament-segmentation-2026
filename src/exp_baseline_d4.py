#!/usr/bin/env python3
"""Reproduce the D4 baseline with the `filament_seg` package and report the full metric set.

Reads the per-view cache written by `research_d4_tta.py cache`, fuses the 8 views, resolves overlaps
at a score threshold, erodes 3x3 and scores against the holdout ground truth. Also writes the fused
instances to a pickle so post-processing studies do not repeat the clustering.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from filament_seg.fuse import fuse_sources, resolve_overlaps
from filament_seg.metrics import bootstrap_pq, observation_parts, score_split, summarize
from filament_seg.post import erode, to_rle
from filament_seg.rle import read_instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--fused-out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--views", type=int, default=8)
    args = parser.parse_args()

    gt = read_instances(args.ground_truth)
    observations = sorted({copy.split("-", 1)[1] for copy in gt})
    fused_by_obs = {}
    predictions = {}
    for observation in observations:
        per_view = pickle.loads((args.cache_dir / f"{observation}.pkl").read_bytes())[: args.views]
        fused = fuse_sources([[(rle, conf) for rle, conf in view] for view in per_view])
        fused_by_obs[observation] = fused
        kept = resolve_overlaps(fused, args.threshold)
        masks = [to_rle(erode(mask, 3)) for _, mask in kept]
        predictions[observation] = [m for m in masks if m is not None]
    args.fused_out.parent.mkdir(parents=True, exist_ok=True)
    args.fused_out.write_bytes(pickle.dumps(fused_by_obs))
    scores = score_split(gt, predictions)
    summary = summarize(scores)
    _, parts = observation_parts(scores)
    point, low, high = bootstrap_pq(parts)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}, indent=1))
    print(f"PQ {point:.4f} 95% CI [{low:.4f}, {high:.4f}] observations={len(parts)}")


if __name__ == "__main__":
    main()
