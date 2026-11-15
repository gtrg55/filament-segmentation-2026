#!/usr/bin/env python3
"""PQ of each dihedral view on its own: how much does a plain transform of the input move an honest model?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from filament_seg.fuse import fuse_sources, resolve_overlaps
from filament_seg.infer import read_cache
from filament_seg.metrics import observation_parts, score_split, summarize
from filament_seg.post import erode, to_rle
from filament_seg.rle import read_instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gt = read_instances(args.ground_truth)
    observations = sorted({copy.split("-", 1)[1] for copy in gt})
    rows = []
    for view in range(8):
        predictions = {}
        for observation in observations:
            per_view = read_cache(args.cache_dir, observation, [view])
            kept = resolve_overlaps(fuse_sources(per_view), args.threshold)
            predictions[observation] = [m for m in (to_rle(erode(mask, 3)) for _, mask in kept) if m is not None]
        summary = summarize(score_split(gt, predictions))
        rows.append({"view": view, "pq": round(summary["pq"], 4), "tp": summary["tp"], "fp": summary["fp"], "fn": summary["fn"]})
        print(rows[-1], flush=True)
    pqs = [row["pq"] for row in rows]
    report = {"views": rows, "min": min(pqs), "max": max(pqs), "spread": round(max(pqs) - min(pqs), 4)}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "views"}))


if __name__ == "__main__":
    main()
