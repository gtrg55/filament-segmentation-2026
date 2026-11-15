#!/usr/bin/env python3
"""Ensemble study on the holdout: vote across models and dihedral views from cached detector output.

A configuration is a set of model caches and a set of views. All (model, view) pairs become sources of one
instance vote. For every configuration the score threshold is swept; the best operating point is reported
with a paired observation-level bootstrap against the reference configuration (the first one listed).
Fusion runs in parallel over observations and the fused instances are kept in a pickle per configuration.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from filament_seg.fuse import fuse_sources, resolve_overlaps
from filament_seg.infer import read_cache
from filament_seg.metrics import observation_parts, paired_bootstrap, score_split, summarize
from filament_seg.post import components, erode, to_rle
from filament_seg.rle import read_instances


def parse_views(text: str) -> list[int]:
    """`0-7` is a range, `0,2,4,6` a list."""
    if "-" in text:
        low, high = text.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(i) for i in text.split(",")]


def fuse_observation(job: tuple) -> tuple[str, list]:
    observation, cache_dirs, views, cluster_iou, vote = job
    sources = []
    for cache_dir in cache_dirs:
        sources += [[(rle, conf) for rle, conf in view] for view in read_cache(Path(cache_dir), observation, views)]
    return observation, fuse_sources(sources, cluster_iou, vote)


def fuse_all(observations: list[str], cache_dirs: list[Path], views: list[int], cluster_iou: float, vote: float,
             workers: int) -> dict:
    jobs = [(o, [str(d) for d in cache_dirs], views, cluster_iou, vote) for o in observations]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(fuse_observation, jobs, chunksize=2))


def predictions_at(fused_by_obs: dict, threshold: float, erosion: int, largest: bool) -> dict:
    out = {}
    for observation, fused in fused_by_obs.items():
        rles = []
        for _, mask in resolve_overlaps(fused, threshold):
            mask = erode(mask, erosion)
            for piece in (components(mask)[:1] if largest else [mask]):
                rle = to_rle(piece) if piece.any() else None
                if rle is not None:
                    rles.append(rle)
        out[observation] = rles
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--config", action="append", required=True,
                        help="name=cacheA+cacheB:views(e.g. 0-7 or 0,2,4,6), first one is the reference")
    parser.add_argument("--thresholds", default="0.10,0.13,0.16,0.18,0.21,0.25,0.30")
    parser.add_argument("--cluster-iou", type=float, default=0.3)
    parser.add_argument("--vote", type=float, default=0.5)
    parser.add_argument("--largest", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fused-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gt = read_instances(args.ground_truth)
    observations = sorted({copy.split("-", 1)[1] for copy in gt})
    thresholds = [float(t) for t in args.thresholds.split(",")]
    reference_parts = None
    rows = []
    for spec in args.config:
        name, rest = spec.split("=", 1)
        caches, views_text = rest.split(":")
        views = parse_views(views_text)
        cache_dirs = [Path(c) for c in caches.split("+")]
        fused_path = args.fused_dir / f"{name}.pkl" if args.fused_dir else None
        if fused_path and fused_path.exists():
            fused_by_obs = pickle.loads(fused_path.read_bytes())
        else:
            fused_by_obs = fuse_all(observations, cache_dirs, views, args.cluster_iou, args.vote, args.workers)
            if fused_path:
                fused_path.parent.mkdir(parents=True, exist_ok=True)
                fused_path.write_bytes(pickle.dumps(fused_by_obs))
        best = None
        for threshold in thresholds:
            scores = score_split(gt, predictions_at(fused_by_obs, threshold, 3, args.largest))
            summary = summarize(scores)
            _, parts = observation_parts(scores)
            if best is None or summary["pq"] > best[0]["pq"]:
                best = (summary, parts, threshold)
        summary, parts, threshold = best
        if reference_parts is None:
            reference_parts = parts
        gain, low, high, p_le0 = paired_bootstrap(parts, reference_parts, seed=11)
        row = {"config": name, "sources": len(cache_dirs) * len(views), "threshold": threshold,
               "pq": round(summary["pq"], 4), "d_pq": round(gain, 4), "ci_low": round(low, 4), "ci_high": round(high, 4),
               "tp": summary["tp"], "fp": summary["fp"], "fn": summary["fn"],
               "sem_dice": round(summary["mean_semantic_dice"], 4), "pair_iou": round(summary["mean_pair_iou"], 4),
               "one_to_many": round(summary["gt_one_to_many"], 4), "many_to_one": round(summary["pred_many_to_one"], 4)}
        rows.append(row)
        print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
