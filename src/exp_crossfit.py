#!/usr/bin/env python3
"""How optimistic is a threshold tuned on the holdout? Split-half cross-fitting of the score threshold.

For many random halves of the holdout observations the (threshold, erosion) pair with the best PQ on
half A is applied to half B. The mean out-of-sample PQ against the in-sample best PQ measures the
selection optimism of tuning on the same data that is used for the final report.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from exp_postproc import build
from filament_seg.metrics import observation_parts, pq_from_parts, score_split
from filament_seg.rle import read_instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.10,0.14,0.18,0.22,0.26")
    parser.add_argument("--erosions", default="1,3")
    parser.add_argument("--splits", type=int, default=500)
    args = parser.parse_args()

    fused_by_obs = pickle.loads(args.fused.read_bytes())
    gt = read_instances(args.ground_truth)
    grid = [(float(t), int(e)) for t in args.thresholds.split(",") for e in args.erosions.split(",")]
    parts_by_cfg = {}
    for threshold, erosion in grid:
        predictions = build(fused_by_obs, [("erode", erosion)], threshold, 5, None)
        names, parts = observation_parts(score_split(gt, predictions))
        parts_by_cfg[(threshold, erosion)] = parts
    stack = np.stack([parts_by_cfg[cfg] for cfg in grid])  # cfg x observation x 4
    n = stack.shape[1]
    rng = np.random.default_rng(0)
    in_sample, out_sample, regret = [], [], []
    for _ in range(args.splits):
        order = rng.permutation(n)
        half_a, half_b = order[: n // 2], order[n // 2 :]
        pq_a = np.array([pq_from_parts(cfg_parts[half_a].sum(axis=0)) for cfg_parts in stack])
        pq_b = np.array([pq_from_parts(cfg_parts[half_b].sum(axis=0)) for cfg_parts in stack])
        chosen = int(pq_a.argmax())
        in_sample.append(pq_a[chosen])
        out_sample.append(pq_b[chosen])
        regret.append(pq_b.max() - pq_b[chosen])
    full = {f"t{t}_e{e}": round(pq_from_parts(parts.sum(axis=0)), 4) for (t, e), parts in parts_by_cfg.items()}
    report = {
        "grid_pq_full_holdout": full,
        "mean_in_sample_pq_of_chosen": round(float(np.mean(in_sample)), 4),
        "mean_out_of_sample_pq_of_chosen": round(float(np.mean(out_sample)), 4),
        "selection_optimism": round(float(np.mean(in_sample) - np.mean(out_sample)), 4),
        "mean_regret_vs_best_on_half_b": round(float(np.mean(regret)), 4),
        "splits": args.splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
