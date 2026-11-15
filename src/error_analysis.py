#!/usr/bin/env python3
"""Where do the errors come from? Recall by annotated filament size, precision by predicted size, PQ by observatory.

A ground-truth filament counts as detected when some prediction of its observation has IoU > 0.5 with it (the
organizers' true-positive rule); a prediction is correct when it matches some annotator copy of its observation
that way. Bins are quantiles of the area so that every bin holds the same number of filaments. Intervals are
observation-level bootstraps.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from filament_seg.rle import area, iou_matrix, read_instances

N_BINS = 5
N_BOOT = 2000


def binned_rate(areas: np.ndarray, hits: np.ndarray, observation: np.ndarray, edges: np.ndarray) -> list[dict]:
    index = np.clip(np.searchsorted(edges, areas, side="right") - 1, 0, len(edges) - 2)
    unique = np.unique(observation)
    position = {o: i for i, o in enumerate(unique)}
    obs_index = np.array([position[o] for o in observation])
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(unique), size=(N_BOOT, len(unique)))
    out = []
    for b in range(len(edges) - 1):
        in_bin = index == b
        per_obs = np.zeros((len(unique), 2))
        np.add.at(per_obs[:, 0], obs_index[in_bin], hits[in_bin])
        np.add.at(per_obs[:, 1], obs_index[in_bin], 1)
        sums = per_obs[draws].sum(axis=1)
        rates = sums[:, 0] / np.maximum(sums[:, 1], 1)
        out.append({"area_low": float(edges[b]), "area_high": float(edges[b + 1]), "n": int(in_bin.sum()),
                    "rate": float(hits[in_bin].mean()), "ci": [float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975))]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--per-observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gt = read_instances(args.ground_truth)
    pred = read_instances(args.predictions)
    gt_area, gt_hit, gt_obs = [], [], []
    pred_area, pred_hit, pred_obs = [], [], []
    for copy, gt_rles in gt.items():
        observation = copy.split("-", 1)[1]
        pred_rles = pred.get(observation, [])
        iou = iou_matrix(gt_rles, pred_rles) if gt_rles and pred_rles else np.zeros((len(gt_rles), len(pred_rles)))
        gt_area += area(gt_rles).tolist()
        gt_hit += (iou.max(axis=1) > 0.5).tolist() if pred_rles else [False] * len(gt_rles)
        gt_obs += [observation] * len(gt_rles)
    matched_by_prediction: dict[str, np.ndarray] = {}
    for observation, pred_rles in pred.items():
        hit = np.zeros(len(pred_rles), dtype=bool)
        for copy, gt_rles in gt.items():
            if copy.split("-", 1)[1] == observation and gt_rles and pred_rles:
                hit |= iou_matrix(gt_rles, pred_rles).max(axis=0) > 0.5
        matched_by_prediction[observation] = hit
        pred_area += area(pred_rles).tolist()
        pred_hit += hit.tolist()
        pred_obs += [observation] * len(pred_rles)

    gt_area, gt_hit = np.array(gt_area, dtype=float), np.array(gt_hit, dtype=float)
    pred_area, pred_hit = np.array(pred_area, dtype=float), np.array(pred_hit, dtype=float)
    gt_edges = np.quantile(gt_area, np.linspace(0, 1, N_BINS + 1))
    pred_edges = np.quantile(pred_area, np.linspace(0, 1, N_BINS + 1))

    by_site: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4))
    counts: dict[str, int] = defaultdict(int)
    with args.per_observation.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            site = row["observation"][-2:]
            by_site[site] += np.array([float(row["sum_iou"]), float(row["tp"]), float(row["fp"]), float(row["fn"])])
            counts[site] += 1
    sites = {}
    for site, (sum_iou, tp, fp, fn) in sorted(by_site.items()):
        denominator = tp + 0.5 * fp + 0.5 * fn
        sites[site] = {"observations": counts[site], "pq": float(sum_iou / denominator) if denominator else 0.0}

    report = {
        "recall_by_gt_area": binned_rate(gt_area, gt_hit, np.array(gt_obs), gt_edges),
        "precision_by_pred_area": binned_rate(pred_area, pred_hit, np.array(pred_obs), pred_edges),
        "pq_by_observatory": sites,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "pq_by_observatory"}, indent=1)[:1800])
    print(sites)


if __name__ == "__main__":
    main()
