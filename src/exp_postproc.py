#!/usr/bin/env python3
"""Post-processing study on cached fused instances (holdout, CPU only).

Each variant is a short pipeline of mask operations applied after overlap resolution. Every variant is
scored with the full organizer metric set and compared with the baseline by a paired
observation-level bootstrap. Run after `exp_baseline_d4.py` has written the fused pickle.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from filament_seg.fuse import resolve_overlaps
from filament_seg.metrics import observation_parts, paired_bootstrap, score_split, summarize
from filament_seg.post import components, distance_shrink, erode, fill_holes, smooth, to_rle
from filament_seg.rle import area as rle_area
from filament_seg.rle import read_instances

BASE_THRESHOLD = 0.18


def apply_ops(mask: np.ndarray, ops: list[tuple]) -> list[np.ndarray]:
    masks = [mask]
    for kind, *params in ops:
        if kind == "fill":
            masks = [fill_holes(m, params[0]) for m in masks]
        elif kind == "smooth":
            masks = [smooth(m, params[0]) for m in masks]
        elif kind == "erode":
            masks = [erode(m, params[0]) for m in masks]
        elif kind == "dt":
            masks = [distance_shrink(m, *params) for m in masks]
        elif kind == "largest":
            masks = [components(m)[0] for m in masks if m.any()]
        elif kind == "split":
            masks = [piece for m in masks for piece in components(m, params[0])]
        elif kind == "dropsmall":
            masks = [np.any(components(m, params[0]), axis=0) for m in masks if components(m, params[0])]
    return [m for m in masks if m.any()]


def build(fused_by_obs: dict, ops: list[tuple], threshold: float, min_area: int, small_thr: tuple | None) -> dict:
    out = {}
    for observation, fused in fused_by_obs.items():
        kept = resolve_overlaps(fused, min(threshold, small_thr[1] if small_thr else threshold))
        rles = []
        for item, mask in kept:
            if small_thr and mask.sum() < small_thr[0] and item.score < small_thr[1]:
                continue
            if item.score < threshold and not small_thr:
                continue
            for piece in apply_ops(mask, ops):
                rle = to_rle(piece)
                if rle is not None and rle_area([rle])[0] >= min_area:
                    rles.append(rle)
        out[observation] = rles
    return out


VARIANTS: dict[str, dict] = {
    "baseline_t018_e3": dict(ops=[("erode", 3)]),
    "no_erode": dict(ops=[]),
    "erode5": dict(ops=[("erode", 5)]),
    "fill50_e3": dict(ops=[("fill", 50), ("erode", 3)]),
    "fill300_e3": dict(ops=[("fill", 300), ("erode", 3)]),
    "smooth1_e3": dict(ops=[("smooth", 1.0), ("erode", 3)]),
    "smooth2_e3": dict(ops=[("smooth", 2.0), ("erode", 3)]),
    "smooth2_only": dict(ops=[("smooth", 2.0)]),
    "largest_e3": dict(ops=[("erode", 3), ("largest",)]),
    "split50_e3": dict(ops=[("erode", 3), ("split", 50)]),
    "split150_e3": dict(ops=[("erode", 3), ("split", 150)]),
    "dropsmall30_e3": dict(ops=[("erode", 3), ("dropsmall", 30)]),
    "dropsmall100_e3": dict(ops=[("erode", 3), ("dropsmall", 100)]),
    "minarea100_e3": dict(ops=[("erode", 3)], min_area=100),
    "minarea250_e3": dict(ops=[("erode", 3)], min_area=250),
    "small_thr_area400_0.25": dict(ops=[("erode", 3)], small_thr=(400, 0.25)),
    "dt1.5": dict(ops=[("dt", 1.5)]),
    "dt1_rel0.15": dict(ops=[("dt", 1.0, 0.15)]),
    "dt1_rel0.25": dict(ops=[("dt", 1.0, 0.25)]),
    "dt1.5_keep0.3": dict(ops=[("dt", 1.5, 0.0, 0.3)]),
    "dt1_keep0.3_largest": dict(ops=[("dt", 1.0, 0.0, 0.3), ("largest",)]),
    "e3_largest_split": dict(ops=[("erode", 3), ("largest",)], min_area=30),
    "t015_e3": dict(ops=[("erode", 3)], threshold=0.15),
    "t021_e3": dict(ops=[("erode", 3)], threshold=0.21),
    "t025_e3": dict(ops=[("erode", 3)], threshold=0.25),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    fused_by_obs = pickle.loads(args.fused.read_bytes())
    gt = read_instances(args.ground_truth)
    reference_parts = None
    rows = []
    names = [n for n in VARIANTS if not args.only or n in args.only.split(",")]
    for name in names:
        cfg = VARIANTS[name]
        predictions = build(
            fused_by_obs, cfg["ops"], cfg.get("threshold", BASE_THRESHOLD), cfg.get("min_area", 5), cfg.get("small_thr")
        )
        scores = score_split(gt, predictions)
        summary = summarize(scores)
        _, parts = observation_parts(scores)
        if reference_parts is None:
            reference_parts = parts
        gain, low, high, p_le0 = paired_bootstrap(parts, reference_parts, seed=1)
        row = {
            "variant": name,
            "pq": round(summary["pq"], 4),
            "d_pq": round(gain, 4),
            "ci_low": round(low, 4),
            "ci_high": round(high, 4),
            "tp": summary["tp"],
            "fp": summary["fp"],
            "fn": summary["fn"],
            "sem_dice": round(summary["mean_semantic_dice"], 4),
            "pair_iou": round(summary["mean_pair_iou"], 4),
            "one_to_many": round(summary["gt_one_to_many"], 4),
            "many_to_one": round(summary["pred_many_to_one"], 4),
        }
        rows.append(row)
        print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
