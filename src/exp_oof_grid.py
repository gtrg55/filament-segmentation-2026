#!/usr/bin/env python3
"""Choose (score threshold, erosion, largest component) on out-of-fold predictions of the fit observations.

The 566 fit observations are predicted by the fold model that did not train on them, so tuning here never touches
the holdout. Rule fixed before looking at the grid: take the configuration with the best pooled PQ, then among all
configurations whose paired bootstrap gap to it includes zero pick the one with the highest semantic Dice.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from exp_postproc import build
from filament_seg.metrics import observation_parts, paired_bootstrap, pq_from_parts, score_split, summarize
from filament_seg.rle import read_instances

STATE: dict = {}


def init(fused_path: str, gt_path: str) -> None:
    import cv2

    cv2.setNumThreads(1)  # twelve workers, each with a full OpenCV thread pool, oversubscribe the CPU
    STATE["fused"] = pickle.loads(Path(fused_path).read_bytes())
    STATE["gt"] = read_instances(Path(gt_path))


def run(cfg: tuple) -> tuple:
    threshold, erosion, largest = cfg
    ops = ([("erode", erosion)] if erosion else []) + ([("largest",)] if largest else [])
    scores = score_split(STATE["gt"], build(STATE["fused"], ops, threshold, 5, None))
    names, parts = observation_parts(scores)
    return cfg, summarize(scores), names, parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--folds-csv", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.25,0.30")
    parser.add_argument("--erosions", default="1,2,3,4")
    parser.add_argument("--reference", default="0.18,3,0", help="threshold,erosion,largest of the current recipe")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grid = [(float(t), int(e), bool(l)) for t in args.thresholds.split(",") for e in args.erosions.split(",") for l in (0, 1)]
    ref_t, ref_e, ref_l = args.reference.split(",")
    reference = (float(ref_t), int(ref_e), bool(int(ref_l)))
    if reference not in grid:
        grid.append(reference)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init,
                             initargs=(str(args.fused), str(args.ground_truth))) as pool:
        results = list(pool.map(run, grid, chunksize=1))
    by_cfg = {cfg: (summary, names, parts) for cfg, summary, names, parts in results}
    names = results[0][2]
    with args.folds_csv.open(newline="", encoding="utf-8") as handle:
        fold_of = {row["observation_id"]: int(row["fold"]) for row in csv.DictReader(handle)}
    folds = np.array([fold_of[n] for n in names])

    best_cfg = max(by_cfg, key=lambda c: by_cfg[c][0]["pq"])
    best_parts = by_cfg[best_cfg][2]
    ref_parts = by_cfg[reference][2]
    rows = []
    for cfg, (summary, _, parts) in by_cfg.items():
        gain_ref, lo_ref, hi_ref, _ = paired_bootstrap(parts, ref_parts, seed=5)
        gain_best, lo_best, hi_best, _ = paired_bootstrap(parts, best_parts, seed=5)
        rows.append({
            "threshold": cfg[0], "erosion": cfg[1], "largest": int(cfg[2]), "pq": round(summary["pq"], 4),
            "d_vs_ref": round(gain_ref, 4), "ref_lo": round(lo_ref, 4), "ref_hi": round(hi_ref, 4),
            "d_vs_best": round(gain_best, 4), "best_lo": round(lo_best, 4), "best_hi": round(hi_best, 4),
            "tied_with_best": int(hi_best >= 0), "sem_dice": round(summary["mean_semantic_dice"], 4),
            "pair_iou": round(summary["mean_pair_iou"], 4), "one_to_many": round(summary["gt_one_to_many"], 4),
            "many_to_one": round(summary["pred_many_to_one"], 4), "tp": summary["tp"], "fp": summary["fp"], "fn": summary["fn"],
            "fold_pq": [round(pq_from_parts(parts[folds == k].sum(axis=0)), 4) for k in range(5)],
        })
    rows.sort(key=lambda r: -r["pq"])
    tied = [r for r in rows if r["tied_with_best"]]
    chosen = max(tied, key=lambda r: r["sem_dice"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"observations": len(names), "reference": list(reference), "best_pq": rows[0], "tied_with_best": len(tied),
               "rule_choice": chosen}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for row in rows[:8]:
        print({k: v for k, v in row.items() if k not in ("ref_lo", "ref_hi", "best_lo", "best_hi")})
    print("reference", next(r for r in rows if (r["threshold"], r["erosion"], bool(r["largest"])) == reference))
    print("rule choice", chosen)


if __name__ == "__main__":
    main()
