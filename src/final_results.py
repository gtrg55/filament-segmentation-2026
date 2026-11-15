#!/usr/bin/env python3
"""Final holdout study: pipeline stages from one detector view to the full ensemble, with paired bootstrap CIs.

The stages come from a JSON spec (`configs/study_stages.json`). Every stage fuses cached detector output
(see `cache_views.py`) with its own settings, so the numbers are those of the released pipeline's settings.
A stage whose `threshold` is a list is swept and the best PQ is kept (the sweep is stored, so the optimism of
that choice can be judged); a number is used as is. Each stage is compared with its parent (default: the
previous stage) by a paired observation-level bootstrap on the same resamples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from exp_ensemble import fuse_all, parse_views, predictions_at
from filament_seg.metrics import bootstrap_pq, observation_parts, paired_bootstrap, score_split, summarize
from filament_seg.rle import read_instances, write_submission

N_BOOT = 4000
SEED = 11


def stage_views(stage: dict) -> list[int]:
    views = stage["views"]
    return parse_views(views) if isinstance(views, str) else list(views)


def fused_for(stage: dict, observations: list[str], fused_dir: Path, workers: int) -> dict:
    key = json.dumps([stage["caches"], stage_views(stage), stage.get("cluster_iou", 0.3), stage.get("vote", 0.5)])
    path = fused_dir / f"{hashlib.md5(key.encode()).hexdigest()[:12]}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    fused = fuse_all(observations, [Path(c) for c in stage["caches"]], stage_views(stage),
                     stage.get("cluster_iou", 0.3), stage.get("vote", 0.5), workers)
    fused_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(fused))
    return fused


def run_point(gt: dict, fused: dict, threshold: float, erosion: int, largest: bool) -> dict:
    predictions = predictions_at(fused, threshold, erosion, largest)
    scores = score_split(gt, predictions)
    names, parts = observation_parts(scores)
    return {"predictions": predictions, "summary": summarize(scores), "names": names, "parts": parts,
            "scores": scores, "threshold": threshold}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="results/final_results.json")
    parser.add_argument("--fused-dir", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, help="holdout CSV of the stage named `final` in the spec")
    parser.add_argument("--per-observation-out", type=Path, help="per-observation parts of the final stage (CSV)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    gt = read_instances(Path(spec["ground_truth"]))
    observations = sorted({copy.split("-", 1)[1] for copy in gt})

    results, runs, parents = [], {}, {}
    for index, stage in enumerate(spec["stages"]):
        fused = fused_for(stage, observations, args.fused_dir, args.workers)
        thresholds = stage["threshold"] if isinstance(stage["threshold"], list) else [stage["threshold"]]
        sweep, best = [], None
        for threshold in thresholds:
            point = run_point(gt, fused, threshold, stage.get("erosion", 0), stage.get("largest", False))
            sweep.append({"threshold": threshold, "pq": round(point["summary"]["pq"], 4)})
            if best is None or point["summary"]["pq"] > best["summary"]["pq"]:
                best = point
        parent_name = stage.get("parent", spec["stages"][index - 1]["name"] if index else None)
        pq, low, high = bootstrap_pq(best["parts"], n_boot=N_BOOT, seed=SEED)
        row = {"name": stage["name"], "label": stage["label"], "sources": len(stage["caches"]) * len(stage_views(stage)),
               "n_models": len(stage["caches"]), "n_views": len(stage_views(stage)), "threshold": best["threshold"],
               "erosion": stage.get("erosion", 0), "largest": stage.get("largest", False), "pq": pq,
               "pq_ci": [low, high], "sweep": sweep if len(sweep) > 1 else None, "parent": parent_name,
               **{k: best["summary"][k] for k in best["summary"]}}
        if parent_name:
            gain, g_low, g_high, p_le0 = paired_bootstrap(best["parts"], parents[parent_name], n_boot=N_BOOT, seed=SEED)
            row.update({"d_pq": gain, "d_pq_ci": [g_low, g_high], "p_gain_le0": p_le0})
        results.append(row)
        runs[stage["name"]] = best
        parents[stage["name"]] = best["parts"]
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("sweep", "parent")},
              flush=True)

    first = spec["stages"][0]["name"]
    for row in results[1:]:
        gain, g_low, g_high, p_le0 = paired_bootstrap(parents[row["name"]], parents[first], n_boot=N_BOOT, seed=SEED)
        row.update({"d_pq_vs_first": gain, "d_pq_vs_first_ci": [g_low, g_high]})

    final = runs[spec["final"]]
    iou = np.concatenate([s.pair_iou for s in final["scores"].values()])
    dice = np.concatenate([s.pair_dice for s in final["scores"].values()])
    edges = np.linspace(0.0, 1.0, 26)
    report = {
        "holdout": {"observations": len(observations), "copies": len(gt), "bootstrap_draws": N_BOOT, "seed": SEED},
        "final": spec["final"],
        "stages": results,
        "final_distribution": {"edges": edges.tolist(), "iou": np.histogram(iou, edges)[0].tolist(),
                               "dice": np.histogram(dice, edges)[0].tolist(), "pairs": int(len(iou))},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.predictions_out:
        args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
        write_submission(args.predictions_out, final["predictions"])
    if args.per_observation_out:
        args.per_observation_out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["observation,sum_iou,tp,fp,fn,pq"]
        for name, (sum_iou, tp, fp, fn) in zip(final["names"], final["parts"]):
            denominator = tp + 0.5 * fp + 0.5 * fn
            lines.append(f"{name},{sum_iou:.6f},{int(tp)},{int(fp)},{int(fn)},{(sum_iou / denominator if denominator else 0.0):.6f}")
        args.per_observation_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
