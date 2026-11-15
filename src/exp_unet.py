#!/usr/bin/env python3
"""Use the consensus-probability U-Net next to the detector ensemble (holdout, CPU only).

Inputs: fused detector instances (`exp_baseline_d4.py` / `exp_ensemble.py` pickle) and the cached D4-averaged
U-Net probability maps. Variants: (1) standalone U-Net (semantic Dice, connected components as instances),
(2) rescoring the detector instances with the mean U-Net probability inside them, (3) restricting each
detector mask to pixels the U-Net supports, (4) adding confident U-Net components the detector missed.
Every variant is compared with the baseline by a paired observation-level bootstrap.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np

from filament_seg.fuse import resolve_overlaps
from filament_seg.metrics import observation_parts, paired_bootstrap, score_split, summarize
from filament_seg.post import components, erode, to_rle
from filament_seg.rle import decode, read_instances


def load_pool(fused: list, probability: np.ndarray, min_score: float) -> list[dict]:
    pool = []
    for item, mask in resolve_overlaps(fused, min_score):
        eroded = erode(mask, 3)
        if not eroded.any():
            continue
        pool.append({"score": item.score, "mask": eroded, "mean_p": float(probability[eroded].mean())})
    return pool


def export(masks: list[np.ndarray]) -> list:
    return [rle for rle in (to_rle(m) for m in masks if m.any()) if rle is not None]


def variant_predictions(name: str, pools: dict, probs: dict, params: dict) -> dict:
    out = {}
    for observation, pool in pools.items():
        p = probs[observation]
        kept: list[np.ndarray] = []
        if name == "baseline":
            kept = [i["mask"] for i in pool if i["score"] >= 0.18]
        elif name == "rescore":
            w, thr = params["w"], params["thr"]
            kept = [i["mask"] for i in pool if (1 - w) * i["score"] + w * i["mean_p"] * 0.5 >= thr]
        elif name == "restrict":
            for item in pool:
                if item["score"] < 0.18:
                    continue
                restricted = item["mask"] & (p > params["t"])
                pieces = components(restricted)
                if pieces and pieces[0].sum() >= 30:
                    kept.append(pieces[0])
        elif name == "extend":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["r"], params["r"]))
            for item in pool:
                if item["score"] < 0.18:
                    continue
                grow = cv2.dilate(item["mask"].astype(np.uint8), kernel).astype(bool) & (p > params["t"])
                kept.append(item["mask"] | grow)
        elif name == "add":
            kept = [i["mask"] for i in pool if i["score"] >= 0.18]
            occupied = np.zeros_like(p, dtype=bool)
            for m in kept:
                occupied |= m
            for piece in components(p > 0.5, params["min_area"]):
                if (piece & occupied).sum() > 0.1 * piece.sum() or float(p[piece].mean()) < params["mean_p"]:
                    continue
                kept.append(piece)
        elif name == "standalone":
            for piece in components(p > params["t"], params["min_area"]):
                if float(p[piece].mean()) >= params["mean_p"]:
                    kept.append(erode(piece, 3))
        out[observation] = export(kept)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--unet-cache", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gt = read_instances(args.ground_truth)
    observations = sorted({copy.split("-", 1)[1] for copy in gt})
    fused_by_obs = pickle.loads(args.fused.read_bytes())
    probs = {o: np.load(args.unet_cache / f"{o}.npy").astype(np.float32) for o in observations}
    pools = {o: load_pool(fused_by_obs[o], probs[o], 0.08) for o in observations}

    # standalone semantic Dice of the U-Net (union masks) against every annotator copy
    for threshold in (0.3, 0.4, 0.5):
        dices = []
        for copy_id, rles in gt.items():
            observation = copy_id.split("-", 1)[1]
            union = np.zeros((2048, 2048), dtype=bool)
            for rle in rles:
                union |= decode(rle)
            pred = probs[observation] > threshold
            dices.append(2.0 * (pred & union).sum() / max(1, pred.sum() + union.sum()))
        print(f"U-Net semantic Dice at p>{threshold}: {np.mean(dices):.4f}", flush=True)

    specs = [("baseline", {})]
    specs += [("rescore", {"w": w, "thr": t}) for w in (0.3, 0.6) for t in (0.12, 0.15, 0.18, 0.21)]
    specs += [("restrict", {"t": t}) for t in (0.15, 0.3)]
    specs += [("extend", {"t": t, "r": 5}) for t in (0.4, 0.5)]
    specs += [("add", {"min_area": 400, "mean_p": m}) for m in (0.55, 0.7)]
    specs += [("standalone", {"t": t, "min_area": 300, "mean_p": 0.5}) for t in (0.4, 0.5)]
    reference = None
    rows = []
    for name, params in specs:
        scores = score_split(gt, variant_predictions(name, pools, probs, params))
        summary = summarize(scores)
        _, parts = observation_parts(scores)
        reference = parts if reference is None else reference
        gain, low, high, _ = paired_bootstrap(parts, reference, seed=5)
        row = {"variant": name, "params": str(params), "pq": round(summary["pq"], 4), "d_pq": round(gain, 4),
               "ci_low": round(low, 4), "ci_high": round(high, 4), "tp": summary["tp"], "fp": summary["fp"],
               "fn": summary["fn"], "sem_dice": round(summary["mean_semantic_dice"], 4),
               "many_to_one": round(summary["pred_many_to_one"], 4)}
        rows.append(row)
        print(row, flush=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
