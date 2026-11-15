#!/usr/bin/env python3
"""Figures for the technical report: pipeline diagram, qualitative examples, ablation and distributions.

Subcommands: `pipeline`, `examples`, `results`. All figures are vector PDFs sized for a two-column ACM page
(single column 3.3 in, double column 7 in) and use a colour-blind-safe palette.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERMILION, SKY, PINK, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#6B7280"
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.linewidth": 0.6, "pdf.fonttype": 42})


def box(ax, x, y, w, h, text, color, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", ec=color, fc=fc, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.4, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=GREY))


def pipeline(out: Path, n_models: int, n_views: int, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    box(ax, 0.2, 6.6, 2.6, 1.1, "H-alpha image\n2048 x 2048", GREY)
    box(ax, 3.7, 6.6, 3.0, 1.1, f"{n_views} dihedral views\n(4 rotations x flip)", BLUE)
    box(ax, 7.3, 6.6, 2.5, 1.1, f"YOLO11s-seg\nx {n_models} models", BLUE)
    box(ax, 0.2, 4.4, 4.4, 1.3, "map masks back to\nimage frame; RLE + score", GREEN)
    box(ax, 5.4, 4.4, 4.4, 1.3, "cluster by IoU > 0.3\n(one member per source)", GREEN)
    box(ax, 0.2, 2.2, 4.4, 1.3, "pixel vote (>= half)\nscore = sum conf / sources", ORANGE)
    box(ax, 5.4, 2.2, 4.4, 1.3, f"overlaps by score;\nscore >= {threshold:.2f}", ORANGE)
    box(ax, 0.2, 0.1, 4.4, 1.3, "erosion 3x3 (ellipse)\nlargest component", VERMILION)
    box(ax, 5.4, 0.1, 4.4, 1.3, "COCO RLE counts\none row per instance", VERMILION)
    arrow(ax, 2.8, 7.15, 3.7, 7.15)
    arrow(ax, 6.7, 7.15, 7.3, 7.15)
    arrow(ax, 8.5, 6.6, 2.4, 5.7)
    arrow(ax, 4.6, 5.05, 5.4, 5.05)
    arrow(ax, 7.6, 4.4, 2.4, 3.5)
    arrow(ax, 4.6, 2.85, 5.4, 2.85)
    arrow(ax, 7.6, 2.2, 2.4, 1.4)
    arrow(ax, 4.6, 0.75, 5.4, 0.75)
    fig.savefig(out, bbox_inches="tight")


def results(out: Path, results_json: Path, human_pq: float) -> None:
    """PQ with bootstrap intervals per stage, IoU/Dice histograms of the final stage, fragmentation / merging rates."""
    report = json.loads(results_json.read_text(encoding="utf-8"))
    stages = report["stages"]
    final = next(s for s in stages if s["name"] == report["final"])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), gridspec_kw={"width_ratios": [1.5, 1.0, 0.9]})

    ax = axes[0]
    ys = np.arange(len(stages))[::-1]
    for y, stage in zip(ys, stages):
        low, high = stage["pq_ci"]
        color = BLUE if stage["name"] == report["final"] else SKY
        ax.errorbar(stage["pq"], y, xerr=[[stage["pq"] - low], [high - stage["pq"]]], fmt="o", color=color, ms=3.2, lw=1.1,
                    capsize=1.8)
        ax.text(high + 0.004, y, f"{stage['pq']:.3f}", va="center", fontsize=6.2)
    ax.axvline(human_pq, color=VERMILION, lw=0.9, ls="--")
    ax.text(human_pq - 0.002, ys.mean(), f"expert vs expert {human_pq:.3f}", color=VERMILION, ha="right", va="center",
            rotation=90, fontsize=5.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([s["label"] for s in stages], fontsize=6.0)
    ax.set_xlabel("Panoptic Quality (holdout, 95% bootstrap CI)")
    lows = min(s["pq_ci"][0] for s in stages)
    highs = max(s["pq_ci"][1] for s in stages)
    ax.set_xlim(min(lows, human_pq) - 0.02, highs + 0.035)
    ax.set_ylim(-0.6, len(stages) - 0.1)
    ax.spines[["top", "right"]].set_visible(False)

    dist = report["final_distribution"]
    edges = np.array(dist["edges"])
    centres = (edges[1:] + edges[:-1]) / 2
    ax = axes[1]
    for key, color, name in (("iou", BLUE, "IoU"), ("dice", ORANGE, "Dice")):
        counts = np.array(dist[key], dtype=float)
        mean = float((counts * centres).sum() / counts.sum())
        ax.step(centres, counts / counts.sum(), where="mid", color=color, lw=1.1, label=f"{name} (mean {mean:.2f})")
    ax.set_xlabel("IoU or Dice of overlapping\nprediction/GT pairs")
    ax.set_ylabel("share of pairs")
    ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    first = stages[0]
    width = 0.36
    for offset, stage, color, name in ((-width / 2, first, SKY, first["label"]), (width / 2, final, BLUE, final["label"])):
        ax.bar(np.arange(2) + offset, [stage["gt_one_to_many"], stage["pred_many_to_one"]], width, color=color, label=name)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["one-to-many\n(split)", "many-to-one\n(merged)"], fontsize=6.0)
    ax.set_ylabel("share")
    ax.set_ylim(0, max(first["gt_one_to_many"], first["pred_many_to_one"], final["gt_one_to_many"], final["pred_many_to_one"]) * 1.6)
    ax.legend(frameon=False, fontsize=5.4, loc="upper right", handlelength=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig(out, bbox_inches="tight")


def errors(out: Path, analysis_json: Path) -> None:
    """Recall of annotated filaments and precision of predicted instances by area quintile."""
    report = json.loads(analysis_json.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    for key, color, label, marker in (("recall_by_gt_area", BLUE, "recall (annotated filaments)", "o"),
                                      ("precision_by_pred_area", ORANGE, "precision (predicted instances)", "s")):
        bins = report[key]
        centre = [np.sqrt(max(b["area_low"], 1.0) * b["area_high"]) for b in bins]
        rate = np.array([b["rate"] for b in bins])
        low = np.array([b["ci"][0] for b in bins])
        high = np.array([b["ci"][1] for b in bins])
        ax.errorbar(centre, rate, yerr=[rate - low, high - rate], color=color, marker=marker, ms=3.0, lw=1.0, capsize=1.6, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("filament area (px$^2$, quintile centres)")
    ax.set_ylabel("share matched (IoU > 0.5)")
    ax.set_ylim(0.3, 0.9)
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")


def contours(mask: np.ndarray, scale: float) -> list[np.ndarray]:
    import cv2

    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c[:, 0, :] * scale for c in found if len(c) >= 3]


def examples(out: Path, per_observation: Path, ground_truth: Path, predictions: Path, image_dir: Path,
             quantiles: list[float], window: int = 768) -> None:
    """Holdout observations at fixed PQ quantiles (not hand-picked): one annotator's outlines vs predictions.

    Each panel is the window x window crop of the 2048 px image with the most annotated and predicted instances.
    """
    import cv2

    from filament_seg.rle import decode, read_instances

    with per_observation.open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: float(r["pq"]))
    chosen = [rows[min(len(rows) - 1, int(round(q * (len(rows) - 1))))] for q in quantiles]
    gt = read_instances(ground_truth)
    pred = read_instances(predictions)
    fig, axes = plt.subplots(1, len(chosen), figsize=(7.0, 1.95))
    for ax, row in zip(np.atleast_1d(axes), chosen):
        observation = row["observation"]
        image = cv2.imread(str(next(image_dir.glob(f"{observation}.jp*g"))), cv2.IMREAD_GRAYSCALE)
        copy = next(c for c in sorted(gt) if c.split("-", 1)[1] == observation)
        gt_masks = [decode(r) for r in gt[copy]]
        pred_masks = [decode(r) for r in pred.get(observation, [])]
        centres = [np.argwhere(m).mean(axis=0) for m in gt_masks + pred_masks if m.any()]
        best, size = (-1, 0, 0), image.shape[0]
        for y0 in range(0, size - window + 1, 64):
            for x0 in range(0, size - window + 1, 64):
                count = sum(y0 <= cy < y0 + window and x0 <= cx < x0 + window for cy, cx in centres)
                best = max(best, (count, y0, x0))
        _, y0, x0 = best
        crop = image[y0:y0 + window, x0:x0 + window]
        low, high = np.percentile(crop, [1, 99.7])
        ax.imshow(crop, cmap="gray", vmin=low, vmax=high)
        for masks, color, width in ((pred_masks, SKY, 1.0), (gt_masks, ORANGE, 0.8)):
            for mask in masks:
                for line in contours(mask[y0:y0 + window, x0:x0 + window], 1.0):
                    ax.plot(*np.vstack([line, line[:1]]).T, color=color, lw=width)
        ax.set_xlim(0, window)
        ax.set_ylim(window, 0)
        ax.set_title(f"PQ {float(row['pq']):.2f}", fontsize=6.5, pad=2)
        ax.axis("off")
    fig.tight_layout(pad=0.3, w_pad=0.3)
    fig.savefig(out, bbox_inches="tight", dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("pipeline", "results", "examples", "errors"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=int, default=6)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.18)
    parser.add_argument("--results-json", type=Path)
    parser.add_argument("--human-pq", type=float, default=0.340)
    parser.add_argument("--per-observation", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--quantiles", default="0.9,0.65,0.35,0.1")
    parser.add_argument("--analysis-json", type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "pipeline":
        pipeline(args.out, args.models, args.views, args.threshold)
    elif args.command == "results":
        results(args.out, args.results_json, args.human_pq)
    elif args.command == "errors":
        errors(args.out, args.analysis_json)
    else:
        examples(args.out, args.per_observation, args.ground_truth, args.predictions, args.image_dir,
                 [float(q) for q in args.quantiles.split(",")])


if __name__ == "__main__":
    main()
