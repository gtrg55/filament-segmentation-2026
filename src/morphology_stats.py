#!/usr/bin/env python3
"""Shape statistics of annotated and predicted filament masks on the holdout (for the report's morphology section).

Per instance: area, number of connected components, solidity (area / convex hull of the whole mask), elongation
(long / short side of the minimum-area rectangle) and thickness (twice the largest distance-transform value).
Annotator copies are pooled; predictions are the final pipeline's holdout CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from filament_seg.rle import decode, read_instances


def shape(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = np.pad(mask[y0:y1, x0:x1].astype(np.uint8), 1)
    area = int(crop.sum())
    components = cv2.connectedComponents(crop, connectivity=8)[0] - 1
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = np.vstack(contours)
    hull_area = cv2.contourArea(cv2.convexHull(points))
    (_, _), (w, h), _ = cv2.minAreaRect(points)
    return {
        "area": area,
        "components": int(components),
        "solidity": float(min(1.0, area / hull_area)) if hull_area > 0 else 1.0,
        "elongation": float(max(w, h) / max(min(w, h), 1.0)),
        "thickness": float(2 * cv2.distanceTransform(crop, cv2.DIST_L2, 3).max()),
    }


def summarize(instances_by_key: dict, copies: int) -> dict:
    rows = [shape(decode(rle)) for rles in instances_by_key.values() for rle in rles]

    def quantiles(name: str) -> dict:
        values = np.array([r[name] for r in rows], dtype=float)
        return {"p10": float(np.quantile(values, 0.1)), "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.9)), "mean": float(values.mean())}

    return {
        "instances": len(rows),
        "per_image": len(rows) / copies,
        "single_component_share": float(np.mean([r["components"] == 1 for r in rows])),
        **{name: quantiles(name) for name in ("area", "thickness", "solidity", "elongation")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gt = read_instances(args.ground_truth)
    pred = read_instances(args.predictions)
    observations = {copy.split("-", 1)[1] for copy in gt}
    report = {
        "annotated": summarize(gt, len(gt)),
        "predicted": summarize({o: pred.get(o, []) for o in sorted(observations)}, len(observations)),
        "copies": len(gt),
        "observations": len(observations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
