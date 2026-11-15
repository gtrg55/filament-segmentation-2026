#!/usr/bin/env python3
"""Expert-agreement ceilings on the training annotations (no model involved).

Scores follow the organizer rule (strict IoU > 0.5, pooled over annotator copies):
  * human-vs-human: every ordered pair of annotator copies of one observation;
  * consensus-vs-held-out: for observations with three copies, two copies are merged into one
    "consensus" prediction and scored against the third (an oracle no honest model can see).
Consensus variants: `agreed` keeps the instances both experts drew (IoU > 0.5; union, intersection
or one expert's mask), `all` adds the unmatched instances of both experts.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from research_diagnostics import SIZE, overlap, pq_from, pq_parts


def load_copies(annotation_path: Path) -> dict[str, dict[str, list[dict]]]:
    """observation -> annotator-image id -> list of RLE instances."""
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    observation_of = {image["id"]: image["file_name"].rsplit(".", 1)[0] for image in data["images"]}
    per_image: dict[str, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        polygons = annotation["segmentation"]
        if isinstance(polygons, str):
            polygons = json.loads(polygons)
        polygons = [p for p in polygons if len(p) >= 6]
        if polygons:
            per_image[annotation["image_id"]].append(mask_utils.merge(mask_utils.frPyObjects(polygons, SIZE, SIZE)))
    copies: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for image_id, observation in observation_of.items():
        copies[observation][image_id] = per_image.get(image_id, [])
    return copies


def consensus(first: list[dict], second: list[dict], mode: str, merge: str) -> list[dict]:
    """Merge two expert annotations of one observation into one prediction."""
    if not first or not second:
        return list(first) + list(second) if mode == "all" else []
    iou, _, _, _ = overlap(first, second)  # rows: first, columns: second
    used_second: set[int] = set()
    merged: list[dict] = []
    for i, row in enumerate(iou):
        j = int(row.argmax())
        if row[j] > 0.5 and j not in used_second:
            used_second.add(j)
            if merge == "union":
                merged.append(mask_utils.merge([first[i], second[j]]))
            elif merge == "intersection":
                inter = mask_utils.merge([first[i], second[j]], intersect=True)
                merged.append(inter if mask_utils.area(inter) > 0 else first[i])
            else:
                merged.append(first[i])
        elif mode == "all":
            merged.append(first[i])
    if mode == "all":
        merged += [mask for j, mask in enumerate(second) if j not in used_second]
    return merged


def run(annotation_path: Path) -> dict:
    copies = load_copies(annotation_path)
    pooled = {"human_vs_human_all": [0.0, 0, 0, 0], "human_vs_human_three_copy": [0.0, 0, 0, 0]}
    variants = {
        "agreed_first_mask": ("agreed", "first"),
        "agreed_union": ("agreed", "union"),
        "agreed_intersection": ("agreed", "intersection"),
        "all_union": ("all", "union"),
    }
    for name in variants:
        pooled[f"consensus_{name}"] = [0.0, 0, 0, 0]
    n_three = 0
    for observation, by_copy in copies.items():
        ids = sorted(by_copy)
        if len(ids) < 2:
            continue
        for pred_id, gt_id in permutations(ids, 2):
            parts = pq_parts(overlap(by_copy[gt_id], by_copy[pred_id])[0], len(by_copy[gt_id]), len(by_copy[pred_id]))
            for index, value in enumerate(parts):
                pooled["human_vs_human_all"][index] += value
                if len(ids) == 3:
                    pooled["human_vs_human_three_copy"][index] += value
        if len(ids) != 3:
            continue
        n_three += 1
        for held_out in ids:
            first, second = [by_copy[i] for i in ids if i != held_out]
            gt = by_copy[held_out]
            for name, (mode, merge) in variants.items():
                pred = consensus(first, second, mode, merge)
                parts = pq_parts(overlap(gt, pred)[0], len(gt), len(pred))
                for index, value in enumerate(parts):
                    pooled[f"consensus_{name}"][index] += value
    report = {"three_copy_observations": n_three}
    for name, parts in pooled.items():
        report[name] = {
            "pq": round(pq_from(*parts), 4),
            "tp": parts[1],
            "fp": parts[2],
            "fn": parts[3],
            "mean_tp_iou": round(parts[0] / parts[1], 4) if parts[1] else 0.0,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
