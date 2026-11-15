#!/usr/bin/env python3
"""Research diagnostics: human inter-annotator PQ ceiling and error decomposition of a prediction CSV.

Part A rasterizes the training annotations and scores every annotator against every other
annotator of the same observation with the organizer PQ rule (IoU strictly above 0.5).
Part B decomposes a validation prediction CSV against gt_val.csv into near-misses,
fragmentation (one GT split into several predictions), over-merging and background FPs,
and reports what PQ each failure mode costs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

SIZE = 2048
AREA_BINS = [0, 500, 2000, 6000, 20000, 10**9]


def area_bin(area: float) -> str:
    for low, high in zip(AREA_BINS[:-1], AREA_BINS[1:]):
        if low <= area < high:
            return f"{low}-{high if high < 10**9 else 'inf'}"
    return "?"


def rle_from_counts(counts: str) -> dict:
    return {"size": [SIZE, SIZE], "counts": counts.encode("ascii")}


def read_csv_rles(path: Path) -> dict[str, list[dict]]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id, _ = row["filament_id"].rsplit("_", 1)
            by_image[image_id].append(rle_from_counts(row["segmentation_rle"]))
    return by_image


def overlap(gt: list[dict], pred: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return IoU (gt x pred), intersection (gt x pred), gt areas, pred areas."""
    gt_area = mask_utils.area(gt).astype(np.float64) if gt else np.zeros(0)
    pred_area = mask_utils.area(pred).astype(np.float64) if pred else np.zeros(0)
    if not gt or not pred:
        shape = (len(gt), len(pred))
        return np.zeros(shape), np.zeros(shape), gt_area, pred_area
    iou = np.asarray(mask_utils.iou(pred, gt, [0] * len(gt)), dtype=np.float64).T
    inter = iou * (gt_area[:, None] + pred_area[None, :]) / (1.0 + iou)
    return iou, inter, gt_area, pred_area


def pq_parts(iou: np.ndarray, n_gt: int, n_pred: int) -> tuple[float, int, int, int]:
    if n_gt == 0:
        return 0.0, 0, n_pred, 0
    if n_pred == 0:
        return 0.0, 0, 0, n_gt
    hit = iou > 0.5
    return float(iou[hit].sum()), int(hit.sum()), int((hit.sum(axis=0) == 0).sum()), int((hit.sum(axis=1) == 0).sum())


def pq_from(total: float, tp: int, fp: int, fn: int) -> float:
    denominator = tp + 0.5 * fp + 0.5 * fn
    return total / denominator if denominator else 0.0


def inter_annotator(annotation_path: Path) -> dict:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    observation_of = {image["id"]: image["file_name"].rsplit(".", 1)[0] for image in data["images"]}
    by_annotator_image: dict[str, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        polygons = annotation["segmentation"]
        if isinstance(polygons, str):
            polygons = json.loads(polygons)
        polygons = [p for p in polygons if len(p) >= 6]
        if not polygons:
            continue
        rle = mask_utils.merge(mask_utils.frPyObjects(polygons, SIZE, SIZE))
        by_annotator_image[annotation["image_id"]].append(rle)
    annotators_of: dict[str, list[str]] = defaultdict(list)
    for image_id in observation_of:
        annotators_of[observation_of[image_id]].append(image_id)

    pooled = [0.0, 0, 0, 0]
    by_observatory: dict[str, list] = defaultdict(lambda: [0.0, 0, 0, 0])
    per_pair_pq: list[float] = []
    multi = {obs: ids for obs, ids in annotators_of.items() if len(ids) >= 2}
    for observation, image_ids in multi.items():
        for pred_id, gt_id in permutations(image_ids, 2):
            gt = by_annotator_image.get(gt_id, [])
            pred = by_annotator_image.get(pred_id, [])
            iou, _, _, _ = overlap(gt, pred)
            parts = pq_parts(iou, len(gt), len(pred))
            per_pair_pq.append(pq_from(*parts))
            for bucket in (pooled, by_observatory[observation[-2:]]):
                for index, value in enumerate(parts):
                    bucket[index] += value
    instance_counts = [len(by_annotator_image.get(i, [])) for ids in multi.values() for i in ids]
    return {
        "multi_annotated_observations": len(multi),
        "ordered_annotator_pairs": len(per_pair_pq),
        "human_vs_human_pq": pq_from(*pooled),
        "tp": pooled[1],
        "fp": pooled[2],
        "fn": pooled[3],
        "mean_tp_iou": pooled[0] / pooled[1] if pooled[1] else 0.0,
        "median_pair_pq": float(np.median(per_pair_pq)),
        "mean_instances_per_annotator_image": float(np.mean(instance_counts)),
        "by_observatory": {k: round(pq_from(*v), 4) for k, v in sorted(by_observatory.items())},
    }


def decompose(gt_path: Path, pred_path: Path) -> dict:
    ground_truth = read_csv_rles(gt_path)
    predictions = read_csv_rles(pred_path)
    totals = [0.0, 0, 0, 0]
    gt_best = defaultdict(int)
    fn_by_area = defaultdict(int)
    gt_by_area = defaultdict(int)
    fp_by_area = defaultdict(int)
    fp_kind = defaultdict(int)
    near_miss_gain = 0.0
    near_miss_count = 0
    split_gt = 0
    split_union_tp = 0
    split_union_iou_sum = 0.0
    merged_pred = 0
    background_fp = 0
    for annotator_image_id, gt in ground_truth.items():
        pred = predictions.get(annotator_image_id.split("-", 1)[1], [])
        iou, inter, gt_area, pred_area = overlap(gt, pred)
        parts = pq_parts(iou, len(gt), len(pred))
        for index, value in enumerate(parts):
            totals[index] += value
        hit = iou > 0.5 if iou.size else np.zeros((len(gt), len(pred)), dtype=bool)
        for g in range(len(gt)):
            best = float(iou[g].max()) if len(pred) else 0.0
            gt_by_area[area_bin(gt_area[g])] += 1
            label = (
                "tp>0.5" if best > 0.5 else "near0.3-0.5" if best > 0.3 else "weak0.1-0.3" if best > 0.1 else "missed<0.1"
            )
            gt_best[label] += 1
            if best <= 0.5:
                fn_by_area[area_bin(gt_area[g])] += 1
            if 0.3 < best <= 0.5:
                near_miss_count += 1
                near_miss_gain += best
            # fragmentation: >=2 predictions lying mostly (>50% of their area) inside this GT
            if len(pred):
                inside = np.where(inter[g] > 0.5 * np.maximum(pred_area, 1))[0]
                if len(inside) >= 2 and best <= 0.5:
                    split_gt += 1
                    union = mask_utils.merge([pred[i] for i in inside])
                    union_iou = float(np.asarray(mask_utils.iou([union], [gt[g]], [0]))[0, 0])
                    if union_iou > 0.5:
                        split_union_tp += 1
                        split_union_iou_sum += union_iou
        for p in range(len(pred)):
            if len(gt) and hit[:, p].any():
                continue
            fp_by_area[area_bin(pred_area[p])] += 1
            best = float(iou[:, p].max()) if len(gt) else 0.0
            covered = inter[:, p] > 0.5 * np.maximum(gt_area, 1) if len(gt) else np.zeros(0, dtype=bool)
            if covered.sum() >= 2:
                merged_pred += 1
                fp_kind["over_merge(>=2 GT mostly inside)"] += 1
            elif best > 0.3:
                fp_kind["near_miss_pred0.3-0.5"] += 1
            elif best > 0.05:
                fp_kind["partial_overlap0.05-0.3"] += 1
            else:
                background_fp += 1
                fp_kind["background<0.05"] += 1
    total, tp, fp, fn = totals
    base = pq_from(total, tp, fp, fn)
    # what-if scenarios (approximate, each applied alone)
    no_background = pq_from(total, tp, fp - background_fp, fn)
    near_fixed = pq_from(total + near_miss_count * 0.55, tp + near_miss_count, max(fp - near_miss_count, 0), fn - near_miss_count)
    splits_merged = pq_from(total + split_union_iou_sum, tp + split_union_tp, fp, fn - split_union_tp)
    return {
        "pq": base,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_tp_iou(SQ)": total / tp if tp else 0.0,
        "RQ": tp / (tp + 0.5 * fp + 0.5 * fn) if tp else 0.0,
        "gt_best_iou_bins": dict(gt_best),
        "fn_by_gt_area": dict(sorted(fn_by_area.items())),
        "gt_by_area": dict(sorted(gt_by_area.items())),
        "fp_by_pred_area": dict(sorted(fp_by_area.items())),
        "fp_kinds": dict(fp_kind),
        "fragmented_gt(>=2 preds inside, no TP)": split_gt,
        "what_if": {
            "drop_background_fps": no_background,
            "near_misses_become_tp_at_0.55": near_fixed,
            "merge_fragments_into_one_pred": splits_merged,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report: dict = {}
    if args.annotations:
        report["inter_annotator"] = inter_annotator(args.annotations)
    if args.ground_truth and args.predictions:
        report["decomposition"] = decompose(args.ground_truth, args.predictions)
        report["decomposition"]["predictions"] = str(args.predictions)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
