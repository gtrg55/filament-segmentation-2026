"""Organizer-compatible evaluation of instance predictions.

Mirrors the official self-evaluation notebook: every annotator copy of an observation is scored
against the same prediction. A GT/prediction pair is a true positive when IoU > 0.5, PQ is
`sum(IoU of TP) / (TP + 0.5 FP + 0.5 FN)`. The one-to-many and many-to-one relations count any
overlap (IoU > 0), and the IoU/Dice distributions use all overlapping pairs. On top of the
notebook we add the semantic (union-mask) Dice and an observation-level bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_utils

from .rle import RLE, area, iou_matrix


@dataclass
class CopyScore:
    """Scores of one prediction against one annotator copy."""

    sum_iou: float
    tp: int
    fp: int
    fn: int
    n_gt: int
    n_pred: int
    pair_iou: np.ndarray  # IoU of every overlapping (IoU > 0) GT/prediction pair
    pair_dice: np.ndarray
    gt_links: np.ndarray  # predictions overlapping each GT filament
    pred_links: np.ndarray  # GT filaments overlapping each prediction
    semantic_dice: float  # Dice of the union masks

    @property
    def parts(self) -> np.ndarray:
        return np.array([self.sum_iou, self.tp, self.fp, self.fn], dtype=np.float64)


def semantic_dice(gt: list[RLE], pred: list[RLE]) -> float:
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    gt_union, pred_union = mask_utils.merge(gt), mask_utils.merge(pred)
    inter = float(mask_utils.area(mask_utils.merge([gt_union, pred_union], intersect=True)))
    total = float(mask_utils.area(gt_union)) + float(mask_utils.area(pred_union))
    return 2.0 * inter / total if total else 1.0


def score_copy(gt: list[RLE], pred: list[RLE]) -> CopyScore:
    iou = iou_matrix(gt, pred)
    n_gt, n_pred = len(gt), len(pred)
    hit = iou > 0.5
    overlap = iou > 0
    gt_area, pred_area = area(gt), area(pred)
    if n_gt and n_pred:
        inter = iou * (gt_area[:, None] + pred_area[None, :]) / (1.0 + iou)
        dice = np.where(overlap, 2.0 * inter / (gt_area[:, None] + pred_area[None, :]), 0.0)
    else:
        dice = np.zeros_like(iou)
    return CopyScore(
        sum_iou=float(iou[hit].sum()),
        tp=int(hit.sum()),
        fp=int((hit.sum(axis=0) == 0).sum()) if n_gt else n_pred,
        fn=int((hit.sum(axis=1) == 0).sum()) if n_pred else n_gt,
        n_gt=n_gt,
        n_pred=n_pred,
        pair_iou=iou[overlap],
        pair_dice=dice[overlap],
        gt_links=overlap.sum(axis=1) if n_pred else np.zeros(n_gt, dtype=int),
        pred_links=overlap.sum(axis=0) if n_gt else np.zeros(n_pred, dtype=int),
        semantic_dice=semantic_dice(gt, pred),
    )


def observation_of(copy_id: str) -> str:
    """`<annotator>-<observation>` -> `<observation>`."""
    return copy_id.split("-", 1)[1]


def score_split(
    gt_by_copy: dict[str, list[RLE]], pred_by_observation: dict[str, list[RLE]]
) -> dict[str, CopyScore]:
    """Score every annotator copy against the prediction of its observation."""
    return {
        copy_id: score_copy(gt, pred_by_observation.get(observation_of(copy_id), []))
        for copy_id, gt in gt_by_copy.items()
    }


def pq_from_parts(parts: np.ndarray) -> float:
    sum_iou, tp, fp, fn = parts
    denominator = tp + 0.5 * fp + 0.5 * fn
    return float(sum_iou / denominator) if denominator else 0.0


def summarize(scores: dict[str, CopyScore]) -> dict[str, float]:
    """Pooled PQ (+SQ, RQ), Dice/IoU distribution means and one-to-many / many-to-one rates."""
    values = list(scores.values())
    parts = np.sum([s.parts for s in values], axis=0)
    sum_iou, tp, fp, fn = parts
    pair_iou = np.concatenate([s.pair_iou for s in values]) if values else np.zeros(0)
    pair_dice = np.concatenate([s.pair_dice for s in values]) if values else np.zeros(0)
    gt_links = np.concatenate([s.gt_links for s in values]) if values else np.zeros(0)
    pred_links = np.concatenate([s.pred_links for s in values]) if values else np.zeros(0)
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if tp else 0.0
    return {
        "pq": pq_from_parts(parts),
        "sq": float(sum_iou / tp) if tp else 0.0,
        "rq": float(rq),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "mean_pair_iou": float(pair_iou.mean()) if len(pair_iou) else 0.0,
        "mean_pair_dice": float(pair_dice.mean()) if len(pair_dice) else 0.0,
        "mean_semantic_dice": float(np.mean([s.semantic_dice for s in values])) if values else 0.0,
        "gt_one_to_many": float((gt_links >= 2).mean()) if len(gt_links) else 0.0,
        "pred_many_to_one": float((pred_links >= 2).mean()) if len(pred_links) else 0.0,
        "gt_unmatched": float((gt_links == 0).mean()) if len(gt_links) else 0.0,
        "pred_unmatched": float((pred_links == 0).mean()) if len(pred_links) else 0.0,
        "n_pred_per_copy": float(np.mean([s.n_pred for s in values])) if values else 0.0,
    }


def observation_parts(scores: dict[str, CopyScore]) -> tuple[list[str], np.ndarray]:
    """Per-observation [sum IoU, TP, FP, FN] (copies pooled), for resampling by observation."""
    grouped: dict[str, np.ndarray] = {}
    for copy_id, score in scores.items():
        observation = observation_of(copy_id)
        grouped[observation] = grouped.get(observation, np.zeros(4)) + score.parts
    names = sorted(grouped)
    return names, np.stack([grouped[name] for name in names])


def bootstrap_pq(parts: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Observation-level bootstrap of pooled PQ: (point estimate, 2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(parts), size=(n_boot, len(parts)))
    resampled = parts[draws].sum(axis=1)
    values = np.array([pq_from_parts(row) for row in resampled])
    return pq_from_parts(parts.sum(axis=0)), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_bootstrap(
    parts_new: np.ndarray, parts_ref: np.ndarray, n_boot: int = 4000, seed: int = 0
) -> tuple[float, float, float, float]:
    """Paired observation-level bootstrap of the PQ gain of `new` over `ref`.

    Returns (gain, 2.5%, 97.5%, share of resamples with gain <= 0). Both arrays must list the same
    observations in the same order.
    """
    if parts_new.shape != parts_ref.shape:
        raise ValueError("parts must cover the same observations in the same order")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(parts_new), size=(n_boot, len(parts_new)))
    new = parts_new[draws].sum(axis=1)
    ref = parts_ref[draws].sum(axis=1)
    gains = np.array([pq_from_parts(a) - pq_from_parts(b) for a, b in zip(new, ref)])
    gain = pq_from_parts(parts_new.sum(axis=0)) - pq_from_parts(parts_ref.sum(axis=0))
    return gain, float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975)), float((gains <= 0).mean())
