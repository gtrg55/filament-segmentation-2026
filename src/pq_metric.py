"""Numpy reproduction of the organizer's August 2026 PQ implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PQResult:
    pq: float
    sum_tp_iou: float
    true_positive_pairs: int
    false_positive_instances: int
    false_negative_instances: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _validate_stack(masks: np.ndarray, name: str) -> np.ndarray:
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"{name} must have shape NxHxW, got {masks.shape}")
    return masks


def pairwise_iou_dice(gt_masks: np.ndarray, pred_masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise overlap matrices without materializing all mask pairs at once."""
    gt_masks = _validate_stack(gt_masks, "gt_masks")
    pred_masks = _validate_stack(pred_masks, "pred_masks")
    if gt_masks.shape[1:] != pred_masks.shape[1:]:
        raise ValueError("GT and prediction mask geometry differs")
    iou = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float64)
    dice = np.zeros_like(iou)
    gt_areas = gt_masks.reshape(len(gt_masks), -1).sum(axis=1)
    pred_areas = pred_masks.reshape(len(pred_masks), -1).sum(axis=1)
    for gt_index, gt in enumerate(gt_masks):
        for pred_index, pred in enumerate(pred_masks):
            intersection = int(np.count_nonzero(gt & pred))
            union = int(gt_areas[gt_index] + pred_areas[pred_index] - intersection)
            total = int(gt_areas[gt_index] + pred_areas[pred_index])
            if union:
                iou[gt_index, pred_index] = intersection / union
            if total:
                dice[gt_index, pred_index] = 2 * intersection / total
    return iou, dice


def pq_score(
    gt_by_image: list[np.ndarray], pred_by_image: list[np.ndarray], iou_threshold: float = 0.5
) -> PQResult:
    """Match the organizer rule: every pair with IoU strictly above 0.5 is a TP hit."""
    if len(gt_by_image) != len(pred_by_image):
        raise ValueError("GT and prediction image counts differ")
    tp_ious: list[float] = []
    fp = 0
    fn = 0
    for gt_masks, pred_masks in zip(gt_by_image, pred_by_image, strict=True):
        gt_masks = _validate_stack(gt_masks, "gt_masks")
        pred_masks = _validate_stack(pred_masks, "pred_masks")
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            raise ValueError("GT and prediction mask geometry differs")
        if len(gt_masks) == 0:
            fp += len(pred_masks)
            continue
        if len(pred_masks) == 0:
            fn += len(gt_masks)
            continue
        iou, _ = pairwise_iou_dice(gt_masks, pred_masks)
        hit = iou > iou_threshold
        tp_ious.extend(iou[hit].tolist())
        fp += int(np.count_nonzero(hit.sum(axis=0) == 0))
        fn += int(np.count_nonzero(hit.sum(axis=1) == 0))
    tp = len(tp_ious)
    denominator = tp + 0.5 * fp + 0.5 * fn
    total_iou = float(sum(tp_ious))
    return PQResult(
        pq=total_iou / denominator if denominator else 0.0,
        sum_tp_iou=total_iou,
        true_positive_pairs=tp,
        false_positive_instances=fp,
        false_negative_instances=fn,
    )
