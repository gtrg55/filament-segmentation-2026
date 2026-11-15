"""Mask post-processing and COCO compressed-RLE helpers."""

from __future__ import annotations

import numpy as np


def remove_instance_overlaps(
    masks: np.ndarray, scores: np.ndarray, min_area: int = 5
) -> np.ndarray:
    """Give overlapping pixels to the highest-confidence instance and drop tiny remnants."""
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float32)
    if masks.ndim != 3:
        raise ValueError(f"expected masks shaped NxHxW, got {masks.shape}")
    if scores.shape != (masks.shape[0],):
        raise ValueError(f"expected {masks.shape[0]} scores, got {scores.shape}")
    if min_area < 1:
        raise ValueError("min_area must be positive")
    occupied = np.zeros(masks.shape[1:], dtype=bool)
    kept: list[np.ndarray] = []
    for index in np.argsort(-scores, kind="stable"):
        clean = masks[index] & ~occupied
        if int(clean.sum()) < min_area:
            continue
        kept.append(clean)
        occupied |= clean
    if not kept:
        return np.zeros((0, *masks.shape[1:]), dtype=bool)
    return np.stack(kept)


def mask_to_rle(mask: np.ndarray) -> str:
    """Encode one binary mask to the compressed COCO counts string used by Kaggle."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("pycocotools is required for compressed COCO RLE") from exc
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("mask must be a non-empty 2-D array")
    encoded = mask_utils.encode(np.asfortranarray(binary))
    counts = encoded["counts"]
    return counts.decode("utf-8") if isinstance(counts, bytes) else str(counts)


def rle_to_mask(counts: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    """Decode one compressed COCO counts string and enforce the expected geometry."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("pycocotools is required for compressed COCO RLE") from exc
    decoded = mask_utils.decode({"size": [height, width], "counts": counts.encode("utf-8")})
    decoded = np.asarray(decoded, dtype=bool)
    if decoded.shape != (height, width):
        raise ValueError(f"decoded shape {decoded.shape} differs from {(height, width)}")
    return decoded
