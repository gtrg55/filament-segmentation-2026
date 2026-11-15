"""Mask post-processing applied to fused instances before export."""

from __future__ import annotations

import cv2
import numpy as np

from .rle import RLE, encode


def erode(mask: np.ndarray, size: int) -> np.ndarray:
    """Erode with an elliptical structuring element (size 1 is the identity)."""
    if size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def components(mask: np.ndarray, min_area: int = 1) -> list[np.ndarray]:
    """Connected components (8-neighbourhood) with at least `min_area` pixels, largest first."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    order = sorted(range(1, count), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    return [labels == i for i in order if stats[i, cv2.CC_STAT_AREA] >= min_area]


def fill_holes(mask: np.ndarray, max_hole: int) -> np.ndarray:
    """Fill background holes of at most `max_hole` pixels that are enclosed by the mask."""
    inverse = (~mask).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=4)
    out = mask.copy()
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])).tolist())
    for index in range(1, count):
        if index not in border and stats[index, cv2.CC_STAT_AREA] <= max_hole:
            out |= labels == index
    return out


def smooth(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of the mask followed by a 0.5 threshold (removes jagged vote edges)."""
    if sigma <= 0:
        return mask
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return blurred >= 0.5


def distance_shrink(mask: np.ndarray, radius: float, relative: float = 0.0, keep_fraction: float = 0.0) -> np.ndarray:
    """Remove boundary pixels closer than `radius` to the background (Euclidean distance transform).

    `radius = 1.0` equals the 3x3 elliptical erosion, `1.5` the 3x3 square, `2.0` roughly the 5x5 ellipse.
    `relative` additionally removes the outer fraction of the instance's maximum thickness, so fat blobs
    shrink more than thin filaments. If the shrunk mask keeps less than `keep_fraction` of the pixels the
    radius is halved once and, failing that, the original mask is returned (thin filaments must survive).
    """
    padded = np.pad(mask.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    cut = max(radius, relative * float(distance.max()))
    out = distance > cut if cut >= 1.0 else mask  # boundary pixels have distance 1
    if keep_fraction and out.sum() < keep_fraction * mask.sum():
        out = distance > max(1.0, cut / 2)
        if out.sum() < keep_fraction * mask.sum():
            out = mask
    return out


def to_rle(mask: np.ndarray) -> RLE | None:
    return encode(mask) if mask.any() else None
