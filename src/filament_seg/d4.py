"""Dihedral (D4) test-time views: 4 rotations x optional horizontal flip."""

from __future__ import annotations

import numpy as np

VIEWS: list[tuple[int, bool]] = [(k, flip) for flip in (False, True) for k in range(4)]  # (rot90 count, flip)


def forward(image: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(image) if flip else image
    return np.ascontiguousarray(np.rot90(out, k))


def inverse(mask: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(mask, -k)
    return np.ascontiguousarray(np.fliplr(out) if flip else out)
