"""Instance voting across test-time views and models.

Every source (a model run on one dihedral view, mapped back to the image frame) contributes a list
of `(RLE, confidence)` instances. Instances are clustered across sources by IoU, one member per
source per cluster. A cluster becomes one instance: its mask is the pixel vote of the members, its
score the sum of member confidences divided by the number of sources (a missing source counts as
zero confidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_utils

from .rle import RLE, SIZE, decode, encode


@dataclass
class Fused:
    rle: RLE
    score: float  # sum of member confidences / number of sources
    support: int  # members in the cluster
    confidences: list[float] = field(default_factory=list)


def cluster_instances(
    sources: list[list[tuple[RLE, float]]], cluster_iou: float = 0.3, min_conf: float = 0.03
) -> list[list[tuple[int, RLE, float]]]:
    """Greedy clustering in descending confidence; a cluster takes at most one member per source."""
    pool = [(rle, conf, source) for source, items in enumerate(sources) for rle, conf in items if conf >= min_conf]
    pool.sort(key=lambda item: -item[1])
    clusters: list[list[tuple[int, RLE, float]]] = []
    seeds: list[RLE] = []
    for rle, conf, source in pool:
        target = None
        if seeds:
            ious = np.asarray(mask_utils.iou([rle], seeds, [0] * len(seeds)))[0]
            for index in np.argsort(-ious):
                if ious[index] <= cluster_iou:
                    break
                if all(member[0] != source for member in clusters[index]):
                    target = int(index)
                    break
        if target is None:
            clusters.append([(source, rle, conf)])
            seeds.append(rle)
        else:
            clusters[target].append((source, rle, conf))
    return clusters


def fuse_cluster(members: list[tuple[int, RLE, float]], n_sources: int, vote: float = 0.5) -> Fused | None:
    support = len(members)
    rles = [member[1] for member in members]
    votes = mask_utils.decode(rles).sum(axis=2) if support > 1 else mask_utils.decode(rles[0])
    mask = votes >= max(1, int(np.ceil(vote * support)))
    if not mask.any():
        return None
    confidences = [member[2] for member in members]
    return Fused(encode(mask), float(sum(confidences) / n_sources), support, confidences)


def fuse_sources(
    sources: list[list[tuple[RLE, float]]], cluster_iou: float = 0.3, vote: float = 0.5, min_conf: float = 0.03
) -> list[Fused]:
    fused = []
    for members in cluster_instances(sources, cluster_iou, min_conf):
        item = fuse_cluster(members, len(sources), vote)
        if item is not None:
            fused.append(item)
    return fused


def resolve_overlaps(
    fused: list[Fused], min_score: float, min_area: int = 5, duplicate_fraction: float | None = None
) -> list[tuple[Fused, np.ndarray]]:
    """Give contested pixels to the higher score and drop remnants below `min_area` pixels.

    With `duplicate_fraction` an instance whose pixels are covered by higher-scoring kept masks by more
    than that fraction is a duplicate and is dropped whole instead of surviving as a thin remnant.
    """
    occupied = np.zeros((SIZE, SIZE), dtype=bool)
    kept = []
    for item in sorted((f for f in fused if f.score >= min_score), key=lambda f: -f.score):
        mask = decode(item.rle)
        clean = mask & ~occupied
        if duplicate_fraction is not None and clean.sum() < (1.0 - duplicate_fraction) * mask.sum():
            continue
        if int(clean.sum()) < min_area:
            continue
        occupied |= clean
        kept.append((item, clean))
    return kept
