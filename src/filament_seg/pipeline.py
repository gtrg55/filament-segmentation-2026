"""End-to-end inference: detectors x dihedral views -> instance voting -> post-processing -> RLE.

Only the pixels of the test image are used at inference time: no file-name metadata, no neighbouring
frames and no external annotations. The model weights are the only learned state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .d4 import VIEWS
from .fuse import fuse_sources, resolve_overlaps
from .infer import load_gray_rgb, view_instances
from .post import components, erode, to_rle
from .rle import RLE, write_submission


@dataclass
class PipelineConfig:
    weights: list[Path]
    imgsz: int = 2048
    view_indices: list[int] = field(default_factory=lambda: list(range(len(VIEWS))))
    conf: float = 0.03  # detector confidence floor; the decision threshold is `threshold`
    iou: float = 0.0  # detector NMS IoU (0 disables NMS: overlaps are resolved after fusion)
    max_det: int = 100
    cluster_iou: float = 0.3
    vote: float = 0.5
    threshold: float = 0.18
    erosion: int = 3  # elliptical kernel size, 1 = none
    largest_component: bool = False
    device: str = "auto"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


class FilamentPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        from ultralytics import YOLO

        self.config = config
        self.device = pick_device(config.device)
        self.models = [YOLO(str(path)) for path in config.weights]

    def sources(self, image: np.ndarray) -> list[list[tuple[dict, float]]]:
        """One instance list per (model, view)."""
        c = self.config
        out = []
        for model in self.models:
            for index in c.view_indices:
                k, flip = VIEWS[index]
                out.append(view_instances(model, image, k, flip, c.imgsz, c.conf, c.iou, c.max_det, self.device))
        return out

    def finalize(self, sources: list[list[tuple[dict, float]]]) -> list[RLE]:
        """Fuse cached or fresh sources into the final instances (everything after the detector)."""
        c = self.config
        fused = fuse_sources(sources, c.cluster_iou, c.vote, c.conf)
        instances: list[RLE] = []
        for _, mask in resolve_overlaps(fused, c.threshold):
            mask = erode(mask, c.erosion)
            pieces = components(mask)[:1] if c.largest_component else [mask]
            instances += [rle for rle in (to_rle(p) for p in pieces if p.any()) if rle is not None]
        return instances

    def predict(self, image_rgb: np.ndarray) -> list[RLE]:
        return self.finalize(self.sources(image_rgb))

    def predict_directory(
        self, image_dir: Path, out_csv: Path, log: Callable[[str], None] = print, limit: int | None = None
    ) -> int:
        paths = sorted(image_dir.glob("*.jp*g"))[:limit]
        predictions = {}
        for index, path in enumerate(paths, start=1):
            predictions[path.stem] = self.predict(load_gray_rgb(path))
            log(f"{index}/{len(paths)} {path.name}: {len(predictions[path.stem])} instances")
        return write_submission(out_csv, predictions)
