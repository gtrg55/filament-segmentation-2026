"""Run a YOLO-seg checkpoint on dihedral views and cache the instances in the image frame.

For every image and view the model is applied to the transformed image; each predicted mask is mapped
back to the original orientation and stored as COCO RLE together with its confidence. The per-image
pickle holds `views -> [(rle, confidence), ...]` so fusion and post-processing can be iterated on
without touching the GPU again.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .d4 import VIEWS, forward, inverse
from .rle import SIZE, encode


def load_gray_rgb(path: Path) -> np.ndarray:
    """The images are grayscale JPEGs; the detector expects three identical channels."""
    return np.asarray(Image.open(path).convert("RGB"))


def view_instances(
    model, image: np.ndarray, k: int, flip: bool, imgsz: int, conf: float, iou: float, max_det: int,
    device: str, retina_masks: bool = False,
) -> list[tuple[dict, float]]:
    view = forward(image, k, flip)
    result = model.predict(
        view[:, :, ::-1].copy(),  # Ultralytics expects BGR numpy input
        imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False, retina_masks=retina_masks,
    )[0]
    instances: list[tuple[dict, float]] = []
    if result.masks is None or result.boxes is None:
        return instances
    masks = result.masks.data.detach().cpu().numpy().astype(np.uint8)
    scores = result.boxes.conf.detach().cpu().numpy()
    for mask, score in zip(masks, scores):
        if mask.shape != (SIZE, SIZE):
            mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        full = inverse(mask.astype(bool), k, flip)
        if full.any():
            instances.append((encode(full), float(score)))
    return instances


def cache_images(
    weights: Path, image_paths: list[Path], cache_dir: Path, imgsz: int = 2048, conf: float = 0.03,
    iou: float = 0.0, max_det: int = 100, device: str = "mps", views: list[tuple[int, bool]] | None = None,
    retina_masks: bool = False, log=print,
) -> None:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    views = VIEWS if views is None else views
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(image_paths, start=1):
        target = cache_dir / f"{path.stem}.pkl"
        if target.exists():
            continue
        image = load_gray_rgb(path)
        per_view = [
            view_instances(model, image, k, flip, imgsz, conf, iou, max_det, device, retina_masks) for k, flip in views
        ]
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(per_view))
        tmp.rename(target)
        log(f"cached={index}/{len(image_paths)} {path.name} views={[len(v) for v in per_view]}")


def read_cache(cache_dir: Path, observation: str, view_indices: list[int] | None = None) -> list[list[tuple[dict, float]]]:
    per_view = pickle.loads((cache_dir / f"{observation}.pkl").read_bytes())
    return per_view if view_indices is None else [per_view[i] for i in view_indices]
