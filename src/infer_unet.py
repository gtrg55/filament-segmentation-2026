#!/usr/bin/env python3
"""Cache D4-averaged consensus-probability maps of the U-Net (float16, 2048 x 2048 per image)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cache_views import observation_images
from filament_seg.d4 import VIEWS, forward, inverse
from filament_seg.unet import FilamentUNet


@torch.no_grad()
def probability_map(model: FilamentUNet, image: np.ndarray, device: str, views: list[tuple[int, bool]]) -> np.ndarray:
    total = np.zeros(image.shape, dtype=np.float32)
    for k, flip in views:
        view = forward(image, k, flip).astype(np.float32) / 255.0
        logits = model(torch.from_numpy(view)[None, None].to(device))
        prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        total += inverse(prob, k, flip)
    return total / len(views)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--fold-csv", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--views", default="all")
    args = parser.parse_args()

    views = VIEWS if args.views == "all" else [VIEWS[int(i)] for i in args.views.split(",")]
    model = FilamentUNet().to(args.device)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    paths = observation_images(args)
    for index, path in enumerate(paths, start=1):
        target = args.cache_dir / f"{path.stem}.npy"
        if target.exists():
            continue
        image = np.asarray(Image.open(path).convert("L"))
        np.save(target, probability_map(model, image, args.device, views).astype(np.float16))
        print(f"cached={index}/{len(paths)} {path.name}", flush=True)


if __name__ == "__main__":
    main()
