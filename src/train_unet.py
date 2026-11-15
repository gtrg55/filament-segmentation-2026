#!/usr/bin/env python3
"""Train the consensus-probability U-Net on random full-resolution crops.

Each sample is a crop of one annotator copy; its target is that copy's union mask. Different copies of the
same image therefore pull the output towards the fraction of annotators that outlined a pixel.
`--holdout-fold k` removes fold k of the training rows (for out-of-fold predictions); the competition
holdout (`split == val`) is never used for training.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils

from filament_seg.d4 import forward
from filament_seg.unet import FilamentUNet, soft_dice_loss

SIZE = 2048


def load_rows(manifest: Path, fold_csv: Path | None, holdout_fold: int | None) -> list[dict]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    if holdout_fold is not None:
        with fold_csv.open(newline="", encoding="utf-8") as handle:
            fold_of = {row["observation_id"]: int(row["fold"]) for row in csv.DictReader(handle)}
        rows = [row for row in rows if fold_of[row["observation_id"]] != holdout_fold]
    return rows


def union_masks(annotation_path: Path, image_ids: set[str]) -> dict[str, np.ndarray]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    per_image: dict[str, list] = {}
    for annotation in data["annotations"]:
        image_id = str(annotation["image_id"])
        if image_id not in image_ids:
            continue
        polygons = annotation["segmentation"]
        if isinstance(polygons, str):
            polygons = json.loads(polygons)
        polygons = [p for p in polygons if len(p) >= 6]
        if polygons:
            per_image.setdefault(image_id, []).append(mask_utils.merge(mask_utils.frPyObjects(polygons, SIZE, SIZE)))
    return {
        image_id: mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8) for image_id, rles in per_image.items()
    }


class CropSampler:
    def __init__(self, rows: list[dict], image_dir: Path, masks: dict[str, np.ndarray], crop: int, seed: int) -> None:
        self.rows = [row for row in rows if row["image_id"] in masks]
        self.crop = crop
        self.rng = np.random.default_rng(seed)
        self.images: dict[str, np.ndarray] = {}
        for row in self.rows:
            name = row["file_name"]
            if name not in self.images:
                self.images[name] = np.asarray(Image.open(image_dir / name).convert("L"))
        self.masks = masks

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        images, targets = [], []
        for _ in range(batch):
            row = self.rows[int(self.rng.integers(len(self.rows)))]
            image, mask = self.images[row["file_name"]], self.masks[row["image_id"]]
            y = int(self.rng.integers(0, SIZE - self.crop + 1))
            x = int(self.rng.integers(0, SIZE - self.crop + 1))
            k, flip = int(self.rng.integers(4)), bool(self.rng.integers(2))
            crop_image = forward(image[y : y + self.crop, x : x + self.crop], k, flip).astype(np.float32) / 255.0
            crop_mask = forward(mask[y : y + self.crop, x : x + self.crop], k, flip).astype(np.float32)
            gamma = float(self.rng.uniform(0.8, 1.25))
            gain = float(self.rng.uniform(0.9, 1.1))
            images.append(np.clip(gain * crop_image**gamma, 0, 1))
            targets.append(crop_mask)
        return torch.from_numpy(np.stack(images))[:, None], torch.from_numpy(np.stack(targets))[:, None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--fold-csv", type=Path)
    parser.add_argument("--holdout-fold", type=int)
    parser.add_argument("--encoder-weights", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--crop", type=int, default=768)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rows = load_rows(args.manifest, args.fold_csv, args.holdout_fold)
    masks = union_masks(args.annotations, {row["image_id"] for row in rows})
    sampler = CropSampler(rows, args.image_dir, masks, args.crop, args.seed)
    print(f"copies={len(sampler.rows)} images={len(sampler.images)}", flush=True)

    model = FilamentUNet(args.encoder_weights).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * args.steps_per_epoch
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for _ in range(args.steps_per_epoch):
            images, targets = sampler.sample(args.batch)
            images, targets = images.to(args.device), targets.to(args.device)
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, targets) + soft_dice_loss(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            running += float(loss.detach())
            step += 1
        print(f"epoch {epoch}/{args.epochs} loss {running / args.steps_per_epoch:.4f} elapsed {time.time() - started:.0f}s",
              flush=True)
        torch.save(model.state_dict(), args.out / "last.pt")
    (args.out / "done.json").write_text(json.dumps({"epochs": args.epochs, "seconds": round(time.time() - started)}))


if __name__ == "__main__":
    main()
