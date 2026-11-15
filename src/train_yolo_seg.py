#!/usr/bin/env python3
"""Train a reproducible YOLO instance-segmentation baseline for MAGFiLO."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/segment"))
    parser.add_argument("--name", default="yolo11s_seg_scene_v1")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"training_device={device}", flush=True)
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        seed=42,
        deterministic=True,
        pretrained=True,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        mosaic=args.mosaic,
        scale=args.scale,
        translate=args.translate,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        overlap_mask=True,
        mask_ratio=4,
        plots=True,
    )


if __name__ == "__main__":
    main()
