#!/usr/bin/env python3
"""Two-stage YOLO11-seg training for one cross-validation fold.

Stage 1 trains from the COCO-pretrained checkpoint at 1024 px; stage 2 fine-tunes the stage-1
`last.pt` at the native 2048 px with a low learning rate and no mosaic. Both stages use fixed
epoch counts and `last.pt`, so the fold's held-out observations never influence checkpoint
selection and its out-of-fold predictions stay honest. The recipe is the one of the original
single model (40 epochs at 1024 px, then 2048 px fine-tuning), only the data subset differs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def start_model(args: argparse.Namespace, start: Path, first_stage: bool):
    """Stage-1 model: the `--arch` YAML initialised from `--init` if given, else the `--pretrained` checkpoint."""
    from ultralytics import YOLO

    if first_stage and args.arch:
        return YOLO(args.arch).load(str(args.init))
    return YOLO(str(start))


def run_stage(args: argparse.Namespace, name: str, start: Path, fresh: dict, first_stage: bool = False) -> Path:
    """Train one stage, resuming an interrupted run and skipping a finished one."""
    from ultralytics import YOLO

    run_dir = args.project.resolve() / name
    weights = run_dir / "weights" / "last.pt"
    marker = args.project.resolve() / f"{name}_done.json"
    if marker.exists():
        return weights
    if weights.exists():
        YOLO(str(weights)).train(resume=True)
    else:
        start_model(args, start, first_stage).train(
            project=str(args.project.resolve()), name=name, exist_ok=True, seed=args.seed_base + args.fold,
            deterministic=True, pretrained=True, overlap_mask=True, mask_ratio=4, val=False, plots=False,
            patience=1000, workers=args.workers, **fresh,
        )
    marker.write_text(json.dumps({"stage": name}), encoding="utf-8")
    return weights


def stage1(args: argparse.Namespace, data: Path, device: str) -> Path:
    fresh = dict(
        data=str(data), epochs=args.epochs1, imgsz=1024, batch=4, device=device, optimizer="auto", lr0=0.01,
        lrf=0.01, warmup_epochs=3.0, mosaic=1.0, scale=0.5, translate=0.1, hsv_s=0.7, hsv_v=0.4,
    )
    return run_stage(args, f"{args.tag}_s1", args.pretrained, fresh, first_stage=True)


def stage2(args: argparse.Namespace, data: Path, device: str, start: Path) -> Path:
    fresh = dict(
        data=str(data), epochs=args.epochs2, imgsz=2048, batch=1, device=device, optimizer="AdamW", lr0=1e-4,
        lrf=0.2, warmup_epochs=0.0, mosaic=0.0, scale=0.1, translate=0.02,
    )
    return run_stage(args, f"{args.tag}_s2", start, fresh)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="fold index; also offsets the seed")
    parser.add_argument("--tag", help="run name and dataset sub-directory (default fold<N>)")
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, default=Path("yolo11s-seg.pt"))
    parser.add_argument("--arch", help="model YAML (e.g. yolo11m-seg.yaml) built for stage 1 instead of --pretrained")
    parser.add_argument("--init", type=Path, help="checkpoint whose matching tensors initialise the --arch model")
    parser.add_argument("--epochs1", type=int, default=40)
    parser.add_argument("--epochs2", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if bool(args.arch) != bool(args.init):
        parser.error("--arch and --init go together")
    args.tag = args.tag or f"fold{args.fold}"

    data = args.folds_dir.resolve() / args.tag / "data.yaml"
    device = pick_device(args.device)
    record = {"fold": args.fold, "tag": args.tag, "device": device, "epochs1": args.epochs1, "epochs2": args.epochs2}
    if args.arch:
        record.update(arch=args.arch, init=str(args.init))
    started = time.time()
    last1 = stage1(args, data, device)
    record["stage1_seconds"] = round(time.time() - started)
    started = time.time()
    last2 = stage2(args, data, device, last1)
    record["stage2_seconds"] = round(time.time() - started)
    record["weights"] = str(last2)
    args.project.mkdir(parents=True, exist_ok=True)
    (args.project / f"{args.tag}_done.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
