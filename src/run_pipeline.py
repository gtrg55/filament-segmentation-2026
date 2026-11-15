#!/usr/bin/env python3
"""Run the final pipeline on a directory of test images and write the competition CSV.

Only the pixels of each image are used. Weights and all parameters come from the JSON config
(`configs/final.json`); paths in the config are relative to the repository root.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from filament_seg.pipeline import FilamentPipeline, PipelineConfig


def load_config(path: Path, device: str) -> PipelineConfig:
    root = path.resolve().parents[1]
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["weights"] = [root / w for w in raw["weights"]]
    raw["device"] = device
    return PipelineConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, help="process only the first N images (smoke test)")
    args = parser.parse_args()

    pipeline = FilamentPipeline(load_config(args.config, args.device))
    started = time.time()
    rows = pipeline.predict_directory(args.image_dir, args.out, limit=args.limit)
    print(f"wrote {rows} instances to {args.out} in {time.time() - started:.0f}s on {pipeline.device}")


if __name__ == "__main__":
    main()
