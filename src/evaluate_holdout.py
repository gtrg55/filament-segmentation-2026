#!/usr/bin/env python3
"""Score a prediction CSV against ground-truth copies with the organizer metrics and a bootstrap CI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from filament_seg.metrics import bootstrap_pq, observation_parts, score_split, summarize
from filament_seg.rle import read_instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True, help="CSV with <annotator>-<observation>_<k> ids")
    parser.add_argument("--predictions", type=Path, required=True, help="submission CSV with <observation>_<k> ids")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scores = score_split(read_instances(args.ground_truth), read_instances(args.predictions))
    summary = summarize(scores)
    _, parts = observation_parts(scores)
    point, low, high = bootstrap_pq(parts)
    report = {**summary, "pq_ci95": [low, high], "observations": len(parts), "annotator_copies": len(scores)}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
