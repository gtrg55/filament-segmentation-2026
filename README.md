# Solar filament instance segmentation (Filament Segmentation Challenge 2026)

Honest instance segmentation of solar filaments in GONG H-alpha images (MAGFiLO dataset): a vote over
6 YOLO11s-seg detectors and 8 dihedral test-time views, followed by overlap resolution, a 3x3
elliptical erosion and a largest-component clean-up. The detectors are trained **only on the provided training
annotations** (COCO-pretrained YOLO11 initialisation); inference uses **only the pixels of the test image** (no file-name
metadata, no neighbouring frames, no external annotations).

On an observation-grouped holdout (never used for training) the pipeline reaches a Panoptic Quality of
**0.425** (95% bootstrap CI [0.404, 0.444]), above the 0.340 agreement between independent experts
measured with the same metric.

## Results

Holdout of 141 physical observations (229 annotator copies), organizer-compatible PQ
(every annotator copy scored separately against the prediction of its observation, pooled). 95% intervals and
paired gains are observation-level bootstraps (4000 draws).

| Stage | sources | PQ [95% CI] | dPQ vs previous [95% CI] | Dice | one-to-many | many-to-one |
|---|---|---|---|---|---|---|
| 1 detector, 1 view | 1 | 0.3815 [0.3607, 0.4026] | - | 0.6141 | 0.0317 | 0.0344 |
| + 3x3 erosion | 1 | 0.4046 [0.3839, 0.4259] | +0.0231 [+0.0157, +0.0309] | 0.6208 | 0.0317 | 0.0330 |
| + 8 dihedral views | 8 | 0.4232 [0.4020, 0.4428] | +0.0186 [+0.0078, +0.0288] | 0.6316 | 0.0379 | 0.0323 |
| + 5 more detectors | 48 | 0.4252 [0.4050, 0.4442] | +0.0020 [-0.0047, +0.0090] | 0.6557 | 0.0354 | 0.0295 |
| **+ largest component** | 48 | 0.4246 [0.4043, 0.4436] | -0.0006 [-0.0043, +0.0029] | 0.6493 | 0.0335 | 0.0179 |

Expert agreement (each annotator scored against the others of the same observation, 296 observations): PQ
0.340; a consensus of two experts scored against the third: 0.31-0.34. The score threshold, the erosion and the
largest-component step were tuned on out-of-fold predictions of the 566 fit observations (one detector, 8 views: PQ
0.4336 at the chosen operating point; the optimum is flat for thresholds between 0.16 and 0.22). Split-half
cross-fitting on the holdout puts the selection optimism of a threshold/erosion choice at 0.002 PQ.
Every number is produced by the scripts in `src/` from cached detector output; see `results/` and the report in
`docs/report/`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_pipeline.py --config configs/final.json --image-dir <folder with the 180 test JPEGs> --out submission.csv
```

`configs/second.json` is the second selected submission: the same recipe and post-processing applied to two detectors
(`weights/all_s0.pt`, `all_s1.pt`) refit on all 707 observations including the holdout, so it has no honest holdout score
(on the test set its predictions agree with the first submission at PQ 0.905 when scored against each other).

`notebooks/demo_pipeline.ipynb` runs the same pipeline step by step (single-view vs fused result, CSV export, format
checks). Weights are stored in `weights/` (about 21 MB each, stripped checkpoints). A CUDA GPU or Apple MPS is used
automatically; on an Apple M4 Max the six detectors x eight views take about 14 s per 2048 px image (about 45 minutes for
the 180 test images); CPU works but is slow.

## Repository layout

| Path | Content |
|---|---|
| `src/filament_seg/` | package: RLE helpers, organizer-compatible metrics with paired bootstrap, dihedral views, instance voting, post-processing, pipeline |
| `src/run_pipeline.py` | inference CLI (`configs/final.json`: weights, views, threshold, erosion) |
| `src/prepare_yolo_seg.py`, `make_folds.py`, `make_alldata.py` | observation-grouped splits and YOLO datasets |
| `src/train_yolo_seg.py`, `train_folds.py` | detector training (two stages: 1024 px, then 2048 px fine-tune) |
| `src/cache_views.py`, `exp_*.py`, `final_results.py` | cached detector output, ensemble / post-processing / out-of-fold studies, final table with CIs |
| `src/make_figures.py`, `morphology_stats.py`, `tools/` | report figures, mask morphology, report/notebook generation |
| `results/` | final table (`final_results.json`), holdout predictions, per-observation scores, morphology, cross-fit, expert agreement |
| `docs/report/` | 4-page technical report (ACM sigconf): `main.pdf`, LaTeX source and figures |
| `submissions/` | the test CSVs of the final and the second submission, as written by `run_pipeline.py` |
| `tests/` | unit tests (metrics, fusion, post-processing) |

## Method in short

1. **Detector.** YOLO11s-seg, every annotator copy is a separate training example (the network learns the typical annotator).
   Stage 1: 40 epochs at 1024 px; stage 2: fine-tune at native 2048 px (AdamW, lr 1e-4, no mosaic).
2. **Views and detectors.** Each detector runs on the 8 dihedral views of the image (4 rotations x flip); masks are mapped back.
3. **Instance vote.** Instances of all (detector, view) sources are clustered by IoU > 0.3 (one member per source); the fused mask
   is the pixel vote of the members (at least half), its score the sum of member confidences divided by the number of sources.
4. **Decision.** Contested pixels go to the higher score, instances below 0.18 are dropped, each mask is eroded with a 3x3 ellipse
   (the detector's quarter-resolution masks are wider than the annotators' polygons) and reduced to its largest connected component.

## Validation protocol

* Splits are by **physical observation**, stratified by observatory (the same JPEG appears up to three times with different annotators).
  Holdout: 141 observations; five observation-grouped folds over the other 566 give out-of-fold predictions and the ensemble members.
* Fixed epoch counts and last checkpoints (fold and refit detectors). The first detector (`weights/a1.pt`) starts its 2048 px stage from
  the validation-best epoch (39 of 40) of its 1024 px stage, a 0.003 mask mAP50-95 selection on the holdout that we disclose.
* Every gain is a **paired** observation-level bootstrap against its parent; the threshold was tuned on out-of-fold data.

## Training from scratch

```bash
python src/prepare_yolo_seg.py --coco-json <MAGFiLO train json> --image-dir <train images> --output-dir data/processed/yolo_seg_observation_split_v1
python src/make_folds.py --dataset data/processed/yolo_seg_observation_split_v1 --output data/processed/folds_v1
for K in 0 1 2 3 4; do python src/train_folds.py --fold $K --folds-dir data/processed/folds_v1 --project runs/folds_v1; done
```

The first detector was trained with `src/train_yolo_seg.py` (40 epochs at 1024 px, then 5 epochs at 2048 px, see `docs/`);
`weights/` holds all final checkpoints, so training is optional. Seeds and epoch counts are fixed; MPS/CUDA kernels are not
bit-deterministic, so retraining reproduces the metrics only up to the fold-to-fold variation reported above (+-0.004 PQ).

## Rules compliance

* Training data: the official training release only (plus the COCO-pretrained YOLO11 initialisation shipped with Ultralytics).
  The released models and the final submissions use neither the public MAGFiLO release nor other participants' notebooks, weights or outputs.
* Inference: only the test image; no file names, timestamps, neighbouring frames or external annotations.
* The final selection consists of our own models only.
* Earlier in the competition the team also submitted outputs derived from public notebooks as attributed controls (public scores
  between 0.34 and 0.56). They are not part of this solution and are not among the two final submissions.

## License and third-party notices

Code: GNU AGPL-3.0 (`LICENSE`), because the pipeline imports and the weights derive from Ultralytics YOLO11 (AGPL-3.0). The
weights are trained on the MAGFiLO annotations (CC BY-NC 4.0): non-commercial use only. Dependencies and their licenses are listed in
`THIRD_PARTY_NOTICES.md`.

## Citation

MAGFiLO dataset and the Solar Filament Segmentation Challenge 2026 (ESAIR Lab); Ultralytics YOLO11. See `docs/report/main.bib`.
