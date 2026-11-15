# Third-party notices

This repository's code is licensed under the GNU Affero General Public License v3.0 (`LICENSE`, the text distributed with
Ultralytics 8.4.138). AGPL-3.0 was chosen because the pipeline imports Ultralytics (AGPL-3.0) and the released weights are
derived from the Ultralytics YOLO11 architecture and its COCO-pretrained initialisation.

## Software dependencies (pinned in `requirements.txt`)

| Package | Version | License | Compatible with AGPL-3.0 code |
|---|---|---|---|
| ultralytics (and ultralytics-thop) | 8.4.138 (2.1.6) | AGPL-3.0 | same license |
| torch | 2.14.0 | BSD-3-Clause (with Apache-2.0 and other permissive components) | yes |
| torchvision | 0.29.0 | BSD-3-Clause | yes |
| numpy | 2.5.2 | BSD-3-Clause (with permissive components) | yes |
| opencv-python | 5.0.0.93 | Apache-2.0 | yes |
| pycocotools | 2.0.11 | BSD-2-Clause | yes |
| pillow | 12.3.0 | MIT-CMU (HPND) | yes |
| matplotlib | 3.11.1 | Matplotlib license (PSF-based) | yes |
| pandas | 3.0.5 | BSD-3-Clause | yes |
| pyyaml | 6.0.3 | MIT | yes |

## Data and weights

* **MAGFiLO dataset / Solar Filament Segmentation Challenge 2026 training release.** The detectors were trained on the
  official training images and annotations only (CC BY-NC 4.0 as shown on the competition's data page). The weights in `weights/` are
  therefore released for **non-commercial** use; the images and annotations themselves are not redistributed here.
* **Initialisation.** The detectors start from the COCO-pretrained `yolo11s-seg.pt` checkpoint published by Ultralytics
  (AGPL-3.0). No other pretrained weights, public notebooks, outputs or submissions of other participants are used.
* **GONG data.** The images originate from the GONG network operated by NSO/AURA; see the acknowledgements of the report.
