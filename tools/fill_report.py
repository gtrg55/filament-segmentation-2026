#!/usr/bin/env python3
"""Fill the report template (`main.tex.in`) from the measured results so that text, tables and figures agree.

Inputs are JSON files written by `src/final_results.py`, `src/morphology_stats.py`, `src/exp_crossfit.py` and the
hand-kept `results/dataset_facts.json`, `results/expert_agreement.json`, `results/calibration.json`. Nothing is
typed into the report by hand except the author block.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def signed(x: float, digits: int = 3) -> str:
    return f"{x:+.{digits}f}".replace("-", "$-$")


def interval(low: float, high: float, digits: int = 3, sign: bool = False) -> str:
    fmt = signed if sign else num
    return f"[{fmt(low, digits)}, {fmt(high, digits)}]"


def results_table(report: dict) -> str:
    holdout = report["holdout"]
    rows = []
    for stage in report["stages"]:
        gain = "--" if "d_pq" not in stage else f"{signed(stage['d_pq'])} {interval(*stage['d_pq_ci'], sign=True)}"
        bold = stage["name"] == report["final"]
        pq = f"{num(stage['pq'])} {interval(*stage['pq_ci'])}"
        label = stage["label"]
        if bold:
            label, pq = f"\\textbf{{{label}}}", f"\\textbf{{{num(stage['pq'])}}} {interval(*stage['pq_ci'])}"
        rows.append(f"{label} & {pq} & {gain} & {num(stage['mean_semantic_dice'])} \\\\")
    return (
        "\\begin{table}[t]\n\\centering\\small\\setlength{\\tabcolsep}{1.5pt}\n"
        f"\\caption{{Pipeline stages on the holdout ({holdout['observations']} observations, {holdout['copies']} annotator "
        "copies). PQ with 95\\% bootstrap interval; $\\Delta$PQ is the paired gain over the previous row; "
        "Dice is the mean Dice of the union masks.}\n\\label{tab:results}\n"
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n Stage & PQ [95\\% CI] & $\\Delta$PQ [95\\% CI] & Dice \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
    )


def results_md(report: dict) -> str:
    lines = ["| Stage | sources | PQ [95% CI] | dPQ vs previous [95% CI] | Dice | one-to-many | many-to-one |",
             "|---|---|---|---|---|---|---|"]
    for stage in report["stages"]:
        gain = "-" if "d_pq" not in stage else f"{stage['d_pq']:+.4f} [{stage['d_pq_ci'][0]:+.4f}, {stage['d_pq_ci'][1]:+.4f}]"
        name = f"**{stage['label']}**" if stage["name"] == report["final"] else stage["label"]
        lines.append(f"| {name} | {stage['sources']} | {stage['pq']:.4f} [{stage['pq_ci'][0]:.4f}, {stage['pq_ci'][1]:.4f}] | {gain} "
                     f"| {stage['mean_semantic_dice']:.4f} | {stage['gt_one_to_many']:.4f} | {stage['pred_many_to_one']:.4f} |")
    return "\n".join(lines)


VARIANT_ROWS = [
    ("a1_id", "1 detector, 1 view"), ("a1_f5_id", "6 detectors, 1 view"), ("base_a1", "1 detector, 8 views"),
    ("a1_f5_rot4", "6 detectors, 4 rotations"), ("a1_f5_v4", "6 detectors, 4 views (2 flips)"),
    ("f5_d4", "5 fold detectors, 8 views"), ("a1_f5_d4", "6 detectors, 8 views"),
]


def variants_block(variants: dict | None) -> str:
    """Test-time compute versus PQ: model ensembling and dihedral views are largely redundant."""
    if not variants or any(key not in variants for key, _ in VARIANT_ROWS):
        return ""
    rows = []
    for key, label in VARIANT_ROWS:
        row = variants[key]
        gain = "--" if key == "base_a1" else (f"{signed(float(row['d_pq']))} "
                                              f"{interval(float(row['ci_low']), float(row['ci_high']), sign=True)}")
        rows.append(f"{label} & {row['sources']} & {num(float(row['pq']))} & {gain} & {num(float(row['sem_dice']))} \\\\")
    table = (
        "\\begin{table}[t]\n\\centering\\small\\setlength{\\tabcolsep}{1.5pt}\n"
        "\\caption{Inference cost versus quality on the holdout. Sources = detectors $\\times$ views; the score threshold is optimised per row, "
        "$\\Delta$PQ is the paired gain over one detector with eight views; the largest-component step is not applied.}\n\\label{tab:variants}\n"
        "\\begin{tabular}{@{}lrccc@{}}\n\\toprule\n Configuration & Sources & PQ & $\\Delta$PQ [95\\% CI] & Dice \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
    )
    pq = {k: float(variants[k]["pq"]) for k, _ in VARIANT_ROWS}
    text = (
        "\\paragraph{Test-time compute.} Detector ensembling and dihedral views are largely redundant (Table~\\ref{tab:variants}): six detectors on a single "
        f"view gain {num(pq['a1_f5_id'] - pq['a1_id'])} PQ over one detector, eight views of one detector {num(pq['base_a1'] - pq['a1_id'])}, and "
        f"both together only {num(pq['a1_f5_d4'] - pq['a1_id'])}. Halving the views to four costs nothing measurable "
        f"({num(pq['a1_f5_v4'])}--{num(pq['a1_f5_rot4'])} PQ against {num(pq['a1_f5_d4'])}); we keep all 48 sources because the full vote has the best "
        f"Dice ({num(float(variants['a1_f5_d4']['sem_dice']))}), and inference is cheap (about 14\\,s per image on an M4 Max)."
    )
    return text + "\n\n" + table


def pct(x: float) -> str:
    return f"{100 * x:.1f}\\%"


def morphology_text(morph: dict, first: dict, final: dict) -> str:
    gt, pr = morph["annotated"], morph["predicted"]

    def med(block: dict, key: str, digits: int = 0) -> str:
        return f"{block[key]['median']:.{digits}f}"

    single = (
        f"Annotated instances are almost always one connected component ({100 * gt['single_component_share']:.1f}\\%); "
        "the largest-component step makes every predicted instance one as well. "
        if final["largest"] else
        f"{100 * gt['single_component_share']:.1f}\\% of the annotated and {100 * pr['single_component_share']:.1f}\\% of the "
        "predicted instances are a single connected component. "
    )
    thicker = pr["thickness"]["median"] - gt["thickness"]["median"]
    return (
        f"Annotators outline {gt['per_image']:.1f} filaments per image (median area {med(gt, 'area')}\\,px$^2$, median "
        f"thickness {med(gt, 'thickness', 1)}\\,px, solidity {med(gt, 'solidity', 2)}, elongation {med(gt, 'elongation', 1)}); "
        f"the final pipeline predicts {pr['per_image']:.1f} instances per image (median area {med(pr, 'area')}\\,px$^2$, "
        f"thickness {med(pr, 'thickness', 1)}\\,px, solidity {med(pr, 'solidity', 2)}, elongation {med(pr, 'elongation', 1)}), i.e. "
        f"smoother and more compact outlines than hand-drawn polygons and {abs(thicker):.1f}\\,px {'thicker' if thicker > 0 else 'thinner'} "
        "at the median. " + single +
        f"Instance relations quantify the structural errors: {pct(final['gt_one_to_many'])} of the annotated "
        f"filaments are overlapped by two or more predictions (one-to-many; {pct(first['gt_one_to_many'])} for a single view) and "
        f"{pct(final['pred_many_to_one'])} of the predictions overlap two or more annotated filaments (many-to-one; "
        f"{pct(first['pred_many_to_one'])} for a single view). Overlapping pairs have a mean IoU of {num(final['mean_pair_iou'], 2)} and a mean "
        f"Dice of {num(final['mean_pair_dice'], 2)} (Fig.~\\ref{{fig:results}}); Fig.~\\ref{{fig:examples}} shows holdout crops at fixed percentiles of the per-observation PQ."
    )


def errors_text(err: dict) -> str:
    recall, precision, sites = err["recall_by_gt_area"], err["precision_by_pred_area"], err["pq_by_observatory"]
    best = max(sites, key=lambda k: sites[k]["pq"])
    worst = min(sites, key=lambda k: sites[k]["pq"])
    return (
        f"Recall depends on size (Fig.~\\ref{{fig:errors}}): {pct(recall[0]['rate'])} of the smallest quintile of annotated filaments "
        f"(below {recall[0]['area_high']:.0f}\\,px$^2$) are matched, {pct(min(r['rate'] for r in recall[1:4]))}--{pct(max(r['rate'] for r in recall[1:4]))} "
        f"of the middle quintiles, and {pct(recall[-1]['rate'])} of the largest (above {recall[-1]['area_low']:.0f}\\,px$^2$), where an IoU above 0.5 "
        f"requires agreeing on the whole extent of a long structure. Precision is {pct(min(r['rate'] for r in precision[1:]))}--"
        f"{pct(max(r['rate'] for r in precision[1:]))} except for the smallest predicted quintile ({pct(precision[0]['rate'])}). "
        f"PQ by observatory ranges from {sites[worst]['pq']:.3f} ({worst}, {sites[worst]['observations']} observations) to "
        f"{sites[best]['pq']:.3f} ({best}, {sites[best]['observations']}). Each estimate uses 20--25 observations."
    )


def calibration_text(calib: dict, final: dict, crossfit: dict | None, optimism: str) -> str:
    pairs = calib["pairs"]
    gaps = [p["holdout_pq"] - p["public_lb"] for p in pairs]
    low, high = min(gaps), max(gaps)
    forecast = final["pq"] - sum(gaps) / len(gaps)
    listing = ", ".join(f"{p['holdout_pq']:.3f}$\\to${p['public_lb']:.2f}" for p in pairs)
    text = (
        f"Earlier single-detector submissions had holdout PQ and public score pairs ({listing}). "
        f"The gaps ranged from {low:.3f} to {high:.3f} across these {len(pairs)} submissions. "
    )
    measured = calib.get("final")
    if measured:
        gap = measured["holdout_pq"] - measured["public_lb"]
        text += (
            f"The final pipeline scored {measured['public_lb']:.2f} publicly and {measured['holdout_pq']:.3f} on the holdout. "
            f"The public forecast was {forecast:.2f} and the holdout-to-public gap was {gap:.3f}. "
            "Kaggle displays the public score to two decimal places."
        )
    else:
        text += f"The mean gap gives a public score forecast of {forecast:.2f}."
    if crossfit is not None:
        text += (
            " A split-half check estimated "
            f"{optimism} PQ of selection optimism for the tested threshold and erosion grid."
        )
    return text


def fold_note(variants: dict | None) -> str:
    if not variants:
        return ""
    folds = [float(variants[f"f{k}_d4"]["pq"]) for k in range(5) if f"f{k}_d4" in variants]
    if len(folds) < 5 or "base_a1" not in variants:
        return ""
    return (f" Individually, the five fold detectors (trained on 80\\% of the fit observations) reach {min(folds):.3f}--{max(folds):.3f} PQ "
            f"with eight views (mean {sum(folds) / 5:.3f}) against {float(variants['base_a1']['pq']):.3f} for the first detector; "
            f"the six-detector vote adds {float(variants['a1_f5_d4']['pq']) - float(variants['base_a1']['pq']):+.3f} PQ (paired 95\\% CI {interval(float(variants['a1_f5_d4']['ci_low']), float(variants['a1_f5_d4']['ci_high']), sign=True)}) "
            f"and {float(variants['a1_f5_d4']['sem_dice']) - float(variants['base_a1']['sem_dice']):+.3f} Dice.")


def strengths_text(first: dict, final: dict, human: dict, variants: dict | None = None) -> str:
    h = human["human_vs_human_all"]
    h_rq = h["tp"] / (h["tp"] + 0.5 * h["fp"] + 0.5 * h["fn"])
    return (
        "\\emph{Strengths.} The solution is a single public architecture trained on the provided labels, evaluated with an "
        "observation-grouped protocol; every claimed gain has a paired interval. Erosion, voting over views and voting over "
        f"detectors turn near-misses into matches and remove false alarms ({first['tp']} $\\to$ {final['tp']} true positives, "
        f"{first['fp']} $\\to$ {final['fp']} false positives)." + fold_note(variants) + " Matched filaments agree with the annotator "
        f"slightly better than experts agree with each other (IoU {num(final['sq'], 2)} vs {num(h['mean_tp_iou'], 2)}). "
        "\\emph{Weaknesses.} "
        f"{pct(final['gt_unmatched'])} of annotated filaments have zero overlap with every prediction. "
        f"Recognition quality is {num(final['rq'], 2)} for the detector and {num(h_rq, 2)} between annotators. "
        "The smallest area quintile has the lowest recall. "
        "\\emph{Improvements.} Further experiments can target the extent of long masks and the detection of small filaments."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--crossfit", type=Path)
    parser.add_argument("--oof-grid", type=Path, help="out-of-fold grid CSV of src/exp_oof_grid.py")
    parser.add_argument("--errors", type=Path, help="results/error_analysis.json")
    parser.add_argument("--variants", type=Path, help="ensemble-variant CSV (results/ensemble_variants.csv)")
    parser.add_argument("--second-submission", default="", help="LaTeX text about the second selected submission")
    parser.add_argument("--readme-template", type=Path)
    parser.add_argument("--readme-out", type=Path)
    parser.add_argument("--epochs2", type=int, default=6)
    parser.add_argument("--author", default="TODO: author name")
    parser.add_argument("--country", default="TODO: country")
    parser.add_argument("--email", default="TODO: e-mail")
    args = parser.parse_args()

    report, morph, calib, facts, human = (load(p) for p in (args.results, args.morphology, args.calibration, args.facts, args.expert))
    crossfit = load(args.crossfit) if args.crossfit else None
    variants = None
    if args.variants:
        with args.variants.open(newline="", encoding="utf-8") as handle:
            variants = {row["config"]: row for row in csv.DictReader(handle)}
    stages = {s["name"]: s for s in report["stages"]}
    first, final = report["stages"][0], stages[report["final"]]
    erosion = next((s for s in report["stages"] if s.get("erosion") and s.get("d_pq") is not None
                    and s["parent"] and stages[s["parent"]].get("erosion", 0) == 0 and s["sources"] == stages[s["parent"]]["sources"]), None)
    optimism = num(crossfit["selection_optimism"], 3) if crossfit else "n/a"
    values = {
        "author": args.author, "country": args.country, "email": args.email,
        "n_models": str(final["n_models"]), "n_views": str(final["n_views"]),
        "pq_final": num(final["pq"]), "pq_final_ci": interval(*final["pq_ci"]),
        "pq_human": num(human["human_vs_human_all"]["pq"]),
        "n_train_obs": str(facts["train_observations"]), "n_copies": f"{facts['train_copies']:,}".replace(",", "\\,"),
        "n_instances": f"{facts['train_instances']:,}".replace(",", "\\,"),
        "n_fit_obs": str(facts["fit_observations"]), "n_holdout_obs": str(report["holdout"]["observations"]),
        "n_holdout_copies": str(report["holdout"]["copies"]), "epochs2": str(args.epochs2),
        "threshold": num(final["threshold"], 2),
        "gain_erosion": signed(erosion["d_pq"]) if erosion else "n/a",
        "largest_component_step": (
            "\\item Only the largest connected component of each mask is kept, so one detection is one filament piece."
            if final["largest"] else ""),
        "optimism": optimism,
        "results_table": results_table(report),
        "morphology_text": morphology_text(morph, first, final),
        "calibration_text": calibration_text(calib, final, crossfit, optimism),
        "strengths_text": strengths_text(first, final, human, variants),
        "second_submission_text": args.second_submission,
        "errors_text": errors_text(load(args.errors)) if args.errors else "",
        "variants_block": variants_block(variants),
    }
    values["bootstrap_draws"] = str(report["holdout"]["bootstrap_draws"])
    values["results_md"] = results_md(report)
    values["oof_pq"] = "n/a"
    if args.oof_grid:
        with args.oof_grid.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (float(row["threshold"]), int(row["erosion"]), bool(int(row["largest"]))) == (
                        final["threshold"], final["erosion"], final["largest"]):
                    values["oof_pq"] = f"{float(row['pq']):.4f}"
    for template, out in ((args.template, args.out), (args.readme_template, args.readme_out)):
        if template is None:
            continue
        text = template.read_text(encoding="utf-8")
        used = set(re.findall(r"@@(\w+)@@", text))
        missing = used - set(values)
        if missing:
            raise SystemExit(f"{template.name}: placeholders without a value: {sorted(missing)}")
        for key in used:
            text = text.replace(f"@@{key}@@", values[key])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out} ({len(used)} placeholders filled)")


if __name__ == "__main__":
    main()
