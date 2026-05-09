from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from spartial_omics.utils.io import ensure_dir

from spartial_omics.data.io import load_study
from spartial_omics.reporting.plots import plot_metric_bars, plot_region_cells


def build_markdown_report(
    results: dict,
    output_dir: str | Path,
    *,
    study_dir: str | None = None,
    features_path: str | None = None,
) -> Path:
    output_dir = ensure_dir(output_dir)
    lines = [
        "# Spatial Omics Feasibility Report",
        "",
        "## Summary",
        "",
    ]
    summary = results.get("summary", {})
    best = summary.get("best_run", {})
    if best:
        lines.append(
            f"Best run: **{best['feature_set']} / {best['model_name']}** "
            f"(AUROC={best['metrics']['auroc']:.4f}, "
            f"Balanced Accuracy={best['metrics']['balanced_accuracy']:.4f}, "
            f"Macro F1={best['metrics']['macro_f1']:.4f})."
        )
    delta = summary.get("topology_added_value_auroc")
    if delta is not None:
        lines.append(f"Topology added value over the logistic F0 baseline: **{delta:+.4f} AUROC**.")
    lines.extend(["", "## Runs", "", "| Feature Set | Model | AUROC | Balanced Accuracy | Macro F1 | Brier | Stability |", "|---|---|---:|---:|---:|---:|---:|"])
    for run in results.get("runs", []):
        metrics = run["metrics"]
        lines.append(
            f"| {run['feature_set']} | {run['model_name']} | {metrics['auroc']:.4f} | "
            f"{metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['brier_score']:.4f} | {metrics['feature_stability']:.4f} |"
        )

    lines.extend(["", "## Interpretation", ""])
    if delta is None:
        lines.append("Combined F0+F1 results were not available, so added-value comparison could not be computed.")
    elif delta > 0:
        lines.append("Topology features improved the logistic baseline and should be investigated further.")
    else:
        lines.append("Topology features did not improve the logistic baseline in this run; the feasibility gate is not yet passed.")

    top_feature_lines = []
    if best:
        for row in best.get("top_features", []):
            top_feature_lines.append(f"- `{row['feature']}`: {row['score']:.4f}")
    if top_feature_lines:
        lines.extend(["", "## Top Features", ""])
        lines.extend(top_feature_lines)

    if study_dir is not None and features_path is not None:
        feature_table = pd.read_csv(features_path)
        study = load_study(study_dir)
        f1_rows = feature_table.loc[feature_table["feature_family"] == "F1"].set_index("region_id")
        score_cols = [col for col in f1_rows.columns if col.startswith("union_tumor_cd8_h1_total_persistence")]
        if score_cols:
            score_col = score_cols[0]
            ranked = f1_rows[score_col].sort_values()
            low_region = str(ranked.index[0])
            high_region = str(ranked.index[-1])
            figures_dir = ensure_dir(Path(output_dir) / "figures")
            low_path = plot_region_cells(study.samples[low_region], f"Low topology score: {low_region}", figures_dir / "low_region.png")
            high_path = plot_region_cells(study.samples[high_region], f"High topology score: {high_region}", figures_dir / "high_region.png")
            lines.extend(
                [
                    "",
                    "## Case Studies",
                    "",
                    f"- Lowest `{score_col}` region: `{low_region}` -> `{low_path.name}`",
                    f"- Highest `{score_col}` region: `{high_region}` -> `{high_path.name}`",
                ]
            )

    plot_metric_bars(results, Path(output_dir) / "figures")
    report_path = Path(output_dir) / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


__all__ = ["build_markdown_report"]
