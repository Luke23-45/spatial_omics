from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spatial_omics.data.io import load_study
from spatial_omics.evaluation.splits import build_grouped_splits


def _cell_type_composition(study) -> pd.DataFrame:
    rows = []
    for region_id, adata in study.samples.items():
        counts = adata.obs["cell_type"].value_counts()
        row = {"region_id": region_id}
        row.update(counts.to_dict())
        rows.append(row)
    comp = pd.DataFrame(rows).fillna(0).set_index("region_id")
    comp = comp.astype(int)
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
    return comp, comp_pct


def _fold_structure(study, n_splits: int, n_repeats: int, random_state: int) -> list[dict]:
    st = study.sample_table
    labels = st["label"].to_numpy()
    groups = st["patient_id"].to_numpy()
    splits = build_grouped_splits(labels, groups, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    fold_info = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        train_groups = groups[train_idx]
        test_groups = groups[test_idx]
        _, train_counts = np.unique(train_labels, return_counts=True)
        _, test_counts = np.unique(test_labels, return_counts=True)
        train_labels_dict = dict(zip(*np.unique(train_labels, return_counts=True)))
        test_labels_dict = dict(zip(*np.unique(test_labels, return_counts=True)))
        fold_info.append(
            {
                "fold": fold_idx + 1,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "train_labels": train_labels_dict,
                "test_labels": test_labels_dict,
                "test_patients": list(test_groups),
            }
        )
    return fold_info


def explore_dataset(
    study_dir: str | Path,
    n_splits: int = 4,
    n_repeats: int = 1,
    random_state: int = 42,
    output_path: str | Path | None = None,
) -> str:
    study = load_study(str(study_dir))
    st = study.sample_table

    lines: list[str] = []
    def w(text: str = "") -> None:
        lines.append(text)

    w(f"# Dataset Report: {study.dataset_name}")
    w()
    w(f"**Source**: `{study.study_meta.get('source_file', 'unknown')}`")
    w(f"**Schema**: `{study.study_meta.get('source_schema', 'unknown')}`")
    w()

    # --- 1. Overview ---
    w("## 1. Dataset Overview")
    w()
    w(f"| Property | Value |")
    w(f"|----------|-------|")
    w(f"| Dataset name | {study.dataset_name} |")
    w(f"| Total regions | {len(study.samples)} |")
    w(f"| Total cells | {int(st['n_cells'].sum()):,} |")
    w(f"| Unique patients | {st['patient_id'].nunique()} |")
    w(f"| Marker channels | {study.study_meta.get('n_marker_channels', '?')} |")
    w(f"| Classes | {st['label'].nunique()} |")
    w()

    # --- 2. Class Distribution ---
    w("## 2. Class Distribution")
    w()
    label_counts = st["label"].value_counts()
    w(f"| Label | Regions | Cells (total) | Cells (mean) | Cells (min) | Cells (max) |")
    w(f"|-------|---------|---------------|--------------|-------------|-------------|")
    for label in sorted(label_counts.index):
        subset = st[st["label"] == label]
        n_regions = len(subset)
        total = int(subset["n_cells"].sum())
        mean_val = int(subset["n_cells"].mean())
        min_val = int(subset["n_cells"].min())
        max_val = int(subset["n_cells"].max())
        w(f"| {label} | {n_regions} | {total:,} | {mean_val:,} | {min_val:,} | {max_val:,} |")
    w()

    # --- 3. Marker Channels ---
    w("## 3. Marker Channels")
    w()
    first = next(iter(study.samples.values()))
    markers = list(first.var_names)
    w(f"Total markers: **{len(markers)}**")
    w()
    w("| # | Marker |")
    w("|---|--------|")
    for i, m in enumerate(markers, 1):
        w(f"| {i} | {m} |")
    w()

    # --- 4. Cell Type Composition ---
    w("## 4. Cell Type Composition")
    w()
    comp, comp_pct = _cell_type_composition(study)
    w("### Absolute counts per region")
    w()
    w(comp.to_markdown())
    w()
    w("### Percentage composition per region")
    w()
    w(comp_pct.to_markdown(floatfmt=".1f"))
    w()

    # Aggregate across all regions
    total_by_type = comp.sum()
    total_all = total_by_type.sum()
    w("### Aggregate cell type distribution")
    w()
    w(f"| Cell Type | Total Cells | Percentage |")
    w(f"|-----------|-------------|------------|")
    for ct in sorted(total_by_type.index):
        count = int(total_by_type[ct])
        pct = count / total_all * 100
        w(f"| {ct} | {count:,} | {pct:.1f}% |")
    w()

    # Per-class cell type composition
    w("### Per-class cell type composition")
    w()
    for label in sorted(st["label"].unique()):
        region_ids = st[st["label"] == label]["region_id"].tolist()
        label_comp = comp.loc[comp.index.isin(region_ids)].sum()
        label_total = int(label_comp.sum())
        w(f"**{label}** ({len(region_ids)} regions, {label_total:,} cells total)")
        w()
        w("| Cell Type | Cells | Percentage |")
        w("|-----------|-------|------------|")
        for ct in sorted(label_comp.index):
            count = int(label_comp[ct])
            pct = count / label_total * 100
            w(f"| {ct} | {count:,} | {pct:.1f}% |")
        w()

    # --- 5. Spatial Statistics ---
    w("## 5. Spatial Statistics")
    w()
    spatial_stats = []
    for region_id, adata in study.samples.items():
        coords = adata.obsm["spatial"]
        n_cells = coords.shape[0]
        x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
        width = x_max - x_min
        height = y_max - y_min
        density = n_cells / (width * height) if width > 0 and height > 0 else 0
        spatial_stats.append(
            {
                "region_id": region_id,
                "n_cells": n_cells,
                "x_range": f"{x_min:.0f}–{x_max:.0f}",
                "y_range": f"{y_min:.0f}–{y_max:.0f}",
                "width": f"{width:.0f}",
                "height": f"{height:.0f}",
                "density_per_px2": f"{density:.6f}",
            }
        )
    spatial_df = pd.DataFrame(spatial_stats)
    w(spatial_df.to_markdown(index=False))
    w()

    # --- 6. Cross-Validation Fold Structure ---
    w("## 6. Cross-Validation Fold Structure")
    w()
    w(f"Split strategy: `build_grouped_splits(n_splits={n_splits}, n_repeats={n_repeats}, random_state={random_state})`")
    w()
    w("Each fold holds out **2 patients** (1 per class) for testing and trains on the remaining **32 patients**. This ensures strict patient-level separation — no patient's cells appear in both train and test.")
    w()
    st = study.sample_table
    labels_all = st["label"].to_numpy()
    groups_all = st["patient_id"].to_numpy()
    folds = _fold_structure(study, n_splits, n_repeats, random_state)

    # Compute coverage
    all_patients = set(groups_all)
    tested_patients = set()
    for f in folds:
        tested_patients.update(f["test_patients"])
    never_tested = all_patients - tested_patients

    w(f"**Coverage**: {len(tested_patients)}/{len(all_patients)} patients are ever tested ({len(never_tested)} patients never appear in a test set).")
    w()
    w(f"**Critical note**: With `n_splits={n_splits}` on {len(all_patients)} patients and only 15 Compartmentalized cases, the binary shortcut in `build_grouped_splits` creates at most 4 folds — testing only 8 patients total. The remaining 26 patients are **always in training**. This means reported metrics reflect performance on a small, non-exhaustive subset of the patient population. Consider LOPO (`n_splits >= {len(all_patients)}`) for full coverage, or document this limitation explicitly.")
    w()
    for f in folds:
        w()
        w(f"| | Train | Test |")
        w(f"|---|-------|------|")
        w(f"| Regions | {f['n_train']} | {f['n_test']} |")
        for label, count in sorted(f["train_labels"].items()):
            test_count = f["test_labels"].get(label, 0)
            w(f"| Label: {label} | {count} | {test_count} |")
        w(f"| Test patients | | {', '.join(f['test_patients'])} |")
        w()

    # --- 7. Data Provenance ---
    w("## 7. Data Provenance")
    w()
    w("### Original Source")
    w()
    w("The raw data originates from the study by **Keren et al. (2018)**:")
    w()
    w("> Keren, L., Bosse, M., Marquez, D. et al. A Structured Tumor-Immune Microenvironment in Triple Negative Breast Cancer Revealed by Multiplexed Ion Beam Imaging. *Cell* 174, 1373–1387 (2018).")
    w("> DOI: [10.1016/j.cell.2018.08.039](https://doi.org/10.1016/j.cell.2018.08.039)")
    w()
    w("### Preprocessing Pipeline")
    w()
    w("1. **Raw file**: `data/spatial_omics/raw/tnbc.h5ad.gz` — AnnData file containing 173,205 cells × 36 markers with cell-level metadata (SampleID, cell type annotations, spatial coordinates).")
    w("2. **Region construction**: Cells are grouped by `SampleID` into 34 regions (one per patient). Each region is stored as a separate AnnData with the same 36 marker channels.")
    w("3. **Label assignment**: Each region is assigned a tissue organization label (`Compartmentalized` or `Mixed`) based on the spatial arrangement of tumor and immune cells.")
    w("4. **Cell type annotation**: Three broad cell-type classes are assigned (`tumor`, `immune`, `stroma`) from the original fine-grained annotations in the raw data.")
    w("5. **Canonical format**: The result is saved as a `SpatialStudy` via `save_study()` — each region as an individual `.h5ad` file plus `sample_table.csv` and `task_table.csv`.")
    w()
    w("### Dataset Usage in Project")
    w()
    w("This is the **locked single dataset** for the paper (per the research charter in `docs/hypothesis.md`). All model comparisons (GraphSAGE vs TopoNet-Hodge) are performed on this dataset under strict patient-level cross-validation.")
    w()

    report = "\n".join(lines)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore a prepared spatial-omics dataset and produce a detailed report.")
    parser.add_argument("--study-dir", default="data/spatial_omics/prepared/keren_tnbc")
    parser.add_argument("--output", "-o", default=None, help="Path to write the markdown report")
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    report = explore_dataset(
        study_dir=args.study_dir,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        output_path=args.output,
    )

    if not args.output:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
