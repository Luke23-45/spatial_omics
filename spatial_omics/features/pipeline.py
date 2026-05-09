from __future__ import annotations

from pathlib import Path

import pandas as pd

from spatial_omics.utils.io import ensure_dir

from spatial_omics.data.io import load_study
from spatial_omics.features.baseline import extract_baseline_features
from spatial_omics.features.topology import extract_topology_features


def extract_feature_families(study_dir: str, cfg) -> pd.DataFrame:
    study = load_study(study_dir)
    rows: list[dict[str, object]] = []
    for region_id in sorted(study.samples):
        adata = study.samples[region_id]
        if cfg.include_f0:
            rows.append(extract_baseline_features(adata, cfg))
        if cfg.include_f1:
            rows.append(extract_topology_features(adata, cfg))
    feature_table = pd.DataFrame(rows).sort_values(["region_id", "feature_family"]).reset_index(drop=True)
    return feature_table


def save_feature_table(feature_table: pd.DataFrame, output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    path = Path(output_dir) / "feature_table.csv"
    feature_table.to_csv(path, index=False)
    return path


__all__ = ["extract_feature_families", "save_feature_table"]
