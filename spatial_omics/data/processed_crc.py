"""Adapter for processed CRC CODEX cell tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from spatial_omics.data.anndata_compat import AnnData
from spatial_omics.data.io import marker_matrix_from_cells
from spatial_omics.data.types import SpatialStudy

REQUIRED_CELL_COLUMNS = {
    "cell_id",
    "sample_id",
    "patient_id",
    "region_id",
    "x",
    "y",
    "cell_type",
}


@dataclass
class ProcessedCRCCODEXAdapter:
    """Load a processed CRC CODEX dataset from local CSV files."""

    input_dir: str | Path
    dataset_name: str = "crc_codex_processed"
    label_column: str = "label"

    def load_study(self) -> SpatialStudy:
        input_dir = Path(self.input_dir)
        cells = pd.read_csv(input_dir / "cells.csv")
        missing = REQUIRED_CELL_COLUMNS.difference(cells.columns)
        if missing:
            raise ValueError(f"cells.csv is missing required columns: {sorted(missing)}")
        if cells["cell_id"].duplicated().any():
            duplicates = cells.loc[cells["cell_id"].duplicated(), "cell_id"].head(5).tolist()
            raise ValueError(f"Duplicate cell_id values detected: {duplicates}")

        regions_path = input_dir / "regions.csv"
        if self.label_column not in cells.columns:
            if not regions_path.exists():
                raise ValueError(
                    f"Missing '{self.label_column}' in cells.csv and regions.csv was not found."
                )
            regions = pd.read_csv(regions_path)
            if "region_id" not in regions.columns or self.label_column not in regions.columns:
                raise ValueError("regions.csv must contain region_id and label columns.")
            cells = cells.merge(
                regions[["region_id", self.label_column]],
                on="region_id",
                how="left",
                validate="many_to_one",
            )

        samples: dict[str, AnnData] = {}
        sample_rows = []
        task_rows = []

        for region_id, region_cells in cells.groupby("region_id", sort=True):
            region_cells = region_cells.reset_index(drop=True)
            sample_id = str(region_cells["sample_id"].iloc[0])
            patient_id = str(region_cells["patient_id"].iloc[0])
            label = str(region_cells[self.label_column].iloc[0])
            coords = region_cells[["x", "y"]].to_numpy(dtype="float32")
            feature_skip = REQUIRED_CELL_COLUMNS | {self.label_column, "compartment"}
            X, marker_cols = marker_matrix_from_cells(region_cells, skip=feature_skip)

            obs = region_cells.copy()
            obs["cell_id"] = obs["cell_id"].astype(str)
            obs["sample_id"] = obs["sample_id"].astype(str)
            obs["patient_id"] = obs["patient_id"].astype(str)
            obs["region_id"] = obs["region_id"].astype(str)
            obs["cell_type"] = obs["cell_type"].astype(str)
            if "compartment" in obs.columns:
                obs["compartment"] = obs["compartment"].fillna("unknown").astype(str)

            samples[str(region_id)] = AnnData(
                X=X,
                obs=obs,
                var=pd.DataFrame(index=marker_cols),
                obsm={"spatial": coords},
                uns={
                    "sample_meta": {
                        "region_id": str(region_id),
                        "sample_id": sample_id,
                        "patient_id": patient_id,
                        "label": label,
                        "dataset_name": self.dataset_name,
                        "marker_columns": marker_cols,
                    }
                },
            )

            sample_rows.append(
                {
                    "region_id": str(region_id),
                    "sample_id": sample_id,
                    "patient_id": patient_id,
                    "label": label,
                    "n_cells": int(len(region_cells)),
                }
            )
            task_rows.append(
                {
                    "region_id": str(region_id),
                    "sample_id": sample_id,
                    "patient_id": patient_id,
                    "label": label,
                }
            )

        sample_table = pd.DataFrame(sample_rows).sort_values("region_id").reset_index(drop=True)
        task_table = pd.DataFrame(task_rows).sort_values("region_id").reset_index(drop=True)
        return SpatialStudy(
            samples=samples,
            sample_table=sample_table,
            task_table=task_table,
            dataset_name=self.dataset_name,
            study_meta={"source_dir": str(input_dir), "label_column": self.label_column},
        )


__all__ = ["ProcessedCRCCODEXAdapter", "REQUIRED_CELL_COLUMNS"]
