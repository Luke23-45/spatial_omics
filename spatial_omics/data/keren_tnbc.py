from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from spatial_omics.data.anndata_compat import AnnData
from spatial_omics.data.types import SpatialStudy


EXPECTED_FILENAMES = (
    "TNBC.h5ad.gz",
    "tnbc.h5ad.gz",
    "TNBC.h5ad",
    "tnbc.h5ad",
)


@dataclass
class KerenTNBCH5ADAdapter:
    input_dir: str | Path
    dataset_name: str = "keren_tnbc"
    label_column: str = "subtype"
    cell_type_column: str = "all_group_name2"
    sample_column: str = "SampleID"

    def load_study(self) -> SpatialStudy:
        input_path = self._resolve_input_file()
        adata = ad.read_h5ad(input_path)
        obs = adata.obs.copy()
        var_names = [str(name) for name in adata.var_names.tolist()]
        marker_frame = pd.DataFrame(
            np.asarray(adata.X, dtype=np.float32),
            index=obs.index,
            columns=var_names,
        )
        obs = obs.join(marker_frame)
        obs["x"] = np.asarray(adata.obsm["spatial"][:, 0], dtype=np.float32)
        obs["y"] = np.asarray(adata.obsm["spatial"][:, 1], dtype=np.float32)
        obs["sample_id"] = obs[self.sample_column].astype(str).map(lambda value: f"keren_{value}")
        obs["patient_id"] = obs["sample_id"]
        obs["region_id"] = obs["sample_id"]
        obs["cell_id"] = obs[self.sample_column].astype(str) + "_" + obs["cellLabelInImage"].astype(str)
        obs["cell_type"] = (
            obs[self.cell_type_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
            .str.strip("_")
        )
        obs["label"] = obs[self.label_column].astype(str)

        samples: dict[str, AnnData] = {}
        sample_rows: list[dict[str, object]] = []
        task_rows: list[dict[str, object]] = []
        for sample_id, region_cells in obs.groupby("region_id", sort=True):
            region_cells = region_cells.reset_index(drop=True)
            label_values = sorted(region_cells["label"].astype(str).unique().tolist())
            if len(label_values) != 1:
                raise ValueError(f"Sample '{sample_id}' has multiple labels: {label_values}")
            label = label_values[0]
            region_cells.index = region_cells["cell_id"].astype(str)
            coords = region_cells[["x", "y"]].to_numpy(dtype=np.float32)
            sample_markers = region_cells[var_names].to_numpy(dtype=np.float32)
            samples[str(sample_id)] = AnnData(
                X=sample_markers,
                obs=region_cells,
                var=pd.DataFrame(index=var_names),
                obsm={"spatial": coords},
                uns={
                    "sample_meta": {
                        "region_id": str(sample_id),
                        "sample_id": str(sample_id),
                        "patient_id": str(region_cells["patient_id"].iloc[0]),
                        "label": label,
                        "dataset_name": self.dataset_name,
                        "marker_columns": var_names,
                    }
                },
            )
            sample_rows.append(
                {
                    "region_id": str(sample_id),
                    "sample_id": str(sample_id),
                    "patient_id": str(region_cells["patient_id"].iloc[0]),
                    "label": label,
                    "n_cells": int(len(region_cells)),
                }
            )
            task_rows.append(
                {
                    "region_id": str(sample_id),
                    "sample_id": str(sample_id),
                    "patient_id": str(region_cells["patient_id"].iloc[0]),
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
            study_meta={
                "source_file": str(input_path),
                "label_column": "label",
                "source_schema": "keren_tnbc_h5ad",
                "n_marker_channels": len(var_names),
            },
        )

    def _resolve_input_file(self) -> Path:
        input_dir = Path(self.input_dir)
        candidates: list[Path] = []
        if input_dir.is_file():
            candidates.append(input_dir)
        else:
            for filename in EXPECTED_FILENAMES:
                candidates.append(input_dir / filename)
                candidates.append(input_dir / "keren_tnbc" / filename)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Could not find Keren TNBC h5ad file under '{input_dir}'. Tried: {[str(path) for path in candidates]}"
        )


__all__ = ["EXPECTED_FILENAMES", "KerenTNBCH5ADAdapter"]
