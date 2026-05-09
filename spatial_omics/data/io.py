from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spatial_omics.utils.io import ensure_dir

from spatial_omics.data.anndata_compat import AnnData, HAS_ANNDATA
from spatial_omics.data.types import SpatialStudy


_PICKLE_MODULE_ALIASES = {
    "synapse.spatial_omics.data.anndata_compat": "spatial_omics.data.anndata_compat",
    "synapse.spatial_omics.data.types": "spatial_omics.data.types",
    "spartial_omics.data.anndata_compat": "spatial_omics.data.anndata_compat",
    "spartial_omics.data.types": "spatial_omics.data.types",
}


class _AliasUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        module = _PICKLE_MODULE_ALIASES.get(module, module)
        return super().find_class(module, name)


def _metadata_path(root: Path) -> Path:
    return root / "study_meta.json"


def _sample_path(root: Path, region_id: str) -> Path:
    suffix = ".h5ad" if HAS_ANNDATA and hasattr(AnnData, "write_h5ad") else ".pkl"
    return root / "samples" / f"{region_id}{suffix}"


def save_adata(adata: AnnData, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_ANNDATA and hasattr(adata, "write_h5ad"):
        adata.write_h5ad(path)
    else:
        with path.open("wb") as fh:
            pickle.dump(adata, fh)
    return path


def load_adata(path: str | Path) -> AnnData:
    path = Path(path)
    if path.suffix == ".h5ad":
        if not HAS_ANNDATA:
            raise RuntimeError("Cannot load .h5ad without the optional 'anndata' dependency.")
        from anndata import read_h5ad

        return read_h5ad(path)
    with path.open("rb") as fh:
        return _AliasUnpickler(fh).load()


def save_study(study: SpatialStudy, output_dir: str | Path) -> Path:
    root = ensure_dir(output_dir)
    samples_dir = ensure_dir(root / "samples")
    study.sample_table.to_csv(root / "sample_table.csv", index=False)
    study.task_table.to_csv(root / "task_table.csv", index=False)
    payload = {
        "dataset_name": study.dataset_name,
        "study_meta": study.study_meta,
        "sample_files": {region_id: _sample_path(root, region_id).name for region_id in study.samples},
    }
    _metadata_path(root).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for region_id, adata in study.samples.items():
        save_adata(adata, samples_dir / _sample_path(root, region_id).name)
    return root


def load_study(input_dir: str | Path) -> SpatialStudy:
    root = Path(input_dir)
    meta = json.loads(_metadata_path(root).read_text(encoding="utf-8"))
    sample_table = pd.read_csv(root / "sample_table.csv")
    task_table = pd.read_csv(root / "task_table.csv")
    samples = {
        region_id: load_adata(root / "samples" / filename)
        for region_id, filename in meta["sample_files"].items()
    }
    return SpatialStudy(
        samples=samples,
        sample_table=sample_table,
        task_table=task_table,
        dataset_name=meta["dataset_name"],
        study_meta=meta.get("study_meta", {}),
    )


def marker_matrix_from_cells(cells: pd.DataFrame, skip: set[str]) -> tuple[np.ndarray, list[str]]:
    marker_cols: list[str] = []
    for col in cells.columns:
        if col in skip:
            continue
        if pd.api.types.is_numeric_dtype(cells[col]):
            marker_cols.append(col)
    if not marker_cols:
        return np.zeros((len(cells), 0), dtype=np.float32), []
    matrix = cells[marker_cols].fillna(0.0).to_numpy(dtype=np.float32)
    return matrix, marker_cols


__all__ = ["load_adata", "load_study", "marker_matrix_from_cells", "save_adata", "save_study"]
