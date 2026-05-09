"""Adapter for the CellSighter CRC multiplexed image test dataset."""

from __future__ import annotations

import ast
import zipfile
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from spatial_omics.data.anndata_compat import AnnData
from spatial_omics.data.types import SpatialStudy


IMMUNE_TYPES = {"Bcell", "CD3T", "CD4T", "CD8T", "DC", "Macrophage", "Neutrophil", "Plasma", "Treg"}
TUMOR_STROMA_TYPES = {"Tumor", "Stroma"}
VASCULAR_TYPES = {"Endothelial", "Lymphatic"}


def _load_mapping(path: Path) -> dict[int, str]:
    return {int(k): str(v) for k, v in ast.literal_eval(path.read_text(encoding="utf-8")).items()}


def _compartment_for_type(cell_type: str) -> str:
    if cell_type in {"Tumor"}:
        return "tumor"
    if cell_type in {"Stroma"}:
        return "stroma"
    if cell_type in IMMUNE_TYPES:
        return "immune"
    if cell_type in VASCULAR_TYPES:
        return "vascular"
    if cell_type in {"Neuron"}:
        return "neural"
    return "other"


def _proxy_patch_label(cell_types: pd.Series) -> str:
    immune_count = int(cell_types.isin(IMMUNE_TYPES).sum())
    tumor_stroma_count = int(cell_types.isin(TUMOR_STROMA_TYPES).sum())
    return "immune_rich" if immune_count >= tumor_stroma_count else "tumor_stroma_rich"


def _normalize_patient_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        return text
    if text.endswith(".0"):
        whole, frac = text.rsplit(".", 1)
        if frac == "0" and whole.isdigit():
            text = whole
    if text.upper().startswith("P"):
        suffix = text[1:]
        if suffix.isdigit():
            return f"P{int(suffix):02d}"
    if text.isdigit():
        return f"P{int(text):02d}"
    return text


def _load_annotation_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xls", ".xlsx"}:
        try:
            return pd.read_excel(path)
        except ImportError:
            if suffix != ".xlsx":
                raise
            return _load_xlsx_fallback(path)
    raise ValueError(f"Unsupported annotation file format: {path.suffix}")


def _load_xlsx_fallback(path: Path) -> pd.DataFrame:
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }

    def _col_index(cell_ref: str) -> int:
        letters = "".join(ch for ch in cell_ref if ch.isalpha())
        idx = 0
        for ch in letters:
            idx = idx * 26 + (ord(ch.upper()) - 64)
        return idx - 1

    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        relationships = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in relationships}
        first_sheet = workbook.find("a:sheets/a:sheet", ns)
        if first_sheet is None:
            return pd.DataFrame()
        rid = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        worksheet = ET.fromstring(zf.read("xl/" + rel_map[rid]))
        rows = worksheet.findall("a:sheetData/a:row", ns)

        header: dict[int, str] = {}
        records: list[dict[str, object]] = []
        for row_idx, row in enumerate(rows):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r", "")
                idx = _col_index(ref)
                node = cell.find("a:v", ns)
                if node is None:
                    value = ""
                elif cell.attrib.get("t") == "s":
                    value = shared[int(node.text)]
                else:
                    value = node.text or ""
                values[idx] = value
            if row_idx == 0:
                header = values
                continue
            record = {header[i]: values.get(i, "") for i in header}
            records.append(record)

    frame = pd.DataFrame(records)
    return frame.replace("", np.nan)


def _resolve_patient_labels(
    annotation_path: str | Path | None,
    patient_id_column: str,
    label_column: str,
    label_map: dict[str, str] | None = None,
) -> dict[str, str]:
    if annotation_path is None:
        return {}
    frame = _load_annotation_frame(Path(annotation_path))
    if patient_id_column not in frame.columns:
        raise KeyError(f"Patient id column '{patient_id_column}' was not found in {annotation_path}")
    if label_column not in frame.columns:
        raise KeyError(f"Label column '{label_column}' was not found in {annotation_path}")

    labels: dict[str, str] = {}
    for _, row in frame[[patient_id_column, label_column]].dropna().iterrows():
        patient_id = _normalize_patient_id(row[patient_id_column])
        label = str(row[label_column]).strip()
        if label_map is not None:
            label = label_map.get(label, label)
        if patient_id and label:
            labels[patient_id] = label
    return labels


@dataclass
class CellSighterCRCTestDatasetAdapter:
    input_dir: str | Path
    dataset_name: str = "cellsighter_crc_test"
    patch_size_px: int = 256
    min_cells_per_region: int = 40
    label_mode: str = "proxy_patch_tme"
    clinical_annotations_path: str | Path | None = None
    clinical_patient_id_column: str = "patient_id"
    clinical_label_column: str = "label"
    clinical_label_map: dict[str, str] | None = None

    def _read_channel_names(self, root: Path) -> list[str]:
        channel_path = root / "channels_codex_CRC_celltune.txt"
        return [line.strip() for line in channel_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _extract_cells(self, mask: np.ndarray, labels: np.ndarray, image: np.ndarray, channel_names: list[str], sample_id: str, patient_id: str) -> pd.DataFrame:
        max_id = int(mask.max())
        flat_mask = mask.reshape(-1)
        counts = np.bincount(flat_mask, minlength=max_id + 1).astype(np.float64)
        y_coords, x_coords = np.indices(mask.shape)
        sum_x = np.bincount(flat_mask, weights=x_coords.reshape(-1), minlength=max_id + 1)
        sum_y = np.bincount(flat_mask, weights=y_coords.reshape(-1), minlength=max_id + 1)

        marker_means = {}
        flat_image = image.reshape(-1, image.shape[-1])
        for idx, channel_name in enumerate(channel_names):
            sums = np.bincount(flat_mask, weights=flat_image[:, idx], minlength=max_id + 1)
            means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            marker_means[channel_name] = means

        rows = []
        for cell_id in range(1, max_id + 1):
            if counts[cell_id] <= 0:
                continue
            label_id = int(labels[cell_id])
            if label_id < 0:
                continue
            rows.append(
                {
                    "cell_id": f"{sample_id}_cell_{cell_id}",
                    "sample_id": sample_id,
                    "patient_id": patient_id,
                    "x": float(sum_x[cell_id] / counts[cell_id]),
                    "y": float(sum_y[cell_id] / counts[cell_id]),
                    "cell_type_id": label_id,
                    **{f"marker_{channel}": float(marker_means[channel][cell_id]) for channel in channel_names},
                }
            )
        return pd.DataFrame(rows)

    def load_study(self) -> SpatialStudy:
        root = Path(self.input_dir)
        mapping = _load_mapping(root / "mapping_id_to_cell_type_name.txt")
        channel_names = self._read_channel_names(root)
        patient_labels = _resolve_patient_labels(
            self.clinical_annotations_path,
            patient_id_column=self.clinical_patient_id_column,
            label_column=self.clinical_label_column,
            label_map=self.clinical_label_map,
        )
        active_label_mode = self.label_mode if not patient_labels else "clinical_patient_label"

        samples: dict[str, AnnData] = {}
        sample_rows: list[dict[str, object]] = []
        task_rows: list[dict[str, object]] = []

        for cells_path in sorted((root / "cells").glob("*.npz")):
            sample_id = cells_path.stem
            patient_id = _normalize_patient_id(sample_id.split("_")[0])
            labels_path = root / "cells2labels" / cells_path.name
            image_path = root / "data" / "antibodies" / cells_path.name
            if not labels_path.exists() or not image_path.exists():
                continue

            mask = np.load(cells_path)["data"]
            labels = np.load(labels_path)["data"]
            image = np.load(image_path)["data"]
            cell_frame = self._extract_cells(mask, labels, image, channel_names, sample_id, patient_id)
            cell_frame["cell_type"] = cell_frame["cell_type_id"].map(mapping).fillna("Unknown")
            cell_frame["compartment"] = cell_frame["cell_type"].map(_compartment_for_type)

            patch_x = (cell_frame["x"] // self.patch_size_px).astype(int)
            patch_y = (cell_frame["y"] // self.patch_size_px).astype(int)
            cell_frame["region_id"] = [
                f"{sample_id}_patch_{px}_{py}" for px, py in zip(patch_x, patch_y, strict=False)
            ]

            for region_id, patch_cells in cell_frame.groupby("region_id", sort=True):
                if len(patch_cells) < self.min_cells_per_region:
                    continue
                coords = patch_cells[["x", "y"]].to_numpy(dtype=np.float32)
                marker_cols = [f"marker_{channel}" for channel in channel_names]
                X = patch_cells[marker_cols].to_numpy(dtype=np.float32)
                label = patient_labels.get(patient_id, _proxy_patch_label(patch_cells["cell_type"]))
                obs = patch_cells[
                    ["cell_id", "sample_id", "patient_id", "region_id", "x", "y", "cell_type", "compartment"]
                    + marker_cols
                ].reset_index(drop=True)
                adata = AnnData(
                    X=X,
                    obs=obs,
                    var=pd.DataFrame(index=marker_cols),
                    obsm={"spatial": coords},
                    uns={
                        "sample_meta": {
                            "region_id": region_id,
                            "sample_id": sample_id,
                            "patient_id": patient_id,
                            "label": label,
                            "dataset_name": self.dataset_name,
                            "label_mode": active_label_mode,
                            "marker_columns": marker_cols,
                        }
                    },
                )
                samples[region_id] = adata
                sample_rows.append(
                    {
                        "region_id": region_id,
                        "sample_id": sample_id,
                        "patient_id": patient_id,
                        "label": label,
                        "n_cells": int(len(patch_cells)),
                    }
                )
                task_rows.append(
                    {
                        "region_id": region_id,
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
            study_meta={
                "source_dir": str(root),
                "label_mode": active_label_mode,
                "patch_size_px": self.patch_size_px,
                "min_cells_per_region": self.min_cells_per_region,
                "clinical_annotations_path": str(self.clinical_annotations_path) if self.clinical_annotations_path else None,
            },
        )


__all__ = ["CellSighterCRCTestDatasetAdapter"]
