from __future__ import annotations

import argparse
from pathlib import Path

from spatial_omics.config import PrepareConfig, load_config
from spatial_omics.data.cellsighter_crc import CellSighterCRCTestDatasetAdapter
from spatial_omics.data.io import save_study
from spatial_omics.data.processed_crc import ProcessedCRCCODEXAdapter


def _parse_label_map(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    mapping: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid clinical label map item: {item!r}")
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a processed CRC CODEX study into canonical spatial-omics artifacts.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--clinical-annotations-path", default=None)
    parser.add_argument("--clinical-patient-id-column", default=None)
    parser.add_argument("--clinical-label-column", default=None)
    parser.add_argument("--clinical-label-map", default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config, PrepareConfig)
    else:
        cfg = PrepareConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name or "crc_codex_processed",
            label_column=args.label_column or "label",
            clinical_annotations_path=args.clinical_annotations_path,
            clinical_patient_id_column=args.clinical_patient_id_column or "patient_id",
            clinical_label_column=args.clinical_label_column or "label",
            clinical_label_map=_parse_label_map(args.clinical_label_map),
        )
    input_dir = Path(cfg.input_dir)
    if (input_dir / "cells").exists() and (input_dir / "cells2labels").exists() and (input_dir / "data").exists():
        adapter = CellSighterCRCTestDatasetAdapter(
            input_dir=cfg.input_dir,
            dataset_name=cfg.dataset_name,
            patch_size_px=cfg.patch_size_px,
            min_cells_per_region=cfg.min_cells_per_region,
            label_mode=cfg.label_mode,
            clinical_annotations_path=cfg.clinical_annotations_path,
            clinical_patient_id_column=cfg.clinical_patient_id_column,
            clinical_label_column=cfg.clinical_label_column,
            clinical_label_map=cfg.clinical_label_map,
        )
    else:
        adapter = ProcessedCRCCODEXAdapter(
            input_dir=cfg.input_dir,
            dataset_name=cfg.dataset_name,
            label_column=cfg.label_column,
        )
    study = adapter.load_study()
    save_study(study, cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
