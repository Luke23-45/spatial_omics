from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from spatial_omics.config import MultiDatasetBenchmarkConfig, MultiDatasetStudyConfig, load_config
from spatial_omics.data import create_study_adapter, registered_study_adapters, resolve_catalog_entry, source_inventory
from spatial_omics.data.io import load_study
from spatial_omics.data.processed_crc import ProcessedCRCCODEXAdapter
from spatial_omics.evaluation import study_preflight_report
from spatial_omics.scripts.run_multidataset_benchmark import _effective_splits


def _write_fixture_dataset(root: Path) -> Path:
    rows = []
    specs = [
        ("r1", "s1", "p1", "CLR"),
        ("r2", "s2", "p2", "DII"),
    ]
    cell_id = 0
    for region_id, sample_id, patient_id, label in specs:
        for i in range(4):
            rows.append(
                {
                    "cell_id": f"c{cell_id}",
                    "sample_id": sample_id,
                    "patient_id": patient_id,
                    "region_id": region_id,
                    "x": float(i),
                    "y": float(i + 1),
                    "cell_type": "tumor" if i < 2 else "immune",
                    "marker_a": 1.0 + i,
                    "label": label,
                }
            )
            cell_id += 1
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "cells.csv", index=False)
    return root


def test_recursive_multidataset_config_load(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: crc_main",
                "    source_id: crc_main",
                "run_models:",
                "  - graphsage",
                "  - toponet_hodge",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(str(config_path), MultiDatasetBenchmarkConfig)
    assert len(cfg.datasets) == 1
    assert cfg.datasets[0].name == "crc_main"
    assert cfg.run_models == ("graphsage", "toponet_hodge")


def test_processed_cell_table_adapter_materializes_study(tmp_path: Path) -> None:
    input_dir = _write_fixture_dataset(tmp_path / "input")
    adapter = create_study_adapter(
        "processed_cell_table",
        name="fixture",
        input_dir=str(input_dir),
        dataset_name="fixture_ds",
        label_column="label",
    )
    materialized = adapter.materialize_study(tmp_path / "prepared")
    study = load_study(materialized)
    assert study.dataset_name == "fixture_ds"
    assert len(study.samples) == 2


def test_prepared_study_adapter_roundtrip(tmp_path: Path) -> None:
    input_dir = _write_fixture_dataset(tmp_path / "input")
    study = ProcessedCRCCODEXAdapter(input_dir=input_dir, dataset_name="fixture_ds").load_study()
    from spatial_omics.data.io import save_study

    saved = save_study(study, tmp_path / "saved")
    adapter = create_study_adapter(
        "prepared_study",
        name="fixture_saved",
        study_dir=str(saved),
    )
    loaded = adapter.load_study()
    assert loaded.dataset_name == "fixture_ds"
    assert len(loaded.sample_table) == 2


def test_registered_study_adapters_contains_expected_entries() -> None:
    assert {"prepared_study", "processed_cell_table"}.issubset(set(registered_study_adapters()))


def test_catalog_resolution_for_local_crc_source() -> None:
    resolution = resolve_catalog_entry("crc_main")
    assert resolution.status in {"prepared", "missing", "raw_available"}


def test_preflight_report_has_basic_counts(tmp_path: Path) -> None:
    input_dir = _write_fixture_dataset(tmp_path / "input")
    study = ProcessedCRCCODEXAdapter(input_dir=input_dir, dataset_name="fixture_ds").load_study()
    report = study_preflight_report(study, requested_splits=2)
    assert report.n_regions == 2
    assert report.n_patients == 2
    assert not report.can_run_grouped_cv
    assert report.recommended_max_splits == 1


def test_source_inventory_contains_dataset_metadata() -> None:
    inventory = source_inventory(["crc_main"])
    assert len(inventory) == 1
    item = inventory[0]
    assert item.name == "crc_main"
    assert item.resolution == "cell-resolved"


def test_effective_splits_auto_caps_to_preflight_max() -> None:
    cfg = MultiDatasetBenchmarkConfig(n_splits=4, auto_cap_splits=True)
    dataset_cfg = MultiDatasetStudyConfig(name="fixture")
    assert _effective_splits(cfg, dataset_cfg, recommended_max_splits=2) == 2
