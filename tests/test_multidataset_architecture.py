from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import spatial_omics.scripts.run_multidataset_benchmark as multidataset_runner

from spatial_omics.config import MultiDatasetBenchmarkConfig, MultiDatasetStudyConfig, load_config
from spatial_omics.data import DatasetResolution, create_study_adapter, registered_study_adapters, resolve_catalog_entry, source_inventory
from spatial_omics.data.io import load_study, save_study
from spatial_omics.data.types import SpatialStudy
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


def _write_keren_fixture(path: Path) -> Path:
    obs = pd.DataFrame(
        {
            "SampleID": ["1", "1", "2", "2"],
            "cellLabelInImage": ["1", "2", "1", "2"],
            "all_group_name2": ["Tumor", "Immune", "Tumor", "Stroma"],
            "subtype": ["Mixed", "Mixed", "Compartmentalized", "Compartmentalized"],
        }
    )
    var = pd.DataFrame(index=["marker_a", "marker_b"])
    x = np.array([[1.0, 2.0], [2.0, 3.0], [0.5, 0.2], [0.8, 0.3]], dtype=np.float32)
    adata = ad.AnnData(X=x, obs=obs, var=var, obsm={"spatial": np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)})
    adata.write_h5ad(path)
    return path


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
    assert {"prepared_study", "processed_cell_table", "keren_tnbc_h5ad"}.issubset(set(registered_study_adapters()))


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


def test_keren_h5ad_adapter_materializes_study(tmp_path: Path) -> None:
    raw_path = _write_keren_fixture(tmp_path / "tnbc.h5ad")
    adapter = create_study_adapter(
        "keren_tnbc_h5ad",
        name="keren_tnbc",
        input_dir=str(raw_path),
        dataset_name="keren_tnbc",
    )
    materialized = adapter.materialize_study(tmp_path / "prepared")
    study = load_study(materialized)
    assert study.dataset_name == "keren_tnbc"
    assert study.sample_table["patient_id"].nunique() == 2
    assert set(study.sample_table["label"]) == {"Mixed", "Compartmentalized"}


def test_resolve_dataset_config_can_auto_download(monkeypatch) -> None:
    cfg = MultiDatasetStudyConfig(name="tnbc_support", source_id="keren_tnbc")
    missing = DatasetResolution(name="keren_tnbc", adapter="keren_tnbc_h5ad", dataset_name="keren_tnbc", status="missing")
    ready = DatasetResolution(
        name="keren_tnbc",
        adapter="keren_tnbc_h5ad",
        dataset_name="keren_tnbc",
        status="raw_available",
        input_dir="C:\\tmp\\keren",
    )

    calls = {"downloaded": False}

    monkeypatch.setattr(multidataset_runner, "resolve_catalog_entry", lambda name: missing)

    def _ensure(name: str):
        calls["downloaded"] = True
        return ready

    monkeypatch.setattr(multidataset_runner, "ensure_catalog_source_available", _ensure)
    resolution = multidataset_runner._resolve_dataset_config(cfg, auto_download_sources=True)
    assert calls["downloaded"] is True
    assert resolution.status == "raw_available"
    assert cfg.input_dir == "C:\\tmp\\keren"


def test_cli_config_builder_for_keren_source() -> None:
    class Args:
        config = None
        dataset_source = "keren_tnbc"
        dataset_name = "keren_tnbc"
        adapter = None
        study_dir = None
        input_dir = None
        label_column = "subtype"
        output_subdir = "keren_tnbc"
        output_dir = "outputs/active/test_cli"
        materialized_data_root = None
        run_models = ["graphsage", "toponet_hodge"]
        preflight_only = False
        no_auto_download = False
        no_strict_preflight = False
        no_auto_cap_splits = False
        random_state = 42
        n_splits = 4
        n_repeats = 1
        batch_size = 12
        epochs = 12
        patience = 4
        learning_rate = 1e-3
        weight_decay = 5e-5
        knn_k = 6
        neighborhood_mode = "adaptive_knn"
        model_dim = 48
        type_embedding_dim = 16
        max_cells = 160
        dropout = 0.15
        focal_gamma = 2.0
        label_smoothing = 0.0
        threshold_grid_size = 61
        temperature = 0.5
        blend_with_engineered = False
        num_layers = 2
        polynomial_order = 2
        restriction_hidden_dim = 32
        use_geometric_weights = False
        use_orthogonal_restrictions = False
        use_morse_gating = False
        global_knn_multiplier = 4
        global_distance_multiplier = 2.5
        global_max_neighbors = 24

    cfg = multidataset_runner._config_from_args(Args())
    assert len(cfg.datasets) == 1
    assert cfg.datasets[0].source_id == "keren_tnbc"
    assert cfg.datasets[0].adapter == "keren_tnbc_h5ad"
    assert cfg.auto_download_sources is True


def test_keren_source_validation_rejects_wrong_labels(tmp_path: Path) -> None:
    sample_table = pd.DataFrame(
        [
            {"region_id": "r1", "sample_id": "s1", "patient_id": "p1", "label": "CLR", "n_cells": 2},
            {"region_id": "r2", "sample_id": "s2", "patient_id": "p2", "label": "DII", "n_cells": 2},
        ]
    )
    task_table = sample_table[["region_id", "sample_id", "patient_id", "label"]].copy()
    obs = pd.DataFrame(
        {
            "cell_id": ["c1", "c2"],
            "sample_id": ["s1", "s1"],
            "patient_id": ["p1", "p1"],
            "region_id": ["r1", "r1"],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "cell_type": ["tumor", "immune"],
            "label": ["CLR", "CLR"],
        }
    ).set_index("cell_id")
    adata = ad.AnnData(
        X=np.array([[1.0], [2.0]], dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["marker_a"]),
        obsm={"spatial": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)},
        uns={"sample_meta": {"region_id": "r1", "sample_id": "s1", "patient_id": "p1", "label": "CLR"}},
    )
    study = SpatialStudy(samples={"r1": adata}, sample_table=sample_table.iloc[:1].copy(), task_table=task_table.iloc[:1].copy(), dataset_name="keren_tnbc")
    study_dir = save_study(study, tmp_path / "bad_keren")
    dataset_cfg = MultiDatasetStudyConfig(name="keren_tnbc", source_id="keren_tnbc", study_dir=str(study_dir))
    loaded = load_study(study_dir)
    try:
        multidataset_runner._validate_source_specific_study(dataset_cfg, loaded)
    except ValueError as exc:
        assert "label mismatch" in str(exc)
    else:
        raise AssertionError("Expected keren_tnbc label validation to fail for wrong labels.")
