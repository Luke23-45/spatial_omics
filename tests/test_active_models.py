from __future__ import annotations

from pathlib import Path

import pandas as pd

from spatial_omics import ACTIVE_BASELINE_MODEL, ACTIVE_TOPOLOGY_MODEL, SUPPORTED_MODELS
from spatial_omics.data.io import save_study
from spatial_omics.data.processed_crc import ProcessedCRCCODEXAdapter
from spatial_omics.models.gnn_baselines import GNNBaselineConfig, run_gnn_baselines_study
from spatial_omics.models.spatial_z4 import SpatialZ4Config, build_region_examples, run_spatial_z4_study


def _write_fixture_dataset(root: Path) -> Path:
    rows = []
    region_specs = [
        ("r1", "s1", "p1", "CLR"),
        ("r2", "s2", "p1", "CLR"),
        ("r3", "s3", "p2", "DII"),
        ("r4", "s4", "p2", "DII"),
        ("r5", "s5", "p3", "CLR"),
        ("r6", "s6", "p3", "CLR"),
        ("r7", "s7", "p4", "DII"),
        ("r8", "s8", "p4", "DII"),
    ]
    base_patterns = {
        "tumor": [(0, 0), (1, 0.2), (0.8, 1.1), (1.2, 1.5)],
        "cd8_t": [(3, 3), (3.5, 3.2), (2.8, 3.7), (2.9, 2.4)],
        "macrophage": [(5, 1), (5.5, 1.2), (5.8, 0.6), (4.8, 0.9)],
        "vessel": [(1, 5), (1.5, 5.2), (0.4, 4.8), (1.2, 4.3)],
    }
    cell_counter = 0
    for region_idx, (region_id, sample_id, patient_id, label) in enumerate(region_specs):
        offset_x = region_idx * 5.0
        offset_y = (region_idx % 2) * 3.0
        for cell_type, coords in base_patterns.items():
            for x, y in coords:
                rows.append(
                    {
                        "cell_id": f"c{cell_counter}",
                        "sample_id": sample_id,
                        "patient_id": patient_id,
                        "region_id": region_id,
                        "x": x + offset_x,
                        "y": y + offset_y,
                        "cell_type": cell_type,
                        "compartment": "tumor_core" if cell_type == "tumor" else "stroma",
                        "marker_cd8": 1.0 if cell_type == "cd8_t" else 0.2,
                        "marker_tumor": 1.0 if cell_type == "tumor" else 0.2,
                        "label": label,
                    }
                )
                cell_counter += 1
    frame = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "cells.csv", index=False)
    return root


def _prepare_study(tmp_path: Path) -> Path:
    input_dir = _write_fixture_dataset(tmp_path / "input")
    study = ProcessedCRCCODEXAdapter(input_dir=input_dir).load_study()
    prepared_dir = tmp_path / "prepared"
    save_study(study, prepared_dir)
    return prepared_dir


def test_active_registry_is_narrow() -> None:
    assert ACTIVE_BASELINE_MODEL == "graphsage"
    assert ACTIVE_TOPOLOGY_MODEL == "spatial_z4_v2"
    assert set(SUPPORTED_MODELS) == {"graphsage", "spatial_z4_v2"}


def test_build_region_examples_smoke(tmp_path: Path) -> None:
    prepared_dir = _prepare_study(tmp_path)
    examples, vocab = build_region_examples(str(prepared_dir), knn_k=3, max_cells=32)
    assert len(examples) == 8
    assert len(vocab) >= 4
    assert examples[0].neighborhood_mode == "adaptive_knn"
    assert examples[0].neighbor_mask.any()


def test_graphsage_smoke_run(tmp_path: Path) -> None:
    prepared_dir = _prepare_study(tmp_path)
    cfg = GNNBaselineConfig(
        study_dir=str(prepared_dir),
        output_dir=str(tmp_path / "graphsage_out"),
        n_splits=2,
        n_repeats=1,
        epochs=1,
        patience=1,
        batch_size=4,
        knn_k=3,
        model_dim=24,
        max_cells=32,
        model_names=("graphsage",),
        blend_with_engineered=False,
    )
    results = run_gnn_baselines_study(cfg)
    assert results["runs"][0]["model_name"] == "graphsage"
    assert 0.0 <= results["runs"][0]["metrics"]["balanced_accuracy"] <= 1.0


def test_spatial_z4_v2_smoke_run(tmp_path: Path) -> None:
    prepared_dir = _prepare_study(tmp_path)
    cfg = SpatialZ4Config(
        study_dir=str(prepared_dir),
        output_dir=str(tmp_path / "spatial_out"),
        n_splits=2,
        n_repeats=1,
        epochs=1,
        patience=1,
        batch_size=4,
        knn_k=3,
        model_dim=24,
        anchor_count=4,
        router_steps=2,
        lift_dim=8,
        max_cells=32,
        blend_with_engineered=False,
        use_engineered_context=False,
    )
    results = run_spatial_z4_study(cfg)
    assert results["summary"]["best_run"]["model_name"] == "spatial_z4_v2"
    assert 0.0 <= results["summary"]["best_run"]["metrics"]["auroc"] <= 1.0
