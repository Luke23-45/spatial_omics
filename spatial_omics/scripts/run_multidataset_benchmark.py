from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from spatial_omics.config import (
    GNNBaselineConfig,
    MultiDatasetBenchmarkConfig,
    MultiDatasetStudyConfig,
    TopoNetHodgeConfig,
    load_config,
)
from spatial_omics.data import (
    create_study_adapter,
    ensure_catalog_source_available,
    resolve_catalog_entry,
    write_dataset_source_state,
    write_source_inventory,
)
from spatial_omics.data.io import load_study
from spatial_omics.evaluation import study_preflight_report
from spatial_omics.models.gnn_baselines import run_gnn_baselines_study, save_gnn_baseline_results
from spatial_omics.models.toponet_hodge import run_toponet_hodge_study, save_toponet_hodge_results
from spatial_omics.utils.io import ensure_dir, validate_expected_labels


EXPECTED_SOURCE_LABELS: dict[str, tuple[str, ...]] = {
    "keren_tnbc": ("Compartmentalized", "Mixed"),
}


def _resolve_dataset_config(dataset_cfg, *, auto_download_sources: bool):
    if not dataset_cfg.source_id:
        return None
    resolution = resolve_catalog_entry(dataset_cfg.source_id)
    if auto_download_sources and resolution.status == "missing":
        resolution = ensure_catalog_source_available(dataset_cfg.source_id)
    if dataset_cfg.study_dir is None:
        dataset_cfg.study_dir = resolution.study_dir
    if dataset_cfg.input_dir is None:
        dataset_cfg.input_dir = resolution.input_dir
    if dataset_cfg.dataset_name is None:
        dataset_cfg.dataset_name = resolution.dataset_name
    if resolution.adapter:
        dataset_cfg.adapter = resolution.adapter
    return resolution


def _materialize_dataset(dataset_cfg, prepared_root: Path) -> Path:
    adapter = create_study_adapter(
        dataset_cfg.adapter,
        name=dataset_cfg.name,
        dataset_name=dataset_cfg.dataset_name or dataset_cfg.name,
        study_dir=dataset_cfg.study_dir,
        input_dir=dataset_cfg.input_dir,
        label_column=dataset_cfg.label_column,
    )
    return adapter.materialize_study(prepared_root)


def _effective_splits(cfg: MultiDatasetBenchmarkConfig, dataset_cfg, recommended_max_splits: int) -> int:
    requested = dataset_cfg.max_splits_override or cfg.n_splits
    if cfg.auto_cap_splits:
        return max(1, min(int(requested), int(recommended_max_splits)))
    return int(requested)


def _build_gnn_config(
    cfg: MultiDatasetBenchmarkConfig,
    study_dir: Path,
    output_dir: Path,
    *,
    effective_splits: int,
) -> GNNBaselineConfig:
    return GNNBaselineConfig(
        study_dir=str(study_dir),
        output_dir=str(output_dir),
        features_path=cfg.features_path,
        random_state=cfg.random_state,
        n_splits=effective_splits,
        n_repeats=cfg.n_repeats,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        patience=cfg.patience,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        knn_k=cfg.knn_k,
        neighborhood_mode=cfg.neighborhood_mode,
        model_dim=cfg.model_dim,
        type_embedding_dim=cfg.type_embedding_dim,
        max_cells=cfg.max_cells,
        dropout=cfg.dropout,
        focal_gamma=cfg.focal_gamma,
        label_smoothing=cfg.label_smoothing,
        temperature=cfg.temperature,
        threshold_grid_size=cfg.threshold_grid_size,
        blend_with_engineered=cfg.blend_with_engineered,
        model_names=("graphsage",),
        split_mode="grouped",
    )


def _build_toponet_config(
    cfg: MultiDatasetBenchmarkConfig,
    study_dir: Path,
    output_dir: Path,
    *,
    effective_splits: int,
) -> TopoNetHodgeConfig:
    return TopoNetHodgeConfig(
        study_dir=str(study_dir),
        output_dir=str(output_dir),
        features_path=cfg.features_path,
        random_state=cfg.random_state,
        n_splits=effective_splits,
        n_repeats=cfg.n_repeats,
        split_mode="grouped",
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        patience=cfg.patience,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        knn_k=cfg.knn_k,
        neighborhood_mode=cfg.neighborhood_mode,
        model_dim=cfg.model_dim,
        type_embedding_dim=cfg.type_embedding_dim,
        max_cells=cfg.max_cells,
        dropout=cfg.dropout,
        focal_gamma=cfg.focal_gamma,
        label_smoothing=cfg.label_smoothing,
        threshold_grid_size=cfg.threshold_grid_size,
        num_layers=cfg.num_layers,
        polynomial_order=cfg.polynomial_order,
        restriction_hidden_dim=cfg.restriction_hidden_dim,
        use_geometric_weights=cfg.use_geometric_weights,
        use_orthogonal_restrictions=cfg.use_orthogonal_restrictions,
        use_morse_gating=cfg.use_morse_gating,
        global_knn_multiplier=cfg.global_knn_multiplier,
        global_distance_multiplier=cfg.global_distance_multiplier,
        global_max_neighbors=cfg.global_max_neighbors,
    )


def _dataset_record_base(dataset_cfg, resolution) -> dict[str, Any]:
    record = {
        "name": dataset_cfg.name,
        "source_id": dataset_cfg.source_id,
        "adapter": dataset_cfg.adapter,
        "status": "pending",
        "study_dir": dataset_cfg.study_dir,
        "input_dir": dataset_cfg.input_dir,
        "required": dataset_cfg.required,
        "runs": {},
        "resolution": None,
        "notes": list(dataset_cfg.notes),
    }
    if resolution is not None:
        record["resolution"] = {
            "status": resolution.status,
            "study_dir": resolution.study_dir,
            "input_dir": resolution.input_dir,
            "source_uris": resolution.source_uris,
            "notes": resolution.notes,
        }
    return record


def _skip_or_raise(cfg: MultiDatasetBenchmarkConfig, dataset_cfg, dataset_record: dict[str, Any], reason: str) -> None:
    dataset_record["status"] = "skipped"
    dataset_record["skip_reason"] = reason
    should_raise = dataset_cfg.required and cfg.fail_on_missing_required
    if should_raise:
        raise RuntimeError(reason)


def _run_model(model_name: str, cfg: MultiDatasetBenchmarkConfig, study_dir: Path, output_dir: Path, *, effective_splits: int):
    if model_name == "graphsage":
        run_cfg = _build_gnn_config(cfg, study_dir, output_dir, effective_splits=effective_splits)
        results = run_gnn_baselines_study(run_cfg)
        save_gnn_baseline_results(results, output_dir)
        return Path(output_dir) / "gnn_baseline_results.json"
    if model_name == "toponet_hodge":
        run_cfg = _build_toponet_config(cfg, study_dir, output_dir, effective_splits=effective_splits)
        results = run_toponet_hodge_study(run_cfg)
        save_toponet_hodge_results(results, output_dir)
        return Path(output_dir) / "toponet_hodge_results.json"
    raise ValueError(f"Unsupported model '{model_name}'.")


def _validate_source_specific_study(dataset_cfg, study) -> None:
    if not dataset_cfg.source_id:
        return
    expected = EXPECTED_SOURCE_LABELS.get(dataset_cfg.source_id)
    if expected is None:
        return
    validate_expected_labels(
        study.sample_table["label"].astype(str),
        expected=expected,
        context=f"Prepared study for source_id={dataset_cfg.source_id}",
    )


def _config_from_args(args: argparse.Namespace) -> MultiDatasetBenchmarkConfig:
    if args.config:
        return load_config(args.config, MultiDatasetBenchmarkConfig)

    if not args.dataset_source and not args.study_dir and not args.input_dir:
        raise ValueError("Provide either --config or one of --dataset-source / --study-dir / --input-dir.")

    if args.study_dir:
        adapter = "prepared_study"
    elif args.adapter:
        adapter = args.adapter
    elif args.dataset_source == "keren_tnbc":
        adapter = "keren_tnbc_h5ad"
    else:
        adapter = "processed_cell_table"

    dataset_cfg = MultiDatasetStudyConfig(
        name=args.dataset_name or args.dataset_source or "dataset",
        source_id=args.dataset_source,
        adapter=adapter,
        dataset_name=args.dataset_name or args.dataset_source,
        study_dir=args.study_dir,
        input_dir=args.input_dir,
        label_column=args.label_column,
        output_subdir=args.output_subdir,
        required=True,
        skip_if_unavailable=False,
        max_splits_override=args.n_splits,
    )
    return MultiDatasetBenchmarkConfig(
        datasets=[dataset_cfg],
        output_dir=args.output_dir,
        materialized_data_root=args.materialized_data_root,
        auto_download_sources=not args.no_auto_download,
        fail_on_missing_required=True,
        fail_on_model_error=True,
        strict_preflight=not args.no_strict_preflight,
        auto_cap_splits=not args.no_auto_cap_splits,
        emit_dataset_state=True,
        preflight_only=args.preflight_only,
        run_models=tuple(args.run_models),
        random_state=args.random_state,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        knn_k=args.knn_k,
        neighborhood_mode=args.neighborhood_mode,
        model_dim=args.model_dim,
        type_embedding_dim=args.type_embedding_dim,
        max_cells=args.max_cells,
        dropout=args.dropout,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        threshold_grid_size=args.threshold_grid_size,
        temperature=args.temperature,
        blend_with_engineered=args.blend_with_engineered,
        num_layers=args.num_layers,
        polynomial_order=args.polynomial_order,
        restriction_hidden_dim=args.restriction_hidden_dim,
        use_geometric_weights=args.use_geometric_weights,
        use_orthogonal_restrictions=args.use_orthogonal_restrictions,
        use_morse_gating=args.use_morse_gating,
        global_knn_multiplier=args.global_knn_multiplier,
        global_distance_multiplier=args.global_distance_multiplier,
        global_max_neighbors=args.global_max_neighbors,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the active spatial-omics benchmark across multiple prepared studies.")
    parser.add_argument("--config", default=None, help="Path to the MultiDatasetBenchmarkConfig yaml file.")
    parser.add_argument("--dataset-source", default=None, help="Catalog source id, e.g. keren_tnbc.")
    parser.add_argument("--dataset-name", default=None, help="Override dataset name for single-dataset CLI mode.")
    parser.add_argument("--adapter", default=None, help="Explicit adapter for single-dataset CLI mode.")
    parser.add_argument("--study-dir", default=None, help="Prepared study directory for single-dataset CLI mode.")
    parser.add_argument("--input-dir", default=None, help="Raw input directory or file for single-dataset CLI mode.")
    parser.add_argument("--label-column", default="label", help="Label column for raw-input adapters.")
    parser.add_argument("--output-subdir", default=None, help="Optional dataset subdirectory inside runs/.")
    parser.add_argument("--output-dir", default="outputs/active/multidataset_cli", help="Benchmark output root for single-dataset CLI mode.")
    parser.add_argument("--materialized-data-root", default=None, help="Prepared-study root for materialized datasets.")
    parser.add_argument("--run-models", nargs="+", default=["graphsage", "toponet_hodge"], choices=["graphsage", "toponet_hodge"])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-auto-download", action="store_true")
    parser.add_argument("--no-strict-preflight", action="store_true")
    parser.add_argument("--no-auto-cap-splits", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--knn-k", type=int, default=6)
    parser.add_argument("--neighborhood-mode", default="adaptive_knn")
    parser.add_argument("--model-dim", type=int, default=48)
    parser.add_argument("--type-embedding-dim", type=int, default=16)
    parser.add_argument("--max-cells", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--threshold-grid-size", type=int, default=61)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--blend-with-engineered", action="store_true")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--polynomial-order", type=int, default=2)
    parser.add_argument("--restriction-hidden-dim", type=int, default=32)
    parser.add_argument("--use-geometric-weights", action="store_true")
    parser.add_argument("--use-orthogonal-restrictions", action="store_true")
    parser.add_argument("--use-morse-gating", action="store_true")
    parser.add_argument("--global-knn-multiplier", type=int, default=4)
    parser.add_argument("--global-distance-multiplier", type=float, default=2.5)
    parser.add_argument("--global-max-neighbors", type=int, default=24)
    args = parser.parse_args()

    cfg = _config_from_args(args)
    root = ensure_dir(cfg.output_dir)
    prepared_root = ensure_dir(cfg.materialized_data_root or (root / "prepared"))
    runs_root = ensure_dir(root / "runs")
    if cfg.emit_dataset_state:
        write_source_inventory(Path(root) / "source_inventory.json")

    manifest: dict[str, Any] = {
        "config_path": str(Path(args.config).resolve()) if args.config else None,
        "preflight_only": cfg.preflight_only,
        "strict_preflight": cfg.strict_preflight,
        "auto_cap_splits": cfg.auto_cap_splits,
        "models": list(cfg.run_models),
        "datasets": [],
    }

    for dataset_cfg in cfg.datasets:
        resolution = _resolve_dataset_config(dataset_cfg, auto_download_sources=cfg.auto_download_sources)
        dataset_record = _dataset_record_base(dataset_cfg, resolution)
        dataset_root = ensure_dir(runs_root / (dataset_cfg.output_subdir or dataset_cfg.name))

        if resolution is not None and cfg.emit_dataset_state:
            write_dataset_source_state(dataset_root, resolution, source_id=dataset_cfg.source_id)

        try:
            study_dir = _materialize_dataset(dataset_cfg, prepared_root)
        except Exception as exc:
            reason = f"Dataset materialization failed: {exc}"
            if dataset_cfg.skip_if_unavailable:
                _skip_or_raise(cfg, dataset_cfg, dataset_record, reason)
                manifest["datasets"].append(dataset_record)
                continue
            raise

        dataset_record["study_dir"] = str(study_dir)
        study = load_study(study_dir)
        _validate_source_specific_study(dataset_cfg, study)
        preflight = study_preflight_report(study, requested_splits=dataset_cfg.max_splits_override or cfg.n_splits)
        effective_splits = _effective_splits(cfg, dataset_cfg, preflight.recommended_max_splits)
        preflight_payload = asdict(preflight)
        preflight_payload["warnings"] = list(preflight.warnings)
        preflight_payload["effective_splits"] = effective_splits
        dataset_record["preflight"] = preflight_payload

        if cfg.strict_preflight and not preflight.can_run_grouped_cv:
            reason = "Preflight failed: insufficient patient-level class coverage for grouped cross-validation."
            if dataset_cfg.skip_if_unavailable:
                _skip_or_raise(cfg, dataset_cfg, dataset_record, reason)
                manifest["datasets"].append(dataset_record)
                continue
            raise RuntimeError(reason)

        if cfg.preflight_only:
            dataset_record["status"] = "prepared"
            manifest["datasets"].append(dataset_record)
            continue

        dataset_record["status"] = "running"
        for model_name in cfg.run_models:
            model_output_dir = ensure_dir(dataset_root / model_name)
            try:
                result_path = _run_model(
                    model_name,
                    cfg,
                    study_dir,
                    model_output_dir,
                    effective_splits=effective_splits,
                )
                dataset_record["runs"][model_name] = {
                    "status": "completed",
                    "results_path": str(result_path),
                }
            except Exception as exc:
                dataset_record["runs"][model_name] = {
                    "status": "failed",
                    "error": str(exc),
                }
                dataset_record["status"] = "failed"
                if cfg.fail_on_model_error:
                    manifest["datasets"].append(dataset_record)
                    manifest_path = Path(root) / "benchmark_manifest.json"
                    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    raise

        if dataset_record["status"] != "failed":
            dataset_record["status"] = "completed"
        manifest["datasets"].append(dataset_record)

    manifest_path = Path(root) / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
