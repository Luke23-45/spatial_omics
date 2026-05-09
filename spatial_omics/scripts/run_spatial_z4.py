from __future__ import annotations

import argparse

from spatial_omics.models.spatial_z4 import SpatialZ4Config, run_spatial_z4_study, save_spatial_z4_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the canonical retained spatial_z4_v2 grouped-CV topology model on a prepared spatial-omics study.")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-results-json", default=None)
    parser.add_argument("--features-path", default=None)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--knn-k", type=int, default=6)
    parser.add_argument("--neighborhood-mode", choices=["adaptive_knn", "knn"], default="adaptive_knn")
    parser.add_argument("--model-dim", type=int, default=48)
    parser.add_argument("--anchor-count", type=int, default=8)
    parser.add_argument("--router-steps", type=int, default=3)
    parser.add_argument("--lift-mode", choices=["identity", "normalized_linear"], default="identity")
    parser.add_argument("--lift-dim", type=int, default=12)
    parser.add_argument("--type-embedding-dim", type=int, default=16)
    parser.add_argument("--max-cells", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--threshold-grid-size", type=int, default=61)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--router-heads", type=int, default=4)
    parser.add_argument("--global-knn-multiplier", type=int, default=4)
    parser.add_argument("--global-distance-multiplier", type=float, default=2.5)
    parser.add_argument("--global-max-neighbors", type=int, default=24)
    parser.add_argument("--cell-drop-rate", type=float, default=0.15)
    parser.add_argument("--calibration-mode", choices=["temperature", "confidence_temperature", "platt"], default="platt")
    parser.add_argument("--temperature-confidence-scale", type=float, default=0.75)
    parser.add_argument("--temperature-confidence-power", type=float, default=2.0)
    parser.add_argument("--head-min-auroc", type=float, default=0.55)
    parser.add_argument("--head-max-auroc-gap", type=float, default=0.20)
    parser.add_argument("--blend-auroc-tolerance", type=float, default=0.005)
    parser.add_argument("--checkpoint-auroc-weight", type=float, default=0.4)
    parser.add_argument("--aux-graph-weight", type=float, default=0.35)
    parser.add_argument("--aux-topology-weight", type=float, default=0.15)
    parser.add_argument("--aux-graph-weight-late", type=float, default=0.20)
    parser.add_argument("--aux-topology-weight-late", type=float, default=0.10)
    parser.add_argument("--aux-decay-start-epoch", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    cfg = SpatialZ4Config(
        study_dir=args.study_dir,
        output_dir=args.output_dir,
        baseline_results_json=args.baseline_results_json,
        features_path=args.features_path,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        knn_k=args.knn_k,
        neighborhood_mode=args.neighborhood_mode,
        model_dim=args.model_dim,
        anchor_count=args.anchor_count,
        router_steps=args.router_steps,
        lift_mode=args.lift_mode,
        lift_dim=args.lift_dim,
        type_embedding_dim=args.type_embedding_dim,
        max_cells=args.max_cells,
        dropout=args.dropout,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        temperature=args.temperature,
        threshold_grid_size=args.threshold_grid_size,
        local_block_mode="mean_residual",
        graph_layers=args.graph_layers,
        router_mode="gru_anchor",
        fusion_mode="feature_readout",
        use_engineered_context=False,
        blend_with_engineered=False,
        gat_heads=args.gat_heads,
        router_heads=args.router_heads,
        global_knn_multiplier=args.global_knn_multiplier,
        global_distance_multiplier=args.global_distance_multiplier,
        global_max_neighbors=args.global_max_neighbors,
        cell_drop_rate=args.cell_drop_rate,
        calibration_mode=args.calibration_mode,
        temperature_confidence_scale=args.temperature_confidence_scale,
        temperature_confidence_power=args.temperature_confidence_power,
        head_min_auroc=args.head_min_auroc,
        head_max_auroc_gap=args.head_max_auroc_gap,
        blend_auroc_tolerance=args.blend_auroc_tolerance,
        checkpoint_auroc_weight=args.checkpoint_auroc_weight,
        aux_graph_weight=args.aux_graph_weight,
        aux_topology_weight=args.aux_topology_weight,
        aux_graph_weight_late=args.aux_graph_weight_late,
        aux_topology_weight_late=args.aux_topology_weight_late,
        aux_decay_start_epoch=args.aux_decay_start_epoch,
        random_state=args.random_state,
    )
    results = run_spatial_z4_study(cfg)
    save_spatial_z4_results(results, cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
