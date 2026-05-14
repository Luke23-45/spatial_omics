from __future__ import annotations

import argparse

from spatial_omics.models.toponet_hodge import TopoNetHodgeConfig, run_toponet_hodge_study, save_toponet_hodge_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fair simplicial Hodge topology benchmark on a prepared spatial-omics study.")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument("--type-embedding-dim", type=int, default=16)
    parser.add_argument("--max-cells", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--threshold-grid-size", type=int, default=61)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--polynomial-order", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    cfg = TopoNetHodgeConfig(
        study_dir=args.study_dir,
        output_dir=args.output_dir,
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
        type_embedding_dim=args.type_embedding_dim,
        max_cells=args.max_cells,
        dropout=args.dropout,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        threshold_grid_size=args.threshold_grid_size,
        num_layers=args.num_layers,
        polynomial_order=args.polynomial_order,
        random_state=args.random_state,
    )
    results = run_toponet_hodge_study(cfg)
    save_toponet_hodge_results(results, cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
