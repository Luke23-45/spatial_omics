from __future__ import annotations

import argparse

from spartial_omics.evaluation.lift_isolation import LiftIsolationConfig, run_lift_isolation, save_lift_isolation_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated geometric-lift study for Spectra.")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--knn-k", type=int, default=6)
    parser.add_argument("--neighborhood-mode", choices=["adaptive_knn", "knn"], default="adaptive_knn")
    parser.add_argument("--lift-dim", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-cells", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    cfg = LiftIsolationConfig(
        study_dir=args.study_dir,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        knn_k=args.knn_k,
        neighborhood_mode=args.neighborhood_mode,
        lift_dim=args.lift_dim,
        hidden_dim=args.hidden_dim,
        max_cells=args.max_cells,
        dropout=args.dropout,
        random_state=args.random_state,
    )
    results = run_lift_isolation(cfg)
    save_lift_isolation_results(results, cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
