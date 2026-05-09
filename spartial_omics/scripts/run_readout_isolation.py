from __future__ import annotations

import argparse

from spartial_omics.evaluation.readout_isolation import ReadoutIsolationConfig, run_readout_isolation, save_readout_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated readout and calibration study for Spectra.")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--knn-k", type=int, default=6)
    parser.add_argument("--neighborhood-mode", choices=["adaptive_knn", "knn"], default="adaptive_knn")
    parser.add_argument("--model-dim", type=int, default=40)
    parser.add_argument("--anchor-count", type=int, default=8)
    parser.add_argument("--router-steps", type=int, default=3)
    parser.add_argument("--type-embedding-dim", type=int, default=16)
    parser.add_argument("--max-cells", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    cfg = ReadoutIsolationConfig(
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
        model_dim=args.model_dim,
        anchor_count=args.anchor_count,
        router_steps=args.router_steps,
        type_embedding_dim=args.type_embedding_dim,
        max_cells=args.max_cells,
        dropout=args.dropout,
        random_state=args.random_state,
    )
    results = run_readout_isolation(cfg)
    save_readout_results(results, cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
