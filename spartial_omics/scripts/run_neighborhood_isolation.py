from __future__ import annotations

import argparse

from spartial_omics.evaluation.neighborhood_isolation import NeighborhoodIsolationConfig, evaluate_methods, save_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run neighborhood construction isolation experiments for spatial omics.")
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--jitter-std", type=float, default=0.05)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    cfg = NeighborhoodIsolationConfig(
        study_dir=args.study_dir,
        output_dir=args.output_dir,
        knn_k=args.knn_k,
        jitter_std=args.jitter_std,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )
    results = evaluate_methods(cfg.study_dir, cfg)
    save_results(results, cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
