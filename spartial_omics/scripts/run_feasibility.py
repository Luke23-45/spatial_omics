from __future__ import annotations

import argparse

import pandas as pd

from spartial_omics.config import EvaluationConfig, load_config
from spartial_omics.models.feasibility import run_feasibility_study, save_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run grouped-CV feasibility evaluation on extracted spatial-omics features.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--features-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config, EvaluationConfig)
    else:
        cfg = EvaluationConfig(features_path=args.features_path, output_dir=args.output_dir)
    feature_table = pd.read_csv(cfg.features_path)
    results = run_feasibility_study(feature_table, cfg)
    save_results(results, cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
