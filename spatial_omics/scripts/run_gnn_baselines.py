from __future__ import annotations

import argparse

from spatial_omics.models.gnn_baselines import run_gnn_baselines_study, save_gnn_baseline_results
from spatial_omics.config import load_config, GNNBaselineConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the canonical GraphSAGE grouped-CV baseline.")
    parser.add_argument("--config", required=True, help="Path to the GNNBaselineConfig yaml file.")
    args = parser.parse_args()

    cfg = load_config(args.config, GNNBaselineConfig)
    results = run_gnn_baselines_study(cfg)
    save_gnn_baseline_results(results, cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
