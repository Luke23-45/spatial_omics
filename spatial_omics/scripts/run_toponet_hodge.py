from __future__ import annotations

import argparse

from spatial_omics.models.toponet_hodge import run_toponet_hodge_study, save_toponet_hodge_results
from spatial_omics.config import load_config, TopoNetHodgeConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the TopoNet Hodge benchmark.")
    parser.add_argument("--config", required=True, help="Path to the TopoNetHodgeConfig yaml file.")
    args = parser.parse_args()

    cfg = load_config(args.config, TopoNetHodgeConfig)
    results = run_toponet_hodge_study(cfg)
    save_toponet_hodge_results(results, cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
