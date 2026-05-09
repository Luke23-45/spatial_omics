from __future__ import annotations

import argparse

from spatial_omics.config import FeatureConfig, load_config
from spatial_omics.features.pipeline import extract_feature_families, save_feature_table


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract F0/F1 feature families from a prepared spatial study.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--study-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config, FeatureConfig)
    else:
        cfg = FeatureConfig(study_dir=args.study_dir, output_dir=args.output_dir)
    feature_table = extract_feature_families(cfg.study_dir, cfg)
    save_feature_table(feature_table, cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
