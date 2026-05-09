from __future__ import annotations

import argparse
import json

from spartial_omics.config import ReportConfig, load_config
from spartial_omics.reporting.markdown import build_markdown_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Markdown report and plots from spatial-omics feasibility results.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--study-dir", default=None)
    parser.add_argument("--features-path", default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config, ReportConfig)
    else:
        cfg = ReportConfig(
            results_json=args.results_json,
            output_dir=args.output_dir,
            study_dir=args.study_dir,
            features_path=args.features_path,
        )
    results = json.loads(open(cfg.results_json, encoding="utf-8").read())
    build_markdown_report(results, cfg.output_dir, study_dir=cfg.study_dir, features_path=cfg.features_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
