from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataclasses import asdict

from spatial_omics.data import known_dataset_names, source_inventory


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect centralized dataset source readiness for the multi-dataset benchmark.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset source ids to inspect.")
    parser.add_argument("--output", default=None, help="Optional path to save the JSON inspection report.")
    args = parser.parse_args()

    dataset_names = args.datasets or known_dataset_names()
    report = {"datasets": [asdict(item) for item in source_inventory(dataset_names)]}
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
