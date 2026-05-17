from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def validate_expected_labels(
    labels: pd.Series | list[str] | tuple[str, ...],
    *,
    expected: set[str] | tuple[str, ...] | list[str],
    context: str,
) -> list[str]:
    observed = sorted({str(label) for label in labels})
    expected_set = {str(label) for label in expected}
    observed_set = set(observed)
    if observed_set != expected_set:
        raise ValueError(
            f"{context} label mismatch. Expected {sorted(expected_set)}, observed {observed}."
        )
    return observed
