from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

from spatial_omics.utils.io import load_yaml

T = TypeVar("T")


def _construct(cls: type[T], payload: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")
    valid = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in payload.items() if k in valid}
    return cls(**filtered)


def load_config(path: str, cls: type[T]) -> T:
    payload = load_yaml(path) or {}
    return _construct(cls, payload)


__all__ = ["load_config"]
