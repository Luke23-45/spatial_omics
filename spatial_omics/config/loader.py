from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

from spatial_omics.utils.io import load_yaml

T = TypeVar("T")


def _construct(cls: type[T], payload: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")
    type_hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for field_def in fields(cls):
        if field_def.name not in payload:
            continue
        annotation = type_hints.get(field_def.name, field_def.type)
        values[field_def.name] = _coerce_value(annotation, payload[field_def.name])
    return cls(**values)


def _coerce_value(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if value is None:
        return None

    if is_dataclass(annotation) and isinstance(value, dict):
        return _construct(annotation, value)

    if origin is list and args and isinstance(value, list):
        return [_coerce_value(args[0], item) for item in value]

    if origin is tuple and args and isinstance(value, (list, tuple)):
        item_type = args[0] if len(args) == 2 and args[1] is Ellipsis else None
        if item_type is not None:
            return tuple(_coerce_value(item_type, item) for item in value)
        return tuple(_coerce_value(arg, item) for arg, item in zip(args, value, strict=False))

    if origin is dict and isinstance(value, dict):
        key_type, val_type = args if len(args) == 2 else (Any, Any)
        return {
            _coerce_value(key_type, k): _coerce_value(val_type, v)
            for k, v in value.items()
        }

    if origin is not None and type(None) in args:
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            return _coerce_value(non_none[0], value)

    return value


def load_config(path: str, cls: type[T]) -> T:
    payload = load_yaml(path) or {}
    return _construct(cls, payload)


__all__ = ["load_config"]
