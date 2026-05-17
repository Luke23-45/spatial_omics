from __future__ import annotations

import inspect
from typing import Any

from spatial_omics.data.adapters import (
    KerenTNBCH5ADStudyAdapter,
    PreparedStudyAdapter,
    ProcessedCellTableStudyAdapter,
    SpatialStudyAdapter,
)


_ADAPTERS: dict[str, type[SpatialStudyAdapter]] = {
    "keren_tnbc_h5ad": KerenTNBCH5ADStudyAdapter,
    "prepared_study": PreparedStudyAdapter,
    "processed_cell_table": ProcessedCellTableStudyAdapter,
}


def register_study_adapter(name: str, cls: type[SpatialStudyAdapter]) -> None:
    _ADAPTERS[name] = cls


def create_study_adapter(adapter_name: str, **kwargs: Any) -> SpatialStudyAdapter:
    if adapter_name not in _ADAPTERS:
        raise ValueError(f"Unknown study adapter '{adapter_name}'. Available: {sorted(_ADAPTERS)}")
    cls = _ADAPTERS[adapter_name]
    sig = inspect.signature(cls.__init__)
    valid = {
        p.name
        for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return cls(**filtered)


def registered_study_adapters() -> list[str]:
    return sorted(_ADAPTERS)


__all__ = ["create_study_adapter", "register_study_adapter", "registered_study_adapters"]
