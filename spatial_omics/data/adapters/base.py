from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spatial_omics.data.types import SpatialStudy


@dataclass(frozen=True)
class StudyDatasetSpec:
    """Immutable dataset specification for multi-study benchmarking."""

    name: str
    adapter: str = "prepared_study"
    dataset_name: str | None = None
    study_dir: str | None = None
    input_dir: str | None = None
    label_column: str = "label"
    metadata: dict[str, Any] = field(default_factory=dict)


class SpatialStudyAdapter(ABC):
    """Adapter contract for any dataset that can yield a SpatialStudy."""

    @property
    @abstractmethod
    def spec(self) -> StudyDatasetSpec:
        ...

    @abstractmethod
    def load_study(self) -> SpatialStudy:
        ...

    @abstractmethod
    def materialize_study(self, output_root: str | Path | None = None) -> Path:
        """Return a prepared study directory suitable for model training."""
        ...


__all__ = ["SpatialStudyAdapter", "StudyDatasetSpec"]
