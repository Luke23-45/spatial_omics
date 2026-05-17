from __future__ import annotations

from pathlib import Path

from spatial_omics.data.adapters.base import SpatialStudyAdapter, StudyDatasetSpec
from spatial_omics.data.io import load_study
from spatial_omics.data.types import SpatialStudy


class PreparedStudyAdapter(SpatialStudyAdapter):
    """Adapter for already-prepared studies persisted via save_study()."""

    def __init__(
        self,
        *,
        name: str,
        study_dir: str,
        dataset_name: str | None = None,
        adapter: str = "prepared_study",
        label_column: str = "label",
        metadata: dict | None = None,
    ) -> None:
        self._spec = StudyDatasetSpec(
            name=name,
            adapter=adapter,
            dataset_name=dataset_name or name,
            study_dir=study_dir,
            label_column=label_column,
            metadata=metadata or {},
        )

    @property
    def spec(self) -> StudyDatasetSpec:
        return self._spec

    def load_study(self) -> SpatialStudy:
        return load_study(self._required_study_dir())

    def materialize_study(self, output_root: str | Path | None = None) -> Path:
        return self._required_study_dir()

    def _required_study_dir(self) -> Path:
        if not self._spec.study_dir:
            raise ValueError(f"PreparedStudyAdapter for '{self._spec.name}' requires study_dir.")
        return Path(self._spec.study_dir)


__all__ = ["PreparedStudyAdapter"]
