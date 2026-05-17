from __future__ import annotations

from pathlib import Path

from spatial_omics.data.adapters.base import SpatialStudyAdapter, StudyDatasetSpec
from spatial_omics.data.io import save_study
from spatial_omics.data.processed_crc import ProcessedCRCCODEXAdapter
from spatial_omics.data.types import SpatialStudy


class ProcessedCellTableStudyAdapter(SpatialStudyAdapter):
    """Adapter for generic cell-level processed tables stored as CSV files."""

    def __init__(
        self,
        *,
        name: str,
        input_dir: str,
        dataset_name: str | None = None,
        adapter: str = "processed_cell_table",
        label_column: str = "label",
        metadata: dict | None = None,
    ) -> None:
        self._spec = StudyDatasetSpec(
            name=name,
            adapter=adapter,
            dataset_name=dataset_name or name,
            input_dir=input_dir,
            label_column=label_column,
            metadata=metadata or {},
        )

    @property
    def spec(self) -> StudyDatasetSpec:
        return self._spec

    def load_study(self) -> SpatialStudy:
        input_dir = self._required_input_dir()
        loader = ProcessedCRCCODEXAdapter(
            input_dir=input_dir,
            dataset_name=self._spec.dataset_name or self._spec.name,
            label_column=self._spec.label_column,
        )
        return loader.load_study()

    def materialize_study(self, output_root: str | Path | None = None) -> Path:
        if output_root is None:
            raise ValueError(
                f"ProcessedCellTableStudyAdapter for '{self._spec.name}' requires output_root to materialize a study."
            )
        target = Path(output_root) / self._spec.name
        if (target / "study_meta.json").is_file():
            return target
        study = self.load_study()
        save_study(study, target)
        return target

    def _required_input_dir(self) -> Path:
        if not self._spec.input_dir:
            raise ValueError(f"ProcessedCellTableStudyAdapter for '{self._spec.name}' requires input_dir.")
        return Path(self._spec.input_dir)


__all__ = ["ProcessedCellTableStudyAdapter"]
