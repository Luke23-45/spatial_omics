from __future__ import annotations

import json
from pathlib import Path

from spatial_omics.data.adapters.base import SpatialStudyAdapter, StudyDatasetSpec
from spatial_omics.data.io import save_study
from spatial_omics.data.keren_tnbc import KerenTNBCH5ADAdapter
from spatial_omics.data.types import SpatialStudy


class KerenTNBCH5ADStudyAdapter(SpatialStudyAdapter):
    def __init__(
        self,
        *,
        name: str,
        input_dir: str,
        dataset_name: str | None = None,
        adapter: str = "keren_tnbc_h5ad",
        label_column: str = "subtype",
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
        loader = KerenTNBCH5ADAdapter(
            input_dir=self._required_input_dir(),
            dataset_name=self._spec.dataset_name or self._spec.name,
            label_column=self._spec.label_column,
        )
        return loader.load_study()

    def materialize_study(self, output_root: str | Path | None = None) -> Path:
        if output_root is None:
            raise ValueError(
                f"KerenTNBCH5ADStudyAdapter for '{self._spec.name}' requires output_root to materialize a study."
            )
        target = Path(output_root) / self._spec.name
        meta_path = target / "study_meta.json"
        if meta_path.is_file() and self._is_compatible_materialization(meta_path):
            return target
        study = self.load_study()
        save_study(study, target)
        return target

    def _required_input_dir(self) -> Path:
        if not self._spec.input_dir:
            raise ValueError(f"KerenTNBCH5ADStudyAdapter for '{self._spec.name}' requires input_dir.")
        return Path(self._spec.input_dir)

    def _is_compatible_materialization(self, meta_path: Path) -> bool:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if meta.get("dataset_name") != (self._spec.dataset_name or self._spec.name):
            return False
        study_meta = meta.get("study_meta", {})
        return study_meta.get("source_schema") == "keren_tnbc_h5ad"


__all__ = ["KerenTNBCH5ADStudyAdapter"]
