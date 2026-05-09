from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from spatial_omics.data.anndata_compat import AnnData


@dataclass
class SpatialStudy:
    """Thin study container for spatial-omics analysis."""

    samples: dict[str, AnnData]
    sample_table: pd.DataFrame
    task_table: pd.DataFrame
    dataset_name: str
    study_meta: dict[str, object] = field(default_factory=dict)


__all__ = ["SpatialStudy"]
