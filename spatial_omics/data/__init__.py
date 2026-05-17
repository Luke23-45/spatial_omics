"""Data interfaces and adapters for spatial omics."""

from spatial_omics.data.adapters import PreparedStudyAdapter, ProcessedCellTableStudyAdapter, SpatialStudyAdapter, StudyDatasetSpec
from spatial_omics.data.anndata_compat import AnnData
from spatial_omics.data.catalog import (
    CATALOG,
    DatasetCatalogEntry,
    DatasetResolution,
    DatasetSource,
    get_catalog_entry,
    known_dataset_names,
    resolve_catalog_entry,
)
from spatial_omics.data.cellsighter_crc import CellSighterCRCTestDatasetAdapter
from spatial_omics.data.processed_crc import ProcessedCRCCODEXAdapter
from spatial_omics.data.registry import create_study_adapter, registered_study_adapters
from spatial_omics.data.source_manager import (
    DatasetSourceState,
    SOURCE_MANIFEST_FILENAME,
    ensure_catalog_source_available,
    dataset_source_state,
    source_inventory,
    write_dataset_source_state,
    write_source_inventory,
)
from spatial_omics.data.types import SpatialStudy

__all__ = [
    "AnnData",
    "CellSighterCRCTestDatasetAdapter",
    "CATALOG",
    "DatasetCatalogEntry",
    "DatasetResolution",
    "DatasetSource",
    "DatasetSourceState",
    "PreparedStudyAdapter",
    "ProcessedCRCCODEXAdapter",
    "ProcessedCellTableStudyAdapter",
    "SOURCE_MANIFEST_FILENAME",
    "SpatialStudy",
    "SpatialStudyAdapter",
    "StudyDatasetSpec",
    "create_study_adapter",
    "ensure_catalog_source_available",
    "dataset_source_state",
    "get_catalog_entry",
    "known_dataset_names",
    "registered_study_adapters",
    "resolve_catalog_entry",
    "source_inventory",
    "write_dataset_source_state",
    "write_source_inventory",
]
