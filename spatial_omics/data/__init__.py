"""Data interfaces and adapters for spatial omics."""

from spatial_omics.data.anndata_compat import AnnData
from spatial_omics.data.cellsighter_crc import CellSighterCRCTestDatasetAdapter
from spatial_omics.data.processed_crc import ProcessedCRCCODEXAdapter
from spatial_omics.data.types import SpatialStudy

__all__ = ["AnnData", "CellSighterCRCTestDatasetAdapter", "ProcessedCRCCODEXAdapter", "SpatialStudy"]
