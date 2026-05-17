from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetSource:
    source_type: str
    uri: str
    description: str


@dataclass(frozen=True)
class DatasetCatalogEntry:
    name: str
    adapter: str
    dataset_name: str
    modality: str
    disease: str
    resolution: str
    task_scope: str
    candidate_study_dirs: tuple[str, ...] = ()
    candidate_input_dirs: tuple[str, ...] = ()
    sources: tuple[DatasetSource, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass
class DatasetResolution:
    name: str
    adapter: str
    dataset_name: str
    status: str
    study_dir: str | None = None
    input_dir: str | None = None
    source_uris: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _abs(path: str) -> str:
    return str((PROJECT_ROOT / path).resolve())


CATALOG: dict[str, DatasetCatalogEntry] = {
    "crc_main": DatasetCatalogEntry(
        name="crc_main",
        adapter="prepared_study",
        dataset_name="cellsighter_crc_clinical",
        modality="multiplex imaging",
        disease="colorectal cancer",
        resolution="cell-resolved",
        task_scope="primary",
        candidate_study_dirs=(
            _abs("data/spatial_omics/prepared/cellsighter_crc_clinical"),
        ),
        notes=(
            "Primary CRC cohort for the thesis benchmark.",
        ),
    ),
    "crc_subset8": DatasetCatalogEntry(
        name="crc_subset8",
        adapter="prepared_study",
        dataset_name="cellsighter_crc_clinical_subset8",
        modality="multiplex imaging",
        disease="colorectal cancer",
        resolution="cell-resolved",
        task_scope="smoke",
        candidate_study_dirs=(
            _abs("data/spatial_omics/prepared/cellsighter_crc_clinical_subset8"),
        ),
        notes=("Small prepared subset for smoke tests and CI.",),
    ),
    "crc_test": DatasetCatalogEntry(
        name="crc_test",
        adapter="prepared_study",
        dataset_name="cellsighter_crc_test",
        modality="multiplex imaging",
        disease="colorectal cancer",
        resolution="cell-resolved",
        task_scope="smoke",
        candidate_study_dirs=(
            _abs("data/spatial_omics/prepared/cellsighter_crc_test"),
        ),
        notes=("Prepared tiny test study.",),
    ),
    "keren_tnbc": DatasetCatalogEntry(
        name="keren_tnbc",
        adapter="keren_tnbc_h5ad",
        dataset_name="keren_tnbc",
        modality="MIBI-TOF",
        disease="triple-negative breast cancer",
        resolution="cell-resolved",
        task_scope="supporting",
        candidate_study_dirs=(
            _abs("data/spatial_omics/prepared/keren_tnbc"),
        ),
        candidate_input_dirs=(
            _abs("data/spatial_omics/raw"),
            _abs("data/spatial_omics/raw/keren_tnbc"),
            _abs("data/spatial_omics/external/keren_tnbc"),
        ),
        sources=(
            DatasetSource(
                source_type="dataset-card",
                uri="https://figshare.com/articles/dataset/TNBC_h5ad_gz/26068006",
                description="Processed public TNBC MIBI-TOF h5ad derived from Keren et al. 2018.",
            ),
            DatasetSource(
                source_type="paper",
                uri="https://pubmed.ncbi.nlm.nih.gov/30193111/",
                description="Original Keren et al. Cell 2018 TNBC MIBI-TOF study.",
            ),
        ),
        notes=(
            "Best conceptual supporting dataset for topology-aware tumor architecture.",
            "Prepared study directory is preferred before model training.",
        ),
    ),
    "ali_breast_imc": DatasetCatalogEntry(
        name="ali_breast_imc",
        adapter="prepared_study",
        dataset_name="ali_breast_imc",
        modality="imaging mass cytometry",
        disease="breast cancer",
        resolution="cell-resolved",
        task_scope="supporting",
        candidate_study_dirs=(
            _abs("data/spatial_omics/prepared/ali_breast_imc"),
        ),
        candidate_input_dirs=(
            _abs("data/spatial_omics/raw/ali_breast_imc"),
            _abs("data/spatial_omics/external/ali_breast_imc"),
        ),
        sources=(
            DatasetSource(
                source_type="dataset-docs",
                uri="https://bodenmillergroup.github.io/imcdatasets/reference/JacksonFischer_2020_BreastCancer.html",
                description="imcdatasets entry for the JacksonFischer/Ali breast cancer IMC dataset.",
            ),
            DatasetSource(
                source_type="repository",
                uri="https://idr.openmicroscopy.org/study/idr0076/",
                description="IDR study entry for the Ali/Jackson-Fischer breast cancer IMC images.",
            ),
        ),
        notes=(
            "Best scale-oriented supporting dataset if label extraction is defensible.",
            "Prepared study directory is preferred before model training.",
        ),
    ),
}


def get_catalog_entry(name: str) -> DatasetCatalogEntry:
    if name not in CATALOG:
        raise KeyError(f"Unknown dataset catalog entry '{name}'. Available: {sorted(CATALOG)}")
    return CATALOG[name]


def resolve_catalog_entry(name: str) -> DatasetResolution:
    entry = get_catalog_entry(name)

    for study_dir in entry.candidate_study_dirs:
        study_meta = Path(study_dir) / "study_meta.json"
        if study_meta.is_file():
            return DatasetResolution(
                name=entry.name,
                adapter="prepared_study",
                dataset_name=entry.dataset_name,
                status="prepared",
                study_dir=study_dir,
                source_uris=[src.uri for src in entry.sources],
                notes=list(entry.notes),
            )

    for input_dir in entry.candidate_input_dirs:
        if Path(input_dir).exists():
            return DatasetResolution(
                name=entry.name,
                adapter=entry.adapter,
                dataset_name=entry.dataset_name,
                status="raw_available",
                input_dir=input_dir,
                source_uris=[src.uri for src in entry.sources],
                notes=list(entry.notes),
            )

    return DatasetResolution(
        name=entry.name,
        adapter=entry.adapter,
        dataset_name=entry.dataset_name,
        status="missing",
        source_uris=[src.uri for src in entry.sources],
        notes=list(entry.notes),
    )


def known_dataset_names() -> list[str]:
    return sorted(CATALOG)


__all__ = [
    "CATALOG",
    "DatasetCatalogEntry",
    "DatasetResolution",
    "DatasetSource",
    "get_catalog_entry",
    "known_dataset_names",
    "resolve_catalog_entry",
]
