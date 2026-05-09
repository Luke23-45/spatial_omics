from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PrepareConfig:
    input_dir: str
    output_dir: str
    dataset_name: str = "crc_codex_processed"
    label_column: str = "label"
    patch_size_px: int = 256
    min_cells_per_region: int = 40
    label_mode: str = "proxy_patch_tme"
    clinical_annotations_path: str | None = None
    clinical_patient_id_column: str = "patient_id"
    clinical_label_column: str = "label"
    clinical_label_map: dict[str, str] = field(default_factory=dict)


@dataclass
class FeatureConfig:
    study_dir: str
    output_dir: str
    include_f0: bool = True
    include_f1: bool = True
    include_f2: bool = False
    knn_k: int = 10
    patch_size: float = 12.0
    min_patch_cells: int = 6
    top_k_lifetimes: int = 3
    betti_grid_size: int = 32
    tumor_terms: list[str] = field(default_factory=lambda: ["tumor", "epithelial"])
    cd8_terms: list[str] = field(default_factory=lambda: ["cd8"])
    treg_terms: list[str] = field(default_factory=lambda: ["treg", "regulatory t"])
    macrophage_terms: list[str] = field(default_factory=lambda: ["macrophage", "mono"])
    vessel_terms: list[str] = field(default_factory=lambda: ["vessel", "endothelial"])
    immune_terms: list[str] = field(
        default_factory=lambda: ["cd4", "cd8", "treg", "t cell", "b cell", "nk", "macrophage", "mono", "immune"]
    )


@dataclass
class EvaluationConfig:
    features_path: str
    output_dir: str
    random_state: int = 42
    n_splits: int = 4
    n_repeats: int = 2
    top_feature_k: int = 10
    model_names: list[str] = field(default_factory=list)
    split_mode: str = "grouped"


@dataclass
class ReportConfig:
    results_json: str
    output_dir: str
    study_dir: str | None = None
    features_path: str | None = None


__all__ = ["EvaluationConfig", "FeatureConfig", "PrepareConfig", "ReportConfig"]
