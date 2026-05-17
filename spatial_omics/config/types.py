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
class ReportConfig:
    results_json: str
    output_dir: str
    study_dir: str | None = None
    features_path: str | None = None


@dataclass
class TopoNetHodgeConfig:
    study_dir: str
    output_dir: str
    features_path: str | None = None
    random_state: int = 42
    n_splits: int = 4
    n_repeats: int = 1
    split_mode: str = "grouped"
    batch_size: int = 12
    epochs: int = 12
    patience: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 5e-5
    knn_k: int = 6
    neighborhood_mode: str = "adaptive_knn"
    model_dim: int = 48
    type_embedding_dim: int = 16
    max_cells: int = 160
    dropout: float = 0.15
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    threshold_grid_size: int = 61
    num_layers: int = 2
    polynomial_order: int = 2
    restriction_hidden_dim: int = 32
    use_geometric_weights: bool = False
    use_orthogonal_restrictions: bool = False
    use_morse_gating: bool = False
    global_knn_multiplier: int = 4
    global_distance_multiplier: float = 2.5
    global_max_neighbors: int = 24


@dataclass
class GNNBaselineConfig:
    study_dir: str
    output_dir: str
    features_path: str | None = None
    random_state: int = 42
    n_splits: int = 4
    n_repeats: int = 1
    batch_size: int = 12
    epochs: int = 12
    patience: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 5e-5
    knn_k: int = 6
    neighborhood_mode: str = "adaptive_knn"
    model_dim: int = 48
    type_embedding_dim: int = 16
    max_cells: int = 160
    dropout: float = 0.15
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    temperature: float = 0.5
    threshold_grid_size: int = 61
    blend_with_engineered: bool = True
    model_names: tuple[str, ...] = ("graphsage",)
    split_mode: str = "grouped"


@dataclass
class MultiDatasetStudyConfig:
    name: str
    source_id: str | None = None
    adapter: str = "prepared_study"
    dataset_name: str | None = None
    study_dir: str | None = None
    input_dir: str | None = None
    label_column: str = "label"
    output_subdir: str | None = None
    required: bool = False
    skip_if_unavailable: bool = True
    max_splits_override: int | None = None
    notes: tuple[str, ...] = ()


@dataclass
class MultiDatasetBenchmarkConfig:
    datasets: list[MultiDatasetStudyConfig] = field(default_factory=list)
    output_dir: str = "outputs/active/multidataset_benchmark"
    features_path: str | None = None
    run_models: tuple[str, ...] = ("graphsage", "toponet_hodge")
    materialized_data_root: str | None = None
    auto_download_sources: bool = True
    fail_on_missing_required: bool = True
    fail_on_model_error: bool = True
    strict_preflight: bool = True
    auto_cap_splits: bool = True
    emit_dataset_state: bool = True
    preflight_only: bool = False
    random_state: int = 42
    n_splits: int = 4
    n_repeats: int = 1
    batch_size: int = 12
    epochs: int = 12
    patience: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 5e-5
    knn_k: int = 6
    neighborhood_mode: str = "adaptive_knn"
    model_dim: int = 48
    type_embedding_dim: int = 16
    max_cells: int = 160
    dropout: float = 0.15
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    threshold_grid_size: int = 61
    temperature: float = 0.5
    blend_with_engineered: bool = False
    num_layers: int = 2
    polynomial_order: int = 2
    restriction_hidden_dim: int = 32
    use_geometric_weights: bool = False
    use_orthogonal_restrictions: bool = False
    use_morse_gating: bool = False
    global_knn_multiplier: int = 4
    global_distance_multiplier: float = 2.5
    global_max_neighbors: int = 24


__all__ = [
    "FeatureConfig",
    "GNNBaselineConfig",
    "MultiDatasetBenchmarkConfig",
    "MultiDatasetStudyConfig",
    "PrepareConfig",
    "ReportConfig",
    "TopoNetHodgeConfig",
]
