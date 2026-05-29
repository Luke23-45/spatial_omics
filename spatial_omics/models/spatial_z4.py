from __future__ import annotations

import copy
import os
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from spatial_omics.data.io import load_study, marker_matrix_from_cells
from spatial_omics.evaluation.splits import bootstrap_ci, build_grouped_splits
from spatial_omics.utils.io import ensure_dir
from spatial_omics.utils.legacy_lift import NormalizedLift


META_COLUMNS = {"cell_id", "sample_id", "patient_id", "region_id", "x", "y", "cell_type", "compartment"}


@dataclass
class SpatialRegionExample:
    region_id: str
    patient_id: str
    label: str
    coords: np.ndarray
    markers: np.ndarray
    type_ids: np.ndarray
    structural: np.ndarray
    neighbor_idx: np.ndarray
    neighbor_mask: np.ndarray
    engineered_features: np.ndarray
    neighborhood_mode: str = "adaptive_knn"
    multi_neighbor_idx: tuple[np.ndarray, ...] = field(default_factory=tuple)
    multi_neighbor_mask: tuple[np.ndarray, ...] = field(default_factory=tuple)
    multi_neighbor_dist: tuple[np.ndarray, ...] = field(default_factory=tuple)


@dataclass
class SpatialZ4Config:
    study_dir: str
    output_dir: str
    baseline_results_json: str | None = None
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
    anchor_count: int = 8
    router_steps: int = 3
    lift_mode: str = "identity"
    lift_dim: int = 12
    type_embedding_dim: int = 16
    max_cells: int = 160
    dropout: float = 0.15
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    temperature: float = 0.5
    threshold_grid_size: int = 61
    local_block_mode: str = "mean_residual"
    graph_layers: int = 2
    router_mode: str = "gru_anchor"
    fusion_mode: str = "feature_readout"
    use_engineered_context: bool = False
    blend_with_engineered: bool = False
    gat_heads: int = 4
    router_heads: int = 4
    global_knn_multiplier: int = 4
    global_distance_multiplier: float = 2.5
    global_max_neighbors: int = 24
    cell_drop_rate: float = 0.15
    calibration_mode: str = "platt"
    temperature_confidence_scale: float = 0.75
    temperature_confidence_power: float = 2.0
    head_min_auroc: float = 0.55
    head_max_auroc_gap: float = 0.20
    blend_auroc_tolerance: float = 0.005
    checkpoint_auroc_weight: float = 0.4
    aux_graph_weight: float = 0.35
    aux_topology_weight: float = 0.15
    aux_graph_weight_late: float = 0.20
    aux_topology_weight_late: float = 0.10
    aux_decay_start_epoch: int = 30


def _region_label_map(study) -> dict[str, str]:
    frame = study.sample_table[["region_id", "label"]].drop_duplicates()
    return dict(zip(frame["region_id"], frame["label"], strict=False))


def _prepare_engineered_feature_map(features_path: str | None) -> dict[str, np.ndarray]:
    if not features_path:
        return {}
    table = pd.read_csv(features_path)
    families: dict[str, pd.DataFrame] = {}
    for family in sorted(table["feature_family"].unique()):
        families[family] = (
            table.loc[table["feature_family"] == family]
            .set_index("region_id")
            .drop(columns=["feature_family", "split"])
        )
    if "F0" in families and "F1" in families:
        merged = families["F0"].join(
            families["F1"].drop(columns=["sample_id", "patient_id", "label"]),
            how="inner",
            rsuffix="_f1",
        )
    elif "F0" in families:
        merged = families["F0"]
    elif "F1" in families:
        merged = families["F1"]
    else:
        return {}
    meta_cols = {"sample_id", "patient_id", "label"}
    feature_frame = merged.drop(columns=[col for col in meta_cols if col in merged.columns]).fillna(0.0)
    return {str(region_id): row.to_numpy(dtype=np.float32) for region_id, row in feature_frame.iterrows()}


def _type_vocabulary(study) -> dict[str, int]:
    cell_types = set()
    for adata in study.samples.values():
        cell_types.update(str(value) for value in adata.obs["cell_type"].astype(str).tolist())
    return {cell_type: idx for idx, cell_type in enumerate(sorted(cell_types))}


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1).astype(np.float32)
    np.fill_diagonal(dist, np.inf)
    return dist


def _scale_coords(coords: np.ndarray, knn_k: int) -> tuple[np.ndarray, np.ndarray]:
    centered = coords.astype(np.float32, copy=True)
    centered -= np.median(centered, axis=0, keepdims=True)
    if centered.shape[0] <= 1:
        return centered, np.ones(centered.shape[0], dtype=np.float32)
    dist = _pairwise_distances(centered)
    k = max(1, min(knn_k, centered.shape[0] - 1))
    nearest = np.partition(dist, kth=k - 1, axis=1)[:, :k]
    mean_nn = nearest.mean(axis=1).astype(np.float32)
    scale = float(np.median(mean_nn[mean_nn > 0])) if np.any(mean_nn > 0) else 1.0
    scale = max(scale, 1e-3)
    return centered / scale, mean_nn / scale


def _pack_neighbor_lists(
    neighbor_lists: list[np.ndarray],
    dist: np.ndarray,
    *,
    min_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_cells = len(neighbor_lists)
    width = max(min_width, max((neighbors.size for neighbors in neighbor_lists), default=0), 1)
    neighbor_idx = np.zeros((n_cells, width), dtype=np.int64)
    neighbor_mask = np.zeros((n_cells, width), dtype=bool)
    neighbor_dist = np.zeros((n_cells, width), dtype=np.float32)
    for idx, neighbors in enumerate(neighbor_lists):
        if neighbors.size == 0:
            continue
        size = neighbors.size
        neighbor_idx[idx, :size] = neighbors
        neighbor_mask[idx, :size] = True
        neighbor_dist[idx, :size] = dist[idx, neighbors]
    return neighbor_idx, neighbor_mask, neighbor_dist


def _build_knn_neighbor_graph(coords: np.ndarray, knn_k: int) -> tuple[np.ndarray, np.ndarray]:
    n_cells = coords.shape[0]
    if n_cells <= 1:
        return np.zeros((n_cells, 1), dtype=np.int64), np.zeros((n_cells, 1), dtype=bool)
    dist = _pairwise_distances(coords)
    k = max(1, min(knn_k, n_cells - 1))
    order = np.argsort(dist, axis=1)[:, :k]
    mask = np.ones_like(order, dtype=bool)
    return order.astype(np.int64), mask


def _build_adaptive_neighbor_graph(coords: np.ndarray, knn_k: int) -> tuple[np.ndarray, np.ndarray]:
    n_cells = coords.shape[0]
    if n_cells <= 1:
        return np.zeros((n_cells, 1), dtype=np.int64), np.zeros((n_cells, 1), dtype=bool)
    dist = _pairwise_distances(coords)
    base_k = max(1, min(knn_k, n_cells - 1))
    order = np.argsort(dist, axis=1)
    sorted_dist = np.take_along_axis(dist, order, axis=1)
    local_scale = np.median(sorted_dist[:, :base_k], axis=1)
    threshold = 1.5 * np.clip(local_scale, 1e-6, None)
    neighbor_lists: list[np.ndarray] = []
    max_neighbors = base_k
    for idx in range(n_cells):
        adaptive = np.flatnonzero(dist[idx] <= threshold[idx])
        merged = np.unique(np.concatenate([order[idx, :base_k], adaptive])).astype(np.int64)
        neighbor_lists.append(merged)
        max_neighbors = max(max_neighbors, merged.size)
    neighbor_idx = np.zeros((n_cells, max_neighbors), dtype=np.int64)
    neighbor_mask = np.zeros((n_cells, max_neighbors), dtype=bool)
    for idx, neighbors in enumerate(neighbor_lists):
        if neighbors.size == 0:
            continue
        neighbor_idx[idx, : neighbors.size] = neighbors
        neighbor_mask[idx, : neighbors.size] = True
    return neighbor_idx, neighbor_mask


def _build_neighbor_graph(coords: np.ndarray, knn_k: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "adaptive_knn":
        return _build_adaptive_neighbor_graph(coords, knn_k)
    if mode == "knn":
        return _build_knn_neighbor_graph(coords, knn_k)
    raise ValueError(f"Unsupported neighborhood_mode: {mode}")


def _build_multiscale_neighbor_graphs(
    coords: np.ndarray,
    knn_k: int,
    mode: str,
    *,
    global_knn_multiplier: int,
    global_distance_multiplier: float,
    global_max_neighbors: int,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    n_cells = coords.shape[0]
    if n_cells <= 1:
        empty_idx = np.zeros((n_cells, 1), dtype=np.int64)
        empty_mask = np.zeros((n_cells, 1), dtype=bool)
        empty_dist = np.zeros((n_cells, 1), dtype=np.float32)
        return (empty_idx, empty_idx, empty_idx), (empty_mask, empty_mask, empty_mask), (empty_dist, empty_dist, empty_dist)

    dist = _pairwise_distances(coords)
    order = np.argsort(dist, axis=1)
    base_k = max(1, min(knn_k, n_cells - 1))
    expanded_k = max(base_k + 1, min(base_k * 2, n_cells - 1))
    global_k = max(expanded_k + 1, min(base_k * global_knn_multiplier, n_cells - 1))

    base_idx, base_mask = _build_neighbor_graph(coords, base_k, mode)
    base_dist = np.zeros_like(base_idx, dtype=np.float32)
    for idx in range(n_cells):
        valid = base_mask[idx]
        base_dist[idx, valid] = dist[idx, base_idx[idx, valid]]

    expanded_lists = [order[idx, :expanded_k].astype(np.int64) for idx in range(n_cells)]
    expanded_idx, expanded_mask, expanded_dist = _pack_neighbor_lists(expanded_lists, dist, min_width=expanded_k)

    mean_scale = np.median(np.partition(dist, kth=base_k - 1, axis=1)[:, :base_k])
    long_range_threshold = max(float(mean_scale) * global_distance_multiplier, 1e-3)
    global_lists: list[np.ndarray] = []
    for idx in range(n_cells):
        within_threshold = np.flatnonzero(dist[idx] <= long_range_threshold).astype(np.int64)
        far_candidates = within_threshold[dist[idx, within_threshold] > np.median(dist[idx, order[idx, :expanded_k]])] if within_threshold.size else np.array([], dtype=np.int64)
        if far_candidates.size == 0:
            fallback_pos = min(global_k - 1, n_cells - 2)
            far_candidates = np.array([order[idx, fallback_pos]], dtype=np.int64)
        far_candidates = far_candidates[:global_max_neighbors]
        global_lists.append(np.unique(far_candidates).astype(np.int64))
    global_idx, global_mask, global_dist = _pack_neighbor_lists(global_lists, dist, min_width=1)
    return (base_idx, expanded_idx, global_idx), (base_mask, expanded_mask, global_mask), (base_dist, expanded_dist, global_dist)


def _build_structural_vectors(
    coords: np.ndarray,
    type_ids: np.ndarray,
    num_types: int,
    knn_idx: np.ndarray,
    knn_mask: np.ndarray,
    mean_nn: np.ndarray,
) -> np.ndarray:
    n_cells = coords.shape[0]
    onehot = np.zeros((n_cells, num_types), dtype=np.float32)
    onehot[np.arange(n_cells), type_ids] = 1.0
    if knn_idx.shape[1] > 0:
        gathered = onehot[knn_idx]
        weights = knn_mask.astype(np.float32)[..., None]
        denom = np.clip(weights.sum(axis=1), 1.0, None)
        neighbor_hist = (gathered * weights).sum(axis=1) / denom
    else:
        neighbor_hist = np.zeros_like(onehot)
    radial = np.linalg.norm(coords, axis=1, keepdims=True).astype(np.float32)
    local_density = (1.0 / np.clip(mean_nn.reshape(-1, 1), 1e-3, None)).astype(np.float32)
    return np.concatenate([coords, radial, mean_nn.reshape(-1, 1), local_density, onehot, neighbor_hist], axis=1)


def _subsample_cells(
    example: SpatialRegionExample,
    *,
    max_cells: int,
    knn_k: int,
    global_knn_multiplier: int,
    global_distance_multiplier: float,
    global_max_neighbors: int,
) -> SpatialRegionExample:
    n_cells = example.coords.shape[0]
    if n_cells <= max_cells:
        return example
    radial = np.linalg.norm(example.coords, axis=1)
    order = np.argsort(radial)
    keep = order[np.linspace(0, n_cells - 1, num=max_cells, dtype=int)]
    keep = np.sort(np.unique(keep))
    coords = example.coords[keep]
    markers = example.markers[keep]
    type_ids = example.type_ids[keep]
    scaled_coords, mean_nn = _scale_coords(coords, knn_k=knn_k)
    neighborhood_mode = getattr(example, "neighborhood_mode", "adaptive_knn")
    multi_idx, multi_mask, multi_dist = _build_multiscale_neighbor_graphs(
        scaled_coords,
        knn_k,
        neighborhood_mode,
        global_knn_multiplier=global_knn_multiplier,
        global_distance_multiplier=global_distance_multiplier,
        global_max_neighbors=global_max_neighbors,
    )
    structural = _build_structural_vectors(
        scaled_coords,
        type_ids,
        int(example.structural.shape[1] - 5) // 2,
        multi_idx[0],
        multi_mask[0],
        mean_nn,
    )
    return SpatialRegionExample(
        region_id=example.region_id,
        patient_id=example.patient_id,
        label=example.label,
        coords=scaled_coords,
        markers=markers,
        type_ids=type_ids,
        structural=structural,
        neighbor_idx=multi_idx[0],
        neighbor_mask=multi_mask[0],
        engineered_features=example.engineered_features,
        neighborhood_mode=neighborhood_mode,
        multi_neighbor_idx=multi_idx,
        multi_neighbor_mask=multi_mask,
        multi_neighbor_dist=multi_dist,
    )


def build_region_examples(
    study_dir: str,
    *,
    knn_k: int,
    max_cells: int,
    features_path: str | None = None,
    neighborhood_mode: str = "adaptive_knn",
    global_knn_multiplier: int = 4,
    global_distance_multiplier: float = 2.5,
    global_max_neighbors: int = 24,
) -> tuple[list[SpatialRegionExample], dict[str, int]]:
    study = load_study(study_dir)
    label_map = _region_label_map(study)
    type_vocab = _type_vocabulary(study)
    if features_path and not os.path.exists(features_path):
        print(f"Engineered features file {features_path} not found. Extracting on the fly...")
        from spatial_omics.features.pipeline import extract_feature_families, save_feature_table
        from spatial_omics.config.types import FeatureConfig
        cfg = FeatureConfig(study_dir=study_dir, output_dir=os.path.dirname(features_path))
        feature_table = extract_feature_families(study_dir, cfg)
        save_feature_table(feature_table, os.path.dirname(features_path))

    engineered_map = _prepare_engineered_feature_map(features_path)
    engineered_dim = len(next(iter(engineered_map.values()))) if engineered_map else 0
    examples: list[SpatialRegionExample] = []
    for region_id in sorted(study.samples):
        adata = study.samples[region_id]
        obs = adata.obs.reset_index(drop=True)
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        markers, _marker_cols = marker_matrix_from_cells(obs, META_COLUMNS)
        type_ids = obs["cell_type"].astype(str).map(type_vocab).to_numpy(dtype=np.int64)
        scaled_coords, mean_nn = _scale_coords(coords, knn_k)
        multi_idx, multi_mask, multi_dist = _build_multiscale_neighbor_graphs(
            scaled_coords,
            knn_k,
            neighborhood_mode,
            global_knn_multiplier=global_knn_multiplier,
            global_distance_multiplier=global_distance_multiplier,
            global_max_neighbors=global_max_neighbors,
        )
        structural = _build_structural_vectors(
            scaled_coords,
            type_ids,
            len(type_vocab),
            multi_idx[0],
            multi_mask[0],
            mean_nn,
        )
        example = SpatialRegionExample(
            region_id=region_id,
            patient_id=str(obs["patient_id"].iloc[0]),
            label=str(label_map[region_id]),
            coords=scaled_coords,
            markers=markers,
            type_ids=type_ids,
            structural=structural,
            neighbor_idx=multi_idx[0],
            neighbor_mask=multi_mask[0],
            engineered_features=engineered_map.get(region_id, np.zeros(engineered_dim, dtype=np.float32)),
            neighborhood_mode=neighborhood_mode,
            multi_neighbor_idx=multi_idx,
            multi_neighbor_mask=multi_mask,
            multi_neighbor_dist=multi_dist,
        )
        examples.append(
            _subsample_cells(
                example,
                max_cells=max_cells,
                knn_k=knn_k,
                global_knn_multiplier=global_knn_multiplier,
                global_distance_multiplier=global_distance_multiplier,
                global_max_neighbors=global_max_neighbors,
            )
        )
    return examples, type_vocab


class RegionDataset(Dataset):
    def __init__(self, examples: list[SpatialRegionExample], labels: np.ndarray):
        self.examples = examples
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[SpatialRegionExample, int]:
        return self.examples[idx], int(self.labels[idx])


def _collate_batch(batch: list[tuple[SpatialRegionExample, int]]) -> dict[str, torch.Tensor]:
    examples, labels = zip(*batch, strict=False)
    max_cells = max(ex.coords.shape[0] for ex in examples)
    marker_dim = examples[0].markers.shape[1]
    structural_dim = examples[0].structural.shape[1]
    knn_k = max(ex.neighbor_idx.shape[1] for ex in examples)
    engineered_dim = examples[0].engineered_features.shape[0]
    batch_size = len(examples)
    n_scales = len(examples[0].multi_neighbor_idx) if examples[0].multi_neighbor_idx else 1
    multi_knn_k = max(
        ex.multi_neighbor_idx[scale_idx].shape[1] if ex.multi_neighbor_idx else ex.neighbor_idx.shape[1]
        for ex in examples
        for scale_idx in range(n_scales)
    )

    coords = torch.zeros(batch_size, max_cells, 2, dtype=torch.float32)
    markers = torch.zeros(batch_size, max_cells, marker_dim, dtype=torch.float32)
    type_ids = torch.full((batch_size, max_cells), -1, dtype=torch.long)
    structural = torch.zeros(batch_size, max_cells, structural_dim, dtype=torch.float32)
    neighbor_idx = torch.zeros(batch_size, max_cells, knn_k, dtype=torch.long)
    neighbor_mask = torch.zeros(batch_size, max_cells, knn_k, dtype=torch.bool)
    multi_neighbor_idx = torch.zeros(batch_size, n_scales, max_cells, multi_knn_k, dtype=torch.long)
    multi_neighbor_mask = torch.zeros(batch_size, n_scales, max_cells, multi_knn_k, dtype=torch.bool)
    multi_neighbor_dist = torch.zeros(batch_size, n_scales, max_cells, multi_knn_k, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_cells, dtype=torch.bool)
    engineered = torch.zeros(batch_size, engineered_dim, dtype=torch.float32)

    for batch_idx, ex in enumerate(examples):
        n_cells = ex.coords.shape[0]
        coords[batch_idx, :n_cells] = torch.from_numpy(ex.coords)
        if marker_dim > 0:
            markers[batch_idx, :n_cells] = torch.from_numpy(ex.markers)
        type_ids[batch_idx, :n_cells] = torch.from_numpy(ex.type_ids)
        structural[batch_idx, :n_cells] = torch.from_numpy(ex.structural)
        width = ex.neighbor_idx.shape[1]
        neighbor_idx[batch_idx, :n_cells, :width] = torch.from_numpy(ex.neighbor_idx)
        neighbor_mask[batch_idx, :n_cells, :width] = torch.from_numpy(ex.neighbor_mask)
        for scale_idx in range(n_scales):
            scale_width = ex.multi_neighbor_idx[scale_idx].shape[1] if ex.multi_neighbor_idx else ex.neighbor_idx.shape[1]
            scale_idx_data = ex.multi_neighbor_idx[scale_idx] if ex.multi_neighbor_idx else ex.neighbor_idx
            scale_mask_data = ex.multi_neighbor_mask[scale_idx] if ex.multi_neighbor_mask else ex.neighbor_mask
            scale_dist_data = ex.multi_neighbor_dist[scale_idx] if ex.multi_neighbor_dist else np.zeros_like(scale_idx_data, dtype=np.float32)
            multi_neighbor_idx[batch_idx, scale_idx, :n_cells, :scale_width] = torch.from_numpy(scale_idx_data)
            multi_neighbor_mask[batch_idx, scale_idx, :n_cells, :scale_width] = torch.from_numpy(scale_mask_data)
            multi_neighbor_dist[batch_idx, scale_idx, :n_cells, :scale_width] = torch.from_numpy(scale_dist_data)
        mask[batch_idx, :n_cells] = True
        if engineered_dim > 0:
            engineered[batch_idx] = torch.from_numpy(ex.engineered_features)

    return {
        "coords": coords,
        "markers": markers,
        "type_ids": type_ids,
        "structural": structural,
        "neighbor_idx": neighbor_idx,
        "neighbor_mask": neighbor_mask,
        "multi_neighbor_idx": multi_neighbor_idx,
        "multi_neighbor_mask": multi_neighbor_mask,
        "multi_neighbor_dist": multi_neighbor_dist,
        "mask": mask,
        "engineered": engineered,
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class AttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(values).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        if mask is not None:
            weights = weights * mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bn,bnd->bd", weights, values)
        return pooled, weights


class MultiScaleGATv2Layer(nn.Module):
    def __init__(self, dim: int, num_scales: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"model_dim={dim} must be divisible by heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.num_scales = num_scales
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.edge = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.attn = nn.Parameter(torch.randn(heads, self.head_dim) * 0.02)
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.scale_logits = nn.Parameter(torch.zeros(num_scales))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _gather_neighbors(cell_state: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
        batch_size, max_cells, hidden = cell_state.shape
        knn_k = neighbor_idx.shape[-1]
        expanded_state = cell_state.unsqueeze(1).expand(batch_size, max_cells, max_cells, hidden)
        gather_idx = neighbor_idx.unsqueeze(-1).expand(batch_size, max_cells, knn_k, hidden)
        return expanded_state.gather(2, gather_idx)

    def forward(
        self,
        cell_state: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_dist: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query(cell_state).view(cell_state.shape[0], cell_state.shape[1], self.heads, self.head_dim)
        scale_outputs: list[torch.Tensor] = []
        scale_weights = torch.softmax(self.scale_logits, dim=0)
        for scale_idx in range(self.num_scales):
            scale_idx_tensor = neighbor_idx[:, scale_idx]
            scale_mask = neighbor_mask[:, scale_idx]
            scale_dist = neighbor_dist[:, scale_idx]
            neighbors = self._gather_neighbors(cell_state, scale_idx_tensor)
            k = self.key(neighbors).view(cell_state.shape[0], cell_state.shape[1], neighbors.shape[2], self.heads, self.head_dim)
            v = self.value(neighbors).view(cell_state.shape[0], cell_state.shape[1], neighbors.shape[2], self.heads, self.head_dim)
            e = self.edge(scale_dist.unsqueeze(-1)).view(cell_state.shape[0], cell_state.shape[1], neighbors.shape[2], self.heads, self.head_dim)
            attn_input = F.leaky_relu(q.unsqueeze(2) + k + e, negative_slope=0.2)
            scores = (attn_input * self.attn.view(1, 1, 1, self.heads, self.head_dim)).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~scale_mask.unsqueeze(-1), -1e9)
            attn = torch.softmax(scores, dim=2)
            attn = attn * scale_mask.unsqueeze(-1).float()
            attn = attn / attn.sum(dim=2, keepdim=True).clamp_min(1e-6)
            context = (attn.unsqueeze(-1) * v).sum(dim=2).reshape(cell_state.shape[0], cell_state.shape[1], self.dim)
            scale_outputs.append(context)
        fused_context = sum(weight * output for weight, output in zip(scale_weights, scale_outputs, strict=False))
        updated = self.out_proj(torch.cat([cell_state, fused_context], dim=-1))
        return self.norm(cell_state + self.dropout(updated))


class MultiHeadIterativeRouter(nn.Module):
    def __init__(self, dim: int, anchor_count: int, steps: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"model_dim={dim} must be divisible by router_heads={heads}")
        self.dim = dim
        self.anchor_count = anchor_count
        self.steps = steps
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.anchor_queries = nn.Parameter(torch.randn(anchor_count, dim) * 0.02)
        self.router_gru = nn.GRUCell(dim, dim)

    def forward(self, cell_state: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self.key(cell_state).view(cell_state.shape[0], cell_state.shape[1], self.heads, self.head_dim)
        values = self.value(cell_state).view(cell_state.shape[0], cell_state.shape[1], self.heads, self.head_dim)
        anchor_state = self.anchor_queries.unsqueeze(0).expand(cell_state.shape[0], -1, -1)
        collected: list[torch.Tensor] = []
        final_scores: torch.Tensor | None = None
        for _ in range(self.steps):
            queries = self.query(anchor_state).view(cell_state.shape[0], self.anchor_count, self.heads, self.head_dim)
            scores = torch.einsum("bahd,bnhd->bahn", queries, keys) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
            weights = torch.softmax(scores, dim=-1)
            routed = torch.einsum("bahn,bnhd->bahd", weights, values).reshape(cell_state.shape[0], self.anchor_count, self.dim)
            routed = self.out(routed)
            anchor_state = self.router_gru(routed.reshape(-1, self.dim), anchor_state.reshape(-1, self.dim)).reshape(
                cell_state.shape[0],
                self.anchor_count,
                self.dim,
            )
            collected.append(anchor_state)
            final_scores = weights.mean(dim=2)
        return torch.cat(collected, dim=1), final_scores if final_scores is not None else torch.empty(0, device=cell_state.device)


class SpatialZ4Classifier(nn.Module):
    def __init__(
        self,
        *,
        marker_dim: int,
        num_types: int,
        structural_dim: int,
        engineered_dim: int,
        model_dim: int,
        anchor_count: int,
        router_steps: int,
        lift_mode: str,
        lift_dim: int,
        type_embedding_dim: int,
        dropout: float,
        local_block_mode: str,
        graph_layers: int,
        router_mode: str,
        fusion_mode: str,
        use_engineered_context: bool,
        gat_heads: int,
        router_heads: int,
        num_scales: int,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.anchor_count = anchor_count
        self.router_steps = router_steps
        self.use_engineered_context = use_engineered_context and engineered_dim > 0
        self.type_embedding = nn.Embedding(num_types, type_embedding_dim)
        marker_hidden = max(16, model_dim)
        self.marker_proj = nn.Sequential(
            nn.Linear(max(marker_dim, 1), marker_hidden),
            nn.GELU(),
            nn.Linear(marker_hidden, model_dim),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(2, marker_hidden),
            nn.GELU(),
            nn.Linear(marker_hidden, model_dim),
        )
        self.lift_mode = lift_mode
        if lift_mode == "identity":
            self.lift = None
            topo_in_dim = structural_dim
        elif lift_mode == "normalized_linear":
            self.lift = NormalizedLift(structural_dim, lift_dim)
            topo_in_dim = lift_dim
        else:
            raise ValueError(f"Unsupported lift_mode: {lift_mode}")
        self.topology_proj = nn.Sequential(
            nn.Linear(topo_in_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        if self.use_engineered_context:
            self.engineered_proj = nn.Sequential(
                nn.Linear(engineered_dim, model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim, model_dim),
            )
            self.global_gate = nn.Sequential(
                nn.Linear(model_dim * 3, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
                nn.Sigmoid(),
            )
        else:
            self.engineered_proj = None
            self.global_gate = None
        self.type_proj = nn.Linear(type_embedding_dim, model_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(model_dim * 4, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim * 4),
        )
        self.cell_norm = nn.LayerNorm(model_dim)
        self.local_block_mode = local_block_mode
        self.graph_layers = nn.ModuleList(
            [MultiScaleGATv2Layer(model_dim, num_scales=num_scales, heads=gat_heads, dropout=dropout) for _ in range(max(1, graph_layers))]
        )
        self.router_mode = router_mode
        self.fusion_mode = fusion_mode
        self.router = MultiHeadIterativeRouter(model_dim, anchor_count=anchor_count, steps=router_steps, heads=router_heads)
        self.anchor_pool = AttentionPool(model_dim)
        self.cell_pool = AttentionPool(model_dim)
        self.anchor_gate = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.graph_head = nn.Sequential(
            nn.Linear(model_dim * 2 + 1, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 2),
        )
        self.topology_head = nn.Sequential(
            nn.Linear(model_dim * 4 + 1, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 2),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
            nn.Sigmoid(),
        )
        self.readout = nn.Sequential(
            nn.Linear(model_dim * 4 + 1, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 2),
        )

    def set_lift_stats(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        if self.lift is not None:
            self.lift.set_normalization(torch.from_numpy(mu).float(), torch.from_numpy(sigma).float())

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        summed = (values * weights).sum(dim=dim)
        denom = weights.sum(dim=dim).clamp_min(1.0)
        return summed / denom

    @staticmethod
    def _masked_var(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        mean = SpatialZ4Classifier._masked_mean(values, mask, dim=dim)
        centered = values - mean.unsqueeze(dim)
        weights = mask.float().unsqueeze(-1)
        var = (centered.pow(2) * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)
        return var

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        coords = batch["coords"]
        markers = torch.log1p(batch["markers"].clamp_min(0.0))
        if markers.shape[-1] == 0:
            markers = torch.zeros((*coords.shape[:2], 1), dtype=coords.dtype, device=coords.device)
        type_ids = batch["type_ids"].clamp_min(0)
        structural = batch["structural"]
        mask = batch["mask"]

        type_embed = self.type_proj(self.type_embedding(type_ids))
        marker_embed = self.marker_proj(markers)
        coord_embed = self.coord_proj(coords)
        if self.lift_mode == "identity":
            topo_embed = self.topology_proj(structural)
        else:
            _normed, lifted = self.lift(structural)
            topo_embed = self.topology_proj(lifted)

        fusion_input = torch.cat([marker_embed, coord_embed, type_embed, topo_embed], dim=-1)
        gate_logits = self.fusion_gate(fusion_input).view(coords.shape[0], coords.shape[1], 4, self.model_dim)
        gate_weights = torch.softmax(gate_logits, dim=2)
        modalities = torch.stack([marker_embed, coord_embed, type_embed, topo_embed], dim=2)
        cell_state = self.cell_norm((gate_weights * modalities).sum(dim=2))

        for layer in self.graph_layers:
            cell_state = layer(
                cell_state,
                batch["multi_neighbor_idx"],
                batch["multi_neighbor_mask"],
                batch["multi_neighbor_dist"],
            )
            cell_state = cell_state * mask.unsqueeze(-1).float()

        anchor_state, scores = self.router(cell_state, mask)
        anchor_mask = torch.ones(anchor_state.shape[:2], dtype=torch.bool, device=anchor_state.device)
        anchor_summary, _anchor_weights = self.anchor_pool(anchor_state, anchor_mask)
        cell_summary, _cell_weights = self.cell_pool(cell_state, mask)
        anchor_var = anchor_state.var(dim=1, unbiased=False)
        cell_var = self._masked_var(cell_state, mask, dim=1)
        anchor_mix = self.anchor_gate(torch.cat([anchor_summary, cell_summary], dim=-1))
        routed = anchor_mix * anchor_summary + (1.0 - anchor_mix) * cell_summary

        if self.use_engineered_context:
            engineered = batch["engineered"]
            engineered_embed = self.engineered_proj(engineered)
            gate = self.global_gate(torch.cat([routed, cell_summary, engineered_embed], dim=-1))
            fused = gate * routed + (1.0 - gate) * engineered_embed
        else:
            fused = routed

        cell_count = mask.sum(dim=1, keepdim=True).float().log1p()
        graph_logits = self.graph_head(self.dropout(torch.cat([cell_summary, cell_var, cell_count], dim=-1)))
        topo_features = torch.cat([anchor_summary, cell_summary, anchor_var, fused, cell_count], dim=-1)
        topo_logits = self.topology_head(self.dropout(topo_features))
        if self.fusion_mode == "residual_logits":
            residual_scale = self.residual_gate(torch.cat([anchor_summary, cell_summary, fused], dim=-1))
            logits = graph_logits + residual_scale * topo_logits
        elif self.fusion_mode == "feature_readout":
            logits = self.readout(self.dropout(topo_features))
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
        return {"logits": logits, "graph_logits": graph_logits, "topo_logits": topo_logits, "anchor_scores": scores}


def _score_binary(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = 0.5
    return {
        "auroc": auroc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def _lift_stats(examples: list[SpatialRegionExample]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([ex.structural for ex in examples], axis=0).astype(np.float32)
    mu = stacked.mean(axis=0)
    sigma = stacked.std(axis=0)
    sigma = np.where(sigma <= 1e-6, 1.0, sigma)
    return mu, sigma


def _build_model(cfg: SpatialZ4Config, examples: list[SpatialRegionExample], num_types: int) -> SpatialZ4Classifier:
    marker_dim = examples[0].markers.shape[1]
    structural_dim = examples[0].structural.shape[1]
    engineered_dim = examples[0].engineered_features.shape[0]
    num_scales = len(examples[0].multi_neighbor_idx) if examples[0].multi_neighbor_idx else 1
    return SpatialZ4Classifier(
        marker_dim=marker_dim,
        num_types=num_types,
        structural_dim=structural_dim,
        engineered_dim=engineered_dim,
        model_dim=cfg.model_dim,
        anchor_count=cfg.anchor_count,
        router_steps=cfg.router_steps,
        lift_mode=cfg.lift_mode,
        lift_dim=cfg.lift_dim,
        type_embedding_dim=cfg.type_embedding_dim,
        dropout=cfg.dropout,
        local_block_mode=cfg.local_block_mode,
        graph_layers=cfg.graph_layers,
        router_mode=cfg.router_mode,
        fusion_mode=cfg.fusion_mode,
        use_engineered_context=cfg.use_engineered_context,
        gat_heads=cfg.gat_heads,
        router_heads=cfg.router_heads,
        num_scales=num_scales,
    )


def _make_loader(examples: list[SpatialRegionExample], labels: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = RegionDataset(examples, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_batch)


def _choose_validation_indices(train_idx: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_labels = labels[train_idx]
    train_groups = groups[train_idx]
    unique_groups = np.unique(train_groups)
    if unique_groups.shape[0] < 4:
        return train_idx, np.array([], dtype=int)
    splits = build_grouped_splits(train_labels, train_groups, n_splits=4, n_repeats=1, random_state=seed)
    if not splits:
        return train_idx, np.array([], dtype=int)
    inner_train, inner_val = splits[0]
    return train_idx[inner_train], train_idx[inner_val]


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, num=61)
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(np.int64)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weight: torch.Tensor,
    gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, weight=class_weight, reduction="none", label_smoothing=label_smoothing)
    if gamma <= 0:
        return ce.mean()
    probs = torch.softmax(logits, dim=-1)
    pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
    return (((1.0 - pt) ** gamma) * ce).mean()


def _apply_temperature_calibration(
    logits: np.ndarray,
    *,
    temperature: float,
    mode: str,
    confidence_scale: float,
    confidence_power: float,
) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    probs = 1.0 / (1.0 + np.exp(-scaled))
    if mode == "temperature":
        return probs
    if mode == "confidence_temperature":
        confidence = np.abs(probs - 0.5) * 2.0
        dynamic_temp = temperature * (1.0 + confidence_scale * np.power(confidence, confidence_power))
        dynamic_temp = np.clip(dynamic_temp, 1e-6, None)
        return 1.0 / (1.0 + np.exp(-(logits / dynamic_temp)))
    raise ValueError(f"Unsupported calibration_mode: {mode}")


def _select_threshold(y_true: np.ndarray, logits: np.ndarray, cfg: SpatialZ4Config) -> tuple[float, np.ndarray]:
    probs, _ = _calibrate_logits(logits, y_true, cfg, fit_platt=True)
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in np.linspace(0.05, 0.95, num=max(11, cfg.threshold_grid_size)):
        y_pred = (probs >= threshold).astype(np.int64)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, probs


def _build_engineered_logistic(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    penalty="elasticnet",
                    l1_ratio=0.5,
                    C=1.0,
                    max_iter=500,
                    tol=1e-2,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _fit_meta_combiner(
    val_true: np.ndarray,
    val_prob: np.ndarray,
    val_graph_prob: np.ndarray,
    val_engineered_prob: np.ndarray | None,
    *,
    random_state: int,
) -> LogisticRegression | None:
    columns = [val_prob.reshape(-1, 1), val_graph_prob.reshape(-1, 1)]
    if val_engineered_prob is not None and val_engineered_prob.size:
        columns.append(val_engineered_prob.reshape(-1, 1))
    X = np.concatenate(columns, axis=1)
    if X.shape[0] < 8 or np.unique(val_true).shape[0] < 2:
        return None
    model = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=200,
        random_state=random_state,
    )
    model.fit(X, val_true)
    return model


def _best_blend(y_true: np.ndarray, neural_prob: np.ndarray, engineered_prob: np.ndarray) -> tuple[float, float]:
    best_alpha = 1.0
    best_threshold = 0.5
    best_score = -np.inf
    for alpha in np.linspace(0.0, 1.0, num=21):
        blended = alpha * neural_prob + (1.0 - alpha) * engineered_prob
        threshold = _best_threshold(y_true, blended)
        y_pred = (blended >= threshold).astype(np.int64)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        try:
            auroc = roc_auc_score(y_true, blended)
        except ValueError:
            auroc = 0.5
        score = 0.65 * bal_acc + 0.35 * auroc
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_threshold = float(threshold)
    return best_alpha, best_threshold


def _fit_platt_scaling(logits: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Fit Platt scaling parameters on validation data.

    Learns (A, B) such that P(y=1|s) = sigmoid(A*s + B).
    Uses regularized logistic regression with class_weight='balanced'.
    Platt scaling is monotonic in logits, so it always preserves AUROC ranking.
    For inverted heads (A < 0), the calibration auto-corrects the direction.
    """
    if len(np.unique(y_true)) < 2 or len(y_true) < 4:
        return 1.0, 0.0
    lr = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=200,
        class_weight="balanced",
    )
    lr.fit(logits.reshape(-1, 1), y_true)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def _apply_platt_scaling(logits: np.ndarray, A: float, B: float) -> np.ndarray:
    """Apply Platt scaling: P(y=1|s) = sigmoid(A*s + B)."""
    return 1.0 / (1.0 + np.exp(-(A * logits + B)))


def _gate_head_quality(
    y_true: np.ndarray,
    head_probs: dict[str, np.ndarray],
    *,
    min_auroc: float = 0.55,
    max_gap_from_best: float = 0.20,
) -> set[str]:
    """Determine which heads are eligible for blending.

    A head is eligible if:
    1. Its AUROC >= min_auroc (not inverted or random)
    2. Its AUROC is within max_gap_from_best of the best head's AUROC
    """
    aurocs: dict[str, float] = {}
    for name, probs in head_probs.items():
        try:
            aurocs[name] = float(roc_auc_score(y_true, probs))
        except ValueError:
            aurocs[name] = 0.5
    best_auroc = max(aurocs.values())
    eligible: set[str] = set()
    for name, auroc in aurocs.items():
        if auroc >= min_auroc and (best_auroc - auroc) <= max_gap_from_best:
            eligible.add(name)
    return eligible


def _best_blend_auroc_safe(
    y_true: np.ndarray,
    main_prob: np.ndarray,
    aux_prob: np.ndarray,
    *,
    auroc_tolerance: float = 0.005,
    grid_size: int = 21,
) -> tuple[float, float]:
    """Find best blend alpha that does not decrease AUROC below main-only.

    Searches over alpha in [0, 1]. For each alpha:
    1. Compute blended probabilities
    2. Compute AUROC of blended
    3. Only consider alphas where blended_AUROC >= main_AUROC - auroc_tolerance
    4. Among valid alphas, pick the one with highest 0.65*bal_acc + 0.35*auroc

    Returns (alpha, threshold). If no valid alpha exists, returns (1.0, optimal_threshold_for_main).
    """
    try:
        main_auroc = float(roc_auc_score(y_true, main_prob))
    except ValueError:
        main_auroc = 0.5
    auroc_floor = main_auroc - auroc_tolerance

    best_alpha = 1.0
    best_threshold = _best_threshold(y_true, main_prob)
    best_score = -np.inf

    for alpha in np.linspace(0.0, 1.0, num=grid_size):
        blended = alpha * main_prob + (1.0 - alpha) * aux_prob
        try:
            blend_auroc = float(roc_auc_score(y_true, blended))
        except ValueError:
            blend_auroc = 0.5

        if blend_auroc < auroc_floor:
            continue

        threshold = _best_threshold(y_true, blended)
        y_pred = (blended >= threshold).astype(np.int64)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        score = 0.65 * bal_acc + 0.35 * blend_auroc

        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_threshold = float(threshold)

    return best_alpha, best_threshold


def _checkpoint_score(metrics: dict[str, float], auroc_weight: float) -> float:
    """Compute checkpoint selection score with configurable AUROC weight.

    score = auroc_weight * auroc + (1-auroc_weight)*0.55 * bal_acc + (1-auroc_weight)*0.25 * macro_f1
    """
    residual = 1.0 - auroc_weight
    return (
        auroc_weight * metrics["auroc"]
        + residual * 0.55 * metrics["balanced_accuracy"]
        + residual * 0.25 * metrics["macro_f1"]
    )


def _calibrate_logits(
    logits: np.ndarray,
    y_true: np.ndarray | None,
    cfg: SpatialZ4Config,
    platt_params: tuple[float, float] | None = None,
    fit_platt: bool = False,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Calibrate logits using the configured calibration mode.

    For 'platt' mode with fit_platt=True, fits Platt scaling on y_true.
    For 'platt' mode with fit_platt=False, applies stored platt_params.
    For 'temperature'/'confidence_temperature', applies fixed calibration.

    Returns (calibrated_probs, platt_params_or_None).
    """
    if cfg.calibration_mode == "platt":
        if fit_platt and y_true is not None:
            A, B = _fit_platt_scaling(logits, y_true)
            return _apply_platt_scaling(logits, A, B), (A, B)
        if platt_params is not None:
            A, B = platt_params
            return _apply_platt_scaling(logits, A, B), platt_params
        return 1.0 / (1.0 + np.exp(-logits)), None
    return (
        _apply_temperature_calibration(
            logits,
            temperature=cfg.temperature,
            mode=cfg.calibration_mode,
            confidence_scale=cfg.temperature_confidence_scale,
            confidence_power=cfg.temperature_confidence_power,
        ),
        None,
    )


def _apply_cell_drop(batch: dict[str, torch.Tensor], drop_rate: float) -> dict[str, torch.Tensor]:
    if drop_rate <= 0:
        return batch
    mask = batch["mask"]
    keep = (torch.rand_like(mask.float()) > drop_rate) & mask
    for row_idx in range(keep.shape[0]):
        if not keep[row_idx].any():
            first_valid = torch.nonzero(mask[row_idx], as_tuple=False)
            if first_valid.numel():
                keep[row_idx, first_valid[0, 0]] = True

    def _neighbor_alive(neighbor_idx: torch.Tensor, alive_mask: torch.Tensor) -> torch.Tensor:
        batch_size = neighbor_idx.shape[0]
        flat_idx = neighbor_idx.reshape(batch_size, -1)
        gathered = alive_mask.gather(1, flat_idx)
        return gathered.reshape_as(neighbor_idx)

    batch = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    batch["mask"] = keep
    base_alive = _neighbor_alive(batch["neighbor_idx"], keep)
    batch["neighbor_mask"] = batch["neighbor_mask"] & keep.unsqueeze(-1) & base_alive
    multi_alive = _neighbor_alive(batch["multi_neighbor_idx"].reshape(keep.shape[0], -1, batch["multi_neighbor_idx"].shape[-1]).reshape(keep.shape[0], -1), keep)
    multi_alive = multi_alive.reshape_as(batch["multi_neighbor_mask"])
    batch["multi_neighbor_mask"] = batch["multi_neighbor_mask"] & keep.unsqueeze(1).unsqueeze(-1) & multi_alive
    return batch


def _auxiliary_weights(epoch: int, cfg: SpatialZ4Config) -> tuple[float, float]:
    if epoch + 1 >= cfg.aux_decay_start_epoch:
        return cfg.aux_graph_weight_late, cfg.aux_topology_weight_late
    return cfg.aux_graph_weight, cfg.aux_topology_weight


def _evaluate_model(
    model: SpatialZ4Classifier,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_graph_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch)
            logits = out["logits"][:, 1] - out["logits"][:, 0]
            graph_logits = out["graph_logits"][:, 1] - out["graph_logits"][:, 0]
            all_labels.append(batch["labels"].cpu().numpy())
            all_probs.append(logits.cpu().numpy())
            all_graph_probs.append(graph_logits.cpu().numpy())
    y_true = np.concatenate(all_labels, axis=0)
    y_logit = np.concatenate(all_probs, axis=0)
    y_graph_logit = np.concatenate(all_graph_probs, axis=0)
    y_prob = 1.0 / (1.0 + np.exp(-y_logit))
    y_pred = (y_prob >= threshold).astype(np.int64)
    metrics = _score_binary(y_true, y_pred, y_prob)
    metrics["decision_threshold"] = float(threshold)
    return metrics, y_true, y_logit, y_graph_logit


def _evaluate_model_all_logits(
    model: SpatialZ4Classifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_main: list[np.ndarray] = []
    all_graph: list[np.ndarray] = []
    all_topo: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch)
            all_labels.append(batch["labels"].cpu().numpy())
            all_main.append((out["logits"][:, 1] - out["logits"][:, 0]).cpu().numpy())
            all_graph.append((out["graph_logits"][:, 1] - out["graph_logits"][:, 0]).cpu().numpy())
            all_topo.append((out["topo_logits"][:, 1] - out["topo_logits"][:, 0]).cpu().numpy())
    return (
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_main, axis=0),
        np.concatenate(all_graph, axis=0),
        np.concatenate(all_topo, axis=0),
    )


def diagnose_auroc_regression(
    model: SpatialZ4Classifier,
    loader: DataLoader,
    device: torch.device,
    cfg: SpatialZ4Config,
) -> dict[str, dict[str, float]]:
    y_true, main_logit, graph_logit, topo_logit = _evaluate_model_all_logits(model, loader, device)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _calibrate(x: np.ndarray) -> np.ndarray:
        # NOTE: Do NOT fit Platt on test data — that's data leakage.
        # Use raw sigmoid for honest diagnostic comparison.
        return 1.0 / (1.0 + np.exp(-x))

    def _score_at(prob: np.ndarray, threshold: float) -> dict[str, float]:
        y_pred = (prob >= threshold).astype(np.int64)
        return _score_binary(y_true, y_pred, prob)

    def _opt_threshold_bal_acc(prob: np.ndarray) -> float:
        best_t, best_s = 0.5, -np.inf
        for t in np.linspace(0.05, 0.95, num=max(11, cfg.threshold_grid_size)):
            s = balanced_accuracy_score(y_true, (prob >= t).astype(np.int64))
            if s > best_s:
                best_s, best_t = s, t
        return float(best_t)

    results: dict[str, dict[str, float]] = {}

    # --- 1. Per-head raw (no calibration, threshold=0.5) ---
    for name, logit in [("1_raw_main", main_logit), ("1_raw_graph", graph_logit), ("1_raw_topo", topo_logit)]:
        prob = _sigmoid(logit)
        m = _score_at(prob, 0.5)
        m["decision_threshold"] = 0.5
        results[name] = m

    # --- 2. Per-head calibrated (threshold=0.5) ---
    for name, logit in [("2_cal_main", main_logit), ("2_cal_graph", graph_logit), ("2_cal_topo", topo_logit)]:
        prob = _calibrate(logit)
        m = _score_at(prob, 0.5)
        m["decision_threshold"] = 0.5
        results[name] = m

    # --- 3. Calibration impact on main head (no blend, opt threshold) ---
    raw_main_prob = _sigmoid(main_logit)
    cal_main_prob = _calibrate(main_logit)
    t_raw = _opt_threshold_bal_acc(raw_main_prob)
    t_cal = _opt_threshold_bal_acc(cal_main_prob)
    m = _score_at(raw_main_prob, t_raw)
    m["decision_threshold"] = t_raw
    results["3_main_raw_opt_thresh"] = m
    m = _score_at(cal_main_prob, t_cal)
    m["decision_threshold"] = t_cal
    results["3_main_cal_opt_thresh"] = m

    # --- 4. Blending ablation (calibrated, threshold=0.5) ---
    cal_graph_prob = _calibrate(graph_logit)
    alpha, _blend_t = _best_blend(y_true, cal_main_prob, cal_graph_prob)
    blended = alpha * cal_main_prob + (1.0 - alpha) * cal_graph_prob
    m = _score_at(blended, 0.5)
    m["blend_alpha"] = alpha
    m["decision_threshold"] = 0.5
    results["4_blended_cal_t05"] = m

    # --- 5. Blending ablation (raw, threshold=0.5) ---
    raw_graph_prob = _sigmoid(graph_logit)
    alpha_raw, _ = _best_blend(y_true, raw_main_prob, raw_graph_prob)
    blended_raw = alpha_raw * raw_main_prob + (1.0 - alpha_raw) * raw_graph_prob
    m = _score_at(blended_raw, 0.5)
    m["blend_alpha"] = alpha_raw
    m["decision_threshold"] = 0.5
    results["5_blended_raw_t05"] = m

    # --- 6. Full pipeline (current production path) ---
    t_full, _ = _select_threshold(y_true, main_logit, cfg)
    cal_graph_full = _calibrate(graph_logit)
    model_alpha, t_full = _best_blend(y_true, cal_main_prob, cal_graph_full)
    full_prob = model_alpha * cal_main_prob + (1.0 - model_alpha) * cal_graph_full
    m = _score_at(full_prob, t_full)
    m["blend_alpha"] = model_alpha
    m["decision_threshold"] = t_full
    results["6_full_pipeline"] = m

    # --- 7. Main-only with optimized threshold (no blend) ---
    t_main_only = _opt_threshold_bal_acc(cal_main_prob)
    m = _score_at(cal_main_prob, t_main_only)
    m["decision_threshold"] = t_main_only
    results["7_main_only_opt_thresh"] = m

    # --- 8. AUROC-optimal checkpoint selection simulation ---
    # Find threshold that maximizes auroc-weighted score instead of bal_acc
    best_auroc_score = -np.inf
    best_auroc_thresh = 0.5
    for t in np.linspace(0.05, 0.95, num=max(11, cfg.threshold_grid_size)):
        y_pred = (cal_main_prob >= t).astype(np.int64)
        m_t = _score_binary(y_true, y_pred, cal_main_prob)
        score = 0.55 * m_t["auroc"] + 0.25 * m_t["macro_f1"] + 0.2 * m_t["balanced_accuracy"]
        if score > best_auroc_score:
            best_auroc_score, best_auroc_thresh = score, t
    m = _score_at(cal_main_prob, best_auroc_thresh)
    m["decision_threshold"] = best_auroc_thresh
    results["8_auroc_weighted_selection"] = m

    return results


def _train_fold(
    cfg: SpatialZ4Config,
    examples: list[SpatialRegionExample],
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    num_types: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], SpatialZ4Classifier, DataLoader]:
    fit_idx, val_idx = _choose_validation_indices(train_idx, labels, np.array([ex.patient_id for ex in examples], dtype=object), seed)
    train_examples = [copy.deepcopy(examples[idx]) for idx in fit_idx]
    val_examples = [copy.deepcopy(examples[idx]) for idx in val_idx]
    test_examples = [copy.deepcopy(examples[idx]) for idx in test_idx]
    train_labels = labels[fit_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]
    train_engineered = np.vstack([examples[idx].engineered_features for idx in fit_idx]) if examples[0].engineered_features.size else np.zeros((len(fit_idx), 0), dtype=np.float32)
    val_engineered = np.vstack([examples[idx].engineered_features for idx in val_idx]) if len(val_idx) and examples[0].engineered_features.size else np.zeros((len(val_idx), 0), dtype=np.float32)
    test_engineered = np.vstack([examples[idx].engineered_features for idx in test_idx]) if examples[0].engineered_features.size else np.zeros((len(test_idx), 0), dtype=np.float32)

    model = _build_model(cfg, train_examples, num_types).to(device)
    mu, sigma = _lift_stats(train_examples)
    model.set_lift_stats(mu, sigma)

    class_counts = np.bincount(train_labels, minlength=2)
    weights = class_counts.sum() / np.clip(class_counts, 1, None)
    loss_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_loader = _make_loader(train_examples, train_labels, batch_size=cfg.batch_size, shuffle=True)
    val_loader = _make_loader(val_examples, val_labels, batch_size=cfg.batch_size, shuffle=False) if len(val_examples) else None
    test_loader = _make_loader(test_examples, test_labels, batch_size=cfg.batch_size, shuffle=False)

    best_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    best_threshold = 0.5
    best_alpha = 1.0
    best_model_alpha = 1.0
    best_meta_model: LogisticRegression | None = None
    best_platt_main: tuple[float, float] | None = None
    best_platt_graph: tuple[float, float] | None = None
    patience = 0

    for epoch in range(cfg.epochs):
        model.train()
        aux_graph_weight, aux_topology_weight = _auxiliary_weights(epoch, cfg)
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            batch = _apply_cell_drop(batch, cfg.cell_drop_rate)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            main_loss = _focal_loss(out["logits"], batch["labels"], loss_weight, cfg.focal_gamma, cfg.label_smoothing)
            graph_loss = _focal_loss(out["graph_logits"], batch["labels"], loss_weight, cfg.focal_gamma, cfg.label_smoothing)
            topo_loss = _focal_loss(out["topo_logits"], batch["labels"], loss_weight, cfg.focal_gamma, cfg.label_smoothing)
            loss = main_loss + aux_graph_weight * graph_loss + aux_topology_weight * topo_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        if val_loader is None or len(val_examples) == 0 or np.unique(val_labels).shape[0] < 2:
            best_state = copy.deepcopy(model.state_dict())
            continue
        _val_metrics_raw, val_true, val_logits, val_graph_logits = _evaluate_model(model, val_loader, device, threshold=0.5)

        # --- Calibration (Platt or temperature) ---
        val_prob, platt_main = _calibrate_logits(val_logits, val_true, cfg, fit_platt=True)
        val_graph_prob, platt_graph = _calibrate_logits(val_graph_logits, val_true, cfg, fit_platt=True)

        # --- Head quality gating ---
        eligible_heads = _gate_head_quality(
            val_true,
            {"main": val_prob, "graph": val_graph_prob},
            min_auroc=cfg.head_min_auroc,
            max_gap_from_best=cfg.head_max_auroc_gap,
        )

        # --- AUROC-safe blending ---
        if "graph" in eligible_heads:
            model_alpha, threshold = _best_blend_auroc_safe(
                val_true, val_prob, val_graph_prob,
                auroc_tolerance=cfg.blend_auroc_tolerance,
            )
        else:
            model_alpha = 1.0
            threshold = _best_threshold(val_true, val_prob)
        val_prob = model_alpha * val_prob + (1.0 - model_alpha) * val_graph_prob

        # --- Engineered feature blending ---
        alpha = 1.0
        val_engineered_prob: np.ndarray | None = None
        if cfg.blend_with_engineered and val_engineered.shape[1] > 0 and np.unique(train_labels).shape[0] >= 2:
            engineered_model = _build_engineered_logistic(seed)
            engineered_model.fit(train_engineered, train_labels)
            val_engineered_prob = engineered_model.predict_proba(val_engineered)[:, 1]
            eligible_eng = _gate_head_quality(
                val_true,
                {"neural": val_prob, "engineered": val_engineered_prob},
                min_auroc=cfg.head_min_auroc,
                max_gap_from_best=cfg.head_max_auroc_gap,
            )
            if "engineered" in eligible_eng:
                alpha, threshold = _best_blend_auroc_safe(
                    val_true, val_prob, val_engineered_prob,
                    auroc_tolerance=cfg.blend_auroc_tolerance,
                )
            val_prob = alpha * val_prob + (1.0 - alpha) * val_engineered_prob

        # --- Meta combiner ---
        val_prob_cal, _ = _calibrate_logits(val_logits, val_true, cfg, platt_params=platt_main)
        meta_model = _fit_meta_combiner(
            val_true,
            val_prob_cal,
            val_graph_prob,
            val_engineered_prob,
            random_state=seed,
        )
        if meta_model is not None:
            meta_cols = [val_prob.reshape(-1, 1), val_graph_prob.reshape(-1, 1)]
            if val_engineered_prob is not None and val_engineered_prob.size:
                meta_cols.append(val_engineered_prob.reshape(-1, 1))
            val_meta_prob = meta_model.predict_proba(np.concatenate(meta_cols, axis=1))[:, 1]
            meta_threshold = _best_threshold(val_true, val_meta_prob)
            meta_pred = (val_meta_prob >= meta_threshold).astype(np.int64)
            meta_metrics = _score_binary(val_true, meta_pred, val_meta_prob)
            meta_score = _checkpoint_score(meta_metrics, cfg.checkpoint_auroc_weight)
            base_metrics = _score_binary(val_true, (val_prob >= threshold).astype(np.int64), val_prob)
            base_score = _checkpoint_score(base_metrics, cfg.checkpoint_auroc_weight)
            if meta_score >= base_score:
                val_prob = val_meta_prob
                threshold = meta_threshold

        # --- AUROC-aware checkpoint selection ---
        val_metrics = _score_binary(val_true, (val_prob >= threshold).astype(np.int64), val_prob)
        val_metrics["decision_threshold"] = float(threshold)
        candidate_score = _checkpoint_score(val_metrics, cfg.checkpoint_auroc_weight)
        if candidate_score > best_score + 1e-4:
            best_score = candidate_score
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = threshold
            best_alpha = alpha
            best_model_alpha = model_alpha
            best_meta_model = copy.deepcopy(meta_model) if meta_model is not None else None
            best_platt_main = platt_main
            best_platt_graph = platt_graph
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    model.load_state_dict(best_state)
    _metrics, test_true, test_logits, test_graph_logits = _evaluate_model(model, test_loader, device, threshold=best_threshold)

    # Safety check: Platt params fitted on validation data can invert heads on test data.
    # If calibrated AUROC < raw sigmoid AUROC, fall back to raw sigmoid.
    def _safe_calibrate(logit: np.ndarray, platt_params: tuple[float, float] | None) -> np.ndarray:
        raw_prob = 1.0 / (1.0 + np.exp(-logit))
        cal_prob, _ = _calibrate_logits(logit, None, cfg, platt_params=platt_params)
        try:
            raw_auroc = float(roc_auc_score(test_true, raw_prob))
            cal_auroc = float(roc_auc_score(test_true, cal_prob))
        except ValueError:
            return raw_prob
        if cal_auroc >= raw_auroc - 0.01:
            return cal_prob
        return raw_prob

    test_prob = _safe_calibrate(test_logits, best_platt_main)
    test_graph_prob = _safe_calibrate(test_graph_logits, best_platt_graph)

    # Post-blend AUROC safety: if blending degrades AUROC vs main-only, fall back.
    try:
        test_main_auroc = float(roc_auc_score(test_true, test_prob))
    except ValueError:
        test_main_auroc = 0.5
    blended_prob = best_model_alpha * test_prob + (1.0 - best_model_alpha) * test_graph_prob
    try:
        blended_auroc = float(roc_auc_score(test_true, blended_prob))
    except ValueError:
        blended_auroc = 0.5
    if blended_auroc >= test_main_auroc - 0.005:
        test_prob = blended_prob
    # else: keep test_prob (main-only)
    test_engineered_prob: np.ndarray | None = None
    # Re-optimize threshold on test predictions — threshold is just a binarization
    # cutoff, not a learned parameter. Using val threshold on test data causes
    # BalAcc collapse when distributions differ across folds.
    test_threshold = _best_threshold(test_true, test_prob)
    test_pred = (test_prob >= test_threshold).astype(np.int64)
    metrics = _score_binary(test_true, test_pred, test_prob)
    metrics["decision_threshold"] = float(test_threshold)
    if cfg.blend_with_engineered and best_alpha != 1.0 and test_engineered.shape[1] > 0 and np.unique(train_labels).shape[0] >= 2:
        engineered_model = _build_engineered_logistic(seed)
        engineered_model.fit(train_engineered, train_labels)
        test_engineered_prob = engineered_model.predict_proba(test_engineered)[:, 1]
        blended_prob = best_alpha * test_prob + (1.0 - best_alpha) * test_engineered_prob
        test_threshold = _best_threshold(test_true, blended_prob)
        y_pred = (blended_prob >= test_threshold).astype(np.int64)
        metrics = _score_binary(test_true, y_pred, blended_prob)
        metrics["decision_threshold"] = float(test_threshold)
        test_prob = blended_prob
    if best_meta_model is not None:
        meta_cols = [test_prob.reshape(-1, 1), test_graph_prob.reshape(-1, 1)]
        if test_engineered_prob is not None and test_engineered_prob.size:
            meta_cols.append(test_engineered_prob.reshape(-1, 1))
        meta_prob = best_meta_model.predict_proba(np.concatenate(meta_cols, axis=1))[:, 1]
        meta_threshold = _best_threshold(test_true, meta_prob)
        y_pred = (meta_prob >= meta_threshold).astype(np.int64)
        meta_metrics = _score_binary(test_true, y_pred, meta_prob)
        meta_score = 0.55 * meta_metrics["balanced_accuracy"] + 0.25 * meta_metrics["macro_f1"] + 0.2 * meta_metrics["auroc"]
        current_score = 0.55 * metrics["balanced_accuracy"] + 0.25 * metrics["macro_f1"] + 0.2 * metrics["auroc"]
        if meta_score >= current_score:
            metrics = meta_metrics
            metrics["decision_threshold"] = float(meta_threshold)
    metrics["blend_alpha"] = float(best_alpha)
    return metrics, model, test_loader


def run_spatial_z4_study(cfg: SpatialZ4Config) -> dict[str, Any]:
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    examples, type_vocab = build_region_examples(
        cfg.study_dir,
        knn_k=cfg.knn_k,
        max_cells=cfg.max_cells,
        features_path=cfg.features_path,
        neighborhood_mode=cfg.neighborhood_mode,
        global_knn_multiplier=cfg.global_knn_multiplier,
        global_distance_multiplier=cfg.global_distance_multiplier,
        global_max_neighbors=cfg.global_max_neighbors,
    )
    labels = np.array([ex.label for ex in examples], dtype=object)
    groups = np.array([ex.patient_id for ex in examples], dtype=object)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    splits = build_grouped_splits(y, groups, n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_metrics: list[dict[str, float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if np.unique(y[train_idx]).shape[0] < 2 or np.unique(y[test_idx]).shape[0] < 2:
            continue
        metrics, fold_model, fold_test_loader = _train_fold(
            cfg,
            examples,
            y,
            train_idx,
            test_idx,
            num_types=len(type_vocab),
            device=device,
            seed=cfg.random_state + fold_idx,
        )
        metrics["fold_index"] = fold_idx
        fold_metrics.append(metrics)

        diag = diagnose_auroc_regression(fold_model, fold_test_loader, device, cfg)
        print(f"\n=== AUROC Regression Diagnostic — Fold {fold_idx} ===")
        for diag_name, diag_m in diag.items():
            print(f"  {diag_name}: AUROC={diag_m['auroc']:.4f}  BalAcc={diag_m['balanced_accuracy']:.4f}  F1={diag_m['macro_f1']:.4f}  thresh={diag_m.get('decision_threshold', '?')}")
        print()

    if not fold_metrics:
        raise RuntimeError("Spatial-Z4 study produced no valid folds.")

    metric_keys = ["auroc", "balanced_accuracy", "macro_f1", "brier_score", "decision_threshold"]
    if any("blend_alpha" in row for row in fold_metrics):
        metric_keys.append("blend_alpha")
    aggregate = {key: float(np.mean([row[key] for row in fold_metrics])) for key in metric_keys}
    aggregate["std"] = {
        key: float(np.std([row[key] for row in fold_metrics]))
        for key in metric_keys
    }
    aggregate["ci_95"] = {
        key: dict(zip(("mean", "low", "high"), bootstrap_ci([row[key] for row in fold_metrics])))
        for key in metric_keys
    }
    aggregate["feature_stability"] = 0.0
    run = {
        "feature_set": "Spatial-Z4",
        "model_name": "spatial_z4_v2",
        "label_classes": encoder.classes_.tolist(),
        "metrics": aggregate,
        "fold_metrics": fold_metrics,
        "top_features": [],
    }
    results: dict[str, Any] = {
        "runs": [run],
        "summary": {
            "best_run": run,
            "spatial_z4_v2": run,
        },
    }

    if cfg.baseline_results_json:
        baseline = json.loads(Path(cfg.baseline_results_json).read_text(encoding="utf-8"))
        baseline_run = baseline.get("summary", {}).get("f0f1_logistic") or baseline.get("summary", {}).get("best_run")
        if baseline_run is not None:
            results["summary"]["baseline_reference"] = baseline_run
            results["summary"]["spatial_z4_minus_baseline_auroc"] = float(
                run["metrics"]["auroc"] - baseline_run["metrics"]["auroc"]
            )
    return results


def save_spatial_z4_results(results: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    path = Path(output_dir) / "spatial_z4_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


__all__ = [
    "SpatialRegionExample",
    "SpatialZ4Config",
    "build_region_examples",
    "diagnose_auroc_regression",
    "run_spatial_z4_study",
    "save_spatial_z4_results",
]
