from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from spartial_omics.utils.legacy_lift import NormalizedLift
from spartial_omics.utils.io import ensure_dir

from spartial_omics.data.io import load_study, marker_matrix_from_cells
from spartial_omics.evaluation.splits import bootstrap_ci, build_grouped_splits, build_lopo_splits


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
    blend_with_engineered: bool = True


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


def _scale_coords(coords: np.ndarray, knn_k: int) -> tuple[np.ndarray, np.ndarray]:
    centered = coords.astype(np.float32, copy=True)
    centered -= np.median(centered, axis=0, keepdims=True)
    if centered.shape[0] <= 1:
        return centered, np.ones(centered.shape[0], dtype=np.float32)
    diff = centered[:, None, :] - centered[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    k = max(1, min(knn_k, centered.shape[0] - 1))
    nearest = np.partition(dist, kth=k - 1, axis=1)[:, :k]
    mean_nn = nearest.mean(axis=1).astype(np.float32)
    scale = float(np.median(mean_nn[mean_nn > 0])) if np.any(mean_nn > 0) else 1.0
    scale = max(scale, 1e-3)
    return centered / scale, mean_nn / scale


def _build_knn_neighbor_graph(coords: np.ndarray, knn_k: int) -> tuple[np.ndarray, np.ndarray]:
    n_cells = coords.shape[0]
    if n_cells <= 1:
        return np.zeros((n_cells, 1), dtype=np.int64), np.zeros((n_cells, 1), dtype=bool)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    k = max(1, min(knn_k, n_cells - 1))
    order = np.argsort(dist, axis=1)[:, :k]
    mask = np.ones_like(order, dtype=bool)
    return order.astype(np.int64), mask


def _build_adaptive_neighbor_graph(coords: np.ndarray, knn_k: int) -> tuple[np.ndarray, np.ndarray]:
    n_cells = coords.shape[0]
    if n_cells <= 1:
        return np.zeros((n_cells, 1), dtype=np.int64), np.zeros((n_cells, 1), dtype=bool)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
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
        size = neighbors.size
        if size == 0:
            continue
        neighbor_idx[idx, :size] = neighbors
        neighbor_mask[idx, :size] = True
    return neighbor_idx, neighbor_mask


def _build_neighbor_graph(coords: np.ndarray, knn_k: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "adaptive_knn":
        return _build_adaptive_neighbor_graph(coords, knn_k)
    if mode == "knn":
        return _build_knn_neighbor_graph(coords, knn_k)
    raise ValueError(f"Unsupported neighborhood_mode: {mode}")


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


def _subsample_cells(example: SpatialRegionExample, max_cells: int) -> SpatialRegionExample:
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
    scaled_coords, mean_nn = _scale_coords(coords, knn_k=min(example.neighbor_idx.shape[1], 6))
    neighborhood_mode = getattr(example, "neighborhood_mode", "adaptive_knn")
    knn_idx, knn_mask = _build_neighbor_graph(scaled_coords, knn_k=min(example.neighbor_idx.shape[1], 6), mode=neighborhood_mode)
    structural = _build_structural_vectors(
        scaled_coords,
        type_ids,
        int(example.structural.shape[1] - 5) // 2,
        knn_idx,
        knn_mask,
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
        neighbor_idx=knn_idx,
        neighbor_mask=knn_mask,
        engineered_features=example.engineered_features,
        neighborhood_mode=neighborhood_mode,
    )


def build_region_examples(
    study_dir: str,
    *,
    knn_k: int,
    max_cells: int,
    features_path: str | None = None,
    neighborhood_mode: str = "adaptive_knn",
) -> tuple[list[SpatialRegionExample], dict[str, int]]:
    study = load_study(study_dir)
    label_map = _region_label_map(study)
    type_vocab = _type_vocabulary(study)
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
        neighbor_idx, neighbor_mask = _build_neighbor_graph(scaled_coords, knn_k, neighborhood_mode)
        structural = _build_structural_vectors(
            scaled_coords,
            type_ids,
            len(type_vocab),
            neighbor_idx,
            neighbor_mask,
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
            neighbor_idx=neighbor_idx,
            neighbor_mask=neighbor_mask,
            engineered_features=engineered_map.get(region_id, np.zeros(engineered_dim, dtype=np.float32)),
            neighborhood_mode=neighborhood_mode,
        )
        examples.append(_subsample_cells(example, max_cells=max_cells))
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

    coords = torch.zeros(batch_size, max_cells, 2, dtype=torch.float32)
    markers = torch.zeros(batch_size, max_cells, marker_dim, dtype=torch.float32)
    type_ids = torch.full((batch_size, max_cells), -1, dtype=torch.long)
    structural = torch.zeros(batch_size, max_cells, structural_dim, dtype=torch.float32)
    neighbor_idx = torch.zeros(batch_size, max_cells, knn_k, dtype=torch.long)
    neighbor_mask = torch.zeros(batch_size, max_cells, knn_k, dtype=torch.bool)
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
        "mask": mask,
        "engineered": engineered,
        "labels": torch.tensor(labels, dtype=torch.long),
    }


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
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.anchor_count = anchor_count
        self.router_steps = router_steps
        self.use_engineered_context = use_engineered_context and engineered_dim > 0
        self.type_embedding = nn.Embedding(num_types, type_embedding_dim)
        marker_hidden = max(16, model_dim // 2)
        self.marker_proj = nn.Sequential(
            nn.Linear(marker_dim, marker_hidden),
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
        self.cell_norm = nn.LayerNorm(model_dim)
        self.local_block_mode = local_block_mode
        self.local_update = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.local_norm = nn.LayerNorm(model_dim)
        self.graph_layers = nn.ModuleList()
        if local_block_mode == "graphsage":
            for _ in range(max(1, graph_layers)):
                self.graph_layers.append(
                    nn.ModuleDict(
                        {
                            "proj": nn.Sequential(
                                nn.Linear(model_dim * 2, model_dim),
                                nn.GELU(),
                                nn.Linear(model_dim, model_dim),
                            ),
                            "norm": nn.LayerNorm(model_dim),
                        }
                    )
                )
        elif local_block_mode != "mean_residual":
            raise ValueError(f"Unsupported local_block_mode: {local_block_mode}")
        self.router_mode = router_mode
        self.fusion_mode = fusion_mode
        self.key_proj = nn.Linear(model_dim, model_dim)
        self.value_proj = nn.Linear(model_dim, model_dim)
        self.anchor_queries = nn.Parameter(torch.randn(anchor_count, model_dim) * 0.02)
        self.router_gru = nn.GRUCell(model_dim, model_dim)
        self.topo_pool = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
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
            nn.Linear(model_dim * 5 + 1, model_dim),
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
            nn.Linear(model_dim * 5 + 1, model_dim),
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

    def _neighbor_mean(self, cell_state: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_cells, hidden = cell_state.shape
        knn_k = neighbor_idx.shape[-1]
        expanded_state = cell_state.unsqueeze(1).expand(batch_size, max_cells, max_cells, hidden)
        gather_idx = neighbor_idx.unsqueeze(-1).expand(batch_size, max_cells, knn_k, hidden)
        neighbor_state = expanded_state.gather(2, gather_idx)
        mask = neighbor_mask.unsqueeze(-1).float()
        summed = (neighbor_state * mask).sum(dim=2)
        denom = mask.sum(dim=2).clamp_min(1.0)
        return summed / denom

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        coords = batch["coords"]
        markers = torch.log1p(batch["markers"].clamp_min(0.0))
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
        cell_state = self.cell_norm(type_embed + marker_embed + coord_embed + topo_embed)
        neighbor_mean = self._neighbor_mean(cell_state, batch["neighbor_idx"], batch["neighbor_mask"])
        if self.local_block_mode == "graphsage":
            for layer in self.graph_layers:
                updated = layer["proj"](torch.cat([cell_state, neighbor_mean], dim=-1))
                cell_state = layer["norm"](cell_state + updated)
                neighbor_mean = self._neighbor_mean(cell_state, batch["neighbor_idx"], batch["neighbor_mask"])
        else:
            updated = self.local_update(torch.cat([cell_state, neighbor_mean, cell_state - neighbor_mean], dim=-1))
            cell_state = self.local_norm(cell_state + updated)

        scores = None
        if self.router_mode == "topo_pool":
            topo_scores = self.topo_pool(torch.cat([cell_state, topo_embed], dim=-1)).squeeze(-1)
            topo_scores = topo_scores.masked_fill(~mask, -1e9)
            topo_weights = torch.softmax(topo_scores, dim=-1).unsqueeze(-1)
            pooled_mean = (topo_weights * cell_state).sum(dim=1)
            anchor_state = cell_state
            pooled_max = cell_state.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
            pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        else:
            keys = self.key_proj(cell_state)
            values = self.value_proj(cell_state)
            anchor_state = self.anchor_queries.unsqueeze(0).expand(cell_state.shape[0], -1, -1)
            for _ in range(self.router_steps):
                scores = torch.einsum("bld,bnd->bln", anchor_state, keys) / math.sqrt(self.model_dim)
                scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
                weights = torch.softmax(scores, dim=-1)
                anchors = torch.einsum("bln,bnd->bld", weights, values)
                anchor_state = self.router_gru(
                    anchors.reshape(-1, self.model_dim),
                    anchor_state.reshape(-1, self.model_dim),
                ).reshape(cell_state.shape[0], self.anchor_count, self.model_dim)
            pooled_mean = anchor_state.mean(dim=1)
            pooled_max = anchor_state.max(dim=1).values
        regional_mean = self._masked_mean(cell_state, mask, dim=1)
        masked_cell_state = cell_state.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        regional_max = masked_cell_state.max(dim=1).values
        regional_max = torch.where(torch.isfinite(regional_max), regional_max, torch.zeros_like(regional_max))
        anchor_mix = self.anchor_gate(torch.cat([pooled_mean, regional_mean], dim=-1))
        routed = anchor_mix * pooled_mean + (1.0 - anchor_mix) * regional_mean
        if self.use_engineered_context:
            engineered = batch["engineered"]
            engineered_embed = self.engineered_proj(engineered)
            gate = self.global_gate(torch.cat([routed, regional_mean, engineered_embed], dim=-1))
            fused = gate * routed + (1.0 - gate) * engineered_embed
        else:
            fused = routed
        cell_count = mask.sum(dim=1, keepdim=True).float().log1p()
        graph_logits = self.graph_head(self.dropout(torch.cat([regional_mean, regional_max, cell_count], dim=-1)))
        topo_features = torch.cat([pooled_mean, pooled_max, regional_mean, regional_max, fused, cell_count], dim=-1)
        topo_logits = self.topology_head(self.dropout(topo_features))
        if self.fusion_mode == "residual_logits":
            residual_scale = self.residual_gate(torch.cat([pooled_mean, regional_mean, fused], dim=-1))
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


def _select_threshold(y_true: np.ndarray, logits: np.ndarray, temperature: float, grid_size: int) -> tuple[float, np.ndarray]:
    scaled = logits / max(temperature, 1e-6)
    probs = 1.0 / (1.0 + np.exp(-scaled))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in np.linspace(0.05, 0.95, num=max(11, grid_size)):
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
) -> tuple[dict[str, float], SpatialZ4Classifier]:
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
    patience = 0

    for _epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            main_loss = _focal_loss(out["logits"], batch["labels"], loss_weight, cfg.focal_gamma, cfg.label_smoothing)
            graph_loss = _focal_loss(out["graph_logits"], batch["labels"], loss_weight, cfg.focal_gamma, cfg.label_smoothing)
            topo_loss = _focal_loss(out["topo_logits"], batch["labels"], loss_weight, cfg.focal_gamma, cfg.label_smoothing)
            loss = main_loss + 0.35 * graph_loss + 0.15 * topo_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        if val_loader is None or len(val_examples) == 0 or np.unique(val_labels).shape[0] < 2:
            best_state = copy.deepcopy(model.state_dict())
            continue
        _val_metrics_raw, val_true, val_logits, val_graph_logits = _evaluate_model(model, val_loader, device, threshold=0.5)
        threshold, val_prob = _select_threshold(val_true, val_logits, cfg.temperature, cfg.threshold_grid_size)
        val_graph_prob = 1.0 / (1.0 + np.exp(-(val_graph_logits / max(cfg.temperature, 1e-6))))
        model_alpha, threshold = _best_blend(val_true, val_prob, val_graph_prob)
        val_prob = model_alpha * val_prob + (1.0 - model_alpha) * val_graph_prob
        alpha = 1.0
        val_engineered_prob: np.ndarray | None = None
        if cfg.blend_with_engineered and val_engineered.shape[1] > 0 and np.unique(train_labels).shape[0] >= 2:
            engineered_model = _build_engineered_logistic(seed)
            engineered_model.fit(train_engineered, train_labels)
            val_engineered_prob = engineered_model.predict_proba(val_engineered)[:, 1]
            alpha, threshold = _best_blend(val_true, val_prob, val_engineered_prob)
            val_prob = alpha * val_prob + (1.0 - alpha) * val_engineered_prob
        meta_model = _fit_meta_combiner(
            val_true,
            1.0 / (1.0 + np.exp(-(val_logits / max(cfg.temperature, 1e-6)))),
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
            meta_score = 0.55 * meta_metrics["balanced_accuracy"] + 0.25 * meta_metrics["macro_f1"] + 0.2 * meta_metrics["auroc"]
            base_metrics = _score_binary(val_true, (val_prob >= threshold).astype(np.int64), val_prob)
            base_score = 0.55 * base_metrics["balanced_accuracy"] + 0.25 * base_metrics["macro_f1"] + 0.2 * base_metrics["auroc"]
            if meta_score >= base_score:
                val_prob = val_meta_prob
                threshold = meta_threshold
        val_metrics = _score_binary(val_true, (val_prob >= threshold).astype(np.int64), val_prob)
        val_metrics["decision_threshold"] = float(threshold)
        if alpha != 1.0:
            y_pred = (val_prob >= threshold).astype(np.int64)
            val_metrics = _score_binary(val_true, y_pred, val_prob)
            val_metrics["decision_threshold"] = float(threshold)
        candidate_score = 0.55 * val_metrics["balanced_accuracy"] + 0.25 * val_metrics["macro_f1"] + 0.2 * val_metrics["auroc"]
        if candidate_score > best_score + 1e-4:
            best_score = candidate_score
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = threshold
            best_alpha = alpha
            best_model_alpha = model_alpha
            best_meta_model = copy.deepcopy(meta_model) if meta_model is not None else None
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    model.load_state_dict(best_state)
    _metrics, test_true, test_logits, test_graph_logits = _evaluate_model(model, test_loader, device, threshold=best_threshold)
    test_prob = 1.0 / (1.0 + np.exp(-(test_logits / max(cfg.temperature, 1e-6))))
    test_graph_prob = 1.0 / (1.0 + np.exp(-(test_graph_logits / max(cfg.temperature, 1e-6))))
    test_prob = best_model_alpha * test_prob + (1.0 - best_model_alpha) * test_graph_prob
    test_engineered_prob: np.ndarray | None = None
    test_pred = (test_prob >= best_threshold).astype(np.int64)
    metrics = _score_binary(test_true, test_pred, test_prob)
    metrics["decision_threshold"] = float(best_threshold)
    if cfg.blend_with_engineered and best_alpha != 1.0 and test_engineered.shape[1] > 0 and np.unique(train_labels).shape[0] >= 2:
        engineered_model = _build_engineered_logistic(seed)
        engineered_model.fit(train_engineered, train_labels)
        test_engineered_prob = engineered_model.predict_proba(test_engineered)[:, 1]
        blended_prob = best_alpha * test_prob + (1.0 - best_alpha) * test_engineered_prob
        y_pred = (blended_prob >= best_threshold).astype(np.int64)
        metrics = _score_binary(test_true, y_pred, blended_prob)
        metrics["decision_threshold"] = float(best_threshold)
        test_prob = blended_prob
    if best_meta_model is not None:
        meta_cols = [test_prob.reshape(-1, 1), test_graph_prob.reshape(-1, 1)]
        if test_engineered_prob is not None and test_engineered_prob.size:
            meta_cols.append(test_engineered_prob.reshape(-1, 1))
        meta_prob = best_meta_model.predict_proba(np.concatenate(meta_cols, axis=1))[:, 1]
        y_pred = (meta_prob >= best_threshold).astype(np.int64)
        meta_metrics = _score_binary(test_true, y_pred, meta_prob)
        meta_score = 0.55 * meta_metrics["balanced_accuracy"] + 0.25 * meta_metrics["macro_f1"] + 0.2 * meta_metrics["auroc"]
        current_score = 0.55 * metrics["balanced_accuracy"] + 0.25 * metrics["macro_f1"] + 0.2 * metrics["auroc"]
        if meta_score >= current_score:
            metrics = meta_metrics
            metrics["decision_threshold"] = float(best_threshold)
    metrics["blend_alpha"] = float(best_alpha)
    return metrics, model


def run_spatial_z4_study(cfg: SpatialZ4Config) -> dict[str, Any]:
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    examples, type_vocab = build_region_examples(
        cfg.study_dir,
        knn_k=cfg.knn_k,
        max_cells=cfg.max_cells,
        features_path=cfg.features_path,
        neighborhood_mode=cfg.neighborhood_mode,
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
        metrics, _model = _train_fold(
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

    if not fold_metrics:
        raise RuntimeError("Spatial-Z4 study produced no valid folds.")

    metric_keys = ["auroc", "balanced_accuracy", "macro_f1", "brier_score", "decision_threshold"]
    if any("blend_alpha" in row for row in fold_metrics):
        metric_keys.append("blend_alpha")
    aggregate = {key: float(np.mean([row[key] for row in fold_metrics])) for key in metric_keys}
    aggregate["feature_stability"] = 0.0
    run = {
        "feature_set": "Spatial-Z4",
        "model_name": "spatial_z4",
        "label_classes": encoder.classes_.tolist(),
        "metrics": aggregate,
        "fold_metrics": fold_metrics,
        "top_features": [],
    }
    results: dict[str, Any] = {
        "runs": [run],
        "summary": {
            "best_run": run,
            "spatial_z4": run,
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
    "run_spatial_z4_study",
    "save_spatial_z4_results",
]
