from __future__ import annotations

import copy
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from ..utils.io import ensure_dir

from ..data.io import load_study, marker_matrix_from_cells
from ..evaluation.splits import bootstrap_ci, build_grouped_splits, build_lopo_splits
from .spatial_z4 import (
    META_COLUMNS,
    _best_blend,
    _best_threshold,
    _build_engineered_logistic,
    _build_neighbor_graph,
    _choose_validation_indices,
    _focal_loss,
    _prepare_engineered_feature_map,
    _region_label_map,
    _scale_coords,
    _score_binary,
    _type_vocabulary,
)

try:  # pragma: no cover - optional runtime dependency already declared in project
    from ripser import ripser
except ImportError:  # pragma: no cover
    ripser = None


TOPOLOGY_GROUPS = (
    ("tumor", ("tumor",)),
    ("cd8_t", ("cd8t",)),
    ("treg", ("treg",)),
    ("macrophage", ("macrophage",)),
    ("endothelial", ("endothelial",)),
    ("stroma", ("stroma",)),
    ("tumor_cd8", ("tumor", "cd8t")),
    ("tumor_treg", ("tumor", "treg")),
    ("tumor_macrophage", ("tumor", "macrophage")),
    ("tumor_endothelial", ("tumor", "endothelial")),
    ("tumor_stroma", ("tumor", "stroma")),
    ("cd8_treg", ("cd8t", "treg")),
)


@dataclass
class SpectraZ5RegionExample:
    region_id: str
    patient_id: str
    label: str
    coords: np.ndarray
    markers: np.ndarray
    type_ids: np.ndarray
    neighbor_idx: np.ndarray
    neighbor_mask: np.ndarray
    engineered_features: np.ndarray
    topo_values: np.ndarray
    topo_dim_ids: np.ndarray
    topo_group_ids: np.ndarray
    topo_summary: np.ndarray


@dataclass
class SpectraZ5Config:
    study_dir: str
    output_dir: str
    baseline_results_json: str | None = None
    features_path: str | None = None
    random_state: int = 42
    n_splits: int = 4
    n_repeats: int = 1
    split_mode: str = "grouped"
    batch_size: int = 12
    epochs: int = 10
    patience: int = 3
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
    graph_layers: int = 2
    use_engineered_context: bool = True
    blend_with_engineered: bool = True
    max_topology_points: int = 24


def _match_aliases(cell_types: np.ndarray, aliases: tuple[str, ...]) -> np.ndarray:
    lowered = np.array([str(value).lower() for value in cell_types], dtype=object)
    mask = np.zeros(lowered.shape[0], dtype=bool)
    for alias in aliases:
        mask |= np.char.find(lowered.astype(str), alias) >= 0
    return mask


def _topology_subsample(coords: np.ndarray, max_points: int) -> np.ndarray:
    if coords.shape[0] <= max_points:
        return coords
    radial = np.linalg.norm(coords, axis=1)
    order = np.argsort(radial)
    keep = order[np.linspace(0, coords.shape[0] - 1, num=max_points, dtype=int)]
    keep = np.sort(np.unique(keep))
    return coords[keep]


def _compute_diagram_points(coords: np.ndarray, group_id: int, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ripser is None or coords.shape[0] < 3:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    coords = _topology_subsample(coords.astype(np.float32, copy=False), max_points=max_points)
    if coords.shape[0] < 3:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    dgms = ripser(coords, maxdim=1)["dgms"]
    value_rows: list[list[float]] = []
    dim_ids: list[int] = []
    group_ids: list[int] = []
    size_signal = math.log1p(coords.shape[0])
    for dim, diagram in enumerate(dgms[:2]):
        for birth, death in diagram:
            if not np.isfinite(death):
                continue
            persistence = float(death - birth)
            if persistence <= 1e-6:
                continue
            value_rows.append([float(birth), float(death), persistence, float(size_signal)])
            dim_ids.append(int(dim))
            group_ids.append(int(group_id))
    if not value_rows:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.asarray(value_rows, dtype=np.float32),
        np.asarray(dim_ids, dtype=np.int64),
        np.asarray(group_ids, dtype=np.int64),
    )


def _compute_region_topology(
    coords: np.ndarray,
    cell_type_names: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_values: list[np.ndarray] = []
    all_dims: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    for group_id, (_name, aliases) in enumerate(TOPOLOGY_GROUPS):
        mask = _match_aliases(cell_type_names, aliases)
        if mask.sum() < 2:
            continue
        values, dims, groups = _compute_diagram_points(coords[mask], group_id, max_points)
        if values.shape[0] == 0:
            continue
        all_values.append(values)
        all_dims.append(dims)
        all_groups.append(groups)
    if not all_values:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.concatenate(all_values, axis=0),
        np.concatenate(all_dims, axis=0),
        np.concatenate(all_groups, axis=0),
    )


def _landscape_summary(
    topo_values: np.ndarray,
    topo_dim_ids: np.ndarray,
    topo_group_ids: np.ndarray,
    *,
    num_groups: int,
    grid_size: int = 8,
) -> np.ndarray:
    feature_dim = num_groups * 2 * grid_size
    if topo_values.shape[0] == 0:
        return np.zeros(feature_dim, dtype=np.float32)
    max_death = float(np.max(topo_values[:, 1]))
    t_grid = np.linspace(0.0, max(max_death, 1e-3), num=grid_size, dtype=np.float32)
    features = np.zeros((num_groups, 2, grid_size), dtype=np.float32)
    births = topo_values[:, 0]
    deaths = topo_values[:, 1]
    for group_id in range(num_groups):
        for dim in range(2):
            mask = (topo_group_ids == group_id) & (topo_dim_ids == dim)
            if not np.any(mask):
                continue
            local_births = births[mask][:, None]
            local_deaths = deaths[mask][:, None]
            tri = np.maximum(0.0, np.minimum(t_grid[None, :] - local_births, local_deaths - t_grid[None, :]))
            features[group_id, dim] = tri.max(axis=0)
    return features.reshape(-1).astype(np.float32)


def build_z5_region_examples(
    study_dir: str,
    *,
    knn_k: int,
    max_cells: int,
    max_topology_points: int,
    features_path: str | None = None,
    neighborhood_mode: str = "adaptive_knn",
    cache_path: str | Path | None = None,
) -> tuple[list[SpectraZ5RegionExample], dict[str, int]]:
    cache_file = Path(cache_path) if cache_path else None
    if cache_file is not None and cache_file.exists():
        with cache_file.open("rb") as fh:
            cached = pickle.load(fh)
        return cached["examples"], cached["type_vocab"]

    study = load_study(study_dir)
    label_map = _region_label_map(study)
    type_vocab = _type_vocabulary(study)
    engineered_map = _prepare_engineered_feature_map(features_path)
    engineered_dim = len(next(iter(engineered_map.values()))) if engineered_map else 0
    examples: list[SpectraZ5RegionExample] = []
    for region_id in sorted(study.samples):
        adata = study.samples[region_id]
        obs = adata.obs.reset_index(drop=True)
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        markers, _marker_cols = marker_matrix_from_cells(obs, META_COLUMNS)
        type_names = obs["cell_type"].astype(str).to_numpy(dtype=object)
        type_ids = obs["cell_type"].astype(str).map(type_vocab).to_numpy(dtype=np.int64)
        scaled_coords, _mean_nn = _scale_coords(coords, knn_k)
        if scaled_coords.shape[0] > max_cells:
            radial = np.linalg.norm(scaled_coords, axis=1)
            order = np.argsort(radial)
            keep = order[np.linspace(0, scaled_coords.shape[0] - 1, num=max_cells, dtype=int)]
            keep = np.sort(np.unique(keep))
            scaled_coords = scaled_coords[keep]
            if markers.shape[1] > 0:
                markers = markers[keep]
            type_ids = type_ids[keep]
            type_names = type_names[keep]
        neighbor_idx, neighbor_mask = _build_neighbor_graph(scaled_coords, knn_k, neighborhood_mode)
        topo_values, topo_dim_ids, topo_group_ids = _compute_region_topology(
            scaled_coords,
            type_names,
            max_points=max_topology_points,
        )
        topo_summary = _landscape_summary(
            topo_values,
            topo_dim_ids,
            topo_group_ids,
            num_groups=len(TOPOLOGY_GROUPS),
        )
        examples.append(
            SpectraZ5RegionExample(
                region_id=region_id,
                patient_id=str(obs["patient_id"].iloc[0]),
                label=str(label_map[region_id]),
                coords=scaled_coords,
                markers=markers,
                type_ids=type_ids,
                neighbor_idx=neighbor_idx,
                neighbor_mask=neighbor_mask,
                engineered_features=engineered_map.get(region_id, np.zeros(engineered_dim, dtype=np.float32)),
                topo_values=topo_values,
                topo_dim_ids=topo_dim_ids,
                topo_group_ids=topo_group_ids,
                topo_summary=topo_summary,
            )
        )

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as fh:
            pickle.dump({"examples": examples, "type_vocab": type_vocab}, fh)
    return examples, type_vocab


class SpectraZ5Dataset(Dataset):
    def __init__(self, examples: list[SpectraZ5RegionExample], labels: np.ndarray):
        self.examples = examples
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[SpectraZ5RegionExample, int]:
        return self.examples[idx], int(self.labels[idx])


def _collate_batch(batch: list[tuple[SpectraZ5RegionExample, int]]) -> dict[str, torch.Tensor]:
    examples, labels = zip(*batch, strict=False)
    batch_size = len(examples)
    max_cells = max(ex.coords.shape[0] for ex in examples)
    marker_dim = examples[0].markers.shape[1]
    knn_k = max(ex.neighbor_idx.shape[1] for ex in examples)
    engineered_dim = examples[0].engineered_features.shape[0]
    max_topo = max(ex.topo_values.shape[0] for ex in examples) if examples else 0

    coords = torch.zeros(batch_size, max_cells, 2, dtype=torch.float32)
    markers = torch.zeros(batch_size, max_cells, marker_dim if marker_dim > 0 else 1, dtype=torch.float32)
    type_ids = torch.full((batch_size, max_cells), -1, dtype=torch.long)
    neighbor_idx = torch.zeros(batch_size, max_cells, knn_k, dtype=torch.long)
    neighbor_mask = torch.zeros(batch_size, max_cells, knn_k, dtype=torch.bool)
    mask = torch.zeros(batch_size, max_cells, dtype=torch.bool)
    engineered = torch.zeros(batch_size, engineered_dim, dtype=torch.float32)
    topo_summary_dim = examples[0].topo_summary.shape[0]
    topo_summary = torch.zeros(batch_size, topo_summary_dim, dtype=torch.float32)
    topo_values = torch.zeros(batch_size, max_topo, 4, dtype=torch.float32)
    topo_dim_ids = torch.zeros(batch_size, max_topo, dtype=torch.long)
    topo_group_ids = torch.zeros(batch_size, max_topo, dtype=torch.long)
    topo_mask = torch.zeros(batch_size, max_topo, dtype=torch.bool)

    for batch_idx, ex in enumerate(examples):
        n_cells = ex.coords.shape[0]
        coords[batch_idx, :n_cells] = torch.from_numpy(ex.coords)
        if marker_dim > 0:
            markers[batch_idx, :n_cells, :marker_dim] = torch.from_numpy(ex.markers)
        type_ids[batch_idx, :n_cells] = torch.from_numpy(ex.type_ids)
        width = ex.neighbor_idx.shape[1]
        neighbor_idx[batch_idx, :n_cells, :width] = torch.from_numpy(ex.neighbor_idx)
        neighbor_mask[batch_idx, :n_cells, :width] = torch.from_numpy(ex.neighbor_mask)
        mask[batch_idx, :n_cells] = True
        if engineered_dim > 0:
            engineered[batch_idx] = torch.from_numpy(ex.engineered_features)
        topo_summary[batch_idx] = torch.from_numpy(ex.topo_summary)
        topo_count = ex.topo_values.shape[0]
        if topo_count > 0:
            topo_values[batch_idx, :topo_count] = torch.from_numpy(ex.topo_values)
            topo_dim_ids[batch_idx, :topo_count] = torch.from_numpy(ex.topo_dim_ids)
            topo_group_ids[batch_idx, :topo_count] = torch.from_numpy(ex.topo_group_ids)
            topo_mask[batch_idx, :topo_count] = True

    return {
        "coords": coords,
        "markers": markers,
        "type_ids": type_ids,
        "neighbor_idx": neighbor_idx,
        "neighbor_mask": neighbor_mask,
        "mask": mask,
        "engineered": engineered,
        "topo_summary": topo_summary,
        "topo_values": topo_values,
        "topo_dim_ids": topo_dim_ids,
        "topo_group_ids": topo_group_ids,
        "topo_mask": topo_mask,
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class GraphSAGELayer(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _gather_neighbors(cell_state: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_cells, hidden = cell_state.shape
        knn_k = neighbor_idx.shape[-1]
        expanded_state = cell_state.unsqueeze(1).expand(batch_size, max_cells, max_cells, hidden)
        gather_idx = neighbor_idx.unsqueeze(-1).expand(batch_size, max_cells, knn_k, hidden)
        neighbor_state = expanded_state.gather(2, gather_idx)
        mask = neighbor_mask.unsqueeze(-1).float()
        return neighbor_state, mask

    def forward(self, cell_state: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        neighbor_state, mask = self._gather_neighbors(cell_state, neighbor_idx, neighbor_mask)
        mean = (neighbor_state * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        updated = self.proj(torch.cat([cell_state, mean], dim=-1))
        return self.norm(cell_state + self.dropout(updated))


class PersLayLike(nn.Module):
    def __init__(self, *, hidden_dim: int, num_groups: int, dropout: float) -> None:
        super().__init__()
        self.value_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dim_embedding = nn.Embedding(2, hidden_dim)
        self.group_embedding = nn.Embedding(num_groups, hidden_dim)
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        topo_values: torch.Tensor,
        topo_dim_ids: torch.Tensor,
        topo_group_ids: torch.Tensor,
        topo_mask: torch.Tensor,
    ) -> torch.Tensor:
        if topo_values.shape[1] == 0:
            return torch.zeros(topo_values.shape[0], self.out_proj[-1].out_features, device=topo_values.device)
        point_state = self.value_proj(topo_values)
        point_state = point_state + self.dim_embedding(topo_dim_ids.clamp_min(0)) + self.group_embedding(topo_group_ids.clamp_min(0))
        weights = torch.softmax(self.weight_net(point_state).squeeze(-1).masked_fill(~topo_mask, -1e9), dim=-1).unsqueeze(-1)
        pooled = (weights * point_state).sum(dim=1)
        masked_max = point_state.masked_fill(~topo_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        masked_max = torch.where(torch.isfinite(masked_max), masked_max, torch.zeros_like(masked_max))
        topo_count = topo_mask.sum(dim=1, keepdim=True).float().log1p()
        return self.out_proj(torch.cat([pooled, masked_max, topo_count], dim=-1))


class SpectraZ5Classifier(nn.Module):
    def __init__(
        self,
        *,
        marker_dim: int,
        num_types: int,
        engineered_dim: int,
        topo_summary_dim: int,
        model_dim: int,
        type_embedding_dim: int,
        graph_layers: int,
        dropout: float,
        use_engineered_context: bool,
    ) -> None:
        super().__init__()
        self.use_engineered_context = use_engineered_context and engineered_dim > 0
        self.type_embedding = nn.Embedding(num_types, type_embedding_dim)
        marker_hidden = max(16, model_dim // 2)
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
        self.type_proj = nn.Linear(type_embedding_dim, model_dim)
        self.cell_norm = nn.LayerNorm(model_dim)
        self.graph_layers = nn.ModuleList([GraphSAGELayer(model_dim, dropout) for _ in range(max(1, graph_layers))])
        self.topology_branch = PersLayLike(hidden_dim=model_dim, num_groups=len(TOPOLOGY_GROUPS), dropout=dropout)
        self.landscape_branch = nn.Sequential(
            nn.Linear(topo_summary_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        if self.use_engineered_context:
            self.engineered_proj = nn.Sequential(
                nn.Linear(engineered_dim, model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim, model_dim),
            )
        else:
            self.engineered_proj = None
        self.graph_head = nn.Sequential(
            nn.Linear(model_dim * 2 + 1, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 2),
        )
        fusion_in = model_dim * (5 if self.use_engineered_context else 4)
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_in, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.Sigmoid(),
        )
        self.topo_head = nn.Sequential(
            nn.Linear(model_dim * 3 + 1, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 2),
        )
        self.residual_scale = nn.Sequential(
            nn.Linear(fusion_in, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        summed = (values * weights).sum(dim=dim)
        denom = weights.sum(dim=dim).clamp_min(1.0)
        return summed / denom

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        coords = batch["coords"]
        markers = torch.log1p(batch["markers"].clamp_min(0.0))
        type_ids = batch["type_ids"].clamp_min(0)
        mask = batch["mask"]

        type_embed = self.type_proj(self.type_embedding(type_ids))
        marker_embed = self.marker_proj(markers)
        coord_embed = self.coord_proj(coords)
        cell_state = self.cell_norm(type_embed + marker_embed + coord_embed)
        for layer in self.graph_layers:
            cell_state = layer(cell_state, batch["neighbor_idx"], batch["neighbor_mask"])

        regional_mean = self._masked_mean(cell_state, mask, dim=1)
        regional_max = cell_state.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        regional_max = torch.where(torch.isfinite(regional_max), regional_max, torch.zeros_like(regional_max))
        cell_count = mask.sum(dim=1, keepdim=True).float().log1p()
        graph_logits = self.graph_head(torch.cat([regional_mean, regional_max, cell_count], dim=-1))

        topo_embed = self.topology_branch(
            batch["topo_values"],
            batch["topo_dim_ids"],
            batch["topo_group_ids"],
            batch["topo_mask"],
        )
        landscape_embed = self.landscape_branch(batch["topo_summary"])

        fusion_parts = [regional_mean, regional_max, topo_embed, landscape_embed]
        if self.use_engineered_context:
            engineered_embed = self.engineered_proj(batch["engineered"])
            fusion_parts.append(engineered_embed)
        fusion_input = torch.cat(fusion_parts, dim=-1)
        gate = self.fusion_gate(fusion_input)
        fused_raw = 0.5 * (topo_embed + landscape_embed)
        fused_topo = gate * fused_raw + (1.0 - gate) * regional_mean
        topo_logits = self.topo_head(torch.cat([regional_mean, fused_topo, regional_max, cell_count], dim=-1))
        residual = self.residual_scale(fusion_input) * topo_logits
        logits = graph_logits + residual
        return {"logits": logits, "graph_logits": graph_logits, "topo_logits": topo_logits}


def _make_loader(examples: list[SpectraZ5RegionExample], labels: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        SpectraZ5Dataset(examples, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_batch,
    )


def _evaluate_model(
    model: SpectraZ5Classifier,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch)
            logits = out["logits"][:, 1] - out["logits"][:, 0]
            all_labels.append(batch["labels"].cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    y_true = np.concatenate(all_labels, axis=0)
    y_logit = np.concatenate(all_logits, axis=0)
    y_prob = 1.0 / (1.0 + np.exp(-y_logit))
    y_pred = (y_prob >= threshold).astype(np.int64)
    metrics = _score_binary(y_true, y_pred, y_prob)
    metrics["decision_threshold"] = float(threshold)
    return metrics, y_true, y_logit


def _train_fold(
    cfg: SpectraZ5Config,
    examples: list[SpectraZ5RegionExample],
    labels: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    num_types: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], SpectraZ5Classifier]:
    fit_idx, val_idx = _choose_validation_indices(train_idx, labels, groups, seed)
    train_examples = [copy.deepcopy(examples[idx]) for idx in fit_idx]
    val_examples = [copy.deepcopy(examples[idx]) for idx in val_idx]
    test_examples = [copy.deepcopy(examples[idx]) for idx in test_idx]
    train_labels = labels[fit_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    train_engineered = (
        np.vstack([examples[idx].engineered_features for idx in fit_idx]) if examples[0].engineered_features.size else np.zeros((len(fit_idx), 0), dtype=np.float32)
    )
    val_engineered = (
        np.vstack([examples[idx].engineered_features for idx in val_idx]) if len(val_idx) and examples[0].engineered_features.size else np.zeros((len(val_idx), 0), dtype=np.float32)
    )
    test_engineered = (
        np.vstack([examples[idx].engineered_features for idx in test_idx]) if examples[0].engineered_features.size else np.zeros((len(test_idx), 0), dtype=np.float32)
    )

    model = SpectraZ5Classifier(
        marker_dim=examples[0].markers.shape[1],
        num_types=num_types,
        engineered_dim=examples[0].engineered_features.shape[0],
        topo_summary_dim=examples[0].topo_summary.shape[0],
        model_dim=cfg.model_dim,
        type_embedding_dim=cfg.type_embedding_dim,
        graph_layers=cfg.graph_layers,
        dropout=cfg.dropout,
        use_engineered_context=cfg.use_engineered_context,
    ).to(device)

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
        _metrics, val_true, val_logits = _evaluate_model(model, val_loader, device, threshold=0.5)
        val_prob = 1.0 / (1.0 + np.exp(-(val_logits / max(cfg.temperature, 1e-6))))
        threshold = _best_threshold(val_true, val_prob)
        alpha = 1.0
        if cfg.blend_with_engineered and val_engineered.shape[1] > 0 and np.unique(train_labels).shape[0] >= 2:
            engineered_model = _build_engineered_logistic(seed)
            engineered_model.fit(train_engineered, train_labels)
            engineered_prob = engineered_model.predict_proba(val_engineered)[:, 1]
            alpha, threshold = _best_blend(val_true, val_prob, engineered_prob)
            val_prob = alpha * val_prob + (1.0 - alpha) * engineered_prob
        val_pred = (val_prob >= threshold).astype(np.int64)
        val_metrics = _score_binary(val_true, val_pred, val_prob)
        candidate_score = 0.55 * val_metrics["balanced_accuracy"] + 0.25 * val_metrics["macro_f1"] + 0.2 * val_metrics["auroc"]
        if candidate_score > best_score + 1e-4:
            best_score = candidate_score
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = float(threshold)
            best_alpha = float(alpha)
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    model.load_state_dict(best_state)
    _metrics, test_true, test_logits = _evaluate_model(model, test_loader, device, threshold=best_threshold)
    test_prob = 1.0 / (1.0 + np.exp(-(test_logits / max(cfg.temperature, 1e-6))))
    if cfg.blend_with_engineered and best_alpha != 1.0 and test_engineered.shape[1] > 0 and np.unique(train_labels).shape[0] >= 2:
        engineered_model = _build_engineered_logistic(seed)
        engineered_model.fit(train_engineered, train_labels)
        engineered_prob = engineered_model.predict_proba(test_engineered)[:, 1]
        test_prob = best_alpha * test_prob + (1.0 - best_alpha) * engineered_prob
    test_pred = (test_prob >= best_threshold).astype(np.int64)
    metrics = _score_binary(test_true, test_pred, test_prob)
    metrics["decision_threshold"] = float(best_threshold)
    metrics["blend_alpha"] = float(best_alpha)
    return metrics, model


def run_spectra_z5_study(cfg: SpectraZ5Config) -> dict[str, Any]:
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    cache_path = Path(cfg.output_dir) / "topology_cache.pkl"
    examples, type_vocab = build_z5_region_examples(
        cfg.study_dir,
        knn_k=cfg.knn_k,
        max_cells=cfg.max_cells,
        max_topology_points=cfg.max_topology_points,
        features_path=cfg.features_path,
        neighborhood_mode=cfg.neighborhood_mode,
        cache_path=cache_path,
    )
    labels = np.array([example.label for example in examples], dtype=object)
    groups = np.array([example.patient_id for example in examples], dtype=object)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    splits = build_grouped_splits(
        y,
        groups,
        n_splits=cfg.n_splits,
        n_repeats=cfg.n_repeats,
        random_state=cfg.random_state,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_metrics: list[dict[str, float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if np.unique(y[train_idx]).shape[0] < 2 or np.unique(y[test_idx]).shape[0] < 2:
            continue
        metrics, _model = _train_fold(
            cfg,
            examples,
            y,
            groups,
            train_idx,
            test_idx,
            num_types=len(type_vocab),
            device=device,
            seed=cfg.random_state + fold_idx,
        )
        metrics["fold_index"] = fold_idx
        fold_metrics.append(metrics)

    if not fold_metrics:
        raise RuntimeError("No valid folds were produced for Spectra-Z5.")

    aggregate = {
        key: float(np.mean([fold[key] for fold in fold_metrics]))
        for key in fold_metrics[0]
        if key != "fold_index"
    }
    run = {
        "feature_set": "Spectra-Z5",
        "model_name": "spectra_z5",
        "label_classes": list(encoder.classes_),
        "metrics": aggregate,
        "fold_metrics": fold_metrics,
        "top_features": [],
    }
    results: dict[str, Any] = {"runs": [run], "summary": {"best_run": run, "spectra_z5": run}}
    if cfg.baseline_results_json:
        baseline_payload = json.loads(Path(cfg.baseline_results_json).read_text(encoding="utf-8"))
        baseline_reference = baseline_payload.get("summary", {}).get("best_run") or baseline_payload.get("runs", [{}])[0]
        results["summary"]["baseline_reference"] = baseline_reference
    return results


def save_spectra_z5_results(results: dict[str, Any], output_dir: str | Path) -> Path:
    root = ensure_dir(output_dir)
    path = root / "spectra_z5_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


__all__ = [
    "SpectraZ5Config",
    "SpectraZ5RegionExample",
    "build_z5_region_examples",
    "run_spectra_z5_study",
    "save_spectra_z5_results",
]
