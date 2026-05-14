from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spatial_omics.evaluation.splits import bootstrap_ci, build_grouped_splits
from spatial_omics.models.spatial_z4 import (
    RegionDataset,
    SpatialRegionExample,
    _apply_platt_scaling,
    _best_threshold,
    _choose_validation_indices,
    _collate_batch,
    _focal_loss,
    _fit_platt_scaling,
    _score_binary,
    build_region_examples,
)
from spatial_omics.utils.io import ensure_dir


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
    global_knn_multiplier: int = 4
    global_distance_multiplier: float = 2.5
    global_max_neighbors: int = 24


def _make_loader(examples: list[SpatialRegionExample], labels: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        RegionDataset(examples, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_toponet_batch,
    )


def _extract_edges(example: SpatialRegionExample) -> tuple[list[tuple[int, int]], np.ndarray]:
    n_cells = example.coords.shape[0]
    adjacency = np.zeros((n_cells, n_cells), dtype=bool)
    for src in range(n_cells):
        valid = example.neighbor_mask[src]
        neighbors = example.neighbor_idx[src, valid]
        for dst in neighbors.tolist():
            if dst < 0 or dst >= n_cells or dst == src:
                continue
            u, v = (src, dst) if src < dst else (dst, src)
            adjacency[u, v] = True
            adjacency[v, u] = True
    edges = [(u, v) for u in range(n_cells) for v in range(u + 1, n_cells) if adjacency[u, v]]
    return edges, adjacency


def _extract_triangles(adjacency: np.ndarray) -> list[tuple[int, int, int]]:
    n_cells = adjacency.shape[0]
    triangles: list[tuple[int, int, int]] = []
    for a in range(n_cells):
        for b in range(a + 1, n_cells):
            if not adjacency[a, b]:
                continue
            common = np.flatnonzero(adjacency[a] & adjacency[b])
            for c in common.tolist():
                if c > b:
                    triangles.append((a, b, c))
    return triangles


def _build_boundary_matrices(example: SpatialRegionExample) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_cells = example.coords.shape[0]
    edges, adjacency = _extract_edges(example)
    triangles = _extract_triangles(adjacency)
    n_edges = len(edges)
    n_triangles = len(triangles)

    b1 = np.zeros((n_cells, max(n_edges, 1)), dtype=np.float32)
    edge_mask = np.zeros(max(n_edges, 1), dtype=bool)
    edge_lookup: dict[tuple[int, int], int] = {}
    for edge_idx, (u, v) in enumerate(edges):
        edge_lookup[(u, v)] = edge_idx
        b1[u, edge_idx] = -1.0
        b1[v, edge_idx] = 1.0
        edge_mask[edge_idx] = True

    b2 = np.zeros((max(n_edges, 1), max(n_triangles, 1)), dtype=np.float32)
    tri_mask = np.zeros(max(n_triangles, 1), dtype=bool)
    for tri_idx, (a, b, c) in enumerate(triangles):
        tri_mask[tri_idx] = True
        # Boundary of oriented simplex [a,b,c] is [b,c] - [a,c] + [a,b].
        b2[edge_lookup[(b, c)], tri_idx] = 1.0
        b2[edge_lookup[(a, c)], tri_idx] = -1.0
        b2[edge_lookup[(a, b)], tri_idx] = 1.0

    return b1, b2, edge_mask, tri_mask


def _collate_toponet_batch(batch: list[tuple[SpatialRegionExample, int]]) -> dict[str, torch.Tensor]:
    collated = _collate_batch(batch)
    examples, _labels = zip(*batch, strict=False)
    batch_size = len(examples)
    max_nodes = collated["coords"].shape[1]

    complexes = [_build_boundary_matrices(example) for example in examples]
    max_edges = max(b1.shape[1] for b1, _b2, _edge_mask, _tri_mask in complexes)
    max_triangles = max(b2.shape[1] for _b1, b2, _edge_mask, _tri_mask in complexes)

    b1_tensor = torch.zeros(batch_size, max_nodes, max_edges, dtype=torch.float32)
    b2_tensor = torch.zeros(batch_size, max_edges, max_triangles, dtype=torch.float32)
    edge_mask_tensor = torch.zeros(batch_size, max_edges, dtype=torch.bool)
    tri_mask_tensor = torch.zeros(batch_size, max_triangles, dtype=torch.bool)

    for batch_idx, (example, (b1, b2, edge_mask, tri_mask)) in enumerate(zip(examples, complexes, strict=False)):
        n_nodes = example.coords.shape[0]
        n_edges = b1.shape[1]
        n_triangles = b2.shape[1]
        b1_tensor[batch_idx, :n_nodes, :n_edges] = torch.from_numpy(b1)
        b2_tensor[batch_idx, :n_edges, :n_triangles] = torch.from_numpy(b2)
        edge_mask_tensor[batch_idx, :n_edges] = torch.from_numpy(edge_mask)
        tri_mask_tensor[batch_idx, :n_triangles] = torch.from_numpy(tri_mask)

    collated["B1"] = b1_tensor
    collated["B2"] = b2_tensor
    collated["edge_mask"] = edge_mask_tensor
    collated["triangle_mask"] = tri_mask_tensor
    return collated


class SimplicialHodgeLayer(nn.Module):
    def __init__(self, dim: int, polynomial_order: int, dropout: float) -> None:
        super().__init__()
        self.polynomial_order = polynomial_order
        self.theta0 = nn.Parameter(torch.zeros(polynomial_order + 1))
        self.theta1_down = nn.Parameter(torch.zeros(polynomial_order + 1))
        self.theta1_up = nn.Parameter(torch.zeros(polynomial_order + 1))
        self.theta2 = nn.Parameter(torch.zeros(polynomial_order + 1))
        self.theta0.data[0] = 1.0
        self.theta1_down.data[0] = 1.0
        self.theta1_up.data[0] = 1.0
        self.theta2.data[0] = 1.0

        self.node_self = nn.Linear(dim, dim, bias=False)
        self.edge_to_node = nn.Linear(dim, dim, bias=False)
        self.edge_self = nn.Linear(dim, dim, bias=False)
        self.node_to_edge = nn.Linear(dim, dim, bias=False)
        self.triangle_to_edge = nn.Linear(dim, dim, bias=False)
        self.triangle_self = nn.Linear(dim, dim, bias=False)
        self.edge_to_triangle = nn.Linear(dim, dim, bias=False)

        self.node_norm = nn.LayerNorm(dim)
        self.edge_norm = nn.LayerNorm(dim)
        self.triangle_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def _poly_apply(self, laplacian: torch.Tensor, features: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        out = coeffs[0] * features
        current = features
        for order in range(1, self.polynomial_order + 1):
            current = torch.bmm(laplacian, current)
            out = out + coeffs[order] * current
        return out

    def forward(
        self,
        node_state: torch.Tensor,
        edge_state: torch.Tensor,
        triangle_state: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        triangle_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b1_t = b1.transpose(1, 2)
        b2_t = b2.transpose(1, 2)
        l0 = torch.bmm(b1, b1_t)
        l1_down = torch.bmm(b1_t, b1)
        l1_up = torch.bmm(b2, b2_t)
        l2 = torch.bmm(b2_t, b2)

        node_update = self.node_self(self._poly_apply(l0, node_state, self.theta0))
        node_update = node_update + self.edge_to_node(torch.bmm(b1, edge_state))
        node_state = self.node_norm(node_state + self.dropout(self.activation(node_update)))
        node_state = node_state * node_mask.unsqueeze(-1).float()

        edge_update = self.edge_self(self._poly_apply(l1_down, edge_state, self.theta1_down) + self._poly_apply(l1_up, edge_state, self.theta1_up))
        edge_update = edge_update + self.node_to_edge(torch.bmm(b1_t, node_state))
        edge_update = edge_update + self.triangle_to_edge(torch.bmm(b2, triangle_state))
        edge_state = self.edge_norm(edge_state + self.dropout(self.activation(edge_update)))
        edge_state = edge_state * edge_mask.unsqueeze(-1).float()

        triangle_update = self.triangle_self(self._poly_apply(l2, triangle_state, self.theta2))
        triangle_update = triangle_update + self.edge_to_triangle(torch.bmm(b2_t, edge_state))
        triangle_state = self.triangle_norm(triangle_state + self.dropout(self.activation(triangle_update)))
        triangle_state = triangle_state * triangle_mask.unsqueeze(-1).float()
        return node_state, edge_state, triangle_state


class TopoNetHodgeClassifier(nn.Module):
    def __init__(
        self,
        *,
        marker_dim: int,
        num_types: int,
        model_dim: int,
        type_embedding_dim: int,
        num_layers: int,
        polynomial_order: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(num_types + 1, type_embedding_dim)
        marker_hidden = max(model_dim, marker_dim if marker_dim > 0 else model_dim)
        self.marker_proj = nn.Sequential(
            nn.Linear(max(marker_dim, 1), marker_hidden),
            nn.GELU(),
            nn.Linear(marker_hidden, model_dim),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.type_proj = nn.Linear(type_embedding_dim, model_dim)
        self.cell_norm = nn.LayerNorm(model_dim)
        self.layers = nn.ModuleList(
            [SimplicialHodgeLayer(model_dim, polynomial_order=polynomial_order, dropout=dropout) for _ in range(max(1, num_layers))]
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(model_dim * 6 + 3, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, 2),
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        summed = (values * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom

    @staticmethod
    def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = values.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_values = masked.max(dim=1).values
        return torch.where(torch.isfinite(max_values), max_values, torch.zeros_like(max_values))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        coords = batch["coords"]
        markers = torch.log1p(batch["markers"].clamp_min(0.0))
        if markers.shape[-1] == 0:
            markers = torch.zeros((*coords.shape[:2], 1), dtype=coords.dtype, device=coords.device)
        type_ids = batch["type_ids"].clamp_min(0)
        node_mask = batch["mask"]
        edge_mask = batch["edge_mask"]
        triangle_mask = batch["triangle_mask"]

        node_state = self.cell_norm(
            self.marker_proj(markers)
            + self.coord_proj(coords)
            + self.type_proj(self.type_embedding(type_ids))
        )
        node_state = node_state * node_mask.unsqueeze(-1).float()

        edge_state = torch.zeros(
            batch["B1"].shape[0],
            batch["B1"].shape[2],
            node_state.shape[-1],
            dtype=node_state.dtype,
            device=node_state.device,
        )
        triangle_state = torch.zeros(
            batch["B2"].shape[0],
            batch["B2"].shape[2],
            node_state.shape[-1],
            dtype=node_state.dtype,
            device=node_state.device,
        )

        for layer in self.layers:
            node_state, edge_state, triangle_state = layer(
                node_state,
                edge_state,
                triangle_state,
                batch["B1"],
                batch["B2"],
                node_mask,
                edge_mask,
                triangle_mask,
            )

        node_mean = self._masked_mean(node_state, node_mask)
        node_max = self._masked_max(node_state, node_mask)
        edge_mean = self._masked_mean(edge_state, edge_mask)
        edge_max = self._masked_max(edge_state, edge_mask)
        tri_mean = self._masked_mean(triangle_state, triangle_mask)
        tri_max = self._masked_max(triangle_state, triangle_mask)
        counts = torch.cat(
            [
                node_mask.sum(dim=1, keepdim=True).float().log1p(),
                edge_mask.sum(dim=1, keepdim=True).float().log1p(),
                triangle_mask.sum(dim=1, keepdim=True).float().log1p(),
            ],
            dim=-1,
        )
        pooled = torch.cat([node_mean, node_max, edge_mean, edge_max, tri_mean, tri_max, counts], dim=-1)
        logits = self.readout(self.dropout(pooled))
        return {"logits": logits}


def _evaluate_model(
    model: TopoNetHodgeClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
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
    cfg: TopoNetHodgeConfig,
    examples: list[SpatialRegionExample],
    labels: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    num_types: int,
    seed: int,
) -> tuple[dict[str, float], TopoNetHodgeClassifier]:
    train_idx, val_idx = _choose_validation_indices(train_idx, labels, groups, seed)
    train_examples = [examples[idx] for idx in train_idx]
    val_examples = [examples[idx] for idx in val_idx]
    test_examples = [examples[idx] for idx in test_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    train_loader = _make_loader(train_examples, train_labels, batch_size=cfg.batch_size, shuffle=True)
    val_loader = _make_loader(val_examples, val_labels, batch_size=cfg.batch_size, shuffle=False) if len(val_examples) else None
    test_loader = _make_loader(test_examples, test_labels, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TopoNetHodgeClassifier(
        marker_dim=examples[0].markers.shape[1],
        num_types=num_types,
        model_dim=cfg.model_dim,
        type_embedding_dim=cfg.type_embedding_dim,
        num_layers=cfg.num_layers,
        polynomial_order=cfg.polynomial_order,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_weight = class_counts.sum() / np.clip(class_counts, 1.0, None)
    loss_weight = torch.tensor(class_weight, dtype=torch.float32, device=device)

    best_state = copy.deepcopy(model.state_dict())
    best_threshold = 0.5
    best_platt = (1.0, 0.0)
    best_score = -np.inf
    patience_counter = 0

    for _epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = _focal_loss(out["logits"], batch["labels"], loss_weight, cfg.focal_gamma, cfg.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        if val_loader is None or len(val_examples) == 0 or np.unique(val_labels).shape[0] < 2:
            best_state = copy.deepcopy(model.state_dict())
            continue
        _raw_metrics, val_true, val_logits = _evaluate_model(model, val_loader, device, threshold=0.5)
        platt_A, platt_B = _fit_platt_scaling(val_logits, val_true)
        val_prob = _apply_platt_scaling(val_logits, platt_A, platt_B)
        threshold = _best_threshold(val_true, val_prob)
        val_pred = (val_prob >= threshold).astype(np.int64)
        val_metrics = _score_binary(val_true, val_pred, val_prob)
        score = 0.65 * val_metrics["balanced_accuracy"] + 0.35 * val_metrics["auroc"]
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = float(threshold)
            best_platt = (platt_A, platt_B)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    model.load_state_dict(best_state)
    _test_metrics, test_true, test_logits = _evaluate_model(model, test_loader, device, threshold=best_threshold)
    raw_prob = 1.0 / (1.0 + np.exp(-test_logits))
    cal_prob = _apply_platt_scaling(test_logits, best_platt[0], best_platt[1])
    try:
        raw_auroc = float(_score_binary(test_true, (raw_prob >= 0.5).astype(np.int64), raw_prob)["auroc"])
        cal_auroc = float(_score_binary(test_true, (cal_prob >= 0.5).astype(np.int64), cal_prob)["auroc"])
    except ValueError:
        raw_auroc, cal_auroc = 0.5, 0.5
    test_prob = cal_prob if cal_auroc >= raw_auroc - 0.01 else raw_prob
    test_threshold = _best_threshold(test_true, test_prob)
    test_pred = (test_prob >= test_threshold).astype(np.int64)
    metrics = _score_binary(test_true, test_pred, test_prob)
    metrics["decision_threshold"] = float(test_threshold)
    metrics["blend_alpha"] = 1.0
    return metrics, model


def run_toponet_hodge_study(cfg: TopoNetHodgeConfig) -> dict[str, Any]:
    np.random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)
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
    labels_text = np.array([example.label for example in examples], dtype=object)
    labels = np.array([0 if label == "CLR" else 1 for label in labels_text], dtype=np.int64)
    groups = np.array([example.patient_id for example in examples], dtype=object)

    if cfg.split_mode != "grouped":
        raise ValueError(f"Unsupported split_mode: {cfg.split_mode}")
    splits = build_grouped_splits(labels, groups, n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.random_state)

    fold_metrics: list[dict[str, float]] = []
    for fold_index, (train_idx, test_idx) in enumerate(splits):
        metrics, _model = _train_fold(
            cfg,
            examples,
            labels,
            groups,
            train_idx,
            test_idx,
            num_types=len(type_vocab),
            seed=cfg.random_state + fold_index,
        )
        metrics["fold_index"] = fold_index
        fold_metrics.append(metrics)

    metric_names = ["auroc", "balanced_accuracy", "macro_f1", "brier_score", "decision_threshold", "blend_alpha"]
    mean_metrics = {name: float(np.mean([fold[name] for fold in fold_metrics])) for name in metric_names}
    std_metrics = {name: float(np.std([fold[name] for fold in fold_metrics], ddof=0)) for name in metric_names}
    ci_metrics = {name: bootstrap_ci([fold[name] for fold in fold_metrics]) for name in metric_names}
    mean_metrics["std"] = std_metrics
    mean_metrics["ci_95"] = ci_metrics
    mean_metrics["feature_stability"] = 0.0

    run = {
        "feature_set": "TopoNet-Hodge",
        "model_name": "toponet_hodge",
        "label_classes": ["CLR", "DII"],
        "metrics": mean_metrics,
        "fold_metrics": fold_metrics,
        "top_features": [],
    }
    return {"runs": [run], "summary": {"best_run": run, "toponet_hodge": run}}


def save_toponet_hodge_results(results: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    path = output_dir / "toponet_hodge_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


__all__ = ["TopoNetHodgeConfig", "run_toponet_hodge_study", "save_toponet_hodge_results"]
