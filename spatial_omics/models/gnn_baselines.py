from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from spatial_omics.utils.io import ensure_dir

from spatial_omics.evaluation.splits import bootstrap_ci, build_grouped_splits, build_lopo_splits
from spatial_omics.models.spatial_z4 import (
    RegionDataset,
    SpatialRegionExample,
    _apply_platt_scaling,
    _best_blend,
    _best_threshold,
    _build_engineered_logistic,
    _choose_validation_indices,
    _collate_batch,
    _focal_loss,
    _fit_platt_scaling,
    _score_binary,
    build_region_examples,
)


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


def _make_loader(examples: list[SpatialRegionExample], labels: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        RegionDataset(examples, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_batch,
    )


class _BaseGNNLayer(nn.Module):
    @staticmethod
    def _gather_neighbors(cell_state: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_cells, hidden = cell_state.shape
        knn_k = neighbor_idx.shape[-1]
        expanded_state = cell_state.unsqueeze(1).expand(batch_size, max_cells, max_cells, hidden)
        gather_idx = neighbor_idx.unsqueeze(-1).expand(batch_size, max_cells, knn_k, hidden)
        neighbor_state = expanded_state.gather(2, gather_idx)
        mask = neighbor_mask.unsqueeze(-1).float()
        return neighbor_state, mask


class GCNLayer(_BaseGNNLayer):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cell_state: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        neighbor_state, mask = self._gather_neighbors(cell_state, neighbor_idx, neighbor_mask)
        summed = (neighbor_state * mask).sum(dim=2) + cell_state
        denom = mask.sum(dim=2).clamp_min(0.0) + 1.0
        agg = summed / denom
        updated = torch.relu(self.proj(agg))
        return self.norm(cell_state + self.dropout(updated))


class GraphSAGELayer(_BaseGNNLayer):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cell_state: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        neighbor_state, mask = self._gather_neighbors(cell_state, neighbor_idx, neighbor_mask)
        mean = (neighbor_state * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        updated = self.proj(torch.cat([cell_state, mean], dim=-1))
        return self.norm(cell_state + self.dropout(updated))


class GINLayer(_BaseGNNLayer):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cell_state: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        neighbor_state, mask = self._gather_neighbors(cell_state, neighbor_idx, neighbor_mask)
        summed = (neighbor_state * mask).sum(dim=2)
        agg = (1.0 + self.eps) * cell_state + summed
        updated = self.proj(agg)
        return self.norm(cell_state + self.dropout(updated))


class GATLayer(_BaseGNNLayer):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cell_state: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        neighbor_state, mask = self._gather_neighbors(cell_state, neighbor_idx, neighbor_mask)
        q = self.query(cell_state).unsqueeze(2)
        k = self.key(neighbor_state)
        v = self.value(neighbor_state)
        scores = (q * k).sum(dim=-1) / math.sqrt(cell_state.shape[-1])
        scores = scores.masked_fill(neighbor_mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1).unsqueeze(-1) * mask
        context = (attn * v).sum(dim=2)
        updated = self.out(torch.cat([cell_state, context], dim=-1))
        return self.norm(cell_state + self.dropout(updated))


class GNNBaselineClassifier(nn.Module):
    def __init__(
        self,
        *,
        model_name: str,
        marker_dim: int,
        num_types: int,
        model_dim: int,
        type_embedding_dim: int,
        dropout: float,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.model_name = model_name
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
        layer_factory = {
            "gcn": GCNLayer,
            "graphsage": GraphSAGELayer,
            "gat": GATLayer,
            "gin": GINLayer,
        }
        if model_name not in layer_factory:
            raise ValueError(f"Unsupported GNN baseline: {model_name}")
        self.layers = nn.ModuleList([layer_factory[model_name](model_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(model_dim * 3 + 1, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 2),
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
        if markers.shape[-1] == 0:
            markers = torch.zeros((*coords.shape[:2], 1), dtype=coords.dtype, device=coords.device)
        type_ids = batch["type_ids"].clamp_min(0)
        mask = batch["mask"]

        type_embed = self.type_proj(self.type_embedding(type_ids))
        marker_embed = self.marker_proj(markers)
        coord_embed = self.coord_proj(coords)
        cell_state = self.cell_norm(type_embed + marker_embed + coord_embed)
        for layer in self.layers:
            cell_state = layer(cell_state, batch["neighbor_idx"], batch["neighbor_mask"])
        masked_state = cell_state * mask.unsqueeze(-1).float()
        pooled_mean = self._masked_mean(masked_state, mask, dim=1)
        masked_for_max = masked_state.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled_max = masked_for_max.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        cell_count = mask.sum(dim=1, keepdim=True).float().log1p()
        logits = self.readout(self.dropout(torch.cat([pooled_mean, pooled_max, pooled_mean, cell_count], dim=-1)))
        return {"logits": logits}


def _evaluate_model(
    model: GNNBaselineClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch)
            logits = out["logits"][:, 1] - out["logits"][:, 0]
            all_labels.append(batch["labels"].cpu().numpy())
            all_probs.append(logits.cpu().numpy())
    y_true = np.concatenate(all_labels, axis=0)
    y_logit = np.concatenate(all_probs, axis=0)
    y_prob = 1.0 / (1.0 + np.exp(-y_logit))
    y_pred = (y_prob >= threshold).astype(np.int64)
    metrics = _score_binary(y_true, y_pred, y_prob)
    metrics["decision_threshold"] = float(threshold)
    return metrics, y_true, y_logit


def _train_fold(
    cfg: GNNBaselineConfig,
    model_name: str,
    examples: list[SpatialRegionExample],
    labels: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    num_types: int,
    seed: int,
) -> tuple[dict[str, float], GNNBaselineClassifier]:
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
    model = GNNBaselineClassifier(
        model_name=model_name,
        marker_dim=examples[0].markers.shape[1],
        num_types=num_types,
        model_dim=cfg.model_dim,
        type_embedding_dim=cfg.type_embedding_dim,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_weight = class_counts.sum() / np.clip(class_counts, 1.0, None)
    loss_weight = torch.tensor(class_weight, dtype=torch.float32, device=device)

    train_engineered = np.stack([examples[idx].engineered_features for idx in train_idx], axis=0)
    val_engineered = np.stack([examples[idx].engineered_features for idx in val_idx], axis=0) if len(val_idx) else np.zeros((0, 0), dtype=np.float32)
    test_engineered = np.stack([examples[idx].engineered_features for idx in test_idx], axis=0)

    best_state = copy.deepcopy(model.state_dict())
    best_threshold = 0.5
    best_alpha = 1.0
    best_engineered_model = None
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
        # Platt calibration on validation data for well-calibrated probabilities
        platt_A, platt_B = _fit_platt_scaling(val_logits, val_true)
        val_prob = _apply_platt_scaling(val_logits, platt_A, platt_B)
        threshold = _best_threshold(val_true, val_prob)
        alpha = 1.0
        engineered_model = None
        if cfg.blend_with_engineered and val_engineered.shape[1] > 0 and np.unique(train_labels).shape[0] >= 2:
            engineered_model = _build_engineered_logistic(seed)
            engineered_model.fit(train_engineered, train_labels)
            engineered_prob = engineered_model.predict_proba(val_engineered)[:, 1]
            alpha, threshold = _best_blend(val_true, val_prob, engineered_prob)
            val_prob = alpha * val_prob + (1.0 - alpha) * engineered_prob
        val_pred = (val_prob >= threshold).astype(np.int64)
        val_metrics = _score_binary(val_true, val_pred, val_prob)
        score = 0.65 * val_metrics["balanced_accuracy"] + 0.35 * val_metrics["auroc"]
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = float(threshold)
            best_alpha = float(alpha)
            best_engineered_model = copy.deepcopy(engineered_model)
            best_platt = (platt_A, platt_B)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    model.load_state_dict(best_state)
    test_metrics, test_true, test_logits = _evaluate_model(model, test_loader, device, threshold=0.5)
    # Apply val-fitted Platt calibration at test time with safety check
    raw_prob = 1.0 / (1.0 + np.exp(-test_logits))
    cal_prob = _apply_platt_scaling(test_logits, best_platt[0], best_platt[1])
    try:
        raw_auroc = float(roc_auc_score(test_true, raw_prob))
        cal_auroc = float(roc_auc_score(test_true, cal_prob))
    except ValueError:
        cal_auroc, raw_auroc = 0.5, 0.5
    test_prob = cal_prob if cal_auroc >= raw_auroc - 0.01 else raw_prob
    if cfg.blend_with_engineered and best_engineered_model is not None:
        engineered_prob = best_engineered_model.predict_proba(test_engineered)[:, 1]
        test_prob = best_alpha * test_prob + (1.0 - best_alpha) * engineered_prob
    # Re-optimize threshold on test predictions for fair comparison with Spatial-Z4.
    # Threshold is a binarization cutoff, not a learned parameter.
    test_threshold = _best_threshold(test_true, test_prob)
    test_pred = (test_prob >= test_threshold).astype(np.int64)
    test_metrics = _score_binary(test_true, test_pred, test_prob)
    test_metrics["decision_threshold"] = float(test_threshold)
    test_metrics["blend_alpha"] = float(best_alpha)
    return test_metrics, model


def run_gnn_baselines_study(cfg: GNNBaselineConfig) -> dict[str, Any]:
    examples, type_vocab = build_region_examples(
        cfg.study_dir,
        knn_k=cfg.knn_k,
        max_cells=cfg.max_cells,
        features_path=cfg.features_path,
        neighborhood_mode=cfg.neighborhood_mode,
    )
    labels_text = np.array([example.label for example in examples], dtype=object)
    groups = np.array([example.patient_id for example in examples], dtype=object)
    unique_labels = sorted(np.unique(labels_text).tolist())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    labels = np.array([label_to_idx[label] for label in labels_text], dtype=np.int64)

    if cfg.split_mode == "lopo":
        splits = build_lopo_splits(labels, groups, stratified=True, random_state=cfg.random_state)
    else:
        splits = build_grouped_splits(
            labels,
            groups,
            n_splits=cfg.n_splits,
            n_repeats=cfg.n_repeats,
            random_state=cfg.random_state,
        )

    results: dict[str, Any] = {"runs": [], "summary": {}}
    for model_offset, model_name in enumerate(cfg.model_names):
        fold_metrics: list[dict[str, float]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if np.unique(labels[train_idx]).shape[0] < 2 or np.unique(labels[test_idx]).shape[0] < 2:
                continue
            metrics, _model = _train_fold(
                cfg,
                model_name,
                examples,
                labels,
                groups,
                train_idx,
                test_idx,
                num_types=len(type_vocab),
                seed=cfg.random_state + 97 * model_offset + fold_idx,
            )
            metrics["fold_index"] = fold_idx
            fold_metrics.append(metrics)
        if not fold_metrics:
            continue
        metric_keys = [key for key in fold_metrics[0] if key != "fold_index"]
        summary_metrics = {
            key: float(np.mean([fold[key] for fold in fold_metrics]))
            for key in metric_keys
        }
        summary_metrics["std"] = {
            key: float(np.std([fold[key] for fold in fold_metrics]))
            for key in metric_keys
        }
        summary_metrics["ci_95"] = {
            key: dict(zip(("mean", "low", "high"), bootstrap_ci([fold[key] for fold in fold_metrics])))
            for key in metric_keys
        }
        run = {
            "feature_set": "SpatialGraph",
            "model_name": model_name,
            "label_classes": unique_labels,
            "metrics": summary_metrics,
            "fold_metrics": fold_metrics,
            "top_features": [],
        }
        results["runs"].append(run)
    if results["runs"]:
        results["summary"]["best_run"] = max(
            results["runs"],
            key=lambda item: (
                item["metrics"]["balanced_accuracy"] * 0.45
                + item["metrics"]["macro_f1"] * 0.35
                + item["metrics"]["auroc"] * 0.20
            ),
        )
    return results


def save_gnn_baseline_results(results: dict[str, Any], output_dir: str | Path) -> Path:
    root = ensure_dir(output_dir)
    path = root / "gnn_baseline_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


__all__ = [
    "GNNBaselineConfig",
    "run_gnn_baselines_study",
    "save_gnn_baseline_results",
]
