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
from torch.utils.data import DataLoader

from ..legacy_lift import NormalizedLift
from ..utils.io import ensure_dir

from ..evaluation.splits import bootstrap_ci, build_grouped_splits, build_lopo_splits
from .gnn_baselines import GNNBaselineConfig, _make_loader
from .spatial_z4 import (
    SpatialRegionExample,
    _best_blend,
    _build_engineered_logistic,
    _choose_validation_indices,
    _focal_loss,
    _score_binary,
    _select_threshold,
    build_region_examples,
)


@dataclass
class GNNSurgeryConfig(GNNBaselineConfig):
    anchor_count: int = 8
    router_steps: int = 3
    variant_names: tuple[str, ...] | None = None


SURGERY_VARIANTS: tuple[dict[str, Any], ...] = (
    {"name": "graphsage_base", "lift_input": False, "topo_weighted_agg": False, "topo_update_gate": False, "anchor_readout": False, "topo_pool": False},
    {"name": "graphsage_lift_input", "lift_input": True, "topo_weighted_agg": False, "topo_update_gate": False, "anchor_readout": False, "topo_pool": False},
    {"name": "graphsage_topo_agg", "lift_input": False, "topo_weighted_agg": True, "topo_update_gate": False, "anchor_readout": False, "topo_pool": False},
    {"name": "graphsage_topo_gate", "lift_input": False, "topo_weighted_agg": False, "topo_update_gate": True, "anchor_readout": False, "topo_pool": False},
    {"name": "graphsage_anchor_pool", "lift_input": False, "topo_weighted_agg": False, "topo_update_gate": False, "anchor_readout": True, "topo_pool": True},
    {"name": "graphsage_lift_topo_agg", "lift_input": True, "topo_weighted_agg": True, "topo_update_gate": False, "anchor_readout": False, "topo_pool": False},
    {"name": "graphsage_topo_agg_gate", "lift_input": False, "topo_weighted_agg": True, "topo_update_gate": True, "anchor_readout": False, "topo_pool": False},
    {"name": "graphsage_topo_agg_anchor_pool", "lift_input": False, "topo_weighted_agg": True, "topo_update_gate": False, "anchor_readout": True, "topo_pool": True},
    {"name": "graphsage_lift_topo_agg_gate_anchor_pool", "lift_input": True, "topo_weighted_agg": True, "topo_update_gate": True, "anchor_readout": True, "topo_pool": True},
)


class _BaseLayer(nn.Module):
    @staticmethod
    def _gather_neighbors(values: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_cells, hidden = values.shape
        width = neighbor_idx.shape[-1]
        expanded = values.unsqueeze(1).expand(batch_size, max_cells, max_cells, hidden)
        gather_idx = neighbor_idx.unsqueeze(-1).expand(batch_size, max_cells, width, hidden)
        gathered = expanded.gather(2, gather_idx)
        mask = neighbor_mask.unsqueeze(-1).float()
        return gathered, mask


class SpectraSAGELayer(_BaseLayer):
    def __init__(self, dim: int, dropout: float, *, topo_weighted_agg: bool, topo_update_gate: bool) -> None:
        super().__init__()
        self.topo_weighted_agg = topo_weighted_agg
        self.topo_update_gate = topo_update_gate
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        if topo_weighted_agg:
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.topo_score = nn.Linear(dim * 2, 1)
        else:
            self.query = None
            self.key = None
            self.topo_score = None
        if topo_update_gate:
            self.update_gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.Sigmoid(),
            )
        else:
            self.update_gate = None
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        cell_state: torch.Tensor,
        topo_state: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_state, mask = self._gather_neighbors(cell_state, neighbor_idx, neighbor_mask)
        neighbor_topo, _ = self._gather_neighbors(topo_state, neighbor_idx, neighbor_mask)
        mean_neighbor = (neighbor_state * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)

        if self.topo_weighted_agg:
            q = self.query(cell_state).unsqueeze(2)
            k = self.key(neighbor_state)
            topo_pair = torch.cat([topo_state.unsqueeze(2).expand_as(neighbor_topo), neighbor_topo], dim=-1)
            topo_bias = self.topo_score(topo_pair).squeeze(-1)
            scores = ((q * k).sum(dim=-1) / math.sqrt(cell_state.shape[-1])) + topo_bias
            scores = scores.masked_fill(neighbor_mask == 0, -1e9)
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1) * mask
            weighted_neighbor = (weights * neighbor_state).sum(dim=2)
        else:
            weighted_neighbor = mean_neighbor

        updated = self.proj(torch.cat([cell_state, mean_neighbor, weighted_neighbor], dim=-1))
        if self.topo_update_gate:
            gate = self.update_gate(torch.cat([cell_state, topo_state], dim=-1))
            updated = gate * updated + (1.0 - gate) * cell_state
        return self.norm(cell_state + self.dropout(updated))


class GraphSAGESpectraSurgery(nn.Module):
    def __init__(
        self,
        *,
        marker_dim: int,
        num_types: int,
        structural_dim: int,
        model_dim: int,
        type_embedding_dim: int,
        dropout: float,
        num_layers: int,
        anchor_count: int,
        router_steps: int,
        lift_input: bool,
        topo_weighted_agg: bool,
        topo_update_gate: bool,
        anchor_readout: bool,
        topo_pool: bool,
    ) -> None:
        super().__init__()
        self.lift_input = lift_input
        self.anchor_readout = anchor_readout
        self.topo_pool_enabled = topo_pool
        self.anchor_count = anchor_count
        self.router_steps = router_steps

        self.type_embedding = nn.Embedding(num_types + 1, type_embedding_dim)
        self.marker_proj = nn.Sequential(
            nn.Linear(max(marker_dim, 1), model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.type_proj = nn.Linear(type_embedding_dim, model_dim)
        self.lift = NormalizedLift(structural_dim, model_dim)
        self.topology_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.cell_norm = nn.LayerNorm(model_dim)
        self.layers = nn.ModuleList(
            [
                SpectraSAGELayer(
                    model_dim,
                    dropout,
                    topo_weighted_agg=topo_weighted_agg,
                    topo_update_gate=topo_update_gate,
                )
                for _ in range(max(1, num_layers))
            ]
        )
        self.dropout = nn.Dropout(dropout)
        if self.anchor_readout:
            self.key_proj = nn.Linear(model_dim, model_dim)
            self.value_proj = nn.Linear(model_dim, model_dim)
            self.anchor_queries = nn.Parameter(torch.randn(anchor_count, model_dim) * 0.02)
            self.router_gru = nn.GRUCell(model_dim, model_dim)
        if self.topo_pool_enabled:
            self.topo_pool = nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, 1),
            )
        extra_blocks = 0
        if self.anchor_readout:
            extra_blocks += 2
        if self.topo_pool_enabled:
            extra_blocks += 1
        readout_in = model_dim * (2 + extra_blocks) + 1
        self.readout = nn.Sequential(
            nn.Linear(readout_in, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 2),
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = values.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled = masked.max(dim=1).values
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

    def _anchor_summary(self, cell_state: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = cell_state.shape[0]
        keys = self.key_proj(cell_state)
        values = self.value_proj(cell_state)
        anchors = self.anchor_queries.unsqueeze(0).expand(batch_size, -1, -1)
        for _ in range(self.router_steps):
            scores = torch.einsum("bad,bnd->ban", anchors, keys) / math.sqrt(cell_state.shape[-1])
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
            weights = torch.softmax(scores, dim=-1)
            routed = torch.einsum("ban,bnd->bad", weights, values)
            anchors = self.router_gru(
                routed.reshape(batch_size * self.anchor_count, -1),
                anchors.reshape(batch_size * self.anchor_count, -1),
            ).reshape(batch_size, self.anchor_count, -1)
        return anchors.mean(dim=1), anchors.max(dim=1).values

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
        _normed, lifted_flat = self.lift(structural.reshape(-1, structural.shape[-1]))
        topo_embed = self.topology_proj(lifted_flat.reshape(structural.shape[0], structural.shape[1], -1))
        base_state = type_embed + marker_embed + coord_embed
        if self.lift_input:
            base_state = base_state + topo_embed
        cell_state = self.cell_norm(base_state)
        for layer in self.layers:
            cell_state = layer(cell_state, topo_embed, batch["neighbor_idx"], batch["neighbor_mask"])

        pooled_mean = self._masked_mean(cell_state, mask)
        pooled_max = self._masked_max(cell_state, mask)
        features = [pooled_mean, pooled_max]
        if self.anchor_readout:
            anchor_mean, anchor_max = self._anchor_summary(cell_state, mask)
            features.extend([anchor_mean, anchor_max])
        if self.topo_pool_enabled:
            topo_scores = self.topo_pool(torch.cat([cell_state, topo_embed], dim=-1)).squeeze(-1)
            topo_scores = topo_scores.masked_fill(~mask, -1e9)
            topo_weights = torch.softmax(topo_scores, dim=-1).unsqueeze(-1)
            topo_summary = (topo_weights * cell_state).sum(dim=1)
            features.append(topo_summary)

        cell_count = mask.sum(dim=1, keepdim=True).float().log1p()
        logits = self.readout(self.dropout(torch.cat([*features, cell_count], dim=-1)))
        return {"logits": logits}


def _evaluate_model(
    model: GraphSAGESpectraSurgery,
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
            logits = model(batch)["logits"][:, 1] - model(batch)["logits"][:, 0]
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
    cfg: GNNSurgeryConfig,
    variant: dict[str, Any],
    examples: list[SpatialRegionExample],
    labels: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    num_types: int,
    seed: int,
) -> tuple[dict[str, float], GraphSAGESpectraSurgery]:
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
    model = GraphSAGESpectraSurgery(
        marker_dim=examples[0].markers.shape[1],
        num_types=num_types,
        structural_dim=examples[0].structural.shape[1],
        model_dim=cfg.model_dim,
        type_embedding_dim=cfg.type_embedding_dim,
        dropout=cfg.dropout,
        num_layers=2,
        anchor_count=cfg.anchor_count,
        router_steps=cfg.router_steps,
        lift_input=variant["lift_input"],
        topo_weighted_agg=variant["topo_weighted_agg"],
        topo_update_gate=variant["topo_update_gate"],
        anchor_readout=variant["anchor_readout"],
        topo_pool=variant["topo_pool"],
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
        threshold, val_prob = _select_threshold(val_true, val_logits, cfg.temperature, cfg.threshold_grid_size)
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
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    model.load_state_dict(best_state)
    _raw_test_metrics, test_true, test_logits = _evaluate_model(model, test_loader, device, threshold=0.5)
    test_prob = 1.0 / (1.0 + np.exp(-(test_logits / max(cfg.temperature, 1e-6))))
    if cfg.blend_with_engineered and best_engineered_model is not None:
        engineered_prob = best_engineered_model.predict_proba(test_engineered)[:, 1]
        test_prob = best_alpha * test_prob + (1.0 - best_alpha) * engineered_prob
    test_pred = (test_prob >= best_threshold).astype(np.int64)
    test_metrics = _score_binary(test_true, test_pred, test_prob)
    test_metrics["decision_threshold"] = float(best_threshold)
    test_metrics["blend_alpha"] = float(best_alpha)
    return test_metrics, model


def run_gnn_surgery_study(cfg: GNNSurgeryConfig) -> dict[str, Any]:
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
    if getattr(cfg, "split_mode", "grouped") == "lopo":
        splits = build_lopo_splits(labels, groups, stratified=True, random_state=cfg.random_state)
    else:
        splits = build_grouped_splits(labels, groups, n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.random_state)

    variants = list(SURGERY_VARIANTS)
    if cfg.variant_names:
        allowed = set(cfg.variant_names)
        variants = [variant for variant in variants if variant["name"] in allowed]

    results: dict[str, Any] = {"runs": [], "summary": {}, "variant_space": variants}
    for variant_idx, variant in enumerate(variants):
        fold_metrics: list[dict[str, float]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if np.unique(labels[train_idx]).shape[0] < 2 or np.unique(labels[test_idx]).shape[0] < 2:
                continue
            metrics, _model = _train_fold(
                cfg,
                variant,
                examples,
                labels,
                groups,
                train_idx,
                test_idx,
                num_types=len(type_vocab),
                seed=cfg.random_state + 97 * variant_idx + fold_idx,
            )
            metrics["fold_index"] = fold_idx
            fold_metrics.append(metrics)
        if not fold_metrics:
            continue
        summary_metrics = {key: float(np.mean([fold[key] for fold in fold_metrics])) for key in fold_metrics[0] if key != "fold_index"}
        results["runs"].append({"model_name": variant["name"], "metrics": summary_metrics, "fold_metrics": fold_metrics})
    if results["runs"]:
        results["summary"]["best_run"] = max(
            results["runs"],
            key=lambda item: item["metrics"]["balanced_accuracy"] * 0.45 + item["metrics"]["macro_f1"] * 0.35 + item["metrics"]["auroc"] * 0.20,
        )
    return results


def save_gnn_surgery_results(results: dict[str, Any], output_dir: str | Path) -> Path:
    root = ensure_dir(output_dir)
    path = root / "gnn_surgery_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


__all__ = ["GNNSurgeryConfig", "SURGERY_VARIANTS", "run_gnn_surgery_study", "save_gnn_surgery_results"]
