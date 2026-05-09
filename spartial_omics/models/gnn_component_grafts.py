from __future__ import annotations

import copy
import itertools
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

from .gnn_baselines import GNNBaselineConfig, GraphSAGELayer, _make_loader
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
from ..evaluation.splits import bootstrap_ci, build_grouped_splits, build_lopo_splits


@dataclass
class GNNGraftConfig(GNNBaselineConfig):
    anchor_count: int = 8
    router_steps: int = 3
    variant_names: tuple[str, ...] | None = None


def _variant_name(
    *,
    structural_input: bool,
    lift_mode: str,
    anchor_readout: bool,
    topo_pool: bool,
    residual_topology: bool,
    engineered_context: bool,
) -> str:
    parts = ["graphsage"]
    if structural_input:
        parts.append("struct")
        if lift_mode == "normalized_linear":
            parts.append("lift")
    if anchor_readout:
        parts.append("anchor")
    if topo_pool:
        parts.append("pool")
    if residual_topology:
        parts.append("resid")
    if engineered_context:
        parts.append("ctx")
    if len(parts) == 1:
        parts.append("base")
    return "_".join(parts)


def build_graft_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for structural_input, lift_mode, anchor_readout, topo_pool, residual_topology, engineered_context in itertools.product(
        [False, True],
        ["identity", "normalized_linear"],
        [False, True],
        [False, True],
        [False, True],
        [False, True],
    ):
        if not structural_input and lift_mode != "identity":
            continue
        variants.append(
            {
                "name": _variant_name(
                    structural_input=structural_input,
                    lift_mode=lift_mode,
                    anchor_readout=anchor_readout,
                    topo_pool=topo_pool,
                    residual_topology=residual_topology,
                    engineered_context=engineered_context,
                ),
                "structural_input": structural_input,
                "lift_mode": lift_mode,
                "anchor_readout": anchor_readout,
                "topo_pool": topo_pool,
                "residual_topology": residual_topology,
                "engineered_context": engineered_context,
            }
        )
    return variants


class GraphSAGEGraftClassifier(nn.Module):
    def __init__(
        self,
        *,
        marker_dim: int,
        num_types: int,
        structural_dim: int,
        engineered_dim: int,
        model_dim: int,
        type_embedding_dim: int,
        dropout: float,
        num_layers: int,
        anchor_count: int,
        router_steps: int,
        structural_input: bool,
        lift_mode: str,
        anchor_readout: bool,
        topo_pool: bool,
        residual_topology: bool,
        engineered_context: bool,
    ) -> None:
        super().__init__()
        self.structural_input = structural_input
        self.anchor_readout = anchor_readout
        self.topo_pool_enabled = topo_pool
        self.residual_topology = residual_topology
        self.engineered_context = engineered_context and engineered_dim > 0
        self.anchor_count = anchor_count
        self.router_steps = router_steps

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
        self.lift_mode = lift_mode
        if lift_mode == "identity":
            self.lift = None
            topo_in_dim = structural_dim
        elif lift_mode == "normalized_linear":
            self.lift = NormalizedLift(structural_dim, model_dim)
            topo_in_dim = model_dim
        else:
            raise ValueError(f"Unsupported lift_mode: {lift_mode}")
        self.topology_proj = nn.Sequential(
            nn.Linear(topo_in_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.cell_norm = nn.LayerNorm(model_dim)
        self.layers = nn.ModuleList([GraphSAGELayer(model_dim, dropout) for _ in range(max(1, num_layers))])
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
        if self.engineered_context:
            self.engineered_proj = nn.Sequential(
                nn.Linear(engineered_dim, model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim, model_dim),
            )
            self.context_gate = nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
                nn.Sigmoid(),
            )
        else:
            self.engineered_proj = None
            self.context_gate = None

        base_extra_blocks = 0
        if self.anchor_readout:
            base_extra_blocks += 2
        if self.topo_pool_enabled:
            base_extra_blocks += 1
        readout_in = model_dim * (2 + base_extra_blocks) + 1
        self.readout = nn.Sequential(
            nn.Linear(readout_in, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 2),
        )
        if self.residual_topology:
            topo_head_in = model_dim * (2 if self.topo_pool_enabled else 1) + 1
            self.topology_head = nn.Sequential(
                nn.Linear(topo_head_in, model_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, 2),
            )
        else:
            self.topology_head = None

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        summed = (values * weights).sum(dim=dim)
        denom = weights.sum(dim=dim).clamp_min(1.0)
        return summed / denom

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
        flat_values = values.reshape(batch_size * values.shape[1], values.shape[2])
        for _ in range(self.router_steps):
            scores = torch.einsum("bad,bnd->ban", anchors, keys) / math.sqrt(cell_state.shape[-1])
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
            weights = torch.softmax(scores, dim=-1)
            routed = torch.einsum("ban,bnd->bad", weights, values)
            anchors = self.router_gru(
                routed.reshape(batch_size * self.anchor_count, -1),
                anchors.reshape(batch_size * self.anchor_count, -1),
            ).reshape(batch_size, self.anchor_count, -1)
        anchor_mean = anchors.mean(dim=1)
        anchor_max = anchors.max(dim=1).values
        return anchor_mean, anchor_max

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        coords = batch["coords"]
        markers = torch.log1p(batch["markers"].clamp_min(0.0))
        if markers.shape[-1] == 0:
            markers = torch.zeros((*coords.shape[:2], 1), dtype=coords.dtype, device=coords.device)
        type_ids = batch["type_ids"].clamp_min(0)
        structural = batch["structural"]
        mask = batch["mask"]
        engineered = batch["engineered"]

        type_embed = self.type_proj(self.type_embedding(type_ids))
        marker_embed = self.marker_proj(markers)
        coord_embed = self.coord_proj(coords)
        if self.lift is not None:
            flat_struct = structural.reshape(-1, structural.shape[-1])
            _normed, lifted_flat = self.lift(flat_struct)
            lifted = lifted_flat.reshape(structural.shape[0], structural.shape[1], -1)
        else:
            lifted = structural
        topo_embed = self.topology_proj(lifted)
        base_state = type_embed + marker_embed + coord_embed
        if self.structural_input:
            base_state = base_state + topo_embed
        cell_state = self.cell_norm(base_state)
        for layer in self.layers:
            cell_state = layer(cell_state, batch["neighbor_idx"], batch["neighbor_mask"])

        pooled_mean = self._masked_mean(cell_state, mask, dim=1)
        pooled_max = self._masked_max(cell_state, mask)
        if self.engineered_context:
            engineered_state = self.engineered_proj(engineered)
            gate = self.context_gate(torch.cat([pooled_mean, engineered_state], dim=-1))
            pooled_mean = gate * pooled_mean + (1.0 - gate) * engineered_state
        feature_blocks = [pooled_mean, pooled_max]

        topo_mean = self._masked_mean(topo_embed, mask, dim=1)
        topo_pool_summary = None
        if self.anchor_readout:
            anchor_mean, anchor_max = self._anchor_summary(cell_state, mask)
            feature_blocks.extend([anchor_mean, anchor_max])
        if self.topo_pool_enabled:
            topo_scores = self.topo_pool(torch.cat([cell_state, topo_embed], dim=-1)).squeeze(-1)
            topo_scores = topo_scores.masked_fill(~mask, -1e9)
            topo_weights = torch.softmax(topo_scores, dim=-1).unsqueeze(-1)
            topo_pool_summary = (topo_weights * cell_state).sum(dim=1)
            feature_blocks.append(topo_pool_summary)

        cell_count = mask.sum(dim=1, keepdim=True).float().log1p()
        logits = self.readout(self.dropout(torch.cat([*feature_blocks, cell_count], dim=-1)))

        topo_logits = None
        if self.topology_head is not None:
            topo_features = [topo_mean]
            if topo_pool_summary is not None:
                topo_features.append(topo_pool_summary)
            topo_logits = self.topology_head(self.dropout(torch.cat([*topo_features, cell_count], dim=-1)))
            logits = logits + topo_logits
        return {"logits": logits, "topology_logits": topo_logits}


def _evaluate_model(
    model: GraphSAGEGraftClassifier,
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
    cfg: GNNGraftConfig,
    variant: dict[str, Any],
    examples: list[SpatialRegionExample],
    labels: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    num_types: int,
    seed: int,
) -> tuple[dict[str, float], GraphSAGEGraftClassifier]:
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
    model = GraphSAGEGraftClassifier(
        marker_dim=examples[0].markers.shape[1],
        num_types=num_types,
        structural_dim=examples[0].structural.shape[1],
        engineered_dim=examples[0].engineered_features.shape[0],
        model_dim=cfg.model_dim,
        type_embedding_dim=cfg.type_embedding_dim,
        dropout=cfg.dropout,
        num_layers=2,
        anchor_count=cfg.anchor_count,
        router_steps=cfg.router_steps,
        structural_input=variant["structural_input"],
        lift_mode=variant["lift_mode"],
        anchor_readout=variant["anchor_readout"],
        topo_pool=variant["topo_pool"],
        residual_topology=variant["residual_topology"],
        engineered_context=variant["engineered_context"],
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
            if out["topology_logits"] is not None:
                loss = loss + 0.2 * _focal_loss(out["topology_logits"], batch["labels"], loss_weight, cfg.focal_gamma, cfg.label_smoothing)
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


def run_gnn_component_grafts_study(cfg: GNNGraftConfig) -> dict[str, Any]:
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
        splits = build_grouped_splits(
            labels,
            groups,
            n_splits=cfg.n_splits,
            n_repeats=cfg.n_repeats,
            random_state=cfg.random_state,
        )
    variants = build_graft_variants()
    if cfg.variant_names:
        allowed = set(cfg.variant_names)
        variants = [variant for variant in variants if variant["name"] in allowed]

    results: dict[str, Any] = {
        "runs": [],
        "summary": {},
        "variant_space": variants,
        "components": ["structural_input", "lift_mode", "anchor_readout", "topo_pool", "residual_topology", "engineered_context"],
    }
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
        summary_metrics = {
            key: float(np.mean([fold[key] for fold in fold_metrics]))
            for key in fold_metrics[0]
            if key != "fold_index"
        }
        run = {
            "model_name": variant["name"],
            "components_enabled": {key: variant[key] for key in results["components"]},
            "label_classes": unique_labels,
            "metrics": summary_metrics,
            "fold_metrics": fold_metrics,
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


def save_gnn_component_graft_results(results: dict[str, Any], output_dir: str | Path) -> Path:
    root = ensure_dir(output_dir)
    path = root / "gnn_component_graft_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


__all__ = [
    "GNNGraftConfig",
    "build_graft_variants",
    "run_gnn_component_grafts_study",
    "save_gnn_component_graft_results",
]
