from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from ..utils.io import ensure_dir

from ..evaluation.splits import bootstrap_ci, build_grouped_splits, build_lopo_splits
from ..models.spatial_z4 import SpatialRegionExample, build_region_examples


@dataclass
class LocalInteractionConfig:
    study_dir: str
    output_dir: str
    random_state: int = 42
    n_splits: int = 4
    n_repeats: int = 1
    split_mode: str = "grouped"
    batch_size: int = 12
    epochs: int = 10
    patience: int = 3
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    knn_k: int = 6
    neighborhood_mode: str = "adaptive_knn"
    model_dim: int = 40
    anchor_count: int = 8
    router_steps: int = 3
    type_embedding_dim: int = 16
    max_cells: int = 160
    dropout: float = 0.1


@dataclass(frozen=True)
class LocalVariant:
    name: str
    mode: str


def default_variants() -> list[LocalVariant]:
    return [
        LocalVariant(name="no_local", mode="none"),
        LocalVariant(name="mean_residual", mode="mean"),
        LocalVariant(name="gated_mean", mode="gated"),
        LocalVariant(name="neighbor_attention", mode="attention"),
    ]


class LocalDataset(Dataset):
    def __init__(self, examples: list[SpatialRegionExample], labels: np.ndarray):
        self.examples = examples
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[SpatialRegionExample, int]:
        return self.examples[idx], int(self.labels[idx])


def _collate(batch: list[tuple[SpatialRegionExample, int]]) -> dict[str, torch.Tensor]:
    examples, labels = zip(*batch, strict=False)
    max_cells = max(ex.coords.shape[0] for ex in examples)
    marker_dim = examples[0].markers.shape[1]
    structural_dim = examples[0].structural.shape[1]
    max_neighbors = max(ex.neighbor_idx.shape[1] for ex in examples)
    batch_size = len(examples)

    coords = torch.zeros(batch_size, max_cells, 2, dtype=torch.float32)
    markers = torch.zeros(batch_size, max_cells, marker_dim, dtype=torch.float32)
    structural = torch.zeros(batch_size, max_cells, structural_dim, dtype=torch.float32)
    type_ids = torch.full((batch_size, max_cells), -1, dtype=torch.long)
    neighbor_idx = torch.zeros(batch_size, max_cells, max_neighbors, dtype=torch.long)
    neighbor_mask = torch.zeros(batch_size, max_cells, max_neighbors, dtype=torch.bool)
    mask = torch.zeros(batch_size, max_cells, dtype=torch.bool)

    for batch_idx, ex in enumerate(examples):
        n_cells = ex.coords.shape[0]
        width = ex.neighbor_idx.shape[1]
        coords[batch_idx, :n_cells] = torch.from_numpy(ex.coords)
        if marker_dim > 0:
            markers[batch_idx, :n_cells] = torch.from_numpy(ex.markers)
        structural[batch_idx, :n_cells] = torch.from_numpy(ex.structural)
        type_ids[batch_idx, :n_cells] = torch.from_numpy(ex.type_ids)
        neighbor_idx[batch_idx, :n_cells, :width] = torch.from_numpy(ex.neighbor_idx)
        neighbor_mask[batch_idx, :n_cells, :width] = torch.from_numpy(ex.neighbor_mask)
        mask[batch_idx, :n_cells] = True

    return {
        "coords": coords,
        "markers": markers,
        "structural": structural,
        "type_ids": type_ids,
        "neighbor_idx": neighbor_idx,
        "neighbor_mask": neighbor_mask,
        "mask": mask,
        "labels": torch.tensor(labels, dtype=torch.long),
    }


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


def _structural_layout(dim: int) -> dict[str, slice]:
    num_types = (dim - 5) // 2
    return {"hist": slice(5 + num_types, 5 + 2 * num_types)}


def _boundary_labels(example: SpatialRegionExample) -> np.ndarray:
    hist = example.structural[:, _structural_layout(example.structural.shape[1])["hist"]]
    purity = hist[np.arange(hist.shape[0]), example.type_ids]
    return (purity < 0.6).astype(np.int64)


class LocalInteractionProbe(nn.Module):
    def __init__(
        self,
        *,
        marker_dim: int,
        structural_dim: int,
        num_types: int,
        variant: LocalVariant,
        model_dim: int,
        anchor_count: int,
        router_steps: int,
        type_embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.model_dim = model_dim
        self.anchor_count = anchor_count
        self.router_steps = router_steps
        hidden = max(16, model_dim // 2)
        self.type_embedding = nn.Embedding(num_types, type_embedding_dim)
        self.type_proj = nn.Linear(type_embedding_dim, model_dim)
        self.marker_proj = nn.Sequential(nn.Linear(marker_dim, hidden), nn.GELU(), nn.Linear(hidden, model_dim))
        self.coord_proj = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, model_dim))
        self.structural_proj = nn.Sequential(nn.Linear(structural_dim, hidden), nn.GELU(), nn.Linear(hidden, model_dim))
        self.cell_norm = nn.LayerNorm(model_dim)

        self.local_update = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.local_gate = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim), nn.Sigmoid())
        self.query_proj = nn.Linear(model_dim, model_dim)
        self.key_proj_local = nn.Linear(model_dim, model_dim)
        self.local_norm = nn.LayerNorm(model_dim)

        self.router_key = nn.Linear(model_dim, model_dim)
        self.router_val = nn.Linear(model_dim, model_dim)
        self.anchor_queries = nn.Parameter(torch.randn(anchor_count, model_dim) * 0.02)
        self.router_gru = nn.GRUCell(model_dim, model_dim)
        self.readout = nn.Sequential(
            nn.Linear(model_dim * 4 + 1, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 2),
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)

    def _base_cells(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        markers = torch.log1p(batch["markers"].clamp_min(0.0))
        type_ids = batch["type_ids"].clamp_min(0)
        encoded = (
            self.marker_proj(markers)
            + self.coord_proj(batch["coords"])
            + self.structural_proj(batch["structural"])
            + self.type_proj(self.type_embedding(type_ids))
        )
        return self.cell_norm(encoded)

    def _neighbor_state(self, cell_state: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_cells, hidden = cell_state.shape
        width = neighbor_idx.shape[-1]
        expanded = cell_state.unsqueeze(1).expand(batch_size, max_cells, max_cells, hidden)
        gather_idx = neighbor_idx.unsqueeze(-1).expand(batch_size, max_cells, width, hidden)
        gathered = expanded.gather(2, gather_idx)
        mask = neighbor_mask.unsqueeze(-1).float()
        return (gathered * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)

    def _attention_neighbor_state(self, cell_state: torch.Tensor, neighbor_idx: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_cells, hidden = cell_state.shape
        width = neighbor_idx.shape[-1]
        expanded = cell_state.unsqueeze(1).expand(batch_size, max_cells, max_cells, hidden)
        gather_idx = neighbor_idx.unsqueeze(-1).expand(batch_size, max_cells, width, hidden)
        neigh = expanded.gather(2, gather_idx)
        query = self.query_proj(cell_state).unsqueeze(2)
        keys = self.key_proj_local(neigh)
        score = (query * keys).sum(dim=-1) / math.sqrt(hidden)
        score = score.masked_fill(~neighbor_mask, -1e9)
        weights = torch.softmax(score, dim=-1).unsqueeze(-1)
        return (neigh * weights).sum(dim=2)

    def _apply_local(self, cell_state: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.variant.mode == "none":
            return cell_state
        if self.variant.mode == "attention":
            neighbor_mean = self._attention_neighbor_state(cell_state, batch["neighbor_idx"], batch["neighbor_mask"])
            updated = self.local_update(torch.cat([cell_state, neighbor_mean, cell_state - neighbor_mean], dim=-1))
            return self.local_norm(cell_state + updated)
        neighbor_mean = self._neighbor_state(cell_state, batch["neighbor_idx"], batch["neighbor_mask"])
        updated = self.local_update(torch.cat([cell_state, neighbor_mean, cell_state - neighbor_mean], dim=-1))
        if self.variant.mode == "gated":
            gate = self.local_gate(torch.cat([cell_state, neighbor_mean], dim=-1))
            return self.local_norm(cell_state + gate * updated)
        return self.local_norm(cell_state + updated)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cell_state = self._apply_local(self._base_cells(batch), batch)
        mask = batch["mask"]
        keys = self.router_key(cell_state)
        values = self.router_val(cell_state)
        anchor_state = self.anchor_queries.unsqueeze(0).expand(cell_state.shape[0], -1, -1)
        scores = None
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
        cell_count = mask.sum(dim=1, keepdim=True).float().log1p()
        logits = self.readout(torch.cat([pooled_mean, pooled_max, regional_mean, pooled_mean, cell_count], dim=-1))
        return {"logits": logits, "cell_state": cell_state}


def _loader(examples: list[SpatialRegionExample], labels: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(LocalDataset(examples, labels), batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)


def _choose_validation_indices(train_idx: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if np.unique(groups[train_idx]).shape[0] < 4:
        return train_idx, np.array([], dtype=int)
    splits = build_grouped_splits(labels[train_idx], groups[train_idx], n_splits=4, n_repeats=1, random_state=seed)
    if not splits:
        return train_idx, np.array([], dtype=int)
    inner_train, inner_val = splits[0]
    return train_idx[inner_train], train_idx[inner_val]


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best = 0.5
    best_score = -np.inf
    for tau in np.linspace(0.2, 0.8, num=25):
        score = balanced_accuracy_score(y_true, (y_prob >= tau).astype(np.int64))
        if score > best_score:
            best_score = score
            best = float(tau)
    return best


def _evaluate_probs(model: LocalInteractionProbe, loader: DataLoader, device: torch.device, threshold: float) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            probs = torch.softmax(model(batch)["logits"], dim=-1)[:, 1]
            all_labels.append(batch["labels"].cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = (y_prob >= threshold).astype(np.int64)
    metrics = _score_binary(y_true, y_pred, y_prob)
    metrics["decision_threshold"] = float(threshold)
    return metrics, y_true, y_prob


def _collect_cell_state(model: LocalInteractionProbe, example: SpatialRegionExample, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        batch = {
            "coords": torch.from_numpy(example.coords).float().unsqueeze(0).to(device),
            "markers": torch.from_numpy(example.markers).float().unsqueeze(0).to(device),
            "structural": torch.from_numpy(example.structural).float().unsqueeze(0).to(device),
            "type_ids": torch.from_numpy(example.type_ids).long().unsqueeze(0).to(device),
            "neighbor_idx": torch.from_numpy(example.neighbor_idx).long().unsqueeze(0).to(device),
            "neighbor_mask": torch.from_numpy(example.neighbor_mask).bool().unsqueeze(0).to(device),
            "mask": torch.ones(1, example.coords.shape[0], dtype=torch.bool, device=device),
        }
        return model(batch)["cell_state"].squeeze(0).cpu().numpy()


def _intrinsic_metrics(
    model: LocalInteractionProbe,
    train_examples: list[SpatialRegionExample],
    test_examples: list[SpatialRegionExample],
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    train_embed = np.concatenate([_collect_cell_state(model, ex, device) for ex in train_examples], axis=0)
    test_embed = np.concatenate([_collect_cell_state(model, ex, device) for ex in test_examples], axis=0)
    train_types = np.concatenate([ex.type_ids for ex in train_examples], axis=0)
    test_types = np.concatenate([ex.type_ids for ex in test_examples], axis=0)
    train_boundary = np.concatenate([_boundary_labels(ex) for ex in train_examples], axis=0)
    test_boundary = np.concatenate([_boundary_labels(ex) for ex in test_examples], axis=0)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embed)
    test_scaled = scaler.transform(test_embed)
    type_probe = LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed)
    type_probe.fit(train_scaled, train_types)
    type_pred = type_probe.predict(test_scaled)
    type_f1 = float(f1_score(test_types, type_pred, average="macro"))
    if np.unique(train_boundary).shape[0] >= 2 and np.unique(test_boundary).shape[0] >= 2:
        boundary_probe = LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed)
        boundary_probe.fit(train_scaled, train_boundary)
        boundary_prob = boundary_probe.predict_proba(test_scaled)[:, 1]
        try:
            boundary_auroc = float(roc_auc_score(test_boundary, boundary_prob))
        except ValueError:
            boundary_auroc = 0.5
    else:
        boundary_auroc = 0.5
    same_sims: list[float] = []
    cross_sims: list[float] = []
    offset = 0
    for example in test_examples:
        state = test_scaled[offset : offset + example.coords.shape[0]]
        offset += example.coords.shape[0]
        norms = np.linalg.norm(state, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-6, None)
        cosine = (state / norms) @ (state / norms).T
        for idx in range(example.coords.shape[0]):
            for jdx in range(idx + 1, example.coords.shape[0]):
                if example.type_ids[idx] == example.type_ids[jdx]:
                    same_sims.append(float(cosine[idx, jdx]))
                else:
                    cross_sims.append(float(cosine[idx, jdx]))
    return {
        "cell_type_macro_f1": type_f1,
        "boundary_auroc": boundary_auroc,
        "oversmoothing_gap": float(np.mean(same_sims) - np.mean(cross_sims)) if same_sims and cross_sims else 0.0,
    }


def _train_variant_fold(
    cfg: LocalInteractionConfig,
    variant: LocalVariant,
    examples: list[SpatialRegionExample],
    labels: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    num_types: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    fit_idx, val_idx = _choose_validation_indices(train_idx, labels, groups, seed)
    train_examples = [copy.deepcopy(examples[idx]) for idx in fit_idx]
    val_examples = [copy.deepcopy(examples[idx]) for idx in val_idx]
    test_examples = [copy.deepcopy(examples[idx]) for idx in test_idx]
    train_labels = labels[fit_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    model = LocalInteractionProbe(
        marker_dim=train_examples[0].markers.shape[1],
        structural_dim=train_examples[0].structural.shape[1],
        num_types=num_types,
        variant=variant,
        model_dim=cfg.model_dim,
        anchor_count=cfg.anchor_count,
        router_steps=cfg.router_steps,
        type_embedding_dim=cfg.type_embedding_dim,
        dropout=cfg.dropout,
    ).to(device)
    class_counts = np.bincount(train_labels, minlength=2)
    weights = class_counts.sum() / np.clip(class_counts, 1, None)
    loss_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    train_loader = _loader(train_examples, train_labels, batch_size=cfg.batch_size, shuffle=True)
    val_loader = _loader(val_examples, val_labels, batch_size=cfg.batch_size, shuffle=False) if len(val_examples) else None
    test_loader = _loader(test_examples, test_labels, batch_size=cfg.batch_size, shuffle=False)

    best_state = copy.deepcopy(model.state_dict())
    best_threshold = 0.5
    best_score = -np.inf
    patience = 0
    for _ in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch)["logits"], batch["labels"], weight=loss_weight, label_smoothing=0.02)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        if val_loader is None or len(val_examples) == 0 or np.unique(val_labels).shape[0] < 2:
            best_state = copy.deepcopy(model.state_dict())
            continue
        _, val_true, val_prob = _evaluate_probs(model, val_loader, device, threshold=0.5)
        threshold = _best_threshold(val_true, val_prob)
        val_metrics, _, _ = _evaluate_probs(model, val_loader, device, threshold=threshold)
        score = 0.55 * val_metrics["balanced_accuracy"] + 0.25 * val_metrics["macro_f1"] + 0.2 * val_metrics["auroc"]
        if score > best_score + 1e-4:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = threshold
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break
    model.load_state_dict(best_state)
    metrics, _, _ = _evaluate_probs(model, test_loader, device, threshold=best_threshold)
    metrics.update(_intrinsic_metrics(model, train_examples, test_examples, device, seed))
    return metrics


def run_local_interaction_isolation(cfg: LocalInteractionConfig, variants: list[LocalVariant] | None = None) -> dict[str, Any]:
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    if variants is None:
        variants = default_variants()
    examples, type_vocab = build_region_examples(
        cfg.study_dir,
        knn_k=cfg.knn_k,
        max_cells=cfg.max_cells,
        features_path=None,
        neighborhood_mode=cfg.neighborhood_mode,
    )
    labels_raw = np.array([ex.label for ex in examples], dtype=object)
    groups = np.array([ex.patient_id for ex in examples], dtype=object)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels_raw)
    if getattr(cfg, "split_mode", "grouped") == "lopo":
        splits = build_lopo_splits(labels, groups, stratified=True, random_state=cfg.random_state)
    else:
        splits = build_grouped_splits(labels, groups, n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs: list[dict[str, Any]] = []
    for variant in variants:
        fold_metrics: list[dict[str, float]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if np.unique(labels[train_idx]).shape[0] < 2 or np.unique(labels[test_idx]).shape[0] < 2:
                continue
            metrics = _train_variant_fold(
                cfg,
                variant,
                examples,
                labels,
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
            continue
        aggregate = {
            key: float(np.mean([row[key] for row in fold_metrics]))
            for key in (
                "auroc",
                "balanced_accuracy",
                "macro_f1",
                "brier_score",
                "decision_threshold",
                "cell_type_macro_f1",
                "boundary_auroc",
                "oversmoothing_gap",
            )
        }
        runs.append({"variant": asdict(variant), "metrics": aggregate, "fold_metrics": fold_metrics})
    if not runs:
        raise RuntimeError("Local interaction isolation produced no valid runs.")
    best_downstream = max(runs, key=lambda row: (row["metrics"]["balanced_accuracy"], row["metrics"]["auroc"]))
    best_intrinsic = max(runs, key=lambda row: (row["metrics"]["boundary_auroc"], row["metrics"]["oversmoothing_gap"]))
    return {
        "runs": runs,
        "summary": {
            "best_downstream": best_downstream,
            "best_intrinsic": best_intrinsic,
            "label_classes": encoder.classes_.tolist(),
        },
    }


def save_local_interaction_results(results: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    path = Path(output_dir) / "local_interaction_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


__all__ = [
    "LocalInteractionConfig",
    "LocalVariant",
    "default_variants",
    "run_local_interaction_isolation",
    "save_local_interaction_results",
]
