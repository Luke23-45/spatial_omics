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
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from ..utils.io import ensure_dir

from ..evaluation.splits import bootstrap_ci, build_grouped_splits, build_lopo_splits
from .spatial_z4 import SpatialRegionExample, build_region_examples


@dataclass
class SpectraSearchConfig:
    study_dir: str
    output_dir: str
    baseline_results_json: str | None = None
    random_state: int = 42
    n_splits: int = 4
    n_repeats: int = 1
    split_mode: str = "grouped"
    batch_size: int = 10
    epochs: int = 8
    patience: int = 3
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    knn_k: int = 6
    neighborhood_mode: str = "adaptive_knn"
    model_dim: int = 64
    anchor_count: int = 10
    router_steps: int = 4
    type_embedding_dim: int = 20
    max_cells: int = 160
    dropout: float = 0.15
    spectral_components: int = 6


@dataclass(frozen=True)
class SpectraVariant:
    positional_mode: str
    neighborhood_mode: str
    spectral_mode: str

    @property
    def name(self) -> str:
        return f"pos={self.positional_mode}|nbr={self.neighborhood_mode}|spec={self.spectral_mode}"


def default_variants() -> list[SpectraVariant]:
    variants: list[SpectraVariant] = []
    for positional_mode in ("raw", "fourier"):
        for neighborhood_mode in ("soft", "hybrid"):
            for spectral_mode in ("off", "laplacian"):
                variants.append(
                    SpectraVariant(
                        positional_mode=positional_mode,
                        neighborhood_mode=neighborhood_mode,
                        spectral_mode=spectral_mode,
                    )
                )
    return variants


class SpectraDataset(Dataset):
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
    batch_size = len(examples)

    coords = torch.zeros(batch_size, max_cells, 2, dtype=torch.float32)
    markers = torch.zeros(batch_size, max_cells, marker_dim, dtype=torch.float32)
    type_ids = torch.full((batch_size, max_cells), -1, dtype=torch.long)
    mask = torch.zeros(batch_size, max_cells, dtype=torch.bool)
    local_knn = torch.zeros(batch_size, max_cells, max_cells, dtype=torch.bool)

    for batch_idx, ex in enumerate(examples):
        n_cells = ex.coords.shape[0]
        coords[batch_idx, :n_cells] = torch.from_numpy(ex.coords)
        if marker_dim > 0:
            markers[batch_idx, :n_cells] = torch.from_numpy(ex.markers)
        type_ids[batch_idx, :n_cells] = torch.from_numpy(ex.type_ids)
        mask[batch_idx, :n_cells] = True
        for i in range(n_cells):
            valid = ex.neighbor_mask[i]
            nbrs = ex.neighbor_idx[i, valid]
            local_knn[batch_idx, i, nbrs] = True

    return {
        "coords": coords,
        "markers": markers,
        "type_ids": type_ids,
        "mask": mask,
        "local_knn": local_knn,
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class FourierPositionEncoding(nn.Module):
    def __init__(self, out_dim: int, n_freqs: int = 8) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.freqs = 2.0 ** torch.arange(n_freqs, dtype=torch.float32)
        in_dim = 2 + 4 * n_freqs
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.to(coords.device)
        scaled = coords.unsqueeze(-1) * freqs
        fourier = torch.cat([coords, torch.sin(scaled).flatten(-2), torch.cos(scaled).flatten(-2)], dim=-1)
        return self.proj(fourier)


class EndToEndSpectraModel(nn.Module):
    def __init__(
        self,
        *,
        marker_dim: int,
        num_types: int,
        variant: SpectraVariant,
        model_dim: int,
        anchor_count: int,
        router_steps: int,
        type_embedding_dim: int,
        spectral_components: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.model_dim = model_dim
        self.anchor_count = anchor_count
        self.router_steps = router_steps
        self.spectral_components = spectral_components

        self.type_embedding = nn.Embedding(num_types, type_embedding_dim)
        self.marker_proj = nn.Sequential(
            nn.Linear(marker_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        if variant.positional_mode == "fourier":
            self.coord_encoder = FourierPositionEncoding(model_dim)
        else:
            self.coord_encoder = nn.Sequential(
                nn.Linear(2, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
            )
        self.type_proj = nn.Linear(type_embedding_dim, model_dim)

        self.query_proj = nn.Linear(model_dim, model_dim)
        self.key_proj = nn.Linear(model_dim, model_dim)
        self.value_proj = nn.Linear(model_dim, model_dim)
        self.spatial_bias = nn.Sequential(
            nn.Linear(3, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 1),
        )

        spectral_in = spectral_components + 1
        self.spectral_proj = nn.Sequential(
            nn.Linear(spectral_in, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

        self.cell_norm = nn.LayerNorm(model_dim)
        self.local_update = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
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

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        summed = (values * weights).sum(dim=dim)
        denom = weights.sum(dim=dim).clamp_min(1.0)
        return summed / denom

    def _pairwise_scores(self, cell_embed: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor, local_knn: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(cell_embed)
        k = self.key_proj(cell_embed)
        attn = torch.einsum("bnd,bmd->bnm", q, k) / math.sqrt(self.model_dim)
        rel = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = torch.norm(rel, dim=-1, keepdim=True)
        bias_in = torch.cat([rel, dist], dim=-1)
        bias = self.spatial_bias(bias_in).squeeze(-1)
        scores = attn + bias
        if self.variant.neighborhood_mode == "hybrid":
            scores = scores + local_knn.float() * 0.5 - (~local_knn).float() * 0.1
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        scores = scores.masked_fill(~pair_mask, -1e9)
        eye = torch.eye(scores.shape[-1], device=scores.device, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(eye, -1e9)
        return scores

    def _spectral_features(self, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_cells, _ = adjacency.shape
        out = adjacency.new_zeros(batch_size, max_cells, self.spectral_components + 1)
        for batch_idx in range(batch_size):
            valid = int(mask[batch_idx].sum().item())
            if valid <= 1:
                continue
            a = adjacency[batch_idx, :valid, :valid]
            sym = 0.5 * (a + a.transpose(0, 1))
            sym = torch.nan_to_num(sym, nan=0.0, posinf=0.0, neginf=0.0)
            deg = sym.sum(dim=-1)
            deg_inv = torch.rsqrt(deg.clamp_min(1e-6))
            identity = torch.eye(valid, device=adjacency.device)
            lap = identity - deg_inv[:, None] * sym * deg_inv[None, :]
            lap = torch.nan_to_num(lap, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                eigvals, eigvecs = torch.linalg.eigh(lap)
            except RuntimeError:
                continue
            comp = min(self.spectral_components, max(0, valid - 1))
            if comp > 0:
                out[batch_idx, :valid, :comp] = eigvecs[:, 1 : comp + 1]
                out[batch_idx, :valid, self.spectral_components] = eigvals[1 : comp + 1].mean()
            else:
                out[batch_idx, :valid, self.spectral_components] = eigvals.mean()
        return out

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        coords = batch["coords"]
        markers = torch.log1p(batch["markers"].clamp_min(0.0))
        type_ids = batch["type_ids"].clamp_min(0)
        mask = batch["mask"]

        cell_embed = (
            self.marker_proj(markers)
            + self.coord_encoder(coords)
            + self.type_proj(self.type_embedding(type_ids))
        )
        scores = self._pairwise_scores(cell_embed, coords, mask, batch["local_knn"])
        adjacency = torch.softmax(scores, dim=-1)
        adjacency = adjacency * mask.unsqueeze(1).float()
        adjacency = torch.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
        value = self.value_proj(cell_embed)
        neighbor_mean = torch.einsum("bnm,bmd->bnd", adjacency, value)

        if self.variant.spectral_mode == "laplacian":
            spectral = self.spectral_proj(self._spectral_features(adjacency, mask))
        else:
            spectral = torch.zeros_like(cell_embed)

        cell_state = self.cell_norm(cell_embed + spectral)
        update = self.local_update(torch.cat([cell_state, neighbor_mean, cell_state - neighbor_mean], dim=-1))
        cell_state = self.local_norm(cell_state + update)
        cell_state = torch.nan_to_num(cell_state, nan=0.0, posinf=0.0, neginf=0.0)

        router_keys = self.router_key(cell_state)
        router_vals = self.router_val(cell_state)
        anchor_state = self.anchor_queries.unsqueeze(0).expand(cell_state.shape[0], -1, -1)
        router_scores = None
        for _ in range(self.router_steps):
            router_scores = torch.einsum("bld,bnd->bln", anchor_state, router_keys) / math.sqrt(self.model_dim)
            router_scores = router_scores.masked_fill(~mask.unsqueeze(1), -1e9)
            weights = torch.softmax(router_scores, dim=-1)
            anchors = torch.einsum("bln,bnd->bld", weights, router_vals)
            anchor_state = self.router_gru(
                anchors.reshape(-1, self.model_dim),
                anchor_state.reshape(-1, self.model_dim),
            ).reshape(cell_state.shape[0], self.anchor_count, self.model_dim)

        anchor_mean = anchor_state.mean(dim=1)
        anchor_max = anchor_state.max(dim=1).values
        cell_mean = self._masked_mean(cell_state, mask, dim=1)
        spectral_mean = self._masked_mean(spectral, mask, dim=1)
        cell_count = mask.sum(dim=1, keepdim=True).float().log1p()
        logits = self.readout(torch.cat([anchor_mean, anchor_max, cell_mean, spectral_mean, cell_count], dim=-1))
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        return {"logits": logits, "adjacency": adjacency, "router_scores": router_scores}


def _score_binary(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
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


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_tau = 0.5
    best_score = -np.inf
    for tau in np.linspace(0.2, 0.8, num=25):
        y_pred = (y_prob >= tau).astype(np.int64)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau


def _build_loader(examples: list[SpatialRegionExample], labels: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(SpectraDataset(examples, labels), batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)


def _evaluate(model: EndToEndSpectraModel, loader: DataLoader, device: torch.device, *, threshold: float) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            out = model(batch)
            probs = torch.softmax(out["logits"], dim=-1)[:, 1]
            probs_list.append(probs.cpu().numpy())
            labels_list.append(batch["labels"].cpu().numpy())
    y_true = np.concatenate(labels_list, axis=0)
    y_prob = np.concatenate(probs_list, axis=0)
    y_pred = (y_prob >= threshold).astype(np.int64)
    metrics = _score_binary(y_true, y_pred, y_prob)
    metrics["decision_threshold"] = float(threshold)
    return metrics, y_true, y_prob


def _inner_split(train_idx: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if np.unique(groups[train_idx]).shape[0] < 4:
        return train_idx, np.array([], dtype=int)
    splits = build_grouped_splits(labels[train_idx], groups[train_idx], n_splits=4, n_repeats=1, random_state=seed)
    if not splits:
        return train_idx, np.array([], dtype=int)
    sub_train, sub_val = splits[0]
    return train_idx[sub_train], train_idx[sub_val]


def _train_variant_fold(
    cfg: SpectraSearchConfig,
    variant: SpectraVariant,
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
    fit_idx, val_idx = _inner_split(train_idx, labels, groups, seed)
    train_examples = [copy.deepcopy(examples[idx]) for idx in fit_idx]
    val_examples = [copy.deepcopy(examples[idx]) for idx in val_idx]
    test_examples = [copy.deepcopy(examples[idx]) for idx in test_idx]
    train_labels = labels[fit_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    model = EndToEndSpectraModel(
        marker_dim=train_examples[0].markers.shape[1],
        num_types=num_types,
        variant=variant,
        model_dim=cfg.model_dim,
        anchor_count=cfg.anchor_count,
        router_steps=cfg.router_steps,
        type_embedding_dim=cfg.type_embedding_dim,
        spectral_components=cfg.spectral_components,
        dropout=cfg.dropout,
    ).to(device)

    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = class_counts.sum() / np.clip(class_counts, 1, None)
    loss_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    train_loader = _build_loader(train_examples, train_labels, batch_size=cfg.batch_size, shuffle=True)
    val_loader = _build_loader(val_examples, val_labels, batch_size=cfg.batch_size, shuffle=False) if len(val_examples) else None
    test_loader = _build_loader(test_examples, test_labels, batch_size=cfg.batch_size, shuffle=False)

    best_state = copy.deepcopy(model.state_dict())
    best_threshold = 0.5
    best_score = -np.inf
    patience = 0

    for _ in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = F.cross_entropy(out["logits"], batch["labels"], weight=loss_weight, label_smoothing=0.03)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        if val_loader is None or len(val_examples) == 0 or np.unique(val_labels).shape[0] < 2:
            best_state = copy.deepcopy(model.state_dict())
            continue
        _, val_true, val_prob = _evaluate(model, val_loader, device, threshold=0.5)
        threshold = _best_threshold(val_true, val_prob)
        val_metrics, _, _ = _evaluate(model, val_loader, device, threshold=threshold)
        composite = 0.5 * val_metrics["balanced_accuracy"] + 0.3 * val_metrics["macro_f1"] + 0.2 * val_metrics["auroc"]
        if composite > best_score + 1e-4:
            best_score = composite
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = threshold
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    model.load_state_dict(best_state)
    test_metrics, _, _ = _evaluate(model, test_loader, device, threshold=best_threshold)
    return test_metrics


def run_spectra_search(cfg: SpectraSearchConfig, variants: list[SpectraVariant] | None = None) -> dict[str, Any]:
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
            for key in ("auroc", "balanced_accuracy", "macro_f1", "brier_score", "decision_threshold")
        }
        aggregate["selection_score"] = float(0.5 * aggregate["balanced_accuracy"] + 0.3 * aggregate["macro_f1"] + 0.2 * aggregate["auroc"])
        runs.append(
            {
                "variant": variant.name,
                "feature_set": "Spectra-E2E",
                "model_name": "spectra_search",
                "label_classes": encoder.classes_.tolist(),
                "metrics": aggregate,
                "fold_metrics": fold_metrics,
            }
        )

    if not runs:
        raise RuntimeError("Spectra search produced no valid runs.")

    best = max(
        runs,
        key=lambda row: (
            row["metrics"]["balanced_accuracy"],
            row["metrics"]["macro_f1"],
            row["metrics"]["auroc"],
        ),
    )
    results: dict[str, Any] = {"runs": runs, "summary": {"best_run": best}}
    if cfg.baseline_results_json:
        baseline = json.loads(Path(cfg.baseline_results_json).read_text(encoding="utf-8"))
        baseline_run = baseline.get("summary", {}).get("best_run")
        if baseline_run is not None:
            results["summary"]["baseline_reference"] = baseline_run
            results["summary"]["best_minus_baseline_auroc"] = float(best["metrics"]["auroc"] - baseline_run["metrics"]["auroc"])
            results["summary"]["best_minus_baseline_balanced_accuracy"] = float(
                best["metrics"]["balanced_accuracy"] - baseline_run["metrics"]["balanced_accuracy"]
            )
    return results


def save_spectra_search_results(results: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    path = Path(output_dir) / "spectra_search_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


__all__ = [
    "SpectraSearchConfig",
    "SpectraVariant",
    "default_variants",
    "run_spectra_search",
    "save_spectra_search_results",
]
