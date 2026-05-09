from __future__ import annotations

import copy
import json
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
class LiftIsolationConfig:
    study_dir: str
    output_dir: str
    random_state: int = 42
    n_splits: int = 4
    n_repeats: int = 1
    split_mode: str = "grouped"
    batch_size: int = 12
    epochs: int = 12
    patience: int = 3
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    knn_k: int = 6
    neighborhood_mode: str = "adaptive_knn"
    lift_dim: int = 12
    hidden_dim: int = 32
    max_cells: int = 160
    dropout: float = 0.1


@dataclass(frozen=True)
class LiftVariant:
    name: str
    mode: str
    drop_hist: bool = False
    drop_density: bool = False
    drop_coords: bool = False


def default_variants() -> list[LiftVariant]:
    return [
        LiftVariant(name="identity_full", mode="identity"),
        LiftVariant(name="linear_full", mode="linear"),
        LiftVariant(name="mlp_full", mode="mlp"),
        LiftVariant(name="linear_no_hist", mode="linear", drop_hist=True),
        LiftVariant(name="linear_no_density", mode="linear", drop_density=True),
        LiftVariant(name="linear_no_coords", mode="linear", drop_coords=True),
    ]


class LiftDataset(Dataset):
    def __init__(self, examples: list[SpatialRegionExample], labels: np.ndarray):
        self.examples = examples
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[SpatialRegionExample, int]:
        return self.examples[idx], int(self.labels[idx])


def _collate(batch: list[tuple[SpatialRegionExample, int]]) -> dict[str, torch.Tensor]:
    examples, labels = zip(*batch, strict=False)
    max_cells = max(ex.structural.shape[0] for ex in examples)
    structural_dim = examples[0].structural.shape[1]
    batch_size = len(examples)
    structural = torch.zeros(batch_size, max_cells, structural_dim, dtype=torch.float32)
    type_ids = torch.full((batch_size, max_cells), -1, dtype=torch.long)
    mask = torch.zeros(batch_size, max_cells, dtype=torch.bool)
    for batch_idx, ex in enumerate(examples):
        n_cells = ex.structural.shape[0]
        structural[batch_idx, :n_cells] = torch.from_numpy(ex.structural)
        type_ids[batch_idx, :n_cells] = torch.from_numpy(ex.type_ids)
        mask[batch_idx, :n_cells] = True
    return {
        "structural": structural,
        "type_ids": type_ids,
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
    return {
        "coords": slice(0, 2),
        "radial": slice(2, 3),
        "mean_nn": slice(3, 4),
        "density": slice(4, 5),
        "onehot": slice(5, 5 + num_types),
        "hist": slice(5 + num_types, 5 + 2 * num_types),
    }


def _apply_variant(structural: np.ndarray, variant: LiftVariant) -> np.ndarray:
    out = structural.copy()
    layout = _structural_layout(out.shape[1])
    if variant.drop_coords:
        out[:, layout["coords"]] = 0.0
        out[:, layout["radial"]] = 0.0
    if variant.drop_density:
        out[:, layout["mean_nn"]] = 0.0
        out[:, layout["density"]] = 0.0
    if variant.drop_hist:
        out[:, layout["hist"]] = 0.0
    return out


class LiftProbe(nn.Module):
    def __init__(self, in_dim: int, lift_dim: int, hidden_dim: int, *, mode: str, dropout: float) -> None:
        super().__init__()
        self.mode = mode
        if mode == "identity":
            self.lift = nn.Identity()
            out_dim = in_dim
        elif mode == "linear":
            self.lift = nn.Linear(in_dim, lift_dim)
            out_dim = lift_dim
        elif mode == "mlp":
            self.lift = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, lift_dim),
            )
            out_dim = lift_dim
        else:
            raise ValueError(f"Unsupported lift mode: {mode}")
        self.head = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def encode(self, structural: torch.Tensor) -> torch.Tensor:
        return self.lift(structural)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        lifted = self.encode(batch["structural"])
        pooled_mean = self._masked_mean(lifted, batch["mask"])
        centered = lifted - pooled_mean.unsqueeze(1)
        pooled_std = torch.sqrt(self._masked_mean(centered * centered, batch["mask"]).clamp_min(1e-6))
        logits = self.head(torch.cat([pooled_mean, pooled_std], dim=-1))
        return {"logits": logits, "lifted": lifted}


def _choose_validation_indices(train_idx: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups[train_idx])
    if unique_groups.shape[0] < 4:
        return train_idx, np.array([], dtype=int)
    splits = build_grouped_splits(labels[train_idx], groups[train_idx], n_splits=4, n_repeats=1, random_state=seed)
    if not splits:
        return train_idx, np.array([], dtype=int)
    inner_train, inner_val = splits[0]
    return train_idx[inner_train], train_idx[inner_val]


def _threshold_search(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_tau = 0.5
    best_score = -np.inf
    for tau in np.linspace(0.2, 0.8, num=25):
        y_pred = (y_prob >= tau).astype(np.int64)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau


def _loader(examples: list[SpatialRegionExample], labels: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(LiftDataset(examples, labels), batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)


def _boundary_labels(example: SpatialRegionExample) -> np.ndarray:
    dim = example.structural.shape[1]
    layout = _structural_layout(dim)
    hist = example.structural[:, layout["hist"]]
    purity = hist[np.arange(hist.shape[0]), example.type_ids]
    return (purity < 0.6).astype(np.int64)


def _collect_cell_embeddings(
    model: LiftProbe,
    examples: list[SpatialRegionExample],
    variant: LiftVariant,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeds: list[np.ndarray] = []
    cell_types: list[np.ndarray] = []
    boundary: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            structural = _apply_variant(example.structural, variant)
            tensor = torch.from_numpy(structural).float().unsqueeze(0).to(device)
            lifted = model.encode(tensor).squeeze(0).cpu().numpy()
            embeds.append(lifted)
            cell_types.append(example.type_ids)
            boundary.append(_boundary_labels(example))
    return np.concatenate(embeds, axis=0), np.concatenate(cell_types, axis=0), np.concatenate(boundary, axis=0)


def _cell_probe_metrics(
    train_embed: np.ndarray,
    train_types: np.ndarray,
    train_boundary: np.ndarray,
    test_embed: np.ndarray,
    test_types: np.ndarray,
    test_boundary: np.ndarray,
    seed: int,
) -> dict[str, float]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embed)
    test_scaled = scaler.transform(test_embed)
    if np.unique(train_types).shape[0] >= 2 and np.unique(test_types).shape[0] >= 2:
        type_probe = LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            random_state=seed,
        )
        type_probe.fit(train_scaled, train_types)
        type_pred = type_probe.predict(test_scaled)
        type_macro_f1 = float(f1_score(test_types, type_pred, average="macro"))
    else:
        type_macro_f1 = 0.0
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
    sing = np.linalg.svd(test_scaled, compute_uv=False)
    stable_rank = float((np.square(sing).sum() / np.square(sing).max()) / max(1, test_scaled.shape[1])) if sing.size else 0.0
    return {
        "cell_type_macro_f1": type_macro_f1,
        "boundary_auroc": boundary_auroc,
        "stable_rank_ratio": stable_rank,
    }


def _evaluate_probs(model: LiftProbe, loader: DataLoader, device: torch.device, threshold: float) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
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


def _transform_examples(examples: list[SpatialRegionExample], variant: LiftVariant) -> list[SpatialRegionExample]:
    transformed: list[SpatialRegionExample] = []
    for example in examples:
        updated = copy.deepcopy(example)
        updated.structural = _apply_variant(example.structural, variant)
        transformed.append(updated)
    return transformed


def _train_variant_fold(
    cfg: LiftIsolationConfig,
    variant: LiftVariant,
    examples: list[SpatialRegionExample],
    labels: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    fit_idx, val_idx = _choose_validation_indices(train_idx, labels, groups, seed)
    train_examples = _transform_examples([examples[idx] for idx in fit_idx], variant)
    val_examples = _transform_examples([examples[idx] for idx in val_idx], variant)
    test_examples = _transform_examples([examples[idx] for idx in test_idx], variant)
    train_labels = labels[fit_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    model = LiftProbe(
        in_dim=train_examples[0].structural.shape[1],
        lift_dim=cfg.lift_dim,
        hidden_dim=cfg.hidden_dim,
        mode=variant.mode,
        dropout=cfg.dropout,
    ).to(device)
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = class_counts.sum() / np.clip(class_counts, 1, None)
    loss_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
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
        val_metrics, val_true, val_prob = _evaluate_probs(model, val_loader, device, threshold=0.5)
        threshold = _threshold_search(val_true, val_prob)
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
    region_metrics, _, _ = _evaluate_probs(model, test_loader, device, threshold=best_threshold)
    train_embed, train_types, train_boundary = _collect_cell_embeddings(model, train_examples, variant, device)
    test_embed, test_types, test_boundary = _collect_cell_embeddings(model, test_examples, variant, device)
    intrinsic = _cell_probe_metrics(
        train_embed,
        train_types,
        train_boundary,
        test_embed,
        test_types,
        test_boundary,
        seed,
    )
    region_metrics.update(intrinsic)
    return region_metrics


def run_lift_isolation(cfg: LiftIsolationConfig, variants: list[LiftVariant] | None = None) -> dict[str, Any]:
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    if variants is None:
        variants = default_variants()
    examples, _type_vocab = build_region_examples(
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
                seed=cfg.random_state + fold_idx,
                device=device,
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
                "stable_rank_ratio",
            )
        }
        runs.append(
            {
                "variant": asdict(variant),
                "metrics": aggregate,
                "fold_metrics": fold_metrics,
            }
        )

    if not runs:
        raise RuntimeError("Lift isolation produced no valid runs.")

    best_downstream = max(runs, key=lambda row: (row["metrics"]["balanced_accuracy"], row["metrics"]["auroc"]))
    best_intrinsic = max(runs, key=lambda row: (row["metrics"]["boundary_auroc"], row["metrics"]["cell_type_macro_f1"]))
    return {
        "runs": runs,
        "summary": {
            "best_downstream": best_downstream,
            "best_intrinsic": best_intrinsic,
            "label_classes": encoder.classes_.tolist(),
        },
    }


def save_lift_isolation_results(results: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    path = Path(output_dir) / "lift_isolation_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


__all__ = [
    "LiftIsolationConfig",
    "LiftVariant",
    "default_variants",
    "run_lift_isolation",
    "save_lift_isolation_results",
]
