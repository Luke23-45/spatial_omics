from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from ..utils.io import ensure_dir

from ..evaluation.splits import bootstrap_ci, build_grouped_splits, build_lopo_splits
from .end_to_end_spectra import (
    EndToEndSpectraModel,
    SpectraDataset,
    SpectraSearchConfig,
    SpectraVariant,
    _collate,
)
from .spatial_z4 import SpatialRegionExample, build_region_examples


@dataclass(frozen=True)
class SpectraPureTrial:
    learning_rate: float
    weight_decay: float
    focal_gamma: float
    label_smoothing: float
    temperature: float
    threshold_grid_size: int

    @property
    def name(self) -> str:
        return (
            f"lr={self.learning_rate:g}|wd={self.weight_decay:g}|fg={self.focal_gamma:g}|"
            f"ls={self.label_smoothing:g}|temp={self.temperature:g}|grid={self.threshold_grid_size}"
        )


@dataclass
class SpectraPureConfig:
    study_dir: str
    output_dir: str
    baseline_results_json: str | None = None
    random_state: int = 42
    n_splits: int = 4
    n_repeats: int = 1
    split_mode: str = "grouped"
    batch_size: int = 10
    epochs: int = 10
    patience: int = 3
    knn_k: int = 6
    neighborhood_mode: str = "adaptive_knn"
    model_dim: int = 56
    anchor_count: int = 8
    router_steps: int = 3
    type_embedding_dim: int = 16
    max_cells: int = 128
    dropout: float = 0.15


def default_trials() -> list[SpectraPureTrial]:
    return [
        SpectraPureTrial(2e-3, 1e-4, 0.0, 0.03, 1.0, 25),
        SpectraPureTrial(2e-3, 1e-4, 1.0, 0.0, 1.0, 41),
        SpectraPureTrial(1e-3, 1e-4, 1.5, 0.0, 0.75, 41),
        SpectraPureTrial(1e-3, 5e-5, 2.0, 0.0, 0.75, 61),
        SpectraPureTrial(1e-3, 5e-5, 2.0, 0.0, 0.5, 61),
        SpectraPureTrial(8e-4, 5e-5, 2.5, 0.0, 0.5, 81),
    ]


PURE_VARIANT = SpectraVariant(positional_mode="raw", neighborhood_mode="soft", spectral_mode="off")


def _build_loader(examples: list[SpatialRegionExample], labels: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(SpectraDataset(examples, labels), batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)


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


def _focal_loss(logits: torch.Tensor, labels: torch.Tensor, class_weight: torch.Tensor, gamma: float, label_smoothing: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, weight=class_weight, reduction="none", label_smoothing=label_smoothing)
    if gamma <= 0:
        return ce.mean()
    probs = torch.softmax(logits, dim=-1)
    pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
    focal = ((1.0 - pt) ** gamma) * ce
    return focal.mean()


def _threshold_candidates(grid_size: int) -> np.ndarray:
    return np.linspace(0.05, 0.95, num=max(11, grid_size))


def _select_threshold(y_true: np.ndarray, logits: np.ndarray, temperature: float, grid_size: int) -> tuple[float, np.ndarray]:
    scaled = logits / max(temperature, 1e-6)
    probs = 1.0 / (1.0 + np.exp(-scaled))
    best_tau = 0.5
    best_score = -np.inf
    for tau in _threshold_candidates(grid_size):
        y_pred = (probs >= tau).astype(np.int64)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau, probs


def _evaluate_logits(model: EndToEndSpectraModel, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch)["logits"][:, 1] - model(batch)["logits"][:, 0]
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
    return np.concatenate(all_labels, axis=0), np.concatenate(all_logits, axis=0)


def _inner_split(train_idx: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if np.unique(groups[train_idx]).shape[0] < 4:
        return train_idx, np.array([], dtype=int)
    splits = build_grouped_splits(labels[train_idx], groups[train_idx], n_splits=4, n_repeats=1, random_state=seed)
    if not splits:
        return train_idx, np.array([], dtype=int)
    fit, val = splits[0]
    return train_idx[fit], train_idx[val]


def _build_model(cfg: SpectraPureConfig, examples: list[SpatialRegionExample], num_types: int) -> EndToEndSpectraModel:
    return EndToEndSpectraModel(
        marker_dim=examples[0].markers.shape[1],
        num_types=num_types,
        variant=PURE_VARIANT,
        model_dim=cfg.model_dim,
        anchor_count=cfg.anchor_count,
        router_steps=cfg.router_steps,
        type_embedding_dim=cfg.type_embedding_dim,
        spectral_components=1,
        dropout=cfg.dropout,
    )


def _train_trial_fold(
    cfg: SpectraPureConfig,
    trial: SpectraPureTrial,
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

    model = _build_model(cfg, train_examples, num_types).to(device)
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = class_counts.sum() / np.clip(class_counts, 1, None)
    class_weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trial.learning_rate, weight_decay=trial.weight_decay)

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
            loss = _focal_loss(out["logits"], batch["labels"], class_weight, trial.focal_gamma, trial.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        if val_loader is None or len(val_examples) == 0 or np.unique(val_labels).shape[0] < 2:
            best_state = copy.deepcopy(model.state_dict())
            continue
        val_true, val_logits = _evaluate_logits(model, val_loader, device)
        threshold, val_prob = _select_threshold(val_true, val_logits, trial.temperature, trial.threshold_grid_size)
        val_pred = (val_prob >= threshold).astype(np.int64)
        val_metrics = _score_binary(val_true, val_pred, val_prob)
        composite = 0.55 * val_metrics["balanced_accuracy"] + 0.25 * val_metrics["macro_f1"] + 0.20 * val_metrics["auroc"]
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
    test_true, test_logits = _evaluate_logits(model, test_loader, device)
    scaled = test_logits / max(trial.temperature, 1e-6)
    test_prob = 1.0 / (1.0 + np.exp(-scaled))
    test_pred = (test_prob >= best_threshold).astype(np.int64)
    metrics = _score_binary(test_true, test_pred, test_prob)
    metrics["decision_threshold"] = float(best_threshold)
    metrics["temperature"] = float(trial.temperature)
    return metrics


def run_spectra_pure_tuning(cfg: SpectraPureConfig, trials: list[SpectraPureTrial] | None = None) -> dict[str, Any]:
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    if trials is None:
        trials = default_trials()
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
    for trial in trials:
        fold_metrics: list[dict[str, float]] = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if np.unique(labels[train_idx]).shape[0] < 2 or np.unique(labels[test_idx]).shape[0] < 2:
                continue
            metrics = _train_trial_fold(
                cfg,
                trial,
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
            for key in ("auroc", "balanced_accuracy", "macro_f1", "brier_score", "decision_threshold", "temperature")
        }
        aggregate["selection_score"] = float(0.55 * aggregate["balanced_accuracy"] + 0.25 * aggregate["macro_f1"] + 0.20 * aggregate["auroc"])
        runs.append(
            {
                "trial": trial.name,
                "trial_config": asdict(trial),
                "feature_set": "Spectra-Pure",
                "model_name": "spectra_pure",
                "label_classes": encoder.classes_.tolist(),
                "metrics": aggregate,
                "fold_metrics": fold_metrics,
            }
        )

    if not runs:
        raise RuntimeError("Spectra-Pure tuning produced no valid runs.")
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


def save_spectra_pure_results(results: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    path = Path(output_dir) / "spectra_pure_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


__all__ = [
    "SpectraPureConfig",
    "SpectraPureTrial",
    "default_trials",
    "run_spectra_pure_tuning",
    "save_spectra_pure_results",
]
