from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from spartial_omics.utils.io import ensure_dir

from spartial_omics.evaluation.splits import bootstrap_ci, build_grouped_splits, build_lopo_splits

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


NUMERIC_META = {"region_id", "sample_id", "patient_id", "label", "split", "feature_family"}


@dataclass
class FeatureSet:
    name: str
    frame: pd.DataFrame


def _prepare_feature_sets(feature_table: pd.DataFrame) -> list[FeatureSet]:
    families = {}
    for family in sorted(feature_table["feature_family"].unique()):
        families[family] = (
            feature_table.loc[feature_table["feature_family"] == family]
            .set_index("region_id")
            .drop(columns=["feature_family", "split"])
        )
    out: list[FeatureSet] = []
    if "F0" in families:
        out.append(FeatureSet("F0", families["F0"]))
    if "F1" in families:
        out.append(FeatureSet("F1", families["F1"]))
    if "F0" in families and "F1" in families:
        merged = families["F0"].join(
            families["F1"].drop(columns=["sample_id", "patient_id", "label"]),
            how="inner",
            rsuffix="_f1",
        )
        out.append(FeatureSet("F0+F1", merged))
    return out


def _make_models(random_state: int) -> dict[str, Any]:
    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
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
    forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                ),
            ),
        ]
    )
    hist = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", HistGradientBoostingClassifier(random_state=random_state)),
        ]
    )
    return {"logistic_elasticnet": logistic, "random_forest": forest, "hist_gradient_boosting": hist}


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


def _feature_importances(model: Pipeline, feature_names: list[str], top_k: int) -> list[dict[str, float]]:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        scores = np.abs(estimator.coef_[0])
    elif hasattr(estimator, "feature_importances_"):
        scores = np.abs(estimator.feature_importances_)
    else:
        return []
    order = np.argsort(scores)[::-1][:top_k]
    return [{"feature": feature_names[idx], "score": float(scores[idx])} for idx in order if scores[idx] > 0]


def _feature_stability(top_feature_sets: list[list[str]]) -> float:
    if len(top_feature_sets) < 2:
        return 0.0
    pair_scores = []
    for i in range(len(top_feature_sets)):
        for j in range(i + 1, len(top_feature_sets)):
            left = set(top_feature_sets[i])
            right = set(top_feature_sets[j])
            union = left | right
            pair_scores.append(len(left & right) / max(len(union), 1))
    return float(np.mean(pair_scores)) if pair_scores else 0.0


def run_feasibility_study(feature_table: pd.DataFrame, cfg) -> dict[str, Any]:
    results: dict[str, Any] = {"runs": [], "summary": {}}
    feature_sets = _prepare_feature_sets(feature_table)
    models = _make_models(cfg.random_state)
    selected_model_names = list(getattr(cfg, "model_names", []) or models.keys())

    for feature_set in feature_sets:
        frame = feature_set.frame.copy()
        meta_cols = ["sample_id", "patient_id", "label"]
        labels = frame["label"].to_numpy()
        groups = frame["patient_id"].to_numpy()
        feature_frame = frame.drop(columns=meta_cols)
        numeric_columns = list(feature_frame.columns)
        X = feature_frame.fillna(0.0).to_numpy(dtype=float)
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)
        split_mode = getattr(cfg, "split_mode", "grouped")
        if split_mode == "lopo":
            splits = build_lopo_splits(y, groups, stratified=True, random_state=cfg.random_state)
        else:
            splits = build_grouped_splits(
                y,
                groups,
                n_splits=cfg.n_splits,
                n_repeats=cfg.n_repeats,
                random_state=cfg.random_state,
            )

        for model_name in selected_model_names:
            if model_name not in models:
                continue
            model = models[model_name]
            fold_metrics = []
            top_features_per_fold: list[list[str]] = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                if np.unique(y[train_idx]).shape[0] < 2 or np.unique(y[test_idx]).shape[0] < 2:
                    continue
                try:
                    fitted = model.fit(X[train_idx], y[train_idx])
                except PermissionError:
                    if model_name != "hist_gradient_boosting":
                        raise
                    fallback = Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                            ("model", GradientBoostingClassifier(random_state=cfg.random_state)),
                        ]
                    )
                    fitted = fallback.fit(X[train_idx], y[train_idx])
                y_prob = fitted.predict_proba(X[test_idx])[:, 1]
                y_pred = fitted.predict(X[test_idx])
                metrics = _score_binary(y[test_idx], y_pred, y_prob)
                metrics["fold_index"] = fold_idx
                fold_metrics.append(metrics)
                top_features_per_fold.append(
                    [row["feature"] for row in _feature_importances(fitted, numeric_columns, cfg.top_feature_k)]
                )
            if not fold_metrics:
                continue
            try:
                final_model = model.fit(X, y)
            except PermissionError:
                final_model = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("model", GradientBoostingClassifier(random_state=cfg.random_state)),
                    ]
                ).fit(X, y)
            aggregate = {
                key: float(np.mean([row[key] for row in fold_metrics]))
                for key in ("auroc", "balanced_accuracy", "macro_f1", "brier_score")
            }
            aggregate["feature_stability"] = _feature_stability(top_features_per_fold)
            aggregate["std"] = {
                key: float(np.std([row[key] for row in fold_metrics]))
                for key in ("auroc", "balanced_accuracy", "macro_f1", "brier_score")
            }
            aggregate["ci_95"] = {
                key: dict(zip(("mean", "low", "high"), bootstrap_ci([row[key] for row in fold_metrics])))
                for key in ("auroc", "balanced_accuracy", "macro_f1", "brier_score")
            }
            run = {
                "feature_set": feature_set.name,
                "model_name": model_name,
                "label_classes": encoder.classes_.tolist(),
                "metrics": aggregate,
                "fold_metrics": fold_metrics,
                "top_features": _feature_importances(final_model, numeric_columns, cfg.top_feature_k),
            }
            results["runs"].append(run)

    if not results["runs"]:
        raise RuntimeError("Feasibility study produced no valid runs; check grouped split configuration and label balance.")
    best = max(results["runs"], key=lambda row: row["metrics"]["auroc"])
    baseline = next((row for row in results["runs"] if row["feature_set"] == "F0" and row["model_name"] == "logistic_elasticnet"), None)
    combined = next((row for row in results["runs"] if row["feature_set"] == "F0+F1" and row["model_name"] == "logistic_elasticnet"), None)
    results["summary"] = {
        "best_run": best,
        "f0_logistic": baseline,
        "f0f1_logistic": combined,
        "topology_added_value_auroc": (
            float(combined["metrics"]["auroc"] - baseline["metrics"]["auroc"])
            if baseline is not None and combined is not None
            else None
        ),
    }
    return results


def save_results(results: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    path = Path(output_dir) / "feasibility_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


__all__ = ["run_feasibility_study", "save_results"]
