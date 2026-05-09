from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..utils.io import ensure_dir

from ..data.io import load_study, marker_matrix_from_cells
from ..evaluation.splits import bootstrap_ci, build_grouped_splits, build_lopo_splits
from ..features.common import infer_role_mask, normalize_coordinates, safe_float


@dataclass(frozen=True)
class NeighborhoodMethod:
    name: str
    builder: Callable[[np.ndarray, np.ndarray, int], np.ndarray]


@dataclass
class NeighborhoodIsolationConfig:
    study_dir: str
    output_dir: str
    random_state: int = 42
    knn_k: int = 10
    jitter_std: float = 0.05
    n_splits: int = 4
    n_repeats: int = 1
    split_mode: str = "grouped"


META_COLUMNS = {"cell_id", "sample_id", "patient_id", "region_id", "x", "y", "cell_type", "compartment"}


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def _topk_mask_from_weights(weights: np.ndarray, k: int) -> np.ndarray:
    n = weights.shape[0]
    k = max(1, min(k, max(1, n - 1)))
    out = np.zeros((n, n), dtype=bool)
    masked = weights.copy()
    np.fill_diagonal(masked, -np.inf)
    for idx in range(n):
        row = masked[idx]
        top = np.argsort(row)[::-1][:k]
        out[idx, top] = np.isfinite(row[top])
    return out


def _adjacency_from_neighbors(indices: np.ndarray) -> np.ndarray:
    n, k = indices.shape
    adj = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    adj[rows, indices.reshape(-1)] = True
    np.fill_diagonal(adj, False)
    return adj


def _knn_adjacency(coords: np.ndarray, _markers: np.ndarray, k: int) -> np.ndarray:
    n = coords.shape[0]
    if n < 2:
        return np.zeros((n, n), dtype=bool)
    n_neighbors = min(max(k + 1, 2), n)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    return _adjacency_from_neighbors(indices[:, 1:])


def _adaptive_knn_adjacency(coords: np.ndarray, _markers: np.ndarray, k: int) -> np.ndarray:
    n = coords.shape[0]
    if n < 2:
        return np.zeros((n, n), dtype=bool)
    dists = _pairwise_distances(coords)
    np.fill_diagonal(dists, np.inf)
    base_k = max(1, min(k, n - 1))
    sorted_d = np.sort(dists, axis=1)[:, :base_k]
    local_scale = np.median(sorted_d, axis=1)
    threshold = 1.5 * np.clip(local_scale, 1e-6, None)
    adj = dists <= threshold[:, None]
    knn = np.argsort(dists, axis=1)[:, :base_k]
    adj[np.arange(n)[:, None], knn] = True
    np.fill_diagonal(adj, False)
    return adj


def _radius_adjacency(coords: np.ndarray, _markers: np.ndarray, k: int) -> np.ndarray:
    n = coords.shape[0]
    if n < 2:
        return np.zeros((n, n), dtype=bool)
    dists = _pairwise_distances(coords)
    np.fill_diagonal(dists, np.inf)
    base_k = max(1, min(k, n - 1))
    sorted_d = np.sort(dists, axis=1)[:, :base_k]
    radius = 1.25 * float(np.median(sorted_d))
    adj = dists <= max(radius, 1e-6)
    np.fill_diagonal(adj, False)
    return adj


def _shared_neighbor_adjacency(coords: np.ndarray, _markers: np.ndarray, k: int) -> np.ndarray:
    base = _knn_adjacency(coords, _markers, k)
    shared = base @ base.T
    threshold = max(1, min(3, k // 2 if k > 1 else 1))
    adj = shared >= threshold
    np.fill_diagonal(adj, False)
    return adj


def _soft_attention_adjacency(coords: np.ndarray, markers: np.ndarray, k: int) -> np.ndarray:
    n = coords.shape[0]
    if n < 2:
        return np.zeros((n, n), dtype=bool)
    dists = _pairwise_distances(coords)
    np.fill_diagonal(dists, 0.0)
    marker_norm = np.linalg.norm(markers, axis=1, keepdims=True)
    marker_norm = np.where(marker_norm <= 1e-6, 1.0, marker_norm)
    normalized_markers = markers / marker_norm
    marker_sim = normalized_markers @ normalized_markers.T if markers.shape[1] > 0 else np.zeros((n, n), dtype=float)
    dist_scale = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
    logits = -((dists / max(dist_scale, 1e-6)) ** 2) + 0.5 * marker_sim
    topk = _topk_mask_from_weights(logits, k)
    return topk


def default_methods() -> list[NeighborhoodMethod]:
    return [
        NeighborhoodMethod("knn", _knn_adjacency),
        NeighborhoodMethod("adaptive_knn", _adaptive_knn_adjacency),
        NeighborhoodMethod("radius", _radius_adjacency),
        NeighborhoodMethod("shared_neighbor", _shared_neighbor_adjacency),
        NeighborhoodMethod("soft_attention", _soft_attention_adjacency),
    ]


def _edge_list(adj: np.ndarray) -> list[tuple[int, int]]:
    sym = adj | adj.T
    rows, cols = np.where(np.triu(sym, k=1))
    return list(zip(rows.tolist(), cols.tolist(), strict=False))


def _neighbor_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    scores = []
    for i in range(left.shape[0]):
        a = set(np.flatnonzero(left[i]))
        b = set(np.flatnonzero(right[i]))
        if not a and not b:
            scores.append(1.0)
            continue
        union = a | b
        scores.append(len(a & b) / max(len(union), 1))
    return float(np.mean(scores)) if scores else 0.0


def _boundary_mask(coords: np.ndarray, cell_types: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    if n < 2:
        return np.zeros(n, dtype=bool)
    dists = _pairwise_distances(coords)
    np.fill_diagonal(dists, np.inf)
    nearest = np.argsort(dists, axis=1)[:, : min(3, n - 1)]
    neighbors_types = cell_types[nearest]
    return np.any(neighbors_types != cell_types[:, None], axis=1)


def _boundary_capture(adj: np.ndarray, boundary: np.ndarray, cell_types: np.ndarray) -> float:
    if boundary.sum() == 0:
        return 0.0
    captured = []
    for idx in np.flatnonzero(boundary):
        nbrs = np.flatnonzero(adj[idx] | adj[:, idx])
        if nbrs.size == 0:
            captured.append(0.0)
            continue
        captured.append(float(np.any(cell_types[nbrs] != cell_types[idx])))
    return float(np.mean(captured)) if captured else 0.0


def _rare_type_coverage(adj: np.ndarray, cell_types: np.ndarray) -> float:
    counts = Counter(cell_types.tolist())
    rare = {cell_type for cell_type, count in counts.items() if count <= max(3, int(0.1 * len(cell_types)))}
    if not rare:
        return 0.0
    covered = []
    for idx, cell_type in enumerate(cell_types):
        if cell_type not in rare:
            continue
        nbrs = np.flatnonzero(adj[idx] | adj[:, idx])
        covered.append(float(nbrs.size > 0))
    return float(np.mean(covered)) if covered else 0.0


def _graph_probe_features(region_meta: dict[str, object], adj: np.ndarray, coords: np.ndarray, cell_types: np.ndarray) -> dict[str, object]:
    features: dict[str, object] = {
        "region_id": region_meta["region_id"],
        "sample_id": region_meta["sample_id"],
        "patient_id": region_meta["patient_id"],
        "label": region_meta["label"],
    }
    edges = _edge_list(adj)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cell_types)))
    graph.add_edges_from(edges)
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    features["edge_count"] = float(len(edges))
    features["degree_mean"] = safe_float(degrees.mean()) if degrees.size else 0.0
    features["degree_std"] = safe_float(degrees.std()) if degrees.size else 0.0
    features["components"] = float(nx.number_connected_components(graph)) if graph.number_of_nodes() else 0.0
    features["clustering"] = safe_float(nx.average_clustering(graph)) if graph.number_of_nodes() > 1 else 0.0
    labels = {idx: str(cell_type) for idx, cell_type in enumerate(cell_types)}
    nx.set_node_attributes(graph, labels, "cell_type")
    features["assortativity"] = safe_float(nx.attribute_assortativity_coefficient(graph, "cell_type")) if graph.number_of_edges() else 0.0
    edge_pairs = []
    same_type = 0
    cross_type = 0
    tumor_immune = 0
    tumor_mask = infer_role_mask(pd.Series(cell_types), ["tumor", "epithelial"])
    immune_mask = infer_role_mask(pd.Series(cell_types), ["cd4", "cd8", "treg", "bcell", "macrophage", "neutrophil", "immune"])
    for a, b in edges:
        left = str(cell_types[a]).lower().replace(" ", "_")
        right = str(cell_types[b]).lower().replace(" ", "_")
        edge_pairs.append(tuple(sorted((left, right))))
        if cell_types[a] == cell_types[b]:
            same_type += 1
        else:
            cross_type += 1
        if (tumor_mask[a] and immune_mask[b]) or (tumor_mask[b] and immune_mask[a]):
            tumor_immune += 1
    total_edges = max(len(edges), 1)
    features["same_type_edge_rate"] = float(same_type / total_edges)
    features["cross_type_edge_rate"] = float(cross_type / total_edges)
    features["tumor_immune_edge_rate"] = float(tumor_immune / total_edges)
    for pair, count in Counter(edge_pairs).items():
        features[f"edge_pair_{pair[0]}__{pair[1]}"] = float(count / total_edges)
    boundary = _boundary_mask(coords, cell_types)
    features["boundary_capture"] = _boundary_capture(adj, boundary, cell_types)
    features["rare_type_coverage"] = _rare_type_coverage(adj, cell_types)
    return features


def evaluate_methods(study_dir: str, cfg: NeighborhoodIsolationConfig, methods: list[NeighborhoodMethod] | None = None) -> dict[str, object]:
    if methods is None:
        methods = default_methods()
    study = load_study(study_dir)
    rng = np.random.default_rng(cfg.random_state)

    intrinsic_rows: list[dict[str, object]] = []
    probe_frames: dict[str, list[dict[str, object]]] = {method.name: [] for method in methods}

    for region_id in sorted(study.samples):
        adata = study.samples[region_id]
        obs = adata.obs.reset_index(drop=True)
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        norm_coords, scale = normalize_coordinates(coords)
        markers, _ = marker_matrix_from_cells(obs, META_COLUMNS)
        cell_types = obs["cell_type"].astype(str).to_numpy()
        meta = dict(adata.uns.get("sample_meta", {}))

        for method in methods:
            adj = method.builder(norm_coords, markers, cfg.knn_k)
            jittered = norm_coords + rng.normal(0.0, cfg.jitter_std, size=norm_coords.shape)
            adj_jitter = method.builder(jittered, markers, cfg.knn_k)
            edges = _edge_list(adj)
            boundary = _boundary_mask(norm_coords, cell_types)
            intrinsic_rows.append(
                {
                    "region_id": region_id,
                    "patient_id": meta.get("patient_id"),
                    "label": meta.get("label"),
                    "method": method.name,
                    "cell_count": int(len(cell_types)),
                    "median_scale": float(scale),
                    "edge_count": float(len(edges)),
                    "mean_out_degree": float(adj.sum(axis=1).mean()) if adj.size else 0.0,
                    "neighbor_stability_jitter": _neighbor_jaccard(adj, adj_jitter),
                    "boundary_capture": _boundary_capture(adj, boundary, cell_types),
                    "rare_type_coverage": _rare_type_coverage(adj, cell_types),
                }
            )
            probe_frames[method.name].append(_graph_probe_features(meta, adj, norm_coords, cell_types))

    downstream_runs: list[dict[str, object]] = []
    for method_name, rows in probe_frames.items():
        frame = pd.DataFrame(rows).fillna(0.0)
        labels = frame["label"].to_numpy()
        groups = frame["patient_id"].to_numpy()
        feature_frame = frame.drop(columns=["region_id", "sample_id", "patient_id", "label"])
        X = feature_frame.to_numpy(dtype=float)
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)
        splits = build_grouped_splits(y, groups, n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.random_state)
        model = Pipeline(
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
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        )
        fold_metrics = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if np.unique(y[train_idx]).shape[0] < 2 or np.unique(y[test_idx]).shape[0] < 2:
                continue
            fitted = model.fit(X[train_idx], y[train_idx])
            probs = fitted.predict_proba(X[test_idx])[:, 1]
            pred = fitted.predict(X[test_idx])
            try:
                auroc = float(roc_auc_score(y[test_idx], probs))
            except ValueError:
                auroc = 0.5
            fold_metrics.append(
                {
                    "fold_index": fold_idx,
                    "auroc": auroc,
                    "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                    "macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
                    "brier_score": float(brier_score_loss(y[test_idx], probs)),
                }
            )
        if not fold_metrics:
            continue
        downstream_runs.append(
            {
                "method": method_name,
                "metrics": {
                    key: float(np.mean([row[key] for row in fold_metrics]))
                    for key in ("auroc", "balanced_accuracy", "macro_f1", "brier_score")
                },
                "fold_metrics": fold_metrics,
            }
        )

    summary = {
        "best_intrinsic_stability": max(intrinsic_rows, key=lambda row: row["neighbor_stability_jitter"])["method"] if intrinsic_rows else None,
        "best_intrinsic_boundary_capture": max(intrinsic_rows, key=lambda row: row["boundary_capture"])["method"] if intrinsic_rows else None,
        "best_downstream": max(downstream_runs, key=lambda row: row["metrics"]["balanced_accuracy"])["method"] if downstream_runs else None,
    }
    return {"intrinsic_rows": intrinsic_rows, "downstream_runs": downstream_runs, "summary": summary}


def save_results(results: dict[str, object], output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    path = Path(output_dir) / "neighborhood_isolation_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


__all__ = ["NeighborhoodIsolationConfig", "NeighborhoodMethod", "default_methods", "evaluate_methods", "save_results"]
