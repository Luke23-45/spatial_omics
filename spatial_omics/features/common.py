from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors


META_COLUMNS = ["region_id", "sample_id", "patient_id", "label", "split", "feature_family"]


def infer_role_mask(cell_types: pd.Series, terms: list[str]) -> np.ndarray:
    lowered = cell_types.fillna("").astype(str).str.lower()
    return lowered.apply(lambda value: any(term in value for term in terms)).to_numpy(dtype=bool)


def normalize_coordinates(coords: np.ndarray) -> tuple[np.ndarray, float]:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] < 2:
        return coords.astype(np.float32), 1.0
    k = min(6, coords.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    local_scale = np.median(distances[:, 1:]) if distances.shape[1] > 1 else 1.0
    local_scale = float(local_scale) if np.isfinite(local_scale) and local_scale > 0 else 1.0
    return (coords / local_scale).astype(np.float32), local_scale


def mutual_knn_edges(coords: np.ndarray, k: int) -> list[tuple[int, int]]:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] < 2:
        return []
    n_neighbors = min(max(k + 1, 2), coords.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    neighbor_sets = [set(row[1:]) for row in indices]
    edges: set[tuple[int, int]] = set()
    for i, neighbors in enumerate(neighbor_sets):
        for j in neighbors:
            if i in neighbor_sets[j]:
                edges.add((i, j) if i < j else (j, i))
    return sorted(edges)


def delaunay_edges(coords: np.ndarray) -> list[tuple[int, int]]:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] < 3:
        return mutual_knn_edges(coords, k=2)
    try:
        tri = Delaunay(coords)
    except Exception:
        return mutual_knn_edges(coords, k=3)
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        simplex = list(simplex)
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                a, b = simplex[i], simplex[j]
                edges.add((a, b) if a < b else (b, a))
    return sorted(edges)


def safe_float(value: float | int | np.floating | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def mixing_entropy(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    counts = Counter(pairs)
    total = float(sum(counts.values()))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12)
    return entropy


def metadata_row(adata, feature_family: str) -> dict[str, object]:
    meta = dict(adata.uns.get("sample_meta", {}))
    return {
        "region_id": meta.get("region_id"),
        "sample_id": meta.get("sample_id"),
        "patient_id": meta.get("patient_id"),
        "label": meta.get("label"),
        "split": "unassigned",
        "feature_family": feature_family,
    }


def patch_indices(coords: np.ndarray, patch_size: float, min_cells: int) -> list[np.ndarray]:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return []
    mins = coords.min(axis=0)
    normalized = coords - mins
    bins_x = np.floor(normalized[:, 0] / max(patch_size, 1e-6)).astype(int)
    bins_y = np.floor(normalized[:, 1] / max(patch_size, 1e-6)).astype(int)
    groups: dict[tuple[int, int], list[int]] = {}
    for idx, key in enumerate(zip(bins_x, bins_y)):
        groups.setdefault(key, []).append(idx)
    return [np.asarray(idxs, dtype=int) for idxs in groups.values() if len(idxs) >= min_cells]


def aggregate_patch_features(rows: list[dict[str, float]], prefix: str) -> dict[str, float]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    out: dict[str, float] = {}
    for col in frame.columns:
        values = frame[col].to_numpy(dtype=float)
        out[f"{prefix}_{col}_mean"] = safe_float(np.mean(values))
        out[f"{prefix}_{col}_std"] = safe_float(np.std(values))
        out[f"{prefix}_{col}_q25"] = safe_float(np.quantile(values, 0.25))
        out[f"{prefix}_{col}_q75"] = safe_float(np.quantile(values, 0.75))
    return out

