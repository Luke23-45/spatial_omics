from __future__ import annotations

import numpy as np
import pandas as pd
from ripser import ripser

from spartial_omics.features.common import (
    aggregate_patch_features,
    infer_role_mask,
    metadata_row,
    normalize_coordinates,
    patch_indices,
    safe_float,
)


def _lifetimes(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.zeros(0, dtype=float)
    finite = diagram[np.isfinite(diagram[:, 1])]
    if finite.size == 0:
        return np.zeros(0, dtype=float)
    life = finite[:, 1] - finite[:, 0]
    return life[life > 0]


def _persistence_entropy(lifetimes: np.ndarray) -> float:
    if lifetimes.size == 0:
        return 0.0
    total = lifetimes.sum()
    if total <= 0:
        return 0.0
    probs = lifetimes / total
    return safe_float(-(probs * np.log(probs + 1e-12)).sum())


def _betti_curve_stats(diagram: np.ndarray, grid_size: int) -> dict[str, float]:
    if diagram.size == 0:
        return {"betti_area": 0.0, "betti_peak": 0.0, "betti_mean": 0.0}
    finite = diagram[np.isfinite(diagram[:, 1])]
    if finite.size == 0:
        return {"betti_area": 0.0, "betti_peak": 0.0, "betti_mean": 0.0}
    start = float(np.min(finite[:, 0]))
    end = float(np.max(finite[:, 1]))
    if end <= start:
        return {"betti_area": 0.0, "betti_peak": 0.0, "betti_mean": 0.0}
    grid = np.linspace(start, end, grid_size)
    curve = np.zeros_like(grid)
    for birth, death in finite:
        curve += ((grid >= birth) & (grid <= death)).astype(float)
    return {
        "betti_area": safe_float(np.trapezoid(curve, grid)),
        "betti_peak": safe_float(np.max(curve)),
        "betti_mean": safe_float(np.mean(curve)),
    }


def _silhouette_stats(diagram: np.ndarray, grid_size: int) -> dict[str, float]:
    lifetimes = _lifetimes(diagram)
    if lifetimes.size == 0:
        return {"silhouette_mean": 0.0, "silhouette_peak": 0.0}
    finite = diagram[np.isfinite(diagram[:, 1])]
    start = float(np.min(finite[:, 0]))
    end = float(np.max(finite[:, 1]))
    if end <= start:
        return {"silhouette_mean": 0.0, "silhouette_peak": 0.0}
    grid = np.linspace(start, end, grid_size)
    curve = np.zeros_like(grid)
    weight_total = max(lifetimes.sum(), 1e-12)
    for (birth, death), weight in zip(finite, lifetimes, strict=False):
        mid = 0.5 * (birth + death)
        support = np.maximum(0.0, np.minimum(grid - birth, death - grid))
        curve += (weight / weight_total) * support
    return {
        "silhouette_mean": safe_float(np.mean(curve)),
        "silhouette_peak": safe_float(np.max(curve)),
    }


def _diagram_summary(points: np.ndarray, prefix: str, grid_size: int, top_k_lifetimes: int) -> dict[str, float]:
    out: dict[str, float] = {}
    if points.shape[0] < 2:
        for dim in (0, 1):
            out[f"{prefix}_h{dim}_count"] = 0.0
            out[f"{prefix}_h{dim}_total_persistence"] = 0.0
            out[f"{prefix}_h{dim}_entropy"] = 0.0
            out[f"{prefix}_h{dim}_betti_area"] = 0.0
            out[f"{prefix}_h{dim}_betti_peak"] = 0.0
            out[f"{prefix}_h{dim}_betti_mean"] = 0.0
            out[f"{prefix}_h{dim}_silhouette_mean"] = 0.0
            out[f"{prefix}_h{dim}_silhouette_peak"] = 0.0
            for idx in range(top_k_lifetimes):
                out[f"{prefix}_h{dim}_top_lifetime_{idx + 1}"] = 0.0
        return out

    diagrams = ripser(points, maxdim=1)["dgms"]
    for dim in (0, 1):
        diagram = diagrams[dim]
        lifetimes = np.sort(_lifetimes(diagram))[::-1]
        out[f"{prefix}_h{dim}_count"] = float(len(lifetimes))
        out[f"{prefix}_h{dim}_total_persistence"] = safe_float(lifetimes.sum() if lifetimes.size else 0.0)
        out[f"{prefix}_h{dim}_entropy"] = _persistence_entropy(lifetimes)
        out.update({f"{prefix}_h{dim}_{k}": v for k, v in _betti_curve_stats(diagram, grid_size).items()})
        out.update({f"{prefix}_h{dim}_{k}": v for k, v in _silhouette_stats(diagram, grid_size).items()})
        for idx in range(top_k_lifetimes):
            value = lifetimes[idx] if idx < lifetimes.size else 0.0
            out[f"{prefix}_h{dim}_top_lifetime_{idx + 1}"] = safe_float(value)
    return out


def _subset_maps(obs: pd.DataFrame, cfg) -> dict[str, np.ndarray]:
    return {
        "tumor": infer_role_mask(obs["cell_type"], cfg.tumor_terms),
        "cd8": infer_role_mask(obs["cell_type"], cfg.cd8_terms),
        "treg": infer_role_mask(obs["cell_type"], cfg.treg_terms),
        "macrophage": infer_role_mask(obs["cell_type"], cfg.macrophage_terms),
        "vessel": infer_role_mask(obs["cell_type"], cfg.vessel_terms),
    }


def _subset_feature_rows(coords: np.ndarray, subset_masks: dict[str, np.ndarray], cfg) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, mask in subset_masks.items():
        out.update(_diagram_summary(coords[mask], prefix=key, grid_size=cfg.betti_grid_size, top_k_lifetimes=cfg.top_k_lifetimes))
    out.update(
        _diagram_summary(
            coords[subset_masks["tumor"] | subset_masks["cd8"]],
            prefix="union_tumor_cd8",
            grid_size=cfg.betti_grid_size,
            top_k_lifetimes=cfg.top_k_lifetimes,
        )
    )
    out.update(
        _diagram_summary(
            coords[subset_masks["tumor"] | subset_masks["treg"]],
            prefix="union_tumor_treg",
            grid_size=cfg.betti_grid_size,
            top_k_lifetimes=cfg.top_k_lifetimes,
        )
    )
    out.update(
        _diagram_summary(
            coords[subset_masks["tumor"] | subset_masks["macrophage"]],
            prefix="union_tumor_macrophage",
            grid_size=cfg.betti_grid_size,
            top_k_lifetimes=cfg.top_k_lifetimes,
        )
    )
    out.update(
        _diagram_summary(
            coords[subset_masks["tumor"] | subset_masks["vessel"]],
            prefix="union_tumor_vessel",
            grid_size=cfg.betti_grid_size,
            top_k_lifetimes=cfg.top_k_lifetimes,
        )
    )
    return out


def extract_topology_features(adata, cfg) -> dict[str, float]:
    obs = adata.obs.reset_index(drop=True)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    norm_coords, _ = normalize_coordinates(coords)
    features = metadata_row(adata, "F1")
    subset_masks = _subset_maps(obs, cfg)
    features.update(_subset_feature_rows(norm_coords, subset_masks, cfg))

    patch_rows = []
    for idxs in patch_indices(norm_coords, patch_size=cfg.patch_size, min_cells=cfg.min_patch_cells):
        patch_obs = obs.iloc[idxs].reset_index(drop=True)
        patch_masks = _subset_maps(patch_obs, cfg)
        patch_rows.append(_subset_feature_rows(norm_coords[idxs], patch_masks, cfg))
    features.update(aggregate_patch_features(patch_rows, prefix="patch"))
    return features
