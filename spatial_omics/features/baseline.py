from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np

from spatial_omics.features.common import delaunay_edges, infer_role_mask, metadata_row, mixing_entropy, mutual_knn_edges, normalize_coordinates, safe_float


def _distance_summary(a: np.ndarray, b: np.ndarray, prefix: str) -> dict[str, float]:
    if a.size == 0 or b.size == 0:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
        }
    dists = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))
    mins = dists.min(axis=1)
    return {
        f"{prefix}_min": safe_float(mins.min()),
        f"{prefix}_mean": safe_float(mins.mean()),
        f"{prefix}_median": safe_float(np.median(mins)),
    }


def extract_baseline_features(adata, cfg) -> dict[str, float]:
    obs = adata.obs.reset_index(drop=True)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    norm_coords, scale = normalize_coordinates(coords)
    features = metadata_row(adata, "F0")

    counts = obs["cell_type"].value_counts()
    total_cells = max(int(obs.shape[0]), 1)
    features["cell_count"] = float(total_cells)
    bbox = (coords.max(axis=0) - coords.min(axis=0)) if coords.shape[0] else np.array([0.0, 0.0])
    area = max(float(bbox[0] * bbox[1]), 1e-6)
    features["bbox_area"] = area
    features["cell_density"] = float(total_cells / area)
    features["median_5nn_scale"] = float(scale)

    for cell_type, count in counts.items():
        slug = str(cell_type).lower().replace(" ", "_")
        features[f"prop_celltype_{slug}"] = float(count / total_cells)

    if "compartment" in obs.columns:
        for compartment, count in obs["compartment"].value_counts().items():
            slug = str(compartment).lower().replace(" ", "_")
            features[f"prop_compartment_{slug}"] = float(count / total_cells)

    edges = mutual_knn_edges(norm_coords, k=cfg.knn_k)
    edge_pairs = []
    for a, b in edges:
        left = str(obs.iloc[a]["cell_type"]).lower().replace(" ", "_")
        right = str(obs.iloc[b]["cell_type"]).lower().replace(" ", "_")
        pair = tuple(sorted((left, right)))
        edge_pairs.append(pair)
    pair_counts = Counter(edge_pairs)
    total_edges = max(len(edges), 1)
    for pair, count in pair_counts.items():
        features[f"knn_pair_{pair[0]}__{pair[1]}"] = float(count / total_edges)

    tumor_mask = infer_role_mask(obs["cell_type"], cfg.tumor_terms)
    immune_mask = infer_role_mask(obs["cell_type"], cfg.immune_terms)
    cd8_mask = infer_role_mask(obs["cell_type"], cfg.cd8_terms)
    macrophage_mask = infer_role_mask(obs["cell_type"], cfg.macrophage_terms)
    features.update(_distance_summary(norm_coords[tumor_mask], norm_coords[immune_mask], "tumor_to_immune"))
    features.update(_distance_summary(norm_coords[tumor_mask], norm_coords[cd8_mask], "tumor_to_cd8"))
    features.update(_distance_summary(norm_coords[tumor_mask], norm_coords[macrophage_mask], "tumor_to_macrophage"))

    graph = nx.Graph()
    graph.add_nodes_from(range(obs.shape[0]))
    graph.add_edges_from(delaunay_edges(norm_coords))
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    features["graph_degree_mean"] = safe_float(np.mean(degrees)) if degrees.size else 0.0
    features["graph_degree_std"] = safe_float(np.std(degrees)) if degrees.size else 0.0
    features["graph_degree_max"] = safe_float(np.max(degrees)) if degrees.size else 0.0
    features["graph_components"] = float(nx.number_connected_components(graph))
    clustering = nx.average_clustering(graph) if graph.number_of_nodes() > 1 else 0.0
    features["graph_clustering_mean"] = safe_float(clustering)
    labels = {idx: str(cell_type) for idx, cell_type in enumerate(obs["cell_type"])}
    try:
        assortativity = nx.attribute_assortativity_coefficient(graph, "cell_type")
    except Exception:
        nx.set_node_attributes(graph, labels, "cell_type")
        assortativity = nx.attribute_assortativity_coefficient(graph, "cell_type")
    features["graph_assortativity"] = safe_float(assortativity)
    graph_pairs = []
    for a, b in graph.edges():
        pair = tuple(sorted((labels[a].lower().replace(" ", "_"), labels[b].lower().replace(" ", "_"))))
        graph_pairs.append(pair)
    features["graph_edge_type_entropy"] = mixing_entropy(graph_pairs)
    return features
