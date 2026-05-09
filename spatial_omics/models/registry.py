"""Canonical active model registry for the dedicated repo."""

ACTIVE_BASELINE_MODEL = "graphsage"
ACTIVE_TOPOLOGY_MODEL = "spatial_z4_v2"

SUPPORTED_MODELS = {
    ACTIVE_BASELINE_MODEL: {
        "family": "baseline",
        "description": "Primary GraphSAGE reference baseline for grouped CRC evaluation.",
        "entrypoint": "python -m spatial_omics.scripts.run_gnn_baselines",
    },
    ACTIVE_TOPOLOGY_MODEL: {
        "family": "topology",
        "description": "Retained standalone topology-oriented Spatial-Z4 v2 configuration.",
        "entrypoint": "python -m spatial_omics.scripts.run_spatial_z4",
    },
}

__all__ = ["ACTIVE_BASELINE_MODEL", "ACTIVE_TOPOLOGY_MODEL", "SUPPORTED_MODELS"]
