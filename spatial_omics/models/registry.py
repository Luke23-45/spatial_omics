"""Canonical active model registry for the dedicated repo."""

ACTIVE_BASELINE_MODEL = "graphsage"
ACTIVE_TOPOLOGY_MODEL = "toponet_hodge"
EXPERIMENTAL_TOPOLOGY_MODEL = None

SUPPORTED_MODELS = {
    ACTIVE_BASELINE_MODEL: {
        "family": "baseline",
        "description": "Primary GraphSAGE reference baseline for grouped CRC evaluation.",
        "entrypoint": "python -m spatial_omics.scripts.run_gnn_baselines",
    },
    ACTIVE_TOPOLOGY_MODEL: {
        "family": "topology",
        "description": "Active simplicial Hodge topology benchmark built from node-edge-triangle complexes.",
        "entrypoint": "python -m spatial_omics.scripts.run_toponet_hodge",
    },
}

__all__ = ["ACTIVE_BASELINE_MODEL", "ACTIVE_TOPOLOGY_MODEL", "EXPERIMENTAL_TOPOLOGY_MODEL", "SUPPORTED_MODELS"]
