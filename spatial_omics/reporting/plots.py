from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from spatial_omics.utils.io import ensure_dir


def plot_metric_bars(results: dict, output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    frame = pd.DataFrame(
        [
            {
                "feature_set": run["feature_set"],
                "model_name": run["model_name"],
                "auroc": run["metrics"]["auroc"],
            }
            for run in results.get("runs", [])
        ]
    )
    path = Path(output_dir) / "metric_comparison.png"
    if frame.empty:
        return path
    labels = frame["feature_set"] + " / " + frame["model_name"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, frame["auroc"])
    ax.set_ylabel("AUROC")
    ax.set_title("Feasibility comparison")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_region_cells(adata, title: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coords = adata.obsm["spatial"]
    obs = adata.obs
    fig, ax = plt.subplots(figsize=(5, 5))
    for cell_type, frame in obs.groupby("cell_type"):
        idx = frame.index.to_numpy()
        ax.scatter(coords[idx, 0], coords[idx, 1], s=8, alpha=0.7, label=str(cell_type))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=6, markerscale=2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


__all__ = ["plot_metric_bars", "plot_region_cells"]
