from __future__ import annotations

from pathlib import Path

from spatial_omics.reporting.markdown import build_markdown_report


def test_build_report_smoke(tmp_path: Path) -> None:
    results = {
        "runs": [
            {
                "feature_set": "SpatialGraph",
                "model_name": "graphsage",
                "metrics": {
                    "auroc": 0.93,
                    "balanced_accuracy": 0.85,
                    "macro_f1": 0.84,
                    "brier_score": 0.11,
                    "feature_stability": 0.0,
                },
                "top_features": [],
            },
            {
                "feature_set": "Spatial-Z4",
                "model_name": "spatial_z4_v2",
                "metrics": {
                    "auroc": 0.94,
                    "balanced_accuracy": 0.82,
                    "macro_f1": 0.81,
                    "brier_score": 0.12,
                    "feature_stability": 0.0,
                },
                "top_features": [],
            },
        ],
        "summary": {
            "best_run": {
                "feature_set": "SpatialGraph",
                "model_name": "graphsage",
                "metrics": {
                    "auroc": 0.93,
                    "balanced_accuracy": 0.85,
                    "macro_f1": 0.84,
                    "brier_score": 0.11,
                    "feature_stability": 0.0,
                },
            }
        },
    }
    report_path = build_markdown_report(results, tmp_path / "report_out")
    assert Path(report_path).exists()
    assert "Spatial Omics Benchmark Report" in Path(report_path).read_text(encoding="utf-8")
