from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatial_omics.data.types import SpatialStudy


@dataclass(frozen=True)
class DatasetPreflightReport:
    dataset_name: str
    n_regions: int
    n_patients: int
    label_counts: dict[str, int]
    patient_counts_by_label: dict[str, int]
    requested_splits: int
    recommended_max_splits: int
    can_run_grouped_cv: bool
    warnings: tuple[str, ...] = ()


def study_preflight_report(study: SpatialStudy, *, requested_splits: int) -> DatasetPreflightReport:
    sample_table = study.sample_table.copy()
    labels = sample_table["label"].astype(str).to_numpy(object)
    groups = sample_table["patient_id"].astype(str).to_numpy(object)

    label_counts = sample_table["label"].astype(str).value_counts().to_dict()
    patient_counts_by_label = {
        str(label): int(sample_table.loc[sample_table["label"].astype(str) == str(label), "patient_id"].nunique())
        for label in sorted(sample_table["label"].astype(str).unique().tolist())
    }

    warnings: list[str] = []
    if len(label_counts) < 2:
        warnings.append("Only one class is present in the study.")

    unique_groups = np.unique(groups)
    max_splits = int(unique_groups.shape[0])
    if len(label_counts) > 1:
        group_label_counts = []
        for label in sorted(np.unique(labels).tolist()):
            label_groups = np.unique(groups[labels == label])
            group_label_counts.append(int(label_groups.shape[0]))
        if group_label_counts:
            max_splits = min(max_splits, min(group_label_counts))

    if max_splits < 2:
        warnings.append("Insufficient patient coverage for grouped cross-validation.")

    if requested_splits > max_splits:
        warnings.append(
            f"Requested n_splits={requested_splits} exceeds recommended max_splits={max_splits} for patient-balanced grouped CV."
        )

    return DatasetPreflightReport(
        dataset_name=study.dataset_name,
        n_regions=int(len(sample_table)),
        n_patients=int(sample_table["patient_id"].nunique()),
        label_counts={str(k): int(v) for k, v in label_counts.items()},
        patient_counts_by_label=patient_counts_by_label,
        requested_splits=int(requested_splits),
        recommended_max_splits=int(max_splits),
        can_run_grouped_cv=bool(max_splits >= 2 and len(label_counts) >= 2),
        warnings=tuple(warnings),
    )


__all__ = ["DatasetPreflightReport", "study_preflight_report"]
