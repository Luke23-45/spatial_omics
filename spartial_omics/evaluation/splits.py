from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

try:  # pragma: no cover - depends on sklearn version
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


def _group_to_label_map(labels: np.ndarray, groups: np.ndarray) -> dict[object, object]:
    """Return {group: label} for groups that carry exactly one label."""
    mapping: dict[object, object] = {}
    for group in np.unique(groups):
        group_labels = np.unique(labels[np.asarray(groups) == group])
        if group_labels.shape[0] == 1:
            mapping[group] = group_labels[0]
    return mapping


def build_lopo_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    stratified: bool = True,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-patient-out cross-validation.

    Each fold holds out exactly one patient (group) for testing.
    When *stratified* is True and all groups are pure-label, the folds
    are ordered so that test-label alternation is preserved (CLR, DII,
    CLR, DII, …), giving more balanced per-fold class representation
    than a naive ordering.

    Returns a list of (train_idx, test_idx) arrays.
    """
    unique_groups = np.unique(groups)
    group_to_label = _group_to_label_map(labels, groups)

    if stratified and len(group_to_label) == unique_groups.shape[0]:
        unique_labels = np.unique(labels)
        groups_by_label: dict[object, list[object]] = {label: [] for label in unique_labels}
        for group in unique_groups:
            groups_by_label[group_to_label[group]].append(group)
        rng = np.random.default_rng(random_state)
        for label in unique_labels:
            rng.shuffle(groups_by_label[label])
        ordered_groups: list[object] = []
        max_len = max(len(v) for v in groups_by_label.values())
        for idx in range(max_len):
            for label in unique_labels:
                if idx < len(groups_by_label[label]):
                    ordered_groups.append(groups_by_label[label][idx])
    else:
        rng = np.random.default_rng(random_state)
        ordered_groups = list(rng.permutation(unique_groups))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for test_group in ordered_groups:
        test_mask = groups == test_group
        splits.append((np.flatnonzero(~test_mask), np.flatnonzero(test_mask)))
    return splits


def build_grouped_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Patient-aware grouped cross-validation with stratification.

    When *n_splits* equals the number of unique groups this falls back
    to LOPO (leave-one-patient-out).  Otherwise it uses
    StratifiedGroupKFold (or GroupKFold as fallback) with the
    requested number of splits and repeats.
    """
    unique_groups = np.unique(groups)
    unique_labels = np.unique(labels)
    group_to_label = _group_to_label_map(labels, groups)

    # LOPO shortcut
    if n_splits >= unique_groups.shape[0]:
        return build_lopo_splits(labels, groups, stratified=True, random_state=random_state)

    max_splits = unique_groups.shape[0]
    if unique_labels.shape[0] > 1:
        group_label_counts = []
        for label in unique_labels:
            label_groups = np.unique(groups[np.asarray(labels) == label])
            group_label_counts.append(label_groups.shape[0])
        max_splits = min(max_splits, min(group_label_counts))
    resolved_splits = max(2, min(int(n_splits), max_splits))

    # Pure binary shortcut — pick one patient per class per fold
    if (
        unique_labels.shape[0] == 2
        and len(group_to_label) == unique_groups.shape[0]
        and resolved_splits <= max_splits
    ):
        groups_by_label = {
            label: np.asarray(
                [group for group, group_label in group_to_label.items() if group_label == label],
                dtype=object,
            )
            for label in unique_labels
        }
        limit = min(
            len(groups_by_label[unique_labels[0]]),
            len(groups_by_label[unique_labels[1]]),
            resolved_splits,
        )
        if limit >= 2:
            splits: list[tuple[np.ndarray, np.ndarray]] = []
            for repeat in range(max(1, int(n_repeats))):
                rng = np.random.default_rng(random_state + repeat)
                shuffled = {label: rng.permutation(groups_by_label[label]) for label in unique_labels}
                for fold_idx in range(limit):
                    test_groups = {shuffled[unique_labels[0]][fold_idx], shuffled[unique_labels[1]][fold_idx]}
                    test_mask = np.isin(groups, list(test_groups))
                    train_idx = np.flatnonzero(~test_mask)
                    test_idx = np.flatnonzero(test_mask)
                    splits.append((train_idx, test_idx))
            return splits

    # General path — StratifiedGroupKFold / GroupKFold
    all_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for repeat in range(max(1, int(n_repeats))):
        seed = random_state + repeat
        if StratifiedGroupKFold is not None:
            splitter = StratifiedGroupKFold(
                n_splits=resolved_splits,
                shuffle=True,
                random_state=seed,
            )
            fold_iter = splitter.split(np.zeros_like(labels), labels, groups)
        else:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(unique_groups)
            remapped = {group: idx for idx, group in enumerate(perm)}
            ordered = np.array([remapped[g] for g in groups], dtype=int)
            splitter = GroupKFold(n_splits=resolved_splits)
            fold_iter = splitter.split(np.zeros_like(labels), labels, ordered)
        all_splits.extend((train_idx, test_idx) for train_idx, test_idx in fold_iter)
    return all_splits


def build_nested_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_repeats: int = 1,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]]:
    """Nested patient-aware cross-validation.

    Returns a list of ``(outer_train_idx, outer_test_idx, inner_splits)``
    tuples.  Each *inner_splits* list contains ``(inner_train_idx,
    inner_test_idx)`` pairs derived **only** from the outer training
    indices, so hyper-parameter tuning never peeks at the outer test
    fold.

    When *outer_splits* >= number of unique groups the outer loop
    becomes LOPO.
    """
    unique_groups = np.unique(groups)

    # Build outer splits
    if outer_splits >= unique_groups.shape[0]:
        outer = build_lopo_splits(labels, groups, stratified=True, random_state=random_state)
    else:
        outer = build_grouped_splits(
            labels, groups, n_splits=outer_splits, n_repeats=n_repeats, random_state=random_state,
        )

    nested: list[tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]] = []
    for outer_train_idx, outer_test_idx in outer:
        outer_labels = labels[outer_train_idx]
        outer_groups = groups[outer_train_idx]

        inner_ug = np.unique(outer_groups)
        if inner_splits >= inner_ug.shape[0]:
            inner = build_lopo_splits(outer_labels, outer_groups, stratified=True, random_state=random_state)
        else:
            inner = build_grouped_splits(
                outer_labels, outer_groups,
                n_splits=inner_splits, n_repeats=n_repeats, random_state=random_state,
            )

        # Remap inner indices back to the original sample index space
        remapped_inner = [
            (outer_train_idx[itrain], outer_train_idx[itest])
            for itrain, itest in inner
        ]
        nested.append((outer_train_idx, outer_test_idx, remapped_inner))

    return nested


def bootstrap_ci(
    values: list[float],
    *,
    confidence: float = 0.95,
    n_boot: int = 2000,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean of *values*.

    Returns ``(mean, ci_low, ci_high)``.
    """
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_state)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.shape[0], replace=True)
        boot_means[i] = float(sample.mean())
    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.percentile(boot_means, 100.0 * alpha))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return float(arr.mean()), ci_low, ci_high


__all__ = ["build_grouped_splits", "build_lopo_splits", "build_nested_splits", "bootstrap_ci"]
