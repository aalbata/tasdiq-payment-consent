"""Interval estimates and tests for clustered benchmark outcomes.

The seed scenario is the independent unit of the trace track and the live task is the
independent unit of the live track, so every interval here resamples or permutes whole
clusters.  Every resampling or permutation step takes an explicit seed.  Binomial intervals (Wilson, Clopper-Pearson, rule of three) are
provided for counts whose observations are independent, and the Holm step-down
procedure adjusts families of p-values.  All results are returned at full precision;
rounding is left to the table layer.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
from scipy.stats import beta, norm

Statistic = Callable[[np.ndarray], float]


class BootstrapInterval(tuple):
    """Point estimate with a percentile bootstrap interval, as the triple (est, lo, hi).

    ``n_dropped`` counts the bootstrap replicates that were not finite (a ratio statistic
    over a resample with a zero denominator) and were left out of the percentiles;
    ``n_boot`` is the number of replicates drawn, so ``n_boot - n_dropped`` replicates
    formed the interval. The triple compares and unpacks as a plain tuple.
    """

    n_dropped: int
    n_boot: int | None

    def __new__(cls, est: float, lo: float, hi: float, n_dropped: int = 0, n_boot: int | None = None) -> "BootstrapInterval":
        self = super().__new__(cls, (float(est), float(lo), float(hi)))
        self.n_dropped = int(n_dropped)
        self.n_boot = None if n_boot is None else int(n_boot)
        return self

    @property
    def est(self) -> float:
        return self[0]

    @property
    def lo(self) -> float:
        return self[1]

    @property
    def hi(self) -> float:
        return self[2]

    def __repr__(self) -> str:
        return f"BootstrapInterval(est={self[0]!r}, lo={self[1]!r}, hi={self[2]!r}, n_dropped={self.n_dropped!r}, n_boot={self.n_boot!r})"


class PermutationResult(NamedTuple):
    """Observed paired difference with its two-sided permutation p-value."""

    diff: float
    p_value: float
    n_perm: int


def _as_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 2:
        raise ValueError("values must be one- or two-dimensional")
    return arr


def _cluster_index(clusters: Sequence[Hashable], n: int) -> list[np.ndarray]:
    labels = list(clusters)
    if len(labels) != n:
        raise ValueError("clusters must have one label per observation")
    if n == 0:
        raise ValueError("at least one observation is required")
    members: dict[Hashable, list[int]] = {}
    for i, label in enumerate(labels):
        members.setdefault(label, []).append(i)
    return [np.asarray(idx, dtype=int) for idx in members.values()]


def _percentile_interval(samples: np.ndarray, alpha: float) -> tuple[float, float, int]:
    """Percentile bounds over the finite replicates and the number of replicates dropped as non-finite."""
    finite = samples[np.isfinite(samples)]
    dropped = int(samples.size - finite.size)
    if finite.size == 0:
        return float("nan"), float("nan"), dropped
    lo = float(np.percentile(finite, 100.0 * alpha / 2.0))
    hi = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi, dropped


def cluster_bootstrap(
    values: Sequence[float] | np.ndarray,
    clusters: Sequence[Hashable],
    stat: Statistic = np.mean,
    n_boot: int = 2000,
    *,
    seed: int,
    alpha: float = 0.05,
) -> BootstrapInterval:
    """Percentile cluster bootstrap of a statistic over observations grouped by cluster.

    ``values`` holds one row per observation (a one-dimensional array for a plain
    statistic, or a two-dimensional array when the statistic needs several columns,
    for example a ratio of sums).  ``clusters`` gives the cluster label of each row.
    Each replicate draws clusters with replacement, concatenates their rows and
    applies ``stat``.  The point estimate is ``stat`` on the original rows and the
    interval is the (alpha/2, 1 - alpha/2) percentile pair of the finite replicates;
    replicates that are not finite are dropped and counted in ``n_dropped`` of the result.
    With a single cluster every replicate equals the original data and the interval
    collapses onto the estimate.  ``seed`` initialises the generator that draws the
    clusters and must be given explicitly.
    """
    arr = _as_array(values)
    groups = _cluster_index(clusters, arr.shape[0])
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    est = float(stat(arr))
    rng = np.random.default_rng(seed)
    n_clusters = len(groups)
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([groups[g] for g in draw])
        samples[b] = float(stat(arr[idx]))
    lo, hi, dropped = _percentile_interval(samples, alpha)
    return BootstrapInterval(est, lo, hi, dropped, n_boot)


def paired_cluster_diff(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    clusters: Sequence[Hashable],
    stat: Statistic = np.mean,
    n_boot: int = 2000,
    *,
    seed: int,
    alpha: float = 0.05,
) -> BootstrapInterval:
    """Cluster bootstrap interval for ``stat(a) - stat(b)`` on paired observations.

    Rows of ``values_a`` and ``values_b`` are aligned (the same variant under two
    arms or two rules) and share the cluster label in ``clusters``.  Every replicate
    draws clusters with replacement once and evaluates both arms on the same draw, so
    the interval reflects the paired design.  ``seed`` initialises the generator that
    draws the clusters and must be given explicitly.
    """
    arr_a = _as_array(values_a)
    arr_b = _as_array(values_b)
    if arr_a.shape != arr_b.shape:
        raise ValueError("paired value arrays must have the same shape")
    groups = _cluster_index(clusters, arr_a.shape[0])
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    est = float(stat(arr_a)) - float(stat(arr_b))
    rng = np.random.default_rng(seed)
    n_clusters = len(groups)
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([groups[g] for g in draw])
        samples[b] = float(stat(arr_a[idx])) - float(stat(arr_b[idx]))
    lo, hi, dropped = _percentile_interval(samples, alpha)
    return BootstrapInterval(est, lo, hi, dropped, n_boot)


def paired_cluster_permutation(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
    clusters: Sequence[Hashable],
    stat: Statistic = np.mean,
    n_perm: int = 2000,
    *,
    seed: int,
) -> PermutationResult:
    """Two-sided permutation test of ``stat(a) - stat(b)`` with cluster-level swaps.

    Under the null hypothesis the two arms are exchangeable within each cluster, so
    each permutation swaps the arm labels of whole clusters chosen with probability
    one half and recomputes the difference.  The p-value is
    ``(1 + number of permuted |differences| >= |observed|) / (1 + n_perm)``.  ``seed``
    initialises the generator that chooses the swapped clusters and must be given
    explicitly.
    """
    arr_a = _as_array(values_a)
    arr_b = _as_array(values_b)
    if arr_a.shape != arr_b.shape:
        raise ValueError("paired value arrays must have the same shape")
    groups = _cluster_index(clusters, arr_a.shape[0])
    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    observed = float(stat(arr_a)) - float(stat(arr_b))
    rng = np.random.default_rng(seed)
    n_clusters = len(groups)
    count = 0
    for _ in range(n_perm):
        flips = rng.integers(0, 2, size=n_clusters).astype(bool)
        perm_a = arr_a.copy()
        perm_b = arr_b.copy()
        for g, flip in enumerate(flips):
            if flip:
                idx = groups[g]
                perm_a[idx] = arr_b[idx]
                perm_b[idx] = arr_a[idx]
        diff = float(stat(perm_a)) - float(stat(perm_b))
        if abs(diff) >= abs(observed) - 1e-12:
            count += 1
    p_value = (1.0 + count) / (1.0 + n_perm)
    return PermutationResult(observed, p_value, n_perm)


def _group_diff(values: np.ndarray, in_group: np.ndarray, stat: Statistic) -> float:
    """``stat(values of group 1) - stat(values of group 0)``; not finite when a group is empty."""
    a = values[in_group]
    b = values[~in_group]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(stat(a)) - float(stat(b))


def cluster_swap_permutation(
    values: Sequence[float] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    clusters: Sequence[Hashable],
    stat: Statistic = np.mean,
    n_perm: int = 2000,
    *,
    seed: int,
) -> PermutationResult:
    """Two-sided permutation test of a two-group difference with cluster-level label swaps.

    ``values`` holds one observation per row and ``groups`` its group flag (truthy for
    group 1, falsy for group 0); every cluster holds members of both groups, as the seed
    scenarios of the capture track hold members in every language.  The statistic is
    ``stat(group 1) - stat(group 0)`` on the whole sample.  Under the null hypothesis the
    group labels are exchangeable within a cluster, so each permutation swaps the labels
    of every member of the clusters chosen with probability one half and recomputes the
    difference; with equal group sizes in every cluster this is the paired cluster
    permutation of :func:`paired_cluster_permutation`.  The p-value is
    ``(1 + number of permuted |differences| >= |observed|) / (1 + n_perm)``.  ``seed``
    initialises the generator that chooses the swapped clusters and must be given
    explicitly.
    """
    arr = _as_array(values)
    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")
    flags = np.asarray([bool(g) for g in groups], dtype=bool)
    if flags.shape != arr.shape:
        raise ValueError("groups must have one flag per observation")
    group_index = _cluster_index(clusters, arr.shape[0])
    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    cluster_of = np.empty(arr.shape[0], dtype=int)
    for g, idx in enumerate(group_index):
        cluster_of[idx] = g
    observed = _group_diff(arr, flags, stat)
    rng = np.random.default_rng(seed)
    n_clusters = len(group_index)
    count = 0
    for _ in range(n_perm):
        swap = rng.integers(0, 2, size=n_clusters).astype(bool)
        permuted = flags ^ swap[cluster_of]
        diff = _group_diff(arr, permuted, stat)
        if abs(diff) >= abs(observed) - 1e-12:
            count += 1
    p_value = (1.0 + count) / (1.0 + n_perm)
    return PermutationResult(observed, p_value, n_perm)


def cluster_label_permutation(
    values: Sequence[float] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    clusters: Sequence[Hashable],
    stat: Statistic = np.mean,
    n_perm: int = 2000,
    *,
    seed: int,
) -> PermutationResult:
    """Two-sided permutation test of a two-group difference where whole clusters belong to one group.

    ``values`` holds one observation per row and ``groups`` its group flag; every cluster
    carries one group label (the live tasks of the change families against the tasks of
    the no-change control).  The statistic is ``stat(group 1) - stat(group 0)`` on the
    whole sample.  Under the null hypothesis the clusters are exchangeable between the
    groups, so each permutation reassigns the cluster-level labels at random, keeping the
    number of clusters in each group, and recomputes the difference.  The p-value is
    ``(1 + number of permuted |differences| >= |observed|) / (1 + n_perm)``.  ``seed``
    initialises the generator that permutes the labels and must be given explicitly.
    """
    arr = _as_array(values)
    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")
    flags = np.asarray([bool(g) for g in groups], dtype=bool)
    if flags.shape != arr.shape:
        raise ValueError("groups must have one flag per observation")
    group_index = _cluster_index(clusters, arr.shape[0])
    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    cluster_of = np.empty(arr.shape[0], dtype=int)
    labels = np.empty(len(group_index), dtype=bool)
    for g, idx in enumerate(group_index):
        cluster_of[idx] = g
        members = flags[idx]
        if members.any() and not members.all():
            raise ValueError("every cluster must belong to one group")
        labels[g] = bool(members[0])
    observed = _group_diff(arr, flags, stat)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        permuted_labels = rng.permutation(labels)
        diff = _group_diff(arr, permuted_labels[cluster_of], stat)
        if abs(diff) >= abs(observed) - 1e-12:
            count += 1
    p_value = (1.0 + count) / (1.0 + n_perm)
    return PermutationResult(observed, p_value, n_perm)


def align_pairs(
    records_a: Iterable[Mapping[str, Any]],
    records_b: Iterable[Mapping[str, Any]],
    key: str,
    value_field: str,
    cluster_field: str = "seed_id",
) -> tuple[np.ndarray, np.ndarray, list[Hashable]]:
    """Align two record sets on ``key`` and return paired values with cluster labels.

    Only keys present in both sets are kept, in the order of ``records_a``.  Values
    are converted to float (booleans become 0.0 or 1.0).  The cluster label is taken
    from the record in ``records_a``.
    """
    index_b = {r[key]: r for r in records_b}
    vals_a: list[float] = []
    vals_b: list[float] = []
    clusters: list[Hashable] = []
    for r in records_a:
        k = r[key]
        if k not in index_b:
            continue
        vals_a.append(float(r[value_field]))
        vals_b.append(float(index_b[k][value_field]))
        clusters.append(r[cluster_field])
    return np.asarray(vals_a, dtype=float), np.asarray(vals_b, dtype=float), clusters


def _check_binomial(k: int, n: int) -> None:
    if n < 0 or k < 0 or k > n:
        raise ValueError("require 0 <= k <= n")


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n``.

    With ``z`` the standard normal quantile at ``1 - alpha / 2`` the interval is
    ``(p + z^2 / 2n +- z sqrt(p (1 - p) / n + z^2 / 4n^2)) / (1 + z^2 / n)``.  An
    empty sample yields ``(0.0, 1.0)``.
    """
    _check_binomial(k, n)
    if n == 0:
        return 0.0, 1.0
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, float(centre - half))
    hi = 1.0 if k == n else min(1.0, float(centre + half))
    return lo, hi


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson interval for a binomial proportion ``k / n``.

    The lower bound is the ``alpha / 2`` quantile of Beta(k, n - k + 1), or 0 when
    ``k = 0``; the upper bound is the ``1 - alpha / 2`` quantile of Beta(k + 1, n - k),
    or 1 when ``k = n``.  An empty sample yields ``(0.0, 1.0)``.
    """
    _check_binomial(k, n)
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def rule_of_three(n: int) -> float:
    """Approximate 95 percent upper bound ``3 / n`` on a rate observed as 0 in n."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 3.0 / n


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order.

    With ``m`` hypotheses and p-values sorted ascending as ``p_(1) <= ... <= p_(m)``,
    the adjusted value of ``p_(i)`` is ``max_{j <= i} min(1, (m - j + 1) p_(j))``.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    if p.size == 0:
        return []
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("p-values must lie in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return [float(v) for v in adjusted]


def holm_reject(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Rejection decisions of the Holm procedure at family-wise level ``alpha``."""
    return [adj <= alpha for adj in holm(pvalues)]
