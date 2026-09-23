"""Statistics for the frozen evaluation: Wilson intervals, rliable aggregates, paired contrasts.

Three tools, one per question the protocol asks (``landing-protocol`` skill, P3-D1):

1. **Per cell, per seed -- how sure are we of one success rate?** :func:`wilson_interval`,
   the Wilson score interval at 95 %. Never report a success rate without it and its ``N``.
2. **Across training seeds -- what is a method's aggregate performance?** :func:`iqm` and
   :func:`optimality_gap` are rliable's own definitions (``rliable.metrics``), with a
   **stratified** bootstrap over runs within each task (:func:`stratified_bootstrap_ci`),
   2 000 replicates by default.
3. **Between two methods on the same episodes -- do they separate?**
   :func:`paired_bootstrap`: a two-level bootstrap that resamples **seeds** (each method's
   independently, because seed 0 of one method has nothing to do with seed 0 of another)
   and then **episodes** (shared between the two methods, which is what makes the contrast
   paired), 10 000 replicates by default. "Separates" means the 95 % CI of the difference
   excludes 0 and nothing weaker.

Why the stratified bootstrap is implemented here rather than called from rliable
---------------------------------------------------------------------------------
``rliable.library`` cannot be imported in the pinned environment: it imports
``arch.bootstrap``, and ``arch`` 7.2.0 fails at import time against ``pandas`` 3.0.5
(``deprecate_kwarg() missing 1 required positional argument: 'new_arg_name'``). The point
estimators come from ``rliable.metrics``, which imports cleanly and is used as-is. The
resampling is a line-for-line port of ``rliable.library.StratifiedBootstrap.update_indices``
(runs resampled with replacement **independently within each task**, tasks never
resampled) with a percentile interval, as ``get_interval_estimates(method="percentile")``
computes. One deliberate difference: rliable draws its indices from NumPy's **global**
``np.random`` state even when a ``random_state`` is passed, so its intervals are not
reproducible from an argument; here every draw comes from an explicit seeded generator.

All quantities are dimensionless (rates in ``[0, 1]`` or differences of rates) unless the
caller passes a physical metric, in which case the units are the caller's; nothing here has a
time or length scale.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from dmf.typedefs import FloatArray
from rliable import metrics as rly_metrics

__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "PAIRED_REPS",
    "STRATIFIED_REPS",
    "WILSON_Z_95",
    "IntervalEstimate",
    "PairedContrast",
    "aggregate_estimates",
    "iqm",
    "optimality_gap",
    "paired_bootstrap",
    "stratified_bootstrap_ci",
    "wilson_interval",
]

#: Two-sided 95 % standard-normal quantile, ``Phi^-1(0.975)``, to float64 precision.
WILSON_Z_95: float = 1.959963984540054

#: Stratified-bootstrap replicates for rliable aggregates (P3-D1).
STRATIFIED_REPS: int = 2000

#: Paired-bootstrap replicates for method contrasts H1-H4 (P3-D1).
PAIRED_REPS: int = 10000

#: Seed of every bootstrap generator unless a caller passes one. Recorded so that every
#: interval in a committed table is reproducible to the last digit.
DEFAULT_BOOTSTRAP_SEED: int = 20260926

#: Replicates evaluated per vectorised batch in :func:`paired_bootstrap`; bounds memory at
#: ``batch x n_seeds x n_episodes`` float64 values and does not change any result.
_PAIRED_BATCH: int = 250


@dataclass(frozen=True)
class IntervalEstimate:
    """A point estimate with a bootstrap confidence interval.

    Attributes:
        point: The statistic on the observed data, in the statistic's units.
        lo: Lower confidence bound.
        hi: Upper confidence bound.
        reps: Bootstrap replicates behind the interval.
        confidence: Two-sided coverage, e.g. 0.95.
    """

    point: float
    lo: float
    hi: float
    reps: int
    confidence: float


@dataclass(frozen=True)
class PairedContrast:
    """A paired method contrast ``mean(a) - mean(b)`` with its bootstrap interval.

    Attributes:
        diff: Observed difference of means, ``a`` minus ``b``, in the outcome's units
            (a difference of success rates for 0/1 outcomes).
        lo: Lower bound of the percentile interval.
        hi: Upper bound of the percentile interval.
        reps: Bootstrap replicates.
        confidence: Two-sided coverage.
        n_episodes: Paired episodes.
        n_seeds_a: Seeds (runs) of method ``a``.
        n_seeds_b: Seeds (runs) of method ``b``.
    """

    diff: float
    lo: float
    hi: float
    reps: int
    confidence: float
    n_episodes: int
    n_seeds_a: int
    n_seeds_b: int

    @property
    def separates(self) -> bool:
        """Return whether the interval excludes 0 -- the only sense of "separates" used."""
        return self.lo > 0.0 or self.hi < 0.0


def wilson_interval(k: int, n: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion.

    ``(p + z^2/2n +/- z sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)`` with ``p = k/n``. Unlike
    the Wald interval it stays inside ``[0, 1]`` and is not degenerate at ``k = 0`` or
    ``k = n``, which is exactly where a strong baseline lives.

    Args:
        k: Successes, ``0 <= k <= n``.
        n: Trials.
        z: Standard-normal quantile; the default is two-sided 95 %.

    Returns:
        ``(lo, hi)``, dimensionless, clipped to ``[0, 1]``. ``(nan, nan)`` when ``n == 0``,
        because an empty cell has no rate rather than a rate of zero.

    Raises:
        ValueError: If ``k`` or ``n`` is negative or ``k > n``.
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * float(np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def iqm(scores: FloatArray) -> float:
    """Return rliable's interquartile mean over all runs and tasks.

    Args:
        scores: ``(n_runs, n_tasks)`` scores, e.g. success rates in ``[0, 1]``.

    Returns:
        The 25 %-trimmed mean, in the scores' units.
    """
    return float(rly_metrics.aggregate_iqm(np.asarray(scores, dtype=np.float64)))


def optimality_gap(scores: FloatArray, gamma: float = 1.0) -> float:
    """Return rliable's optimality gap: ``gamma - mean(min(scores, gamma))``.

    Args:
        scores: ``(n_runs, n_tasks)`` scores.
        gamma: The target score; 1.0 for success rates. Lower gap is better.

    Returns:
        The optimality gap, in the scores' units.
    """
    return float(rly_metrics.aggregate_optimality_gap(np.asarray(scores, dtype=np.float64), gamma))


def _as_score_matrix(scores: FloatArray) -> FloatArray:
    """Validate an rliable-shaped score matrix.

    Args:
        scores: Candidate ``(n_runs, n_tasks)`` array.

    Returns:
        The array as float64.

    Raises:
        ValueError: If it is not 2-D, is empty, or holds a non-finite value.
    """
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"scores must be (n_runs >= 1, n_tasks >= 1), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("scores must be finite: a missing seed or cell is not a zero")
    return array


def stratified_bootstrap_ci(
    scores: FloatArray,
    statistic: Callable[[FloatArray], float],
    *,
    reps: int = STRATIFIED_REPS,
    confidence: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> IntervalEstimate:
    """Return a stratified-bootstrap percentile interval for an aggregate over runs x tasks.

    Each replicate resamples the runs **independently within every task**, with
    replacement, and never resamples tasks -- rliable's ``StratifiedBootstrap`` with
    ``task_bootstrap=False``.

    Args:
        scores: ``(n_runs, n_tasks)``; a run is a training seed, a task is an evaluation
            cell (regime x sea state).
        statistic: Aggregate of a score matrix, e.g. :func:`iqm`.
        reps: Bootstrap replicates.
        confidence: Two-sided coverage in ``(0, 1)``.
        seed: Seed of the resampling generator.

    Returns:
        The :class:`IntervalEstimate`. With a single run the interval collapses onto the
        point estimate, which is correct: a deterministic baseline has no seed variance.

    Raises:
        ValueError: If ``scores`` is malformed, ``reps < 1`` or ``confidence`` is not in
            ``(0, 1)``.
    """
    array = _as_score_matrix(scores)
    if reps < 1 or not 0.0 < confidence < 1.0:
        raise ValueError(f"need reps >= 1 and 0 < confidence < 1, got {reps}, {confidence}")
    n_runs, n_tasks = array.shape
    rng = np.random.default_rng(seed)
    task_index = np.arange(n_tasks)[np.newaxis, :]
    values = np.empty(reps, dtype=np.float64)
    for rep in range(reps):
        run_index = rng.integers(0, n_runs, size=(n_runs, n_tasks))
        values[rep] = statistic(array[run_index, task_index])
    tail = 100.0 * (1.0 - confidence) / 2.0
    lo, hi = np.percentile(values, [tail, 100.0 - tail])
    return IntervalEstimate(
        point=float(statistic(array)),
        lo=float(lo),
        hi=float(hi),
        reps=reps,
        confidence=confidence,
    )


def aggregate_estimates(
    scores: FloatArray,
    *,
    reps: int = STRATIFIED_REPS,
    confidence: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, IntervalEstimate]:
    """Return the two rliable aggregates the protocol reports, each with its interval.

    Args:
        scores: ``(n_runs, n_tasks)`` success rates in ``[0, 1]``.
        reps: Stratified-bootstrap replicates.
        confidence: Two-sided coverage.
        seed: Resampling seed, shared by both aggregates so they see the same replicates.

    Returns:
        ``{"iqm": ..., "optimality_gap": ...}``.
    """
    return {
        "iqm": stratified_bootstrap_ci(scores, iqm, reps=reps, confidence=confidence, seed=seed),
        "optimality_gap": stratified_bootstrap_ci(
            scores, optimality_gap, reps=reps, confidence=confidence, seed=seed
        ),
    }


def paired_bootstrap(
    a: FloatArray,
    b: FloatArray,
    *,
    reps: int = PAIRED_REPS,
    confidence: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PairedContrast:
    """Return the paired contrast ``mean(a) - mean(b)`` over identical episodes.

    Two-level resampling per replicate: (1) draw ``n_seeds_a`` of ``a``'s seeds and
    ``n_seeds_b`` of ``b``'s seeds with replacement, **independently**, since training
    seeds of two methods are not paired; (2) draw ``n_episodes`` episode indices with
    replacement, **shared** by both methods, since episode ``e`` is the identical committed
    episode for both -- that is the pairing. The replicate's statistic is the mean of
    ``a`` over its resampled seeds and episodes minus the same for ``b``.

    Args:
        a: ``(n_seeds_a, n_episodes)`` per-episode outcomes of method ``a`` (1.0/0.0 for
            success, or a continuous metric). A deterministic baseline is one row.
        b: ``(n_seeds_b, n_episodes)``, same episodes in the same column order.
        reps: Bootstrap replicates.
        confidence: Two-sided coverage in ``(0, 1)``.
        seed: Resampling seed.

    Returns:
        The :class:`PairedContrast`, in the outcome's units.

    Raises:
        ValueError: If either array is not 2-D, the episode axes differ, a value is not
            finite, ``reps < 1`` or ``confidence`` is not in ``(0, 1)``.
    """
    arr_a = _as_score_matrix(a)
    arr_b = _as_score_matrix(b)
    if arr_a.shape[1] != arr_b.shape[1]:
        raise ValueError(
            f"a and b must cover the same episodes, got {arr_a.shape[1]} and {arr_b.shape[1]}"
        )
    if reps < 1 or not 0.0 < confidence < 1.0:
        raise ValueError(f"need reps >= 1 and 0 < confidence < 1, got {reps}, {confidence}")
    n_a, n_e = arr_a.shape
    n_b = arr_b.shape[0]
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, _PAIRED_BATCH):
        size = min(_PAIRED_BATCH, reps - start)
        seeds_a = rng.integers(0, n_a, size=(size, n_a))
        seeds_b = rng.integers(0, n_b, size=(size, n_b))
        episodes = rng.integers(0, n_e, size=(size, n_e))
        per_episode_a = arr_a[seeds_a].mean(axis=1)  # (size, n_e)
        per_episode_b = arr_b[seeds_b].mean(axis=1)
        diff = per_episode_a - per_episode_b
        values[start : start + size] = np.take_along_axis(diff, episodes, axis=1).mean(axis=1)
    tail = 100.0 * (1.0 - confidence) / 2.0
    lo, hi = np.percentile(values, [tail, 100.0 - tail])
    return PairedContrast(
        diff=float(arr_a.mean() - arr_b.mean()),
        lo=float(lo),
        hi=float(hi),
        reps=reps,
        confidence=confidence,
        n_episodes=n_e,
        n_seeds_a=n_a,
        n_seeds_b=n_b,
    )
