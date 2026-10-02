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
4. **The Phase 7 hypothesis statistics (P3-D1 §4 and §8, P3-D4 #6 and #8, P7-D1 §2).**

   * :func:`paired_relative_p95` -- ``r = 1 - p95(a) / p95(b)`` of the touchdown closing
     speed. Each replicate resamples seeds per method, then **one** set of episode indices
     shared by both methods; each method's p95 is over **its own pooled resampled touched-down
     episodes only** (an episode without a touchdown has no closing speed and is excluded,
     never imputed). The point estimate is the same statistic on the unresampled pool of all
     seeds x episodes. ``timeouts_as_worst`` gives the P3-D4 #8 sensitivity variant, in which a
     ``timeout`` episode is ranked as the worst (infinite) closing speed.
   * :func:`paired_iqm_difference` -- IQM-over-seeds success of ``a`` minus that of ``b`` on
     identical episodes (seeds per method, one shared set of episode indices).
   * :func:`drop_difference_bootstrap` -- the H2 / H4 / lambda "drop" statistic,
     ``drop(m) = IQM success in cell 1 - IQM success in cell 2``, alone or as
     ``drop(a) - drop(b)``. Mode ``"independent_cells"`` (H2, lambda): seeds resampled per
     method and shared by that method's two cells; episodes resampled **independently per
     cell** (within a cell one set of indices is shared by both methods, because both flew
     that cell's identical list). Mode ``"shared_episodes"`` (H4, and the paired arm
     contrasts): one set of episode indices shared by both cells and both methods.

   "Success of a learned method in a cell" is rliable's IQM over its seeds of each seed's
   success rate (P7-D1 §2); a deterministic baseline is one seed, so its value is its rate.

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
from typing import Literal

import numpy as np
import scipy.stats
from dmf.typedefs import FloatArray, IntArray
from rliable import metrics as rly_metrics

__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "PAIRED_REPS",
    "STRATIFIED_REPS",
    "WILSON_Z_95",
    "DropContrast",
    "DropMode",
    "IntervalEstimate",
    "PairedContrast",
    "RelativeP95",
    "aggregate_estimates",
    "drop_difference_bootstrap",
    "iqm",
    "optimality_gap",
    "paired_bootstrap",
    "paired_iqm_difference",
    "paired_relative_p95",
    "quantile_linear",
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


# --------------------------------------------------------------------------- Phase 7

#: Two-sided percentile of the touchdown closing speed the relative-p95 statistic compares.
P95_QUANTILE: float = 0.95

#: The two resampling designs of :func:`drop_difference_bootstrap` (P7-D1 §2).
type DropMode = Literal["independent_cells", "shared_episodes"]


def _lerp(a: FloatArray, b: FloatArray, t: FloatArray) -> FloatArray:
    """Linear interpolation exactly as NumPy's ``_lerp`` (``np.percentile`` "linear").

    Infinite end points are handled explicitly, where NumPy's formula would give NaN
    (``inf - inf`` or ``inf * 0``): an infinite bound wins unless its weight is exactly 0.

    Args:
        a: Lower neighbours.
        b: Upper neighbours (``b >= a`` elementwise).
        t: Weights in ``[0, 1)``.

    Returns:
        The interpolated values.
    """
    with np.errstate(invalid="ignore"):
        diff = b - a
        out = a + diff * t
        out = np.where(t >= 0.5, b - diff * (1.0 - t), out)
    out = np.where(a == b, a, out)
    out = np.where(np.isneginf(a), -np.inf, out)
    out = np.where(np.isposinf(b) & (t > 0.0), np.inf, out)
    out = np.where(np.isposinf(b) & (t == 0.0), a, out)
    return np.asarray(out, dtype=np.float64)


def quantile_linear(values: FloatArray, q: float) -> FloatArray:
    """Return the ``q``-quantile of each row over its non-NaN entries.

    Hyndman-Fan type 7 ("linear"), bit-identical to ``np.percentile(finite, 100 q)`` on a row
    of finite values (tested), vectorised over rows of unequal valid length, and defined for
    ``+inf`` / ``-inf`` entries (:func:`_lerp`), which the timeout-ranked sensitivity needs.

    Args:
        values: ``(n_rows, n)`` or ``(n,)``; NaN marks a missing value (excluded).
        q: Quantile in ``[0, 1]``.

    Returns:
        ``(n_rows,)`` (or a 0-d array for 1-D input); NaN for a row with no valid entry.
    """
    arr = np.asarray(values, dtype=np.float64)
    one_d = arr.ndim == 1
    rows = arr.reshape(1, -1) if one_d else arr
    ordered = np.sort(rows, axis=1)  # NaN sorts last; -inf first, +inf before NaN
    counts = np.sum(~np.isnan(rows), axis=1)
    safe = np.maximum(counts, 1)
    virtual = (safe - 1).astype(np.float64) * np.float64(q)
    lower = np.floor(virtual).astype(np.int64)
    upper = np.minimum(lower + 1, safe - 1)
    gamma = virtual - lower
    idx = np.arange(rows.shape[0])
    out = _lerp(ordered[idx, lower], ordered[idx, upper], gamma)
    out = np.where(counts == 0, np.nan, out)
    return out.reshape(()) if one_d else out


def _check_pair(a: FloatArray, b: FloatArray, *, finite: bool) -> tuple[FloatArray, FloatArray]:
    """Validate two ``(n_seeds, n_episodes)`` arrays over the same episodes.

    Args:
        a: First method.
        b: Second method.
        finite: Require every value finite (0/1 outcomes); otherwise NaN is allowed.

    Returns:
        Both as float64.

    Raises:
        ValueError: On a shape or finiteness violation.
    """
    out = []
    for name, arr in (("a", a), ("b", b)):
        array = np.asarray(arr, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError(f"{name} must be (n_seeds >= 1, n_episodes >= 1), got {array.shape}")
        if finite and not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite: a missing seed or episode is not a zero")
        if not finite and np.any(np.isinf(array)):
            raise ValueError(f"{name}: a closing speed is never infinite (use timeouts_as_worst)")
        out.append(array)
    if out[0].shape[1] != out[1].shape[1]:
        raise ValueError(
            f"a and b must cover the same episodes, got {out[0].shape[1]} and {out[1].shape[1]}"
        )
    return out[0], out[1]


def _check_reps(reps: int, confidence: float) -> None:
    """Refuse a non-positive replicate count or a coverage outside ``(0, 1)``."""
    if reps < 1 or not 0.0 < confidence < 1.0:
        raise ValueError(f"need reps >= 1 and 0 < confidence < 1, got {reps}, {confidence}")


def _percentile_interval(values: FloatArray, confidence: float) -> tuple[float, float, int]:
    """Return the percentile interval of bootstrap replicates and the number undefined.

    Replicates that are NaN (statistic undefined, e.g. no touchdown in a resample) are
    excluded and counted; the bounds use :func:`quantile_linear`, so infinite replicates
    (timeout-ranked sensitivity) are ordered correctly.
    """
    tail = (1.0 - confidence) / 2.0
    lo = float(quantile_linear(values, tail))
    hi = float(quantile_linear(values, 1.0 - tail))
    return lo, hi, int(np.sum(np.isnan(values)))


@dataclass(frozen=True)
class RelativeP95:
    """The paired relative-p95 closing-speed contrast ``r = 1 - p95(a) / p95(b)``.

    Attributes:
        r: Point estimate on the unresampled pools (every seed x episode of each method),
            dimensionless; positive means ``a`` lands softer than ``b``.
        lo: Lower percentile bound.
        hi: Upper percentile bound.
        p95_a: ``a``'s pooled p95 closing speed, m/s model scale (``inf`` possible only with
            ``timeouts_as_worst``).
        p95_b: ``b``'s pooled p95, m/s model scale.
        reps: Bootstrap replicates.
        confidence: Two-sided coverage.
        n_episodes: Paired episodes.
        n_seeds_a: Seeds of ``a``.
        n_seeds_b: Seeds of ``b``.
        n_touchdowns_a: Touched-down (seed, episode) pairs of ``a`` in the pool.
        n_touchdowns_b: The same for ``b``.
        n_timeouts_a: ``timeout`` (seed, episode) pairs of ``a`` (ranked worst only under the
            sensitivity variant).
        n_timeouts_b: The same for ``b``.
        timeouts_as_worst: Whether this is the P3-D4 #8 sensitivity variant.
        n_reps_undefined: Replicates whose statistic was undefined (excluded from the bounds).
    """

    r: float
    lo: float
    hi: float
    p95_a: float
    p95_b: float
    reps: int
    confidence: float
    n_episodes: int
    n_seeds_a: int
    n_seeds_b: int
    n_touchdowns_a: int
    n_touchdowns_b: int
    n_timeouts_a: int
    n_timeouts_b: int
    timeouts_as_worst: bool
    n_reps_undefined: int

    @property
    def separates(self) -> bool:
        """Return whether the interval excludes 0."""
        return self.lo > 0.0 or self.hi < 0.0


def _relative(p95_a: FloatArray, p95_b: FloatArray) -> FloatArray:
    """Return ``1 - p95_a / p95_b`` elementwise, with the infinite cases made explicit.

    ``p95_a = inf`` against a finite ``p95_b`` gives ``-inf``; a finite ``p95_a`` against
    ``p95_b = inf`` gives ``1``; both infinite, or ``p95_b = 0``, is undefined (NaN).
    """
    a = np.asarray(p95_a, dtype=np.float64)
    b = np.asarray(p95_b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - a / b
    out = np.where(np.isinf(a) & np.isfinite(b) & (b > 0.0), -np.inf, out)
    out = np.where(np.isfinite(a) & np.isinf(b), 1.0, out)
    out = np.where(np.isinf(a) & np.isinf(b), np.nan, out)
    out = np.where(b == 0.0, np.nan, out)
    return np.asarray(out, dtype=np.float64)


def _ranked(speeds: FloatArray, timeouts: FloatArray | None, worst: bool) -> FloatArray:
    """Return the closing speeds, with timeouts set to ``+inf`` under the sensitivity variant.

    Raises:
        ValueError: If the masks are missing or misshapen.
    """
    if not worst:
        return speeds
    if timeouts is None:
        raise ValueError("timeouts_as_worst needs the timeout masks")
    mask = np.asarray(timeouts, dtype=bool)
    if mask.shape != speeds.shape:
        raise ValueError(f"timeout mask shape {mask.shape} != speeds {speeds.shape}")
    return np.where(mask, np.inf, speeds)


def paired_relative_p95(
    a_speeds: FloatArray,
    b_speeds: FloatArray,
    *,
    a_timeouts: FloatArray | None = None,
    b_timeouts: FloatArray | None = None,
    timeouts_as_worst: bool = False,
    reps: int = PAIRED_REPS,
    confidence: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> RelativeP95:
    """Return P3-D1 §4's paired relative-p95 statistic with its bootstrap interval.

    Per replicate (P3-D1 §4, P3-D4 #6): draw ``n_seeds_a`` of ``a``'s seeds and ``n_seeds_b``
    of ``b``'s with replacement, independently; draw **one** set of ``n_episodes`` episode
    indices with replacement, shared by both methods; each method's p95 is
    :func:`quantile_linear` at 0.95 over its pooled resampled (seed, episode) closing speeds,
    **touched-down episodes only** (NaN entries are excluded, never imputed); the replicate's
    statistic is ``1 - p95_a / p95_b``. The draw order per batch is seeds of ``a``, seeds of
    ``b``, then episodes, from one generator seeded with ``seed``.

    With ``timeouts_as_worst`` (P3-D4 #8, a sensitivity analysis, never scored) every
    ``timeout`` episode enters its method's pool as an infinite closing speed, so a method
    cannot lower its p95 by timing out on its worst deck windows.

    Args:
        a_speeds: ``(n_seeds_a, n_episodes)`` touchdown closing speeds along the deck normal,
            m/s model scale; NaN where the episode has no touchdown.
        b_speeds: ``(n_seeds_b, n_episodes)``, the same episodes in the same column order.
        a_timeouts: ``a``'s ``timeout`` mask, same shape; needed for the sensitivity variant.
        b_timeouts: ``b``'s ``timeout`` mask.
        timeouts_as_worst: Rank timeouts as the worst closing speed.
        reps: Bootstrap replicates (P3-D1: 10 000).
        confidence: Two-sided coverage.
        seed: Resampling seed (P3-D1: 20260926).

    Returns:
        The :class:`RelativeP95`.

    Raises:
        ValueError: On malformed input, or a method with no touchdown at all.
    """
    arr_a, arr_b = _check_pair(a_speeds, b_speeds, finite=False)
    _check_reps(reps, confidence)
    ranked_a = _ranked(arr_a, a_timeouts, timeouts_as_worst)
    ranked_b = _ranked(arr_b, b_timeouts, timeouts_as_worst)
    if not np.any(np.isfinite(arr_a)) or not np.any(np.isfinite(arr_b)):
        raise ValueError("each method needs at least one touchdown for a p95")
    n_a, n_e = arr_a.shape
    n_b = arr_b.shape[0]
    p95_a = float(quantile_linear(ranked_a.reshape(-1), P95_QUANTILE))
    p95_b = float(quantile_linear(ranked_b.reshape(-1), P95_QUANTILE))
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, _PAIRED_BATCH):
        size = min(_PAIRED_BATCH, reps - start)
        seeds_a = rng.integers(0, n_a, size=(size, n_a))
        seeds_b = rng.integers(0, n_b, size=(size, n_b))
        episodes = rng.integers(0, n_e, size=(size, n_e))
        pool_a = ranked_a[seeds_a[:, :, None], episodes[:, None, :]].reshape(size, -1)
        pool_b = ranked_b[seeds_b[:, :, None], episodes[:, None, :]].reshape(size, -1)
        values[start : start + size] = _relative(
            quantile_linear(pool_a, P95_QUANTILE), quantile_linear(pool_b, P95_QUANTILE)
        )
    lo, hi, undefined = _percentile_interval(values, confidence)
    timeouts_a = 0 if a_timeouts is None else int(np.sum(np.asarray(a_timeouts, dtype=bool)))
    timeouts_b = 0 if b_timeouts is None else int(np.sum(np.asarray(b_timeouts, dtype=bool)))
    return RelativeP95(
        r=float(_relative(np.array([p95_a]), np.array([p95_b]))[0]),
        lo=lo,
        hi=hi,
        p95_a=p95_a,
        p95_b=p95_b,
        reps=reps,
        confidence=confidence,
        n_episodes=n_e,
        n_seeds_a=n_a,
        n_seeds_b=n_b,
        n_touchdowns_a=int(np.sum(np.isfinite(arr_a))),
        n_touchdowns_b=int(np.sum(np.isfinite(arr_b))),
        n_timeouts_a=timeouts_a,
        n_timeouts_b=timeouts_b,
        timeouts_as_worst=timeouts_as_worst,
        n_reps_undefined=undefined,
    )


def _iqm_rows(rates: FloatArray) -> FloatArray:
    """Return rliable's IQM (25 % trimmed mean) of each row, vectorised over replicates."""
    return np.asarray(scipy.stats.trim_mean(rates, 0.25, axis=1), dtype=np.float64)


def _iqm_point(success: FloatArray) -> float:
    """Return the IQM over seeds of each seed's success rate (rliable's own function)."""
    rates = np.asarray(success, dtype=np.float64).mean(axis=1)
    return iqm(rates.reshape(-1, 1))


def _resampled_rates(success: FloatArray, seeds: IntArray, episodes: IntArray) -> FloatArray:
    """Return ``(batch, n_seeds)`` success rates of resampled seeds over resampled episodes."""
    return np.asarray(
        success[seeds[:, :, None], episodes[:, None, :]].mean(axis=2), dtype=np.float64
    )


@dataclass(frozen=True)
class DropContrast:
    """A success contrast between two cells, or the difference of two methods' contrasts.

    ``drop(m) = success(m, cell 1) - success(m, cell 2)``, success being the IQM over seeds of
    each seed's success rate (P7-D1 §2). With one method the statistic is ``drop(a)``; with
    two it is ``drop(a) - drop(b)``. Rates and differences are fractions in ``[-1, 1]``
    (multiply by 100 for points).

    Attributes:
        statistic: Point estimate on the unresampled data.
        lo: Lower percentile bound.
        hi: Upper percentile bound.
        mode: ``"independent_cells"`` or ``"shared_episodes"``.
        drop_a: ``drop(a)`` point estimate.
        drop_b: ``drop(b)`` point estimate, NaN for a single-method contrast.
        success_a1: ``a``'s IQM success in cell 1.
        success_a2: ``a``'s IQM success in cell 2.
        success_b1: ``b``'s IQM success in cell 1 (NaN without ``b``).
        success_b2: ``b``'s IQM success in cell 2 (NaN without ``b``).
        reps: Bootstrap replicates.
        confidence: Two-sided coverage.
        n_episodes_1: Episodes in cell 1.
        n_episodes_2: Episodes in cell 2.
        n_seeds_a: Seeds of ``a``.
        n_seeds_b: Seeds of ``b`` (0 without ``b``).
    """

    statistic: float
    lo: float
    hi: float
    mode: str
    drop_a: float
    drop_b: float
    success_a1: float
    success_a2: float
    success_b1: float
    success_b2: float
    reps: int
    confidence: float
    n_episodes_1: int
    n_episodes_2: int
    n_seeds_a: int
    n_seeds_b: int

    @property
    def separates(self) -> bool:
        """Return whether the interval excludes 0."""
        return self.lo > 0.0 or self.hi < 0.0


def _method_cells(cell1: FloatArray, cell2: FloatArray, name: str) -> tuple[FloatArray, FloatArray]:
    """Validate one method's two cells: 0/1 outcomes, the same seeds (rows) in both."""
    c1 = _as_score_matrix(cell1)
    c2 = _as_score_matrix(cell2)
    if c1.shape[0] != c2.shape[0]:
        raise ValueError(
            f"{name}: a seed is one policy, so both cells need the same seeds (rows), got "
            f"{c1.shape[0]} and {c2.shape[0]}"
        )
    return c1, c2


def drop_difference_bootstrap(
    a_cell1: FloatArray,
    a_cell2: FloatArray,
    b_cell1: FloatArray | None = None,
    b_cell2: FloatArray | None = None,
    *,
    mode: DropMode,
    reps: int = PAIRED_REPS,
    confidence: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> DropContrast:
    """Return ``drop(a)`` or ``drop(a) - drop(b)`` with its bootstrap interval (P7-D1 §2).

    Per replicate: draw each method's seeds with replacement, independently between methods
    and **shared by the method's two cells** (a seed is one policy). Then draw episodes:

    * ``"independent_cells"`` (H2, the lambda arm): one set of ``n_episodes_1`` indices for
      cell 1 and an **independent** set of ``n_episodes_2`` for cell 2, because the two cells
      are different episodes. Within a cell the set is shared by both methods, which flew
      that cell's identical list.
    * ``"shared_episodes"`` (H4, and the pad, motion and noise arm contrasts): the two cells
      are the **same** listed episodes under two conditions, so one set of indices is shared
      by both cells and both methods.

    Each method's success in a cell is the IQM over its resampled seeds of each seed's
    success rate over the resampled episodes. Draw order per batch: seeds of ``a``, seeds of
    ``b`` (if any), episodes of cell 1, episodes of cell 2 (independent mode only).

    Args:
        a_cell1: ``(n_seeds_a, n_episodes_1)`` 0/1 success of ``a`` in cell 1.
        a_cell2: ``(n_seeds_a, n_episodes_2)``, same seeds in the same row order.
        b_cell1: ``(n_seeds_b, n_episodes_1)`` or ``None`` for a single-method drop.
        b_cell2: ``(n_seeds_b, n_episodes_2)`` or ``None``.
        mode: The resampling design.
        reps: Bootstrap replicates (P7-D1: 10 000).
        confidence: Two-sided coverage.
        seed: Resampling seed (P7-D1: 20260926).

    Returns:
        The :class:`DropContrast`.

    Raises:
        ValueError: On malformed input, a mode that does not fit the shapes, or only one of
            ``b``'s cells.
    """
    _check_reps(reps, confidence)
    if mode not in ("independent_cells", "shared_episodes"):
        raise ValueError(f"unknown mode {mode!r}")
    a1, a2 = _method_cells(a_cell1, a_cell2, "a")
    if (b_cell1 is None) != (b_cell2 is None):
        raise ValueError("give both of b's cells or neither")
    has_b = b_cell1 is not None and b_cell2 is not None
    if has_b:
        assert b_cell1 is not None and b_cell2 is not None
        b1, b2 = _method_cells(b_cell1, b_cell2, "b")
        if b1.shape[1] != a1.shape[1] or b2.shape[1] != a2.shape[1]:
            raise ValueError("a and b must cover the same episodes in each cell")
    else:
        b1 = b2 = np.zeros((0, 0))
    n_e1, n_e2 = a1.shape[1], a2.shape[1]
    if mode == "shared_episodes" and n_e1 != n_e2:
        raise ValueError(f"shared episodes need equal cells, got {n_e1} and {n_e2}")
    n_a = a1.shape[0]
    n_b = b1.shape[0] if has_b else 0

    s_a1, s_a2 = _iqm_point(a1), _iqm_point(a2)
    drop_a = s_a1 - s_a2
    nan = float("nan")
    s_b1, s_b2, drop_b = nan, nan, nan
    if has_b:
        s_b1, s_b2 = _iqm_point(b1), _iqm_point(b2)
        drop_b = s_b1 - s_b2
    point = drop_a - drop_b if has_b else drop_a

    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, _PAIRED_BATCH):
        size = min(_PAIRED_BATCH, reps - start)
        seeds_a = rng.integers(0, n_a, size=(size, n_a))
        seeds_b = rng.integers(0, n_b, size=(size, n_b)) if has_b else None
        ep1 = rng.integers(0, n_e1, size=(size, n_e1))
        ep2 = rng.integers(0, n_e2, size=(size, n_e2)) if mode == "independent_cells" else ep1
        rep = _iqm_rows(_resampled_rates(a1, seeds_a, ep1)) - _iqm_rows(
            _resampled_rates(a2, seeds_a, ep2)
        )
        if seeds_b is not None:
            rep = rep - (
                _iqm_rows(_resampled_rates(b1, seeds_b, ep1))
                - _iqm_rows(_resampled_rates(b2, seeds_b, ep2))
            )
        values[start : start + size] = rep
    lo, hi, _ = _percentile_interval(values, confidence)
    return DropContrast(
        statistic=float(point),
        lo=lo,
        hi=hi,
        mode=mode,
        drop_a=float(drop_a),
        drop_b=float(drop_b),
        success_a1=float(s_a1),
        success_a2=float(s_a2),
        success_b1=float(s_b1),
        success_b2=float(s_b2),
        reps=reps,
        confidence=confidence,
        n_episodes_1=n_e1,
        n_episodes_2=n_e2,
        n_seeds_a=n_a,
        n_seeds_b=n_b,
    )


def paired_iqm_difference(
    a: FloatArray,
    b: FloatArray,
    *,
    reps: int = PAIRED_REPS,
    confidence: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PairedContrast:
    """Return ``IQM success(a) - IQM success(b)`` on identical episodes (H1a NI, H1b).

    The paired design of :func:`paired_bootstrap` -- seeds resampled per method
    independently, then one set of episode indices shared by both -- with the success of a
    method taken as the IQM over its (resampled) seeds of each seed's success rate
    (P7-D1 §2) rather than the plain mean. Draw order per batch: seeds of ``a``, seeds of
    ``b``, episodes. For a single-seed method the IQM is its rate, so against a
    deterministic baseline and with 1 seed both definitions agree.

    Args:
        a: ``(n_seeds_a, n_episodes)`` 0/1 outcomes.
        b: ``(n_seeds_b, n_episodes)``, the same episodes in the same column order.
        reps: Bootstrap replicates (P3-D1: 10 000).
        confidence: Two-sided coverage.
        seed: Resampling seed.

    Returns:
        A :class:`PairedContrast` whose ``diff`` is the IQM difference.
    """
    arr_a, arr_b = _check_pair(a, b, finite=True)
    _check_reps(reps, confidence)
    n_a, n_e = arr_a.shape
    n_b = arr_b.shape[0]
    rng = np.random.default_rng(seed)
    values = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, _PAIRED_BATCH):
        size = min(_PAIRED_BATCH, reps - start)
        seeds_a = rng.integers(0, n_a, size=(size, n_a))
        seeds_b = rng.integers(0, n_b, size=(size, n_b))
        episodes = rng.integers(0, n_e, size=(size, n_e))
        values[start : start + size] = _iqm_rows(
            _resampled_rates(arr_a, seeds_a, episodes)
        ) - _iqm_rows(_resampled_rates(arr_b, seeds_b, episodes))
    lo, hi, _ = _percentile_interval(values, confidence)
    return PairedContrast(
        diff=float(_iqm_point(arr_a) - _iqm_point(arr_b)),
        lo=lo,
        hi=hi,
        reps=reps,
        confidence=confidence,
        n_episodes=n_e,
        n_seeds_a=n_a,
        n_seeds_b=n_b,
    )
