"""Per-cell metrics: the outcome breakdown, success with its Wilson CI, and the touchdown audit.

One call, :func:`cell_metrics`, reduces the per-episode rows of one cell -- one method, one
run seed, one (regime, sea state) -- to one :class:`CellMetrics`. The rows are the runner's
(:data:`rld.eval.runner.EPISODE_COLUMNS`), in memory or read back from CSV; values are
coerced, so a CSV's ``"True"``/``"nan"`` strings reduce identically to the in-memory values.

What is reported, and why every piece is there
----------------------------------------------
* **Outcome-class fractions** for all six classes, asserted to sum to 1 (counts are asserted
  to sum to N exactly). Success without the breakdown hides hovering (``timeout``) and
  crashing, so the breakdown is never optional.
* **Success** as ``k``, ``N``, the rate and its **Wilson 95 %** interval.
* **Touchdown quality**, over the episodes where the contact detector fired: the closing
  speed along the deck normal ``max(0, -(v_drone - v_pad) . n)`` -- the criterion-1
  quantity (P2-D5) -- at p50 and p95, and the same closing speed along world z, so the whole
  table is recomputable under the other reading; lateral offset p50/p95; relative tilt p95.
* **Time to touchdown** (contact detector), mean and p50.
* **Control effort** (the per-episode mean of ``||a||^2``, then averaged over episodes) and
  **action jerk** (the per-episode mean of ``||a_t - a_{t-1}||``), in normalised action
  units; multiply by ``v_max^2`` or ``v_max`` for (m/s)^2 or m/s.
* **Analytic-versus-contact touchdown disagreement** ``n`` and rate (Gate 2 required < 1 %;
  Phase 3 is the first volume test) and **tunnelling** ``n``.
* **termination_reason counts**, so the ``crash`` bar decomposes.
* **Touchdown in a truly quiescent window** (Gate 3 remediation, results-skeptic M1): of the
  episodes where the contact detector fired, how many touched down at an instant from which
  the **true** deck satisfied the permissive quiescence predicate for the next 12
  control-rate samples (per-episode ``td_in_quiescent_window``, :mod:`rld.eval.truth`). The
  denominator is ``n_touchdowns``; an episode without a touchdown carries NaN and is not
  counted as either.

Percentiles are NumPy's default (linear interpolation) over finite samples; a cell with no
touchdown reports NaN, never 0. Units: speeds metres per second model scale, lengths metres
model scale, times seconds model scale, angles degrees.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from rld.envs.touchdown import OUTCOMES, TERMINATION_REASONS
from rld.eval.stats import wilson_interval

__all__ = [
    "CELL_METRIC_COLUMNS",
    "CellMetrics",
    "as_bool",
    "as_float",
    "as_optional_bool",
    "cell_metrics",
    "nan_percentile",
]


@dataclass(frozen=True)
class CellMetrics:
    """Every per-cell number the protocol reports. Field order is the committed column order.

    Attributes:
        n_episodes: Episodes in the cell, N.
        n_success: Episodes classified ``success``, k.
        success_rate: ``k / N``, dimensionless.
        success_wilson_lo: Wilson 95 % lower bound, dimensionless.
        success_wilson_hi: Wilson 95 % upper bound, dimensionless.
        frac_crash: Fraction ``crash``.
        frac_off_pad: Fraction ``off_pad``.
        frac_hard_landing: Fraction ``hard_landing``.
        frac_bounce: Fraction ``bounce``.
        frac_success: Fraction ``success`` (equals ``success_rate``).
        frac_timeout: Fraction ``timeout``.
        n_touchdowns: Episodes where the contact detector fired.
        rel_vz_normal_p50_m_s: Closing speed along the deck normal at first contact, p50,
            metres per second model scale.
        rel_vz_normal_p95_m_s: The same, p95.
        rel_vz_world_z_p50_m_s: Closing speed along world z at first contact, p50.
        rel_vz_world_z_p95_m_s: The same, p95.
        lateral_p50_m: Drone-centre offset from the pad centre in the deck plane at first
            contact, p50, metres model scale.
        lateral_p95_m: The same, p95.
        rel_tilt_p95_deg: Drone-versus-deck-normal tilt at first contact, p95, degrees.
        time_to_touchdown_mean_s: Mean episode time of first contact, seconds model scale.
        time_to_touchdown_p50_s: Median of the same.
        effort_mean_sq: Mean over episodes of the per-episode mean ``||a||^2``,
            normalised action units squared.
        action_jerk_mean: Mean over episodes of the per-episode mean
            ``||a_t - a_{t-1}||``, normalised action units per control step.
        disagreement_n: Episodes where the two touchdown detectors disagree.
        disagreement_rate: ``disagreement_n / N``.
        tunnelling_n: Episodes whose penetration exceeded the tunnelling threshold.
        max_penetration_m: Deepest penetration in the cell (most negative separation),
            metres model scale.
        n_td_in_quiescent_window: Touched-down episodes whose true deck satisfied the
            permissive quiescence predicate over the 12 control-rate samples starting at the
            contact touchdown.
        frac_td_in_quiescent_window: ``n_td_in_quiescent_window / n_touchdowns``,
            dimensionless; NaN when nothing touched down.
        frac_in_training_distribution: Fraction of the cell's episodes whose grid cell is a
            dev-pool cell (P3-D2).
        n_reason_ground_contact: ``termination_reason == "ground_contact"`` count.
        n_reason_off_plate_strike: ``"off_plate_strike"`` count.
        n_reason_tilt_gt_crash: ``"tilt_gt_crash"`` count.
        n_reason_below_deck: ``"below_deck"`` count.
        n_reason_out_of_bounds: ``"out_of_bounds"`` count.
        n_reason_dwell_complete: ``"dwell_complete"`` count.
        n_reason_release: ``"release"`` count.
        n_reason_time_limit: ``"time_limit"`` count.
    """

    n_episodes: int
    n_success: int
    success_rate: float
    success_wilson_lo: float
    success_wilson_hi: float
    frac_crash: float
    frac_off_pad: float
    frac_hard_landing: float
    frac_bounce: float
    frac_success: float
    frac_timeout: float
    n_touchdowns: int
    rel_vz_normal_p50_m_s: float
    rel_vz_normal_p95_m_s: float
    rel_vz_world_z_p50_m_s: float
    rel_vz_world_z_p95_m_s: float
    lateral_p50_m: float
    lateral_p95_m: float
    rel_tilt_p95_deg: float
    time_to_touchdown_mean_s: float
    time_to_touchdown_p50_s: float
    effort_mean_sq: float
    action_jerk_mean: float
    disagreement_n: int
    disagreement_rate: float
    tunnelling_n: int
    max_penetration_m: float
    n_td_in_quiescent_window: int
    frac_td_in_quiescent_window: float
    frac_in_training_distribution: float
    n_reason_ground_contact: int
    n_reason_off_plate_strike: int
    n_reason_tilt_gt_crash: int
    n_reason_below_deck: int
    n_reason_out_of_bounds: int
    n_reason_dwell_complete: int
    n_reason_release: int
    n_reason_time_limit: int


#: Committed column order of a per-cell summary.
CELL_METRIC_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(CellMetrics))

#: Tolerance on the outcome fractions' sum; the counts themselves are checked exactly.
_SUM_TOL: float = 1e-12


def as_float(value: Any) -> float:
    """Return a row value as a float, mapping ``None`` and ``""`` to NaN.

    Args:
        value: A float, int, numeric string, ``None`` or ``""``.

    Returns:
        The float.
    """
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return float("nan")
    return float(value)


def as_bool(value: Any) -> bool:
    """Return a row value as a bool, accepting CSV spellings.

    Args:
        value: A bool, int, or one of ``"True"``/``"False"``/``"true"``/``"false"``/
            ``"1"``/``"0"``.

    Returns:
        The bool.

    Raises:
        ValueError: If a string is none of the accepted spellings.
    """
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1"):
            return True
        if text in ("false", "0"):
            return False
        raise ValueError(f"not a boolean: {value!r}")
    return bool(value)


def as_optional_bool(value: Any) -> bool | None:
    """Return a row value as a bool, or ``None`` for a missing (NaN) value.

    Args:
        value: A bool, ``None``, a float NaN, or a CSV spelling accepted by :func:`as_bool`
            or ``"nan"``/``""``.

    Returns:
        The bool, or ``None`` when the value is missing.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and value.strip().lower() in ("nan", ""):
        return None
    return as_bool(value)


def nan_percentile(values: Sequence[float], q: float) -> float:
    """Return a percentile over the finite values, or NaN if there are none.

    Args:
        values: Samples, in the caller's units.
        q: Percentile in ``[0, 100]``.

    Returns:
        The percentile (NumPy linear interpolation), or NaN for an empty sample -- a cell
        where nothing landed has no touchdown speed, not a touchdown speed of zero.
    """
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float("nan") if finite.size == 0 else float(np.percentile(finite, q))


def _nan_mean(values: Sequence[float]) -> float:
    """Return the mean over the finite values, or NaN if there are none."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float("nan") if finite.size == 0 else float(finite.mean())


def cell_metrics(rows: Sequence[Mapping[str, Any]]) -> CellMetrics:
    """Reduce one cell's per-episode rows to its committed metrics.

    Args:
        rows: The cell's episodes -- one method, one run seed, one (regime, sea state).

    Returns:
        The :class:`CellMetrics`.

    Raises:
        ValueError: If ``rows`` is empty, or an outcome or termination reason is not one of
            the frozen classes.
        AssertionError: If the outcome counts do not sum to N or the fractions do not sum
            to 1 -- the six classes are exhaustive and mutually exclusive by construction,
            so either failure is a bug upstream.
    """
    n = len(rows)
    if n == 0:
        raise ValueError("a cell needs at least one episode")
    outcomes = [str(row["outcome"]) for row in rows]
    unknown = sorted(set(outcomes) - set(OUTCOMES))
    if unknown:
        raise ValueError(f"unknown outcomes {unknown}, expected {list(OUTCOMES)}")
    reasons = [str(row["termination_reason"]) for row in rows]
    unknown = sorted(set(reasons) - set(TERMINATION_REASONS))
    if unknown:
        raise ValueError(f"unknown termination reasons {unknown}")

    counts = {outcome: outcomes.count(outcome) for outcome in OUTCOMES}
    if sum(counts.values()) != n:
        raise AssertionError(f"outcome counts {counts} do not sum to N = {n}")
    fracs = {outcome: counts[outcome] / n for outcome in OUTCOMES}
    if not math.isclose(sum(fracs.values()), 1.0, rel_tol=0.0, abs_tol=_SUM_TOL):
        raise AssertionError(f"outcome fractions sum to {sum(fracs.values())!r}, not 1")

    landed = [row for row in rows if as_bool(row["touchdown_contact"])]
    closing_normal = [max(0.0, -as_float(r["rel_vz_normal_m_s"])) for r in landed]
    closing_world = [max(0.0, -as_float(r["rel_vz_world_z_m_s"])) for r in landed]
    lateral = [as_float(r["lateral_offset_m"]) for r in landed]
    tilt = [as_float(r["rel_tilt_deg"]) for r in landed]
    t_td = [as_float(r["td_t_episode_s"]) for r in landed]
    disagreements = sum(as_bool(row["detectors_disagree"]) for row in rows)
    n_quiet = 0
    for row in rows:
        verdict = as_optional_bool(row["td_in_quiescent_window"])
        if as_bool(row["touchdown_contact"]) != (verdict is not None):
            raise AssertionError(
                "td_in_quiescent_window must be a verdict exactly when the contact detector "
                f"fired: touchdown_contact={row['touchdown_contact']!r}, verdict={verdict!r}"
            )
        n_quiet += bool(verdict)
    k = counts["success"]
    lo, hi = wilson_interval(k, n)
    return CellMetrics(
        n_episodes=n,
        n_success=k,
        success_rate=k / n,
        success_wilson_lo=lo,
        success_wilson_hi=hi,
        frac_crash=fracs["crash"],
        frac_off_pad=fracs["off_pad"],
        frac_hard_landing=fracs["hard_landing"],
        frac_bounce=fracs["bounce"],
        frac_success=fracs["success"],
        frac_timeout=fracs["timeout"],
        n_touchdowns=len(landed),
        rel_vz_normal_p50_m_s=nan_percentile(closing_normal, 50.0),
        rel_vz_normal_p95_m_s=nan_percentile(closing_normal, 95.0),
        rel_vz_world_z_p50_m_s=nan_percentile(closing_world, 50.0),
        rel_vz_world_z_p95_m_s=nan_percentile(closing_world, 95.0),
        lateral_p50_m=nan_percentile(lateral, 50.0),
        lateral_p95_m=nan_percentile(lateral, 95.0),
        rel_tilt_p95_deg=nan_percentile(tilt, 95.0),
        time_to_touchdown_mean_s=_nan_mean(t_td),
        time_to_touchdown_p50_s=nan_percentile(t_td, 50.0),
        effort_mean_sq=_nan_mean([as_float(row["effort_mean_sq"]) for row in rows]),
        action_jerk_mean=_nan_mean([as_float(row["action_jerk_mean"]) for row in rows]),
        disagreement_n=disagreements,
        disagreement_rate=disagreements / n,
        tunnelling_n=sum(as_bool(row["tunnelled"]) for row in rows),
        max_penetration_m=min(as_float(row["max_penetration_m"]) for row in rows),
        n_td_in_quiescent_window=n_quiet,
        frac_td_in_quiescent_window=n_quiet / len(landed) if landed else float("nan"),
        frac_in_training_distribution=(
            sum(as_bool(row["in_training_distribution"]) for row in rows) / n
        ),
        n_reason_ground_contact=reasons.count("ground_contact"),
        n_reason_off_plate_strike=reasons.count("off_plate_strike"),
        n_reason_tilt_gt_crash=reasons.count("tilt_gt_crash"),
        n_reason_below_deck=reasons.count("below_deck"),
        n_reason_out_of_bounds=reasons.count("out_of_bounds"),
        n_reason_dwell_complete=reasons.count("dwell_complete"),
        n_reason_release=reasons.count("release"),
        n_reason_time_limit=reasons.count("time_limit"),
    )
