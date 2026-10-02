r"""Score H1a, H1b, H2, H3 and H4 exactly as pre-registered; compute every Phase 7 contrast.

Authority: P3-D1 §4 and §8 (frozen), P3-D4 #5, #6, #8, #9, P6-D6 (H4 stays at ``id`` SS5,
bounded at <= 0), P7-D1 §2 (estimators), §7 (H5 deferred to Gate 8), §8 (caveats), and
**P7-D1a** (2026-10-02, committed before any Phase 7 number was read), which corrects P7-D1
§2 where it over-reached. Where P3-D1 and P7-D1 disagree, P3-D1 wins. Every verdict is
computed here, by :func:`verdict_h1a`, :func:`verdict_magnitude`, :func:`verdict_p7d1a` and
:func:`verdict_half_rule`, from numbers computed from the committed per-episode rows -- never
typed by hand. A test pins the verdict functions with a truth table.

Inputs (per-episode rows, nothing else)
---------------------------------------
* learned methods, aft, JONSWAP, lambda = 1/25: ``results/e07/matrix/episodes.csv``;
* baselines, aft: ``results/e01/episodes.csv`` (five) and ``results/e01_lowvz_cut`` (``lowvz_cut``);
* pad at CG: ``results/e07/cg`` (learned, ``lowvz_cut``) and ``results/e02`` (five baselines);
* the sinusoid leg, the noise conditions and the lambda conditions: their ``results/e07``
  directories.

Every ``results/e07`` ``episodes.csv`` is read from its committed ``episodes.csv.gz`` when
present (:mod:`rld.eval.storage`, P7-D1a §12); sources are always named by the logical
``episodes.csv``, so both files below are byte-identical before and after compression.

Outputs
-------
* ``results/e07/contrasts.csv`` -- one row per contrast (:data:`CONTRAST_COLUMNS`): the
  hypothesis contrasts, their post-hoc and sensitivity variants, and the descriptive arm
  contrasts (pad aft - CG, JONSWAP - sinusoid, clean - noisy, lambda 1/25 - other lambda),
  each labelled with its role. 10 000 replicates, seed 20260926 for **every** contrast (so
  replicate draws are correlated across contrasts; stated, not corrected: P7-D1a #11),
  percentile 95 % CI.
* ``results/e07/hypotheses.csv`` -- one row per scored part, secondary, post-hoc variant,
  sensitivity and pending item (:data:`HYPOTHESIS_COLUMNS`), with the rule that produced the
  verdict, the contrast ids behind it and the caveat keys (:data:`CAVEATS`) carried with it.
  There is **no combined H1 and no combined H3 row** (P3-D4 #5; P7-D1a #2).

Estimators (P3-D1 §4; P7-D1a #3, #4)
-----------------------------------
* **H1a's non-inferiority, H1b, H4** -- P3-D1 §4's paired bootstrap on per-episode
  differences: seeds resampled per method, then one set of episode indices shared by both;
  the statistic is a **mean** over the resampled seeds x episodes
  (:func:`rld.eval.stats.paired_bootstrap`; for H4 :func:`~rld.eval.stats.drop_difference_bootstrap`
  with ``estimator="mean"``, ``"shared_episodes"``, which is the same per-episode paired
  mean). The IQM-over-seeds variant of each is printed beside it with role
  :data:`ROLE_POSTHOC` ("post hoc, not scored").
* **H2** -- IQM success (the only hypothesis that names IQM): seeds resampled per method and
  shared by the method's two cells; inside a cell ``ppo`` and ``residual_ppo`` share the
  resampled episode indices; between ``id`` SS5 and ``unseen_seastate`` SS6 the indices are
  independent.
* **H1a / H3 closing speed** -- the paired relative-p95 (P3-D1 §4, P3-D4 #6), p95 by NumPy's
  ``linear`` estimator (P7-D1a #9); the P3-D4 #8 variant ranks only ``timeout`` episodes as
  the worst closing speed (a ``crash`` without contact stays excluded) and is never scored
  (P7-D1a #8).

Scoring rules (as written; fractions, so 0.15 = 15 %, 0.05 = 5 points)
---------------------------------------------------------------------
"The CI excludes 0" means, for a predicted improvement, that its lower bound is > 0; a CI
wholly below 0 is "not supported" (P7-D1a #6).

* **H1a** (P3-D1 §8, P3-D4 #5): ``r = 1 - p95(residual_ppo) / p95(pid_feedforward_lowvz)``
  at ``id`` SS5; non-inferiority: the lower 95 % bound of the mean paired success difference
  ``residual_ppo - lowvz`` >= -0.02. Supported: r >= 0.15, lower bound of r > 0, NI holds.
  Inconclusive: 0 < r < 0.15, lower bound > 0, NI holds. Not supported: every other case.
* **H1b**: mean paired success(``residual_ppo``) - success(``pid_feedforward``) at ``id`` SS6.
  Supported: >= 0.05 and lower bound > 0. Inconclusive: positive, lower bound > 0, < 0.05.
  Not supported: CI includes 0 or negative (:func:`verdict_magnitude`).
* **H2** and **H4** (P7-D1a #1, user 2026-10-02; :func:`verdict_p7d1a`): supported: point
  >= 0.10 and lower bound > 0; inconclusive: lower bound > 0 and point < 0.10; not supported:
  every other case. H2: drop(``ppo``) - drop(``residual_ppo``), drop = IQM success at ``id``
  SS5 - at ``unseen_seastate`` SS6. H4 (P6-D6): drop_sin - drop_jon at ``id`` SS5, one shared
  episode set for both motions and both methods. H4's novelty claim is withdrawn if the CI
  includes 0 (P3-D1 §8) **or** D0.4's transfer trigger fires: ``ppo_sinusoid``'s JONSWAP
  ``id`` SS5 success >= ``ppo``'s (P7-D1a #7; read on the scored mean, the IQM beside it).
* **H3** (P7-D1a #2, #5): r(``ppo_forecast`` vs ``ppo``) at ``id`` SS5 and at ``id`` SS6,
  each its own verdict by H1a's point-estimate / CI rule at 0.10, with **no**
  non-inferiority term; the ``unseen_vessel`` half-rule per SS, its own verdict, on point
  estimates, "not applicable" when r(``id``) <= 0 (P7-D1 §2). No conjunction row. The same
  parts for ``residual_ppo_forecast`` vs ``residual_ppo`` are the secondary.
* **H5**: "pending — scored at Gate 8" (P7-D1 §7).

Units: success rates and differences are fractions in [-1, 1]; closing speeds m/s model
scale; r dimensionless.
"""

import csv
import io
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dmf.typedefs import FloatArray

from rld.eval import storage
from rld.eval.arms import (
    BASELINES,
    E07_DIR,
    LEARNED_METHODS,
    NOISE_GRID,
    RESULTS_DIR,
    conditions,
)
from rld.eval.stats import (
    DEFAULT_BOOTSTRAP_SEED,
    PAIRED_REPS,
    DropContrast,
    PairedContrast,
    RelativeP95,
    drop_difference_bootstrap,
    paired_bootstrap,
    paired_iqm_difference,
    paired_relative_p95,
)

__all__ = [
    "CAVEATS",
    "CONTRAST_COLUMNS",
    "HYPOTHESIS_COLUMNS",
    "EpisodeStore",
    "check_hypotheses",
    "compute",
    "verdict_h1a",
    "verdict_half_rule",
    "verdict_magnitude",
    "verdict_p7d1a",
    "write_hypotheses",
]

SUPPORTED = "supported"
NOT_SUPPORTED = "not supported"
INCONCLUSIVE = "inconclusive"
HOLDS = "holds"
FAILS = "fails"
NOT_APPLICABLE = "not applicable — no id gain to shrink"
PENDING_H5 = "pending — scored at Gate 8"
NOT_SCORED = "not scored"

#: Thresholds as fractions (P3-D1 §8).
H1A_R_MIN: float = 0.15
H1A_NI_MARGIN: float = -0.02
H1B_MIN: float = 0.05
H2_MIN: float = 0.10
H3_R_MIN: float = 0.10
H3_HALF: float = 0.5
H4_MIN: float = 0.10

#: Caveat keys -> text (P7-D1 §8 and the reporting rules), rendered as a legend.
CAVEATS: dict[str, str] = {
    "simulation-only": "Simulation only (PyBullet, dmf deck motion); no real flight, no real "
    "deck data.",
    "P6-D1-forecast": "Forecast methods: the forecaster's forecasts were in-sample during "
    "training, and the past-only ship-motion feed is an extra ideal sensor the other methods "
    "lack (P6-D1).",
    "P6-D5-residual-descent": "The residual methods descend harder than their base; no residual "
    "seed learned the post-contact throttle cut, and a lowvz-like descent was within the "
    "residual's authority (P6-D5, corrected at P6-D6 M1).",
    "two-phase-descent": "The learned methods land with a two-phase descent that the PID tuning "
    "space cannot express.",
    "bounce-grace-50ms": "`bounce` is decided by the 50 ms contact-loss grace rule and is unstable "
    "at 240 Hz (P5-D1).",
    "closing-speed-7pct": "Recorded closing speed understates impact speed by about 7 % (read "
    "after the first contact substep's impulse; P5-D14), for every method alike.",
    "tunnelling-any-substep": "Penetration is flagged at any contact substep, not only at first "
    "contact (P5-D14).",
    "P6-D5-tunnelling-bound": "Up to 3 / 1 / 0 / 3 SS6 successes per 1 000 of residual_ppo / "
    "ppo_forecast / residual_ppo_forecast / ppo_sinusoid may depend on tunnelling overlap "
    "(P6-D5); ppo 2, sac 41 (all SS), pid_feedforward_lowvz_cut 1 (P5-D14).",
    "tilt-only-hard-landings": "Every hard landing of the four Phase 6 methods, ppo and "
    "pid_feedforward in e06 is a deck-tilt event (relative tilt > 15 deg with a nearly level "
    "drone); sac's and pid_track_descend's are mostly speed-driven (P6-D5).",
    "H1-narrowed": "H1 reads 'lands softer than the lowest-closing-speed gain set in the P3-D3 "
    "log (constant-descent law), without losing success' -- not 'softer than any PID' "
    "(P3-D4 MAJOR-1).",
    "H1b-out-of-distribution": "id SS6 is outside every method's training distribution, so H1b "
    "is a sea-state extrapolation result (P3-D4 #7).",
    "H4-bounded": "H4 is arithmetically bounded at <= 0 at id SS5: ppo_sinusoid and ppo are both "
    "200/200 on JONSWAP id SS5 in every seed (P6-D6); the user kept H4 at id SS5.",
    "no-multiplicity": "No multiplicity correction across H1a, H1b, H2, H3 and H4 (P3-D4 #9).",
    "shared-bootstrap-seed": "Every contrast uses bootstrap seed 20260926 (P3-D1 §4), so "
    "replicate draws are correlated across contrasts; stated, not corrected (P7-D1a #11).",
    "regimes-overlap": "The regimes share realizations (P3-D1 §2); a regime-vs-regime "
    "difference is not a comparison of independent draws.",
    "sac-budget": "SAC trained for 2 M env steps against 10 M for the PPO family (P3-D1 §5).",
    "noise-feed-ideal": "In the perception arm the forecast methods' ship-motion feed stays ideal "
    "(P7-D1 §4).",
    "lambda-t0-redrawn": "At lambda != 1/25 the start offset t0 is re-drawn by the environment, so "
    "the episodes are not the listed ones; contrasts against 1/25 resample episodes "
    "independently per lambda (P7-D1 §3).",
    "privileged-oracle": "oracle_gated is a commit-timing oracle (privileged): it reads the true "
    "future deck motion.",
}

_H1_CAVEATS = (
    "P6-D5-residual-descent",
    "two-phase-descent",
    "bounce-grace-50ms",
    "closing-speed-7pct",
    "tunnelling-any-substep",
    "P6-D5-tunnelling-bound",
    "tilt-only-hard-landings",
    "H1-narrowed",
    "no-multiplicity",
    "shared-bootstrap-seed",
)

#: Caveats of every scored statistical verdict.
_STAT_CAVEATS: tuple[str, ...] = ("no-multiplicity", "shared-bootstrap-seed")

CONTRAST_COLUMNS: tuple[str, ...] = (
    "contrast",
    "role",
    "hypothesis",
    "statistic",
    "description",
    "method_a",
    "method_b",
    "cell_1",
    "cell_2",
    "point",
    "ci_lo",
    "ci_hi",
    "separates",
    "value_a1",
    "value_a2",
    "value_b1",
    "value_b2",
    "n_episodes_1",
    "n_episodes_2",
    "n_seeds_a",
    "n_seeds_b",
    "n_touchdowns_a",
    "n_touchdowns_b",
    "n_timeouts_a",
    "n_timeouts_b",
    "n_reps_undefined",
    "reps",
    "bootstrap_seed",
    "confidence",
    "resampling",
    "sources",
    "note",
)

HYPOTHESIS_COLUMNS: tuple[str, ...] = (
    "hypothesis",
    "part",
    "role",
    "cell",
    "prediction",
    "statistic",
    "threshold",
    "point",
    "ci_lo",
    "ci_hi",
    "separates",
    "non_inferiority_lo",
    "non_inferiority_margin",
    "verdict",
    "rule",
    "rule_source",
    "contrasts",
    "caveats",
    "note",
)

ROLE_SCORED = "scored"
ROLE_SECONDARY = "secondary (pre-registered, reported beside)"
ROLE_SENSITIVITY = "sensitivity (P3-D4 #8; not scored)"
ROLE_POSTHOC = "post hoc, not scored"
ROLE_DESCRIPTIVE = "descriptive (not scored)"
ROLE_PENDING = "pending"


# --------------------------------------------------------------------------- verdict rules


def verdict_magnitude(point: float, lo: float, hi: float, threshold: float) -> str:
    """Return the verdict of H1b (P3-D1 §8), and of H1a / H3's r rule (P3-D4 #5, P7-D1a #5).

    Supported: ``point >= threshold`` and the lower bound > 0. Inconclusive: ``0 < point <
    threshold`` and the lower bound > 0. Not supported: the CI includes 0, or the point
    estimate is <= 0 (a CI wholly below 0 included; P7-D1a #6).

    Args:
        point: Point estimate.
        lo: Lower 95 % bound.
        hi: Upper 95 % bound.
        threshold: The predicted magnitude (> 0).

    Returns:
        ``"supported"``, ``"inconclusive"`` or ``"not supported"``.
    """
    del hi
    if not lo > 0.0 or not point > 0.0:
        return NOT_SUPPORTED
    return SUPPORTED if point >= threshold else INCONCLUSIVE


def verdict_p7d1a(point: float, lo: float, hi: float, threshold: float) -> str:
    """Return the H2 / H4 verdict, exactly as P7-D1a #1 words it (user, 2026-10-02).

    Supported: ``point >= threshold`` **and** the lower bound > 0. Inconclusive: the lower
    bound > 0 but ``point < threshold``. Not supported: every other case. It differs from
    :func:`verdict_magnitude` only when the lower bound is > 0 while the point estimate is
    <= 0 (a percentile interval need not contain its point): inconclusive here, not supported
    there.

    Args:
        point: Point estimate.
        lo: Lower 95 % bound.
        hi: Upper 95 % bound.
        threshold: The predicted magnitude, 0.10 for both.

    Returns:
        ``"supported"``, ``"inconclusive"`` or ``"not supported"``.
    """
    del hi
    if not lo > 0.0:
        return NOT_SUPPORTED
    return SUPPORTED if point >= threshold else INCONCLUSIVE


def verdict_h1a(r: float, lo: float, hi: float, ni_lo: float) -> str:
    """Return H1a's verdict (P3-D4 #5).

    Args:
        r: Relative-p95 point estimate.
        lo: Lower 95 % bound of r.
        hi: Upper 95 % bound of r.
        ni_lo: Lower 95 % bound of success(residual_ppo) - success(lowvz), P3-D1 §4's mean
            paired bootstrap (P7-D1a #3).

    Returns:
        ``"supported"`` (r >= 0.15, CI of r excludes 0, NI holds), ``"inconclusive"``
        (0 < r < 0.15, CI excludes 0, NI holds) or ``"not supported"`` (everything else).
    """
    if not ni_lo >= H1A_NI_MARGIN:
        return NOT_SUPPORTED
    return verdict_magnitude(r, lo, hi, H1A_R_MIN)


def verdict_half_rule(r_id: float, r_unseen: float) -> str:
    """Return H3's ``unseen_vessel`` part for one sea state (P7-D1 §2, point estimates).

    Args:
        r_id: r(``id``, SS).
        r_unseen: r(``unseen_vessel``, SS).

    Returns:
        ``"not applicable — no id gain to shrink"`` if ``r_id <= 0``; ``"holds"`` if
        ``r_unseen <= 0.5 r_id``; ``"fails"`` otherwise.
    """
    if not r_id > 0.0:
        return NOT_APPLICABLE
    return HOLDS if r_unseen <= H3_HALF * r_id else FAILS


# --------------------------------------------------------------------------- episode data


@dataclass(frozen=True)
class Cell:
    """One (source, method, pad, regime, sea state) cell as seeds x episodes arrays.

    Attributes:
        source: The ``episodes.csv`` it came from (repository-relative).
        method: Method.
        pad: Effective pad.
        regime: Regime.
        ss: Sea state label.
        seeds: Run seeds, the row order.
        index: Listed episode index of each column.
        episode_seed: Episode seed of each column (the pairing key).
        t0: Start offset of each column, seconds model scale (first seed).
        success: ``(n_seeds, n_episodes)`` 1.0 / 0.0.
        speed: Touchdown closing speed along the deck normal, m/s model scale, NaN without
            a contact touchdown (as ``rld.eval.metrics.cell_metrics`` reads it).
        timeout: ``timeout`` outcome mask.
    """

    source: str
    method: str
    pad: str
    regime: str
    ss: str
    seeds: tuple[int, ...]
    index: tuple[int, ...]
    episode_seed: tuple[int, ...]
    t0: tuple[float, ...]
    success: FloatArray
    speed: FloatArray
    timeout: FloatArray

    @property
    def label(self) -> str:
        """``source|pad|regime|ss``."""
        return f"{self.source}|{self.pad}|{self.regime}|{self.ss}"


_NEEDED: tuple[str, ...] = (
    "method",
    "run_seed",
    "pad",
    "regime",
    "ss",
    "index",
    "episode_seed",
    "t0_model_s",
    "outcome",
    "touchdown_contact",
    "rel_vz_normal_m_s",
)


class EpisodeStore:
    """Reads committed ``episodes.csv`` files once and serves :class:`Cell` arrays."""

    def __init__(self, repo_root: Path) -> None:
        """Create an empty store.

        Args:
            repo_root: Paths in :attr:`Cell.source` are relative to it.
        """
        self._root = repo_root.resolve()
        self._rows: dict[Path, dict[tuple[str, str, str, str], list[list[str]]]] = {}

    def _load(self, path: Path) -> dict[tuple[str, str, str, str], list[list[str]]]:
        """Index one file's rows by (method, pad, regime, ss), keeping only needed columns.

        ``path`` is the logical ``episodes.csv``; its ``.gz`` is read when present.
        """
        path = path.resolve()
        if path not in self._rows:
            groups: dict[tuple[str, str, str, str], list[list[str]]] = {}
            with storage.open_text(path) as handle:
                reader = csv.reader(handle)
                header = next(reader)
                pos = [header.index(c) for c in _NEEDED]
                for raw in reader:
                    vals = [raw[i] for i in pos]
                    groups.setdefault((vals[0], vals[2], vals[3], vals[4]), []).append(vals)
            self._rows[path] = groups
        return self._rows[path]

    def has(self, path: Path) -> bool:
        """Return whether the logical CSV ``path`` exists (compressed or not)."""
        return storage.exists(path)

    def cell(self, path: Path, method: str, pad: str, regime: str, ss: str) -> Cell:
        """Return one cell's arrays.

        Raises:
            KeyError: If the file has no rows for the cell.
            ValueError: If the seeds do not cover the identical episodes in the same order.
        """
        rows = self._load(path).get((method, pad, regime, ss))
        if not rows:
            raise KeyError(f"{path}: no rows for {method}/{pad}/{regime}/{ss}")
        by_seed: dict[int, list[list[str]]] = {}
        for vals in rows:
            by_seed.setdefault(int(vals[1]), []).append(vals)
        seeds = tuple(sorted(by_seed))
        ordered = {s: sorted(by_seed[s], key=lambda v: int(v[5])) for s in seeds}
        first = ordered[seeds[0]]
        index = tuple(int(v[5]) for v in first)
        eps = tuple(int(v[6]) for v in first)
        for s in seeds:
            if (
                tuple(int(v[5]) for v in ordered[s]) != index
                or tuple(int(v[6]) for v in ordered[s]) != eps
            ):
                raise ValueError(f"{path}: {method} seeds cover different episodes")
        success = np.array([[v[8] == "success" for v in ordered[s]] for s in seeds], dtype=float)
        timeout = np.array([[v[8] == "timeout" for v in ordered[s]] for s in seeds], dtype=float)
        speed = np.array(
            [
                [max(0.0, -float(v[10])) if v[9] == "True" else np.nan for v in ordered[s]]
                for s in seeds
            ],
            dtype=float,
        )
        return Cell(
            source=self._rel(path),
            method=method,
            pad=pad,
            regime=regime,
            ss=ss,
            seeds=seeds,
            index=index,
            episode_seed=eps,
            t0=tuple(float(v[7]) for v in first),
            success=success,
            speed=speed,
            timeout=timeout,
        )

    def _rel(self, path: Path) -> str:
        resolved = path.resolve()
        return (
            str(resolved.relative_to(self._root))
            if resolved.is_relative_to(self._root)
            else str(resolved)
        )


def _restrict(cell: Cell, keep: Sequence[int]) -> Cell:
    """Return ``cell`` restricted to the listed episode indices ``keep`` (in that order)."""
    pos = {idx: k for k, idx in enumerate(cell.index)}
    cols = [pos[idx] for idx in keep]
    return Cell(
        source=cell.source,
        method=cell.method,
        pad=cell.pad,
        regime=cell.regime,
        ss=cell.ss,
        seeds=cell.seeds,
        index=tuple(keep),
        episode_seed=tuple(cell.episode_seed[c] for c in cols),
        t0=tuple(cell.t0[c] for c in cols),
        success=cell.success[:, cols],
        speed=cell.speed[:, cols],
        timeout=cell.timeout[:, cols],
    )


def align(*cells: Cell, same_t0: bool = False, allow_subset: bool = False) -> tuple[Cell, ...]:
    """Return cells over the identical listed episodes, in one column order.

    Args:
        *cells: Cells that must be the same listed episodes (same index and episode seed).
        same_t0: Also require the same start offsets (two conditions of one episode).
        allow_subset: Restrict to the common episodes instead of refusing a difference
            (scratch smoke flights only, which fly the first K episodes per cell).

    Returns:
        The cells, column-aligned.

    Raises:
        ValueError: If the cells are not the identical episodes (or do not start alike).
    """
    common = [i for i in cells[0].index if all(i in c.index for c in cells[1:])]
    if not allow_subset and any(c.index != tuple(common) for c in cells):
        raise ValueError(f"{[c.label for c in cells]} are not the identical episodes")
    if not common:
        raise ValueError(f"{[c.label for c in cells]} share no episode")
    out = tuple(_restrict(c, common) for c in cells)
    for c in out[1:]:
        if c.episode_seed != out[0].episode_seed:
            raise ValueError(f"{out[0].label} and {c.label} differ in episode seeds")
        if same_t0 and c.t0 != out[0].t0:
            raise ValueError(f"{out[0].label} and {c.label} start at different t0")
    return out


# --------------------------------------------------------------------------- sources


@dataclass(frozen=True)
class Sources:
    """Where each condition's per-episode rows live."""

    e07: Path
    results: Path

    def clean(self, method: str, pad: str = "aft") -> Path:
        """Rows of a method at lambda 1/25, JONSWAP, noise off, on ``pad``."""
        if pad == "aft":
            if method in LEARNED_METHODS:
                return self.e07 / "matrix" / "episodes.csv"
            if method == "pid_feedforward_lowvz_cut":
                return self.results / "e01_lowvz_cut" / "episodes.csv"
            return self.results / "e01" / "episodes.csv"
        if method in LEARNED_METHODS or method == "pid_feedforward_lowvz_cut":
            return self.e07 / "cg" / "episodes.csv"
        return self.results / "e02" / "episodes.csv"

    def condition(self, key: str) -> Path:
        """Rows of a written Phase 7 condition, e.g. ``noise/sigma1cm_lat0step``."""
        return self.e07 / key / "episodes.csv"


# --------------------------------------------------------------------------- contrast rows


def _fmt(value: float | int | bool | str) -> str:
    """Write a value as ``rld.eval.report.write_rows`` does (``repr`` floats)."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def _row(**values: Any) -> dict[str, str]:
    """Return a contrast row with every column (missing values empty)."""
    return {c: _fmt(values[c]) if c in values else "" for c in CONTRAST_COLUMNS}


def _relp95_row(
    cid: str,
    role: str,
    hypothesis: str,
    description: str,
    a: Cell,
    b: Cell,
    res: RelativeP95,
) -> dict[str, str]:
    return _row(
        contrast=cid,
        role=role,
        hypothesis=hypothesis,
        statistic="relative_p95_timeouts_worst" if res.timeouts_as_worst else "relative_p95",
        description=description,
        method_a=a.method,
        method_b=b.method,
        cell_1=f"{a.pad}|{a.regime}|{a.ss}",
        point=res.r,
        ci_lo=res.lo,
        ci_hi=res.hi,
        separates=res.separates,
        value_a1=res.p95_a,
        value_b1=res.p95_b,
        n_episodes_1=res.n_episodes,
        n_seeds_a=res.n_seeds_a,
        n_seeds_b=res.n_seeds_b,
        n_touchdowns_a=res.n_touchdowns_a,
        n_touchdowns_b=res.n_touchdowns_b,
        n_timeouts_a=res.n_timeouts_a,
        n_timeouts_b=res.n_timeouts_b,
        n_reps_undefined=res.n_reps_undefined,
        reps=res.reps,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        confidence=res.confidence,
        resampling="seeds per method, then one set of episode indices shared by both methods; "
        "p95 over each method's pooled resampled touched-down episodes (P3-D1 §4, P3-D4 #6)"
        + (
            "; timeouts ranked as the worst closing speed (P3-D4 #8)"
            if res.timeouts_as_worst
            else ""
        ),
        sources=f"{a.source};{b.source}",
        note="r = 1 - p95_a / p95_b; value_a1 / value_b1 are the pooled p95 closing speeds, m/s "
        "model scale",
    )


def _diff_row(
    cid: str,
    role: str,
    hypothesis: str,
    description: str,
    a: Cell,
    b: Cell,
    res: PairedContrast,
    statistic: str,
    resampling: str,
) -> dict[str, str]:
    return _row(
        contrast=cid,
        role=role,
        hypothesis=hypothesis,
        statistic=statistic,
        description=description,
        method_a=a.method,
        method_b=b.method,
        cell_1=f"{a.pad}|{a.regime}|{a.ss}",
        point=res.diff,
        ci_lo=res.lo,
        ci_hi=res.hi,
        separates=res.separates,
        n_episodes_1=res.n_episodes,
        n_seeds_a=res.n_seeds_a,
        n_seeds_b=res.n_seeds_b,
        reps=res.reps,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        confidence=res.confidence,
        resampling=resampling,
        sources=f"{a.source};{b.source}",
        note="success fractions; a minus b",
    )


def _drop_row(
    cid: str,
    role: str,
    hypothesis: str,
    description: str,
    a1: Cell,
    a2: Cell,
    res: DropContrast,
    b1: Cell | None = None,
    b2: Cell | None = None,
) -> dict[str, str]:
    design = (
        "seeds per method, shared by its two cells; episodes resampled independently per cell, "
        "shared by both methods within a cell (P7-D1 §2; P7-D1a #4)"
        if res.mode == "independent_cells"
        else "seeds per method, shared by its two cells; one set of episode indices shared by "
        "both cells and both methods (P7-D1 §2)"
    )
    sources = [a1.source, a2.source] + ([] if b1 is None or b2 is None else [b1.source, b2.source])
    success = (
        "success = mean over seeds x episodes (P3-D1 §4 per-episode paired mean)"
        if res.estimator == "mean"
        else "success = IQM over seeds of per-seed success rate"
    )
    statistic = "drop_difference" if b1 is not None else "drop"
    return _row(
        contrast=cid,
        role=role,
        hypothesis=hypothesis,
        statistic=statistic + ("_mean" if res.estimator == "mean" else ""),
        description=description,
        method_a=a1.method,
        method_b="" if b1 is None else b1.method,
        cell_1=f"{a1.source}|{a1.pad}|{a1.regime}|{a1.ss}",
        cell_2=f"{a2.source}|{a2.pad}|{a2.regime}|{a2.ss}",
        point=res.statistic,
        ci_lo=res.lo,
        ci_hi=res.hi,
        separates=res.separates,
        value_a1=res.success_a1,
        value_a2=res.success_a2,
        value_b1=res.success_b1,
        value_b2=res.success_b2,
        n_episodes_1=res.n_episodes_1,
        n_episodes_2=res.n_episodes_2,
        n_seeds_a=res.n_seeds_a,
        n_seeds_b=res.n_seeds_b,
        reps=res.reps,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        confidence=res.confidence,
        resampling=f"{res.mode}: {design}",
        sources=";".join(dict.fromkeys(sources)),
        note=f"{success}; drop = cell 1 - cell 2"
        + ("; statistic = drop(a) - drop(b)" if b1 is not None else ""),
    )


# --------------------------------------------------------------------------- compute


@dataclass
class _Ctx:
    store: EpisodeStore
    src: Sources
    reps: int
    contrasts: list[dict[str, str]]
    hypotheses: list[dict[str, str]]
    missing: list[str]
    allow_subset: bool = False

    def align(self, *cells: Cell, same_t0: bool = False) -> tuple[Cell, ...]:
        return align(*cells, same_t0=same_t0, allow_subset=self.allow_subset)

    def get(self, path: Path, method: str, pad: str, regime: str, ss: str) -> Cell | None:
        if not self.store.has(path):
            self.missing.append(f"{path}: absent")
            return None
        try:
            return self.store.cell(path, method, pad, regime, ss)
        except KeyError as exc:
            self.missing.append(str(exc))
            return None


def _hyp(**values: Any) -> dict[str, str]:
    return {c: _fmt(values[c]) if c in values else "" for c in HYPOTHESIS_COLUMNS}


def _missing_hyp(ctx: _Ctx, hypothesis: str, part: str, role: str, cell: str) -> None:
    ctx.hypotheses.append(
        _hyp(
            hypothesis=hypothesis,
            part=part,
            role=role,
            cell=cell,
            verdict=f"{NOT_SCORED} — missing data",
            note="; ".join(ctx.missing[-3:]),
        )
    )


def _ni_rows(
    ctx: _Ctx,
    tag: str,
    hypothesis: str,
    desc: str,
    a: Cell,
    b: Cell,
) -> tuple[PairedContrast, PairedContrast]:
    """Append the scored mean success contrast and its post-hoc IQM twin; return both."""
    mean = paired_bootstrap(a.success, b.success, reps=ctx.reps)
    iqm = paired_iqm_difference(a.success, b.success, reps=ctx.reps)
    ctx.contrasts += [
        _diff_row(
            tag,
            ROLE_SCORED,
            hypothesis,
            desc,
            a,
            b,
            mean,
            "mean_success_difference",
            "rld.eval.stats.paired_bootstrap: seeds per method, then one set of episode indices "
            "shared by both; per-episode differences averaged over the resampled seeds x "
            "episodes (P3-D1 §4; P7-D1a #3)",
        ),
        _diff_row(
            tag + ".iqm",
            ROLE_POSTHOC,
            hypothesis,
            desc + ": IQM over seeds instead of the mean",
            a,
            b,
            iqm,
            "iqm_success_difference",
            "seeds per method, then one set of episode indices shared by both; success = IQM "
            "over seeds of per-seed rates (P7-D1 §2 reading, superseded by P7-D1a #3)",
        ),
    ]
    return mean, iqm


def _posthoc_hyp(
    ctx: _Ctx, hypothesis: str, part: str, cell: str, statistic: str, cid: str, res: Any
) -> None:
    """Append the "post hoc, not scored" IQM line printed beside a scored part (P7-D1a #3)."""
    point = res.diff if isinstance(res, PairedContrast) else res.statistic
    ctx.hypotheses.append(
        _hyp(
            hypothesis=hypothesis,
            part=part,
            role=ROLE_POSTHOC,
            cell=cell,
            statistic=statistic,
            point=point,
            ci_lo=res.lo,
            ci_hi=res.hi,
            separates=res.separates,
            verdict=NOT_SCORED,
            rule_source="P7-D1a #3: the IQM-based value, printed beside the scored mean",
            contrasts=cid,
        )
    )


def _h1(ctx: _Ctx) -> None:
    src = ctx.src
    res_ss5 = ctx.get(src.clean("residual_ppo"), "residual_ppo", "aft", "id", "SS5")
    low = ctx.get(src.clean("pid_feedforward_lowvz"), "pid_feedforward_lowvz", "aft", "id", "SS5")
    if res_ss5 is None or low is None:
        _missing_hyp(ctx, "H1a", "relative p95 + non-inferiority", ROLE_SCORED, "id SS5")
    else:
        res_ss5, low = ctx.align(res_ss5, low)
        r = paired_relative_p95(res_ss5.speed, low.speed, reps=ctx.reps)
        sens = paired_relative_p95(
            res_ss5.speed,
            low.speed,
            a_timeouts=res_ss5.timeout,
            b_timeouts=low.timeout,
            timeouts_as_worst=True,
            reps=ctx.reps,
        )
        desc = "residual_ppo vs pid_feedforward_lowvz, id SS5, aft"
        ctx.contrasts.append(_relp95_row("H1a.r", ROLE_SCORED, "H1a", desc, res_ss5, low, r))
        ni, ni_iqm = _ni_rows(
            ctx, "H1a.ni", "H1a", desc + ": success non-inferiority", res_ss5, low
        )
        ctx.contrasts.append(
            _relp95_row("H1a.r.timeouts", ROLE_SENSITIVITY, "H1a", desc, res_ss5, low, sens)
        )
        verdict = verdict_h1a(r.r, r.lo, r.hi, ni.lo)
        ctx.hypotheses.append(
            _hyp(
                hypothesis="H1a",
                part="relative p95 + non-inferiority",
                role=ROLE_SCORED,
                cell="id SS5, aft",
                prediction="r >= 15 % (0.188 -> <= 0.160 m/s), CI of r excludes 0, and the "
                "lower 95 % bound of the paired success difference >= -2 points",
                statistic="relative_p95 r = 1 - p95(residual_ppo) / p95(pid_feedforward_lowvz); "
                "non-inferiority: mean paired success difference",
                threshold=H1A_R_MIN,
                point=r.r,
                ci_lo=r.lo,
                ci_hi=r.hi,
                separates=r.separates,
                non_inferiority_lo=ni.lo,
                non_inferiority_margin=H1A_NI_MARGIN,
                verdict=verdict,
                rule="supported: r >= 0.15, lower bound of r > 0, NI holds; inconclusive: "
                "0 < r < 0.15, lower bound > 0, NI holds; not supported: every other case",
                rule_source="P3-D1 §8 H1a; P3-D4 #5; P7-D1a #3 (NI: mean paired bootstrap), #6",
                contrasts="H1a.r;H1a.ni",
                caveats=";".join(_H1_CAVEATS),
                note=f"p95 residual_ppo {r.p95_a:.4f} m/s, lowvz {r.p95_b:.4f} m/s; success "
                f"difference (mean, scored) {ni.diff:+.4f} [{ni.lo:+.4f}, {ni.hi:+.4f}]; "
                f"IQM over seeds (post hoc, not scored) {ni_iqm.diff:+.4f} "
                f"[{ni_iqm.lo:+.4f}, {ni_iqm.hi:+.4f}]",
            )
        )
        _posthoc_hyp(
            ctx,
            "H1a",
            "non-inferiority, IQM over seeds",
            "id SS5, aft",
            "IQM success(residual_ppo) - success(pid_feedforward_lowvz)",
            "H1a.ni.iqm",
            ni_iqm,
        )
        ctx.hypotheses.append(
            _hyp(
                hypothesis="H1a",
                part="relative p95 with timeouts ranked worst",
                role=ROLE_SENSITIVITY,
                cell="id SS5, aft",
                statistic="relative_p95_timeouts_worst",
                point=sens.r,
                ci_lo=sens.lo,
                ci_hi=sens.hi,
                separates=sens.separates,
                verdict=NOT_SCORED,
                rule_source="P3-D4 #8; P7-D1a #8 (only timeouts ranked worst)",
                contrasts="H1a.r.timeouts",
                note=f"timeouts: residual_ppo {sens.n_timeouts_a}, lowvz {sens.n_timeouts_b}",
            )
        )
    res_ss6 = ctx.get(src.clean("residual_ppo"), "residual_ppo", "aft", "id", "SS6")
    ff = ctx.get(src.clean("pid_feedforward"), "pid_feedforward", "aft", "id", "SS6")
    if res_ss6 is None or ff is None:
        _missing_hyp(ctx, "H1b", "success difference", ROLE_SCORED, "id SS6")
        return
    res_ss6, ff = ctx.align(res_ss6, ff)
    d, d_iqm = _ni_rows(
        ctx, "H1b.d", "H1b", "residual_ppo vs pid_feedforward, id SS6, aft", res_ss6, ff
    )
    ctx.hypotheses.append(
        _hyp(
            hypothesis="H1b",
            part="success difference",
            role=ROLE_SCORED,
            cell="id SS6, aft",
            prediction="residual_ppo exceeds pid_feedforward by >= 5 points (paired)",
            statistic="mean paired success(residual_ppo) - success(pid_feedforward)",
            threshold=H1B_MIN,
            point=d.diff,
            ci_lo=d.lo,
            ci_hi=d.hi,
            separates=d.separates,
            verdict=verdict_magnitude(d.diff, d.lo, d.hi, H1B_MIN),
            rule="supported: >= 0.05 and lower bound > 0; inconclusive: positive, lower bound "
            "> 0, < 0.05; not supported: CI includes 0 or negative",
            rule_source="P3-D1 §8 H1b; P7-D1a #3 (mean paired bootstrap), #6",
            contrasts="H1b.d",
            caveats=";".join(("H1b-out-of-distribution", *_H1_CAVEATS)),
            note=f"IQM over seeds (post hoc, not scored) {d_iqm.diff:+.4f} "
            f"[{d_iqm.lo:+.4f}, {d_iqm.hi:+.4f}]",
        )
    )
    _posthoc_hyp(
        ctx,
        "H1b",
        "success difference, IQM over seeds",
        "id SS6, aft",
        "IQM success(residual_ppo) - success(pid_feedforward)",
        "H1b.d.iqm",
        d_iqm,
    )


def _h2(ctx: _Ctx) -> None:
    src = ctx.src
    cells = {
        (m, reg, ss): ctx.get(src.clean(m), m, "aft", reg, ss)
        for m in ("ppo", "residual_ppo")
        for reg, ss in (("id", "SS5"), ("unseen_seastate", "SS6"))
    }
    if any(c is None for c in cells.values()):
        _missing_hyp(ctx, "H2", "drop difference", ROLE_SCORED, "id SS5 -> unseen_seastate SS6")
        return
    a1, a2 = cells[("ppo", "id", "SS5")], cells[("ppo", "unseen_seastate", "SS6")]
    b1, b2 = cells[("residual_ppo", "id", "SS5")], cells[("residual_ppo", "unseen_seastate", "SS6")]
    assert a1 and a2 and b1 and b2
    a1, b1 = ctx.align(a1, b1)  # within a cell: the identical list, shared indices (P7-D1a #4)
    a2, b2 = ctx.align(a2, b2)
    res = drop_difference_bootstrap(
        a1.success, a2.success, b1.success, b2.success, mode="independent_cells", reps=ctx.reps
    )
    ctx.contrasts.append(
        _drop_row(
            "H2.dd",
            ROLE_SCORED,
            "H2",
            "drop(ppo) - drop(residual_ppo), drop = id SS5 - unseen_seastate SS6, aft",
            a1,
            a2,
            res,
            b1,
            b2,
        )
    )
    ctx.hypotheses.append(
        _hyp(
            hypothesis="H2",
            part="drop difference",
            role=ROLE_SCORED,
            cell="id SS5 -> unseen_seastate SS6, aft",
            prediction="drop(ppo) - drop(residual_ppo) >= 10 points",
            statistic="drop(m) = IQM success id SS5 - IQM success unseen_seastate SS6",
            threshold=H2_MIN,
            point=res.statistic,
            ci_lo=res.lo,
            ci_hi=res.hi,
            separates=res.separates,
            verdict=verdict_p7d1a(res.statistic, res.lo, res.hi, H2_MIN),
            rule="supported: >= 0.10 and lower bound > 0; inconclusive: lower bound > 0 and "
            "< 0.10; not supported: every other case",
            rule_source="P3-D1 §8 H2 (prediction, IQM, test); P7-D1a #1 (mapping, user "
            "2026-10-02), #3 (IQM), #4 (pairing)",
            contrasts="H2.dd",
            caveats=";".join(
                ("regimes-overlap", "two-phase-descent", "bounce-grace-50ms", *_STAT_CAVEATS)
            ),
            note=f"drop(ppo) {res.drop_a:+.4f}, drop(residual_ppo) {res.drop_b:+.4f}",
        )
    )


def _h3_family(ctx: _Ctx, a_name: str, b_name: str, role: str, tag: str) -> None:
    """Score H3's four parts for one pair, each its own verdict; no conjunction (P7-D1a #2)."""
    src = ctx.src
    r_by: dict[tuple[str, str], RelativeP95] = {}
    for reg in ("id", "unseen_vessel"):
        for ss in ("SS5", "SS6"):
            a = ctx.get(src.clean(a_name), a_name, "aft", reg, ss)
            b = ctx.get(src.clean(b_name), b_name, "aft", reg, ss)
            if a is None or b is None:
                continue
            a, b = ctx.align(a, b)
            res = paired_relative_p95(a.speed, b.speed, reps=ctx.reps)
            sens = paired_relative_p95(
                a.speed,
                b.speed,
                a_timeouts=a.timeout,
                b_timeouts=b.timeout,
                timeouts_as_worst=True,
                reps=ctx.reps,
            )
            cid = f"{tag}.r.{reg}.{ss}"
            desc = f"{a_name} vs {b_name}, {reg} {ss}, aft"
            ctx.contrasts += [
                _relp95_row(cid, role, "H3", desc, a, b, res),
                _relp95_row(cid + ".timeouts", ROLE_SENSITIVITY, "H3", desc, a, b, sens),
            ]
            r_by[(reg, ss)] = res
    caveats = ";".join(
        ("P6-D1-forecast", "closing-speed-7pct", "tunnelling-any-substep", *_STAT_CAVEATS)
    )
    for ss in ("SS5", "SS6"):
        rid = r_by.get(("id", ss))
        if rid is None:
            _missing_hyp(ctx, "H3", f"{tag}: id {ss} relative p95", role, f"id {ss}")
            continue
        ctx.hypotheses.append(
            _hyp(
                hypothesis="H3",
                part=f"{tag}: id {ss} relative p95",
                role=role,
                cell=f"id {ss}, aft",
                prediction=f"{a_name} lowers p95 closing speed relative to {b_name} by >= 10 %",
                statistic=f"relative_p95 r = 1 - p95({a_name}) / p95({b_name})",
                threshold=H3_R_MIN,
                point=rid.r,
                ci_lo=rid.lo,
                ci_hi=rid.hi,
                separates=rid.separates,
                verdict=verdict_magnitude(rid.r, rid.lo, rid.hi, H3_R_MIN),
                rule="H1a's point-estimate / CI rule at 0.10: supported: r >= 0.10 and lower "
                "bound > 0; inconclusive: 0 < r < 0.10, lower bound > 0; not supported: every "
                "other case (no non-inferiority term)",
                rule_source="P3-D1 §8 H3; P7-D1a #2 (own verdict), #5, #6",
                contrasts=f"{tag}.r.id.{ss}",
                caveats=caveats,
                note=f"p95 {a_name} {rid.p95_a:.4f} m/s, {b_name} {rid.p95_b:.4f} m/s",
            )
        )
    for ss in ("SS5", "SS6"):
        r_id, r_uv = r_by.get(("id", ss)), r_by.get(("unseen_vessel", ss))
        if r_id is None or r_uv is None:
            _missing_hyp(
                ctx, "H3", f"{tag}: unseen_vessel {ss} half-rule", role, f"unseen_vessel {ss}"
            )
            continue
        ctx.hypotheses.append(
            _hyp(
                hypothesis="H3",
                part=f"{tag}: unseen_vessel {ss} half-rule",
                role=role,
                cell=f"unseen_vessel {ss} vs id {ss}, aft",
                prediction="r(unseen_vessel) <= 0.5 r(id), same SS",
                statistic="relative_p95 point estimates",
                threshold=H3_HALF,
                point=r_uv.r,
                ci_lo=r_uv.lo,
                ci_hi=r_uv.hi,
                separates=r_uv.separates,
                verdict=verdict_half_rule(r_id.r, r_uv.r),
                rule="judged on point estimates; not applicable if r(id) <= 0",
                rule_source="P3-D1 §8 H3; P7-D1 §2; P7-D1a #2 (own verdict)",
                contrasts=f"{tag}.r.unseen_vessel.{ss};{tag}.r.id.{ss}",
                caveats=caveats + ";regimes-overlap",
                note=f"r(id {ss}) {r_id.r:+.4f} [{r_id.lo:+.4f}, {r_id.hi:+.4f}]; "
                f"r(unseen_vessel {ss}) {r_uv.r:+.4f} [{r_uv.lo:+.4f}, {r_uv.hi:+.4f}]",
            )
        )


def _h4(ctx: _Ctx) -> None:
    src = ctx.src
    sin_path = src.condition("sinusoid")
    a_sin = ctx.get(sin_path, "ppo_sinusoid", "aft", "id", "SS5")
    a_jon = ctx.get(src.clean("ppo_sinusoid"), "ppo_sinusoid", "aft", "id", "SS5")
    b_jon = ctx.get(src.clean("ppo"), "ppo", "aft", "id", "SS5")
    b_sin = ctx.get(sin_path, "ppo", "aft", "id", "SS5")
    if a_sin is None or a_jon is None or b_jon is None or b_sin is None:
        _missing_hyp(ctx, "H4", "drop difference", ROLE_SCORED, "id SS5")
        return
    a_sin, a_jon, b_jon, b_sin = ctx.align(a_sin, a_jon, b_jon, b_sin, same_t0=True)
    cells = (a_sin.success, a_jon.success, b_jon.success, b_sin.success)
    res = drop_difference_bootstrap(*cells, mode="shared_episodes", reps=ctx.reps, estimator="mean")
    res_iqm = drop_difference_bootstrap(*cells, mode="shared_episodes", reps=ctx.reps)
    desc = (
        "drop_sin - drop_jon at id SS5, aft: drop_sin = ppo_sinusoid on sinusoid - on "
        "JONSWAP; drop_jon = ppo on JONSWAP - on sinusoid"
    )
    ctx.contrasts += [
        _drop_row("H4.dd", ROLE_SCORED, "H4", desc, a_sin, a_jon, res, b_jon, b_sin),
        _drop_row(
            "H4.dd.iqm",
            ROLE_POSTHOC,
            "H4",
            desc + "; IQM over seeds instead of the mean",
            a_sin,
            a_jon,
            res_iqm,
            b_jon,
            b_sin,
        ),
    ]
    verdict = verdict_p7d1a(res.statistic, res.lo, res.hi, H4_MIN)
    ci_includes_zero = not res.lo > 0.0
    transfers = res.success_a2 >= res.success_b1  # D0.4 on the scored (mean) success
    transfers_iqm = res_iqm.success_a2 >= res_iqm.success_b1
    withdrawn = ci_includes_zero or transfers
    ctx.hypotheses.append(
        _hyp(
            hypothesis="H4",
            part="drop difference",
            role=ROLE_SCORED,
            cell="id SS5, aft",
            prediction="drop_sin - drop_jon >= 10 points, same episode lists for both motions",
            statistic="drop_sin - drop_jon (mean paired success, one shared episode set)",
            threshold=H4_MIN,
            point=res.statistic,
            ci_lo=res.lo,
            ci_hi=res.hi,
            separates=res.separates,
            verdict=verdict,
            rule="supported: >= 0.10 and lower bound > 0; inconclusive: lower bound > 0 and "
            "< 0.10; not supported: every other case",
            rule_source="P3-D1 §8 H4 (prediction, test); P6-D6 (scored as written at id SS5); "
            "P7-D1a #1 (mapping, user 2026-10-02), #3 (mean paired bootstrap), #7 (D0.4)",
            contrasts="H4.dd",
            caveats=";".join(("H4-bounded", *_STAT_CAVEATS)),
            note=f"drop_sin {res.drop_a:+.4f} (sin {res.success_a1:.4f}, "
            f"jon {res.success_a2:.4f}); "
            f"drop_jon {res.drop_b:+.4f} (jon {res.success_b1:.4f}, sin {res.success_b2:.4f}). "
            f"Novelty claim withdrawn: {withdrawn}. Triggers: P3-D1 §8 (CI does not exclude 0, "
            f"i.e. lower bound <= 0, P7-D1a #6) {ci_includes_zero}; D0.4 (ppo_sinusoid's "
            f"JONSWAP id SS5 success {res.success_a2:.4f} >= ppo's {res.success_b1:.4f}) "
            f"{transfers}; with IQM over seeds (post hoc) {res_iqm.success_a2:.4f} >= "
            f"{res_iqm.success_b1:.4f} {transfers_iqm}. IQM over seeds (post hoc, not scored) "
            f"{res_iqm.statistic:+.4f} [{res_iqm.lo:+.4f}, {res_iqm.hi:+.4f}]",
        )
    )
    _posthoc_hyp(
        ctx,
        "H4",
        "drop difference, IQM over seeds",
        "id SS5, aft",
        "drop_sin - drop_jon (IQM success over seeds, one shared episode set)",
        "H4.dd.iqm",
        res_iqm,
    )


def _h5(ctx: _Ctx) -> None:
    ctx.hypotheses.append(
        _hyp(
            hypothesis="H5",
            part="ORT CPU vs GPU p50 latency",
            role=ROLE_PENDING,
            cell="batch 1",
            prediction="ORT CPU (1 thread) p50 lower than every parity-passing GPU provider "
            "by >= 2x",
            verdict=PENDING_H5,
            rule_source="P3-D1 §8 H5; P7-D1 §7 (user decision 2026-10-01: deferred to Gate 8)",
        )
    )


def _descriptive(ctx: _Ctx) -> None:
    """The arm contrasts: CG, sinusoid, noise and lambda (descriptive, not scored)."""
    src = ctx.src
    methods = (*LEARNED_METHODS, *BASELINES)
    # Pad at CG: aft - CG on the identical episodes (P7-D1 §5).
    for reg in conditions("cg")[0].lists:
        for m in methods:
            for ss in _ss_of(reg):
                aft = ctx.get(src.clean(m, "aft"), m, "aft", reg, ss)
                cg = ctx.get(src.clean(m, "cg"), m, "cg", reg, ss)
                if aft is None or cg is None:
                    continue
                aft, cg = ctx.align(aft, cg, same_t0=True)
                res = drop_difference_bootstrap(
                    aft.success, cg.success, mode="shared_episodes", reps=ctx.reps
                )
                ctx.contrasts.append(
                    _drop_row(
                        f"cg.{m}.{reg}.{ss}",
                        ROLE_DESCRIPTIVE,
                        "",
                        f"{m}: success aft - CG, {reg} {ss} (P7-D1 §5)",
                        aft,
                        cg,
                        res,
                    )
                )
    # Sinusoid leg: JONSWAP - sinusoid on the identical episodes (P7-D1 §1).
    for m in methods:
        for ss in _ss_of("id"):
            jon = ctx.get(src.clean(m), m, "aft", "id", ss)
            sin = ctx.get(src.condition("sinusoid"), m, "aft", "id", ss)
            if jon is None or sin is None:
                continue
            jon, sin = ctx.align(jon, sin, same_t0=True)
            res = drop_difference_bootstrap(
                jon.success, sin.success, mode="shared_episodes", reps=ctx.reps
            )
            ctx.contrasts.append(
                _drop_row(
                    f"sinusoid.{m}.id.{ss}",
                    ROLE_DESCRIPTIVE,
                    "",
                    f"{m}: success JONSWAP - sinusoid, id {ss} (P7-D1 §1)",
                    jon,
                    sin,
                    res,
                )
            )
    # Perception stand-in: clean - noisy on the identical episodes (P7-D1 §4).
    for setting in NOISE_GRID:
        if setting.clean:
            continue
        key = f"noise/{setting.name}"
        for m in methods:
            for ss in _ss_of("id"):
                clean = ctx.get(src.clean(m), m, "aft", "id", ss)
                noisy = ctx.get(src.condition(key), m, "aft", "id", ss)
                if clean is None or noisy is None:
                    continue
                clean, noisy = ctx.align(clean, noisy, same_t0=True)
                res = drop_difference_bootstrap(
                    clean.success, noisy.success, mode="shared_episodes", reps=ctx.reps
                )
                ctx.contrasts.append(
                    _drop_row(
                        f"{key}.{m}.id.{ss}",
                        ROLE_DESCRIPTIVE,
                        "",
                        f"{m}: success clean - {setting.name} (sigma_p {setting.sigma_p_m} m, "
                        f"sigma_v {setting.sigma_v_m_s} m/s, {setting.latency_steps} step "
                        f"latency), id {ss} (P7-D1 §4)",
                        clean,
                        noisy,
                        res,
                    )
                )
    # Lambda: 1/25 - other lambda, episodes resampled independently per lambda (P7-D1 §3).
    for cond in conditions("lambda"):
        for m in (*cond.learned, *cond.baselines):
            for reg in cond.lists:
                for ss in _ss_of(reg):
                    base = ctx.get(src.clean(m), m, "aft", reg, ss)
                    other = ctx.get(src.condition(cond.key), m, "aft", reg, ss)
                    if base is None or other is None:
                        continue
                    base, other = ctx.align(base, other)  # same listed episodes; t0 re-drawn
                    res = drop_difference_bootstrap(
                        base.success, other.success, mode="independent_cells", reps=ctx.reps
                    )
                    ctx.contrasts.append(
                        _drop_row(
                            f"{cond.key}.{m}.{reg}.{ss}",
                            ROLE_DESCRIPTIVE,
                            "",
                            f"{m}: success lambda 1/25 - 1/{round(cond.lam_inverse)}, {reg} {ss} "
                            "(P7-D1 §3; t0 re-drawn, unpaired episodes)",
                            base,
                            other,
                            res,
                        )
                    )


def _ss_of(regime: str) -> tuple[str, ...]:
    """Return a frozen regime's sea states, committed order."""
    from rld.eval.episodes import REGIME_CELLS

    return dict(REGIME_CELLS)[regime]


def compute(
    e07: Path = E07_DIR,
    results: Path = RESULTS_DIR,
    *,
    reps: int = PAIRED_REPS,
    repo_root: Path | None = None,
    allow_subset: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    """Compute every contrast and every hypothesis row from the committed episode rows.

    Args:
        e07: The Phase 7 output root.
        results: ``results/`` (the carried baseline sources).
        reps: Bootstrap replicates (10 000; fewer only in tests).
        repo_root: Paths in ``sources`` are relative to it (default: ``results``' parent).
        allow_subset: Scratch smoke flights only: pair cells over their common episodes.

    Returns:
        ``(contrast rows, hypothesis rows, missing-data notes)``, every value as text.
    """
    store = EpisodeStore(repo_root or results.parent)
    ctx = _Ctx(store, Sources(e07, results), reps, [], [], [], allow_subset)
    _h1(ctx)
    _h2(ctx)
    _h3_family(ctx, "ppo_forecast", "ppo", ROLE_SCORED, "primary")
    _h3_family(ctx, "residual_ppo_forecast", "residual_ppo", ROLE_SECONDARY, "secondary")
    _h4(ctx)
    _h5(ctx)
    _descriptive(ctx)
    return ctx.contrasts, ctx.hypotheses, list(dict.fromkeys(ctx.missing))


def _text(rows: Iterable[Mapping[str, str]], columns: Sequence[str]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row[c] for c in columns])
    return buffer.getvalue()


def _texts(
    e07: Path, results: Path, reps: int, allow_subset: bool
) -> tuple[dict[str, str], list[str]]:
    contrasts, hypotheses, missing = compute(e07, results, reps=reps, allow_subset=allow_subset)
    return {
        "contrasts.csv": _text(contrasts, CONTRAST_COLUMNS),
        "hypotheses.csv": _text(hypotheses, HYPOTHESIS_COLUMNS),
    }, missing


def write_hypotheses(
    e07: Path = E07_DIR,
    results: Path = RESULTS_DIR,
    *,
    reps: int = PAIRED_REPS,
    allow_subset: bool = False,
) -> dict[str, Any]:
    """Write ``contrasts.csv`` and ``hypotheses.csv`` into ``e07``.

    Args:
        e07: The Phase 7 output root.
        results: ``results/``.
        reps: Bootstrap replicates.
        allow_subset: Scratch smoke flights only (:func:`compute`).

    Returns:
        ``{"files": {name: rows}, "missing": [...], "verdicts": {hypothesis/part: verdict}}``.
    """
    texts, missing = _texts(e07, results, reps, allow_subset)
    e07.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (e07 / name).write_text(text, encoding="utf-8")
    verdicts = {
        f"{r['hypothesis']}/{r['part']}": r["verdict"]
        for r in csv.DictReader(io.StringIO(texts["hypotheses.csv"]))
    }
    for key, verdict in verdicts.items():
        print(f"{key}: {verdict}")
    if missing:
        print(f"missing data ({len(missing)}): first {missing[:3]}")
    return {
        "files": {n: t.count("\n") - 1 for n, t in texts.items()},
        "missing": missing,
        "verdicts": verdicts,
    }


def check_hypotheses(
    e07: Path = E07_DIR,
    results: Path = RESULTS_DIR,
    *,
    reps: int = PAIRED_REPS,
    allow_subset: bool = False,
) -> bool:
    """Recompute both files and compare bytes; print the result.

    Args:
        e07: The Phase 7 output root.
        results: ``results/``.
        reps: Bootstrap replicates.
        allow_subset: Scratch smoke flights only (:func:`compute`).

    Returns:
        Whether both files exist and are byte-identical to the recomputation.
    """
    texts, _ = _texts(e07, results, reps, allow_subset)
    ok = True
    for name, text in texts.items():
        path = e07 / name
        same = path.is_file() and path.read_text(encoding="utf-8") == text
        print(f"[{name}] {'OK' if same else 'MISMATCH'}")
        ok = ok and same
    return ok


#: Re-exported for the renderer: the role labels in the order tables list them.
ROLE_ORDER: tuple[str, ...] = (
    ROLE_SCORED,
    ROLE_SECONDARY,
    ROLE_SENSITIVITY,
    ROLE_POSTHOC,
    ROLE_PENDING,
)
