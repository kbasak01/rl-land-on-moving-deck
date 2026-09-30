"""Per-episode paired comparison of two methods on the identical frozen episodes.

Written for P5-D3: scoring P5-D2's three pre-registered predictions for
``pid_feedforward_lowvz_cut`` against the committed ``pid_feedforward_lowvz`` rows of
``results/e01/episodes.csv``. The functions are generic over the two methods; the P5-D2
specifics (which columns are "first contact", which outcomes are decided at or before first
contact) are module constants named after the entry that fixed them.

What is computed, and how
-------------------------
* **Pairing.** Rows are matched on ``(pad, regime, ss, index)``; both methods must cover the
  identical episode set, or :func:`pair_rows` raises. ``t0_model_s``, ``realization_seed`` and
  ``episode_seed`` are asserted equal per pair.
* **First-contact identity** (P5-D2 prediction 1). The raw CSV text of every column in
  :data:`FIRST_CONTACT_COLUMNS` is compared per episode. Both files are written by
  :func:`rld.eval.report.write_rows` (``repr`` floats), so equal text is equal float64.
* **Outcomes decided at or before first contact** (P5-D2 prediction 1): an episode whose
  outcome under *either* method is in :data:`PRE_CONTACT_OUTCOMES`, or is ``crash`` without a
  contact touchdown, must carry the same ``outcome`` under both (the pre-registered test).
  Whether ``termination_reason`` also agrees is counted separately and not scored: a
  ``hard_landing`` is decided at first contact, but the episode then runs on until
  ``dwell_complete`` or ``release``, which post-contact thrust can change.
* **Paired success difference per cell**: :func:`rld.eval.stats.paired_bootstrap` on 0/1
  outcomes, P3-D1 §4 replicates and seed (one "seed" per deterministic method).
* **Ratio of event counts pooled over cells** (P5-D2 prediction 2, the bounce ratio):
  :func:`stratified_paired_ratio`. Each replicate resamples episode indices with replacement
  **within each cell** (the cells are a fixed design, never resampled), shared by both
  methods (the pairing), and takes ``sum(a) / sum(b)``. Percentile interval.
* **Paired mean difference of a per-episode metric per cell** (P5-D2 prediction 3: effort and
  jerk), again :func:`~rld.eval.stats.paired_bootstrap`, plus the median per-episode ratio,
  the statistic P5-D2's tune-pool check reported.

Units: success and outcome fractions dimensionless; ``effort_mean_sq`` in normalised action
units squared, ``action_jerk_mean`` in normalised action units (multiply by ``v_max`` =
1.5 m/s for m/s). Nothing here has a time or length scale of its own.
"""

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rld.config import REPO_ROOT
from rld.envs.touchdown import OUTCOMES
from rld.eval.episodes import REGIME_CELLS
from rld.eval.metrics import as_bool, as_float
from rld.eval.report import read_rows, sha256_file, write_rows
from rld.eval.stats import (
    DEFAULT_BOOTSTRAP_SEED,
    PAIRED_REPS,
    IntervalEstimate,
    paired_bootstrap,
    wilson_interval,
)

__all__ = [
    "FIRST_CONTACT_COLUMNS",
    "PRE_CONTACT_OUTCOMES",
    "CellPairs",
    "EpisodePair",
    "cell_comparison",
    "decided_by_first_contact",
    "pair_rows",
    "stratified_paired_ratio",
]

#: Columns fixed at first contact (P5-D2 prediction 1): ``EpisodeRecord.as_row``'s
#: first-contact record of both detectors, and the runner extras derived only from it.
FIRST_CONTACT_COLUMNS: tuple[str, ...] = (
    "touchdown_contact",
    "touchdown_analytic",
    "td_t_episode_s",
    "td_t_episode_analytic_s",
    "rel_vz_normal_m_s",
    "rel_vz_world_z_m_s",
    "closing_speed_normal_m_s",
    "lateral_offset_m",
    "rel_tilt_deg",
    "abs_tilt_deg",
    "deck_tilt_deg",
    "n_contacts",
    "detectors_disagree",
    "closing_speed_world_z_m_s",
    "time_to_touchdown_s",
    "td_in_quiescent_window",
)

#: Outcomes P5-D2 lists as decided at or before first contact (``crash`` counts only when
#: the contact detector never fired; see :func:`decided_by_first_contact`).
PRE_CONTACT_OUTCOMES: tuple[str, ...] = ("off_pad", "hard_landing", "timeout")

#: Columns that must agree for two rows to be the same listed episode.
_EPISODE_IDENTITY: tuple[str, ...] = ("t0_model_s", "realization_seed", "episode_seed")


@dataclass(frozen=True)
class EpisodePair:
    """One listed episode flown by both methods.

    Attributes:
        regime: Regime label.
        ss: Sea-state label.
        index: Index within the cell.
        a: Method ``a``'s ``episodes.csv`` row, as text.
        b: Method ``b``'s row, as text.
    """

    regime: str
    ss: str
    index: int
    a: Mapping[str, str]
    b: Mapping[str, str]


@dataclass(frozen=True)
class CellPairs:
    """All paired episodes of one (regime, sea state) cell, in index order.

    Attributes:
        regime: Regime label.
        ss: Sea-state label.
        pairs: The pairs.
    """

    regime: str
    ss: str
    pairs: tuple[EpisodePair, ...]


def pair_rows(
    rows_a: Sequence[Mapping[str, str]],
    rows_b: Sequence[Mapping[str, str]],
    method_a: str,
    method_b: str,
    *,
    pad: str = "aft",
) -> list[CellPairs]:
    """Match two methods' rows on the identical listed episodes.

    Args:
        rows_a: ``episodes.csv`` rows holding ``method_a`` (other methods are ignored).
        rows_b: ``episodes.csv`` rows holding ``method_b``.
        method_a: Method ``a``.
        method_b: Method ``b``.
        pad: Effective pad to pair on.

    Returns:
        One :class:`CellPairs` per cell present, in the committed cell order.

    Raises:
        ValueError: If a method has a duplicate episode, the two episode sets differ, or a
            paired episode's start offset or seeds differ.
    """

    def keyed(rows: Sequence[Mapping[str, str]], method: str) -> dict[tuple[str, str, int], Any]:
        out: dict[tuple[str, str, int], Mapping[str, str]] = {}
        for row in rows:
            if row["method"] != method or (row.get("pad") or "aft") != pad:
                continue
            key = (row["regime"], row["ss"], int(row["index"]))
            if key in out:
                raise ValueError(f"{method}: duplicate episode {key}")
            out[key] = row
        return out

    a, b = keyed(rows_a, method_a), keyed(rows_b, method_b)
    if not a:
        raise ValueError(f"no {method_a} rows on pad {pad}")
    if set(a) != set(b):
        raise ValueError(
            f"episode sets differ: {len(set(a) - set(b))} only in {method_a}, "
            f"{len(set(b) - set(a))} only in {method_b}"
        )
    for key in a:
        for column in _EPISODE_IDENTITY:
            if a[key][column] != b[key][column]:
                raise ValueError(f"{key}: {column} differs ({a[key][column]} vs {b[key][column]})")
    cells: list[CellPairs] = []
    for regime, sea_states in REGIME_CELLS:
        for ss in sea_states:
            keys = sorted(k for k in a if k[0] == regime and k[1] == ss)
            if keys:
                pairs = tuple(EpisodePair(regime, ss, k[2], a[k], b[k]) for k in keys)
                cells.append(CellPairs(regime, ss, pairs))
    return cells


def decided_by_first_contact(row: Mapping[str, str]) -> bool:
    """Return whether an episode's outcome was decided at or before first contact (P5-D2).

    Args:
        row: An ``episodes.csv`` row.

    Returns:
        True for ``off_pad``, ``hard_landing`` and ``timeout``, and for a ``crash`` without a
        contact-detector touchdown.
    """
    outcome = row["outcome"]
    if outcome in PRE_CONTACT_OUTCOMES:
        return True
    return outcome == "crash" and not as_bool(row["touchdown_contact"])


def stratified_paired_ratio(
    a: Sequence[np.ndarray],
    b: Sequence[np.ndarray],
    *,
    reps: int = PAIRED_REPS,
    confidence: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> IntervalEstimate:
    """Return ``sum(a) / sum(b)`` over all strata with a paired, stratified bootstrap CI.

    Each replicate draws, independently in every stratum, as many episode indices as the
    stratum holds, with replacement; the **same** indices select from ``a`` and ``b`` (the
    pairing). Strata are never resampled.

    Args:
        a: Per stratum, method ``a``'s per-episode values (e.g. 0/1 bounce indicators).
        b: Per stratum, method ``b``'s values, same episodes in the same order.
        reps: Bootstrap replicates.
        confidence: Two-sided coverage in ``(0, 1)``.
        seed: Resampling seed.

    Returns:
        The ratio and its percentile interval (dimensionless). A replicate whose ``b`` sum is
        0 yields NaN and is dropped from the percentile; the interval is NaN if all are.

    Raises:
        ValueError: On mismatched strata, an empty input, ``sum(b) == 0`` on the observed
            data, ``reps < 1`` or a bad ``confidence``.
    """
    if len(a) != len(b) or not a:
        raise ValueError("a and b need the same, non-zero number of strata")
    arrs_a = [np.asarray(x, dtype=np.float64) for x in a]
    arrs_b = [np.asarray(x, dtype=np.float64) for x in b]
    for xa, xb in zip(arrs_a, arrs_b, strict=True):
        if xa.shape != xb.shape or xa.ndim != 1 or xa.size == 0:
            raise ValueError("each stratum needs equal-length non-empty 1-D arrays")
    if reps < 1 or not 0.0 < confidence < 1.0:
        raise ValueError(f"need reps >= 1 and 0 < confidence < 1, got {reps}, {confidence}")
    denom = float(sum(x.sum() for x in arrs_b))
    if denom == 0.0:
        raise ValueError("sum(b) is 0 on the observed data: the ratio is undefined")
    point = float(sum(x.sum() for x in arrs_a)) / denom
    rng = np.random.default_rng(seed)
    num = np.zeros(reps)
    den = np.zeros(reps)
    for xa, xb in zip(arrs_a, arrs_b, strict=True):
        idx = rng.integers(0, xa.size, size=(reps, xa.size))
        num += xa[idx].sum(axis=1)
        den += xb[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    values = values[np.isfinite(values)]
    tail = 100.0 * (1.0 - confidence) / 2.0
    if values.size == 0:
        lo, hi = float("nan"), float("nan")
    else:
        lo, hi = (float(v) for v in np.percentile(values, [tail, 100.0 - tail]))
    return IntervalEstimate(point=point, lo=lo, hi=hi, reps=reps, confidence=confidence)


def _metric(pairs: Sequence[EpisodePair], column: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(a, b)`` per-episode values of a numeric column."""
    a = np.array([as_float(p.a[column]) for p in pairs])
    b = np.array([as_float(p.b[column]) for p in pairs])
    return a, b


def cell_comparison(
    cell: CellPairs, *, reps: int = PAIRED_REPS, seed: int = DEFAULT_BOOTSTRAP_SEED
) -> dict[str, Any]:
    """Return one row of the paired table for a cell.

    Args:
        cell: The paired episodes.
        reps: Paired-bootstrap replicates.
        seed: Paired-bootstrap seed.

    Returns:
        Counts, Wilson CIs, outcome breakdowns and transitions, first-contact identity
        counts, the paired success difference ``a - b`` with its CI, and the effort and jerk
        comparisons. Column meanings are in the P5-D3 entry of ``docs/protocol.md``.
    """
    pairs = cell.pairs
    n = len(pairs)
    row: dict[str, Any] = {"regime": cell.regime, "ss": cell.ss, "n_episodes": n}
    for tag, side in (("a", "a"), ("b", "b")):
        outs = [getattr(p, side)["outcome"] for p in pairs]
        k = outs.count("success")
        lo, hi = wilson_interval(k, n)
        row[f"{tag}_n_success"] = k
        row[f"{tag}_success_rate"] = k / n
        row[f"{tag}_success_wilson_lo"] = lo
        row[f"{tag}_success_wilson_hi"] = hi
        for outcome in OUTCOMES:
            row[f"{tag}_n_{outcome}"] = outs.count(outcome)
    succ_a, succ_b = (
        np.array([[float(getattr(p, s)["outcome"] == "success") for p in pairs]]) for s in "ab"
    )
    contrast = paired_bootstrap(succ_a, succ_b, reps=reps, seed=seed)
    row["success_diff_a_minus_b"] = contrast.diff
    row["success_diff_lo"] = contrast.lo
    row["success_diff_hi"] = contrast.hi
    row["success_diff_separates"] = contrast.separates
    changed = [p for p in pairs if p.a["outcome"] != p.b["outcome"]]
    row["n_outcome_changed"] = len(changed)
    row["n_b_bounce_to_a_success"] = sum(
        p.b["outcome"] == "bounce" and p.a["outcome"] == "success" for p in pairs
    )
    row["n_b_success_to_a_bounce"] = sum(
        p.b["outcome"] == "success" and p.a["outcome"] == "bounce" for p in pairs
    )
    row["n_b_bounce_to_a_bounce"] = sum(
        p.b["outcome"] == "bounce" and p.a["outcome"] == "bounce" for p in pairs
    )
    transitions = Counter(f"{p.b['outcome']}->{p.a['outcome']}" for p in changed)
    row["outcome_transitions_b_to_a"] = ";".join(
        f"{t}:{transitions[t]}" for t in sorted(transitions)
    )
    row["n_first_contact_identical"] = sum(
        all(p.a[c] == p.b[c] for c in FIRST_CONTACT_COLUMNS) for p in pairs
    )
    decided = [p for p in pairs if decided_by_first_contact(p.a) or decided_by_first_contact(p.b)]
    row["n_decided_by_first_contact"] = len(decided)
    row["n_decided_by_first_contact_outcome_identical"] = sum(
        p.a["outcome"] == p.b["outcome"] for p in decided
    )
    row["n_decided_by_first_contact_outcome_and_reason_identical"] = sum(
        p.a["outcome"] == p.b["outcome"] and p.a["termination_reason"] == p.b["termination_reason"]
        for p in decided
    )
    row["n_steps_identical"] = sum(p.a["steps"] == p.b["steps"] for p in pairs)
    for tag, side in (("a", "a"), ("b", "b")):
        rows_ = [getattr(p, side) for p in pairs]
        row[f"{tag}_n_tunnelled"] = sum(as_bool(r["tunnelled"]) for r in rows_)
        row[f"{tag}_max_penetration_m"] = float(
            -min(as_float(r["max_penetration_m"]) for r in rows_)
        )
    for column, tag in (("effort_mean_sq", "effort"), ("action_jerk_mean", "jerk")):
        a, b = _metric(pairs, column)
        c = paired_bootstrap(a[np.newaxis, :], b[np.newaxis, :], reps=reps, seed=seed)
        row[f"a_{tag}_mean"] = float(a.mean())
        row[f"b_{tag}_mean"] = float(b.mean())
        row[f"{tag}_diff_a_minus_b"] = c.diff
        row[f"{tag}_diff_lo"] = c.lo
        row[f"{tag}_diff_hi"] = c.hi
        row[f"{tag}_median_ratio_a_over_b"] = float(np.median(a / b))
        row[f"n_{tag}_rose"] = int(np.sum(a > b))
    return row


# --- P5-D3 entry point (thin wrapper; the logic above is generic) ----------------------------

_A = "pid_feedforward_lowvz_cut"
_B = "pid_feedforward_lowvz"
_FF = "pid_feedforward"
_BOUNCE_PREDICTED_POINT = 0.6
_BOUNCE_PREDICTED_FLOOR = 0.5


def _p5d3(new_dir: Path, reference_dir: Path, reps: int, seed: int) -> None:
    """Score P5-D2's predictions; write the paired tables and ``p5d2_predictions.csv``.

    Writes ``paired_vs_lowvz.csv`` (the scored comparison), ``p5d2_predictions.csv`` (the
    verdicts) and, **post hoc**, ``paired_vs_pid_feedforward.csv`` (same statistics against
    the H1b reference; not part of any P5-D2 prediction).

    Args:
        new_dir: The run holding ``pid_feedforward_lowvz_cut`` (``results/e01_lowvz_cut``).
        reference_dir: The committed run holding ``pid_feedforward_lowvz`` (``results/e01``).
        reps: Bootstrap replicates (P3-D1 §4: 10 000).
        seed: Bootstrap seed (P3-D1 §4: 20260926).
    """
    new_csv, ref_csv = new_dir / "episodes.csv", reference_dir / "episodes.csv"
    cells = pair_rows(read_rows(new_csv), read_rows(ref_csv), _A, _B)
    source = {
        "method_a": _A,
        "method_b": _B,
        "a_episodes_sha256": sha256_file(new_csv),
        "b_episodes_sha256": sha256_file(ref_csv),
        "bootstrap_reps": str(reps),
        "bootstrap_seed": str(seed),
    }
    table = [{**cell_comparison(c, reps=reps, seed=seed), **source} for c in cells]
    write_rows(new_dir / "paired_vs_lowvz.csv", table, list(table[0]))

    # Post hoc (not in P5-D2): the same paired table against pid_feedforward, the H1b reference.
    ff_cells = pair_rows(read_rows(new_csv), read_rows(ref_csv), _A, _FF)
    ff_source = {**source, "method_b": _FF, "post_hoc": "True"}
    ff_table = [{**cell_comparison(c, reps=reps, seed=seed), **ff_source} for c in ff_cells]
    write_rows(new_dir / "paired_vs_pid_feedforward.csv", ff_table, list(ff_table[0]))

    n = sum(r["n_episodes"] for r in table)
    fc = sum(r["n_first_contact_identical"] for r in table)
    dec = sum(r["n_decided_by_first_contact"] for r in table)
    dec_same = sum(r["n_decided_by_first_contact_outcome_identical"] for r in table)
    dec_reason = sum(r["n_decided_by_first_contact_outcome_and_reason_identical"] for r in table)
    bounce = [
        [np.array([float(p.a["outcome"] == "bounce") for p in c.pairs]) for c in cells],
        [np.array([float(p.b["outcome"] == "bounce") for p in c.pairs]) for c in cells],
    ]
    ratio = stratified_paired_ratio(bounce[0], bounce[1], reps=reps, seed=seed)
    nan = float("nan")
    moving = [r for r in table if r["regime"] != "static"]
    tunnel_a = sum(r["a_n_tunnelled"] for r in table)
    tunnel_b = sum(r["b_n_tunnelled"] for r in table)

    def pred(
        name: str, statistic: str, point: float, k: int, m: int, held: bool | None, **ci: float
    ) -> dict[str, Any]:
        return {
            "prediction": name,
            "statistic": statistic,
            "point": point,
            "lo": ci.get("lo", nan),
            "hi": ci.get("hi", nan),
            "k": k,
            "n": m,
            "held": "not_scored" if held is None else str(held),
        }

    preds = [
        pred(
            "1_first_contact_columns_identical",
            "episodes with every FIRST_CONTACT_COLUMNS value text-identical, of N",
            fc / n,
            fc,
            n,
            fc == n,
        ),
        pred(
            "1_pre_contact_outcomes_identical",
            "episodes decided at/before first contact (either method) with identical outcome",
            dec_same / dec if dec else nan,
            dec_same,
            dec,
            dec_same == dec,
        ),
        pred(
            "1_info_termination_reason_also_identical",
            "the same episodes with identical outcome AND termination_reason (not pre-registered)",
            dec_reason / dec if dec else nan,
            dec_reason,
            dec,
            None,
        ),
        pred(
            "2_bounce_ratio",
            "sum bounce(a) / sum bounce(b) over all cells, paired stratified bootstrap; held if "
            f"point >= {_BOUNCE_PREDICTED_FLOOR} and the CI contains {_BOUNCE_PREDICTED_POINT}",
            ratio.point,
            int(sum(x.sum() for x in bounce[0])),
            int(sum(x.sum() for x in bounce[1])),
            ratio.point >= _BOUNCE_PREDICTED_FLOOR
            and ratio.lo <= _BOUNCE_PREDICTED_POINT <= ratio.hi,
            lo=ratio.lo,
            hi=ratio.hi,
        ),
    ]
    for tag in ("effort", "jerk"):
        rose = [r[f"{tag}_diff_lo"] > 0.0 for r in moving]
        preds.append(
            pred(
                f"3_{tag}_rises",
                f"moving-deck cells whose paired mean {tag} difference (a - b) CI lies above 0",
                sum(rose) / len(rose),
                sum(rose),
                len(rose),
                all(rose),
            )
        )
    preds.append(
        pred(
            "info_tunnelled_episodes",
            "tunnelled episodes a (k) vs b (n), all cells (not pre-registered)",
            nan,
            tunnel_a,
            tunnel_b,
            None,
        )
    )
    write_rows(
        new_dir / "p5d2_predictions.csv",
        [{**p, **source} for p in preds],
        [*preds[0], *source],
    )
    for p in preds:
        print(
            f"{p['prediction']:>42s} point={p['point']:.4f} [{p['lo']:.4f}, {p['hi']:.4f}] "
            f"k={p['k']} n={p['n']} held={p['held']}"
        )


def main() -> None:
    """Command-line entry: ``python -m rld.eval.paired_outcomes`` (P5-D3)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new-dir", type=Path, default=REPO_ROOT / "results" / "e01_lowvz_cut")
    parser.add_argument("--reference-dir", type=Path, default=REPO_ROOT / "results" / "e01")
    parser.add_argument("--reps", type=int, default=PAIRED_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    _p5d3(args.new_dir, args.reference_dir, args.reps, args.seed)


if __name__ == "__main__":
    main()
