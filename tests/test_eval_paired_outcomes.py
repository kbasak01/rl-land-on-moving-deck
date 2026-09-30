"""``rld.eval.paired_outcomes``: pairing, first-contact identity, the stratified paired ratio."""

import numpy as np
import pytest

from rld.eval.paired_outcomes import (
    FIRST_CONTACT_COLUMNS,
    cell_comparison,
    decided_by_first_contact,
    pair_rows,
    stratified_paired_ratio,
)


def _row(method: str, index: int, outcome: str, *, ss: str = "SS5", **extra: str) -> dict[str, str]:
    row = {
        "method": method,
        "pad": "aft",
        "regime": "id",
        "ss": ss,
        "index": str(index),
        "t0_model_s": f"{10.0 + index!r}",
        "realization_seed": "3",
        "episode_seed": str(100 + index),
        "outcome": outcome,
        "termination_reason": {
            "success": "dwell_complete",
            "bounce": "release",
            "timeout": "time_limit",
            "hard_landing": "dwell_complete",
            "crash": "ground_contact",
        }[outcome],
        "steps": "150",
        "effort_mean_sq": "0.1",
        "action_jerk_mean": "0.01",
        "tunnelled": "False",
        "max_penetration_m": "-0.001",
    }
    for column in FIRST_CONTACT_COLUMNS:
        row[column] = "True" if column.startswith("touchdown") else "0.25"
    row.update(extra)
    return row


def test_pair_rows_refuses_different_episode_sets() -> None:
    a = [_row("x", 0, "success"), _row("x", 1, "success")]
    b = [_row("y", 0, "success")]
    with pytest.raises(ValueError, match="episode sets differ"):
        pair_rows(a, b, "x", "y")


def test_pair_rows_refuses_a_different_start_offset() -> None:
    a = [_row("x", 0, "success")]
    b = [_row("y", 0, "success", t0_model_s="99.0")]
    with pytest.raises(ValueError, match="t0_model_s differs"):
        pair_rows(a, b, "x", "y")


def test_pair_rows_orders_cells_and_ignores_other_methods() -> None:
    rows = [
        _row("x", 1, "success", ss="SS6"),
        _row("x", 0, "success", ss="SS3"),
        _row("z", 0, "success", ss="SS3"),
        _row("y", 0, "bounce", ss="SS3"),
        _row("y", 1, "success", ss="SS6"),
    ]
    cells = pair_rows(rows, rows, "x", "y")
    assert [(c.regime, c.ss, len(c.pairs)) for c in cells] == [("id", "SS3", 1), ("id", "SS6", 1)]


def test_decided_by_first_contact() -> None:
    assert decided_by_first_contact(_row("x", 0, "hard_landing"))
    assert decided_by_first_contact(_row("x", 0, "timeout"))
    assert decided_by_first_contact(_row("x", 0, "crash", touchdown_contact="False"))
    assert not decided_by_first_contact(_row("x", 0, "crash"))
    assert not decided_by_first_contact(_row("x", 0, "bounce"))
    assert not decided_by_first_contact(_row("x", 0, "success"))


def test_ratio_of_identical_methods_is_exactly_one() -> None:
    strata = [np.array([1.0, 0.0, 1.0, 0.0]), np.array([0.0, 1.0, 1.0])]
    est = stratified_paired_ratio(strata, strata, reps=500, seed=1)
    assert est.point == 1.0 and est.lo == 1.0 and est.hi == 1.0


def test_ratio_known_point_and_pairing() -> None:
    # b bounces in 4 of 8 episodes per stratum; a keeps exactly the first 2 of them.
    b = [np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)] * 3
    a = [np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=float)] * 3
    est = stratified_paired_ratio(a, b, reps=4000, seed=2)
    assert est.point == 0.5
    # Paired: a's events are a subset of b's, so no replicate can exceed 1.
    assert 0.0 <= est.lo < 0.5 < est.hi <= 1.0
    # Reproducible from the seed.
    assert stratified_paired_ratio(a, b, reps=4000, seed=2) == est


def test_ratio_refuses_zero_denominator() -> None:
    zero = [np.zeros(4)]
    with pytest.raises(ValueError, match="undefined"):
        stratified_paired_ratio(zero, zero)


def test_cell_comparison_counts() -> None:
    a = [
        _row("a", 0, "success"),
        _row("a", 1, "bounce"),
        _row("a", 2, "hard_landing"),
        _row("a", 3, "success", effort_mean_sq="0.2", rel_tilt_deg="7.0"),
    ]
    b = [
        _row("b", 0, "bounce"),
        _row("b", 1, "bounce"),
        _row("b", 2, "hard_landing"),
        _row("b", 3, "success"),
    ]
    (cell,) = pair_rows(a, b, "a", "b")
    row = cell_comparison(cell, reps=200, seed=3)
    assert row["n_episodes"] == 4
    assert (row["a_n_success"], row["b_n_success"]) == (2, 1)
    assert row["a_n_bounce"] == 1 and row["b_n_bounce"] == 2
    assert row["success_diff_a_minus_b"] == 0.25
    assert row["n_b_bounce_to_a_success"] == 1 and row["n_b_bounce_to_a_bounce"] == 1
    assert row["outcome_transitions_b_to_a"] == "bounce->success:1"
    assert row["n_first_contact_identical"] == 3  # episode 3's rel_tilt differs
    assert row["n_decided_by_first_contact"] == 1
    assert row["n_decided_by_first_contact_outcome_identical"] == 1
    assert row["n_decided_by_first_contact_outcome_and_reason_identical"] == 1
    assert row["n_effort_rose"] == 1
    assert row["effort_diff_a_minus_b"] == pytest.approx(0.025)


def test_cell_comparison_tunnelling_and_penetration() -> None:
    a = [_row("a", 0, "success", tunnelled="True", max_penetration_m="-0.006")]
    b = [_row("b", 0, "success")]
    (cell,) = pair_rows(a, b, "a", "b")
    row = cell_comparison(cell, reps=50, seed=4)
    assert (row["a_n_tunnelled"], row["b_n_tunnelled"]) == (1, 0)
    assert row["a_max_penetration_m"] == 0.006 and row["b_max_penetration_m"] == 0.001
