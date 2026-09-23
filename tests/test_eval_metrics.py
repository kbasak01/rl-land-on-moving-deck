"""Per-cell metrics on synthetic rows whose answers are known.

Units: speeds metres per second model scale, lengths metres model scale, times seconds model
scale, angles degrees, actions normalised.
"""

import math
from dataclasses import fields
from typing import Any

import pytest

from rld.envs.touchdown import OUTCOMES, TERMINATION_REASONS
from rld.eval.metrics import CellMetrics, cell_metrics
from rld.eval.stats import wilson_interval

NAN = float("nan")


def _row(outcome: str, reason: str, **touchdown: float) -> dict[str, Any]:
    landed = bool(touchdown)
    return {
        "outcome": outcome,
        "termination_reason": reason,
        "touchdown_contact": landed,
        "rel_vz_normal_m_s": touchdown.get("vz", NAN),
        "rel_vz_world_z_m_s": touchdown.get("vzw", NAN),
        "lateral_offset_m": touchdown.get("lat", NAN),
        "rel_tilt_deg": touchdown.get("tilt", NAN),
        "td_t_episode_s": touchdown.get("t", NAN),
        "effort_mean_sq": 0.04,
        "action_jerk_mean": 0.01,
        "detectors_disagree": False,
        "tunnelled": False,
        "max_penetration_m": -0.001,
        "in_training_distribution": True,
        "td_in_quiescent_window": True if landed else NAN,
    }


def _cell() -> list[dict[str, Any]]:
    rows = [
        _row(
            "success",
            "dwell_complete",
            vz=-0.1 * (i + 1),
            vzw=-0.1 * (i + 1),
            lat=0.01 * i,
            tilt=float(i),
            t=2.0 + i,
        )
        for i in range(6)
    ]
    rows += [
        _row("hard_landing", "dwell_complete", vz=-0.9, vzw=-0.95, lat=0.02, tilt=3.0, t=1.0),
        _row("crash", "tilt_gt_crash"),
        _row("timeout", "time_limit"),
        _row("bounce", "release", vz=0.2, vzw=0.2, lat=0.05, tilt=2.0, t=4.0),
    ]
    rows[-1]["detectors_disagree"] = True
    rows[-2]["tunnelled"] = True
    rows[-2]["max_penetration_m"] = -0.0079
    rows[0]["in_training_distribution"] = False
    rows[1]["td_in_quiescent_window"] = False
    rows[6]["td_in_quiescent_window"] = False
    return rows


def test_breakdown_sums_to_one_and_success_carries_wilson() -> None:
    m = cell_metrics(_cell())
    assert m.n_episodes == 10 and m.n_success == 6
    fracs = [getattr(m, f"frac_{o}") for o in OUTCOMES]
    assert fracs == [0.1, 0.0, 0.1, 0.1, 0.6, 0.1]
    assert math.fsum(fracs) == pytest.approx(1.0, abs=1e-12)
    assert (m.success_wilson_lo, m.success_wilson_hi) == wilson_interval(6, 10)
    assert m.success_rate == m.frac_success == 0.6


def test_touchdown_statistics_use_closing_speed_over_landed_episodes() -> None:
    m = cell_metrics(_cell())
    assert m.n_touchdowns == 8
    # Closing speeds: 0.1..0.6 (successes), 0.9 (hard), 0.0 (bounce moving away: +0.2).
    closing = sorted([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 0.0])
    assert m.rel_vz_normal_p50_m_s == pytest.approx(0.35)
    expected_p95 = closing[6] + 0.65 * (closing[7] - closing[6])  # linear interpolation
    assert m.rel_vz_normal_p95_m_s == pytest.approx(expected_p95)
    assert m.rel_vz_world_z_p95_m_s > m.rel_vz_normal_p95_m_s
    assert m.time_to_touchdown_mean_s == pytest.approx((2 + 3 + 4 + 5 + 6 + 7 + 1 + 4) / 8)
    assert m.effort_mean_sq == pytest.approx(0.04)
    assert m.action_jerk_mean == pytest.approx(0.01)


def test_audit_counts() -> None:
    m = cell_metrics(_cell())
    assert m.disagreement_n == 1 and m.disagreement_rate == 0.1
    assert m.tunnelling_n == 1 and m.max_penetration_m == -0.0079
    assert m.frac_in_training_distribution == 0.9
    assert m.n_reason_dwell_complete == 7
    assert m.n_reason_tilt_gt_crash == 1
    assert m.n_reason_time_limit == 1
    assert m.n_reason_release == 1
    reason_total = sum(getattr(m, f"n_reason_{r}") for r in TERMINATION_REASONS)
    assert reason_total == m.n_episodes


def test_quiescent_touchdown_fraction_is_over_touched_down_episodes() -> None:
    m = cell_metrics(_cell())
    # 8 touchdowns, 2 of them outside a quiescent window; crash and timeout carry NaN and are
    # in neither the numerator nor the denominator.
    assert m.n_td_in_quiescent_window == 6
    assert m.frac_td_in_quiescent_window == 6 / 8


def test_quiescent_verdict_must_match_touchdown() -> None:
    rows = _cell()
    rows[7]["td_in_quiescent_window"] = False  # crash, no touchdown, yet a verdict
    with pytest.raises(AssertionError, match="td_in_quiescent_window"):
        cell_metrics(rows)
    rows = _cell()
    rows[0]["td_in_quiescent_window"] = NAN  # touchdown, yet no verdict
    with pytest.raises(AssertionError, match="td_in_quiescent_window"):
        cell_metrics(rows)


def test_reason_columns_cover_every_frozen_reason() -> None:
    names = {f.name for f in fields(CellMetrics) if f.name.startswith("n_reason_")}
    assert names == {f"n_reason_{r}" for r in TERMINATION_REASONS}


def test_no_touchdown_is_nan_not_zero() -> None:
    m = cell_metrics([_row("timeout", "time_limit") for _ in range(5)])
    assert m.n_touchdowns == 0
    assert math.isnan(m.rel_vz_normal_p95_m_s) and math.isnan(m.lateral_p50_m)
    assert math.isnan(m.time_to_touchdown_mean_s)
    assert m.frac_timeout == 1.0 and m.success_wilson_lo == 0.0
    assert m.n_td_in_quiescent_window == 0 and math.isnan(m.frac_td_in_quiescent_window)


def test_csv_strings_reduce_like_in_memory_values() -> None:
    rows = _cell()
    as_text = [
        {k: (repr(v) if isinstance(v, float) else str(v)) for k, v in row.items()} for row in rows
    ]
    a, b = cell_metrics(rows), cell_metrics(as_text)
    for f in fields(CellMetrics):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        assert va == vb or (math.isnan(va) and math.isnan(vb)), f.name


def test_rejects_unknown_classes_and_empty_cells() -> None:
    with pytest.raises(ValueError):
        cell_metrics([])
    with pytest.raises(ValueError):
        cell_metrics([_row("landed_ok", "dwell_complete")])
    with pytest.raises(ValueError):
        cell_metrics([_row("success", "because")])
