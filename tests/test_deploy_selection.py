"""Seed selection (P8-D1 §2) and the closed-loop episode rule (P8-D1 §7), on the committed files.

Units: success fractions; p95 metres per second model scale.
"""

from pathlib import Path

import pytest

from conftest import REPO_ROOT
from rld.deploy.closed_loop import CLOSED_LOOP_PER_SS, closed_loop_episodes
from rld.deploy.selection import (
    SELECTION_SOURCES,
    rank_seeds,
    read_seed_row,
    selection_csv,
)


def _row(success: list[float], p95: list[float]) -> dict[str, str]:
    row = {"seeds": "|".join(str(s) for s in range(len(success)))}
    for s, (a, b) in enumerate(zip(success, p95, strict=True)):
        row[f"success_rate_seed{s}"] = repr(a)
        row[f"rel_vz_normal_p95_m_s_seed{s}"] = repr(b)
    return row


def test_rule_orders_worst_to_best_with_both_tie_breaks() -> None:
    r = rank_seeds("m", _row([0.9, 0.8, 0.9, 0.9, 0.95], [0.30, 0.2, 0.25, 0.30, 0.1]))
    # 0.8 worst; then the 0.9s by higher p95 first (0 and 3 tie at 0.30: higher seed is worse);
    # then 0.95 best.
    assert r.order == (1, 3, 0, 2, 4)
    assert r.median_seed == 0


def test_committed_tables_give_seed_four_for_both() -> None:
    ppo = rank_seeds("ppo", read_seed_row(SELECTION_SOURCES["ppo"], "ppo"))
    rpf = rank_seeds(
        "residual_ppo_forecast",
        read_seed_row(SELECTION_SOURCES["residual_ppo_forecast"], "residual_ppo_forecast"),
    )
    assert ppo.order == (2, 1, 4, 3, 0) and ppo.median_seed == 4
    assert rpf.order == (1, 3, 4, 2, 0) and rpf.median_seed == 4
    text = selection_csv([ppo, rpf])
    assert text.count(",True\n") == 2


def test_committed_selection_csv_reproduces() -> None:
    path = REPO_ROOT / "results" / "latency" / "selection.csv"
    if not path.exists():
        pytest.skip("results/latency/selection.csv not written yet")
    rankings = [rank_seeds(m, read_seed_row(SELECTION_SOURCES[m], m)) for m in SELECTION_SOURCES]
    assert path.read_text(encoding="utf-8") == selection_csv(rankings)


def test_closed_loop_episodes_are_the_p8d1_fifty() -> None:
    eps = closed_loop_episodes()
    assert len(eps) == 50 == sum(CLOSED_LOOP_PER_SS.values())
    assert all(e.regime == "id" and e.pad == "aft" for e in eps)
    for ss, k in CLOSED_LOOP_PER_SS.items():
        assert [e.index for e in eps if e.ss == ss] == list(range(k))


def test_closed_loop_rule_refuses_a_short_list(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        closed_loop_episodes(tmp_path)
