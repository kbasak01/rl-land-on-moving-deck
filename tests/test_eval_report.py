"""Report tables: grouping, byte-reproducible CSVs, markdown rendered only from the CSV.

The last test checks the committed ``results/e01`` artifacts: ``summary.csv`` must be exactly
what ``episodes.csv`` reduces to, and ``success_vs_seastate.md`` must re-render byte for byte
from ``summary.csv``.

Units: speeds metres per second model scale; rates dimensionless.
"""

from pathlib import Path
from typing import Any

import pytest

from rld.eval.metrics import CELL_METRIC_COLUMNS
from rld.eval.report import (
    METHOD_LABELS,
    SUMMARY_KEY_COLUMNS,
    read_rows,
    render_success_vs_seastate,
    summarise,
    summary_columns,
    write_rows,
)
from rld.eval.runner import EPISODE_COLUMNS

E01 = Path(__file__).resolve().parents[1] / "results" / "e01"
NAN = float("nan")


def _episode(method: str, privileged: bool, regime: str, ss: str, outcome: str) -> dict[str, Any]:
    landed = outcome != "timeout"
    return {
        "method": method,
        "privileged": privileged,
        "run_seed": 0,
        "regime": regime,
        "ss": ss,
        "outcome": outcome,
        "termination_reason": "time_limit" if not landed else "dwell_complete",
        "touchdown_contact": landed,
        "rel_vz_normal_m_s": -0.2 if landed else NAN,
        "rel_vz_world_z_m_s": -0.21 if landed else NAN,
        "lateral_offset_m": 0.02 if landed else NAN,
        "rel_tilt_deg": 1.5 if landed else NAN,
        "td_t_episode_s": 3.0 if landed else NAN,
        "effort_mean_sq": 0.03,
        "action_jerk_mean": 0.004,
        "detectors_disagree": False,
        "tunnelled": False,
        "max_penetration_m": -0.001,
        "in_training_distribution": ss != "SS6",
    }


def _rows() -> list[dict[str, Any]]:
    rows = []
    for method, priv in (("pid_feedforward", False), ("oracle_gated", True)):
        for ss in ("SS3", "SS6"):
            rows += [_episode(method, priv, "id", ss, "success") for _ in range(3)]
            rows.append(_episode(method, priv, "id", ss, "timeout"))
    return rows


def test_summary_groups_in_committed_order() -> None:
    summary = summarise(_rows(), ["pid_feedforward", "oracle_gated"])
    keys = [(r["method"], r["regime"], r["ss"]) for r in summary]
    assert keys == [
        ("pid_feedforward", "id", "SS3"),
        ("pid_feedforward", "id", "SS6"),
        ("oracle_gated", "id", "SS3"),
        ("oracle_gated", "id", "SS6"),
    ]
    assert all(r["n_episodes"] == 4 and r["n_success"] == 3 for r in summary)
    assert [r["privileged"] for r in summary] == [False, False, True, True]
    assert list(summary[0])[: len(SUMMARY_KEY_COLUMNS)] == list(SUMMARY_KEY_COLUMNS)
    with pytest.raises(ValueError):
        summarise(_rows(), ["pid_feedforward"])


def test_csv_round_trip_and_render_are_deterministic(tmp_path: Path) -> None:
    summary = summarise(_rows(), ["pid_feedforward", "oracle_gated"], provenance={"x": "1"})
    columns = summary_columns(["x"])
    write_rows(tmp_path / "a.csv", summary, columns)
    write_rows(tmp_path / "b.csv", summary, columns)
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    text = render_success_vs_seastate(read_rows(tmp_path / "a.csv"), "t")
    assert text == render_success_vs_seastate(read_rows(tmp_path / "b.csv"), "t")
    assert "75.0 [" in text and "(3/4)" in text
    assert METHOD_LABELS["oracle_gated"] in text
    assert "not on success" in text
    # Missing cells are printed as missing, never as zero.
    assert "| missing |" in text
    assert "Simulation only" in text


@pytest.mark.skipif(not (E01 / "summary.csv").exists(), reason="results/e01 not generated")
def test_committed_e01_is_consistent() -> None:
    episodes = read_rows(E01 / "episodes.csv")
    assert tuple(episodes[0]) == EPISODE_COLUMNS
    summary = read_rows(E01 / "summary.csv")
    methods = list(dict.fromkeys(r["method"] for r in summary))
    recomputed = summarise(episodes, methods)
    assert len(recomputed) == len(summary)
    for new, old in zip(recomputed, summary, strict=True):
        for column in (*SUMMARY_KEY_COLUMNS, *CELL_METRIC_COLUMNS):
            value = new[column]
            text = repr(float(value)) if isinstance(value, float) else str(value)
            assert text == old[column], (old["method"], old["regime"], old["ss"], column)
    fractions = [
        sum(
            float(r[f"frac_{o}"])
            for o in ("crash", "off_pad", "hard_landing", "bounce", "success", "timeout")
        )
        for r in summary
    ]
    assert all(abs(f - 1.0) < 1e-12 for f in fractions)
    rendered = render_success_vs_seastate(
        summary, "e01 — classical baselines: success versus sea state (frozen episode lists)"
    )
    assert rendered == (E01 / "success_vs_seastate.md").read_text(encoding="utf-8")
