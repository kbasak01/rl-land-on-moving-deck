"""Report tables: grouping, byte-reproducible CSVs, markdown rendered only from the CSV.

The last test checks the committed ``results/e01`` artifacts: ``summary.csv`` must be exactly
what ``episodes.csv`` reduces to, and ``success_vs_seastate.md`` must re-render byte for byte
from ``summary.csv``.

Units: speeds metres per second model scale; rates dimensionless.
"""

from pathlib import Path
from typing import Any

import pytest

from rld.control.registry import REGISTRY
from rld.eval.controller_config import resolved_config_sha256
from rld.eval.metrics import CELL_METRIC_COLUMNS
from rld.eval.report import (
    CELL_STATUS_COLUMNS,
    DEFAULT_PAD,
    FORBIDDEN_RENDERED_PHRASES,
    METHOD_LABELS,
    QUIET_COLUMNS,
    SUMMARY_KEY_COLUMNS,
    SkipRecord,
    method_label,
    read_rows,
    render_success_vs_seastate,
    secondary_seed_arm,
    summarise,
    summary_columns,
    write_rows,
)
from rld.eval.runner import EPISODE_COLUMNS
from rld.eval.stats import wilson_interval

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
        "td_in_quiescent_window": (ss == "SS3") if landed else NAN,
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
    assert all(r["pad"] == DEFAULT_PAD and r["n_listed"] == 4 for r in summary)
    assert all(r["skip_reason"] == "" for r in summary)
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
    assert "the gated rule applied to the true future deck motion" in text
    assert "commit-timing oracle (privileged)" in text
    # The quiet-touchdown fraction is in the breakdown: 3/3 touchdowns at SS3 (3 of 4 listed),
    # 0/3 at SS6.
    assert "quiet \\| td" in text
    breakdown = [line for line in text.splitlines() if line.startswith("| pid_feedforward | SS")]

    def cell(k: int, n: int) -> str:
        lo, hi = wilson_interval(k, n)
        return f"{k / n:.3f} [{lo:.3f}, {hi:.3f}] ({k}/{n})"

    assert [line.split(" | ")[-3] for line in breakdown] == [cell(3, 3), cell(0, 3)]
    assert [line.split(" | ")[-2] for line in breakdown] == [cell(3, 4), cell(0, 4)]
    # A summary without the quiet columns (e01) keeps its original single column.
    legacy = [
        {k: v for k, v in r.items() if k not in QUIET_COLUMNS}
        for r in read_rows(tmp_path / "a.csv")
    ]
    old_text = render_success_vs_seastate(legacy, "t")
    assert "| quiet td |" in old_text and "quiet landings / listed" not in old_text
    # Missing cells are printed as missing, never as zero.
    assert "| missing |" in text
    assert "Simulation only" in text


@pytest.mark.skipif(not (E01 / "summary.csv").exists(), reason="results/e01 not generated")
def test_labels_mark_privilege_and_the_h1a_reference() -> None:
    assert METHOD_LABELS["oracle_gated"] == (
        "oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true "
        "future deck motion"
    )
    assert METHOD_LABELS["pid_feedforward_lowvz"] == (
        "pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8))"
    )


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
            # e01's summary predates the pad key: it is all-aft.
            committed = old.get(column, DEFAULT_PAD if column == "pad" else None)
            assert text == committed, (old["method"], old["regime"], old["ss"], column)
    fractions = [
        sum(
            float(r[f"frac_{o}"])
            for o in ("crash", "off_pad", "hard_landing", "bounce", "success", "timeout")
        )
        for r in summary
    ]
    assert all(abs(f - 1.0) < 1e-12 for f in fractions)
    # The committed numbers were flown with the configs that resolve today.
    for row in summary:
        if row["method"] in REGISTRY:
            assert row["resolved_gains_sha256"] == resolved_config_sha256(row["method"])
    rendered = render_success_vs_seastate(
        summary, "e01 — classical baselines: success versus sea state (frozen episode lists)"
    )
    assert rendered == (E01 / "success_vs_seastate.md").read_text(encoding="utf-8")


def test_rendered_output_never_says_upper_bound() -> None:
    """P3-D4: the oracle is a commit-timing oracle (privileged), never an "upper bound"."""
    assert "upper bound" in FORBIDDEN_RENDERED_PHRASES
    assert "commit-timing oracle (privileged)" in METHOD_LABELS["oracle_gated"]
    for label in METHOD_LABELS.values():
        assert "upper bound" not in label.lower()
    summary = summarise(_rows(), ["pid_feedforward", "oracle_gated"])
    records = [{k: str(v) for k, v in r.items()} for r in summary]
    text = render_success_vs_seastate(records, "t")
    assert "upper bound" not in text.lower()
    assert "commit-timing oracle (privileged)" in text
    # Every oracle row is labelled, in the headline and the breakdown.
    oracle_lines = [line for line in text.splitlines() if line.startswith("| oracle_gated")]
    assert oracle_lines and all(METHOD_LABELS["oracle_gated"] in line for line in oracle_lines)
    # The guard refuses the phrase anywhere, including a title.
    with pytest.raises(ValueError, match="upper bound"):
        render_success_vs_seastate(records, "an Upper Bound on success")


@pytest.mark.skipif(not (E01 / "summary.csv").exists(), reason="results/e01 not generated")
def test_committed_renderings_never_say_upper_bound() -> None:
    for path in sorted(E01.parent.glob("e*/success_vs_seastate.md")):
        assert "upper bound" not in path.read_text(encoding="utf-8").lower(), path


def test_pads_and_skipped_cells_are_keyed_and_rendered() -> None:
    rows = [{**r, "pad": "aft"} for r in _rows()] + [{**r, "pad": "cg"} for r in _rows()]
    rows += [
        {**_episode("gated_forecast", False, "id", "SS3", "timeout"), "pad": pad}
        for pad in ("aft", "cg")
    ]
    skips = [
        SkipRecord("gated_forecast", False, 0, pad, "static", "static", "no ship to feed")
        for pad in ("aft", "cg")
        for _ in range(2)
    ]
    order = ["pid_feedforward", "oracle_gated", "gated_forecast"]
    summary = summarise(rows, order, skipped=skips)
    keys = [(r["pad"], r["method"], r["regime"], r["ss"]) for r in summary]
    assert keys[:5] == [
        ("aft", "pid_feedforward", "id", "SS3"),
        ("aft", "pid_feedforward", "id", "SS6"),
        ("aft", "oracle_gated", "id", "SS3"),
        ("aft", "oracle_gated", "id", "SS6"),
        ("aft", "gated_forecast", "id", "SS3"),
    ]
    assert keys[5] == ("aft", "gated_forecast", "static", "static")
    assert keys[6][0] == "cg" and len(keys) == 12
    skipped = [r for r in summary if r["n_episodes"] == 0]
    assert len(skipped) == 2
    for rec in skipped:
        assert rec["n_listed"] == 2 and rec["n_success"] == 0
        assert rec["skip_reason"] == "2 of 2 not run: no ship to feed"
        assert all(rec[c] != rec[c] for c in ("success_rate", "frac_timeout"))  # NaN
    columns = summary_columns([])
    status_at = len(SUMMARY_KEY_COLUMNS) + len(CELL_METRIC_COLUMNS)
    assert columns[status_at : status_at + len(CELL_STATUS_COLUMNS)] == list(CELL_STATUS_COLUMNS)
    records = [
        {k: (repr(v) if isinstance(v, float) else str(v)) for k, v in r.items()} for r in summary
    ]
    text = render_success_vs_seastate(records, "t")
    assert "## id — aft pad (primary, as frozen in P3-D1)" in text
    assert "## id — pad at CG" in text
    assert "Cells not run" in text
    assert "gated_forecast, pad cg, static static: 2 of 2 not run: no ship to feed" in text
    assert text.count("not run (0/2; see above)") == 2
    assert "pad-at-CG control arm" in text
    # A single-pad (aft) summary renders with no pad headings, as e01 does.
    single = render_success_vs_seastate(
        [r for r in records if r["pad"] == "aft" and r["method"] != "gated_forecast"], "t"
    )
    assert "## id\n" in single and "aft pad (primary" not in single


def test_quiet_columns_separate_per_touchdown_from_per_listed() -> None:
    """Results-skeptic M1: a method that rarely touches down is not flattered."""
    rows = []
    # "rare": 2 touchdowns of 10, both quiet (SS3). "often": 10 of 10, 3 quiet.
    rows += [_episode("rare", False, "id", "SS3", "success") for _ in range(2)]
    rows += [_episode("rare", False, "id", "SS3", "timeout") for _ in range(8)]
    often = [_episode("often", False, "id", "SS3", "success") for _ in range(10)]
    for i, row in enumerate(often):
        row["td_in_quiescent_window"] = i < 3
    summary = summarise(rows + often, ["rare", "often"])
    rare, oft = summary
    assert (rare["n_touchdowns"], rare["n_td_in_quiescent_window"]) == (2, 2)
    assert rare["frac_td_in_quiescent_window"] == 1.0 > oft["frac_td_in_quiescent_window"]
    assert rare["quiet_landings_per_listed"] == 0.2 < oft["quiet_landings_per_listed"] == 0.3
    lo, hi = wilson_interval(2, 2)
    assert (
        rare["frac_td_in_quiescent_window_wilson_lo"],
        rare["frac_td_in_quiescent_window_wilson_hi"],
    ) == (lo, hi)
    assert (
        rare["quiet_landings_per_listed_wilson_lo"],
        rare["quiet_landings_per_listed_wilson_hi"],
    ) == wilson_interval(2, 10)
    assert summary_columns([])[-len(QUIET_COLUMNS) :] == list(QUIET_COLUMNS)
    records = [
        {k: (repr(v) if isinstance(v, float) else str(v)) for k, v in r.items()} for r in summary
    ]
    text = render_success_vs_seastate(records, "t")
    assert "| td n | quiet \\| td | quiet landings / listed |" in text
    assert "1.000 [0.342, 1.000] (2/2) | 0.200 [0.057, 0.510] (2/10)" in text
    assert "of the touched-down episodes only" in text and "quiet td" not in text


def test_quiet_columns_are_nan_for_a_cell_not_run() -> None:
    skips = [SkipRecord("gated_forecast", False, 0, "aft", "static", "static", "no ship")] * 3
    (rec,) = summarise([], ["gated_forecast"], skipped=skips)
    assert all(rec[c] != rec[c] for c in QUIET_COLUMNS)


def test_seed_sensitivity_arms_are_labelled_secondary() -> None:
    assert secondary_seed_arm("gated_forecast_tcn_seed2") == ("gated_forecast_tcn", 2)
    assert secondary_seed_arm("gated_forecast_tcn") is None
    label = method_label("gated_forecast_tcn_seed0", False)
    assert label.startswith("gated_forecast_tcn_seed0 (secondary: seed sensitivity;")
    rows = [_episode("gated_forecast_tcn_seed0", False, "id", "SS3", "success")]
    records = [
        {k: str(v) for k, v in r.items()} for r in summarise(rows, ["gated_forecast_tcn_seed0"])
    ]
    text = render_success_vs_seastate(records, "t")
    assert "**secondary: seed sensitivity**" in text
    assert f"| {label} |" in text


E02 = E01.parent / "e02"


@pytest.mark.skipif(not (E02 / "summary.csv").exists(), reason="results/e02 not generated")
def test_e02_rerenders_and_resummarises_identically() -> None:
    from rld.eval.reproduce import resummarise

    summary = read_rows(E02 / "summary.csv")
    title = (
        "e02 — baselines, gated and forecast-gated controllers, aft pad and pad at CG: "
        "success versus sea state (frozen episode lists)"
    )
    rendered = render_success_vs_seastate(summary, title)
    assert rendered == (E02 / "success_vs_seastate.md").read_text(encoding="utf-8")
    assert "upper bound" not in rendered.lower()
    rows, columns, added = resummarise(E02)
    assert added == []  # already carries every column
    assert columns == list(summary[0])
