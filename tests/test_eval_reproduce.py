"""``rld.eval.reproduce``: row-for-row, byte-for-byte comparison against a committed run.

Synthetic rows only; nothing is flown. Units: none.
"""

import math
from pathlib import Path
from typing import Any

import pytest

from rld.config import REPO_ROOT
from rld.eval.report import QUIET_COLUMNS, SkipRecord, summarise, summary_columns, write_rows
from rld.eval.reproduce import (
    compare_to_reference,
    git_state,
    resummarise,
    summary_provenance_differences,
)
from rld.eval.runner import EPISODE_COLUMNS, episode_rows_csv


def _row(method: str, index: int, pad: str = "aft", ret: float = 1.5) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(EPISODE_COLUMNS, 0)
    row.update(
        method=method, run_seed=0, pad=pad, regime="id", ss="SS3", index=index, outcome="success"
    )
    row["return"] = ret
    row["td_in_quiescent_window"] = math.nan
    return row


def test_identical_rows_pass_and_a_changed_byte_fails(tmp_path: Path) -> None:
    ref = [_row("pid_feedforward", i) for i in range(3)] + [_row("gated", 0)]
    path = tmp_path / "episodes.csv"
    path.write_text(episode_rows_csv(ref), encoding="utf-8")
    new = [
        *ref,
        _row("pid_feedforward", 0, pad="cg", ret=9.0),  # other pad: not compared
        _row("gated_forecast", 0),  # not in the reference: not compared
    ]
    check = compare_to_reference(new, path)
    assert check["all_identical"] is True and check["header_identical"] is True
    assert check["methods"]["pid_feedforward"]["n_compared"] == 3
    assert check["methods"]["pid_feedforward"]["n_reference_rows"] == 3
    assert set(check["methods"]) == {"pid_feedforward", "gated"}
    # One float differing in its last digit is a mismatch.
    bad = [_row("pid_feedforward", 0, ret=1.5000000000000002), *ref[1:]]
    check = compare_to_reference(bad, path)
    assert check["all_identical"] is False
    assert check["methods"]["pid_feedforward"]["n_differ"] == 1
    assert check["methods"]["pid_feedforward"]["first_mismatch"] == "pid_feedforward/0/aft/id/SS3/0"
    # A row the reference does not hold is a mismatch too.
    extra = [*ref, _row("gated", 7)]
    check = compare_to_reference(extra, path)
    assert check["methods"]["gated"]["n_missing_in_reference"] == 1
    assert check["all_identical"] is False


def test_summary_provenance_differences(tmp_path: Path) -> None:
    rows = [
        {"method": "oracle_gated", "controller_config_sha256": "old", "resolved_gains_sha256": "r"},
        {"method": "oracle_gated", "controller_config_sha256": "old", "resolved_gains_sha256": "r"},
        {"method": "gated", "controller_config_sha256": "g", "resolved_gains_sha256": "q"},
    ]
    path = tmp_path / "summary.csv"
    write_rows(path, rows, ["method", "controller_config_sha256", "resolved_gains_sha256"])
    live = {
        "oracle_gated": {
            "controller_config_sha256": "new",
            "resolved_gains_sha256": "r",
            "forecaster_files_sha256": "",  # not in the reference: skipped
        },
        "gated": {"controller_config_sha256": "g", "resolved_gains_sha256": "q"},
    }
    assert summary_provenance_differences(path, live) == [
        {
            "method": "oracle_gated",
            "column": "controller_config_sha256",
            "committed": "old",
            "live": "new",
            "n_rows": "2",
        }
    ]


def test_no_common_method_is_not_applicable(tmp_path: Path) -> None:
    path = tmp_path / "episodes.csv"
    path.write_text(episode_rows_csv([_row("pid_feedforward", 0)]), encoding="utf-8")
    check = compare_to_reference([_row("gated_forecast_tcn_seed0", 0)], path)
    assert check["applicable"] is False and check["methods"] == {}
    assert compare_to_reference([_row("pid_feedforward", 0)], path)["applicable"] is True


def test_git_state_names_the_commit() -> None:
    state = git_state(REPO_ROOT)
    assert len(state["git_sha"]) == 40 and int(state["git_sha"], 16) >= 0
    assert isinstance(state["git_dirty"], bool)
    assert state["git_dirty"] is bool(state["git_dirty_paths"])


def _summary_row(method: str, index: int, quiet: bool) -> dict[str, Any]:
    row = {
        "method": method,
        "privileged": False,
        "run_seed": 0,
        "pad": "aft",
        "regime": "id",
        "ss": "SS3",
        "index": index,
        "outcome": "success",
        "termination_reason": "dwell_complete",
        "touchdown_contact": True,
        "rel_vz_normal_m_s": -0.2,
        "rel_vz_world_z_m_s": -0.2,
        "lateral_offset_m": 0.01,
        "rel_tilt_deg": 1.0,
        "td_t_episode_s": 3.0,
        "effort_mean_sq": 0.03,
        "action_jerk_mean": 0.004,
        "detectors_disagree": False,
        "tunnelled": False,
        "max_penetration_m": -0.001,
        "in_training_distribution": True,
        "td_in_quiescent_window": quiet,
    }
    return {**dict.fromkeys(EPISODE_COLUMNS, 0), **row}


def test_resummarise_adds_columns_and_changes_nothing_old(tmp_path: Path) -> None:
    episodes = [_summary_row("gated", i, i % 2 == 0) for i in range(5)]
    write_rows(tmp_path / "episodes.csv", episodes, EPISODE_COLUMNS)
    skips = [SkipRecord("gated_forecast", False, 0, "aft", "static", "static", "no ship")] * 4
    summary = summarise(episodes, ["gated", "gated_forecast"], None, {"prov": "x"}, skips)
    old_columns = [c for c in summary_columns(["prov"]) if c not in QUIET_COLUMNS]
    write_rows(tmp_path / "summary.csv", summary, old_columns)
    rows, columns, added = resummarise(tmp_path)
    assert added == list(QUIET_COLUMNS)
    assert columns == summary_columns(["prov"])
    assert rows[0]["quiet_landings_per_listed"] == 3 / 5
    assert rows[1]["n_listed"] == 4 and rows[1]["skip_reason"] == "4 of 4 not run: no ship"
    # A committed metric that the episodes do not reproduce is refused.
    text = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    (tmp_path / "summary.csv").write_text(text.replace(",5,5,1.0,", ",5,4,0.8,", 1))
    with pytest.raises(ValueError, match="column"):
        resummarise(tmp_path)
