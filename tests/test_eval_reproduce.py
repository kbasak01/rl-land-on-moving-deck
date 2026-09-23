"""``rld.eval.reproduce``: row-for-row, byte-for-byte comparison against a committed run.

Synthetic rows only; nothing is flown. Units: none.
"""

import math
from pathlib import Path
from typing import Any

from rld.eval.report import write_rows
from rld.eval.reproduce import compare_to_reference, summary_provenance_differences
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
