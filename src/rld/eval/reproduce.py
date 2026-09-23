"""Check that a new evaluation reproduces a committed one, row for row, byte for byte.

Phase 4 re-flies the Phase 3 controllers so that every table prints them beside the new
methods (CLAUDE.md non-negotiable 4). Those rows are only admissible if they are the Phase 3
rows: :func:`compare_to_reference` serialises the new rows exactly as ``episodes.csv`` is
written (:func:`rld.eval.runner.episode_rows_csv`) and compares each line with the committed
line for the same (method, run seed, pad, regime, sea state, index). The result is a plain
dict for ``run_info.json``.

:func:`summary_provenance_differences` lists, per method, the ``summary.csv`` provenance
columns whose committed value differs from the live one -- the way a comment-only config
edit (``oracle_gated.yaml``'s relabel, P4-D5) shows up without re-flying anything.

Units: none (text comparison).
"""

import csv
import hashlib
import io
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rld.eval.runner import EPISODE_COLUMNS, episode_rows_csv

__all__ = ["compare_to_reference", "summary_provenance_differences"]

#: Columns that identify one episode row of one method.
_ROW_KEY: tuple[str, ...] = ("method", "run_seed", "pad", "regime", "ss", "index")


def _keyed_lines(text: str) -> tuple[str, dict[tuple[str, ...], str]]:
    """Split an ``episodes.csv`` text into its header and ``{row key: raw line}``.

    Args:
        text: The file's text, newline line endings.

    Returns:
        ``(header line, lines by key)``.

    Raises:
        ValueError: On a duplicate key.
    """
    raw = text.split("\n")
    header = raw[0]
    lines = [line for line in raw[1:] if line]
    keyed: dict[tuple[str, ...], str] = {}
    for line, record in zip(
        lines, csv.DictReader(io.StringIO(text), lineterminator="\n"), strict=True
    ):
        key = tuple(record[c] for c in _ROW_KEY)
        if key in keyed:
            raise ValueError(f"duplicate episode row {key}")
        keyed[key] = line
    return header, keyed


def compare_to_reference(
    rows: Sequence[Mapping[str, Any]], reference_csv: Path, *, pad: str = "aft"
) -> dict[str, Any]:
    """Compare new rows on ``pad`` with a committed ``episodes.csv``, byte for byte.

    Only methods present in both are compared; every new row of such a method on ``pad``
    must exist in the reference and be the identical line.

    Args:
        rows: New runner rows (:data:`~rld.eval.runner.EPISODE_COLUMNS`).
        reference_csv: The committed ``episodes.csv`` (e.g. ``results/e01/episodes.csv``).
        pad: The pad whose rows are compared (the reference's pad).

    Returns:
        ``{"reference": path, "reference_sha256", "pad", "header_identical",
        "all_identical", "methods": {method: {"n_compared", "n_identical", "n_differ",
        "n_missing_in_reference", "n_reference_rows", "first_mismatch"}}}``.
    """
    ref_text = reference_csv.read_text(encoding="utf-8")
    ref_header, ref_lines = _keyed_lines(ref_text)
    new_header, new_lines = _keyed_lines(episode_rows_csv(rows))
    ref_methods = {key[0] for key in ref_lines}
    methods: dict[str, dict[str, Any]] = {}
    for key, line in new_lines.items():
        method, _, row_pad = key[0], key[1], key[2]
        if method not in ref_methods or row_pad != pad:
            continue
        stats = methods.setdefault(
            method,
            {
                "n_compared": 0,
                "n_identical": 0,
                "n_differ": 0,
                "n_missing_in_reference": 0,
                "n_reference_rows": sum(1 for k in ref_lines if k[0] == method),
                "first_mismatch": None,
            },
        )
        stats["n_compared"] += 1
        old = ref_lines.get(key)
        if old is None:
            stats["n_missing_in_reference"] += 1
        elif old == line:
            stats["n_identical"] += 1
        else:
            stats["n_differ"] += 1
        if old != line and stats["first_mismatch"] is None:
            stats["first_mismatch"] = "/".join(key)
    header_ok = ref_header == new_header == ",".join(EPISODE_COLUMNS)
    return {
        "reference": str(reference_csv),
        "reference_sha256": hashlib.sha256(ref_text.encode("utf-8")).hexdigest(),
        "pad": pad,
        "header_identical": header_ok,
        "all_identical": bool(
            header_ok
            and methods
            and all(s["n_identical"] == s["n_compared"] for s in methods.values())
        ),
        "methods": methods,
    }


def summary_provenance_differences(
    reference_summary: Path, live: Mapping[str, Mapping[str, str]]
) -> list[dict[str, str]]:
    """List per-method provenance columns whose committed value differs from the live one.

    Args:
        reference_summary: A committed ``summary.csv``.
        live: ``{method: {column: live value}}``
            (:func:`rld.eval.controller_config.controller_provenance_columns`).

    Returns:
        One ``{"method", "column", "committed", "live", "n_rows"}`` per (method, column,
        committed value) that differs; columns absent from the reference are skipped.
    """
    with reference_summary.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    counts: dict[tuple[str, str, str], int] = {}
    for record in records:
        method = record["method"]
        for column, value in live.get(method, {}).items():
            if column in record and record[column] != value:
                key = (method, column, record[column])
                counts[key] = counts.get(key, 0) + 1
    return [
        {
            "method": method,
            "column": column,
            "committed": committed,
            "live": live[method][column],
            "n_rows": str(n),
        }
        for (method, column, committed), n in counts.items()
    ]
