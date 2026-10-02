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
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rld.eval import storage
from rld.eval.report import SkipRecord, read_rows, summarise, summary_columns
from rld.eval.runner import EPISODE_COLUMNS, episode_rows_csv

__all__ = [
    "compare_to_reference",
    "git_state",
    "resummarise",
    "skip_records_from_summary",
    "summary_provenance_differences",
]

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
        ``{"reference": path, "reference_sha256", "pad", "applicable", "header_identical",
        "all_identical", "methods": {method: {"n_compared", "n_identical", "n_differ",
        "n_missing_in_reference", "n_reference_rows", "first_mismatch"}}}``.
    """
    ref_text = storage.read_text(reference_csv)  # its .csv.gz when present (P7-D1a §12)
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
        # False when no evaluated method is in the reference (e.g. seed-sensitivity arms
        # only): there is nothing to reproduce, and the check neither passes nor fails.
        "applicable": bool(methods),
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


def git_state(repo_root: Path) -> dict[str, Any]:
    """Return the checkout's commit and whether it has uncommitted changes.

    Args:
        repo_root: The repository root.

    Returns:
        ``{"git_sha": <HEAD or "unknown">, "git_dirty": bool | None, "git_dirty_paths":
        [porcelain lines, at most 50]}``; ``git_dirty`` is None when git is unavailable.
    """

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=normal")
    paths = [] if status is None else [line for line in status.splitlines() if line]
    return {
        "git_sha": "unknown" if sha is None else sha.strip(),
        "git_dirty": None if status is None else bool(paths),
        "git_dirty_paths": paths[:50],
    }


_SKIP_PREFIX = re.compile(r"^\d+ of \d+ not run: (.*)$")


def skip_records_from_summary(records: Sequence[Mapping[str, str]]) -> list[SkipRecord]:
    """Rebuild the :class:`SkipRecord` s behind a summary's not-run cells.

    Args:
        records: ``summary.csv`` rows as text; those with ``n_episodes == 0`` are expanded
            into ``n_listed`` records with the reason parsed from ``skip_reason``.

    Returns:
        The records, in summary order.

    Raises:
        ValueError: If a not-run cell's ``skip_reason`` cannot be parsed.
    """
    skipped: list[SkipRecord] = []
    for record in records:
        if int(record["n_episodes"]) != 0:
            continue
        match = _SKIP_PREFIX.match(record["skip_reason"])
        if match is None:
            raise ValueError(f"cannot parse skip_reason {record['skip_reason']!r}")
        skipped += [
            SkipRecord(
                record["method"],
                record["privileged"] == "True",
                int(record["run_seed"]),
                record.get("pad") or "aft",
                record["regime"],
                record["ss"],
                match.group(1),
            )
        ] * int(record["n_listed"])
    return skipped


def resummarise(out_dir: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Recompute ``summary.csv`` from ``episodes.csv`` with today's columns, changing nothing old.

    No episode is flown. The method order, the skipped cells (``n_episodes == 0`` rows:
    their ``n_listed`` and reason) and every per-method and provenance column are taken
    from the existing ``summary.csv``, as recorded when the run was made; only the
    metrics are recomputed from the episodes. Every column the old file has must come out
    byte-identical, row for row, or nothing is returned.

    Args:
        out_dir: A run directory holding ``episodes.csv`` and ``summary.csv``.

    Returns:
        ``(rows, columns, added_columns)``: the new summary rows (write with
        :func:`rld.eval.report.write_rows` and ``columns``) and the columns that are new.

    Raises:
        ValueError: If a provenance column is not constant within a method, a skip reason
            cannot be parsed, or any existing column would change.
    """
    episodes = read_rows(out_dir / "episodes.csv")
    old = read_rows(out_dir / "summary.csv")
    old_columns = list(old[0])
    methods = list(dict.fromkeys(r["method"] for r in old))
    base = summary_columns([])
    extras = [c for c in old_columns if c not in base]
    per_method: dict[str, dict[str, str]] = {}
    for record in old:
        values = {c: record[c] for c in extras}
        if per_method.setdefault(record["method"], values) != values:
            raise ValueError(f"{record['method']}: provenance columns are not constant")
    skipped = skip_records_from_summary(old)
    rows = summarise(episodes, methods, per_method, None, skipped)
    columns = summary_columns(extras)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(
            [repr(float(row[c])) if isinstance(row[c], float) else row[c] for c in columns]
        )
    new = list(csv.DictReader(io.StringIO(buffer.getvalue())))
    if len(new) != len(old):
        raise ValueError(f"{len(new)} summary rows recomputed, {len(old)} committed")
    for fresh, committed in zip(new, old, strict=True):
        for column in old_columns:
            if fresh.get(column) != committed[column]:
                raise ValueError(
                    f"{committed['method']}/{committed.get('pad', 'aft')}/{committed['regime']}"
                    f"/{committed['ss']}: column {column} {committed[column]!r} -> "
                    f"{fresh.get(column)!r}"
                )
    return rows, columns, [c for c in columns if c not in old_columns]
