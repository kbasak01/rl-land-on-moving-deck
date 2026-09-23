"""Committed evaluation tables: per-cell summaries from episode rows, markdown from the CSV.

Two rules this module exists to enforce:

1. **The CSVs are byte-reproducible.** Every value is written by :func:`write_rows` with
   ``repr`` floats (shortest exact round trip, ``nan`` for missing) and newline line endings,
   and the only provenance written *into* a CSV is deterministic: package versions,
   submodule SHAs and the SHA-256 of every config and episode list the numbers depend on
   (:func:`deterministic_provenance`). Wall-clock, host, worker count and timestamp live in a
   sidecar ``run_info.json``, so two runs at different worker counts produce identical CSVs.
2. **Markdown is rendered from the CSV, never from memory.** :func:`render_success_vs_seastate`
   takes the ``summary.csv`` records as text and nothing else, so re-rendering the committed
   CSV reproduces the committed markdown byte for byte.

Reporting rules carried into the rendering (``landing-protocol`` skill, P3-D1): success is
never printed without its Wilson 95 % CI, its ``N`` and the six-class outcome breakdown;
success is **never pooled across sea states** (there is no "overall" column); every
privileged method is labelled in every table; the required caveats head the file.

Units: speeds metres per second model scale, lengths metres model scale, times seconds model
scale, angles degrees; rates and fractions dimensionless.
"""

import csv
import hashlib
import io
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rld.envs.touchdown import OUTCOMES
from rld.eval.episodes import REGIME_CELLS
from rld.eval.metrics import CELL_METRIC_COLUMNS, as_bool, as_float, cell_metrics

__all__ = [
    "METHOD_LABELS",
    "SUMMARY_KEY_COLUMNS",
    "deterministic_provenance",
    "method_label",
    "read_rows",
    "render_success_vs_seastate",
    "sha256_file",
    "summarise",
    "summary_columns",
    "write_rows",
]

#: Grouping columns of a summary row, in committed order.
SUMMARY_KEY_COLUMNS: tuple[str, ...] = ("method", "privileged", "run_seed", "regime", "ss")

#: How a method is named in rendered tables. A privileged method's label says what it bounds.
METHOD_LABELS: dict[str, str] = {
    "oracle_gated": (
        "oracle_gated (privileged — upper bound on commit timing under the quiescence rule, "
        "not on success)"
    ),
}

#: Caveats every rendered results file carries (``landing-protocol`` skill).
_CAVEATS: tuple[str, ...] = (
    "Simulation only (PyBullet); no real flight and no real deck data.",
    "Deck motion is dmf's 3-DOF (heave, roll, pitch) JONSWAP response, Froude-scaled to a "
    "Crazyflie at lambda = 1/25 (1 s model = 5 s full scale).",
    "State-based observations; the perception-noise stand-in is disabled "
    "(`configs/env/noise.yaml: enabled: false`); not vision.",
    "dmf's roll/pitch-heave phase defect (~90 deg) is carried, not fixed; this table is the "
    "aft pad, which is sensitive to it (P1-D2).",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_provenance(
    versions: Mapping[str, str], hashed_files: Mapping[str, Path], extra: Mapping[str, str]
) -> dict[str, str]:
    """Return the provenance columns a reproducible CSV may carry.

    Args:
        versions: Package versions and submodule SHAs (no host, time or worker count).
        hashed_files: ``{column name: file}``; each becomes the file's SHA-256.
        extra: Further deterministic facts, already strings (e.g. the generator seed).

    Returns:
        ``{column: value}`` in the order given: versions, file hashes, extras.
    """
    return {
        **dict(versions),
        **{name: sha256_file(path) for name, path in hashed_files.items()},
        **dict(extra),
    }


def _cell_order() -> list[tuple[str, str]]:
    """Return every (regime, sea state) cell in committed order."""
    return [(regime, ss) for regime, sea_states in REGIME_CELLS for ss in sea_states]


def summarise(
    episode_rows: Sequence[Mapping[str, Any]],
    method_order: Sequence[str],
    per_method: Mapping[str, Mapping[str, str]] | None = None,
    provenance: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Reduce per-episode rows to one row per (method, run seed, regime, sea state).

    Args:
        episode_rows: Runner rows (in memory, or read back from ``episodes.csv`` as text).
        method_order: Methods in output order; every method in the rows must be listed.
        per_method: Extra columns per method (e.g. its config's SHA-256), inserted after
            the metrics.
        provenance: Deterministic provenance columns appended to every row.

    Returns:
        Rows keyed :data:`SUMMARY_KEY_COLUMNS`, then
        :data:`~rld.eval.metrics.CELL_METRIC_COLUMNS`, then the per-method and provenance
        columns. Ordered by ``method_order``, run seed, then the committed cell order.
        Cells with no episodes are absent -- and :func:`render_success_vs_seastate` prints
        them as missing rather than inventing a zero.

    Raises:
        ValueError: If a row's method is not in ``method_order``, or a method's
            ``privileged`` flag is not constant.
    """
    groups: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = {}
    privileged: dict[str, bool] = {}
    for row in episode_rows:
        method = str(row["method"])
        if method not in method_order:
            raise ValueError(f"method {method!r} not in {list(method_order)}")
        flag = as_bool(row["privileged"])
        if privileged.setdefault(method, flag) != flag:
            raise ValueError(f"{method}: privileged flag is not constant")
        key = (method, int(row["run_seed"]), str(row["regime"]), str(row["ss"]))
        groups.setdefault(key, []).append(row)
    cells = _cell_order()
    out: list[dict[str, Any]] = []
    for method in method_order:
        seeds = sorted({key[1] for key in groups if key[0] == method})
        for seed in seeds:
            for regime, ss in cells:
                rows = groups.get((method, seed, regime, ss))
                if not rows:
                    continue
                record: dict[str, Any] = {
                    "method": method,
                    "privileged": privileged[method],
                    "run_seed": seed,
                    "regime": regime,
                    "ss": ss,
                }
                record.update(asdict(cell_metrics(rows)))
                record.update(dict((per_method or {}).get(method, {})))
                record.update(dict(provenance or {}))
                out.append(record)
    return out


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    """Write rows as a byte-reproducible CSV.

    Args:
        path: Target file; its directory is created if missing.
        rows: Records holding at least ``columns``.
        columns: Column order.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError(f"refusing to write an empty table to {path}")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(
            [repr(float(row[c])) if isinstance(row[c], float) else row[c] for c in columns]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV back as text records, in file order."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def method_label(method: str, privileged: bool) -> str:
    """Return a method's rendered name; a privileged method is always marked.

    Args:
        method: Method name.
        privileged: Its privileged flag.

    Returns:
        The label from :data:`METHOD_LABELS`, or ``"<method> (privileged)"`` for a privileged
        method without one, or the bare name.
    """
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    return f"{method} (privileged)" if privileged else method


def _pct(value: float) -> str:
    """Format a rate in ``[0, 1]`` as a percentage with one decimal, or ``–`` for NaN."""
    return "–" if value != value else f"{100.0 * value:.1f}"


def _num(value: float, digits: int) -> str:
    """Format a number with fixed decimals, or ``–`` for NaN."""
    return "–" if value != value else f"{value:.{digits}f}"


def _success_cell(record: Mapping[str, str]) -> str:
    """Return ``"rate [lo, hi] (k/N)"`` in percent for one summary record."""
    return (
        f"{_pct(as_float(record['success_rate']))} "
        f"[{_pct(as_float(record['success_wilson_lo']))}, "
        f"{_pct(as_float(record['success_wilson_hi']))}] "
        f"({record['n_success']}/{record['n_episodes']})"
    )


def render_success_vs_seastate(summary: Sequence[Mapping[str, str]], title: str) -> str:
    """Render the success-versus-sea-state tables from ``summary.csv`` text records.

    Args:
        summary: ``summary.csv`` rows as read by :func:`read_rows` -- text, not floats.
        title: The document title.

    Returns:
        The markdown text: caveats, then per regime a headline table (success % with
        Wilson 95 % CI and k/N, one column per sea state, never pooled) and a breakdown
        table (six outcome fractions, touchdown quality, audit counts) per (method, sea
        state).
    """
    methods = list(dict.fromkeys(r["method"] for r in summary))
    by_key = {(r["method"], r["run_seed"], r["regime"], r["ss"]): r for r in summary}
    seeds = {
        m: list(dict.fromkeys(r["run_seed"] for r in summary if r["method"] == m)) for m in methods
    }
    lines: list[str] = [f"# {title}", ""]
    lines += [f"- {caveat}" for caveat in _CAVEATS]
    lines += [
        "- Success = all four frozen criteria (`configs/env/success.yaml`); rates in %, "
        "Wilson 95 % CI in brackets, then k/N. Success is never pooled across sea states.",
        "- `in-dist` is the fraction of the cell's episodes whose grid cell is in the "
        "development pool (P3-D2); `id` SS6 and every 90 deg episode are outside it.",
        "- Rendered from `summary.csv` by `rld.eval.report`; do not edit by hand.",
        "",
    ]
    for regime, sea_states in REGIME_CELLS:
        lines += [f"## {regime}", ""]
        header = "| method | " + " | ".join(sea_states) + " |"
        lines += [header, "|---|" + "---|" * len(sea_states)]
        for method in methods:
            for seed in seeds[method]:
                cells = []
                flag = False
                for ss in sea_states:
                    rec = by_key.get((method, seed, regime, ss))
                    cells.append("missing" if rec is None else _success_cell(rec))
                    flag = flag or (rec is not None and as_bool(rec["privileged"]))
                name = method_label(method, flag)
                if len(seeds[method]) > 1:
                    name += f" seed {seed}"
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", f"### {regime}: outcome breakdown and touchdown audit", ""]
        cols = [
            "method",
            "SS",
            "N",
            *OUTCOMES,
            "v_n p50 / p95 (m/s)",
            "v_z p95 (m/s)",
            "lat p95 (m)",
            "tilt p95 (deg)",
            "t_td p50 (s)",
            "disagree",
            "tunnel",
            "in-dist",
        ]
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for method in methods:
            for seed in seeds[method]:
                for ss in sea_states:
                    rec = by_key.get((method, seed, regime, ss))
                    if rec is None:
                        continue
                    name = method_label(method, as_bool(rec["privileged"]))
                    if len(seeds[method]) > 1:
                        name += f" seed {seed}"
                    values = [
                        name,
                        ss,
                        rec["n_episodes"],
                        *(_num(as_float(rec[f"frac_{o}"]), 3) for o in OUTCOMES),
                        f"{_num(as_float(rec['rel_vz_normal_p50_m_s']), 3)} / "
                        f"{_num(as_float(rec['rel_vz_normal_p95_m_s']), 3)}",
                        _num(as_float(rec["rel_vz_world_z_p95_m_s"]), 3),
                        _num(as_float(rec["lateral_p95_m"]), 3),
                        _num(as_float(rec["rel_tilt_p95_deg"]), 1),
                        _num(as_float(rec["time_to_touchdown_p50_s"]), 2),
                        f"{rec['disagreement_n']}/{rec['n_episodes']}",
                        rec["tunnelling_n"],
                        _num(as_float(rec["frac_in_training_distribution"]), 3),
                    ]
                    lines.append("| " + " | ".join(str(v) for v in values) + " |")
        lines.append("")
    return "\n".join(lines)


def summary_columns(extra: Iterable[str]) -> list[str]:
    """Return the committed ``summary.csv`` column order.

    Args:
        extra: Per-method and provenance column names, in order.

    Returns:
        :data:`SUMMARY_KEY_COLUMNS` + :data:`~rld.eval.metrics.CELL_METRIC_COLUMNS` +
        ``extra``.
    """
    return [*SUMMARY_KEY_COLUMNS, *CELL_METRIC_COLUMNS, *extra]
