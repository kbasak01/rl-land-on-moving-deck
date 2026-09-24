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
privileged method is labelled in every table (``oracle_gated`` as a **commit-timing oracle
(privileged)**; the phrase "upper bound" is refused in rendered output, P3-D4); the required
caveats head the file.

Pads and skipped cells
----------------------
Summary rows are keyed by ``pad`` as well (the aft pad as frozen, and the pad-at-CG control
arm flown on the same lists). A ``summary.csv`` written before the column existed (``e01``)
is read as all-aft. A cell a method was **not** flown on (a feed-consuming controller on the
static-pad list) is a summary row with ``n_episodes = 0``, ``n_listed`` and a verbatim
``skip_reason``; it is rendered as "not run" with its reason, never dropped and never a zero.

Units: speeds metres per second model scale, lengths metres model scale, times seconds model
scale, angles degrees; rates and fractions dimensionless.
"""

import csv
import hashlib
import io
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from rld.envs.touchdown import OUTCOMES
from rld.eval.episodes import REGIME_CELLS
from rld.eval.metrics import CELL_METRIC_COLUMNS, CellMetrics, as_bool, as_float, cell_metrics
from rld.eval.stats import wilson_interval

__all__ = [
    "CELL_STATUS_COLUMNS",
    "QUIET_COLUMNS",
    "DEFAULT_PAD",
    "FORBIDDEN_RENDERED_PHRASES",
    "METHOD_LABELS",
    "PAD_ORDER",
    "SUMMARY_KEY_COLUMNS",
    "SkipRecord",
    "deterministic_provenance",
    "method_label",
    "read_rows",
    "render_success_vs_seastate",
    "secondary_seed_arm",
    "sha256_file",
    "summarise",
    "summary_columns",
    "write_rows",
]

#: Grouping columns of a summary row, in committed order. ``pad`` is new in Phase 4; a
#: summary without it (``e01``) is read as :data:`DEFAULT_PAD`.
SUMMARY_KEY_COLUMNS: tuple[str, ...] = ("method", "privileged", "run_seed", "pad", "regime", "ss")

#: Per-cell status columns, after the metrics: the listed N and why any of it was not run.
CELL_STATUS_COLUMNS: tuple[str, ...] = ("n_listed", "skip_reason")

#: Quiet-touchdown columns, after the status columns (results-skeptic M1, Gate 4). The
#: touchdown-conditional fraction ``frac_td_in_quiescent_window`` (a metric column) divides by
#: ``n_touchdowns``, so a method that rarely touches down can print a fraction near the
#: oracle's while landing quietly no more often than ``gated``. These columns give that
#: fraction its Wilson 95 % CI, and add the **per-listed-episode** rate
#: ``quiet_landings_per_listed = n_td_in_quiescent_window / n_listed`` with its own Wilson CI.
#: Both are NaN for a cell that was not run (``n_episodes == 0``).
QUIET_COLUMNS: tuple[str, ...] = (
    "frac_td_in_quiescent_window_wilson_lo",
    "frac_td_in_quiescent_window_wilson_hi",
    "quiet_landings_per_listed",
    "quiet_landings_per_listed_wilson_lo",
    "quiet_landings_per_listed_wilson_hi",
)

#: The pad of a summary or episode row that predates the ``pad`` key (the frozen aft pad).
DEFAULT_PAD: str = "aft"

#: Pads in rendering and summary order; any other pad follows, sorted.
PAD_ORDER: tuple[str, ...] = ("aft", "cg")

#: Pad headings in a multi-pad rendering.
_PAD_TITLES: dict[str, str] = {
    "aft": "aft pad (primary, as frozen in P3-D1)",
    "cg": "pad at CG (control arm for dmf's phase defect; same frozen episodes)",
}

#: Phrases that must never appear in rendered output (P3-D4: ``oracle_gated`` is a
#: commit-timing oracle, not a bound on success). Checked case-insensitively.
FORBIDDEN_RENDERED_PHRASES: tuple[str, ...] = ("upper bound",)

#: How a method is named in rendered tables. A privileged method's label says what it is.
METHOD_LABELS: dict[str, str] = {
    "oracle_gated": (
        "oracle_gated — commit-timing oracle (privileged): the gated rule applied to the true "
        "future deck motion"
    ),
    "pid_feedforward_lowvz": "pid_feedforward_lowvz (H1a closing-speed reference (P3-D1 §8))",
    "pid_feedforward_lowvz_cut": (
        "pid_feedforward_lowvz_cut (lowvz + latched post-contact throttle cut (P5-D2); "
        "not the H1a reference)"
    ),
    "gated_forecast": (
        "gated_forecast (not privileged: past-only ship-motion feed + dmf residual_interval band)"
    ),
    "gated_forecast_tcn": (
        "gated_forecast_tcn (not privileged: past-only ship-motion feed + dmf tcn_quantile band)"
    ),
}


@dataclass(frozen=True)
class SkipRecord:
    """One listed episode a method was not flown on (from :func:`rld.eval.runner.split_runnable`).

    Attributes:
        method: Method name.
        privileged: Its privileged flag.
        run_seed: Its run seed.
        pad: The effective pad of the arm.
        regime: The episode's regime.
        ss: The episode's sea state label.
        reason: Why, verbatim.
    """

    method: str
    privileged: bool
    run_seed: int
    pad: str
    regime: str
    ss: str
    reason: str


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


def _pad_rank(pad: str) -> tuple[int, str]:
    """Sort key: :data:`PAD_ORDER` first, then any other pad by name."""
    return (PAD_ORDER.index(pad), "") if pad in PAD_ORDER else (len(PAD_ORDER), pad)


def _quiet_columns(record: Mapping[str, Any]) -> dict[str, float]:
    """Return :data:`QUIET_COLUMNS` for one summary record (metrics and ``n_listed`` set).

    Args:
        record: A summary record holding ``n_episodes``, ``n_touchdowns``,
            ``n_td_in_quiescent_window`` and ``n_listed`` (counts).

    Returns:
        The Wilson 95 % CI of the touchdown-conditional quiet fraction (NaN without a
        touchdown), and quiet landings per listed episode with its Wilson 95 % CI (NaN for a
        cell that was not run). Dimensionless.
    """
    nan = float("nan")
    n_quiet = int(record["n_td_in_quiescent_window"])
    n_td = int(record["n_touchdowns"])
    n_listed = int(record["n_listed"])
    td_lo, td_hi = wilson_interval(n_quiet, n_td) if n_td > 0 else (nan, nan)
    if int(record["n_episodes"]) == 0:
        rate, lo, hi = nan, nan, nan
    else:
        rate = n_quiet / n_listed
        lo, hi = wilson_interval(n_quiet, n_listed)
    return {
        "frac_td_in_quiescent_window_wilson_lo": td_lo,
        "frac_td_in_quiescent_window_wilson_hi": td_hi,
        "quiet_landings_per_listed": rate,
        "quiet_landings_per_listed_wilson_lo": lo,
        "quiet_landings_per_listed_wilson_hi": hi,
    }


def _empty_metrics() -> dict[str, Any]:
    """Metrics of a cell with no flown episode: counts 0, every rate and statistic NaN."""
    out: dict[str, Any] = {}
    for f in fields(CellMetrics):
        out[f.name] = 0 if f.type in (int, "int") else float("nan")
    return out


def summarise(
    episode_rows: Sequence[Mapping[str, Any]],
    method_order: Sequence[str],
    per_method: Mapping[str, Mapping[str, str]] | None = None,
    provenance: Mapping[str, str] | None = None,
    skipped: Sequence[SkipRecord] = (),
) -> list[dict[str, Any]]:
    """Reduce per-episode rows to one row per (method, run seed, pad, regime, sea state).

    Args:
        episode_rows: Runner rows (in memory, or read back from ``episodes.csv`` as text).
            A row without a ``pad`` value counts as :data:`DEFAULT_PAD`.
        method_order: Methods in output order; every method in the rows must be listed.
        per_method: Extra columns per method (e.g. its config's SHA-256), inserted after
            the metrics.
        provenance: Deterministic provenance columns appended to every row.
        skipped: Listed episodes a method was not flown on, with the reason. A cell with
            no flown episode becomes a row with ``n_episodes = 0``; ``n_listed`` counts
            flown plus skipped, and ``skip_reason`` carries the reason(s) verbatim.

    Returns:
        Rows keyed :data:`SUMMARY_KEY_COLUMNS`, then
        :data:`~rld.eval.metrics.CELL_METRIC_COLUMNS`, then :data:`CELL_STATUS_COLUMNS` and
        :data:`QUIET_COLUMNS`,
        then the per-method and provenance columns. Ordered by pad (:data:`PAD_ORDER`),
        ``method_order``, run seed, then the committed cell order. Cells with neither a
        flown nor a skipped episode are absent -- and :func:`render_success_vs_seastate`
        prints them as missing rather than inventing a zero.

    Raises:
        ValueError: If a row's method is not in ``method_order``, or a method's
            ``privileged`` flag is not constant.
    """
    groups: dict[tuple[str, int, str, str, str], list[Mapping[str, Any]]] = {}
    skips: dict[tuple[str, int, str, str, str], list[SkipRecord]] = {}
    privileged: dict[str, bool] = {}

    def _flag(method: str, flag: bool) -> None:
        if method not in method_order:
            raise ValueError(f"method {method!r} not in {list(method_order)}")
        if privileged.setdefault(method, flag) != flag:
            raise ValueError(f"{method}: privileged flag is not constant")

    for row in episode_rows:
        method = str(row["method"])
        _flag(method, as_bool(row["privileged"]))
        pad = str(row.get("pad") or DEFAULT_PAD)
        key = (method, int(row["run_seed"]), pad, str(row["regime"]), str(row["ss"]))
        groups.setdefault(key, []).append(row)
    for skip in skipped:
        _flag(skip.method, bool(skip.privileged))
        key = (skip.method, int(skip.run_seed), skip.pad, skip.regime, skip.ss)
        skips.setdefault(key, []).append(skip)
    keys = set(groups) | set(skips)
    pads = sorted({key[2] for key in keys}, key=_pad_rank)
    cells = _cell_order()
    out: list[dict[str, Any]] = []
    for pad in pads:
        for method in method_order:
            seeds = sorted({key[1] for key in keys if key[0] == method and key[2] == pad})
            for seed in seeds:
                for regime, ss in cells:
                    key = (method, seed, pad, regime, ss)
                    rows = groups.get(key, [])
                    cell_skips = skips.get(key, [])
                    if not rows and not cell_skips:
                        continue
                    record: dict[str, Any] = {
                        "method": method,
                        "privileged": privileged[method],
                        "run_seed": seed,
                        "pad": pad,
                        "regime": regime,
                        "ss": ss,
                    }
                    record.update(asdict(cell_metrics(rows)) if rows else _empty_metrics())
                    record["n_listed"] = len(rows) + len(cell_skips)
                    reasons = dict.fromkeys(skip.reason for skip in cell_skips)
                    record["skip_reason"] = (
                        ""
                        if not cell_skips
                        else f"{len(cell_skips)} of {record['n_listed']} not run: "
                        + "; ".join(reasons)
                    )
                    record.update(_quiet_columns(record))
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
        The label from :data:`METHOD_LABELS`; for a seed-sensitivity arm
        (:func:`secondary_seed_arm`) ``"<method> (secondary: seed sensitivity; ...)"``; else
        ``"<method> (privileged)"`` for a privileged method, or the bare name.
    """
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    seed = secondary_seed_arm(method)
    if seed is not None:
        label = (
            f"{method} (secondary: seed sensitivity; {seed[0]} with its forecaster fitted at "
            f"training seed {seed[1]}, not the selected seed; not the primary arm)"
        )
        return f"{label} (privileged)" if privileged else label
    return f"{method} (privileged)" if privileged else method


def secondary_seed_arm(method: str) -> tuple[str, int] | None:
    """Return ``(primary method, seed)`` for a seed-sensitivity arm, else ``None``.

    A seed-sensitivity arm is a registry entry named ``<primary>_seed<k>`` (e.g.
    ``gated_forecast_tcn_seed0``): the primary controller with its forecaster fitted at
    another training seed. Its rows are secondary, never a headline.

    Args:
        method: Method name.

    Returns:
        ``(primary, k)`` or ``None``.
    """
    match = re.fullmatch(r"(.+)_seed(\d+)", method)
    return None if match is None else (match.group(1), int(match.group(2)))


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


def _caveats(pads: Sequence[str]) -> list[str]:
    """Return the caveat bullets; the phase-defect line names the pads in the table."""
    if list(pads) in ([], [DEFAULT_PAD]):
        return list(_CAVEATS)
    defect = (
        "dmf's roll/pitch-heave phase defect (~90 deg) is carried, not fixed. The aft pad "
        "(primary) is sensitive to it; the pad-at-CG control arm flies the same frozen "
        "episodes with the pad at the ship's CG, where the pad's v_z is the heave rate and "
        "the defect does not enter (P1-D2). The static-pad list has no lever arm, so its aft "
        "and CG rows fly the identical deck."
    )
    return [*_CAVEATS[:-1], defect]


def _check_phrases(text: str) -> str:
    """Return ``text`` unless it contains a :data:`FORBIDDEN_RENDERED_PHRASES` phrase.

    Raises:
        ValueError: If a forbidden phrase appears (case-insensitive).
    """
    lowered = text.lower()
    for phrase in FORBIDDEN_RENDERED_PHRASES:
        if phrase in lowered:
            raise ValueError(f"rendered output contains the forbidden phrase {phrase!r}")
    return text


_PREDICATE = (
    "TRUE deck satisfied the permissive quiescence predicate "
    "(`rld.control.quiescence.QuiescenceRule`: 12 samples 1/30 s apart, |roll| <= 3.0 deg, "
    "|pitch| <= 2.0 deg, |pad v_z| <= 0.16 m/s model) starting at the contact touchdown; "
    "evaluation ground truth, never a controller input"
)

#: The quiet-touchdown bullet of a summary without :data:`QUIET_COLUMNS` (e01, unchanged).
_QUIET_BULLETS_V1: tuple[str, ...] = (
    f"- `quiet td` is, of the touched-down episodes, the fraction whose {_PREDICATE}. `–` "
    "when nothing touched down.",
)

#: The quiet-touchdown bullets once :data:`QUIET_COLUMNS` exist (results-skeptic M1).
_QUIET_BULLETS_V2: tuple[str, ...] = (
    "- `td n` is the number of episodes that touched down (contact detector).",
    "- `quiet | td` (the column `quiet \\| td`) is, **of the touched-down episodes only**, "
    "the fraction whose "
    f"{_PREDICATE}; Wilson 95 % CI in brackets, then k/`td n`. `–` when nothing touched down. "
    "It is conditional on touching down: a controller that rarely touches down can score "
    "high here while landing quietly no more often than another.",
    "- `quiet landings / listed` is the number of quiet touchdowns divided by the number of "
    "listed episodes in the cell (timeouts count as not quiet); Wilson 95 % CI in brackets, "
    "then k/N. This is the per-episode rate to compare across controllers.",
)


def _quiet_cells(record: Mapping[str, str]) -> list[str]:
    """Return ``td n``, ``quiet | td`` and ``quiet landings / listed`` cells for one record."""
    n_quiet = record["n_td_in_quiescent_window"]
    return [
        record["n_touchdowns"],
        f"{_num(as_float(record['frac_td_in_quiescent_window']), 3)} "
        f"[{_num(as_float(record['frac_td_in_quiescent_window_wilson_lo']), 3)}, "
        f"{_num(as_float(record['frac_td_in_quiescent_window_wilson_hi']), 3)}] "
        f"({n_quiet}/{record['n_touchdowns']})",
        f"{_num(as_float(record['quiet_landings_per_listed']), 3)} "
        f"[{_num(as_float(record['quiet_landings_per_listed_wilson_lo']), 3)}, "
        f"{_num(as_float(record['quiet_landings_per_listed_wilson_hi']), 3)}] "
        f"({n_quiet}/{record['n_listed']})",
    ]


def _is_skipped(record: Mapping[str, str]) -> bool:
    """Return whether a summary record is a cell with no flown episode."""
    return int(record["n_episodes"]) == 0


def render_success_vs_seastate(summary: Sequence[Mapping[str, str]], title: str) -> str:
    """Render the success-versus-sea-state tables from ``summary.csv`` text records.

    Args:
        summary: ``summary.csv`` rows as read by :func:`read_rows` -- text, not floats. A
            record without ``pad`` is :data:`DEFAULT_PAD`.
        title: The document title.

    Returns:
        The markdown text: caveats, then per pad (a heading only when there is more than
        one) and per regime a headline table (success % with Wilson 95 % CI and k/N, one
        column per sea state, never pooled) and a breakdown table (six outcome fractions,
        touchdown quality, audit counts) per (method, sea state). A cell that was not run
        is printed as such, with its reason listed at the top.

    Raises:
        ValueError: If the text would contain a phrase in
            :data:`FORBIDDEN_RENDERED_PHRASES`.
    """

    def pad_of(record: Mapping[str, str]) -> str:
        return record.get("pad") or DEFAULT_PAD

    methods = list(dict.fromkeys(r["method"] for r in summary))
    pads = sorted({pad_of(r) for r in summary}, key=_pad_rank)
    by_key = {(r["method"], r["run_seed"], pad_of(r), r["regime"], r["ss"]): r for r in summary}
    seeds = {
        m: list(dict.fromkeys(r["run_seed"] for r in summary if r["method"] == m)) for m in methods
    }
    multi_pad = len(pads) > 1
    quiet_v2 = bool(summary) and all(c in summary[0] for c in QUIET_COLUMNS)
    lines: list[str] = [f"# {title}", ""]
    lines += [f"- {caveat}" for caveat in _caveats(pads)]
    lines += [
        "- Success = all four frozen criteria (`configs/env/success.yaml`); rates in %, "
        "Wilson 95 % CI in brackets, then k/N. Success is never pooled across sea states.",
        "- `in-dist` is the fraction of the cell's episodes whose grid cell is in the "
        "development pool (P3-D2); `id` SS6 and every 90 deg episode are outside it.",
        *(_QUIET_BULLETS_V2 if quiet_v2 else _QUIET_BULLETS_V1),
        "- `oracle_gated` is privileged: it reads the true future deck motion. It bounds commit "
        "timing under the gated rule, not success, and is never a deployable result.",
    ]
    if any(m.startswith("gated_forecast") for m in methods):
        lines.append(
            "- `gated_forecast` and `gated_forecast_tcn` are not privileged: they read the "
            "observation plus the past-only history of dmf's six clean ship channels (an ideal "
            "noise-free, zero-latency ship motion reference unit that never reaches past the "
            "runner's clock) and a dmf forecaster fitted on the P3-D2 dev pool (P4-D1, P4-D3)."
        )
    secondary = [m for m in methods if secondary_seed_arm(m) is not None]
    if secondary:
        lines.append(
            "- Rows labelled **secondary: seed sensitivity** ("
            + ", ".join(f"`{m}`" for m in secondary)
            + ") fly a primary controller with its forecaster fitted at another training "
            "seed than the selected one (P4-D2), on the same episodes. They show how much the "
            "primary arm's numbers depend on the forecaster's seed. They are not additional "
            "methods, and they never replace the primary row."
        )
    if multi_pad:
        lines.append(
            "- Pad: `aft` is the frozen primary; `cg` is the pad-at-CG control arm, flown on "
            "the identical listed episodes (same t0, realization and initial state)."
        )
    skipped = [r for r in summary if _is_skipped(r)]
    if skipped:
        lines.append("- Cells not run (never dropped; listed here and marked in the tables):")
        for skip in skipped:
            lines.append(
                f"  - {skip['method']}, pad {pad_of(skip)}, {skip['regime']} {skip['ss']}: "
                f"{skip.get('skip_reason', '')}"
            )
    lines += ["- Rendered from `summary.csv` by `rld.eval.report`; do not edit by hand.", ""]
    for pad in pads:
        suffix = f" — {_PAD_TITLES.get(pad, f'pad {pad}')}" if multi_pad else ""
        for regime, sea_states in REGIME_CELLS:
            lines += [f"## {regime}{suffix}", ""]
            header = "| method | " + " | ".join(sea_states) + " |"
            lines += [header, "|---|" + "---|" * len(sea_states)]
            for method in methods:
                for seed in seeds[method]:
                    cells = []
                    flag = False
                    for ss in sea_states:
                        rec = by_key.get((method, seed, pad, regime, ss))
                        if rec is None:
                            cells.append("missing")
                        elif _is_skipped(rec):
                            cells.append(f"not run (0/{rec.get('n_listed', '?')}; see above)")
                        else:
                            cells.append(_success_cell(rec))
                        flag = flag or (rec is not None and as_bool(rec["privileged"]))
                    name = method_label(method, flag)
                    if len(seeds[method]) > 1:
                        name += f" seed {seed}"
                    lines.append(f"| {name} | " + " | ".join(cells) + " |")
            lines += ["", f"### {regime}{suffix}: outcome breakdown and touchdown audit", ""]
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
                *(
                    ("td n", "quiet \\| td", "quiet landings / listed")
                    if quiet_v2
                    else ("quiet td",)
                ),
                "in-dist",
            ]
            lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            for method in methods:
                for seed in seeds[method]:
                    for ss in sea_states:
                        rec = by_key.get((method, seed, pad, regime, ss))
                        if rec is None or _is_skipped(rec):
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
                            *(
                                _quiet_cells(rec)
                                if quiet_v2
                                else [_num(as_float(rec["frac_td_in_quiescent_window"]), 3)]
                            ),
                            _num(as_float(rec["frac_in_training_distribution"]), 3),
                        ]
                        lines.append("| " + " | ".join(str(v) for v in values) + " |")
            lines.append("")
    return _check_phrases("\n".join(lines))


def summary_columns(extra: Iterable[str]) -> list[str]:
    """Return the committed ``summary.csv`` column order.

    Args:
        extra: Per-method and provenance column names, in order.

    Returns:
        :data:`SUMMARY_KEY_COLUMNS` + :data:`~rld.eval.metrics.CELL_METRIC_COLUMNS` +
        :data:`CELL_STATUS_COLUMNS` + :data:`QUIET_COLUMNS` + ``extra``.
    """
    return [
        *SUMMARY_KEY_COLUMNS,
        *CELL_METRIC_COLUMNS,
        *CELL_STATUS_COLUMNS,
        *QUIET_COLUMNS,
        *extra,
    ]
