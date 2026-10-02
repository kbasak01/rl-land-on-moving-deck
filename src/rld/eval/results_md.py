r"""Render ``results/results.md`` from the committed Phase 7 CSVs, and nothing else.

Called as :func:`rld.eval.report.render_results` (``scripts/report.py``). Reads only CSV files
under ``results/e07/`` (each condition's ``summary.csv``, ``aggregate.csv``, ``seeds.csv``,
``baselines_summary.csv`` and ``carried_summary_*.csv``; ``contrasts.csv``;
``hypotheses.csv``; ``lambda/feasibility.csv``) plus, for the MSS comparison only, the matrix
``episodes.csv`` and ``results/e01/episodes.csv``. Re-rendering the committed CSVs reproduces
the committed file byte for byte (``scripts/report.py --check``).

Reporting rules enforced here (P3-D1, ``landing-protocol`` skill, CLAUDE.md):

* success is never pooled across sea states: every success number sits in a sea-state
  column or row;
* a learned method's success is the IQM over its 5 seeds with the stratified-bootstrap 95 %
  CI and the per-seed range; a baseline's is its rate with the Wilson 95 % CI and k/N; per-seed
  Wilson CIs are in the appendix;
* every success table is followed by the six-class outcome breakdown for the same cells;
* the always-printed baselines ``pid_track_descend``, ``pid_feedforward`` and ``oracle_gated``
  are in every method table; ``oracle_gated`` is labelled "commit-timing oracle
  (privileged)"; the phrase refused by P3-D4 never appears;
* the caveats of P7-D1 §8 head the file, and each hypothesis row names its caveat keys.

Units: success and outcome fractions printed in %; closing speeds m/s model scale; sigmas cm
and m/s model scale; latency in control steps and ms model scale.
"""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from rld.envs.touchdown import OUTCOMES
from rld.eval.arms import (
    ALWAYS_PRINTED,
    BASELINES,
    E07_DIR,
    LAMBDA_INVERSES,
    LEARNED_METHODS,
    RESULTS_DIR,
    Condition,
    conditions,
)
from rld.eval.episodes import REGIME_CELLS
from rld.eval.hypotheses import CAVEATS
from rld.eval.metrics import as_float
from rld.eval.report import FORBIDDEN_RENDERED_PHRASES, method_label, read_rows, render_results

__all__ = ["RESULTS_NAME", "main", "render"]

#: The rendered file under ``results/``.
RESULTS_NAME: str = "results.md"

#: How a learned method is described in a table label.
_LEARNED_NOTES: dict[str, str] = {
    "ppo": "learned, PPO",
    "sac": "learned, SAC, 2 M env steps (PPO family: 10 M)",
    "residual_ppo": "learned, residual on pid_feedforward",
    "ppo_forecast": "learned, + past-only ship-motion feed (extra ideal sensor)",
    "residual_ppo_forecast": "learned, residual + ship-motion feed (extra ideal sensor)",
    "ppo_sinusoid": "learned, trained on sinusoid motion only",
}

_ORACLE_SHORT = "commit-timing oracle (privileged)"


def _label(method: str) -> str:
    """Return a method's table label (privileged and learned methods always marked)."""
    if method in _LEARNED_NOTES:
        return f"`{method}` ({_LEARNED_NOTES[method]})"
    if method == "oracle_gated":
        return f"`oracle_gated` — {_ORACLE_SHORT}"
    full = method_label(method, False)
    return f"`{method}`" + (full[len(method) :] if full.startswith(method) else "")


def _pct(value: float) -> str:
    return "–" if value != value else f"{100.0 * value:.1f}"


def _signed_pts(value: float) -> str:
    return "–" if value != value else f"{100.0 * value:+.1f}"


def _num(value: float, digits: int) -> str:
    return "–" if value != value else f"{value:.{digits}f}"


def _f(rec: Mapping[str, str], col: str) -> float:
    return as_float(rec[col])


# --------------------------------------------------------------------------- data


class CondData:
    """The committed CSVs of one condition directory, indexed by cell."""

    def __init__(self, directory: Path) -> None:
        """Read whatever of the condition's CSVs exist.

        Args:
            directory: ``results/e07/<arm>[/<condition>]``.
        """
        self.dir = directory
        self.present = (directory / "summary.csv").is_file()
        self.learned: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
        self.agg: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}
        self.spread: dict[tuple[str, str, str, str], dict[str, str]] = {}
        self.base: dict[tuple[str, str, str, str], dict[str, str]] = {}
        self.base_source: dict[tuple[str, str, str, str], str] = {}
        if not self.present:
            return
        for rec in read_rows(directory / "summary.csv"):
            key = (rec["method"], rec["pad"], rec["regime"], rec["ss"])
            self.learned.setdefault(key, []).append(rec)
        for rec in read_rows(directory / "aggregate.csv"):
            k = (rec["method"], rec["pad"], rec["regime"], rec["ss"], rec["metric"])
            self.agg[(*k, rec["statistic"])] = rec
        for rec in read_rows(directory / "seeds.csv"):
            self.spread[(rec["method"], rec["pad"], rec["regime"], rec["ss"])] = rec
        sources = [("flown here", directory / "baselines_summary.csv")]
        sources += [
            (f"carried from `results/{p.stem.removeprefix('carried_summary_')}`", p)
            for p in sorted(directory.glob("carried_summary_*.csv"))
        ]
        for origin, path in sources:
            if not path.is_file():
                continue
            for rec in read_rows(path):
                key = (rec["method"], rec.get("pad") or "aft", rec["regime"], rec["ss"])
                self.base[key] = rec
                self.base_source[key] = origin

    def has(self, method: str, pad: str, regime: str, ss: str) -> bool:
        """Return whether the cell exists (flown, carried or recorded as not run)."""
        key = (method, pad, regime, ss)
        return key in self.learned or key in self.base

    def records(self, method: str, pad: str, regime: str, ss: str) -> list[dict[str, str]]:
        """Return the cell's per-seed (learned) or single (baseline) summary records."""
        key = (method, pad, regime, ss)
        return self.learned.get(key) or ([self.base[key]] if key in self.base else [])

    def success_cell(self, method: str, pad: str, regime: str, ss: str) -> str:
        """Return the formatted success of a cell (IQM [CI]; seeds range, or Wilson k/N)."""
        recs = self.records(method, pad, regime, ss)
        if not recs:
            return "missing"
        if all(int(r["n_episodes"]) == 0 for r in recs):
            return f"not run (0/{recs[0]['n_listed']}; reason above)"
        if method in LEARNED_METHODS:
            a = self.agg.get((method, pad, regime, ss, "success_rate", "iqm"))
            s = self.spread.get((method, pad, regime, ss))
            if a is None or s is None:
                return "missing"
            n = sum(int(r["n_episodes"]) for r in recs)
            return (
                f"**{_pct(_f(a, 'point'))}** [{_pct(_f(a, 'ci_lo'))}, {_pct(_f(a, 'ci_hi'))}]; "
                f"seeds {_pct(_f(s, 'success_min'))}–{_pct(_f(s, 'success_max'))}; "
                f"N {len(recs)}×{recs[0]['n_episodes']}={n}"
            )
        rec = recs[0]
        return (
            f"{_pct(_f(rec, 'success_rate'))} [{_pct(_f(rec, 'success_wilson_lo'))}, "
            f"{_pct(_f(rec, 'success_wilson_hi'))}] {rec['n_success']}/{rec['n_episodes']}"
        )

    def success_value(self, method: str, pad: str, regime: str, ss: str) -> float:
        """Return the IQM (learned) or rate (baseline) of a flown cell; NaN otherwise."""
        recs = [r for r in self.records(method, pad, regime, ss) if int(r["n_episodes"]) > 0]
        if not recs:
            return float("nan")
        if method in LEARNED_METHODS:
            a = self.agg.get((method, pad, regime, ss, "success_rate", "iqm"))
            return float("nan") if a is None else _f(a, "point")
        return _f(recs[0], "success_rate")

    def outcome_counts(self, method: str, pad: str, regime: str, ss: str) -> tuple[int, list[int]]:
        """Return ``(N, counts per outcome)`` pooled over the cell's seeds."""
        recs = [r for r in self.records(method, pad, regime, ss) if int(r["n_episodes"]) > 0]
        n = sum(int(r["n_episodes"]) for r in recs)
        counts = [
            sum(round(_f(r, f"frac_{o}") * int(r["n_episodes"])) for r in recs) for o in OUTCOMES
        ]
        return n, counts

    def p95_cell(self, method: str, pad: str, regime: str, ss: str) -> str:
        """Return p95 closing speed (IQM of per-seed p95 for learned rows), m/s."""
        recs = [r for r in self.records(method, pad, regime, ss) if int(r["n_episodes"]) > 0]
        if not recs:
            return "–"
        if method in LEARNED_METHODS:
            a = self.agg.get((method, pad, regime, ss, "rel_vz_normal_p95_m_s", "iqm"))
            if a is None or a["reported"] != "True":
                return "–"
            return (
                f"{_num(_f(a, 'point'), 3)} [{_num(_f(a, 'ci_lo'), 3)}, {_num(_f(a, 'ci_hi'), 3)}]"
            )
        return _num(_f(recs[0], "rel_vz_normal_p95_m_s"), 3)

    def skip_reasons(self) -> list[str]:
        """Return the distinct ``skip_reason`` texts of not-run cells."""
        reasons = [
            r["skip_reason"]
            for recs in self.learned.values()
            for r in recs
            if int(r["n_episodes"]) == 0 and r["skip_reason"]
        ]
        return list(dict.fromkeys(reasons))


def _methods(data: Sequence[CondData], pad: str) -> list[str]:
    """Return every method present (learned order, then baseline order), always-printed first."""
    seen = {k[0] for d in data for k in (*d.learned, *d.base) if k[1] == pad}
    learned = [m for m in LEARNED_METHODS if m in seen]
    base = [m for m in BASELINES if m in seen or m in ALWAYS_PRINTED]
    return [*learned, *base]


def _ss(regime: str) -> tuple[str, ...]:
    return dict(REGIME_CELLS).get(regime, ())


# --------------------------------------------------------------------------- sections


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return [*out, ""]


def _breakdown(
    data: CondData, methods: Sequence[str], pad: str, regime: str, sea_states: Sequence[str]
) -> list[str]:
    rows = []
    for m in methods:
        for ss in sea_states:
            if not data.has(m, pad, regime, ss):
                rows.append([_label(m), ss, "missing", *[""] * len(OUTCOMES)])
                continue
            n, counts = data.outcome_counts(m, pad, regime, ss)
            if n == 0:
                rows.append([_label(m), ss, "not run", *[""] * len(OUTCOMES)])
                continue
            rows.append([_label(m), ss, str(n), *(f"{c} ({_pct(c / n)})" for c in counts)])
    return _table(["method", "SS", "N", *OUTCOMES], rows)


def _regime_block(
    title: str, data: CondData, methods: Sequence[str], pad: str, regime: str
) -> list[str]:
    sea_states = _ss(regime)
    lines = [f"### {title}: success per sea state", ""]
    rows = [
        [_label(m), *(data.success_cell(m, pad, regime, ss) for ss in sea_states)] for m in methods
    ]
    lines += _table(["method", *sea_states], rows)
    lines += [f"### {title}: p95 closing speed (m/s), crash %, timeout %", ""]
    rows = []
    for m in methods:
        cells = []
        for ss in sea_states:
            n, counts = data.outcome_counts(m, pad, regime, ss)
            if n == 0:
                cells.append("–")
                continue
            crash = counts[OUTCOMES.index("crash")] / n
            timeout = counts[OUTCOMES.index("timeout")] / n
            cells.append(f"{data.p95_cell(m, pad, regime, ss)} · {_pct(crash)} · {_pct(timeout)}")
        rows.append([_label(m), *cells])
    lines += _table(["method", *(f"{ss}: p95 · crash · timeout" for ss in sea_states)], rows)
    lines += [f"### {title}: outcome breakdown (counts, pooled over seeds; % of N)", ""]
    lines += _breakdown(data, methods, pad, regime, sea_states)
    return lines


def _contrasts(e07: Path) -> dict[str, dict[str, str]]:
    path = e07 / "contrasts.csv"
    return {r["contrast"]: r for r in read_rows(path)} if path.is_file() else {}


def _ci(rec: Mapping[str, str] | None) -> str:
    if rec is None:
        return "–"
    return (
        f"{_signed_pts(_f(rec, 'point'))} [{_signed_pts(_f(rec, 'ci_lo'))}, "
        f"{_signed_pts(_f(rec, 'ci_hi'))}]"
    )


def _not_flown(cond: Condition, e07: Path) -> str:
    return f"*Not flown yet: `{cond.out_dir(e07).relative_to(e07.parent)}` has no `summary.csv`.*"


def _matrix_section(e07: Path) -> list[str]:
    data = CondData(e07 / "matrix")
    lines = ["## 1. Main matrix: aft pad, JONSWAP, lambda = 1/25", ""]
    if not data.present:
        return [*lines, _not_flown(conditions("matrix")[0], e07), ""]
    lines += [
        "Learned rows: IQM over 5 training seeds [stratified-bootstrap 95 % CI]; per-seed "
        "range; N = seeds × episodes. Baselines: rate [Wilson 95 % CI] k/N, carried line for "
        "line from `results/e01` and `results/e01_lowvz_cut`. p95 for learned rows is the IQM "
        "of per-seed p95s (descriptive, not the relative-p95 statistic).",
        "",
    ]
    for reason in data.skip_reasons():
        lines += [f"- Cells not run (recorded, never dropped): {reason}", ""]
    methods = _methods([data], "aft")
    for regime, _ in REGIME_CELLS:
        lines += _regime_block(f"`{regime}`", data, methods, "aft", regime)
    return lines


def _cg_section(e07: Path) -> list[str]:
    aft = CondData(e07 / "matrix")
    cg = CondData(e07 / "cg")
    lines = ["## 2. Pad at CG (control arm for dmf's phase defect; P7-D1 §5; descriptive)", ""]
    if not cg.present:
        return [*lines, _not_flown(conditions("cg")[0], e07), ""]
    lines += [
        "Same listed episodes with the pad at the ship's CG. Learned methods and "
        "`pid_feedforward_lowvz_cut` flown here; the other baselines' CG rows carried from "
        "`results/e02`. aft − CG is the paired success contrast on identical episodes "
        "(`contrasts.csv`, shared episodes, 10 000 replicates), in points. No hypothesis is "
        "scored at CG.",
        "",
    ]
    con = _contrasts(e07)
    methods = _methods([cg, aft], "cg")
    for regime in conditions("cg")[0].lists:
        lines += _regime_block(f"`{regime}`, pad at CG", cg, methods, "cg", regime)
        rows = []
        for m in methods:
            for ss in _ss(regime):
                rows.append(
                    [
                        _label(m),
                        ss,
                        aft.success_cell(m, "aft", regime, ss),
                        cg.success_cell(m, "cg", regime, ss),
                        _ci(con.get(f"cg.{m}.{regime}.{ss}")),
                    ]
                )
        lines += [f"### `{regime}`: aft − CG", ""]
        lines += _table(["method", "SS", "aft", "CG", "aft − CG, points [95 % CI]"], rows)
    return lines


def _sinusoid_section(e07: Path) -> list[str]:
    jon = CondData(e07 / "matrix")
    sin = CondData(e07 / "sinusoid")
    lines = ["## 3. Sinusoid test motion (the H4 cross; P7-D1 §1), `id`, aft pad", ""]
    if not sin.present:
        return [*lines, _not_flown(conditions("sinusoid")[0], e07), ""]
    lines += [
        "Each listed episode flown on the matched sinusoid built by the training builder "
        "(`rld.rl.motion.sinusoid_motion`: amplitude √2 × committed aft RMS, peak encounter "
        "period, phases from the episode seed); same start offsets and initial states. H4 is "
        "read at `id` SS5 only (section 7); the other cells are descriptive. JONSWAP − "
        "sinusoid is the paired success contrast, in points.",
        "",
    ]
    methods = _methods([sin], "aft")
    lines += _regime_block("`id`, sinusoid motion", sin, methods, "aft", "id")
    con = _contrasts(e07)
    rows = [
        [
            _label(m),
            ss,
            jon.success_cell(m, "aft", "id", ss),
            sin.success_cell(m, "aft", "id", ss),
            _ci(con.get(f"sinusoid.{m}.id.{ss}")),
        ]
        for m in methods
        for ss in _ss("id")
    ]
    lines += ["### `id`: JONSWAP − sinusoid", ""]
    lines += _table(["method", "SS", "JONSWAP", "sinusoid", "JONSWAP − sinusoid [95 % CI]"], rows)
    return lines


def _noise_label(sigma_cm: int, steps: int) -> str:
    return f"σp {sigma_cm} cm / {steps} step ({round(1000 * steps / 30, 1)} ms)"


def _noise_section(e07: Path) -> list[str]:
    clean = CondData(e07 / "matrix")
    conds = {c.name: (c, CondData(c.out_dir(e07))) for c in conditions("noise")}
    lines = ["## 4. Perception stand-in (P7-D1 §4), `id`, aft pad", ""]
    flown = [d for _, d in conds.values() if d.present]
    lines += [
        "σp on relative position, σv = σp / 0.2 s on relative velocity (0 / 0.05 / 0.10 / "
        "0.20 m/s), latency 0 / 1 / 2 control steps (configured 0 / 33.4 / 66.7 ms; 1 step = "
        "33.3 ms model = 167 ms full scale), no hold (30 Hz). Applied to the six relative-pad "
        "entries only; the forecast methods' ship-motion feed stays ideal. The clean column is "
        "the main matrix's rows. Cells: success (IQM or rate) with the clean − noisy paired "
        "contrast in points [95 % CI] below it.",
        f"Conditions written: {len(flown)} of {len(conds)}.",
        "",
    ]
    if not flown:
        return [*lines, "*Not flown yet.*", ""]
    con = _contrasts(e07)
    methods = _methods([clean, *flown], "aft")
    header = ["method", _noise_label(0, 0)] + [
        _noise_label(round(c.noise.sigma_p_m * 100), c.noise.latency_steps)
        for c, _ in conds.values()
        if c.noise is not None
    ]
    for ss in _ss("id"):
        rows = []
        for m in methods:
            row = [_label(m), _pct(clean.success_value(m, "aft", "id", ss))]
            for name, (_, data) in conds.items():
                value = _pct(data.success_value(m, "aft", "id", ss)) if data.present else "–"
                row.append(f"{value}<br>{_ci(con.get(f'noise/{name}.{m}.id.{ss}'))}")
            rows.append(row)
        lines += [f"### `id` {ss}: success % by condition (clean − noisy, points [95 % CI])", ""]
        lines += _table(header, rows)
    return lines


def _lambda_section(e07: Path) -> list[str]:
    base = CondData(e07 / "matrix")
    conds = [(c, CondData(c.out_dir(e07))) for c in conditions("lambda")]
    lines = ["## 5. lambda sensitivity (P7-D1 §3), aft pad", ""]
    if not any(d.present for _, d in conds):
        return [*lines, "*Not flown yet.*", ""]
    lines += [
        "`ppo` (the best learned method at `id` SS6) and `pid_feedforward` (the best "
        "classical controller), plus the always-printed `pid_track_descend` and `oracle_gated`. "
        "The project lambda stays 1/25. At lambda ≠ 1/25 the realization, episode seed, "
        "initial position and pad are the listed ones, but **t0 is re-drawn** by the "
        "environment inside that lambda's window, so the episodes are not identical; the "
        "contrasts resample episodes independently per lambda (points [95 % CI]).",
        "",
    ]
    con = _contrasts(e07)
    first = conds[0][0]
    methods = [*first.learned, *(b for b in BASELINES if b in first.baselines)]
    for regime in first.lists:
        rows = []
        for m in methods:
            for ss in _ss(regime):
                row = [_label(m), ss]
                cells = {round(c.lam_inverse): d for c, d in conds}
                row.append(cells[15].success_cell(m, "aft", regime, ss) if 15 in cells else "–")
                row.append(base.success_cell(m, "aft", regime, ss))
                row.append(cells[40].success_cell(m, "aft", regime, ss) if 40 in cells else "–")
                row += [
                    _ci(con.get(f"lambda/lam{round(lam)}.{m}.{regime}.{ss}"))
                    for lam in LAMBDA_INVERSES
                ]
                rows.append(row)
        lines += [f"### `{regime}`: success at lambda 1/15, 1/25, 1/40", ""]
        lines += _table(
            ["method", "SS", "1/15", "1/25", "1/40", "1/25 − 1/15", "1/25 − 1/40"], rows
        )
        for c, d in conds:
            if d.present:
                lines += [
                    f"#### `{regime}`, lambda 1/{round(c.lam_inverse)}: outcome breakdown",
                    "",
                ]
                lines += _breakdown(d, methods, "aft", regime, _ss(regime))
    feas = e07 / "lambda" / "feasibility.csv"
    if feas.is_file():
        rows = [
            [
                f"1/{round(float(r['lam_inverse']))}",
                f"{r['vessel']} {r['ss']} {float(r['heading_deg']):.0f} deg "
                f"{float(r['speed_kn']):.0f} kn {r['pad']}",
                _num(_f(r, "vz_p99_model_m_s"), 4),
                _num(_f(r, "threshold_m_s"), 4),
                r["verdict"],
            ]
            for r in read_rows(feas)
        ]
        lines += [
            "### P1-D1 feasibility rule at each lambda (context only; P7-D1 §3)",
            "",
            "Deck-point v_z p99, model scale, worst head-seas SS6 frigate cell, aft pad, "
            "against 0.25 × the CF2X URDF speed. 1/15 is outside the declared ladder; it is "
            "reported, not adopted.",
            "",
        ]
        lines += _table(["lambda", "cell", "v_z p99 (m/s)", "threshold (m/s)", "verdict"], rows)
    return lines


def _mss_section(e07: Path, results: Path) -> list[str]:
    cond = conditions("mss")[0]
    data = CondData(cond.out_dir(e07))
    lines = ["## 6. MSS transfer (optional; P7-D1 §6; descriptive)", ""]
    if not data.present:
        return [*lines, "*Not flown: no `results/e07/mss/summary.csv`.*", ""]
    lines += [
        "Episodes on another simulator's strip-theory trajectories (MSS, ITTC S-175, SS5), "
        "**not measurements of a real ship**. `mss_transfer` uses MSS's spectrum and RAOs; "
        "`mss_transfer_corpus` dmf's own wave field through MSS's transfer function (the "
        "attribution control). Beside them, the matrix's `unseen_vessel` SS5 restricted to "
        "the 180° and 135° headings (different realizations; descriptive only).",
        "",
    ]
    regimes = list(dict.fromkeys(k[2] for k in (*data.learned, *data.base)))
    for pad in ("aft", "cg"):
        methods = _methods([data], pad)
        for regime in regimes:
            sea = sorted({k[3] for k in (*data.learned, *data.base) if k[2] == regime})
            rows = [
                [_label(m), *(data.success_cell(m, pad, regime, s) for s in sea)] for m in methods
            ]
            lines += [f"### `{regime}`, {pad} pad: success", ""]
            lines += _table(["method", *sea], rows)
            lines += _breakdown(data, methods, pad, regime, sea)
    lines += ["### `unseen_vessel` SS5, headings 180° and 135° only (matrix, aft)", ""]
    lines += _uv_headings(e07, results)
    return lines


def _uv_headings(e07: Path, results: Path) -> list[str]:
    """Success of every method on unseen_vessel SS5 at 180 / 135 deg, pooled over seeds."""
    sources = [e07 / "matrix" / "episodes.csv", results / "e01" / "episodes.csv"]
    sources.append(results / "e01_lowvz_cut" / "episodes.csv")
    counts: dict[str, list[int]] = {}
    for path in sources:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                if r["regime"] != "unseen_vessel" or r["ss"] != "SS5" or r["pad"] != "aft":
                    continue
                if float(r["heading_deg"]) not in (180.0, 135.0):
                    continue
                if (
                    path.parent.name == "e01_lowvz_cut"
                    and r["method"] != "pid_feedforward_lowvz_cut"
                ):
                    continue
                c = counts.setdefault(r["method"], [0, 0])
                c[0] += r["outcome"] == "success"
                c[1] += 1
    rows = [
        [_label(m), f"{_pct(k / n)} ({k}/{n} seed-episodes)"]
        for m in (*LEARNED_METHODS, *BASELINES)
        if m in counts
        for k, n in [counts[m]]
    ]
    return _table(["method", "success, pooled over seeds"], rows)


def _threshold(rec: Mapping[str, str]) -> str:
    """Return a hypothesis row's threshold as printed: points / %, or the half-rule ratio."""
    if not rec["threshold"]:
        return "–"
    if "half-rule" in rec["part"]:
        return f"r(unseen_vessel) ≤ {_f(rec, 'threshold'):g} × r(id)"
    return f"≥ {_signed_pts(_f(rec, 'threshold'))}"


def _hypotheses_section(e07: Path) -> list[str]:
    path = e07 / "hypotheses.csv"
    lines = ["## 7. Hypotheses (P3-D1 §8, P3-D4, P6-D6, P7-D1 §2)", ""]
    if not path.is_file():
        return [*lines, "*Not scored yet: no `results/e07/hypotheses.csv`.*", ""]
    lines += [
        "Every verdict is computed by `rld.eval.hypotheses` from the pre-registered rule "
        "named in its row. Rates and differences in points, r in %; 10 000 bootstrap "
        "replicates, seed 20260926, percentile 95 % CI. No multiplicity correction "
        "(P3-D4 #9). Rows not labelled `scored` are not scored.",
        "",
    ]
    rows = []
    used: list[str] = []
    notes: list[str] = []
    for r in read_rows(path):
        if r["note"]:
            notes.append(f"- {r['hypothesis']}, {r['part']}: {r['note']}")
        stat = r["point"] and (
            f"{_signed_pts(_f(r, 'point'))} [{_signed_pts(_f(r, 'ci_lo'))}, "
            f"{_signed_pts(_f(r, 'ci_hi'))}]"
        )
        ni = (
            r["non_inferiority_lo"] and f"NI lower bound {_signed_pts(_f(r, 'non_inferiority_lo'))}"
        )
        keys = [k for k in r["caveats"].split(";") if k]
        used += keys
        rows.append(
            [
                r["hypothesis"],
                r["part"],
                r["role"],
                r["cell"],
                " ".join(x for x in (stat or "–", ni or "") if x),
                _threshold(r),
                f"**{r['verdict']}**",
                r["rule_source"],
                ", ".join(keys) or "–",
            ]
        )
    lines += _table(
        ["H", "part", "role", "cell", "point [95 % CI]", "threshold", "verdict", "rule", "caveats"],
        rows,
    )
    lines += ["### Numbers behind each verdict (`hypotheses.csv` notes)", "", *notes, ""]
    lines += ["### Caveat keys", ""]
    lines += [f"- `{k}`: {CAVEATS[k]}" for k in dict.fromkeys(used) if k in CAVEATS]
    return [*lines, ""]


def _wilson(rec: Mapping[str, str]) -> str:
    """Return ``rate [Wilson lo, hi] k/N`` in percent for one summary record."""
    return (
        f"{_pct(_f(rec, 'success_rate'))} [{_pct(_f(rec, 'success_wilson_lo'))}, "
        f"{_pct(_f(rec, 'success_wilson_hi'))}] {rec['n_success']}/{rec['n_episodes']}"
    )


def _appendix_seeds(e07: Path) -> list[str]:
    data = CondData(e07 / "matrix")
    if not data.present:
        return []
    lines = ["## Appendix A. Per-seed success with Wilson 95 % CI (main matrix, aft)", ""]
    for regime, sea_states in REGIME_CELLS:
        rows = []
        for m in LEARNED_METHODS:
            seeds = sorted(
                {r["run_seed"] for ss in sea_states for r in data.records(m, "aft", regime, ss)},
                key=int,
            )
            for seed in seeds:
                row = [_label(m), seed]
                for ss in sea_states:
                    rec = next(
                        (r for r in data.records(m, "aft", regime, ss) if r["run_seed"] == seed),
                        None,
                    )
                    if rec is None:
                        row.append("missing")
                    elif int(rec["n_episodes"]) == 0:
                        row.append("not run")
                    else:
                        row.append(_wilson(rec))
                rows.append(row)
        lines += [f"### `{regime}`", ""]
        lines += _table(["method", "seed", *sea_states], rows)
    return lines


def _appendix_noise(e07: Path) -> list[str]:
    lines = ["## Appendix B. Perception stand-in: outcome breakdown per condition", ""]
    any_flown = False
    for cond in conditions("noise"):
        data = CondData(cond.out_dir(e07))
        if not data.present or cond.noise is None:
            continue
        any_flown = True
        label = _noise_label(round(cond.noise.sigma_p_m * 100), cond.noise.latency_steps)
        lines += [f"### `{cond.key}`: {label}, σv {cond.noise.sigma_v_m_s} m/s", ""]
        lines += _breakdown(data, _methods([data], "aft"), "aft", "id", _ss("id"))
    return lines if any_flown else []


def _header() -> list[str]:
    return [
        "# Results — Phase 7: evaluation under shift and ablations (simulation only)",
        "",
        "- **Simulation only** (PyBullet, gym-pybullet-drones Crazyflie 2.x). No sentence here "
        "describes real flight or real deck data.",
        "- Deck motion: dmf's 3-DOF (heave, roll, pitch) JONSWAP response, Froude-scaled to "
        "the drone at **lambda = 1/25** (1 s model = 5 s full scale), except where the lambda "
        "arm says otherwise. Surge, sway and yaw are absent.",
        "- State-based observations; the perception stand-in (noise, latency) is the only "
        "sensor model and is off except in section 4. Not vision.",
        "- dmf's roll/pitch–heave phase defect (~90°) is carried, not fixed; the aft pad is "
        "sensitive to it, and the pad-at-CG arm (section 2) is its control.",
        "- Success = all four frozen criteria (`configs/env/success.yaml`). Success is never "
        "pooled across sea states. Learned rows: IQM over 5 training seeds with the "
        "stratified-bootstrap 95 % CI (2 000 replicates, seed 20260926) and the per-seed "
        "range; baselines: rate with the Wilson 95 % CI and k/N. Every success table is "
        "followed by the six-class outcome breakdown.",
        f"- `oracle_gated` is a **{_ORACLE_SHORT}**: it reads the true future deck motion. "
        "It is never a deployable result.",
        "- `pid_track_descend`, `pid_feedforward` and `oracle_gated` are in every method table.",
        "- Carried into every table and verdict (P7-D1 §8):",
        *(
            f"  - {CAVEATS[k]}"
            for k in (
                "P6-D1-forecast",
                "P6-D5-residual-descent",
                "two-phase-descent",
                "bounce-grace-50ms",
                "closing-speed-7pct",
                "tunnelling-any-substep",
                "P6-D5-tunnelling-bound",
                "tilt-only-hard-landings",
                "sac-budget",
                "regimes-overlap",
            )
        ),
        "- Rendered from the CSVs under `results/e07/` by `rld.eval.report.render_results` "
        "(`scripts/report.py`); do not edit by hand. `scripts/report.py --check` re-renders "
        "and compares bytes.",
        "",
    ]


def render(results: Path = RESULTS_DIR, e07: Path = E07_DIR) -> str:
    """Return the text of ``results/results.md``, from the CSVs only.

    Args:
        results: ``results/``.
        e07: ``results/e07``.

    Returns:
        The markdown.

    Raises:
        ValueError: If the text would contain a phrase refused by P3-D4.
    """
    lines = _header()
    lines += _matrix_section(e07)
    lines += _cg_section(e07)
    lines += _sinusoid_section(e07)
    lines += _noise_section(e07)
    lines += _lambda_section(e07)
    lines += _mss_section(e07, results)
    lines += _hypotheses_section(e07)
    lines += _appendix_seeds(e07)
    lines += _appendix_noise(e07)
    text = "\n".join(lines).rstrip("\n") + "\n"
    lowered = text.lower()
    for phrase in FORBIDDEN_RENDERED_PHRASES:
        if phrase in lowered:
            raise ValueError(f"rendered output contains the forbidden phrase {phrase!r}")
    return text


def main(argv: Sequence[str] | None = None) -> int:
    """Write ``results/results.md``, or with ``--check`` re-render and compare bytes.

    Args:
        argv: Arguments (``None``: ``sys.argv``).

    Returns:
        0 on success; 1 if ``--check`` finds a difference or the file is missing.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="scripts/report.py", description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--e07-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="default: <results-dir>/results.md")
    parser.add_argument("--check", action="store_true", help="re-render and compare; write nothing")
    args = parser.parse_args(argv)
    e07 = args.e07_dir or args.results_dir / "e07"
    out: Path = args.out or args.results_dir / RESULTS_NAME
    text = render_results(args.results_dir, e07)
    if args.check:
        same = out.is_file() and out.read_text(encoding="utf-8") == text
        print(f"{out}: {'byte-identical' if same else 'DIFFERS or missing'}")
        return 0 if same else 1
    out.write_text(text, encoding="utf-8")
    print(f"{out}: {text.count(chr(10))} lines")
    return 0
