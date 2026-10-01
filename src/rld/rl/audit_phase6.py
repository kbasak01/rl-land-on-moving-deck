"""Phase 6 additions to the reward-hacking audit (``results/audit/e06/README.md``).

:mod:`rld.rl.audit` runs P5-D14's checks 1a-8 unchanged on the four Phase 6 methods
(``AuditSpec`` ``PHASE6``) and hands its re-flights here for the checks pre-stated for
Phase 6 only:

* **1b-fine** -- training timeouts at 1 % step bins, every run (threshold 0.50, bin 0
  reported apart from bins 1-99);
* **6 per seed** -- the post-contact down-force rule applied to each residual seed, beside
  the re-flown ``pid_feedforward_lowvz_cut`` and ``pid_feedforward``;
* **9** -- the residual's authority: ``|alpha * pi|``, ``pi`` saturation, the composed clip
  and the norm cap, and the executed setpoint's departure from the base's;
* **10** -- whether the forecast methods depend on their forecast block: open loop (the
  network re-evaluated with the block replaced along the factual trajectory) and closed loop
  (set C, re-flown with the block replaced);
* **11** -- what drives the tilt-violating hard landings (deck tilt vs drone tilt);
* **12** -- the tunnelling mechanism of every tunnelled episode (``tunnelling_episodes.csv``,
  written by :mod:`rld.rl.audit`), restated per method.

Like :mod:`rld.rl.audit`, this module never opens a frozen list: the listed episodes arrive
through :class:`rld.rl.audit.Inputs`.

Units: speeds metres per second model scale (setpoints world frame), angles degrees, angular
rates degrees per second model scale, times seconds model scale (lambda = 1/25); actions in
normalised units (x 1.5 m/s = velocity setpoint); alpha (0.3) in normalised units.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dmf.typedefs import FloatArray

from rld.rl.audit import (
    _K,
    _X,
    ABLATIONS,
    BASELINE_REFLY,
    BOOT_REPS,
    BOOT_SEED,
    DOWNFORCE_IDLE_FRAC,
    DOWNFORCE_MEDIAN_SP_M_S,
    LAST_WINDOW_S,
    LOWVZ_CUT,
    NORM_EPS,
    SAT_AXIS_LEVEL,
    SEA_STATES,
    TIMEOUT_BIN_MAX,
    AuditSpec,
    Inputs,
    _b,
    _columns,
    _f,
    _frac,
    _refly_row,
    read_rows,
    write_csv,
)

__all__ = [
    "RESIDUAL_METHODS",
    "Phase6Context",
    "classify_hard_landing",
    "dependence_bootstrap",
    "phase6_outputs",
    "summarise_extras",
]

#: The residual methods (check 6 per seed, check 9).
RESIDUAL_METHODS: tuple[str, ...] = ("residual_ppo", "residual_ppo_forecast")
#: The forecast methods (check 10).
FORECAST_METHODS: tuple[str, ...] = ("ppo_forecast", "residual_ppo_forecast")
#: Check 10a: pooled open-loop median |delta setpoint| at or above this "depends", m/s
#: (1 % of v_max = 1.5 m/s).
OPEN_LOOP_DEPENDS_M_S: float = 0.015
#: Check 9: the residual is authority-limited if |pi_z| >= 0.99 on at least this fraction
#: of pooled pre-contact steps.
AUTHORITY_LIMITED_FRAC: float = 0.50
#: Check 11: the frozen limits the hard-landing classes are read against (success.yaml).
TILT_LIMIT_DEG: float = 15.0
SPEED_LIMIT_M_S: float = 0.5
#: Check 10b: a time-to-touchdown change larger than one control step, seconds model.
ONE_STEP_S: float = 1.0 / 30.0


def _pct(values: Sequence[float] | FloatArray, q: float) -> float:
    """Linear percentile, NaN for an empty input."""
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def _med(values: Sequence[float] | FloatArray) -> float:
    """Median, NaN for an empty input."""
    return _pct(values, 50.0)


def _max(values: Sequence[float] | FloatArray) -> float:
    """Maximum, NaN for an empty input."""
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.max()) if arr.size else float("nan")


def _mean(values: Sequence[float] | FloatArray) -> float:
    """Mean, NaN for an empty input."""
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _first_contact_step(steps: FloatArray) -> int:
    """Index of the first control step whose observation has ``in_contact = 1``."""
    in_obs = steps[:, _K["obs_in_contact"]] > 0.5
    return int(np.argmax(in_obs)) if bool(in_obs.any()) else int(steps.shape[0])


# ============================================================================ worker side


def _residual_arrays(
    steps: FloatArray, xsteps: FloatArray, v_max_m_s: float, alpha: float
) -> dict[str, FloatArray]:
    """Per-step residual quantities of one flight (check 9).

    Args:
        steps: ``(k, len(STEP_COLS))`` control-step log.
        xsteps: ``(k, len(XSTEP_COLS))`` extras log.
        v_max_m_s: The speed cap, m/s.
        alpha: Residual scale, normalised units.

    Returns:
        ``auth`` (|alpha pi| m/s), ``auth_z`` (signed vertical part, m/s), ``pi_sat_any``,
        ``pi_sat_z``, ``clip_bind``, ``cap_bind`` (0/1), ``dep`` (|v_sp(exec) - v_sp(base)|,
        m/s) and ``dep_x``, ``dep_y``, ``dep_z`` (signed, m/s).
    """
    pi = xsteps[:, [_X["pi_x"], _X["pi_y"], _X["pi_z"]]]
    base = xsteps[:, [_X["base_ax"], _X["base_ay"], _X["base_az"]]]
    base_sp = xsteps[:, [_X["base_sp_x_m_s"], _X["base_sp_y_m_s"], _X["base_sp_z_m_s"]]]
    sp = steps[:, [_K["sp_x_m_s"], _K["sp_y_m_s"], _K["sp_z_m_s"]]]
    raw = base + alpha * pi
    clipped = np.clip(raw, -1.0, 1.0)
    dep = sp - base_sp
    return {
        "auth": v_max_m_s * alpha * np.linalg.norm(pi, axis=1),
        "auth_z": v_max_m_s * alpha * pi[:, 2],
        "pi_sat_any": np.any(np.abs(pi) >= SAT_AXIS_LEVEL, axis=1).astype(np.float64),
        "pi_sat_z": (np.abs(pi[:, 2]) >= SAT_AXIS_LEVEL).astype(np.float64),
        "clip_bind": np.any(np.abs(raw) > 1.0, axis=1).astype(np.float64),
        "cap_bind": (np.linalg.norm(clipped, axis=1) > 1.0 + NORM_EPS).astype(np.float64),
        "dep": np.linalg.norm(dep, axis=1),
        "dep_x": dep[:, 0],
        "dep_y": dep[:, 1],
        "dep_z": dep[:, 2],
    }


def summarise_extras(
    steps: FloatArray,
    xsteps: FloatArray,
    record: Mapping[str, Any],
    *,
    v_max_m_s: float,
    alpha: float,
) -> dict[str, Any]:
    """Per-episode Phase 6 quantities of one re-flight (``x_`` columns).

    Args:
        steps: ``(k, len(STEP_COLS))`` control-step log.
        xsteps: ``(k, len(XSTEP_COLS))`` extras log (same steps).
        record: ``EpisodeRecord.as_row()``.
        v_max_m_s: The speed cap, m/s.
        alpha: Residual scale, normalised units (NaN for a non-residual policy).

    Returns:
        ``x_*`` scalars: residual authority (pre-/post-contact medians and maxima), open-loop
        forecast sensitivity medians, and the attitudes in the last 0.5 s before touchdown.
    """
    nan = float("nan")
    out: dict[str, Any] = {"x_alpha": alpha}
    k1 = _first_contact_step(steps)
    residual = bool(np.all(np.isfinite(xsteps[:, _X["base_ax"]]))) and xsteps.shape[0] > 0
    keys = ("auth", "pi_sat_any", "pi_sat_z", "clip_bind", "cap_bind", "dep")
    if residual:
        arr = _residual_arrays(steps, xsteps, v_max_m_s, alpha)
        for tag, sl in (("pre", slice(0, k1)), ("post", slice(k1, None))):
            out[f"x_auth_{tag}_p50_m_s"] = _med(arr["auth"][sl])
            out[f"x_auth_{tag}_max_m_s"] = _max(arr["auth"][sl])
            out[f"x_auth_z_{tag}_mean_m_s"] = _mean(arr["auth_z"][sl])
            for key in keys[1:5]:
                out[f"x_{key}_{tag}_frac"] = _mean(arr[key][sl])
            out[f"x_dep_{tag}_p50_m_s"] = _med(arr["dep"][sl])
    else:
        for tag in ("pre", "post"):
            out[f"x_auth_{tag}_p50_m_s"] = nan
            out[f"x_auth_{tag}_max_m_s"] = nan
            out[f"x_auth_z_{tag}_mean_m_s"] = nan
            for key in keys[1:5]:
                out[f"x_{key}_{tag}_frac"] = nan
            out[f"x_dep_{tag}_p50_m_s"] = nan
    for ab in ABLATIONS:
        col = xsteps[:, _X[f"d_open_{'mean' if ab == 'mean' else 'zero'}_m_s"]]
        out[f"x_d_open_{ab}_median_m_s"] = (
            _med(col) if xsteps.shape[0] and bool(np.all(np.isfinite(col))) else nan
        )
    td = record["td_t_episode_s"]
    names = (
        "x_win_abs_tilt_max_deg",
        "x_last_abs_tilt_deg",
        "x_last_deck_tilt_deg",
        "x_last_rel_tilt_deg",
        "x_last_deck_normal_rate_deg_s",
        "x_win_sp_h_mean_m_s",
        "x_win_auth_mean_m_s",
        "x_win_auth_z_mean_m_s",
    )
    if td is None:
        out.update(dict.fromkeys(names, nan))
        return out
    t_start = steps[:, _K["t_start_s"]]
    win = (t_start >= float(td) - LAST_WINDOW_S) & (t_start < float(td))
    idx = np.flatnonzero(win)
    if idx.size == 0:
        out.update(dict.fromkeys(names, nan))
        return out
    last = int(idx[-1])
    sp_h = np.hypot(steps[idx, _K["sp_x_m_s"]], steps[idx, _K["sp_y_m_s"]])
    out["x_win_abs_tilt_max_deg"] = _max(xsteps[idx, _X["abs_tilt_deg"]])
    out["x_last_abs_tilt_deg"] = float(xsteps[last, _X["abs_tilt_deg"]])
    out["x_last_deck_tilt_deg"] = float(xsteps[last, _X["deck_tilt_deg"]])
    out["x_last_rel_tilt_deg"] = float(xsteps[last, _X["rel_tilt_deg"]])
    out["x_last_deck_normal_rate_deg_s"] = float(xsteps[last, _X["deck_normal_rate_deg_s"]])
    out["x_win_sp_h_mean_m_s"] = _mean(sp_h)
    if residual:
        arr = _residual_arrays(steps[idx], xsteps[idx], v_max_m_s, alpha)
        out["x_win_auth_mean_m_s"] = _mean(arr["auth"])
        out["x_win_auth_z_mean_m_s"] = _mean(arr["auth_z"])
    else:
        out["x_win_auth_mean_m_s"] = nan
        out["x_win_auth_z_mean_m_s"] = nan
    return out


# ============================================================================ reductions


def classify_hard_landing(
    closing_m_s: float, rel_tilt_deg: float, abs_tilt_deg: float, deck_tilt_deg: float
) -> tuple[str, str]:
    """Check 11's pre-stated classes for one hard landing.

    Args:
        closing_m_s: Committed closing speed along the deck normal, m/s.
        rel_tilt_deg: Drone body z vs deck normal at first contact, degrees.
        abs_tilt_deg: Drone body z vs world up, degrees.
        deck_tilt_deg: Deck normal vs world up, degrees.

    Returns:
        ``(cause, tilt_class)``: cause ``speed_only`` / ``tilt_only`` / ``both`` / ``neither``;
        tilt class (tilt violations only, else ``""``) ``deck_alone_over_limit`` (deck tilt
        > 15 deg), ``deck_dominated`` (deck tilt >= drone absolute tilt) or
        ``drone_dominated``.
    """
    speed = closing_m_s > SPEED_LIMIT_M_S
    tilt = rel_tilt_deg > TILT_LIMIT_DEG
    cause = {
        (True, False): "speed_only",
        (False, True): "tilt_only",
        (True, True): "both",
        (False, False): "neither",
    }[(speed, tilt)]
    if not tilt:
        return cause, ""
    if deck_tilt_deg > TILT_LIMIT_DEG:
        return cause, "deck_alone_over_limit"
    if deck_tilt_deg >= abs_tilt_deg:
        return cause, "deck_dominated"
    return cause, "drone_dominated"


def dependence_bootstrap(
    diffs: Mapping[str, Sequence[float]], *, reps: int, seed_words: Sequence[int]
) -> tuple[float, float, float]:
    """Paired success difference with a bootstrap CI stratified by sea state (check 10b).

    Args:
        diffs: SS to the per-listed-episode differences (counterfactual minus factual
            success, mean over the method's seeds).
        reps: Replicates.
        seed_words: ``numpy.random.SeedSequence`` words.

    Returns:
        ``(mean difference, 2.5 % bound, 97.5 % bound)`` over all episodes (each SS
        resampled within itself, then pooled with its own episode count).
    """
    arrays = [np.asarray(diffs[ss], dtype=np.float64) for ss in sorted(diffs)]
    total = sum(a.size for a in arrays)
    point = float(sum(a.sum() for a in arrays) / total)
    rng = np.random.default_rng(np.random.SeedSequence([int(w) for w in seed_words]))
    boots = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        acc = 0.0
        for a in arrays:
            acc += float(rng.choice(a, size=a.size, replace=True).sum())
        boots[r] = acc / total
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


@dataclass
class Phase6Context:
    """What :func:`phase6_outputs` reads from the shared audit run.

    Attributes:
        repo: Repository root.
        out_dir: ``results/audit/e06``.
        spec: The Phase 6 :class:`~rld.rl.audit.AuditSpec`.
        inp: The audit's inputs (e06, e05 ``ppo``, e01, e01_lowvz_cut rows).
        thr: Tunnelling threshold, metres.
        per_run: Method to its per-run rows plus the pooled row (already written).
        fine_rows: The 1 %-bin training rows of every run.
        flights: Every reproduction-checked re-flight (sets S, T, H).
        counterfactual: The set C flights (``ablate`` set).
        written: File name to text; this function adds its files.
    """

    repo: Path
    out_dir: Path
    spec: AuditSpec
    inp: Inputs
    thr: float
    per_run: Mapping[str, Sequence[Mapping[str, Any]]]
    fine_rows: Sequence[Mapping[str, Any]]
    flights: Sequence[Mapping[str, Any]]
    counterfactual: Sequence[Mapping[str, Any]]
    written: dict[str, str]


def _verdict(
    check: str, method: str, stat: str, value: Any, threshold: str, verdict: str
) -> dict[str, Any]:
    """One ``verdicts.csv`` row."""
    return {
        "check": check,
        "method": method,
        "statistic": stat,
        "value": value,
        "threshold": threshold,
        "verdict": verdict,
    }


def _fl(
    flights: Sequence[Mapping[str, Any]], method: str, seed: int | None, s: str
) -> list[Mapping[str, Any]]:
    """Flights of a method (and seed) in set ``s``."""
    return [
        f
        for f in flights
        if f["method"] == method and (seed is None or f["run_seed"] == seed) and s in f["sets"]
    ]


def _write(ctx: Phase6Context, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one CSV into the context's output directory and record its text."""
    ctx.written[name] = write_csv(ctx.out_dir / name, rows, _columns(rows))


def _check_1b_fine(ctx: Phase6Context) -> list[dict[str, Any]]:
    """Check 1b-fine: 1 %-bin training timeouts (pre-stated threshold 0.50)."""
    verdicts: list[dict[str, Any]] = []
    hover: list[dict[str, Any]] = []
    per_run_rows: list[dict[str, Any]] = []
    for method in ctx.spec.methods:
        worst_all: list[tuple[float, int, int]] = []
        worst_late: list[tuple[float, int, int]] = []
        for seed in range(5):
            bins = [r for r in ctx.fine_rows if r["method"] == method and r["run_seed"] == seed]
            fr = [float(b["timeout_frac"]) for b in bins]
            late = fr[1:]
            i_all = int(np.argmax(fr))
            i_late = 1 + int(np.argmax(late))
            worst_all.append((fr[i_all], seed, i_all))
            worst_late.append((fr[i_late], seed, i_late))
            per_run_rows.append(
                {
                    "method": method,
                    "run_seed": seed,
                    "bin0_timeout_frac": fr[0],
                    "bin0_success_frac": float(bins[0]["success_frac"]),
                    "max_timeout_frac": fr[i_all],
                    "max_bin": i_all,
                    "late_max_timeout_frac": fr[i_late],
                    "late_max_bin": i_late,
                    "bins_ge_threshold": sum(v >= TIMEOUT_BIN_MAX for v in fr),
                    "late_bins_ge_threshold": sum(v >= TIMEOUT_BIN_MAX for v in late),
                    "first_bin_timeout_below_0p10": next(
                        (i for i, v in enumerate(fr) if v < 0.10), -1
                    ),
                }
            )
            for b in bins:
                if float(b["timeout_frac"]) >= TIMEOUT_BIN_MAX:
                    hover.append(dict(b))
        v_all, s_all, b_all = max(worst_all)
        v_late, s_late, b_late = max(worst_late)
        verdicts.append(
            _verdict(
                "1b-fine",
                method,
                "max over runs and 1 % bins, training timeout frac (seed, bin)",
                f"{v_all!r} (seed {s_all}, bin {b_all})",
                f">= {TIMEOUT_BIN_MAX}",
                "finding" if v_all >= TIMEOUT_BIN_MAX else "clean",
            )
        )
        verdicts.append(
            _verdict(
                "1b-fine-late",
                method,
                "max over runs and 1 % bins 1-99, training timeout frac (seed, bin)",
                f"{v_late!r} (seed {s_late}, bin {b_late})",
                f">= {TIMEOUT_BIN_MAX}",
                "finding" if v_late >= TIMEOUT_BIN_MAX else "clean",
            )
        )
    _write(ctx, "training_fine_per_run.csv", per_run_rows)
    hover_cols = list(ctx.fine_rows[0].keys())
    ctx.written["training_hover_bins.csv"] = write_csv(
        ctx.out_dir / "training_hover_bins.csv", hover, hover_cols
    )
    return verdicts


def _bounces(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    """Bounce counts per SS."""
    return {
        f"bounce_n_{ss}": sum(r["ss"] == ss and r["outcome"] == "bounce" for r in rows)
        for ss in SEA_STATES
    }


def _check_6_per_seed(ctx: Phase6Context) -> list[dict[str, Any]]:
    """Check 6 per seed (pre-stated for the residual seeds), beside lowvz_cut and the base."""
    verdicts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    keep = (
        "s_flights",
        "s_post_steps",
        "s_post_idle_frac",
        "s_post_sp_z_median",
        "s_post_full_descent_frac",
        "s_touchdowns",
    )
    for method in ctx.spec.methods:
        for seed in (*range(5), None):
            fl = _fl(ctx.flights, method, seed, "S")
            r = _refly_row(fl)
            idle = float(r["s_post_idle_frac"])
            med = float(r["s_post_sp_z_median"])
            present = idle >= DOWNFORCE_IDLE_FRAC or med <= DOWNFORCE_MEDIAN_SP_M_S
            e = [
                x
                for x in ctx.inp.e05
                if x["method"] == method and (seed is None or int(x["run_seed"]) == seed)
            ]
            rows.append(
                {
                    "method": method,
                    "run_seed": "pooled" if seed is None else seed,
                    "source": "re-flight S (e06)",
                    **{k: r[k] for k in keep},
                    "confound_present": present,
                    **_bounces(e),
                    "episodes": len(e),
                }
            )
            if seed is not None and method in RESIDUAL_METHODS:
                verdicts.append(
                    _verdict(
                        "6-seed",
                        f"{method} seed {seed}",
                        "post-contact idle frac; median post-contact setpoint z (sample S)",
                        f"{idle!r}; {med!r}",
                        f">= {DOWNFORCE_IDLE_FRAC} or <= {DOWNFORCE_MEDIAN_SP_M_S}",
                        "finding" if present else "clean",
                    )
                )
    for method, src in (
        (LOWVZ_CUT, "re-flight S (e01_lowvz_cut)"),
        (BASELINE_REFLY, "re-flight S (e01)"),
    ):
        r = _refly_row(_fl(ctx.flights, method, 0, "S"))
        idle = float(r["s_post_idle_frac"])
        med = float(r["s_post_sp_z_median"])
        base_rows = (
            ctx.inp.cut
            if method == LOWVZ_CUT
            else [x for x in ctx.inp.e01 if x["method"] == method]
        )
        rows.append(
            {
                "method": method,
                "run_seed": "-",
                "source": src,
                **{k: r[k] for k in keep},
                "confound_present": idle >= DOWNFORCE_IDLE_FRAC or med <= DOWNFORCE_MEDIAN_SP_M_S,
                **_bounces(base_rows),
                "episodes": len(base_rows),
            }
        )
    lowvz = [x for x in ctx.inp.e01 if x["method"] == "pid_feedforward_lowvz"]
    rows.append(
        {
            "method": "pid_feedforward_lowvz",
            "run_seed": "-",
            "source": "e01 (not re-flown)",
            **_bounces(lowvz),
            "episodes": len(lowvz),
        }
    )
    p5 = read_rows(ctx.repo / "results" / "audit" / "ppo.csv")
    ppo_rows = ctx.inp.ppo_e05
    for r5 in p5:
        seed_s = r5["run_seed"]
        e = [x for x in ppo_rows if seed_s == "pooled" or x["run_seed"] == seed_s]
        rows.append(
            {
                "method": "ppo",
                "run_seed": seed_s,
                "source": "carried: results/audit/ppo.csv (P5-D14); bounces from e05",
                **{k: r5[k] for k in keep},
                "confound_present": _f(r5["s_post_idle_frac"]) >= DOWNFORCE_IDLE_FRAC
                or _f(r5["s_post_sp_z_median"]) <= DOWNFORCE_MEDIAN_SP_M_S,
                **_bounces(e),
                "episodes": len(e),
            }
        )
    _write(ctx, "downforce_per_seed.csv", rows)
    return verdicts


def _check_9(ctx: Phase6Context) -> list[dict[str, Any]]:
    """Check 9: residual authority (descriptive; one pre-stated reading)."""
    verdicts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    prof: list[dict[str, Any]] = []
    v_max = float(ctx.inp.cfgs.landing.v_max_m_s)
    ctrl_hz = float(ctx.inp.cfgs.landing.ctrl_freq_hz)
    for method in RESIDUAL_METHODS:
        alphas = {float(f["x_alpha"]) for f in _fl(ctx.flights, method, None, "S")}
        if len(alphas) != 1:
            raise ValueError(f"{method}: residual alpha not unique across runs: {alphas}")
        (alpha,) = alphas
        for seed in (*range(5), None):
            fl = _fl(ctx.flights, method, seed, "S")
            acc: dict[str, dict[str, list[float]]] = {"pre": {}, "post": {}}
            for f in fl:
                steps, xsteps = f["_steps"], f["_xsteps"]
                k1 = _first_contact_step(steps)
                arr = _residual_arrays(steps, xsteps, v_max, alpha)
                for tag, sl in (("pre", slice(0, k1)), ("post", slice(k1, None))):
                    for key, val in arr.items():
                        acc[tag].setdefault(key, []).extend(float(v) for v in val[sl])
            row: dict[str, Any] = {
                "method": method,
                "run_seed": "pooled" if seed is None else seed,
                "flights": len(fl),
                "alpha": alpha,
                "authority_per_axis_m_s": v_max * alpha,
            }
            for tag in ("pre", "post"):
                a = acc[tag]
                row[f"{tag}_steps"] = len(a.get("auth", []))
                row[f"{tag}_auth_p50_m_s"] = _med(a.get("auth", []))
                row[f"{tag}_auth_p90_m_s"] = _pct(a.get("auth", []), 90.0)
                row[f"{tag}_auth_p99_m_s"] = _pct(a.get("auth", []), 99.0)
                row[f"{tag}_auth_max_m_s"] = _max(a.get("auth", []))
                row[f"{tag}_auth_z_mean_m_s"] = _mean(a.get("auth_z", []))
                row[f"{tag}_auth_z_p50_m_s"] = _med(a.get("auth_z", []))
                for key in ("pi_sat_any", "pi_sat_z", "clip_bind", "cap_bind"):
                    row[f"{tag}_{key}_frac"] = _mean(a.get(key, []))
                row[f"{tag}_dep_p50_m_s"] = _med(a.get("dep", []))
                row[f"{tag}_dep_p90_m_s"] = _pct(a.get("dep", []), 90.0)
                row[f"{tag}_dep_max_m_s"] = _max(a.get("dep", []))
                for ax in ("x", "y", "z"):
                    row[f"{tag}_dep_{ax}_mean_m_s"] = _mean(a.get(f"dep_{ax}", []))
                    row[f"{tag}_dep_{ax}_absmean_m_s"] = _mean(np.abs(a.get(f"dep_{ax}", [])))
            limited = float(row["pre_pi_sat_z_frac"]) >= AUTHORITY_LIMITED_FRAC
            row["authority_limited"] = limited
            td_times = [
                float(f["td_t_episode_s"]) for f in fl if not math.isnan(float(f["td_t_episode_s"]))
            ]
            row["td_time_median_s"] = _med(td_times)
            rows.append(row)
            if seed is None:
                verdicts.append(
                    _verdict(
                        "9",
                        method,
                        "pooled pre-contact |pi_z|>=0.99 frac; |alpha pi| p50/p99 m/s; "
                        "composed clip / norm cap frac; departure p50 m/s (pre-contact)",
                        f"{row['pre_pi_sat_z_frac']!r}; {row['pre_auth_p50_m_s']!r}/"
                        f"{row['pre_auth_p99_m_s']!r}; {row['pre_clip_bind_frac']!r}/"
                        f"{row['pre_cap_bind_frac']!r}; {row['pre_dep_p50_m_s']!r}",
                        f"authority-limited if |pi_z|>=0.99 on >= {AUTHORITY_LIMITED_FRAC}",
                        "authority-limited" if limited else "not authority-limited",
                    )
                )
        # Descent profile: executed vs base setpoint z by control steps before touchdown.
        pts: dict[int, list[tuple[float, float, float]]] = {}
        for f in _fl(ctx.flights, method, None, "S"):
            td = float(f["td_t_episode_s"])
            if math.isnan(td):
                continue
            steps, xsteps = f["_steps"], f["_xsteps"]
            for k in range(steps.shape[0]):
                t = float(steps[k, _K["t_start_s"]])
                if t >= td:
                    break
                kb = int(math.floor((td - t) * ctrl_hz + 1e-9))
                pts.setdefault(kb, []).append(
                    (
                        float(steps[k, _K["sp_z_m_s"]]),
                        float(xsteps[k, _X["base_sp_z_m_s"]]),
                        v_max * alpha * float(xsteps[k, _X["pi_z"]]),
                    )
                )
        for kb in sorted(pts):
            if kb > 150:
                break
            p = pts[kb]
            prof.append(
                {
                    "method": method,
                    "steps_before_td": kb,
                    "t_before_td_s_lo": kb / ctrl_hz,
                    "n": len(p),
                    "exec_sp_z_median_m_s": _med([q[0] for q in p]),
                    "base_sp_z_median_m_s": _med([q[1] for q in p]),
                    "auth_z_median_m_s": _med([q[2] for q in p]),
                }
            )
    # pid_feedforward's own flights on the same episodes, for the same profile.
    pts_pid: dict[int, list[float]] = {}
    for f in _fl(ctx.flights, BASELINE_REFLY, 0, "S"):
        td = float(f["td_t_episode_s"])
        if math.isnan(td):
            continue
        steps = f["_steps"]
        for k in range(steps.shape[0]):
            t = float(steps[k, _K["t_start_s"]])
            if t >= td:
                break
            pts_pid.setdefault(int(math.floor((td - t) * ctrl_hz + 1e-9)), []).append(
                float(steps[k, _K["sp_z_m_s"]])
            )
    for kb in sorted(pts_pid):
        if kb > 150:
            break
        prof.append(
            {
                "method": BASELINE_REFLY,
                "steps_before_td": kb,
                "t_before_td_s_lo": kb / ctrl_hz,
                "n": len(pts_pid[kb]),
                "exec_sp_z_median_m_s": _med(pts_pid[kb]),
                "base_sp_z_median_m_s": float("nan"),
                "auth_z_median_m_s": float("nan"),
            }
        )
    pid_td = [
        float(f["td_t_episode_s"])
        for f in _fl(ctx.flights, BASELINE_REFLY, 0, "S")
        if not math.isnan(float(f["td_t_episode_s"]))
    ]
    rows.append(
        {
            "method": BASELINE_REFLY,
            "run_seed": "-",
            "flights": len(pid_td),
            "td_time_median_s": _med(pid_td),
        }
    )
    _write(ctx, "residual_authority.csv", rows)
    _write(ctx, "residual_descent_profile.csv", prof)
    return verdicts


def _check_10(ctx: Phase6Context) -> list[dict[str, Any]]:
    """Check 10: forecast dependence, open loop and closed loop (set C)."""
    verdicts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    eps: list[dict[str, Any]] = []
    fact = {
        (f["method"], int(f["run_seed"]), f["ss"], int(f["index"])): f
        for f in ctx.flights
        if "S" in f["sets"] and f["method"] in FORECAST_METHODS
    }
    for f in ctx.counterfactual:
        g = fact[(f["method"], int(f["run_seed"]), f["ss"], int(f["index"]))]
        ttd_f, ttd_c = float(g["td_t_episode_s"]), float(f["td_t_episode_s"])
        eps.append(
            {
                "method": f["method"],
                "run_seed": f["run_seed"],
                "ablate": f["ablate"],
                "ss": f["ss"],
                "index": f["index"],
                "factual_outcome": g["outcome"],
                "counterfactual_outcome": f["outcome"],
                "identical_row": g["_text"] == f["_text"],
                "outcome_changed": g["outcome"] != f["outcome"],
                "factual_td_s": ttd_f,
                "counterfactual_td_s": ttd_c,
                "delta_td_s": ttd_c - ttd_f,
                "factual_closing_m_s": float(g["closing_speed_normal_m_s"]),
                "counterfactual_closing_m_s": float(f["closing_speed_normal_m_s"]),
                "delta_closing_m_s": float(f["closing_speed_normal_m_s"])
                - float(g["closing_speed_normal_m_s"]),
                "factual_rel_tilt_deg": float(g.get("x_last_rel_tilt_deg", float("nan"))),
            }
        )
    _write(ctx, "forecast_counterfactual_episodes.csv", eps)
    for m_i, method in enumerate(ctx.spec.methods):
        if method not in FORECAST_METHODS:
            continue
        pooled_open: dict[str, float] = {}
        pooled_ci: dict[str, tuple[float, float, float]] = {}
        for v_i, ab in enumerate(ABLATIONS):
            col = _X[f"d_open_{'mean' if ab == 'mean' else 'zero'}_m_s"]
            for seed in (*range(5), None):
                fl = _fl(ctx.flights, method, seed, "S")
                pre: list[float] = []
                post: list[float] = []
                for f in fl:
                    k1 = _first_contact_step(f["_steps"])
                    d = f["_xsteps"][:, col]
                    pre += [float(v) for v in d[:k1]]
                    post += [float(v) for v in d[k1:]]
                allv = pre + post
                ce = [
                    e
                    for e in eps
                    if e["method"] == method
                    and e["ablate"] == ab
                    and (seed is None or e["run_seed"] == seed)
                ]
                f_succ = sum(e["factual_outcome"] == "success" for e in ce)
                c_succ = sum(e["counterfactual_outcome"] == "success" for e in ce)
                dtd = [abs(e["delta_td_s"]) for e in ce if not math.isnan(e["delta_td_s"])]
                row: dict[str, Any] = {
                    "method": method,
                    "run_seed": "pooled" if seed is None else seed,
                    "ablate": ab,
                    "open_steps": len(allv),
                    "open_median_m_s": _med(allv),
                    "open_p90_m_s": _pct(allv, 90.0),
                    "open_max_m_s": _max(allv),
                    "open_pre_median_m_s": _med(pre),
                    "open_post_median_m_s": _med(post),
                    "cf_flights": len(ce),
                    "cf_identical_n": sum(bool(e["identical_row"]) for e in ce),
                    "cf_outcome_changed_n": sum(bool(e["outcome_changed"]) for e in ce),
                    "factual_success_n": f_succ,
                    "cf_success_n": c_succ,
                    "cf_success_diff": _frac(c_succ - f_succ, len(ce)),
                    "cf_abs_delta_td_median_s": _med(dtd),
                    "cf_abs_delta_td_gt_one_step_frac": _frac(
                        sum(v > ONE_STEP_S + 1e-12 for v in dtd), len(dtd)
                    ),
                    "cf_delta_td_median_s": _med(
                        [e["delta_td_s"] for e in ce if not math.isnan(e["delta_td_s"])]
                    ),
                    "cf_delta_closing_median_m_s": _med(
                        [
                            e["delta_closing_m_s"]
                            for e in ce
                            if not math.isnan(e["delta_closing_m_s"])
                        ]
                    ),
                }
                for ss in SEA_STATES:
                    cs = [e for e in ce if e["ss"] == ss]
                    row[f"cf_success_diff_n_{ss}"] = sum(
                        (e["counterfactual_outcome"] == "success")
                        - (e["factual_outcome"] == "success")
                        for e in cs
                    )
                if seed is None:
                    diffs: dict[str, list[float]] = {}
                    for ss in SEA_STATES:
                        idxs = sorted({int(e["index"]) for e in ce if e["ss"] == ss})
                        for i in idxs:
                            per = [
                                float(e["counterfactual_outcome"] == "success")
                                - float(e["factual_outcome"] == "success")
                                for e in ce
                                if e["ss"] == ss and int(e["index"]) == i
                            ]
                            diffs.setdefault(ss, []).append(float(np.mean(per)))
                    point, lo, hi = dependence_bootstrap(
                        diffs, reps=BOOT_REPS, seed_words=(BOOT_SEED, 10, m_i, v_i)
                    )
                    row["cf_success_diff_boot_point"] = point
                    row["cf_success_diff_ci95_lo"] = lo
                    row["cf_success_diff_ci95_hi"] = hi
                    pooled_open[ab] = float(row["open_median_m_s"])
                    pooled_ci[ab] = (point, lo, hi)
                rows.append(row)
        depends = any(v >= OPEN_LOOP_DEPENDS_M_S for v in pooled_open.values())
        verdicts.append(
            _verdict(
                "10a",
                method,
                "pooled open-loop median |delta v_sp| m/s (mean; zeros)",
                "; ".join(f"{pooled_open[a]!r}" for a in ABLATIONS),
                f">= {OPEN_LOOP_DEPENDS_M_S} under either: depends",
                "depends on the block" if depends else "does not depend",
            )
        )
        excl = any(lo > 0.0 or hi < 0.0 for _, lo, hi in pooled_ci.values())
        verdicts.append(
            _verdict(
                "10b",
                method,
                "paired success diff (counterfactual - factual) [95 % CI] (mean; zeros)",
                "; ".join(
                    f"{pooled_ci[a][0]!r} [{pooled_ci[a][1]!r}, {pooled_ci[a][2]!r}]"
                    for a in ABLATIONS
                ),
                "CI excludes 0 under either: outcome depends",
                "outcome depends on the block" if excl else "outcome dependence not shown",
            )
        )
    _write(ctx, "forecast_dependence.csv", rows)
    return verdicts


def _check_11(ctx: Phase6Context) -> list[dict[str, Any]]:
    """Check 11: tilt-driven hard landings, committed columns plus the H re-flights."""
    verdicts: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, str]]]] = [
        (m, ctx.spec.tag, [r for r in ctx.inp.e05 if r["method"] == m]) for m in ctx.spec.methods
    ]
    groups.append(("ppo", "e05", list(ctx.inp.ppo_e05)))
    groups.append(
        (BASELINE_REFLY, "e01", [r for r in ctx.inp.e01 if r["method"] == BASELINE_REFLY])
    )
    hflights = {
        (f["method"], int(f["run_seed"]), f["ss"], int(f["index"])): f
        for f in ctx.flights
        if "H" in f["sets"]
    }
    xcols = (
        "x_win_abs_tilt_max_deg",
        "x_last_abs_tilt_deg",
        "x_last_deck_tilt_deg",
        "x_last_rel_tilt_deg",
        "x_last_deck_normal_rate_deg_s",
        "x_win_sp_h_mean_m_s",
        "x_win_auth_mean_m_s",
        "x_win_auth_z_mean_m_s",
    )
    per_ep: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for method, source, rows in groups:
        hard = [r for r in rows if r["outcome"] == "hard_landing"]
        classes: list[tuple[str, str]] = []
        for r in hard:
            c = classify_hard_landing(
                _f(r["closing_speed_normal_m_s"]),
                _f(r["rel_tilt_deg"]),
                _f(r["abs_tilt_deg"]),
                _f(r["deck_tilt_deg"]),
            )
            classes.append(c)
            seed = int(r["run_seed"])
            key = (method, seed, r["ss"], int(r["index"]))
            h = hflights.get(key)
            per_ep.append(
                {
                    "method": method,
                    "source": source,
                    "run_seed": seed,
                    "ss": r["ss"],
                    "index": int(r["index"]),
                    "closing_speed_normal_m_s": _f(r["closing_speed_normal_m_s"]),
                    "rel_tilt_deg": _f(r["rel_tilt_deg"]),
                    "abs_tilt_deg": _f(r["abs_tilt_deg"]),
                    "deck_tilt_deg": _f(r["deck_tilt_deg"]),
                    "lateral_offset_m": _f(r["lateral_offset_m"]),
                    "time_to_touchdown_s": _f(r["time_to_touchdown_s"]),
                    "cause": c[0],
                    "tilt_class": c[1],
                    "reflown": h is not None,
                    **{k: (float("nan") if h is None else h[k]) for k in xcols},
                }
            )
        for ss in (*SEA_STATES, "all"):
            cell = [r for r in rows if ss == "all" or r["ss"] == ss]
            td = [r for r in cell if _b(r["touchdown_contact"])]
            hc = [c for r, c in zip(hard, classes, strict=True) if ss == "all" or r["ss"] == ss]
            hr = [r for r in hard if ss == "all" or r["ss"] == ss]
            rel = [_f(r["rel_tilt_deg"]) for r in td]
            ab = [_f(r["abs_tilt_deg"]) for r in td]
            dk = [_f(r["deck_tilt_deg"]) for r in td]
            summary.append(
                {
                    "method": method,
                    "source": source,
                    "ss": ss,
                    "episodes": len(cell),
                    "touchdowns": len(td),
                    "hard_landing_n": len(hr),
                    "speed_only_n": sum(c[0] == "speed_only" for c in hc),
                    "tilt_only_n": sum(c[0] == "tilt_only" for c in hc),
                    "both_n": sum(c[0] == "both" for c in hc),
                    "deck_alone_over_limit_n": sum(c[1] == "deck_alone_over_limit" for c in hc),
                    "deck_dominated_n": sum(c[1] == "deck_dominated" for c in hc),
                    "drone_dominated_n": sum(c[1] == "drone_dominated" for c in hc),
                    "rel_tilt_p50_deg": _med(rel),
                    "rel_tilt_p95_deg": _pct(rel, 95.0),
                    "rel_tilt_max_deg": _max(rel),
                    "abs_tilt_p50_deg": _med(ab),
                    "abs_tilt_p95_deg": _pct(ab, 95.0),
                    "abs_tilt_max_deg": _max(ab),
                    "deck_tilt_p50_deg": _med(dk),
                    "deck_tilt_p95_deg": _pct(dk, 95.0),
                    "deck_tilt_max_deg": _max(dk),
                    "rel_tilt_gt_limit_n": sum(v > TILT_LIMIT_DEG for v in rel),
                    "deck_tilt_gt_limit_n": sum(v > TILT_LIMIT_DEG for v in dk),
                    "closing_gt_limit_n": sum(
                        _f(r["closing_speed_normal_m_s"]) > SPEED_LIMIT_M_S for r in td
                    ),
                    "hard_deck_tilt_median_deg": _med([_f(r["deck_tilt_deg"]) for r in hr]),
                    "hard_abs_tilt_median_deg": _med([_f(r["abs_tilt_deg"]) for r in hr]),
                    "hard_rel_tilt_median_deg": _med([_f(r["rel_tilt_deg"]) for r in hr]),
                    "hard_closing_max_m_s": _max([_f(r["closing_speed_normal_m_s"]) for r in hr]),
                }
            )
        allrow = summary[-1]
        verdicts.append(
            _verdict(
                "11",
                method,
                "hard landings: n; speed-only / tilt-only / both; deck-alone-over / "
                "deck-dominated / drone-dominated",
                f"{allrow['hard_landing_n']}; {allrow['speed_only_n']}/{allrow['tilt_only_n']}/"
                f"{allrow['both_n']}; {allrow['deck_alone_over_limit_n']}/"
                f"{allrow['deck_dominated_n']}/{allrow['drone_dominated_n']}",
                "descriptive (pre-stated classes)",
                "descriptive",
            )
        )
    _write(ctx, "hard_landing_tilt.csv", per_ep)
    _write(ctx, "tilt_summary.csv", summary)
    return verdicts


def _check_12(ctx: Phase6Context) -> list[dict[str, Any]]:
    """Check 12: every tunnelled episode's mechanism, restated per method."""
    verdicts: list[dict[str, Any]] = []
    lim_mm = 7.03
    for method in ctx.spec.methods:
        pooled = next(r for r in ctx.per_run[method] if r["run_seed"] == "pooled")
        committed_max = float(pooled[f"{ctx.spec.tag}_depth_max_mm_all"])
        t = [
            f
            for f in ctx.flights
            if f["method"] == method
            and "T" in f["sets"]
            and not math.isnan(float(f["depth_max_mm"]))
            and float(f["depth_max_mm"]) > ctx.thr * 1000.0
        ]
        mech = {
            k: sum(f["mechanism"] == k for f in t) for k in ("impact", "post_contact_idle", "other")
        }
        deepest = max(t, key=lambda f: float(f["depth_max_mm"])) if t else None
        verdicts.append(
            _verdict(
                "12",
                method,
                "tunnelled re-flown n; impact / post-contact idle / other; deepest mm (its "
                "mechanism, seed, SS/index, outcome)",
                f"{len(t)}; {mech['impact']}/{mech['post_contact_idle']}/{mech['other']}; "
                + (
                    "none"
                    if deepest is None
                    else f"{float(deepest['depth_max_mm'])!r} ({deepest['mechanism']}, seed "
                    f"{deepest['run_seed']}, {deepest['ss']}/{deepest['index']}, "
                    f"{deepest['outcome']})"
                ),
                f"limit {lim_mm} mm; committed max {committed_max!r}",
                "descriptive",
            )
        )
    return verdicts


def _reflight_counts(ctx: Phase6Context) -> None:
    """Re-flight counts per method and set, and how many reproduced their committed row."""
    rows: list[dict[str, Any]] = []
    methods = sorted({str(f["method"]) for f in ctx.flights})
    for method in methods:
        fl = [f for f in ctx.flights if f["method"] == method]
        rows.append(
            {
                "method": method,
                "flights": len(fl),
                "reproduced": sum(bool(f["reproduced"]) for f in fl),
                "in_S": sum("S" in f["sets"] for f in fl),
                "in_T": sum("T" in f["sets"] for f in fl),
                "in_H": sum("H" in f["sets"] for f in fl),
                "counterfactual_C": sum(f["method"] == method for f in ctx.counterfactual),
            }
        )
    rows.append(
        {
            "method": "all",
            "flights": len(ctx.flights),
            "reproduced": sum(bool(f["reproduced"]) for f in ctx.flights),
            "in_S": sum("S" in f["sets"] for f in ctx.flights),
            "in_T": sum("T" in f["sets"] for f in ctx.flights),
            "in_H": sum("H" in f["sets"] for f in ctx.flights),
            "counterfactual_C": len(ctx.counterfactual),
        }
    )
    _write(ctx, "reflight_counts.csv", rows)


def phase6_outputs(ctx: Phase6Context) -> list[dict[str, Any]]:
    """Write the Phase 6 tables and return their ``verdicts.csv`` rows.

    Args:
        ctx: The shared audit run's data.

    Returns:
        Verdict rows for checks 1b-fine, 6-seed, 9, 10a/10b, 11 and 12, in that order.

    Raises:
        ReproductionError: Never here; every S/T/H flight was checked in the workers.
    """
    if not ctx.spec.phase6:
        raise ValueError("phase6_outputs needs the Phase 6 spec")
    if not all(bool(f["reproduced"]) for f in ctx.flights):
        raise AssertionError("an S/T/H flight was not reproduction-checked")
    _reflight_counts(ctx)
    out: list[dict[str, Any]] = []
    out += _check_1b_fine(ctx)
    out += _check_6_per_seed(ctx)
    out += _check_9(ctx)
    out += _check_10(ctx)
    out += _check_11(ctx)
    out += _check_12(ctx)
    return out
