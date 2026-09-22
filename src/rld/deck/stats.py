"""Deck-point statistics over the whole dmf grid, and the lambda feasibility verdict.

This is the measurement Gate 1 reads. It answers three questions with committed numbers:

1. **How hard is the landing target, per cell?** Deck-point vertical excursion, velocity and
   acceleration at the model scale the PyBullet environment actually flies in.
2. **How much of that is the dmf phase defect?** Every cell is reported twice, once for the
   ``aft`` pad and once for the ``cg`` pad, side by side, plus ``corr_pitch_heave`` -- the
   defect itself, measured. Recorded, never corrected (CLAUDE.md non-negotiable 7).
3. **Is lambda = 1/25 feasible?** The pre-registered rule, evaluated under **both** readings
   of "the CF2X's commanded max speed", with the gate reading marked.

Sampling and determinism
------------------------
Every realization is evaluated at the **model physics rate** (240 Hz model = 48 Hz full at
``lam = 1/25``, 28 800 samples over the 600 s full-scale record), because that is the signal
Phase 2 sees, not the corpus's 10 Hz storage rate. The same statistic at the corpus rate is
committed beside it as ``vz_p99_abs_at_corpus_fs_m_s`` so the claim that the rate barely
matters is evidence rather than an assertion.

Work is parallelised over the 96 grid **cells**, never over realizations, and each cell pools
its realizations in sorted spec order before reducing (plan addendum C6). Pooling in
completion order would make ``std`` summation order and percentile tie-breaking depend on the
scheduler, so the CSV would not be byte-reproducible at a different worker count.

Units and scales
----------------
``*_model_*`` columns are model scale: metres, metres per second, metres per second squared.
Angles (``roll_std_deg``, ``pitch_std_deg``, ``tilt_p99_deg``) are scale-invariant degrees.
``pitch_rate_std_dps`` is the dmf channel at **full** scale, degrees per second;
``pitch_rate_std_model_dps`` is the same quantity at model scale (x ``1/sqrt(lam)`` = x 5).
``te_peak_full_s`` / ``te_peak_model_s`` are seconds. ``hs_m``, ``tp_s``, ``x_pad_full_m`` are
full scale; ``x_pad_model_m`` is model scale.

Percentile columns named ``p95``/``p99`` are percentiles of the **absolute value** of the
channel, which is what a landing budget cares about and what the feasibility rule compares.
"""

import multiprocessing as mp
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from dmf.config import SeaState, SimConfig
from dmf.sim.generate import RealizationSpec
from dmf.typedefs import FloatArray

from rld.deck.bridge import JonswapDeckMotion
from rld.deck.config import PadConfig, ScalingConfig
from rld.deck.sinusoid import (
    SinusoidParams,
    matched_rms_from_channels,
    peak_encounter_period_s,
    sinusoid_params,
)

__all__ = [
    "CG_PAD",
    "DEFAULT_WORKERS",
    "TILT_CRITERION_DEG",
    "CellKey",
    "CellPadStats",
    "FeasibilityRow",
    "RealizationPadStats",
    "cell_keys",
    "compute_cell",
    "compute_grid",
    "feasibility_rows",
    "rows_as_dicts",
]

#: The control pad whose lever arm is zero, so its motion is pure heave. Every cell's
#: ``vz_rms_ratio_pad_over_cg`` is measured against it.
CG_PAD: str = "cg"

#: The Phase 2 success criterion's relative-tilt budget, degrees. Cells whose deck tilt p99
#: alone exceeds it are structurally impossible under the criterion as written, which is a
#: finding for ``eval-auditor`` before P3-D1 freezes (plan addendum C4).
TILT_CRITERION_DEG: float = 15.0

#: Default worker count for the grid sweep.
DEFAULT_WORKERS: int = 24

#: One grid cell: ``(vessel, sea state, heading_deg, speed_kn)``. Seeds live inside it.
CellKey = tuple[str, str, float, float]


@dataclass(frozen=True)
class RealizationPadStats:
    """One realization x pad row of ``results/deck_stats_seeds.csv``.

    The audit trail behind the 192-row table, and the committed per-realization sinusoid
    parameters that Phase 6's ``ppo_sinusoid`` and Phase 7's H4 cross consume. Units are as
    given in the module docstring.
    """

    vessel: str
    ss: str
    heading_deg: float
    speed_kn: float
    seed: int
    pad: str
    lam: float
    sample_rate_full_hz: float
    n_samples: int
    x_pad_full_m: float
    x_pad_model_m: float
    z_std_model_m: float
    vz_std_model_m_s: float
    vz_rms_model_m_s: float
    vz_p95_model_m_s: float
    vz_p99_model_m_s: float
    az_p99_model_m_s2: float
    roll_std_deg: float
    pitch_std_deg: float
    pitch_rate_std_dps: float
    pitch_rate_std_model_dps: float
    tilt_p99_deg: float
    corr_pitch_heave: float
    vz_p99_abs_at_corpus_fs_m_s: float
    te_peak_full_s: float
    te_peak_model_s: float
    sin_roll_amp_deg: float
    sin_pitch_amp_deg: float
    sin_heave_amp_m: float
    sin_te_full_s: float
    sin_episode_seed: int


@dataclass(frozen=True)
class CellPadStats:
    """One cell x pad row of ``results/deck_stats.csv`` -- 192 rows over the committed grid.

    Every dispersion and percentile column is computed on the cell's realizations **pooled**
    in sorted spec order, not averaged over per-realization values, so the number is
    independent of worker count and of scheduling (plan addendum C6). Per-realization spread
    is still visible through ``vz_p99_model_m_s_seed_min`` / ``_seed_max``.
    """

    vessel: str
    ss: str
    hs_m: float
    tp_s: float
    heading_deg: float
    speed_kn: float
    pad: str
    n_seeds: int
    lam: float
    sample_rate_full_hz: float
    x_pad_full_m: float
    x_pad_model_m: float
    z_std_model_m: float
    vz_std_model_m_s: float
    vz_rms_model_m_s: float
    vz_p95_model_m_s: float
    vz_p99_model_m_s: float
    az_p99_model_m_s2: float
    roll_std_deg: float
    pitch_std_deg: float
    pitch_rate_std_dps: float
    pitch_rate_std_model_dps: float
    te_peak_full_s: float
    te_peak_model_s: float
    vz_p99_model_m_s_seed_min: float
    vz_p99_model_m_s_seed_max: float
    tilt_p99_deg: float
    tilt_p99_over_15deg: bool
    corr_pitch_heave: float
    vz_rms_ratio_pad_over_cg: float
    vz_p99_abs_at_corpus_fs_m_s: float


@dataclass(frozen=True)
class FeasibilityRow:
    """One row of ``results/deck_feasibility.csv``: the rule under one reference speed.

    Attributes:
        rule: The pre-registered rule text, committed verbatim.
        denominator_source: Where the reference speed comes from, file and line.
        reference_name: Short identifier of the reference.
        is_gate: True for the single row Gate 1 reads.
        v_max_m_s: Reference speed, metres per second, model scale.
        fraction: Dimensionless fraction of ``v_max_m_s`` allowed.
        threshold_m_s: ``fraction * v_max_m_s``, metres per second, model scale.
        vessel, ss, heading_deg, speed_kn, pad: The measured cell.
        vz_p99_model_m_s: That cell's measured statistic, metres per second, model scale.
        lam: The Froude scale the measurement was made at.
        verdict: ``"PASS"`` or ``"FAIL"``.
    """

    rule: str
    denominator_source: str
    reference_name: str
    is_gate: bool
    v_max_m_s: float
    fraction: float
    threshold_m_s: float
    vessel: str
    ss: str
    heading_deg: float
    speed_kn: float
    pad: str
    vz_p99_model_m_s: float
    lam: float
    verdict: str


def cell_keys(cfg: SimConfig) -> list[CellKey]:
    """Enumerate the grid's cells in a fixed order.

    Args:
        cfg: dmf's corpus config.

    Returns:
        One ``(vessel, sea state, heading_deg, speed_kn)`` per cell, in
        :func:`dmf.sim.generate.realization_grid` nesting order -- 96 for the committed grid.
    """
    return [
        (vessel, sea.name, float(heading), float(speed))
        for vessel in cfg.vessels
        for sea in cfg.sea_states
        for heading in cfg.headings_deg
        for speed in cfg.speeds_kn
    ]


def _abs_percentile(values: FloatArray, q: float) -> float:
    """Return the ``q``-th percentile of ``|values|``, in the units of ``values``."""
    return float(np.percentile(np.abs(values), q))


def _sea_state(name: str, cfg: SimConfig) -> SeaState:
    """Return the named sea state.

    Args:
        name: Sea state label.
        cfg: dmf's corpus config.

    Returns:
        The sea state; ``hs_m`` metres and ``tp_s`` seconds, full scale.

    Raises:
        ValueError: If the label is not configured.
    """
    for sea in cfg.sea_states:
        if sea.name == name:
            return sea
    raise ValueError(f"unknown sea state {name!r}")


@dataclass(frozen=True)
class _PadSamples:
    """Pooled model-scale samples of one pad over one realization or one cell."""

    z_m: FloatArray
    vz_m_s: FloatArray
    az_m_s2: FloatArray
    vz_corpus_fs_m_s: FloatArray


@dataclass(frozen=True)
class _SharedSamples:
    """Pooled attitude samples, which do not depend on the pad."""

    roll_deg: FloatArray
    pitch_deg: FloatArray
    heave_m: FloatArray
    pitch_rate_dps: FloatArray
    tilt_deg: FloatArray


def _model_grid(cfg: SimConfig, scale_lam: float, physics_freq_hz: float) -> FloatArray:
    """Return the model-scale sample times covering the whole committed record.

    Args:
        cfg: dmf's corpus config (``duration_s`` is full scale).
        scale_lam: Froude length scale, dimensionless.
        physics_freq_hz: Model-scale sampling rate, hertz.

    Returns:
        Model-scale seconds, starting at 0 (the first committed sample) and stepping at
        ``1/physics_freq_hz``. 28 800 samples for the committed 600 s record at 240 Hz.
    """
    n = int(round(cfg.duration_s * scale_lam**0.5 * physics_freq_hz))
    return np.arange(n, dtype=np.float64) / physics_freq_hz


def _corpus_grid(cfg: SimConfig, scale_lam: float) -> FloatArray:
    """Return model-scale sample times at the corpus's own 10 Hz full-scale rate."""
    n = int(round(cfg.duration_s * cfg.fs_hz))
    return np.asarray(np.arange(n, dtype=np.float64) / cfg.fs_hz * scale_lam**0.5)


def compute_cell(
    cell: CellKey,
    cfg: SimConfig,
    scaling: ScalingConfig,
    pads: PadConfig,
    physics_freq_hz: float,
    episode_seed: int = 0,
) -> tuple[list[CellPadStats], list[RealizationPadStats]]:
    """Evaluate every realization of one grid cell and reduce it to one row per pad.

    Module-level and dependency-free of any shared state, so it is picklable by
    ``multiprocessing`` and two workers can never interfere: everything it needs comes from
    its arguments, and every seed comes from the realization spec.

    Args:
        cell: ``(vessel, sea state, heading_deg, speed_kn)``.
        cfg: dmf's corpus config, full scale.
        scaling: Froude scale and the feasibility rule.
        pads: Pad geometry.
        physics_freq_hz: Model-scale sampling rate, hertz.
        episode_seed: Episode seed for the committed sinusoid phases.

    Returns:
        ``(cell_rows, realization_rows)``: one :class:`CellPadStats` per pad (in config pad
        order) and one :class:`RealizationPadStats` per realization x pad, the latter in
        sorted spec order.
    """
    vessel, ss, heading_deg, speed_kn = cell
    scale = scaling.froude_scale()
    sea = _sea_state(ss, cfg)
    specs = [
        RealizationSpec(
            seed=seed, sea_state=ss, heading_deg=heading_deg, speed_kn=speed_kn, vessel=vessel
        )
        for seed in range(cfg.seeds_for(vessel))
    ]
    t_model_s = _model_grid(cfg, scale.lam, physics_freq_hz)
    t_corpus_model_s = _corpus_grid(cfg, scale.lam)
    te_full_s = peak_encounter_period_s(sea.tp_s, speed_kn, heading_deg)
    pad_names = pads.names

    pooled_pad: dict[str, list[_PadSamples]] = {name: [] for name in pad_names}
    pooled_shared: list[_SharedSamples] = []
    realization_rows: list[RealizationPadStats] = []
    per_seed_vz_p99: dict[str, list[float]] = {name: [] for name in pad_names}

    for spec in specs:  # sorted by construction: seed ordinals ascending within the cell
        source = JonswapDeckMotion(spec, cfg, scale, pads)
        channels = source.channels(source.full_time_s(t_model_s))
        points = source.deck_points(t_model_s, pad_names)
        corpus_channels = source.channels(source.full_time_s(t_corpus_model_s))
        corpus_points = source.deck_points(t_corpus_model_s, pad_names)
        params = sinusoid_params(
            spec,
            cfg,
            scale,
            pads,
            episode_seed=episode_seed,
            rms_full=matched_rms_from_channels(corpus_channels),
        )
        shared = _SharedSamples(
            roll_deg=channels.roll_deg,
            pitch_deg=channels.pitch_deg,
            heave_m=channels.heave_m,
            pitch_rate_dps=channels.pitch_rate_dps,
            tilt_deg=points[pad_names[0]].state.tilt_deg,
        )
        pooled_shared.append(shared)
        for name in pad_names:
            state = points[name].state
            samples = _PadSamples(
                z_m=state.position_m[:, 2],
                vz_m_s=state.velocity_m_s[:, 2],
                az_m_s2=state.acceleration_m_s2[:, 2],
                vz_corpus_fs_m_s=corpus_points[name].state.velocity_m_s[:, 2],
            )
            pooled_pad[name].append(samples)
            per_seed_vz_p99[name].append(_abs_percentile(samples.vz_m_s, 99.0))
            realization_rows.append(
                _realization_row(
                    spec,
                    name,
                    samples,
                    shared,
                    scaling,
                    pads,
                    te_full_s,
                    params,
                    physics_freq_hz,
                    t_model_s.size,
                )
            )

    cell_rows = [
        _cell_row(
            cell,
            name,
            sea,
            len(specs),
            pooled_pad,
            pooled_shared,
            per_seed_vz_p99[name],
            scaling,
            pads,
            te_full_s,
            physics_freq_hz,
        )
        for name in pad_names
    ]
    return cell_rows, realization_rows


def _pad_arm_m(
    pad_name: str, vessel: str, pads: PadConfig, scaling: ScalingConfig
) -> tuple[float, float]:
    """Return the pad's longitudinal lever arm, ``(full scale, model scale)`` metres."""
    from rld.deck.bridge import load_vessel_cached

    x_full = pads.spec(pad_name).r_pad_full_m(load_vessel_cached(vessel).length_m)[0]
    return x_full, scaling.froude_scale().length(x_full)


def _realization_row(
    spec: RealizationSpec,
    pad_name: str,
    samples: _PadSamples,
    shared: _SharedSamples,
    scaling: ScalingConfig,
    pads: PadConfig,
    te_full_s: float,
    params: SinusoidParams,
    physics_freq_hz: float,
    n_samples: int,
) -> RealizationPadStats:
    """Reduce one realization x pad to its committed row."""
    scale = scaling.froude_scale()
    x_full_m, x_model_m = _pad_arm_m(pad_name, spec.vessel, pads, scaling)
    return RealizationPadStats(
        vessel=spec.vessel,
        ss=spec.sea_state,
        heading_deg=spec.heading_deg,
        speed_kn=spec.speed_kn,
        seed=spec.seed,
        pad=pad_name,
        lam=scale.lam,
        sample_rate_full_hz=physics_freq_hz * scale.sqrt_lam,
        n_samples=n_samples,
        x_pad_full_m=x_full_m,
        x_pad_model_m=x_model_m,
        z_std_model_m=float(np.std(samples.z_m)),
        vz_std_model_m_s=float(np.std(samples.vz_m_s)),
        vz_rms_model_m_s=float(np.sqrt(np.mean(np.square(samples.vz_m_s)))),
        vz_p95_model_m_s=_abs_percentile(samples.vz_m_s, 95.0),
        vz_p99_model_m_s=_abs_percentile(samples.vz_m_s, 99.0),
        az_p99_model_m_s2=_abs_percentile(samples.az_m_s2, 99.0),
        roll_std_deg=float(np.std(shared.roll_deg)),
        pitch_std_deg=float(np.std(shared.pitch_deg)),
        pitch_rate_std_dps=float(np.std(shared.pitch_rate_dps)),
        pitch_rate_std_model_dps=float(scale.angular_rate(np.std(shared.pitch_rate_dps))),
        tilt_p99_deg=float(np.percentile(shared.tilt_deg, 99.0)),
        corr_pitch_heave=float(np.corrcoef(shared.pitch_deg, shared.heave_m)[0, 1]),
        vz_p99_abs_at_corpus_fs_m_s=_abs_percentile(samples.vz_corpus_fs_m_s, 99.0),
        te_peak_full_s=te_full_s,
        te_peak_model_s=float(scale.time(te_full_s)),
        sin_roll_amp_deg=params.roll_amp_deg,
        sin_pitch_amp_deg=params.pitch_amp_deg,
        sin_heave_amp_m=params.heave_amp_m,
        sin_te_full_s=params.te_full_s,
        sin_episode_seed=params.episode_seed,
    )


def _cell_row(
    cell: CellKey,
    pad_name: str,
    sea: SeaState,
    n_seeds: int,
    pooled_pad: dict[str, list[_PadSamples]],
    pooled_shared: list[_SharedSamples],
    per_seed_vz_p99: list[float],
    scaling: ScalingConfig,
    pads: PadConfig,
    te_full_s: float,
    physics_freq_hz: float,
) -> CellPadStats:
    """Reduce one cell x pad from its pooled samples to the committed row."""
    vessel, ss, heading_deg, speed_kn = cell
    scale = scaling.froude_scale()
    x_full_m, x_model_m = _pad_arm_m(pad_name, vessel, pads, scaling)
    z_m = np.concatenate([s.z_m for s in pooled_pad[pad_name]])
    vz_m_s = np.concatenate([s.vz_m_s for s in pooled_pad[pad_name]])
    az_m_s2 = np.concatenate([s.az_m_s2 for s in pooled_pad[pad_name]])
    vz_corpus = np.concatenate([s.vz_corpus_fs_m_s for s in pooled_pad[pad_name]])
    vz_cg = np.concatenate([s.vz_m_s for s in pooled_pad[CG_PAD]])
    roll_deg = np.concatenate([s.roll_deg for s in pooled_shared])
    pitch_deg = np.concatenate([s.pitch_deg for s in pooled_shared])
    heave_m = np.concatenate([s.heave_m for s in pooled_shared])
    pitch_rate_dps = np.concatenate([s.pitch_rate_dps for s in pooled_shared])
    tilt = np.concatenate([s.tilt_deg for s in pooled_shared])
    vz_rms = float(np.sqrt(np.mean(np.square(vz_m_s))))
    vz_rms_cg = float(np.sqrt(np.mean(np.square(vz_cg))))
    tilt_p99 = float(np.percentile(tilt, 99.0))
    return CellPadStats(
        vessel=vessel,
        ss=ss,
        hs_m=sea.hs_m,
        tp_s=sea.tp_s,
        heading_deg=heading_deg,
        speed_kn=speed_kn,
        pad=pad_name,
        n_seeds=n_seeds,
        lam=scale.lam,
        sample_rate_full_hz=physics_freq_hz * scale.sqrt_lam,
        x_pad_full_m=x_full_m,
        x_pad_model_m=x_model_m,
        z_std_model_m=float(np.std(z_m)),
        vz_std_model_m_s=float(np.std(vz_m_s)),
        vz_rms_model_m_s=vz_rms,
        vz_p95_model_m_s=_abs_percentile(vz_m_s, 95.0),
        vz_p99_model_m_s=_abs_percentile(vz_m_s, 99.0),
        az_p99_model_m_s2=_abs_percentile(az_m_s2, 99.0),
        roll_std_deg=float(np.std(roll_deg)),
        pitch_std_deg=float(np.std(pitch_deg)),
        pitch_rate_std_dps=float(np.std(pitch_rate_dps)),
        pitch_rate_std_model_dps=float(scale.angular_rate(np.std(pitch_rate_dps))),
        te_peak_full_s=te_full_s,
        te_peak_model_s=float(scale.time(te_full_s)),
        vz_p99_model_m_s_seed_min=float(min(per_seed_vz_p99)),
        vz_p99_model_m_s_seed_max=float(max(per_seed_vz_p99)),
        tilt_p99_deg=tilt_p99,
        tilt_p99_over_15deg=bool(tilt_p99 > TILT_CRITERION_DEG),
        corr_pitch_heave=float(np.corrcoef(pitch_deg, heave_m)[0, 1]),
        vz_rms_ratio_pad_over_cg=vz_rms / vz_rms_cg if vz_rms_cg else float("nan"),
        vz_p99_abs_at_corpus_fs_m_s=_abs_percentile(vz_corpus, 99.0),
    )


def _worker(task: tuple[CellKey, SimConfig, ScalingConfig, PadConfig, float, int]) -> Any:
    """Unpack one cell task for ``multiprocessing``. Module-level so that it is picklable."""
    return compute_cell(*task)


def compute_grid(
    cfg: SimConfig,
    scaling: ScalingConfig,
    pads: PadConfig,
    physics_freq_hz: float,
    workers: int = DEFAULT_WORKERS,
    episode_seed: int = 0,
    cells: Sequence[CellKey] | None = None,
) -> tuple[list[CellPadStats], list[RealizationPadStats]]:
    """Evaluate the whole grid, cell by cell, and return both committed tables.

    Args:
        cfg: dmf's corpus config, full scale.
        scaling: Froude scale and the feasibility rule.
        pads: Pad geometry.
        physics_freq_hz: Model-scale sampling rate, hertz.
        workers: Process count. Work is split over **cells**; results are re-sorted into
            grid order before returning, so the output does not depend on this value.
        episode_seed: Episode seed for the committed sinusoid phases.
        cells: Optional subset of cells, for tests. Defaults to the whole grid.

    Returns:
        ``(cell_rows, realization_rows)`` in grid order -- 192 and 4608 rows respectively for
        the committed grid.

    Raises:
        ValueError: If ``workers`` is not positive.
    """
    if workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    todo = list(cells) if cells is not None else cell_keys(cfg)
    tasks = [(cell, cfg, scaling, pads, physics_freq_hz, episode_seed) for cell in todo]
    if workers == 1:
        results = [_worker(task) for task in tasks]
    else:
        with mp.get_context("spawn").Pool(processes=min(workers, len(tasks))) as pool:
            results = list(pool.imap(_worker, tasks, chunksize=1))
    order = {cell: index for index, cell in enumerate(todo)}
    cell_rows: list[CellPadStats] = []
    realization_rows: list[RealizationPadStats] = []
    for _index, (cells_out, seeds_out) in sorted(
        zip([order[cell] for cell in todo], results, strict=True), key=lambda pair: pair[0]
    ):
        cell_rows.extend(cells_out)
        realization_rows.extend(seeds_out)
    return cell_rows, realization_rows


def feasibility_rows(
    cell_rows: Iterable[CellPadStats],
    scaling: ScalingConfig,
    pad: str = "aft",
    vessel: str = "frigate",
    sea_state: str = "SS6",
    heading_deg: float = 180.0,
    reference: str = "all",
) -> list[FeasibilityRow]:
    """Evaluate the pre-registered lambda feasibility rule against the measured grid.

    The measured cell is the **worst** speed of the named vessel's head-seas cells at the
    named sea state and pad -- the rule's own wording ("SS6 head-seas deck-point v_z p99").

    Args:
        cell_rows: The committed cell table.
        scaling: The Froude scale and the rule, including both candidate denominators.
        pad: Pad the rule is measured at.
        vessel: Vessel the rule is measured on.
        sea_state: Sea state the rule is measured at.
        heading_deg: Heading the rule is measured at.
        reference: ``"all"`` to emit one row per candidate denominator, ``"gate"`` for only
            the row Gate 1 reads.

    Returns:
        One :class:`FeasibilityRow` per selected reference.

    Raises:
        ValueError: If the named cell is absent from ``cell_rows`` or ``reference`` is
            unknown.
    """
    if reference not in ("all", "gate"):
        raise ValueError(f"reference must be 'all' or 'gate', got {reference!r}")
    candidates = [
        row
        for row in cell_rows
        if row.vessel == vessel
        and row.ss == sea_state
        and row.heading_deg == heading_deg
        and row.pad == pad
    ]
    if not candidates:
        raise ValueError(
            f"no {vessel} {sea_state} {heading_deg} deg {pad!r} cell in the measured grid"
        )
    worst = max(candidates, key=lambda row: row.vz_p99_model_m_s)
    rule = scaling.feasibility
    refs = rule.references if reference == "all" else (rule.gate_reference,)
    return [
        FeasibilityRow(
            rule=rule.rule,
            denominator_source=ref.source,
            reference_name=ref.name,
            is_gate=ref.is_gate,
            v_max_m_s=ref.v_max_m_s,
            fraction=rule.fraction,
            threshold_m_s=rule.threshold_m_s(ref),
            vessel=worst.vessel,
            ss=worst.ss,
            heading_deg=worst.heading_deg,
            speed_kn=worst.speed_kn,
            pad=worst.pad,
            vz_p99_model_m_s=worst.vz_p99_model_m_s,
            lam=scaling.lam,
            verdict="PASS" if worst.vz_p99_model_m_s <= rule.threshold_m_s(ref) else "FAIL",
        )
        for ref in refs
    ]


def rows_as_dicts(rows: Iterable[Any]) -> list[dict[str, Any]]:
    """Return frozen result rows as plain dicts, ready for ``csv.DictWriter``.

    Args:
        rows: Frozen dataclass rows.

    Returns:
        One dict per row, field order preserved, values unrounded.
    """
    return [asdict(row) for row in rows]
