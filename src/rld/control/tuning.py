"""Tuning the classical baselines on the tune pool, with a stated and equal budget.

What this is, and what it must never touch
------------------------------------------
``pid_track_descend`` and ``pid_feedforward`` are tuned with the **same** budget, the same
search space on their shared parameters, and the same episodes, so that the naive baseline
is not strawmanned against the residual-RL base. Episodes come **only** from the P3-D2 tune
pool, :func:`rld.deck.splits.dev_pool` ``[1]`` (frigate, SS3-SS5, heading != 90 deg, seed
ordinals 27-31), which is disjoint from every regime's test partition. Nothing here reads
``results/episodes/`` or builds a test partition. ``gated`` and ``oracle_gated`` are not
tuned: they inherit ``pid_feedforward``'s gains and dmf's thresholds.

Procedure (``configs/control/tuning.yaml``; recorded as ``docs/protocol.md`` P3-D3)
------------------------------------------------------------------------------------
* **Draw.** ``episodes_per_sea_state`` episodes per sea state from the tune pool, with the
  committed ``tuning_seed`` mixed with a SHA-256 label (never ``hash()``, which is salted
  per process). The draw depends on neither the controller nor the trial, so every trial
  of both controllers flies the identical episodes: a paired comparison.
* **Trials.** Trial 0 is the hand-set ``initial`` point; trials ``1..n-1`` are the first
  ``n-1`` points of **one** scrambled Sobol sequence of dimension 5 over
  ``search_space``, seeded with ``sobol_seed``. ``pid_track_descend`` reads its first four
  coordinates and ignores ``k_ff``, so both controllers are evaluated at identical
  ``(kp, ki, radius, descent)`` points. Parameters outside the search space come from the
  controller's committed YAML unchanged.
* **Objective.** Mean success over the sea states; ties broken by the lower pooled p95
  touchdown closing speed along the deck normal, then by the lower trial index.
* **Low-closing-speed reference** (Gate 3 remediation, 2026-09-22). :func:`select_lowvz`
  applies a second, fixed rule to the **existing** ``pid_feedforward`` log -- no new trial,
  no new episode: keep the trials whose objective is ``>= best - 0.02``, take the lowest
  pooled p95 closing speed, then the lowest trial index. It selects trial 10, whose gains
  are ``configs/control/pid_feedforward_lowvz.yaml``.

Worker independence
-------------------
Work is split over contiguous chunks of one trial's episodes, one environment per chunk
re-pointed with ``set_motion`` (the :mod:`rld.envs.sanity` pattern), and every result is
re-sorted into ``(trial, sea state, index)`` order, so the output does not depend on
``workers``.

Units and scales
----------------
Metres, metres per second and seconds are **model** scale; angles degrees.
"""

import hashlib
import multiprocessing as mp
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from dmf.config import SimConfig
from dmf.sim.generate import RealizationSpec
from scipy.stats import qmc

from rld.config import load_yaml, require
from rld.control.base import Controller, ControlSpec, PrivilegedContext
from rld.control.config import CONTROL_CONFIG_DIR, load_feedforward, load_pid
from rld.control.feedforward import PidFeedforward
from rld.control.pid import PidTrackDescend
from rld.control.registry import entry, make_controller
from rld.deck.bridge import JonswapDeckMotion
from rld.deck.splits import dev_pool
from rld.envs.landing_env import DeckLandingAviary
from rld.envs.platform import pad_offset_model_m
from rld.envs.sanity import SweepConfigs
from rld.envs.touchdown import OUTCOMES

__all__ = [
    "DEFAULT_CHUNK",
    "PID_PARAMS",
    "TUNED_CONTROLLERS",
    "TUNING_CONFIG",
    "LOWVZ_SUCCESS_MARGIN",
    "TrialResult",
    "TuningConfig",
    "TuningEpisode",
    "build_controller",
    "draw_tuning_episodes",
    "load_tuning",
    "read_tuning_log",
    "run_trials",
    "select_lowvz",
    "select_trial",
    "summarise_trial",
    "trial_points",
]

#: The committed tuning config.
TUNING_CONFIG: Path = CONTROL_CONFIG_DIR / "tuning.yaml"

#: Controllers this module tunes. ``gated``/``oracle_gated`` inherit ``pid_feedforward``.
TUNED_CONTROLLERS: tuple[str, ...] = ("pid_track_descend", "pid_feedforward")

#: Search-space keys that belong to :class:`~rld.control.config.PidConfig`.
PID_PARAMS: tuple[str, ...] = (
    "kp_xy_per_s",
    "ki_xy_per_s2",
    "commit_radius_m",
    "descent_rate_m_s",
)

#: Search-space keys each tuned controller reads, in Sobol-coordinate order.
CONTROLLER_PARAMS: dict[str, tuple[str, ...]] = {
    "pid_track_descend": PID_PARAMS,
    "pid_feedforward": (*PID_PARAMS, "k_ff"),
}

#: Episodes per parallel task.
DEFAULT_CHUNK: int = 15

#: ``select_lowvz``'s success margin below the best trial, dimensionless (2 points).
#: Fixed by the user's rule; a constant, not a parameter, so it cannot be varied per call.
LOWVZ_SUCCESS_MARGIN: float = 0.02


@dataclass(frozen=True)
class TuningConfig:
    """``configs/control/tuning.yaml``.

    Attributes:
        tuning_seed: Committed episode-draw seed.
        sea_states: Sea-state labels drawn, e.g. ``("SS3", "SS4", "SS5")``.
        episodes_per_sea_state: Episodes per sea state per trial.
        pad: Pad name.
        trials: Trials per controller, including trial 0.
        sobol_seed: Seed of the scrambled Sobol sequence.
        search_space: Parameter name to ``(low, high)``, in the parameter's unit. The key
            order is the Sobol coordinate order.
        initial: Trial 0's parameters.
    """

    tuning_seed: int
    sea_states: tuple[str, ...]
    episodes_per_sea_state: int
    pad: str
    trials: int
    sobol_seed: int
    search_space: dict[str, tuple[float, float]]
    initial: dict[str, float]


def load_tuning(path: Path = TUNING_CONFIG) -> TuningConfig:
    """Load the tuning config.

    Args:
        path: The YAML file.

    Returns:
        The parsed :class:`TuningConfig`.

    Raises:
        ValueError: If a key is missing, a range is not ordered, the search space does not
            cover every tuned controller's parameters, or ``initial`` misses one.
    """
    raw = load_yaml(path)
    space_raw = require(raw, "search_space", path)
    space = {str(k): (float(v[0]), float(v[1])) for k, v in dict(space_raw).items()}
    for name, (low, high) in space.items():
        if low > high:
            raise ValueError(f"{path}: search_space.{name} is not ordered: {(low, high)}")
    initial = {str(k): float(v) for k, v in dict(require(raw, "initial", path)).items()}
    needed = {key for keys in CONTROLLER_PARAMS.values() for key in keys}
    if missing := sorted(needed - set(space)):
        raise ValueError(f"{path}: search_space lacks {missing}")
    if missing := sorted(needed - set(initial)):
        raise ValueError(f"{path}: initial lacks {missing}")
    cfg = TuningConfig(
        tuning_seed=int(require(raw, "tuning_seed", path)),
        sea_states=tuple(str(s) for s in require(raw, "sea_states", path)),
        episodes_per_sea_state=int(require(raw, "episodes_per_sea_state", path)),
        pad=str(require(raw, "pad", path)),
        trials=int(require(raw, "trials", path)),
        sobol_seed=int(require(raw, "sobol_seed", path)),
        search_space=space,
        initial=initial,
    )
    if cfg.trials < 1 or cfg.episodes_per_sea_state < 1:
        raise ValueError(f"{path}: trials and episodes_per_sea_state must be positive")
    return cfg


@dataclass(frozen=True)
class TuningEpisode:
    """One tuning episode: which tune-pool realization and which episode seed.

    Attributes:
        ss: Sea-state label.
        index: Position in the sea state's draw order.
        pad: Pad name.
        vessel: Vessel name (always ``"frigate"`` in the tune pool).
        heading_deg: Encounter heading, degrees.
        speed_kn: Forward speed, knots, full scale.
        realization_seed: dmf realization seed ordinal (27-31 in the tune pool).
        episode_seed: The environment's episode seed: start offset and initial state.
    """

    ss: str
    index: int
    pad: str
    vessel: str
    heading_deg: float
    speed_kn: float
    realization_seed: int
    episode_seed: int

    @property
    def realization(self) -> RealizationSpec:
        """Return the dmf realization this episode flies against."""
        return RealizationSpec(
            sea_state=self.ss,
            heading_deg=self.heading_deg,
            speed_kn=self.speed_kn,
            vessel=self.vessel,
            seed=self.realization_seed,
        )


def _label_entropy(label: str) -> int:
    """Return a stable 32-bit value for a label, via SHA-256 (``hash()`` is salted).

    Args:
        label: Any string.

    Returns:
        A value in ``[0, 2**32)``.
    """
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "big")


def draw_tuning_episodes(sim_cfg: SimConfig, cfg: TuningConfig) -> list[TuningEpisode]:
    """Draw the tuning episodes from the tune pool only.

    Args:
        sim_cfg: dmf's corpus config; the pool is a pure function of its grid.
        cfg: The tuning config.

    Returns:
        ``episodes_per_sea_state`` episodes per sea state, sea states in config order, each
        in draw order.

    Raises:
        ValueError: If the tune pool has no realization at a requested sea state.
    """
    _, tune = dev_pool(sim_cfg)
    episodes: list[TuningEpisode] = []
    for ss in cfg.sea_states:
        candidates = [spec for spec in tune if spec.sea_state == ss]
        if not candidates:
            raise ValueError(f"the tune pool has no realization at {ss}")
        seq = np.random.SeedSequence([cfg.tuning_seed, _label_entropy(f"tune|{ss}")])
        rng = np.random.default_rng(seq)
        picks = rng.integers(0, len(candidates), size=cfg.episodes_per_sea_state)
        seeds = rng.integers(0, 2**31 - 1, size=cfg.episodes_per_sea_state)
        for index, (pick, seed) in enumerate(zip(picks, seeds, strict=True)):
            spec = candidates[int(pick)]
            episodes.append(
                TuningEpisode(
                    ss=ss,
                    index=index,
                    pad=cfg.pad,
                    vessel=spec.vessel,
                    heading_deg=float(spec.heading_deg),
                    speed_kn=float(spec.speed_kn),
                    realization_seed=int(spec.seed),
                    episode_seed=int(seed),
                )
            )
    return episodes


def trial_points(controller: str, cfg: TuningConfig) -> list[dict[str, float]]:
    """Return every trial's parameters for one tuned controller.

    Args:
        controller: ``"pid_track_descend"`` or ``"pid_feedforward"``.
        cfg: The tuning config.

    Returns:
        ``cfg.trials`` parameter dicts: trial 0 is ``initial``, the rest are Sobol points
        mapped linearly into ``search_space``. Units as the parameter names say.

    Raises:
        KeyError: If ``controller`` is not tuned here.
    """
    keys = CONTROLLER_PARAMS[controller]
    space_keys = list(cfg.search_space)
    points: list[dict[str, float]] = [{key: cfg.initial[key] for key in keys}]
    if cfg.trials > 1:
        # Power-of-two draw then truncate: scipy warns on non-power-of-two sample sizes
        # because the balance properties hold only for full blocks.
        m = int(np.ceil(np.log2(cfg.trials - 1))) if cfg.trials > 2 else 0
        sampler = qmc.Sobol(d=len(space_keys), scramble=True, seed=cfg.sobol_seed)
        unit = sampler.random_base2(m=m)[: cfg.trials - 1]
        for row in unit:
            point = {}
            for coord, key in enumerate(space_keys):
                if key in keys:
                    low, high = cfg.search_space[key]
                    point[key] = float(low + row[coord] * (high - low))
            points.append(point)
    return points


def build_controller(controller: str, params: Mapping[str, float], spec: ControlSpec) -> Controller:
    """Build a controller with tuned parameters overriding its committed YAML.

    Args:
        controller: Registry name.
        params: Overrides; must be empty for a controller that is not tuned here.
        spec: Environment facts.

    Returns:
        A fresh controller.

    Raises:
        ValueError: If overrides are given for an untuned controller.
    """
    if controller == "pid_track_descend":
        base = load_pid(entry(controller).config_path)
        return PidTrackDescend(
            replace(base, **{k: params[k] for k in PID_PARAMS if k in params}), spec
        )
    if controller == "pid_feedforward":
        ff = load_feedforward(entry(controller).config_path)
        pid = replace(ff.pid, **{k: params[k] for k in PID_PARAMS if k in params})
        return PidFeedforward(replace(ff, pid=pid, k_ff=params.get("k_ff", ff.k_ff)), spec)
    if params:
        raise ValueError(f"{controller} is not tuned; it takes no parameter overrides")
    return make_controller(controller, spec)


type Task = tuple[str, int, tuple[tuple[str, float], ...], Sequence[TuningEpisode], SweepConfigs]


def _control_spec(cfgs: SweepConfigs) -> ControlSpec:
    """Return the controller-facing view of the sweep configs.

    Args:
        cfgs: The sweep configs.

    Returns:
        The :class:`ControlSpec`.
    """
    return ControlSpec(
        landing=cfgs.landing, observation=cfgs.observation, scale=cfgs.scaling.froude_scale()
    )


def run_chunk(task: Task) -> list[dict[str, Any]]:
    """Run one contiguous chunk of one trial's episodes in one environment.

    Module-level with one packed argument so ``multiprocessing`` can pickle it.

    Args:
        task: ``(controller, trial, params, episodes, configs)``.

    Returns:
        One flat row per episode, in the order given: the draw, then
        ``EpisodeRecord.as_row()``, then the detector-disagreement flag.
    """
    controller_name, trial, params, episodes, cfgs = task
    if not episodes:
        return []
    scale = cfgs.scaling.froude_scale()
    lookback = cfgs.motion.forecast_lookback_full_s
    spec = _control_spec(cfgs)
    controller = build_controller(controller_name, dict(params), spec)
    privileged = entry(controller_name).privileged
    first = episodes[0]
    env = DeckLandingAviary(
        motion=JonswapDeckMotion(first.realization, cfgs.sim, scale, cfgs.pads, lookback),
        pad=first.pad,
        pad_radius_m=cfgs.pads.radius_model_m,
        pad_offset_m=pad_offset_model_m(first.vessel, cfgs.pads, scale, first.pad),
        cfg=cfgs.landing,
        success=cfgs.success,
        obs_cfg=cfgs.observation,
        noise_cfg=cfgs.noise,
        reward_cfg=cfgs.reward,
        episode_seed=first.episode_seed,
    )
    rows: list[dict[str, Any]] = []
    max_steps = int(cfgs.landing.total_len_s * cfgs.landing.ctrl_freq_hz) + 1
    try:
        for episode in episodes:
            env.set_motion(
                JonswapDeckMotion(episode.realization, cfgs.sim, scale, cfgs.pads, lookback),
                episode.pad,
                pad_offset_model_m(episode.vessel, cfgs.pads, scale, episode.pad),
            )
            obs, _ = env.reset(seed=episode.episode_seed)
            context = PrivilegedContext.from_env(env) if privileged else None
            controller.reset(episode.episode_seed, context)
            for _ in range(max_steps):
                obs, _, terminated, truncated, _ = env.step(controller.act(obs))
                if terminated or truncated:
                    break
            rows.append(
                {
                    "controller": controller_name,
                    "trial": trial,
                    "ss": episode.ss,
                    "index": episode.index,
                    "vessel": episode.vessel,
                    "heading_deg": episode.heading_deg,
                    "speed_kn": episode.speed_kn,
                    "realization_seed": episode.realization_seed,
                    "episode_seed": episode.episode_seed,
                    **env.record.as_row(),
                    "detectors_disagree": env.record.detectors_disagree(
                        cfgs.success.detector_disagreement_window_s
                    ),
                }
            )
    finally:
        env.close()
    return rows


def run_trials(
    controller: str,
    points: Sequence[Mapping[str, float]],
    episodes: Sequence[TuningEpisode],
    cfgs: SweepConfigs,
    *,
    workers: int,
    chunk: int = DEFAULT_CHUNK,
) -> list[dict[str, Any]]:
    """Run every trial on the identical episodes and return the per-episode rows.

    Args:
        controller: Registry name.
        points: One parameter dict per trial (empty for an untuned controller).
        episodes: The tuning draw.
        cfgs: Every committed config the environment needs.
        workers: Processes; the output does not depend on it.
        chunk: Episodes per task.

    Returns:
        Rows sorted by ``(trial, sea state order, index)``.

    Raises:
        ValueError: If ``workers`` or ``chunk`` is not positive.
    """
    if workers < 1 or chunk < 1:
        raise ValueError(f"workers and chunk must be positive, got {workers}, {chunk}")
    tasks: list[Task] = []
    for trial, params in enumerate(points):
        packed = tuple(sorted((str(k), float(v)) for k, v in params.items()))
        for start in range(0, len(episodes), chunk):
            tasks.append((controller, trial, packed, episodes[start : start + chunk], cfgs))
    if workers == 1:
        results = [run_chunk(task) for task in tasks]
    else:
        with mp.get_context("spawn").Pool(processes=min(workers, len(tasks))) as pool:
            results = list(pool.imap(run_chunk, tasks, chunksize=1))
    ss_order = {ss: i for i, ss in enumerate(dict.fromkeys(e.ss for e in episodes))}
    rows = [row for chunk_rows in results for row in chunk_rows]
    rows.sort(key=lambda r: (int(r["trial"]), ss_order[str(r["ss"])], int(r["index"])))
    return rows


def _p95(values: Sequence[float]) -> float:
    """Return the 95th percentile of the finite values, or NaN if none.

    Args:
        values: Samples, in the caller's units.

    Returns:
        The percentile.
    """
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    array = array[np.isfinite(array)]
    return float("nan") if array.size == 0 else float(np.percentile(array, 95.0))


@dataclass(frozen=True)
class TrialResult:
    """One trial reduced to its log row.

    Attributes:
        row: Flat mapping for the CSV: parameters, per-sea-state success and outcome
            fractions, p95 closing speed, pooled objective and tie-breaker.
        objective: Mean success over the sea states, dimensionless.
        tie_break_m_s: Pooled p95 touchdown closing speed along the deck normal, metres per
            second model scale (NaN if nothing touched down).
    """

    row: dict[str, Any]
    objective: float
    tie_break_m_s: float


def summarise_trial(
    controller: str,
    trial: int,
    params: Mapping[str, float],
    rows: Sequence[Mapping[str, Any]],
    sea_states: Sequence[str],
) -> TrialResult:
    """Reduce one trial's episodes to its log row.

    Args:
        controller: Registry name.
        trial: Trial index.
        params: The trial's parameters.
        rows: The trial's per-episode rows.
        sea_states: Sea-state order.

    Returns:
        The :class:`TrialResult`. Per sea state: ``n``, the six outcome fractions (summing
        to 1), success, p50/p95 closing speed (m/s model), p95 lateral offset (m model),
        p95 relative tilt (degrees), mean time to touchdown (s model), disagreements and
        tunnelling counts.
    """
    out: dict[str, Any] = {"controller": controller, "trial": trial}
    out.update({f"param_{k}": float(v) for k, v in sorted(params.items())})
    successes = []
    for ss in sea_states:
        subset = [r for r in rows if r["ss"] == ss]
        n = len(subset)
        out[f"{ss}_n"] = n
        for outcome in OUTCOMES:
            out[f"{ss}_frac_{outcome}"] = sum(r["outcome"] == outcome for r in subset) / n
        successes.append(out[f"{ss}_frac_success"])
        landed = [r for r in subset if r["touchdown_contact"]]
        closing = [float(r["closing_speed_normal_m_s"]) for r in landed]
        out[f"{ss}_td_closing_p50_m_s"] = (
            float(np.percentile(closing, 50.0)) if closing else float("nan")
        )
        out[f"{ss}_td_closing_p95_m_s"] = _p95(closing)
        out[f"{ss}_td_lateral_p95_m"] = _p95([r["lateral_offset_m"] for r in landed])
        out[f"{ss}_td_rel_tilt_p95_deg"] = _p95([r["rel_tilt_deg"] for r in landed])
        out[f"{ss}_mean_time_to_touchdown_s"] = (
            float(np.mean([float(r["td_t_episode_s"]) for r in landed])) if landed else float("nan")
        )
        out[f"{ss}_disagreement_n"] = sum(bool(r["detectors_disagree"]) for r in subset)
        out[f"{ss}_tunnelling_n"] = sum(bool(r["tunnelled"]) for r in subset)
    objective = float(np.mean(successes))
    tie = _p95([float(r["closing_speed_normal_m_s"]) for r in rows if r["touchdown_contact"]])
    out["mean_success"] = objective
    out["pooled_td_closing_p95_m_s"] = tie
    out["n_episodes"] = len(rows)
    return TrialResult(row=out, objective=objective, tie_break_m_s=tie)


def select_trial(results: Sequence[TrialResult]) -> int:
    """Return the winning trial's index.

    Args:
        results: One result per trial, in trial order.

    Returns:
        The index maximising mean success, ties broken by the lower pooled p95 closing
        speed (NaN sorts last), then by the lower trial index.
    """

    def key(item: tuple[int, TrialResult]) -> tuple[float, float, int]:
        index, result = item
        tie = result.tie_break_m_s if np.isfinite(result.tie_break_m_s) else float("inf")
        return (-result.objective, tie, index)

    return min(enumerate(results), key=key)[0]


def read_tuning_log(path: Path) -> list[dict[str, str]]:
    """Read a committed tuning log, ``results/e01/tuning_<controller>.csv``.

    Args:
        path: The CSV file.

    Returns:
        One dict per trial row, values as strings, in file order.
    """
    import csv

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def select_lowvz(log: Sequence[Mapping[str, Any]]) -> int:
    """Apply the low-closing-speed selection rule to an existing tuning log.

    The rule, fixed by the user on 2026-09-22 before any candidate's frozen-list number was
    seen (``docs/protocol.md`` P3-D3 amendment):

    1. keep the trials whose ``mean_success`` (the objective: mean tune-pool success over
       SS3-SS5, dimensionless) is ``>= best - LOWVZ_SUCCESS_MARGIN``;
    2. among them take the lowest ``pooled_td_closing_p95_m_s`` (pooled p95 touchdown
       closing speed along the deck normal, metres per second model scale; NaN sorts last);
    3. break any remaining tie by the lowest ``trial`` index.

    No trial is run and no episode is drawn: it only reads the log.

    Args:
        log: The log's rows (e.g. from :func:`read_tuning_log`), each with ``trial``,
            ``mean_success`` and ``pooled_td_closing_p95_m_s``.

    Returns:
        The selected trial index.

    Raises:
        ValueError: If the log is empty or has a non-finite ``mean_success``.
    """
    if not log:
        raise ValueError("empty tuning log")
    success = {int(row["trial"]): float(row["mean_success"]) for row in log}
    if not all(np.isfinite(v) for v in success.values()):
        raise ValueError("non-finite mean_success in the tuning log")
    cutoff = max(success.values()) - LOWVZ_SUCCESS_MARGIN

    def key(row: Mapping[str, Any]) -> tuple[float, int]:
        p95 = float(row["pooled_td_closing_p95_m_s"])
        return (p95 if np.isfinite(p95) else float("inf"), int(row["trial"]))

    eligible = [row for row in log if success[int(row["trial"])] >= cutoff]
    return int(min(eligible, key=key)["trial"])
