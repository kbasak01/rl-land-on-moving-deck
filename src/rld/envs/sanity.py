"""The e00 environment sanity sweep: what the environment does to policies that are not policies.

What Gate 2 reads
-----------------
Two numbers in ``results/e00_env_sanity.csv`` matter more than the rest:

1. **The outcome-class breakdown.** A ``random`` policy that "succeeds" is an environment
   bug, not a lucky agent, and a success rate printed without its outcome classes hides
   both hovering and crashing. The six fractions are committed per row and sum to 1.
2. **The analytic-versus-contact touchdown disagreement rate**, which Gate 2 requires to be
   below 1 %.

The expected shape of the result is stated in advance so that the run can falsify it: the
``hover`` policy should be essentially all ``timeout`` (it never descends), and the
``random`` policy should be dominated by ``crash`` (a uniformly random velocity setpoint at
1.5 m/s leaves the deck and hits the sea). Anything else is a finding.

Episodes (P2-D4)
----------------
Drawn from the ``id`` split's **val** partition, with the draw seed committed in the CSV.
Those realizations are neither trained on nor part of the Phase 3 frozen evaluation lists,
so this sweep cannot contaminate either. There are 60 val realizations per sea state and
200 episodes per cell, so each realization contributes several episodes at **different start
offsets** -- which is the project's episode definition (CLAUDE.md non-negotiable 2), not a
shortage of data.

Determinism and worker independence
-----------------------------------
Work is split over contiguous chunks of episodes and every result is re-sorted into draw
order before returning, so the CSV does not depend on ``--workers``. Each episode's seed is
a pure function of the draw seed and the episode's index.

Units and scales
----------------
Lengths are metres **model** scale, speeds metres per second model scale, times seconds
model scale, angles degrees.
"""

import hashlib
import multiprocessing as mp
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from dmf.config import SimConfig
from dmf.sim.generate import RealizationSpec
from dmf.typedefs import FloatArray

from rld.deck.bridge import JonswapDeckMotion
from rld.deck.config import MotionConfig, PadConfig, ScalingConfig
from rld.deck.splits import build_all_splits, part_specs
from rld.envs.config import (
    LandingConfig,
    NoiseConfig,
    ObservationConfig,
    RewardConfig,
    SuccessConfig,
)
from rld.envs.landing_env import ACTION_DIM, DeckLandingAviary
from rld.envs.platform import pad_offset_model_m
from rld.envs.touchdown import OUTCOMES

__all__ = [
    "DEFAULT_CHUNK",
    "DEFAULT_DRAW_SEED",
    "DEFAULT_EPISODES",
    "DEFAULT_SEA_STATES",
    "DEFAULT_WORKERS",
    "POLICIES",
    "CellRow",
    "EpisodeRow",
    "EpisodeSpec",
    "PolicyName",
    "SweepConfigs",
    "draw_episodes",
    "rows_as_dicts",
    "run_chunk",
    "run_sweep",
]

#: The two null policies. Neither learns anything; they exist to show what the environment
#: does to an agent that does nothing and to an agent that does anything.
type PolicyName = Literal["hover", "random"]

#: Tuple form of :data:`PolicyName`, in committed row order.
POLICIES: tuple[PolicyName, ...] = ("hover", "random")

#: Sea states swept. SS3 and SS5 bracket the difficulty ladder that P1-D2 measured at the
#: aft pad (v_z p99 0.166 -> 0.575 m/s) without paying for all four.
DEFAULT_SEA_STATES: tuple[str, ...] = ("SS3", "SS5")

#: Episodes per (policy, sea state) cell.
DEFAULT_EPISODES: int = 200

#: The committed episode-draw seed (P2-D4). Recorded in every row of the CSV.
DEFAULT_DRAW_SEED: int = 20260922

#: Episodes per parallel task. Small enough to fill the workers (4 cells x 200 episodes /
#: 25 = 32 tasks), large enough that one environment construction is amortised over many
#: episodes.
DEFAULT_CHUNK: int = 25

#: Default worker count, matching ``rld.deck.stats``.
DEFAULT_WORKERS: int = 24


@dataclass(frozen=True)
class SweepConfigs:
    """Every committed config the sweep needs, bundled so a worker gets one picklable object.

    Attributes:
        sim: dmf's corpus config, full scale.
        scaling: Froude scale.
        pads: Pad geometry; ``radius_model_m`` is metres model scale.
        motion: Model-scale rates and the full-scale forecaster lookback.
        landing: The environment config.
        success: The frozen criteria.
        observation: The observation config.
        noise: The perception stand-in config.
        reward: The reward weights.
    """

    sim: SimConfig
    scaling: ScalingConfig
    pads: PadConfig
    motion: MotionConfig
    landing: LandingConfig
    success: SuccessConfig
    observation: ObservationConfig
    noise: NoiseConfig
    reward: RewardConfig


@dataclass(frozen=True)
class EpisodeSpec:
    """One episode of the sweep: which realization, which policy, which seed.

    Attributes:
        policy: ``"hover"`` or ``"random"``.
        ss: Sea state label, e.g. ``"SS5"``.
        pad: Pad name, ``"aft"`` or ``"cg"``.
        index: Position in the cell's draw order, ``0 .. n_episodes - 1``. The CSV is sorted
            on it, so the output does not depend on the worker count.
        vessel: Vessel name.
        heading_deg: Encounter heading, degrees (180 head seas, 90 beam).
        speed_kn: Forward speed, knots, full scale.
        realization_seed: dmf realization seed ordinal within its cell.
        episode_seed: The environment's episode seed. Fixes the start offset, the initial
            drone state and the noise stream.
    """

    policy: PolicyName
    ss: str
    pad: str
    index: int
    vessel: str
    heading_deg: float
    speed_kn: float
    realization_seed: int
    episode_seed: int

    @property
    def realization(self) -> RealizationSpec:
        """Return the dmf realization coordinates this episode flies against."""
        return RealizationSpec(
            sea_state=self.ss,
            heading_deg=self.heading_deg,
            speed_kn=self.speed_kn,
            vessel=self.vessel,
            seed=self.realization_seed,
        )


@dataclass(frozen=True)
class EpisodeRow:
    """One row of ``results/e00_env_sanity_episodes.csv``: the audit trail.

    Units: lengths metres model scale, speeds metres per second model scale, times seconds
    model scale, angles degrees. ``t0_model_s`` is the episode's start offset inside the
    committed dmf record.
    """

    policy: str
    ss: str
    vessel: str
    heading_deg: float
    speed_kn: float
    realization_seed: int
    pad: str
    index: int
    episode_seed: int
    t0_model_s: float
    init_x_m: float
    init_y_m: float
    init_z_m: float
    outcome: str
    termination_reason: str
    steps: int
    episode_return: float
    control_effort: float
    action_jerk: float
    dwell_s: float
    touchdown_contact: bool
    touchdown_analytic: bool
    detectors_disagree: bool
    td_t_episode_s: float
    td_t_episode_analytic_s: float
    rel_vz_normal_m_s: float
    rel_vz_world_z_m_s: float
    closing_speed_normal_m_s: float
    lateral_offset_m: float
    rel_tilt_deg: float
    abs_tilt_deg: float
    deck_tilt_deg: float
    n_contacts: int
    max_penetration_m: float
    tunnelled: bool
    contact_events: int


@dataclass(frozen=True)
class CellRow:
    """One row of ``results/e00_env_sanity.csv``: one (policy, sea state) cell.

    Units as in :class:`EpisodeRow`. The six ``frac_*`` columns are fractions of
    ``n_episodes`` and sum to exactly 1.
    """

    policy: str
    regime: str
    partition: str
    ss: str
    vessel: str
    headings_deg: str
    speeds_kn: str
    pad: str
    n_episodes: int
    n_realizations: int
    v_max_m_s: float
    driver: str
    episode_draw_seed: int
    frac_crash: float
    frac_off_pad: float
    frac_hard_landing: float
    frac_bounce: float
    frac_success: float
    frac_timeout: float
    n_touchdowns: int
    td_rel_vz_normal_p50_m_s: float
    td_rel_vz_normal_p95_m_s: float
    td_rel_vz_world_z_p50_m_s: float
    td_rel_vz_world_z_p95_m_s: float
    td_lateral_p50_m: float
    td_lateral_p95_m: float
    td_rel_tilt_p95_deg: float
    td_deck_tilt_p95_deg: float
    mean_steps: float
    mean_time_to_touchdown_s: float
    touchdown_disagreement_n: int
    touchdown_disagreement_rate: float
    max_penetration_m: float
    tunnelling_n: int
    steps_per_s: float
    wall_s: float


def draw_episodes(
    sim_cfg: SimConfig,
    policy: PolicyName,
    sea_state: str,
    pad: str,
    n_episodes: int,
    draw_seed: int,
) -> list[EpisodeSpec]:
    """Draw one cell's episodes from the ``id`` split's val partition (P2-D4).

    Args:
        sim_cfg: dmf's corpus config, the grid the split is built from.
        policy: Which null policy this cell runs.
        sea_state: Sea state label, e.g. ``"SS5"``.
        pad: Pad name.
        n_episodes: Episodes to draw.
        draw_seed: The committed draw seed. Mixed with the policy and sea state so that the
            two policies fly the **same** realizations in the same order -- which makes the
            two rows of a sea state a paired comparison rather than two independent draws.

    Returns:
        The episode specs in draw order.

    Raises:
        ValueError: If the val partition has no realization at this sea state.
    """
    split = build_all_splits(sim_cfg)["id"]
    candidates = [spec for spec in part_specs(split, "val") if spec.sea_state == sea_state]
    if not candidates:
        raise ValueError(f"no 'id' val realization at {sea_state}")
    # The realization order and the episode seeds depend on the sea state but NOT on the
    # policy, so `hover` and `random` see identical (realization, start offset, initial
    # state) pairs.
    rng = np.random.default_rng(np.random.SeedSequence([draw_seed, _label_entropy(sea_state)]))
    picks = rng.integers(0, len(candidates), size=n_episodes)
    episode_seeds = rng.integers(0, 2**31 - 1, size=n_episodes)
    return [
        EpisodeSpec(
            policy=policy,
            ss=sea_state,
            pad=pad,
            index=index,
            vessel=candidates[int(pick)].vessel,
            heading_deg=float(candidates[int(pick)].heading_deg),
            speed_kn=float(candidates[int(pick)].speed_kn),
            realization_seed=int(candidates[int(pick)].seed),
            episode_seed=int(seed),
        )
        for index, (pick, seed) in enumerate(zip(picks, episode_seeds, strict=True))
    ]


def _label_entropy(label: str) -> int:
    """Return a stable 32-bit entropy value for a string label.

    ``hash()`` is salted per interpreter process (``PYTHONHASHSEED``), so using it here
    would make the episode draw differ between a single-worker run and a spawned worker,
    and between two runs on the same machine. SHA-256 is stable forever.

    Args:
        label: Any label, e.g. a sea-state name.

    Returns:
        A value in ``[0, 2**32)``.
    """
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "big")


def _action(policy: PolicyName, rng: np.random.Generator) -> FloatArray:
    """Return one action from a null policy.

    Args:
        policy: ``"hover"`` commands zero velocity; ``"random"`` samples the action space
            uniformly.
        rng: The episode's action generator, seeded from the episode seed.

    Returns:
        A ``(3,)`` action, dimensionless, in ``[-1, 1]^3``.
    """
    if policy == "hover":
        return np.zeros(ACTION_DIM, dtype=np.float64)
    return np.asarray(rng.uniform(-1.0, 1.0, size=ACTION_DIM), dtype=np.float64)


def run_chunk(task: tuple[Sequence[EpisodeSpec], SweepConfigs]) -> list[EpisodeRow]:
    """Run one contiguous chunk of episodes in a single environment instance.

    Module-level and taking one packed argument so that ``multiprocessing`` can pickle it.

    Args:
        task: ``(episode specs, configs)``. All specs in a chunk share a policy and a pad;
            their realizations may differ, and the environment is re-pointed between
            episodes rather than rebuilt.

    Returns:
        One :class:`EpisodeRow` per spec, in the order given.
    """
    specs, cfgs = task
    if not specs:
        return []
    scale = cfgs.scaling.froude_scale()
    lookback = cfgs.motion.forecast_lookback_full_s
    env = DeckLandingAviary(
        motion=JonswapDeckMotion(specs[0].realization, cfgs.sim, scale, cfgs.pads, lookback),
        pad=specs[0].pad,
        pad_radius_m=cfgs.pads.radius_model_m,
        pad_offset_m=pad_offset_model_m(specs[0].vessel, cfgs.pads, scale, specs[0].pad),
        cfg=cfgs.landing,
        success=cfgs.success,
        obs_cfg=cfgs.observation,
        noise_cfg=cfgs.noise,
        reward_cfg=cfgs.reward,
        episode_seed=specs[0].episode_seed,
    )
    rows: list[EpisodeRow] = []
    try:
        for spec in specs:
            env.set_motion(
                JonswapDeckMotion(spec.realization, cfgs.sim, scale, cfgs.pads, lookback),
                spec.pad,
                pad_offset_model_m(spec.vessel, cfgs.pads, scale, spec.pad),
            )
            env.reset(seed=spec.episode_seed)
            init = np.asarray(env.INIT_XYZS, dtype=np.float64).reshape(3)
            rng = np.random.default_rng(np.random.SeedSequence([spec.episode_seed, 0xAC7]))
            info: dict[str, Any] = {}
            for _ in range(int(cfgs.landing.total_len_s * cfgs.landing.ctrl_freq_hz) + 1):
                _, _, terminated, truncated, info = env.step(_action(spec.policy, rng))
                if terminated or truncated:
                    break
            rows.append(_episode_row(spec, env, init))
    finally:
        env.close()
    return rows


def _episode_row(spec: EpisodeSpec, env: DeckLandingAviary, init_xyz_m: FloatArray) -> EpisodeRow:
    """Flatten one finished episode into its committed row.

    Args:
        spec: The episode's draw.
        env: The environment, immediately after the episode ended.
        init_xyz_m: The drone's initial position, metres model scale, world frame.

    Returns:
        The :class:`EpisodeRow`. Missing values (no touchdown) are NaN rather than blank, so
        a percentile over the column cannot silently treat "did not land" as zero.
    """
    record = env.record
    contact = record.contact_record
    analytic = record.analytic_record
    nan = float("nan")
    return EpisodeRow(
        policy=spec.policy,
        ss=spec.ss,
        vessel=spec.vessel,
        heading_deg=spec.heading_deg,
        speed_kn=spec.speed_kn,
        realization_seed=spec.realization_seed,
        pad=spec.pad,
        index=spec.index,
        episode_seed=spec.episode_seed,
        t0_model_s=record.t0_model_s,
        init_x_m=float(init_xyz_m[0]),
        init_y_m=float(init_xyz_m[1]),
        init_z_m=float(init_xyz_m[2]),
        outcome=str(record.outcome),
        termination_reason=str(record.termination_reason),
        steps=record.steps,
        episode_return=record.return_,
        control_effort=record.control_effort,
        action_jerk=record.action_jerk,
        dwell_s=record.dwell_s,
        touchdown_contact=contact is not None,
        touchdown_analytic=analytic is not None,
        detectors_disagree=record.detectors_disagree(env.success.detector_disagreement_window_s),
        td_t_episode_s=nan if contact is None else contact.t_episode_s,
        td_t_episode_analytic_s=nan if analytic is None else analytic.t_episode_s,
        rel_vz_normal_m_s=nan if contact is None else contact.rel_vz_normal_m_s,
        rel_vz_world_z_m_s=nan if contact is None else contact.rel_vz_world_z_m_s,
        closing_speed_normal_m_s=nan if contact is None else contact.closing_speed_normal_m_s,
        lateral_offset_m=nan if contact is None else contact.lateral_offset_m,
        rel_tilt_deg=nan if contact is None else contact.rel_tilt_deg,
        abs_tilt_deg=nan if contact is None else contact.abs_tilt_deg,
        deck_tilt_deg=nan if contact is None else contact.deck_tilt_deg,
        n_contacts=0 if contact is None else contact.n_contacts,
        max_penetration_m=record.max_penetration_m,
        tunnelled=record.tunnelled,
        contact_events=record.contact_events,
    )


def _nan_percentile(values: Sequence[float], q: float) -> float:
    """Return a percentile that ignores NaN, or NaN if nothing is left.

    Args:
        values: Samples, in the caller's units.
        q: Percentile in ``[0, 100]``.

    Returns:
        The percentile, or NaN when every sample is NaN (no episode in the cell landed).
    """
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float("nan") if finite.size == 0 else float(np.percentile(finite, q))


def _cell_row(
    policy: PolicyName,
    sea_state: str,
    pad: str,
    rows: Sequence[EpisodeRow],
    cfgs: SweepConfigs,
    draw_seed: int,
    wall_s: float,
) -> CellRow:
    """Reduce one cell's episodes to its committed row.

    Args:
        policy: The cell's policy.
        sea_state: The cell's sea state.
        pad: Pad name.
        rows: The cell's episodes, in draw order.
        cfgs: The committed configs, for ``v_max`` and the driver.
        draw_seed: The committed episode-draw seed.
        wall_s: Wall-clock seconds the cell took.

    Returns:
        The :class:`CellRow`.
    """
    n = len(rows)
    counts = {outcome: sum(row.outcome == outcome for row in rows) for outcome in OUTCOMES}
    landed = [row for row in rows if row.touchdown_contact]
    disagreements = sum(row.detectors_disagree for row in rows)
    steps = sum(row.steps for row in rows)
    return CellRow(
        policy=policy,
        regime="id",
        partition="val",
        ss=sea_state,
        vessel="|".join(sorted({row.vessel for row in rows})),
        headings_deg="|".join(f"{v:g}" for v in sorted({row.heading_deg for row in rows})),
        speeds_kn="|".join(f"{v:g}" for v in sorted({row.speed_kn for row in rows})),
        pad=pad,
        n_episodes=n,
        n_realizations=len({(row.heading_deg, row.speed_kn, row.realization_seed) for row in rows}),
        v_max_m_s=cfgs.landing.v_max_m_s,
        driver=cfgs.landing.platform.driver,
        episode_draw_seed=draw_seed,
        frac_crash=counts["crash"] / n,
        frac_off_pad=counts["off_pad"] / n,
        frac_hard_landing=counts["hard_landing"] / n,
        frac_bounce=counts["bounce"] / n,
        frac_success=counts["success"] / n,
        frac_timeout=counts["timeout"] / n,
        n_touchdowns=len(landed),
        td_rel_vz_normal_p50_m_s=_nan_percentile([r.rel_vz_normal_m_s for r in landed], 50.0),
        td_rel_vz_normal_p95_m_s=_nan_percentile(
            [r.closing_speed_normal_m_s for r in landed], 95.0
        ),
        td_rel_vz_world_z_p50_m_s=_nan_percentile([r.rel_vz_world_z_m_s for r in landed], 50.0),
        td_rel_vz_world_z_p95_m_s=_nan_percentile(
            [max(0.0, -r.rel_vz_world_z_m_s) for r in landed], 95.0
        ),
        td_lateral_p50_m=_nan_percentile([r.lateral_offset_m for r in landed], 50.0),
        td_lateral_p95_m=_nan_percentile([r.lateral_offset_m for r in landed], 95.0),
        td_rel_tilt_p95_deg=_nan_percentile([r.rel_tilt_deg for r in landed], 95.0),
        td_deck_tilt_p95_deg=_nan_percentile([r.deck_tilt_deg for r in landed], 95.0),
        mean_steps=steps / n,
        mean_time_to_touchdown_s=(
            float("nan")
            if not landed
            else float(np.mean([row.td_t_episode_s for row in landed], dtype=np.float64))
        ),
        touchdown_disagreement_n=disagreements,
        touchdown_disagreement_rate=disagreements / n,
        max_penetration_m=min((row.max_penetration_m for row in rows), default=0.0),
        tunnelling_n=sum(row.tunnelled for row in rows),
        steps_per_s=steps / wall_s if wall_s > 0.0 else float("nan"),
        wall_s=wall_s,
    )


def run_sweep(
    cfgs: SweepConfigs,
    *,
    pad: str = "aft",
    policies: Sequence[PolicyName] = POLICIES,
    sea_states: Sequence[str] = DEFAULT_SEA_STATES,
    n_episodes: int = DEFAULT_EPISODES,
    draw_seed: int = DEFAULT_DRAW_SEED,
    workers: int = DEFAULT_WORKERS,
    chunk: int = DEFAULT_CHUNK,
) -> tuple[list[CellRow], list[EpisodeRow]]:
    """Run the whole sweep and return both committed tables.

    Args:
        cfgs: Every committed config the environment needs.
        pad: Pad name, ``"aft"`` (the primary arm) or ``"cg"`` (the control arm).
        policies: Which null policies to run.
        sea_states: Which sea states to run.
        n_episodes: Episodes per cell.
        draw_seed: The committed episode-draw seed (P2-D4).
        workers: Process count. Work splits over chunks of episodes and the output is
            re-sorted into draw order, so it does not depend on this value.
        chunk: Episodes per parallel task.

    Returns:
        ``(cell_rows, episode_rows)``: one row per (policy, sea state), and one per episode.

    Raises:
        ValueError: If ``workers``, ``chunk`` or ``n_episodes`` is not positive.
    """
    if workers < 1 or chunk < 1 or n_episodes < 1:
        raise ValueError(
            f"workers, chunk and n_episodes must be positive, got {workers}, {chunk}, {n_episodes}"
        )
    cells = [(policy, ss) for policy in policies for ss in sea_states]
    draws = {
        (policy, ss): draw_episodes(cfgs.sim, policy, ss, pad, n_episodes, draw_seed)
        for policy, ss in cells
    }
    tasks: list[tuple[Sequence[EpisodeSpec], SweepConfigs]] = []
    for cell in cells:
        specs = draws[cell]
        tasks += [(specs[start : start + chunk], cfgs) for start in range(0, len(specs), chunk)]

    started = time.perf_counter()
    if workers == 1:
        results = [run_chunk(task) for task in tasks]
    else:
        with mp.get_context("spawn").Pool(processes=min(workers, len(tasks))) as pool:
            results = list(pool.imap(run_chunk, tasks, chunksize=1))
    wall_s = time.perf_counter() - started

    episode_rows = [row for chunk_rows in results for row in chunk_rows]
    episode_rows.sort(key=lambda row: (row.policy, row.ss, row.index))
    total_steps = sum(row.steps for row in episode_rows) or 1
    cell_rows = []
    for policy, sea_state in cells:
        subset = [row for row in episode_rows if row.policy == policy and row.ss == sea_state]
        # Wall time is apportioned by step count: the cells ran concurrently, so a per-cell
        # stopwatch would report the whole sweep's span for every cell.
        share = wall_s * sum(row.steps for row in subset) / total_steps
        cell_rows.append(_cell_row(policy, sea_state, pad, subset, cfgs, draw_seed, share))
    return cell_rows, episode_rows


def rows_as_dicts(rows: Iterable[Any]) -> list[dict[str, Any]]:
    """Return frozen result rows as plain dicts, ready for ``csv.DictWriter``.

    Args:
        rows: Frozen dataclass rows.

    Returns:
        One dict per row, field order preserved, values unrounded.
    """
    return [asdict(row) for row in rows]
