"""Measure gym-pybullet-drones stepping throughput under vectorised execution.

Phase 0, Gate 0. The number this module produces -- environment steps per second as a
function of worker count -- is what the Phase 5 training budget in ``docs/protocol.md``
P3-D1 is sized from, so it is measured rather than assumed.

Two action types are measured because they cost very different amounts:

- ``rpm``: the four motor speeds, gym-pybullet-drones' default. The cheapest action space
  and the one the plan's wording names.
- ``vel``: a velocity setpoint tracked by ``DSLPIDControl``. Every method in this project
  shares a velocity-setpoint action space, so this is the row the budget should be read
  from; ``rpm`` is the reference that shows how much of the cost is the PID tracker.

  **Correction, recorded rather than rewritten (P2-D9).** The original text of this
  docstring said the ``vel`` path runs ``DSLPIDControl`` "at the physics rate (240 Hz)
  rather than the control rate (30 Hz)", and P0-D2's reasoning quotes that sentence. It is
  wrong: ``BaseRLAviary._preprocessAction`` calls ``computeControl`` **once per control
  step** (30 Hz) and holds the resulting RPMs across the eight physics substeps.
  ``rld.envs.landing_env.DeckLandingAviary`` does the same. The measured numbers in
  ``results/env_throughput.csv`` stand; only the explanation was wrong.

Two environments are measured:

- ``hover``: ``HoverAviary``, the Phase 0 measurement. A **ceiling**, not an estimate: it
  has one body, no deck, no constraint and no motion bridge.
- ``landing``: ``rld.envs.landing_env.DeckLandingAviary``, the Phase 2 environment, with the
  deck plate driven every physics step and the deck-motion bridge evaluated once per reset.
  This is the row P3-D1's training budget must be sized from (plan Phase 2 note (e)). It is
  written to a **separate** CSV so that the committed Gate 0 artifact's shape is preserved.

Both are stepped with **random** actions, which is what makes the two rows comparable
only with care: a random RPM command tumbles the Crazyflie out of bounds in a handful of
steps, so that row spends most of its time in ``reset`` reloading the drone URDF, while a
random velocity setpoint is tracked and the episode runs on. The ``resets`` and
``steps_per_episode`` columns are recorded so that this shows in the CSV instead of being
mistaken for the PID tracker being free. The Phase 5 budget should be read from the ``vel``
rows, whose episode lengths are closer to what a landing episode will be.

All timing is wall clock over a fixed number of vector steps after a discarded warmup.
Times are seconds, rates are steps per second, and *step* always means one environment
step (one ``ctrl_freq`` tick of one drone), never one vector step: a 16-worker vector step
is 16 steps.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rld.provenance import environment_provenance as _environment_provenance

#: Vectorisation backends measured. ``dummy`` runs the envs in-process and is the
#: reference that makes ``subproc`` at one worker readable as IPC overhead rather than as
#: environment cost.
VecClsName = Literal["dummy", "subproc"]

#: Action spaces measured. See the module docstring for why both are recorded. The
#: ``landing`` environment has exactly one action space, so it ignores this.
ActName = Literal["rpm", "vel"]

#: Environments measured. ``hover`` is Phase 0's ceiling; ``landing`` is the Phase 2
#: environment with the deck body and the motion bridge in the loop.
EnvName = Literal["hover", "landing"]

#: Action stream used during the timed region.
#:
#: ``random`` is Phase 0's contract and is kept so the rows stay comparable. ``hold`` sends
#: a constant zero action, which in the landing environment is "hover" -- the episode then
#: runs to the 12 s time limit instead of tumbling out in ~1.5 s. The two bracket the truth
#: for a training run: a policy early in training resets often, a competent one rarely, and
#: the landing environment's reset costs ~90 ms (most of it the analytic deck-motion
#: evaluation), so the two rows differ by far more than the per-step cost.
PolicyName = Literal["random", "hold"]

#: Worker counts required by the Phase 0 task.
DEFAULT_WORKERS: tuple[int, ...] = (1, 8, 16)

#: Vector steps discarded before timing starts, per row. Covers PyBullet's first-contact
#: allocations and, for ``subproc``, the worker handshake.
WARMUP_VEC_STEPS: int = 50

#: Pre-generated random action batches, cycled during the timed region so that action
#: sampling never enters the measurement.
ACTION_POOL_SIZE: int = 512

#: Budget horizon reported alongside each rate, in environment steps. The plan's default
#: PPO budget (section 5, Phase 5).
BUDGET_STEPS: int = 10_000_000

#: Physics and control rates, hertz. Pre-registered in the plan (Phase 2) and fixed here
#: so the Phase 0 number is measured at the rates the project will actually train at.
PYB_FREQ_HZ: int = 240
CTRL_FREQ_HZ: int = 30

#: Grid cell the ``landing`` rows are measured at. The frigate's worst aft-pad cell in
#: P1-D2's table, so the number is taken at the difficulty training will face.
LANDING_BENCH_SEA_STATE: str = "SS5"

#: Pad the ``landing`` rows are measured at. ``aft`` is the primary arm.
LANDING_BENCH_PAD: str = "aft"


@dataclass(frozen=True)
class ThroughputResult:
    """One measured row of ``results/env_throughput.csv``.

    Attributes:
        env: Environment measured, ``hover`` or ``landing``.
        policy: Action stream, ``random`` or ``hold``. See :data:`PolicyName`.
        vec_cls: Vectorisation backend, ``dummy`` or ``subproc``.
        n_envs: Number of parallel environments.
        act: Action space measured, ``rpm`` or ``vel``. Always ``vel`` for ``landing``,
            which has one shared action space by construction.
        total_steps: Timed environment steps, ``n_vec_steps * n_envs``.
        warmup_steps: Environment steps executed and discarded before timing.
        wall_s: Wall-clock seconds spanning the timed steps.
        steps_per_s: ``total_steps / wall_s``, environment steps per second.
        steps_per_s_per_env: ``steps_per_s / n_envs``, the per-worker rate. It falls as
            workers are added; how fast it falls is the scaling story.
        est_hours_budget: Hours to reach :data:`BUDGET_STEPS` at this rate, excluding all
            policy-update time.
        resets: Episode boundaries inside the timed region. A vectorised env auto-resets,
            and a reset reloads the drone URDF, which costs far more than a step.
        steps_per_episode: ``total_steps / resets``, the mean episode length in steps.
            Recorded because it is the confounder: under a *random* policy the two action
            spaces crash at wildly different rates, so a bare steps/s comparison between
            them measures episode length, not per-step cost. Read the rates within an
            action space, and read across action spaces only alongside this column.
    """

    env: EnvName
    policy: PolicyName
    vec_cls: VecClsName
    n_envs: int
    act: ActName
    total_steps: int
    warmup_steps: int
    wall_s: float
    steps_per_s: float
    steps_per_s_per_env: float
    est_hours_budget: float
    resets: int
    steps_per_episode: float


def _env_factory(env: EnvName, act: ActName, seed: int) -> Any:
    """Build a picklable zero-argument factory for one environment.

    Imports happen inside the returned closure so that ``subproc`` workers import
    PyBullet themselves rather than inheriting a connected client across a fork.

    Args:
        env: ``hover`` or ``landing``.
        act: Action space, ``rpm`` or ``vel``. Ignored by ``landing``.
        seed: Reset seed for this worker.

    Returns:
        A callable taking no arguments and returning a fresh environment.
    """

    def _hover() -> Any:
        from gym_pybullet_drones.envs.HoverAviary import HoverAviary
        from gym_pybullet_drones.utils.enums import ActionType

        aviary = HoverAviary(
            gui=False,
            record=False,
            pyb_freq=PYB_FREQ_HZ,
            ctrl_freq=CTRL_FREQ_HZ,
            act=ActionType.RPM if act == "rpm" else ActionType.VEL,
        )
        aviary.reset(seed=seed)
        return aviary

    def _landing() -> Any:
        from dmf.config import load_sim
        from dmf.sim.generate import RealizationSpec

        from rld.deck.bridge import JonswapDeckMotion
        from rld.deck.config import (
            MOTION_JONSWAP_CONFIG,
            PAD_CONFIG,
            SCALING_CONFIG,
            load_motion,
            load_pads,
            load_scaling,
        )
        from rld.envs.config import (
            LANDING_CONFIG,
            NOISE_CONFIG,
            OBSERVATION_CONFIG,
            REWARD_CONFIG,
            SUCCESS_CONFIG,
            load_landing,
            load_noise,
            load_observation,
            load_reward,
            load_success,
        )
        from rld.envs.landing_env import DeckLandingAviary
        from rld.envs.platform import pad_offset_model_m

        motion_cfg = load_motion(MOTION_JONSWAP_CONFIG)
        pads = load_pads(PAD_CONFIG)
        scale = load_scaling(SCALING_CONFIG).froude_scale()
        # The measured cell is the one P1-D2 reports the worst frigate aft-pad v_z p99 for,
        # so the throughput number is taken at the difficulty the training budget will face,
        # not at flat water.
        source = JonswapDeckMotion(
            RealizationSpec(
                sea_state=LANDING_BENCH_SEA_STATE,
                heading_deg=180.0,
                speed_kn=12.0,
                vessel="frigate",
                seed=seed % 40,
            ),
            load_sim(motion_cfg.sim_config_path),
            scale,
            pads,
            lookback_full_s=motion_cfg.forecast_lookback_full_s,
        )
        aviary = DeckLandingAviary(
            motion=source,
            pad=LANDING_BENCH_PAD,
            pad_radius_m=pads.radius_model_m,
            pad_offset_m=pad_offset_model_m("frigate", pads, scale, LANDING_BENCH_PAD),
            cfg=load_landing(LANDING_CONFIG),
            success=load_success(SUCCESS_CONFIG),
            obs_cfg=load_observation(OBSERVATION_CONFIG),
            noise_cfg=load_noise(NOISE_CONFIG),
            reward_cfg=load_reward(REWARD_CONFIG),
            episode_seed=seed,
        )
        aviary.reset(seed=seed)
        return aviary

    return _hover if env == "hover" else _landing


def measure_throughput(
    n_envs: int,
    act: ActName,
    total_steps: int,
    vec_cls: VecClsName,
    warmup_vec_steps: int = WARMUP_VEC_STEPS,
    seed: int = 0,
    env: EnvName = "hover",
    policy: PolicyName = "random",
) -> ThroughputResult:
    """Time one environment's stepping under random actions in DIRECT (headless) mode.

    The vectorised environment auto-resets on termination, so no episode bookkeeping
    enters the timed region. Actions are drawn once into a pool before timing and cycled,
    so the measured cost is stepping and inter-process transport only.

    Args:
        n_envs: Number of parallel environments.
        act: Action space, ``rpm`` or ``vel``.
        total_steps: Target number of timed environment steps. Rounded up to a whole
            number of vector steps.
        vec_cls: Vectorisation backend, ``dummy`` (in-process) or ``subproc``.
        warmup_vec_steps: Vector steps executed and discarded before timing.
        seed: Base seed; worker ``i`` is reset with ``seed + i``.
        env: Environment to measure, ``hover`` (Phase 0's ceiling) or ``landing`` (the
            Phase 2 environment, with the deck body and the motion bridge in the loop).
        policy: Action stream during the timed region, ``random`` or ``hold``.

    Returns:
        The measured row.

    Raises:
        ValueError: If ``n_envs`` or ``total_steps`` is not positive.
        TypeError: If the environment's action space is not a ``Box``.
    """
    if n_envs <= 0:
        raise ValueError(f"n_envs must be positive, got {n_envs}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    from gymnasium.spaces import Box
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    n_vec_steps = -(-total_steps // n_envs)
    fns = [_env_factory(env, act, seed + i) for i in range(n_envs)]
    venv = DummyVecEnv(fns) if vec_cls == "dummy" else SubprocVecEnv(fns)
    try:
        space = venv.action_space
        if not isinstance(space, Box):
            raise TypeError(f"expected a Box action space, got {type(space).__name__}")
        rng = np.random.default_rng(seed)
        if policy == "random":
            pool = rng.uniform(
                low=space.low, high=space.high, size=(ACTION_POOL_SIZE, n_envs, *space.shape)
            ).astype(np.float32)
        else:
            pool = np.zeros((ACTION_POOL_SIZE, n_envs, *space.shape), dtype=np.float32)

        venv.reset()
        for i in range(warmup_vec_steps):
            venv.step(pool[i % ACTION_POOL_SIZE])

        resets = 0
        t0 = time.perf_counter()
        for i in range(n_vec_steps):
            _, _, dones, _ = venv.step(pool[i % ACTION_POOL_SIZE])
            resets += int(np.count_nonzero(dones))
        wall_s = time.perf_counter() - t0
    finally:
        venv.close()

    steps = n_vec_steps * n_envs
    rate = steps / wall_s
    return ThroughputResult(
        env=env,
        policy=policy,
        vec_cls=vec_cls,
        n_envs=n_envs,
        act=act,
        total_steps=steps,
        warmup_steps=warmup_vec_steps * n_envs,
        wall_s=wall_s,
        steps_per_s=rate,
        steps_per_s_per_env=rate / n_envs,
        est_hours_budget=BUDGET_STEPS / rate / 3600.0,
        resets=resets,
        steps_per_episode=steps / resets if resets else float("inf"),
    )


def environment_provenance(repo_root: Path) -> dict[str, str]:
    """Return the provenance block for one throughput row.

    The generic environment block comes from :func:`rld.provenance.environment_provenance`;
    this adds the three fields that are specific to this measurement, in the position they
    already occupy in ``results/env_throughput.csv``.

    Args:
        repo_root: Repository root, used to locate the two submodules.

    Returns:
        A mapping of provenance field to string value. ``pyb_freq_hz`` and ``ctrl_freq_hz``
        are hertz at model scale; the rest are versions, identifiers and a UTC timestamp.
    """
    return _environment_provenance(
        repo_root,
        extra={
            "drone_model": "cf2x",
            "pyb_freq_hz": str(PYB_FREQ_HZ),
            "ctrl_freq_hz": str(CTRL_FREQ_HZ),
        },
    )
