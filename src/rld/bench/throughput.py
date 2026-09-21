"""Measure gym-pybullet-drones stepping throughput under vectorised execution.

Phase 0, Gate 0. The number this module produces -- environment steps per second as a
function of worker count -- is what the Phase 5 training budget in ``docs/protocol.md``
P3-D1 is sized from, so it is measured rather than assumed.

Two action types are measured because they cost very different amounts:

- ``rpm``: the four motor speeds, gym-pybullet-drones' default. The cheapest action space
  and the one the plan's wording names.
- ``vel``: a velocity setpoint tracked by ``DSLPIDControl``, which runs an inner loop at
  the physics rate (240 Hz) rather than the control rate (30 Hz). Every method in this
  project shares a velocity-setpoint action space, so this is the row the budget should be
  read from; ``rpm`` is the reference that shows how much of the cost is the PID tracker.

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

import importlib.metadata as md
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

#: Vectorisation backends measured. ``dummy`` runs the envs in-process and is the
#: reference that makes ``subproc`` at one worker readable as IPC overhead rather than as
#: environment cost.
VecClsName = Literal["dummy", "subproc"]

#: Action spaces measured. See the module docstring for why both are recorded.
ActName = Literal["rpm", "vel"]

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


@dataclass(frozen=True)
class ThroughputResult:
    """One measured row of ``results/env_throughput.csv``.

    Attributes:
        vec_cls: Vectorisation backend, ``dummy`` or ``subproc``.
        n_envs: Number of parallel environments.
        act: Action space measured, ``rpm`` or ``vel``.
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


def _env_factory(act: ActName, seed: int) -> Any:
    """Build a picklable zero-argument factory for one ``HoverAviary``.

    Imports happen inside the returned closure so that ``subproc`` workers import
    PyBullet themselves rather than inheriting a connected client across a fork.

    Args:
        act: Action space, ``rpm`` or ``vel``.
        seed: Reset seed for this worker.

    Returns:
        A callable taking no arguments and returning a fresh environment.
    """

    def _init() -> Any:
        from gym_pybullet_drones.envs.HoverAviary import HoverAviary
        from gym_pybullet_drones.utils.enums import ActionType

        env = HoverAviary(
            gui=False,
            record=False,
            pyb_freq=PYB_FREQ_HZ,
            ctrl_freq=CTRL_FREQ_HZ,
            act=ActionType.RPM if act == "rpm" else ActionType.VEL,
        )
        env.reset(seed=seed)
        return env

    return _init


def measure_throughput(
    n_envs: int,
    act: ActName,
    total_steps: int,
    vec_cls: VecClsName,
    warmup_vec_steps: int = WARMUP_VEC_STEPS,
    seed: int = 0,
) -> ThroughputResult:
    """Time ``HoverAviary`` stepping under random actions in DIRECT (headless) mode.

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
    fns = [_env_factory(act, seed + i) for i in range(n_envs)]
    venv = DummyVecEnv(fns) if vec_cls == "dummy" else SubprocVecEnv(fns)
    try:
        space = venv.action_space
        if not isinstance(space, Box):
            raise TypeError(f"expected a Box action space, got {type(space).__name__}")
        rng = np.random.default_rng(seed)
        pool = rng.uniform(
            low=space.low, high=space.high, size=(ACTION_POOL_SIZE, n_envs, *space.shape)
        ).astype(np.float32)

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


def _version(dist: str) -> str:
    """Return an installed distribution's version, or ``"unknown"``.

    Args:
        dist: Distribution name as it appears on PyPI.

    Returns:
        The version string.
    """
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "unknown"


def _submodule_sha(path: Path) -> str:
    """Return the checked-out commit of a git submodule, or ``"unknown"``.

    Args:
        path: Path to the submodule working tree.

    Returns:
        The full 40-character SHA.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def environment_provenance(repo_root: Path) -> dict[str, str]:
    """Collect the conditions a throughput number is only meaningful alongside.

    ``OMP_NUM_THREADS`` is recorded because it is set to 1 by ``.claude/settings.json``:
    with 16 single-threaded workers on a 36-thread host the workers do not contend for
    BLAS threads, and the same measurement on a host that leaves it unset would be slower,
    not faster.

    Args:
        repo_root: Repository root, used to locate the two submodules.

    Returns:
        A mapping of provenance field to string value, all of which become CSV columns.
    """
    return {
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "torch": _version("torch"),
        "sb3": _version("stable-baselines3"),
        "gymnasium": _version("gymnasium"),
        "pybullet": _version("pybullet"),
        "dmf_sha": _submodule_sha(repo_root / "third_party" / "deck-motion-forecast"),
        "gpd_sha": _submodule_sha(repo_root / "third_party" / "gym-pybullet-drones"),
        "drone_model": "cf2x",
        "pyb_freq_hz": str(PYB_FREQ_HZ),
        "ctrl_freq_hz": str(CTRL_FREQ_HZ),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "cpu_count": str(os.cpu_count()),
        "host": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
