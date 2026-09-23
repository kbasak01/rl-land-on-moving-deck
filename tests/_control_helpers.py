"""Shared helpers for the ``tests/test_control_*.py`` files.

Not a test module (the leading underscore keeps pytest from collecting it). These live here
so the controller test files do not each re-derive them:

* :func:`synthetic_obs` -- an observation vector built **through the declared layout**, from
  world-frame quantities, with the drone-yaw rotation applied the way the environment
  applies it. Lets a test pin a control law on exact inputs without a physics client.
* :func:`run_episode` -- one episode of a controller in a real environment, returning the
  action stream and the final record, with the privileged context built exactly as the
  evaluation runner will build it.
* :func:`quiescence_windows` and :func:`deck_normal` -- the 40 seeded deck windows on which
  ``oracle_gated`` and ``gated_forecast`` are both checked against ``QuiescenceRule``.

Units: metres, metres per second and seconds model scale; angles radians in the vector.
"""

from pathlib import Path
from typing import Any

import numpy as np

from rld.control.base import Controller, ControlSpec, PrivilegedContext
from rld.control.obs_view import obs_layout
from rld.envs.config import ObservationConfig
from rld.envs.landing_env import DeckLandingAviary, EpisodeRecord


def committed_spec() -> ControlSpec:
    """Return the committed controller-facing configs."""
    return ControlSpec.committed()


def synthetic_obs(
    obs_cfg: ObservationConfig,
    *,
    yaw_rad: float = 0.0,
    velocity_w: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rel_position_w: tuple[float, float, float] = (0.0, 0.0, -0.5),
    deck_velocity_w: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normal_w: tuple[float, float, float] = (0.0, 0.0, 1.0),
    height_m: float = 0.5,
    time_fraction: float = 0.0,
    in_contact: bool = False,
) -> np.ndarray:
    """Build one observation from world-frame quantities, through the layout table.

    Args:
        obs_cfg: The observation config.
        yaw_rad: Drone yaw, radians.
        velocity_w: Drone velocity, m/s model, world.
        rel_position_w: ``p_pad - p_drone``, m model, world.
        deck_velocity_w: Pad velocity, m/s model, world. ``rel_velocity`` is written as
            ``v_pad - v_drone``, the environment's definition.
        normal_w: Unit deck normal, world.
        height_m: Signed clearance, m model.
        time_fraction: ``t / episode_len``, dimensionless.
        in_contact: Contact flag.

    Returns:
        A float32 observation vector.
    """
    layout = obs_layout(obs_cfg)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    to_yaw = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    obs = np.zeros(layout.size, dtype=np.float64)
    obs[layout.slices["attitude"]] = [0.0, 0.0, yaw_rad]
    v = np.asarray(velocity_w, dtype=np.float64)
    obs[layout.slices["velocity"]] = to_yaw @ v
    obs[layout.slices["rel_position"]] = to_yaw @ np.asarray(rel_position_w, dtype=np.float64)
    obs[layout.slices["rel_velocity"]] = to_yaw @ (np.asarray(deck_velocity_w) - v)
    obs[layout.slices["deck_normal"]] = to_yaw @ np.asarray(normal_w, dtype=np.float64)
    obs[layout.slices["height"]] = height_m
    obs[layout.slices["time_fraction"]] = time_fraction
    if "in_contact" in layout.slices:
        obs[layout.slices["in_contact"]] = 1.0 if in_contact else 0.0
    return obs.astype(np.float32)


def feed(
    controller: Controller, obs_list: list[np.ndarray], ctrl_dt_s: float, episode_len_s: float
) -> list[np.ndarray]:
    """Feed synthetic observations with consistent ``time_fraction`` into a controller.

    Args:
        controller: A reset controller.
        obs_list: Observations; their ``time_fraction`` is overwritten to ``k * dt / len``.
        ctrl_dt_s: Control period, s model.
        episode_len_s: Flight budget, s model.

    Returns:
        The actions.
    """
    spec: ControlSpec = controller.spec  # type: ignore[attr-defined]
    layout = obs_layout(spec.observation)
    actions = []
    for k, obs in enumerate(obs_list):
        stamped = obs.copy()
        stamped[layout.slices["time_fraction"]] = k * ctrl_dt_s / episode_len_s
        actions.append(controller.act(stamped))
    return actions


def run_episode(
    env: DeckLandingAviary,
    controller: Controller,
    seed: int,
    *,
    privileged: bool,
) -> tuple[list[np.ndarray], EpisodeRecord, dict[str, Any]]:
    """Run one episode of a controller the way the evaluation runner will.

    Args:
        env: The environment.
        controller: The controller.
        seed: Episode seed.
        privileged: Whether to build and pass a :class:`PrivilegedContext`.

    Returns:
        ``(actions, record, last_info)``.
    """
    obs, info = env.reset(seed=seed)
    context = PrivilegedContext.from_env(env) if privileged else None
    controller.reset(seed, context)
    actions: list[np.ndarray] = []
    max_steps = int(env.cfg.total_len_s * env.cfg.ctrl_freq_hz) + 1
    for _ in range(max_steps):
        action = controller.act(obs)
        actions.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return actions, env.record, info


def assert_actions_in_space(actions: list[np.ndarray]) -> None:
    """Assert every action is a finite float32 ``(3,)`` inside the unit ball and the box.

    Args:
        actions: The action stream.
    """
    assert actions, "no actions"
    for action in actions:
        assert action.shape == (3,)
        assert action.dtype == np.float32
        assert np.all(np.isfinite(action))
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
        assert float(np.linalg.norm(action.astype(np.float64))) <= 1.0 + 1e-6


def fuzz_actions(controller: Controller, obs_cfg: ObservationConfig, n: int, seed: int) -> list:
    """Drive a reset controller with observations sampled uniformly from the obs space.

    ``time_fraction`` is stamped consistently so the controller's clock check passes; every
    other entry is anywhere inside the declared bounds (yaw anywhere in ``[-pi, pi]``,
    velocities up to 10 m/s, heights up to 5 m).

    Args:
        controller: A reset controller.
        obs_cfg: The observation config.
        n: Number of observations.
        seed: Generator seed.

    Returns:
        The actions.
    """
    from rld.envs.observation import observation_space

    space = observation_space(obs_cfg)
    rng = np.random.default_rng(seed)
    obs_list = [
        rng.uniform(space.low.astype(np.float64), space.high.astype(np.float64)).astype(np.float32)
        for _ in range(n)
    ]
    spec: ControlSpec = controller.spec  # type: ignore[attr-defined]
    return feed(controller, obs_list, spec.ctrl_dt_s, spec.landing.episode_len_s)


def selected_trial_params(csv_path: str) -> dict[str, float]:
    """Return the ``param_*`` values of the selected row of a committed tuning log.

    Args:
        csv_path: Path to ``results/e01/tuning_<controller>.csv``.

    Returns:
        Parameter name (without the ``param_`` prefix) to value.
    """
    import csv

    with Path(csv_path).open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["selected"] == "True"]
    assert len(rows) == 1, f"{csv_path}: expected exactly one selected trial, got {len(rows)}"
    return {k[len("param_") :]: float(v) for k, v in rows[0].items() if k.startswith("param_")}


def deck_normal(roll_deg: float, pitch_deg: float) -> tuple[float, float, float]:
    """P1-D2's unit deck normal from dmf-sign roll and pitch, degrees in, world frame."""
    r, p = np.radians(roll_deg), np.radians(pitch_deg)
    return (-np.sin(p) * np.cos(r), -np.sin(r), np.cos(p) * np.cos(r))


def quiescence_windows(
    rng: np.random.Generator, limits: Any, n: int
) -> list[tuple[np.ndarray, ...]]:
    """Deck windows (roll deg, pitch deg, pad v_z m/s model), ``n`` samples each, 40 windows.

    Every value is either well inside its limit (<= 0.9x) or well outside (1.1-2x), so
    float32 rounding in the observation cannot flip a sample. Half the windows are all
    quiet; the rest carry one violation in one channel at a random position. Shared by the
    ``oracle_gated`` and ``gated_forecast`` tests, which must see the **same** windows.

    Args:
        rng: Seeded generator (both tests use ``default_rng(20260922)``).
        limits: :class:`rld.control.quiescence.ModelQuiescenceLimits`.
        n: Samples per window, dimensionless.

    Returns:
        ``[(roll, pitch, v_z), ...]``, each ``(n,)``.
    """
    scale = np.array([limits.roll_deg, limits.pitch_deg, limits.pad_vz_m_s])
    out = []
    for k in range(40):
        values = rng.uniform(-0.9, 0.9, size=(n, 3)) * scale
        if k % 2:
            j, c = int(rng.integers(n)), int(rng.integers(3))
            if k in (1, 3):
                j = 0 if k == 1 else n - 1  # the window's edges
            values[j, c] = rng.choice([-1.0, 1.0]) * rng.uniform(1.1, 2.0) * scale[c]
        out.append((values[:, 0], values[:, 1], values[:, 2]))
    return out
