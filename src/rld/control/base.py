"""The controller contract, the environment facts a controller may know, and the oracle's feed.

The contract
------------
Every controller in this project -- the four classical baselines here and, from Phase 6, the
residual wrapper around ``pid_feedforward`` -- implements :class:`Controller`:

* ``reset(seed, context=None)`` at the start of every episode, **after** ``env.reset``;
* ``act(obs)`` exactly once per control step, starting with the observation ``env.reset``
  returned, and returning a ``(3,)`` float32 action in ``[-1, 1]^3``.

The action is the shared one (P2-D2): a world-frame velocity setpoint, dimensionless,
scaled by ``v_max`` = 1.5 m/s model scale with a **norm** cap. Controllers think in metres
per second and convert through :func:`rld.control.obs_view.setpoint_to_action`, which is the
exact inverse of ``DeckLandingAviary.velocity_setpoint_m_s`` on its range, so a PID setpoint
and a PPO action are scaled by the same rule.

Statelessness across episodes
-----------------------------
A controller's only memory is what ``reset`` clears (integrators, latches, the step
counter). Two episodes run with the same seed through the same controller instance give
bit-identical action streams, whatever ran in between; each controller's test asserts it.

Model time
----------
A controller that needs the current episode time counts its own ``act`` calls since
``reset``: ``t = k * ctrl_dt``, seconds model scale. It does **not** read time from the
observation's ``time_fraction`` block, because that block is float32 and saturates at 1.0
during the 0.5 s dwell grace (P2-D7). It **does** cross-check the counter against
``time_fraction`` while the latter is below 1, and raises if they disagree by more than half
a control step -- the symptom of a runner that skipped a ``reset`` or called ``act`` twice
per step. See :class:`StepClock`.

The privileged feed
-------------------
:class:`PrivilegedContext` is the **true future** deck-point trajectory of the current
episode. The evaluation runner builds it with :meth:`PrivilegedContext.from_env` after
``env.reset`` and passes it **only** to controllers the registry marks ``privileged=True``
(``oracle_gated``). It is rebuilt from the environment's public attributes through
:func:`rld.envs.platform.build_trajectory` on the same physics grid the environment uses,
so it is the trajectory the plate is driven along, bit for bit; ``rld.envs`` is not edited
to provide it.

Units and scales
----------------
Metres, metres per second and seconds are **model** scale throughout (one model second is
five full-scale seconds at ``lam = 1/25``). Angles are degrees at this module's boundary,
in dmf's sign convention (roll positive to starboard, pitch positive bow-up).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from dmf.data.splits import RealizationKey
from dmf.typedefs import FloatArray

from rld.deck.config import SCALING_CONFIG, load_scaling
from rld.deck.scaling import FroudeScale
from rld.envs.config import (
    LANDING_CONFIG,
    OBSERVATION_CONFIG,
    LandingConfig,
    ObservationConfig,
    load_landing,
    load_observation,
)
from rld.envs.platform import build_trajectory

if TYPE_CHECKING:
    from rld.envs.landing_env import DeckLandingAviary

__all__ = [
    "ControlSpec",
    "Controller",
    "DeckWindow",
    "PrivilegedContext",
    "StepClock",
]


@dataclass(frozen=True)
class ControlSpec:
    """What a controller is allowed to know about the environment it flies in.

    Deliberately the committed *configs*, not the environment object: a controller that
    held the environment could read the true deck state and silently become privileged.

    Attributes:
        landing: ``configs/env/landing.yaml``: ``v_max_m_s`` (metres per second model
            scale), the norm cap, ``ctrl_freq_hz`` (hertz model scale) and
            ``episode_len_s`` (seconds model scale).
        observation: ``configs/env/observation.yaml``: which blocks are present, hence the
            layout :func:`rld.envs.observation.obs_fields` returns.
        scale: The Froude scale, from ``configs/deck/scaling.yaml``; used to convert dmf's
            full-scale quiescence thresholds.
    """

    landing: LandingConfig
    observation: ObservationConfig
    scale: FroudeScale

    @classmethod
    def committed(cls) -> "ControlSpec":
        """Load the committed configs.

        Returns:
            The :class:`ControlSpec` built from ``configs/env/landing.yaml``,
            ``configs/env/observation.yaml`` and ``configs/deck/scaling.yaml``.
        """
        return cls(
            landing=load_landing(LANDING_CONFIG),
            observation=load_observation(OBSERVATION_CONFIG),
            scale=load_scaling(SCALING_CONFIG).froude_scale(),
        )

    @property
    def ctrl_dt_s(self) -> float:
        """Control period, seconds model scale (1/30 s)."""
        return self.landing.ctrl_dt_s


@dataclass(frozen=True)
class DeckWindow:
    """Samples of the true deck-point trajectory, for the oracle's commit rule.

    Attributes:
        t_episode_s: Sample times, seconds model scale since the episode start, ``(n,)``.
        roll_deg: Deck roll from the Euler angles, degrees, dmf sign (positive to
            starboard), ``(n,)``.
        pitch_deg: Deck pitch from the Euler angles, degrees, dmf sign (positive bow-up),
            ``(n,)``.
        pad_vz_m_s: Pad deck-point vertical velocity, metres per second model scale, world
            ``z``, ``(n,)``.
        normal: Unit deck normal, dimensionless, world frame, ``(n, 3)``. The oracle
            derives its roll and pitch from this through
            :func:`rld.control.obs_view.deck_angles_deg`, exactly as ``gated`` does from
            the observation, so both rules read the same quantity.
        complete: False when the requested window runs past the end of the episode's
            trajectory; the samples then cover only the part that exists.
    """

    t_episode_s: FloatArray
    roll_deg: FloatArray
    pitch_deg: FloatArray
    pad_vz_m_s: FloatArray
    normal: FloatArray
    complete: bool


def _frozen(array: FloatArray) -> FloatArray:
    """Return a read-only float64 copy.

    Args:
        array: Any array, in the caller's units.

    Returns:
        A copy with ``writeable = False``, so a controller cannot edit the trajectory it
        was handed and change what the next controller sees.
    """
    out = np.array(array, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class PrivilegedContext:
    """The true deck-point trajectory of one episode, read-only.

    **Privileged.** Only a controller the registry marks ``privileged=True`` may receive
    it, and any result computed with it is an upper bound, never a deployable result.

    Attributes:
        t0_model_s: The episode's start offset inside the committed dmf record, seconds
            model scale.
        physics_dt_s: Sample spacing, seconds model scale (1/240 s).
        t_episode_s: Sample times since the episode start, seconds model scale, ``(n,)``.
            Sample ``i`` is the deck state the environment drives the plate to at physics
            substep ``i``.
        position_m: Pad position including ``deck_origin_m`` and with the standing lever
            arm removed, metres model scale, world frame, ``(n, 3)``.
        velocity_m_s: Pad velocity, metres per second model scale, world frame, ``(n, 3)``.
        roll_deg: Deck roll, degrees, dmf sign (positive to starboard), ``(n,)``.
        pitch_deg: Deck pitch, degrees, dmf sign (positive bow-up), ``(n,)``.
        normal: Unit deck normal, dimensionless, world frame, ``(n, 3)``.
        realization_key: The dmf realization key the trajectory came from.
    """

    t0_model_s: float
    physics_dt_s: float
    t_episode_s: FloatArray
    position_m: FloatArray
    velocity_m_s: FloatArray
    roll_deg: FloatArray
    pitch_deg: FloatArray
    normal: FloatArray
    realization_key: RealizationKey

    @classmethod
    def from_env(cls, env: "DeckLandingAviary") -> "PrivilegedContext":
        """Rebuild the current episode's trajectory from the environment's public state.

        Call after ``env.reset``. The time grid is the environment's own --
        ``t0 + arange(n_physics_samples) * physics_dt`` -- and the call is the same
        batched :func:`rld.envs.platform.build_trajectory` over the same 3 001 samples, so
        the result equals the trajectory the plate is driven along bit for bit (a test
        asserts it). Costs one bridge evaluation, ~64 ms.

        Args:
            env: The landing environment, immediately after ``reset``.

        Returns:
            The episode's :class:`PrivilegedContext`.
        """
        cfg = env.cfg
        t0 = float(env.record.t0_model_s)
        grid = np.asarray(
            t0 + np.arange(cfg.n_physics_samples) * cfg.physics_dt_s, dtype=np.float64
        )
        traj = build_trajectory(env.motion, env.pad, cfg.platform, grid, env.pad_offset_m)
        euler = np.asarray(traj.state.euler_xyz_rad, dtype=np.float64)
        return cls(
            t0_model_s=t0,
            physics_dt_s=float(cfg.physics_dt_s),
            t_episode_s=_frozen(
                np.arange(cfg.n_physics_samples, dtype=np.float64) * float(cfg.physics_dt_s)
            ),
            position_m=_frozen(traj.position_m),
            velocity_m_s=_frozen(traj.state.velocity_m_s),
            # P1-D2: phi_world = +radians(roll_deg), theta_world = -radians(pitch_deg).
            roll_deg=_frozen(np.asarray(np.degrees(euler[:, 0]), dtype=np.float64)),
            pitch_deg=_frozen(np.asarray(-np.degrees(euler[:, 1]), dtype=np.float64)),
            normal=_frozen(traj.state.normal),
            realization_key=env.motion.key,
        )

    def __len__(self) -> int:
        """Return the number of samples, dimensionless."""
        return int(self.t_episode_s.size)

    def index_at(self, t_episode_s: float) -> int:
        """Return the first sample at or after an episode time.

        Args:
            t_episode_s: Time since the episode start, seconds model scale.

        Returns:
            The sample index, clamped into ``[0, len(self)]``; ``len(self)`` means the time
            is past the end of the trajectory.
        """
        index = int(np.ceil(t_episode_s / self.physics_dt_s - 1e-9))
        return min(max(index, 0), len(self))

    def window(self, t_start_s: float, n_samples: int, stride: int = 1) -> DeckWindow:
        """Return ``n_samples`` deck samples, ``stride`` physics samples apart, from a time.

        Args:
            t_start_s: Window start, seconds model scale since the episode start. The first
                sample is the first physics sample at or after it.
            n_samples: Number of samples, dimensionless.
            stride: Physics samples between consecutive window samples, dimensionless;
                ``physics_freq / ctrl_freq`` (8) gives control-rate spacing, 1/30 s.

        Returns:
            The :class:`DeckWindow`; ``complete`` is False when the window runs past the end
            of the trajectory.

        Raises:
            ValueError: If ``stride`` or ``n_samples`` is not positive.
        """
        if stride < 1 or n_samples < 1:
            raise ValueError(f"stride and n_samples must be positive, got {stride}, {n_samples}")
        start = self.index_at(t_start_s)
        index = start + int(stride) * np.arange(int(n_samples))
        index = index[index < len(self)]
        return DeckWindow(
            t_episode_s=self.t_episode_s[index],
            roll_deg=self.roll_deg[index],
            pitch_deg=self.pitch_deg[index],
            pad_vz_m_s=self.velocity_m_s[index, 2],
            normal=self.normal[index],
            complete=index.size == int(n_samples),
        )


@dataclass
class StepClock:
    """Episode time for a controller, from its own ``act`` count, cross-checked.

    ``t = k * ctrl_dt`` seconds model scale, where ``k`` is the number of ``act`` calls
    since :meth:`reset`. The observation's ``time_fraction`` is float32 and saturates at 1.0
    during the dwell grace, so it is used only as a consistency check.

    Attributes:
        ctrl_dt_s: Control period, seconds model scale.
        episode_len_s: Flight budget, seconds model scale; ``time_fraction`` is
            ``t / episode_len_s``.
        steps: ``act`` calls since the last reset, dimensionless.
    """

    ctrl_dt_s: float
    episode_len_s: float
    steps: int = field(default=0)

    def reset(self) -> None:
        """Start a new episode at ``t = 0``."""
        self.steps = 0

    def tick(self, time_fraction: float) -> float:
        """Return the time of the observation being acted on, and advance.

        Args:
            time_fraction: The observation's ``time_fraction`` block, dimensionless.

        Returns:
            The episode time of this observation, seconds model scale.

        Raises:
            RuntimeError: If the counter and ``time_fraction`` disagree by more than half a
                control step while ``time_fraction < 1`` -- a skipped ``reset`` or a double
                ``act`` per step.
        """
        t = self.steps * self.ctrl_dt_s
        if time_fraction < 1.0 - 1e-6:
            observed = float(time_fraction) * self.episode_len_s
            if abs(observed - t) > 0.5 * self.ctrl_dt_s:
                raise RuntimeError(
                    f"controller clock at {t:.4f} s but the observation says {observed:.4f} s: "
                    "reset() must follow every env.reset and act() must run once per step"
                )
        self.steps += 1
        return t


@runtime_checkable
class Controller(Protocol):
    """A landing controller in the shared normalised action space.

    Attributes:
        name: Registry name, e.g. ``"pid_feedforward"``.
        privileged: True if the controller reads anything a deployed vehicle could not
            (the true future deck trajectory). Must equal the registry's flag.
    """

    name: str
    privileged: bool

    def reset(self, seed: int, context: PrivilegedContext | None = None) -> None:
        """Clear all per-episode state.

        Args:
            seed: The episode seed. The classical controllers are deterministic and only
                record it; a stochastic controller must derive all randomness from it.
            context: The true deck trajectory. Passed by the runner **only** to
                controllers the registry marks privileged; others must ignore it.
        """
        ...

    def act(self, obs: FloatArray) -> FloatArray:
        """Return the action for one observation.

        Args:
            obs: The observation vector, laid out by
                :func:`rld.envs.observation.obs_fields`.

        Returns:
            A ``(3,)`` float32 action in ``[-1, 1]^3``: a world-frame velocity setpoint
            divided by ``v_max``, with ``||action|| <= 1`` when ``v_max`` is a norm cap.
        """
        ...
