"""``DeckLandingAviary``: the environment every method in this project is judged inside.

Why ``BaseAviary`` and not ``BaseRLAviary``
-------------------------------------------
``BaseRLAviary`` hard-codes four things this task cannot use: ``SPEED_LIMIT`` = 0.25 m/s
(below the deck's own aft-pad ``v_z`` p99 from SS3 up -- P1-D1), a 0.5 s action buffer, an
``ActionType`` enum, and ``(NUM_DRONES, N)``-shaped 2-D spaces. The 2-D spaces in particular
are what would fight ``check_env`` and SB3 for a single-drone task. So this class derives
from ``BaseAviary`` and declares its own flat spaces.

The shared action space
-----------------------
``Box(-1, 1, (3,), float32)``: a world-frame velocity setpoint, scaled by ``v_max`` (P2-D2),
yaw held at 0, tracked by ``DSLPIDControl``. **Every** method emits exactly this -- PID
baselines, PPO, SAC, residual, forecast-conditioned -- which is what makes the comparison
fair. Residual composition (``base + alpha * pi(o)``) belongs in a Phase 6 wrapper, never in
here.

``v_max`` is a cap on the **speed**, not on each axis: a commanded vector longer than
``v_max`` is rescaled rather than clipped per component, because the literal box would allow
``sqrt(3) * v_max`` on a diagonal and a diagonal descent would be 73 % faster than a vertical
one for no physical reason.

Yaw is commanded to 0, not to the current yaw. Upstream's ``ActionType.VEL`` passes
``target_rpy = [0, 0, state[9]]``, which re-targets whatever yaw the attitude loop has
drifted to instead of correcting it; a test asserts yaw stays near zero over a full random
episode.

Two things about ``BaseAviary`` that shape this file
----------------------------------------------------
1. ``__init__`` calls ``_actionSpace()``, ``_observationSpace()`` **and** ``_housekeeping()``
   (hence ``_addObstacles()``) at its very end, so every attribute those read must be
   assigned **before** ``super().__init__()``.
2. ``step()`` runs ``PYB_STEPS_PER_CTRL`` physics substeps with no per-substep hook. The
   deck must advance and contacts must be polled at 240 Hz, not 30 Hz, so ``step()`` is
   overridden: it mirrors the base loop and adds, per substep, *advance the deck ->
   ``stepSimulation`` -> poll both detectors*. ``_computeReward``/``_computeTerminated``/
   ``_computeTruncated``/``_computeInfo`` then become pure reads of state recorded inside
   that loop, which keeps the base class's call order intact.

Termination versus truncation
-----------------------------
``timeout`` is ``truncated=True`` with no terminal penalty. Everything else -- ``success``,
``hard_landing``, ``off_pad``, ``bounce``, ``crash`` -- is ``terminated=True``. Getting this
backwards teaches PPO to hover out the clock, and there is a test for it.

Units and scales
----------------
Metres, metres per second and seconds are **model** scale throughout; angles are radians
inside and degrees at the boundary. The world frame is P1-D2's: ``x`` = bow, ``y`` = port,
``z`` = up. One model second is five full-scale seconds at ``lam = 1/25``.
"""

import time
from importlib.resources import files
from typing import Any

import numpy as np
import pybullet as pyb
import pybullet_data  # type: ignore[import-untyped]
from dmf.typedefs import FloatArray
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gymnasium.spaces import Box

from rld.envs.config import (
    LandingConfig,
    NoiseConfig,
    ObservationConfig,
    RewardConfig,
    SuccessConfig,
)
from rld.envs.noise import PerceptionNoise
from rld.envs.observation import OBS_DTYPE, build_observation, observation_space
from rld.envs.platform import (
    DeckPlatform,
    EpisodeMotionSource,
    LazyDeckTrajectory,
    PlatformSample,
)
from rld.envs.reward import shaping_reward, terminal_reward
from rld.envs.touchdown import (
    DroneCollisionGeometry,
    Outcome,
    TerminationReason,
    TouchdownRecord,
    analytic_clearance_m,
    analytic_touchdown,
    classify,
    contact_touchdown,
    poll_contacts,
)

__all__ = ["ACTION_DIM", "EPISODE_SALT", "DeckLandingAviary", "EpisodeRecord"]

#: Dimension of the shared action space.
ACTION_DIM: int = 3

#: Mixed into every episode seed sequence so that an episode seed ``s`` in this environment
#: and the same integer used anywhere else in the project (a dmf realization seed, a
#: training seed) cannot produce correlated draws.
EPISODE_SALT: int = 0x1A2D1E


class EpisodeRecord:
    """Everything one episode produced, for the sanity sweep and the Phase 7 evaluation.

    Populated as the episode runs and read once it ends. Not a frozen dataclass because it
    is written to incrementally; :meth:`as_row` is the frozen view.

    Attributes:
        outcome: The classified outcome, or ``None`` while the episode is still running.
        termination_reason: Free-text reason (P2-D5), or ``None`` while running.
        contact_record: The contact detector's first-contact record, or ``None``.
        analytic_record: The analytic detector's first-contact record, or ``None``.
        dwell_s: Time in contact since first contact, seconds model scale.
        max_penetration_m: Deepest (most negative) separation seen at any contact, metres
            model scale.
        tunnelled: Whether penetration ever exceeded ``tunnelling_penetration_m``.
        contact_events: Distinct touchdown events in the episode. One clean touchdown is
            exactly 1; a bounce that re-lands is 2 or more. A rising edge only counts as a
            new event if contact was lost for longer than ``contact_loss_grace_s`` -- Bullet
            drops and re-adds contact points on a settling body, and counting that flicker
            would make "exactly one touchdown" unmeasurable. Kept separate from
            ``contact_record`` (which latches the *first* event) so the claim is a
            measurement rather than a property of the bookkeeping.
        steps: Control steps executed.
        t0_model_s: Episode start offset inside the dmf record, seconds model scale.
        control_effort: Sum of ``||a_t||`` over the episode, dimensionless.
        action_jerk: Sum of ``||a_t - a_{t-1}||`` over the episode, dimensionless.
        return_: Undiscounted sum of rewards, dimensionless.
    """

    def __init__(self, t0_model_s: float) -> None:
        """Start an empty record.

        Args:
            t0_model_s: Episode start offset inside the dmf record, seconds model scale.
        """
        self.outcome: Outcome | None = None
        self.termination_reason: TerminationReason | None = None
        self.contact_record: TouchdownRecord | None = None
        self.analytic_record: TouchdownRecord | None = None
        self.dwell_s: float = 0.0
        self.max_penetration_m: float = 0.0
        self.tunnelled: bool = False
        self.contact_events: int = 0
        self.steps: int = 0
        self.t0_model_s: float = float(t0_model_s)
        self.control_effort: float = 0.0
        self.action_jerk: float = 0.0
        self.return_: float = 0.0

    def detectors_disagree(self, window_s: float) -> bool:
        """Report whether the two touchdown detectors disagree on this episode.

        Args:
            window_s: Tolerance, seconds model scale. One control step at 30 Hz.

        Returns:
            True if exactly one detector fired, or if both fired more than ``window_s``
            apart.
        """
        contact, analytic = self.contact_record, self.analytic_record
        if (contact is None) != (analytic is None):
            return True
        if contact is None or analytic is None:
            return False
        return abs(contact.t_model_s - analytic.t_model_s) > window_s

    def as_row(self) -> dict[str, float | int | str | bool | None]:
        """Return the episode as one flat CSV row.

        Returns:
            A mapping of column to value. Times are seconds model scale, lengths metres
            model scale, speeds metres per second model scale, angles degrees.
        """
        contact, analytic = self.contact_record, self.analytic_record
        return {
            "outcome": self.outcome,
            "termination_reason": self.termination_reason,
            "t0_model_s": self.t0_model_s,
            "steps": self.steps,
            "return": self.return_,
            "control_effort": self.control_effort,
            "action_jerk": self.action_jerk,
            "dwell_s": self.dwell_s,
            "max_penetration_m": self.max_penetration_m,
            "tunnelled": self.tunnelled,
            "contact_events": self.contact_events,
            "touchdown_contact": contact is not None,
            "touchdown_analytic": analytic is not None,
            "td_t_episode_s": None if contact is None else contact.t_episode_s,
            "td_t_episode_analytic_s": None if analytic is None else analytic.t_episode_s,
            "rel_vz_normal_m_s": None if contact is None else contact.rel_vz_normal_m_s,
            "rel_vz_world_z_m_s": None if contact is None else contact.rel_vz_world_z_m_s,
            "closing_speed_normal_m_s": None
            if contact is None
            else contact.closing_speed_normal_m_s,
            "lateral_offset_m": None if contact is None else contact.lateral_offset_m,
            "rel_tilt_deg": None if contact is None else contact.rel_tilt_deg,
            "abs_tilt_deg": None if contact is None else contact.abs_tilt_deg,
            "deck_tilt_deg": None if contact is None else contact.deck_tilt_deg,
            "n_contacts": None if contact is None else contact.n_contacts,
        }


class DeckLandingAviary(BaseAviary):  # type: ignore[misc]
    """A Crazyflie landing on a moving ship deck.

    ``BaseAviary`` is untyped (``gym_pybullet_drones`` is in mypy's
    ``ignore_missing_imports`` list), so every override below is annotated explicitly and
    every PyBullet return is narrowed at the boundary. Nothing untyped escapes this class.

    Attributes:
        cfg: Rates, episode geometry, the action space and the plate.
        success: The frozen success criteria.
        motion: The deck-motion source this environment flies against.
        pad: Pad name, ``"aft"`` or ``"cg"``.
        record: The current episode's :class:`EpisodeRecord`.
    """

    metadata: dict[str, list[str]] = {"render_modes": []}

    #: Physics substeps executed since the last reset. Declared here because ``BaseAviary``
    #: is untyped, so mypy cannot infer the attribute's type from the base class alone.
    step_counter: int

    def __init__(
        self,
        *,
        motion: EpisodeMotionSource,
        pad: str,
        pad_radius_m: float,
        pad_offset_m: tuple[float, float, float],
        cfg: LandingConfig,
        success: SuccessConfig,
        obs_cfg: ObservationConfig,
        noise_cfg: NoiseConfig,
        reward_cfg: RewardConfig,
        episode_seed: int = 0,
        gui: bool = False,
        visual_shapes: bool | None = None,
    ) -> None:
        """Build the environment. Every attribute the base class reads is set first.

        Args:
            motion: Deck-motion source (JONSWAP, sinusoid, or the static test double).
            pad: Pad name from ``configs/deck/pad.yaml``.
            pad_radius_m: Painted-pad radius, metres model scale.
            pad_offset_m: The pad's standing lever arm from the ship's mean-position CG,
                metres model scale, from
                :func:`rld.envs.platform.pad_offset_model_m`. It is removed so the plate
                oscillates about ``cfg.platform.deck_origin_m``; see that function for why
                leaving it in would put the frigate's and the s175's plates in different
                places. Pass ``(0, 0, 0)`` for a zero-lever-arm source such as
                :class:`~rld.envs.platform.StaticDeckMotion`.
            cfg: ``configs/env/landing.yaml``.
            success: ``configs/env/success.yaml``, the frozen criteria.
            obs_cfg: ``configs/env/observation.yaml``.
            noise_cfg: ``configs/env/noise.yaml``.
            reward_cfg: ``configs/env/reward.yaml``.
            episode_seed: Base entropy for the first episode. A ``reset(seed=s)`` overrides
                it; a ``reset()`` with no seed advances an internal counter, so repeated
                resets give different episodes deterministically.
            gui: Open PyBullet's GUI. DIRECT for training and for every committed number;
                GUI only for rendering a GIF.
            visual_shapes: Load the drone's visual mesh. ``None`` (the default) means
                ``gui``. Skipping it in DIRECT mode saves ~21 ms per reset (P5-D5) and
                changes no physics: visual shapes are not in the dynamics world. Pass
                ``True`` to render offscreen from a DIRECT client.
        """
        self.cfg = cfg
        self.success = success
        self.motion = motion
        self.pad = pad
        self.pad_offset_m = pad_offset_m
        self._obs_cfg = obs_cfg
        self._noise_cfg = noise_cfg
        self._reward_cfg = reward_cfg
        self._visual_shapes = bool(gui) if visual_shapes is None else bool(visual_shapes)
        self._base_seed = int(episode_seed)
        self._episode_index = 0
        self._platform = DeckPlatform(cfg.platform, pad_radius_m)
        self._geometry = DroneCollisionGeometry(height_m=0.0, radius_m=0.0, z_offset_m=0.0)
        self._trajectory: LazyDeckTrajectory
        self._substep = 0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self._prev_distance_m = 0.0
        self._step_reward = 0.0
        self._in_contact = False
        self._contact_lost = False
        self._last_contact_s = 0.0
        self._crash = False
        self._reason: TerminationReason | None = None
        self._terminated = False
        self._truncated = False
        self._noise: PerceptionNoise
        self.record: EpisodeRecord

        self._prepare_episode(self._base_seed)

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=self._init_xyzs,
            initial_rpys=np.zeros((1, 3)),
            physics=Physics.PYB,
            pyb_freq=cfg.physics_freq_hz,
            ctrl_freq=cfg.ctrl_freq_hz,
            gui=gui,
            record=False,
            obstacles=True,
            user_debug_gui=False,
        )
        self._geometry = DroneCollisionGeometry(
            height_m=float(self.COLLISION_H),
            radius_m=float(self.COLLISION_R),
            z_offset_m=float(self.COLLISION_Z_OFFSET),
        )
        self._ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self._begin_episode()

    # ---------------------------------------------------------------- spaces

    def _actionSpace(self) -> Box:  # noqa: N802
        """Return the shared action space.

        Returns:
            ``Box(-1, 1, (3,), float32)``: a world-frame velocity setpoint, dimensionless,
            scaled to ``cfg.v_max_m_s`` metres per second model scale at use.
        """
        return Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=OBS_DTYPE)

    def _observationSpace(self) -> Box:  # noqa: N802
        """Return the observation space declared by :mod:`rld.envs.observation`.

        Returns:
            A flat float32 ``Box`` with finite bounds.
        """
        return observation_space(self._obs_cfg)

    # ---------------------------------------------------------------- episode setup

    def _prepare_episode(self, entropy: int) -> None:
        """Draw one episode's start offset, initial state and noise stream.

        Three independent children are spawned from one sequence, so that changing what one
        of them consumes -- turning noise on, say -- cannot shift the others. That is what
        keeps the Phase 7 noise ablation a paired comparison over identical episodes.

        Args:
            entropy: The episode's seed.
        """
        seq = np.random.SeedSequence([int(entropy), EPISODE_SALT])
        start_seq, init_seq, noise_seq = seq.spawn(3)
        rng_start = np.random.default_rng(start_seq)
        rng_init = np.random.default_rng(init_seq)

        lo, hi = self.motion.episode_start_window_s(self.cfg.total_len_s)
        self._t0_model_s = float(rng_start.uniform(lo, hi))

        init = self.cfg.init
        height = float(rng_init.uniform(*init.height_above_deck_m))
        lateral = rng_init.uniform(-init.lateral_m, init.lateral_m, size=2)
        origin = np.asarray(self.cfg.platform.deck_origin_m, dtype=np.float64)
        self._init_xyzs = np.array(
            [[origin[0] + lateral[0], origin[1] + lateral[1], origin[2] + height]],
            dtype=np.float64,
        )

        grid = np.asarray(
            self._t0_model_s + np.arange(self.cfg.n_physics_samples) * self.cfg.physics_dt_s,
            dtype=np.float64,
        )
        # Lazy (P5-D5): only the chunk holding sample 0 is synthesised here, when the plate
        # is spawned; later chunks as the episode reaches them. Every sample is bit-identical
        # to the eager full-grid `build_trajectory` on this same grid.
        self._trajectory = LazyDeckTrajectory(
            self.motion, self.pad, self.cfg.platform, grid, self.pad_offset_m
        )
        self._noise = PerceptionNoise(
            self._noise_cfg, np.random.default_rng(noise_seq), float(self.cfg.ctrl_freq_hz)
        )

    def _begin_episode(self) -> None:
        """Zero every per-episode accumulator. Called after the simulation is rebuilt."""
        self._substep = 0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self._step_reward = 0.0
        self._in_contact = False
        self._contact_lost = False
        self._last_contact_s = 0.0
        self._crash = False
        self._reason = None
        self._terminated = False
        self._truncated = False
        self._noise.reset()
        self.record = EpisodeRecord(self._t0_model_s)
        self._prev_distance_m = float(
            np.linalg.norm(self._init_xyzs[0] - self._deck_sample().position_m)
        )

    def _housekeeping(self) -> None:
        """Rebuild the world after ``resetSimulation``; upstream's, minus the drone's mesh.

        ``BaseAviary._housekeeping`` re-parses the drone URDF on every reset, and ~21 of its
        ~23 ms are PyBullet loading the **visual** mesh ``cf2.dae``, which a DIRECT client
        never draws (P5-D5). With visual shapes wanted (GUI, or ``visual_shapes=True``) this
        is exactly upstream's method. Otherwise it is upstream's method line for line --
        same calls, same order, same arguments -- except that the drone is loaded with
        ``URDF_IGNORE_VISUAL_SHAPES`` added to its flags. The collision cylinder, the
        file inertia, the body and link order and every body id are unchanged, so the
        dynamics world the solver sees is the same one; ``tests/test_lazy_trajectory.py``
        compares the two worlds' dynamics and collision data, pins the upstream method's
        source hash so an upstream change forces a re-review, and the P5-D5 re-runs of
        e00 and e01 came out byte-identical.

        Restoring a cached world (``saveState``/``restoreState``) or keeping bodies across
        resets was considered and not taken: either skips ``resetSimulation``, which
        leaves the broadphase and contact caches with a history a fresh world does not
        have, so bit-identity could not be argued, only hoped for.
        """
        if self._visual_shapes:
            super()._housekeeping()
            return
        # --- upstream BaseAviary._housekeeping, verbatim apart from the drone's flags. ---
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1 * np.ones(self.NUM_DRONES)
        self.Y_AX = -1 * np.ones(self.NUM_DRONES)
        self.Z_AX = -1 * np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1 * np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM = False
        self.last_input_switch = 0
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        pyb.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        pyb.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        pyb.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        self.PLANE_ID = pyb.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        self.DRONE_IDS = np.array(
            [
                pyb.loadURDF(
                    str(files("gym_pybullet_drones") / "assets" / self.URDF),
                    self.INIT_XYZS[i, :],
                    pyb.getQuaternionFromEuler(self.INIT_RPYS[i, :]),
                    flags=pyb.URDF_USE_INERTIA_FROM_FILE | pyb.URDF_IGNORE_VISUAL_SHAPES,
                    physicsClientId=self.CLIENT,
                )
                for i in range(self.NUM_DRONES)
            ]
        )
        if self.GUI and self.USER_DEBUG:
            for i in range(self.NUM_DRONES):
                self._showDroneLocalAxes(i)
        if self.OBSTACLES:
            self._addObstacles()

    def _addObstacles(self) -> None:  # noqa: N802
        """Spawn the deck plate.

        ``BaseAviary._housekeeping`` calls this immediately after ``resetSimulation`` and
        the drone URDF load, which is exactly when the plate and its constraint have to be
        rebuilt: both died with the client reset.
        """
        self._platform.spawn(int(self.CLIENT), self._trajectory.sample(0))

    def set_motion(
        self,
        motion: EpisodeMotionSource,
        pad: str | None = None,
        pad_offset_m: tuple[float, float, float] | None = None,
    ) -> None:
        """Point the environment at a different deck-motion source.

        Exists so that a sweep can run many realizations through **one** environment
        instance: building a fresh ``DeckLandingAviary`` per episode would re-parse the
        drone URDF and re-open a physics client 200 times per cell. The new source takes
        effect at the next :meth:`reset`, never mid-episode, so no episode can straddle two
        realizations.

        Args:
            motion: The new deck-motion source.
            pad: New pad name, or ``None`` to keep the current one.
            pad_offset_m: The new source's pad lever arm, metres model scale, or ``None`` to
                keep the current one. It differs per vessel (-1.984 m frigate, -2.800 m
                s175), so a sweep that changes hull must pass it.
        """
        self.motion = motion
        if pad is not None:
            self.pad = pad
        if pad_offset_m is not None:
            self.pad_offset_m = pad_offset_m

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Reset to a fresh episode.

        Args:
            seed: Episode seed. Given, it fully determines the episode: the start offset,
                the initial state and the noise stream. Omitted, an internal counter
                advances, so successive resets give different episodes deterministically
                from the constructor's ``episode_seed``.
            options: Unused; accepted for Gymnasium API compliance.

        Returns:
            ``(observation, info)``.
        """
        entropy = int(seed) if seed is not None else self._base_seed + 1 + self._episode_index
        self._episode_index += 1
        self._prepare_episode(entropy)
        self.INIT_XYZS = self._init_xyzs
        # Order matters. `BaseAviary.reset` rebuilds the client, calls `_addObstacles`
        # (which spawns the plate from the new trajectory) and then calls `_computeObs`
        # and `_computeInfo` -- so every per-episode accumulator must already be cleared
        # when it runs, or the first observation of episode n+1 is built from episode n's
        # substep counter. `DSLPIDControl` carries integral state across episodes unless it
        # is reset here, which is one of the three ways determinism breaks.
        self._begin_episode()
        self._ctrl.reset()
        obs, info = super().reset(seed=seed, options=options)
        return np.asarray(obs, dtype=OBS_DTYPE), dict(info)

    # ---------------------------------------------------------------- stepping

    def _deck_sample(self, index: int | None = None) -> PlatformSample:
        """Return the analytic deck state at one physics substep.

        Args:
            index: Substep index; the current substep when omitted.

        Returns:
            The :class:`~rld.envs.platform.PlatformSample`, model scale, world frame.
        """
        return self._trajectory.sample(self._substep if index is None else index)

    def velocity_setpoint_m_s(self, action: FloatArray) -> FloatArray:
        """Map an action in ``[-1, 1]^3`` to a world-frame velocity setpoint.

        Public because Phase 6's residual wrapper and Phase 3's controllers must scale a
        setpoint the same way the environment does, and duplicating the rule is how the
        arms of a comparison silently stop sharing an action space.

        Args:
            action: The action, dimensionless, clipped into ``[-1, 1]^3`` here.

        Returns:
            ``(3,)`` metres per second model scale, world frame. When
            ``cfg.v_max_is_norm_cap`` is set, the returned vector's norm never exceeds
            ``cfg.v_max_m_s``.
        """
        clipped = np.clip(np.asarray(action, dtype=np.float64).reshape(ACTION_DIM), -1.0, 1.0)
        command = self.cfg.v_max_m_s * clipped
        if self.cfg.v_max_is_norm_cap:
            speed = float(np.linalg.norm(command))
            if speed > self.cfg.v_max_m_s:
                command = command * (self.cfg.v_max_m_s / speed)
        return command

    def _drone_state(self) -> FloatArray:
        """Return gym-pybullet-drones' 20-entry state vector for the single drone.

        Returns:
            ``(20,)`` float64: position (m, world), quaternion, roll-pitch-yaw (rad),
            velocity (m/s, world), angular velocity (rad/s), last RPM command.
        """
        return np.asarray(self._getDroneStateVector(0), dtype=np.float64).reshape(20)

    def _drone_rotation(self, quaternion: FloatArray) -> FloatArray:
        """Return the drone's body-to-world rotation matrix.

        Args:
            quaternion: ``(x, y, z, w)``.

        Returns:
            ``(3, 3)`` dimensionless rotation matrix.
        """
        flat = np.asarray(pyb.getMatrixFromQuaternion(list(quaternion)), dtype=np.float64).reshape(
            9
        )
        return flat.reshape(3, 3)

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """Advance one control step: ``PYB_STEPS_PER_CTRL`` physics substeps.

        The base class's loop is reproduced rather than called because it offers no
        per-substep hook, and both the deck drive and the contact poll must run at the
        physics rate: driving the deck at 30 Hz fails the 1 mm tracking test outright, and
        polling contacts at 30 Hz misses or mis-times a fast approach, which shows up as
        analytic-versus-contact disagreement.

        Args:
            action: The action, dimensionless, in ``[-1, 1]^3``.

        Returns:
            ``(observation, reward, terminated, truncated, info)``. ``timeout`` is the only
            ``truncated`` outcome; every other outcome is ``terminated``.
        """
        act = np.clip(np.asarray(action, dtype=np.float64).reshape(ACTION_DIM), -1.0, 1.0)
        self._prev_action = self._last_action
        self._last_action = act
        target_vel = self.velocity_setpoint_m_s(act)

        state = self._drone_state()
        rpm, _, _ = self._ctrl.computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=state[0:3],
            target_rpy=np.array([0.0, 0.0, np.radians(self.cfg.yaw_setpoint_deg)]),
            target_vel=target_vel,
        )
        clipped_rpm = np.clip(np.asarray(rpm, dtype=np.float64).reshape(1, 4), 0.0, self.MAX_RPM)

        for _ in range(self.PYB_STEPS_PER_CTRL):
            if self._substep >= self.cfg.max_substeps:
                break
            self._platform.advance(self._deck_sample())
            self._physics(clipped_rpm[0, :], 0)
            pyb.stepSimulation(physicsClientId=self.CLIENT)
            self.last_clipped_action = clipped_rpm
            self._substep += 1
            self._updateAndStoreKinematicInformation()
            self._poll_detectors()

        self.record.steps += 1
        self.record.control_effort += float(np.linalg.norm(act))
        self.record.action_jerk += float(np.linalg.norm(act - self._prev_action))

        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        self.record.return_ += reward
        return obs, reward, terminated, truncated, info

    def _poll_detectors(self) -> None:
        """Run both touchdown detectors and the bail-out checks for one physics substep."""
        deck = self._deck_sample()
        position = np.asarray(self.pos[0], dtype=np.float64)
        velocity = np.asarray(self.vel[0], dtype=np.float64)
        rotation = self._drone_rotation(np.asarray(self.quat[0], dtype=np.float64))
        t_episode = self._substep * self.cfg.physics_dt_s
        half = self.cfg.platform.half_extents_m

        poll = poll_contacts(
            client=int(self.CLIENT),
            drone_id=int(self.DRONE_IDS[0]),
            deck_id=self._platform.body_id,
            plane_id=int(self.PLANE_ID),
            cfg=self.success,
        )
        contact = contact_touchdown(
            poll=poll,
            t_model_s=deck.t_model_s,
            t_episode_s=t_episode,
            drone_position_m=position,
            drone_velocity_m_s=velocity,
            drone_rotation=rotation,
            deck=deck,
            half_extents_m=half,
        )
        analytic = analytic_touchdown(
            t_model_s=deck.t_model_s,
            t_episode_s=t_episode,
            drone_position_m=position,
            drone_velocity_m_s=velocity,
            drone_rotation=rotation,
            deck=deck,
            half_extents_m=half,
            geometry=self._geometry,
            contact_margin_m=self.success.analytic_contact_margin_m,
        )
        if self.record.analytic_record is None and analytic is not None:
            self.record.analytic_record = analytic

        was_in_contact = self._in_contact
        self._in_contact = contact is not None
        if self._in_contact and not was_in_contact:
            gap_s = t_episode - self._last_contact_s
            if self.record.contact_events == 0 or gap_s > self.success.contact_loss_grace_s:
                self.record.contact_events += 1
        if contact is not None:
            self.record.max_penetration_m = min(
                self.record.max_penetration_m, contact.penetration_m
            )
            if contact.penetration_m < -self.success.tunnelling_penetration_m:
                self.record.tunnelled = True
            if self.record.contact_record is None:
                self.record.contact_record = contact
                if not contact.on_plate:
                    self._crash = True
                    self._reason = "off_plate_strike"
            self._last_contact_s = t_episode
        first = self.record.contact_record
        if first is not None:
            if self._in_contact:
                self.record.dwell_s = t_episode - first.t_episode_s
            elif t_episode - self._last_contact_s > self.success.contact_loss_grace_s:
                self._contact_lost = True

        # Bail-outs. Checked in the order that makes the reason most specific.
        abs_tilt = float(
            np.degrees(
                np.arccos(np.clip(float(np.dot(rotation[:, 2], np.array([0.0, 0.0, 1.0]))), -1, 1))
            )
        )
        if not self._crash:
            if abs_tilt > self.success.crash_tilt_deg:
                self._crash, self._reason = True, "tilt_gt_crash"
            elif poll.touches_ground:
                self._crash, self._reason = True, "ground_contact"
            elif position[2] < self.cfg.platform.deck_origin_z_m - self.cfg.bounds.below_deck_m:
                self._crash, self._reason = True, "below_deck"
            else:
                offset = position - np.asarray(deck.position_m, dtype=np.float64)
                out = (
                    abs(float(offset[0])) > self.cfg.bounds.rel_xy_max_m
                    or abs(float(offset[1])) > self.cfg.bounds.rel_xy_max_m
                    or abs(float(offset[2])) > self.cfg.bounds.rel_z_max_m
                )
                if out:
                    self._crash, self._reason = True, "out_of_bounds"

    # ---------------------------------------------------------------- base-class reads

    def _computeObs(self) -> FloatArray:  # noqa: N802
        """Build the observation for the current state.

        Returns:
            A ``(n,)`` float32 vector matching ``observation_space``.
        """
        deck = self._deck_sample()
        position = np.asarray(self.pos[0], dtype=np.float64)
        velocity = np.asarray(self.vel[0], dtype=np.float64)
        rotation = self._drone_rotation(np.asarray(self.quat[0], dtype=np.float64))
        rpy = np.asarray(self.rpy[0], dtype=np.float64)
        override: FloatArray | None = None
        if self._noise_cfg.enabled:
            yaw = float(rpy[2])
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            rot_yaw = np.array(
                [[cos_yaw, sin_yaw, 0.0], [-sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            clean = np.concatenate(
                [
                    rot_yaw @ (np.asarray(deck.position_m, dtype=np.float64) - position),
                    rot_yaw @ (np.asarray(deck.velocity_m_s, dtype=np.float64) - velocity),
                ]
            )
            override = self._noise.apply(clean)
        return build_observation(
            cfg=self._obs_cfg,
            drone_position_m=position,
            drone_rpy_rad=rpy,
            drone_velocity_m_s=velocity,
            drone_body_rates_rad_s=np.asarray(self.ang_v[0], dtype=np.float64),
            drone_rotation=rotation,
            deck=deck,
            geometry=self._geometry,
            time_fraction=float(
                np.clip(
                    self._substep * self.cfg.physics_dt_s / self.cfg.episode_len_s,
                    0.0,
                    1.0,
                )
            ),
            last_action=self._last_action,
            in_contact=self._in_contact,
            relative_block_override=override,
        )

    def _computeReward(self) -> float:  # noqa: N802
        """Return this control step's reward, dense plus any terminal term.

        Returns:
            The reward, dimensionless.
        """
        deck = self._deck_sample()
        position = np.asarray(self.pos[0], dtype=np.float64)
        velocity = np.asarray(self.vel[0], dtype=np.float64)
        rotation = self._drone_rotation(np.asarray(self.quat[0], dtype=np.float64))
        distance = float(np.linalg.norm(position - np.asarray(deck.position_m, dtype=np.float64)))
        normal = np.asarray(deck.normal, dtype=np.float64)
        closing = max(
            0.0,
            -float(np.dot(velocity - np.asarray(deck.velocity_m_s, dtype=np.float64), normal)),
        )
        clearance = analytic_clearance_m(position, rotation, deck, self._geometry)
        reward = shaping_reward(
            cfg=self._reward_cfg,
            distance_prev_m=self._prev_distance_m,
            distance_now_m=distance,
            closing_speed_m_s=closing,
            clearance_m=clearance,
            action=self._last_action,
            prev_action=self._prev_action,
        )
        self._prev_distance_m = distance
        outcome = self._resolve_outcome()
        if outcome is not None:
            reward += terminal_reward(self._reward_cfg, outcome)
        return reward

    def _resolve_outcome(self) -> Outcome | None:
        """Classify the episode if it has ended, and latch the result.

        Returns:
            The outcome, or ``None`` if the episode is still running.
        """
        if self.record.outcome is not None:
            return self.record.outcome
        elapsed = self._substep * self.cfg.physics_dt_s
        first = self.record.contact_record
        timed_out = first is None and elapsed >= self.cfg.episode_len_s
        dwell_done = first is not None and self.record.dwell_s >= self.success.contact_dwell_min_s
        hard_cap = self._substep >= self.cfg.max_substeps
        finished = self._crash or self._contact_lost or dwell_done or timed_out or hard_cap
        if not finished:
            return None
        outcome = classify(
            first,
            dwell_s=self.record.dwell_s,
            contact_lost=self._contact_lost,
            crash=self._crash,
            timed_out=timed_out or hard_cap,
            cfg=self.success,
        )
        if self._reason is None:
            if self._contact_lost:
                self._reason = "release"
            elif dwell_done:
                self._reason = "dwell_complete"
            else:
                self._reason = "time_limit"
        self.record.outcome = outcome
        self.record.termination_reason = self._reason
        self._truncated = outcome == "timeout"
        self._terminated = not self._truncated
        return outcome

    def _computeTerminated(self) -> bool:  # noqa: N802
        """Report terminal failure or terminal success.

        Returns:
            True for ``crash``, ``off_pad``, ``hard_landing``, ``bounce`` and ``success``;
            False for ``timeout`` and while the episode runs.
        """
        self._resolve_outcome()
        return self._terminated

    def _computeTruncated(self) -> bool:  # noqa: N802
        """Report the time limit, and only the time limit.

        Returns:
            True exactly when the outcome is ``timeout``. A terminal penalty on a
            truncation, or a truncation reported as a termination, is what teaches PPO that
            hovering out the clock is safe; ``tests/test_landing_env.py`` pins this.
        """
        self._resolve_outcome()
        return self._truncated

    def _computeInfo(self) -> dict[str, Any]:  # noqa: N802
        """Return the per-step info dict.

        Returns:
            A mapping carrying the outcome (once decided), the termination reason, both
            touchdown records, the dwell so far, the detector-disagreement flag, and the
            episode's start offset in seconds model scale.
        """
        return {
            "outcome": self.record.outcome,
            "termination_reason": self.record.termination_reason,
            "touchdown_contact": self.record.contact_record,
            "touchdown_analytic": self.record.analytic_record,
            "dwell_s": self.record.dwell_s,
            "in_contact": self._in_contact,
            "detectors_disagree": self.record.detectors_disagree(
                self.success.detector_disagreement_window_s
            ),
            "t0_model_s": self._t0_model_s,
            "t_model_s": self._deck_sample().t_model_s,
            "realization_key": self.motion.key,
        }

    def _preprocessAction(self, action: FloatArray) -> FloatArray:  # noqa: N802
        """Unused: :meth:`step` is overridden and never calls this hook.

        Args:
            action: Ignored.

        Returns:
            Never returns.

        Raises:
            NotImplementedError: Always. The velocity setpoint is tracked inside
                :meth:`step` so that the deck can be advanced between physics substeps,
                which ``BaseAviary.step`` provides no hook for.
        """
        raise NotImplementedError(
            "DeckLandingAviary overrides step(); _preprocessAction is never called"
        )
