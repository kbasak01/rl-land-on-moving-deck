"""The deck plate as a PyBullet body, driven along the bridge's deck-point trajectory.

What this module is for
-----------------------
The whole project's physics rests on one claim: the drone lands on a deck that is *really*
moving, not on a deck that is teleported under it. This module is where that claim is made
and where ``tests/test_platform.py`` checks it.

The trap it exists to avoid
---------------------------
``resetBasePositionAndOrientation`` alone moves the body but leaves its velocity at zero, so
the contact solver believes the plate is stationary. Relative touchdown velocities are then
wrong and the drone is never carried by the deck. ``getBaseVelocity`` agreeing with the
analytic velocity is **necessary but not sufficient** to rule this out -- under a kinematic
drive it merely reads back what was written -- so the decisive check is the *conveyor* test:
a drone resting on a plate translating at 0.5 m/s must be carried along.

The two drivers, both implemented, both measured (P2-D1)
--------------------------------------------------------
``kinematic``
    A dynamic body whose weight is cancelled by an external force, with pose **and**
    velocity (linear and angular) written from the analytic deck state every physics step.
    The plate the solver sees is exactly the analytic plate.
``constraint``
    A dynamic body held to the world by a ``JOINT_FIXED`` constraint, retargeted with
    ``changeConstraint`` every physics step. The constraint is a velocity-level servo, so
    the body trails its target by a few steps.

Both are driven at the **physics** rate (240 Hz model), never at the control rate.

Units, scales and frames
------------------------
Every length is metres **model** scale, every velocity metres per second model scale, every
angular rate radians per second model scale, and every time seconds model scale. The world
frame is the one P1-D2 fixed: ``x`` = bow, ``y`` = port, ``z`` = up, and the plate's
orientation is ``pybullet.getQuaternionFromEuler(DeckPointState.euler_xyz_rad)`` with no
further conversion. The plate's **body origin is on its top surface at the pad centre**, so
it coincides with the bridge's deck point offset by ``deck_origin_m``.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pybullet as pyb
from dmf.data.splits import RealizationKey, realization_key
from dmf.typedefs import FloatArray

from rld.deck.bridge import (
    DeckMotionSource,
    DeckPointTrajectory,
    MotionChannels,
    load_vessel_cached,
)
from rld.deck.config import PadConfig
from rld.deck.kinematics import DeckPointState, deck_point_state
from rld.deck.scaling import FroudeScale
from rld.envs.config import PlatformConfig

__all__ = [
    "PAD_MARKER_RGBA",
    "PLATE_RGBA",
    "DeckPlatform",
    "EpisodeMotionSource",
    "DeckTrajectory",
    "PlatformSample",
    "StaticDeckMotion",
    "build_trajectory",
    "pad_offset_model_m",
    "quaternion_angle_rad",
]

#: Plate colour, RGBA in [0, 1]. Visual only; DIRECT mode ignores it.
PLATE_RGBA: tuple[float, float, float, float] = (0.35, 0.38, 0.42, 1.0)

#: Painted-pad marker colour, RGBA in [0, 1].
PAD_MARKER_RGBA: tuple[float, float, float, float] = (0.95, 0.75, 0.15, 1.0)

#: Thickness of the painted-pad visual disc, metres model scale. Visual only -- it carries
#: no collision shape, so it cannot change a contact.
PAD_MARKER_HEIGHT_M: float = 1.0e-3


class EpisodeMotionSource(DeckMotionSource, Protocol):
    """:class:`~rld.deck.bridge.DeckMotionSource` plus the one method an episode needs.

    ``DeckMotionSource`` as :mod:`rld.deck.bridge` declares it does not list
    ``episode_start_window_s``, although every concrete source implements it
    (``JonswapDeckMotion``, ``SinusoidDeckMotion``, and the static test double here).
    Widening the protocol in ``rld/deck/`` would be an edit to the deck-bridge owner's
    package, so Phase 2 states the requirement on its own side instead. If Phase 4 adds the
    method to the base protocol, this class collapses to an alias.
    """

    def episode_start_window_s(self, episode_len_model_s: float) -> tuple[float, float]:
        """Return the model-scale window of legal episode start offsets, seconds.

        Args:
            episode_len_model_s: Episode length, seconds model scale.

        Returns:
            ``(lo, hi)`` seconds model scale, inclusive, such that an episode starting
            anywhere inside it evaluates only committed times.
        """
        ...


def _vec(value: object, size: int) -> FloatArray:
    """Narrow an untyped PyBullet return into a fixed-length float64 array.

    ``pybullet`` is in mypy's ``ignore_missing_imports`` list, so everything it returns is
    ``Any``. Funnelling those returns through this one function is what stops ``Any`` from
    leaking into the typed parts of :mod:`rld.envs`.

    Args:
        value: A PyBullet return value (a tuple of floats).
        size: Expected length.

    Returns:
        A ``(size,)`` float64 array, in whatever units the caller's PyBullet call used.

    Raises:
        ValueError: If the value does not have exactly ``size`` entries.
    """
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"expected {size} entries from PyBullet, got {array.size}")
    return array


def quaternion_angle_rad(a: FloatArray, b: FloatArray) -> float:
    """Return the rotation angle between two unit quaternions, radians.

    Args:
        a: Quaternion ``(x, y, z, w)``, PyBullet's order.
        b: Quaternion ``(x, y, z, w)``.

    Returns:
        The geodesic angle in ``[0, pi]``, radians. Sign-insensitive: ``q`` and ``-q`` are
        the same rotation.
    """
    dot = float(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


@dataclass(frozen=True)
class PlatformSample:
    """The analytic deck state at one physics instant, ready for PyBullet.

    Attributes:
        t_model_s: Time, seconds model scale, measured from the start of the committed dmf
            record (not from the start of the episode).
        position_m: Plate body-origin position, metres model scale, world frame. Already
            includes ``PlatformConfig.deck_origin_m``.
        quaternion: Plate orientation ``(x, y, z, w)``, PyBullet's order, dimensionless.
        euler_xyz_rad: ``(phi, theta, 0)`` radians, the triple the quaternion came from.
        velocity_m_s: Plate linear velocity, metres per second model scale, world frame.
        angular_velocity_rad_s: Plate angular velocity, radians per second model scale,
            world axes.
        acceleration_m_s2: Plate linear acceleration, metres per second squared, world
            frame.
        normal: Unit deck normal, dimensionless, world frame.
        tilt_deg: Angle between the deck normal and world ``+z``, degrees.
    """

    t_model_s: float
    position_m: FloatArray
    quaternion: FloatArray
    euler_xyz_rad: FloatArray
    velocity_m_s: FloatArray
    angular_velocity_rad_s: FloatArray
    acceleration_m_s2: FloatArray
    normal: FloatArray
    tilt_deg: float


@dataclass(frozen=True)
class DeckTrajectory:
    """One episode's deck-point trajectory, precomputed on the physics grid.

    Precomputed deliberately. Evaluating the bridge one sample at a time inside each of the
    1 920 physics substeps per simulated second costs far more than one batched call per
    reset, and the two are bit-identical because ``synthesize_motion`` is a memoryless sum
    of harmonics in ``t`` (``tests/test_landing_env.py`` asserts the equality).

    Attributes:
        pad: Pad name, ``"aft"`` or ``"cg"``.
        t_model_s: Sample times, seconds model scale, shape ``(n,)``.
        position_m: Plate body-origin positions including ``deck_origin_m``, metres model
            scale, world frame, shape ``(n, 3)``.
        quaternion: Plate orientations ``(x, y, z, w)``, shape ``(n, 4)``.
        state: The underlying model-scale :class:`~rld.deck.kinematics.DeckPointState`,
            relative to the vessel's mean-position CG (i.e. **without** ``deck_origin_m``).
    """

    pad: str
    t_model_s: FloatArray
    position_m: FloatArray
    quaternion: FloatArray
    state: DeckPointState

    def __len__(self) -> int:
        """Return the number of samples, dimensionless."""
        return int(self.t_model_s.size)

    def sample(self, index: int) -> PlatformSample:
        """Return one physics instant.

        Args:
            index: Sample index in ``[0, len(self))``.

        Returns:
            The :class:`PlatformSample` at that instant, model scale, world frame.

        Raises:
            IndexError: If ``index`` is out of range. Raised explicitly rather than letting
                numpy wrap a negative index, because a wrapped index would silently give
                the episode the deck motion from its own end.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"deck sample {index} out of range [0, {len(self)})")
        return PlatformSample(
            t_model_s=float(self.t_model_s[index]),
            position_m=self.position_m[index],
            quaternion=self.quaternion[index],
            euler_xyz_rad=self.state.euler_xyz_rad[index],
            velocity_m_s=self.state.velocity_m_s[index],
            angular_velocity_rad_s=self.state.angular_velocity_rad_s[index],
            acceleration_m_s2=self.state.acceleration_m_s2[index],
            normal=self.state.normal[index],
            tilt_deg=float(self.state.tilt_deg[index]),
        )


def pad_offset_model_m(
    vessel: str, pads: PadConfig, scale: FroudeScale, pad: str
) -> tuple[float, float, float]:
    """Return a pad's lever arm from the ship's mean-position CG, metres **model** scale.

    This is the constant the plate's world anchor has to remove. ``DeckPointState`` measures
    a pad's position **from the vessel's mean-position CG**, so the frigate's aft pad has a
    standing offset of -1.984 m model (-49.6 m full, -0.4 L) in world x and the s175's is
    -2.800 m -- P1-D3's named ``unseen_vessel`` confound. Left in, the plate would sit two
    metres aft of where the drone is released and, worse, at a *different* place for each
    hull, so ``unseen_vessel`` would test a geometry change as well as a dynamics change.

    Args:
        vessel: dmf vessel config stem, ``"frigate"`` or ``"s175"``.
        pads: Pad geometry; offsets are fractions of the vessel's full-scale length.
        scale: The Froude scale, for the single length conversion.
        pad: Pad name, e.g. ``"aft"`` or ``"cg"``.

    Returns:
        ``(x, y, z)`` metres model scale in the ship body frame (x = bow, y = port, z = up).

    Raises:
        FileNotFoundError: If no dmf config exists for ``vessel``.
        ValueError: If the pad is unknown.
    """
    length_full_m = float(load_vessel_cached(vessel).length_m)
    arm_full = pads.spec(pad).r_pad_full_m(length_full_m)
    arm_model = [float(scale.length(np.asarray(value, dtype=np.float64))) for value in arm_full]
    return (arm_model[0], arm_model[1], arm_model[2])


def build_trajectory(
    motion: DeckMotionSource,
    pad: str,
    cfg: PlatformConfig,
    t_model_s: FloatArray,
    pad_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> DeckTrajectory:
    """Evaluate one pad's trajectory on a physics-rate time grid, anchored at the plate.

    Args:
        motion: Any :class:`~rld.deck.bridge.DeckMotionSource` -- JONSWAP, sinusoid or the
            static test double.
        pad: Pad name from ``configs/deck/pad.yaml``.
        cfg: Plate config; only ``deck_origin_m`` is used here.
        t_model_s: Sample times, seconds model scale, measured from the start of the
            committed dmf record. Must lie inside the source's usable window.
        pad_offset_m: The pad's standing lever arm from the ship's mean-position CG, metres
            model scale, from :func:`pad_offset_model_m`. It is **subtracted**, so the plate
            oscillates about ``cfg.deck_origin_m`` instead of about a point two metres away
            from it. Only the constant is removed: every lever-arm-driven motion,
            ``R(t) r - r``, stays.

    Returns:
        The :class:`DeckTrajectory`, model scale, world frame.

    Raises:
        ValueError: If the source rejects a requested time (outside the committed record)
            or the pad is unknown.
    """
    traj: DeckPointTrajectory = motion.deck_point(np.asarray(t_model_s, dtype=np.float64), pad)
    origin = np.asarray(cfg.deck_origin_m, dtype=np.float64) - np.asarray(
        pad_offset_m, dtype=np.float64
    )
    position = np.asarray(traj.state.position_m, dtype=np.float64) + origin
    quaternion = np.asarray(
        [pyb.getQuaternionFromEuler(list(euler)) for euler in traj.state.euler_xyz_rad],
        dtype=np.float64,
    )
    return DeckTrajectory(
        pad=pad,
        t_model_s=np.asarray(traj.t_model_s, dtype=np.float64),
        position_m=position,
        quaternion=quaternion,
        state=traj.state,
    )


class StaticDeckMotion:
    """A zero-motion :class:`~rld.deck.bridge.DeckMotionSource`: a pad on flat water.

    A **test fixture**, not a motion model any committed result is computed from. It lives
    here rather than in :mod:`rld.deck` for that reason: ``rld.deck`` is the deck-bridge
    owner's package and every source in it is something a result can be measured against.

    It is what makes the two scripted static-pad checks possible -- a drop registering
    exactly one touchdown, and 100/100 ``success`` on a 0.3 m/s scripted descent -- by
    removing deck motion as a variable.

    Attributes:
        duration_s: Length of the pretend record, seconds model scale. Any time in
            ``[0, duration_s]`` is legal; outside it raises, as the real bridge does.
    """

    def __init__(self, duration_s: float = 120.0) -> None:
        """Build a static deck.

        Args:
            duration_s: Length of the pretend record, seconds model scale.

        Raises:
            ValueError: If ``duration_s`` is not positive.
        """
        if duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        self.duration_s = float(duration_s)

    @property
    def key(self) -> RealizationKey:
        """Return a dmf-shaped key identifying this fixture, so rows stay self-describing."""
        return realization_key("static", 180.0, 0.0, "static", 0)

    @property
    def t_model_window_s(self) -> tuple[float, float]:
        """Usable model-scale span, seconds, as ``(lo, hi)``."""
        return (0.0, self.duration_s)

    def episode_start_window_s(self, episode_len_model_s: float) -> tuple[float, float]:
        """Return the model-scale window of legal episode start offsets, seconds.

        Args:
            episode_len_model_s: Episode length, seconds model scale.

        Returns:
            ``(lo, hi)`` seconds model scale.

        Raises:
            ValueError: If the episode does not fit in the record.
        """
        lo, hi = self.t_model_window_s
        if not 0.0 < episode_len_model_s <= hi - lo:
            raise ValueError(
                f"episode of {episode_len_model_s} s (model) does not fit in {(lo, hi)}"
            )
        return (lo, hi - episode_len_model_s)

    def full_time_s(self, t_model_s: FloatArray) -> FloatArray:
        """Map model-scale seconds to full-scale seconds; the identity for a static deck.

        Args:
            t_model_s: Times, seconds model scale.

        Returns:
            The same values, seconds. No Froude scaling applies to a motionless deck.
        """
        return np.asarray(t_model_s, dtype=np.float64)

    def channels(self, t_full_s: FloatArray) -> MotionChannels:
        """Return all-zero motion channels.

        Args:
            t_full_s: Times, seconds.

        Returns:
            :class:`~rld.deck.bridge.MotionChannels` of zeros in dmf's units (degrees,
            degrees per second, degrees per second squared, metres, metres per second,
            metres per second squared).

        Raises:
            ValueError: If a time lies outside ``t_model_window_s``.
        """
        t = self._checked(t_full_s)
        zeros = np.zeros_like(t)
        return MotionChannels(
            t_full_s=t,
            roll_deg=zeros,
            pitch_deg=zeros.copy(),
            heave_m=zeros.copy(),
            roll_rate_dps=zeros.copy(),
            pitch_rate_dps=zeros.copy(),
            heave_rate_m_s=zeros.copy(),
            heave_acc_m_s2=zeros.copy(),
            roll_acc_dps2=zeros.copy(),
            pitch_acc_dps2=zeros.copy(),
        )

    def deck_point(self, t_model_s: FloatArray, pad: str) -> DeckPointTrajectory:
        """Return a motionless pad trajectory.

        Args:
            t_model_s: Times, seconds model scale.
            pad: Pad name; carried through to the result but geometrically irrelevant,
                since a motionless ship puts every pad at its mean position.

        Returns:
            The :class:`~rld.deck.bridge.DeckPointTrajectory`: position, velocity and
            acceleration all zero, orientation level, normal ``+z``, tilt 0 deg.

        Raises:
            ValueError: If a time lies outside ``t_model_window_s``.
        """
        t = self._checked(t_model_s)
        zeros = np.zeros_like(t)
        state = deck_point_state(
            roll_deg=zeros,
            pitch_deg=zeros,
            heave_m=zeros,
            roll_rate_dps=zeros,
            pitch_rate_dps=zeros,
            heave_rate_m_s=zeros,
            roll_acc_dps2=zeros,
            pitch_acc_dps2=zeros,
            heave_acc_m_s2=zeros,
            r_pad_m=(0.0, 0.0, 0.0),
        )
        return DeckPointTrajectory(pad=pad, t_model_s=t, t_full_s=t, state=state)

    def _checked(self, t_s: FloatArray) -> FloatArray:
        """Validate a time grid against the pretend record.

        Args:
            t_s: Times, seconds model scale.

        Returns:
            The same times as a float64 array.

        Raises:
            ValueError: If any time lies outside ``[0, duration_s]``.
        """
        t = np.asarray(t_s, dtype=np.float64)
        if t.size and (float(t.min()) < -1e-9 or float(t.max()) > self.duration_s + 1e-9):
            raise ValueError(
                f"times [{float(t.min())}, {float(t.max())}] leave the static record "
                f"[0, {self.duration_s}]"
            )
        return t


class DeckPlatform:
    """The deck plate body and its driver.

    One instance per environment; the body is (re)created on every ``spawn`` because
    ``BaseAviary.reset`` calls ``resetSimulation``, which destroys every body and every
    constraint in the client.

    Attributes:
        cfg: Plate geometry, mass, friction and driver.
        pad_radius_m: Painted-pad radius, metres model scale, from
            ``configs/deck/pad.yaml``. Visual only.
    """

    def __init__(self, cfg: PlatformConfig, pad_radius_m: float) -> None:
        """Configure a plate. No PyBullet call happens until :meth:`spawn`.

        Args:
            cfg: Plate geometry, mass, friction and driver.
            pad_radius_m: Painted-pad radius, metres model scale.

        Raises:
            ValueError: If ``pad_radius_m`` is not positive or does not fit on the plate.
        """
        if pad_radius_m <= 0.0:
            raise ValueError(f"pad_radius_m must be positive, got {pad_radius_m}")
        if pad_radius_m > min(cfg.half_extents_m[0], cfg.half_extents_m[1]):
            raise ValueError(
                f"pad radius {pad_radius_m} m does not fit on a "
                f"{2 * cfg.half_extents_m[0]} x {2 * cfg.half_extents_m[1]} m plate"
            )
        self.cfg = cfg
        self.pad_radius_m = float(pad_radius_m)
        self._client: int = -1
        self._body_id: int = -1
        self._constraint_id: int = -1
        self._weight_n: float = 0.0

    @property
    def body_id(self) -> int:
        """PyBullet body id of the plate; ``-1`` before :meth:`spawn`."""
        return self._body_id

    @property
    def constraint_id(self) -> int:
        """PyBullet constraint id; ``-1`` unless the ``constraint`` driver is in use."""
        return self._constraint_id

    def spawn(self, client: int, sample: PlatformSample) -> int:
        """Create the plate body (and, for ``constraint``, its constraint) at one instant.

        Call this from the environment's ``_addObstacles`` hook: ``BaseAviary`` calls it
        from ``_housekeeping`` immediately after ``resetSimulation`` and the drone URDF
        load, which is exactly when the plate has to be rebuilt.

        Args:
            client: PyBullet physics client id.
            sample: The deck state to spawn at, model scale, world frame.

        Returns:
            The new body id.
        """
        self._client = client
        hx, hy, hz = self.cfg.half_extents_m
        collision = int(
            pyb.createCollisionShape(
                pyb.GEOM_BOX,
                halfExtents=[hx, hy, hz],
                # Push the box down by its own half-thickness so that the BODY ORIGIN sits
                # on the plate's top surface, at the pad centre, and therefore coincides
                # with the bridge's deck point.
                collisionFramePosition=[0.0, 0.0, -hz],
                physicsClientId=client,
            )
        )
        visual = int(
            pyb.createVisualShape(
                pyb.GEOM_BOX,
                halfExtents=[hx, hy, hz],
                visualFramePosition=[0.0, 0.0, -hz],
                rgbaColor=list(PLATE_RGBA),
                physicsClientId=client,
            )
        )
        self._body_id = int(
            pyb.createMultiBody(
                baseMass=self.cfg.mass_kg,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=list(sample.position_m),
                baseOrientation=list(sample.quaternion),
                physicsClientId=client,
            )
        )
        pyb.changeDynamics(
            self._body_id,
            -1,
            lateralFriction=self.cfg.lateral_friction,
            restitution=self.cfg.restitution,
            physicsClientId=client,
        )
        self._spawn_pad_marker(client, sample)
        # Cached once per spawn rather than read every substep: one PyBullet round trip per
        # physics step would be 240 per simulated second, for a constant.
        self._weight_n = self.cfg.mass_kg * self._read_gravity(client)
        self._constraint_id = -1
        if self.cfg.driver == "constraint":
            self._constraint_id = int(
                pyb.createConstraint(
                    self._body_id,
                    -1,
                    -1,
                    -1,
                    pyb.JOINT_FIXED,
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    list(sample.position_m),
                    childFrameOrientation=list(sample.quaternion),
                    physicsClientId=client,
                )
            )
        return self._body_id

    def _spawn_pad_marker(self, client: int, sample: PlatformSample) -> None:
        """Attach the painted-pad disc.

        Visual only: the marker carries no collision shape, so it cannot change a contact.

        Args:
            client: PyBullet physics client id.
            sample: The deck state to spawn at, model scale, world frame.
        """
        marker = int(
            pyb.createVisualShape(
                pyb.GEOM_CYLINDER,
                radius=self.pad_radius_m,
                length=PAD_MARKER_HEIGHT_M,
                rgbaColor=list(PAD_MARKER_RGBA),
                physicsClientId=client,
            )
        )
        pyb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=marker,
            basePosition=list(sample.position_m),
            baseOrientation=list(sample.quaternion),
            physicsClientId=client,
        )

    def advance(self, sample: PlatformSample) -> None:
        """Drive the plate to one deck state.

        Call once per **physics** step, before ``stepSimulation``. The pose and velocity
        written here are the ones the contact solver sees during the step that follows,
        which is the whole point.

        Args:
            sample: The deck state for the interval that is about to be stepped, model
                scale, world frame.

        Raises:
            RuntimeError: If the plate has not been spawned.
        """
        if self._body_id < 0:
            raise RuntimeError("DeckPlatform.advance() before spawn()")
        position = list(sample.position_m)
        orientation = list(sample.quaternion)
        if self.cfg.driver == "constraint":
            pyb.changeConstraint(
                self._constraint_id,
                jointChildPivot=position,
                jointChildFrameOrientation=orientation,
                maxForce=self.cfg.constraint_max_force_n,
                physicsClientId=self._client,
            )
            return
        pyb.resetBasePositionAndOrientation(
            self._body_id, position, orientation, physicsClientId=self._client
        )
        pyb.resetBaseVelocity(
            self._body_id,
            linearVelocity=list(sample.velocity_m_s),
            angularVelocity=list(sample.angular_velocity_rad_s),
            physicsClientId=self._client,
        )
        # Cancel the plate's weight. Without it PyBullet integrates gravity into the
        # plate's velocity before the constraint solver runs, so the plate enters the
        # contact solve at v - g*dt: 0.0408 m/s, which is 7 % of the deck's own peak |v_z|
        # at SS6 and would show up directly in the touchdown relative velocity. The force
        # is applied at the plate's centre of mass in WORLD_FRAME, so it adds no torque and
        # stays vertical however the deck is tilted.
        pyb.applyExternalForce(
            self._body_id,
            -1,
            forceObj=[0.0, 0.0, self._weight_n],
            posObj=position,
            flags=pyb.WORLD_FRAME,
            physicsClientId=self._client,
        )

    @staticmethod
    def _read_gravity(client: int) -> float:
        """Return the client's gravitational acceleration magnitude, m/s^2.

        Read from PyBullet rather than assumed, so that the weight cancellation stays exact
        if a caller changes gravity. gym-pybullet-drones uses 9.8 m/s^2
        (``BaseAviary.py:74``), 0.068 % from dmf's 9.80665 -- a difference P1-D1 records and
        does not fix.

        Args:
            client: PyBullet physics client id.

        Returns:
            ``|g|``, metres per second squared.
        """
        params = pyb.getPhysicsEngineParameters(physicsClientId=client)
        gravity = np.array(
            [
                float(params["gravityAccelerationX"]),
                float(params["gravityAccelerationY"]),
                float(params["gravityAccelerationZ"]),
            ],
            dtype=np.float64,
        )
        return float(np.linalg.norm(gravity))

    def pose(self) -> tuple[FloatArray, FloatArray]:
        """Read the plate's pose back from PyBullet.

        Returns:
            ``(position_m, quaternion)``: metres model scale in the world frame, and
            ``(x, y, z, w)``.
        """
        position, orientation = pyb.getBasePositionAndOrientation(
            self._body_id, physicsClientId=self._client
        )
        return _vec(position, 3), _vec(orientation, 4)

    def velocity(self) -> tuple[FloatArray, FloatArray]:
        """Read the plate's velocity back from PyBullet.

        Returns:
            ``(linear_m_s, angular_rad_s)``: metres per second and radians per second, model
            scale, world frame and world axes.
        """
        linear, angular = pyb.getBaseVelocity(self._body_id, physicsClientId=self._client)
        return _vec(linear, 3), _vec(angular, 3)

    def corner_positions_m(self, sample: PlatformSample) -> FloatArray:
        """Return the plate's four top-surface corners for one analytic sample.

        Tracking the body origin alone cannot see an orientation error: a plate rotated
        about its own origin has zero position error. The corners are the lever that turns
        an orientation error into a length, and 0.40 m of lever makes 1 mrad worth 0.4 mm.

        Args:
            sample: The analytic deck state, model scale, world frame.

        Returns:
            ``(4, 3)`` corner positions, metres model scale, world frame.
        """
        hx, hy, _ = self.cfg.half_extents_m
        local = np.array(
            [[+hx, +hy, 0.0], [+hx, -hy, 0.0], [-hx, +hy, 0.0], [-hx, -hy, 0.0]],
            dtype=np.float64,
        )
        rot = _vec(pyb.getMatrixFromQuaternion(list(sample.quaternion)), 9).reshape(3, 3)
        return np.asarray(sample.position_m + local @ rot.T, dtype=np.float64)

    def measured_corner_positions_m(self) -> FloatArray:
        """Return the plate's four top-surface corners as PyBullet currently holds them.

        Returns:
            ``(4, 3)`` corner positions, metres model scale, world frame.
        """
        hx, hy, _ = self.cfg.half_extents_m
        local = np.array(
            [[+hx, +hy, 0.0], [+hx, -hy, 0.0], [-hx, +hy, 0.0], [-hx, -hy, 0.0]],
            dtype=np.float64,
        )
        position, quaternion = self.pose()
        rot = _vec(pyb.getMatrixFromQuaternion(list(quaternion)), 9).reshape(3, 3)
        return np.asarray(position + local @ rot.T, dtype=np.float64)
