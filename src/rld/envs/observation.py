"""The observation vector: a declared field table, its Box, and the builder.

Why a table
-----------
The layout is a module-level tuple of :class:`ObsField`, not a sequence of ``np.hstack``
calls, so the vector documents itself -- every block's name, slice, unit and frame -- and
Phase 4 appends its forecast block at the end without renumbering anything that already
exists. ``tests/test_landing_env.py`` asserts the table and the built vector agree in
length, so the two cannot drift apart.

The layout (25 entries with both optional blocks on)
----------------------------------------------------
======================  ===  =====================================  ======================
block                   dim  contents                               frame / unit
======================  ===  =====================================  ======================
``attitude``              3  roll, pitch, yaw                       radians, world
``body_rates``            3  angular velocity                       radians/s, body
``velocity``              3  drone linear velocity                  m/s, drone-yaw
``rel_position``          3  ``p_pad - p_drone``                    m, drone-yaw
``rel_velocity``          3  ``v_pad - v_drone``                    m/s, drone-yaw
``deck_normal``           3  unit deck normal                       dimensionless, drone-yaw
``rel_tilt``              1  drone body z vs deck normal            radians
``height``                1  signed clearance to the pad plane      m
``time_fraction``         1  ``t_episode / episode_len``            dimensionless, [0, 1]
``last_action``           3  ``a_{t-1}``                            dimensionless, [-1, 1]
``in_contact``            1  deck contact flag                      dimensionless, {0, 1}
*(Phase 4)*               k  forecast block                         appended, off by default
======================  ===  =====================================  ======================

Frames
------
"world" is P1-D2's frame: ``x`` = bow, ``y`` = port, ``z`` = up. "drone-yaw" is that frame
rotated by ``-yaw`` about ``z`` -- the drone's heading-aligned horizontal frame. It is
deliberately **not** the full body frame: a tilted drone's relative-position block should
not swing with its attitude, because then the same geometric situation would look different
depending on how the vehicle happened to be leaning.

Why ``last_action`` and ``in_contact`` are in here
--------------------------------------------------
The smoothness term in the reward depends on ``a_t - a_{t-1}``, so without ``a_{t-1}`` the
reward is not a function of the observed state and the MDP is not Markov. And the dwell
criterion asks the policy to stay put for 0.5 s after touchdown, which it cannot even
represent without knowing it is touching.

Units and scales
----------------
Metres and metres per second are **model** scale throughout; angles are radians inside the
vector (degrees appear only at module boundaries, per the project convention); time
fraction is dimensionless.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
from dmf.typedefs import FloatArray
from gymnasium.spaces import Box

from rld.envs.config import ObservationConfig
from rld.envs.platform import PlatformSample
from rld.envs.touchdown import DroneCollisionGeometry, analytic_clearance_m, relative_tilt_deg

__all__ = ["OBS_DTYPE", "ObsField", "build_observation", "obs_fields", "observation_space"]

#: The dtype every observation is returned in. float32 is what SB3's buffers store and what
#: the ONNX export in Phase 8 will consume; building in float64 and casting once at the end
#: keeps the arithmetic clean without a dtype mismatch at the boundary.
OBS_DTYPE = np.float32


@dataclass(frozen=True)
class ObsField:
    """One named block of the observation vector.

    Attributes:
        name: Block name, stable across phases.
        size: Number of entries, dimensionless.
        unit: Physical unit, or ``"1"`` for a dimensionless block.
        frame: Reference frame, or ``"-"`` where none applies.
        low: Lower bound of every entry, in ``unit``.
        high: Upper bound of every entry, in ``unit``.
    """

    name: str
    size: int
    unit: str
    frame: str
    low: float
    high: float


@cache
def obs_fields(cfg: ObservationConfig) -> tuple[ObsField, ...]:
    """Return the observation layout for one config.

    Args:
        cfg: Which optional blocks are on, and the clipping bounds.

    Returns:
        The blocks in vector order. Offsets are the running sums of ``size``.
    """
    fields = [
        ObsField("attitude", 3, "rad", "world", -np.pi, np.pi),
        ObsField(
            "body_rates", 3, "rad/s", "body", -cfg.max_body_rate_rad_s, cfg.max_body_rate_rad_s
        ),
        ObsField("velocity", 3, "m/s", "drone-yaw", -cfg.max_velocity_m_s, cfg.max_velocity_m_s),
        ObsField(
            "rel_position", 3, "m", "drone-yaw", -cfg.max_rel_position_m, cfg.max_rel_position_m
        ),
        ObsField(
            "rel_velocity",
            3,
            "m/s",
            "drone-yaw",
            -cfg.max_rel_velocity_m_s,
            cfg.max_rel_velocity_m_s,
        ),
        ObsField("deck_normal", 3, "1", "drone-yaw", -1.0, 1.0),
        ObsField("rel_tilt", 1, "rad", "-", 0.0, np.pi),
        ObsField("height", 1, "m", "deck normal", -cfg.max_height_m, cfg.max_height_m),
        ObsField("time_fraction", 1, "1", "-", 0.0, 1.0),
    ]
    if cfg.include_last_action:
        fields.append(ObsField("last_action", 3, "1", "world", -1.0, 1.0))
    if cfg.include_contact:
        fields.append(ObsField("in_contact", 1, "1", "-", 0.0, 1.0))
    return tuple(fields)


@cache
def observation_space(cfg: ObservationConfig) -> Box:
    """Build the flat ``Box`` the layout describes.

    Bounds are finite everywhere. ``gymnasium.utils.env_checker.check_env`` warns on
    infinite bounds, and more importantly a finite box is a promise the builder keeps by
    clipping, so an observation can never fall outside its own space during a divergence.

    Args:
        cfg: Which optional blocks are on, and the clipping bounds.

    Returns:
        A ``Box(float32, shape=(n,))`` with per-entry bounds from the layout.
    """
    fields = obs_fields(cfg)
    low = np.concatenate([np.full(field.size, field.low) for field in fields]).astype(OBS_DTYPE)
    high = np.concatenate([np.full(field.size, field.high) for field in fields]).astype(OBS_DTYPE)
    return Box(low=low, high=high, dtype=OBS_DTYPE)


def _yaw_frame(yaw_rad: float) -> FloatArray:
    """Return the world-to-drone-yaw rotation matrix.

    Args:
        yaw_rad: Drone yaw, radians.

    Returns:
        ``R_z(-yaw)``, shape ``(3, 3)``, dimensionless. Applying it to a world vector gives
        that vector in the drone's heading-aligned horizontal frame.
    """
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array(
        [[cos_yaw, sin_yaw, 0.0], [-sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def build_observation(
    *,
    cfg: ObservationConfig,
    drone_position_m: FloatArray,
    drone_rpy_rad: FloatArray,
    drone_velocity_m_s: FloatArray,
    drone_body_rates_rad_s: FloatArray,
    drone_rotation: FloatArray,
    deck: PlatformSample,
    geometry: DroneCollisionGeometry,
    time_fraction: float,
    last_action: FloatArray,
    in_contact: bool,
) -> FloatArray:
    """Assemble one observation vector.

    Args:
        cfg: Which optional blocks are on, and the clipping bounds.
        drone_position_m: Drone body-origin position, metres model scale, world frame.
        drone_rpy_rad: Drone roll, pitch, yaw, radians, world frame.
        drone_velocity_m_s: Drone linear velocity, metres per second model scale, world
            frame.
        drone_body_rates_rad_s: Drone angular velocity, radians per second. This is
            PyBullet's ``getBaseVelocity`` angular term, i.e. world axes; it is passed
            through unrotated and the layout table calls it ``body`` because that is how
            gym-pybullet-drones' own observation labels it.
        drone_rotation: The drone's body-to-world rotation matrix, shape ``(3, 3)``.
        deck: The deck state every deck-derived entry (relative position and velocity,
            deck normal, relative tilt, clearance) is computed from: the analytic deck at
            the same instant, or, with the perception stand-in on, the perceived sample
            from :meth:`rld.envs.noise.PerceptionNoise.perceive` (P7-D4). Metres and metres
            per second model scale, world frame.
        geometry: The drone's collision cylinder, for the signed clearance.
        time_fraction: ``t_episode / episode_len``, dimensionless, clipped into ``[0, 1]``.
        last_action: The previous action, dimensionless, in ``[-1, 1]^3``.
        in_contact: Whether the drone is touching the deck this step.

    Returns:
        A ``(n,)`` float32 vector matching :func:`observation_space`, clipped into its
        bounds.
    """
    yaw = float(drone_rpy_rad[2])
    rot_yaw = _yaw_frame(yaw)
    normal = np.asarray(deck.normal, dtype=np.float64)

    rel_position_world = np.asarray(deck.position_m, dtype=np.float64) - np.asarray(
        drone_position_m, dtype=np.float64
    )
    rel_velocity_world = np.asarray(deck.velocity_m_s, dtype=np.float64) - np.asarray(
        drone_velocity_m_s, dtype=np.float64
    )
    relative_block = np.concatenate([rot_yaw @ rel_position_world, rot_yaw @ rel_velocity_world])

    blocks: list[FloatArray] = [
        np.asarray(drone_rpy_rad, dtype=np.float64).reshape(3),
        np.asarray(drone_body_rates_rad_s, dtype=np.float64).reshape(3),
        rot_yaw @ np.asarray(drone_velocity_m_s, dtype=np.float64),
        relative_block[0:3],
        relative_block[3:6],
        rot_yaw @ normal,
        np.array([np.radians(relative_tilt_deg(drone_rotation[:, 2], normal))]),
        np.array([analytic_clearance_m(drone_position_m, drone_rotation, deck, geometry)]),
        np.array([time_fraction]),
    ]
    if cfg.include_last_action:
        blocks.append(np.asarray(last_action, dtype=np.float64).reshape(3))
    if cfg.include_contact:
        blocks.append(np.array([1.0 if in_contact else 0.0]))

    vector = np.concatenate(blocks)
    space = observation_space(cfg)
    return np.clip(vector, space.low.astype(np.float64), space.high.astype(np.float64)).astype(
        OBS_DTYPE
    )
