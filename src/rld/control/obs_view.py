"""Named, world-frame access to the observation, and the setpoint-to-action conversion.

Why this module exists
----------------------
The observation is a flat float32 vector laid out by :func:`rld.envs.observation.obs_fields`
(P2-D9). A controller that slices it with literal indices breaks silently the day Phase 4
appends its forecast block, so every controller reads it through :class:`ObsView`, which
takes its offsets from the layout table and nothing else.

Frames
------
The observation's vector blocks are in the **drone-yaw** frame: the world frame rotated by
``-yaw`` about ``z`` (``v_obs = R_z(-yaw) v_world``). The setpoint the environment tracks is
in the **world** frame (P2-D2), so :class:`ObsView` rotates every vector block back with
``R_z(+yaw)``, using the ``yaw`` entry of the ``attitude`` block. World is P1-D2's:
``x`` = bow, ``y`` = port, ``z`` = up.

Quantities derived here, not read
---------------------------------
* **Deck-point velocity** ``v_pad = velocity + rel_velocity``. The environment defines
  ``rel_velocity = v_pad - v_drone`` (``rld.envs.observation``: "``rel_velocity`` ...
  ``v_pad - v_drone``"), so the pad's velocity is the **sum**, not the difference. A test
  checks the derived value against the environment's own trajectory. This is what an
  ideal (noise-free, zero-latency) ship motion reference unit plus state estimate would
  provide.
* **Deck roll and pitch** from the deck normal, inverting P1-D2's
  ``n = (-sin(pitch) cos(roll), -sin(roll), cos(pitch) cos(roll))``:
  ``roll = -asin(n_y)``, ``pitch = atan2(-n_x, n_z)``, degrees, dmf's signs (roll positive
  to starboard, pitch positive bow-up).

Units and scales
----------------
Metres and metres per second are **model** scale. Angles are radians inside the vector and
degrees on the :class:`ObsView` roll/pitch attributes, per the project convention.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np
from dmf.typedefs import FloatArray

from rld.envs.config import LandingConfig, ObservationConfig
from rld.envs.observation import obs_fields

__all__ = [
    "ACTION_DTYPE",
    "ObsLayout",
    "ObsView",
    "deck_angles_deg",
    "obs_layout",
    "setpoint_to_action",
    "view",
]

#: The dtype actions are returned in; the action space is ``Box(-1, 1, (3,), float32)``.
ACTION_DTYPE = np.float32


@dataclass(frozen=True)
class ObsLayout:
    """Offsets of every observation block, from the declared layout table.

    Attributes:
        slices: Block name to its slice of the flat vector.
        size: Total length of the vector, dimensionless.
    """

    slices: dict[str, slice]
    size: int

    def __hash__(self) -> int:
        """Hash on the ordered block offsets, so the layout can key a cache."""
        return hash(tuple((k, s.start, s.stop) for k, s in self.slices.items()))

    def block(self, obs: FloatArray, name: str) -> FloatArray:
        """Return one block as float64.

        Args:
            obs: The flat observation vector.
            name: Block name from :func:`rld.envs.observation.obs_fields`.

        Returns:
            The block, float64, in the block's declared unit and frame.

        Raises:
            KeyError: If the layout has no such block (e.g. ``in_contact`` switched off).
        """
        return np.asarray(obs[self.slices[name]], dtype=np.float64)


@cache
def obs_layout(cfg: ObservationConfig) -> ObsLayout:
    """Return the block offsets for one observation config.

    Args:
        cfg: The observation config the environment was built with.

    Returns:
        The :class:`ObsLayout`; offsets are running sums of the declared block sizes.
    """
    slices: dict[str, slice] = {}
    start = 0
    for obs_field in obs_fields(cfg):
        slices[obs_field.name] = slice(start, start + obs_field.size)
        start += obs_field.size
    return ObsLayout(slices=slices, size=start)


def _yaw_to_world(yaw_rad: float) -> FloatArray:
    """Return ``R_z(+yaw)``, the drone-yaw-to-world rotation.

    Args:
        yaw_rad: Drone yaw, radians.

    Returns:
        ``(3, 3)`` dimensionless rotation matrix; the transpose of the environment's
        world-to-drone-yaw matrix.
    """
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def deck_angles_deg(normal_world: FloatArray) -> tuple[float, float]:
    """Recover deck roll and pitch from the unit deck normal (P1-D2).

    Args:
        normal_world: Unit deck normal, dimensionless, world frame. Renormalised here, so
            float32 rounding in the observation does not bias ``asin``.

    Returns:
        ``(roll_deg, pitch_deg)``, degrees, dmf's signs: roll positive to starboard, pitch
        positive bow-up.
    """
    n = np.asarray(normal_world, dtype=np.float64).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    roll = -np.arcsin(np.clip(n[1], -1.0, 1.0))
    pitch = np.arctan2(-n[0], n[2])
    return float(np.degrees(roll)), float(np.degrees(pitch))


@dataclass(frozen=True)
class ObsView:
    """One observation, rotated to the world frame and named.

    Attributes:
        yaw_rad: Drone yaw, radians.
        velocity_m_s: Drone velocity, metres per second model scale, world frame.
        rel_position_m: ``p_pad - p_drone``, metres model scale, world frame. Its ``x, y``
            part is the lateral error a tracker drives to zero.
        rel_velocity_m_s: ``v_pad - v_drone``, metres per second model scale, world frame.
        deck_velocity_m_s: ``v_pad = velocity + rel_velocity``, metres per second model
            scale, world frame. What an ideal (noise-free, zero-latency) ship motion
            reference unit plus state estimate would provide.
        deck_normal: Unit deck normal, dimensionless, world frame.
        deck_roll_deg: Deck roll from the normal, degrees, dmf sign.
        deck_pitch_deg: Deck pitch from the normal, degrees, dmf sign.
        height_m: Signed clearance from the drone's lowest point to the pad plane along the
            deck normal, metres model scale.
        time_fraction: ``t_episode / episode_len``, dimensionless, clipped to ``[0, 1]``.
        in_contact: Whether the drone is touching the deck.
    """

    yaw_rad: float
    velocity_m_s: FloatArray
    rel_position_m: FloatArray
    rel_velocity_m_s: FloatArray
    deck_velocity_m_s: FloatArray
    deck_normal: FloatArray
    deck_roll_deg: float
    deck_pitch_deg: float
    height_m: float
    time_fraction: float
    in_contact: bool

    @property
    def lateral_error_m(self) -> FloatArray:
        """``(x, y)`` of ``p_pad - p_drone``, metres model scale, world frame."""
        return self.rel_position_m[:2]


def view(obs: FloatArray, layout: ObsLayout) -> ObsView:
    """Read one observation through its layout, in the world frame.

    Args:
        obs: The flat observation vector.
        layout: Its :class:`ObsLayout`, from :func:`obs_layout`.

    Returns:
        The :class:`ObsView`.

    Raises:
        ValueError: If ``obs`` does not have the layout's length.
    """
    vector = np.asarray(obs).reshape(-1)
    if vector.size != layout.size:
        raise ValueError(f"observation has {vector.size} entries, layout expects {layout.size}")
    yaw = float(layout.block(vector, "attitude")[2])
    rot = _yaw_to_world(yaw)
    velocity = rot @ layout.block(vector, "velocity")
    rel_velocity = rot @ layout.block(vector, "rel_velocity")
    normal = rot @ layout.block(vector, "deck_normal")
    roll_deg, pitch_deg = deck_angles_deg(normal)
    in_contact = "in_contact" in layout.slices and bool(layout.block(vector, "in_contact")[0] > 0.5)
    return ObsView(
        yaw_rad=yaw,
        velocity_m_s=velocity,
        rel_position_m=rot @ layout.block(vector, "rel_position"),
        rel_velocity_m_s=rel_velocity,
        deck_velocity_m_s=velocity + rel_velocity,
        deck_normal=normal,
        deck_roll_deg=roll_deg,
        deck_pitch_deg=pitch_deg,
        height_m=float(layout.block(vector, "height")[0]),
        time_fraction=float(layout.block(vector, "time_fraction")[0]),
        in_contact=in_contact,
    )


def setpoint_to_action(setpoint_m_s: FloatArray, landing: LandingConfig) -> FloatArray:
    """Convert a world-frame velocity setpoint to the shared normalised action.

    The inverse of ``DeckLandingAviary.velocity_setpoint_m_s`` on its range: with the norm
    cap on (P2-D2) a setpoint longer than ``v_max`` is rescaled to ``v_max`` **before**
    dividing, so the direction is preserved and ``velocity_setpoint_m_s(action)`` returns
    the capped setpoint exactly (to float32 rounding) rather than a per-axis-clipped one.
    Without the norm cap the division is followed by the environment's per-axis clip.

    Args:
        setpoint_m_s: Desired velocity, metres per second model scale, world frame, ``(3,)``.
        landing: The landing config, for ``v_max_m_s`` and ``v_max_is_norm_cap``.

    Returns:
        ``(3,)`` float32 action in ``[-1, 1]^3``.
    """
    command = np.asarray(setpoint_m_s, dtype=np.float64).reshape(3)
    v_max = landing.v_max_m_s
    if landing.v_max_is_norm_cap:
        speed = float(np.linalg.norm(command))
        if speed > v_max:
            command = command * (v_max / speed)
    action = np.clip(command / v_max, -1.0, 1.0)
    return action.astype(ACTION_DTYPE)
