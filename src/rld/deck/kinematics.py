"""Deck-point kinematics: pose, velocity and acceleration of a pad on a 3-DOF deck.

Frame and sign convention
-------------------------
**World frame (the PyBullet frame every downstream phase uses):** ``x`` = bow (forward),
``y`` = **port**, ``z`` = up. Right-handed.

**dmf's channels** (``third_party/deck-motion-forecast/docs/corpus_card.md``): ``roll`` in
degrees, positive to **starboard** (starboard side down); ``pitch`` in degrees, positive
**bow-up**; ``heave`` in metres, positive up. Rates are the degrees-per-second and
metres-per-second derivatives of those channels.

**Mapping into the world frame.** Rotating about ``+x`` (bow) by ``phi`` lifts port and
drops starboard, and rotating about ``+y`` (port) by a **negative** angle lifts the bow, so

    phi_world   = +radians(roll_deg)
    theta_world = -radians(pitch_deg)
    psi_world   =  0                      (dmf has no yaw; the deck is 3-DOF, plan D0.3)

**Rotation order is ZYX** -- ``R = R_z(psi) R_y(theta) R_x(phi)``, i.e. here
``R = R_y(theta_world) R_x(phi_world)``. That is both the ship convention and
``pybullet.getQuaternionFromEuler([phi_world, theta_world, 0.0])``, which is what makes the
Phase 2 platform body carry exactly this matrix. Under ZYX, ``R_x`` fixes any point on the
``x`` axis, so a centreline pad ``(x, 0, 0)`` is **exactly roll-blind** and

    z_pad = heave + x_pad * sin(pitch_dmf)          # x_pad signed, negative = aft

with no ``cos(roll)`` factor. The sign is ``+``: with ``x_pad < 0`` (aft) a bow-up pitch
moves the pad **down** and **forward**. The opposite sign, written in plan D0.5 and in the
``deck-scaling-physics`` skill, makes an aft pad rise when the bow rises; see
``docs/protocol.md`` P1-D2. The order matters too: ``R_x R_y`` would give a centreline aft
pad a spurious lateral offset at finite roll.

**Angular velocity** of that composition, in world axes, is

    omega = theta_dot * y_hat + phi_dot * R_y(theta) x_hat
          = (phi_dot*cos(theta), theta_dot, -phi_dot*sin(theta))

and the pad follows ``p = [0,0,heave] + R r_pad`` exactly, so

    v = [0,0,heave_rate]  + omega x p_offset
    a = [0,0,heave_acc]   + alpha x p_offset + omega x (omega x p_offset)

with no finite differences anywhere. ``alpha`` needs angular accelerations, which
:mod:`rld.deck.bridge` reconstructs analytically from dmf's public transfer functions.

Scales and units
----------------
This module is **scale-agnostic pure geometry**: angles in degrees, angular rates in
degrees per second, angular accelerations in degrees per second squared, lengths in metres,
velocities in metres per second, accelerations in metres per second squared -- all at one
consistent scale, which the caller states. :mod:`rld.deck.bridge` evaluates it at **full
scale** with a full-scale lever arm and Froude-scales the result to model scale afterwards.
Radians appear only inside this module.
"""

from dataclasses import dataclass

import numpy as np
from dmf.typedefs import FloatArray

from rld.deck.scaling import FroudeScale

__all__ = [
    "DeckPointState",
    "angular_acceleration_rad_s2",
    "angular_velocity_rad_s",
    "deck_normal",
    "deck_point_state",
    "euler_xyz_rad",
    "rotation_matrix",
    "tilt_deg",
    "to_model_state",
]


@dataclass(frozen=True)
class DeckPointState:
    """Pose and motion of one deck point, in the world frame at one consistent scale.

    Leading dimensions broadcast: a scalar sample gives shape ``(3,)`` vectors and a record
    of ``n`` samples gives ``(n, 3)``.

    Attributes:
        position_m: Pad position, metres, world frame (x bow, y port, z up), relative to the
            vessel's mean-position CG.
        velocity_m_s: Pad velocity, metres per second, world frame. Analytic, never
            finite-differenced.
        acceleration_m_s2: Pad acceleration, metres per second squared, world frame.
        euler_xyz_rad: ``(phi_world, theta_world, 0)`` radians -- exactly the triple to hand
            to ``pybullet.getQuaternionFromEuler``.
        angular_velocity_rad_s: Deck angular velocity, radians per second, world axes.
        angular_acceleration_rad_s2: Deck angular acceleration, radians per second squared,
            world axes.
        normal: Unit deck normal, dimensionless, world frame. ``x`` component is negative
            when the bow is up.
        tilt_deg: Angle between the deck normal and world ``+z``, degrees. Compared against
            the drone's own tilt by the Phase 2 success criterion.
    """

    position_m: FloatArray
    velocity_m_s: FloatArray
    acceleration_m_s2: FloatArray
    euler_xyz_rad: FloatArray
    angular_velocity_rad_s: FloatArray
    angular_acceleration_rad_s2: FloatArray
    normal: FloatArray
    tilt_deg: FloatArray


def _world_angles(roll_deg: FloatArray, pitch_deg: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Convert dmf's roll and pitch, degrees, to world-frame Euler angles, radians.

    Args:
        roll_deg: dmf roll, degrees, positive to starboard.
        pitch_deg: dmf pitch, degrees, positive bow-up.

    Returns:
        ``(phi_world, theta_world)`` radians: rotation about world ``+x`` (bow) and about
        world ``+y`` (port). ``theta_world`` carries the sign flip that makes bow-up a
        negative rotation about the port axis.
    """
    phi = np.radians(np.asarray(roll_deg, dtype=np.float64))
    theta = -np.radians(np.asarray(pitch_deg, dtype=np.float64))
    return phi, theta


def euler_xyz_rad(roll_deg: FloatArray, pitch_deg: FloatArray) -> FloatArray:
    """Return the world-frame ZYX Euler triple for PyBullet.

    Args:
        roll_deg: dmf roll, degrees, positive to starboard.
        pitch_deg: dmf pitch, degrees, positive bow-up.

    Returns:
        ``(phi, theta, 0)`` radians, shape ``(..., 3)``, to be passed unchanged to
        ``pybullet.getQuaternionFromEuler``.
    """
    phi, theta = _world_angles(roll_deg, pitch_deg)
    phi, theta = np.broadcast_arrays(phi, theta)
    return np.stack([phi, theta, np.zeros_like(phi)], axis=-1)


def rotation_matrix(roll_deg: FloatArray, pitch_deg: FloatArray) -> FloatArray:
    """Return the deck's body-to-world rotation ``R = R_y(theta) R_x(phi)`` (ZYX).

    Args:
        roll_deg: dmf roll, degrees, positive to starboard.
        pitch_deg: dmf pitch, degrees, positive bow-up.

    Returns:
        Rotation matrices, dimensionless, shape ``(..., 3, 3)``, mapping body-frame vectors
        (x bow, y port, z up) into the world frame. Equal to
        ``getMatrixFromQuaternion(getQuaternionFromEuler([phi, theta, 0]))``.
    """
    phi, theta = _world_angles(roll_deg, pitch_deg)
    phi, theta = np.broadcast_arrays(phi, theta)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    zero = np.zeros_like(phi)
    rows = [
        [cos_theta, sin_theta * sin_phi, sin_theta * cos_phi],
        [zero, cos_phi, -sin_phi],
        [-sin_theta, cos_theta * sin_phi, cos_theta * cos_phi],
    ]
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def deck_normal(roll_deg: FloatArray, pitch_deg: FloatArray) -> FloatArray:
    """Return the unit deck normal in the world frame.

    Args:
        roll_deg: dmf roll, degrees, positive to starboard.
        pitch_deg: dmf pitch, degrees, positive bow-up.

    Returns:
        ``R @ (0, 0, 1)`` = ``(sin(theta)cos(phi), -sin(phi), cos(theta)cos(phi))``,
        dimensionless, shape ``(..., 3)``. Bow-up gives a negative ``x`` component; roll to
        starboard tilts the normal toward starboard (negative ``y``).
    """
    phi, theta = _world_angles(roll_deg, pitch_deg)
    phi, theta = np.broadcast_arrays(phi, theta)
    return np.stack(
        [np.sin(theta) * np.cos(phi), -np.sin(phi), np.cos(theta) * np.cos(phi)], axis=-1
    )


def tilt_deg(roll_deg: FloatArray, pitch_deg: FloatArray) -> FloatArray:
    """Return the angle between the deck normal and world ``+z``, degrees.

    Args:
        roll_deg: dmf roll, degrees, positive to starboard.
        pitch_deg: dmf pitch, degrees, positive bow-up.

    Returns:
        Tilt magnitude, degrees, shape ``(...)``, in ``[0, 180]``. Equal to
        ``degrees(arccos(cos(phi)cos(theta)))``.
    """
    normal_z = deck_normal(roll_deg, pitch_deg)[..., 2]
    return np.degrees(np.arccos(np.clip(normal_z, -1.0, 1.0)))


def angular_velocity_rad_s(
    roll_deg: FloatArray,
    pitch_deg: FloatArray,
    roll_rate_dps: FloatArray,
    pitch_rate_dps: FloatArray,
) -> FloatArray:
    """Return the deck's angular velocity in world axes, radians per second.

    For ``R = R_y(theta) R_x(phi)``, ``omega = theta_dot*y_hat + phi_dot*R_y(theta)*x_hat``.

    Args:
        roll_deg: dmf roll, degrees.
        pitch_deg: dmf pitch, degrees.
        roll_rate_dps: dmf roll rate, degrees per second.
        pitch_rate_dps: dmf pitch rate, degrees per second.

    Returns:
        ``(phi_dot*cos(theta), theta_dot, -phi_dot*sin(theta))`` radians per second, shape
        ``(..., 3)``, world axes (x bow, y port, z up).
    """
    _, theta = _world_angles(roll_deg, pitch_deg)
    phi_dot = np.radians(np.asarray(roll_rate_dps, dtype=np.float64))
    theta_dot = -np.radians(np.asarray(pitch_rate_dps, dtype=np.float64))
    theta, phi_dot, theta_dot = np.broadcast_arrays(theta, phi_dot, theta_dot)
    return np.stack([phi_dot * np.cos(theta), theta_dot, -phi_dot * np.sin(theta)], axis=-1)


def angular_acceleration_rad_s2(
    roll_deg: FloatArray,
    pitch_deg: FloatArray,
    roll_rate_dps: FloatArray,
    pitch_rate_dps: FloatArray,
    roll_acc_dps2: FloatArray,
    pitch_acc_dps2: FloatArray,
) -> FloatArray:
    """Return the deck's angular acceleration in world axes, radians per second squared.

    The exact time derivative of :func:`angular_velocity_rad_s`, so the angular
    accelerations must come from dmf's analytic transfer functions
    (:mod:`rld.deck.bridge`), never from differencing.

    Args:
        roll_deg: dmf roll, degrees.
        pitch_deg: dmf pitch, degrees.
        roll_rate_dps: dmf roll rate, degrees per second.
        pitch_rate_dps: dmf pitch rate, degrees per second.
        roll_acc_dps2: dmf roll acceleration, degrees per second squared.
        pitch_acc_dps2: dmf pitch acceleration, degrees per second squared.

    Returns:
        ``(phi_ddot*cos(theta) - phi_dot*theta_dot*sin(theta), theta_ddot,
        -phi_ddot*sin(theta) - phi_dot*theta_dot*cos(theta))``, radians per second squared,
        shape ``(..., 3)``.
    """
    _, theta = _world_angles(roll_deg, pitch_deg)
    phi_dot = np.radians(np.asarray(roll_rate_dps, dtype=np.float64))
    theta_dot = -np.radians(np.asarray(pitch_rate_dps, dtype=np.float64))
    phi_ddot = np.radians(np.asarray(roll_acc_dps2, dtype=np.float64))
    theta_ddot = -np.radians(np.asarray(pitch_acc_dps2, dtype=np.float64))
    theta, phi_dot, theta_dot, phi_ddot, theta_ddot = np.broadcast_arrays(
        theta, phi_dot, theta_dot, phi_ddot, theta_ddot
    )
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return np.stack(
        [
            phi_ddot * cos_theta - phi_dot * theta_dot * sin_theta,
            theta_ddot,
            -phi_ddot * sin_theta - phi_dot * theta_dot * cos_theta,
        ],
        axis=-1,
    )


def deck_point_state(
    *,
    roll_deg: FloatArray,
    pitch_deg: FloatArray,
    heave_m: FloatArray,
    roll_rate_dps: FloatArray,
    pitch_rate_dps: FloatArray,
    heave_rate_m_s: FloatArray,
    roll_acc_dps2: FloatArray,
    pitch_acc_dps2: FloatArray,
    heave_acc_m_s2: FloatArray,
    r_pad_m: tuple[float, float, float],
) -> DeckPointState:
    """Evaluate one deck point's pose, velocity and acceleration analytically.

    All inputs are at **one consistent scale**, which the caller states; the outputs are at
    that same scale. :mod:`rld.deck.bridge` calls this at full scale with a full-scale lever
    arm and Froude-scales afterwards.

    Args:
        roll_deg: dmf roll, degrees, positive to starboard.
        pitch_deg: dmf pitch, degrees, positive bow-up.
        heave_m: dmf heave, metres, positive up.
        roll_rate_dps: dmf roll rate, degrees per second.
        pitch_rate_dps: dmf pitch rate, degrees per second.
        heave_rate_m_s: dmf heave rate, metres per second.
        roll_acc_dps2: Roll acceleration, degrees per second squared.
        pitch_acc_dps2: Pitch acceleration, degrees per second squared.
        heave_acc_m_s2: dmf heave acceleration, metres per second squared.
        r_pad_m: Pad offset from the CG in the **body** frame, metres: ``(x, y, z)`` with
            x = bow, y = port, z = up. Negative x is aft, so the default aft pad is
            ``(-0.4*L, 0, 0)``.

    Returns:
        The :class:`DeckPointState`, world frame, same scale as the inputs.
    """
    rot = rotation_matrix(roll_deg, pitch_deg)
    offset = np.einsum("...ij,j->...i", rot, np.asarray(r_pad_m, dtype=np.float64))
    omega = angular_velocity_rad_s(roll_deg, pitch_deg, roll_rate_dps, pitch_rate_dps)
    alpha = angular_acceleration_rad_s2(
        roll_deg, pitch_deg, roll_rate_dps, pitch_rate_dps, roll_acc_dps2, pitch_acc_dps2
    )
    heave = np.broadcast_to(np.asarray(heave_m, dtype=np.float64), offset.shape[:-1])
    heave_rate = np.broadcast_to(np.asarray(heave_rate_m_s, dtype=np.float64), offset.shape[:-1])
    heave_acc = np.broadcast_to(np.asarray(heave_acc_m_s2, dtype=np.float64), offset.shape[:-1])
    zero = np.zeros_like(heave)
    position = offset + np.stack([zero, zero, heave], axis=-1)
    velocity = np.stack([zero, zero, heave_rate], axis=-1) + np.cross(omega, offset)
    acceleration = (
        np.stack([zero, zero, heave_acc], axis=-1)
        + np.cross(alpha, offset)
        + np.cross(omega, np.cross(omega, offset))
    )
    return DeckPointState(
        position_m=position,
        velocity_m_s=velocity,
        acceleration_m_s2=acceleration,
        euler_xyz_rad=euler_xyz_rad(roll_deg, pitch_deg),
        angular_velocity_rad_s=omega,
        angular_acceleration_rad_s2=alpha,
        normal=deck_normal(roll_deg, pitch_deg),
        tilt_deg=tilt_deg(roll_deg, pitch_deg),
    )


def to_model_state(state_full: DeckPointState, scale: FroudeScale) -> DeckPointState:
    """Froude-scale a full-scale deck-point state to model scale.

    Args:
        state_full: A :class:`~rld.deck.kinematics.DeckPointState` at full scale.
        scale: The Froude scale.

    Returns:
        The same state at model scale: ``position_m`` x ``lam``, ``velocity_m_s`` x
        ``sqrt(lam)``, ``acceleration_m_s2`` x 1, ``angular_velocity_rad_s`` x
        ``1/sqrt(lam)``, ``angular_acceleration_rad_s2`` x ``1/lam``, and the angles,
        normal and tilt unchanged. The angular-acceleration factor is not in plan D0.1's
        table; it is derived, ``angular_rate / time = lam**-0.5 / lam**0.5``.
    """
    return DeckPointState(
        position_m=scale.length(state_full.position_m),
        velocity_m_s=scale.velocity(state_full.velocity_m_s),
        acceleration_m_s2=scale.acceleration(state_full.acceleration_m_s2),
        euler_xyz_rad=state_full.euler_xyz_rad,
        angular_velocity_rad_s=scale.angular_rate(state_full.angular_velocity_rad_s),
        angular_acceleration_rad_s2=state_full.angular_acceleration_rad_s2 / scale.lam,
        normal=state_full.normal,
        tilt_deg=state_full.tilt_deg,
    )
