"""Touchdown detection, by two independent paths, and the six-class outcome rule.

Why two detectors
-----------------
A landing result is only as trustworthy as the instant it is measured at. This module
detects first contact twice -- once from PyBullet's own contact manifold and once
analytically from the recorded drone state and the analytic deck state -- and the
environment records **both** on every episode whether or not they agree. Their disagreement
rate is a committed Gate 2 number (< 1 %), so a contact-detection bug shows up as a number
in a CSV instead of as a quietly wrong success rate.

The contact detector's one filter
---------------------------------
Bullet reports candidate contact points up to about 2 cm before the surfaces touch (its
``contactBreakingThreshold``), so an unfiltered ``getContactPoints`` fires early and by a
distance that depends on closing speed. Points carrying less normal force than
``SuccessConfig.contact_normal_force_min_n`` are discarded. Without that filter the two
detectors disagree for a purely numerical reason and Gate 2 fails on an artefact.

The analytic detector's geometry
--------------------------------
The CF2X collision shape is a cylinder of radius ``COLLISION_R`` = 0.06 m and length
``COLLISION_H`` = 0.025 m, centred ``COLLISION_Z_OFFSET`` = 0 m from the body origin
(``cf2x.urdf:34``). Its lowest point along a unit direction ``n`` is
``(h/2)|a . n| + r sqrt(1 - (a . n)^2)`` below the body origin, where ``a`` is the
cylinder's axis in world coordinates. The signed clearance is that support distance
subtracted from the drone's height above the pad plane along the deck normal, and touchdown
is its first non-positive crossing.

Units and scales
----------------
Every length is metres **model** scale, every velocity metres per second model scale, every
time seconds model scale, angles degrees at the boundary. The world frame is P1-D2's:
``x`` = bow, ``y`` = port, ``z`` = up.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pybullet as pyb
from dmf.typedefs import FloatArray

from rld.envs.config import SuccessConfig
from rld.envs.platform import PlatformSample

__all__ = [
    "OUTCOMES",
    "TERMINATION_REASONS",
    "ContactPoll",
    "DroneCollisionGeometry",
    "Outcome",
    "TerminationReason",
    "TouchdownRecord",
    "analytic_clearance_m",
    "analytic_touchdown",
    "classify",
    "contact_touchdown",
    "ground_contact",
    "lowest_point_offset_m",
    "poll_contacts",
    "relative_tilt_deg",
]

#: The six outcome classes, exhaustive and mutually exclusive, in the order
#: :func:`classify` matches them (first match wins).
type Outcome = Literal["crash", "off_pad", "hard_landing", "bounce", "success", "timeout"]

#: Tuple form of :data:`Outcome`, in match order. Used for the committed column order in
#: ``results/e00_env_sanity.csv`` so the six fractions always appear in the same places.
OUTCOMES: tuple[Outcome, ...] = (
    "crash",
    "off_pad",
    "hard_landing",
    "bounce",
    "success",
    "timeout",
)

#: Free-text reason recorded beside every outcome (P2-D5). Keeps the ``crash`` bar
#: decomposable without adding a seventh class.
type TerminationReason = Literal[
    "ground_contact",
    "off_plate_strike",
    "tilt_gt_crash",
    "below_deck",
    "out_of_bounds",
    "dwell_complete",
    "release",
    "time_limit",
]

#: Tuple form of :data:`TerminationReason`.
TERMINATION_REASONS: tuple[TerminationReason, ...] = (
    "ground_contact",
    "off_plate_strike",
    "tilt_gt_crash",
    "below_deck",
    "out_of_bounds",
    "dwell_complete",
    "release",
    "time_limit",
)


@dataclass(frozen=True)
class DroneCollisionGeometry:
    """The drone's collision cylinder, as parsed from its URDF by ``BaseAviary``.

    Attributes:
        height_m: Cylinder length ``COLLISION_H``, metres. 0.025 m for the CF2X.
        radius_m: Cylinder radius ``COLLISION_R``, metres. 0.06 m for the CF2X.
        z_offset_m: Cylinder centre offset from the body origin along body z,
            ``COLLISION_Z_OFFSET``, metres. 0.0 m for the CF2X.
    """

    height_m: float
    radius_m: float
    z_offset_m: float


@dataclass(frozen=True)
class TouchdownRecord:
    """The frozen state of one touchdown event, from one of the two detectors.

    Attributes:
        t_model_s: Time of the event, seconds model scale, measured from the start of the
            committed dmf record.
        t_episode_s: Time of the event, seconds model scale, measured from the start of the
            episode.
        source: Which detector produced it, ``"contact"`` or ``"analytic"``.
        rel_velocity_m_s: ``v_drone - v_pad``, metres per second model scale, world frame.
        rel_vz_normal_m_s: Signed ``(v_drone - v_pad) . n_deck``, metres per second model
            scale. **Negative means closing.** This is the reading the frozen criterion
            uses (P2-D5, ``vz_component: deck_normal``).
        rel_vz_world_z_m_s: Signed ``(v_drone - v_pad) . z_hat``, metres per second model
            scale. The other reading, committed so the evaluation is recomputable under it.
        lateral_offset_m: Distance from the pad centre in the deck plane, metres model
            scale, measured to the drone **centre**.
        rel_tilt_deg: Angle between the drone's body z and the deck normal, degrees.
        abs_tilt_deg: Angle between the drone's body z and world ``+z``, degrees. The
            ``crash`` threshold is read on this one.
        deck_tilt_deg: Angle between the deck normal and world ``+z``, degrees.
        penetration_m: Signed separation at the event, metres model scale. Negative is
            penetration; ``< -tunnelling_penetration_m`` is flagged as tunnelling.
        n_contacts: Number of surviving contact points. Zero for the analytic detector.
        on_plate: Whether the drone centre was over the plate's footprint.
    """

    t_model_s: float
    t_episode_s: float
    source: Literal["contact", "analytic"]
    rel_velocity_m_s: FloatArray
    rel_vz_normal_m_s: float
    rel_vz_world_z_m_s: float
    lateral_offset_m: float
    rel_tilt_deg: float
    abs_tilt_deg: float
    deck_tilt_deg: float
    penetration_m: float
    n_contacts: int
    on_plate: bool

    @property
    def closing_speed_normal_m_s(self) -> float:
        """Closing speed into the deck surface, metres per second model scale, ``>= 0``.

        ``max(0, -rel_vz_normal_m_s)``: a drone moving *away* from the deck at the instant
        of contact has zero closing speed, not a negative one, so the criterion is never
        satisfied by a sign error.
        """
        return max(0.0, -self.rel_vz_normal_m_s)

    @property
    def closing_speed_world_z_m_s(self) -> float:
        """Closing speed under the world-z reading, metres per second model scale, ``>= 0``."""
        return max(0.0, -self.rel_vz_world_z_m_s)


def _unit(vector: FloatArray) -> FloatArray:
    """Return a unit vector, dimensionless.

    Args:
        vector: Any non-degenerate 3-vector.

    Returns:
        ``vector / ||vector||``; the input unchanged if its norm underflows.
    """
    norm = float(np.linalg.norm(vector))
    return vector if norm < 1e-12 else np.asarray(vector / norm, dtype=np.float64)


def relative_tilt_deg(drone_body_z: FloatArray, reference: FloatArray) -> float:
    """Return the angle between the drone's body z axis and a reference direction.

    Args:
        drone_body_z: The drone's body z axis in world coordinates, dimensionless.
        reference: The reference direction in world coordinates (the deck normal for the
            relative criterion, world ``+z`` for the ``crash`` threshold), dimensionless.

    Returns:
        The angle in ``[0, 180]``, degrees.
    """
    cosine = float(np.clip(np.dot(_unit(drone_body_z), _unit(reference)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def lowest_point_offset_m(
    drone_body_z: FloatArray, direction: FloatArray, geometry: DroneCollisionGeometry
) -> float:
    """Return how far the drone's collision cylinder reaches below its body origin.

    Args:
        drone_body_z: The cylinder's axis in world coordinates, dimensionless.
        direction: The unit direction "up" out of the deck (the deck normal),
            dimensionless.
        geometry: The drone's collision cylinder.

    Returns:
        The support distance along ``-direction`` from the body origin, metres model scale,
        always non-negative.
    """
    axis = _unit(drone_body_z)
    normal = _unit(direction)
    cosine = float(np.clip(np.dot(axis, normal), -1.0, 1.0))
    sine = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
    half_height = 0.5 * geometry.height_m
    support = half_height * abs(cosine) + geometry.radius_m * sine
    return float(support - geometry.z_offset_m * cosine)


def analytic_clearance_m(
    drone_position_m: FloatArray,
    drone_rotation: FloatArray,
    deck: PlatformSample,
    geometry: DroneCollisionGeometry,
) -> float:
    """Return the signed clearance from the drone's lowest point to the pad plane.

    Args:
        drone_position_m: Drone body-origin position, metres model scale, world frame.
        drone_rotation: The drone's body-to-world rotation matrix, shape ``(3, 3)``,
            dimensionless.
        deck: The analytic deck state at the same instant.
        geometry: The drone's collision cylinder.

    Returns:
        Clearance in metres model scale: positive above the deck plane, zero at touch,
        negative once the shapes overlap.
    """
    normal = _unit(deck.normal)
    offset = np.asarray(drone_position_m, dtype=np.float64) - np.asarray(
        deck.position_m, dtype=np.float64
    )
    height = float(np.dot(offset, normal))
    return height - lowest_point_offset_m(drone_rotation[:, 2], normal, geometry)


def _deck_frame_offset_m(
    drone_position_m: FloatArray, deck: PlatformSample
) -> tuple[float, float, float]:
    """Return the drone's offset from the pad centre expressed in the deck's body frame.

    Args:
        drone_position_m: Drone body-origin position, metres model scale, world frame.
        deck: The analytic deck state at the same instant.

    Returns:
        ``(dx, dy, dn)`` metres model scale: along the deck's own x (bow) and y (port) axes
        and along its normal.
    """
    rot = np.asarray(pyb.getMatrixFromQuaternion(list(deck.quaternion)), dtype=np.float64).reshape(
        3, 3
    )
    offset = np.asarray(drone_position_m, dtype=np.float64) - np.asarray(
        deck.position_m, dtype=np.float64
    )
    local = rot.T @ offset
    return (float(local[0]), float(local[1]), float(local[2]))


def _build_record(
    *,
    source: Literal["contact", "analytic"],
    t_model_s: float,
    t_episode_s: float,
    drone_position_m: FloatArray,
    drone_velocity_m_s: FloatArray,
    drone_rotation: FloatArray,
    deck: PlatformSample,
    half_extents_m: tuple[float, float, float],
    penetration_m: float,
    n_contacts: int,
) -> TouchdownRecord:
    """Assemble one :class:`TouchdownRecord` from states both detectors already have.

    Both detectors describe the *same* event with the *same* quantities; only the trigger
    differs. Sharing this builder is what makes their disagreement a statement about timing
    rather than about bookkeeping.

    Args:
        source: Which detector fired.
        t_model_s: Event time, seconds model scale, from the start of the dmf record.
        t_episode_s: Event time, seconds model scale, from the start of the episode.
        drone_position_m: Drone body-origin position, metres model scale, world frame.
        drone_velocity_m_s: Drone linear velocity, metres per second model scale, world
            frame.
        drone_rotation: The drone's body-to-world rotation matrix, shape ``(3, 3)``.
        deck: The analytic deck state at the event.
        half_extents_m: Plate half-extents ``(hx, hy, hz)``, metres model scale.
        penetration_m: Signed separation, metres model scale, negative when overlapping.
        n_contacts: Surviving contact points; zero for the analytic detector.

    Returns:
        The frozen record.
    """
    rel_velocity = np.asarray(drone_velocity_m_s, dtype=np.float64) - np.asarray(
        deck.velocity_m_s, dtype=np.float64
    )
    normal = _unit(deck.normal)
    dx, dy, _ = _deck_frame_offset_m(drone_position_m, deck)
    return TouchdownRecord(
        t_model_s=float(t_model_s),
        t_episode_s=float(t_episode_s),
        source=source,
        rel_velocity_m_s=rel_velocity,
        rel_vz_normal_m_s=float(np.dot(rel_velocity, normal)),
        rel_vz_world_z_m_s=float(rel_velocity[2]),
        lateral_offset_m=float(np.hypot(dx, dy)),
        rel_tilt_deg=relative_tilt_deg(drone_rotation[:, 2], normal),
        abs_tilt_deg=relative_tilt_deg(drone_rotation[:, 2], np.array([0.0, 0.0, 1.0])),
        deck_tilt_deg=float(deck.tilt_deg),
        penetration_m=float(penetration_m),
        n_contacts=int(n_contacts),
        on_plate=bool(abs(dx) <= half_extents_m[0] and abs(dy) <= half_extents_m[1]),
    )


@dataclass(frozen=True)
class ContactPoll:
    """One physics substep's drone contacts, from a single PyBullet query.

    One query per substep rather than two: ``getContactPoints(bodyA=drone)`` returns every
    contact the drone is in, and partitioning it here saves 240 round trips per simulated
    second compared with querying the deck and the ground separately.

    Attributes:
        deck_distances: Signed separations of the surviving drone-deck contact points,
            metres model scale. Negative is penetration. Empty means no deck contact.
        touches_ground: Whether any surviving contact is with PyBullet's ground plane.
            Landing on the sea surface is a ``crash``, never a landing, and is counted
            separately so the two can never be confused.
    """

    deck_distances: tuple[float, ...]
    touches_ground: bool


def poll_contacts(
    *, client: int, drone_id: int, deck_id: int, plane_id: int, cfg: SuccessConfig
) -> ContactPoll:
    """Read and partition the drone's contacts for one physics substep.

    Args:
        client: PyBullet physics client id.
        drone_id: Drone body id.
        deck_id: Deck plate body id.
        plane_id: Ground-plane body id.
        cfg: The frozen criteria; only ``contact_normal_force_min_n`` is read here.

    Returns:
        The partitioned :class:`ContactPoll`.
    """
    points = pyb.getContactPoints(bodyA=drone_id, physicsClientId=client)
    deck: list[float] = []
    ground = False
    for point in points:
        if float(point[9]) <= cfg.contact_normal_force_min_n:
            continue
        other = int(point[2])
        if other == deck_id:
            deck.append(float(point[8]))
        elif other == plane_id:
            ground = True
    return ContactPoll(deck_distances=tuple(deck), touches_ground=ground)


def contact_touchdown(
    *,
    poll: ContactPoll,
    t_model_s: float,
    t_episode_s: float,
    drone_position_m: FloatArray,
    drone_velocity_m_s: FloatArray,
    drone_rotation: FloatArray,
    deck: PlatformSample,
    half_extents_m: tuple[float, float, float],
) -> TouchdownRecord | None:
    """Detect drone-versus-deck contact from PyBullet's contact manifold.

    Args:
        poll: This substep's partitioned contacts, from :func:`poll_contacts`.
        t_model_s: Current time, seconds model scale, from the start of the dmf record.
        t_episode_s: Current time, seconds model scale, from the start of the episode.
        drone_position_m: Drone body-origin position, metres model scale, world frame.
        drone_velocity_m_s: Drone linear velocity, metres per second model scale, world
            frame.
        drone_rotation: The drone's body-to-world rotation matrix, shape ``(3, 3)``.
        deck: The analytic deck state at the same instant.
        half_extents_m: Plate half-extents, metres model scale.

    Returns:
        A :class:`TouchdownRecord` if at least one contact point survived the normal-force
        filter, otherwise ``None``.
    """
    distances = poll.deck_distances
    if not distances:
        return None
    return _build_record(
        source="contact",
        t_model_s=t_model_s,
        t_episode_s=t_episode_s,
        drone_position_m=drone_position_m,
        drone_velocity_m_s=drone_velocity_m_s,
        drone_rotation=drone_rotation,
        deck=deck,
        half_extents_m=half_extents_m,
        penetration_m=min(distances),
        n_contacts=len(distances),
    )


def analytic_touchdown(
    *,
    t_model_s: float,
    t_episode_s: float,
    drone_position_m: FloatArray,
    drone_velocity_m_s: FloatArray,
    drone_rotation: FloatArray,
    deck: PlatformSample,
    half_extents_m: tuple[float, float, float],
    geometry: DroneCollisionGeometry,
    contact_margin_m: float,
) -> TouchdownRecord | None:
    """Detect touchdown from recorded states only, with no PyBullet contact query.

    Args:
        t_model_s: Current time, seconds model scale, from the start of the dmf record.
        t_episode_s: Current time, seconds model scale, from the start of the episode.
        drone_position_m: Drone body-origin position, metres model scale, world frame.
        drone_velocity_m_s: Drone linear velocity, metres per second model scale, world
            frame.
        drone_rotation: The drone's body-to-world rotation matrix, shape ``(3, 3)``.
        deck: The analytic deck state at the same instant.
        half_extents_m: Plate half-extents, metres model scale. A drone can touch the plate
            while its *centre* is already past the edge -- its collision cylinder reaches
            ``COLLISION_R`` = 0.06 m further -- so the footprint test is widened by the
            drone's radius. Without that the two detectors disagree on exactly the edge
            strikes, which is where they were measured disagreeing (1 of 800 sanity
            episodes, drone centre at 0.453 m against a 0.40 m half-extent). The record's
            ``on_plate`` flag stays on the **centre**, so the classification of an edge
            strike as ``crash``/``off_plate_strike`` is unchanged.
        geometry: The drone's collision cylinder.
        contact_margin_m: Where PyBullet places the surface, metres model scale
            (``SuccessConfig.analytic_contact_margin_m``). Bullet's collision margin keeps
            resting bodies apart by this much, so a purely geometric trigger at zero would
            never fire on a soft landing.

    Returns:
        A :class:`TouchdownRecord` if the clearance has reached the contact margin over the
        plate, otherwise ``None``.
    """
    clearance = analytic_clearance_m(drone_position_m, drone_rotation, deck, geometry)
    if clearance > contact_margin_m:
        return None
    dx, dy, _ = _deck_frame_offset_m(drone_position_m, deck)
    reach = geometry.radius_m
    if abs(dx) > half_extents_m[0] + reach or abs(dy) > half_extents_m[1] + reach:
        return None
    return _build_record(
        source="analytic",
        t_model_s=t_model_s,
        t_episode_s=t_episode_s,
        drone_position_m=drone_position_m,
        drone_velocity_m_s=drone_velocity_m_s,
        drone_rotation=drone_rotation,
        deck=deck,
        half_extents_m=half_extents_m,
        penetration_m=clearance,
        n_contacts=0,
    )


def ground_contact(*, client: int, drone_id: int, plane_id: int, cfg: SuccessConfig) -> bool:
    """Report whether the drone is touching PyBullet's ground plane.

    Landing on the sea surface is never a landing: it is a ``crash`` with
    ``termination_reason = ground_contact``, and it is counted separately from deck
    contacts so that the two can never be confused.

    Args:
        client: PyBullet physics client id.
        drone_id: Drone body id.
        plane_id: Ground-plane body id.
        cfg: The frozen criteria; only ``contact_normal_force_min_n`` is read here.

    Returns:
        True if at least one drone-ground contact survives the normal-force filter.
    """
    points = pyb.getContactPoints(bodyA=drone_id, bodyB=plane_id, physicsClientId=client)
    return any(float(point[9]) > cfg.contact_normal_force_min_n for point in points)


def classify(
    record: TouchdownRecord | None,
    *,
    dwell_s: float,
    contact_lost: bool,
    crash: bool,
    timed_out: bool,
    cfg: SuccessConfig,
) -> Outcome:
    """Assign one of the six outcome classes. A pure function of its arguments.

    The classes are exhaustive and mutually exclusive, and the **first match wins** in the
    protocol's order::

        crash -> off_pad -> hard_landing -> bounce -> success -> timeout

    ``hard_landing`` covers a violation of *either* touchdown-quality criterion, vertical
    velocity **or** relative tilt (P2-D5). Without that, an on-pad, soft, 20 deg-tilt
    landing that dwells its 0.5 s would match no class at all while the list is documented
    as exhaustive.

    Args:
        record: The first-contact record, or ``None`` if the drone never touched down.
        dwell_s: Time in contact since first contact, seconds model scale.
        contact_lost: Whether contact was released for longer than the grace period before
            the dwell was complete.
        crash: Whether a loss-of-control or divergence condition fired (excessive tilt,
            ground contact, a strike beyond the plate footprint, below-deck, or leaving the
            divergence bounds).
        timed_out: Whether the flight budget expired.
        cfg: The frozen criteria.

    Returns:
        The outcome class.

    Raises:
        ValueError: If no class matches. That is a bug in the environment's episode
            bookkeeping, not a new outcome, so it is raised rather than absorbed into
            ``timeout``.
    """
    if crash:
        return "crash"
    if record is not None:
        if record.lateral_offset_m > cfg.lateral_offset_max_m:
            return "off_pad"
        too_fast = record.closing_speed_normal_m_s > cfg.rel_vertical_velocity_max_m_s
        too_tilted = (
            cfg.hard_landing_includes_relative_tilt
            and record.rel_tilt_deg > cfg.relative_tilt_max_deg
        )
        if too_fast or too_tilted:
            return "hard_landing"
        if contact_lost:
            return "bounce"
        if dwell_s >= cfg.contact_dwell_min_s:
            return "success"
    if timed_out:
        return "timeout"
    raise ValueError(
        "no outcome class matched: no touchdown, no crash and no timeout. The episode "
        "ended for a reason the classifier does not know about."
    )
