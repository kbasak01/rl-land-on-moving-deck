"""Touchdown detection and the six-class outcome rule, as pure functions.

No PyBullet here: :func:`rld.envs.touchdown.classify` and the analytic geometry are pure,
and pinning them without a physics client is what lets the classification be argued about
directly instead of inferred from an episode that happened to end a certain way.

The frozen criteria are also checked against the numbers written in the implementation plan,
so that ``configs/env/success.yaml`` cannot drift from the pre-registration without a test
failing.

Units: metres and metres per second model scale, seconds model scale, degrees.
"""

import numpy as np
import pytest

from rld.envs.config import SUCCESS_CONFIG, SuccessConfig, load_success
from rld.envs.platform import PlatformSample
from rld.envs.touchdown import (
    OUTCOMES,
    DroneCollisionGeometry,
    TouchdownRecord,
    analytic_clearance_m,
    analytic_touchdown,
    classify,
    lowest_point_offset_m,
    relative_tilt_deg,
)

#: The CF2X collision cylinder (``cf2x.urdf:34``): radius 0.06 m, length 0.025 m, centred on
#: the body origin. Hard-coded *in the test only*, as the value the environment must read
#: out of the URDF.
CF2X = DroneCollisionGeometry(height_m=0.025, radius_m=0.06, z_offset_m=0.0)

#: Plate half-extents, metres model scale (P2-D3).
HALF = (0.40, 0.26, 0.01)


def _level_deck(t_model_s: float = 0.0, height_m: float = 1.0) -> PlatformSample:
    """Return a motionless, level deck sample at a given height.

    Args:
        t_model_s: Time, seconds model scale.
        height_m: Pad height, metres model scale, world frame.

    Returns:
        The :class:`~rld.envs.platform.PlatformSample`.
    """
    return PlatformSample(
        t_model_s=t_model_s,
        position_m=np.array([0.0, 0.0, height_m]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        euler_xyz_rad=np.zeros(3),
        velocity_m_s=np.zeros(3),
        angular_velocity_rad_s=np.zeros(3),
        acceleration_m_s2=np.zeros(3),
        normal=np.array([0.0, 0.0, 1.0]),
        tilt_deg=0.0,
    )


def _record(
    *,
    lateral_m: float = 0.0,
    rel_vz_normal_m_s: float = -0.1,
    rel_tilt_deg: float = 0.0,
) -> TouchdownRecord:
    """Build a touchdown record with the three criterion values under test.

    Args:
        lateral_m: Lateral offset from the pad centre, metres model scale.
        rel_vz_normal_m_s: Signed relative velocity along the deck normal, m/s model scale.
            Negative is closing.
        rel_tilt_deg: Relative tilt, degrees.

    Returns:
        The :class:`~rld.envs.touchdown.TouchdownRecord`.
    """
    return TouchdownRecord(
        t_model_s=5.0,
        t_episode_s=1.0,
        source="contact",
        rel_velocity_m_s=np.array([0.0, 0.0, rel_vz_normal_m_s]),
        rel_vz_normal_m_s=rel_vz_normal_m_s,
        rel_vz_world_z_m_s=rel_vz_normal_m_s,
        lateral_offset_m=lateral_m,
        rel_tilt_deg=rel_tilt_deg,
        abs_tilt_deg=rel_tilt_deg,
        deck_tilt_deg=0.0,
        penetration_m=-1e-4,
        n_contacts=4,
        on_plate=True,
    )


def test_success_yaml_matches_the_preregistered_criteria():
    """The four frozen numbers are the plan's four numbers.

    ``docs/IMPLEMENTATION_PLAN.md`` Phase 2 pre-registers 0.5 m/s, 0.10 m, 15 deg and 0.5 s.
    They are frozen at Gate 3 and may only change through a dated deviation entry in
    ``docs/protocol.md``; this test is what makes an undeclared change fail loudly.
    """
    cfg = load_success(SUCCESS_CONFIG)
    assert cfg.rel_vertical_velocity_max_m_s == 0.5
    assert cfg.lateral_offset_max_m == 0.10
    assert cfg.relative_tilt_max_deg == 15.0
    assert cfg.contact_dwell_min_s == 0.5
    assert cfg.crash_tilt_deg == 60.0
    # The three resolutions the criteria carry with them (P2-D5).
    assert cfg.vz_component == "deck_normal"
    assert cfg.crash_tilt_component == "absolute"
    assert cfg.hard_landing_includes_relative_tilt is True


def test_success_yaml_states_its_own_resolutions_in_prose():
    """The frozen file explains the three P2-D5 resolutions, not only their values.

    A frozen criterion that carries its own definition survives being read in Phase 7 by
    someone who was not in the room when it was decided.
    """
    text = SUCCESS_CONFIG.read_text(encoding="utf-8")
    assert "rel_vz_world_z_m_s" in text
    assert "DECK NORMAL" in text
    assert "termination_reason" in text


@pytest.mark.parametrize("outcome", OUTCOMES)
def test_every_outcome_class_is_reachable(outcome, env_success_cfg):
    """Each of the six classes is produced by some input. A class no input reaches is dead.

    ``timeout`` here is the no-touchdown branch; the environment's own truncation test in
    ``tests/test_landing_env.py`` is what checks that it arrives as ``truncated=True``.
    """
    inputs: dict[str, dict] = {
        "crash": {"record": _record(), "crash": True},
        "off_pad": {"record": _record(lateral_m=0.25)},
        "hard_landing": {"record": _record(rel_vz_normal_m_s=-0.9)},
        "bounce": {"record": _record(), "contact_lost": True},
        "success": {"record": _record(), "dwell_s": 0.6},
        "timeout": {"record": None, "timed_out": True},
    }
    kwargs = {
        "dwell_s": 0.0,
        "contact_lost": False,
        "crash": False,
        "timed_out": False,
        "cfg": env_success_cfg,
        **inputs[outcome],
    }
    record = kwargs.pop("record")
    assert classify(record, **kwargs) == outcome


def test_classification_is_first_match_wins(env_success_cfg):
    """Order matters: a record that violates several criteria gets the earliest class.

    The protocol's order is ``crash -> off_pad -> hard_landing -> bounce -> success ->
    timeout``. Without a test, a refactor that reorders the branches silently relabels every
    multi-violation landing, and the outcome fractions in every table move.
    """
    everything = _record(lateral_m=0.25, rel_vz_normal_m_s=-2.0, rel_tilt_deg=30.0)
    common = {"dwell_s": 0.0, "cfg": env_success_cfg}
    assert classify(everything, contact_lost=True, crash=True, timed_out=True, **common) == "crash"
    assert (
        classify(everything, contact_lost=True, crash=False, timed_out=True, **common) == "off_pad"
    )
    soft_but_tilted = _record(rel_vz_normal_m_s=-0.1, rel_tilt_deg=30.0)
    assert (
        classify(soft_but_tilted, contact_lost=True, crash=False, timed_out=True, **common)
        == "hard_landing"
    )


def test_hard_landing_absorbs_a_relative_tilt_violation(env_success_cfg):
    """P2-D5: a soft, on-pad landing at 20 deg relative tilt is a ``hard_landing``.

    Without this resolution such a landing would match **no** class while the six-class list
    is documented as exhaustive: it is not a crash, it is on the pad, it is not too fast, it
    does not bounce, and it is not a success because it fails criterion 3.
    """
    tilted = _record(rel_tilt_deg=20.0, rel_vz_normal_m_s=-0.05)
    outcome = classify(
        tilted,
        dwell_s=0.6,
        contact_lost=False,
        crash=False,
        timed_out=False,
        cfg=env_success_cfg,
    )
    assert outcome == "hard_landing"


def test_a_landing_that_matches_nothing_raises_rather_than_becoming_a_timeout(env_success_cfg):
    """An unclassifiable episode is a bookkeeping bug and is raised, not absorbed.

    Silently returning ``timeout`` would hide the bug inside a legitimate-looking outcome
    fraction, which is exactly the failure mode the six-class breakdown exists to prevent.
    """
    with pytest.raises(ValueError, match="no outcome class matched"):
        classify(
            None,
            dwell_s=0.0,
            contact_lost=False,
            crash=False,
            timed_out=False,
            cfg=env_success_cfg,
        )


def test_closing_speed_is_never_negative():
    """A drone leaving the deck at the contact instant has zero closing speed, not negative.

    If the sign leaked through, a bounce upward would read as a *large negative* closing
    speed and pass criterion 1 by arithmetic rather than by physics.
    """
    rising = _record(rel_vz_normal_m_s=+0.8)
    assert rising.closing_speed_normal_m_s == 0.0
    assert rising.closing_speed_world_z_m_s == 0.0
    falling = _record(rel_vz_normal_m_s=-0.8)
    assert falling.closing_speed_normal_m_s == pytest.approx(0.8)


def test_both_vz_readings_are_recorded_and_differ_on_a_tilted_deck(env_success_cfg):
    """P2-D5 commits ``rel_vz_normal_m_s`` **and** ``rel_vz_world_z_m_s`` on every episode.

    They coincide on a level deck and separate as it tilts, which is why the criterion has
    to name one -- and why the other is committed beside it, so Phase 7 can recompute the
    whole evaluation under the other definition without re-running anything.
    """
    tilt_deg = 24.49  # The grid's worst deck tilt (frigate SS6 90 deg 0 kn, P1-D1).
    tilt = np.radians(tilt_deg)
    deck = PlatformSample(
        t_model_s=0.0,
        position_m=np.array([0.0, 0.0, 1.0]),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        euler_xyz_rad=np.zeros(3),
        velocity_m_s=np.zeros(3),
        angular_velocity_rad_s=np.zeros(3),
        acceleration_m_s2=np.zeros(3),
        normal=np.array([np.sin(tilt), 0.0, np.cos(tilt)]),
        tilt_deg=tilt_deg,
    )
    descent = np.array([0.0, 0.0, -0.4])
    along_normal = float(np.dot(descent, deck.normal))
    assert along_normal == pytest.approx(-0.4 * np.cos(tilt))
    # ~9 % apart at the grid's worst tilt, exactly as the frozen file states.
    assert abs(along_normal / descent[2] - 1.0) == pytest.approx(1.0 - np.cos(tilt), abs=1e-9)
    assert 0.08 < 1.0 - np.cos(tilt) < 0.10
    assert env_success_cfg.vz_component == "deck_normal"


def test_lowest_point_offset_is_half_height_when_level_and_radius_when_on_its_side():
    """The support distance of the collision cylinder, at the two orientations we can check.

    Level, the lowest point is half the cylinder's length below the origin. Tipped 90 deg,
    it is the cylinder's radius. Anything in between interpolates as
    ``h/2 |cos| + r sin``, which is what the analytic detector uses.
    """
    up = np.array([0.0, 0.0, 1.0])
    assert lowest_point_offset_m(up, up, CF2X) == pytest.approx(0.0125)
    sideways = np.array([1.0, 0.0, 0.0])
    assert lowest_point_offset_m(sideways, up, CF2X) == pytest.approx(0.06)
    tilted = np.array([np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
    expected = 0.0125 * np.cos(np.pi / 4) + 0.06 * np.sin(np.pi / 4)
    assert lowest_point_offset_m(tilted, up, CF2X) == pytest.approx(expected)


def test_analytic_clearance_is_zero_when_the_drone_rests_on_the_pad():
    """A level drone at ``pad_z + h/2`` has exactly zero geometric clearance."""
    deck = _level_deck()
    position = np.array([0.0, 0.0, 1.0 + 0.0125])
    assert analytic_clearance_m(position, np.eye(3), deck, CF2X) == pytest.approx(0.0)
    assert analytic_clearance_m(position + np.array([0, 0, 0.1]), np.eye(3), deck, CF2X) > 0.0
    assert analytic_clearance_m(position - np.array([0, 0, 0.1]), np.eye(3), deck, CF2X) < 0.0


def test_analytic_detector_does_not_fire_past_the_plate_footprint(env_success_cfg):
    """A drone below deck level and clear of the plate's edge has not landed on anything.

    The plate is 0.80 x 0.52 m, so the y half-extent (0.26 m) is the binding one; without
    the footprint check a drone that missed the deck entirely and is falling past it would
    register a touchdown the moment it crossed the deck plane. The test point is beyond
    ``0.26 + COLLISION_R`` = 0.32 m, because the footprint is widened by the drone's radius:
    a drone whose centre is just past the edge can still strike it with its rim, and
    PyBullet reports that contact, so the analytic detector must too.
    """
    deck = _level_deck()
    below = np.array([0.0, 0.40, 1.0])
    assert (
        analytic_touchdown(
            t_model_s=0.0,
            t_episode_s=0.0,
            drone_position_m=below,
            drone_velocity_m_s=np.array([0.0, 0.0, -1.0]),
            drone_rotation=np.eye(3),
            deck=deck,
            half_extents_m=HALF,
            geometry=CF2X,
            contact_margin_m=env_success_cfg.analytic_contact_margin_m,
        )
        is None
    )
    # Just past the geometric edge but within the drone's reach: this is an edge strike,
    # the analytic detector fires, and `on_plate` is False so it classifies as a crash.
    rim = analytic_touchdown(
        t_model_s=0.0,
        t_episode_s=0.0,
        drone_position_m=np.array([0.0, 0.29, 1.0]),
        drone_velocity_m_s=np.array([0.0, 0.0, -1.0]),
        drone_rotation=np.eye(3),
        deck=deck,
        half_extents_m=HALF,
        geometry=CF2X,
        contact_margin_m=env_success_cfg.analytic_contact_margin_m,
    )
    assert rim is not None
    assert rim.on_plate is False

    on_plate = np.array([0.0, 0.20, 1.0])
    record = analytic_touchdown(
        t_model_s=0.0,
        t_episode_s=0.0,
        drone_position_m=on_plate,
        drone_velocity_m_s=np.array([0.0, 0.0, -1.0]),
        drone_rotation=np.eye(3),
        deck=deck,
        half_extents_m=HALF,
        geometry=CF2X,
        contact_margin_m=env_success_cfg.analytic_contact_margin_m,
    )
    assert record is not None
    assert record.source == "analytic"
    assert record.on_plate
    assert record.lateral_offset_m == pytest.approx(0.20)
    assert record.rel_vz_normal_m_s == pytest.approx(-1.0)


def test_relative_tilt_is_measured_against_the_deck_normal_not_world_up():
    """Criterion 3 is relative, and on a 20 deg deck the difference is 20 deg.

    A drone parallel to that deck has 0 deg relative tilt and 20 deg absolute tilt.
    P1-D1 measured the deck alone above 15 deg in 10 of 96 cells, so under an absolute
    criterion those cells would be structurally 0 % success for every method -- an
    environment artefact that would have been reported as a result.
    """
    tilt = np.radians(20.0)
    deck_normal = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    assert relative_tilt_deg(deck_normal, deck_normal) == pytest.approx(0.0, abs=1e-9)
    assert relative_tilt_deg(deck_normal, np.array([0.0, 0.0, 1.0])) == pytest.approx(20.0)


def test_success_config_rejects_a_tilt_budget_above_the_crash_threshold(tmp_path):
    """A criterion above the crash threshold would make ``hard_landing`` unreachable.

    Validated at load time rather than discovered as a missing bar in a Phase 7 table.
    """
    text = SUCCESS_CONFIG.read_text(encoding="utf-8").replace(
        "relative_tilt_max_deg: 15.0", "relative_tilt_max_deg: 75.0"
    )
    broken = tmp_path / "success.yaml"
    broken.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="must be below"):
        load_success(broken)


def test_success_config_is_frozen_and_hashable(env_success_cfg):
    """The criteria are a frozen dataclass, so nothing downstream can mutate them in place."""
    assert isinstance(env_success_cfg, SuccessConfig)
    with pytest.raises(AttributeError):
        env_success_cfg.lateral_offset_max_m = 0.5  # type: ignore[misc]
    assert hash(env_success_cfg) == hash(load_success(SUCCESS_CONFIG))
