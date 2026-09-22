"""Deck-point kinematics: the sign convention and the rotation order, pinned by hand.

Three hand-computed cases arbitrate the two corrections this phase rests on
(``docs/protocol.md`` P1-D2). All three are **full scale** on the frigate, whose aft pad sits
at ``x_pad = -0.4 x 124.0 = -49.6 m`` in the body frame (x = bow, y = port, z = up):

* **Case A -- the sign.** Bow-up pitch, no roll. The aft pad must move **down** and
  **forward**: ``z_pad = heave + x_pad*sin(pitch)`` with ``x_pad < 0``. Plan D0.5 and the
  ``deck-scaling-physics`` skill write a minus sign there, which would make an aft point rise
  when the bow rises; the anti-assertion in this case is what makes that unpassable.
* **Case B -- the roll sign and the lateral term.** Roll to starboard with an off-centreline
  pad on the **starboard** side (``y_pad = -7 m``, since y is port). That pad must go down.
* **Case C -- the rotation order.** Finite roll *and* pitch on a centreline pad. Under ZYX
  (``R = R_y(theta) R_x(phi)``, PyBullet's ``getQuaternionFromEuler`` order) ``R_x`` fixes the
  pad, so ``p_y`` is **exactly** zero and ``z`` is roll-independent. The skill's
  ``R_x R_y`` order invents ``p_y = +2.592305889 m`` and shifts ``z`` by 0.694606270 m.

Units: degrees, degrees per second, degrees per second squared, metres, metres per second,
metres per second squared, radians only inside the module under test. Every case here is full
scale except where a model-scale value is named explicitly.
"""

import numpy as np
import pytest
from dmf.typedefs import FloatArray

from conftest import BASE_SEED
from rld.deck.config import PadConfig, ScalingConfig
from rld.deck.kinematics import (
    DeckPointState,
    angular_velocity_rad_s,
    deck_normal,
    deck_point_state,
    euler_xyz_rad,
    rotation_matrix,
    tilt_deg,
    to_model_state,
)
from rld.deck.scaling import FroudeScale

#: Frigate length between perpendiculars, metres, full scale.
FRIGATE_LENGTH_FULL_M = 124.0

#: Default aft-pad lever arm, metres, full scale. Negative because x is the bow direction.
X_PAD_FULL_M = -0.4 * FRIGATE_LENGTH_FULL_M

#: Componentwise tolerance for the hand-computed cases, metres / m per s / radians.
HAND_ABS = 1e-9


def _quiet_state(**overrides: object) -> DeckPointState:
    """Evaluate a deck point with every channel zero except the named overrides."""
    channels: dict[str, object] = {
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "heave_m": 0.0,
        "roll_rate_dps": 0.0,
        "pitch_rate_dps": 0.0,
        "heave_rate_m_s": 0.0,
        "roll_acc_dps2": 0.0,
        "pitch_acc_dps2": 0.0,
        "heave_acc_m_s2": 0.0,
        "r_pad_m": (X_PAD_FULL_M, 0.0, 0.0),
    }
    channels.update(overrides)
    return deck_point_state(**channels)  # type: ignore[arg-type]


def _wrong_order_offset(roll_deg: float, pitch_deg: float, r_pad_m: FloatArray) -> FloatArray:
    """Return ``R_x(phi) R_y(theta) r_pad`` -- the skill's rotation order, for anti-assertions."""
    phi, theta = np.radians(roll_deg), -np.radians(pitch_deg)
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]]
    )
    rot_y = np.array(
        [[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]]
    )
    return np.asarray(rot_x @ rot_y @ np.asarray(r_pad_m, dtype=np.float64))


def test_case_a_arbitrates_the_aft_pad_sign() -> None:
    """Aft pad, pitch 6 deg bow-up, heave 0.5 m, pitch rate 2 deg/s, heave rate -0.2 m/s."""
    state = _quiet_state(pitch_deg=6.0, heave_m=0.5, pitch_rate_dps=2.0, heave_rate_m_s=-0.2)
    np.testing.assert_allclose(
        state.position_m, [-49.328286010266, 0.0, -4.684611778076], atol=HAND_ABS
    )
    np.testing.assert_allclose(
        state.velocity_m_s, [0.180977091930, 0.0, -1.921882010489], atol=HAND_ABS
    )
    # Centripetal only: alpha = 0, so a = -|omega|**2 * p_offset.
    np.testing.assert_allclose(
        state.acceleration_m_s2, [0.060105020828, 0.0, 0.006317292250], atol=HAND_ABS
    )
    np.testing.assert_allclose(state.euler_xyz_rad, [0.0, -0.104719755120, 0.0], atol=HAND_ABS)
    np.testing.assert_allclose(
        state.angular_velocity_rad_s, [0.0, -0.034906585040, 0.0], atol=HAND_ABS
    )
    # The lever-arm term alone, without heave: z_pad - heave = x_pad*sin(pitch).
    assert float(state.position_m[2]) - 0.5 == pytest.approx(
        X_PAD_FULL_M * np.sin(np.radians(6.0)), abs=HAND_ABS
    )
    # Anti-assertion: the sign written in plan D0.5 and in the skill.
    wrong_z = 0.5 - X_PAD_FULL_M * np.sin(np.radians(6.0))
    assert float(state.position_m[2]) < 0.5 < wrong_z
    assert float(state.position_m[2]) != pytest.approx(wrong_z, abs=1e-6)
    # Aft pad goes down and forward as the bow pitches up.
    assert float(state.velocity_m_s[2]) < 0.0
    assert float(state.velocity_m_s[0]) > 0.0


def test_case_a_at_model_scale() -> None:
    """The same case after one Froude scaling, lambda = 1/25."""
    state = to_model_state(
        _quiet_state(pitch_deg=6.0, heave_m=0.5, pitch_rate_dps=2.0, heave_rate_m_s=-0.2),
        FroudeScale(lam=1.0 / 25.0),
    )
    assert float(state.position_m[2]) == pytest.approx(-0.187384471123, abs=HAND_ABS)
    assert float(state.velocity_m_s[0]) == pytest.approx(0.036195418386, abs=HAND_ABS)
    assert float(state.velocity_m_s[2]) == pytest.approx(-0.384376402098, abs=HAND_ABS)


def test_case_b_arbitrates_the_roll_sign_and_the_lateral_term() -> None:
    """Starboard pad (y = -7 m), roll 10 deg to starboard, roll rate 3 deg/s."""
    state = _quiet_state(roll_deg=10.0, roll_rate_dps=3.0, r_pad_m=(X_PAD_FULL_M, -7.0, 0.0))
    np.testing.assert_allclose(
        state.position_m, [-49.6, -6.893654271085, -1.215537243669], atol=HAND_ABS
    )
    np.testing.assert_allclose(
        state.velocity_m_s, [0.0, 0.063645381248, -0.360950893574], atol=HAND_ABS
    )
    np.testing.assert_allclose(
        state.acceleration_m_s2, [0.0, 0.018899344593, 0.003332464369], atol=HAND_ABS
    )
    np.testing.assert_allclose(state.euler_xyz_rad, [0.174532925199, 0.0, 0.0], atol=HAND_ABS)
    # Roll positive to starboard puts the starboard pad below the CG and the port pad above.
    port = _quiet_state(roll_deg=10.0, r_pad_m=(X_PAD_FULL_M, 7.0, 0.0))
    assert float(state.position_m[2]) < 0.0 < float(port.position_m[2])
    # With no pitch, a centreline pad stays at the CG height: the lateral term is the whole
    # roll contribution.
    assert float(_quiet_state(roll_deg=10.0).position_m[2]) == 0.0


def test_case_c_arbitrates_the_rotation_order() -> None:
    """Centreline aft pad at roll 30 deg and pitch 6 deg: ZYX keeps ``p_y`` exactly zero."""
    state = _quiet_state(roll_deg=30.0, pitch_deg=6.0)
    np.testing.assert_allclose(
        state.position_m, [-49.328286010266, 0.0, -5.184611778076], atol=HAND_ABS
    )
    assert abs(float(state.position_m[1])) < 1e-12
    assert float(state.position_m[1]) == 0.0  # exactly, not merely within tolerance
    # Anti-assertion: the skill's R_x R_y order, which is what these numbers reject.
    wrong = _wrong_order_offset(30.0, 6.0, np.array([X_PAD_FULL_M, 0.0, 0.0]))
    assert float(wrong[1]) == pytest.approx(2.592305889, abs=1e-9)
    assert float(state.position_m[2]) - float(wrong[2]) == pytest.approx(-0.694606270, abs=1e-9)
    assert float(state.position_m[1]) != pytest.approx(float(wrong[1]), abs=1e-6)
    np.testing.assert_allclose(
        state.euler_xyz_rad, [0.523598775598, -0.104719755120, 0.0], atol=HAND_ABS
    )


def test_centreline_pad_is_exactly_roll_blind() -> None:
    """Roll and roll rate cannot move a pad on the ``x`` axis at all (addendum C2).

    Beam seas -- the ``unseen_heading`` regime -- is therefore the *easiest* pad-motion cell,
    which is a Phase 5 fact worth asserting here rather than discovering later.
    """
    rng = np.random.default_rng(BASE_SEED + 11)
    base = _quiet_state(pitch_deg=4.0, heave_m=0.3, pitch_rate_dps=1.5, heave_rate_m_s=0.1)
    for _ in range(8):
        rolled = _quiet_state(
            roll_deg=float(rng.uniform(-35.0, 35.0)),
            pitch_deg=4.0,
            heave_m=0.3,
            roll_rate_dps=float(rng.uniform(-10.0, 10.0)),
            pitch_rate_dps=1.5,
            heave_rate_m_s=0.1,
            roll_acc_dps2=float(rng.uniform(-5.0, 5.0)),
        )
        np.testing.assert_array_equal(rolled.position_m, base.position_m)
        np.testing.assert_allclose(rolled.velocity_m_s, base.velocity_m_s, atol=1e-15)
        np.testing.assert_allclose(rolled.acceleration_m_s2, base.acceleration_m_s2, atol=1e-13)


def test_cg_pad_is_exactly_pure_heave() -> None:
    """With ``r_pad = 0`` the pad is the CG: pure heave, attitude and normal unaffected."""
    state = _quiet_state(
        roll_deg=12.0,
        pitch_deg=-5.0,
        heave_m=1.25,
        roll_rate_dps=4.0,
        pitch_rate_dps=-2.0,
        heave_rate_m_s=-0.4,
        roll_acc_dps2=1.0,
        pitch_acc_dps2=-0.5,
        heave_acc_m_s2=0.6,
        r_pad_m=(0.0, 0.0, 0.0),
    )
    np.testing.assert_array_equal(state.position_m, [0.0, 0.0, 1.25])
    np.testing.assert_array_equal(state.velocity_m_s, [0.0, 0.0, -0.4])
    np.testing.assert_array_equal(state.acceleration_m_s2, [0.0, 0.0, 0.6])
    # The CG arm isolates the dmf phase defect: it removes the lever arm, not the attitude.
    np.testing.assert_array_equal(state.euler_xyz_rad, euler_xyz_rad(12.0, -5.0))
    np.testing.assert_array_equal(state.normal, deck_normal(12.0, -5.0))
    assert float(state.tilt_deg) == pytest.approx(float(tilt_deg(12.0, -5.0)))


def _analytic_channels(t_full_s: FloatArray) -> dict[str, FloatArray]:
    """Return a smooth analytic 3-DOF motion and its exact derivatives, full scale.

    Amplitudes and frequencies are chosen to be ship-like (8 deg roll, 4 deg pitch, 1.5 m
    heave near 0.6 rad/s) so the finite-difference comparison runs at a realistic curvature.
    """
    spec = {"roll": (8.0, 0.55, 0.3), "pitch": (4.0, 0.70, 1.1), "heave": (1.5, 0.62, 2.4)}
    out: dict[str, FloatArray] = {}
    for name, (amp, w, phase) in spec.items():
        out[name] = amp * np.sin(w * t_full_s + phase)
        out[f"{name}_rate"] = amp * w * np.cos(w * t_full_s + phase)
        out[f"{name}_acc"] = -amp * w**2 * np.sin(w * t_full_s + phase)
    return out


def _state_at(t_full_s: FloatArray, r_pad_m: tuple[float, float, float]) -> DeckPointState:
    """Evaluate the analytic motion's deck point at ``t_full_s``, full scale."""
    ch = _analytic_channels(t_full_s)
    return deck_point_state(
        roll_deg=ch["roll"],
        pitch_deg=ch["pitch"],
        heave_m=ch["heave"],
        roll_rate_dps=ch["roll_rate"],
        pitch_rate_dps=ch["pitch_rate"],
        heave_rate_m_s=ch["heave_rate"],
        roll_acc_dps2=ch["roll_acc"],
        pitch_acc_dps2=ch["pitch_acc"],
        heave_acc_m_s2=ch["heave_acc"],
        r_pad_m=r_pad_m,
    )


def test_analytic_velocity_and_acceleration_match_finite_differences() -> None:
    """A 5-point central difference of position reproduces the analytic velocity.

    Finite differences live **only** in this test: the deck-motion path itself is analytic
    end to end (CLAUDE.md architecture rules). The pad is off the centreline so that every
    term -- roll, pitch and the cross products -- is exercised.
    """
    t_full_s = np.linspace(10.0, 40.0, 601)
    step_s = 1e-3
    r_pad_m = (X_PAD_FULL_M, -7.0, 2.0)
    samples = [_state_at(t_full_s + k * step_s, r_pad_m) for k in (-2, -1, 0, 1, 2)]
    fd_velocity = (
        -samples[4].position_m
        + 8.0 * samples[3].position_m
        - 8.0 * samples[1].position_m
        + samples[0].position_m
    ) / (12.0 * step_s)
    fd_acceleration = (
        -samples[4].velocity_m_s
        + 8.0 * samples[3].velocity_m_s
        - 8.0 * samples[1].velocity_m_s
        + samples[0].velocity_m_s
    ) / (12.0 * step_s)
    np.testing.assert_allclose(samples[2].velocity_m_s, fd_velocity, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(samples[2].acceleration_m_s2, fd_acceleration, rtol=1e-6, atol=1e-9)


def test_kinematics_commutes_with_froude_scaling() -> None:
    """Scaling the inputs or scaling the result gives the same model-scale state.

    This is what licenses the bridge's order of operations: evaluate the geometry once at
    full scale with the full-scale lever arm, then Froude-scale once.
    """
    scale = FroudeScale(lam=1.0 / 25.0)
    t_full_s = np.linspace(20.0, 25.0, 129)
    r_pad_full_m = (X_PAD_FULL_M, -7.0, 0.0)
    from_full = to_model_state(_state_at(t_full_s, r_pad_full_m), scale)
    ch = _analytic_channels(t_full_s)
    direct = deck_point_state(
        roll_deg=ch["roll"],
        pitch_deg=ch["pitch"],
        heave_m=scale.length(ch["heave"]),
        roll_rate_dps=scale.angular_rate(ch["roll_rate"]),
        pitch_rate_dps=scale.angular_rate(ch["pitch_rate"]),
        heave_rate_m_s=scale.velocity(ch["heave_rate"]),
        roll_acc_dps2=ch["roll_acc"] / scale.lam,
        pitch_acc_dps2=ch["pitch_acc"] / scale.lam,
        heave_acc_m_s2=scale.acceleration(ch["heave_acc"]),
        r_pad_m=tuple(scale.length(v) for v in r_pad_full_m),  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(from_full.position_m, direct.position_m, rtol=1e-12)
    np.testing.assert_allclose(from_full.velocity_m_s, direct.velocity_m_s, rtol=1e-12)
    np.testing.assert_allclose(
        from_full.acceleration_m_s2, direct.acceleration_m_s2, rtol=1e-12, atol=1e-18
    )
    np.testing.assert_allclose(
        from_full.angular_velocity_rad_s, direct.angular_velocity_rad_s, rtol=1e-12
    )


def test_normal_is_unit_norm_and_signed_for_bow_up() -> None:
    rng = np.random.default_rng(BASE_SEED + 12)
    roll_deg = rng.uniform(-30.0, 30.0, size=64)
    pitch_deg = rng.uniform(-10.0, 10.0, size=64)
    normal = deck_normal(roll_deg, pitch_deg)
    np.testing.assert_allclose(np.linalg.norm(normal, axis=-1), 1.0, rtol=1e-14)
    # Bow-up tilts the normal aft (negative x); roll to starboard tilts it to starboard
    # (negative y, since y is port).
    assert float(deck_normal(0.0, 6.0)[0]) < 0.0
    assert float(deck_normal(6.0, 0.0)[1]) < 0.0
    # Tilt is the angle to world +z and reduces to the single angle in each pure case.
    np.testing.assert_allclose(
        tilt_deg(roll_deg, pitch_deg),
        np.degrees(np.arccos(np.clip(normal[..., 2], -1.0, 1.0))),
        rtol=1e-14,
    )
    assert float(tilt_deg(0.0, 6.0)) == pytest.approx(6.0)
    assert float(tilt_deg(-7.0, 0.0)) == pytest.approx(7.0)
    # Roll and pitch together exceed either alone -- the Phase 2 success criterion's 15 deg
    # relative-tilt budget is spent by the deck itself in the worst cells (addendum C4).
    assert float(tilt_deg(30.0, 6.0)) == pytest.approx(30.539255336567, abs=1e-9)


def test_rotation_matrix_is_orthonormal_and_columns_are_the_body_axes() -> None:
    rng = np.random.default_rng(BASE_SEED + 13)
    roll_deg = rng.uniform(-30.0, 30.0, size=16)
    pitch_deg = rng.uniform(-10.0, 10.0, size=16)
    rot = rotation_matrix(roll_deg, pitch_deg)
    identity = np.einsum("...ij,...kj->...ik", rot, rot)
    np.testing.assert_allclose(identity, np.broadcast_to(np.eye(3), identity.shape), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, rtol=1e-14)
    np.testing.assert_allclose(rot[..., :, 2], deck_normal(roll_deg, pitch_deg), rtol=1e-14)
    # omega is the axis of the ZYX composition: pure roll rate about the bow, pure pitch rate
    # about the port axis with the sign flip.
    np.testing.assert_allclose(
        angular_velocity_rad_s(0.0, 0.0, 3.0, 0.0), [np.radians(3.0), 0.0, 0.0], atol=1e-15
    )
    np.testing.assert_allclose(
        angular_velocity_rad_s(0.0, 0.0, 0.0, 3.0), [0.0, -np.radians(3.0), 0.0], atol=1e-15
    )


@pytest.mark.pybullet
def test_euler_triple_matches_pybullet_rotation() -> None:
    """``getQuaternionFromEuler(euler_xyz_rad(...))`` is exactly our rotation matrix.

    This is the test that makes the Phase 2 platform body carry the same orientation as the
    kinematics: the triple handed to PyBullet is ``(phi, theta, 0)`` in that order.
    """
    import pybullet

    rng = np.random.default_rng(BASE_SEED + 14)
    for roll_deg, pitch_deg in zip(
        rng.uniform(-30.0, 30.0, size=12), rng.uniform(-12.0, 12.0, size=12), strict=True
    ):
        euler = euler_xyz_rad(float(roll_deg), float(pitch_deg))
        quaternion = pybullet.getQuaternionFromEuler(euler.tolist())
        matrix = np.asarray(pybullet.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        np.testing.assert_allclose(
            matrix, rotation_matrix(float(roll_deg), float(pitch_deg)), atol=1e-12
        )


def test_committed_pad_config_places_the_arms_where_the_plan_says(
    deck_pads: PadConfig, deck_scaling: ScalingConfig
) -> None:
    aft = deck_pads.spec("aft")
    assert aft.r_pad_frac == (-0.4, 0.0, 0.0)
    assert aft.r_pad_full_m(FRIGATE_LENGTH_FULL_M) == pytest.approx((-49.6, 0.0, 0.0))
    # The lever arm differs across hulls: a named confound of the unseen_vessel regime.
    assert aft.r_pad_full_m(175.0) == pytest.approx((-70.0, 0.0, 0.0))
    assert deck_pads.spec("cg").r_pad_full_m(FRIGATE_LENGTH_FULL_M) == (0.0, 0.0, 0.0)
    assert deck_pads.radius_model_m == pytest.approx(0.15)
    assert deck_scaling.froude_scale().length(-49.6) == pytest.approx(-1.984)
    with pytest.raises(ValueError, match="unknown pad"):
        deck_pads.spec("bow")
