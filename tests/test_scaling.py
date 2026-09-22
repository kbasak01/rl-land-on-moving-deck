"""Froude-scaling invariants (plan D0.1).

Every quantity here is named for its scale: ``*_full_*`` is ship scale, ``*_model_*`` is the
Crazyflie-scale stand-in the PyBullet environment flies in. The table under test is
``length x lam``, ``time x sqrt(lam)``, ``velocity x sqrt(lam)``, ``acceleration x 1``,
``angle x 1``, ``angular rate and frequency x 1/sqrt(lam)``.
"""

import numpy as np
import pytest

from conftest import BASE_SEED
from rld.deck.config import SCALING_CONFIG, ScalingConfig, load_scaling
from rld.deck.scaling import SCALE_EXPONENTS, FroudeScale

FRIGATE_LENGTH_FULL_M = 124.0


def test_committed_config_is_one_over_25() -> None:
    cfg = load_scaling(SCALING_CONFIG)
    assert cfg.lam == pytest.approx(0.04)
    assert cfg.lam_ladder == pytest.approx((1 / 25, 1 / 40, 1 / 60, 1 / 100))
    assert cfg.gravity_m_s2 == pytest.approx(9.80665)
    gate = cfg.feasibility.gate_reference
    assert gate.name == "cf2x_urdf_max_speed_kmh"
    assert gate.v_max_m_s == pytest.approx(30.0 / 3.6)
    assert cfg.feasibility.threshold_m_s(gate) == pytest.approx(2.0833, abs=1e-4)
    # The rejected reading is carried, not deleted, so the CSV can report both.
    other = next(r for r in cfg.feasibility.references if not r.is_gate)
    assert cfg.feasibility.threshold_m_s(other) == pytest.approx(0.0625)


def test_factor_table_matches_plan_d01(deck_scaling: ScalingConfig) -> None:
    scale = deck_scaling.froude_scale()
    assert scale.factor("length") == pytest.approx(0.04)
    assert scale.factor("time") == pytest.approx(0.2)
    assert scale.factor("velocity") == pytest.approx(0.2)
    assert scale.factor("acceleration") == pytest.approx(1.0)
    assert scale.factor("angle") == pytest.approx(1.0)
    assert scale.factor("angular_rate") == pytest.approx(5.0)
    assert scale.factor("frequency") == pytest.approx(5.0)
    assert set(SCALE_EXPONENTS) == {
        "length",
        "time",
        "velocity",
        "acceleration",
        "angle",
        "angular_rate",
        "frequency",
    }
    with pytest.raises(ValueError, match="unknown scaled quantity"):
        scale.factor("mass")


def test_angles_and_accelerations_are_invariant() -> None:
    rng = np.random.default_rng(BASE_SEED + 1)
    scale = FroudeScale(lam=1.0 / 25.0)
    angles_full_deg = rng.uniform(-20.0, 20.0, size=64)
    accel_full_m_s2 = rng.normal(0.0, 2.0, size=64)
    np.testing.assert_array_equal(scale.angle(angles_full_deg), angles_full_deg)
    np.testing.assert_array_equal(scale.acceleration(accel_full_m_s2), accel_full_m_s2)


def test_froude_number_is_unchanged_by_scaling() -> None:
    rng = np.random.default_rng(BASE_SEED + 2)
    scale = FroudeScale(lam=1.0 / 25.0)
    for speed_full_m_s in rng.uniform(0.5, 8.0, size=16):
        speed_model_m_s = scale.velocity(float(speed_full_m_s))
        length_model_m = scale.length(FRIGATE_LENGTH_FULL_M)
        assert scale.froude_number(speed_model_m_s, length_model_m) == pytest.approx(
            scale.froude_number(float(speed_full_m_s), FRIGATE_LENGTH_FULL_M), rel=1e-12
        )
    with pytest.raises(ValueError, match="length_m must be positive"):
        scale.froude_number(1.0, 0.0)


def test_periods_and_rates_scale_as_the_table() -> None:
    scale = FroudeScale(lam=1.0 / 25.0)
    # SS6 180 deg 12 kn peak encounter period, full scale seconds.
    assert scale.time(9.4) == pytest.approx(1.88)
    # A 10 s full-scale period is a 0.1 Hz full-scale signal; both must agree after scaling.
    period_model_s = scale.time(10.0)
    assert scale.frequency(1.0 / 10.0) == pytest.approx(1.0 / period_model_s)
    assert scale.angular_rate(2.0) == pytest.approx(10.0)


def test_time_maps_round_trip_and_include_no_offset() -> None:
    rng = np.random.default_rng(BASE_SEED + 3)
    scale = FroudeScale(lam=1.0 / 25.0)
    t_model_s = rng.uniform(0.0, 120.0, size=32)
    np.testing.assert_allclose(
        scale.to_model_time(scale.to_full_time(t_model_s)), t_model_s, rtol=1e-14
    )
    # The corpus's 120 s spin-up offset belongs to the bridge, not to the scale.
    assert scale.to_full_time(0.0) == 0.0
    assert scale.to_full_time(120.0) == pytest.approx(600.0)


def test_lam_one_is_the_identity() -> None:
    rng = np.random.default_rng(BASE_SEED + 4)
    scale = FroudeScale(lam=1.0)
    values = rng.normal(size=16)
    for method in (
        scale.length,
        scale.time,
        scale.velocity,
        scale.acceleration,
        scale.angle,
        scale.angular_rate,
        scale.frequency,
        scale.to_full_time,
        scale.to_model_time,
    ):
        np.testing.assert_array_equal(method(values), values)


def test_invalid_scales_are_rejected() -> None:
    for bad_lam in (0.0, -0.04, 1.5):
        with pytest.raises(ValueError, match="lam must lie in"):
            FroudeScale(lam=bad_lam)
    with pytest.raises(ValueError, match="gravity_m_s2 must be positive"):
        FroudeScale(lam=0.04, gravity_m_s2=0.0)
