"""``td_in_quiescent_window`` on hand-built true deck trajectories whose answers are known.

The trajectory is a :class:`~rld.control.base.PrivilegedContext` built by hand on the
committed physics grid (240 Hz, 3 001 samples = 12.5 s): a flat, still deck, with single
samples perturbed to put roll, pitch or pad ``v_z`` outside a limit. The permissive rule at
``lam = 1/25`` and 30 Hz is 12 samples 1/30 s apart (every 8th physics sample),
``|roll| <= 3.0 deg``, ``|pitch| <= 2.0 deg``, ``|v_z| <= 0.16 m/s``.

Units: seconds model scale, degrees, metres per second model scale.
"""

import math

import numpy as np
import pytest
from dmf.data.splits import realization_key

from rld.control.base import PrivilegedContext
from rld.control.registry import make_controller
from rld.eval.envs import EvalConfigs, load_eval_configs
from rld.eval.truth import td_in_quiescent_window, touchdown_rule

PHYSICS_HZ = 240
N_SAMPLES = 3001
STRIDE = 8
T_TD = 2.0  # s; physics sample 480
I_TD = 480


@pytest.fixture(scope="module")
def cfgs() -> EvalConfigs:
    """Return the committed configs."""
    return load_eval_configs()


def _context(
    vz: dict[int, float] | None = None,
    roll_deg: dict[int, float] | None = None,
    pitch_deg: dict[int, float] | None = None,
) -> PrivilegedContext:
    """Return a still, flat deck with the given samples perturbed.

    Args:
        vz: ``{physics sample: pad v_z}``, metres per second model scale.
        roll_deg: ``{physics sample: roll}``, degrees, written into the deck normal.
        pitch_deg: ``{physics sample: pitch}``, degrees, written into the deck normal.
    """
    velocity = np.zeros((N_SAMPLES, 3))
    normal = np.tile([0.0, 0.0, 1.0], (N_SAMPLES, 1))
    for i, value in (vz or {}).items():
        velocity[i, 2] = value
    # P1-D2: n = (-sin(theta) cos(phi), -sin(phi), cos(theta) cos(phi)); deck_angles_deg
    # returns roll = -asin(n_y) and pitch = atan2(-n_x, n_z).
    for i, value in (roll_deg or {}).items():
        r = math.radians(value)
        normal[i] = [0.0, -math.sin(r), math.cos(r)]
    for i, value in (pitch_deg or {}).items():
        p = math.radians(value)
        normal[i] = [-math.sin(p), 0.0, math.cos(p)]
    zeros = np.zeros(N_SAMPLES)
    return PrivilegedContext(
        t0_model_s=0.0,
        physics_dt_s=1.0 / PHYSICS_HZ,
        t_episode_s=np.arange(N_SAMPLES) / PHYSICS_HZ,
        position_m=np.zeros((N_SAMPLES, 3)),
        velocity_m_s=velocity,
        roll_deg=zeros,
        pitch_deg=zeros,
        normal=normal,
        realization_key=realization_key("SS5", 180.0, 12.0, "frigate", 0),
    )


def _verdict(cfgs: EvalConfigs, context: PrivilegedContext, t_td: float | None = T_TD) -> object:
    return td_in_quiescent_window(context, touchdown_rule(cfgs), t_td, STRIDE)


def test_rule_is_gateds_rule(cfgs: EvalConfigs) -> None:
    rule = touchdown_rule(cfgs)
    assert rule == make_controller("gated").rule  # type: ignore[attr-defined]
    assert rule == make_controller("oracle_gated").rule  # type: ignore[attr-defined]
    assert rule.n_samples == 12
    assert rule.spacing_s == pytest.approx(1.0 / 30.0)
    assert int(cfgs.landing.pyb_steps_per_ctrl) == STRIDE
    assert rule.limits.pad_vz_m_s == pytest.approx(0.16)
    assert (rule.limits.roll_deg, rule.limits.pitch_deg) == (3.0, 2.0)


def test_still_deck_is_quiescent(cfgs: EvalConfigs) -> None:
    assert _verdict(cfgs, _context()) is True


@pytest.mark.parametrize("j", [0, 5, 11])
def test_a_violation_on_any_window_sample_fails(cfgs: EvalConfigs, j: int) -> None:
    i = I_TD + STRIDE * j
    assert _verdict(cfgs, _context(vz={i: 0.17})) is False
    assert _verdict(cfgs, _context(vz={i: -0.17})) is False
    assert _verdict(cfgs, _context(roll_deg={i: 3.1})) is False
    assert _verdict(cfgs, _context(pitch_deg={i: -2.1})) is False


@pytest.mark.parametrize(
    "i",
    [
        I_TD - 1,  # just before touchdown
        I_TD + 1,  # between control-rate samples: not read
        I_TD + STRIDE * 11 - 1,  # between the last two samples
        I_TD + STRIDE * 12,  # the 13th control-rate sample: outside the window
    ],
)
def test_samples_outside_the_window_are_not_read(cfgs: EvalConfigs, i: int) -> None:
    assert _verdict(cfgs, _context(vz={i: 1.0}, roll_deg={i: 10.0})) is True


def test_limits_are_inclusive(cfgs: EvalConfigs) -> None:
    limit = touchdown_rule(cfgs).limits.pad_vz_m_s
    i = I_TD + STRIDE * 3
    assert _verdict(cfgs, _context(vz={i: limit})) is True
    assert _verdict(cfgs, _context(vz={i: float(np.nextafter(limit, 1.0))})) is False
    assert _verdict(cfgs, _context(roll_deg={i: 2.99}, pitch_deg={i + STRIDE: 1.99})) is True


def test_off_grid_touchdown_starts_at_the_next_physics_sample(cfgs: EvalConfigs) -> None:
    # 2.001 s is physics sample 480.24, so the window starts at sample 481.
    assert _verdict(cfgs, _context(vz={480: 1.0}), t_td=2.001) is True
    assert _verdict(cfgs, _context(vz={481: 1.0}), t_td=2.001) is False
    assert _verdict(cfgs, _context(vz={481 + STRIDE * 11: 1.0}), t_td=2.001) is False


def test_no_touchdown_is_nan_not_false(cfgs: EvalConfigs) -> None:
    for missing in (None, float("nan")):
        value = _verdict(cfgs, _context(), t_td=missing)
        assert isinstance(value, float) and math.isnan(value)


def test_a_window_past_the_trajectory_is_loud(cfgs: EvalConfigs) -> None:
    # The last window that fits starts at sample 3000 - 88 = 2912 (12.1333 s).
    assert _verdict(cfgs, _context(), t_td=2912 / PHYSICS_HZ) is True
    with pytest.raises(RuntimeError, match="past"):
        _verdict(cfgs, _context(), t_td=2913 / PHYSICS_HZ)
