"""``oracle_gated``: the commit-timing oracle (privileged).

Contract checks (shape and bounds, determinism under reset, static-pad landing), and the
three properties that keep it honest:

* it is ``privileged=True`` in the registry **and** on the instance;
* it **refuses to run without a context** -- ``reset`` raises rather than letting it degrade
  silently into a different controller;
* its rule reads the **future**: the same present with a different future gives a different
  decision, with the window placed at the predicted touchdown
  ``t + height / descent_rate``;
* its rule is **exactly** ``gated``'s (Gate 3 remediation, 2026-09-22): on an identical
  deck window -- past for ``gated``, future for the oracle -- both return the same verdict,
  and physics samples between the control-rate samples are not read.

Units: metres, metres per second and seconds model scale; degrees.
"""

import numpy as np
import pytest

from _control_helpers import (
    assert_actions_in_space,
    committed_spec,
    deck_normal,
    feed,
    quiescence_windows,
    run_episode,
    synthetic_obs,
)
from rld.control.base import PrivilegedContext
from rld.control.gated import Gated
from rld.control.obs_view import obs_layout, view
from rld.control.oracle import OracleGated
from rld.control.registry import entry, make_controller

NAME = "oracle_gated"


def test_privileged_flag_is_true():
    """Marked privileged in the registry and on the instance."""
    assert entry(NAME).privileged is True
    assert make_controller(NAME).privileged is True


def test_refuses_to_run_without_a_context(env_obs_cfg):
    """reset(seed) without a context raises; act() before a valid reset raises."""
    controller = make_controller(NAME, committed_spec())
    with pytest.raises(ValueError, match="PrivilegedContext"):
        controller.reset(0)
    with pytest.raises(ValueError, match="PrivilegedContext"):
        controller.reset(0, None)
    with pytest.raises(RuntimeError, match="before reset"):
        controller.act(synthetic_obs(env_obs_cfg))


def _context(pad_vz: np.ndarray, dt: float = 1.0 / 240.0) -> PrivilegedContext:
    """A synthetic flat-deck context with a prescribed pad v_z history (m/s model)."""
    n = pad_vz.size
    zeros = np.zeros(n)
    return PrivilegedContext(
        t0_model_s=10.0,
        physics_dt_s=dt,
        t_episode_s=np.arange(n) * dt,
        position_m=np.tile([0.0, 0.0, 1.0], (n, 1)),
        velocity_m_s=np.column_stack([zeros, zeros, pad_vz]),
        roll_deg=zeros,
        pitch_deg=zeros.copy(),
        normal=np.tile([0.0, 0.0, 1.0], (n, 1)),
        realization_key=("SS5", 180.0, 0.0, "synthetic", 0),
    )


def test_rule_reads_the_future_window_at_predicted_touchdown(env_obs_cfg):
    """Identical present, different future -> different decision.

    At hover (0.3 m) with the committed descent rate the predicted touchdown is
    ``0.3 / descent_rate`` after now; a burst of pad motion inside that future window blocks
    the commit, the same burst placed after the window does not.
    """
    spec = committed_spec()
    controller = make_controller(NAME, spec)
    assert isinstance(controller, OracleGated)
    assert controller.window_samples == 12
    assert controller.window_stride == 8
    hover = controller.gated.hover_height_m
    obs = synthetic_obs(env_obs_cfg, rel_position_w=(0.0, 0.0, -hover), height_m=hover)
    t_td = hover / controller.pid.descent_rate_m_s
    n = 3001
    t = np.arange(n) / 240.0
    burst_in = np.where((t >= t_td + 0.1) & (t < t_td + 0.2), 0.5, 0.0)
    burst_after = np.where((t >= t_td + 0.5) & (t < t_td + 0.6), 0.5, 0.0)

    controller.reset(0, _context(burst_in))
    feed(controller, [obs], spec.ctrl_dt_s, 12.0)
    assert not controller.committed

    controller.reset(0, _context(burst_after))
    feed(controller, [obs], spec.ctrl_dt_s, 12.0)
    assert controller.committed
    predicted = controller.predicted_touchdown_s(view(obs, obs_layout(env_obs_cfg)), 0.0)
    assert predicted == pytest.approx(t_td, rel=1e-6)


@pytest.mark.pybullet
def test_deterministic_under_reset(landing_env, static_motion):
    """Same seed, same instance, another episode in between: bit-identical actions."""
    env = landing_env(static_motion)
    controller = make_controller(NAME, committed_spec())
    first, _, _ = run_episode(env, controller, 3, privileged=True)
    run_episode(env, controller, 4, privileged=True)
    again, _, _ = run_episode(env, controller, 3, privileged=True)
    np.testing.assert_array_equal(np.stack(first), np.stack(again))
    assert_actions_in_space(first)


@pytest.mark.pybullet
@pytest.mark.slow
def test_static_pad_landing_succeeds(landing_env, static_motion):
    """A static deck's future is always quiescent: ten out of ten."""
    env = landing_env(static_motion)
    controller = make_controller(NAME, committed_spec())
    for seed in range(10):
        _, record, _ = run_episode(env, controller, seed, privileged=True)
        assert record.outcome == "success", (seed, record.as_row())


def test_oracle_and_gated_give_the_same_verdict_on_the_same_window(env_obs_cfg):
    """The oracle's predicate IS gated's: same count, spacing, thresholds and v_z.

    ``gated`` observes the window as its trailing 12 control samples at hover; the oracle
    sees the same 12 states in its true future, at ``t_td + j / 30``, with every other
    physics sample of the context set far outside the limits. Both commit, or neither,
    and both agree with :meth:`QuiescenceRule.verdict` on the raw window.
    """
    spec = committed_spec()
    gated = make_controller("gated", spec)
    oracle = make_controller(NAME, spec)
    assert isinstance(gated, Gated) and isinstance(oracle, OracleGated)
    rule = gated.rule
    assert rule == oracle.rule
    n, stride = rule.n_samples, oracle.window_stride
    hover = gated.gated.hover_height_m
    t_td = hover / oracle.pid.descent_rate_m_s
    size = 3001
    dt = 1.0 / 240.0
    verdicts = []
    for roll, pitch, vz in quiescence_windows(np.random.default_rng(20260922), gated.limits, n):
        expected = rule.verdict(roll, pitch, vz)

        # gated: the window is its observed past, ending now.
        obs_list = [
            synthetic_obs(
                env_obs_cfg,
                rel_position_w=(0.0, 0.0, -hover),
                deck_velocity_w=(0.0, 0.0, float(vz[j])),
                normal_w=deck_normal(float(roll[j]), float(pitch[j])),
                height_m=hover,
            )
            for j in range(n)
        ]
        gated.reset(0)
        feed(gated, obs_list, spec.ctrl_dt_s, 12.0)

        # oracle: the same states are its true future at control-rate spacing; everything
        # in between is loud, so reading it would block the commit.
        roll_all = np.full(size, 10.0)
        pitch_all = np.full(size, -10.0)
        vz_all = np.full(size, 0.5)
        start = int(np.ceil(t_td / dt - 1e-9))
        index = start + stride * np.arange(n)
        roll_all[index], pitch_all[index], vz_all[index] = roll, pitch, vz
        normal = np.array([deck_normal(r, q) for r, q in zip(roll_all, pitch_all, strict=True)])
        context = PrivilegedContext(
            t0_model_s=10.0,
            physics_dt_s=dt,
            t_episode_s=np.arange(size) * dt,
            position_m=np.tile([0.0, 0.0, 1.0], (size, 1)),
            velocity_m_s=np.column_stack([np.zeros(size), np.zeros(size), vz_all]),
            roll_deg=roll_all,
            pitch_deg=pitch_all,
            normal=normal,
            realization_key=("SS5", 180.0, 0.0, "synthetic", 0),
        )
        assert context.index_at(t_td) == start
        oracle.reset(0, context)
        feed(oracle, obs_list[:1], spec.ctrl_dt_s, 12.0)

        assert gated.committed == oracle.committed == expected, (roll, pitch, vz)
        verdicts.append(expected)
    assert any(verdicts) and not all(verdicts)
