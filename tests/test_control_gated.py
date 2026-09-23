"""``gated``: hover above the pad, commit on dmf quiescence of the current deck.

Contract checks (shape and bounds, determinism under reset, static-pad landing, the
privileged flag), and the two things a reviewer would attack first:

* **The Froude conversion of dmf's thresholds** -- imported from ``dmf.eval.quiescence``,
  not copied; angles unchanged, heave-rate and sustain times ``sqrt(lam)``:
  0.8 -> 0.16 m/s and 2.0 -> 0.4 s (permissive), 0.4 -> 0.08 m/s and 3.0 -> 0.6 s (strict)
  at ``lam = 1/25``.
* **The rule** -- commits only after ``sustain`` of in-limit samples at hover, never commits
  on a deck that stays outside the limits (with the default ``fallback_commit_s: null``),
  and inherits ``pid_feedforward``'s gains by reference.

Units: metres, metres per second and seconds model scale; degrees.
"""

import numpy as np
import pytest
from dmf.eval import quiescence as dmf_q

from _control_helpers import (
    assert_actions_in_space,
    committed_spec,
    feed,
    fuzz_actions,
    run_episode,
    synthetic_obs,
)
from rld.control.config import load_feedforward, load_gated
from rld.control.gated import Gated
from rld.control.quiescence import QuiescenceRule, froude_limits, limits_for
from rld.control.registry import entry, make_controller
from rld.deck.scaling import FroudeScale

NAME = "gated"


def test_privileged_flag_is_false():
    """Reads the current deck state from the observation only."""
    assert entry(NAME).privileged is False
    assert make_controller(NAME).privileged is False


def test_froude_conversion_of_dmf_thresholds():
    """Angles unchanged; rate and sustain x sqrt(1/25) = 0.2; dmf's objects, not copies."""
    scale = committed_spec().scale
    assert scale.lam == pytest.approx(1.0 / 25.0)
    permissive = limits_for("permissive", scale)
    strict = limits_for("strict", scale)
    assert permissive.source is dmf_q.PERMISSIVE
    assert strict.source is dmf_q.STRICT
    assert (permissive.roll_deg, permissive.pitch_deg) == (3.0, 2.0)
    assert (strict.roll_deg, strict.pitch_deg) == (1.5, 1.0)
    assert permissive.pad_vz_m_s == pytest.approx(0.16, abs=1e-12)
    assert strict.pad_vz_m_s == pytest.approx(0.08, abs=1e-12)
    assert permissive.sustain_s == pytest.approx(0.4, abs=1e-12)
    assert strict.sustain_s == pytest.approx(0.6, abs=1e-12)
    # dmf's run-length convention: ceil(sustain * fs).
    assert permissive.sustain_samples(30.0) == 12
    assert strict.sustain_samples(30.0) == 18
    assert permissive.sustain_samples(240.0) == 96
    assert strict.sustain_samples(240.0) == 144
    # Identity at lam = 1 and the sqrt law at another lam, so the factor is not a literal.
    identity = froude_limits(dmf_q.PERMISSIVE, FroudeScale(lam=1.0))
    assert (identity.pad_vz_m_s, identity.sustain_s) == (0.8, 2.0)
    other = froude_limits(dmf_q.STRICT, FroudeScale(lam=1.0 / 100.0))
    assert other.pad_vz_m_s == pytest.approx(0.04) and other.sustain_s == pytest.approx(0.3)
    assert other.roll_deg == 1.5


def test_shared_rule_is_n_samples_all_inside_all_limits():
    """Exactly n control-rate samples, every sample, every channel, limits inclusive."""
    spec = committed_spec()
    lim = limits_for("permissive", spec.scale)
    rule = QuiescenceRule(lim, float(spec.landing.ctrl_freq_hz))
    assert rule.n_samples == 12
    assert rule.spacing_s == pytest.approx(1.0 / 30.0)
    ok = np.zeros(12)
    assert rule.verdict(ok, ok, ok)
    at_limit = np.full(12, 1.0)
    assert rule.verdict(at_limit * 3.0, -at_limit * 2.0, at_limit * lim.pad_vz_m_s)
    bad_vz = ok.copy()
    bad_vz[5] = 0.1601
    assert not rule.verdict(ok, ok, bad_vz)
    bad_pitch = ok.copy()
    bad_pitch[0] = -2.01
    assert not rule.verdict(ok, bad_pitch, ok)
    bad_roll = ok.copy()
    bad_roll[11] = 3.01
    assert not rule.verdict(bad_roll, ok, ok)
    # Too short (history not yet long enough, or a future window past the end): no verdict.
    assert not rule.verdict(ok[:11], ok[:11], ok[:11])
    assert not rule.verdict(ok[:0], ok[:0], ok[:0])
    with pytest.raises(ValueError, match="12"):
        rule.verdict(np.zeros(13), np.zeros(13), np.zeros(13))
    with pytest.raises(ValueError, match="differ"):
        rule.verdict(ok, ok, ok[:11])


def test_both_gated_controllers_hold_the_same_rule():
    """The two gated controllers build one predicate, at the control rate, in one place."""
    gated = make_controller(NAME, committed_spec())
    oracle = make_controller("oracle_gated", committed_spec())
    assert isinstance(gated, Gated)
    assert gated.rule == oracle.rule  # type: ignore[attr-defined]
    assert gated.rule.rate_hz == 30.0
    assert gated.rule.n_samples == gated.sustain_samples_needed == 12


def test_gains_are_pid_feedforwards_by_reference():
    """Loaded from pid_feedforward.yaml, not a copy that could go stale."""
    cfg = load_gated(entry(NAME).config_path)
    assert cfg.gains == load_feedforward(entry("pid_feedforward").config_path)
    assert cfg.gains_from == entry("pid_feedforward").config_path.resolve()
    assert cfg.thresholds == "permissive"
    assert cfg.fallback_commit_s is None


def test_actions_stay_in_the_shared_space_on_arbitrary_observations(env_obs_cfg):
    """Any observation inside the declared bounds gives a float32 action in the unit ball."""
    controller = make_controller(NAME, committed_spec())
    controller.reset(0)
    assert_actions_in_space(fuzz_actions(controller, env_obs_cfg, 500, seed=3))


def _at_hover(env_obs_cfg, controller: Gated, vz_pad: float, pitch_deg: float = 0.0):
    p = np.radians(pitch_deg)
    return synthetic_obs(
        env_obs_cfg,
        rel_position_w=(0.0, 0.0, -controller.gated.hover_height_m),
        deck_velocity_w=(0.0, 0.0, vz_pad),
        normal_w=(-np.sin(p), 0.0, np.cos(p)),
        height_m=controller.gated.hover_height_m,
    )


def test_commits_only_after_sustain_of_quiet_samples(env_obs_cfg):
    """12 quiet control samples at hover (0.4 s at 30 Hz) and not one fewer."""
    spec = committed_spec()
    controller = make_controller(NAME, spec)
    assert isinstance(controller, Gated)
    needed = controller.sustain_samples_needed
    assert needed == 12
    loud = _at_hover(env_obs_cfg, controller, vz_pad=0.3)
    quiet = _at_hover(env_obs_cfg, controller, vz_pad=0.05)
    controller.reset(0)
    feed(controller, [loud] * 5 + [quiet] * (needed - 1), spec.ctrl_dt_s, 12.0)
    assert not controller.committed
    controller.reset(0)
    feed(controller, [loud] * 5 + [quiet] * needed, spec.ctrl_dt_s, 12.0)
    assert controller.committed
    assert controller.commit_time_s == pytest.approx((5 + needed - 1) * spec.ctrl_dt_s)


def test_never_commits_on_a_deck_that_never_goes_quiet(env_obs_cfg):
    """fallback_commit_s: null -> the controller hovers to the time limit (a timeout)."""
    spec = committed_spec()
    controller = make_controller(NAME, spec)
    assert isinstance(controller, Gated)
    pitching = _at_hover(env_obs_cfg, controller, vz_pad=0.0, pitch_deg=3.0)
    controller.reset(0)
    actions = feed(controller, [pitching] * 360, spec.ctrl_dt_s, 12.0)
    assert not controller.committed
    # Hovering at the hover height with a still pad: no vertical command at all.
    assert max(abs(float(a[2])) for a in actions) < 1e-6


@pytest.mark.pybullet
def test_deterministic_under_reset(landing_env, static_motion):
    """Same seed, same instance, another episode in between: bit-identical actions."""
    env = landing_env(static_motion)
    controller = make_controller(NAME, committed_spec())
    first, _, _ = run_episode(env, controller, 3, privileged=False)
    run_episode(env, controller, 4, privileged=False)
    again, _, _ = run_episode(env, controller, 3, privileged=False)
    np.testing.assert_array_equal(np.stack(first), np.stack(again))
    assert_actions_in_space(first)


@pytest.mark.pybullet
@pytest.mark.slow
def test_static_pad_landing_succeeds(landing_env, static_motion):
    """A static deck is always quiescent: hover, commit, land, ten out of ten."""
    env = landing_env(static_motion)
    controller = make_controller(NAME, committed_spec())
    for seed in range(10):
        _, record, _ = run_episode(env, controller, seed, privileged=False)
        assert record.outcome == "success", (seed, record.as_row())
