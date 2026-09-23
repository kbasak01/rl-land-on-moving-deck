"""``pid_feedforward``: the residual-RL base.

Contract checks (shape and bounds, determinism under reset, static-pad landing, the
privileged flag), the feedforward law itself -- the setpoint differs from
``pid_track_descend``'s by exactly ``k_ff * v_pad`` on all three axes, with ``v_pad``
derived from the observation -- and that the committed YAML carries exactly the tuning log's
selected trial (P3-D3).

Units: metres, metres per second and seconds model scale.
"""

import numpy as np
import pytest

from _control_helpers import (
    assert_actions_in_space,
    committed_spec,
    fuzz_actions,
    run_episode,
    selected_trial_params,
    synthetic_obs,
)
from conftest import REPO_ROOT
from rld.control.config import load_feedforward
from rld.control.feedforward import PidFeedforward
from rld.control.pid import PidTrackDescend
from rld.control.registry import entry, make_controller

NAME = "pid_feedforward"


def test_privileged_flag_is_false():
    """v_pad is derivable from the observation, so not privileged.

    Physically: an ideal (noise-free, zero-latency) ship motion reference unit plus state
    estimate.
    """
    assert entry(NAME).privileged is False
    assert make_controller(NAME).privileged is False


def test_actions_stay_in_the_shared_space_on_arbitrary_observations(env_obs_cfg):
    """Any observation inside the declared bounds gives a float32 action in the unit ball."""
    controller = make_controller(NAME, committed_spec())
    controller.reset(0)
    assert_actions_in_space(fuzz_actions(controller, env_obs_cfg, 500, seed=2))


@pytest.mark.parametrize("lateral_m", [0.0, 0.2])
def test_setpoint_is_the_pid_setpoint_plus_k_ff_times_pad_velocity(env_obs_cfg, lateral_m):
    """Same gains, same observation: the difference is exactly ``k_ff * v_pad`` (3 axes).

    Checked both inside the commit radius (descending) and outside it (holding height above
    the pad rather than in the world).
    """
    spec = committed_spec()
    controller = make_controller(NAME, spec)
    assert isinstance(controller, PidFeedforward)
    naive = PidTrackDescend(controller.pid, spec)
    v_pad = (0.02, -0.01, 0.31)
    obs = synthetic_obs(
        env_obs_cfg,
        yaw_rad=0.3,
        velocity_w=(0.05, 0.0, -0.1),
        rel_position_w=(lateral_m, 0.0, -0.4),
        deck_velocity_w=v_pad,
    )
    controller.reset(0)
    naive.reset(0)
    difference = controller.setpoint_m_s(obs) - naive.setpoint_m_s(obs)
    np.testing.assert_allclose(difference, controller.ff.k_ff * np.asarray(v_pad), atol=1e-6)


@pytest.mark.pybullet
def test_deterministic_under_reset(landing_env, deck_source):
    """Same seed, same instance, another episode in between: bit-identical actions.

    On a moving deck, because the feedforward is what varies with the deck.
    """
    from dmf.sim.generate import RealizationSpec

    spec = RealizationSpec(
        sea_state="SS4", heading_deg=180.0, speed_kn=12.0, vessel="frigate", seed=2
    )
    env = landing_env(deck_source(spec))
    controller = make_controller(NAME, committed_spec())
    first, _, _ = run_episode(env, controller, 7, privileged=False)
    run_episode(env, controller, 8, privileged=False)
    again, _, _ = run_episode(env, controller, 7, privileged=False)
    np.testing.assert_array_equal(np.stack(first), np.stack(again))
    assert_actions_in_space(first)


@pytest.mark.pybullet
@pytest.mark.slow
def test_static_pad_landing_succeeds(landing_env, static_motion):
    """Ten static-pad episodes, including starts outside the plate footprint, all succeed."""
    env = landing_env(static_motion)
    controller = make_controller(NAME, committed_spec())
    for seed in range(10):
        _, record, _ = run_episode(env, controller, seed, privileged=False)
        assert record.outcome == "success", (seed, record.as_row())
        assert record.contact_events == 1


def test_committed_yaml_is_the_selected_tuning_trial():
    """The YAML carries exactly the gains the tuning log selected (P3-D3)."""
    params = selected_trial_params(str(REPO_ROOT / "results" / "e01" / f"tuning_{NAME}.csv"))
    cfg = load_feedforward(entry(NAME).config_path)
    assert set(params) == {
        "kp_xy_per_s",
        "ki_xy_per_s2",
        "commit_radius_m",
        "descent_rate_m_s",
        "k_ff",
    }
    assert cfg.k_ff == params.pop("k_ff")
    for key, value in params.items():
        assert getattr(cfg.pid, key) == value, key
