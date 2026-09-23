"""``pid_track_descend``: the naive baseline.

Contract checks (shape and bounds, determinism under reset, static-pad landing, the
privileged flag), the law's one non-obvious property -- it holds height until the lateral
error is inside the commit radius, because ~13 % of starts are outside the plate (P2-D3) --
and that the committed YAML carries exactly the tuning log's selected trial (P3-D3).

Units: metres, metres per second and seconds model scale.
"""

import numpy as np
import pytest

from _control_helpers import (
    assert_actions_in_space,
    committed_spec,
    feed,
    fuzz_actions,
    run_episode,
    selected_trial_params,
    synthetic_obs,
)
from conftest import REPO_ROOT
from rld.control.config import load_pid
from rld.control.pid import PidTrackDescend
from rld.control.registry import entry, make_controller

NAME = "pid_track_descend"


def test_privileged_flag_is_false():
    """Reads only the observation; the registry and the controller agree."""
    assert entry(NAME).privileged is False
    assert make_controller(NAME).privileged is False


def test_actions_stay_in_the_shared_space_on_arbitrary_observations(env_obs_cfg):
    """Any observation inside the declared bounds gives a float32 action in the unit ball."""
    controller = make_controller(NAME, committed_spec())
    controller.reset(0)
    assert_actions_in_space(fuzz_actions(controller, env_obs_cfg, 500, seed=1))


def test_holds_height_until_laterally_inside_the_commit_radius(env_obs_cfg):
    """v_z = 0 while outside the radius, -descent_rate once inside, released past 2x radius."""
    spec = committed_spec()
    controller = make_controller(NAME, spec)
    assert isinstance(controller, PidTrackDescend)
    cfg = controller.pid
    far = synthetic_obs(env_obs_cfg, rel_position_w=(0.25, 0.1, -0.5))
    near = synthetic_obs(env_obs_cfg, rel_position_w=(0.5 * cfg.commit_radius_m, 0.0, -0.5))
    mid = synthetic_obs(
        env_obs_cfg, rel_position_w=(0.5 * (1 + cfg.release_factor) * cfg.commit_radius_m, 0, -0.5)
    )
    controller.reset(0)
    v_max = spec.landing.v_max_m_s
    actions = feed(controller, [far, near, mid, far], spec.ctrl_dt_s, spec.landing.episode_len_s)
    assert actions[0][2] == 0.0
    assert actions[1][2] * v_max == pytest.approx(-cfg.descent_rate_m_s, rel=1e-6)
    assert actions[2][2] * v_max == pytest.approx(-cfg.descent_rate_m_s, rel=1e-6)  # hysteresis
    assert actions[3][2] == 0.0
    # Lateral command points at the pad.
    assert actions[0][0] > 0.0 and actions[0][1] > 0.0


@pytest.mark.pybullet
def test_deterministic_under_reset(landing_env, static_motion):
    """Same seed, same instance, another episode in between: bit-identical action stream."""
    env = landing_env(static_motion)
    controller = make_controller(NAME, committed_spec())
    first, _, _ = run_episode(env, controller, 3, privileged=False)
    run_episode(env, controller, 4, privileged=False)
    again, _, _ = run_episode(env, controller, 3, privileged=False)
    assert len(first) == len(again)
    np.testing.assert_array_equal(np.stack(first), np.stack(again))
    assert_actions_in_space(first)


@pytest.mark.pybullet
@pytest.mark.slow
def test_static_pad_landing_succeeds(landing_env, static_motion):
    """Ten static-pad episodes, including starts outside the plate footprint, all succeed."""
    env = landing_env(static_motion)
    controller = make_controller(NAME, committed_spec())
    outside = 0
    for seed in range(10):
        _, record, _ = run_episode(env, controller, seed, privileged=False)
        init = np.asarray(env.INIT_XYZS).reshape(3)
        outside += int(abs(init[1]) > env.cfg.platform.half_extents_m[1])
        assert record.outcome == "success", (seed, record.as_row())
        assert record.contact_events == 1
    assert outside >= 1, "the seeds should include a start outside the plate's beam"


def test_committed_yaml_is_the_selected_tuning_trial():
    """The YAML carries exactly the gains the tuning log selected (P3-D3)."""
    params = selected_trial_params(str(REPO_ROOT / "results" / "e01" / f"tuning_{NAME}.csv"))
    cfg = load_pid(entry(NAME).config_path)
    assert set(params) == {"kp_xy_per_s", "ki_xy_per_s2", "commit_radius_m", "descent_rate_m_s"}
    for key, value in params.items():
        assert getattr(cfg, key) == value, key
