"""The controller contract's shared pieces: observation access, action scaling, the oracle feed.

These are the places a baseline can be silently wrong while every controller test passes:

* **Action scaling.** A controller that scaled its setpoint differently from
  ``DeckLandingAviary.velocity_setpoint_m_s`` would fly a different action space from PPO.
  :func:`setpoint_to_action` must invert it exactly, norm cap included (P2-D2).
* **Frames and derived quantities.** The observation is in the drone-yaw frame; deck
  velocity is ``velocity + rel_velocity`` (the environment defines
  ``rel_velocity = v_pad - v_drone``); deck roll/pitch come from the normal through P1-D2.
  Each is checked against the environment's own analytic deck state, not re-derived.
* **The privileged feed** must be the trajectory the plate is actually driven along, and
  read-only.
* **The registry** must mark exactly one controller privileged.

Units: metres, metres per second and seconds model scale; degrees at boundaries.
"""

import numpy as np
import pytest
from dmf.sim.generate import RealizationSpec

from _control_helpers import committed_spec, synthetic_obs
from rld.control.base import PrivilegedContext, StepClock
from rld.control.obs_view import deck_angles_deg, obs_layout, setpoint_to_action, view
from rld.control.registry import REGISTRY, make_controller
from rld.envs.observation import obs_fields

#: A moving deck with real roll, pitch and heave: frigate SS5 at 135 deg (roll-bearing).
SS5_135 = RealizationSpec(
    sea_state="SS5", heading_deg=135.0, speed_kn=6.0, vessel="frigate", seed=1
)


def test_layout_comes_from_the_declared_table(env_obs_cfg):
    """Offsets are running sums of the declared sizes; no literal index anywhere."""
    layout = obs_layout(env_obs_cfg)
    fields = obs_fields(env_obs_cfg)
    assert layout.size == sum(f.size for f in fields)
    start = 0
    for f in fields:
        assert layout.slices[f.name] == slice(start, start + f.size)
        start += f.size


@pytest.mark.pybullet
def test_setpoint_to_action_inverts_the_env_scaling(landing_env, static_motion, rng):
    """``velocity_setpoint_m_s(setpoint_to_action(v))`` returns the norm-capped ``v``.

    Includes setpoints far beyond ``v_max`` on diagonals, where a per-axis clip would change
    the direction and the norm cap must not.
    """
    env = landing_env(static_motion)
    v_max = env.cfg.v_max_m_s
    for _ in range(200):
        v = rng.normal(0.0, 1.5, size=3)
        action = setpoint_to_action(v, env.cfg)
        assert action.dtype == np.float32
        speed = float(np.linalg.norm(v))
        expected = v if speed <= v_max else v * (v_max / speed)
        np.testing.assert_allclose(env.velocity_setpoint_m_s(action), expected, atol=1e-6)
    # Direction is preserved on a long diagonal; a per-axis clip would give (1, 1, -1).
    long = np.array([3.0, 3.0, -0.3])
    got = env.velocity_setpoint_m_s(setpoint_to_action(long, env.cfg))
    np.testing.assert_allclose(got / np.linalg.norm(got), long / np.linalg.norm(long), atol=1e-6)


def test_view_rotates_drone_yaw_back_to_world(env_obs_cfg):
    """Every vector block comes back in the world frame at a non-zero yaw."""
    yaw = 0.7
    obs = synthetic_obs(
        env_obs_cfg,
        yaw_rad=yaw,
        velocity_w=(0.3, -0.2, 0.1),
        rel_position_w=(0.12, 0.05, -0.6),
        deck_velocity_w=(0.01, 0.0, 0.25),
        normal_w=(-0.05, 0.02, float(np.sqrt(1 - 0.05**2 - 0.02**2))),
    )
    v = view(obs, obs_layout(env_obs_cfg))
    assert v.yaw_rad == pytest.approx(yaw, abs=1e-6)
    np.testing.assert_allclose(v.velocity_m_s, [0.3, -0.2, 0.1], atol=1e-6)
    np.testing.assert_allclose(v.rel_position_m, [0.12, 0.05, -0.6], atol=1e-6)
    np.testing.assert_allclose(v.deck_velocity_m_s, [0.01, 0.0, 0.25], atol=1e-6)
    np.testing.assert_allclose(v.lateral_error_m, [0.12, 0.05], atol=1e-6)


@pytest.mark.parametrize(
    ("roll_deg", "pitch_deg"), [(0.0, 0.0), (3.0, 0.0), (0.0, 2.0), (-4.5, 1.5), (7.0, -6.0)]
)
def test_deck_angles_invert_the_p1_d2_normal(roll_deg, pitch_deg):
    """``n = (-sin(pitch) cos(roll), -sin(roll), cos(pitch) cos(roll))`` inverts exactly."""
    r, p = np.radians(roll_deg), np.radians(pitch_deg)
    normal = np.array([-np.sin(p) * np.cos(r), -np.sin(r), np.cos(p) * np.cos(r)])
    got_roll, got_pitch = deck_angles_deg(normal)
    assert got_roll == pytest.approx(roll_deg, abs=1e-9)
    assert got_pitch == pytest.approx(pitch_deg, abs=1e-9)


@pytest.mark.pybullet
def test_view_matches_the_env_deck_state_on_a_moving_deck(landing_env, deck_source):
    """Derived deck velocity, roll and pitch equal the environment's analytic deck state.

    Also the anti-assertion: ``velocity - rel_velocity`` is **not** the deck velocity, since
    the environment defines ``rel_velocity = v_pad - v_drone``.
    """
    env = landing_env(deck_source(SS5_135))
    obs, _ = env.reset(seed=5)
    layout = obs_layout(env._obs_cfg)  # noqa: SLF001
    worst_wrong = 0.0
    for step in range(40):
        action = np.array([0.1 * np.sin(step / 5.0), 0.1, -0.1], dtype=np.float32)
        obs, *_ = env.step(action)
        v = view(obs, layout)
        deck = env._deck_sample()  # noqa: SLF001
        np.testing.assert_allclose(v.deck_velocity_m_s, deck.velocity_m_s, atol=2e-6)
        euler = np.asarray(deck.euler_xyz_rad)
        assert v.deck_roll_deg == pytest.approx(np.degrees(euler[0]), abs=1e-4)
        assert v.deck_pitch_deg == pytest.approx(-np.degrees(euler[1]), abs=1e-4)
        wrong = v.velocity_m_s - v.rel_velocity_m_s
        worst_wrong = max(worst_wrong, float(np.max(np.abs(wrong - deck.velocity_m_s))))
    assert worst_wrong > 1e-2


@pytest.mark.pybullet
def test_privileged_context_is_the_driven_trajectory(landing_env, deck_source):
    """``from_env`` rebuilds the plate's trajectory bit for bit, and it is read-only."""
    env = landing_env(deck_source(SS5_135))
    env.reset(seed=11)
    ctx = PrivilegedContext.from_env(env)
    traj = env._trajectory  # noqa: SLF001
    assert len(ctx) == len(traj) == env.cfg.n_physics_samples
    np.testing.assert_array_equal(ctx.position_m, traj.position_m)
    np.testing.assert_array_equal(ctx.velocity_m_s, traj.state.velocity_m_s)
    assert ctx.t0_model_s == env.record.t0_model_s
    assert ctx.realization_key == env.motion.key
    with pytest.raises(ValueError, match="read-only"):
        ctx.velocity_m_s[0, 2] = 99.0
    # Windows: 96 samples = 0.4 s at 240 Hz; one past the end is incomplete.
    assert ctx.window(1.0, 96).complete
    assert ctx.window(1.0, 96).pad_vz_m_s.size == 96
    end = ctx.window(float(ctx.t_episode_s[-1]) - 0.1, 96)
    assert not end.complete
    # The oracle's window: 12 samples every 8th physics sample = 1/30 s apart (control rate).
    strided = ctx.window(1.0, 12, stride=8)
    assert strided.complete and strided.normal.shape == (12, 3)
    np.testing.assert_allclose(np.diff(strided.t_episode_s), 1.0 / 30.0, atol=1e-12)
    np.testing.assert_array_equal(strided.pad_vz_m_s, ctx.velocity_m_s[240:336:8, 2])
    assert not ctx.window(float(ctx.t_episode_s[-1]) - 0.3, 12, stride=8).complete


def test_step_clock_counts_and_catches_a_missed_reset():
    """The counter matches time_fraction; a double act() per step is caught."""
    clock = StepClock(ctrl_dt_s=1.0 / 30.0, episode_len_s=12.0)
    for k in range(10):
        assert clock.tick(k / 30.0 / 12.0) == pytest.approx(k / 30.0)
    with pytest.raises(RuntimeError, match="reset"):
        clock.tick(3.0 / 30.0 / 12.0)
    # Saturated time_fraction (the dwell grace) is not checked.
    clock.reset()
    clock.steps = 365
    assert clock.tick(1.0) == pytest.approx(365 / 30.0)


def test_registry_marks_exactly_the_oracle_privileged():
    """One privileged entry, and each controller's own flag agrees with the registry."""
    spec = committed_spec()
    assert list(REGISTRY) == [
        "pid_track_descend",
        "pid_feedforward",
        "pid_feedforward_lowvz",
        "gated",
        "oracle_gated",
    ]
    assert [n for n, e in REGISTRY.items() if e.privileged] == ["oracle_gated"]
    for name, item in REGISTRY.items():
        assert item.config_path.is_file()
        controller = make_controller(name, spec)
        assert controller.name == name
        assert controller.privileged == item.privileged
