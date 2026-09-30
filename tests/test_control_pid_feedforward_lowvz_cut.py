"""``pid_feedforward_lowvz_cut``: lowvz plus the latched post-contact throttle cut (P5-D2).

Contract checks (shape and bounds, determinism under reset, static-pad landing, the flags),
plus what makes this controller what P5-D2 says it is:

* **the config**: lowvz's gains by reference, never copied; the pinned SHA-256; the cut
  speed equals ``v_max``; the resolved-config hash carries the cut;
* **pre-contact identity**: until the latch, actions are bit-identical to lowvz's, both on
  synthetic streams and in a moving-deck episode (identical first-contact record too);
* **the latch**: it sets on the first ``in_contact`` = 1, holds when the flag drops, and
  ``reset`` clears it;
* **the command**: after the latch the setpoint is lowvz's minus ``cut_speed_m_s * n``, and
  in P5-D2's post-contact envelope ``DSLPIDControl`` turns it into all four motors at
  ``MIN_PWM`` (idle), checked through ``computeControl`` itself and in the environment.

Units: metres, metres per second and seconds model scale; newtons; RPM.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from _control_helpers import (
    assert_actions_in_space,
    committed_spec,
    deck_normal,
    feed,
    fuzz_actions,
    synthetic_obs,
)
from rld.control.base import ControlSpec
from rld.control.config import CONTROL_CONFIG_DIR, load_feedforward, load_lowvz_cut
from rld.control.feedforward import PidFeedforward
from rld.control.lowvz_cut import PidFeedforwardLowvzCut
from rld.control.obs_view import ObsLayout, obs_layout, setpoint_to_action
from rld.control.registry import entry, make_controller
from rld.envs.landing_env import DeckLandingAviary, EpisodeRecord

NAME = "pid_feedforward_lowvz_cut"
LOWVZ = "pid_feedforward_lowvz"
#: docs/protocol.md P5-D2. Changing the YAML's bytes requires a dated protocol entry.
CONFIG_SHA256 = "e5f19903a89165ecdc50218f2d626b7b41b03b566895f7be520f5d9a59e499bb"

#: gym-pybullet-drones CF2X ``DSLPIDControl``: ``MIN_PWM`` = 20 000 as RPM.
IDLE_RPM = 0.2685 * 20000 + 4070.3

#: First-contact columns of ``EpisodeRecord.as_row`` that the cut must leave unchanged.
TOUCHDOWN_COLUMNS = (
    "t0_model_s",
    "touchdown_contact",
    "touchdown_analytic",
    "td_t_episode_s",
    "td_t_episode_analytic_s",
    "rel_vz_normal_m_s",
    "rel_vz_world_z_m_s",
    "closing_speed_normal_m_s",
    "lateral_offset_m",
    "rel_tilt_deg",
    "abs_tilt_deg",
    "deck_tilt_deg",
    "n_contacts",
)


def _pair() -> tuple[PidFeedforwardLowvzCut, PidFeedforward]:
    """Return a fresh (cut, lowvz) pair on the committed configs."""
    spec = committed_spec()
    cut = make_controller(NAME, spec)
    lowvz = make_controller(LOWVZ, spec)
    assert isinstance(cut, PidFeedforwardLowvzCut)
    assert isinstance(lowvz, PidFeedforward)
    return cut, lowvz


# ---------------------------------------------------------------------------- flags, config


def test_flags_not_privileged_and_no_motion_feed():
    """Reads only the committed observation: not privileged, no ship-motion feed."""
    item = entry(NAME)
    assert item.privileged is False
    assert item.needs_motion_feed is False
    controller = make_controller(NAME)
    assert controller.name == NAME
    assert controller.privileged is False
    assert controller.needs_motion_feed is False
    with pytest.raises(ValueError, match="ShipMotionFeed"):
        controller.reset(0, None, object())  # type: ignore[arg-type]


def test_config_is_pinned_and_references_lowvz_gains():
    """SHA-256 as recorded in P5-D2; gains are lowvz's by reference; cut speed = v_max."""
    path = entry(NAME).config_path
    assert path == CONTROL_CONFIG_DIR / "pid_feedforward_lowvz_cut.yaml"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CONFIG_SHA256
    cfg = load_lowvz_cut(path)
    lowvz = load_feedforward(entry(LOWVZ).config_path)
    assert cfg.pid == lowvz.pid
    assert cfg.k_ff == lowvz.k_ff
    assert cfg.gains_from == (CONTROL_CONFIG_DIR / "pid_feedforward_lowvz.yaml").resolve()
    assert cfg.cut_speed_m_s == committed_spec().landing.v_max_m_s == 1.5


def test_loader_refuses_copied_gains_and_bad_cut(tmp_path: Path):
    """A gain key in the cut YAML could drift from lowvz's, so it is refused."""
    (tmp_path / "pid_feedforward_lowvz.yaml").write_bytes(
        (CONTROL_CONFIG_DIR / "pid_feedforward_lowvz.yaml").read_bytes()
    )
    good = "gains_from: pid_feedforward_lowvz.yaml\ncut_speed_m_s: 1.5\n"
    (tmp_path / "ok.yaml").write_text(good, encoding="utf-8")
    assert load_lowvz_cut(tmp_path / "ok.yaml").cut_speed_m_s == 1.5
    (tmp_path / "copied.yaml").write_text(good + "kp_xy_per_s: 1.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gains_from"):
        load_lowvz_cut(tmp_path / "copied.yaml")
    bad = "gains_from: pid_feedforward_lowvz.yaml\ncut_speed_m_s: 0.0\n"
    (tmp_path / "bad.yaml").write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="cut_speed_m_s"):
        load_lowvz_cut(tmp_path / "bad.yaml")


def test_resolved_config_hash_carries_the_cut_and_lowvz_gains():
    """The eval provenance hash moves with the cut, and carries lowvz's gains."""
    from rld.eval.controller_config import resolved_config_json, resolved_config_sha256

    doc = json.loads(resolved_config_json(NAME))
    lowvz = json.loads(resolved_config_json(LOWVZ))
    assert doc["class"] == "rld.control.lowvz_cut.PidFeedforwardLowvzCut"
    assert doc["config"]["gains_from"] == "pid_feedforward_lowvz.yaml"
    assert doc["config"]["cut_speed_m_s"] == 1.5
    assert doc["config"]["pid"] == lowvz["config"]["pid"]
    assert doc["config"]["k_ff"] == lowvz["config"]["k_ff"]
    assert resolved_config_sha256(NAME) != resolved_config_sha256(LOWVZ)


def test_refuses_an_observation_without_the_contact_flag():
    """Without ``in_contact`` the rule could never fire and would silently fly as lowvz."""
    spec = committed_spec()
    no_contact = ControlSpec(
        landing=spec.landing,
        observation=dataclasses.replace(spec.observation, include_contact=False),
        scale=spec.scale,
    )
    with pytest.raises(ValueError, match="in_contact"):
        make_controller(NAME, no_contact)


# ---------------------------------------------------------------------------- the law


def test_actions_stay_in_the_shared_space_on_arbitrary_observations(env_obs_cfg):
    """Any observation inside the declared bounds gives a float32 action in the unit ball."""
    controller = make_controller(NAME, committed_spec())
    controller.reset(0)
    assert_actions_in_space(fuzz_actions(controller, env_obs_cfg, 500, seed=11))


def test_pre_contact_actions_are_bit_identical_to_lowvz(env_obs_cfg):
    """With ``in_contact`` = 0 throughout, every action equals lowvz's exactly."""
    cut, lowvz = _pair()
    layout = obs_layout(env_obs_cfg)
    rng = np.random.default_rng(20260923)
    from rld.envs.observation import observation_space

    space = observation_space(env_obs_cfg)
    obs_list = []
    for _ in range(400):
        obs = rng.uniform(space.low.astype(np.float64), space.high.astype(np.float64))
        obs[layout.slices["in_contact"]] = 0.0
        obs_list.append(obs.astype(np.float32))
    spec = cut.spec
    cut.reset(3)
    lowvz.reset(3)
    a_cut = feed(cut, obs_list, spec.ctrl_dt_s, spec.landing.episode_len_s)
    a_low = feed(lowvz, obs_list, spec.ctrl_dt_s, spec.landing.episode_len_s)
    np.testing.assert_array_equal(np.stack(a_cut), np.stack(a_low))
    assert not cut.latched and cut.latch_step is None


def _descent_stream(env_obs_cfg, n: int, contact_at: set[int]) -> list[np.ndarray]:
    """A plausible descent onto a tilted, moving deck, with the contact flag where given."""
    normal = deck_normal(4.0, -3.0)
    out = []
    for k in range(n):
        out.append(
            synthetic_obs(
                env_obs_cfg,
                yaw_rad=0.05,
                velocity_w=(0.02, -0.01, -0.12 + 0.1 * np.sin(0.3 * k)),
                rel_position_w=(0.01, -0.005, -max(0.3 - 0.01 * k, 0.0)),
                deck_velocity_w=(0.03, 0.01, 0.1 * np.sin(0.3 * k)),
                normal_w=normal,
                height_m=max(0.3 - 0.01 * k, 0.0),
                in_contact=k in contact_at,
            )
        )
    return out


def test_latch_sets_on_first_contact_holds_and_commands_the_cut(env_obs_cfg):
    """Identical before the flag; from it on, lowvz minus ``cut * n`` even after the flag drops."""
    cut, lowvz = _pair()
    spec = cut.spec
    first = 20
    # Contact for two steps, a gap (flag drops), then contact again.
    stream = _descent_stream(env_obs_cfg, 40, {first, first + 1, first + 5, first + 6})
    cut.reset(0)
    lowvz.reset(0)
    layout = obs_layout(env_obs_cfg)
    for k, obs in enumerate(stream):
        stamped = obs.copy()
        stamped[layout.slices["time_fraction"]] = k * spec.ctrl_dt_s / spec.landing.episode_len_s
        s_cut = cut.setpoint_m_s(stamped)
        s_low = lowvz.setpoint_m_s(stamped)
        if k < first:
            np.testing.assert_array_equal(s_cut, s_low)
            np.testing.assert_array_equal(
                setpoint_to_action(s_cut, spec.landing), setpoint_to_action(s_low, spec.landing)
            )
            assert not cut.latched
        else:
            assert cut.latched and cut.latch_step == first
            normal = _observed_normal(stamped, layout)
            np.testing.assert_allclose(s_cut, s_low - 1.5 * normal, rtol=0, atol=1e-12)
            # Strong descent: the normalised action points down, near the cap.
            action = setpoint_to_action(s_cut, spec.landing).astype(np.float64)
            assert action[2] < -0.75
    # reset clears the latch; the next episode starts as lowvz again.
    cut.reset(0)
    lowvz.reset(0)
    assert not cut.latched and cut.latch_step is None
    a_cut = feed(cut, stream[:first], spec.ctrl_dt_s, spec.landing.episode_len_s)
    a_low = feed(lowvz, stream[:first], spec.ctrl_dt_s, spec.landing.episode_len_s)
    np.testing.assert_array_equal(np.stack(a_cut), np.stack(a_low))


def _observed_normal(obs: np.ndarray, layout: ObsLayout) -> np.ndarray:
    """The world-frame deck normal the controller reads (through ``ObsView``)."""
    from rld.control.obs_view import view

    return np.asarray(view(obs, layout).deck_normal, dtype=np.float64)


def _body_axes(normal: np.ndarray, rel_deg: float) -> list[np.ndarray]:
    """The deck normal and four axes tilted ``rel_deg`` from it (unit vectors)."""
    from scipy.spatial.transform import Rotation

    out = [normal]
    perp = np.cross(normal, [1.0, 0.0, 0.0])
    perp /= np.linalg.norm(perp)
    for azimuth in np.radians([0.0, 90.0, 180.0, 270.0]):
        axis = Rotation.from_rotvec(azimuth * normal).apply(perp)
        out.append(Rotation.from_rotvec(np.radians(rel_deg) * axis).apply(normal))
    return out


def _quat_with_z(axis: np.ndarray) -> np.ndarray:
    """A quaternion ``[x, y, z, w]`` whose body z is ``axis``, with no twist about it."""
    from scipy.spatial.transform import Rotation

    rotation, _ = Rotation.align_vectors([axis], [[0.0, 0.0, 1.0]])
    return np.asarray(rotation.as_quat(), dtype=np.float64)


def test_post_latch_command_puts_every_motor_at_idle_in_the_p5d2_envelope(env_obs_cfg):
    """Through ``DSLPIDControl.computeControl`` exactly as the environment calls it.

    P5-D2's post-contact envelope: deck tilt up to 15 deg, body axis within 15 deg of the
    deck normal, |v_pad,z| <= 0.7 m/s, |v_pad,xy| <= 0.3 m/s, drone-vs-pad velocity
    <= 0.1 m/s per axis, lateral error inside the commit radius or far outside it (PI at its
    0.5 m/s cap, gate closed).
    """
    import pybullet as pyb
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.utils.enums import DroneModel

    spec = committed_spec()
    dsl = DSLPIDControl(drone_model=DroneModel.CF2X)
    cut = make_controller(NAME, spec)
    assert isinstance(cut, PidFeedforwardLowvzCut)
    checked = 0
    for roll, pitch in [(-15, -15), (-15, 15), (15, -15), (15, 15), (0, 0), (8, -4)]:
        normal = np.asarray(deck_normal(roll, pitch), dtype=np.float64)
        for v_pad in [(0.3, -0.3, -0.7), (-0.3, 0.3, 0.7), (0.0, 0.0, 0.0), (0.3, 0.3, -0.35)]:
            for dv in [(0, 0, 0.1), (0, 0, -0.1), (0.1, 0, 0), (-0.1, 0, 0), (0, 0.1, 0)]:
                v_drone = np.asarray(v_pad) + np.asarray(dv, dtype=np.float64)
                for lateral in [(0.0, 0.0), (0.05, 0.0), (0.5, 0.0), (0.0, -0.5), (-0.4, 0.4)]:
                    obs = synthetic_obs(
                        env_obs_cfg,
                        velocity_w=tuple(v_drone),
                        rel_position_w=(lateral[0], lateral[1], 0.0),
                        deck_velocity_w=v_pad,
                        normal_w=tuple(normal),
                        height_m=0.0,
                        in_contact=True,
                    )
                    cut.reset(0)
                    action = cut.act(obs)
                    assert cut.latched
                    target = spec.landing.v_max_m_s * action.astype(np.float64)
                    for body_z in _body_axes(normal, 15.0):
                        quat = _quat_with_z(body_z)
                        dsl.reset()
                        # No spurious attitude rate from the controller's previous call.
                        dsl.last_rpy = np.asarray(pyb.getEulerFromQuaternion(quat))
                        rpm, _, _ = dsl.computeControl(
                            control_timestep=spec.ctrl_dt_s,
                            cur_pos=np.zeros(3),
                            cur_quat=quat,
                            cur_vel=v_drone,
                            cur_ang_vel=np.zeros(3),
                            target_pos=np.zeros(3),
                            target_rpy=np.zeros(3),
                            target_vel=target,
                        )
                        np.testing.assert_allclose(rpm, IDLE_RPM, rtol=0, atol=1e-6)
                        checked += 1
    assert checked == 6 * 4 * 5 * 5 * 5


# ---------------------------------------------------------------------------- in the env


def _run(
    env: DeckLandingAviary, controller, seed: int
) -> tuple[list[np.ndarray], EpisodeRecord, list[np.ndarray]]:
    """One episode; returns actions, the record, and the applied RPMs after each step."""
    obs, _ = env.reset(seed=seed)
    controller.reset(seed)
    actions, rpms = [], []
    for _ in range(int(env.cfg.total_len_s * env.cfg.ctrl_freq_hz) + 1):
        action = controller.act(obs)
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        rpms.append(np.asarray(env.last_clipped_action, dtype=np.float64).reshape(-1).copy())
        if terminated or truncated:
            break
    return actions, env.record, rpms


@pytest.mark.pybullet
def test_moving_deck_episode_is_lowvz_until_the_latch(landing_env, deck_source):
    """Same episode: actions bit-identical up to the latch step, first contact identical."""
    from dmf.sim.generate import RealizationSpec

    spec = RealizationSpec(
        sea_state="SS4", heading_deg=180.0, speed_kn=12.0, vessel="frigate", seed=2
    )
    env = landing_env(deck_source(spec))
    cut, lowvz = _pair()
    a_cut, rec_cut, _ = _run(env, cut, 7)
    row_cut = rec_cut.as_row()
    a_low, rec_low, _ = _run(env, lowvz, 7)
    row_low = rec_low.as_row()
    assert row_cut["touchdown_contact"] is True
    assert cut.latched and cut.latch_step is not None
    k = cut.latch_step
    np.testing.assert_array_equal(np.stack(a_cut[:k]), np.stack(a_low[:k]))
    for column in TOUCHDOWN_COLUMNS:
        assert row_cut[column] == row_low[column], column
    # The latch is the first observation carrying the flag, so never before first contact.
    # (Not necessarily the first boundary after it: a rocking gap can hide the flag, P5-D2.)
    assert k * env.cfg.ctrl_dt_s >= row_cut["td_t_episode_s"] - 1e-9


@pytest.mark.pybullet
def test_deterministic_under_reset(landing_env, deck_source):
    """Same seed, same instance, another episode in between: bit-identical actions."""
    from dmf.sim.generate import RealizationSpec

    spec = RealizationSpec(
        sea_state="SS4", heading_deg=180.0, speed_kn=12.0, vessel="frigate", seed=2
    )
    env = landing_env(deck_source(spec))
    controller = make_controller(NAME, committed_spec())
    first, _, _ = _run(env, controller, 7)
    _run(env, controller, 8)
    again, _, _ = _run(env, controller, 7)
    np.testing.assert_array_equal(np.stack(first), np.stack(again))
    assert_actions_in_space(first)


@pytest.mark.pybullet
@pytest.mark.slow
def test_static_pad_landing_succeeds_with_motors_idle_after_the_latch(landing_env, static_motion):
    """Ten static-pad episodes succeed; after the latch every motor is at idle."""
    env = landing_env(static_motion)
    controller = make_controller(NAME, committed_spec())
    assert isinstance(controller, PidFeedforwardLowvzCut)
    for seed in range(10):
        _, record, rpms = _run(env, controller, seed)
        assert record.outcome == "success", (seed, record.as_row())
        assert record.contact_events == 1
        k = controller.latch_step
        assert k is not None
        post = np.stack(rpms[k:])
        assert post.size, seed
        np.testing.assert_allclose(post, IDLE_RPM, rtol=0, atol=1e-6)
