"""The landing environment: API contract, determinism, and the two scripted static-pad runs.

What these tests are for
------------------------
This environment is what every method in the project is judged inside, so a bug here does
not produce a wrong number in one table -- it produces a wrong number in every table, with
no signal that anything is wrong. The checks below are the ones whose failure would be
invisible in a results CSV:

* ``check_env`` -- the Gymnasium contract, so SB3 does not silently reshape anything.
* determinism -- the same seed must give bit-identical observations, run as the environment's
  1st reset and again as its 3rd, because the three ways this breaks (``DSLPIDControl``
  integral state, the noise stream, the episode-start draw) all survive a reset.
* truncation semantics -- ``timeout`` must arrive as ``truncated``, or PPO learns to hover.
* the scripted static-pad runs -- with the deck removed as a variable, a 0.3 m/s descent
  must succeed 100 times out of 100, and a drop must register exactly one touchdown.

Units: metres, metres per second and seconds are model scale; angles radians in the
observation and degrees at the boundary.
"""

import numpy as np
import pytest
from dmf.sim.generate import RealizationSpec
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env

from rld.envs.config import NoiseConfig
from rld.envs.landing_env import ACTION_DIM, DeckLandingAviary
from rld.envs.observation import obs_fields, observation_space
from rld.envs.platform import StaticDeckMotion, build_trajectory, pad_offset_model_m
from rld.envs.reward import terminal_reward

#: A JONSWAP cell used where real deck motion matters. SS5 head seas at 12 kn is the
#: frigate's worst aft-pad cell in P1-D2's table.
SS5_SPEC = RealizationSpec(
    sea_state="SS5", heading_deg=180.0, speed_kn=12.0, vessel="frigate", seed=0
)


def _scripted_descent(obs: np.ndarray, env: DeckLandingAviary, descent_m_s: float = 0.3):
    """Return the scripted controller's action: null the lateral error, descend at a rate.

    Deliberately trivial. It is not a baseline -- Phase 3 owns those -- it is the minimum
    controller that turns "the environment can be landed in" into a testable statement.

    Args:
        obs: The current observation.
        env: The environment, for ``v_max`` and the observation layout.
        descent_m_s: Descent rate, metres per second model scale.

    Returns:
        A ``(3,)`` float32 action in ``[-1, 1]^3``.
    """
    offsets = {}
    start = 0
    for field in obs_fields(env._obs_cfg):  # noqa: SLF001
        offsets[field.name] = (start, start + field.size)
        start += field.size
    lo, hi = offsets["rel_position"]
    rel = np.asarray(obs[lo:hi], dtype=np.float64)
    lateral = np.clip(1.5 * rel[:2], -0.4, 0.4)
    command = np.array([lateral[0], lateral[1], -descent_m_s]) / env.cfg.v_max_m_s
    return np.clip(command, -1.0, 1.0).astype(np.float32)


@pytest.mark.pybullet
def test_check_env_passes(landing_env, static_motion):
    """The Gymnasium API contract holds, including reset-seed and space membership."""
    env = landing_env(static_motion)
    check_env(env, skip_render_check=True)


@pytest.mark.pybullet
def test_action_space_is_three_dimensional_and_scaled_by_v_max(
    landing_env, static_motion, env_landing_cfg
):
    """The shared action space: ``Box(-1, 1, (3,))``, scaled by ``v_max`` from the config.

    Every method emits exactly this, so a change here changes every arm of every comparison
    at once. ``v_max`` is read from ``configs/env/landing.yaml`` and is never hard-coded.
    """
    env = landing_env(static_motion)
    assert isinstance(env.action_space, Box)
    assert env.action_space.shape == (ACTION_DIM,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)
    assert env.cfg.v_max_m_s == env_landing_cfg.v_max_m_s
    assert env_landing_cfg.v_max_m_s == 1.5  # P2-D2
    straight = env.velocity_setpoint_m_s(np.array([0.0, 0.0, -1.0]))
    assert straight == pytest.approx([0.0, 0.0, -env_landing_cfg.v_max_m_s])


@pytest.mark.pybullet
def test_v_max_is_a_speed_cap_not_a_per_axis_cap(landing_env, static_motion, env_landing_cfg):
    """A full-diagonal action is rescaled to ``v_max``, not left at ``sqrt(3) * v_max``.

    P2-D2. The literal box would let a diagonal descent be 73 % faster than a vertical one,
    which is a property of the box, not of the vehicle, and would make the action space
    anisotropic for every method at once.
    """
    env = landing_env(static_motion)
    diagonal = env.velocity_setpoint_m_s(np.array([1.0, 1.0, 1.0]))
    assert float(np.linalg.norm(diagonal)) == pytest.approx(env_landing_cfg.v_max_m_s)
    assert env_landing_cfg.v_max_is_norm_cap
    # An action inside the unit ball is untouched.
    small = env.velocity_setpoint_m_s(np.array([0.2, 0.0, 0.0]))
    assert small == pytest.approx([0.2 * env_landing_cfg.v_max_m_s, 0.0, 0.0])
    # Out-of-range actions are clipped into the box before scaling.
    assert float(np.linalg.norm(env.velocity_setpoint_m_s(np.array([9.0, 9.0, 9.0])))) == (
        pytest.approx(env_landing_cfg.v_max_m_s)
    )


@pytest.mark.pybullet
def test_observation_matches_its_declared_layout(landing_env, static_motion, env_obs_cfg):
    """The vector, the layout table and the ``Box`` all agree on length and bounds.

    The layout table is what Phase 4 appends its forecast block to, so a silent drift
    between table and vector would renumber every downstream slice.
    """
    env = landing_env(static_motion)
    obs, _ = env.reset(seed=1)
    fields = obs_fields(env_obs_cfg)
    assert sum(field.size for field in fields) == obs.size
    assert observation_space(env_obs_cfg).shape == obs.shape
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    names = [field.name for field in fields]
    assert names[:9] == [
        "attitude",
        "body_rates",
        "velocity",
        "rel_position",
        "rel_velocity",
        "deck_normal",
        "rel_tilt",
        "height",
        "time_fraction",
    ]
    # The two blocks the addendum added, and why: the smoothness reward needs a_{t-1} and
    # the 0.5 s dwell needs a contact flag.
    assert names[9:] == ["last_action", "in_contact"]
    assert np.all(np.isfinite(env.observation_space.low))
    assert np.all(np.isfinite(env.observation_space.high))


@pytest.mark.pybullet
def test_reset_is_deterministic_first_100_obs(landing_env, static_motion):
    """Same seed -> bit-identical observations, as the 1st reset and again as the 3rd.

    Running it as the 3rd reset is the point: ``DSLPIDControl``'s integral state, the noise
    generator and the episode-start draw all survive a reset, and each of them would make
    the second run of the same seed differ while the first run looked perfect.
    """
    env = landing_env(static_motion)
    actions = (
        np.random.default_rng(11).uniform(-1.0, 1.0, size=(100, ACTION_DIM)).astype(np.float32)
    )

    def rollout(seed: int) -> np.ndarray:
        obs, _ = env.reset(seed=seed)
        frames = [obs.copy()]
        for action in actions[1:]:
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(obs.copy())
            if terminated or truncated:
                break
        return np.array(frames)

    first = rollout(42)
    rollout(99)  # a different episode in between
    third = rollout(42)
    assert first.shape == third.shape
    assert np.array_equal(first, third)
    assert first.shape[0] > 10


@pytest.mark.pybullet
def test_timeout_truncates_and_does_not_terminate(landing_env, static_motion, env_landing_cfg):
    """A hovering drone runs out of clock as ``truncated``, with zero terminal reward.

    The two halves matter together. ``terminated=True`` on a time limit tells the value
    function the episode genuinely ended, so bootstrapping stops and hovering looks
    terminal; a terminal penalty attached to the truncation teaches the same lesson from the
    other side. Both are checked.
    """
    env = landing_env(static_motion)
    env.reset(seed=7)
    hover = np.zeros(ACTION_DIM, dtype=np.float32)
    terminated = truncated = False
    steps = 0
    info: dict = {}
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(hover)
        steps += 1
    assert truncated is True
    assert terminated is False
    assert info["outcome"] == "timeout"
    assert info["termination_reason"] == "time_limit"
    assert steps == pytest.approx(
        env_landing_cfg.episode_len_s * env_landing_cfg.ctrl_freq_hz, abs=1
    )
    assert terminal_reward(env._reward_cfg, "timeout") == 0.0  # noqa: SLF001


@pytest.mark.pybullet
def test_yaw_setpoint_is_zero_not_the_current_yaw(landing_env, static_motion, monkeypatch):
    """The yaw target handed to ``DSLPIDControl`` is 0, not the drone's present yaw.

    Upstream's ``ActionType.VEL`` passes ``target_rpy = [0, 0, state[9]]``
    (``BaseRLAviary.py:220``), which re-targets whatever yaw the attitude loop has drifted
    to instead of correcting it -- a pure integrator with no restoring term. We deviate
    (P2-D2) and this spies on the actual call rather than inferring the deviation from an
    outcome.
    """
    env = landing_env(static_motion)
    env.reset(seed=3)
    seen: list[float] = []
    original = env._ctrl.computeControl  # noqa: SLF001

    def spy(*args, **kwargs):
        seen.append(float(np.asarray(kwargs["target_rpy"], dtype=np.float64)[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(env._ctrl, "computeControl", spy)  # noqa: SLF001
    rng = np.random.default_rng(5)
    for _ in range(40):
        _, _, terminated, truncated, _ = env.step(
            rng.uniform(-1.0, 1.0, ACTION_DIM).astype(np.float32)
        )
        if terminated or truncated:
            break
    assert seen
    assert all(target == 0.0 for target in seen)
    # And the yaw the drone actually reached is non-zero, so the target is not trivially
    # equal to the current yaw.
    assert abs(float(env.rpy[0, 2])) > 1e-6


@pytest.mark.pybullet
def test_yaw_stays_small_under_a_scripted_descent(landing_env, static_motion):
    """Under a controller that is trying to land, yaw stays within a few degrees.

    Measured, not assumed: under a *uniformly random* action stream the same drone reaches
    41 deg of yaw before it tumbles past the 60 deg crash threshold, because a random 1.5 m/s
    setpoint drives large roll and pitch and the DSL yaw loop is the weakest of the three.
    That is a property of the random policy, not of the yaw target, so the claim worth
    testing is the one about a policy that is flying.
    """
    env = landing_env(static_motion)
    obs, _ = env.reset(seed=3)
    yaws = []
    for _ in range(200):
        obs, _, terminated, truncated, _ = env.step(_scripted_descent(obs, env))
        yaws.append(abs(float(env.rpy[0, 2])))
        if terminated or truncated:
            break
    assert max(yaws) < np.radians(5.0), f"yaw drifted to {np.degrees(max(yaws)):.2f} deg"


@pytest.mark.pybullet
def test_drop_on_static_pad_registers_exactly_one_touchdown(landing_env, static_motion):
    """A drone released from rest above a static pad touches down once and is classified.

    "Exactly one" is measured as the number of rising edges of deck contact
    (``EpisodeRecord.contact_events``), not inferred from the record latching the first
    event -- otherwise the claim would be true by construction.
    """
    env = landing_env(static_motion)
    env.reset(seed=21)
    idle = np.zeros(ACTION_DIM, dtype=np.float32)
    # Cut the motors: apply a hard-down setpoint so the PID lets it fall rather than hover.
    fall = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    terminated = truncated = False
    info: dict = {}
    for step in range(400):
        _, _, terminated, truncated, info = env.step(fall if step < 60 else idle)
        if terminated or truncated:
            break
    assert terminated or truncated
    assert env.record.contact_events == 1, env.record.as_row()
    assert env.record.contact_record is not None
    assert env.record.analytic_record is not None
    assert info["outcome"] in {"success", "hard_landing", "off_pad", "bounce", "crash"}
    # A fast arrival does penetrate, and the tunnelling flag is exactly the recorded
    # penetration against the frozen threshold -- reported, not asserted away.
    assert env.record.max_penetration_m < 0.0
    assert env.record.tunnelled == (
        env.record.max_penetration_m < -env.success.tunnelling_penetration_m
    )


@pytest.mark.pybullet
@pytest.mark.slow
def test_scripted_descent_static_pad_all_success(landing_env, static_motion, env_success_cfg):
    """100 scripted 0.3 m/s descents onto a static pad, 100 successes.

    With the deck held still this is a pure test of the environment: the criteria, the
    contact geometry, the dwell bookkeeping and the classification. Anything below 100/100
    means one of those is wrong, because at 0.3 m/s onto a motionless pad there is nothing
    left to be hard.

    The analytic-versus-contact disagreement rate is asserted here too, on the only episodes
    where every episode lands.
    """
    env = landing_env(static_motion)
    outcomes: list[str] = []
    disagreements = 0
    gaps: list[float] = []
    for episode in range(100):
        obs, _ = env.reset(seed=1000 + episode)
        info: dict = {}
        for _ in range(400):
            obs, _, terminated, truncated, info = env.step(_scripted_descent(obs, env))
            if terminated or truncated:
                break
        outcomes.append(str(info["outcome"]))
        disagreements += int(info["detectors_disagree"])
        contact, analytic = info["touchdown_contact"], info["touchdown_analytic"]
        if contact is not None and analytic is not None:
            gaps.append(abs(contact.t_episode_s - analytic.t_episode_s))
    assert outcomes.count("success") == 100, {o: outcomes.count(o) for o in set(outcomes)}
    assert disagreements == 0
    assert max(gaps) <= env_success_cfg.detector_disagreement_window_s
    # A 0.3 m/s arrival does not tunnel; the flag discriminates rather than always firing.
    assert not env.record.tunnelled


@pytest.mark.pybullet
@pytest.mark.slow
def test_analytic_and_contact_touchdown_agree_on_a_moving_deck(landing_env, deck_source):
    """The two detectors agree on a real SS5 deck, under a scripted descent.

    The static-pad run removes deck motion; this one puts it back. Gate 2 asks for a
    disagreement rate below 1 %, and a moving deck is where a detector that reads the deck
    state at the wrong instant would show it.
    """
    env = landing_env(deck_source(SS5_SPEC))
    disagreements = 0
    episodes = 30
    landed = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=2000 + episode)
        info: dict = {}
        for _ in range(400):
            obs, _, terminated, truncated, info = env.step(_scripted_descent(obs, env))
            if terminated or truncated:
                break
        disagreements += int(info["detectors_disagree"])
        landed += int(info["touchdown_contact"] is not None)
    assert landed >= episodes // 2, "the scripted descent should reach the deck most of the time"
    assert disagreements / episodes < 0.01 or disagreements == 0, disagreements


@pytest.mark.pybullet
def test_episode_start_offsets_stay_inside_the_committed_record(landing_env, deck_source):
    """Every episode's start offset comes from ``episode_start_window_s``.

    The bridge refuses off-corpus times, so an offset outside the window is not a subtly
    wrong number -- it is a raise mid-episode. The window for a 12.5 s episode is
    ``(4.0, 107.5)`` s model, the 4.0 s being the forecaster's reserved lookback.
    """
    source = deck_source(SS5_SPEC)
    env = landing_env(source)
    lo, hi = source.episode_start_window_s(env.cfg.total_len_s)
    assert (lo, hi) == pytest.approx((4.0, 107.5))
    offsets = []
    for seed in range(30):
        _, info = env.reset(seed=seed)
        offsets.append(float(info["t0_model_s"]))
    assert min(offsets) >= lo
    assert max(offsets) <= hi
    assert np.ptp(offsets) > 10.0, "start offsets should actually vary across seeds"


@pytest.mark.pybullet
def test_precomputed_trajectory_matches_per_sample_bridge_calls_to_round_off(
    deck_source, env_landing_cfg, static_motion
):
    """One batched bridge call per reset agrees with single-sample calls to float64 round-off.

    This is why the deck trajectory is precomputed rather than evaluated inside each of the
    1 920 physics substeps per simulated second: ``synthesize_motion`` is a memoryless sum of
    harmonics in ``t``, so batching changes the cost and not the physics.

    **Measured correction to the approved plan (P2-D8).** The plan's addendum says the two
    are "bit-identical". They are not: 0 of 20 probed samples matched bit for bit, because
    the superposition ends in a matrix product and BLAS picks a different kernel -- hence a
    different summation order over the 299 wave components -- for a (1, 299) operand than for
    a (3001, 299) one. The measured agreement is <= 1e-16 m and <= 1e-16 m/s in absolute
    terms on quantities of order 2 m and 0.4 m/s, i.e. under one ULP. Determinism is
    unaffected because the environment always evaluates the same 3 001-sample batch; the
    claim that had to be weakened is the *equality*, not the *reproducibility*.
    """
    source = deck_source(SS5_SPEC)
    plate = env_landing_cfg.platform
    grid = 10.0 + np.arange(200) * env_landing_cfg.physics_dt_s
    batched = build_trajectory(source, "aft", plate, grid)
    for index in (0, 37, 199):
        single = build_trajectory(source, "aft", plate, grid[index : index + 1])
        assert single.position_m[0] == pytest.approx(batched.position_m[index], abs=1e-15)
        assert single.state.velocity_m_s[0] == pytest.approx(
            batched.state.velocity_m_s[index], abs=1e-15
        )
        assert single.quaternion[0] == pytest.approx(batched.quaternion[index], abs=1e-15)
    assert isinstance(static_motion, StaticDeckMotion)


@pytest.mark.pybullet
def test_plate_is_anchored_at_the_deck_origin_not_at_the_ship_cg(
    landing_env, deck_source, env_landing_cfg, deck_pads, deck_scaling
):
    """The plate oscillates about ``deck_origin_m``, with the pad's lever arm removed.

    ``DeckPointState`` measures a pad from the vessel's **mean-position CG**, so the
    frigate's aft pad carries a standing -1.984 m offset in world x (-0.4 L at lam = 1/25)
    and the s175's carries -2.800 m. Left in, the plate would sit two metres aft of where
    the drone is released -- every scripted descent then falls past the deck and ends
    ``crash``/``below_deck``, which is exactly what happened before this was caught -- and it
    would sit in a *different place* for each hull, turning P1-D3's named ``unseen_vessel``
    lever-arm confound into a world-geometry confound as well.

    Only the constant is removed: the lever-arm-driven *motion* is untouched, which the
    non-zero x excursion below asserts.
    """
    offset = pad_offset_model_m("frigate", deck_pads, deck_scaling.froude_scale(), "aft")
    assert offset[0] == pytest.approx(-1.984)
    env = landing_env(deck_source(SS5_SPEC))
    env.reset(seed=4)
    traj = env._trajectory.full()  # noqa: SLF001  (lazy since P5-D5)
    origin = np.array(env_landing_cfg.platform.deck_origin_m, dtype=np.float64)
    mean = traj.position_m.mean(axis=0)
    assert mean == pytest.approx(origin, abs=0.05)
    # The lever arm still drives real horizontal motion about that anchor.
    assert float(np.ptp(traj.position_m[:, 0])) > 1e-3


@pytest.mark.pybullet
def test_noise_defaults_off_and_is_reproducible(landing_env, static_motion, env_noise_cfg):
    """Noise is off by default, and turning it on does not move the episode.

    The noise generator is a **spawned child** of the episode seed sequence, so enabling it
    leaves the start offset and the initial drone state untouched. If it shared the
    generator, the Phase 7 ablation would compare different episodes and would stop being
    the paired comparison it is reported as.
    """
    assert env_noise_cfg.enabled is False
    clean = landing_env(static_motion)
    _, clean_info = clean.reset(seed=77)

    noisy_cfg = NoiseConfig(
        enabled=True,
        position_sigma_m=0.02,
        velocity_sigma_m_s=0.05,
        latency_ms=33.0,
        hold_freq_hz=30.0,
    )
    noisy = landing_env(static_motion, noise_cfg=noisy_cfg, episode_seed=0)
    obs_a, noisy_info = noisy.reset(seed=77)
    assert noisy_info["t0_model_s"] == clean_info["t0_model_s"]
    # Same seed, same noise draw.
    obs_b, _ = noisy.reset(seed=77)
    assert np.array_equal(obs_a, obs_b)
    # And the noise actually perturbs the relative block relative to the clean env.
    clean.reset(seed=77)
    clean_obs, _ = clean.reset(seed=77)
    assert not np.array_equal(clean_obs[9:15], obs_a[9:15])


@pytest.mark.pybullet
def test_crash_bail_outs_carry_a_specific_termination_reason(landing_env, static_motion):
    """Divergence is a ``crash`` with a named reason, not a seventh outcome class (P2-D5).

    A hard sustained lateral command runs the drone off the plate and out of bounds; the
    class is ``crash`` and the reason says which bail-out fired, so the crash bar stays
    decomposable in Phase 7 without adding a class to the frozen six.
    """
    env = landing_env(static_motion)
    env.reset(seed=13)
    push = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    info: dict = {}
    for _ in range(400):
        _, _, terminated, truncated, info = env.step(push)
        if terminated or truncated:
            break
    assert info["outcome"] == "crash"
    assert info["termination_reason"] in {
        "ground_contact",
        "off_plate_strike",
        "tilt_gt_crash",
        "below_deck",
        "out_of_bounds",
    }


@pytest.mark.pybullet
def test_reset_without_a_seed_advances_deterministically(landing_env, static_motion):
    """Seedless resets give different episodes, and the same sequence from a fresh env.

    Training never passes a seed per episode, so this is the path PPO actually takes; it
    still has to be reproducible from the constructor's ``episode_seed`` alone.
    """
    first = landing_env(static_motion, episode_seed=5)
    starts_a = [float(first.reset()[1]["t0_model_s"]) for _ in range(5)]
    second = landing_env(static_motion, episode_seed=5)
    starts_b = [float(second.reset()[1]["t0_model_s"]) for _ in range(5)]
    assert starts_a == starts_b
    assert len(set(starts_a)) == len(starts_a)
