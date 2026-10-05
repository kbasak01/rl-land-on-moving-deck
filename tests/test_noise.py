"""The perception stand-in perceives the deck (P7-D4): delay, hold, world-frame noise.

What is pinned here
-------------------
* **Unit (no PyBullet).** :class:`~rld.envs.noise.PerceptionNoise` operates on
  :class:`~rld.envs.platform.PlatformSample`: the perceived sample of control step ``k`` is
  the true sample of step ``k - L`` (oldest available during warm-up), held at the refresh
  rate, with world-frame Gaussian noise on pad position and velocity only, drawn 3 position
  then 3 velocity values per step from the given generator. Disabled, it returns its input
  and draws nothing. The latencies 0 / 33.4 / 66.7 ms quantise to 0 / 1 / 2 steps.
* **Noise off is untouched.** Full-episode observations on 2 frozen ``id`` episodes
  (``pid_feedforward``) hash to the digests recorded with the **pre-P7-D4** code (the
  Phase 2 stand-in, commit ``3a25620``), byte for byte.
* **L = 0, sigma = 0 enabled is clean**, byte for byte over the same full episodes.
* **L = 2, sigma = 0 is a consistently stale deck.** At every step ``k >= 2``, own velocity
  + relative velocity (world) is the true pad velocity of step ``k - 2``, and the whole
  observation -- clearance, normal and tilt included -- is the one built from the
  ``k - 2`` deck sample and the drone's current state.
* **Noise lands in the world frame and moves the clearance**: a twin stand-in on the same
  spawned stream reproduces the environment's observation byte for byte.
* **Pairing.** Turning the stand-in on does not move t0 or the initial state.

Units: metres, metres per second and seconds model scale; latency milliseconds model scale.
"""

import dataclasses
import hashlib
from typing import Any

import numpy as np
import pytest
from dmf.typedefs import FloatArray

from rld.control.obs_view import obs_layout, view
from rld.envs.config import NoiseConfig
from rld.envs.landing_env import EPISODE_SALT, DeckLandingAviary
from rld.envs.noise import PerceptionNoise, copy_sample
from rld.envs.observation import build_observation
from rld.envs.platform import PlatformSample
from rld.eval.envs import EvalConfigs, load_eval_configs, make_env, motion_for, with_noise
from rld.eval.episodes import EPISODES_DIR, ListedEpisode, read_list
from rld.eval.runner import controller_spec

CTRL_HZ = 30.0

#: SHA-256 of the full-episode float32 observation stack (reset observation first, then one
#: per step), ``pid_feedforward``, noise **off**, recorded with the pre-P7-D4 code at
#: ``3a25620`` before ``rld.envs.noise`` or ``_computeObs`` changed. Keyed by (ss, index) of
#: the first listed ``id`` episode of that sea state; value ``(n_observations, sha256)``.
PRE_P7D4_NOISE_OFF: dict[tuple[str, int], tuple[int, str]] = {
    ("SS3", 0): (172, "410e65570188f57be079d01ec9201c650788ad595f8b1ce5f3ec2d382296a4ac"),
    ("SS6", 0): (153, "d8be2350d331127036b13e2a4d00aa4208f16f682c5477e4e9d320423223a331"),
}


# --------------------------------------------------------------------------- unit (fast)


def _cfg(
    *,
    sigma_p: float = 0.0,
    sigma_v: float = 0.0,
    latency_ms: float = 0.0,
    hold_hz: float = 30.0,
    enabled: bool = True,
) -> NoiseConfig:
    return NoiseConfig(
        enabled=enabled,
        position_sigma_m=sigma_p,
        velocity_sigma_m_s=sigma_v,
        latency_ms=latency_ms,
        hold_freq_hz=hold_hz,
    )


def _sample(k: int) -> PlatformSample:
    """A synthetic deck sample whose every field encodes its step ``k``."""
    tilt = 0.01 * k
    normal = np.array([-np.sin(tilt), 0.0, np.cos(tilt)])
    return PlatformSample(
        t_model_s=0.1 * k,
        position_m=np.array([1.0 + k, 2.0 + k, 3.0 + k]),
        quaternion=np.array([0.0, np.sin(tilt / 2), 0.0, np.cos(tilt / 2)]),
        euler_xyz_rad=np.array([0.0, tilt, 0.0]),
        velocity_m_s=np.array([0.1 * k, -0.2 * k, 0.3 * k]),
        angular_velocity_rad_s=np.array([0.0, 0.01 * k, 0.0]),
        acceleration_m_s2=np.array([0.0, 0.0, -0.5 * k]),
        normal=normal,
        tilt_deg=float(np.degrees(tilt)),
    )


def _same(a: PlatformSample, b: PlatformSample) -> bool:
    """Every field equal bit for bit."""
    for f in dataclasses.fields(PlatformSample):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            if np.asarray(x).tobytes() != np.asarray(y).tobytes():
                return False
        elif x != y:
            return False
    return True


@pytest.mark.parametrize(("latency_ms", "steps"), [(0.0, 0), (33.4, 1), (66.7, 2), (33.0, 0)])
def test_latency_quantises_down_to_whole_control_steps(latency_ms: float, steps: int) -> None:
    """P7-D1 §4: 0 / 33.4 / 66.7 ms are 0 / 1 / 2 steps at 30 Hz; a literal 33 ms is 0."""
    noise = PerceptionNoise(_cfg(latency_ms=latency_ms), np.random.default_rng(0), CTRL_HZ)
    assert noise.latency_steps == steps


def test_disabled_returns_its_input_and_draws_nothing() -> None:
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state
    noise = PerceptionNoise(
        _cfg(sigma_p=0.04, sigma_v=0.2, latency_ms=66.7, enabled=False), rng, CTRL_HZ
    )
    for k in range(5):
        deck = _sample(k)
        assert noise.perceive(deck) is deck
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("latency_ms", [0.0, 33.4, 66.7])
def test_delay_returns_the_sample_of_step_k_minus_l_with_warm_up(latency_ms: float) -> None:
    noise = PerceptionNoise(_cfg(latency_ms=latency_ms), np.random.default_rng(0), CTRL_HZ)
    lag = noise.latency_steps
    for k in range(8):
        deck = _sample(k)
        out = noise.perceive(deck)
        assert _same(out, _sample(max(0, k - lag)))
        assert _same(deck, _sample(k))  # input never modified
    noise.reset()
    assert _same(noise.perceive(_sample(10)), _sample(10))  # reset empties the delay line


def test_hold_refreshes_every_hold_interval() -> None:
    noise = PerceptionNoise(_cfg(hold_hz=15.0), np.random.default_rng(0), CTRL_HZ)
    assert noise.hold_steps == 2
    got = [noise.perceive(_sample(k)).t_model_s for k in range(6)]
    assert got == pytest.approx([0.0, 0.0, 0.2, 0.2, 0.4, 0.4])


def test_noise_is_world_frame_on_position_and_velocity_only_in_draw_order() -> None:
    """3 position draws then 3 velocity draws per step; every other field the delayed truth."""
    sigma_p, sigma_v = 0.02, 0.1
    noise = PerceptionNoise(
        _cfg(sigma_p=sigma_p, sigma_v=sigma_v, latency_ms=33.4), np.random.default_rng(11), CTRL_HZ
    )
    twin = np.random.default_rng(11)
    for k in range(6):
        out = noise.perceive(_sample(k))
        truth = _sample(max(0, k - 1))
        dp = twin.normal(0.0, sigma_p, size=3)
        dv = twin.normal(0.0, sigma_v, size=3)
        assert out.position_m.tobytes() == (truth.position_m + dp).tobytes()
        assert out.velocity_m_s.tobytes() == (truth.velocity_m_s + dv).tobytes()
        unperturbed = dataclasses.replace(
            out, position_m=truth.position_m, velocity_m_s=truth.velocity_m_s
        )
        assert _same(unperturbed, truth)  # normal, quaternion, tilt, t: no noise


def test_position_only_noise_draws_only_position() -> None:
    noise = PerceptionNoise(_cfg(sigma_p=0.01), np.random.default_rng(3), CTRL_HZ)
    twin = np.random.default_rng(3)
    for k in range(4):
        out = noise.perceive(_sample(k))
        expected = _sample(k).position_m + twin.normal(0.0, 0.01, size=3)
        assert out.position_m.tobytes() == expected.tobytes()
        assert out.velocity_m_s.tobytes() == _sample(k).velocity_m_s.tobytes()


def test_perceived_sample_never_aliases_the_input_or_the_held_sample() -> None:
    noise = PerceptionNoise(_cfg(latency_ms=33.4, hold_hz=15.0), np.random.default_rng(0), CTRL_HZ)
    deck = _sample(0)
    out = noise.perceive(deck)
    for f in dataclasses.fields(PlatformSample):
        x = getattr(out, f.name)
        if isinstance(x, np.ndarray):
            assert not np.shares_memory(x, getattr(deck, f.name))
            x[...] = np.nan  # mutating what the caller got cannot reach the stand-in's state
    assert _same(noise.perceive(_sample(1)), _sample(0))
    assert _same(copy_sample(deck), deck)


# --------------------------------------------------------------------------- env (PyBullet)


@pytest.fixture(scope="module")
def cfgs() -> EvalConfigs:
    return load_eval_configs()


@pytest.fixture(scope="module")
def episodes() -> list[ListedEpisode]:
    listed = read_list(EPISODES_DIR / "id.parquet")
    return [next(e for e in listed if e.ss == ss) for ss in ("SS3", "SS6")]


def _env(cfgs: EvalConfigs, listed: ListedEpisode) -> DeckLandingAviary:
    motion = motion_for(
        cfgs,
        listed.vessel,
        listed.ss,
        listed.heading_deg,
        listed.speed_kn,
        listed.realization_seed,
        kind="jonswap",
        episode_seed=listed.episode_seed,
    )
    return make_env(cfgs, motion, listed.vessel, listed.pad, listed.episode_seed)


def _rebuild(env: DeckLandingAviary, deck: PlatformSample) -> FloatArray:
    """The observation ``_computeObs`` builds from ``deck`` and the env's current true state."""
    rpy = np.asarray(env.rpy[0], dtype=np.float64)
    return build_observation(
        cfg=env._obs_cfg,  # noqa: SLF001
        drone_position_m=np.asarray(env.pos[0], dtype=np.float64),
        drone_rpy_rad=rpy,
        drone_velocity_m_s=np.asarray(env.vel[0], dtype=np.float64),
        drone_body_rates_rad_s=np.asarray(env.ang_v[0], dtype=np.float64),
        drone_rotation=env._drone_rotation(np.asarray(env.quat[0], dtype=np.float64)),  # noqa: SLF001
        deck=deck,
        geometry=env._geometry,  # noqa: SLF001
        time_fraction=float(
            np.clip(env._substep * env.cfg.physics_dt_s / env.cfg.episode_len_s, 0.0, 1.0)  # noqa: SLF001
        ),
        last_action=env._last_action,  # noqa: SLF001
        in_contact=env._in_contact,  # noqa: SLF001
    )


def _fly(
    cfgs: EvalConfigs, listed: ListedEpisode, on_step: Any = None
) -> tuple[FloatArray, dict[str, Any]]:
    """Fly ``pid_feedforward`` on one listed episode, as the runner does.

    Args:
        cfgs: The configs to build the env from.
        listed: The episode.
        on_step: Optional ``(env, k, obs) -> None`` called after the reset (``k = 0``) and
            after every step (``k`` = control steps taken).

    Returns:
        ``(observations, last info)``: the float32 stack, reset observation first.
    """
    env = _env(cfgs, listed)
    try:
        policy = controller_spec("pid_feedforward").build(cfgs)
        obs, info = env.reset(seed=listed.episode_seed)
        policy.reset(listed.episode_seed, None)
        if on_step is not None:
            on_step(env, 0, obs)
        rows = [obs]
        max_steps = int(round(cfgs.landing.total_len_s * cfgs.landing.ctrl_freq_hz)) + 1
        for k in range(1, max_steps + 1):
            obs, _, terminated, truncated, info = env.step(
                np.asarray(policy.act(obs), dtype=np.float64)
            )
            if on_step is not None:
                on_step(env, k, obs)
            rows.append(obs)
            if terminated or truncated:
                break
        return np.stack(rows), dict(info)
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.pybullet
def test_noise_off_observations_match_the_pre_p7d4_recording(
    cfgs: EvalConfigs, episodes: list[ListedEpisode]
) -> None:
    assert cfgs.noise.enabled is False
    for listed in episodes:
        stack, _ = _fly(cfgs, listed)
        n, digest = PRE_P7D4_NOISE_OFF[(listed.ss, listed.index)]
        assert stack.dtype == np.float32 and stack.shape == (n, 25)
        assert hashlib.sha256(stack.tobytes()).hexdigest() == digest


@pytest.mark.slow
@pytest.mark.pybullet
def test_enabled_at_zero_latency_and_zero_sigma_is_bit_identical_to_clean(
    cfgs: EvalConfigs, episodes: list[ListedEpisode]
) -> None:
    zero = with_noise(cfgs, position_sigma_m=0.0, velocity_sigma_m_s=0.0, latency_ms=0.0)
    assert zero.noise.enabled
    for listed in episodes:
        clean, _ = _fly(cfgs, listed)
        noisy, _ = _fly(zero, listed)
        assert noisy.tobytes() == clean.tobytes()


#: The deck-derived blocks P7-D4 §4 names; each must come from the perceived sample.
DECK_BLOCKS = ("rel_position", "rel_velocity", "deck_normal", "rel_tilt", "height")


@pytest.mark.slow
@pytest.mark.pybullet
@pytest.mark.parametrize("which", [0, 1])
def test_two_step_latency_is_a_consistently_stale_deck(
    cfgs: EvalConfigs, episodes: list[ListedEpisode], which: int
) -> None:
    """At k >= 2 every deck-derived entry is the k - 2 deck with the current drone state."""
    late = with_noise(cfgs, position_sigma_m=0.0, velocity_sigma_m_s=0.0, latency_ms=66.7)
    layout = obs_layout(cfgs.observation)
    decks: list[PlatformSample] = []
    fresh_differs = [0]

    def check(env: DeckLandingAviary, k: int, obs: FloatArray) -> None:
        decks.append(copy_sample(env._deck_sample()))  # noqa: SLF001  (true deck, step k)
        assert len(decks) == k + 1
        expected = _rebuild(env, decks[max(0, k - 2)])  # warm-up: the oldest available
        # The whole vector, byte for byte: clearance, normal and tilt from the k - 2 deck and
        # the drone's current state, and the own-state blocks clean and current.
        assert obs.tobytes() == expected.tobytes()
        for name in DECK_BLOCKS:
            assert obs[layout.slices[name]].tobytes() == expected[layout.slices[name]].tobytes()
        if k >= 2:
            v = view(obs, layout)
            np.testing.assert_allclose(
                v.velocity_m_s + v.rel_velocity_m_s, decks[k - 2].velocity_m_s, rtol=0, atol=2e-6
            )
            if obs.tobytes() != _rebuild(env, decks[k]).tobytes():
                fresh_differs[0] += 1

    stack, _ = _fly(late, episodes[which], check)
    assert len(decks) == len(stack) > 30
    # The delay acted: the fresh-deck observation differs on (nearly) every step.
    assert fresh_differs[0] >= len(stack) - 3


@pytest.mark.slow
@pytest.mark.pybullet
@pytest.mark.parametrize("which", [0, 1])
def test_noise_is_world_frame_and_moves_the_clearance(
    cfgs: EvalConfigs, episodes: list[ListedEpisode], which: int
) -> None:
    """A twin stand-in on the env's own spawned stream reproduces every observation."""
    listed = episodes[which]
    noisy = with_noise(cfgs, position_sigma_m=0.02, velocity_sigma_m_s=0.1, latency_ms=33.4)
    height = obs_layout(cfgs.observation).slices["height"]
    seq = np.random.SeedSequence([int(listed.episode_seed), EPISODE_SALT])
    twin = PerceptionNoise(noisy.noise, np.random.default_rng(seq.spawn(3)[2]), CTRL_HZ)
    decks: list[PlatformSample] = []
    moved = [0]

    def check(env: DeckLandingAviary, k: int, obs: FloatArray) -> None:
        deck = env._deck_sample()  # noqa: SLF001  (true deck, step k)
        decks.append(copy_sample(deck))
        perceived = twin.perceive(deck)
        # Byte for byte: the env adds the same world-frame draws to the k - 1 deck.
        assert obs.tobytes() == _rebuild(env, perceived).tobytes()
        stale = decks[max(0, k - 1)]
        assert perceived.normal.tobytes() == stale.normal.tobytes()  # no attitude noise
        # The clearance follows the noisy pad position.
        if _rebuild(env, stale)[height] != obs[height]:
            moved[0] += 1

    stack, _ = _fly(noisy, listed, check)
    assert moved[0] >= len(stack) - 3


@pytest.mark.slow
@pytest.mark.pybullet
def test_noise_does_not_shift_t0_or_the_initial_state(
    cfgs: EvalConfigs, episodes: list[ListedEpisode]
) -> None:
    loud = with_noise(cfgs, position_sigma_m=0.04, velocity_sigma_m_s=0.2, latency_ms=66.7)
    for listed in episodes:
        starts: list[tuple[Any, ...]] = []
        for c in (cfgs, loud):
            env = _env(c, listed)
            try:
                obs, _ = env.reset(seed=listed.episode_seed)
                starts.append(
                    (
                        float(env.record.t0_model_s),
                        np.asarray(env.pos[0]).tobytes(),
                        np.asarray(env.quat[0]).tobytes(),
                        np.asarray(env.vel[0]).tobytes(),
                        env._init_xyzs.tobytes(),  # noqa: SLF001
                        obs[:9].tobytes(),  # own attitude, rates and velocity: clean
                    )
                )
            finally:
                env.close()
        assert starts[0] == starts[1]
        assert starts[0][0] == listed.t0_model_s
        init = np.frombuffer(starts[0][4]).reshape(-1)
        assert tuple(init) == (listed.init_x_m, listed.init_y_m, listed.init_z_m)
