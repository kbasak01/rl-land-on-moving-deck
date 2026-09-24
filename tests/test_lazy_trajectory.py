"""P5-D5: the reset speed-up changes no bit of any episode.

Two changes make ``DeckLandingAviary.reset`` fast, and each has to be proven exact rather than
"close" because every committed result was produced by the old code:

1. **Lazy deck trajectory.** The environment no longer synthesises all 3 001 physics samples
   at reset; :class:`rld.envs.platform.LazyDeckTrajectory` synthesises chunks as the episode
   reaches them. The trap is BLAS: the superposition's reduction over the 299 wave
   components is a matrix product whose kernel -- hence summation order -- OpenBLAS picks
   from the operand shape, so a naive slice of the grid differs from the full grid in the
   last bits. The tests below compare with ``np.array_equal``, never ``allclose``, over the
   **whole** grid, for both vessels, SS3-SS6, the aft and CG pads, several headings, start
   offsets up to the top of the legal window, chunk layouts that force one-row chunks, and
   the sinusoid and static sources.
2. **No visual mesh in DIRECT mode.** The drone URDF is loaded with
   ``URDF_IGNORE_VISUAL_SHAPES``; the dynamics world must be the one upstream builds.

Then, end to end: an episode flown on the new environment equals the same episode flown on
the pre-P5-D5 configuration (eager full-grid trajectory **and** visual shapes), observation
for observation, including a touchdown on a moving deck.

Units: metres, metres per second, seconds model scale unless stated; the channel arrays of
the bridge are full scale.
"""

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import fields
from typing import Any

import numpy as np
import pybullet as pyb
import pytest
from dmf.sim.generate import RealizationSpec
from gym_pybullet_drones.envs.BaseAviary import BaseAviary

import rld.envs.landing_env as landing_env_module
from rld.deck.bridge import (
    TIME_CHUNK,
    JonswapDeckMotion,
    harmonic_sum,
    harmonic_sum_rows,
    load_vessel_cached,
)
from rld.deck.kinematics import DeckPointState
from rld.deck.sinusoid import SinusoidDeckMotion, sinusoid_params
from rld.envs.config import LandingConfig
from rld.envs.landing_env import ACTION_DIM, DeckLandingAviary
from rld.envs.observation import obs_fields
from rld.envs.platform import (
    DECK_CHUNK_SAMPLES,
    DeckTrajectory,
    LazyDeckTrajectory,
    StaticDeckMotion,
    build_trajectory,
    pad_offset_model_m,
)

#: SHA-256 of upstream ``BaseAviary._housekeeping``'s source at the pinned gym-pybullet-drones
#: commit 7ebad1e. ``DeckLandingAviary._housekeeping`` mirrors it line for line in DIRECT
#: mode; if upstream changes, this fails and the mirror must be re-reviewed.
UPSTREAM_HOUSEKEEPING_SHA256 = "82032a879f0b3f647ee0b3588f219425a732acf02431293d6a3c018785f9692a"

#: Realizations the bit-identity sweep runs on: both vessels, SS3-SS6, head, beam, following
#: and the encounter-non-monotonic 45 deg quartering cell, with and without forward speed.
SWEEP_SPECS: tuple[RealizationSpec, ...] = (
    RealizationSpec(sea_state="SS3", heading_deg=180.0, speed_kn=12.0, vessel="frigate", seed=0),
    RealizationSpec(sea_state="SS4", heading_deg=45.0, speed_kn=6.0, vessel="s175", seed=3),
    RealizationSpec(sea_state="SS5", heading_deg=90.0, speed_kn=0.0, vessel="frigate", seed=7),
    RealizationSpec(sea_state="SS6", heading_deg=135.0, speed_kn=12.0, vessel="s175", seed=1),
    RealizationSpec(sea_state="SS6", heading_deg=180.0, speed_kn=12.0, vessel="frigate", seed=2),
    RealizationSpec(sea_state="SS3", heading_deg=0.0, speed_kn=6.0, vessel="s175", seed=5),
)

#: First-chunk sizes: the default, one sample (every early chunk tiny, including one-row
#: chunks), and n - 1 (a one-row tail chunk). One-row products take a different BLAS path.
CHUNK_LAYOUTS: tuple[int | None, ...] = (None, 1, 3000)

#: The JONSWAP cell of the end-to-end episode comparison (P1-D2's worst frigate aft cell).
EPISODE_SPEC = RealizationSpec(
    sea_state="SS5", heading_deg=180.0, speed_kn=12.0, vessel="frigate", seed=0
)


def _assert_trajectories_equal(got: DeckTrajectory, want: DeckTrajectory) -> None:
    """Assert two trajectories are equal in every array, bit for bit.

    Args:
        got: The trajectory under test.
        want: The eager full-grid reference.
    """
    assert got.pad == want.pad
    assert np.array_equal(got.t_model_s, want.t_model_s)
    assert np.array_equal(got.position_m, want.position_m)
    assert np.array_equal(got.quaternion, want.quaternion)
    for field in fields(DeckPointState):
        a, b = getattr(got.state, field.name), getattr(want.state, field.name)
        assert a.shape == b.shape, field.name
        assert np.array_equal(a, b), field.name


def _grid(cfg: LandingConfig, t0_s: float) -> np.ndarray:
    """Return the environment's physics grid for a start offset, exactly as it builds it.

    Args:
        cfg: The landing config.
        t0_s: Start offset, seconds model scale.

    Returns:
        ``t0 + arange(n_physics_samples) * physics_dt``, seconds model scale.
    """
    return np.asarray(t0_s + np.arange(cfg.n_physics_samples) * cfg.physics_dt_s, dtype=np.float64)


# ------------------------------------------------------------------------ the bridge layer


def test_harmonic_sum_rows_equals_the_full_grid_rows() -> None:
    """Every row range of ``harmonic_sum_rows`` is the full call's rows, bit for bit.

    Includes one-row ranges, ranges on both sides of OpenBLAS's small-matrix switch
    (``m * 2 * 299 <= 1e6``, i.e. 1 672 rows), and a grid longer than ``TIME_CHUNK`` so a
    range can straddle two superposition blocks.
    """
    rng = np.random.default_rng(20260923)
    n_components = 299
    phasors = rng.normal(size=(2, n_components)) + 1j * rng.normal(size=(2, n_components))
    w_e = rng.uniform(-0.5, 3.0, n_components)
    for n in (3001, TIME_CHUNK + 777):
        t = 120.0 + 5.0 * rng.uniform(0.0, 100.0) + np.arange(n) / 48.0
        full = harmonic_sum(phasors, w_e, t)
        ranges = [(0, 1), (0, 2), (n - 1, n), (5, 245), (240, 480), (0, 1672), (0, n)]
        ranges += [
            (int(a), int(a) + int(k))
            for a, k in zip(rng.integers(0, n - 600, 8), rng.integers(1, 600, 8), strict=True)
        ]
        if n > TIME_CHUNK:
            ranges.append((TIME_CHUNK - 100, TIME_CHUNK + 100))
        for start, stop in ranges:
            rows = harmonic_sum_rows(phasors, w_e, t, start, stop)
            assert np.array_equal(rows, full[start:stop]), (n, start, stop)


def test_harmonic_sum_rows_rejects_a_bad_range() -> None:
    """An out-of-grid or inverted row range raises instead of returning a short array."""
    phasors = np.ones((2, 3), dtype=np.complex128)
    w_e = np.ones(3)
    t = np.arange(10.0)
    with pytest.raises(ValueError, match="outside a grid"):
        harmonic_sum_rows(phasors, w_e, t, 5, 11)
    with pytest.raises(ValueError, match="outside a grid"):
        harmonic_sum_rows(phasors, w_e, t, 6, 5)


def test_deck_point_rows_validates_like_deck_point(deck_source, env_landing_cfg) -> None:
    """Row ranges are checked, and a grid leaving the record fails on the first chunk.

    The eager path rejects an out-of-record grid at reset; the lazy path must not defer that
    to whichever chunk happens to reach the bad time mid-episode.
    """
    source: JonswapDeckMotion = deck_source(SWEEP_SPECS[0])
    lo, hi = source.episode_start_window_s(env_landing_cfg.total_len_s)
    grid = _grid(env_landing_cfg, hi)
    with pytest.raises(ValueError, match="outside a grid"):
        source.deck_point_rows(grid, "aft", 10, 10)
    with pytest.raises(ValueError, match="outside a grid"):
        source.deck_point_rows(grid, "aft", 0, grid.size + 1)
    beyond = grid + 1.0  # the last second of this grid leaves the committed record
    with pytest.raises(ValueError, match="committed record"):
        source.deck_point(beyond, "aft")
    with pytest.raises(ValueError, match="committed record"):
        source.deck_point_rows(beyond, "aft", 0, 240)
    with pytest.raises(ValueError, match="unknown|pad"):
        source.deck_point_rows(grid, "bridge_wing", 0, 240)
    assert lo < hi


# ------------------------------------------------------------ the lazy trajectory, offline


@pytest.mark.parametrize(
    "spec", SWEEP_SPECS, ids=lambda s: f"{s.vessel}-{s.sea_state}-{s.heading_deg:g}"
)
@pytest.mark.parametrize("pad", ["aft", "cg"])
def test_lazy_trajectory_is_bit_identical_to_the_full_grid(
    spec, pad, deck_source, deck_pads, deck_scaling, env_landing_cfg
) -> None:
    """The lazy trajectory equals ``build_trajectory`` on the same grid in every array.

    Three chunk layouts per case, and start offsets at the bottom, the top and a random
    point of the legal window. ``sample(i)`` is also read in a scrambled order on a fresh
    object, which is how a read-ahead consumer (a controller looking at a future sample)
    meets it.
    """
    source = deck_source(spec)
    plate = env_landing_cfg.platform
    offset = pad_offset_model_m(spec.vessel, deck_pads, deck_scaling.froude_scale(), pad)
    lo, hi = source.episode_start_window_s(env_landing_cfg.total_len_s)
    rng = np.random.default_rng(hash((spec.vessel, spec.sea_state, spec.seed, pad)) % 2**32)
    for t0 in (lo, hi, float(rng.uniform(lo, hi))):
        grid = _grid(env_landing_cfg, t0)
        eager = build_trajectory(source, pad, plate, grid, offset)
        for first in CHUNK_LAYOUTS:
            kwargs: dict[str, Any] = {} if first is None else {"chunk_samples": first}
            lazy = LazyDeckTrajectory(source, pad, plate, grid, offset, **kwargs)
            assert lazy.lazy
            _assert_trajectories_equal(lazy.full(), eager)
        fresh = LazyDeckTrajectory(source, pad, plate, grid, offset)
        for index in rng.permutation(len(eager))[:64].tolist() + [len(eager) - 1, 0]:
            got, want = fresh.sample(index), eager.sample(index)
            assert got.t_model_s == want.t_model_s
            for name in (
                "position_m",
                "quaternion",
                "euler_xyz_rad",
                "velocity_m_s",
                "angular_velocity_rad_s",
                "acceleration_m_s2",
                "normal",
            ):
                assert np.array_equal(getattr(got, name), getattr(want, name)), (index, name)
            assert got.tilt_deg == want.tilt_deg


@pytest.mark.parametrize("pad", ["aft", "cg"])
def test_lazy_trajectory_sinusoid_and_static_are_bit_identical(
    pad, sim_cfg, deck_scaling, deck_pads, deck_motion_cfg, env_landing_cfg
) -> None:
    """The sinusoid (H4 arm) and the static test double go lazy too, exactly."""
    spec = RealizationSpec(
        sea_state="SS6", heading_deg=180.0, speed_kn=12.0, vessel="frigate", seed=0
    )
    scale = deck_scaling.froude_scale()
    sinusoid = SinusoidDeckMotion(
        sinusoid_params(spec, sim_cfg, scale, deck_pads, episode_seed=3),
        sim_cfg,
        scale,
        deck_pads,
        length_full_m=load_vessel_cached(spec.vessel).length_m,
        lookback_full_s=deck_motion_cfg.forecast_lookback_full_s,
    )
    sources: list[tuple[Any, tuple[float, float, float]]] = [
        (sinusoid, pad_offset_model_m("frigate", deck_pads, scale, pad)),
        (StaticDeckMotion(), (0.0, 0.0, 0.0)),
    ]
    plate = env_landing_cfg.platform
    for source, offset in sources:
        lo, hi = source.episode_start_window_s(env_landing_cfg.total_len_s)
        for t0 in (lo, hi, 0.5 * (lo + hi) + 0.123):
            grid = _grid(env_landing_cfg, t0)
            eager = build_trajectory(source, pad, plate, grid, offset)
            for first in CHUNK_LAYOUTS:
                kwargs: dict[str, Any] = {} if first is None else {"chunk_samples": first}
                lazy = LazyDeckTrajectory(source, pad, plate, grid, offset, **kwargs)
                assert lazy.lazy
                _assert_trajectories_equal(lazy.full(), eager)


class _OpaqueSource:
    """A deck source the lazy trajectory knows nothing about (delegates to the static one)."""

    def __init__(self) -> None:
        self._inner = StaticDeckMotion()
        self.calls: list[int] = []

    def deck_point(self, t_model_s: np.ndarray, pad: str) -> Any:
        self.calls.append(int(np.asarray(t_model_s).size))
        return self._inner.deck_point(t_model_s, pad)


def test_an_unknown_source_is_evaluated_eagerly_on_the_whole_grid(env_landing_cfg) -> None:
    """No row-invariance guarantee, no chunking: one full-grid call, as before P5-D5."""
    source = _OpaqueSource()
    grid = _grid(env_landing_cfg, 1.0)
    lazy = LazyDeckTrajectory(source, "aft", env_landing_cfg.platform, grid)  # type: ignore[arg-type]
    assert not lazy.lazy
    lazy.sample(0)
    lazy.sample(len(grid) - 1)
    assert source.calls == [len(grid)]


def test_chunk_schedule_doubles_and_covers_the_grid(env_landing_cfg) -> None:
    """Chunks are [0, C), [C, 2C), [2C, 4C), ... ending at the grid's end, and read-ahead works."""
    grid = _grid(env_landing_cfg, 1.0)
    lazy = LazyDeckTrajectory(StaticDeckMotion(), "aft", env_landing_cfg.platform, grid)
    assert lazy.chunks_evaluated == 0
    lazy.sample(0)
    assert lazy.chunks_evaluated == 1
    lazy.sample(DECK_CHUNK_SAMPLES - 1)
    assert lazy.chunks_evaluated == 1
    lazy.sample(DECK_CHUNK_SAMPLES)
    assert lazy.chunks_evaluated == 2
    lazy.sample(len(grid) - 1)  # read far ahead: evaluates only the last chunk
    assert lazy.chunks_evaluated == 3
    bounds = [lazy._bounds(k) for k in range(lazy._chunk_index(len(grid) - 1) + 1)]  # noqa: SLF001
    assert bounds[0] == (0, DECK_CHUNK_SAMPLES)
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False))
    assert bounds[-1][1] == len(grid)
    assert [b - a for a, b in bounds[1:-1]] == [
        DECK_CHUNK_SAMPLES << k for k in range(len(bounds) - 2)
    ]
    with pytest.raises(IndexError):
        lazy.sample(len(grid))
    with pytest.raises(IndexError):
        lazy.sample(-1)


# ----------------------------------------------------------------- the environment, end to end


def _descent_action(obs: np.ndarray, env: DeckLandingAviary) -> np.ndarray:
    """Null the lateral error and descend at 0.3 m/s: enough to touch a moving deck down.

    Args:
        obs: The current observation.
        env: The environment, for the observation layout and ``v_max``.

    Returns:
        A ``(3,)`` float32 action in ``[-1, 1]^3``.
    """
    start = 0
    for field in obs_fields(env._obs_cfg):  # noqa: SLF001
        if field.name == "rel_position":
            break
        start += field.size
    rel = np.asarray(obs[start : start + 3], dtype=np.float64)
    command = np.array([*np.clip(1.5 * rel[:2], -0.4, 0.4), -0.3]) / env.cfg.v_max_m_s
    return np.clip(command, -1.0, 1.0).astype(np.float32)


def _fly(
    env: DeckLandingAviary, seed: int, policy: Callable[[np.ndarray], np.ndarray]
) -> dict[str, Any]:
    """Fly one episode and return everything it produced.

    Args:
        env: The environment.
        seed: Episode seed.
        policy: Maps an observation to an action.

    Returns:
        Observations, rewards, flags, the per-step deck samples and the episode row.
    """
    obs, info = env.reset(seed=seed)
    frames, rewards, flags, decks = [obs.copy()], [], [], []
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        deck = env._deck_sample(min(env._substep, len(env._trajectory) - 1))  # noqa: SLF001
        frames.append(obs.copy())
        rewards.append(reward)
        flags.append((terminated, truncated))
        decks.append(np.concatenate([deck.position_m, deck.velocity_m_s, deck.quaternion]))
        if terminated or truncated:
            break
    return {
        "obs": np.array(frames),
        "rewards": np.array(rewards),
        "flags": flags,
        "decks": np.array(decks),
        "row": env.record.as_row(),
        "outcome": info["outcome"],
    }


def _jobs(
    env: DeckLandingAviary,
) -> list[tuple[DeckLandingAviary, int, Callable[[np.ndarray], np.ndarray]]]:
    """Return the episodes both environments fly: three descents, two random policies.

    Args:
        env: The environment the policies act in.

    Returns:
        ``(env, seed, policy)`` triples. The random policies draw from their own seeded
        generators, so the two environments see the same action streams.
    """

    def descent(obs: np.ndarray) -> np.ndarray:
        return _descent_action(obs, env)

    def gaussian(seed: int) -> Callable[[np.ndarray], np.ndarray]:
        rng = np.random.default_rng(seed)
        return lambda _obs: np.clip(rng.normal(0.0, 0.37, ACTION_DIM), -1.0, 1.0).astype(np.float32)

    return [(env, 1, descent), (env, 2, descent), (env, 3, descent)] + [
        (env, seed, gaussian(seed)) for seed in (4, 5)
    ]


@pytest.mark.pybullet
def test_env_synthesises_the_deck_lazily(landing_env, deck_source) -> None:
    """After reset only the first chunk exists; later chunks appear as the episode reaches them."""
    env = landing_env(deck_source(EPISODE_SPEC))
    env.reset(seed=3)
    traj = env._trajectory  # noqa: SLF001
    assert isinstance(traj, LazyDeckTrajectory)
    assert traj.lazy
    assert traj.chunks_evaluated == 1
    hover = np.zeros(ACTION_DIM, dtype=np.float32)
    steps = DECK_CHUNK_SAMPLES // env.PYB_STEPS_PER_CTRL  # 30 control steps = 240 substeps
    for _ in range(steps - 1):
        env.step(hover)
    assert traj.chunks_evaluated == 1
    env.step(hover)  # the observation after this step reads sample 240
    assert traj.chunks_evaluated == 2


@pytest.mark.pybullet
@pytest.mark.slow
@pytest.mark.parametrize("pad", ["aft", "cg"])
def test_episodes_equal_the_pre_p5_d5_environment(
    pad, landing_env, deck_source, monkeypatch
) -> None:
    """The new environment flies the same episodes as the old one, bit for bit.

    "Old" is rebuilt exactly: the eager full-grid ``build_trajectory`` in place of the lazy
    trajectory, **and** the drone URDF loaded with its visual mesh (``visual_shapes=True``,
    upstream's own ``_housekeeping``). Each seed flies a scripted 0.3 m/s descent onto a
    moving SS5 deck (touchdown, dwell, contact detectors) and a random 0.37-std policy
    (the PPO base config's initial exploration, P5-D4). Observations, rewards, flags, the
    deck states the plate was driven through, and the whole episode row must be identical.
    """
    source = deck_source(EPISODE_SPEC)
    runs_new = [_fly(*job) for job in _jobs(landing_env(source, pad=pad))]

    def eager(motion, pad_name, cfg, grid, offset):  # type: ignore[no-untyped-def]
        return build_trajectory(motion, pad_name, cfg, grid, offset)

    monkeypatch.setattr(landing_env_module, "LazyDeckTrajectory", eager)
    old_env = landing_env(source, pad=pad)
    old_env._visual_shapes = True  # noqa: SLF001  (upstream housekeeping from here on)
    runs_old = [_fly(*job) for job in _jobs(old_env)]
    assert isinstance(old_env._trajectory, DeckTrajectory)  # noqa: SLF001
    assert pyb.getVisualShapeData(int(old_env.DRONE_IDS[0]), physicsClientId=old_env.CLIENT)

    outcomes = []
    for new, old in zip(runs_new, runs_old, strict=True):
        assert new["obs"].shape == old["obs"].shape
        assert np.array_equal(new["obs"], old["obs"])
        assert np.array_equal(new["rewards"], old["rewards"])
        assert new["flags"] == old["flags"]
        assert np.array_equal(new["decks"], old["decks"])
        assert new["row"] == old["row"]
        outcomes.append(new["outcome"])
    # The descents must actually reach the deck, or the contact path went untested.
    assert sum(o in ("success", "hard_landing", "bounce", "off_pad") for o in outcomes[:3]) >= 2


@pytest.mark.pybullet
def test_same_seed_same_observations_on_a_moving_deck(landing_env, deck_source) -> None:
    """Determinism with the lazy deck: a seed flown 1st and 3rd gives identical observations."""
    env = landing_env(deck_source(EPISODE_SPEC))
    actions = np.random.default_rng(8).normal(0.0, 0.37, size=(120, ACTION_DIM)).astype(np.float32)

    def rollout(seed: int) -> np.ndarray:
        obs, _ = env.reset(seed=seed)
        frames = [obs.copy()]
        for action in actions:
            obs, _, terminated, truncated, _ = env.step(np.clip(action, -1, 1))
            frames.append(obs.copy())
            if terminated or truncated:
                break
        return np.array(frames)

    first = rollout(17)
    rollout(18)
    third = rollout(17)
    assert first.shape == third.shape
    assert np.array_equal(first, third)


# ----------------------------------------------------------------------- the DIRECT world


def test_upstream_housekeeping_is_the_one_mirrored() -> None:
    """Pin upstream's ``_housekeeping`` so a submodule bump cannot desynchronise the mirror."""
    source = inspect.getsource(BaseAviary._housekeeping)  # noqa: SLF001
    assert hashlib.sha256(source.encode()).hexdigest() == UPSTREAM_HOUSEKEEPING_SHA256


def _world(env: DeckLandingAviary) -> list[Any]:
    """Return every dynamics-relevant fact PyBullet exposes about the env's world.

    Args:
        env: The environment, after a reset.

    Returns:
        A list of plain values: body ids and names, per-link dynamics info, collision shape
        data, joint info, base poses and velocities, and the engine parameters.
    """
    client = int(env.CLIENT)
    facts: list[Any] = [pyb.getNumBodies(physicsClientId=client)]
    for index in range(pyb.getNumBodies(physicsClientId=client)):
        body = pyb.getBodyUniqueId(index, physicsClientId=client)
        facts.append((body, pyb.getBodyInfo(body, physicsClientId=client)))
        n_joints = pyb.getNumJoints(body, physicsClientId=client)
        facts.append(n_joints)
        for link in range(-1, n_joints):
            facts.append(pyb.getDynamicsInfo(body, link, physicsClientId=client))
            facts.append(pyb.getCollisionShapeData(body, link, physicsClientId=client))
            if link >= 0:
                facts.append(pyb.getJointInfo(body, link, physicsClientId=client))
        facts.append(pyb.getBasePositionAndOrientation(body, physicsClientId=client))
        facts.append(pyb.getBaseVelocity(body, physicsClientId=client))
    facts.append(sorted(pyb.getPhysicsEngineParameters(physicsClientId=client).items()))
    return facts


@pytest.mark.pybullet
def test_direct_world_without_the_visual_mesh_is_upstreams_world(landing_env, deck_source) -> None:
    """Skipping the drone's visual mesh leaves every dynamics fact of the world unchanged.

    And it does skip it: the upstream world has the drone's visual shape, the DIRECT one has
    none -- otherwise the ~21 ms saving would be coming from somewhere else.
    """
    source = deck_source(EPISODE_SPEC)
    fast = landing_env(source)
    slow = landing_env(source)
    slow._visual_shapes = True  # noqa: SLF001
    for seed in (0, 9):
        fast.reset(seed=seed)
        slow.reset(seed=seed)
        assert _world(fast) == _world(slow)
        assert int(fast.DRONE_IDS[0]) == int(slow.DRONE_IDS[0])
        assert int(fast.PLANE_ID) == int(slow.PLANE_ID)
        # Visual data rows are (body, link, geometry type, dimensions, mesh file, ...).
        fast_visual = pyb.getVisualShapeData(int(fast.DRONE_IDS[0]), physicsClientId=fast.CLIENT)
        slow_visual = pyb.getVisualShapeData(int(slow.DRONE_IDS[0]), physicsClientId=slow.CLIENT)
        assert [row[2] for row in slow_visual] == [pyb.GEOM_MESH]
        assert slow_visual[0][4].endswith(b"cf2.dae")
        # Without it PyBullet shows the collision cylinder as a proxy; no mesh is parsed.
        assert [row[2] for row in fast_visual] == [pyb.GEOM_CYLINDER]
        assert fast_visual[0][4] == b""
    assert not DeckLandingAviary.__init__.__kwdefaults__["gui"]  # type: ignore[index]
