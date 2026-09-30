"""Phase 6: ``ppo_sinusoid`` -- every training and tune-pool episode flies a matched sinusoid.

* The committed per-realization RMS (``results/deck_stats_seeds.csv``) is the RMS
  :func:`rld.deck.sinusoid.matched_rms` computes from the JONSWAP record, and covers the
  whole P3-D2 train and tune pools.
* Phases are a deterministic function of the episode seed and differ across seeds;
  amplitudes are the committed ones.
* :class:`~rld.rl.wrappers.EnvFactory` with ``motion: sinusoid`` hands the landing env a
  :class:`~rld.deck.sinusoid.SinusoidDeckMotion` for every episode -- training (with reset
  prefetch) and queue-mode evaluation alike -- and never a JONSWAP source.

Units: RMS and amplitudes full scale (degrees, metres); periods seconds full scale.
"""

import csv
import math
from typing import Any

import numpy as np
import pytest
from dmf.sim.generate import RealizationSpec

from _rl_helpers import phase6_config
from rld.control.tuning import TuningEpisode
from rld.deck.bridge import JonswapDeckMotion
from rld.deck.sinusoid import SinusoidDeckMotion, matched_rms, sinusoid_params
from rld.deck.splits import key_for_spec
from rld.eval.envs import load_eval_configs
from rld.rl.motion import DECK_STATS_SEEDS, committed_rms_full, sinusoid_motion
from rld.rl.train import _factory_components, build_env_configs, training_pools
from rld.rl.wrappers import EnvFactory

SPECS = (
    RealizationSpec(sea_state="SS3", heading_deg=180.0, speed_kn=0.0, vessel="frigate", seed=0),
    RealizationSpec(sea_state="SS5", heading_deg=135.0, speed_kn=12.0, vessel="frigate", seed=3),
)


@pytest.mark.slow
def test_committed_rms_is_matched_rms() -> None:
    cfgs = load_eval_configs()
    scale = cfgs.scaling.froude_scale()
    table = committed_rms_full(key_for_spec(s) for s in SPECS)
    for spec in SPECS:
        want = matched_rms(spec, cfgs.sim, scale, cfgs.pads)
        assert np.allclose(table[key_for_spec(spec)], want, rtol=1e-9, atol=0.0), spec


def test_committed_rms_covers_the_pools_and_round_trips_the_amplitudes() -> None:
    cfg = phase6_config("ppo_sinusoid")
    cfgs = build_env_configs(cfg)
    train, tune = training_pools(cfg, cfgs)
    keys = {key_for_spec(s) for s in (*train, *tune)}
    table = committed_rms_full(keys)
    assert set(table) == keys and len(keys) == 864
    with DECK_STATS_SEEDS.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["pad"] == "aft"]
    row = next(r for r in rows if (r["ss"], r["seed"], r["heading_deg"]) == ("SS5", "3", "135.0"))
    key = key_for_spec(
        RealizationSpec(
            sea_state="SS5",
            heading_deg=135.0,
            speed_kn=float(row["speed_kn"]),
            vessel=row["vessel"],
            seed=3,
        )
    )
    params = sinusoid_params(
        _spec(key),
        cfgs.sim,
        cfgs.scaling.froude_scale(),
        cfgs.pads,
        episode_seed=0,
        rms_full=committed_rms_full([key])[key],
    )
    for column, value in (
        ("sin_roll_amp_deg", params.roll_amp_deg),
        ("sin_pitch_amp_deg", params.pitch_amp_deg),
        ("sin_heave_amp_m", params.heave_amp_m),
        ("sin_te_full_s", params.te_full_s),
    ):
        assert math.isclose(value, float(row[column]), rel_tol=4e-16), column
    with pytest.raises(KeyError):
        committed_rms_full([("SS3", 180.0, 0.0, "frigate", 99_999)])


def _spec(key: Any) -> RealizationSpec:
    ss, heading, speed, vessel, seed = key
    return RealizationSpec(
        sea_state=str(ss),
        heading_deg=float(heading),
        speed_kn=float(speed),
        vessel=str(vessel),
        seed=int(seed),
    )


def test_phases_are_a_function_of_the_episode_seed() -> None:
    cfgs = load_eval_configs()
    spec = SPECS[1]
    rms = committed_rms_full([key_for_spec(spec)])[key_for_spec(spec)]
    a = sinusoid_motion(cfgs, spec, 7, rms)
    b = sinusoid_motion(cfgs, spec, 7, rms)
    c = sinusoid_motion(cfgs, spec, 8, rms)
    assert a.params == b.params
    assert a.params.phase_rad != c.params.phase_rad
    assert (a.params.roll_amp_deg, a.params.te_full_s) == (
        c.params.roll_amp_deg,
        c.params.te_full_s,
    )
    assert a.key == key_for_spec(spec)
    jonswap = JonswapDeckMotion(
        spec,
        cfgs.sim,
        cfgs.scaling.froude_scale(),
        cfgs.pads,
        cfgs.motion.forecast_lookback_full_s,
    )
    assert a.t_model_window_s == pytest.approx(jonswap.t_model_window_s, abs=1e-12)


def test_sinusoid_factory_needs_the_rms_table() -> None:
    cfg = phase6_config("ppo_sinusoid")
    cfgs = build_env_configs(cfg)
    train, _ = training_pools(cfg, cfgs)
    with pytest.raises(ValueError, match="committed RMS"):
        EnvFactory(cfgs, tuple(train), "aft", 0, 0, "SS3", None, motion="sinusoid")


@pytest.mark.slow
@pytest.mark.pybullet
def test_factory_flies_a_matched_sinusoid_every_episode() -> None:
    cfg = phase6_config("ppo_sinusoid")
    assert cfg.motion == "sinusoid" and cfg.residual is None and cfg.forecast_obs is None
    cfgs = build_env_configs(cfg)
    train, tune = training_pools(cfg, cfgs)
    components = _factory_components(cfg, train, tune)
    assert components["train"]["motion"] == components["tune"]["motion"] == "sinusoid"
    scale = cfgs.scaling.froude_scale()

    def check(env: Any, info: dict[str, Any], seen: set[tuple[float, ...]]) -> None:
        motion = env.unwrapped.motion
        assert isinstance(motion, SinusoidDeckMotion), type(motion)
        spec = _spec(motion.key)
        rms = committed_rms_full([motion.key])[motion.key]
        want = sinusoid_params(
            spec, cfgs.sim, scale, cfgs.pads, episode_seed=info["episode_seed"], rms_full=rms
        )
        assert motion.params == want
        seen.add(motion.params.phase_rad)

    # Training worker, reset prefetch on: SS3 stage, several episodes on both slots.
    train_env = EnvFactory(
        cfgs, tuple(train), "aft", 0, 0, "SS3", None, True, **components["train"]
    )()
    seen: set[tuple[float, ...]] = set()
    try:
        assert isinstance(train_env.unwrapped.motion, SinusoidDeckMotion)  # never JONSWAP
        for _ in range(4):
            _, info = train_env.reset()
            check(train_env, info, seen)
            assert info["ss"] == "SS3"
            for _ in range(3):
                train_env.step(np.zeros(3, np.float32))
        assert train_env.get_wrapper_attr("prefetch_counts")["used"] >= 3
    finally:
        train_env.close()
    assert len(seen) == 4

    # Tune-pool evaluation worker (queue mode): the same episode seed gives the same motion.
    first = next(s for s in tune if s.sea_state == "SS4")
    ep = TuningEpisode(
        "SS4", 0, "aft", first.vessel, first.heading_deg, first.speed_kn, first.seed, 1234
    )
    eval_env = EnvFactory(cfgs, tuple(tune), "aft", 0, 0, None, None, True, **components["tune"])()
    try:
        eval_env.get_wrapper_attr("load_episode_queue")([ep, ep])
        phases: set[tuple[float, ...]] = set()
        for _ in range(2):
            _, info = eval_env.reset()
            check(eval_env, info, phases)
            assert info["episode_seed"] == 1234
        assert len(phases) == 1
    finally:
        eval_env.close()
