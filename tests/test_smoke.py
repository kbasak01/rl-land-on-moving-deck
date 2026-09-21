"""Bootstrap smoke tests -- Gate 0.

These assert the four items of Phase 0 task 4 in ``docs/IMPLEMENTATION_PLAN.md``, plus the
two environment traps recorded in ``CLAUDE.md`` that an import alone would not catch:

1. ``pybullet`` has no cp312 wheel and is built from sdist, so a successful ``import`` does
   not prove the compiled extension links -- a DIRECT connection does.
2. ``dmf`` resolves ``configs/sim/vessels`` via ``Path(__file__).parents[3]``, so
   :func:`dmf.sim.generate.simulate_realization` only works from an **editable** install.
   The parity of that call is a Phase 1 concern; here it only has to run.

Everything here is simulated. Nothing in this suite touches real deck or flight data.
"""

import math

import numpy as np
import pytest

from conftest import BASE_SEED, DMF_ROOT

#: Steps of the random-policy rollout in item 3 of the Phase 0 smoke set.
ROLLOUT_STEPS = 1000

#: Env steps of the SB3 PPO smoke run in item 4 of the Phase 0 smoke set.
PPO_STEPS = 20_000


def test_import_rld() -> None:
    import rld

    assert rld.__version__ == "0.1.0"


def test_import_stack() -> None:
    import dmf
    import gym_pybullet_drones
    import gymnasium
    import pybullet
    import rliable
    import stable_baselines3
    import torch

    assert dmf.__version__ == "0.1.0"
    for module in (gym_pybullet_drones, gymnasium, pybullet, rliable, stable_baselines3, torch):
        assert module is not None


@pytest.mark.pybullet
def test_pybullet_direct_client_connects() -> None:
    """A built-from-sdist pybullet must actually connect, not merely import."""
    import pybullet

    client = pybullet.connect(pybullet.DIRECT)
    try:
        assert client >= 0
        pybullet.setGravity(0, 0, -9.81, physicsClientId=client)
        pybullet.stepSimulation(physicsClientId=client)
    finally:
        pybullet.disconnect(client)


@pytest.mark.slow
def test_dmf_simulate_realization_runs_from_editable_install() -> None:
    """One dmf realization end to end, which also proves the editable-install path."""
    from dmf.config import load_sim
    from dmf.sim.generate import RealizationSpec, simulate_realization

    cfg = load_sim(DMF_ROOT / "configs" / "sim" / "corpus.yaml")
    spec = RealizationSpec(
        seed=0, sea_state="SS5", heading_deg=180.0, speed_kn=12.0, vessel="frigate"
    )
    frame = simulate_realization(spec, cfg)

    assert len(frame) == round(cfg.duration_s * cfg.fs_hz)
    assert math.isclose(float(frame["t"].iloc[0]), cfg.spinup_s, abs_tol=1e-9)
    for column in ("roll", "pitch", "heave", "roll_rate", "pitch_rate", "heave_rate"):
        values = frame[column].to_numpy()
        assert np.isfinite(values).all(), f"{column} is not finite"
    # Degrees, metres: a frigate in SS5 head seas moves, and does not move absurdly.
    assert 0.0 < float(np.std(frame["heave"].to_numpy())) < 10.0
    assert 0.0 < float(np.std(frame["pitch"].to_numpy())) < 45.0


@pytest.mark.pybullet
def test_hover_aviary_random_rollout_headless() -> None:
    """1000 random steps in DIRECT mode, resetting on termination or truncation."""
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary

    env = HoverAviary(gui=False, record=False)
    try:
        obs, _ = env.reset(seed=BASE_SEED)
        env.action_space.seed(BASE_SEED)
        episodes = 0
        for _ in range(ROLLOUT_STEPS):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(np.asarray(obs)).all()
            assert math.isfinite(float(reward))
            if terminated or truncated:
                obs, _ = env.reset(seed=BASE_SEED + episodes)
                episodes += 1
        assert episodes > 0, "1000 steps produced no episode boundary; check EPISODE_LEN_SEC"
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.pybullet
def test_ppo_trains_hover_aviary() -> None:
    """SB3 PPO completes 20k env steps on HoverAviary and predicts an in-bounds action.

    ``device="cpu"`` deliberately: an SB3 MLP policy is slower on the GPU than on the CPU
    at these widths (CLAUDE.md, known traps), and Phase 5 trains PPO on the CPU.
    """
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from stable_baselines3 import PPO

    env = HoverAviary(gui=False, record=False)
    try:
        model = PPO("MlpPolicy", env, device="cpu", seed=BASE_SEED, verbose=0)
        model.learn(total_timesteps=PPO_STEPS)
        assert model.num_timesteps >= PPO_STEPS

        obs, _ = env.reset(seed=BASE_SEED)
        action, _ = model.predict(obs, deterministic=True)
        assert env.action_space.contains(np.asarray(action, dtype=np.float32))
    finally:
        env.close()
