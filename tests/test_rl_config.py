"""Phase 5: training and search configs -- the P3-D1 pins and the loader's protocol checks.

The committed ``configs/rl/ppo.yaml``, ``sac.yaml`` and ``tune_*.yaml`` carry pre-registered
numbers (P3-D1 sections 5-6): 10 M PPO / 2 M SAC env steps, <= 20 trials at 2 M / 0.5 M,
promotion at 0.80 over >= 100 episodes, SS6 never trained on, PPO on the CPU, reward
normalisation PPO-only, and tuning of reward *weights* only. A silent edit to any of them
fails here.

Units: steps are env control steps (1/30 s model scale each).
"""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from _rl_helpers import tiny_raw
from rld.control.tuning import TUNING_CONFIG
from rld.rl.config import (
    RL_CONFIG_DIR,
    STRUCTURE_REWARD_KEYS,
    apply_overrides,
    config_to_dict,
    load_train_config,
    train_config_from_dict,
)
from rld.rl.tuning import (
    MAX_TRIALS,
    load_tune_config,
    materialize_trials,
    trial_points,
    trial_raw_configs,
)


def test_ppo_config_pins_p3d1() -> None:
    cfg = load_train_config(RL_CONFIG_DIR / "ppo.yaml")
    assert cfg.algo == "ppo" and cfg.method == "ppo" and cfg.run_group == "ppo"
    assert cfg.total_steps == 10_000_000
    assert cfg.device == "cpu"
    assert cfg.normalize.norm_obs and cfg.normalize.norm_reward
    assert cfg.normalize.clip_obs == 10.0
    assert cfg.curriculum.enabled and cfg.curriculum.stages == ("SS3", "SS4", "SS5")
    assert cfg.curriculum.promote_success == 0.80
    assert cfg.curriculum.window_episodes == 100
    assert cfg.eval.tuning_config == TUNING_CONFIG and cfg.eval.episodes_per_ss is None
    assert cfg.pad == "aft"
    assert cfg.workers <= 34


@pytest.mark.parametrize(
    ("final_name", "results_dir"),
    [("ppo.yaml", "results/tune/ppo"), ("sac.yaml", "results/tune/sac_v2")],
)
def test_final_config_is_the_tuning_winner(final_name: str, results_dir: str) -> None:
    # P5-D11 (PPO), P5-D12 (SAC): each final-run config equals its search's selected trial
    # config in every field except the name, run group and step budget.
    import dataclasses
    import json

    from rld.rl.tuning import REPO_ROOT

    selection = json.loads((REPO_ROOT / results_dir / "selection.json").read_text())
    winner = selection["winner_trial"]
    final = dataclasses.asdict(load_train_config(RL_CONFIG_DIR / final_name))
    trial = dataclasses.asdict(
        load_train_config(REPO_ROOT / results_dir / f"configs/trial_{winner:02d}.yaml")
    )
    differs = {k for k in final if final[k] != trial[k]}
    assert differs == {"method", "run_group", "total_steps", "source"}


def test_sac_config_pins_p3d1() -> None:
    cfg = load_train_config(RL_CONFIG_DIR / "sac.yaml")
    assert cfg.algo == "sac" and cfg.total_steps == 2_000_000
    assert cfg.normalize.norm_obs and not cfg.normalize.norm_reward
    assert cfg.normalize.clip_obs == 10.0
    assert cfg.curriculum.stages == ("SS3", "SS4", "SS5")
    assert (cfg.curriculum.promote_success, cfg.curriculum.window_episodes) == (0.80, 100)
    assert cfg.eval.tuning_config == TUNING_CONFIG and cfg.eval.episodes_per_ss is None


def test_smoke_config_gives_eight_to_twelve_eval_points() -> None:
    cfg = load_train_config(RL_CONFIG_DIR / "ppo_smoke.yaml")
    assert cfg.total_steps == 200_000 and cfg.n_envs == 16
    assert cfg.ppo is not None
    rollout = cfg.ppo.n_steps * cfg.n_envs
    final = math.ceil(cfg.total_steps / rollout) * rollout
    points = final // cfg.eval.interval_steps + (final % cfg.eval.interval_steps != 0)
    assert 8 <= points <= 12


def test_search_configs_pin_budgets_and_equal_trials() -> None:
    ppo = load_tune_config(RL_CONFIG_DIR / "tune_ppo.yaml")
    sac = load_tune_config(RL_CONFIG_DIR / "tune_sac.yaml")
    # P5-D4: PPO's smoke run is charged as one trial, so the TOTAL budgets are equal.
    assert ppo.trials + ppo.trials_used_before_search == MAX_TRIALS == 20
    assert sac.trials + sac.trials_used_before_search == MAX_TRIALS
    assert ppo.trials_used_before_search == 2 and sac.trials_used_before_search == 0
    assert ppo.trial_steps == 2_000_000 and sac.trial_steps == 500_000
    assert ppo.trial_seed == sac.trial_seed == 0
    # The reward-weight search is the same for both methods.
    reward = {p.name: p for p in ppo.params if p.targets[0].startswith("reward.")}
    reward_sac = {p.name: p for p in sac.params if p.targets[0].startswith("reward.")}
    assert reward == reward_sac and reward
    for params in (ppo.params, sac.params):
        for param in params:
            for target in param.targets:
                if target.startswith("reward."):
                    assert target.split(".", 1)[1] not in STRUCTURE_REWARD_KEYS


def test_config_round_trip() -> None:
    cfg = load_train_config(RL_CONFIG_DIR / "ppo.yaml")
    again = train_config_from_dict(yaml.safe_load(yaml.safe_dump(config_to_dict(cfg))))
    assert replace(again, source=cfg.source) == cfg


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"curriculum.stages": ["SS3", "SS6"]}, "SS6"),
        ({"curriculum.stages": ["SS3", "SS7"]}, "not training sea states"),
        ({"device": "cuda"}, "CPU"),
        ({"reward.v_safe_m_s": 0.2}, "structure"),
        ({"reward.gate_height_m": 0.2}, "structure"),
        ({"reward.not_a_weight": 1.0}, "not reward weights"),
        ({"reward.w_vz": -1.0}, "non-negative"),
        ({"ppo.batch_size": 100}, "divide"),
    ],
)
def test_ppo_config_rejects(override: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        train_config_from_dict(apply_overrides(tiny_raw("ppo"), override))


def test_sac_rejects_reward_normalisation() -> None:
    with pytest.raises(ValueError, match="PPO-only"):
        train_config_from_dict(apply_overrides(tiny_raw("sac"), {"normalize.norm_reward": True}))


def test_unknown_key_rejected() -> None:
    raw = tiny_raw("ppo")
    raw["learning_rate"] = 1e-3
    with pytest.raises(ValueError, match="unknown keys"):
        train_config_from_dict(raw)


def _tune_yaml(tmp_path: Path, **changes: object) -> Path:
    raw = yaml.safe_load((RL_CONFIG_DIR / "tune_ppo.yaml").read_text())
    raw["results_dir"] = str(tmp_path / "tune")
    raw.update(changes)
    path = tmp_path / "tune.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_search_rejects_more_than_twenty_trials(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="trials"):
        load_tune_config(_tune_yaml(tmp_path, trials=21))


def test_search_charges_trials_used_before_search(tmp_path: Path) -> None:
    # P5-D4/P5-D7: 18 drawn + 2 charged = 20 is allowed; 19 drawn + 2 charged is not.
    assert load_tune_config(_tune_yaml(tmp_path, trials=18)).trials == 18
    with pytest.raises(ValueError, match="exceeds"):
        load_tune_config(_tune_yaml(tmp_path, trials=19))


def test_search_rejects_structure_targets(tmp_path: Path) -> None:
    raw = yaml.safe_load((RL_CONFIG_DIR / "tune_ppo.yaml").read_text())
    space = dict(raw["search_space"])
    space["v_safe"] = {"targets": ["reward.v_safe_m_s"], "low": 0.1, "high": 0.4}
    with pytest.raises(ValueError, match="structure"):
        load_tune_config(_tune_yaml(tmp_path, search_space=space))


def test_trial_points_deterministic_and_trial_zero_is_base(tmp_path: Path) -> None:
    cfg = load_tune_config(_tune_yaml(tmp_path))
    assert trial_points(cfg) == trial_points(cfg)
    base = yaml.safe_load((RL_CONFIG_DIR / "ppo.yaml").read_text())
    index, raw0 = trial_raw_configs(cfg)[0]
    assert index == 0
    differing = {k for k in base if base[k] != raw0[k]}
    assert differing == {"total_steps", "run_group", "method"}
    assert raw0["total_steps"] == 2_000_000 and raw0["run_group"] == "tune_ppo/trial_00"
    # Every Sobol trial stays inside its declared ranges.
    for point in trial_points(cfg)[1:]:
        for param in cfg.params:
            value = point[param.name]
            if param.kind == "choice":
                assert value in param.choices
            else:
                assert param.low <= value <= param.high


def test_materialize_is_idempotent_and_refuses_changes(tmp_path: Path) -> None:
    cfg = load_tune_config(_tune_yaml(tmp_path))
    first = [p.read_text() for p in materialize_trials(cfg)]
    second = [p.read_text() for p in materialize_trials(cfg)]
    assert first == second and len(first) == cfg.trials
    changed = replace(cfg, sobol_seed=cfg.sobol_seed + 1)
    with pytest.raises(FileExistsError):
        materialize_trials(changed)


def test_ppo_log_std_init_reaches_the_policy(tmp_path: Path) -> None:
    # P5-D7/P5-D11: the committed PPO configs set log_std_init, and the value is really passed to
    # SB3 rather than silently left at its default of 0 (std 1.0). Each smoke config keeps the
    # value its run was flown with.
    from stable_baselines3.common.vec_env import DummyVecEnv

    from _rl_helpers import FakeLanding, tiny_config
    from rld.rl.train import _build_model

    flown = {"ppo_smoke.yaml": 0.0, "ppo_smoke_p5d4.yaml": -1.0, "ppo_smoke_p5d7.yaml": -2.3}
    for name, value in flown.items():
        raw = yaml.safe_load((RL_CONFIG_DIR / name).read_text())
        assert raw["ppo"]["log_std_init"] == value, name
    cfg = tiny_config("ppo")
    assert cfg.ppo is not None
    assert -3.0 <= cfg.ppo.log_std_init <= -1.6  # the P5-D7 search range; P5-D11 winner
    model = _build_model(cfg, DummyVecEnv([FakeLanding]), 0, tmp_path)
    log_std = model.policy.log_std.detach().numpy()  # type: ignore[union-attr]
    assert np.allclose(log_std, cfg.ppo.log_std_init)


def test_prefetch_reset_is_optional_and_round_trips() -> None:
    # Engineering switch, not a hyperparameter: absent (run dirs written before it existed)
    # means the default; present, it must survive config.yaml.
    from rld.rl.config import PREFETCH_RESET_DEFAULT

    raw = yaml.safe_load((RL_CONFIG_DIR / "ppo.yaml").read_text())
    raw.pop("prefetch_reset", None)
    assert train_config_from_dict(raw).prefetch_reset is PREFETCH_RESET_DEFAULT is True
    off = train_config_from_dict({**raw, "prefetch_reset": False})
    assert off.prefetch_reset is False
    again = train_config_from_dict(yaml.safe_load(yaml.safe_dump(config_to_dict(off))))
    assert again == replace(off, source=None)
    assert off.workers == 8  # ceil(0.4 * 16) + 1, whatever the switch
