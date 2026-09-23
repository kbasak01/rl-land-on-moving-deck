"""``pid_feedforward_lowvz``: the H1 closing-speed reference.

Contract checks (shape and bounds, determinism under reset, static-pad landing, the
privileged flag), plus what makes it this controller and not a new one:

* the selection rule, applied by :func:`rld.control.tuning.select_lowvz` to the committed
  ``pid_feedforward`` tuning log, picks **trial 10** (pinned), and the rule's three steps
  behave as worded on synthetic rows;
* the committed YAML equals that row's parameters exactly, and its fixed keys equal
  ``pid_feedforward.yaml``'s;
* it is ``pid_feedforward``'s class and law: same class, registered under its own name.

Units: metres, metres per second and seconds model scale.
"""

import numpy as np
import pytest

from _control_helpers import (
    assert_actions_in_space,
    committed_spec,
    fuzz_actions,
    run_episode,
)
from conftest import REPO_ROOT
from rld.control.config import load_feedforward
from rld.control.feedforward import PidFeedforward
from rld.control.registry import entry, make_controller
from rld.control.tuning import LOWVZ_SUCCESS_MARGIN, read_tuning_log, select_lowvz

NAME = "pid_feedforward_lowvz"
LOG = REPO_ROOT / "results" / "e01" / "tuning_pid_feedforward.csv"
SELECTED_TRIAL = 10


def test_privileged_flag_is_false():
    """Same observation-only feedforward as pid_feedforward: not privileged."""
    assert entry(NAME).privileged is False
    controller = make_controller(NAME)
    assert controller.privileged is False
    assert controller.name == NAME


def test_is_pid_feedforwards_class_and_law():
    """No new control law: the same class, only the committed gains differ."""
    controller = make_controller(NAME, committed_spec())
    assert type(controller) is PidFeedforward
    base = make_controller("pid_feedforward", committed_spec())
    assert type(base) is PidFeedforward and base.name == "pid_feedforward"


def test_rule_selects_trial_10_from_the_committed_log():
    """The user's rule on the existing 20-trial log picks trial 10, not 15 or 5."""
    log = read_tuning_log(LOG)
    assert len(log) == 20
    assert LOWVZ_SUCCESS_MARGIN == 0.02
    assert select_lowvz(log) == SELECTED_TRIAL
    row = {int(r["trial"]): r for r in log}[SELECTED_TRIAL]
    assert float(row["mean_success"]) == pytest.approx(178 / 180)
    assert float(row["pooled_td_closing_p95_m_s"]) == pytest.approx(0.152995, abs=1e-6)


def test_rule_steps_on_synthetic_rows():
    """Cutoff inclusive at best - 0.02; lowest p95 wins; NaN last; lowest index on ties."""

    def row(trial: int, success: float, p95: float) -> dict[str, str]:
        return {
            "trial": str(trial),
            "mean_success": repr(success),
            "pooled_td_closing_p95_m_s": repr(p95),
        }

    # Trial 2 is below the cutoff and would win on p95; trial 1 is the answer.
    assert select_lowvz([row(0, 1.0, 0.3), row(1, 0.985, 0.2), row(2, 0.97, 0.1)]) == 1
    # Exactly at the cutoff counts (the same float expression as the cutoff, so equal).
    assert select_lowvz([row(0, 0.5, 0.3), row(1, 0.5 - 0.02, 0.2)]) == 1
    # NaN p95 (nothing touched down) sorts last.
    assert select_lowvz([row(0, 1.0, float("nan")), row(1, 1.0, 0.4)]) == 1
    # Tie on p95: the lower trial index.
    assert select_lowvz([row(3, 1.0, 0.2), row(1, 0.99, 0.2), row(2, 1.0, 0.25)]) == 1
    with pytest.raises(ValueError, match="empty"):
        select_lowvz([])


def test_committed_yaml_is_the_rules_row_of_the_log():
    """YAML = trial 10's five searched values at full precision; fixed keys = pid_feedforward's."""
    log = read_tuning_log(LOG)
    selected = select_lowvz(log)
    row = {int(r["trial"]): r for r in log}[selected]
    params = {k[len("param_") :]: float(v) for k, v in row.items() if k.startswith("param_")}
    assert set(params) == {
        "kp_xy_per_s",
        "ki_xy_per_s2",
        "commit_radius_m",
        "descent_rate_m_s",
        "k_ff",
    }
    cfg = load_feedforward(entry(NAME).config_path)
    assert cfg.k_ff == params.pop("k_ff")
    for key, value in params.items():
        assert getattr(cfg.pid, key) == value, key
    base = load_feedforward(entry("pid_feedforward").config_path)
    for key in ("integral_limit_m_s", "lateral_speed_max_m_s", "release_factor"):
        assert getattr(cfg.pid, key) == getattr(base.pid, key), key
    # It is not the success-tuned row.
    assert row["selected"] == "False"


def test_actions_stay_in_the_shared_space_on_arbitrary_observations(env_obs_cfg):
    """Any observation inside the declared bounds gives a float32 action in the unit ball."""
    controller = make_controller(NAME, committed_spec())
    controller.reset(0)
    assert_actions_in_space(fuzz_actions(controller, env_obs_cfg, 500, seed=5))


@pytest.mark.pybullet
def test_deterministic_under_reset(landing_env, deck_source):
    """Same seed, same instance, another episode in between: bit-identical actions."""
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
    """Ten static-pad episodes all succeed at the slow (0.111 m/s) descent."""
    env = landing_env(static_motion)
    controller = make_controller(NAME, committed_spec())
    for seed in range(10):
        _, record, _ = run_episode(env, controller, seed, privileged=False)
        assert record.outcome == "success", (seed, record.as_row())
        assert record.contact_events == 1
