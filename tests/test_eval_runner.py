"""The evaluation runner: worker/chunk independence, list assertions, privilege, the hook.

The tiny list these tests fly is built from **dev-pool train** realizations plus the static
pad -- never from the committed evaluation lists -- so running the test suite cannot put a
controller in front of an evaluation episode. It still goes through the same draw ->
real-reset resolution the committed lists went through, and it switches hull geometry
(static pad, zero lever arm; frigate aft pad, -1.984 m) between episodes, so a stale
``set_motion`` would show up.

Units: times seconds model scale, lengths metres model scale; actions normalised.
"""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from dmf.typedefs import FloatArray

from rld.control.base import PrivilegedContext
from rld.control.registry import REGISTRY
from rld.deck.splits import dev_pool
from rld.eval.envs import STATIC_LABEL, EvalConfigs, load_eval_configs
from rld.eval.episodes import EpisodeDraw, ListedEpisode, resolve_draws
from rld.eval.metrics import cell_metrics
from rld.eval.runner import (
    EPISODE_COLUMNS,
    CallablePolicy,
    EpisodeListMismatchError,
    PolicySpec,
    callable_spec,
    controller_spec,
    episode_rows_csv,
    run_list,
)

pytestmark = [pytest.mark.pybullet, pytest.mark.slow]


@pytest.fixture(scope="module")
def cfgs() -> EvalConfigs:
    """Return the committed configs."""
    return load_eval_configs()


@pytest.fixture(scope="module")
def tiny_list(cfgs: EvalConfigs) -> list[ListedEpisode]:
    """Return five resolved episodes: two static, three dev-pool-train frigate realizations."""
    train, _ = dev_pool(cfgs.sim)
    picks = [train[0], train[len(train) // 2], train[-1]]
    draws = [
        EpisodeDraw("test", STATIC_LABEL, 0, "aft", STATIC_LABEL, 180.0, 0.0, 0, 11, False),
        *(
            EpisodeDraw(
                "test",
                spec.sea_state,
                i + 1,
                "aft",
                spec.vessel,
                float(spec.heading_deg),
                float(spec.speed_kn),
                int(spec.seed),
                1000 + i,
                True,
            )
            for i, spec in enumerate(picks)
        ),
        EpisodeDraw("test", STATIC_LABEL, 4, "aft", STATIC_LABEL, 180.0, 0.0, 0, 12, False),
    ]
    return resolve_draws(draws, cfgs, workers=1)


def test_registry_privilege_flags() -> None:
    assert {name: e.privileged for name, e in REGISTRY.items()} == {
        "pid_track_descend": False,
        "pid_feedforward": False,
        "pid_feedforward_lowvz": False,
        "gated": False,
        "oracle_gated": True,
    }
    for name in REGISTRY:
        assert controller_spec(name).privileged is REGISTRY[name].privileged


@pytest.mark.parametrize("name", ["pid_feedforward", "oracle_gated"])
def test_rows_do_not_depend_on_workers_or_chunk(
    name: str, tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    spec = controller_spec(name)
    one = run_list(tiny_list, spec, cfgs, workers=1, chunk=1)
    two = run_list(tiny_list, spec, cfgs, workers=2, chunk=2)
    whole = run_list(tiny_list, spec, cfgs, workers=1, chunk=len(tiny_list))
    assert episode_rows_csv(one) == episode_rows_csv(two) == episode_rows_csv(whole)
    assert [tuple(row) for row in one] == [EPISODE_COLUMNS] * len(tiny_list)
    assert [row["index"] for row in one] == [row.index for row in tiny_list]
    assert all(row["privileged"] is (name == "oracle_gated") for row in one)
    # Ground truth: a verdict exactly when the contact detector fired, NaN otherwise; the
    # static deck is always quiescent.
    for row in one:
        verdict = row["td_in_quiescent_window"]
        if row["touchdown_contact"]:
            assert isinstance(verdict, bool)
            if row["vessel"] == STATIC_LABEL:
                assert verdict is True
        else:
            assert isinstance(verdict, float) and np.isnan(verdict)
    metrics = cell_metrics(one)
    fractions = (
        metrics.frac_crash,
        metrics.frac_off_pad,
        metrics.frac_hard_landing,
        metrics.frac_bounce,
        metrics.frac_success,
        metrics.frac_timeout,
    )
    assert sum(fractions) == pytest.approx(1.0, abs=1e-12)


def test_every_reset_is_checked_against_the_list(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    spec = controller_spec("pid_track_descend")
    bad_t0 = [tiny_list[0], replace(tiny_list[1], t0_model_s=tiny_list[1].t0_model_s + 1e-12)]
    with pytest.raises(EpisodeListMismatchError, match="t0"):
        run_list(bad_t0, spec, cfgs, workers=1)
    bad_init = [replace(tiny_list[2], init_z_m=np.nextafter(tiny_list[2].init_z_m, 9.0))]
    with pytest.raises(EpisodeListMismatchError, match="init"):
        run_list(bad_init, spec, cfgs, workers=1)
    bad_seed = [replace(tiny_list[3], episode_seed=tiny_list[3].episode_seed + 1)]
    with pytest.raises(EpisodeListMismatchError):
        run_list(bad_seed, spec, cfgs, workers=1)


class _Spy:
    """Hover in place and record what ``reset`` was handed."""

    seen: list[Any] = []

    def __init__(self, privileged: bool | None) -> None:
        if privileged is not None:
            self.privileged = privileged

    def reset(self, seed: int, context: PrivilegedContext | None = None) -> None:
        del seed
        _Spy.seen.append(context)

    def act(self, obs: FloatArray) -> FloatArray:
        del obs
        return np.zeros(3)


def _spy_builder(privileged: bool | None) -> Any:
    def build(cfgs: EvalConfigs) -> _Spy:
        del cfgs
        return _Spy(privileged)

    return build


def test_context_only_under_a_privileged_spec(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    episodes = tiny_list[:2]
    _Spy.seen = []
    run_list(episodes, PolicySpec("spy", False, _spy_builder(None)), cfgs, workers=1)
    assert _Spy.seen == [None, None]
    _Spy.seen = []
    run_list(episodes, PolicySpec("spy", True, _spy_builder(True)), cfgs, workers=1)
    assert all(isinstance(ctx, PrivilegedContext) for ctx in _Spy.seen)
    assert [ctx.t0_model_s for ctx in _Spy.seen] == [row.t0_model_s for row in episodes]


def test_a_policy_cannot_claim_privilege_its_spec_lacks(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    with pytest.raises(ValueError, match="privileged"):
        run_list(tiny_list[:1], PolicySpec("spy", False, _spy_builder(True)), cfgs, workers=1)


def _hover(obs: FloatArray) -> FloatArray:
    return np.zeros(3, dtype=np.float32) * float(obs.size > 0)


def _hover_builder(cfgs: EvalConfigs) -> CallablePolicy:
    del cfgs
    return CallablePolicy(_hover)


def test_phase5_callable_hook(tiny_list: list[ListedEpisode], cfgs: EvalConfigs) -> None:
    spec = callable_spec("hover", _hover_builder, run_seed=3)
    assert spec.privileged is False
    rows = run_list(tiny_list[:2], spec, cfgs, workers=1)
    assert [row["outcome"] for row in rows] == ["timeout", "timeout"]
    assert all(row["run_seed"] == 3 and row["method"] == "hover" for row in rows)
    assert all(row["effort_mean_sq"] == 0.0 and row["action_jerk_mean"] == 0.0 for row in rows)
    assert all(np.isnan(row["rel_vz_normal_m_s"]) and row["n_contacts"] == 0 for row in rows)
    assert all(np.isnan(row["td_in_quiescent_window"]) for row in rows)
