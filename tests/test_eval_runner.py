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
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from dmf.typedefs import FloatArray

from rld.control.base import PrivilegedContext
from rld.control.registry import REGISTRY
from rld.deck.splits import dev_pool
from rld.eval.envs import (
    STATIC_LABEL,
    EvalConfigs,
    load_eval_configs,
    make_env,
    motion_for,
    pad_offset_for,
)
from rld.eval.episodes import EpisodeDraw, ListedEpisode, resolve_draws
from rld.eval.metrics import cell_metrics
from rld.eval.runner import (
    EPISODE_COLUMNS,
    STATIC_FEED_SKIP_REASON,
    CallablePolicy,
    EpisodeListMismatchError,
    FeedClockError,
    PolicySpec,
    _advance_feed,
    callable_spec,
    controller_spec,
    episode_rows_csv,
    feed_skip_reason,
    run_list,
    split_runnable,
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
        "gated_forecast": False,
        "gated_forecast_tcn": False,
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


# --------------------------------------------------------------------------- the feed


class _FeedSpy:
    """Hover in place; record the feed handed to ``reset`` and its clock at every act."""

    needs_motion_feed = True
    feeds: list[Any] = []
    t0s: list[float] = []
    clocks: list[list[float]] = []

    def reset(
        self, seed: int, context: PrivilegedContext | None = None, motion_feed: Any = None
    ) -> None:
        del seed
        assert context is None
        assert motion_feed is not None
        _FeedSpy.feeds.append(motion_feed)
        _FeedSpy.t0s.append(float(motion_feed.clock_model_s))
        _FeedSpy.clocks.append([])
        self._feed = motion_feed

    def act(self, obs: FloatArray) -> FloatArray:
        del obs
        _FeedSpy.clocks[-1].append(float(self._feed.clock_model_s))
        # Causality: the newest sample in the window is at or before the clock.
        assert self._feed.t_origin_model_s <= self._feed.clock_model_s
        return np.zeros(3)


def _feed_spy_builder(cfgs: EvalConfigs) -> _FeedSpy:
    del cfgs
    return _FeedSpy()


def test_feed_is_built_per_episode_and_clocked_to_the_env(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    spec = PolicySpec("feed_spy", False, _feed_spy_builder, needs_motion_feed=True)
    runnable, skipped = split_runnable(tiny_list, spec)
    assert [row.vessel for row in runnable] == ["frigate"] * 3
    assert [s.episode.index for s in skipped] == [0, 4]
    assert all(s.reason == STATIC_FEED_SKIP_REASON and s.pad == "aft" for s in skipped)
    _FeedSpy.feeds, _FeedSpy.t0s, _FeedSpy.clocks = [], [], []
    rows = run_list(runnable, spec, cfgs, workers=1, chunk=len(runnable))
    # A new feed per episode, never shared.
    assert len({id(feed) for feed in _FeedSpy.feeds}) == len(runnable)
    dt = cfgs.landing.ctrl_dt_s
    for row, t0, clocks in zip(rows, _FeedSpy.t0s, _FeedSpy.clocks, strict=True):
        assert t0 == row["t0_model_s"]
        assert len(clocks) == row["steps"]
        assert clocks == [t0 + k * dt for k in range(len(clocks))]
    # The feed never becomes a privilege.
    assert all(row["privileged"] is False for row in rows)


def test_a_feed_spec_refuses_the_static_list(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    spec = PolicySpec("feed_spy", False, _feed_spy_builder, needs_motion_feed=True)
    assert feed_skip_reason(tiny_list[0], spec) == STATIC_FEED_SKIP_REASON
    assert feed_skip_reason(tiny_list[1], spec) is None
    assert feed_skip_reason(tiny_list[0], controller_spec("gated")) is None
    with pytest.raises(ValueError, match="static-pad list"):
        run_list(tiny_list[:1], spec, cfgs, workers=1)


def test_feed_flags_must_agree(tiny_list: list[ListedEpisode], cfgs: EvalConfigs) -> None:
    # A feed-consuming policy under a spec without the feed is refused, and vice versa.
    with pytest.raises(ValueError, match="needs_motion_feed"):
        run_list(tiny_list[1:2], PolicySpec("spy", False, _feed_spy_builder), cfgs, workers=1)
    with pytest.raises(ValueError, match="ShipMotionFeed"):
        CallablePolicy(_hover).reset(0, None, object())  # type: ignore[arg-type]
    for name in REGISTRY:
        assert controller_spec(name).needs_motion_feed is REGISTRY[name].needs_motion_feed
    assert controller_spec("gated_forecast").needs_motion_feed is True
    assert controller_spec("gated").needs_motion_feed is False


def test_feed_clock_mismatch_raises() -> None:
    class Env:
        cfg = SimpleNamespace(ctrl_dt_s=1.0 / 30.0)
        record = SimpleNamespace(steps=0)

    class StuckFeed:
        clock_model_s = 10.0

        def advance_to(self, t: float) -> None:
            del t  # forgets to move the clock

    env: Any = Env()
    feed: Any = StuckFeed()
    _advance_feed(feed, env, 10.0, 0)  # k = 0: the clock is already t0
    env.record.steps = 1
    with pytest.raises(FeedClockError, match="env control time"):
        _advance_feed(feed, env, 10.0, 1)
    with pytest.raises(FeedClockError, match="control steps"):
        _advance_feed(feed, env, 10.0, 2)


# --------------------------------------------------------------------------- pad override


def test_pad_override_flies_the_same_listed_episodes_on_the_cg(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    spec = controller_spec("pid_feedforward")
    aft = run_list(tiny_list, spec, cfgs, workers=1, chunk=len(tiny_list))
    same = run_list(tiny_list, spec, cfgs, workers=1, chunk=len(tiny_list), pad_override="aft")
    cg = run_list(tiny_list, spec, cfgs, workers=1, chunk=len(tiny_list), pad_override="cg")
    # "aft" is the listed pad: overriding with it changes nothing.
    assert episode_rows_csv(same) == episode_rows_csv(aft)
    assert [row["pad"] for row in cg] == ["cg"] * len(tiny_list)
    for a, c in zip(aft, cg, strict=True):
        # Same start (the reset check passed on the listed t0 and initial position).
        assert c["t0_model_s"] == a["t0_model_s"]
        assert (c["init_x_m"], c["init_y_m"], c["init_z_m"]) == (
            a["init_x_m"],
            a["init_y_m"],
            a["init_z_m"],
        )
        if a["vessel"] == STATIC_LABEL:
            # No lever arm on the static fixture: identical but for the label (NaN-safe).
            assert episode_rows_csv([{**c, "pad": a["pad"]}]) == episode_rows_csv([a])
        else:
            # A moving deck at another pad is a different episode.
            assert c["steps"] != a["steps"] or c["return"] != a["return"]


def test_ground_truth_reads_the_effective_pad(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    """``td_in_quiescent_window`` of a pad-at-CG row is the CG pad's truth, not the aft's."""
    from rld.eval.truth import td_in_quiescent_window, touchdown_rule

    moving = tiny_list[1:4]
    rows = run_list(moving, controller_spec("pid_feedforward"), cfgs, workers=1, pad_override="cg")
    rule = touchdown_rule(cfgs)
    stride = int(cfgs.landing.pyb_steps_per_ctrl)
    checked = 0
    for listed, row in zip(moving, rows, strict=True):
        contexts = {}
        for pad in ("aft", "cg"):
            env = make_env(
                cfgs,
                motion_for(
                    cfgs,
                    listed.vessel,
                    listed.ss,
                    listed.heading_deg,
                    listed.speed_kn,
                    listed.realization_seed,
                ),
                listed.vessel,
                pad,
                listed.episode_seed,
            )
            try:
                env.reset(seed=listed.episode_seed)
                assert env.pad == pad
                assert tuple(env.pad_offset_m) == pad_offset_for(cfgs, listed.vessel, pad)
                contexts[pad] = PrivilegedContext.from_env(env)
            finally:
                env.close()
        # The two pads' true v_z differ (the pitch lever arm), so the pad matters here.
        assert not np.array_equal(contexts["aft"].velocity_m_s, contexts["cg"].velocity_m_s)
        if row["touchdown_contact"]:
            expected = td_in_quiescent_window(
                contexts["cg"], rule, float(row["time_to_touchdown_s"]), stride
            )
            assert row["td_in_quiescent_window"] is expected
            checked += 1
    assert checked > 0


def test_real_gated_forecast_runs_through_the_runner(
    tiny_list: list[ListedEpisode], cfgs: EvalConfigs
) -> None:
    """The registry's ``gated_forecast`` with its real feed and ORT forecaster, both pads.

    Skipped when the forecaster is not fitted. Dev-pool train episodes only; the outcome is
    not asserted, only that the runner wires the feed (the controller's own clock check
    raises otherwise) and that rows are complete and independent of the chunking.
    """
    from rld.config import REPO_ROOT

    if not (REPO_ROOT / "artifacts" / "dmf" / "residual_interval" / "meta.json").is_file():
        pytest.skip("residual_interval forecaster not fitted")
    spec = controller_spec("gated_forecast")
    runnable, _ = split_runnable(tiny_list, spec)
    for pad in (None, "cg"):
        one = run_list(runnable, spec, cfgs, workers=1, chunk=1, pad_override=pad)
        whole = run_list(runnable, spec, cfgs, workers=1, chunk=len(runnable), pad_override=pad)
        assert episode_rows_csv(one) == episode_rows_csv(whole)
        assert all(row["method"] == "gated_forecast" and not row["privileged"] for row in one)
        assert {row["pad"] for row in one} == {pad or "aft"}
