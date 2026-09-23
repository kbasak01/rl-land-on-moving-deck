"""Phase 5: the SS3 -> SS4 -> SS5 curriculum and the pool-sampling wrapper.

P3-D1 section 6: promote at rolling tune-pool success >= 0.80 over >= 100 episodes at the
current sea state; SS6 is never trained on. The wrapper must never sample outside the pool
it holds and must refuse to hold, or be switched to, SS6.

Units: episode counts dimensionless; steps are env control steps.
"""

from collections import Counter

import pytest
from dmf.sim.generate import RealizationSpec

from _rl_helpers import FakeLanding
from conftest import BASE_SEED
from rld.control.tuning import TuningEpisode
from rld.deck.splits import key_for_spec
from rld.rl.config import CurriculumConfig
from rld.rl.curriculum import Curriculum, should_promote
from rld.rl.wrappers import PoolSamplingEnv

CFG = CurriculumConfig(
    enabled=True, stages=("SS3", "SS4", "SS5"), promote_success=0.80, window_episodes=100
)


def test_should_promote_boundaries() -> None:
    assert not should_promote([True] * 99, threshold=0.8, min_episodes=100)
    assert should_promote([True] * 80 + [False] * 20, threshold=0.8, min_episodes=100)
    assert not should_promote([True] * 79 + [False] * 21, threshold=0.8, min_episodes=100)
    # Rolling: only the most recent 100 count.
    assert should_promote([False] * 50 + [True] * 100, threshold=0.8, min_episodes=100)
    assert not should_promote([True] * 100 + [False] * 21, threshold=0.8, min_episodes=100)


def test_promotion_needs_a_full_window_at_the_current_stage() -> None:
    cur = Curriculum(CFG)
    assert cur.stage == "SS3"
    assert not cur.observe([True] * 60, steps=1)  # one 60-episode evaluation: 60 < 100
    assert cur.stage == "SS3"
    assert cur.observe([True] * 60, steps=2)  # second evaluation completes the window
    assert cur.stage == "SS4" and cur.window == ()  # window cleared at promotion
    assert cur.history[0]["from"] == "SS3" and cur.history[0]["to"] == "SS4"
    assert not cur.observe([True] * 40 + [False] * 20, steps=3)
    # The last 100 are 20 T + 20 F + 40 T + 20 F = 0.60: no promotion.
    assert cur.observe([True] * 40 + [False] * 20, steps=4) is False
    assert cur.stage == "SS4"


def test_curriculum_never_passes_ss5() -> None:
    cur = Curriculum(CFG)
    for step in range(20):
        cur.observe([True] * 60, steps=step)
    assert cur.stage == "SS5" and cur.is_final
    assert [h["to"] for h in cur.history] == ["SS4", "SS5"]


@pytest.mark.parametrize("stages", [("SS3", "SS6"), ("SS6",), ("SS3", "SS4", "SS5", "SS6")])
def test_curriculum_rejects_ss6(stages: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="never trained"):
        Curriculum(CurriculumConfig(True, stages, 0.8, 100))


def test_disabled_curriculum_samples_all_stages_and_never_promotes() -> None:
    cur = Curriculum(CurriculumConfig(False, ("SS3", "SS4", "SS5"), 0.8, 100))
    assert cur.sampling_stage is None
    assert not cur.observe([True] * 200, steps=1)
    assert cur.stage == "SS3"


def _pool() -> list[RealizationSpec]:
    return [
        RealizationSpec(sea_state=ss, heading_deg=180.0, speed_kn=6.0, vessel="frigate", seed=s)
        for ss in ("SS3", "SS4", "SS5")
        for s in range(4)
    ]


def _wrapper(pool: list[RealizationSpec], stage: str | None = "SS3") -> PoolSamplingEnv:
    return PoolSamplingEnv(
        FakeLanding(),
        pool,
        lambda spec: spec,  # type: ignore[arg-type,return-value]
        lambda _vessel: (0.0, 0.0, 0.0),
        pad="aft",
        seed=BASE_SEED,
        rank=0,
        stage=stage,
        deck_origin_m=(0.0, 0.0, 1.0),
    )


def test_wrapper_samples_only_its_pool_at_the_current_stage() -> None:
    pool = _pool()
    env = _wrapper(pool)
    keys = {key_for_spec(s) for s in pool}
    seen: Counter[str] = Counter()
    for stage in ("SS3", "SS4", "SS5", None):
        env.set_stage(stage)
        for _ in range(200):
            env.reset()
            spec: RealizationSpec = env.unwrapped.motion  # the fake stores the spec
            assert key_for_spec(spec) in keys
            if stage is not None:
                assert spec.sea_state == stage
            seen[spec.sea_state] += 1
    assert set(seen) == {"SS3", "SS4", "SS5"}


def test_wrapper_refuses_ss6() -> None:
    env = _wrapper(_pool())
    with pytest.raises(ValueError, match="never trained"):
        env.set_stage("SS6")
    ss6 = RealizationSpec(
        sea_state="SS6", heading_deg=180.0, speed_kn=6.0, vessel="frigate", seed=0
    )
    with pytest.raises(ValueError, match="SS6"):
        _wrapper([*_pool(), ss6])


def test_wrapper_sampling_is_seeded_per_worker() -> None:
    def draws(rank: int) -> list[tuple[object, ...]]:
        env = PoolSamplingEnv(
            FakeLanding(),
            _pool(),
            lambda spec: spec,  # type: ignore[arg-type,return-value]
            lambda _vessel: (0.0, 0.0, 0.0),
            pad="aft",
            seed=3,
            rank=rank,
            stage=None,
            deck_origin_m=(0.0, 0.0, 1.0),
        )
        out = []
        for _ in range(20):
            _, info = env.reset()
            out.append((key_for_spec(env.unwrapped.motion), info["episode_seed"]))
        return out

    assert draws(0) == draws(0)
    assert draws(0) != draws(1)


def test_queue_mode_flies_listed_episodes_then_idles() -> None:
    pool = _pool()
    env = _wrapper(pool, stage=None)
    episodes = [
        TuningEpisode(
            ss="SS4",
            index=i,
            pad="aft",
            vessel="frigate",
            heading_deg=180.0,
            speed_kn=6.0,
            realization_seed=i,
            episode_seed=1000 + i,
        )
        for i in range(2)
    ]
    env.load_episode_queue(episodes)
    rows = []
    for _ in range(3):
        env.reset()
        done = False
        info: dict[str, object] = {}
        while not done:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
        rows.append(info)
    assert [r["eval_idle"] for r in rows] == [False, False, True]
    assert [r["episode_seed"] for r in rows[:2]] == [1000, 1001]
    assert rows[0]["eval_index"] == 0 and rows[1]["eval_index"] == 1
    fake = env.unwrapped
    assert fake.seeds[:2] == [1000, 1001]  # type: ignore[attr-defined]
    outside = TuningEpisode(
        ss="SS4",
        index=0,
        pad="aft",
        vessel="frigate",
        heading_deg=45.0,
        speed_kn=0.0,
        realization_seed=30,
        episode_seed=1,
    )
    with pytest.raises(ValueError, match="not in the pool"):
        env.load_episode_queue([outside])
