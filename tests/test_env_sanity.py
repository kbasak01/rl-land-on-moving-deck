"""The e00 sanity sweep: its draw, its reductions, and the committed artifacts' shape.

The sweep itself is what Gate 2 reads, so the tests here are about the two properties that
would make its numbers untrustworthy: a draw that depends on the process or the worker
count, and a cell reduction that loses episodes. The full 800-episode sweep is a ``make``
target, not a test; what is tested is a small sweep with the same code path.

Units: seconds model scale, metres model scale.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from rld.envs.sanity import (
    DEFAULT_DRAW_SEED,
    POLICIES,
    SweepConfigs,
    draw_episodes,
    run_sweep,
)
from rld.envs.touchdown import OUTCOMES

#: Repository root, resolved from this file (``tests/`` is not a package, so
#: ``conftest.REPO_ROOT`` is not importable from here).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The committed artifacts, if they have been generated.
CELL_CSV = REPO_ROOT / "results" / "e00_env_sanity.csv"
EPISODE_CSV = REPO_ROOT / "results" / "e00_env_sanity_episodes.csv"


@pytest.fixture
def sweep_cfgs(
    sim_cfg,
    deck_scaling,
    deck_pads,
    deck_motion_cfg,
    env_landing_cfg,
    env_success_cfg,
    env_obs_cfg,
    env_noise_cfg,
    env_reward_cfg,
) -> SweepConfigs:
    """Bundle the committed configs the sweep needs."""
    return SweepConfigs(
        sim=sim_cfg,
        scaling=deck_scaling,
        pads=deck_pads,
        motion=deck_motion_cfg,
        landing=env_landing_cfg,
        success=env_success_cfg,
        observation=env_obs_cfg,
        noise=env_noise_cfg,
        reward=env_reward_cfg,
    )


def test_episode_draw_is_reproducible_and_from_the_id_val_partition(sim_cfg):
    """The draw is a pure function of the seed, and lands only on ``id`` val realizations.

    P2-D4: the sanity episodes must be neither trained on nor part of the Phase 3 frozen
    evaluation lists. If the draw depended on anything but the seed -- a salted string hash,
    say -- a single-worker run and a spawned worker would disagree and the committed CSV
    would not be reproducible.
    """
    from rld.deck.splits import build_all_splits, part_specs

    first = draw_episodes(sim_cfg, "random", "SS5", "aft", 25, DEFAULT_DRAW_SEED)
    again = draw_episodes(sim_cfg, "random", "SS5", "aft", 25, DEFAULT_DRAW_SEED)
    assert first == again
    other = draw_episodes(sim_cfg, "random", "SS5", "aft", 25, DEFAULT_DRAW_SEED + 1)
    assert first != other

    val_keys = {
        (spec.vessel, spec.sea_state, spec.heading_deg, spec.speed_kn, spec.seed)
        for spec in part_specs(build_all_splits(sim_cfg)["id"], "val")
    }
    for spec in first:
        key = (spec.vessel, spec.ss, spec.heading_deg, spec.speed_kn, spec.realization_seed)
        assert key in val_keys


def test_the_two_policies_fly_the_same_episodes(sim_cfg):
    """``hover`` and ``random`` get identical realizations, offsets and initial states.

    That makes the two rows of a sea state a paired comparison: the only difference between
    them is the action stream. A draw keyed on the policy would make them two independent
    samples and any difference would carry the draw's noise as well as the policy's effect.
    """
    hover = draw_episodes(sim_cfg, "hover", "SS3", "aft", 20, DEFAULT_DRAW_SEED)
    random = draw_episodes(sim_cfg, "random", "SS3", "aft", 20, DEFAULT_DRAW_SEED)
    assert [spec.realization for spec in hover] == [spec.realization for spec in random]
    assert [spec.episode_seed for spec in hover] == [spec.episode_seed for spec in random]


@pytest.mark.pybullet
@pytest.mark.slow
def test_small_sweep_produces_well_formed_rows(sweep_cfgs):
    """A miniature sweep exercises the whole path and its reductions.

    The two properties asserted are the ones a Gate 2 reader relies on: the six outcome
    fractions cover every episode exactly once, and the per-episode table has exactly as
    many rows as were drawn -- so a cell's success rate cannot be a fraction of a different
    denominator than the one printed beside it.
    """
    cells, episodes = run_sweep(sweep_cfgs, sea_states=("SS3",), n_episodes=6, workers=1, chunk=6)
    assert len(cells) == len(POLICIES)
    assert len(episodes) == len(POLICIES) * 6
    for row in cells:
        fractions = [getattr(row, f"frac_{outcome}") for outcome in OUTCOMES]
        assert sum(fractions) == pytest.approx(1.0)
        assert row.n_episodes == 6
        assert row.regime == "id"
        assert row.partition == "val"
        assert row.v_max_m_s == sweep_cfgs.landing.v_max_m_s
        assert row.driver == sweep_cfgs.landing.platform.driver
        assert row.episode_draw_seed == DEFAULT_DRAW_SEED
    for row in episodes:
        assert row.outcome in OUTCOMES
        assert row.termination_reason != "None"
        assert row.steps > 0
        assert np.isfinite(row.t0_model_s)


@pytest.mark.pybullet
@pytest.mark.slow
def test_sweep_output_does_not_depend_on_worker_count(sweep_cfgs):
    """Two workers give the same rows as one, in the same order.

    Work splits over chunks of episodes and the result is re-sorted into draw order, so a
    re-run at a different ``--workers`` is byte-identical. Without that, the committed CSV
    would silently encode the machine it was produced on.
    """
    serial = run_sweep(
        sweep_cfgs, policies=("hover",), sea_states=("SS3",), n_episodes=4, workers=1, chunk=2
    )[1]
    parallel = run_sweep(
        sweep_cfgs, policies=("hover",), sea_states=("SS3",), n_episodes=4, workers=2, chunk=2
    )[1]
    assert [row.index for row in serial] == [row.index for row in parallel]
    assert [row.outcome for row in serial] == [row.outcome for row in parallel]
    assert [row.t0_model_s for row in serial] == [row.t0_model_s for row in parallel]
    assert [row.steps for row in serial] == [row.steps for row in parallel]


def test_committed_e00_artifacts_have_expected_shape():
    """The committed CSVs, if present, have the shape Gate 2 reads.

    Skipped rather than failed when absent, so a fresh clone can run the suite before
    ``make env-sanity``. When present, the two Gate 2 numbers are checked directly: the six
    outcome fractions sum to 1 in every row, and the analytic-versus-contact disagreement
    rate is below 1 % everywhere.
    """
    if not CELL_CSV.exists() or not EPISODE_CSV.exists():
        pytest.skip("results/e00_env_sanity*.csv not generated; run `make env-sanity`")
    with CELL_CSV.open() as handle:
        cells = list(csv.DictReader(handle))
    with EPISODE_CSV.open() as handle:
        episodes = list(csv.DictReader(handle))
    assert len(cells) == 4
    assert len(episodes) == sum(int(row["n_episodes"]) for row in cells)
    for row in cells:
        total = sum(float(row[f"frac_{outcome}"]) for outcome in OUTCOMES)
        assert total == pytest.approx(1.0)
        assert float(row["touchdown_disagreement_rate"]) < 0.01
        assert row["regime"] == "id" and row["partition"] == "val"
    policies = {row["policy"] for row in cells}
    assert policies == set(POLICIES)
    # The pre-registered expectation, checked rather than assumed: a policy that does
    # nothing runs out the clock, and a policy that does anything crashes. A random policy
    # that "succeeds" would mean the criteria or the contact logic are wrong.
    for row in cells:
        if row["policy"] == "hover":
            assert float(row["frac_timeout"]) > 0.9
        if row["policy"] == "random":
            assert float(row["frac_success"]) == 0.0
