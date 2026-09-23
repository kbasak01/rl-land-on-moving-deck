"""The committed evaluation episode lists (P3-D1): provenance, isolation, size, hashes, replay.

What these tests hold the committed ``results/episodes/`` files to:

* every realization key is in **that regime's** dmf test partition;
* no key is in the development pool (``dev_pool`` train or tune, P3-D2);
* N = 200 per (regime, sea state) cell, 13 cells plus the 200-episode static list;
* the draw is balanced round-robin: coverage is ``min(R, N)`` realizations per cell, the
  per-realization count spread is at most 1, and episode ``i`` flies the same realization
  as episode ``i mod R``;
* ``in_training_distribution`` is exactly the (vessel, ss, heading) dev-pool membership;
* a regeneration of a prefix of every cell reproduces the committed rows and their content
  hash (episode ``i`` depends only on its own index);
* ``MANIFEST.csv``'s file and content hashes match the committed files;
* a fresh environment reset with a listed episode seed reproduces the listed ``t0`` and
  initial position **exactly** -- built through the conftest factory, not through
  :mod:`rld.eval`, so the check does not share code with the generator it checks.

Units: ``t0_model_s`` seconds model scale, ``init_*_m`` metres model scale, world frame.
"""

from collections.abc import Callable

import numpy as np
import pytest
from dmf.config import SimConfig
from dmf.sim.generate import RealizationSpec

from rld.deck.splits import build_all_splits, dev_pool, key_for_spec, part_specs
from rld.envs.landing_env import DeckLandingAviary
from rld.envs.platform import StaticDeckMotion
from rld.eval.envs import STATIC_LABEL
from rld.eval.episodes import (
    DEFAULT_GENERATOR_SEED,
    DRAW_RULE,
    EPISODES_DIR,
    EPISODES_PER_CELL,
    LIST_COLUMNS,
    MANIFEST_NAME,
    REGIME_CELLS,
    ListedEpisode,
    canonical_csv,
    cell_order,
    content_sha256,
    draw_cell,
    file_sha256,
    generate_lists,
    list_names,
    read_list,
    read_manifest,
)

DMF_REGIMES = tuple(name for name in list_names() if name != STATIC_LABEL)


@pytest.fixture(scope="module")
def committed() -> dict[str, list[ListedEpisode]]:
    """Return every committed list, read from ``results/episodes/``."""
    return {name: read_list(EPISODES_DIR / f"{name}.parquet") for name in list_names()}


def test_thirteen_cells_of_200_plus_static(committed: dict[str, list[ListedEpisode]]) -> None:
    n_cells = 0
    for regime, sea_states in REGIME_CELLS:
        rows = committed[regime]
        assert [row.ss for row in rows] == [ss for ss in sea_states for _ in range(200)]
        for ss in sea_states:
            indices = [row.index for row in rows if row.ss == ss]
            assert indices == list(range(EPISODES_PER_CELL))
        if regime != STATIC_LABEL:
            n_cells += len(sea_states)
        assert {row.regime for row in rows} == {regime}
    assert n_cells == 13
    assert len(committed[STATIC_LABEL]) == 200
    assert EPISODES_PER_CELL == 200


def test_every_key_is_in_its_regimes_test_partition(
    committed: dict[str, list[ListedEpisode]], sim_cfg: SimConfig
) -> None:
    splits = build_all_splits(sim_cfg)
    for regime in DMF_REGIMES:
        test_keys = splits[regime].test_keys
        missing = [row.key for row in committed[regime] if row.key not in test_keys]
        assert not missing, f"{regime}: {len(missing)} keys outside its test partition"


def test_draw_is_balanced_round_robin(
    committed: dict[str, list[ListedEpisode]], sim_cfg: SimConfig
) -> None:
    splits = build_all_splits(sim_cfg)
    for regime, sea_states in REGIME_CELLS:
        for ss in sea_states:
            cell = [row for row in committed[regime] if row.ss == ss]
            n = len(cell)
            if regime == STATIC_LABEL:
                n_real = 1
                candidates = None
            else:
                candidates = [s for s in part_specs(splits[regime], "test") if s.sea_state == ss]
                n_real = len(candidates)
            counts: dict[object, int] = {}
            for row in cell:
                counts[row.key] = counts.get(row.key, 0) + 1
            assert len(counts) == min(n_real, n), (regime, ss, len(counts), n_real)
            assert max(counts.values()) - min(counts.values()) <= 1, (regime, ss)
            # Round-robin over one fixed shuffled order.
            for row in cell:
                assert row.key == cell[row.index % n_real].key, (regime, ss, row.index)
            if candidates is not None:
                order = cell_order(DEFAULT_GENERATOR_SEED, regime, ss, n_real)
                expected = [key_for_spec(candidates[order[i % n_real]]) for i in range(n)]
                assert [row.key for row in cell] == expected, (regime, ss)
                # The extra episodes go to the first N mod R realizations in shuffled order.
                if n_real <= n:
                    heavy = {key_for_spec(candidates[j]) for j in order[: n % n_real]}
                    assert {k for k, c in counts.items() if c == -(-n // n_real)} == (
                        heavy if n % n_real else set(counts)
                    )


def test_lists_are_disjoint_from_the_development_pool(
    committed: dict[str, list[ListedEpisode]], sim_cfg: SimConfig
) -> None:
    train, tune = dev_pool(sim_cfg)
    dev_keys = {key_for_spec(spec) for spec in (*train, *tune)}
    assert len(dev_keys) == 729 + 135
    for regime in DMF_REGIMES:
        leaked = {row.key for row in committed[regime]} & dev_keys
        assert not leaked, f"{regime}: {sorted(leaked)[:3]} are dev-pool realizations"


def test_in_training_distribution_is_dev_pool_cell_membership(
    committed: dict[str, list[ListedEpisode]], sim_cfg: SimConfig
) -> None:
    train, tune = dev_pool(sim_cfg)
    cells3 = {(s.vessel, s.sea_state, float(s.heading_deg)) for s in (*train, *tune)}
    for regime in list_names():
        for row in committed[regime]:
            expected = (row.vessel, row.ss, row.heading_deg) in cells3
            assert row.in_training_distribution is expected, row
    # And the named consequences of P3-D2.
    ids = committed["id"]
    assert not any(r.in_training_distribution for r in ids if r.ss == "SS6")
    assert not any(r.in_training_distribution for r in ids if r.heading_deg == 90.0)
    assert all(r.in_training_distribution for r in ids if r.ss != "SS6" and r.heading_deg != 90.0)
    for regime in ("unseen_seastate", "unseen_heading", "unseen_vessel", STATIC_LABEL):
        assert not any(r.in_training_distribution for r in committed[regime])


def test_no_episode_is_listed_twice(committed: dict[str, list[ListedEpisode]]) -> None:
    for regime, rows in committed.items():
        pairs = [(row.key, row.episode_seed) for row in rows]
        assert len(set(pairs)) == len(pairs), regime


def test_manifest_matches_the_committed_files(
    committed: dict[str, list[ListedEpisode]],
) -> None:
    manifest = read_manifest(EPISODES_DIR / MANIFEST_NAME)
    assert [row["file"] for row in manifest] == [f"{n}.parquet" for n in list_names()]
    for row in manifest:
        name = row["regime"]
        path = EPISODES_DIR / row["file"]
        assert row["file_sha256"] == file_sha256(path)
        assert row["content_sha256"] == content_sha256(committed[name])
        assert int(row["rows"]) == len(committed[name])
        assert int(row["generator_seed"]) == DEFAULT_GENERATOR_SEED
        assert int(row["n_per_cell"]) == EPISODES_PER_CELL
        assert row["draw"] == DRAW_RULE == "balanced_round_robin"


def test_canonical_csv_is_fixed_format() -> None:
    row = ListedEpisode(
        regime="id",
        ss="SS3",
        index=0,
        pad="aft",
        vessel="frigate",
        heading_deg=180.0,
        speed_kn=6.0,
        realization_seed=33,
        episode_seed=7,
        t0_model_s=0.1,
        init_x_m=-0.0,
        init_y_m=1e-300,
        init_z_m=1.7,
        in_training_distribution=True,
    )
    text = canonical_csv([row])
    assert text == (
        ",".join(LIST_COLUMNS) + "\n"
        "id,SS3,0,aft,frigate,180,6,33,7,0.10000000000000001,-0,1e-300,"
        "1.7,true\n"
    )
    # %.17g round-trips float64 exactly.
    assert float(text.splitlines()[1].split(",")[9]) == 0.1


def test_draws_are_a_pure_function_of_the_index(sim_cfg: SimConfig) -> None:
    full = draw_cell(sim_cfg, "id", "SS5", DEFAULT_GENERATOR_SEED, range(12))
    alone = draw_cell(sim_cfg, "id", "SS5", DEFAULT_GENERATOR_SEED, [7])
    assert alone == [full[7]]
    other = draw_cell(sim_cfg, "id", "SS5", DEFAULT_GENERATOR_SEED + 1, range(12))
    assert [d.episode_seed for d in other] != [d.episode_seed for d in full]


@pytest.mark.pybullet
@pytest.mark.slow
def test_regenerating_a_prefix_reproduces_the_committed_rows(
    committed: dict[str, list[ListedEpisode]], sim_cfg: SimConfig
) -> None:
    k = 2
    fresh = generate_lists(sim_cfg, DEFAULT_GENERATOR_SEED, n=k, workers=1)
    for regime, sea_states in REGIME_CELLS:
        prefix: list[ListedEpisode] = []
        for ss in sea_states:
            prefix += [row for row in committed[regime] if row.ss == ss and row.index < k]
        assert fresh[regime] == prefix, regime
        assert content_sha256(fresh[regime]) == content_sha256(prefix)


@pytest.mark.pybullet
@pytest.mark.slow
def test_env_reset_reproduces_listed_t0_and_initial_state_exactly(
    committed: dict[str, list[ListedEpisode]],
    deck_source: Callable[[RealizationSpec], object],
    landing_env: Callable[..., DeckLandingAviary],
) -> None:
    # The first and last episode of every cell, through the conftest factory (which closes
    # every environment it built when the test ends). The constructor seed is deliberately
    # NOT the episode seed, so only the explicit reset seed can reproduce the row.
    probes = []
    for regime, sea_states in REGIME_CELLS:
        for ss in sea_states:
            cell = [row for row in committed[regime] if row.ss == ss]
            probes += [cell[0], cell[-1]]
    for row in probes:
        if row.vessel == STATIC_LABEL:
            motion: object = StaticDeckMotion()
        else:
            motion = deck_source(
                RealizationSpec(
                    sea_state=row.ss,
                    heading_deg=row.heading_deg,
                    speed_kn=row.speed_kn,
                    vessel=row.vessel,
                    seed=row.realization_seed,
                )
            )
        env = landing_env(motion, pad=row.pad, episode_seed=row.episode_seed + 1)
        env.reset(seed=row.episode_seed)
        assert env.record.t0_model_s == row.t0_model_s, row
        position = np.asarray(env.pos[0], dtype=np.float64)
        assert tuple(float(v) for v in position) == row.init_xyz_m, row
