"""The MSS transfer arm's episode lists (P7-D1 §6): ``results/episodes_mss/``.

What these tests hold the committed MSS lists to:

* their own directory and their own ``MANIFEST.csv``: generator seed 20261001,
  ``balanced_round_robin``, pad ``aft``, N = 200 per list, file and content SHA-256 matching
  the committed files; the frozen ``results/episodes/`` manifest is untouched;
* every row is an MSS realization of the arm's 18 per grid kind: ``ss =
  rld.deck.mss.mss_ss_label(grid_kind)``, vessel ``s175``, ``in_training_distribution``
  false; balanced round-robin over the 18 (counts 11 or 12);
* **no key collides** with any frozen-list key, with any dev-pool key (P3-D2), or with the
  other grid kind's keys;
* a regeneration of the first ``k`` rows of each list reproduces the committed rows (episode
  ``i`` depends only on its own index);
* ``scripts/make_episodes.py --mss --check`` regenerates both lists and matches both hashes;
* the runner's **strict** start check passes on 2 episodes per list (listed ``t0``,
  initial position, realization key and pad), flown under each list's motion kind; and an
  independent reset through the conftest factory reproduces ``t0`` and the initial state.

Tests that need the MSS clone (``artifacts/mss/upstream``) skip without it (``make
mss-export``). Everything is simulation; these are another simulator's trajectories, not
measurements of a real ship.

Units: ``t0_model_s`` seconds model scale, ``init_*_m`` metres model scale, world frame.
"""

import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from dmf.config import SimConfig

from rld.deck import mss as deck_mss
from rld.deck.splits import dev_pool, key_for_spec
from rld.envs.landing_env import DeckLandingAviary
from rld.eval import arms
from rld.eval.episodes import (
    DRAW_RULE,
    EPISODES_DIR,
    EPISODES_MSS_DIR,
    EPISODES_PER_CELL,
    MANIFEST_NAME,
    MSS_GENERATOR_SEED,
    MSS_REGIME_GRID,
    MSS_VESSEL,
    ListedEpisode,
    content_sha256,
    draw_mss_cell,
    file_sha256,
    generate_mss_lists,
    list_names,
    motion_kind_for,
    mss_candidates,
    mss_list_names,
    mss_ss,
    read_list,
    read_manifest,
)
from rld.eval.learned import verify_lists

REPO = Path(__file__).resolve().parents[1]

#: The frozen manifest's SHA-256 (P3-D1 §2); the MSS lists must never change it.
FROZEN_MANIFEST_SHA256 = "e6f30e55e478d39061b94e38de33959ca99b16e8cac38a3bd99b34f6543ad4a2"


def _clone_present() -> bool:
    return (deck_mss.MSS_UPSTREAM_DIR / "HYDRO" / "vessels_shipx" / "s175" / "s175.mat").is_file()


@pytest.fixture(scope="module")
def committed() -> dict[str, list[ListedEpisode]]:
    """Every committed MSS list, read from ``results/episodes_mss/``."""
    if not (EPISODES_MSS_DIR / MANIFEST_NAME).is_file():
        pytest.skip(f"no MSS lists at {EPISODES_MSS_DIR} (scripts/make_episodes.py --mss)")
    return {name: read_list(EPISODES_MSS_DIR / f"{name}.parquet") for name in mss_list_names()}


@pytest.fixture(scope="module")
def clone() -> None:
    if not _clone_present():
        pytest.skip(f"MSS clone absent at {deck_mss.MSS_UPSTREAM_DIR} (run `make mss-export`)")


# --------------------------------------------------------------------------- fast


def test_mss_constants_agree_with_the_arm_and_the_deck_side() -> None:
    assert MSS_GENERATOR_SEED == 20261001
    assert mss_list_names() == ("mss_transfer", "mss_transfer_corpus")
    assert tuple(arms.MSS_REGIME_MOTION) == mss_list_names()
    assert MSS_VESSEL == deck_mss.MSS_VESSEL == "s175"
    assert EPISODES_MSS_DIR == arms.EPISODES_MSS_DIR != EPISODES_DIR
    for regime, grid in MSS_REGIME_GRID.items():
        assert mss_ss(regime) == deck_mss.mss_ss_label(grid) == f"SS5/mss:{grid}"
        assert motion_kind_for(regime) == arms.MSS_REGIME_MOTION[regime]
    for regime in list_names():
        assert motion_kind_for(regime) == "jonswap"
    with pytest.raises(ValueError):
        mss_ss("id")


def test_candidates_are_the_arms_18_realizations() -> None:
    cands = mss_candidates()
    assert len(cands) == 18 == len(set(cands))
    assert cands == sorted(cands)
    assert {h for h, _, _ in cands} == {135.0, 180.0}
    assert {u for _, u, _ in cands} == {0.0, 6.0, 12.0}
    assert {s for _, _, s in cands} == {0, 1, 2}


def test_mss_draws_are_a_pure_function_of_the_index() -> None:
    full = draw_mss_cell("mss_transfer", MSS_GENERATOR_SEED, range(40))
    assert draw_mss_cell("mss_transfer", MSS_GENERATOR_SEED, [23]) == [full[23]]
    assert all(d.vessel == "s175" and not d.in_training_distribution for d in full)
    other = draw_mss_cell("mss_transfer", MSS_GENERATOR_SEED + 1, range(40))
    assert [d.episode_seed for d in other] != [d.episode_seed for d in full]


def test_manifest_matches_the_committed_files(committed: dict[str, list[ListedEpisode]]) -> None:
    manifest = read_manifest(EPISODES_MSS_DIR / MANIFEST_NAME)
    assert [row["file"] for row in manifest] == [f"{n}.parquet" for n in mss_list_names()]
    for row in manifest:
        name = row["regime"]
        assert row["file_sha256"] == file_sha256(EPISODES_MSS_DIR / row["file"])
        assert row["content_sha256"] == content_sha256(committed[name])
        assert int(row["rows"]) == int(row["n_per_cell"]) == EPISODES_PER_CELL == 200
        assert int(row["n_cells"]) == 1 and row["cells"] == mss_ss(name)
        assert int(row["generator_seed"]) == MSS_GENERATOR_SEED
        assert row["draw"] == DRAW_RULE == "balanced_round_robin"
        assert row["pad"] == "aft" and float(row["frac_in_training_distribution"]) == 0.0
    assert len(verify_lists(EPISODES_MSS_DIR)) == 2  # the runner's pre-flight check
    # The frozen lists and their manifest are untouched by the MSS mode.
    assert file_sha256(EPISODES_DIR / MANIFEST_NAME) == FROZEN_MANIFEST_SHA256


def test_rows_are_balanced_mss_realizations(committed: dict[str, list[ListedEpisode]]) -> None:
    cands = mss_candidates()
    for regime, rows in committed.items():
        ss = mss_ss(regime)
        assert [r.index for r in rows] == list(range(200))
        assert {(r.regime, r.ss, r.vessel, r.pad) for r in rows} == {(regime, ss, "s175", "aft")}
        assert not any(r.in_training_distribution for r in rows)
        assert {(r.heading_deg, r.speed_kn, r.realization_seed) for r in rows} == set(cands)
        counts = Counter(r.key for r in rows)
        assert sorted(Counter(counts.values()).items()) == [(11, 16), (12, 2)]
        for r in rows:
            grid = MSS_REGIME_GRID[regime]
            assert r.key == deck_mss.mss_key(grid, r.heading_deg, r.speed_kn, r.realization_seed)
            assert deck_mss.parse_mss_ss_label(r.ss) == ("SS5", grid)
            assert r.key == rows[r.index % 18].key  # round-robin over one shuffled order
        assert len({(r.key, r.episode_seed) for r in rows}) == 200
        lo, hi = 4.0, 120.0 - 12.5  # the JONSWAP source's model window at lambda = 1/25
        assert all(lo <= r.t0_model_s <= hi for r in rows)


def test_no_key_collides_with_a_frozen_or_dev_pool_key(
    committed: dict[str, list[ListedEpisode]], sim_cfg: SimConfig
) -> None:
    frozen = {r.key for name in list_names() for r in read_list(EPISODES_DIR / f"{name}.parquet")}
    train, tune = dev_pool(sim_cfg)
    dev = {key_for_spec(s) for s in (*train, *tune)}
    assert len(dev) == 729 + 135 and len(frozen) > 1000
    keys = {name: {r.key for r in rows} for name, rows in committed.items()}
    for name, mine in keys.items():
        assert len(mine) == 18
        assert not mine & frozen, name
        assert not mine & dev, name
        assert all("/mss:" in k[0] for k in mine)  # no dmf corpus sea-state label has a "/"
    assert not {k[0] for k in (*frozen, *dev)} & {k[0] for m in keys.values() for k in m}
    assert not keys["mss_transfer"] & keys["mss_transfer_corpus"]


# --------------------------------------------------------------------------- slow


@pytest.mark.pybullet
@pytest.mark.slow
def test_regenerating_a_prefix_reproduces_the_committed_rows(
    committed: dict[str, list[ListedEpisode]], clone: None
) -> None:
    k = 3
    fresh = generate_mss_lists(MSS_GENERATOR_SEED, n=k, workers=1)
    for regime, rows in committed.items():
        assert fresh[regime] == rows[:k], regime
        assert content_sha256(fresh[regime]) == content_sha256(rows[:k])


@pytest.mark.pybullet
@pytest.mark.slow
def test_make_episodes_mss_check_passes(
    committed: dict[str, list[ListedEpisode]], clone: None
) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_episodes.py"), "--mss", "--check"]
        + ["--workers", "4"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [line for line in proc.stdout.splitlines() if ".parquet" in line]
    assert len(lines) == 2 and all("content=OK file=OK" in line for line in lines)


@pytest.mark.pybullet
@pytest.mark.slow
def test_strict_start_check_passes_on_two_episodes_per_list(
    committed: dict[str, list[ListedEpisode]], clone: None
) -> None:
    from rld.eval.envs import load_eval_configs
    from rld.eval.runner import controller_spec, run_list

    cfgs = load_eval_configs()
    spec = controller_spec("pid_feedforward")
    for regime, rows in committed.items():
        episodes = [rows[0], rows[-1]]
        out = run_list(episodes, spec, cfgs, workers=1, motion_kind=motion_kind_for(regime))
        for row, ep in zip(out, episodes, strict=True):  # strict check raised if not listed
            assert row["t0_model_s"] == ep.t0_model_s
            assert (row["init_x_m"], row["init_y_m"], row["init_z_m"]) == ep.init_xyz_m
            assert (row["regime"], row["ss"], row["pad"]) == (regime, ep.ss, "aft")


@pytest.mark.pybullet
@pytest.mark.slow
def test_independent_reset_reproduces_t0_and_initial_state(
    committed: dict[str, list[ListedEpisode]],
    clone: None,
    landing_env: Callable[..., DeckLandingAviary],
) -> None:
    from rld.eval.envs import load_eval_configs

    cfgs = load_eval_configs()
    for regime, rows in committed.items():
        grid = MSS_REGIME_GRID[regime]
        for row in (rows[1], rows[-2]):
            motion = deck_mss.mss_motion(
                cfgs, grid, row.heading_deg, row.speed_kn, row.realization_seed
            )
            assert motion.key == row.key
            env = landing_env(motion, pad=row.pad, episode_seed=row.episode_seed + 1)
            env.reset(seed=row.episode_seed)
            assert env.record.t0_model_s == row.t0_model_s, row
            position = np.asarray(env.pos[0], dtype=np.float64)
            assert tuple(float(v) for v in position) == row.init_xyz_m, row
