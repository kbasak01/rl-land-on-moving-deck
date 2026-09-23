"""Phase 5: no training or tuning path touches a frozen-list realization or file.

* The P3-D2 train pool, the tune pool and the committed frozen lists
  (``results/episodes/*.parquet``) share no realization key (CLAUDE.md non-negotiable 2).
* The training pool a run actually uses and its fixed evaluation draw come from the right
  pools.
* No module or script of the RL path reads ``results/episodes/`` -- checked on the source
  text and on the import graph. (Reading the lists *here*, in a test, is the point.)

Units: none (realization keys).
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from dmf.config import load_sim

from _rl_helpers import tiny_config
from conftest import REPO_ROOT
from rld.deck.config import MOTION_JONSWAP_CONFIG, load_motion
from rld.deck.splits import dev_pool, key_for_spec
from rld.eval.episodes import read_list
from rld.rl.train import build_env_configs, eval_episodes, training_pools

EPISODES_DIR = REPO_ROOT / "results" / "episodes"

RL_SOURCES = [
    *sorted((REPO_ROOT / "src" / "rld" / "rl").glob("*.py")),
    *(
        REPO_ROOT / "scripts" / f"{name}.py"
        for name in ("train", "sweep", "tune", "plot_learning_curves")
    ),
]


@pytest.fixture(scope="module")
def frozen_keys() -> set[tuple[object, ...]]:
    files = sorted(EPISODES_DIR.glob("*.parquet"))
    assert len(files) == 5, files
    keys: set[tuple[object, ...]] = set()
    for path in files:
        keys |= {row.key for row in read_list(path) if row.vessel != "static"}
    assert keys
    return keys


@pytest.fixture(scope="module")
def pools() -> tuple[set[tuple[object, ...]], set[tuple[object, ...]]]:
    sim = load_sim(load_motion(MOTION_JONSWAP_CONFIG).sim_config_path)
    train, tune = dev_pool(sim)
    return {key_for_spec(s) for s in train}, {key_for_spec(s) for s in tune}


def test_pools_disjoint_from_frozen_lists_and_each_other(
    frozen_keys: set[tuple[object, ...]],
    pools: tuple[set[tuple[object, ...]], set[tuple[object, ...]]],
) -> None:
    train, tune = pools
    assert len(train) == 729 and len(tune) == 135
    assert not train & frozen_keys
    assert not tune & frozen_keys
    assert not train & tune
    assert all(k[0] != "SS6" for k in train | tune)


def test_run_pools_and_eval_draw(
    frozen_keys: set[tuple[object, ...]],
    pools: tuple[set[tuple[object, ...]], set[tuple[object, ...]]],
) -> None:
    cfg = tiny_config("ppo", **{"eval__episodes_per_ss": None})
    cfgs = build_env_configs(cfg)
    train, tune = training_pools(cfg, cfgs)
    assert {key_for_spec(s) for s in train} <= pools[0]
    assert {key_for_spec(s) for s in tune} == pools[1]
    episodes = eval_episodes(cfg, cfgs)
    assert len(episodes) == 180  # the P3-D3 draw: 60 per sea state
    eval_keys = {key_for_spec(e.realization) for e in episodes}
    assert eval_keys <= pools[1]
    assert not eval_keys & frozen_keys


def _code_tokens(path: Path) -> list[str]:
    """Return every non-docstring string constant, imported module/name and identifier."""
    tree = ast.parse(path.read_text())
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                out.append(node.value)
        elif isinstance(node, ast.ImportFrom):
            out += [node.module or "", *(alias.name for alias in node.names)]
        elif isinstance(node, ast.Import):
            out += [alias.name for alias in node.names]
        elif isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
    return out


def test_rl_code_never_references_the_frozen_lists() -> None:
    needles = ("results/episodes", "episodes/", "read_list", "rld.eval.episodes", ".parquet")
    for path in RL_SOURCES:
        for token in _code_tokens(path):
            for needle in needles:
                assert needle not in token, f"{path.name}: {token!r} contains {needle!r}"


def test_rl_import_graph_excludes_episode_lists() -> None:
    code = (
        "import sys; import rld.rl.train, rld.rl.tuning, rld.rl.scheduler, rld.rl.curves; "
        "print('rld.eval.episodes' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_frozen_list_dir_is_what_the_scan_expects() -> None:
    assert Path(EPISODES_DIR / "MANIFEST.csv").exists()
