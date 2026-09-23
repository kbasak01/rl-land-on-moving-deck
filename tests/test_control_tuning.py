"""``scripts/tune_controller.py --final``: the pinned Phase 3 controller set.

``--final`` re-runs controllers at their committed configs on the P3-D3 tune draw and writes
``results/e01/tune_pool_final.csv``. It used to iterate the whole registry; since Phase 4 the
registry also holds ``gated_forecast*``, which need a ship-motion feed that the tuning loop
does not build, so the loop would raise before the CSV was written. The set is now an
explicit tuple. These tests pin that tuple, tie it to the committed CSV's rows, and check
that the script iterates the tuple and not the registry. No tuning is run here.
"""

import ast
import csv
from pathlib import Path

from rld.control.registry import REGISTRY, entry
from rld.control.tuning import FINAL_CONTROLLERS

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tune_controller.py"
FINAL_CSV = REPO_ROOT / "results" / "e01" / "tune_pool_final.csv"

PHASE3 = ("pid_track_descend", "pid_feedforward", "pid_feedforward_lowvz", "gated", "oracle_gated")


def test_final_set_is_exactly_the_five_phase3_controllers() -> None:
    """The tuple, in order; every name is registered and none needs the feed."""
    assert FINAL_CONTROLLERS == PHASE3
    for name in FINAL_CONTROLLERS:
        assert name in REGISTRY
        assert entry(name).needs_motion_feed is False
    # The feed-needing entries exist and are deliberately outside the set.
    assert {n for n, e in REGISTRY.items() if e.needs_motion_feed}.isdisjoint(FINAL_CONTROLLERS)


def test_final_set_matches_the_committed_csv_rows() -> None:
    """The committed tune_pool_final.csv has exactly these controllers, in this order."""
    with FINAL_CSV.open(newline="", encoding="utf-8") as handle:
        rows = [line for line in handle if not line.startswith("#")]
    names = [row["controller"] for row in csv.DictReader(rows)]
    assert tuple(names) == FINAL_CONTROLLERS


def test_script_iterates_the_pinned_tuple_not_the_registry() -> None:
    """No ``for ... in REGISTRY`` in the script; its loop is over ``FINAL_CONTROLLERS``."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    loop_targets = [
        node.iter.id
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name)
    ]
    assert "FINAL_CONTROLLERS" in loop_targets
    assert "REGISTRY" not in loop_targets
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "REGISTRY" not in names
