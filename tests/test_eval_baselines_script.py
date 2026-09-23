"""``scripts/eval_baselines.py``: the Phase 3 default, the pad arms, the scratch-only guard.

Nothing is flown: every refusal happens before the episode lists are read.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from rld.config import REPO_ROOT
from rld.control.registry import REGISTRY

SCRIPT = REPO_ROOT / "scripts" / "eval_baselines.py"
GATED_YAML = REPO_ROOT / "configs" / "control" / "gated.yaml"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_baselines_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_controllers_are_the_five_phase3_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    """``make baselines`` must keep reproducing e01 whatever the registry grows."""
    script = _load()
    assert script.PHASE3_CONTROLLERS == (
        "pid_track_descend",
        "pid_feedforward",
        "pid_feedforward_lowvz",
        "gated",
        "oracle_gated",
    )
    assert set(script.PHASE3_CONTROLLERS) < set(REGISTRY)
    assert not any(REGISTRY[n].needs_motion_feed for n in script.PHASE3_CONTROLLERS)
    monkeypatch.setattr(sys, "argv", ["eval_baselines.py"])
    args = script.parse_args()
    assert args.controllers == list(script.PHASE3_CONTROLLERS)
    assert args.pad == "aft"
    assert args.out_dir == REPO_ROOT / "results" / "e01"
    assert script.PAD_ARMS == {"aft": (None,), "cg": ("cg",), "both": (None, "cg")}
    assert script.title_for(Path("results/e01"), None) == script.TITLES["e01"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--per-cell", "3"],
        ["--lists", "id"],
        ["--config-override", f"gated={GATED_YAML}"],
    ],
)
def test_scratch_options_are_refused_under_results(
    extra: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load()
    monkeypatch.setattr(
        sys, "argv", ["eval_baselines.py", "--out-dir", str(REPO_ROOT / "results" / "eX"), *extra]
    )
    with pytest.raises(SystemExit, match="scratch-only"):
        script.main()
    assert not (REPO_ROOT / "results" / "eX").exists()
