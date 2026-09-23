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


def test_carry_rows_copies_only_this_runs_pads_and_episodes(tmp_path: Path) -> None:
    """Carried (not re-flown) rows: right methods, pads, episodes; not-run cells rescaled."""
    from rld.eval.episodes import ListedEpisode
    from rld.eval.report import SkipRecord, summarise, summary_columns, write_rows
    from rld.eval.runner import EPISODE_COLUMNS

    script = _load()

    def row(method: str, pad: str, index: int) -> dict[str, object]:
        base: dict[str, object] = dict.fromkeys(EPISODE_COLUMNS, 0)
        base.update(
            method=method,
            privileged=False,
            run_seed=0,
            pad=pad,
            regime="id",
            ss="SS3",
            index=index,
            outcome="success",
            termination_reason="dwell_complete",
            touchdown_contact=False,
            td_in_quiescent_window=float("nan"),
            in_training_distribution=True,
            detectors_disagree=False,
            tunnelled=False,
        )
        return base

    rows = [
        row(m, p, i)
        for m in ("gated", "gated_forecast_tcn")
        for p in ("aft", "cg")
        for i in (0, 1, 2)
    ]
    skips = [
        SkipRecord("gated_forecast_tcn", False, 0, p, "static", "static", "no ship")
        for p in ("aft", "cg")
    ] * 200
    write_rows(tmp_path / "episodes.csv", rows, EPISODE_COLUMNS)
    summary = summarise(
        rows,
        ["gated", "gated_forecast_tcn"],
        {m: dict.fromkeys(script.PROVENANCE_COLUMNS, m) for m in ("gated", "gated_forecast_tcn")},
        {"v": "1"},
        skips,
    )
    write_rows(
        tmp_path / "summary.csv", summary, summary_columns([*script.PROVENANCE_COLUMNS, "v"])
    )

    def listed(regime: str, ss: str, index: int) -> ListedEpisode:
        fields = dict.fromkeys(ListedEpisode.__dataclass_fields__, 0)
        fields.update(regime=regime, ss=ss, index=index, pad="aft", vessel="frigate")
        return ListedEpisode(**fields)  # type: ignore[arg-type]

    episodes = [listed("id", "SS3", 0), listed("id", "SS3", 2), listed("static", "static", 0)]
    got, methods, prov, carried_skips = script.carry_rows(
        tmp_path, None, ["gated_forecast_tcn_seed0"], {"aft"}, episodes, {"v": "1"}
    )
    assert methods == ["gated", "gated_forecast_tcn"]
    assert {(r["method"], r["pad"], r["index"]) for r in got} == {
        (m, "aft", i) for m in methods for i in ("0", "2")
    }
    assert prov["gated"] == dict.fromkeys(script.PROVENANCE_COLUMNS, "gated")
    assert len(carried_skips) == 1 and carried_skips[0].pad == "aft"  # one listed static here
    with pytest.raises(SystemExit, match="live"):
        script.carry_rows(tmp_path, None, [], {"aft"}, episodes, {"v": "2"})
