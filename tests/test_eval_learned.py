"""Learned policies through the frozen-list runner: seed labels, spread, aggregates, carry, render.

The end-to-end test trains two tiny SAC runs (training seeds 0 and 3) and flies their
``final/`` checkpoints through :func:`rld.eval.learned.main` on a **temporary** list and
manifest built from dev-pool **train** realizations -- never from the committed evaluation
lists, so the suite cannot put a policy in front of an evaluation episode. Two carried
"baseline" sources are made the same way (``pid_feedforward`` and the privileged
``oracle_gated`` flown on the temporary list), so the carry checks, the privileged label and
the byte-identical re-render are all exercised.

Units: rates dimensionless; speeds metres per second model scale; steps env control steps.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from _rl_helpers import tiny_config
from rld.eval.envs import EvalConfigs, load_eval_configs
from rld.eval.episodes import EpisodeDraw, ListedEpisode, resolve_draws, write_lists
from rld.eval.learned import (
    AGGREGATE_COLUMNS,
    CARRIED_PREFIX,
    aggregate_rows,
    evaluate_runs,
    inspect_run,
    live_provenance,
    main,
    order_runs,
    parse_learned,
    seed_spread,
)
from rld.eval.report import read_rows, summarise, summary_columns, write_rows
from rld.eval.runner import EPISODE_COLUMNS, controller_spec, run_list
from rld.eval.stats import iqm
from rld.rl.train import prepare_run_dir, train

# --------------------------------------------------------------------------- fast


def test_parse_learned() -> None:
    parsed = parse_learned(["ppo=a/0,a/1", "sac=b/4"])
    assert parsed == [("ppo", (Path("a/0"), Path("a/1"))), ("sac", (Path("b/4"),))]
    for bad in (["ppo"], ["ppo="], ["=a/0"], ["ppo=a", "ppo=b"], ["pid_feedforward=a"]):
        with pytest.raises(SystemExit):
            parse_learned(bad)


def _record(seed: int, success: float, p95: float, ss: str = "SS5") -> dict[str, Any]:
    return {
        "method": "m",
        "pad": "aft",
        "regime": "id",
        "ss": ss,
        "run_seed": seed,
        "n_episodes": 200,
        "n_success": round(200 * success),
        "success_rate": success,
        "rel_vz_normal_p95_m_s": p95,
    }


def test_seed_spread_and_aggregate_on_known_values() -> None:
    success = [1.0, 0.9, 0.8, 0.95, 0.85]
    p95 = [0.20, 0.30, 0.25, 0.22, 0.28]
    recs = [_record(s, v, p) for s, (v, p) in enumerate(zip(success, p95, strict=True))]
    recs += [_record(s, 0.5, math.nan if s == 2 else 0.4, "SS6") for s in range(5)]
    spread, columns = seed_spread(recs)
    assert [r["ss"] for r in spread] == ["SS5", "SS6"]
    ss5 = spread[0]
    assert ss5["seeds"] == "0|1|2|3|4" and ss5["n_seeds"] == 5
    assert [ss5[f"success_rate_seed{s}"] for s in range(5)] == success
    assert ss5["success_min"] == 0.8 and ss5["success_max"] == 1.0
    assert ss5["success_sd"] == pytest.approx(float(np.std(success, ddof=1)))
    assert ss5["p95_sd_m_s"] == pytest.approx(float(np.std(p95, ddof=1)))
    assert spread[1]["p95_n_seeds_finite"] == 4
    assert set(columns) == set(ss5)

    agg = aggregate_rows(recs)
    assert all(tuple(r) == AGGREGATE_COLUMNS for r in agg)
    by = {(r["ss"], r["metric"], r["statistic"]): r for r in agg}
    iqm5 = by[("SS5", "success_rate", "iqm")]
    # rliable IQM of 5 runs: the mean of the middle 3.
    assert iqm5["point"] == pytest.approx((0.85 + 0.9 + 0.95) / 3)
    assert iqm5["point"] == pytest.approx(iqm(np.array(success)[:, None]))
    assert 0.8 <= iqm5["ci_lo"] <= iqm5["point"] <= iqm5["ci_hi"] <= 1.0
    assert iqm5["reps"] == 2000 and iqm5["bootstrap_seed"] == 20260926
    gap = by[("SS5", "success_rate", "optimality_gap")]
    assert gap["point"] == pytest.approx(1.0 - float(np.mean(success)))
    assert by[("SS5", "rel_vz_normal_p95_m_s", "iqm")]["reported"] is True
    missing = by[("SS6", "rel_vz_normal_p95_m_s", "iqm")]
    assert missing["reported"] is False and math.isnan(missing["point"])
    assert "seed(s) 2" in missing["note"]
    # Same inputs, same numbers: every interval is reproducible from its seed.
    assert [repr(r) for r in aggregate_rows(recs)] == [repr(r) for r in agg]


# --------------------------------------------------------------------------- slow


@pytest.fixture(scope="module")
def cfgs() -> EvalConfigs:
    return load_eval_configs()


@pytest.fixture(scope="module")
def runs_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two tiny SAC runs, training seeds 0 and 3."""
    root = tmp_path_factory.mktemp("runs")
    cfg = tiny_config(
        "sac",
        total_steps=300,
        sac__learning_starts=100,
        sac__batch_size=32,
        eval__interval_steps=300,
        checkpoint_interval_steps=300,
    )
    for seed in (3, 0):
        run_dir = prepare_run_dir(cfg, seed, root)
        assert train(cfg, seed, run_dir)["state"] == "done"
    return root / cfg.run_group


@pytest.fixture(scope="module")
def tmp_lists(tmp_path_factory: pytest.TempPathFactory, cfgs: EvalConfigs) -> Path:
    """A temporary ``id`` list (one episode at each of two sea states) and its manifest."""
    from rld.deck.splits import dev_pool

    train_keys, _ = dev_pool(cfgs.sim)
    picks = []
    for ss in ("SS3", "SS4"):
        picks.append(next(s for s in train_keys if s.sea_state == ss))
    draws = [
        EpisodeDraw(
            "id",
            spec.sea_state,
            0,
            "aft",
            spec.vessel,
            float(spec.heading_deg),
            float(spec.speed_kn),
            int(spec.seed),
            2000 + i,
            True,
        )
        for i, spec in enumerate(picks)
    ]
    episodes = resolve_draws(draws, cfgs, workers=1)
    out = tmp_path_factory.mktemp("episodes")
    write_lists({"id": episodes}, out, 1)
    return out


def _fly_source(
    out: Path, name: str, episodes: list[ListedEpisode], cfgs: EvalConfigs, prov: dict[str, str]
) -> Path:
    """Write a carried-baseline source dir (episodes.csv, summary.csv) for one controller."""
    rows = run_list(episodes, controller_spec(name), cfgs, workers=1)
    summary = [{**r, **prov} for r in summarise(rows, [name])]
    src = out / f"src_{name}"
    write_rows(src / "episodes.csv", rows, EPISODE_COLUMNS)
    write_rows(src / "summary.csv", summary, summary_columns(prov))
    return src


@pytest.mark.slow
@pytest.mark.pybullet
def test_inspect_run_labels_training_seed(runs_root: Path, tmp_path: Path) -> None:
    import shutil

    runs = [inspect_run("test_sac", runs_root / d) for d in ("3", "0")]
    assert [r.seed for r in runs] == [3, 0]
    assert [r.seed for r in order_runs(runs)] == [0, 3]
    assert runs[0].algo == "sac" and runs[0].ckpt == "final" and runs[0].resumed_from is None
    assert dict(runs[0].digests)["run_model_sha256"] != dict(runs[1].digests)["run_model_sha256"]
    with pytest.raises(ValueError, match="twice"):
        order_runs([runs[0], runs[0]])
    with pytest.raises(ValueError, match="label"):
        inspect_run("sac", runs_root / "0")
    broken = tmp_path / "broken"
    shutil.copytree(runs_root / "0", broken)
    status = json.loads((broken / "status.json").read_text())
    (broken / "status.json").write_text(json.dumps({**status, "seed": 7}))
    with pytest.raises(ValueError, match="seed"):
        inspect_run("test_sac", broken)
    (broken / "status.json").write_text(json.dumps({**status, "state": "failed"}))
    with pytest.raises(ValueError, match="done"):
        inspect_run("test_sac", broken)


@pytest.mark.slow
@pytest.mark.pybullet
def test_learned_rows_do_not_depend_on_workers(
    runs_root: Path, tmp_lists: Path, cfgs: EvalConfigs
) -> None:
    from rld.eval.episodes import read_list

    episodes = read_list(tmp_lists / "id.parquet")
    runs = order_runs([inspect_run("test_sac", runs_root / d) for d in ("0", "3")])
    one = evaluate_runs(runs, episodes, cfgs, workers=1, chunk=1)
    two = evaluate_runs(runs, episodes, cfgs, workers=2, chunk=2)
    assert len(one) == 4
    for a, b in zip(one, two, strict=True):
        for key in EPISODE_COLUMNS:
            x, y = a[key], b[key]
            assert x == y or (isinstance(x, float) and x != x and y != y), key


@pytest.mark.slow
@pytest.mark.pybullet
def test_main_end_to_end(
    runs_root: Path, tmp_lists: Path, cfgs: EvalConfigs, tmp_path: Path
) -> None:
    from rld.eval.episodes import read_list

    episodes = read_list(tmp_lists / "id.parquet")
    prov = live_provenance(cfgs, tmp_lists)
    src_a = _fly_source(tmp_path, "pid_feedforward", episodes, cfgs, prov)
    src_b = _fly_source(tmp_path, "oracle_gated", episodes, cfgs, prov)
    out = tmp_path / "e05"
    run_dirs = f"{runs_root / '3'},{runs_root / '0'}"
    argv = [
        "--learned",
        f"test_sac={run_dirs}",
        "--episodes-dir",
        str(tmp_lists),
        "--out-dir",
        str(out),
        "--workers",
        "2",
        "--chunk",
        "1",
        "--carry",
        f"{src_a}=pid_feedforward",
        f"{src_b}=oracle_gated",
        "--title",
        "tiny",
    ]
    assert main(argv) == 0

    rows = read_rows(out / "episodes.csv")
    assert len(rows) == 2 * len(episodes)
    assert [r["run_seed"] for r in rows] == ["0", "0", "3", "3"]  # ordered by training seed
    assert {r["method"] for r in rows} == {"test_sac"}
    assert {r["privileged"] for r in rows} == {"False"}
    summary = read_rows(out / "summary.csv")
    assert len(summary) == 2 * 2  # 2 seeds x 2 sea states
    for rec in summary:
        seed_dir = runs_root / rec["run_seed"]
        assert Path(rec["run_dir"]) == seed_dir.resolve()  # the label is the run's own seed
        fracs = sum(float(rec[f"frac_{o}"]) for o in ("crash", "off_pad", "hard_landing"))
        fracs += sum(float(rec[f"frac_{o}"]) for o in ("bounce", "success", "timeout"))
        assert fracs == pytest.approx(1.0, abs=1e-12)
        assert rec["run_ckpt"] == "final" and rec["run_algo"] == "sac"
    spread = read_rows(out / "seeds.csv")
    assert {r["ss"] for r in spread} == {"SS3", "SS4"}
    assert "success_rate_seed0" in spread[0] and "success_rate_seed3" in spread[0]
    agg = read_rows(out / "aggregate.csv")
    assert {(r["ss"], r["metric"], r["statistic"]) for r in agg} >= {
        ("SS3", "success_rate", "iqm"),
        ("SS4", "success_rate", "optimality_gap"),
    }
    assert {r["n_runs"] for r in agg} == {"2"}

    # Carried rows are line-for-line copies of their sources.
    for src in (src_a, src_b):
        carried = (out / f"{CARRIED_PREFIX}{src.name}.csv").read_text().splitlines()
        source = (src / "summary.csv").read_text().splitlines()
        assert carried == source  # the whole (tiny) source is on the flown cells
    info = json.loads((out / "run_info.json").read_text())
    assert [(r["method"], r["seed"]) for r in info["runs"]] == [("test_sac", 0), ("test_sac", 3)]
    assert all(v["lines_byte_identical_to_source"] for v in info["carried"].values())
    assert all(info["rederived_byte_identical"].values())

    # The markdown is rendered from the CSVs only and re-renders byte for byte.
    md = (out / "success_vs_seastate.md").read_text()
    assert "commit-timing oracle (privileged)" in md
    assert "test_sac | 3 |" in md and "IQM (2 seeds)" in md
    assert "outside every method's training distribution" in md
    assert main(["--out-dir", str(out), "--title", "tiny", "--render-only"]) == 0
    assert (out / "success_vs_seastate.md").read_text() == md
    assert main(["--out-dir", str(out), "--title", "tiny", "--check"]) == 0
