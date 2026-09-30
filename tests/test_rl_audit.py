"""Tests for the Phase 5 reward-hacking audit (``rld.rl.audit``).

Known-value tests of every reduction the verdicts rest on, plus one end-to-end re-flight
that reproduces a committed e01 row through :class:`rld.rl.audit.AuditAviary` (which checks
that the logging subclass changes nothing).

Units: steps are env control steps; lengths metres model scale; speeds m/s model scale.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from rld.rl import audit

REPO = Path(__file__).resolve().parents[1]
MONITOR_HEADER = (
    "r,l,t,outcome,termination_reason,ss,heading_deg,speed_kn,realization_seed,episode_seed,"
    "t0_model_s,init_height_m,init_lateral_m,closing_speed_normal_m_s,max_penetration_m,"
    "detectors_disagree"
)


def _monitor_row(length: int, outcome: str, depth_m: float = 0.001, touchdown: bool = True) -> str:
    closing = "0.3" if touchdown else ""
    return f"1.0,{length},0.1,{outcome},x,SS3,180.0,0.0,1,2,10.0,0.8,0.1,{closing},{-depth_m},False"


def _run_dir(tmp_path: Path, files: dict[str, list[str]], n_envs: int, steps: int) -> Path:
    run = tmp_path / "run"
    (run / "monitor").mkdir(parents=True)
    (run / "status.json").write_text(
        json.dumps({"n_envs": n_envs, "steps": steps, "resumed_from": None})
    )
    for name, rows in files.items():
        (run / "monitor" / name).write_text(
            '#{"t_start": 0, "env_id": "None"}\n' + MONITOR_HEADER + "\n" + "\n".join(rows) + "\n"
        )
    return run


def test_bin_arithmetic_is_exact() -> None:
    assert audit.bin_of(1, 100) == 0
    assert audit.bin_of(10, 100) == 1
    assert audit.bin_of(99, 100) == 9
    assert audit.bin_of(100, 100) == 9
    assert not audit.in_final_tenth(90, 100)
    assert audit.in_final_tenth(91, 100)


def test_training_episodes_global_step_and_bins(tmp_path: Path) -> None:
    run = _run_dir(
        tmp_path,
        {
            "0.monitor.csv": [
                _monitor_row(10, "timeout", touchdown=False),
                _monitor_row(20, "success"),
            ],
            "1.monitor.csv": [_monitor_row(25, "success", depth_m=0.006)],
        },
        n_envs=2,
        steps=60,
    )
    eps, total, n_envs = audit.read_training_episodes(run)
    assert (total, n_envs) == (60, 2)
    # rank 0: ends at env steps 10 and 30 -> global 20 and 60; rank 1: 25 -> 50.
    assert [e["global_step_end"] for e in eps] == [20, 50, 60]
    assert [e["rank"] for e in eps] == [0, 1, 0]
    bins = audit.training_bins(eps, total, 0.005, n_bins=2)
    assert [b["n_episodes"] for b in bins] == [1, 2]
    assert bins[0]["timeout_frac"] == 1.0
    assert bins[1]["tunnelled_n"] == 1
    assert eps[0]["touchdown"] is False


def test_training_episodes_refuse_resume_segments(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, {"0.resume1.monitor.csv": [_monitor_row(10, "success")]}, 1, 10)
    with pytest.raises(ValueError, match="first-segment"):
        audit.read_training_episodes(run)


def test_training_episodes_refuse_resumed_run(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, {"0.monitor.csv": [_monitor_row(10, "success")]}, 1, 10)
    (run / "status.json").write_text(json.dumps({"n_envs": 1, "steps": 10, "resumed_from": "x"}))
    with pytest.raises(ValueError, match="resumed"):
        audit.read_training_episodes(run)


def test_depth_stats_known_values() -> None:
    depths = [0.001 * k for k in range(1, 101)]  # 1..100 mm
    out = audit.depth_stats(depths, 0.005)
    assert out["n"] == 100
    assert out["tunnelled_n"] == 95  # 6..100 mm
    assert out["max_mm"] == pytest.approx(100.0)
    assert out["p99_mm"] == pytest.approx(float(np.percentile(np.arange(1, 101), 99.0)))
    empty = audit.depth_stats([], 0.005)
    assert empty["n"] == 0 and np.isnan(empty["max_mm"])


def test_quintiles_are_equal_sized_and_ordered() -> None:
    rng = np.random.default_rng(0)
    values = rng.permutation(200).astype(float).tolist()
    qs = audit.quintile_of(values)
    assert [qs.count(q) for q in range(5)] == [40] * 5
    order = np.argsort(values)
    assert [qs[i] for i in order[:40]] == [0] * 40
    assert [qs[i] for i in order[-40:]] == [4] * 40


def test_quintile_contrast_known_values() -> None:
    qs = [q for q in range(5) for _ in range(40)]
    same = [1.0 if q >= 2 else 0.0 for q in qs]
    res = audit.quintile_contrast(qs, same, same, reps=200, seed_words=(1,), alphas=(0.05,))
    assert res["learned_gap"] == 1.0 and res["reference_gap"] == 1.0
    assert res["diff"] == 0.0
    assert res["lo_0.05"] == 0.0 and res["hi_0.05"] == 0.0
    flat = [1.0] * 200
    res2 = audit.quintile_contrast(qs, same, flat, reps=200, seed_words=(1,), alphas=(0.05,))
    assert res2["diff"] == 1.0
    again = audit.quintile_contrast(qs, same, flat, reps=200, seed_words=(1,), alphas=(0.05,))
    assert again == res2


def test_singled_out_rule() -> None:
    assert audit.singled_out({0: 1.0, 1: 1.1, 2: 0.9, 3: 1.0, 4: 5.0}) == 4
    # others' range 0.6; seed 4 is 0.5 above it -> not singled out.
    assert audit.singled_out({0: 1.0, 1: 1.5, 2: 0.9, 3: 1.0, 4: 2.0}) is None
    # ... and 0.7 above it -> singled out.
    assert audit.singled_out({0: 1.0, 1: 1.5, 2: 0.9, 3: 1.0, 4: 2.2}) == 4
    assert audit.singled_out({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}) is None
    assert audit.singled_out({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.01}) == 4
    assert audit.singled_out({0: -3.0, 1: 1.0, 2: 1.2, 3: 1.1, 4: 0.9}) == 0
    assert audit.singled_out({0: float("nan"), 1: 1.0, 2: 1.0}) is None


def test_saturation_counts() -> None:
    a = np.array(
        [
            [0.0, 0.0, -1.0],  # axis z saturated, norm exactly 1: cap not active
            [0.8, 0.0, -0.8],  # norm 1.13: cap active
            [0.99, 0.0, 0.0],  # axis x at the level
            [0.1, 0.1, 0.1],
        ]
    )
    out = audit.saturation_counts(a)
    assert out == {"n": 4, "cap": 1, "ax_x": 1, "ax_y": 0, "ax_z": 1}


def _logs(
    td_index: int,
    n_sub: int,
    depth_peak_index: int,
    idle_from: int,
    *,
    need: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / 240.0
    sub = np.zeros((n_sub, len(audit.SUB_COLS)))
    sub[:, audit.SUB_COLS.index("t_ep_s")] = np.arange(1, n_sub + 1) * dt
    sub[:, audit.SUB_COLS.index("depth_m")] = np.nan
    sub[td_index:, audit.SUB_COLS.index("depth_m")] = 0.001
    sub[depth_peak_index, audit.SUB_COLS.index("depth_m")] = 0.006
    sub[td_index:, audit.SUB_COLS.index("in_contact")] = 1.0
    sub[idle_from:, audit.SUB_COLS.index("idle")] = 1.0
    sub[:, audit.SUB_COLS.index("thrust_n")] = 0.26
    sub[:, audit.SUB_COLS.index("need_n")] = need
    sub[:, audit.SUB_COLS.index("vz_pre_m_s")] = -0.3
    sub[:, audit.SUB_COLS.index("rel_vn_pre_m_s")] = -0.3
    n_steps = n_sub // 8
    steps = np.zeros((n_steps, len(audit.STEP_COLS)))
    steps[:, audit.STEP_COLS.index("t_start_s")] = np.arange(n_steps) * 8 * dt
    steps[:, audit.STEP_COLS.index("sp_z_m_s")] = -0.3
    steps[:, audit.STEP_COLS.index("az")] = -0.2
    first_obs = td_index // 8 + 1
    steps[first_obs:, audit.STEP_COLS.index("obs_in_contact")] = 1.0
    steps[first_obs:, audit.STEP_COLS.index("sp_z_m_s")] = -1.5
    return steps, sub


def _record(td_t: float, depth: float, outcome: str = "success") -> dict[str, object]:
    return {
        "td_t_episode_s": td_t,
        "max_penetration_m": -depth,
        "outcome": outcome,
        "termination_reason": "dwell_complete",
    }


def _summarise(steps: np.ndarray, sub: np.ndarray, rec: dict[str, object]) -> dict[str, object]:
    return audit.summarise_flight(
        steps,
        sub,
        rec,
        substeps_per_ctrl=8,
        v_max_m_s=1.5,
        weight_n=0.2646,
        tunnelling_m=0.005,
        physics_dt_s=1.0 / 240.0,
        grace_s=0.05,
        hard_landing_m_s=0.5,
    )


def test_mechanism_impact_vs_post_contact_idle() -> None:
    td = 80
    t_td = (td + 1) / 240.0
    steps, sub = _logs(td, 240, td + 3, idle_from=10_000)
    out = _summarise(steps, sub, _record(t_td, 0.006))
    assert out["mechanism"] == "impact"
    assert out["in_impact_window"] is True
    assert out["loaded"] is True and out["possibly_dependent"] is False
    steps, sub = _logs(td, 240, td + 40, idle_from=td + 5)
    out = _summarise(steps, sub, _record(t_td, 0.006))
    assert out["mechanism"] == "post_contact_idle"
    assert out["idle_at_max"] is True
    assert out["post_full_descent_n"] == out["post_n"]  # -1.5 m/s after the flag
    assert out["passive"] is False


def test_unloaded_success_is_flagged_and_run_length_measured() -> None:
    td = 80
    steps, sub = _logs(td, 240, td + 3, idle_from=10_000)
    need = sub[:, audit.SUB_COLS.index("need_n")]
    need[td + 10 : td + 10 + 13] = -0.01  # 13 substeps = 54 ms > 50 ms grace
    out = _summarise(steps, sub, _record((td + 1) / 240.0, 0.006))
    assert out["loaded"] is False
    assert out["dependence_reason"] == "success_unloaded"
    assert out["max_unloaded_run_s"] == pytest.approx(13 / 240.0)
    assert out["unloaded_run_gt_grace"] is True


def test_depth_mismatch_raises() -> None:
    td = 80
    steps, sub = _logs(td, 240, td + 3, idle_from=10_000)
    with pytest.raises(audit.ReproductionError):
        _summarise(steps, sub, _record((td + 1) / 240.0, 0.007))


def test_no_touchdown_summary_is_nan() -> None:
    steps, sub = _logs(10_000, 240, 0, idle_from=10_000)
    sub[:, audit.SUB_COLS.index("depth_m")] = np.nan
    out = _summarise(steps, sub, _record(0.0, 0.0) | {"td_t_episode_s": None})
    assert np.isnan(float(out["depth_max_mm"]))
    assert out["post_sub_n"] == 0


def test_sample_is_the_pre_stated_draw() -> None:
    a = audit.sample_indices()
    assert a == audit.sample_indices()
    assert list(a) == list(audit.SEA_STATES)
    for idx in a.values():
        assert len(idx) == audit.SAMPLE_PER_SS == 25
        assert idx == sorted(set(idx)) and idx[0] >= 0 and idx[-1] < 200
    assert audit.SAMPLE_SEED == 20260930


def test_record_text_matches_committed_formatting() -> None:
    record = dict.fromkeys(audit.RECORD_COLUMNS)
    record.update(outcome="timeout", termination_reason="time_limit", steps=361, tunnelled=False)
    text = audit.record_text(
        record,
        [np.zeros(3), np.ones(3)],
        {
            "detectors_disagree": False,
            "closing_speed_world_z_m_s": float("nan"),
            "time_to_touchdown_s": float("nan"),
        },
    )
    assert text["n_contacts"] == "0"
    assert text["td_t_episode_s"] == "nan"
    assert text["steps"] == "361" and text["tunnelled"] == "False"
    assert text["effort_mean_sq"] == repr(1.5)


def test_write_csv_is_deterministic(tmp_path: Path) -> None:
    rows = [{"a": 0.1, "b": True, "c": None}, {"a": 1, "b": False}]
    t1 = audit.write_csv(tmp_path / "x.csv", rows, ["a", "b", "c"])
    t2 = audit.write_csv(tmp_path / "y.csv", rows, ["a", "b", "c"])
    assert t1 == t2 == "a,b,c\n0.1,True,nan\n1,False,nan\n"


@pytest.mark.skipif(
    not (REPO / "results" / "e01" / "episodes.csv").exists(), reason="committed e01 absent"
)
def test_refly_reproduces_a_committed_row_and_logs() -> None:
    """The logging subclass flies a committed e01 episode to the identical row."""
    from rld.eval.episodes import read_list

    listed = {(r.ss, r.index): r for r in read_list(REPO / "results" / "episodes" / "id.parquet")}
    rows = audit.read_rows(REPO / "results" / "e01" / "episodes.csv")
    key = ("SS5", 3)
    committed = next(
        r
        for r in rows
        if r["method"] == "pid_feedforward"
        and r["regime"] == "id"
        and (r["ss"], int(r["index"])) == key
    )
    task = audit.FlightTask(
        source=audit.PolicySource("pid_feedforward", 0),
        episodes=((listed[key], "S"),),
        committed=(committed,),
        logs_dir=None,
    )
    (out,) = audit.fly_chunk(task)
    assert out["reproduced"] is True
    assert out["outcome"] == committed["outcome"]
    assert out["pre_n"] + out["post_n"] == int(committed["steps"])
    assert float(out["depth_max_mm"]) == pytest.approx(
        -float(committed["max_penetration_m"]) * 1000.0, abs=0.0
    )
    # A committed row that differs in any checked column must stop the audit.
    tampered = dict(committed, return_="x", steps=str(int(committed["steps"]) + 1))
    bad = audit.FlightTask(task.source, task.episodes, (tampered,), None)
    with pytest.raises(audit.ReproductionError):
        audit.fly_chunk(bad)
