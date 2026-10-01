"""Tests for the Phase 6 additions to the reward-hacking audit.

Known-value tests of the reductions the Phase 6 readings rest on (``rld.rl.audit_phase6``),
the spec / output-directory guards, and end-to-end re-flights through the committed
``build_policy`` path: a residual and a forecast run each reproduce a committed e06 row
with the read-only hooks installed, and the set-C block replacement does what the
pre-statement says (the training-mean replacement normalises to exactly 0).

Units: speeds m/s model scale, angles degrees, actions normalised (x 1.5 m/s).
"""

from pathlib import Path

import numpy as np
import pytest

from rld.rl import audit, audit_phase6

REPO = Path(__file__).resolve().parents[1]
E06 = REPO / "results" / "e06" / "episodes.csv"
RUNS = REPO / "artifacts" / "runs"


def test_specs() -> None:
    assert audit.PHASE5.methods == ("ppo", "sac") and not audit.PHASE5.phase6
    assert audit.PHASE5.quintile_tests == audit.QUINTILE_TESTS == 24
    assert audit.PHASE6.methods == (
        "residual_ppo",
        "ppo_forecast",
        "residual_ppo_forecast",
        "ppo_sinusoid",
    )
    assert audit.PHASE6.quintile_tests == 48 and audit.PHASE6.phase6
    assert audit.seed_metrics("e05") == audit.SEED_METRICS
    assert all("e05" not in m for m in audit.seed_metrics("e06"))


def test_out_dir_guard(tmp_path: Path) -> None:
    p5 = tmp_path / "results" / "audit"
    with pytest.raises(ValueError):
        audit.check_out_dir(p5, audit.PHASE6, repo=tmp_path)
    with pytest.raises(ValueError):
        audit.check_out_dir(p5 / "e06", audit.PHASE5, repo=tmp_path)
    audit.check_out_dir(p5 / "e06", audit.PHASE6, repo=tmp_path)
    audit.check_out_dir(tmp_path / "scratch", audit.PHASE6, repo=tmp_path)
    audit.check_out_dir(p5, audit.PHASE5, repo=tmp_path)


def test_parse_args_default_out_dirs() -> None:
    a5 = audit.parse_args(["--scratch-dir", "/tmp/x"])
    a6 = audit.parse_args(["--phase", "6", "--scratch-dir", "/tmp/x"])
    assert a5.out_dir.resolve() == (audit.REPO_ROOT / "results" / "audit").resolve()
    assert a6.out_dir == audit.REPO_ROOT / "results" / "audit" / "e06"


@pytest.mark.parametrize(
    ("closing", "rel", "ab", "deck", "expected"),
    [
        (0.30, 10.0, 2.0, 9.0, ("neither", "")),
        (0.60, 10.0, 2.0, 9.0, ("speed_only", "")),
        (0.30, 16.0, 2.0, 17.0, ("tilt_only", "deck_alone_over_limit")),
        (0.30, 16.0, 5.0, 12.0, ("tilt_only", "deck_dominated")),
        (0.30, 16.0, 12.0, 5.0, ("tilt_only", "drone_dominated")),
        (0.60, 16.0, 12.0, 12.0, ("both", "deck_dominated")),
        (0.50, 15.0, 1.0, 1.0, ("neither", "")),  # at the limits is not a violation
    ],
)
def test_classify_hard_landing(
    closing: float, rel: float, ab: float, deck: float, expected: tuple[str, str]
) -> None:
    assert audit_phase6.classify_hard_landing(closing, rel, ab, deck) == expected


def test_dependence_bootstrap_known_values() -> None:
    zero = {ss: [0.0] * 25 for ss in audit.SEA_STATES}
    assert audit_phase6.dependence_bootstrap(zero, reps=200, seed_words=(1,)) == (0.0, 0.0, 0.0)
    const = {ss: [-0.2] * 25 for ss in audit.SEA_STATES}
    point, lo, hi = audit_phase6.dependence_bootstrap(const, reps=200, seed_words=(1,))
    assert point == pytest.approx(-0.2) and lo == pytest.approx(-0.2) and hi == pytest.approx(-0.2)
    mixed = {"SS3": [0.0] * 24 + [-1.0], "SS6": [0.0] * 25}
    point, lo, hi = audit_phase6.dependence_bootstrap(mixed, reps=2000, seed_words=(1,))
    assert point == pytest.approx(-1.0 / 50.0)
    assert lo < point <= hi <= 0.0
    again = audit_phase6.dependence_bootstrap(mixed, reps=2000, seed_words=(1,))
    assert again == (point, lo, hi)


def _extras_logs(k: int, k1: int) -> tuple[np.ndarray, np.ndarray]:
    """A residual flight: pi_z = -1 before contact, 0.5 after; base (0.1, 0, -0.2)."""
    steps = np.zeros((k, len(audit.STEP_COLS)))
    steps[:, audit.STEP_COLS.index("t_start_s")] = np.arange(k) / 30.0
    steps[k1:, audit.STEP_COLS.index("obs_in_contact")] = 1.0
    x = np.zeros((k, len(audit.XSTEP_COLS)))
    col = audit.XSTEP_COLS.index
    x[:, col("base_ax")] = 0.1
    x[:, col("base_az")] = -0.2
    x[:, col("base_sp_x_m_s")] = 0.15
    x[:, col("base_sp_z_m_s")] = -0.3
    x[:k1, col("pi_z")] = -1.0
    x[k1:, col("pi_z")] = 0.5
    x[:, col("d_open_mean_m_s")] = np.nan
    x[:, col("d_open_zero_m_s")] = np.nan
    x[:, col("abs_tilt_deg")] = np.arange(k, dtype=np.float64)
    x[:, col("deck_tilt_deg")] = 7.0
    # The executed setpoint: base + 0.3 pi, no clip or cap binding here.
    exe = np.stack(
        [x[:, col("base_ax")], np.zeros(k), x[:, col("base_az")] + 0.3 * x[:, col("pi_z")]], 1
    )
    steps[:, audit.STEP_COLS.index("sp_x_m_s")] = 1.5 * exe[:, 0]
    steps[:, audit.STEP_COLS.index("sp_z_m_s")] = 1.5 * exe[:, 2]
    return steps, x


def test_summarise_extras_residual_known_values() -> None:
    k, k1 = 60, 45
    steps, x = _extras_logs(k, k1)
    td = (k1 - 0.5) / 30.0
    out = audit_phase6.summarise_extras(steps, x, {"td_t_episode_s": td}, v_max_m_s=1.5, alpha=0.3)
    assert out["x_alpha"] == 0.3
    assert out["x_auth_pre_p50_m_s"] == pytest.approx(0.45)
    assert out["x_auth_z_pre_mean_m_s"] == pytest.approx(-0.45)
    assert out["x_pi_sat_z_pre_frac"] == 1.0 and out["x_pi_sat_any_pre_frac"] == 1.0
    assert out["x_auth_post_p50_m_s"] == pytest.approx(0.225)
    assert out["x_pi_sat_z_post_frac"] == 0.0
    assert out["x_clip_bind_pre_frac"] == 0.0 and out["x_cap_bind_pre_frac"] == 0.0
    assert out["x_dep_pre_p50_m_s"] == pytest.approx(0.45)
    assert np.isnan(out["x_d_open_mean_median_m_s"])
    # Last 0.5 s before touchdown = 15 control steps, the last being index k1 - 1.
    assert out["x_last_abs_tilt_deg"] == float(k1 - 1)
    assert out["x_win_abs_tilt_max_deg"] == float(k1 - 1)
    assert out["x_last_deck_tilt_deg"] == 7.0
    assert out["x_win_auth_z_mean_m_s"] == pytest.approx(-0.45)


def test_residual_arrays_clip_and_cap() -> None:
    steps, x = _extras_logs(4, 2)
    col = audit.XSTEP_COLS.index
    x[0, col("base_ax")] = 0.9
    x[0, col("pi_x")] = 1.0  # 0.9 + 0.3 > 1: the composed clip binds
    x[1, col("base_ax")] = 0.8
    x[1, col("base_az")] = -0.8  # norm 1.13 after the clip: the cap binds
    x[1, col("pi_z")] = 0.0
    arr = audit_phase6._residual_arrays(steps, x, 1.5, 0.3)
    assert list(arr["clip_bind"]) == [1.0, 0.0, 0.0, 0.0]
    assert list(arr["cap_bind"][:2]) == [1.0, 1.0]
    assert list(arr["pi_sat_any"]) == [1.0, 0.0, 0.0, 0.0]  # rows 2-3: pi_z = 0.5


def _committed_row(method: str, seed: int, ss: str, index: int) -> dict[str, str]:
    rows = audit.read_rows(E06)
    return next(
        r
        for r in rows
        if r["method"] == method
        and int(r["run_seed"]) == seed
        and (r["ss"], int(r["index"])) == (ss, index)
    )


needs_e06 = pytest.mark.skipif(
    not E06.exists() or not (RUNS / "ppo_forecast" / "0" / "final").exists(),
    reason="committed e06 or the Phase 6 run artifacts absent",
)


@needs_e06
@pytest.mark.parametrize("method", ["residual_ppo", "ppo_forecast", "residual_ppo_forecast"])
def test_phase6_refly_reproduces_with_hooks(method: str) -> None:
    """The hooks read what was flown: the committed e06 row reproduces, and alpha*pi fits."""
    from rld.eval.episodes import read_list

    listed = {(r.ss, r.index): r for r in read_list(REPO / "results" / "episodes" / "id.parquet")}
    key = ("SS5", 3)
    src = audit.PolicySource(method, 0, str(RUNS / method / "0"))
    assert src.needs_feed() == ("forecast" in method)
    task = audit.FlightTask(
        src, ((listed[key], "S"),), (_committed_row(method, 0, *key),), None, extras=True
    )
    (out,) = audit.fly_chunk(task)
    assert out["reproduced"] is True
    xs = out["_xsteps"]
    assert xs.shape == (out["pre_n"] + out["post_n"], len(audit.XSTEP_COLS))
    if "residual" in method:
        assert out["x_alpha"] == 0.3
        assert float(out["x_auth_pre_max_m_s"]) <= 0.45 * np.sqrt(3.0) + 1e-9
    else:
        assert np.isnan(out["x_alpha"])
    finite = np.isfinite(xs[:, audit.XSTEP_COLS.index("d_open_mean_m_s")])
    assert bool(finite.all()) == ("forecast" in method)


@needs_e06
def test_ablation_mean_normalises_block_to_exact_zero() -> None:
    """Set C's training-mean replacement is exactly 0 after the frozen normaliser."""
    from rld.eval.envs import load_eval_configs

    cfgs = load_eval_configs()
    src = audit.PolicySource("ppo_forecast", 0, str(RUNS / "ppo_forecast" / "0"), "mean")
    policy = src.build(cfgs)
    calls: list[int] = []

    def fake_input(obs: np.ndarray) -> np.ndarray:
        del obs
        calls.append(1)
        return np.arange(31, dtype=np.float32)

    policy.policy_input = fake_input
    audit._install_ablation(policy, "mean", 25)
    x = policy.policy_input(np.zeros(25, dtype=np.float32))
    assert calls == [1]
    z = policy.normalizer.normalize_obs(x.reshape(1, -1))[0]
    assert np.all(z[25:] == 0.0)
    assert np.array_equal(x[:25], np.arange(25, dtype=np.float64))
    audit._install_ablation(policy, "zeros", 25)
    assert np.all(policy.policy_input(np.zeros(25, dtype=np.float32))[25:] == 0.0)
    with pytest.raises(ValueError):
        audit._install_ablation(policy, "noise", 25)
