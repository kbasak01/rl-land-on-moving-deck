"""Phase 7 evaluation code: arms, per-arm configs, motion kinds, start checks, verdicts, report.

What is pinned here (P7-D1, committed at ``27934bf`` before any Phase 7 flight):

* **Noise.** The configured latencies 0 / 33.4 / 66.7 ms quantise to exactly 0 / 1 / 2 control
  steps (a literal 33 ms would be 0); sigma_v = sigma_p / 0.2 s. Turning the stand-in on leaves
  t0 and the initial state the listed ones (the strict start check holds), and the enabled
  sigma = 0 / latency = 0 condition is bit-identical to the clean run, every column.
* **Sinusoid leg.** The runner's sinusoid source is the training builder's
  (:class:`rld.rl.wrappers.EnvFactory` with ``motion = "sinusoid"``) bit for bit on 3 listed
  episodes: same parameters, same deck-point trajectory on the episode's physics grid.
* **lambda.** At lambda = 1/40 the strict start check refuses the listed episodes (t0 is
  re-drawn) and the ``"lambda"`` check accepts them with the listed realization, initial
  position and pad.
* **Default arms unchanged.** With every new field at its default, the runner reproduces
  committed e01 and e06 rows line for line; so does :mod:`rld.eval.phase7`'s matrix arm.
* **Verdicts** follow the pre-registered rules (truth table), and the report re-renders
  byte-identically.

The flights use committed episode lists and committed policies only to **reproduce committed
rows** (as the Phase 6 gate test does); no new number is read from them. Tests that need the
trained runs under ``artifacts/runs/`` skip without them.

Units: seconds and metres model scale; rates dimensionless.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from rld.envs.noise import PerceptionNoise
from rld.eval import arms
from rld.eval.envs import (
    MOTION_KINDS,
    EvalConfigs,
    load_eval_configs,
    motion_for,
    with_lambda,
    with_noise,
)
from rld.eval.episodes import EPISODES_DIR, ListedEpisode, read_list
from rld.eval.hypotheses import (
    INCONCLUSIVE,
    NO_SUPPORTED_ID_GAIN,
    NOT_APPLICABLE,
    NOT_SUPPORTED,
    SUPPORTED,
    verdict_h1a,
    verdict_half_rule,
    verdict_magnitude,
    verdict_p7d1a,
)
from rld.eval.runner import (
    EpisodeListMismatchError,
    EvalArm,
    controller_spec,
    episode_rows_csv,
    run_arms,
    run_list,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
_RUNS = REPO / "artifacts" / "runs"
_HAVE_RUNS = all((_RUNS / m / "0" / "final" / "model.zip").is_file() for m in arms.LEARNED_METHODS)


@pytest.fixture(scope="module")
def cfgs() -> EvalConfigs:
    return load_eval_configs()


@pytest.fixture(scope="module")
def id_list() -> list[ListedEpisode]:
    return read_list(EPISODES_DIR / "id.parquet")


def _first(episodes: list[ListedEpisode], ss: str, k: int) -> list[ListedEpisode]:
    return [e for e in episodes if e.ss == ss][:k]


def _lines(text: str) -> list[str]:
    return [line for line in text.split("\n")[1:] if line]


# --------------------------------------------------------------------------- arms (fast)


def test_arm_definitions_follow_p7_d1() -> None:
    noise = arms.conditions("noise")
    assert len(noise) == 11  # 4 sigma x 3 latency, the clean (0, 0) condition is the matrix's
    assert {c.noise.latency_steps for c in noise if c.noise} == {0, 1, 2}
    lam = arms.conditions("lambda")
    assert [c.lam_inverse for c in lam] == [15.0, 40.0]
    assert all(c.start_check == "lambda" and c.lists == arms.NONSTATIC_LISTS for c in lam)
    assert all(c.learned == ("ppo",) for c in lam)  # P7-D1 §3 selection rule
    assert set(lam[0].baselines) == {"pid_feedforward", *arms.ALWAYS_PRINTED}
    (cg,) = arms.conditions("cg")
    assert cg.pads == ("cg",) and "static" not in cg.lists
    assert cg.baselines == ("pid_feedforward_lowvz_cut",)
    assert {m for _, ms, _ in cg.carry for m in ms} | set(cg.baselines) == set(arms.BASELINES)
    (matrix,) = arms.conditions("matrix")
    assert matrix.lists == arms.ALL_LISTS and matrix.baselines == ()
    assert {m for _, ms, _ in matrix.carry for m in ms} == set(arms.BASELINES)
    (sin,) = arms.conditions("sinusoid")
    assert sin.motion == "sinusoid" and sin.lists == ("id",) and sin.baselines == arms.BASELINES
    (mss,) = arms.conditions("mss", Path("/nonexistent"))
    assert mss.optional and mss.pads == (None, "cg")
    assert {mss.motion_for_regime(r) for r in mss.lists} == {"mss", "mss_corpus"}
    assert MOTION_KINDS == ("jonswap", "sinusoid", "mss", "mss_corpus")


def test_noise_latency_steps_and_velocity_rule(cfgs: EvalConfigs) -> None:
    hz = float(cfgs.landing.ctrl_freq_hz)
    for setting in arms.NOISE_GRID:
        noisy = with_noise(
            cfgs,
            position_sigma_m=setting.sigma_p_m,
            velocity_sigma_m_s=setting.sigma_v_m_s,
            latency_ms=setting.latency_ms,
        )
        steps = PerceptionNoise(noisy.noise, np.random.default_rng(0), hz).latency_steps
        assert steps == setting.latency_steps
        assert noisy.noise.enabled and noisy.noise.hold_freq_hz == 30.0
        assert setting.sigma_v_m_s == pytest.approx(setting.sigma_p_m / arms.TAU_V_MODEL_S)
    assert [s.latency_ms for s in arms.NOISE_GRID[::4]] == [0.0, 33.4, 66.7]
    literal = with_noise(cfgs, position_sigma_m=0.0, velocity_sigma_m_s=0.0, latency_ms=33.0)
    assert PerceptionNoise(literal.noise, np.random.default_rng(0), hz).latency_steps == 0
    assert cfgs.noise.enabled is False  # the committed config is untouched


def test_with_lambda_changes_only_the_scale(cfgs: EvalConfigs) -> None:
    out = with_lambda(cfgs, 40.0)
    assert out.scaling.lam == 1.0 / 40.0 and cfgs.scaling.lam == 1.0 / 25.0
    assert dataclasses.replace(out, scaling=cfgs.scaling) == cfgs
    with pytest.raises(ValueError):
        with_lambda(cfgs, 0.0)


def test_motion_for_kinds(cfgs: EvalConfigs, id_list: list[ListedEpisode]) -> None:
    ep = id_list[0]
    args = (ep.vessel, ep.ss, ep.heading_deg, ep.speed_kn, ep.realization_seed)
    assert type(motion_for(cfgs, *args)).__name__ == "JonswapDeckMotion"
    with pytest.raises(ValueError, match="episode seed"):
        motion_for(cfgs, *args, kind="sinusoid")
    with pytest.raises(ValueError, match="unknown motion kind"):
        motion_for(cfgs, *args, kind="waves")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="static pad"):
        motion_for(cfgs, "static", "static", 0.0, 0.0, 0, kind="sinusoid", episode_seed=1)


# --------------------------------------------------------------------------- verdicts (fast)


@pytest.mark.parametrize(
    ("r", "lo", "hi", "ni_lo", "want"),
    [
        (0.20, 0.05, 0.30, -0.01, SUPPORTED),
        (0.15, 0.01, 0.25, -0.02, SUPPORTED),  # both thresholds inclusive
        (0.10, 0.02, 0.18, 0.0, INCONCLUSIVE),
        (0.20, 0.05, 0.30, -0.021, NOT_SUPPORTED),  # non-inferiority fails
        (0.20, -0.01, 0.30, 0.0, NOT_SUPPORTED),  # CI of r includes 0
        (0.20, 0.0, 0.30, 0.0, NOT_SUPPORTED),  # a bound at 0 does not exclude 0
        (-0.10, -0.20, -0.05, 0.0, NOT_SUPPORTED),  # lands harder, separates below 0
        (-0.30, -0.50, -0.10, -0.50, NOT_SUPPORTED),
    ],
)
def test_h1a_truth_table(r: float, lo: float, hi: float, ni_lo: float, want: str) -> None:
    assert verdict_h1a(r, lo, hi, ni_lo) == want


@pytest.mark.parametrize(
    ("point", "lo", "hi", "threshold", "want"),
    [
        (0.06, 0.01, 0.10, 0.05, SUPPORTED),  # H1b
        (0.05, 0.001, 0.09, 0.05, SUPPORTED),
        (0.03, 0.005, 0.06, 0.05, INCONCLUSIVE),
        (0.06, -0.01, 0.10, 0.05, NOT_SUPPORTED),
        (0.06, 0.0, 0.10, 0.05, NOT_SUPPORTED),  # a bound at 0 does not exclude 0
        (-0.02, -0.05, -0.01, 0.05, NOT_SUPPORTED),  # wholly below 0 (P7-D1a #6)
        (-0.01, 0.002, 0.03, 0.05, NOT_SUPPORTED),  # H1b: "negative" is not supported
        (0.12, 0.02, 0.20, 0.10, SUPPORTED),  # H3 threshold
        (0.07, 0.01, 0.12, 0.10, INCONCLUSIVE),  # H3: 0 < r < 10 %, separates
        (float("nan"), float("nan"), float("nan"), 0.10, NOT_SUPPORTED),
    ],
)
def test_magnitude_truth_table(
    point: float, lo: float, hi: float, threshold: float, want: str
) -> None:
    """H1b (P3-D1 §8) and H1a / H3's r rule (P3-D4 #5, P7-D1a #5, #6)."""
    assert verdict_magnitude(point, lo, hi, threshold) == want


@pytest.mark.parametrize(
    ("point", "lo", "hi", "want"),
    [
        (0.12, 0.02, 0.20, SUPPORTED),  # >= 10 points and lower bound > 0
        (0.10, 0.001, 0.19, SUPPORTED),  # both inclusive / strict as worded
        (0.09, 0.01, 0.15, INCONCLUSIVE),  # lower bound > 0, point < 10 points
        (0.0, 0.001, 0.02, INCONCLUSIVE),  # P7-D1a #1 as worded (verdict_magnitude: not)
        (0.15, 0.0, 0.30, NOT_SUPPORTED),  # bound at 0
        (0.15, -0.02, 0.30, NOT_SUPPORTED),  # CI includes 0
        (0.0, 0.0, 0.0, NOT_SUPPORTED),  # H4 at its bound: all 200/200 everywhere
        (-0.40, -0.60, -0.20, NOT_SUPPORTED),  # both legs lose; CI wholly below 0
        (float("nan"), float("nan"), float("nan"), NOT_SUPPORTED),
    ],
)
def test_h2_h4_truth_table(point: float, lo: float, hi: float, want: str) -> None:
    """P7-D1a #1 (user, 2026-10-02): H2 and H4 at the pre-registered 10-point magnitude."""
    assert verdict_p7d1a(point, lo, hi, 0.10) == want


@pytest.mark.parametrize(
    ("r_id", "r_uv", "id_verdict", "want"),
    [
        (0.20, 0.10, SUPPORTED, SUPPORTED),  # exactly half meets the rule
        (0.20, 0.05, SUPPORTED, SUPPORTED),
        (0.20, -0.10, SUPPORTED, SUPPORTED),
        (0.20, 0.11, SUPPORTED, NOT_SUPPORTED),
        # Review m6: a met rule behind an id gain that is not supported is not a verdict.
        (0.0188, 0.0004, INCONCLUSIVE, NO_SUPPORTED_ID_GAIN),
        (0.05, 0.20, NOT_SUPPORTED, NO_SUPPORTED_ID_GAIN),
        (0.0, 0.0, NOT_SUPPORTED, NOT_APPLICABLE),
        (-0.05, 0.10, NOT_SUPPORTED, NOT_APPLICABLE),
    ],
)
def test_half_rule_truth_table(r_id: float, r_uv: float, id_verdict: str, want: str) -> None:
    assert verdict_half_rule(r_id, r_uv, id_verdict) == want


def test_verdict_words_are_p3_d1_vocabulary() -> None:
    """Review m6: every verdict word is P3-D1 §8's, or an explicit not-scored / pending label."""
    import csv

    from rld.eval import hypotheses

    allowed = {SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE}
    with (RESULTS / "e07" / "hypotheses.csv").open(newline="", encoding="utf-8") as handle:
        verdicts = [row["verdict"] for row in csv.DictReader(handle)]
    for verdict in verdicts:
        assert verdict in allowed or verdict.startswith(
            (hypotheses.NOT_SCORED, "not applicable", "pending")
        ), verdict
    assert verdicts.count(NO_SUPPORTED_ID_GAIN) == 1
    assert not hasattr(hypotheses, "HOLDS") and not hasattr(hypotheses, "FAILS")


def test_no_combined_h3_or_h1_verdict_exists() -> None:
    """P3-D4 #5 and P7-D1a #2: every part is its own verdict; no conjunction is written."""
    from rld.eval import hypotheses

    assert not hasattr(hypotheses, "combine_h3")
    assert not hasattr(hypotheses, "ROLE_COMBINED")
    assert "combine_h3" not in hypotheses.__all__
    assert hypotheses.ROLE_POSTHOC == "post hoc, not scored"


# --------------------------------------------------------------------------- flights (slow)


@pytest.mark.slow
@pytest.mark.pybullet
def test_noise_keeps_the_listed_start_and_zero_noise_is_clean(
    cfgs: EvalConfigs, id_list: list[ListedEpisode]
) -> None:
    episodes = _first(id_list, "SS5", 2)
    spec = controller_spec("pid_feedforward")
    clean = run_list(episodes, spec, cfgs, workers=1)
    zero = with_noise(cfgs, position_sigma_m=0.0, velocity_sigma_m_s=0.0, latency_ms=0.0)
    assert zero.noise.enabled
    assert episode_rows_csv(run_list(episodes, spec, zero, workers=1)) == episode_rows_csv(clean)
    loud = with_noise(cfgs, position_sigma_m=0.04, velocity_sigma_m_s=0.20, latency_ms=66.7)
    noisy = run_list(episodes, spec, loud, workers=1)  # strict start check: t0 and init listed
    for a, b in zip(clean, noisy, strict=True):
        assert a["t0_model_s"] == b["t0_model_s"]
        assert (a["init_x_m"], a["init_y_m"], a["init_z_m"]) == (
            b["init_x_m"],
            b["init_y_m"],
            b["init_z_m"],
        )
    assert episode_rows_csv(noisy) != episode_rows_csv(clean)  # the noise did act


def _assert_bit_identical(a: object, b: object) -> None:
    """Assert two (nested) dataclasses of arrays and scalars are equal bit for bit."""
    assert type(a) is type(b)
    if dataclasses.is_dataclass(a):
        for field in dataclasses.fields(a):
            _assert_bit_identical(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.shape == b.shape
        assert np.array_equal(a, b, equal_nan=True) and a.tobytes() == b.tobytes()
    else:
        assert a == b


@pytest.mark.slow
@pytest.mark.pybullet
def test_sinusoid_leg_is_the_training_builder(
    cfgs: EvalConfigs, id_list: list[ListedEpisode]
) -> None:
    from dmf.sim.generate import RealizationSpec

    from rld.deck.splits import key_for_spec
    from rld.rl.motion import committed_rms_full
    from rld.rl.wrappers import EnvFactory

    episodes = [_first(id_list, ss, 1)[0] for ss in ("SS3", "SS5", "SS6")]
    specs = tuple(
        RealizationSpec(
            sea_state=e.ss,
            heading_deg=e.heading_deg,
            speed_kn=e.speed_kn,
            vessel=e.vessel,
            seed=e.realization_seed,
        )
        for e in episodes
    )
    table = committed_rms_full(key_for_spec(s) for s in specs)
    factory = EnvFactory(
        cfgs, specs, "aft", 0, 0, None, None, motion="sinusoid", sinusoid_rms=tuple(table.items())
    )
    build = factory._episode_motion_factory()
    assert build is not None
    for ep, spec in zip(episodes, specs, strict=True):
        train = build(spec, ep.episode_seed)
        evaluated = motion_for(
            cfgs,
            ep.vessel,
            ep.ss,
            ep.heading_deg,
            ep.speed_kn,
            ep.realization_seed,
            kind="sinusoid",
            episode_seed=ep.episode_seed,
        )
        assert type(evaluated) is type(train)
        assert evaluated.params == train.params  # type: ignore[attr-defined]
        grid = ep.t0_model_s + np.arange(int(12.0 * 240) + 1) / 240.0
        for pad in ("aft", "cg"):
            _assert_bit_identical(evaluated.deck_point(grid, pad), train.deck_point(grid, pad))
    rows = run_list(
        episodes, controller_spec("pid_feedforward"), cfgs, workers=1, motion_kind="sinusoid"
    )
    assert [r["t0_model_s"] for r in rows] == [e.t0_model_s for e in episodes]  # strict check held


@pytest.mark.slow
@pytest.mark.pybullet
def test_lambda_check_accepts_what_the_strict_check_refuses(
    cfgs: EvalConfigs, id_list: list[ListedEpisode]
) -> None:
    episodes = _first(id_list, "SS4", 2)
    spec = controller_spec("pid_feedforward")
    at40 = with_lambda(cfgs, 40.0)
    with pytest.raises(EpisodeListMismatchError, match="t0"):
        run_list(episodes, spec, at40, workers=1)
    rows = run_list(episodes, spec, at40, workers=1, start_check="lambda")
    for row, ep in zip(rows, episodes, strict=True):
        assert row["t0_model_s"] != ep.t0_model_s  # re-drawn inside the 1/40 window
        assert (row["init_x_m"], row["init_y_m"], row["init_z_m"]) == ep.init_xyz_m
        assert (row["realization_seed"], row["episode_seed"]) == (
            ep.realization_seed,
            ep.episode_seed,
        )


@pytest.mark.slow
@pytest.mark.pybullet
def test_default_arm_reproduces_e01_rows(cfgs: EvalConfigs, id_list: list[ListedEpisode]) -> None:
    episodes = [e for ss in ("SS3", "SS6") for e in _first(id_list, ss, 2)]
    arm = EvalArm(controller_spec("pid_feedforward"), None, episodes)
    explicit = EvalArm(
        controller_spec("pid_feedforward"), None, episodes, "jonswap", None, "strict"
    )
    committed = set((RESULTS / "e01" / "episodes.csv").read_text(encoding="utf-8").split("\n"))
    for a in (arm, explicit):
        lines = _lines(episode_rows_csv(run_arms([a], cfgs, workers=1)))
        assert len(lines) == 4 and all(line in committed for line in lines)


@pytest.mark.slow
@pytest.mark.pybullet
@pytest.mark.skipif(not _HAVE_RUNS, reason="trained runs not under artifacts/runs/")
def test_default_arms_reproduce_e06_rows(cfgs: EvalConfigs, id_list: list[ListedEpisode]) -> None:
    from rld.eval.learned import inspect_run, learned_spec

    episodes = [_first(id_list, ss, 1)[0] for ss in ("SS3", "SS4", "SS5", "SS6")]
    committed = set((RESULTS / "e06" / "episodes.csv").read_text(encoding="utf-8").split("\n"))
    specs = [
        learned_spec(inspect_run(m, _RUNS / m / "0")) for m in ("residual_ppo", "ppo_forecast")
    ]
    rows = run_arms([EvalArm(s, None, episodes) for s in specs], cfgs, workers=2)
    lines = _lines(episode_rows_csv(rows))
    assert len(lines) == 8 and all(line in committed for line in lines)


# --------------------------------------------------------------------------- phase7 + report


@pytest.fixture(scope="module")
def scratch_e07(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A scratch e07 tree: the matrix and sinusoid arms, one episode per cell, seed 0."""
    if not _HAVE_RUNS:
        pytest.skip("trained runs not under artifacts/runs/")
    from rld.eval.phase7 import main

    root = tmp_path_factory.mktemp("e07")
    common = ["--out-root", str(root), "--per-cell", "1", "--seeds", "0", "--workers", "4"]
    assert main(["--arm", "matrix", "--methods", "residual_ppo", "ppo", *common]) == 0
    sub = ["--methods", "ppo", "ppo_sinusoid", "--baselines", *arms.ALWAYS_PRINTED]
    assert main(["--arm", "sinusoid", *sub, *common]) == 0
    return root


@pytest.mark.slow
@pytest.mark.pybullet
def test_phase7_matrix_slice_reproduces_e05_e06_and_verifies(scratch_e07: Path) -> None:
    from rld.eval.phase7 import main

    info = json.loads((scratch_e07 / "matrix" / "run_info.json").read_text(encoding="utf-8"))
    refs = info["reference_checks"]
    assert refs["e05"]["all_identical"] and refs["e06"]["all_identical"]
    assert refs["e05"]["methods"]["ppo"]["n_identical"] == 4
    assert all(info["rederived_byte_identical"].values()) and info["complete"]
    assert info["carried"].startswith("not carried")  # scratch subset
    for arm in ("matrix", "sinusoid"):
        assert main(["--arm", arm, "--check", "--out-root", str(scratch_e07)]) == 0
    summary = (scratch_e07 / "sinusoid" / "summary.csv").read_text(encoding="utf-8")
    header = summary.split("\n", 1)[0].split(",")
    assert header[-14:-12] == ["arm", "condition"]  # the condition is summary-level only
    assert "arm_motion" in header and "noise_latency_steps" in header
    rows = summary.split("\n")[1:]
    assert all(",sinusoid," in r for r in rows if r)  # eval_deck_motion and arm_motion
    path = scratch_e07 / "sinusoid" / "aggregate.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("0.95", "0.94", 1), encoding="utf-8")
    assert main(["--arm", "sinusoid", "--check", "--out-root", str(scratch_e07)]) == 1


@pytest.mark.slow
@pytest.mark.pybullet
def test_report_round_trip(scratch_e07: Path, tmp_path: Path) -> None:
    from rld.eval.hypotheses import write_hypotheses
    from rld.eval.report import render_results
    from rld.eval.results_md import main

    write_hypotheses(scratch_e07, RESULTS, reps=200, allow_subset=True)
    out = tmp_path / "results.md"
    argv = ["--results-dir", str(RESULTS), "--e07-dir", str(scratch_e07), "--out", str(out)]
    assert main(argv) == 0
    assert main([*argv, "--check"]) == 0
    text = out.read_text(encoding="utf-8")
    assert text == render_results(RESULTS, scratch_e07)
    assert "commit-timing oracle (privileged)" in text and "upper bound" not in text.lower()
    assert "Simulation only" in text and "pending — scored at Gate 8" in text
    for method in arms.ALWAYS_PRINTED:
        assert f"`{method}`" in text
    out.write_text(text + "edited by hand\n", encoding="utf-8")
    assert main([*argv, "--check"]) == 1


@pytest.mark.slow
@pytest.mark.pybullet
def test_compress_keeps_every_check_byte_identical(scratch_e07: Path, tmp_path: Path) -> None:
    """P7-D1a §12: gzip each written condition; --check, hypotheses and report unchanged."""
    import shutil

    from rld.eval import storage
    from rld.eval.hypotheses import check_hypotheses, write_hypotheses
    from rld.eval.phase7 import main
    from rld.eval.report import render_results

    root = tmp_path / "e07"
    shutil.copytree(scratch_e07, root)
    write_hypotheses(root, RESULTS, reps=200, allow_subset=True)
    hyp_before = (root / "hypotheses.csv").read_bytes()
    md_before = render_results(RESULTS, root)
    plain = (root / "matrix" / "episodes.csv").read_bytes()
    assert main(["--arm", "matrix", "--compress", "--out-root", str(root)]) == 0
    csv_path = root / "matrix" / "episodes.csv"
    assert not csv_path.exists() and storage.gz_path(csv_path).is_file()
    assert storage.read_bytes(csv_path) == plain  # round trip byte-identical
    info = json.loads((root / "matrix" / "run_info.json").read_text(encoding="utf-8"))
    sha = info["files_sha256"]["episodes.csv"]
    assert info["episodes_storage"]["sha256"] == sha
    assert storage.sha256_path(csv_path).read_text(encoding="utf-8").split()[0] == sha
    assert main(["--arm", "matrix", "--check", "--out-root", str(root)]) == 0
    assert main(["--arm", "matrix", "--compress", "--out-root", str(root)]) == 0  # idempotent
    # A condition that no longer re-derives byte-identically (a derived file edited by hand)
    # is refused, and nothing is written for it.
    agg = root / "sinusoid" / "aggregate.csv"
    agg.write_text(agg.read_text(encoding="utf-8") + "edited by hand\n", encoding="utf-8")
    sin_csv = root / "sinusoid" / "episodes.csv"
    assert main(["--arm", "sinusoid", "--compress", "--out-root", str(root)]) == 1
    assert sin_csv.is_file() and not storage.gz_path(sin_csv).exists()
    assert not storage.sha256_path(sin_csv).exists()
    assert check_hypotheses(root, RESULTS, reps=200, allow_subset=True)
    assert (root / "hypotheses.csv").read_bytes() == hyp_before
    assert render_results(RESULTS, root) == md_before
    log = json.loads((root / "run_info.json").read_text(encoding="utf-8"))
    assert "compress" in log["invocations"][-1]
    # A tampered sidecar is caught by --check.
    side = storage.sha256_path(root / "matrix" / "episodes.csv")
    side.write_text("0" * 64 + "  episodes.csv\n", encoding="utf-8")
    assert main(["--arm", "matrix", "--check", "--out-root", str(root)]) == 1
