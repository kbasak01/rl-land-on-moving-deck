"""Hypothesis scoring as P7-D1a fixes it, on synthetic per-episode rows (no flight needed).

What is pinned here (P3-D1 §4 and §8, P3-D4 #5, P7-D1a #1-#4, #6, #7, #11, #12):

* **estimators.** H1a's non-inferiority, H1b and H4 are the mean-based paired bootstrap
  (``H1a.ni``, ``H1b.d``: :func:`rld.eval.stats.paired_bootstrap`; ``H4.dd``: the drop
  difference with ``estimator="mean"``) and their IQM variants sit beside them as
  ``post hoc, not scored`` (``*.iqm``); H2 is IQM with ``independent_cells``;
* **verdicts** in every scored row equal the pre-registered verdict function applied to that
  row's own numbers; H2 and H4 use :func:`verdict_p7d1a`;
* **no combined H3 row** (and no combined H1 row): eight H3 parts, each with its own verdict;
* every scored row carries the shared-bootstrap-seed and no-multiplicity caveats;
* **storage.** Compressing the Phase 7 ``episodes.csv`` files changes neither output by a
  byte, and ``check_hypotheses`` passes on the compressed tree.

The rows are random but fixed (seeded); they carry only the columns the scorer reads, and
no number from them is a project result.

Units: success fractions; closing speeds m/s model scale.
"""

import csv
import io
from pathlib import Path

import numpy as np
import pytest

from rld.eval import storage
from rld.eval.hypotheses import (
    INCONCLUSIVE,
    NOT_SCORED,
    NOT_SUPPORTED,
    ROLE_POSTHOC,
    ROLE_SCORED,
    ROLE_SECONDARY,
    SUPPORTED,
    check_hypotheses,
    compute,
    verdict_h1a,
    verdict_half_rule,
    verdict_magnitude,
    verdict_p7d1a,
    write_hypotheses,
)
from rld.eval.stats import drop_difference_bootstrap, paired_bootstrap

COLUMNS = (
    "method",
    "run_seed",
    "pad",
    "regime",
    "ss",
    "index",
    "episode_seed",
    "t0_model_s",
    "outcome",
    "touchdown_contact",
    "rel_vz_normal_m_s",
)
N_EP = 40
SEEDS = (0, 1, 2, 3, 4)
REPS = 300

#: (method, regime, ss) -> success probability of the synthetic rows.
P_SUCCESS = {
    "residual_ppo": 0.97,
    "ppo": 0.93,
    "ppo_forecast": 0.95,
    "residual_ppo_forecast": 0.96,
    "ppo_sinusoid": 0.9,
    "pid_feedforward_lowvz": 0.94,
    "pid_feedforward": 0.9,
}


def _rows(method: str, seeds: tuple[int, ...], cells: list[tuple[str, str]], salt: int) -> list:
    rng = np.random.default_rng(1000 + salt)
    out = []
    for regime, ss in cells:
        for seed in seeds:
            for i in range(N_EP):
                u = rng.random()
                outcome = "success" if u < P_SUCCESS[method] else rng.choice(["timeout", "bounce"])
                touched = outcome != "timeout"
                speed = -(0.1 + 0.2 * rng.random()) if touched else float("nan")
                out.append(
                    {
                        "method": method,
                        "run_seed": str(seed),
                        "pad": "aft",
                        "regime": regime,
                        "ss": ss,
                        "index": str(i),
                        "episode_seed": str(7000 + i + 100 * len(regime) + int(ss[-1])),
                        "t0_model_s": repr(5.0 + i * 0.25),
                        "outcome": str(outcome),
                        "touchdown_contact": str(touched),
                        "rel_vz_normal_m_s": repr(speed),
                    }
                )
    return out


def _write(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


@pytest.fixture()
def tree(tmp_path: Path) -> tuple[Path, Path]:
    """A synthetic ``results/`` with ``e01`` baselines and ``e07`` matrix + sinusoid rows."""
    results = tmp_path / "results"
    e07 = results / "e07"
    learned_cells = [
        ("id", "SS5"),
        ("id", "SS6"),
        ("unseen_seastate", "SS6"),
        ("unseen_vessel", "SS5"),
        ("unseen_vessel", "SS6"),
    ]
    matrix = []
    for k, m in enumerate(
        ("ppo", "residual_ppo", "ppo_forecast", "residual_ppo_forecast", "ppo_sinusoid")
    ):
        matrix += _rows(m, SEEDS, learned_cells, k)
    _write(e07 / "matrix" / "episodes.csv", matrix)
    sinusoid = _rows("ppo", SEEDS, [("id", "SS5")], 20)
    sinusoid += _rows("ppo_sinusoid", SEEDS, [("id", "SS5")], 21)
    _write(e07 / "sinusoid" / "episodes.csv", sinusoid)
    base = _rows("pid_feedforward_lowvz", (0,), [("id", "SS5")], 30)
    base += _rows("pid_feedforward", (0,), [("id", "SS6")], 31)
    _write(results / "e01" / "episodes.csv", base)
    return results, e07


def _by(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {r[key]: r for r in rows}


def _f(rec: dict[str, str], col: str) -> float:
    return float(rec[col])


def test_estimators_follow_p7_d1a(tree: tuple[Path, Path]) -> None:
    results, e07 = tree
    contrasts, hyps, _ = compute(e07, results, reps=REPS, repo_root=results.parent)
    con = _by(contrasts, "contrast")
    # H1a NI and H1b: P3-D1 §4's mean paired bootstrap is scored, IQM is post hoc.
    for cid in ("H1a.ni", "H1b.d"):
        assert con[cid]["role"] == ROLE_SCORED
        assert con[cid]["statistic"] == "mean_success_difference"
        assert con[cid + ".iqm"]["role"] == ROLE_POSTHOC
        assert con[cid + ".iqm"]["statistic"] == "iqm_success_difference"
    assert "H1a.ni.mean" not in con and "H1b.d.mean" not in con
    # H4: mean drop difference scored, IQM post hoc; H2: IQM, independent cells.
    assert con["H4.dd"]["statistic"] == "drop_difference_mean"
    assert con["H4.dd"]["role"] == ROLE_SCORED and con["H4.dd.iqm"]["role"] == ROLE_POSTHOC
    assert con["H4.dd.iqm"]["statistic"] == "drop_difference"
    assert con["H2.dd"]["statistic"] == "drop_difference"
    assert con["H2.dd"]["resampling"].startswith("independent_cells")
    assert "P7-D1a #4" in con["H2.dd"]["resampling"]
    assert con["H4.dd"]["resampling"].startswith("shared_episodes")


def test_scored_numbers_are_recomputable_by_hand(tree: tuple[Path, Path]) -> None:
    from rld.eval.hypotheses import EpisodeStore

    results, e07 = tree
    contrasts, hyps, _ = compute(e07, results, reps=REPS, repo_root=results.parent)
    con = _by(contrasts, "contrast")
    store = EpisodeStore(results.parent)
    res6 = store.cell(e07 / "matrix" / "episodes.csv", "residual_ppo", "aft", "id", "SS6")
    ff6 = store.cell(results / "e01" / "episodes.csv", "pid_feedforward", "aft", "id", "SS6")
    want = paired_bootstrap(res6.success, ff6.success, reps=REPS)
    assert _f(con["H1b.d"], "point") == want.diff == res6.success.mean() - ff6.success.mean()
    assert (_f(con["H1b.d"], "ci_lo"), _f(con["H1b.d"], "ci_hi")) == (want.lo, want.hi)
    sin = e07 / "sinusoid" / "episodes.csv"
    mat = e07 / "matrix" / "episodes.csv"
    cells = (
        store.cell(sin, "ppo_sinusoid", "aft", "id", "SS5").success,
        store.cell(mat, "ppo_sinusoid", "aft", "id", "SS5").success,
        store.cell(mat, "ppo", "aft", "id", "SS5").success,
        store.cell(sin, "ppo", "aft", "id", "SS5").success,
    )
    h4 = drop_difference_bootstrap(*cells, mode="shared_episodes", reps=REPS, estimator="mean")
    hand = (cells[0].mean() - cells[1].mean()) - (cells[2].mean() - cells[3].mean())
    assert _f(con["H4.dd"], "point") == h4.statistic == pytest.approx(hand, abs=1e-12)
    assert _f(con["H4.dd"], "ci_lo") == h4.lo
    h1a = _by([h for h in hyps if h["role"] == ROLE_SCORED], "hypothesis")["H1a"]
    assert _f(h1a, "non_inferiority_lo") == _f(con["H1a.ni"], "ci_lo")


def test_verdicts_are_the_rule_functions_and_h3_has_no_combined_row(
    tree: tuple[Path, Path],
) -> None:
    results, e07 = tree
    contrasts, hyps, _ = compute(e07, results, reps=REPS, repo_root=results.parent)
    con = _by(contrasts, "contrast")
    parts = [(h["hypothesis"], h["part"], h["role"]) for h in hyps]
    assert not any("combined" in p or "combined" in r for _, p, r in parts)
    h3 = [h for h in hyps if h["hypothesis"] == "H3"]
    assert len(h3) == 8 and len({h["part"] for h in h3}) == 8
    assert {h["role"] for h in h3} == {ROLE_SCORED, ROLE_SECONDARY}
    for h in hyps:
        if h["role"] not in (ROLE_SCORED, ROLE_SECONDARY):
            continue
        point, lo, hi = _f(h, "point"), _f(h, "ci_lo"), _f(h, "ci_hi")
        if h["hypothesis"] == "H1a":
            want = verdict_h1a(point, lo, hi, _f(h, "non_inferiority_lo"))
        elif h["hypothesis"] in ("H2", "H4"):
            want = verdict_p7d1a(point, lo, hi, 0.10)
            assert "P7-D1a #1" in h["rule_source"]
        elif h["hypothesis"] == "H1b":
            want = verdict_magnitude(point, lo, hi, 0.05)
        elif "half-rule" in h["part"]:
            uv_id, id_id = h["contrasts"].split(";")
            assert _f(con[uv_id], "point") == point
            r_id = con[id_id]
            id_verdict = verdict_magnitude(
                _f(r_id, "point"), _f(r_id, "ci_lo"), _f(r_id, "ci_hi"), 0.10
            )
            want = verdict_half_rule(_f(r_id, "point"), point, id_verdict)
        else:
            want = verdict_magnitude(point, lo, hi, 0.10)
        assert h["verdict"] == want, h
        assert "shared-bootstrap-seed" in h["caveats"].split(";"), h["hypothesis"]
        assert "no-multiplicity" in h["caveats"].split(";"), h["hypothesis"]
    posthoc = [h for h in hyps if h["role"] == ROLE_POSTHOC]
    assert {h["hypothesis"] for h in posthoc} == {"H1a", "H1b", "H4"}
    assert all(h["verdict"] == NOT_SCORED for h in posthoc)
    h4 = next(h for h in hyps if h["hypothesis"] == "H4" and h["role"] == ROLE_SCORED)
    assert "Novelty claim withdrawn" in h4["note"] and "D0.4" in h4["note"]
    assert {h["verdict"] for h in hyps} <= {
        SUPPORTED,
        INCONCLUSIVE,
        NOT_SUPPORTED,
        NOT_SCORED,
        "not scored (no supported id gain)",
        "not applicable — no id gain to shrink",
        "pending — scored at Gate 8",
    }


def test_compressed_episodes_give_byte_identical_outputs(tree: tuple[Path, Path]) -> None:
    results, e07 = tree
    write_hypotheses(e07, results, reps=REPS)
    before = {n: (e07 / n).read_bytes() for n in ("contrasts.csv", "hypotheses.csv")}
    for cond in ("matrix", "sinusoid"):
        facts = storage.compress_csv(e07 / cond / "episodes.csv")
        assert facts["removed_source"] and not (e07 / cond / "episodes.csv").exists()
    assert check_hypotheses(e07, results, reps=REPS)
    write_hypotheses(e07, results, reps=REPS)
    after = {n: (e07 / n).read_bytes() for n in ("contrasts.csv", "hypotheses.csv")}
    assert after == before
    assert b".csv.gz" not in after["contrasts.csv"]  # sources name the logical episodes.csv
