"""Phase 7 review fixes (results-skeptic, P7-D4).

Superseded record, read-only verification, tunnelled successes, and the results.md tables the
review found incomplete.

What is pinned here:

* **P7-D4 storage.** ``results/e07/noise_superseded_p7d1/`` holds the 11 superseded noise
  conditions byte for byte as P7-D2 §5 recorded them; a changed byte is caught; nothing in
  the arm code treats it as a condition (it is never flown or re-derived).
* **m8.** A verify-only invocation writes nothing (no ``run_info.json`` append, no rewrite),
  needs no checkpoint, and a condition that would need flying stops before anything is
  written when the checkpoints are absent.
* **M4.** ``tunnelled_success.csv`` counts success and tunnelled per cell; its totals agree
  with every committed summary's ``n_success`` and ``tunnelling_n``.
* **m4 / m5 / M2 / M4 in results.md.** The three always-printed baselines are in every
  Appendix A table; the ``unseen_vessel`` heading subset has Wilson CIs and an outcome
  breakdown; section 4 states what the P7-D4 stand-in delays and noises and the effective
  latency in steps and ms; the superseded arm is a labelled note, not a table; every outcome
  breakdown carries the tunnelled-success column; the committed file re-renders identically.

Every test reads the committed ``results/`` read-only (copies go to ``tmp_path``).

Units: none (bytes, counts).
"""

import csv
import hashlib
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from rld.eval import arms
from rld.eval.arms import ALWAYS_PRINTED, E07_DIR, RESULTS_DIR
from rld.eval.envs import EvalConfigs
from rld.eval.superseded import (
    SUPERSEDED_NOISE_DIR,
    SUPERSEDED_NOISE_SHA256,
    verify_superseded_noise,
)

_WRITTEN = ("matrix", "cg", "sinusoid", "lambda/lam15", "lambda/lam40", "mss")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------------------- P7-D4 storage


def test_superseded_noise_is_byte_identical_to_p7_d2() -> None:
    result = verify_superseded_noise(E07_DIR)
    assert set(result) == {SUPERSEDED_NOISE_DIR} | {
        f"{SUPERSEDED_NOISE_DIR}/{name}" for name in SUPERSEDED_NOISE_SHA256
    }
    bad = {k: [c for c, ok in v.items() if not ok] for k, v in result.items()}
    assert not any(bad.values()), bad
    assert len(SUPERSEDED_NOISE_SHA256) == 11
    # The superseded record is not a condition of any arm: nothing flies or re-derives it.
    keys = {c.key for arm in arms.ARMS for c in arms.conditions(arm)}
    assert not any(k.startswith(SUPERSEDED_NOISE_DIR) for k in keys)
    assert {c.out_dir().parent.name for c in arms.conditions("noise")} == {"noise"}


def test_superseded_noise_check_catches_a_changed_byte(tmp_path: Path) -> None:
    src = E07_DIR / SUPERSEDED_NOISE_DIR
    dst = tmp_path / SUPERSEDED_NOISE_DIR
    shutil.copytree(src, dst, copy_function=os.symlink)  # read-only view of the record
    assert all(all(v.values()) for v in verify_superseded_noise(tmp_path).values())
    target = dst / "sigma2cm_lat1step" / "aggregate.csv"
    data = target.read_bytes()
    target.unlink()  # replace the link, never the committed file
    target.write_bytes(data.replace(b"0.", b"1.", 1))
    result = verify_superseded_noise(tmp_path)
    assert not result[f"{SUPERSEDED_NOISE_DIR}/sigma2cm_lat1step"]["aggregate.csv = P7-D2"]
    (dst / "sigma2cm_lat1step" / "episodes.csv").write_text("x\n", encoding="utf-8")
    result = verify_superseded_noise(tmp_path)
    assert not result[f"{SUPERSEDED_NOISE_DIR}/sigma2cm_lat1step"]["exactly the committed files"]
    assert verify_superseded_noise(tmp_path / "absent") == {
        SUPERSEDED_NOISE_DIR: {"present": False}
    }


# --------------------------------------------------------------------------- M4 tunnelled


def _ep(method: str, seed: str, ss: str, outcome: str, tunnelled: bool) -> dict[str, str]:
    return {
        "method": method,
        "privileged": "False",
        "run_seed": seed,
        "pad": "aft",
        "regime": "id",
        "ss": ss,
        "outcome": outcome,
        "tunnelled": str(tunnelled),
    }


def test_derive_tunnelled_counts_success_and_tunnelled() -> None:
    from rld.eval.phase7 import TUNNELLED_COLUMNS, derive_tunnelled

    rows = [
        _ep("ppo", "0", "SS5", "success", True),
        _ep("ppo", "0", "SS5", "success", False),
        _ep("ppo", "0", "SS5", "hard_landing", True),
        _ep("ppo", "1", "SS5", "success", True),
        _ep("ppo", "0", "SS6", "bounce", False),
    ]
    cfgs = cast(
        EvalConfigs, SimpleNamespace(success=SimpleNamespace(tunnelling_penetration_m=0.005))
    )
    cond = arms.conditions("sinusoid")[0]
    text = derive_tunnelled(cond, rows, [], cfgs, carried=False)
    lines = text.splitlines()
    assert lines[0] == ",".join(TUNNELLED_COLUMNS)
    assert lines[1:] == [
        "ppo,False,0,aft,id,SS5,flown here,3,2,2,1,0.005",
        "ppo,False,1,aft,id,SS5,flown here,1,1,1,1,0.005",
        "ppo,False,0,aft,id,SS6,flown here,1,0,0,0,0.005",
    ]


@pytest.mark.parametrize("key", _WRITTEN)
def test_committed_tunnelled_agrees_with_the_summaries(key: str) -> None:
    """Totals per cell equal the summaries' n_success and tunnelling_n (flown and carried)."""
    root = E07_DIR / key
    tun: dict[tuple[str, ...], list[int]] = {}
    for r in _read(root / "tunnelled_success.csv"):
        cell = (r["method"], r["pad"], r["regime"], r["ss"])
        got = tun.setdefault(cell, [0, 0, 0])
        got[0] += int(r["n_success"])
        got[1] += int(r["n_tunnelled"])
        got[2] += int(r["n_success_tunnelled"])
        assert 0 <= int(r["n_success_tunnelled"]) <= min(int(r["n_success"]), int(r["n_tunnelled"]))
    want: dict[tuple[str, ...], list[int]] = {}
    files = [root / "summary.csv", root / "baselines_summary.csv", *root.glob("carried_summary_*")]
    for path in files:
        if not path.is_file():
            continue
        for r in _read(path):
            if int(r["n_episodes"]) == 0:
                continue
            cell = (r["method"], r.get("pad") or "aft", r["regime"], r["ss"])
            got = want.setdefault(cell, [0, 0])
            got[0] += int(r["n_success"])
            got[1] += int(r["tunnelling_n"])
    assert {k: v[:2] for k, v in tun.items()} == want
    for method in ALWAYS_PRINTED:  # the always-printed baselines are counted in every arm
        assert any(k[0] == method for k in tun), method


# --------------------------------------------------------------------------- m8 read-only


@pytest.mark.slow
def test_verify_only_invocations_write_nothing_and_need_no_checkpoint(tmp_path: Path) -> None:
    from rld.eval.phase7 import main

    root = tmp_path / "e07"
    shutil.copytree(E07_DIR / "sinusoid", root / "sinusoid")
    before = _tree_digest(root)
    no_runs = ["--runs-root", str(tmp_path / "no_runs")]
    common = ["--arm", "sinusoid", "--out-root", str(root)]
    assert main([*common, *no_runs]) == 0  # verify-only flight invocation
    assert main([*common, "--check"]) == 0
    assert main([*common, "--compress"]) == 0  # already compressed
    assert main([*common, "--add-tunnelled"]) == 0  # already present
    assert _tree_digest(root) == before
    assert not (root / "run_info.json").exists()
    # A condition that needs flying stops before writing when the checkpoints are absent.
    shutil.copytree(E07_DIR / "lambda" / "lam15", root / "lambda" / "lam15")
    before = _tree_digest(root)
    with pytest.raises(SystemExit, match="checkpoint files"):
        main(["--arm", "lambda", "--out-root", str(root), *no_runs])
    assert _tree_digest(root) == before
    assert (
        main(["--arm", "lambda", "--condition", "lambda/lam15", "--check", "--out-root", str(root)])
        == 0
    )
    assert (
        main(["--arm", "lambda", "--condition", "lambda/lam40", "--check", "--out-root", str(root)])
        == 1
    )  # not written: the check fails rather than passing silently


# --------------------------------------------------------------------------- results.md


@pytest.fixture(scope="module")
def results_md() -> str:
    return (RESULTS_DIR / "results.md").read_text(encoding="utf-8")


def _section(text: str, start: str, stop: str = "\n## ") -> str:
    i = text.index(start)
    j = text.find(stop, i + len(start))
    return text[i : j if j >= 0 else len(text)]


def test_committed_results_md_rerenders_identically(results_md: str) -> None:
    from rld.eval.report import render_results

    assert render_results(RESULTS_DIR, E07_DIR) == results_md


def test_appendix_a_prints_the_always_printed_baselines_in_every_table(results_md: str) -> None:
    appendix = _section(results_md, "## Appendix A.")
    tables = appendix.split("\n### ")[1:]
    assert len(tables) == 5
    for table in tables:
        for method in ALWAYS_PRINTED:
            assert f"| `{method}`" in table, (method, table[:40])
        assert "| `oracle_gated` — commit-timing oracle (privileged) | – |" in table


def test_unseen_vessel_heading_subset_has_wilson_and_breakdown(results_md: str) -> None:
    block = _section(results_md, "### `unseen_vessel` SS5, headings 180° and 135° only", "\n## ")
    assert "| method | seed | success [Wilson 95 % CI] k/N |" in block
    assert "| `pid_feedforward` | – | 100.0 [96.2, 100.0] 97/97 |" in block
    assert (
        "| method | N (seed-episodes) | crash | off_pad | hard_landing | bounce | success |"
        in block
    )
    assert "success ∧ tunnelled (> 5 mm)" in block
    for method in ALWAYS_PRINTED:
        assert f"| `{method}`" in block


def test_noise_section_states_the_p7_d4_stand_in_and_latency(results_md: str) -> None:
    sec = _section(results_md, "## 4. Perception stand-in")
    assert "P7-D4" in sec
    assert "Pad **position and velocity**: delayed by L and **noisy**" in sec
    assert "Pad **orientation and deck normal**: delayed by L, **no noise**" in sec
    assert "**own state** (attitude, rates, velocity), time, last action and contact" in sec
    assert "effective (control steps)" in sec and "effective (ms, model)" in sec
    assert "| `noise/sigma0cm_lat2step` | 0 | 0 | 66.7 | 2 | 66.7 | 333.3 |" in sec
    assert "*Superseded (P7-D4):*" in sec
    # The superseded arm is a note, never a table: no row or heading names its conditions.
    assert f"{SUPERSEDED_NOISE_DIR}/sigma" not in results_md
    assert results_md.count(SUPERSEDED_NOISE_DIR) == 1


def test_every_outcome_breakdown_has_the_tunnelled_column(results_md: str) -> None:
    headers = [
        line
        for line in results_md.splitlines()
        if line.startswith("| method |") and "| crash |" in line and "| timeout |" in line
    ]
    assert headers
    assert all(line.endswith("| success ∧ tunnelled (> 5 mm) |") for line in headers)


def test_hypotheses_use_p3_d1_words_in_results_md(results_md: str) -> None:
    sec = _section(results_md, "## 7. Hypotheses")
    assert "**holds**" not in sec and "**fails**" not in sec
    assert "**not scored (no supported id gain)**" in sec


def test_header_names_the_phase4_forecast_gated_controllers(results_md: str) -> None:
    """Gate 9 review M1: the two Phase 4 forecast-gated controllers are named, and where."""
    header = results_md[: results_md.index("\n## 1. ")]
    (bullet,) = [x for x in header.splitlines() if x.startswith("- `gated_forecast` ")]
    assert "`gated_forecast_tcn`" in bullet and "no Phase 7 arm" in bullet
    assert "`results/e02/success_vs_seastate.md` (P4-D4)" in bullet
    assert "gated_forecast" not in results_md[len(header) :]  # not in any table of this report


def test_h5_is_rendered_from_latency_h5_csv(results_md: str) -> None:
    """Phase 9 (a): H5 comes from ``results/latency/h5.csv``; Phase 7's CSV stays "pending"."""
    sec = _section(results_md, "## 7. Hypotheses")
    h5_lines = [line for line in sec.splitlines() if line.startswith("| H5 |")]
    assert len(h5_lines) == 2  # the scored row and the torch-eager CUDA context row
    scored, context = h5_lines
    assert "| scored |" in scored and "| **supported** |" in scored
    for text in ("ORT CUDA 4.72× (0.217 vs 0.046 ms)", "ORT TensorRT 3.94× (0.181 vs 0.046 ms)"):
        assert text in scored
    assert "ORT CUDA 5.57×, ORT TensorRT 6.97×" in scored and "≥ 2×" in scored
    assert "one measurement per configuration, no CI" in scored
    assert "`results/latency/h5.csv`" in scored and "P8-D2" in scored
    # MINOR 17: H5's caveat says what "parity-passing" means, on every H5 row.
    for line in h5_lines:
        caveats = line.rstrip(" |").rsplit(" | ", 1)[1]
        assert caveats.startswith("parity-passing = numeric parity only (P8-D1 §6)")
        assert "P8-D1 §7 not met" in caveats and "P8-D5 met for ORT CPU only" in caveats
    # MINOR 17: the intro names both scorers, and the header names P8-D2.
    assert sec.startswith("## 7. Hypotheses (P3-D1 §8, P3-D4, P6-D6, P7-D1 §2, P7-D1a, P8-D2)\n")
    assert "Every verdict is computed by" not in sec
    assert (
        "H1a–H4 verdicts are computed by `rld.eval.hypotheses` (`results/e07/hypotheses.csv`)"
        in sec
    )
    assert "H5's by `rld.deploy.latency.score_h5` (`results/latency/h5.csv`, P8-D2 §7)" in sec
    assert "| context (not scored) |" in context and "torch:cuda" in context
    assert "| **not scored** |" in context and "**supported**" not in context
    assert "pending — scored at Gate 8" not in results_md
    note = next(line for line in sec.splitlines() if line.startswith("- H5,"))
    assert 'keeps its Phase 7 "pending" H5 row unedited' in note
    # The committed Phase 7 file is not edited: its H5 row still reads "pending".
    with (E07_DIR / "hypotheses.csv").open(encoding="utf-8", newline="") as handle:
        (h5,) = [r for r in csv.DictReader(handle) if r["hypothesis"] == "H5"]
    assert h5["verdict"] == "pending — scored at Gate 8"


def test_h5_rows_are_formatted_from_the_csv_not_hardcoded(tmp_path: Path) -> None:
    """Different numbers in ``h5.csv`` give different rows; no ``h5.csv`` keeps the pending row."""
    from rld.eval.results_md import _hypotheses_section

    src = RESULTS_DIR / "latency" / "h5.csv"
    with src.open(encoding="utf-8", newline="") as handle:
        recs = list(csv.DictReader(handle))
        fields = list(recs[0])
    for r in recs:
        if r["verdict"] != "context (not scored)":
            r["verdict"] = "not supported"
    recs[0].update(other_p50_ms="0.07", ratio_p50="1.5", ratio_p99="1.25", meets_threshold="False")
    h5 = tmp_path / "h5.csv"
    with h5.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(recs)
    sec = "\n".join(_hypotheses_section(E07_DIR, h5))
    (scored,) = [x for x in sec.splitlines() if x.startswith("| H5 |") and "| scored |" in x]
    assert "ORT CUDA 1.50× (0.070 vs 0.046 ms)" in scored and "ORT CUDA 1.25×" in scored
    assert "| **not supported** |" in scored
    without = "\n".join(_hypotheses_section(E07_DIR, tmp_path / "missing.csv"))
    assert "**pending — scored at Gate 8**" in without and "latency/h5.csv" not in without
    assert without.startswith("## 7. Hypotheses (P3-D1 §8, P3-D4, P6-D6, P7-D1 §2, P7-D1a)\n")
    assert "Every verdict is computed by `rld.eval.hypotheses`" in without
    assert "score_h5" not in without and "parity-passing" not in without


def test_noise_flight_refuses_the_superseded_stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """P7-D4: the noise arm is never flown under the Phase 2 stand-in (``apply`` only)."""
    from rld.eval import phase7

    class Phase2StandIn:
        def apply(self) -> None: ...

    noise = arms.conditions("noise")
    phase7._noise_preflight(arms.conditions("sinusoid"), committed=False)  # not noise: no-op
    monkeypatch.setattr(phase7, "PerceptionNoise", Phase2StandIn)
    with pytest.raises(SystemExit, match="perceive"):
        phase7._noise_preflight(noise, committed=False)
    phase7._noise_preflight(arms.conditions("matrix"), committed=True)  # other arms unaffected
