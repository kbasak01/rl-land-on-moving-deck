"""Phase 9 figures, landing GIFs and the runtime table (P9-D1).

- The headline figure reads committed CSVs only and draws every method in every cell.
- A GIF re-flight reproduces its committed episode row column for column, and a doctored
  committed row is refused rather than relabelled.
- The pad marker is drawn on the plate, and moving it (visual only) changes nothing recorded.
- The committed gallery manifest and runtime table agree with their sources.

Units: success in percent; times in seconds; lengths in metres model scale.
"""

import csv
import math
from pathlib import Path

import numpy as np
import pybullet as pyb
import pytest

from rld.eval.envs import load_eval_configs
from rld.runtimes import collect
from rld.viz import gifs
from rld.viz.curves import PANELS, plot_success_vs_seastate, success_table
from rld.viz.gifs import Camera, Case, PanelSpec, ReflightMismatchError, fly, load_gallery

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RUNS = REPO / "artifacts" / "runs"
SMALL = Camera(width_px=64, height_px=48, supersample=1)
CASE = Case("test", "id", "SS5", 0, (PanelSpec("pid_feedforward"),))


def test_success_table_covers_every_method_and_cell() -> None:
    table = success_table(RESULTS)
    methods = {m for p in PANELS for m in p.methods}
    assert len(methods) == 12
    for regime, sea_states in (
        ("id", ("SS3", "SS4", "SS5", "SS6")),
        ("unseen_heading", ("SS3", "SS4", "SS5", "SS6")),
        ("unseen_vessel", ("SS3", "SS4", "SS5", "SS6")),
        ("unseen_seastate", ("SS6",)),
    ):
        cell = table[table["regime"] == regime]
        for method in methods:
            rows = cell[cell["method"] == method]
            assert sorted(rows["ss"]) == list(sea_states), (regime, method)
            assert (rows["lo"] <= rows["point"]).all() and (rows["point"] <= rows["hi"]).all()
    # Two values the README quotes (findings §2).
    ppo = table[(table["method"] == "ppo") & (table["regime"] == "id") & (table["ss"] == "SS6")]
    pff = table[
        (table["method"] == "pid_feedforward") & (table["regime"] == "id") & (table["ss"] == "SS6")
    ]
    assert round(float(ppo["point"].iloc[0]), 1) == 98.2
    assert round(float(pff["point"].iloc[0]), 1) == 90.5


def test_headline_figure_is_byte_reproducible(tmp_path: Path) -> None:
    out = plot_success_vs_seastate(RESULTS, tmp_path / "id.png")
    committed = RESULTS / "figures" / "success_vs_seastate_id.png"
    assert out.read_bytes() == committed.read_bytes()


@pytest.fixture(scope="module")
def cfgs():  # type: ignore[no-untyped-def]
    return load_eval_configs()


def test_pid_reflight_reproduces_its_committed_row(cfgs) -> None:  # type: ignore[no-untyped-def]
    flight = fly(
        CASE,
        CASE.panels[0],
        cfgs,
        results_dir=RESULTS,
        runs_dir=RUNS,
        camera=SMALL,
        frame_stride=10,
    )
    assert flight.committed["outcome"] == flight.record["outcome"] == "success"
    assert len(flight.frames) == len(flight.times_s) == len(flight.heights_m)
    assert flight.frames[0].shape == (48, 64, 3)
    # One frame per 10 control steps plus the first and the last.
    steps = int(flight.committed["steps"])
    assert len(flight.frames) == 1 + steps // 10 + (steps % 10 != 0)
    assert abs(flight.heights_m[-1]) < 0.05  # resting on the pad


def test_a_doctored_committed_row_is_refused(cfgs, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    real = gifs.committed_row

    def doctored(*args, **kwargs):  # type: ignore[no-untyped-def]
        row = real(*args, **kwargs)
        row["outcome"] = "bounce"
        return row

    monkeypatch.setattr(gifs, "committed_row", doctored)
    with pytest.raises(ReflightMismatchError, match="outcome"):
        fly(
            CASE,
            CASE.panels[0],
            cfgs,
            results_dir=RESULTS,
            runs_dir=RUNS,
            camera=SMALL,
            frame_stride=50,
        )


def test_pad_marker_is_drawn_on_the_plate(cfgs, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The visual-only marker is moved onto the plate before every frame."""
    seen: list[float] = []
    grab = Camera.grab

    def spy(self: Camera, client: int) -> np.ndarray:
        n = pyb.getNumBodies(physicsClientId=client)
        ids = [pyb.getBodyUniqueId(i, physicsClientId=client) for i in range(n)]
        markers = [
            b
            for b in ids
            if any(
                tuple(round(c, 2) for c in v[7]) == (0.95, 0.75, 0.15, 1.0)
                for v in pyb.getVisualShapeData(b, physicsClientId=client)
            )
        ]
        assert len(markers) == 1
        marker = markers[0]
        plate = marker - 1
        mp, _ = pyb.getBasePositionAndOrientation(marker, physicsClientId=client)
        pp, _ = pyb.getBasePositionAndOrientation(plate, physicsClientId=client)
        seen.append(float(np.linalg.norm(np.subtract(mp, pp))))
        return grab(self, client)

    monkeypatch.setattr(Camera, "grab", spy)
    fly(
        CASE,
        CASE.panels[0],
        cfgs,
        results_dir=RESULTS,
        runs_dir=RUNS,
        camera=SMALL,
        frame_stride=30,
    )
    assert seen and max(seen) < 0.002  # 1.2 mm along the deck normal


@pytest.mark.skipif(
    not (RUNS / "ppo" / "0" / "final" / "model.zip").exists(),
    reason="needs the gitignored ppo seed 0 checkpoint",
)
def test_learned_reflight_reproduces_its_committed_row(cfgs) -> None:  # type: ignore[no-untyped-def]
    panel = PanelSpec("ppo", 0)
    flight = fly(
        CASE, panel, cfgs, results_dir=RESULTS, runs_dir=RUNS, camera=SMALL, frame_stride=20
    )
    assert flight.record["outcome"] == flight.committed["outcome"]


def test_gallery_manifest_matches_the_committed_rows() -> None:
    """Every manifest row (what the README captions quote) is its committed episode row."""
    gallery = load_gallery(REPO / "configs" / "viz" / "gifs.yaml")
    manifest = RESULTS / "figures" / "gifs" / "manifest.csv"
    with manifest.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["slug"] for r in rows] == [c.slug for c in gallery.cases for _ in c.panels]
    for row in rows:
        seed = int(row["run_seed"])
        committed = gifs.committed_row(
            RESULTS, row["regime"], row["ss"], int(row["index"]), row["method"], seed
        )
        for col in ("outcome", "rel_vz_normal_m_s", "rel_tilt_deg", "steps"):
            assert row[col] == committed[col], (row["slug"], row["method"], col)
        case = next(c for c in gallery.cases if c.slug == row["slug"])
        k, n = gifs.seed_agreement(RESULTS, case, PanelSpec(row["method"], seed), row["outcome"])
        assert (int(row["seeds_with_outcome"]), int(row["seeds_flown"])) == (k, n)
        assert (RESULTS / "figures" / "gifs" / f"{row['slug']}.gif").exists()


def test_runtime_table_matches_its_committed_sources() -> None:
    """Every committed-source row of results/runtime_stages.csv re-collects to the same value."""
    path = RESULTS / "runtime_stages.csv"
    with path.open(encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    fresh = {(r.stage, r.item, r.source): r for r in collect(REPO) if r.committed}
    checked = 0
    for row in table:
        if row["committed"] != "True" or row["source"].startswith("measured"):
            continue
        ref = fresh[(row["stage"], row["item"], row["source"])]
        assert math.isclose(float(row["wall_s"]), round(ref.wall_s, 1), abs_tol=0.051)
        checked += 1
    assert checked > 0
    stages = {r["stage"] for r in table}
    for stage in ("deck-stats", "env-sanity", "baselines", "eval", "bench", "bench-investigate"):
        assert stage in stages


def test_closing_speed_table_counts_each_episode_once() -> None:
    """Baselines are not double-counted (e01_lowvz_cut re-carries e01); counts match outcomes."""
    from rld.viz.curves import closing_speed_table

    table = closing_speed_table(RESULTS)
    counts = table.groupby(["method", "ss"]).size()
    for (method, _ss), n in counts.items():
        assert n <= (200 if method.startswith(("pid_", "gated", "oracle_")) else 1000), method
    assert counts[("pid_feedforward", "SS5")] == 200
    assert counts[("pid_feedforward_lowvz_cut", "SS6")] == 200
    assert counts[("gated", "SS5")] == 179  # 21 timeouts never touch down (findings §2)
    assert counts[("sac", "SS6")] == 985
    # The README's exceedance shares.
    sac6 = table[(table["method"] == "sac") & (table["ss"] == "SS6")]["closing_m_s"]
    assert round(float((sac6 > 0.5).mean()), 3) == 0.234


def test_closing_speed_figure_is_byte_reproducible(tmp_path: Path) -> None:
    from rld.viz.curves import plot_closing_speed_ecdf

    out = plot_closing_speed_ecdf(RESULTS, tmp_path / "ecdf.png")
    assert out.read_bytes() == (RESULTS / "figures" / "closing_speed_ecdf_id.png").read_bytes()
