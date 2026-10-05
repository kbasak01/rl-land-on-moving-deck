"""The real checkpoints (P8-D1 §3): export, sidecar, CPU parity, and one ONNX-flown episode.

Skipped when ``artifacts/runs/<method>/4`` is absent (artifacts are gitignored). When
``results/latency/run_info.json`` exists, the re-export must reproduce its graph SHA-256s.

Units: actions normalised in [-1, 1]; speeds metres per second model scale.
"""

import functools
import json
from pathlib import Path

import pytest

from conftest import REPO_ROOT
from rld.deploy.export import export_run, normalizer_constants
from rld.deploy.onnx_policy import build_onnx_policy
from rld.deploy.parity import open_session, parity_draw, reference_actions
from rld.eval.envs import load_eval_configs
from rld.eval.learned import inspect_run
from rld.eval.runner import callable_spec, run_list
from rld.rl.train import build_policy

RUNS = REPO_ROOT / "artifacts" / "runs"
METHODS = ("ppo", "residual_ppo_forecast")

pytestmark = [pytest.mark.slow, pytest.mark.pybullet]


def _need(method: str) -> Path:
    run = RUNS / method / "4"
    if not (run / "final" / "model.zip").exists():
        pytest.skip(f"{run} absent (artifacts are gitignored)")
    return run


@pytest.mark.parametrize("method", METHODS)
def test_export_sidecar_and_cpu_parity(method: str, tmp_path: Path) -> None:
    run_dir = _need(method)
    graph = export_run(method, 4, out_dir=tmp_path)
    side = json.loads(graph.sidecar_path.read_text())
    run = inspect_run(method, run_dir.resolve())
    digests = dict(run.digests)
    assert side["model_sha256"] == digests["run_model_sha256"]
    assert side["vecnormalize_sha256"] == digests["run_vecnormalize_sha256"]
    assert side["obs_dim"] == (25 if method == "ppo" else 31) == graph.obs_dim
    assert graph.architecture == f"mlp512x2-tanh-in{graph.obs_dim}"
    info = REPO_ROOT / "results" / "latency" / "run_info.json"
    if info.exists():
        recorded = {g["file"]: g["onnx_sha256"] for g in json.loads(info.read_text())["graphs"]}
        assert recorded[graph.onnx_path.name] == graph.onnx_sha256
    policy = build_policy(load_eval_configs(), run_dir=run_dir.resolve())
    x, _, _ = parity_draw(normalizer_constants(policy), seed=0)
    session, _ = open_session(graph.onnx_path, "CPUExecutionProvider")
    (out,) = session.run([], {"input": x})
    assert abs(out - reference_actions(policy, x)).max() < 1e-4


def test_one_onnx_episode_through_the_runner(tmp_path: Path) -> None:
    run_dir = _need("residual_ppo_forecast").resolve()
    graph = export_run("residual_ppo_forecast", 4, out_dir=tmp_path)
    from rld.deploy.closed_loop import closed_loop_episodes

    episode = closed_loop_episodes()[0]
    spec = callable_spec(
        "residual_ppo_forecast",
        functools.partial(build_onnx_policy, run_dir=run_dir, onnx_path=graph.onnx_path),
        run_seed=4,
        needs_motion_feed=True,
    )
    (row,) = run_list([episode], spec, load_eval_configs(), workers=1)
    assert row["outcome"] in {"success", "hard_landing", "bounce", "off_pad", "crash", "timeout"}
    assert row["run_seed"] == 4
