"""``resolved_gains_sha256``: the hash follows inherited gains, not just the YAML bytes.

Units: gains in the units their key names state; nothing here is flown.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from rld.control.config import CONTROL_CONFIG_DIR, load_feedforward, load_gated_forecast
from rld.control.registry import REGISTRY
from rld.eval.controller_config import (
    PROVENANCE_COLUMNS,
    controller_provenance_columns,
    forecaster_provenance,
    resolved_config_json,
    resolved_config_sha256,
)


def _fitted(name: str) -> bool:
    """Whether a registry controller's forecaster (if any) is fitted on this machine."""
    if not REGISTRY[name].needs_motion_feed:
        return True
    cfg = load_gated_forecast(REGISTRY[name].config_path)
    return (cfg.forecaster_dir / "meta.json").is_file()


#: Controllers whose resolved config can be built here (a forecaster may still be training).
AVAILABLE = [name for name in REGISTRY if _fitted(name)]


def test_every_controller_resolves_to_a_distinct_stable_hash() -> None:
    hashes = {name: resolved_config_sha256(name) for name in AVAILABLE}
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in hashes.values())
    assert len(set(hashes.values())) == len(AVAILABLE)
    assert hashes == {name: resolved_config_sha256(name) for name in AVAILABLE}
    for name in AVAILABLE:
        text = resolved_config_json(name)
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == hashes[name]
        assert str(CONTROL_CONFIG_DIR.resolve()) not in text  # no absolute paths


def test_gated_controllers_carry_pid_feedforwards_gains() -> None:
    ff = load_feedforward(CONTROL_CONFIG_DIR / "pid_feedforward.yaml")
    for name in ("gated", "oracle_gated"):
        doc = json.loads(resolved_config_json(name))
        assert doc["config"]["gains_from"] == "pid_feedforward.yaml"
        assert doc["config"]["gains"]["k_ff"] == ff.k_ff
        assert doc["config"]["gains"]["pid"]["kp_xy_per_s"] == ff.pid.kp_xy_per_s
        assert doc["quiescence_rule"]["n_samples"] == 12
    lowvz = json.loads(resolved_config_json("pid_feedforward_lowvz"))
    assert lowvz["class"] == json.loads(resolved_config_json("pid_feedforward"))["class"]
    assert lowvz["config"] != json.loads(resolved_config_json("pid_feedforward"))["config"]


def test_an_inherited_gain_change_moves_the_resolved_hash(tmp_path: Path) -> None:
    for name in ("gated.yaml", "pid_feedforward.yaml"):
        shutil.copy(CONTROL_CONFIG_DIR / name, tmp_path / name)
    gated = tmp_path / "gated.yaml"
    # An identical copy elsewhere resolves to the committed hash (paths are not hashed).
    assert resolved_config_sha256("gated", config_path=gated) == resolved_config_sha256("gated")
    raw = yaml.safe_load((tmp_path / "pid_feedforward.yaml").read_text(encoding="utf-8"))
    raw["kp_xy_per_s"] = float(raw["kp_xy_per_s"]) + 0.1
    (tmp_path / "pid_feedforward.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    # gated.yaml's bytes did not change; the resolved hash did.
    assert gated.read_bytes() == (CONTROL_CONFIG_DIR / "gated.yaml").read_bytes()
    assert resolved_config_sha256("gated", config_path=gated) != resolved_config_sha256("gated")


def test_provenance_columns() -> None:
    columns = controller_provenance_columns(AVAILABLE)
    assert list(columns) == AVAILABLE
    for name, cols in columns.items():
        assert tuple(cols) == PROVENANCE_COLUMNS
        assert (
            cols["controller_config_sha256"]
            == hashlib.sha256(REGISTRY[name].config_path.read_bytes()).hexdigest()
        )
        assert cols["resolved_gains_sha256"] == resolved_config_sha256(name)
        assert (cols["forecaster_files_sha256"] == "") is (not REGISTRY[name].needs_motion_feed)


def test_phase3_resolved_hashes_are_untouched_by_the_forecaster_block() -> None:
    """The five Phase 3 documents carry no forecaster key, so e01's hashes still verify."""
    for name in ("pid_track_descend", "pid_feedforward", "pid_feedforward_lowvz", "gated"):
        assert "forecaster" not in json.loads(resolved_config_json(name))
    assert "forecaster" not in json.loads(resolved_config_json("oracle_gated"))


def _fake_model_dir(root: Path, files: dict[str, bytes], smoke: bool = False) -> Path:
    model = root / "tcn_quantile"
    model.mkdir()
    for name, data in files.items():
        (model / name).write_bytes(data)
    meta = {
        "model": "tcn_quantile",
        "smoke": smoke,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    (model / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return model


def test_forecaster_provenance_hashes_every_listed_file(tmp_path: Path) -> None:
    files = {
        "state_dict.pt": b"ckpt",
        "model.onnx": b"onnx",
        "norm_stats.npz": b"norm",
        "conformal_padvz.npz": b"gamma",
        "some_future_file.bin": b"new",  # handled generically
    }
    model = _fake_model_dir(tmp_path, files, smoke=True)
    prov = forecaster_provenance(model)
    assert prov["dir"] == "tcn_quantile" and prov["smoke"] is True
    assert list(prov["files"]) == sorted(files)
    assert prov["files"]["conformal_padvz.npz"] == hashlib.sha256(b"gamma").hexdigest()
    assert (
        prov["meta_json_sha256"] == hashlib.sha256((model / "meta.json").read_bytes()).hexdigest()
    )
    (model / "model.onnx").write_bytes(b"refit")
    with pytest.raises(ValueError, match="model.onnx"):
        forecaster_provenance(model)
    with pytest.raises(FileNotFoundError, match="meta.json"):
        forecaster_provenance(tmp_path / "absent")


def test_a_forecaster_file_moves_the_resolved_hash(tmp_path: Path) -> None:
    """Refitting the forecaster (same YAML bytes) moves resolved_gains_sha256."""
    files = {"model.onnx": b"a", "norm_stats.npz": b"n", "conformal_padvz.npz": b"g"}
    model = _fake_model_dir(tmp_path, files)
    for name in ("gated_forecast_tcn.yaml", "pid_feedforward.yaml"):
        shutil.copy(CONTROL_CONFIG_DIR / name, tmp_path / name)
    yaml_path = tmp_path / "gated_forecast_tcn.yaml"
    text = yaml_path.read_text(encoding="utf-8").replace(
        "forecaster_dir: artifacts/dmf/tcn_quantile", f"forecaster_dir: {model}"
    )
    yaml_path.write_text(text, encoding="utf-8")
    first = resolved_config_sha256("gated_forecast_tcn", config_path=yaml_path)
    doc = json.loads(resolved_config_json("gated_forecast_tcn", config_path=yaml_path))
    assert set(doc["forecaster"]["files"]) == set(files)
    (model / "conformal_padvz.npz").write_bytes(b"g2")
    meta = json.loads((model / "meta.json").read_text(encoding="utf-8"))
    meta["files"]["conformal_padvz.npz"] = hashlib.sha256(b"g2").hexdigest()
    (model / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert resolved_config_sha256("gated_forecast_tcn", config_path=yaml_path) != first
    cols = controller_provenance_columns(
        ["gated_forecast_tcn"], config_paths={"gated_forecast_tcn": yaml_path}
    )["gated_forecast_tcn"]
    assert (
        "conformal_padvz.npz=" + hashlib.sha256(b"g2").hexdigest()
        in cols["forecaster_files_sha256"]
    )
