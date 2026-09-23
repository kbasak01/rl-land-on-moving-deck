"""``resolved_gains_sha256``: the hash follows inherited gains, not just the YAML bytes.

Units: gains in the units their key names state; nothing here is flown.
"""

import hashlib
import json
import shutil
from pathlib import Path

import yaml

from rld.control.config import CONTROL_CONFIG_DIR, load_feedforward
from rld.control.registry import REGISTRY
from rld.eval.controller_config import (
    controller_provenance_columns,
    resolved_config_json,
    resolved_config_sha256,
)


def test_every_controller_resolves_to_a_distinct_stable_hash() -> None:
    hashes = {name: resolved_config_sha256(name) for name in REGISTRY}
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in hashes.values())
    assert len(set(hashes.values())) == len(REGISTRY)
    assert hashes == {name: resolved_config_sha256(name) for name in REGISTRY}
    for name in REGISTRY:
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
    columns = controller_provenance_columns(list(REGISTRY))
    assert list(columns) == list(REGISTRY)
    for name, cols in columns.items():
        assert list(cols) == ["controller_config_sha256", "resolved_gains_sha256"]
        assert (
            cols["controller_config_sha256"]
            == hashlib.sha256(REGISTRY[name].config_path.read_bytes()).hexdigest()
        )
        assert cols["resolved_gains_sha256"] == resolved_config_sha256(name)
