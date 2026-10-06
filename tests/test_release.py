"""Release pins (plan §7, audit SHOULD FIX 2): the submodules are the commits P0-D1 recorded.

``tests/test_bridge.py`` checks the bridge against whatever dmf is installed. This file pins *which*
dmf (and gym-pybullet-drones) that is, and that the import resolves to the editable submodule
rather than to some other copy on the path.
"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Submodule commits recorded in P0-D1 (``docs/protocol.md``) and in every provenance block.
PINNED: dict[str, str] = {
    "third_party/deck-motion-forecast": "e9fa15cc35312a5c1ecf0c42668a690b5061c090",
    "third_party/gym-pybullet-drones": "7ebad1ecabd28a7000add2d05f888aa2e837c2cc",
}


@pytest.mark.parametrize("path", sorted(PINNED))
def test_submodule_is_at_the_pinned_commit(path: str) -> None:
    head = subprocess.run(
        ["git", "-C", str(REPO / path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == PINNED[path]
    assert PINNED[path] in (REPO / "docs" / "protocol.md").read_text(encoding="utf-8")


def test_dmf_and_gpd_import_from_the_submodules() -> None:
    import dmf
    import gym_pybullet_drones

    assert (
        Path(dmf.__file__).resolve().is_relative_to(REPO / "third_party" / "deck-motion-forecast")
    )
    assert (
        Path(gym_pybullet_drones.__file__)
        .resolve()
        .is_relative_to(REPO / "third_party" / "gym-pybullet-drones")
    )
