"""Superseded Phase 7 records: kept unchanged, verified by hash, never re-flown (P7-D4).

Authority: ``docs/protocol.md`` P7-D4 (2026-10-02). The perception stand-in arm flown at
``6b83e5c`` (P7-D2) replaced only the six relative entries with a delayed, noisy copy, so its
latency conditions mixed timestamps (review M1) and its clearance, deck normal and relative
tilt stayed ideal (M2). P7-D4 redefines the stand-in and re-flies the arm into
``results/e07/noise/``. The original arm is **not deleted**: it was moved unchanged, with
``git mv``, to ``results/e07/noise_superseded_p7d1/``.

That directory is a read-only record:

* nothing in :mod:`rld.eval.phase7` flies, re-derives, compresses or writes into it;
* ``scripts/eval_phase7.py --check`` (and a verify-only ``make eval``) compares its bytes with
  the SHA-256s P7-D2 §5 recorded at ``6b83e5c`` -- ``summary.csv``, ``aggregate.csv``, the
  uncompressed ``episodes.csv`` (the authority, P7-D1a §12) and ``episodes.csv.gz`` -- plus
  the ``episodes.csv.sha256`` sidecar and, for ``seeds.csv`` and ``baselines_summary.csv``
  (which P7-D2 covers through ``files_sha256``), the condition's own ``run_info.json``;
* it is not re-derived from its episodes: re-derivation would run the current code, whose
  condition columns describe the P7-D4 stand-in, against rows flown under the old one;
* :mod:`rld.eval.hypotheses` never reads it (no hypothesis reads the noise arm, P7-D4) and
  ``results/results.md`` shows it only as a labelled "superseded (P7-D4)" note.

Units: none (bytes and SHA-256 hex digests).
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from rld.eval import storage
from rld.eval.arms import E07_DIR

__all__ = [
    "SUPERSEDED_NOISE_DIR",
    "SUPERSEDED_NOISE_SHA256",
    "SUPERSEDED_NOTE",
    "verify_superseded_noise",
]

#: The superseded noise arm's directory under ``results/e07`` (P7-D4).
SUPERSEDED_NOISE_DIR: str = "noise_superseded_p7d1"

#: One line describing the record, for logs and ``results.md``.
SUPERSEDED_NOTE: str = (
    "superseded (P7-D4): the noise arm flown at 6b83e5c under the Phase 2 stand-in, which "
    "delayed and noised only the six relative entries (mixed timestamps; clearance, deck normal "
    "and relative tilt ideal). Kept unchanged for the record; hashes in P7-D2 §5; never re-flown"
)

#: P7-D2 §5, verbatim: condition -> (summary.csv, aggregate.csv, episodes.csv uncompressed,
#: episodes.csv.gz) SHA-256, as committed at ``05361a5`` (flown at ``6b83e5c``).
SUPERSEDED_NOISE_SHA256: dict[str, tuple[str, str, str, str]] = {
    "sigma1cm_lat0step": (
        "d2edf45870e5124ee1cdeee9f9bc73169d54bda635c294d025584654d77fc56a",
        "2c0063288b328c6b3243f073cf846f4e4594d64033086ea8c186f02fe24c5d68",
        "9dcf6941d9222f2f99d70e6820b21410f0bc69f29c560f2562bbced45ef3da1f",
        "03d091bfca35960639755de16edd41c2d2928b4b45b22facba0b748b09411c58",
    ),
    "sigma2cm_lat0step": (
        "845704cdfea07b9333f7e4fb2dc38753e639e03d77f1665bc278dbdf2ddae7c3",
        "9544d4296da55abeeab35292a425fb1114f81111ef6e02bcb2afb5ac15ea1f7d",
        "08faedbfb979774c008d242cc6c7a7169a99df5d963d287bed8c14c29b0dd1e9",
        "0de28870959ae8a042ef625b5ceb96a5b54b847d30e515810f6bdc674edf306f",
    ),
    "sigma4cm_lat0step": (
        "48e4c707191c71ead6b0fc67bbc3ff5da7067b6758eb8e6eb4e2d092c4dd2394",
        "f65076c685c3919722850561baa82ec7303d04e980c6dbc767591824fd4758a2",
        "03a2b24849316097d62c5df23edbfd464d463b21ffb527583f4dd8e01a90003f",
        "607f14d600ace4ecd5eb7805bb08229f681aa36382e7905827a8b4e65f82e251",
    ),
    "sigma0cm_lat1step": (
        "8da8a13c439cba44dbe2adf2403a80fcea999ab55f76a7f4d4d5022a073eb8e2",
        "90339ed2e85763ecda7f6019e295882e5b032e04104b25bf267131fbf249db0f",
        "fcefd562858d8e5ba3badfce52136863ef5061ed540223eea8552b90b62bc2e1",
        "16c5b2efdb2bb4173f2bbf3d8676895afd0cf5816e8fa00bf0660b0176a55fb6",
    ),
    "sigma1cm_lat1step": (
        "da80f43784668c107631a03cf724e2ca02f86ec338b3a525712da3c582b797e2",
        "025221b79f3b322960e933167143e3a659b4f740d96036c377580c70a32dc902",
        "c1b9d831243114cc38810d5662f83bf69e387be9a2b1cb3ba232c2e6590d1fdb",
        "4e9bad2d100a00acf668681bbc4e5e54d1226afa0cb14c85a4e9776b23e623aa",
    ),
    "sigma2cm_lat1step": (
        "6ad4b167f1edbad579e6d26214394b45e2a3cf6d560956001381aa8d24dfd9b5",
        "4d91e6b9244c5f67c19378b03ae5ae11fd56c1fc69bb8399decb3d2b74fb4b72",
        "deeecd1564f0b1f5afd7fb9baa74603dbd512b305316820b2499dd826a443c96",
        "f102d2293dfda7da80c899fba7becaaf18d23f6fe3518c9223c7f780bba107ed",
    ),
    "sigma4cm_lat1step": (
        "77a7395f1a084abc06ebf5e452240fd8fa2f10af5a56ffe775dddcf8d1f0a906",
        "06404f4965007afdb189d0fb362130a33d9c77fa4c1eacec96c8bdafd23baa49",
        "efe71388722b018db48114c62025df3ca680dcd5f8377d40923172c7007f1158",
        "7a5cd216622e614881220fb095443f9bc853c16691f44e151ae6a1592119cff6",
    ),
    "sigma0cm_lat2step": (
        "8d8b11adff40515de64bc8b789ccf30bf1733ea5feca110235c7e575e2952682",
        "33532960b5d862651a1ef45d1537fbc1a57cb62dfd3520151b5d08c1c921cba7",
        "70fb5eb6427165d436f555543ae1e920329cf92c81e9cd07b13275279d504f38",
        "0bfbe0ac215581556efe9b322fcf445d4ad1617762b09d6c79f6fe3d926a904d",
    ),
    "sigma1cm_lat2step": (
        "d48f57050defabf19594f2fb4c87450db672ec600f1f82956a506484e2f1cf79",
        "75e6a663cd5e92bec3f435ba2b74cbcfee3533f2970e1142be32f6eb0d3ae391",
        "57106556cbe21a4e5002f3d8ed57321f8dd24438a295e0158dc1366652519f17",
        "dd4c951bf3abb320e170b06b0c36e994d181637c291671e27cbc93ea124a68f7",
    ),
    "sigma2cm_lat2step": (
        "84c32e9c20f94682a1b01455d8112c7875e44f0305fb8c4a0a56ba923b0ac458",
        "cfd9a7ebe8536e9f5d95d6408e9afbd015302618cd0386bd4d18d08f22615ba8",
        "136d251f7894afe565d1b56e8ef1ac9c6af3e82f5b72a8ae52668e0d732287c3",
        "eae89ce387d2371e88143bea45cca1efe72544bbd274e0fb09ad0906718e8f0f",
    ),
    "sigma4cm_lat2step": (
        "2ee64db7e7cdbe5118e4fbe3c5b9d3b0ddd4b4b397ab96e025f9f59bbe37461f",
        "e6bf7911b85f5e8b10f1db1e590ef0d3f6cc2955badc6a0ef7c794449471923c",
        "42b114336793ac08e3e044d4226fefe763ec21029e4e9afbd7d34d1b72d89b8d",
        "245d498584ba74e638a1db943b50d63d395a278f5f6cf2a3f526ca7e056178e6",
    ),
}

#: Files every superseded condition holds, and nothing else.
_FILES: frozenset[str] = frozenset(
    {
        "summary.csv",
        "seeds.csv",
        "aggregate.csv",
        "baselines_summary.csv",
        "episodes.csv.gz",
        "episodes.csv.sha256",
        "run_info.json",
    }
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_one(directory: Path, name: str, hashes: tuple[str, str, str, str]) -> dict[str, bool]:
    """Return ``{check: ok}`` for one superseded condition directory."""
    summary, aggregate, plain, gz = hashes
    result: dict[str, bool] = {"present": directory.is_dir()}
    if not directory.is_dir():
        return result
    present = {p.name for p in directory.iterdir()}
    result["exactly the committed files"] = present == _FILES
    info_path = directory / "run_info.json"
    info: Mapping[str, object] = (
        json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    )
    recorded = info.get("files_sha256", {})
    recorded = recorded if isinstance(recorded, dict) else {}
    result["run_info.json names the condition"] = info.get("condition") == f"noise/{name}"
    result["run_info.json complete"] = info.get("complete") is True
    for file, want in (("summary.csv", summary), ("aggregate.csv", aggregate)):
        path = directory / file
        result[f"{file} = P7-D2"] = path.is_file() and _sha256(path.read_bytes()) == want
    for file in ("seeds.csv", "baselines_summary.csv"):
        path = directory / file
        result[f"{file} = run_info.json"] = path.is_file() and _sha256(
            path.read_bytes()
        ) == recorded.get(file)
    gz_file = directory / "episodes.csv.gz"
    result["episodes.csv.gz = P7-D2"] = gz_file.is_file() and _sha256(gz_file.read_bytes()) == gz
    if gz_file.is_file():
        result["episodes.csv (uncompressed) = P7-D2"] = (
            _sha256(storage.gunzip_bytes(gz_file.read_bytes())) == plain
        )
    sidecar = directory / "episodes.csv.sha256"
    result["episodes.csv.sha256 = P7-D2"] = sidecar.is_file() and sidecar.read_text(
        encoding="utf-8"
    ).split()[:1] == [plain]
    result["run_info.json episodes.csv = P7-D2"] = recorded.get("episodes.csv") == plain
    return result


def verify_superseded_noise(e07: Path = E07_DIR) -> dict[str, dict[str, bool]]:
    """Verify the superseded noise arm's bytes against P7-D2 (read-only; nothing is written).

    Args:
        e07: The Phase 7 output root (``results/e07``).

    Returns:
        ``{condition key: {check: ok}}``, keys ``noise_superseded_p7d1/<condition>``, plus
        ``noise_superseded_p7d1`` itself with ``"exactly the 11 conditions"``. An absent
        directory gives ``{"noise_superseded_p7d1": {"present": False}}``.
    """
    root = e07 / SUPERSEDED_NOISE_DIR
    if not root.is_dir():
        return {SUPERSEDED_NOISE_DIR: {"present": False}}
    out: dict[str, dict[str, bool]] = {
        SUPERSEDED_NOISE_DIR: {
            "present": True,
            "exactly the 11 conditions": {p.name for p in root.iterdir()}
            == set(SUPERSEDED_NOISE_SHA256),
        }
    }
    for name, hashes in SUPERSEDED_NOISE_SHA256.items():
        out[f"{SUPERSEDED_NOISE_DIR}/{name}"] = _verify_one(root / name, name, hashes)
    return out
