"""Committed storage of per-episode CSVs (P7-D1a §12): deterministic gzip, transparent reads.

What is pinned here:

* the gzip round trip is **byte-identical** on a real committed ``episodes.csv`` (``e01``)
  and on synthetic text with UTF-8 and empty lines;
* the ``.gz`` bytes are reproducible: no file name (``FLG = 0``), ``MTIME = 0``, ``OS = 255``,
  fixed level; compressing twice, or the same text from another file name, gives the same
  bytes; the standard library's ``gzip`` reads them;
* :func:`rld.eval.storage.compress_csv` records the uncompressed SHA-256 in the sidecar,
  deletes the source only when asked and only after the check, refuses a ``.gz`` that does
  not hold the source's bytes, and is idempotent;
* every reader prefers ``<name>.gz`` and falls back to ``<name>``:
  :func:`rld.eval.report.read_rows`, :func:`rld.eval.reproduce.compare_to_reference` and
  :class:`rld.eval.hypotheses.EpisodeStore` read the same records either way;
* :func:`rld.eval.storage.verify_compressed` catches a tampered sidecar or ``.gz``.

Units: none (bytes and text).
"""

import gzip
import hashlib
import struct
from pathlib import Path

import pytest

from rld.eval import storage
from rld.eval.report import read_rows

REPO = Path(__file__).resolve().parents[1]
E01 = REPO / "results" / "e01" / "episodes.csv"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_round_trip_is_byte_identical_on_a_committed_episodes_csv() -> None:
    data = E01.read_bytes()
    packed = storage.gzip_bytes(data)
    assert storage.gunzip_bytes(packed) == data
    assert gzip.decompress(packed) == data  # any gzip reader, not only ours
    assert len(packed) < len(data) // 2  # about 2.9x on e01 (float-heavy text)


@pytest.mark.parametrize(
    "text", ["", "a\n", "x,y\n1,2\n\n3,4\n", "θ,σ_p\n0.5,1e-300\n" * 1000, "no newline"]
)
def test_round_trip_on_edge_cases(text: str) -> None:
    data = text.encode("utf-8")
    assert storage.gunzip_bytes(storage.gzip_bytes(data)) == data


def test_gzip_bytes_are_deterministic_and_carry_no_name_or_time() -> None:
    data = E01.read_bytes()[:200_000]
    one, two = storage.gzip_bytes(data), storage.gzip_bytes(data)
    assert one == two
    assert one[:4] == b"\x1f\x8b\x08\x00"  # magic, deflate, FLG = 0: no FNAME / FCOMMENT
    assert struct.unpack("<I", one[4:8])[0] == 0  # MTIME
    assert one[8:10] == b"\x02\xff"  # XFL for level 9, OS = 255 (unknown)
    crc, isize = struct.unpack("<II", one[-8:])
    assert isize == len(data)
    import zlib

    assert crc == zlib.crc32(data)
    assert storage.GZIP_LEVEL == 9
    with pytest.raises(ValueError):
        storage.gzip_bytes(data, level=10)
    with pytest.raises(ValueError):
        storage.gunzip_bytes(one + b"trailing")


def test_compress_csv_round_trip_sidecar_and_removal(tmp_path: Path) -> None:
    src = tmp_path / "a" / "episodes.csv"
    src.parent.mkdir()
    data = E01.read_bytes()
    src.write_bytes(data)
    facts = storage.compress_csv(src, remove=False)
    gz, sidecar = storage.gz_path(src), storage.sha256_path(src)
    assert (gz.name, sidecar.name) == ("episodes.csv.gz", "episodes.csv.sha256")
    assert facts["sha256"] == _sha(data) and facts["bytes"] == len(data)
    assert facts["gz_sha256"] == _sha(gz.read_bytes()) and not facts["removed_source"]
    assert sidecar.read_text(encoding="utf-8") == f"{_sha(data)}  episodes.csv\n"
    assert storage.read_bytes(src) == data and src.is_file()
    assert storage.verify_compressed(src, _sha(data)) == {
        "episodes.csv.gz present": True,
        "episodes.csv.gz = episodes.csv.sha256": True,
        "episodes.csv.gz = run_info": True,
        "episodes.csv = episodes.csv.gz": True,
    }
    # Same bytes under another name and directory: the .gz is the same file.
    other = tmp_path / "b" / "other.csv"
    other.parent.mkdir()
    other.write_bytes(data)
    storage.compress_csv(other)
    assert storage.gz_path(other).read_bytes() == gz.read_bytes()
    assert not other.exists()
    # Remove the source; reading is unchanged; running again is idempotent.
    again = storage.compress_csv(src, remove=True)
    assert not src.exists() and again["removed_source"]
    assert storage.read_bytes(src) == data and storage.exists(src)
    third = storage.compress_csv(src)
    assert third["already_compressed"] and third["sha256"] == _sha(data)


def test_compress_refuses_a_gz_that_does_not_hold_the_source(tmp_path: Path) -> None:
    src = tmp_path / "episodes.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    storage.gz_path(src).write_bytes(storage.gzip_bytes(b"a,b\n1,3\n"))
    with pytest.raises(RuntimeError, match="does not hold"):
        storage.compress_csv(src)
    assert src.is_file()  # nothing removed
    with pytest.raises(FileNotFoundError):
        storage.compress_csv(tmp_path / "missing.csv")


def test_verify_catches_tampering(tmp_path: Path) -> None:
    src = tmp_path / "episodes.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    storage.compress_csv(src, remove=False)
    assert all(storage.verify_compressed(src).values())
    assert not all(storage.verify_compressed(src, "0" * 64).values())
    src.write_text("a,b\n1,9\n", encoding="utf-8")  # the plain copy now differs
    assert storage.verify_compressed(src)["episodes.csv = episodes.csv.gz"] is False
    src.unlink()
    storage.sha256_path(src).write_text("f" * 64 + "  episodes.csv\n", encoding="utf-8")
    assert storage.verify_compressed(src)["episodes.csv.gz = episodes.csv.sha256"] is False
    assert storage.verify_compressed(tmp_path / "none.csv") == {"none.csv.gz present": False}


def test_readers_prefer_the_gz_and_fall_back_to_the_csv(tmp_path: Path) -> None:
    from rld.eval.hypotheses import EpisodeStore
    from rld.eval.reproduce import compare_to_reference

    plain = tmp_path / "plain" / "episodes.csv"
    packed = tmp_path / "packed" / "episodes.csv"
    for path in (plain, packed):
        path.parent.mkdir()
        path.write_bytes(E01.read_bytes())
    storage.compress_csv(packed)
    assert storage.resolve(plain) == plain and storage.resolve(packed).name.endswith(".gz")
    rows = read_rows(plain)
    assert read_rows(packed) == rows and len(rows) > 1000
    with storage.open_text(packed) as handle:
        assert handle.readline() == E01.read_text(encoding="utf-8").split("\n", 1)[0] + "\n"
    ref_plain = compare_to_reference(rows[:50], plain)
    ref_packed = compare_to_reference(rows[:50], packed)
    assert ref_plain["reference_sha256"] == ref_packed["reference_sha256"] == _sha(E01.read_bytes())
    assert ref_plain["all_identical"] and ref_packed["all_identical"]
    store = EpisodeStore(tmp_path)
    a = store.cell(plain, "pid_feedforward", "aft", "id", "SS5")
    b = store.cell(packed, "pid_feedforward", "aft", "id", "SS5")
    assert (a.index, a.episode_seed, a.t0) == (b.index, b.episode_seed, b.t0)
    assert (a.success == b.success).all() and (a.timeout == b.timeout).all()
    assert b.source == "packed/episodes.csv"  # the logical name, never the .gz
    assert store.has(packed) and not store.has(tmp_path / "missing" / "episodes.csv")
