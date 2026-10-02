"""Committed storage of large per-episode CSVs: deterministic gzip, read back transparently.

Authority: P7-D1a §12 (user, 2026-10-02). The Phase 7 per-episode files are committed
**gzipped, one file per condition**, as ``results/e07/<arm>[/<condition>]/episodes.csv.gz``,
with the SHA-256 of the **uncompressed** ``episodes.csv`` recorded beside it. Everything
else of a condition (summaries, contrasts, ``results.md``) is re-derived from a clone.

What is guaranteed
------------------
* **Reproducible bytes.** :func:`gzip_bytes` writes the RFC 1952 member by hand: magic,
  ``CM = 8`` (deflate), ``FLG = 0`` (no file name, no comment, no extra field, no header
  CRC), ``MTIME = 0``, ``XFL`` from the level, ``OS = 255`` ("unknown"), then a raw deflate
  stream at the fixed level :data:`GZIP_LEVEL`, then CRC-32 and ISIZE. Nothing in the header
  depends on the file's name, the clock or the platform, so the same CSV gives the same
  ``.gz`` bytes for the same zlib. (``gzip.compress`` is not used: since Python 3.11 it
  delegates to ``zlib.compress(wbits=31)`` when ``mtime=0``, whose header carries the
  build's OS byte.) The **authority** is the uncompressed SHA-256, which does not depend on
  zlib at all.
* **Lossless, checked.** :func:`compress_csv` decompresses what it wrote and compares the
  SHA-256 with the source's before it records anything or removes the source.
* **Transparent reads.** Every reader of a logical ``.../episodes.csv`` goes through
  :func:`resolve`: the ``.csv.gz`` beside it when present, else the ``.csv`` itself. The
  logical path (never the ``.gz``) is what tables and contrast rows name, so a recomputation
  is byte-identical before and after compression.

The sidecar ``<name>.sha256`` holds one ``sha256sum`` line (``<hex>  <name>``), so
``gunzip -c episodes.csv.gz | sha256sum`` can be compared with it by hand.

Units: none (bytes and text).
"""

import gzip
import hashlib
import struct
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, TextIO

__all__ = [
    "GZIP_LEVEL",
    "GZ_SUFFIX",
    "SHA_SUFFIX",
    "compress_csv",
    "exists",
    "gunzip_bytes",
    "gz_path",
    "gzip_bytes",
    "open_text",
    "read_bytes",
    "read_text",
    "resolve",
    "sha256_path",
    "verify_compressed",
]

#: The fixed deflate level of every committed ``.csv.gz`` (zlib's maximum).
GZIP_LEVEL: Final[int] = 9

#: Suffix appended to a CSV's name for its compressed copy.
GZ_SUFFIX: Final[str] = ".gz"

#: Suffix appended to a CSV's name for its uncompressed SHA-256 sidecar.
SHA_SUFFIX: Final[str] = ".sha256"

#: Read and compress in chunks of this many bytes.
_CHUNK: Final[int] = 1 << 20

#: RFC 1952 header: ID1 ID2 CM FLG, MTIME = 0, then XFL and OS (filled per level).
_MAGIC: Final[bytes] = b"\x1f\x8b\x08\x00" + struct.pack("<I", 0)


def gz_path(path: Path) -> Path:
    """Return the compressed sibling of a logical CSV path (``episodes.csv.gz``)."""
    return path.with_name(path.name + GZ_SUFFIX)


def sha256_path(path: Path) -> Path:
    """Return the SHA-256 sidecar of a logical CSV path (``episodes.csv.sha256``)."""
    return path.with_name(path.name + SHA_SUFFIX)


def resolve(path: Path) -> Path:
    """Return the file that holds a logical CSV: its ``.gz`` when present, else itself.

    Args:
        path: The logical path, e.g. ``results/e07/matrix/episodes.csv``.

    Returns:
        ``path + ".gz"`` if that file exists, otherwise ``path`` (which may not exist).
    """
    compressed = gz_path(path)
    return compressed if compressed.is_file() else path


def exists(path: Path) -> bool:
    """Return whether a logical CSV exists, compressed or not."""
    return resolve(path).is_file()


@contextmanager
def open_text(path: Path) -> Iterator[TextIO]:
    """Open a logical CSV for reading as UTF-8 text with universal newlines off.

    Args:
        path: The logical path (``.csv``); its ``.csv.gz`` is read when present.

    Yields:
        A text handle (``newline=""``, as :mod:`csv` wants).
    """
    actual = resolve(path)
    if actual.name.endswith(GZ_SUFFIX):
        with gzip.open(actual, "rt", encoding="utf-8", newline="") as text:
            yield text
    else:
        with actual.open(newline="", encoding="utf-8") as handle:
            yield handle


def read_bytes(path: Path) -> bytes:
    """Return the uncompressed bytes of a logical CSV (``.gz`` preferred)."""
    actual = resolve(path)
    data = actual.read_bytes()
    return gunzip_bytes(data) if actual.name.endswith(GZ_SUFFIX) else data


def read_text(path: Path) -> str:
    """Return the uncompressed UTF-8 text of a logical CSV (``.gz`` preferred)."""
    return read_bytes(path).decode("utf-8")


def _xfl(level: int) -> bytes:
    """Return RFC 1952's XFL byte for a deflate level (2: slowest, 4: fastest, else 0)."""
    if level == 9:
        return b"\x02"
    if level == 1:
        return b"\x04"
    return b"\x00"


def gzip_bytes(data: bytes, level: int = GZIP_LEVEL) -> bytes:
    """Return one deterministic gzip member of ``data`` (see the module docstring).

    Args:
        data: Uncompressed bytes.
        level: Deflate level, 0-9.

    Returns:
        The gzip bytes: fixed header (no name, ``MTIME`` 0, ``OS`` 255), raw deflate,
        CRC-32 and ISIZE, little-endian.

    Raises:
        ValueError: If ``level`` is outside 0-9.
    """
    if not 0 <= level <= 9:
        raise ValueError(f"deflate level must be 0-9, got {level}")
    deflate = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS, 9, zlib.Z_DEFAULT_STRATEGY)
    parts = [_MAGIC, _xfl(level), b"\xff"]
    crc = 0
    view = memoryview(data)
    for start in range(0, len(view), _CHUNK):
        chunk = view[start : start + _CHUNK]
        crc = zlib.crc32(chunk, crc)
        parts.append(deflate.compress(chunk))
    parts.append(deflate.flush())
    parts.append(struct.pack("<II", crc & 0xFFFFFFFF, len(data) & 0xFFFFFFFF))
    return b"".join(parts)


def gunzip_bytes(data: bytes) -> bytes:
    """Return the uncompressed bytes of a gzip stream, checking its CRC and length.

    Args:
        data: Gzip bytes (any conforming member, not only :func:`gzip_bytes`').

    Returns:
        The uncompressed bytes.

    Raises:
        ValueError: If the stream is not one complete gzip member.
    """
    inflate = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = inflate.decompress(data) + inflate.flush()
    if not inflate.eof or inflate.unused_data:
        raise ValueError("not exactly one complete gzip member")
    return out


def compress_csv(path: Path, *, level: int = GZIP_LEVEL, remove: bool = True) -> dict[str, Any]:
    """Gzip a CSV deterministically beside itself and record its uncompressed SHA-256.

    Steps: hash the source; write ``<name>.gz`` through a temporary file and rename it;
    decompress the written file and require the identical SHA-256; write the sidecar
    ``<name>.sha256``; only then, with ``remove``, delete the source. Re-running on an
    already compressed CSV (no source, ``.gz`` and sidecar present) verifies and returns.

    Args:
        path: The logical CSV, e.g. ``results/e07/matrix/episodes.csv``.
        level: Deflate level (:data:`GZIP_LEVEL`).
        remove: Delete the uncompressed source once the round trip is verified.

    Returns:
        ``{"file", "gz_file", "sha256" (uncompressed), "gz_sha256", "bytes", "gz_bytes",
        "level", "removed_source", "already_compressed"}``.

    Raises:
        FileNotFoundError: If neither the CSV nor a verified ``.gz`` exists.
        RuntimeError: If the round trip changes a byte, or an existing ``.gz`` disagrees
            with the source or its sidecar.
    """
    target = gz_path(path)
    sidecar = sha256_path(path)
    if not path.is_file():
        if target.is_file():
            check = verify_compressed(path)
            if not all(check.values()):
                raise RuntimeError(f"{target}: existing compressed copy fails its check {check}")
            data = gunzip_bytes(target.read_bytes())
            return _facts(path, data, target, level, removed=False, already=True)
        raise FileNotFoundError(f"{path}: no CSV and no compressed copy")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if target.is_file():
        existing = gunzip_bytes(target.read_bytes())
        if hashlib.sha256(existing).hexdigest() != digest:
            raise RuntimeError(f"{target} exists and does not hold {path}'s bytes")
    else:
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(gzip_bytes(data, level))
        tmp.replace(target)
    back = gunzip_bytes(target.read_bytes())
    if back != data or hashlib.sha256(back).hexdigest() != digest:
        raise RuntimeError(f"{target}: the gzip round trip changed {path}")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    if remove:
        path.unlink()
    return _facts(path, data, target, level, removed=remove, already=False)


def _facts(
    path: Path, data: bytes, target: Path, level: int, *, removed: bool, already: bool
) -> dict[str, Any]:
    """Return the storage facts of one compressed CSV (for ``run_info.json``)."""
    gz = target.read_bytes()
    return {
        "file": path.name,
        "gz_file": target.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "gz_sha256": hashlib.sha256(gz).hexdigest(),
        "bytes": len(data),
        "gz_bytes": len(gz),
        "level": level,
        "removed_source": removed,
        "already_compressed": already,
    }


def verify_compressed(path: Path, expected_sha256: str | None = None) -> dict[str, bool]:
    """Check a compressed CSV against its sidecar (and an expected SHA-256, if given).

    Args:
        path: The logical CSV path.
        expected_sha256: Another record of the uncompressed SHA-256 (e.g. ``run_info.json``'s
            ``files_sha256["episodes.csv"]``), or ``None``.

    Returns:
        ``{check: ok}``: ``"<name>.gz present"``, ``"<name>.gz = sidecar"``,
        ``"<name>.gz = run_info"`` (only with ``expected_sha256``), and, when the
        uncompressed CSV is also present, ``"<name> = <name>.gz"`` (identical bytes).
    """
    target = gz_path(path)
    sidecar = sha256_path(path)
    result: dict[str, bool] = {f"{target.name} present": target.is_file()}
    if not target.is_file():
        return result
    data = gunzip_bytes(target.read_bytes())
    digest = hashlib.sha256(data).hexdigest()
    recorded = sidecar.read_text(encoding="utf-8").split()[0] if sidecar.is_file() else ""
    result[f"{target.name} = {sidecar.name}"] = recorded == digest
    if expected_sha256 is not None:
        result[f"{target.name} = run_info"] = expected_sha256 == digest
    if path.is_file():
        result[f"{path.name} = {target.name}"] = path.read_bytes() == data
    return result
