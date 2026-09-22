"""Generate the frozen evaluation episode lists, or check them against the committed manifest.

Phase 3, Step B. Argparse wrapper only; the logic lives in :mod:`rld.eval.episodes`.

Usage::

    python scripts/make_episodes.py --workers 24            # write results/episodes/
    python scripts/make_episodes.py --check --workers 24    # regenerate, compare, write nothing

Writes ``results/episodes/{id,unseen_seastate,unseen_heading,unseen_vessel,static}.parquet``
and ``results/episodes/MANIFEST.csv``. The lists are generated **once**, committed on their
own commit, and their hashes are written into P3-D1; the script therefore refuses to
overwrite an existing manifest unless ``--force`` is given. ``--check`` regenerates every list
and compares both the content SHA-256 and the file SHA-256 with the committed manifest; it
exits non-zero on any mismatch.

``--workers`` does not change any output: resets are split over chunks and reassembled in
draw order.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

from rld.eval.envs import load_eval_configs
from rld.eval.episodes import (
    DEFAULT_CHUNK,
    DEFAULT_GENERATOR_SEED,
    DEFAULT_PAD,
    DEFAULT_WORKERS,
    EPISODES_DIR,
    EPISODES_PER_CELL,
    MANIFEST_NAME,
    generate_lists,
    read_manifest,
    write_lists,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=EPISODES_DIR)
    parser.add_argument("--generator-seed", type=int, default=DEFAULT_GENERATOR_SEED)
    parser.add_argument("--n", type=int, default=EPISODES_PER_CELL, help="episodes per cell")
    parser.add_argument("--pad", default=DEFAULT_PAD)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--force", action="store_true", help="overwrite committed lists")
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate into a temporary directory and compare with the committed manifest",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.out_dir / MANIFEST_NAME
    if manifest_path.exists() and not (args.force or args.check):
        print(f"{manifest_path} exists; the lists are frozen. Use --check, or --force.")
        return 2
    cfgs = load_eval_configs()
    started = time.perf_counter()
    lists = generate_lists(
        cfgs.sim,
        args.generator_seed,
        args.n,
        cfgs=cfgs,
        pad=args.pad,
        workers=args.workers,
        chunk=args.chunk,
    )
    wall_s = time.perf_counter() - started

    if args.check:
        committed = {row["file"]: row for row in read_manifest(manifest_path)}
        with tempfile.TemporaryDirectory() as tmp:
            fresh = write_lists(lists, Path(tmp), args.generator_seed)
        failures = 0
        for row in fresh:
            ref = committed.get(row.file)
            content_ok = ref is not None and ref["content_sha256"] == row.content_sha256
            file_ok = ref is not None and ref["file_sha256"] == row.file_sha256
            failures += int(not (content_ok and file_ok))
            print(
                f"{row.file:<26s} rows={row.rows:<5d} content={'OK' if content_ok else 'MISMATCH'} "
                f"file={'OK' if file_ok else 'MISMATCH'}"
            )
        missing = sorted(set(committed) - {row.file for row in fresh})
        failures += len(missing)
        for name in missing:
            print(f"{name:<26s} committed but not regenerated")
        print(f"{wall_s:.1f} s wall with {args.workers} worker(s)")
        return 1 if failures else 0

    manifest = write_lists(lists, args.out_dir, args.generator_seed)
    for row in manifest:
        print(
            f"{row.file:<26s} rows={row.rows:<5d} cells={row.cells:<20s} "
            f"in_train={row.frac_in_training_distribution:.4f} "
            f"file={row.file_sha256} content={row.content_sha256}"
        )
    print(f"manifest -> {manifest_path}")
    print(f"{wall_s:.1f} s wall with {args.workers} worker(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
