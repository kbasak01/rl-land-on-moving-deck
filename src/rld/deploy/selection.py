"""Which seed of a method is exported and timed: the median seed of the selection cell (P8-D1 §2).

Rule (user, 2026-10-05): rank the five training seeds **worst to best** by per-seed success at
``id`` SS6, aft pad (the P7-D1 §3 selection cell), read from the committed ``seeds.csv``. Among
equal success the higher p95 ``rel_vz_normal`` ranks worse (lower is better); if both are equal
the lower seed ranks better. The 3rd of 5 is taken.

Units: success is a fraction in [0, 1]; ``rel_vz_normal`` p95 is metres per second model scale.
"""

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rld.config import REPO_ROOT

__all__ = [
    "SELECTION_CELL",
    "SELECTION_COLUMNS",
    "SELECTION_SOURCES",
    "SeedRanking",
    "rank_seeds",
    "read_seed_row",
    "selection_csv",
]

#: ``(pad, regime, ss)`` of the selection cell (P7-D1 §3, P8-D1 §1).
SELECTION_CELL: tuple[str, str, str] = ("aft", "id", "SS6")

#: The committed per-seed table each exported method is read from.
SELECTION_SOURCES: dict[str, Path] = {
    "ppo": REPO_ROOT / "results" / "e05" / "seeds.csv",
    "residual_ppo_forecast": REPO_ROOT / "results" / "e06" / "seeds.csv",
}

#: Columns of ``results/latency/selection.csv``.
SELECTION_COLUMNS: tuple[str, ...] = (
    "method",
    "source",
    "pad",
    "regime",
    "ss",
    "rank_worst_to_best",
    "seed",
    "success_rate",
    "rel_vz_normal_p95_m_s",
    "selected",
)


@dataclass(frozen=True)
class SeedRanking:
    """A method's seeds ranked worst to best at the selection cell.

    Attributes:
        method: Method label.
        order: Seeds, worst first.
        success: ``{seed: success rate}``, fraction.
        p95: ``{seed: p95 rel_vz_normal}``, metres per second model scale.
    """

    method: str
    order: tuple[int, ...]
    success: dict[int, float]
    p95: dict[int, float]

    @property
    def median_seed(self) -> int:
        """The 3rd of 5 (the middle element of an odd-length ranking)."""
        if len(self.order) % 2 != 1:
            raise ValueError(f"{self.method}: an even number of seeds has no median seed")
        return self.order[len(self.order) // 2]


def read_seed_row(path: Path, method: str) -> Mapping[str, str]:
    """Return the selection-cell row of ``method`` in a committed ``seeds.csv``.

    Args:
        path: The ``seeds.csv``.
        method: Method label.

    Returns:
        The row, as text.

    Raises:
        ValueError: If there is not exactly one such row.
    """
    pad, regime, ss = SELECTION_CELL
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            r
            for r in csv.DictReader(handle)
            if r["method"] == method and r["pad"] == pad and r["regime"] == regime and r["ss"] == ss
        ]
    if len(rows) != 1:
        raise ValueError(f"{path}: {len(rows)} rows for {method} at {SELECTION_CELL}, expected 1")
    return rows[0]


def rank_seeds(method: str, row: Mapping[str, str]) -> SeedRanking:
    """Rank a method's seeds worst to best by the P8-D1 §2 rule.

    Args:
        method: Method label.
        row: Its ``seeds.csv`` row at the selection cell.

    Returns:
        The ranking.
    """
    seeds = [int(s) for s in row["seeds"].split("|")]
    success = {s: float(row[f"success_rate_seed{s}"]) for s in seeds}
    p95 = {s: float(row[f"rel_vz_normal_p95_m_s_seed{s}"]) for s in seeds}
    # Worst first: lower success; then higher p95 (lower is better); then higher seed (the
    # lower seed ranks better on a full tie).
    order = sorted(seeds, key=lambda s: (success[s], -p95[s], -s))
    return SeedRanking(method=method, order=tuple(order), success=success, p95=p95)


def selection_csv(
    rankings: Sequence[SeedRanking], sources: Mapping[str, Path] = SELECTION_SOURCES
) -> str:
    """Render ``selection.csv`` deterministically.

    Args:
        rankings: One ranking per method.
        sources: Where each ranking was read from.

    Returns:
        The CSV text: one row per (method, seed), worst first, floats as ``repr``.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SELECTION_COLUMNS)
    pad, regime, ss = SELECTION_CELL
    for ranking in rankings:
        source = sources[ranking.method]
        rel = source.relative_to(REPO_ROOT) if source.is_relative_to(REPO_ROOT) else source
        for rank, seed in enumerate(ranking.order, start=1):
            writer.writerow(
                [
                    ranking.method,
                    str(rel),
                    pad,
                    regime,
                    ss,
                    rank,
                    seed,
                    repr(ranking.success[seed]),
                    repr(ranking.p95[seed]),
                    seed == ranking.median_seed,
                ]
            )
    return buffer.getvalue()
