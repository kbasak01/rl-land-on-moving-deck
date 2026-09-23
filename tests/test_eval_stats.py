"""Statistics for the frozen evaluation: Wilson, rliable aggregates, the paired bootstrap.

Wilson values are checked against numbers computed by hand, not against the implementation:

* 81/263 -> (0.2553, 0.3662), Newcombe (1998), *Stat. Med.* 17:857-872, method 3 (Wilson score);
* k = 0 -> upper bound ``z^2 / (n + z^2)``; k = n -> lower bound ``n / (n + z^2)``;
* 5/10 -> symmetric about 0.5, half-width ``z sqrt(0.025 + z^2/400) / (1 + z^2/10)``.

The paired bootstrap is checked on synthetic data whose difference is known exactly: a
constant per-episode shift must return that shift with a zero-width interval, because
pairing cancels every between-episode difference -- an unpaired bootstrap would not.
All quantities are dimensionless.
"""

import math

import numpy as np
import pytest
import scipy.stats

from rld.eval.stats import (
    WILSON_Z_95,
    aggregate_estimates,
    iqm,
    optimality_gap,
    paired_bootstrap,
    stratified_bootstrap_ci,
    wilson_interval,
)

Z = 1.959963984540054


def test_z_is_the_975_quantile() -> None:
    assert abs(float(scipy.stats.norm.ppf(0.975)) - WILSON_Z_95) < 1e-15


def test_wilson_matches_newcombe_1998() -> None:
    lo, hi = wilson_interval(81, 263)
    assert round(lo, 4) == 0.2553
    assert round(hi, 4) == 0.3662


def test_wilson_closed_forms_at_the_boundaries() -> None:
    z2 = Z * Z
    assert wilson_interval(0, 10) == pytest.approx((0.0, z2 / (10 + z2)), abs=1e-15)
    assert wilson_interval(0, 10)[1] == pytest.approx(0.27753, abs=5e-6)
    assert wilson_interval(10, 10) == pytest.approx((10 / (10 + z2), 1.0), abs=1e-15)
    assert wilson_interval(200, 200)[0] == pytest.approx(200 / (200 + z2), abs=1e-15)
    assert wilson_interval(200, 200)[0] == pytest.approx(0.98115, abs=5e-6)


def test_wilson_half_and_half() -> None:
    half = Z * math.sqrt(0.025 + Z * Z / 400.0) / (1.0 + Z * Z / 10.0)
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.5 - half, abs=1e-14)
    assert hi == pytest.approx(0.5 + half, abs=1e-14)
    assert (round(lo, 4), round(hi, 4)) == (0.2366, 0.7634)


def test_wilson_rejects_nonsense_and_empties() -> None:
    assert all(math.isnan(v) for v in wilson_interval(0, 0))
    for k, n in [(-1, 5), (6, 5), (0, -1)]:
        with pytest.raises(ValueError):
            wilson_interval(k, n)


def test_iqm_and_optimality_gap_are_rliables() -> None:
    scores = np.arange(1.0, 9.0).reshape(2, 4) / 10.0  # 0.1 .. 0.8
    # 25 % trimmed from 8 values leaves 0.3 .. 0.6.
    assert iqm(scores) == pytest.approx(0.45, abs=1e-15)
    assert optimality_gap(scores) == pytest.approx(1.0 - 0.45, abs=1e-15)
    assert optimality_gap(np.array([[1.0, 0.5]]), gamma=0.8) == pytest.approx(0.8 - 0.65)


def test_stratified_bootstrap_never_resamples_tasks() -> None:
    # Every run is identical within a task, so resampling runs within tasks changes
    # nothing; only resampling *tasks* could move the statistic.
    scores = np.tile(np.array([[0.1, 0.5, 0.9, 1.0]]), (5, 1))
    est = stratified_bootstrap_ci(scores, iqm, reps=500, seed=1)
    assert est.lo == est.hi == est.point


def test_stratified_bootstrap_single_run_collapses() -> None:
    est = stratified_bootstrap_ci(np.array([[0.3, 0.7, 0.9]]), iqm, reps=200, seed=2)
    assert est.lo == est.hi == est.point


def test_stratified_bootstrap_is_seeded_and_covers_the_point() -> None:
    rng = np.random.default_rng(20260921)
    scores = np.clip(rng.normal(0.7, 0.1, size=(5, 13)), 0.0, 1.0)
    a = aggregate_estimates(scores, reps=2000, seed=3)
    b = aggregate_estimates(scores, reps=2000, seed=3)
    assert a == b
    for est in a.values():
        assert est.lo <= est.point <= est.hi
        assert est.reps == 2000
        assert est.lo < est.hi


def test_stratified_bootstrap_rejects_missing_values() -> None:
    with pytest.raises(ValueError):
        stratified_bootstrap_ci(np.array([[0.5, np.nan]]), iqm)
    with pytest.raises(ValueError):
        stratified_bootstrap_ci(np.array([0.5, 0.6]), iqm)


def test_paired_constant_shift_is_recovered_exactly() -> None:
    rng = np.random.default_rng(20260921 + 1)
    base = rng.normal(0.0, 1.0, size=300)  # large between-episode spread
    b = base[np.newaxis, :]
    a = np.tile(base + 0.1, (5, 1))
    contrast = paired_bootstrap(a, b, reps=2000, seed=4)
    assert contrast.diff == pytest.approx(0.1, abs=1e-12)
    assert contrast.lo == pytest.approx(0.1, abs=1e-12)
    assert contrast.hi == pytest.approx(0.1, abs=1e-12)
    assert contrast.separates
    assert (contrast.n_episodes, contrast.n_seeds_a, contrast.n_seeds_b) == (300, 5, 1)


def test_paired_binary_known_difference() -> None:
    # b succeeds with p = 0.6; a keeps every b success and converts each b failure to a
    # success with probability 0.25, independently per seed: E[a - b] = 0.4 * 0.25 = 0.10.
    rng = np.random.default_rng(20260921 + 2)
    n_e = 2000
    b = (rng.random(n_e) < 0.6).astype(np.float64)[np.newaxis, :]
    a = np.maximum(b, (rng.random((5, n_e)) < 0.25).astype(np.float64))
    contrast = paired_bootstrap(a, b, reps=10000, seed=5)
    assert contrast.lo < 0.10 < contrast.hi
    assert contrast.lo < contrast.diff < contrast.hi
    assert contrast.separates
    assert contrast.hi - contrast.lo < 0.05


def test_paired_null_difference_does_not_separate() -> None:
    rng = np.random.default_rng(20260921 + 3)
    b = (rng.random((5, 400)) < 0.7).astype(np.float64)
    a = (rng.random((5, 400)) < 0.7).astype(np.float64)
    contrast = paired_bootstrap(a, b, reps=5000, seed=6)
    assert contrast.lo < 0.0 < contrast.hi
    assert not contrast.separates


def test_paired_is_seeded_and_checks_shapes() -> None:
    a = np.array([[1.0, 0.0, 1.0, 1.0]])
    b = np.array([[0.0, 0.0, 1.0, 0.0]])
    assert paired_bootstrap(a, b, reps=300, seed=7) == paired_bootstrap(a, b, reps=300, seed=7)
    with pytest.raises(ValueError):
        paired_bootstrap(a, b[:, :3])
    with pytest.raises(ValueError):
        paired_bootstrap(a, np.array([[0.0, np.nan, 1.0, 0.0]]))
