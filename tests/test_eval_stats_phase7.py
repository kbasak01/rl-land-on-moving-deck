"""The Phase 7 statistics: paired relative-p95, IQM success differences, drop contrasts.

P3-D1 §4 asks for a unit test of the relative-p95 statistic on a synthetic case with a known
r (the plan's Phase 7 "first task"). The known cases here are built so the answer follows
from the construction, not from the implementation:

* every closing speed of ``a`` is ``k`` times the matching speed of ``b`` on the same
  episodes, so every replicate's p95 ratio is ``k`` and ``r = 1 - k`` with a zero-width CI;
* the point estimate on random data equals ``1 - np.percentile(pooled a) /
  np.percentile(pooled b)`` over the touched-down entries only;
* NaN closing speeds (no touchdown) are excluded, never imputed: imputing them as 0 or as the
  worst speed would move the p95, and the statistic does not move;
* a drop of exactly 10 points, and a drop contrast whose two cells are the same episodes, where
  the shared-episode design cancels to a zero-width interval and the independent-cell design
  does not.

All quantities are dimensionless (rates, ratios) or m/s model scale (closing speeds).
"""

import numpy as np
import pytest

from rld.eval.stats import (
    drop_difference_bootstrap,
    iqm,
    paired_iqm_difference,
    paired_relative_p95,
    quantile_linear,
)

BASE_SEED = 20261001


def _rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(BASE_SEED + offset)


# --------------------------------------------------------------------------- quantile


@pytest.mark.parametrize("n", [1, 2, 3, 7, 20, 199, 200, 1000])
@pytest.mark.parametrize("q", [0.025, 0.5, 0.95, 0.975])
def test_quantile_linear_is_numpy_percentile_bit_for_bit(n: int, q: float) -> None:
    x = _rng(n).normal(size=n)
    assert float(quantile_linear(x, q)) == float(np.percentile(x, 100.0 * q))


def test_quantile_linear_rows_exclude_nan_and_order_infinities() -> None:
    rows = np.array(
        [[1.0, np.nan, 3.0, 2.0], [np.nan, np.nan, np.nan, np.nan], [1.0, 2.0, np.inf, np.inf]]
    )
    out = quantile_linear(rows, 0.5)
    assert out[0] == np.percentile([1.0, 2.0, 3.0], 50.0)
    assert np.isnan(out[1])
    assert out[2] == np.inf  # the median of [1, 2, inf, inf] lies between 2 and inf
    assert float(quantile_linear(np.array([1.0, 2.0, np.inf]), 0.5)) == 2.0  # weight 0 on inf
    assert float(quantile_linear(np.array([-np.inf, 1.0, 2.0]), 0.25)) == -np.inf


# --------------------------------------------------------------------------- relative p95


def test_relative_p95_known_ratio() -> None:
    """Policy a lands at 0.8 x b episode by episode: r = 0.2 exactly, in every replicate."""
    b = _rng(1).uniform(0.05, 0.5, size=(1, 200))
    a = np.repeat(0.8 * b, 5, axis=0)  # 5 seeds of a policy that lands 20 % softer everywhere
    res = paired_relative_p95(a, b, reps=2000)
    assert res.r == pytest.approx(0.2, abs=1e-12)
    assert res.lo == pytest.approx(0.2, abs=1e-12)
    assert res.hi == pytest.approx(0.2, abs=1e-12)
    assert res.separates
    assert (res.n_episodes, res.n_seeds_a, res.n_seeds_b) == (200, 5, 1)
    assert res.n_touchdowns_a == 1000 and res.n_touchdowns_b == 200


def test_relative_p95_point_estimate_pools_seeds_and_episodes() -> None:
    a = _rng(2).uniform(0.1, 0.4, size=(5, 200))
    b = _rng(3).uniform(0.1, 0.5, size=(1, 200))
    res = paired_relative_p95(a, b, reps=500)
    expected = 1.0 - np.percentile(a.reshape(-1), 95.0) / np.percentile(b.reshape(-1), 95.0)
    assert res.r == expected
    assert res.p95_a == float(np.percentile(a.reshape(-1), 95.0))
    assert res.lo <= res.r <= res.hi


def test_relative_p95_excludes_episodes_without_touchdown() -> None:
    a = _rng(4).uniform(0.1, 0.4, size=(5, 200))
    b = _rng(5).uniform(0.1, 0.5, size=(1, 200))
    a_missing = a.copy()
    a_missing[:, :30] = np.nan  # 30 timeouts in every seed: no touchdown, no closing speed
    res = paired_relative_p95(a_missing, b, reps=500)
    finite = a_missing[np.isfinite(a_missing)]
    assert res.r == 1.0 - np.percentile(finite, 95.0) / np.percentile(b, 95.0)
    assert res.n_touchdowns_a == 5 * 170
    # Imputing the missing speeds -- as 0 or as the worst observed speed -- would move p95.
    for fill in (0.0, float(np.nanmax(a_missing))):
        imputed = np.where(np.isnan(a_missing), fill, a_missing)
        assert paired_relative_p95(imputed, b, reps=10).r != res.r


def test_relative_p95_timeouts_ranked_worst_is_the_sensitivity_variant() -> None:
    a = _rng(6).uniform(0.1, 0.4, size=(5, 200))
    b = _rng(7).uniform(0.1, 0.5, size=(1, 200))
    timeouts = np.zeros_like(a, dtype=bool)
    timeouts[:, :4] = True  # 2 % of episodes time out: below 5 %, p95 stays finite but rises
    a_nan = np.where(timeouts, np.nan, a)
    primary = paired_relative_p95(a_nan, b, reps=500)
    sens = paired_relative_p95(
        a_nan,
        b,
        a_timeouts=timeouts,
        b_timeouts=np.zeros_like(b, dtype=bool),
        timeouts_as_worst=True,
        reps=500,
    )
    assert sens.timeouts_as_worst and sens.n_timeouts_a == 20
    assert np.isfinite(sens.p95_a) and sens.p95_a >= primary.p95_a
    assert sens.r <= primary.r
    timeouts[:, :20] = True  # 10 % time out: the 95th percentile is a timeout
    sens = paired_relative_p95(
        np.where(timeouts, np.nan, a),
        b,
        a_timeouts=timeouts,
        b_timeouts=np.zeros_like(b, dtype=bool),
        timeouts_as_worst=True,
        reps=200,
    )
    assert sens.p95_a == np.inf and sens.r == -np.inf


def test_relative_p95_is_deterministic_under_its_seed() -> None:
    a = _rng(8).uniform(0.1, 0.4, size=(5, 50))
    b = _rng(9).uniform(0.1, 0.5, size=(1, 50))
    first = paired_relative_p95(a, b, reps=300, seed=11)
    assert paired_relative_p95(a, b, reps=300, seed=11) == first
    other = paired_relative_p95(a, b, reps=300, seed=12)
    assert other.r == first.r and (other.lo, other.hi) != (first.lo, first.hi)


def test_relative_p95_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="same episodes"):
        paired_relative_p95(np.ones((2, 3)), np.ones((1, 4)))
    with pytest.raises(ValueError, match="touchdown"):
        paired_relative_p95(np.full((2, 3), np.nan), np.ones((1, 3)))
    with pytest.raises(ValueError, match="timeout masks"):
        paired_relative_p95(np.ones((2, 3)), np.ones((1, 3)), timeouts_as_worst=True)


# --------------------------------------------------------------------------- success contrasts


def _with_failures(n_seeds: int, n_eps: int, failures: list[list[int]]) -> np.ndarray:
    out = np.ones((n_seeds, n_eps))
    for seed, idx in enumerate(failures):
        out[seed, idx] = 0.0
    return out


def test_drop_of_a_known_size() -> None:
    """Cell 1 all successes; cell 2 loses exactly 20 of 200 in every seed: drop = 0.1."""
    a1 = np.ones((5, 200))
    a2 = _with_failures(5, 200, [list(range(20))] * 5)
    res = drop_difference_bootstrap(a1, a2, mode="independent_cells", reps=2000)
    assert res.statistic == pytest.approx(0.1, abs=1e-15)
    assert res.drop_a == res.statistic and np.isnan(res.drop_b)
    assert res.lo < 0.1 < res.hi and res.separates
    # Against a method with no drop, the difference of drops is the same 10 points.
    b = np.ones((1, 200))
    diff = drop_difference_bootstrap(a1, a2, b, b, mode="independent_cells", reps=2000)
    assert diff.statistic == pytest.approx(0.1, abs=1e-15) and diff.drop_b == 0.0


def test_drop_uses_the_iqm_over_seeds() -> None:
    rates = [0.5, 0.9, 0.95, 1.0, 1.0]
    a2 = _with_failures(5, 20, [list(range(round(20 * (1 - r)))) for r in rates])
    res = drop_difference_bootstrap(np.ones((5, 20)), a2, mode="shared_episodes", reps=100)
    assert res.success_a2 == pytest.approx((0.9 + 0.95 + 1.0) / 3.0, abs=1e-15)
    assert res.success_a2 == iqm(np.array(rates).reshape(-1, 1))


def test_shared_episodes_cancel_and_independent_cells_do_not() -> None:
    a = (_rng(10).random((5, 200)) < 0.9).astype(float)
    shared = drop_difference_bootstrap(a, a.copy(), mode="shared_episodes", reps=1000)
    assert (shared.statistic, shared.lo, shared.hi) == (0.0, 0.0, 0.0)
    independent = drop_difference_bootstrap(a, a.copy(), mode="independent_cells", reps=1000)
    assert independent.statistic == 0.0 and independent.lo < 0.0 < independent.hi


def test_drop_is_deterministic_and_validates() -> None:
    a1 = (_rng(11).random((5, 100)) < 0.95).astype(float)
    a2 = (_rng(12).random((5, 100)) < 0.85).astype(float)
    one = drop_difference_bootstrap(a1, a2, mode="independent_cells", reps=500, seed=3)
    two = drop_difference_bootstrap(a1, a2, mode="independent_cells", reps=500, seed=3)
    assert (two.statistic, two.lo, two.hi) == (one.statistic, one.lo, one.hi)
    other = drop_difference_bootstrap(a1, a2, mode="independent_cells", reps=500, seed=4)
    assert other.statistic == one.statistic and (other.lo, other.hi) != (one.lo, one.hi)
    with pytest.raises(ValueError, match="same seeds"):
        drop_difference_bootstrap(a1, a2[:4], mode="independent_cells")
    with pytest.raises(ValueError, match="equal cells"):
        drop_difference_bootstrap(a1, a2[:, :50], mode="shared_episodes")
    with pytest.raises(ValueError, match="both of b"):
        drop_difference_bootstrap(a1, a2, a1, None, mode="shared_episodes")


def test_paired_iqm_difference() -> None:
    b = (_rng(13).random((1, 200)) < 0.9).astype(float)
    a = np.repeat(np.ones((1, 200)), 5, axis=0)
    res = paired_iqm_difference(a, b, reps=1000)
    assert res.diff == pytest.approx(1.0 - b.mean(), abs=1e-15)
    assert res.lo > 0.0 and res.separates
    same = paired_iqm_difference(b, b.copy(), reps=500)
    assert (same.diff, same.lo, same.hi) == (0.0, 0.0, 0.0)


def test_drop_mean_estimator_is_the_per_episode_paired_mean() -> None:
    """P7-D1a #3: H4 uses P3-D1 §4's mean over resampled seeds x shared episodes."""
    from rld.eval import stats

    rates = [0.5, 0.9, 0.95, 1.0, 1.0]
    a2 = _with_failures(5, 20, [list(range(round(20 * (1 - r)))) for r in rates])
    res = drop_difference_bootstrap(
        np.ones((5, 20)), a2, mode="shared_episodes", reps=100, estimator="mean"
    )
    assert res.estimator == "mean"
    assert res.success_a2 == pytest.approx(float(np.mean(rates)), abs=1e-15)
    # An independent reference: same draw order (seeds of a, seeds of b, shared episodes),
    # statistic = mean over the resampled seeds x episodes of the per-episode differences.
    rng0 = _rng(20)
    a1, a2 = (rng0.random((5, 60)) < 0.9).astype(float), (rng0.random((5, 60)) < 0.8).astype(float)
    b1, b2 = (rng0.random((5, 60)) < 0.95).astype(float), (rng0.random((5, 60)) < 0.9).astype(float)
    reps, seed = 600, 7
    got = drop_difference_bootstrap(
        a1, a2, b1, b2, mode="shared_episodes", reps=reps, seed=seed, estimator="mean"
    )
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, reps, stats._PAIRED_BATCH):
        size = min(stats._PAIRED_BATCH, reps - start)
        sa = rng.integers(0, 5, size=(size, 5))
        sb = rng.integers(0, 5, size=(size, 5))
        ep = rng.integers(0, 60, size=(size, 60))
        for k in range(size):
            da = (a1 - a2)[sa[k]][:, ep[k]].mean()
            db = (b1 - b2)[sb[k]][:, ep[k]].mean()
            values.append(da - db)
    lo, hi = np.percentile(values, [2.5, 97.5])
    assert got.lo == pytest.approx(lo, abs=1e-12) and got.hi == pytest.approx(hi, abs=1e-12)
    point = ((a1 - a2).mean()) - ((b1 - b2).mean())
    assert got.statistic == pytest.approx(point, abs=1e-12)
    # The IQM default is untouched by the new argument.
    default = drop_difference_bootstrap(
        a1, a2, b1, b2, mode="shared_episodes", reps=reps, seed=seed
    )
    explicit = drop_difference_bootstrap(
        a1, a2, b1, b2, mode="shared_episodes", reps=reps, seed=seed, estimator="iqm"
    )
    assert default == explicit and default.estimator == "iqm"
    with pytest.raises(ValueError, match="estimator"):
        drop_difference_bootstrap(a1, a2, mode="shared_episodes", estimator="median")  # type: ignore[arg-type]
