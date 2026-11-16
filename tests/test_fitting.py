import numpy as np
import pytest

from labquake_explorer.analysis.fitting import linear_fit


def test_ols_exact_and_noisy():
    x = np.linspace(0, 10, 101)
    fit = linear_fit(x, 2.5 * x - 1.0)
    assert fit.valid and fit.n == 101 and fit.x_range == pytest.approx(10.0)
    assert (fit.slope, fit.intercept, fit.r2, fit(4.0)) == pytest.approx((2.5, -1.0, 1.0, 9.0))
    x = np.linspace(0, 1, 500)
    fit = linear_fit(x, 3.0 * x + np.random.default_rng(1).normal(0, 0.1, x.size))
    assert fit.slope == pytest.approx(3.0, abs=5 * fit.stderr)
    assert 0.9 < fit.r2 < 1.0 and fit.stderr > 0


def test_theilsen_ignores_outliers():
    x = np.linspace(0, 10, 101)
    y = 2.0 * x + 1.0
    y[-10:] += 50.0  # 10% gross outliers at one end (biases OLS)
    ts = linear_fit(x, y, "theilsen")
    assert ts.valid and ts.method == "theilsen" and np.isnan(ts.stderr)
    assert ts.slope == pytest.approx(2.0, abs=1e-6)
    assert abs(linear_fit(x, y, "ols").slope - 2.0) > 0.1


def test_degenerate_and_bad_inputs():
    assert not linear_fit([], []).valid
    assert not linear_fit([1.0], [2.0]).valid
    fit = linear_fit([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert not fit.valid and "identical" in fit.reason
    assert np.isnan(fit.slope) and np.isnan(fit(0.0))
    fit = linear_fit([0.0, np.nan, 1.0], [0.0, 5.0, 1.0])
    assert fit.valid and fit.n == 2 and fit.slope == pytest.approx(1.0)
    fit = linear_fit([0.0, 1.0], [0.0, 2.0])                 # two points: valid, but no stderr
    assert fit.valid and fit.slope == 2.0 and np.isnan(fit.stderr)
    with pytest.raises(ValueError):
        linear_fit([0, 1], [0, 1], method="ransac")
    with pytest.raises(ValueError):
        linear_fit([0, 1, 2], [0, 1])
