import warnings

import numpy as np
import pytest
from scipy import stats

from labquake_explorer.analysis.scaling import (
    PowerLawFit, fit_power_law, bootstrap_exponent, reference_line,
)

FIELDS = {"exponent", "log10_coefficient", "coefficient", "exponent_stderr",
          "r2", "exponent_rma", "n", "valid", "reason"}


def test_exact_power_law_recovered():
    x = np.logspace(-3, 3, 25)
    fit = fit_power_law(x, 4.0 * x ** (-1.0 / 3.0))
    assert isinstance(fit, PowerLawFit) and fit.valid and fit.reason == "" and fit.n == 25
    assert fit.exponent == pytest.approx(-1.0 / 3.0, abs=1e-10)
    assert fit.coefficient == pytest.approx(4.0, rel=1e-9)
    assert fit.log10_coefficient == pytest.approx(np.log10(4.0), abs=1e-10)
    assert fit.r2 == pytest.approx(1.0)
    assert fit(8.0) == pytest.approx(4.0 * 8.0 ** (-1.0 / 3.0), rel=1e-9)
    # zero to rounding: 1.5e-9 with numpy 2.5 on x86, less elsewhere
    assert np.isfinite(fit.exponent_stderr) and fit.exponent_stderr == pytest.approx(0.0, abs=1e-7)
    assert fit.exponent_rma == pytest.approx(-1.0 / 3.0, abs=1e-10)     # RMA coincides with OLS on exact data
    d = fit_power_law([1.0, 10.0, 100.0, 1000.0], 2.0 * np.array([1.0, 10.0, 100.0, 1000.0]) ** 1.5).as_dict()
    assert set(d) == FIELDS and d["valid"] is True
    assert d["exponent"] == pytest.approx(1.5) and d["exponent_rma"] == pytest.approx(1.5)
    # evaluation matches direct evaluation where finite, x = 0 included
    fit = fit_power_law(np.logspace(0, 3, 10), 3.0 * np.logspace(0, 3, 10) ** 2.0)
    xs = np.array([0.0, 0.5, 1.0, 10.0, 1e3])
    assert np.allclose(fit(xs), 3.0 * xs ** 2.0, rtol=1e-9) and fit(0.0) == pytest.approx(0.0)
    # zero exponent: constant everywhere, including x = 0
    flat = fit_power_law([1.0, 10.0, 100.0], [0.4, 0.4, 0.4])
    assert flat.valid and flat.exponent == pytest.approx(0.0, abs=1e-12)
    assert flat(0.0) == pytest.approx(0.4) and flat(7.0) == pytest.approx(0.4)
    assert flat.exponent_rma == pytest.approx(0.0, abs=1e-12)


def test_noisy_fit_pinned_to_ols_and_rma_formulas():
    rng = np.random.default_rng(3)
    x = np.logspace(0, 4, 200)
    y = 10.0 ** (0.7 + (-1.0 / 3.0) * np.log10(x) + rng.normal(0, 0.1, x.size))
    fit = fit_power_law(x, y)
    assert fit.valid and fit.exponent_stderr > 0 and 0.5 < fit.r2 < 1.0
    assert abs(fit.exponent - (-1.0 / 3.0)) < 3 * fit.exponent_stderr
    # stderr: the textbook OLS formula with n-2 degrees of freedom, not n-1
    lx, ly = np.log10(x), np.log10(y)
    resid = ly - (fit.exponent * lx + fit.log10_coefficient)
    sxx = np.sum((lx - lx.mean()) ** 2)
    assert fit.exponent_stderr == pytest.approx(np.sqrt(np.sum(resid ** 2) / (x.size - 2) / sxx), rel=1e-9)
    assert fit.exponent_stderr != pytest.approx(np.sqrt(np.sum(resid ** 2) / (x.size - 1) / sxx), rel=1e-9)
    ref = stats.linregress(lx, ly)                             # r2, slope and intercept as scipy's
    assert fit.r2 == pytest.approx(ref.rvalue ** 2, rel=1e-12)
    assert fit.exponent == pytest.approx(ref.slope, rel=1e-12)
    assert fit.log10_coefficient == pytest.approx(ref.intercept, rel=1e-12)
    # RMA = sign(r) * std(ly)/std(lx) = OLS / |r|, steeper than OLS, same sign
    rng = np.random.default_rng(5)
    x = np.logspace(0, 4, 150)
    y = 10.0 ** (0.7 - 0.5 * np.log10(x) + rng.normal(0, 0.2, x.size))
    fit = fit_power_law(x, y)
    lx, ly = np.log10(x), np.log10(y)
    r = stats.linregress(lx, ly).rvalue
    assert r < 0
    assert fit.exponent_rma == pytest.approx(np.sign(r) * np.std(ly) / np.std(lx), rel=1e-12)
    assert fit.exponent_rma == pytest.approx(fit.exponent / abs(r), rel=1e-9)
    assert abs(fit.exponent_rma) > abs(fit.exponent) and np.sign(fit.exponent_rma) == np.sign(fit.exponent)


def test_regression_dilution_with_scatter_in_x():
    # y is an exact -1/3 power of the TRUE x; the observed x scatters.
    # OLS of log y on observed log x is attenuated toward 0; RMA is less so.
    rng = np.random.default_rng(7)
    lx_true = rng.uniform(0, 4, 400)
    lx_obs = lx_true + rng.normal(0, 0.5, lx_true.size)
    y = 10.0 ** (1.0 - lx_true / 3.0)
    fit = fit_power_law(10.0 ** lx_obs, y)
    assert fit.valid
    assert abs(fit.exponent) < abs(fit.exponent_rma) < 1.0 / 3.0
    assert abs(fit.exponent) < 1.0 / 3.0 - 5 * fit.exponent_stderr   # clearly attenuated
    # the reverse regression (x on y) gives a different exponent for the same data
    rev = fit_power_law(y, 10.0 ** lx_obs)
    assert abs(1.0 / rev.exponent - fit.exponent) > 0.02


def test_coefficient_overflow_and_underflow_do_not_raise():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # intercept +400: 10**400 overflows a float; the fit must stay usable
        fit = fit_power_law([1e-300, 1e-250, 1e-200], [1e100, 1e150, 1e200])
        assert fit.valid and fit.reason == "" and np.isposinf(fit.coefficient)
        assert fit.exponent == pytest.approx(1.0, abs=1e-9) and fit.exponent_rma == pytest.approx(1.0, abs=1e-9)
        assert fit.log10_coefficient == pytest.approx(400.0, abs=1e-6)
        # evaluation falls back to log space and gives finite, correct values
        assert fit(1e-300) == pytest.approx(1e100, rel=1e-6) and fit(1e-200) == pytest.approx(1e200, rel=1e-6)
        assert np.all(np.isfinite(fit(np.array([1e-300, 1e-250, 1e-200]))))
        # intercept -400: underflows to 0.0 coefficient, still usable
        fit = fit_power_law([1e100, 1e150, 1e200], [1e-300, 1e-250, 1e-200])
        assert fit.valid and fit.coefficient == 0.0 and fit.as_dict()["coefficient"] == 0.0
        assert fit.log10_coefficient == pytest.approx(-400.0, abs=1e-6)
        assert fit(1e100) == pytest.approx(1e-300, rel=1e-6)


def test_unusable_pairs_dropped_and_invalid_fits():
    x = np.array([1.0, 10.0, 100.0, -5.0, 1000.0, np.nan, 1e4, 1e5])
    y = np.array([2.0, 20.0, 200.0, 7.0, 2000.0, 1.0, 0.0, np.inf])
    fit = fit_power_law(x, y)                                   # non-positive and non-finite pairs dropped
    assert fit.valid and fit.n == 4
    assert fit.exponent == pytest.approx(1.0, abs=1e-10) and fit.coefficient == pytest.approx(2.0, rel=1e-9)
    fit = fit_power_law([1.0, 10.0], [1.0, 10.0])
    assert not fit.valid and fit.n == 2 and "fewer than 3" in fit.reason
    assert np.isnan(fit.exponent) and np.isnan(fit.coefficient)
    assert np.isnan(fit.exponent_rma) and np.isnan(fit.exponent_stderr) and np.isnan(fit(5.0))
    assert set(fit.as_dict()) == FIELDS
    assert not fit_power_law([], []).valid
    fit = fit_power_law([1.0, 10.0, 100.0], [1.0, 10.0, -1.0])   # three pairs, two usable
    assert not fit.valid and fit.n == 2
    fit = fit_power_law([-1.0, -2.0, 0.0], [1.0, 2.0, 3.0])
    assert not fit.valid and fit.n == 0
    fit = fit_power_law([3.0, 3.0, 3.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    assert not fit.valid and "identical" in fit.reason and fit.n == 4
    with pytest.raises(ValueError):
        fit_power_law([1.0, 2.0, 3.0], [1.0, 2.0])


def test_bootstrap_exponent():
    rng = np.random.default_rng(11)
    x = np.logspace(0, 4, 80)
    y = 10.0 ** (1.0 - 0.5 * np.log10(x) + rng.normal(0, 0.15, x.size))
    lo, hi = bootstrap_exponent(x, y, n_boot=400, seed=0)
    assert (lo, hi) == bootstrap_exponent(x, y, n_boot=400, seed=0)        # deterministic
    assert bootstrap_exponent(x, y, n_boot=400, seed=1) != (lo, hi)
    fit = fit_power_law(x, y)
    assert lo < fit.exponent < hi
    # 16-84 half-width is close to +-1 stderr of the OLS fit (observed ratio ~1.04);
    # a 95% or 90% interval (ratio ~1.9 / ~1.6) or an IQR (~0.66) would fail
    assert 0.85 < (hi - lo) / 2 / fit.exponent_stderr < 1.25
    # exactly the 16-84 percentiles of the resampled exponents, not another common interval
    rng = np.random.default_rng(0)
    exponents = [fit_power_law(x[idx], y[idx]).exponent
                 for idx in (rng.integers(0, x.size, size=x.size) for _ in range(300))]
    lo, hi = bootstrap_exponent(x, y, n_boot=300, seed=0)
    assert (lo, hi) == tuple(float(v) for v in np.percentile(exponents, [16.0, 84.0]))
    for pair in ([2.5, 97.5], [5.0, 95.0], [10.0, 90.0], [25.0, 75.0]):
        assert (lo, hi) != tuple(float(v) for v in np.percentile(exponents, pair))
    # exact data collapses; an invalid fit gives NaN
    x = np.logspace(0, 3, 12)
    lo, hi = bootstrap_exponent(x, 3.0 * x ** 2.0, n_boot=50)
    assert lo == pytest.approx(2.0, abs=1e-8) and hi == pytest.approx(2.0, abs=1e-8)
    assert all(np.isnan(bootstrap_exponent([1.0, 2.0], [1.0, 2.0])))
    with pytest.raises(ValueError):
        bootstrap_exponent(x, x, n_boot=0)


def test_reference_line():
    x = np.logspace(9, 15, 7)
    y = reference_line(x, -1.0 / 3.0, anchor_x=1e12, anchor_y=2e4)
    assert y[3] == pytest.approx(2e4)
    assert np.allclose(np.diff(np.log10(y)) / np.diff(np.log10(x)), -1.0 / 3.0)
    assert reference_line(1e12, -1.0 / 3.0, 1e12, 2e4) == pytest.approx(2e4)
    # one decade in x -> 10**(-1/3) in y
    assert reference_line(1e13, -1.0 / 3.0, 1e12, 2e4) == pytest.approx(2e4 * 10 ** (-1.0 / 3.0))
    for bad in [(0.0, 1.0), (1.0, -1.0), (np.nan, 1.0), (1.0, np.inf)]:
        pytest.raises(ValueError, reference_line, [1.0, 2.0], -1.0 / 3.0, *bad)
    pytest.raises(ValueError, reference_line, [1.0], np.nan, 1.0, 1.0)
