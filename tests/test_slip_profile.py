import json
import math

import numpy as np
import pytest

from labquake_explorer.analysis import slip_profile as sp

X = np.array([31.0, 93.0, 155.0, 217.0, 279.0, 341.0, 403.0, 465.0])
# coseismic slip (um) of real t0207e events at X, run/event
REAL = {
    "0/0": [9.71, 5.28, 37.02, 57.87, 6.91, 12.08, 43.31, -7.59],
    "0/3": [6.18, 2.91, 27.01, 55.38, 95.58, 64.09, 38.25, 20.31],
    "0/7": [-0.6243, -1.0231, -0.3146, 1.3223, 3.1136, 15.3814, 4.8029, 0.8165],
    "0/9": [13.94, 7.25, 9.84, -17.81, -71.25, -32.69, 8.84, 23.59],
    "0/10": [4.31, 0.29, 24.13, 48.47, 89.23, 64.96, 35.63, 19.4],
    "0/19": [0.15, 0.12, 0.72, 2.54, 1.55, 16.02, 3.41, 0.08],
    "1/0": [3.38, 1.96, 22.36, 54.2, 22.0, 54.83, 32.28, 14.91],
    "1/4": [0.03, -3.55, -24.45, -56.26, -111.99, -73.72, -36.30, -15.61],
    "4/8": [8.84, 16.91, 36.54, 73.88, 129.78, 103.85, 52.93, 29.44],
}


def cost(fit, y, w=1.0):
    return 0.5 * float(np.sum((w * (fit.evaluate(X) - np.asarray(y))) ** 2))


@pytest.mark.parametrize("lam", [0.1, 0.4, 0.7, 0.95])
def test_eaz_is_c1_continuous_and_reaches_zero_at_the_tips(lam):
    D, a = 100.0, 200.0
    rj = sp.eaz_joint(a, lam)
    assert rj == pytest.approx(a * (math.sqrt(1 + 3 * lam ** 2) - 1))
    assert 0 < rj < lam * a < a
    h = 1e-6 * a
    left = sp.eaz_half(np.array([rj - h, rj]), D, a, lam)
    right = sp.eaz_half(np.array([rj, rj + h]), D, a, lam)
    assert left[1] == pytest.approx(right[0], rel=1e-12)                       # continuous value
    assert (left[1] - left[0]) / h == pytest.approx((right[1] - right[0]) / h, rel=1e-4)   # continuous slope
    assert sp.eaz_half(np.array([0.0, a, a + 1.0, 10 * a]), D, a, lam).tolist() == [D, 0.0, 0.0, 0.0]
    # symmetric about the centre
    x = np.linspace(-300, 300, 61)
    np.testing.assert_allclose(sp.eaz_profile(x, D, 0.0, a, lam), sp.eaz_profile(-x, D, 0.0, a, lam))


def test_eaz_limits_and_asymmetric_form():
    x = np.linspace(-300, 300, 61)
    ellipse = 100 * np.sqrt(np.clip(1 - (x / 250) ** 2, 0, None))
    np.testing.assert_allclose(sp.eaz_profile(x, 100, 0, 250, 1.0), ellipse)
    # the asymmetric form with equal sides is the symmetric one; each side follows its own a and lambda
    np.testing.assert_allclose(sp.eaz_asymmetric_profile(x, 90, 20, 200, 200, 0.5, 0.5), sp.eaz_profile(x, 90, 20, 200, 0.5))
    asym = sp.eaz_asymmetric_profile(x, 90, 20, 150, 250, 0.3, 0.8)
    np.testing.assert_allclose(asym[x < 20], sp.eaz_profile(x[x < 20], 90, 20, 150, 0.3))
    np.testing.assert_allclose(asym[x >= 20], sp.eaz_profile(x[x >= 20], 90, 20, 250, 0.8))
    assert sp.eaz_half(np.array([1.0]), 1.0, 0.0, 0.5).tolist() == [0.0]


def test_quadratic_mask():
    y = sp.quadratic_masked(X, -524.0, 5.0, -0.01)
    assert np.all(y >= 0) and y[0] == 0.0 and y[4] > 0


@pytest.mark.parametrize("true", [
    {"D": 110.0, "xc": 300.0, "a": 260.0, "lambda": 0.6},
    {"D": 80.0, "xc": 240.0, "a": 180.0, "lambda": 0.4},
])
def test_eaz_fit_recovers_the_parameters(true):
    rng = np.random.default_rng(3)
    y = sp.eaz_profile(X, *true.values()) + rng.normal(0, 0.1, X.size)
    fit = sp.fit_profile(X, y, model="eaz")
    assert fit.valid and fit.n == 8 and fit.r2 > 0.9999 and not fit.at_bounds
    for k, v in true.items():
        assert fit.params[k] == pytest.approx(v, rel=0.02), k
    assert fit.tips_mm == pytest.approx((true["xc"] - true["a"], true["xc"] + true["a"]), rel=0.02)
    assert fit.peak_x_mm == pytest.approx(true["xc"], rel=0.01) and fit.peak == pytest.approx(true["D"], rel=0.01)
    np.testing.assert_allclose(fit.evaluate(X), sp.eaz_profile(X, *fit.params.values()))


def test_eaz_asymmetric_fit_recovers_the_parameters():
    true = dict(D=100.0, xc=260.0, a_minus=200.0, a_plus=150.0, lambda_minus=0.5, lambda_plus=0.8)
    y = sp.eaz_asymmetric_profile(X, *true.values())
    fit = sp.fit_profile(X, y, model="eaz_asymmetric")
    assert fit.valid and fit.r2 > 0.9999
    for k, v in true.items():
        assert fit.params[k] == pytest.approx(v, rel=0.02), k
    assert fit.tips_mm == pytest.approx((60.0, 410.0), rel=0.02)


def test_quadratic_fit_with_zero_sensors_outside_the_rupture():
    rng = np.random.default_rng(1)
    y = np.maximum(0, 100 - 0.01 * (X - 250) ** 2) + rng.normal(0, 0.1, X.size)
    fit = sp.fit_profile(X, y, model="quadratic")
    assert fit.valid and fit.r2 > 0.9999
    assert fit.params["c2"] == pytest.approx(-0.01, rel=0.01)
    assert fit.tips_mm == pytest.approx((150.0, 350.0), abs=1.0)
    assert fit.peak_x_mm == pytest.approx(250.0, abs=0.5) and fit.peak == pytest.approx(100.0, abs=0.5)
    assert fit.evaluate(X)[[0, 1, 6, 7]].tolist() == [0.0, 0.0, 0.0, 0.0]
    # an ordinary parabola that stays positive is fitted exactly too
    y2 = 100 - 0.0008 * (X - 250) ** 2
    fit2 = sp.fit_profile(X, y2, model="quadratic")
    assert fit2.params == pytest.approx({"c0": 50.0, "c1": 0.4, "c2": -0.0008}, rel=1e-6)


def test_fit_options_and_failures():
    y = sp.eaz_profile(X, 100, 280, 250, 0.5)
    sigma = np.full(8, 0.1)
    w = sp.fit_profile(X, y, sigma, model="eaz", weighted=True)
    assert w.valid and w.weighted and w.r2 > 0.9999
    use = np.ones(8, dtype=bool)
    use[[0, 7]] = False
    part = sp.fit_profile(X, y, model="eaz", use=use)
    assert part.n == 6 and part.valid
    assert not sp.fit_profile(X[:4], y[:4], model="eaz").valid
    assert "at least 5" in sp.fit_profile(X[:4], y[:4], model="eaz").reason
    assert "at least 7" in sp.fit_profile(X[:6], y[:6], model="eaz_asymmetric").reason
    assert "positive" in sp.fit_profile(X, np.zeros(8), model="quadratic").reason
    bad_sigma = sigma.copy()
    bad_sigma[:5] = 0.0
    assert not sp.fit_profile(X, y, bad_sigma, model="eaz", weighted=True).valid
    nan_y = y.copy()
    nan_y[3] = np.nan
    assert sp.fit_profile(X, nan_y, model="eaz").n == 7
    with pytest.raises(ValueError):
        sp.fit_profile(X, y, model="cubic")
    with pytest.raises(ValueError):
        sp.fit_profile(X, y[:-1], model="eaz")
    assert np.isnan(sp.fit_profile(X[:3], y[:3], model="eaz").evaluate(X)).all()


def test_fit_record_is_json_like():
    fit = sp.fit_profile(X, sp.eaz_profile(X, 100, 280, 250, 0.5), model="eaz")
    d = fit.as_dict()
    json.dumps(d)
    assert d["model"] == "eaz" and set(d["params"]) == {"D", "xc", "a", "lambda"}
    assert {"r2", "rms_um", "tips_mm", "peak_x_mm", "peak_um", "n", "weighted", "valid", "reason", "at_bounds"} <= set(d)


def test_front_single_and_bilateral():
    t = 0.02 + np.abs(X - 279.0) * 1e-3 / 50.0
    single = sp.fit_front(X[4:], t[4:])
    assert single.valid and single.segments[0].velocity_m_s == pytest.approx(50.0)
    assert single.segments[0].r2 == pytest.approx(1.0)
    both = sp.fit_front(X, t, mode="bilateral")
    assert both.split_x_mm == 279.0 and len(both.segments) == 2
    left, right = both.segments
    assert left.velocity_m_s == pytest.approx(-50.0) and right.velocity_m_s == pytest.approx(50.0)
    assert (left.n, right.n) == (5, 4)
    use = np.ones(8, dtype=bool)
    use[0] = False
    assert sp.fit_front(X, t, use, mode="bilateral").segments[0].n == 4
    assert not sp.fit_front(X[:1], t[:1]).valid
    one_side = sp.fit_front(X[4:], t[4:], mode="bilateral")                 # the earliest sensor is at one end
    assert one_side.valid and len(one_side.segments) == 1
    assert sp.fit_front(X, np.full(8, 0.1)).segments[0].velocity_m_s == math.inf
    d = both.as_dict()
    json.dumps(d)
    assert d["mode"] == "bilateral" and len(d["segments"]) == 2
    with pytest.raises(ValueError):
        sp.fit_front(X, t, mode="trilateral")


# ------------------------------------------------------------ profile search
@pytest.mark.parametrize("name, minimum, tips", [
    ("0/7", 1.9753034, (271.31, 415.27)),
    ("0/19", 3.50665, (275.66, 410.60)),
    ("edge", 9.41742, (311.31, 642.07)),       # vertex past the last sensor
])
def test_quadratic_reaches_the_global_minimum(name, minimum, tips):
    # the old starts, all centred on the array, ended in a wide low parabola (or a convex one)
    y = REAL.get(name, [1.48, 0.242, 0.204, 3.096, -2.638, 33.773, 82.818, 102.817])
    fit = sp.fit_profile(X, y, model="quadratic")
    assert fit.valid and fit.params["c2"] < 0
    assert cost(fit, y) == pytest.approx(minimum, rel=1e-6)
    assert fit.tips_mm == pytest.approx(tips, abs=0.01)


def test_quadratic_minimum_with_the_parabola_zero_at_a_sensor():
    # the minimum is a convex parabola that touches zero at a sensor with negative slip: the
    # least-squares parabola through the positive sensors that is zero there
    fit = sp.fit_profile(X, REAL["0/9"], model="quadratic")
    assert fit.params["c2"] > 0 and math.isnan(fit.tips_mm[0])
    assert cost(fit, REAL["0/9"]) == pytest.approx(3286.0292678, rel=1e-7)


def test_arrest_zone_fits_find_narrow_and_two_sided_minima():
    eaz = sp.fit_profile(X, REAL["0/0"], model="eaz")             # a narrow rupture: the old starts stopped at R2 0.362
    assert eaz.r2 > 0.3858 and eaz.tips_mm == pytest.approx((87.8, 304.5), abs=0.5)
    asym = sp.fit_profile(X, REAL["1/0"], model="eaz_asymmetric")  # the old fit stopped at 0.711, lambda+ at its bound
    assert asym.r2 > 0.726


def test_fit_that_does_not_converge_is_invalid(monkeypatch):
    # one sensor weighted 2e5 times more than the others: the cost still falls fast after the budget
    y = np.array([-0.401, 17.127, 66.386, 94.101, 100.397, 88.793, 50.919, 7.494])
    sigma = np.full(8, 0.2)
    sigma[3] = 1e-6
    fit = sp.fit_profile(X, y, sigma, model="eaz", weighted=True)
    assert not fit.valid and fit.reason == "the fit did not converge"
    monkeypatch.setattr(sp, "MAX_NFEV_FINAL", 2000)
    fit = sp.fit_profile(X, y, sigma, model="eaz", weighted=True)
    assert fit.valid
    assert list(fit.params.values()) == pytest.approx([100.30, 270.10, 218.68, 0.701], rel=1e-3)
    # a flat valley (lambda free on one side) creeps with a stable cost: converged
    flat = sp.fit_profile(X, REAL["0/3"], model="eaz_asymmetric")
    assert flat.valid and flat.r2 > 0.99


def test_weighted_fit_leaves_out_a_negligible_sigma():
    # a constant (dead or saturated) record: slip and sigma both ~1e-12
    y = sp.eaz_profile(X, 100.0, 270.0, 220.0, 0.7)
    y[3] = 2.3e-12
    sigma = np.full(8, 0.2)
    sigma[3] = 2.7e-13
    fit = sp.fit_profile(X, y, sigma, model="eaz", weighted=True)
    use = np.ones(8, dtype=bool)
    use[3] = False
    without = sp.fit_profile(X, y, sigma, model="eaz", weighted=True, use=use)
    assert fit.n == 7 and fit.params == pytest.approx(without.params)
    assert fit.tips_mm == pytest.approx((50.0, 490.0), abs=0.1)


def test_profile_within_the_noise_is_invalid():
    # a neighbour-contaminated event: the only positive slip, 0.03 um, is far inside its noise
    sigma = [1.5, 1.28, 4.49, 14.86, 32.26, 22.73, 6.99, 3.28]
    for model in sp.MODELS:
        fit = sp.fit_profile(X, REAL["1/4"], sigma, model=model)
        assert not fit.valid and fit.reason == "the fitted profile is within the noise at every sensor", model
        assert np.isnan(fit.evaluate(X)).all()
    # without sigma the 0.03 um is fitted, by a profile the sensors do not pin
    bump = sp.fit_profile(X, REAL["1/4"], model="eaz")
    assert bump.valid and np.count_nonzero(bump.evaluate(X) > 0) == 1 and "tips" in bump.unresolved


def test_analytic_jacobians_match_differences():
    rng = np.random.default_rng(4)
    x = np.linspace(0.0, 500.0, 26)
    for model, jacobian, f in (("eaz", sp._eaz_jacobian, sp.eaz_profile),
                               ("eaz_asymmetric", sp._eaz_asymmetric_jacobian, sp.eaz_asymmetric_profile)):
        sides = 1 if model == "eaz" else 2
        for _ in range(20):
            p = np.concatenate([[rng.uniform(20, 100), rng.uniform(150, 350)],          # D, xc
                                rng.uniform(80, 300, sides), rng.uniform(0.05, 0.95, sides)])   # a, lambda
            h = 1e-7 * np.maximum(np.abs(p), 1.0)
            numeric = np.column_stack([(f(x, *(p + d)) - f(x, *(p - d))) / (2 * d[k])
                                       for k, d in enumerate(np.diag(h))])
            np.testing.assert_allclose(jacobian(x, p), numeric, rtol=1e-4, atol=1e-6 * np.abs(numeric).max())


# ---------------------------------------------------------- identifiability
def test_well_sampled_fits_have_nothing_unresolved():
    for true in ((110.0, 300.0, 260.0, 0.6), (80.0, 240.0, 180.0, 0.4)):
        fit = sp.fit_profile(X, sp.eaz_profile(X, *true), model="eaz")
        assert fit.unresolved == [] and fit.ranges == {}
    fit = sp.fit_profile(X, sp.eaz_profile(X, 100.0, 280.0, 250.0, 0.5), model="quadratic")
    assert fit.unresolved == []


def test_lambda_is_free_when_no_sensor_is_within_the_joint():
    # r_joint is 3.4 mm, the nearest sensor 31 mm from xc: the sensors see only the 3/2 tip part
    fit = sp.fit_profile(X, sp.eaz_profile(X, 100.0, 248.0, 230.0, 0.1), model="eaz")
    assert fit.unresolved == ["lambda"]
    lam_max = math.sqrt(((1 + 31.0 / 230.0) ** 2 - 1) / 3)
    assert fit.ranges["lambda"] == pytest.approx([0.0, lam_max], abs=1e-6)
    assert fit.ranges["D"][0] < fit.params["D"] < fit.ranges["D"][1]
    # both ends of the range, with D following, give the same profile at the sensors
    for lam, D in ((lam_max, fit.ranges["D"][0]), (1e-9, fit.ranges["D"][1])):
        moved = sp.eaz_profile(X, D, fit.params["xc"], fit.params["a"], lam)
        np.testing.assert_allclose(moved, fit.evaluate(X), atol=1e-9)
    d = fit.as_dict()
    json.dumps(d)
    assert d["unresolved"] == ["lambda"] and set(d["ranges"]) == {"lambda", "D"}


def test_ellipse_fixes_only_lambda_times_a():
    # every sensor inside r_joint of an elliptical crack whose tips are off the array
    fit = sp.fit_profile(X, sp.eaz_profile(X, 100.0, 250.0, 260.0, 1.0), model="eaz")
    assert fit.unresolved == ["a", "lambda", "tips"]
    assert fit.params["lambda"] * fit.params["a"] == pytest.approx(260.0, rel=1e-6)
    assert fit.ranges["lambda"][1] == 1.0 and fit.ranges["a"][0] == pytest.approx(260.0, rel=1e-6)
    assert fit.ranges["tip_minus"][0] <= -10.0 <= fit.ranges["tip_minus"][1]
    assert fit.ranges["tip_plus"][0] <= 510.0 <= fit.ranges["tip_plus"][1]


def test_too_few_slipping_sensors_leave_the_extent_free():
    # three slipping sensors (279, 341, 403): the tips are only bracketed by the sensors
    for model, extent in (("eaz", ["a", "lambda"]), ("quadratic", [])):
        fit = sp.fit_profile(X, REAL["0/7"], model=model)
        assert fit.unresolved == extent + ["tips"], model
        assert fit.ranges["tip_minus"] == [217.0, 279.0] and fit.ranges["tip_plus"] == [403.0, 465.0]


@pytest.mark.parametrize("name, free", [("0/3", True), ("0/10", True), ("4/8", True), ("1/0", False)])
def test_lambda_on_real_events(name, free):
    fit = sp.fit_profile(X, REAL[name], model="eaz")
    assert ("lambda" in fit.unresolved) == free
    if free:
        assert fit.unresolved == ["lambda"] and set(fit.ranges) == {"lambda", "D"}     # D follows within its range
        assert fit.ranges["lambda"][0] == 0.0 and fit.ranges["lambda"][1] < 0.25


def test_asymmetric_peak_free_when_neither_side_has_a_sensor_within_its_joint():
    true = (100.0, 248.0, 200.0, 260.0, 0.1, 0.15)
    fit = sp.fit_profile(X, sp.eaz_asymmetric_profile(X, *true), model="eaz_asymmetric")
    assert fit.r2 > 0.999999 and fit.unresolved == ["D", "lambda_minus", "lambda_plus"]
    for name, v in zip(("D", "lambda_minus", "lambda_plus"), (true[0], true[4], true[5])):
        assert fit.ranges[name][0] <= v <= fit.ranges[name][1], name
    # here the sensor at the peak fixes D on the plus side: lambda- moves the cost, so no range
    one_side = sp.fit_profile(X, REAL["0/3"], model="eaz_asymmetric")
    assert "lambda_minus" not in one_side.ranges and "D" not in one_side.unresolved


# ---------------------------------------------------------------- the front
def test_front_slope_zero_to_rounding_and_one_sample():
    # real t0207e run 1 event 0: symmetric about 279 mm, so the least-squares slope is 0
    x = np.array([155.0, 217.0, 279.0, 341.0, 403.0])
    t = np.array([0.042, 0.042, 0.0425, 0.042, 0.042])
    seg = sp.fit_front(x, t).segments[0]
    assert seg.slope_s_per_mm == 0.0 and seg.velocity_m_s == math.inf and seg.intercept_s == pytest.approx(t.mean())
    assert seg.resolved and math.isnan(seg.min_speed_m_s)
    seg = sp.fit_front(x, t, dt_s=5e-4).segments[0]
    assert not seg.resolved and seg.min_speed_m_s == pytest.approx(496.0)
    both = sp.fit_front(x, t, mode="bilateral", dt_s=5e-4)       # the middle sensor half a sample late
    assert both.split_x_mm == 279.0 and [s.velocity_m_s > 0 for s in both.segments] == [True, False]
    assert not any(s.resolved for s in both.segments) and both.segments[0].min_speed_m_s == pytest.approx(248.0)


@pytest.mark.parametrize("x0, v_minus, v_plus", [(248.0, -50.0, 50.0), (300.0, -100.0, 100.0), (240.0, -50.0, 150.0)])
def test_bilateral_hinge_is_exact(x0, v_minus, v_plus):
    # the apex between sensors: the old fit put the earliest sensor on both lines
    t = 0.02 + np.where(X < x0, (x0 - X) / -v_minus, (X - x0) / v_plus) * 1e-3
    fit = sp.fit_front(X, t, mode="bilateral", dt_s=5e-4)
    assert fit.valid and fit.split_x_mm == pytest.approx(x0, abs=1e-9) and fit.t0_s == pytest.approx(0.02, abs=1e-15)
    left, right = fit.segments
    assert (left.velocity_m_s, right.velocity_m_s) == pytest.approx((v_minus, v_plus), rel=1e-9)
    assert (left.x_from, left.x_to, right.x_from, right.x_to) == pytest.approx((31.0, x0, x0, 465.0))
    assert (left.n, right.n) == (np.count_nonzero(X <= x0), np.count_nonzero(X >= x0))
    assert left.r2 == pytest.approx(1.0) and right.r2 == pytest.approx(1.0) and left.resolved and right.resolved
    assert left.min_speed_m_s == pytest.approx((x0 - 31.0) * 1e-3 / 5e-4)
    d = fit.as_dict()
    json.dumps(d)
    assert d["t0_s"] == fit.t0_s and {"resolved", "min_speed_m_s"} <= set(d["segments"][0])


def test_bilateral_hinge_with_ties_and_one_way_fronts():
    t_ms = np.array([3.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0])     # 62 m/s both ways, two sensors tied first
    fit = sp.fit_front(X, t_ms * 1e-3, mode="bilateral")
    assert fit.split_x_mm == pytest.approx(248.0)
    assert [s.velocity_m_s for s in fit.segments] == pytest.approx([-62.0, 62.0])
    t_ms = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])     # the old fit gave one line, 67.6 m/s
    fit = sp.fit_front(X, t_ms * 1e-3, mode="bilateral", dt_s=5e-4)
    left, right = fit.segments
    assert fit.split_x_mm == pytest.approx(93.0) and left.velocity_m_s == math.inf and not left.resolved
    assert right.velocity_m_s == pytest.approx(62.0)
    one_way = sp.fit_front(X, 0.01 + (X - 31.0) * 1e-3 / 50.0, mode="bilateral")
    assert len(one_way.segments) == 1 and one_way.segments[0].velocity_m_s == pytest.approx(50.0)
    assert one_way.split_x_mm == 31.0 and one_way.t0_s == pytest.approx(0.01)
    flat = sp.fit_front(X, np.full(8, 0.1), mode="bilateral")
    assert len(flat.segments) == 1 and flat.segments[0].velocity_m_s == math.inf and math.isnan(flat.split_x_mm)


def test_bilateral_with_three_sensors_keeps_two_lines_through_the_earliest():
    x = np.array([217.0, 279.0, 341.0])
    fit = sp.fit_front(x, np.array([77.0, 75.0, 75.5]) * 1e-3, mode="bilateral")      # real run 0 event 10
    assert fit.split_x_mm == 279.0 and fit.t0_s == pytest.approx(0.075)
    assert [s.velocity_m_s for s in fit.segments] == pytest.approx([-31.0, 124.0])
    tied = sp.fit_front(x, np.array([75.0, 75.0, 77.0]) * 1e-3, mode="bilateral")      # the middle of the tied ones
    assert tied.split_x_mm == 279.0 and tied.segments[0].velocity_m_s == math.inf
    at_end = sp.fit_front(x, np.array([75.0, 76.0, 77.0]) * 1e-3, mode="bilateral")
    assert at_end.split_x_mm == 217.0 and len(at_end.segments) == 1


def test_held_parameters_and_user_bounds():
    x = X
    y = sp.eaz_profile(x, 100.0, 270.0, 230.0, 0.6) + np.array([0.3, -0.2, 0.1, 0.0, -0.1, 0.2, -0.3, 0.1])
    free = sp.fit_profile(x, y, model="eaz")
    held = sp.fit_profile(x, y, model="eaz", fixed={"lambda": 0.6})
    assert held.valid and held.fixed == ["lambda"] and held.params["lambda"] == 0.6
    assert held.params["xc"] == pytest.approx(270.0, abs=2.0) and held.params["a"] == pytest.approx(230.0, rel=0.02)
    assert "lambda" not in held.unresolved and held.r2 <= free.r2 + 1e-9
    capped = sp.fit_profile(x, y, model="eaz", bounds={"a": (None, 200.0)})
    assert capped.params["a"] == pytest.approx(200.0) and capped.at_bounds == ["a"]
    params = {"D": 90.0, "xc": 280.0, "a": 250.0, "lambda": 0.5}
    manual = sp.fit_profile(x, y, model="eaz", fixed=params)            # everything held: no fit
    assert manual.valid and manual.params == params and manual.fixed == list(params) and manual.unresolved == []
    np.testing.assert_allclose(manual.evaluate(x), sp.eaz_profile(x, **{"D": 90.0, "xc": 280.0, "a": 250.0,
                                                                          "lam": 0.5}))
    quad = sp.fit_profile(x, np.maximum(0.0, 100 - 0.004 * (x - 250.0) ** 2), model="quadratic",
                          fixed={"c2": -0.004})
    assert quad.params["c2"] == -0.004 and quad.params["c0"] + quad.params["c1"] * 250 + -0.004 * 250 ** 2 == \
        pytest.approx(100.0, abs=1e-3)


@pytest.mark.parametrize("kwargs, reason", [({"fixed": {"lambda": 1.5}}, "lambda must be > 0 and <= 1"),
                                            ({"fixed": {"a": 0.0}}, "a must be > 0"),
                                            ({"fixed": {"D": float("nan")}}, "D must be a number"),
                                            ({"bounds": {"a": (300.0, 200.0)}}, "the minimum must be below"),
                                            ({"bounds": {"lambda": (2.0, None)}}, "no allowed value")])
def test_unusable_held_values_and_bounds(kwargs, reason):
    fit = sp.fit_profile(X, sp.eaz_profile(X, 100.0, 270.0, 230.0, 0.6), model="eaz", **kwargs)
    assert not fit.valid and reason in fit.reason
    with pytest.raises(ValueError, match="no parameter"):
        sp.fit_profile(X, sp.eaz_profile(X, 100.0, 270.0, 230.0, 0.6), model="eaz", fixed={"c0": 1.0})
