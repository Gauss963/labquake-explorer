import json
import math

import numpy as np
import pytest

from labquake_explorer.analysis import coseismic_slip as cs
from labquake_explorer.data.data_manager import DataManager
from tests.synthetic import SLIP_X_MM, add_slip_array

FS = 2000.0
EVENT_TIME = 10.0


def make_event(**kwargs):
    t = EVENT_TIME + np.arange(-5.0, 5.0, 1.0 / FS)
    event = {"event_time": EVENT_TIME, "time": t}
    truth = add_slip_array(event, **kwargs)
    return event, truth


def run(event, **kwargs):
    s = event["slip"]
    return cs.analyze_slip(event["time"], s["data"], event["event_time"], -5.0, 5.0,
                           channels=s["channels"], x_mm=s["positions"]["x"], **kwargs)


def test_default_windows_and_validation():
    assert cs.default_fit_windows(-5.0, 5.0) == ((-3.75, -1.25), (1.25, 3.75))
    assert cs.default_fit_windows(-2.0, 6.0) == ((-1.5, -0.5), (1.5, 4.5))
    for bad in ((1.0, 5.0), (-5.0, -1.0), (0.0, 5.0)):
        with pytest.raises(ValueError):
            cs.default_fit_windows(*bad)
    assert cs.check_window((-2, -1)) == (-2.0, -1.0)
    for bad in ((-1, -2), (0, 0), ("a", 1), (1,), (float("nan"), 1)):
        with pytest.raises(ValueError):
            cs.check_window(bad)


def test_fit_line_and_rate_helpers():
    t = np.arange(0.0, 1.0, 0.001)
    y = 3.0 * (t - 0.4) + 2.0
    fit = cs.fit_line(t, y, 0.4, (-0.3, -0.1))
    assert fit.valid and fit.slope == pytest.approx(3.0) and fit.value == pytest.approx(2.0)
    assert fit.rms == pytest.approx(0.0, abs=1e-9) and fit.n == 201 and not fit.clipped   # both edge samples included
    assert fit.used == pytest.approx((-0.3, -0.1))
    clipped = cs.fit_line(t, y, 0.4, (-1.0, -0.1))
    assert clipped.clipped and clipped.valid and clipped.used[0] == pytest.approx(-0.4)
    short = cs.fit_line(t, y, 0.4, (-0.0015, -0.0005))
    assert not short.valid and short.n < cs.MIN_FIT_SAMPLES and math.isnan(short.value)
    assert cs.smoothing_samples(0.0025, 0.0005) == 5 and cs.smoothing_samples(0.002, 0.0005) == 5
    assert cs.smoothing_samples(0.0, 0.0005) == 1
    rate = cs.slip_rate(t, y, 0.01)
    assert rate.shape == y.shape and np.allclose(rate, 3.0)
    two = cs.slip_rate(t, np.vstack([y, 2 * y]), 0.0)
    assert two.shape == (2, t.size) and np.allclose(two[1], 6.0)


def test_max_rate_picks_recover_each_sensor():
    event, truth = make_event(noise_um=0.05, slips_um=[0.0, 40.0, 70.0, 90.0, 100.0, 90.0, 70.0, 0.0])
    res = run(event)
    # the reference sits at the peak of the mean rate, a few ms after the first arrival
    assert len(res.sensors) == 8 and truth["t_i"].min() <= res.t_ref <= truth["t_i"].min() + 0.005
    big = truth["slips_um"] > 20.0
    for s, t_i, slip, is_big in zip(res.sensors, truth["t_i"], truth["slips_um"], big):
        assert s.method == cs.METHOD_MAX_RATE
        if is_big:
            assert EVENT_TIME + s.t_shift_s == pytest.approx(t_i, abs=1.0 / FS)
            assert not s.flags, (s.channel, s.flags)
        assert s.coseismic_slip == pytest.approx(slip, abs=0.05)
        assert s.noise == pytest.approx(0.05, rel=0.2)
        assert s.before.window == (-3.75, -1.25) and s.after.window == (1.25, 3.75)
    # sensors without a step: the picks are flagged so the user knows to drag them
    flat = [s for s, sl in zip(res.sensors, truth["slips_um"]) if sl == 0.0]
    assert flat and all(cs.FLAG_WEAK in s.flags or cs.FLAG_EDGE in s.flags for s in flat)


def test_manual_times_and_fallback():
    event, truth = make_event()
    shifts = list(truth["t_i"] - EVENT_TIME)
    shifts[4] = shifts[4] + 0.010                       # a manual time 10 ms late
    shifts[5] = float("nan")                            # no shift: falls back to max_rate
    methods = ["max_rate"] * 4 + ["manual", "manual", "max_rate", "max_rate"]
    res = run(event, methods=methods, t_shift_s=shifts)
    assert res.sensors[4].method == "manual" and res.sensors[4].t_shift_s == pytest.approx(shifts[4])
    assert res.sensors[5].method == "max_rate"
    assert res.sensors[4].coseismic_slip == pytest.approx(truth["slips_um"][4], abs=0.1)   # the lines are far from t_i
    with pytest.raises(ValueError):
        run(event, methods=["bogus"] * 8)
    with pytest.raises(ValueError):
        run(event, methods=["manual"] * 3)


def test_windows_one_pair_or_per_sensor_and_clipping():
    event, truth = make_event()
    res = run(event, fit_before_s=(-2.0, -0.5), fit_after_s=(0.5, 2.0))
    assert all(s.before.window == (-2.0, -0.5) and s.after.window == (0.5, 2.0) for s in res.sensors)
    per = [(-1.0 - 0.1 * i, -0.2) for i in range(8)]
    res = run(event, fit_before_s=per)
    assert [s.before.window for s in res.sensors] == [pytest.approx(w) for w in per]
    res = run(event, fit_after_s=(1.0, 9.0))
    assert all(cs.FLAG_AFTER_CLIPPED in s.flags for s in res.sensors)
    assert all(s.after.valid for s in res.sensors)
    res = run(event, fit_before_s=(-0.0004, -0.0001))
    assert all(cs.FLAG_BEFORE_INVALID in s.flags and math.isnan(s.coseismic_slip) for s in res.sensors)
    with pytest.raises(ValueError):
        run(event, fit_before_s=[(-1, -0.5)] * 3)
    with pytest.raises(ValueError):
        run(event, fit_before_s=(-0.5, -1.0))


def test_settings_validation():
    event, _ = make_event()
    for kwargs in ({"rate_smoothing_s": -1}, {"search_halfwidth_s": 0}, {"rate_snr": -2},
                   {"rate_smoothing_s": float("nan")}):
        with pytest.raises(ValueError):
            run(event, **kwargs)
    s = event["slip"]
    with pytest.raises(ValueError):
        cs.analyze_slip(event["time"], s["data"][:, :-1], EVENT_TIME, -5.0, 5.0)
    with pytest.raises(ValueError):
        cs.analyze_slip(event["time"], s["data"], EVENT_TIME, -5.0, 5.0, channels=["a"])


def test_reference_time_near_the_event_time():
    t = np.arange(0.0, 10.0, 0.001)
    rate = np.zeros((2, t.size))
    rate[:, np.argmin(np.abs(t - 6.0))] = 100.0           # a neighbouring event 1 s later: ignored
    rate[:, np.argmin(np.abs(t - 5.4))] = 50.0
    assert cs.reference_time(t, rate, 5.0) == pytest.approx(5.4)
    assert cs.reference_time(t, rate, 5.0, halfwidth_s=1.5) == pytest.approx(6.0)
    assert cs.reference_time(t, np.full((2, t.size), np.nan), 5.0) == 5.0


def test_a_neighbouring_event_does_not_capture_the_picks():
    """Real run 5 event 7 of t0207: the next event 1.1 s later is larger, and a
    reference searched over +-1.25 s jumped to it, copying its slips."""
    event, truth = make_event()
    s = event["slip"]
    t = event["time"]
    neighbour = 3.0 * truth["slips_um"][:, None] * 0.5 * (1.0 + np.tanh((t - (EVENT_TIME + 1.1)) / 0.002))
    s["data"] = s["data"] + neighbour
    res = run(event, fit_after_s=(0.2, 0.8))
    assert abs(res.t_ref - EVENT_TIME) < 0.05
    for sensor, t_i in zip(res.sensors, truth["t_i"]):
        assert EVENT_TIME + sensor.t_shift_s == pytest.approx(t_i, abs=1.0 / FS)
    wide = run(event, reference_halfwidth_s=1.5)
    assert wide.t_ref == pytest.approx(EVENT_TIME + 1.1, abs=0.01)     # what the old +-1.25 s search did
    with pytest.raises(ValueError):
        run(event, reference_halfwidth_s=0.0)


def test_nan_samples_stay_local():
    event, truth = make_event()
    s = event["slip"]
    s["data"][4, 3000] = np.nan                           # one bad sample 3.5 s before the event
    s["data"][2, :20] = np.nan                            # a short gap at the start of the record
    rate = cs.slip_rate(event["time"], s["data"], 0.0025)
    assert np.isfinite(rate[4]).all()                    # the average of the finite neighbours fills it
    assert abs(rate[4, 3000] - 3.0) < 100.0              # creep of 3 um/s plus the noise derivative
    assert np.isfinite(rate[2, 30:]).all() and not np.isfinite(rate[2, :10]).any()
    res = run(event)
    for k in (2, 4):
        assert EVENT_TIME + res.sensors[k].t_shift_s == pytest.approx(truth["t_i"][k], abs=1.0 / FS)
        assert res.sensors[k].coseismic_slip == pytest.approx(truth["slips_um"][k], abs=0.05)
    # a window mostly emptied by NaN samples is flagged
    s["data"][5, (event["time"] > EVENT_TIME + 1.3)] = np.nan
    res = run(event)
    assert cs.FLAG_AFTER_GAPS in res.sensors[5].flags and res.sensors[5].after.gaps
    assert cs.FLAG_AFTER_GAPS not in res.sensors[4].flags


def test_small_inputs():
    event, truth = make_event()
    s = event["slip"]
    # a search covering one sample is at its own edge
    res = run(event, search_halfwidth_s=0.0002)
    assert all(cs.FLAG_EDGE in x.flags for x in res.sensors)
    # one value for all sensors, as a scalar
    res = run(event, methods="manual", t_shift_s=0.02)
    assert all(x.method == "manual" and x.t_shift_s == pytest.approx(0.02) for x in res.sensors)
    res = run(event, methods="manual", t_shift_s=np.float64(0.03))
    assert all(x.t_shift_s == pytest.approx(0.03) for x in res.sensors)
    for n in (0, 1):
        with pytest.raises(ValueError):
            cs.analyze_slip(EVENT_TIME + np.arange(n) / FS, np.zeros((2, n)), EVENT_TIME, -5.0, 5.0)
        with pytest.raises(ValueError):
            cs.slip_rate(np.arange(n) / FS, np.zeros(n))
    rec = run(event).as_dict()
    assert len(rec["rate_noise_um_s"]) == 8 and rec["reference_halfwidth_s"] == 0.5


def test_record_is_json_like_and_round_trips(tmp_path):
    event, truth = make_event()
    methods = ["max_rate"] * 7 + ["manual"]
    shifts = [0.0] * 7 + [0.123]
    res = run(event, methods=methods, t_shift_s=shifts, fit_before_s=(-3.0, -1.0))
    rec = res.as_dict()
    json.dumps(rec)
    assert rec["version"] == cs.RESULT_VERSION and rec["channels"] == [f"slip_{i}" for i in range(1, 9)]
    assert rec["x_mm"] == list(SLIP_X_MM) and rec["time_method"] == methods
    assert rec["t_shift_s"][7] == pytest.approx(0.123) and rec["fit_before_s"][0] == [-3.0, -1.0]
    assert len(rec["coseismic_slip_um"]) == 8 and all(isinstance(f, str) for f in rec["flags"])
    # through an HDF5 file the lists come back as arrays; the inputs are still restored
    dm = DataManager()
    dm.data = {"name": "x", "runs": [{"time": event["time"], "events": [dict(event, slip_analysis=rec)]}]}
    path = tmp_path / "x.h5"
    dm.save_file(path)
    back = DataManager()
    back.load_file(path)
    loaded = back.get_data("runs/[0]/events/[0]/slip_analysis")
    inputs = cs.restore_inputs(loaded, rec["channels"])
    assert inputs["methods"] == methods and inputs["t_shift_s"][7] == pytest.approx(0.123)
    assert inputs["reference_halfwidth_s"] == 0.5
    assert inputs["fit_before_s"][0] == (-3.0, -1.0) and inputs["fit_after_s"][0] == (1.25, 3.75)
    again = run(event, **inputs)
    np.testing.assert_allclose(again.as_dict()["coseismic_slip_um"], rec["coseismic_slip_um"])
    assert cs.restore_inputs(loaded, rec["channels"][:-1]) is None
    assert cs.restore_inputs({"channels": rec["channels"]}, rec["channels"]) is None
    assert cs.restore_inputs(None, rec["channels"]) is None


def test_windows_are_relative_to_the_event_time():
    event, truth = make_event()
    base = run(event, fit_before_s=(-2.0, -0.5), fit_after_s=(0.5, 2.0))
    moved = run(event, methods="manual", t_shift_s=[s.t_shift_s + 0.1 for s in base.sensors],
                fit_before_s=(-2.0, -0.5), fit_after_s=(0.5, 2.0))
    for a, b in zip(base.sensors, moved.sensors):
        assert b.fit_before_s == a.fit_before_s == (-2.0, -0.5)       # the windows stay put
        assert b.before.n == a.before.n and b.before.used == a.before.used
        assert b.before.slope == pytest.approx(a.before.slope)          # the same lines, evaluated at the new t_i
        assert b.before.value == pytest.approx(a.before.value + 0.1 * a.before.slope)
        assert b.after.value == pytest.approx(a.after.value + 0.1 * a.after.slope)
    # a window that reaches past t_i is fitted and flagged
    late = run(event, fit_before_s=(-2.0, 1.0), fit_after_s=(-0.5, 2.0))
    assert all(cs.FLAG_BEFORE_OVERLAP in s.flags and cs.FLAG_AFTER_OVERLAP in s.flags for s in late.sensors)
    assert not any(cs.FLAG_BEFORE_OVERLAP in s.flags for s in base.sensors)


def test_version_1_records_had_windows_relative_to_t_i():
    event, truth = make_event()
    rec = run(event, methods="manual", t_shift_s=0.02).as_dict()
    rec["version"] = 1
    rec["fit_before_s"] = [[-3.75, -1.25]] * 8          # relative to t_i, as version 1 stored them
    inputs = cs.restore_inputs(rec, rec["channels"])
    assert all(w == pytest.approx((-3.73, -1.23)) for w in inputs["fit_before_s"])
    rec["version"] = 2
    assert cs.restore_inputs(rec, rec["channels"])["fit_before_s"][0] == (-3.75, -1.25)


@pytest.mark.parametrize("rate_filter, kwargs", [("lowpass", {"rate_cutoff_hz": 50.0}),
                                                 ("savgol", {"rate_smoothing_s": 0.02}),
                                                 ("boxcar", {"rate_smoothing_s": 0.01})])
def test_rate_filters_cut_the_noise_and_keep_the_picks(rate_filter, kwargs):
    event, truth = make_event(noise_um=0.2, rise_s=0.005)
    raw = run(event, rate_filter="boxcar", rate_smoothing_s=0.0)
    out = run(event, rate_filter=rate_filter, **kwargs)
    assert out.rate_filter == rate_filter
    noise = np.median([s.rate_noise for s in out.sensors])
    assert noise < 0.25 * np.median([s.rate_noise for s in raw.sensors])
    moving = np.array(truth["slips_um"]) > 5.0
    np.testing.assert_allclose(out.t_i[moving], np.asarray(truth["t_i"])[moving], atol=0.003)
    rec = out.as_dict()
    assert rec["rate_filter"] == rate_filter
    assert cs.restore_inputs(rec, rec["channels"])["rate_filter"] == rate_filter


def test_rate_filter_gaps_and_validation():
    t = np.arange(0.0, 2.0, 0.0005)
    d = np.where(t > 1.0, 10.0, 0.0) + 0.01 * np.sin(40 * t)
    d[1000:1010] = np.nan
    for rate_filter in ("lowpass", "savgol"):
        rate = cs.slip_rate(t, d, 0.01, rate_filter, 50.0)
        assert np.all(np.isnan(rate[1000:1010])) and np.all(np.isfinite(np.delete(rate, range(1000, 1010))))
    with pytest.raises(ValueError, match="half the sampling rate"):
        cs.slip_rate(t, d, 0.01, "lowpass", 1500.0)
    with pytest.raises(ValueError, match="unknown rate filter"):
        cs.slip_rate(t, d, 0.01, "median")
    event, _ = make_event()
    rec = run(event).as_dict()
    assert rec["rate_filter"] == cs.DEFAULT_RATE_FILTER == "lowpass" and rec["rate_cutoff_hz"] == 50.0
    del rec["rate_filter"], rec["rate_cutoff_hz"]                   # made before there was a choice
    assert cs.restore_inputs(rec, rec["channels"])["rate_filter"] == "boxcar"


def test_lowpass_type_and_orders():
    t = np.arange(0.0, 2.0, 0.0005)
    rng = np.random.default_rng(1)
    d = 10.0 / (1.0 + np.exp(-(t - 1.0) / 0.005)) + rng.normal(0, 0.2, t.size)
    rates = {kind: cs.slip_rate(t, d, rate_filter="lowpass", cutoff_hz=50.0, lowpass_type=kind)
             for kind in cs.LOWPASS_TYPES}
    for rate in rates.values():
        assert abs(t[np.argmax(rate)] - 1.0) < 0.003
    assert not np.allclose(rates["butterworth"], rates["bessel"])
    gentle = cs.slip_rate(t, d, rate_filter="lowpass", cutoff_hz=50.0, lowpass_order=1)
    assert np.std(gentle[:1000]) > np.std(rates["butterworth"][:1000])     # order 1 rolls off less
    with pytest.raises(ValueError, match="whole number from 1 to"):
        cs.slip_rate(t, d, rate_filter="lowpass", lowpass_order=0)
    with pytest.raises(ValueError, match="unknown low-pass type"):
        cs.slip_rate(t, d, rate_filter="lowpass", lowpass_type="chebyshev")
    cubic = cs.slip_rate(t, d, 0.02, "savgol", savgol_order=3)
    assert abs(t[np.argmax(cubic)] - 1.0) < 0.003
    with pytest.raises(ValueError, match="window must span"):
        cs.slip_rate(t, d, 0.001, "savgol", savgol_order=3)
    event, _ = make_event()
    rec = run(event, rate_lowpass_type="bessel", rate_lowpass_order=2).as_dict()
    inputs = cs.restore_inputs(rec, rec["channels"])
    assert (inputs["rate_lowpass_type"], inputs["rate_lowpass_order"], inputs["rate_savgol_order"]) == ("bessel", 2, 2)


def test_moving_average_ends_are_as_quiet_as_the_rest():
    t = np.arange(0.0, 4.0, 0.0005)
    d = 5.0 * t + np.random.default_rng(2).normal(0.0, 0.2, t.size)       # 5 um/s and noise
    rate = cs.slip_rate(t, d, 0.01, "boxcar")
    interior = rate[100:-100]
    spread = np.max(np.abs(interior - 5.0))
    # the end samples average fewer samples, so allow some more scatter, but not the one-sample difference
    assert np.max(np.abs(rate[:20] - 5.0)) < 2.0 * spread and np.max(np.abs(rate[-20:] - 5.0)) < 2.0 * spread


def test_onset_step_and_shared_event_times():
    event, truth = make_event(noise_um=0.05, rise_s=0.005)
    peak = run(event)
    onset = run(event, methods="onset", onset_fraction=0.2)
    step = run(event, methods="step", step_fraction=0.5)
    moving = np.array(truth["slips_um"]) > 5.0
    t_true = np.asarray(truth["t_i"])[moving]
    # a tanh step: its rate peaks at the midpoint, which the step crossing finds too; the onset comes earlier
    np.testing.assert_allclose(step.t_i[moving], t_true, atol=0.002)
    assert np.all(onset.t_i[moving] < peak.t_i[moving]) and np.all(onset.t_i[moving] > t_true - 0.01)
    shared = run(event, methods="shared")
    assert np.all(shared.t_i == shared.t_ref)
    fixed = run(event, methods="shared", shared_shift_s=0.03)
    assert np.allclose(fixed.t_i, EVENT_TIME + 0.03) and not any(s.flags for s in fixed.sensors if s.coseismic_slip > 5)
    rec = fixed.as_dict()
    assert cs.restore_inputs(rec, rec["channels"])["shared_shift_s"] == pytest.approx(0.03)
    with pytest.raises(ValueError, match="onset threshold"):
        run(event, methods="onset", onset_fraction=1.5)


def test_burst_time_and_levels_mode():
    fs = 2e6
    t = np.arange(0, 0.02, 1 / fs)
    te = 0.010
    slip = np.array([5.0 * (1 + np.tanh((t - te - k * 1e-4) / 2e-4)) for k in range(4)])     # 10 um steps
    pzt = np.array([np.exp(-((t - te - 3e-4) / 5e-5) ** 2) * np.sin(2 * np.pi * 50e3 * t),
                    np.zeros(t.size)])                                                       # burst 0.3 ms after
    assert cs.burst_time(t, pzt, te, 3e-3) == pytest.approx(te + 3e-4, abs=2e-5)
    assert cs.burst_time(t, None, te) is None and cs.burst_time(t, np.zeros((0, t.size)), te) is None
    assert cs.burst_time(t, pzt, te + 0.009, 1e-4) is None                                  # nothing in the search
    r = cs.analyze_slip(t, slip, te, -0.01, 0.01, methods="burst", pzt=pzt, window_mode="levels",
                        burst_halfwidth_s=1e-3, level_width_s=1e-4, rate_filter="boxcar", rate_smoothing_s=1e-4)
    assert r.window_mode == "levels" and r.burst_source == "pzt_envelope"
    assert r.t_burst == pytest.approx(te + 3e-4, abs=2e-5)
    for s in r.sensors:
        assert s.method == "burst" and s.window_mode == "levels" and s.flags == []
        assert s.t_shift_s == pytest.approx(3e-4, abs=2e-5)
        assert s.fit_before_s == pytest.approx((s.t_shift_s - 1.1e-3, s.t_shift_s - 1e-3))
        assert s.fit_after_s == pytest.approx((s.t_shift_s + 1e-3, s.t_shift_s + 1.1e-3))
        assert s.before_level == pytest.approx(0.0, abs=0.05) and s.after_level == pytest.approx(10.0, abs=0.05)
        assert s.coseismic_slip == pytest.approx(10.0, abs=0.05)
    d = r.as_dict()
    for key in ("window_mode", "burst_halfwidth_s", "level_width_s", "burst_search_s", "burst_source",
                "t_burst_shift_s", "before_level_um", "after_level_um"):
        assert key in d
    assert d["t_burst_shift_s"] == pytest.approx(3e-4, abs=2e-5) and d["before_level_um"][0] == pytest.approx(0.0, abs=0.05)
    json.dumps(d)
    back = cs.restore_inputs(d, d["channels"])
    assert back["window_mode"] == "levels" and back["burst_halfwidth_s"] == 1e-3 and back["burst_source"] == "pzt_envelope"
    # without PZT traces the burst is the peak slip rate, flagged; 'peak_rate' asks for that outright
    r2 = cs.analyze_slip(t, slip, te, -0.01, 0.01, methods="burst", window_mode="levels", rate_filter="boxcar",
                         rate_smoothing_s=1e-4)
    assert r2.t_burst == pytest.approx(r2.t_ref) and cs.FLAG_NO_PZT in r2.sensors[0].flags
    r3 = cs.analyze_slip(t, slip, te, -0.01, 0.01, methods="burst", pzt=pzt, burst_source="peak_rate",
                         rate_filter="boxcar", rate_smoothing_s=1e-4)
    assert r3.t_burst == pytest.approx(r3.t_ref) and r3.sensors[0].flags == [] and r3.window_mode == "lines"
    assert math.isnan(r3.sensors[0].before_level) and r3.sensors[0].coseismic_slip == pytest.approx(10.0, abs=0.2)
    # the lines mode keeps the given windows; a record without the new keys reads as lines
    r4 = cs.analyze_slip(t, slip, te, -0.01, 0.01, fit_before_s=(-0.008, -0.002), fit_after_s=(0.002, 0.008),
                         rate_filter="boxcar", rate_smoothing_s=1e-4)
    assert r4.sensors[0].fit_before_s == (-0.008, -0.002) and math.isnan(r4.t_burst)
    old = {k: v for k, v in r4.as_dict().items() if "burst" not in k and "level" not in k and k != "window_mode"}
    assert cs.restore_inputs(old, old["channels"])["window_mode"] == "lines"
    for bad in ({"window_mode": "x"}, {"burst_source": "y"}, {"level_width_s": 0.0}, {"burst_search_s": -1.0},
                {"burst_halfwidth_s": -1.0}):
        with pytest.raises(ValueError):
            cs.analyze_slip(t, slip, te, -0.01, 0.01, rate_filter="boxcar", rate_smoothing_s=1e-4, **bad)
