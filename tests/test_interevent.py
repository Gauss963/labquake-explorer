import numpy as np
import pytest

from labquake_explorer.analysis.interevent import (
    creep_per_cycle, event_times_from_run, interevent_metrics, sample_after,
)
from tests.synthetic import make_stick_slip_run, extract_events


def test_sample_after():
    t = np.arange(0, 10, 0.1)
    s = t * 2.0
    # width 0 -> the sample at t+delay (within half a sample)
    assert sample_after(t, s, 1.0, delay=0.06, width=0.0) == pytest.approx(2.2)
    assert sample_after(t, s, 1.0, delay=0.04, width=0.0) == pytest.approx(2.0)
    # width covers a fixed number of samples: round(width/dt)+1 (documented formula);
    # on a unit-step ramp the mean of 0..expect-1 is (expect-1)/2 only for exactly `expect` samples
    ramp = np.arange(t.size, dtype=float)
    for width, expect in [(0.0, 1), (0.1, 2), (0.2, 3), (0.5, 6)]:
        assert sample_after(t, ramp, 0.0, delay=0.0, width=width) == pytest.approx((expect - 1) / 2)
    assert sample_after(t, s, 1.0, delay=0.0, width=0.2) == pytest.approx(2 * np.mean([1.0, 1.1, 1.2]))
    assert sample_after(t, s, 1.0, delay=0.0, width=0.5) == pytest.approx(2 * np.mean(t[(t >= 1.0) & (t <= 1.5)]))
    # outside the run: NaN, not clamped
    assert np.isnan(sample_after(t, s, 9.95, delay=0.05, width=0.1))
    assert np.isnan(sample_after(t, s, -1.0)) and np.isnan(sample_after(t, s, 20.0))
    assert np.isfinite(sample_after(t, s, 9.8, delay=0.05, width=0.05))
    pytest.raises(ValueError, sample_after, [0, 1], [0], 0.5)
    pytest.raises(ValueError, sample_after, [0, 1], [0, 1], 0.5, width=-1)


def test_interevent_metrics_recover_synthetic_cycle():
    ssr = make_stick_slip_run(n_events=5, recurrence=6.0, fs=1000.0)
    run = ssr.as_dict()
    res = interevent_metrics(run["time"], ssr.event_times, {"lp": run["LP_displacement"], "slip": run["displacement"]},
                             delay=0.05, width=0.05)
    assert res["event_times"] == pytest.approx(list(ssr.event_times))
    assert np.isnan(res["recurrence"][0]) and res["recurrence"][1:] == pytest.approx([6.0] * 4, abs=1e-9)
    # load point advances lp_velocity * recurrence per cycle
    assert res["lp_per_cycle"][1:] == pytest.approx([ssr.lp_velocity * 6.0] * 4, rel=1e-9)
    # fault slips one coseismic slip plus creep per cycle
    expected_slip = ssr.slip + ssr.creep_fraction * ssr.lp_velocity * 6.0
    assert res["slip_per_cycle"][1:] == pytest.approx([expected_slip] * 4, rel=1e-9)
    creep = creep_per_cycle(res["slip_per_cycle"], [ssr.slip] * 5)
    assert np.isnan(creep[0]) and creep[1:] == pytest.approx([ssr.creep_fraction * ssr.lp_velocity * 6.0] * 4, rel=1e-9)
    # event times: from the run, then from its events; one event without a time gives None
    assert event_times_from_run(run) == pytest.approx(ssr.event_times)
    run["events"] = extract_events(run, 2.0)
    assert event_times_from_run(run) == pytest.approx(ssr.event_times)
    run["events"][1] = {}
    assert event_times_from_run(run) is None
    assert event_times_from_run({"time": np.arange(5.0)}) is None


def test_interevent_metrics_edges():
    t = np.arange(0, 10, 0.01)
    res = interevent_metrics(t, [2.0, 9.98], {"s": np.ones_like(t)}, delay=0.05, width=0.05)
    assert np.isfinite(res["s_after"][0]) and np.isnan(res["s_after"][1])     # last event near the run end
    assert np.isnan(res["s_per_cycle"][1])
    pytest.raises(ValueError, interevent_metrics, np.arange(10.0), [3.0, 2.0], {})   # times must increase
    res = interevent_metrics(np.arange(10.0), [], {"a": np.arange(10.0)})
    assert res["event_times"] == [] and res["a_after"] == []
