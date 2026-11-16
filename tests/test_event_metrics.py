import numpy as np
import pytest

from labquake_explorer.analysis.event_metrics import (
    EventPicks, analyze_event, picks_from_result, picks_from_windows, windows_from_picks, RESULT_VERSION,
)
from tests.synthetic import make_stick_slip_run, extract_events

RESULT_KEYS = {"version", "x_field", "y_field", "loading_indices", "unloading_indices", "delta_indices",
               "loading_slope", "unloading_slope", "delta_x", "delta_y"}


def test_analyze_event_and_windows_on_a_synthetic_event():
    truth = make_stick_slip_run(recurrence=12.0, post_slope=0.02, fs=1000.0)
    event = extract_events(truth.as_dict(), 5.0)[1]
    x, y = event["displacement"], event["shear_stress"]
    t_rel = event["time"] - event["event_time"]
    at = lambda t: int(np.argmin(np.abs(t_rel - t)))
    picks = EventPicks(loading=(at(-4.0), at(-1.0)), unloading=(at(-0.002), at(0.002)), delta=(at(-0.001), at(0.001)))
    res = analyze_event(x, y, picks, x_field="displacement", y_field="shear_stress")
    assert set(res) == RESULT_KEYS and res["version"] == RESULT_VERSION == 3
    assert res["x_field"] == "displacement" and res["y_field"] == "shear_stress"
    assert res["loading_slope"] == pytest.approx(truth.stiffness_fault, rel=1e-6)
    assert np.isfinite(res["unloading_slope"]) and res["unloading_slope"] < 0     # stress falls while slip grows
    # the differences across the picks are the imposed drop and slip; loading over 2 ms is negligible
    assert -res["delta_y"] == pytest.approx(truth.stress_drop, abs=2 * truth.loading_rate * 0.002)
    assert res["delta_x"] == pytest.approx(truth.slip, abs=1e-6 + 2 * truth.creep_fraction * truth.lp_velocity * 0.002)
    assert res["loading_indices"] == list(picks.loading) and res["delta_indices"] == list(picks.delta)
    assert all(isinstance(v, (int, float, str, list)) for v in res.values())
    # swapped delta picks flip both differences
    res = analyze_event(x, y, EventPicks(loading=(100, 200), unloading=(300, 400), delta=(at(0.001), at(-0.001))))
    assert res["delta_y"] > 0 and res["delta_x"] < 0
    # degenerate picks never raise; empty or mismatched data does
    res = analyze_event(x, y, EventPicks((5, 5), (5, 5), (5, 5)))
    assert np.isnan(res["loading_slope"]) and np.isnan(res["unloading_slope"])
    assert res["delta_x"] == 0.0 and res["delta_y"] == 0.0
    pytest.raises(ValueError, analyze_event, [], [], EventPicks((0, 0), (0, 0), (0, 0)))
    pytest.raises(ValueError, analyze_event, [1.0, 2.0], [1.0], EventPicks((0, 0), (0, 0), (0, 0)))
    # picks <-> windows in seconds round trip
    windows = windows_from_picks(t_rel, picks)
    assert windows["loading"] == pytest.approx((-4.0, -1.0), abs=1e-3)
    assert windows["delta"][0] < 0 < windows["delta"][1]
    assert picks_from_windows(t_rel, windows) == picks
    assert picks_from_windows(t_rel[::2], windows).loading[0] == pytest.approx(picks.loading[0] / 2, abs=1)
    pytest.raises(ValueError, picks_from_windows, [], windows)
    pytest.raises(ValueError, windows_from_picks, [], picks)


def test_picks_defaults_clipping_and_saved_results():
    assert EventPicks.defaults(1000).to_list() == [250, 350, 500, 600, 400, 700]
    assert EventPicks.from_list(range(8)).to_list() == [0, 1, 2, 3, 4, 5]     # a legacy post range is dropped
    pytest.raises(ValueError, EventPicks.from_list, [1, 2, 3])
    clipped = EventPicks.from_list([0, 5000, 2, 3, 4, 5]).clipped(100)
    assert clipped.loading == (0, 99) and clipped.pairs()["delta"] == (4, 5)
    # picks from saved results, versions 1 to 3
    v1 = {"loading_indices": [1, 2], "unloading_indices": [3, 4],
          "rupture_start_index": 5, "rupture_end_index": 6, "stress_drop": 0.1}
    p = picks_from_result(v1, 1000)
    assert p.loading == (1, 2) and p.delta == (5, 6)
    assert picks_from_result(dict(v1, post_indices=[7, 8], version=2), 1000).to_list() == [1, 2, 3, 4, 5, 6]
    v3 = {"version": 3, "loading_indices": [1, 2], "unloading_indices": [3, 4], "delta_indices": [5, 6]}
    assert picks_from_result(v3, 1000).delta == (5, 6)
    assert picks_from_result(v3, 5) is None       # out of range
    assert picks_from_result({"x": 1}, 10) is None and picks_from_result(None, 10) is None
