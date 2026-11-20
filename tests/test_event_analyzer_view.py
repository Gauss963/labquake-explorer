import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from labquake_explorer.analysis.event_metrics import RESULT_VERSION
from labquake_explorer.ui.views import EventAnalyzerView


@pytest.fixture
def view(app):
    v = EventAnalyzerView(app, 0, 1)
    yield v
    v.on_close()


def idx_at(view, t_rel):
    return int(np.argmin(np.abs(view.data_t - view.event["event_time"] - t_rel)))


def place(view):
    """Loading range well before the event, the differenced samples 1 ms either side of it."""
    for point, t_rel in ((0, -4.0), (1, -1.0), (4, -0.001), (5, 0.001)):
        view.move_point(point, idx_at(view, t_rel))


def test_compute_save_reopen_and_switch(app, view):
    truth = app.truth[0]
    assert view.event_idx == 1 and view.title() == "Event Analyzer - Event 1"
    assert view.data_x_combo.get() == "displacement" and view.data_y_combo.get() == "shear_stress"
    assert len(view.picked_idx) == 6 and len(view.markers) == 6
    assert set(view.result_entries) == {"loading_slope", "unloading_slope", "delta_x", "delta_y"}
    assert view in app.child_windows
    # the default loading range lies inside the loading phase: slope = d(tau)/d(displacement)
    assert float(view.loading_slope_text.get()) == pytest.approx(truth.stiffness_fault, rel=1e-6)
    assert float(view.delta_y_text.get()) < 0 < float(view.delta_x_text.get())
    # degenerate picks show n/a
    view.move_point(1, view.picked_idx[0])
    assert view.loading_slope_text.get() == "n/a"
    assert np.isnan(view.result["loading_slope"])

    # moving markers recomputes
    place(view)
    r = view.result
    assert r["delta_y"] == pytest.approx(-truth.stress_drop, abs=1e-3)
    assert r["delta_x"] == pytest.approx(truth.slip, abs=3e-3)
    assert view.delta_x_text.get() == f"{r['delta_x']:.6g}"
    assert r["delta_indices"] == view.picked_idx[4:]

    # save writes a versioned record without the old fields
    view.save_event()
    saved = app.data_manager.get_data("runs/[0]/events/[1]/event_analysis")
    assert saved is view.event["event_analysis"]
    assert saved["version"] == RESULT_VERSION == 3
    assert saved["x_field"] == "displacement" and saved["y_field"] == "shear_stress"
    assert saved["loading_indices"] == view.picked_idx[:2] and saved["delta_indices"] == view.picked_idx[4:]
    assert not any(k in saved for k in ("fit_method", "loading_window", "stress_drop", "displacement", "post_indices"))
    # reloading the event restores the picks
    view.picked_idx[4] = 3
    view.set_event(1)
    assert view.picked_idx == saved["loading_indices"] + saved["unloading_indices"] + saved["delta_indices"]

    view.set_event(2)
    assert view.event_idx == 2
    assert view.title() == "Event Analyzer - Event 2"
    assert view.event is app.data_manager.get_data("runs/[0]/events/[2]")
    assert view.event_combobox.get() == "2"


def test_apply_to_all_events_uses_relative_ranges(app, view, monkeypatch):
    from tkinter import messagebox
    truth = app.truth[0]
    place(view)
    monkeypatch.setattr(messagebox, "askokcancel", lambda *a, **k: False)
    assert view.apply_to_all_events() == 0
    assert "event_analysis" not in app.data_manager.get_data("runs/[0]/events/[0]")

    n = view.apply_to_all_events(confirm=False)
    events = app.data_manager.get_data("runs/[0]/events")
    assert n == len(events)
    for event in events:
        r = event["event_analysis"]
        t_rel = event["time"] - event["event_time"]
        assert r["version"] == RESULT_VERSION
        assert t_rel[r["loading_indices"][0]] == pytest.approx(-4.0, abs=2e-3)
        assert t_rel[r["loading_indices"][1]] == pytest.approx(-1.0, abs=2e-3)
        assert r["delta_x"] == pytest.approx(truth.slip, abs=3e-3)
        assert -r["delta_y"] == pytest.approx(truth.stress_drop, abs=1e-3)
    # the view is still on the same event, with the saved picks
    assert view.event_idx == 1
    r1 = events[1]["event_analysis"]
    assert view.picked_idx == r1["loading_indices"] + r1["unloading_indices"] + r1["delta_indices"]


def test_reopen_restores_saved_x_and_y_fields(app):
    """A record analysed on X = LP_displacement is shown (and re-saved) against
    LP_displacement, not the constructor default."""
    truth = app.truth[0]
    v = EventAnalyzerView(app, 0, 1, item_x="LP_displacement")
    place(v)
    v.save_event()
    saved = v.event["event_analysis"]
    assert saved["x_field"] == "LP_displacement" and saved["y_field"] == "shear_stress"
    assert abs(saved["delta_x"]) < 1.0                            # LP advance across 2 ms, not the slip
    assert saved["loading_slope"] == pytest.approx(truth.stiffness_lp, rel=1e-6)
    v.on_close()

    v2 = EventAnalyzerView(app, 0, 1)                             # default constructor: 'displacement'
    assert v2.item_x == "LP_displacement" and v2.data_x_combo.get() == "LP_displacement"
    assert v2.item_y == "shear_stress"
    assert float(v2.delta_x_text.get()) == pytest.approx(saved["delta_x"], rel=1e-5)
    assert float(v2.loading_slope_text.get()) == pytest.approx(saved["loading_slope"], rel=1e-6)
    assert v2.result["x_field"] == "LP_displacement"
    v2.save_event()
    assert v2.event["event_analysis"]["x_field"] == "LP_displacement"
    # switching to an event without a record keeps the current selection
    v2.set_event(2)
    assert v2.data_x_combo.get() == "LP_displacement"
    v2.on_close()
    # a remembered field that no longer exists falls back without failing
    saved["x_field"] = "gone"
    v3 = EventAnalyzerView(app, 0, 1)
    assert v3.data_x_combo.get() in v3.data_x_combo["values"]
    v3.on_close()


def test_dragging_onto_nan_gap_snaps_to_finite_sample(app, view):
    """A marker dragged over a NaN gap must land on a finite sample: argmin over
    distances with NaN returns the NaN index, which strands the marker."""
    y = view.event["shear_stress"]
    y[100:200] = np.nan
    view.plot_data()
    view.plot_picked_points()
    marker = view.markers[0]
    view.picker.current_artist = marker
    view.picker.currently_dragging = True
    view.picker.offset = [0.0, 0.0]
    ax = view.ax
    target_x = float(view.data_x[150])
    target_y = float(np.nanmean(y))
    px, py = ax.transData.transform((target_x, target_y))
    view.picker.on_motion(MouseEvent("motion_notify_event", view.canvas, px, py))
    idx = view.picked_idx[0]
    assert not 100 <= idx < 200
    assert view.loading_line.get_data()[0][0] == view.data_x[idx]          # the line follows the drag
    assert np.isfinite(view.data_y[idx]) and np.isfinite(marker.center[1])
    # an all-NaN trace leaves the pick alone instead of stranding it
    y[:] = np.nan
    view.plot_data()
    before = list(view.picked_idx)
    view.picker.on_motion(MouseEvent("motion_notify_event", view.canvas, px, py))
    assert view.picked_idx == before


def test_legacy_results_load_their_picks(app):
    event = app.data_manager.get_data("runs/[0]/events/[0]")
    event["event_analysis"] = {
        "version": 2, "loading_indices": [10, 20], "unloading_indices": [30, 40],
        "rupture_start_index": 25, "rupture_end_index": 45, "post_indices": [50, 60],
        "loading_stiffness": 1.0, "stress_drop": 0.1, "displacement": 1.0,
    }
    v = EventAnalyzerView(app, 0, 0)
    assert v.picked_idx == [10, 20, 30, 40, 25, 45]
    v.on_close()
    assert v not in app.child_windows
