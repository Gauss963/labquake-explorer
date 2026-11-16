import numpy as np
import pytest

from labquake_explorer.ui.actions import actions_for
from labquake_explorer.ui.views.interevent_view import InterEventView, aligned_arrays


def test_open_from_action_compute_and_save(app):
    truth = app.truth[0]
    ctx = app.context_at(app.find_item("runs/[0]"))
    action = [a for a in actions_for(ctx.kind) if a.label == "Inter-event Metrics"][0]
    app.run_action(action, ctx)
    view = app.child_windows[-1]
    assert isinstance(view, InterEventView) and view.run_idx == 0
    assert view.title() == "Inter-event Metrics - run00"
    assert view.lp_combo.get() == "LP_displacement"
    assert view.slip_combo.get() == "displacement"
    r = view.result
    n = len(truth.event_times)
    assert len(r["event_times"]) == n
    assert np.isnan(r["recurrence"][0])
    assert r["recurrence"][1:] == pytest.approx([12.0] * (n - 1), abs=1e-9)
    assert r["lp_per_cycle"][1:] == pytest.approx([truth.lp_velocity * 12.0] * (n - 1), rel=1e-9)
    expected_slip = truth.slip + truth.creep_fraction * truth.lp_velocity * 12.0
    assert r["slip_per_cycle"][1:] == pytest.approx([expected_slip] * (n - 1), rel=1e-9)
    assert all(np.isnan(r["coseismic_slip"]))   # no event_analysis saved yet
    assert len(view.table.get_children()) == n

    view.delay_var.set("abc")
    view.compute()
    assert "number" in view.status_var.get()
    view.delay_var.set("0.1")
    view.width_var.set("0")
    view.compute()
    assert view.result["delay_s"] == 0.1 and view.result["width_s"] == 0.0
    view.save()
    saved = app.data_manager.get_data("runs/[0]/interevent")
    assert saved["version"] == 1
    assert saved["lp_field"] == "LP_displacement" and saved["delay_s"] == 0.1
    assert "saved" in view.status_var.get()
    view.on_close()
    assert view not in app.child_windows
    # reopening restores the parameters
    v2 = InterEventView(app, 0)
    assert v2.delay_var.get() == "0.1" and v2.width_var.get() == "0"
    v2.on_close()


def test_creep_from_saved_event_analysis_skips_legacy_v1(app):
    truth = app.truth[0]
    events = app.data_manager.get_data("runs/[0]/events")
    for event in events:
        event["event_analysis"] = {"version": 3, "x_field": "displacement", "delta_x": truth.slip}
    v = InterEventView(app, 0)
    creep = v.result["creep"]
    assert np.isnan(creep[0])
    assert creep[1:] == pytest.approx([truth.creep_fraction * truth.lp_velocity * 12.0] * (len(creep) - 1), rel=1e-9)
    assert v.result["coseismic_field"] == "displacement"
    assert v.result["coseismic_skipped"] == {}
    assert f"coseismic slip from event_analysis (displacement) for {len(events)} events" in v.status_var.get()
    assert "skipped" not in v.status_var.get()

    # v1 records stored abs(x5 - x4) without x_field: unusable for creep, counted
    for event in events[:2]:
        event["event_analysis"] = {"displacement": truth.slip, "stress_drop": 0.4}   # main's v1 layout
    for event in events[2:]:
        event["event_analysis"] = {"version": 2, "x_field": "displacement", "displacement": truth.slip}
    v.compute()
    r = v.result
    assert np.isnan(r["coseismic_slip"][0]) and np.isnan(r["coseismic_slip"][1])
    assert r["coseismic_slip"][2] == pytest.approx(truth.slip)
    assert r["coseismic_skipped"] == {"legacy v1 record without x_field (unsigned displacement)": 2}
    assert "skipped for 2 event(s): legacy v1" in v.status_var.get()
    assert f"for {len(events) - 2} events" in v.status_var.get()
    v.on_close()


def test_creep_ignores_event_analysis_on_another_x_field(app):
    """``event_analysis['delta_x']`` is X(end) - X(start) of WHATEVER X the
    analyser ran on: a record analysed on LP_displacement (machine stiffness)
    or time is not a fault slip and must not be subtracted from the
    per-cycle slip of ``run['displacement']``."""
    from labquake_explorer.ui.views import EventAnalyzerView
    truth = app.truth[0]
    events = app.data_manager.get_data("runs/[0]/events")
    # the real analyser on X = LP_displacement, rupture picks 2 samples either side of the event
    ea = EventAnalyzerView(app, 0, 1, item_x="LP_displacement")
    i_ev = int(np.argmin(np.abs(ea.data_t - ea.event["event_time"])))
    for point, t_rel in ((0, -4.0), (1, -1.0)):
        ea.move_point(point, int(np.argmin(np.abs(ea.data_t - ea.event["event_time"] - t_rel))))
    ea.move_point(4, i_ev - 2)
    ea.move_point(5, i_ev + 2)
    assert ea.apply_to_all_events(confirm=False) == len(events)
    ea.on_close()
    saved = events[1]["event_analysis"]
    assert saved["x_field"] == "LP_displacement"
    assert abs(saved["delta_x"]) < 1.0                 # LP advance over 4 samples, not the 30 um slip

    v = InterEventView(app, 0)
    assert v.slip_combo.get() == "displacement"
    r = v.result
    assert all(np.isnan(r["coseismic_slip"])) and all(np.isnan(r["creep"]))
    assert r["coseismic_field"] == "displacement"
    assert r["coseismic_skipped"] == {"analysed on LP_displacement": len(events)}
    status = v.status_var.get()
    assert "for 0 events" in status
    assert f"coseismic slip skipped for {len(events)} event(s): analysed on LP_displacement" in status
    # the same records ARE the coseismic advance of LP_displacement
    v.slip_combo.set("LP_displacement")
    v.compute()
    r = v.result
    assert r["coseismic_field"] == "LP_displacement" and r["coseismic_skipped"] == {}
    assert np.all(np.isfinite(r["coseismic_slip"]))
    assert r["coseismic_slip"][1] == pytest.approx(saved["delta_x"])
    assert r["creep"][1:] == pytest.approx(
        [truth.lp_velocity * 12.0 - saved["delta_x"]] * (len(events) - 1), rel=1e-6)
    v.on_close()


def test_opens_on_run_with_empty_events_after_hdf5_reload(app, app_from_file, tmp_path):
    """An empty ``events`` list comes back from HDF5 as an empty ndarray, whose
    truth value raises; the view must still open (event times fall back to
    time[event_indices])."""
    app.data_manager.set_data("runs/[0]/events", [])
    path = tmp_path / "empty_events.h5"
    app.data_manager.save_file(path)
    app2 = app_from_file(path)
    run = app2.data_manager.get_data("runs/[0]")
    assert len(run["events"]) == 0                       # an empty ndarray or an empty list, by loader version
    run["events"] = np.zeros(0)                          # the form that raises on bool(): must still open
    v = InterEventView(app2, 0)
    assert v in app2.child_windows
    n = len(app.truth[0].event_times)
    assert len(v.result["event_times"]) == n
    assert all(np.isnan(v.result["coseismic_slip"]))
    assert v.result["recurrence"][1:] == pytest.approx([12.0] * (n - 1), abs=1e-9)
    v.on_close()


def test_aligned_arrays_filters_by_length_and_kind():
    run = {"time": np.arange(5.0), "a": np.arange(5.0), "b": np.arange(4.0),
           "c": np.array([True] * 5), "events": [{}], "name": "x", "d": np.zeros((5, 2))}
    assert aligned_arrays(run) == ["a"]
