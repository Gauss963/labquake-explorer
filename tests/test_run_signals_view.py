import numpy as np
import pytest
from matplotlib.lines import Line2D

from labquake_explorer.data.picks import picked_indices
from labquake_explorer.ui.actions import actions_for
from labquake_explorer.ui.context import RUN
from labquake_explorer.ui.views.run_signals_view import (
    RunSignalsView, event_positions, event_times, nearest_samples, normalize01,
    signal_candidates,
)


@pytest.fixture
def view(app):
    v = RunSignalsView(app, 0)
    yield v
    if v in app.child_windows:
        v.on_close()


def n_events(app):
    return len(app.data_manager.get_data("runs/[0]/events"))


def marker_xs(view):
    return sorted(l.get_xdata()[0] for l in view.event_lines)


# ---------------------------------------------------------------- pure helpers
def test_signal_candidates_and_normalize01():
    n = 10
    run = {
        "name": "run00",
        "time": np.arange(n, dtype=float),
        "shear_stress": np.ones(n),
        "counts": np.arange(n),                      # integer dtype ok
        "with_nan": np.r_[np.nan, np.ones(n - 1)],   # NaN allowed
        "short": np.ones(n - 1),                     # wrong length
        "matrix": np.ones((n, 2)),                   # not 1-D
        "flags": np.ones(n, dtype=bool),             # bool excluded
        "event_indices": [2, 5],
        "events": [{"event_time": 2.0}, {"event_time": 5.0}],
        "nested": {"time": np.arange(n)},
    }
    assert signal_candidates(run) == ["time", "shear_stress", "counts", "with_nan"]
    assert signal_candidates({"name": "x"}) == []
    assert signal_candidates({"time": np.array([])}) == []

    out = normalize01(np.array([2.0, np.nan, 4.0, 6.0]))
    assert np.isnan(out[1])
    assert out[[0, 2, 3]] == pytest.approx([0.0, 0.5, 1.0])
    assert np.all(normalize01(np.full(5, 3.0)) == 0.0)
    assert np.all(np.isnan(normalize01(np.full(3, np.nan))))


def test_event_positions_and_nearest_samples():
    t = np.linspace(0.0, 9.0, 10)
    lp = 10.0 * t
    # event times are preferred over indices and mapped onto x
    run = {"time": t, "event_indices": [3, 7],
           "events": [{"event_time": 3.02}, {"event_time": 6.98}]}
    assert event_positions(run, t) == pytest.approx([3.0, 7.0])
    assert event_positions(run, lp) == pytest.approx([30.0, 70.0])
    assert event_positions({"time": t, "event_indices": [1, 8]}, lp) == pytest.approx([10.0, 80.0])
    assert event_positions({"time": t}, t).size == 0

    # a clock reset (time not monotonic) still maps to the right sample, and
    # nearest_samples indexes the ORIGINAL (unsorted) array
    time = np.concatenate([t[5:], t[:5]])
    run = {"time": time, "events": [{"event_time": 7.0}, {"event_time": 2.1}]}
    assert event_positions(run, 10.0 * time) == pytest.approx([70.0, 20.0])
    assert event_positions(run, time) == pytest.approx([7.0, 2.0])
    assert nearest_samples(time, [7.0, 2.1]).tolist() == [2, 7]

    # events without a usable time fall back to their index one by one
    run = {"time": t, "event_indices": [2, 5, 8],
           "events": [{"event_time": 2.0}, {"foo": 1}, {"event_time": "bad"}]}
    assert event_times(run) == pytest.approx([2.0, 5.0, 8.0])
    assert event_positions(run, t) == pytest.approx([2.0, 5.0, 8.0])
    # an event with neither a time nor an index is reported as NaN and dropped
    run = {"time": t, "event_indices": [2], "events": [{"event_time": 2.0}, {"foo": 1}]}
    times = event_times(run)
    assert times.size == 2 and times[0] == 2.0 and np.isnan(times[1])
    assert event_positions(run, t) == pytest.approx([2.0])

    # x undefined at the event sample: that event is dropped
    x = lp.copy()
    x[3] = np.nan
    run = {"time": t, "events": [{"event_time": 3.0}, {"event_time": 7.0}]}
    assert event_positions(run, x) == pytest.approx([70.0])
    # NaN in the time axis is never chosen as the nearest sample
    time_nan = t.copy()
    time_nan[5] = np.nan
    assert nearest_samples(time_nan, [5.0]).tolist() in ([4], [6])
    assert nearest_samples(np.full(4, np.nan), [1.0]).tolist() == [-1]
    assert event_positions({"time": time_nan, "events": [{"event_time": 5.0}]}, time_nan).size == 1


# ----------------------------------------------------------------------- view
def test_open_from_action_defaults_and_close(app):
    ctx = app.context_at(app.find_item("runs/[0]"))
    assert ctx.kind == RUN
    action = [a for a in actions_for(ctx.kind) if a.label == "Plot Run Signals"][0]
    before = list(app.child_windows)
    app.run_action(action, ctx)
    opened = [w for w in app.child_windows if w not in before]
    assert len(opened) == 1 and isinstance(opened[0], RunSignalsView)
    view = opened[0]
    assert view.run_idx == 0
    assert RunSignalsView.result_key is None
    assert view.title() == "Run Signals - run00"

    listed = list(view.signal_listbox.get(0, "end"))
    assert listed == view.candidates
    for expected in ("shear_stress", "displacement", "LP_displacement", "LP_velocity",
                     "normal_stress", "friction"):
        assert expected in listed
    for excluded in ("events", "event_extraction", "name", "time"):
        assert excluded not in listed
    x_values = list(view.x_combo["values"])
    assert x_values[:2] == ["time", "index"] and "LP_displacement" in x_values
    assert view.x_combo.get() == "time"
    assert view.normalize_var.get() is False
    assert view.mark_events_var.get() is True
    # default selection plots shear_stress against time with event markers
    assert view.selected_signals() == ["shear_stress"]
    assert [l.get_label() for l in view.signal_lines] == ["shear_stress"]
    assert len(view.event_lines) == n_events(app)
    assert view.ax.get_xlabel() == "time" and view.ax.get_ylabel() == "shear_stress"
    assert "not markable" not in view.status_var.get()
    assert "Skipped" not in view.status_var.get()

    view.on_close()
    assert view not in app.child_windows


def test_plot_options(app, view):
    run = app.data_manager.get_data("runs/[0]")
    # two signals with dotted event markers and an upper-left legend
    view.select_signals(["shear_stress", "displacement"])
    view.plot_button.invoke()
    assert all(isinstance(l, Line2D) for l in view.signal_lines)
    assert [l.get_label() for l in view.signal_lines] == ["shear_stress", "displacement"]
    assert len(view.ax.lines) == 2 + n_events(app)
    assert marker_xs(view) == pytest.approx([e["event_time"] for e in run["events"]])
    assert all(l.get_linestyle() == ":" for l in view.event_lines)
    np.testing.assert_array_equal(view.signal_lines[0].get_ydata(), run["shear_stress"])
    np.testing.assert_array_equal(view.signal_lines[1].get_xdata(), run["time"])
    legend = view.ax.get_legend()
    assert legend._get_loc() == 2  # matplotlib code for 'upper left'
    assert [t.get_text() for t in legend.get_texts()] == ["shear_stress", "displacement"]

    # normalize puts every line in [0, 1]
    view.select_signals(["shear_stress", "displacement", "LP_displacement"])
    view.normalize_var.set(True)
    view.plot_button.invoke()
    assert len(view.signal_lines) == 3
    for line in view.signal_lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        y = y[np.isfinite(y)]
        assert y.min() == pytest.approx(0.0) and y.max() == pytest.approx(1.0)
    assert "normalized" in view.ax.get_ylabel()
    view.normalize_var.set(False)

    # LP_displacement as x: markers move to the event samples on that axis
    view.select_signals(["shear_stress"])
    view.x_combo.set("LP_displacement")
    view.plot_button.invoke()
    np.testing.assert_array_equal(view.signal_lines[0].get_xdata(), run["LP_displacement"])
    assert view.ax.get_xlabel() == "LP_displacement"
    expected = run["LP_displacement"][np.asarray(picked_indices(run))]
    assert marker_xs(view) == pytest.approx(sorted(expected))
    view.x_combo.set("index")
    view.plot_button.invoke()
    np.testing.assert_array_equal(view.signal_lines[0].get_xdata(), np.arange(len(run["time"])))
    assert view.ax.get_xlabel() == "sample index"

    # mark events off
    view.mark_events_var.set(False)
    view.plot_button.invoke()
    assert view.event_lines == [] and len(view.ax.lines) == 1

    # nothing selected
    view.select_signals([])
    view.plot_button.invoke()
    assert view.signal_lines == [] and view.ax.get_legend() is None
    assert "Select" in view.status_var.get()

    # refresh keeps the selection and the x choice
    view.select_signals(["displacement", "friction"])
    view.x_combo.set("LP_displacement")
    view.refresh_candidates()
    assert view.selected_signals() == ["displacement", "friction"]
    assert view.x_combo.get() == "LP_displacement"

    # arrays with NaN gaps are listed and plotted with the gap
    gappy = np.array(run["shear_stress"], dtype=float)
    gappy[100:200] = np.nan
    run["gappy"] = gappy
    view.refresh_candidates()
    assert "gappy" in view.candidates and "gappy" in list(view.x_combo["values"])
    view.select_signals(["gappy"])
    view.normalize_var.set(True)
    view.plot_button.invoke()
    y = np.asarray(view.signal_lines[0].get_ydata(), dtype=float)
    assert np.isnan(y[150])
    finite = y[np.isfinite(y)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0
    del run["gappy"]


def test_zoom_kept_across_replots_until_home_or_reset(app, view):
    view.select_signals(["shear_stress"])
    view.plot_button.invoke()
    auto_x, auto_y = view.ax.get_xlim(), view.ax.get_ylim()
    view.ax.set_xlim(10.0, 20.0)
    view.ax.set_ylim(4.0, 6.0)
    # re-plot with the same signals: both zooms survive
    view.plot_button.invoke()
    assert view.ax.get_xlim() == pytest.approx((10.0, 20.0))
    assert view.ax.get_ylim() == pytest.approx((4.0, 6.0))
    # adding a signal changes what y shows: y re-autoscales so the new line is
    # visible, x zoom kept
    view.select_signals(["shear_stress", "normal_stress"])
    view.plot_button.invoke()
    assert view.ax.get_xlim() == pytest.approx((10.0, 20.0))
    lo, hi = view.ax.get_ylim()
    for line in view.signal_lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        assert lo <= np.nanmin(y) and np.nanmax(y) <= hi
    # y zoom with the two signals survives a re-plot of the same two signals
    view.ax.set_ylim(3.0, 12.0)
    view.plot_button.invoke()
    assert view.ax.get_ylim() == pytest.approx((3.0, 12.0))
    # removing a signal also resets y
    view.select_signals(["shear_stress"])
    view.plot_button.invoke()
    assert view.ax.get_ylim() != pytest.approx((3.0, 12.0))
    assert view.ax.get_xlim() == pytest.approx((10.0, 20.0))
    # toggling normalize changes the y quantity: y resets, x zoom kept
    view.ax.set_ylim(4.0, 6.0)
    view.normalize_var.set(True)
    view.plot_button.invoke()
    assert view.ax.get_xlim() == pytest.approx((10.0, 20.0))
    assert view.ax.get_ylim() != pytest.approx((4.0, 6.0))
    # changing the x quantity resets x
    view.x_combo.set("LP_displacement")
    view.plot_button.invoke()
    assert view.ax.get_xlim() != pytest.approx((10.0, 20.0))
    # an un-zoomed plot re-autoscales
    view.x_combo.set("time")
    view.normalize_var.set(False)
    view.select_signals(["shear_stress"])
    view.plot_button.invoke()
    assert view.ax.get_xlim() == pytest.approx(auto_x)
    assert view.ax.get_ylim() == pytest.approx(auto_y)

    # toolbar Home after a zoomed re-plot returns to the full view: the nav
    # stack is not empty after a re-plot, so the preserved zoom is not
    # recorded as Home
    view.ax.set_xlim(10.0, 20.0)
    view.plot_button.invoke()
    assert view.toolbar._nav_stack() is not None
    view.toolbar.home()
    assert view.ax.get_xlim() == pytest.approx(auto_x)
    assert view.ax.get_ylim() == pytest.approx(auto_y)
    # a subsequent Plot stays at the full view (the zoom is forgotten)
    view.plot_button.invoke()
    assert view.ax.get_xlim() == pytest.approx(auto_x)
    # a toolbar zoom on top of a preserved zoom, then Home
    view.ax.set_xlim(10.0, 20.0)
    view.plot_button.invoke()
    view.toolbar.push_current()
    view.ax.set_xlim(12.0, 14.0)
    view.toolbar.push_current()
    view.toolbar.home()
    assert view.ax.get_xlim() == pytest.approx(auto_x)
    assert view.ax.get_ylim() == pytest.approx(auto_y)

    # the Reset view button forgets the zoom too
    view.ax.set_xlim(10.0, 20.0)
    view.ax.set_ylim(4.0, 6.0)
    view.plot_button.invoke()
    assert view.ax.get_xlim() == pytest.approx((10.0, 20.0))
    view.reset_button.invoke()
    assert view.ax.get_xlim() == pytest.approx(auto_x)
    assert view.ax.get_ylim() == pytest.approx(auto_y)
    assert [l.get_label() for l in view.signal_lines] == ["shear_stress"]


def test_stale_signals_and_unmarkable_events_reported(app, view):
    run = app.data_manager.get_data("runs/[0]")
    view.select_signals(["shear_stress", "friction", "displacement"])
    view.plot_button.invoke()
    assert len(view.signal_lines) == 3
    run["shear_stress"] = np.asarray(run["shear_stress"])[:-1]   # no longer aligned
    run.pop("friction")                                     # gone
    view.plot_button.invoke()                               # must not raise
    assert [l.get_label() for l in view.signal_lines] == ["displacement"]
    assert len(view.event_lines) == n_events(app)
    status = view.status_var.get()
    assert "Skipped" in status and "shear_stress" in status and "friction" in status
    assert "1 signal(s)" in status
    # the listbox was refreshed: stale entries gone, the good one stays selected
    assert "friction" not in view.candidates and "shear_stress" not in view.candidates
    assert view.selected_signals() == ["displacement"]
    # a stale X-axis choice falls back to the sample index
    view.x_combo.set("LP_displacement")
    lp = run["LP_displacement"]
    run["LP_displacement"] = "not an array"
    view.plot_button.invoke()
    assert view.ax.get_xlabel() == "sample index"
    assert [l.get_label() for l in view.signal_lines] == ["displacement"]
    run["LP_displacement"] = lp

    # an event with neither a time nor an index cannot be marked
    run["events"].append({"note": "no time, no index"})
    view.x_combo.set("time")
    view.plot_button.invoke()
    assert len(view.event_lines) == n_events(app) - 1
    assert "1 event(s) not markable" in view.status_var.get()
    # nor can an event whose x sample is NaN
    run["events"].pop()
    x = np.array(run["LP_displacement"], dtype=float)
    x[np.asarray(picked_indices(run))[0]] = np.nan
    run["lp_gappy"] = x
    view.refresh_candidates()
    view.x_combo.set("lp_gappy")
    view.plot_button.invoke()
    assert len(view.event_lines) == n_events(app) - 1
    assert "1 event(s) not markable" in view.status_var.get()
    assert all(np.isfinite(l.get_xdata()[0]) for l in view.event_lines)


# ------------------------------------------------------------ degenerate runs
def test_degenerate_runs(app):
    runs = app.data_manager.get_data("runs")
    indices_only = {k: v for k, v in runs[0].items() if k != "events"}
    indices_only["name"] = "run_indices_only"
    runs.extend([{"name": "run_no_time", "shear_stress": np.ones(5)}, indices_only])
    app.refresh_tree()

    # no time axis: opens empty and Plot does not raise
    v = RunSignalsView(app, 1)
    assert v.candidates == []
    assert list(v.x_combo["values"]) == ["index"] and v.x_combo.get() == "index"
    assert v.signal_lines == [] and v.event_lines == []
    assert "Select" in v.status_var.get()
    v.plot_button.invoke()
    v.on_close()

    # no events list: markers fall back to event_indices
    v = RunSignalsView(app, 2)
    expected = np.asarray(indices_only["time"])[np.asarray(picked_indices(indices_only))]
    assert marker_xs(v) == pytest.approx(sorted(expected))
    assert "not markable" not in v.status_var.get()
    v.on_close()
