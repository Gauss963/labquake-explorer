import pytest

from labquake_explorer.ui.actions import actions_for
from labquake_explorer.ui.views import EventPickerView


def open_from(app, path, label="Pick Events"):
    item = app.find_item(path)
    app.data_tree.selection_set(item)
    ctx = app.context_at(item)
    action = [a for a in actions_for(ctx.kind) if a.label == label][0]
    app.run_action(action, ctx)
    return app.child_windows[-1]


@pytest.fixture
def view(app):
    v = EventPickerView(app, 0)
    yield v
    v.on_close()


def test_browse_events_zoom_and_markers(app, view):
    truth = app.truth[0]
    n = len(truth.event_indices)
    assert view.title() == "Pick Events - run00"
    assert view.x_combo.get() == "time" and view.y_combo.get() == "shear_stress"
    assert view.start_var.get() == "-5" and view.end_var.get() == "5"
    assert view.picks == sorted(int(i) for i in truth.event_indices)
    assert list(view.event_combo["values"]) == [str(i) for i in range(n)]
    assert view.event_combo.get() == "0"
    assert "time" not in view.y_combo["values"] and "index" in view.x_combo["values"]
    assert view.ax.get_xlabel() == "time"
    assert len(view.spans) == n == len(view.picker.markers)
    assert not view.extract_button.instate(["disabled"])
    assert view.status_var.get().startswith(f"{n} picks")
    k = 1234
    assert view.picker.nearest_index(truth.time[k], truth.shear_stress[k]) == k

    # prev/next step the event, clamped at both ends
    view.prev_button.invoke()
    assert view.event_combo.get() == "0"
    view.next_button.invoke()
    assert view.event_combo.get() == "1"
    lo, hi = view.ax.get_xlim()
    assert lo < truth.event_times[1] < hi and hi - lo == pytest.approx(30.0, abs=0.05)
    for _ in range(n + 2):
        view.next_button.invoke()
    assert view.event_combo.get() == str(n - 1)

    # the event zoom survives a replot; a new X axis autoscales again
    view.event_combo.set("1")
    view.on_event_selected()
    lo, hi = view.ax.get_xlim()
    assert lo == pytest.approx(truth.event_times[1] - 15.0, abs=0.01)
    assert hi == pytest.approx(truth.event_times[1] + 15.0, abs=0.01)
    view.picker.add_point(view.picks[1] + 3)
    assert view.ax.get_xlim() == (lo, hi)
    view.x_combo.set("index")
    view.plot()
    assert view.ax.get_xlim() != (lo, hi)
    view.show_whole_run()
    lo, hi = view.ax.get_xlim()
    assert lo < 1 and hi > len(truth.time) - 2

    # markers follow the axes limits (what the toolbar's zoom, pan and home do)
    marker = view.picker.markers[0]
    w0 = marker.width
    view.ax.set_xlim(lo, lo + (hi - lo) / 10)
    assert view.picker.markers[0] is marker and marker.width == pytest.approx(w0 / 10, rel=0.01)
    view.picker.add_point(view.picks[-1] + 10)        # a replot keeps watching the new axes
    marker = view.picker.markers[0]
    view.ax.set_xlim(lo, hi)
    assert marker.width == pytest.approx(w0, rel=0.01)


def test_opened_from_array_run_and_event_indices(app):
    v = open_from(app, "runs/[0]/friction")
    assert isinstance(v, EventPickerView) and v.y_combo.get() == "friction"
    v.on_close()
    for path in ("runs/[0]", "runs/[0]/event_extraction", "runs/[0]/event_extraction/event_indices"):
        v = open_from(app, path)
        assert isinstance(v, EventPickerView) and v.run_idx == 0 and v.y_combo.get() == "shear_stress"
        v.on_close()


def test_spinbox_ticks_and_pick_edits(app, view):
    t = app.truth[0].time
    idx = view.picks[0]
    view.start_entry.event_generate("<<Increment>>")
    view.end_entry.event_generate("<<Decrement>>")
    assert view.start_var.get() == "-4.5" and view.end_var.get() == "4.5"
    assert view.window_s() == (-4.5, 4.5)
    span = view.spans[0]
    assert span.get_x() == pytest.approx(t[idx] - 4.5, abs=2e-3)
    assert span.get_x() + span.get_width() == pytest.approx(t[idx] + 4.5, abs=2e-3)

    # picks stay sorted, follow edits, and stay unsaved
    saved = list(app.truth[0].event_indices)
    first = view.picks[0]
    view.picker.add_point(5)                            # earlier than every existing pick
    assert view.picks[0] == 5 and view.event_combo.get() == "0"
    assert len(view.spans) == len(view.picks) == len(view.picker.markers)
    assert "unsaved" in view.status_var.get()
    view.picker.remove_point(0)
    assert view.picks[0] == first and "unsaved" not in view.status_var.get()
    # a drag ends with a "move" notification carrying the new sample
    view.picker.picks[-1] = 1
    view.picker.on_release(None)                        # nothing moved: no notification
    assert view.event_combo.get() == "0"
    view.on_picks_changed("move", 1)
    assert view.picks[0] == 1 and view.event_combo.get() == "0"
    assert app.data_manager.get_data("runs/[0]/event_extraction/event_indices") == saved


def test_extract(app, view, monkeypatch):
    from tkinter import messagebox
    from labquake_explorer.ui.views.event_picker_view import FIELDS_ALL, FIELDS_REFERENCE
    truth = app.truth[0]
    # a bad window is reported
    view.start_var.set("abc")
    assert view.extract(confirm=False) == 0 and "numbers" in view.status_var.get()
    view.start_var.set("3")
    view.end_var.set("1")
    assert view.extract(confirm=False) == 0 and "before" in view.status_var.get()
    # cancelling the confirmation keeps the events
    view.start_var.set("-2")
    view.end_var.set("3")
    monkeypatch.setattr(messagebox, "askokcancel", lambda *a, **k: False)
    before = app.data_manager.get_data("runs/[0]/events")
    assert view.extract() == 0
    assert app.data_manager.get_data("runs/[0]/events") is before

    # extract saves the picks and uses the window
    view.picker.add_point(5)
    assert view.fields_combo.get() == FIELDS_REFERENCE and view.waveform_fields() is None
    n = view.extract(confirm=False)
    assert n == len(truth.event_indices) + 1
    events = app.data_manager.get_data("runs/[0]/events")
    assert len(events) == n
    e = events[2]                                       # the second original event
    assert e["event_time"] == pytest.approx(truth.event_times[1])
    assert e["time"][0] == pytest.approx(truth.event_times[1] - 2.0, abs=2e-3)
    assert e["time"][-1] == pytest.approx(truth.event_times[1] + 3.0, abs=2e-3)
    assert e["shear_stress"].shape == e["time"].shape
    saved = app.data_manager.get_data("runs/[0]/event_extraction")
    assert saved == {"version": 1, "event_indices": view.picks, "start_s": -2.0, "end_s": 3.0, "x_field": "time",
                     "y_field": "shear_stress", "n_events": n, "waveform_fields": "reference"}
    assert saved["event_indices"] is not view.picks
    assert f"{n} events extracted" in view.status_var.get()

    # the waveform channels box reaches the record; the synthetic run has no
    # raw recorder, so no waveform and no error
    view.fields_combo.set(FIELDS_ALL)
    assert view.waveform_fields() == "all"
    view.fields_combo.set("pzt_1, pzt_2")
    assert view.waveform_fields() == ["pzt_1", "pzt_2"]
    assert view.extract(confirm=False) == n and "extracted" in view.status_var.get()
    assert app.data_manager.get_data("runs/[0]/event_extraction")["waveform_fields"] == ["pzt_1", "pzt_2"]
    view.fields_combo.set(FIELDS_ALL)
    view.extract(confirm=False)
    assert app.data_manager.get_data("runs/[0]/event_extraction")["waveform_fields"] == "all"
    # reopening restores the window and the channels choice
    v2 = EventPickerView(app, 0)
    assert v2.start_var.get() == "-2" and v2.end_var.get() == "3"
    assert v2.fields_combo.get() == FIELDS_ALL
    v2.on_close()
    assert EventPickerView.fields_text(["pzt_1", b"pzt_2"]) == "pzt_1, pzt_2"
    assert EventPickerView.fields_text("reference") == FIELDS_REFERENCE
    assert EventPickerView.fields_text(None) == FIELDS_REFERENCE


# ------------------------------------------------------- real Tk mouse events
def tk_xy(view, xd, yd):
    px, py = view.ax.transData.transform((xd, yd))
    return int(round(px)), int(round(view.canvas.figure.bbox.height - py))


def tk_drag(widget, x0, y0, x1, y1):
    widget.event_generate("<Button-1>", x=x0, y=y0)
    widget.update()
    for f in (0.3, 0.6, 1.0):
        widget.event_generate("<Motion>", x=int(x0 + f * (x1 - x0)), y=int(y0 + f * (y1 - y0)), state=0x100)
        widget.update()
    widget.event_generate("<ButtonRelease-1>", x=x1, y=y1)
    widget.update()


def shown(view):
    view.deiconify()
    view.update()
    view.canvas.draw()
    view.update()
    if not view.canvas.get_tk_widget().winfo_ismapped():
        pytest.skip("window not mapped: Tk mouse events need a display")
    return view.canvas.get_tk_widget()


def test_mouse_double_clicks_and_drag(app, view):
    t = app.truth[0].time
    y = app.truth[0].shear_stress
    widget = shown(view)
    n = len(view.picks)
    k = 1234
    x0, y0 = tk_xy(view, t[k], y[k])
    for _ in range(2):                                # Tk makes <Double-Button-1> from two quick presses
        widget.event_generate("<Button-1>", x=x0, y=y0)
        widget.event_generate("<ButtonRelease-1>", x=x0, y=y0)
    widget.update()
    xlim = view.ax.get_xlim()
    per_px = (xlim[1] - xlim[0]) / float(view.ax.bbox.width) / float(t[1] - t[0])   # samples per screen pixel
    assert len(view.picks) == n + 1 and abs(view.picks[0] - k) <= max(5.0, per_px)
    view.after(600)                                   # outside the double-click interval
    view.update()
    x0, y0 = tk_xy(view, t[view.picks[0]], y[view.picks[0]])
    for _ in range(2):
        widget.event_generate("<Button-3>", x=x0, y=y0)
        widget.event_generate("<ButtonRelease-3>", x=x0, y=y0)
    widget.update()
    assert len(view.picks) == n

    # a drag right after editing the window: the first canvas click moves the focus away
    k = view.picks[0]
    view.start_entry.focus_force()
    view.update()
    x0, y0 = tk_xy(view, t[k], y[k])
    x1, y1 = tk_xy(view, t[k + 300], y[k + 300])
    tk_drag(widget, x0, y0, x1, y1)
    moved = view.picks[0]
    assert abs(moved - (k + 300)) < 60                # one screen pixel spans ~50 samples here
    assert view.spans[0].get_x() == pytest.approx(t[moved] - 5, abs=2e-3)       # replotted on release


def test_form_starts_from_the_extraction_record_or_defaults(app):
    run = app.data_manager.get_data("runs/[0]")
    picks = [int(i) for i in run["event_extraction"]["event_indices"]]
    run["event_extraction"] = {"version": 1, "event_indices": picks[1:], "start_s": -2.0, "end_s": 3.0,
                               "x_field": "index", "y_field": "friction", "n_events": 3}
    v = EventPickerView(app, 0)
    assert v.picks == picks[1:] and v.start_var.get() == "-2" and v.end_var.get() == "3"
    assert v.x_combo.get() == "index" and v.y_combo.get() == "friction"
    # Save picks refreshes the record's picks and keeps the rest of the record
    v.picker.add_point(7)
    v.save_picks()
    record = app.data_manager.get_data("runs/[0]/event_extraction")
    assert record["event_indices"] == [7] + picks[1:] and record["start_s"] == -2.0 and record["n_events"] == 3
    assert "event_indices" not in run                  # the picks live in the record only
    v.on_close()
    # an older file: picks at the run's top level and the window under the record's old name
    del run["event_extraction"]
    run["event_indices"] = picks
    run["event_window"] = {"version": 1, "start_s": -1.0, "end_s": 4.0}
    v = EventPickerView(app, 0)
    assert v.picks == picks and v.start_var.get() == "-1" and v.end_var.get() == "4"
    v.save_picks()                                     # saving moves everything into event_extraction
    assert "event_indices" not in run and "event_window" not in run
    assert run["event_extraction"]["event_indices"] == picks and run["event_extraction"]["start_s"] == -1.0
    v.on_close()
    # nothing saved at all: the defaults, Extract disabled until a pick exists
    del run["event_extraction"]
    v = EventPickerView(app, 0)
    assert v.start_var.get() == "-5" and v.end_var.get() == "5"
    assert v.picks == [] and v.event_combo.get() == ""
    assert v.extract_button.instate(["disabled"]) and v.extract(confirm=False) == 0
    v.picker.add_point(100)
    assert not v.extract_button.instate(["disabled"]) and v.event_combo.get() == "0"
    v.save_picks()
    assert app.data_manager.get_data("runs/[0]/event_extraction/event_indices") == [100]
    v.on_close()


def test_y_can_be_a_converted_source_channel(tmp_path, app_from_file):
    import numpy as np
    from labquake_explorer.data.data_manager import DataManager
    from labquake_explorer.ui.views import event_picker_view
    from tests.test_event_processor import sample_aligned_run
    run = sample_aligned_run(tmp_path)
    run["elsys"]["conversion"] = {"channels": ["sigma_xy"], "inputs": ["pzt_1"], "matrix": [[-3.0]], "unit": "MPa"}
    dm = DataManager()
    dm.data = {"name": "t0001", "runs": [run]}
    dm.save_file(tmp_path / "t0001.h5")
    app = app_from_file(tmp_path / "t0001.h5")
    v = EventPickerView(app, 0)
    assert "elsys/sigma_xy" in v.y_combo["values"] and "elsys/sigma_xy" not in v.x_combo["values"]
    v.y_combo.set("elsys/sigma_xy")
    v.plot()
    y = v.y_values()
    assert y.shape == v.time.shape and v.time[np.nanargmin(y)] == pytest.approx(1.2, abs=1e-4)
    assert v.ax.get_ylabel() == "elsys/sigma_xy (MPa)"
    assert any(k[1] == "sigma_xy" for k in event_picker_view._SOURCE_SERIES)   # read once, then cached
    v.picker.add_point(int(np.nanargmin(y)))
    assert v.extract(confirm=False) == 1
    assert app.data_manager.get_data("runs/[0]/event_extraction")["y_field"] == "elsys/sigma_xy"
    # raw file gone (moved, or the experiment copied elsewhere): nothing is extracted, the events stay
    events = app.data_manager.get_data("runs/[0]/events")
    raw = tmp_path / run["elsys"]["filename"]
    raw.rename(raw.with_suffix(".moved"))
    assert v.extract(confirm=False) == 0 and "cannot be opened" in v.status_var.get()
    assert app.data_manager.get_data("runs/[0]/events") is events and "waveform" in events[0]
    v.on_close()


def test_converted_channel_of_a_dual_mode_file_is_drawn_over_the_whole_run(tmp_path, app_from_file):
    import numpy as np
    from labquake_explorer.data.data_manager import DataManager
    from tests.test_event_processor import dual_mode_run
    dm = DataManager()
    dm.data = {"name": "t0001", "runs": [dual_mode_run(tmp_path)]}
    dm.save_file(tmp_path / "t0001.h5")
    app = app_from_file(tmp_path / "t0001.h5")
    v = EventPickerView(app, 0)
    v.y_combo.set("elsys/sigma_xy")
    v.plot()
    y = v.y_values()
    inside = (v.time >= 0.5) & (v.time <= 2.4995)             # the Elsys continuous block on the run clock
    assert y.shape == v.time.shape and np.isfinite(y[inside]).all() and np.isnan(y[~inside]).all()
    v.on_close()
