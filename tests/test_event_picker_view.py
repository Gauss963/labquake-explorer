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


def test_defaults(app, view):
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


def test_opened_from_array_run_and_event_indices(app):
    v = open_from(app, "runs/[0]/friction")
    assert isinstance(v, EventPickerView) and v.y_combo.get() == "friction"
    v.on_close()
    for path in ("runs/[0]", "runs/[0]/event_extraction", "runs/[0]/event_extraction/event_indices"):
        v = open_from(app, path)
        assert isinstance(v, EventPickerView) and v.run_idx == 0 and v.y_combo.get() == "shear_stress"
        v.on_close()


def test_picks_stay_sorted_and_follow_edits(app, view):
    truth = app.truth[0]
    saved = list(truth.event_indices)
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


def test_zoom_survives_a_replot(app, view):
    t_event = app.truth[0].event_times[1]
    view.event_combo.set("1")
    view.on_event_selected()
    lo, hi = view.ax.get_xlim()
    assert lo == pytest.approx(t_event - 15.0, abs=0.01) and hi == pytest.approx(t_event + 15.0, abs=0.01)
    view.picker.add_point(view.picks[1] + 3)            # replots: the zoom must stay
    assert view.ax.get_xlim() == (lo, hi)
    view.x_combo.set("index")
    view.plot()                                         # a new X axis autoscales again
    assert view.ax.get_xlim() != (lo, hi)
    view.show_whole_run()
    lo, hi = view.ax.get_xlim()
    assert lo < 1 and hi > len(app.truth[0].time) - 2


def test_spinbox_ticks_redraw_the_windows(app, view):
    t = app.truth[0].time
    idx = view.picks[0]
    view.start_entry.event_generate("<<Increment>>")
    view.end_entry.event_generate("<<Decrement>>")
    assert view.start_var.get() == "-4.5" and view.end_var.get() == "4.5"
    assert view.window_s() == (-4.5, 4.5)
    span = view.spans[0]
    assert span.get_x() == pytest.approx(t[idx] - 4.5, abs=2e-3)
    assert span.get_x() + span.get_width() == pytest.approx(t[idx] + 4.5, abs=2e-3)


def test_prev_next_buttons_step_the_event(app, view):
    n = len(view.picks)
    view.prev_button.invoke()
    assert view.event_combo.get() == "0"                # clamped at the first event
    view.next_button.invoke()
    assert view.event_combo.get() == "1"
    lo, hi = view.ax.get_xlim()
    assert lo < app.truth[0].event_times[1] < hi and hi - lo == pytest.approx(30.0, abs=0.05)
    for _ in range(n + 2):
        view.next_button.invoke()
    assert view.event_combo.get() == str(n - 1)         # clamped at the last event


def test_extract_saves_picks_and_uses_the_window(app, view):
    truth = app.truth[0]
    view.start_var.set("-2")
    view.end_var.set("3")
    view.picker.add_point(5)
    n = view.extract(confirm=False)
    assert n == len(truth.event_indices) + 1
    assert app.data_manager.get_data("runs/[0]/event_extraction/event_indices") == view.picks
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
    v2 = EventPickerView(app, 0)                        # reopening restores the window
    assert v2.start_var.get() == "-2" and v2.end_var.get() == "3"
    v2.on_close()


def test_extract_asks_before_replacing(app, view, monkeypatch):
    from tkinter import messagebox
    monkeypatch.setattr(messagebox, "askokcancel", lambda *a, **k: False)
    before = app.data_manager.get_data("runs/[0]/events")
    assert view.extract() == 0
    assert app.data_manager.get_data("runs/[0]/events") is before


def test_bad_window_is_reported(app, view):
    view.start_var.set("abc")
    assert view.extract(confirm=False) == 0 and "numbers" in view.status_var.get()
    view.start_var.set("3")
    view.end_var.set("1")
    assert view.extract(confirm=False) == 0 and "before" in view.status_var.get()


def test_without_picks(app):
    run = app.data_manager.get_data("runs/[0]")
    del run["event_extraction"]
    v = EventPickerView(app, 0)
    assert v.picks == [] and v.event_combo.get() == ""
    assert v.extract_button.instate(["disabled"]) and v.extract(confirm=False) == 0
    v.picker.add_point(100)
    assert not v.extract_button.instate(["disabled"]) and v.event_combo.get() == "0"
    v.save_picks()
    assert app.data_manager.get_data("runs/[0]/event_extraction/event_indices") == [100]
    v.on_close()


def test_nearest_index_from_the_curve(app, view):
    t = app.truth[0].time
    y = app.truth[0].shear_stress
    k = 1234
    assert view.picker.nearest_index(t[k], y[k]) == k


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


def test_drag_right_after_editing_the_window(app, view):
    t = app.truth[0].time
    y = app.truth[0].shear_stress
    widget = shown(view)
    k = view.picks[0]
    view.start_entry.focus_force()                    # the first canvas click moves the focus away
    view.update()
    x0, y0 = tk_xy(view, t[k], y[k])
    x1, y1 = tk_xy(view, t[k + 300], y[k + 300])
    tk_drag(widget, x0, y0, x1, y1)
    moved = view.picks[0]
    assert abs(moved - (k + 300)) < 60                # one screen pixel spans ~50 samples here
    assert view.spans[0].get_x() == pytest.approx(t[moved] - 5, abs=2e-3)       # replotted on release


def test_double_click_adds_and_right_double_click_removes(app, view):
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


def test_markers_follow_the_axes_limits(app, view):
    marker = view.picker.markers[0]
    w0 = marker.width
    lo, hi = view.ax.get_xlim()
    view.ax.set_xlim(lo, lo + (hi - lo) / 10)         # what the toolbar's zoom, pan and home do
    assert view.picker.markers[0] is marker and marker.width == pytest.approx(w0 / 10, rel=0.01)
    view.picker.add_point(view.picks[-1] + 10)        # a replot keeps watching the new axes
    marker = view.picker.markers[0]
    view.ax.set_xlim(lo, hi)
    assert marker.width == pytest.approx(w0, rel=0.01)


def test_form_starts_from_the_extraction_record(app):
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
    # nothing saved at all: the defaults
    del run["event_extraction"]
    v = EventPickerView(app, 0)
    assert v.start_var.get() == "-5" and v.end_var.get() == "5" and v.picks == []
    v.on_close()


def test_waveform_channels_box_reaches_the_extraction_and_is_restored(app, view):
    from labquake_explorer.ui.views.event_picker_view import FIELDS_ALL, FIELDS_REFERENCE
    assert view.fields_combo.get() == FIELDS_REFERENCE and view.waveform_fields() is None
    view.fields_combo.set(FIELDS_ALL)
    assert view.waveform_fields() == "all"
    view.fields_combo.set("pzt_1, pzt_2")
    assert view.waveform_fields() == ["pzt_1", "pzt_2"]
    n = view.extract(confirm=False)                     # the synthetic run has no raw recorder: no waveform, no error
    assert n == len(view.picks) and "extracted" in view.status_var.get()
    saved = app.data_manager.get_data("runs/[0]/event_extraction")
    assert saved["waveform_fields"] == ["pzt_1", "pzt_2"]
    view.fields_combo.set(FIELDS_ALL)
    view.extract(confirm=False)
    assert app.data_manager.get_data("runs/[0]/event_extraction")["waveform_fields"] == "all"
    v2 = EventPickerView(app, 0)                        # reopening restores the choice
    try:
        assert v2.fields_combo.get() == FIELDS_ALL
        assert EventPickerView.fields_text(["pzt_1", b"pzt_2"]) == "pzt_1, pzt_2"
        assert EventPickerView.fields_text("reference") == FIELDS_REFERENCE and EventPickerView.fields_text(None) == FIELDS_REFERENCE
    finally:
        v2.on_close()
