"""Headless-Tk tests for the Slip Distribution view (coseismic slip per sensor)."""
import contextlib
import functools
import inspect
import math
import warnings

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from labquake_explorer.analysis import coseismic_slip as cs
from labquake_explorer.analysis import slip_profile as sprof
from labquake_explorer.ui.actions import actions_for
from labquake_explorer.ui.views import SlipDistributionView
from labquake_explorer.ui.views.slip_distribution_view import T_I_FLAGS, slip_fields
from tests.synthetic import add_slip_array

EVENT = 1


@pytest.fixture
def slip_app(app):
    """The synthetic experiment with an 8-sensor slip array on every event and
    an extraction window of +-5 s recorded for the run."""
    run = app.data_manager.get_data("runs/[0]")
    run["event_extraction"].update({"start_s": -5.0, "end_s": 5.0})
    app.slip_truth = [add_slip_array(event, noise_um=0.05, seed=j) for j, event in enumerate(run["events"])]
    app.refresh_tree()
    return app


@contextlib.contextmanager
def opened(app, event_idx=EVENT):
    v = SlipDistributionView(app, 0, event_idx)
    try:
        yield v
    finally:
        if v.winfo_exists():
            v.on_close()


@pytest.fixture
def view(slip_app):
    with opened(slip_app) as v:
        yield v


def events(app):
    return app.data_manager.get_data("runs/[0]")["events"]


def type_patch(v, lo, hi, apply=True):
    v.patch_lo_var.set(lo)
    v.patch_hi_var.set(hi)
    if apply:
        v.schedule(v.on_patch_edited, now=True)


def enter(v, handler, **texts):
    """Type ``texts`` into the named entry variables and apply them at once."""
    for name, text in texts.items():
        getattr(v, name).set(text)
    v.schedule(handler, now=True)


def set_front_mode(v, mode):
    v.front_mode_combobox.set(mode)
    v.on_front_settings()


def type_param(v, model, name, text, field="value"):
    v.param_vars[model][name][field].set(text)
    handler = v.on_param_value_edited if field == "value" else v.on_param_bounds_edited
    v.schedule(functools.partial(handler, model, name), now=True)


def press(view, ax, xd, yd, button=1, name="button_press_event"):
    """matplotlib's own dispatch of a mouse event at data (xd, yd) of ``ax``."""
    held = {}
    if name == "motion_notify_event" and "buttons" in inspect.signature(MouseEvent).parameters:
        held["buttons"] = {button}        # the buttons held during a move: the toolbar's zoom reads them (matplotlib 3.11)
    event = MouseEvent(name, view.canvas, *ax.transData.transform((xd, yd)), button=button, **held)
    view.canvas.callbacks.process(name, event)


def point(view, panel, k):
    xs, ys = view.point_artists[panel].get_data()
    return xs[k], ys[k]


def test_opens_with_defaults_navigates_and_refuses_events_without_slip(slip_app, view, no_dialogs):
    truth = slip_app.slip_truth[EVENT]
    ctx = slip_app.context_at(slip_app.find_item(f"runs/[0]/events/[{EVENT}]"))
    assert "Slip Distribution" in [a.label for a in actions_for(ctx.kind)]
    assert view.title() == f"Slip Distribution - Event {EVENT}"
    assert view.field == "slip" and list(view.field_combobox["values"]) == ["slip"]
    assert len(view.channels) == 8 and view.x_mm[0] == 31.0 and (view.start_s, view.end_s) == (-5.0, 5.0)
    assert all(m == cs.METHOD_MAX_RATE for m in view.methods) and view.method_combobox.get() == "max slip rate"
    assert view.fit_before[0] == (-3.75, -1.25) and view.fit_after[0] == (1.25, 3.75)
    assert view.same_windows_var.get() is True
    assert (view.reference_var.get(), view.smoothing_var.get(), view.search_var.get(), view.snr_var.get()) == \
        ("500", "10", "50", "5")
    assert view.rate_filter() == "lowpass" and view.cutoff_var.get() == "50"
    assert (view.lowpass_type(), view.lowpass_order_var.get()) == ("butterworth", "4")
    assert (view.savgol_window_var.get(), view.savgol_order_var.get()) == ("20", "2")
    np.testing.assert_allclose([s.coseismic_slip for s in view.result.sensors], truth["slips_um"], atol=0.2)
    assert len(view.table.get_children()) == 8
    assert "t_i from" in view.table["columns"] and "t_i - event (ms)" in view.table["columns"]
    assert "not saved" in view.status_var.get() and "reference at +" in view.status_var.get()
    assert set(view.handles) == {"t_i", "before_start", "before_end", "after_start", "after_end"}
    # the rate axes sit behind the trace axes (which get the mouse) and keep their label on the right
    assert view.ax_rate.get_zorder() < view.ax_trace.get_zorder() and not view.ax_trace.patch.get_visible()
    assert view.ax_rate.yaxis.get_label_position() == "right"
    # profile and front
    assert view.profile_model() == "eaz" and view.weighted_var.get() is False and view.profile_use == [True] * 8
    assert view.front_mode_combobox.get() == "bilateral" and view.front_use_auto and view.front_use == [True] * 8
    fit, front = view.profile_fit, view.front_fit
    assert fit.model == "eaz" and fit.valid and fit.r2 > 0.99 and fit.n == 8
    # the hinge's apex lands within a sensor spacing of where the synthetic front starts (279 mm)
    assert front.valid and len(front.segments) == 2 and abs(front.split_x_mm - 279.0) < 62.0
    left, right = front.segments
    assert left.x_to == right.x_from == front.split_x_mm
    assert left.velocity_m_s == pytest.approx(-50.0, rel=0.25) and right.velocity_m_s == pytest.approx(50.0, rel=0.25)
    assert "fronts start at x = " in view.front_text_var.get() and "m/s" in view.front_text_var.get()
    # the profile panel shows the slip with sigma error bars and the model curve; the front panel the t_i
    xs, ys = view.point_artists["profile"].get_data()
    np.testing.assert_allclose(xs, truth["x_mm"])
    np.testing.assert_allclose(ys, [s.coseismic_slip for s in view.result.sensors])
    segs = view.profile_errorbar.lines[2][0].get_segments()
    np.testing.assert_allclose([s[1][1] - s[0][1] for s in segs], 2 * np.array([s.noise for s in view.result.sensors]))
    assert view.profile_curve is not None and "EAZ" in view.profile_text_var.get()
    np.testing.assert_allclose(view.point_artists["front"].get_data()[1], [1e3 * s.t_shift_s for s in view.result.sensors])
    # sensor navigation and the table
    view.step_sensor(+1)
    assert view.sensor == 1 and view.table.selection() == ("1",)
    view.step_sensor(-5)
    assert view.sensor == 0
    view.table.selection_set("6")
    view.on_table_select()
    assert view.sensor == 6 and view.sensor_combobox.get().startswith("7: slip_7") and "slip_7" in view.ax_trace.get_title()
    # zoom buttons; the zoom survives a sensor change
    view.set_sensor(4)
    view.zoom_to_event_time()
    lo, hi = view.ax_trace.get_xlim()
    ti = view.result.sensors[4].t_shift_s
    assert lo == pytest.approx(ti - 0.2) and hi == pytest.approx(ti + 0.2)
    view.step_sensor(+1)
    assert view.ax_trace.get_xlim() == (lo, hi)
    view.show_whole_window()
    lo, hi = view.ax_trace.get_xlim()
    assert lo == pytest.approx(-5.0, abs=0.01) and hi > 4.9 and math.isfinite(view.ax_rate.get_ylim()[1])
    # no extraction window recorded: the default windows follow the event's time axis
    extraction = slip_app.data_manager.get_data("runs/[0]")["event_extraction"]
    del extraction["start_s"], extraction["end_s"]
    view.set_event(0)
    t_rel = view.time - view.t_event
    assert view.start_s == pytest.approx(t_rel[0]) and view.end_s == pytest.approx(t_rel[-1] + cs.sampling_interval(view.time))
    assert view.fit_before[0] == pytest.approx((0.75 * view.start_s, 0.25 * view.start_s))
    assert view.fit_after[0] == pytest.approx((0.25 * view.end_s, 0.75 * view.end_s))
    # an event without slip: not switched to, and not opened
    del events(slip_app)[2]["slip"]
    view.set_event(2)
    assert view.event_idx == 0 and any("No slip data" in str(c) for c in no_dialogs)
    assert slip_fields(events(slip_app)[2]) == []
    with pytest.raises(ValueError, match="no slip channel array"):
        SlipDistributionView(slip_app, 0, 2)


def test_a_failed_switch_keeps_nothing_of_the_previous_event(slip_app, view):
    assert view.ax_profile.lines and view.profile_text_var.get()
    view.search_var.set("0")                              # typed, never applied
    view.set_event(2)
    assert view.event_idx == 2 and view.result is None and view.profile_fit is None and view.front_fit is None
    assert view.table.get_children() == () and not view.ax_trace.lines
    assert not view.ax_profile.lines and not view.ax_front.lines and view.profile_text_var.get() == ""
    assert "search half-width" in view.status_var.get() and view.record() is None
    assert not view.save() and "slip_analysis" not in view.event
    enter(view, view.on_settings_edited, search_var="50")     # the same text as before still recomputes
    assert view.result is not None and view.result.event_time == view.t_event
    assert view.save() and "slip_analysis" in view.event


def test_typing_event_time_windows_and_method(slip_app, view):
    truth = slip_app.slip_truth[EVENT]
    view.set_sensor(3)
    pick = view.result.sensors[3].t_shift_s
    enter(view, view.on_t_shift_edited)             # the rounded text read back, e.g. on focus-out: no edit
    enter(view, view.on_window_edited)
    assert view.methods[3] == cs.METHOD_MAX_RATE and view.result.sensors[3].t_shift_s == pick
    view.set_sensor(4)
    enter(view, view.on_t_shift_edited, t_shift_var=f"{(truth['t_i'][4] - view.t_event) * 1e3 + 5.0:.3f}")
    s = view.result.sensors[4]
    assert s.method == cs.METHOD_MANUAL and view.method_combobox.get() == "manual"
    assert s.t_shift_s == pytest.approx(truth["t_i"][4] - view.t_event + 0.005, abs=1e-9)
    # a window edit applies to every sensor while "same windows" is on, else only to the selected one
    enter(view, view.on_window_edited, before_start_var="-2", before_end_var="-0.5")
    assert all(w == (-2.0, -0.5) for w in view.fit_before)
    enter(view, view.on_window_edited, same_windows_var=False, after_end_var="2")
    assert view.fit_after[4] == (1.25, 2.0) and view.fit_after[0] == (1.25, 3.75)
    view.same_windows_var.set(True)                       # turned back on: the selected sensor's windows for all
    view.on_same_windows_toggled()
    assert all(w == (1.25, 2.0) for w in view.fit_after)
    # a before window that reaches past t_i is fitted and flagged; one past the after window is refused
    enter(view, view.on_window_edited, before_end_var="0.5")
    assert all(cs.FLAG_BEFORE_OVERLAP in s.flags for s in view.result.sensors)
    enter(view, view.on_window_edited, before_end_var="1.5")
    assert "before window must end" in view.status_var.get() and view.fit_before[0] == (-2.0, 0.5)
    enter(view, view.on_window_edited, before_end_var="abc")
    assert "enter a number" in view.status_var.get()
    view.before_end_var.set("0.5")
    # a method switch keeps the pick; repick puts every sensor back
    view.set_sensor(2)
    pick = view.result.sensors[2].t_shift_s
    view.method_combobox.set("manual")
    view.on_method_selected()
    assert view.methods[2] == cs.METHOD_MANUAL and view.result.sensors[2].t_shift_s == pytest.approx(pick)
    view.repick_all()
    assert all(m == cs.METHOD_MAX_RATE for m in view.methods)
    enter(view, view.on_settings_edited, search_var="10")
    assert view.result.search_halfwidth_s == pytest.approx(0.010)
    enter(view, view.on_settings_edited, smoothing_var="-1")
    assert "window" in view.status_var.get()


def test_a_typed_value_is_applied_before_save_navigation_and_dropped_on_close(slip_app, view):
    view.set_sensor(2)
    view.t_shift_var.set("12.5")
    view.schedule(view.on_t_shift_edited)                 # still waiting for the keys to settle
    assert view.methods[2] == cs.METHOD_MAX_RATE
    assert view.save()
    rec = view.event["slip_analysis"]
    assert rec["time_method"][2] == "manual" and rec["t_shift_s"][2] == pytest.approx(0.0125)
    view.t_shift_var.set("13")
    view.schedule(view.on_t_shift_edited)
    view.step_sensor(+1)                                  # goes to sensor 2, then the view moves on
    assert view.sensor == 3 and view.t_shift[2] == pytest.approx(0.013) and view.methods[3] == cs.METHOD_MAX_RATE
    view.before_end_var.set("abc")                        # a bad value in an entry is never saved past
    assert not view.save() and "not saved" in view.status_var.get()
    assert view.event["slip_analysis"]["t_shift_s"][2] == pytest.approx(0.0125)
    view.t_shift_var.set("5")                             # closing cancels a pending recompute
    view.schedule(view.on_t_shift_edited)
    after_id = view._compute_after_id
    assert after_id is not None
    view.on_close()
    assert after_id not in view.tk.splitlist(view.tk.call("after", "info"))


def test_dragging_event_time_and_windows(slip_app, view):
    view.set_sensor(3)
    ti = view.result.sensors[3].t_shift_s
    view.begin_drag("t_i", ti)
    state = view.drag_to(ti + 0.010)
    assert state["t_i"] == pytest.approx(ti + 0.010)
    assert float(view.t_shift_var.get()) == pytest.approx((ti + 0.010) * 1e3, abs=1e-3)
    # t_i moves alone: the windows stay where they are
    assert state["before"] == (-3.75, -1.25) and state["after"] == (1.25, 3.75)
    assert [view.handles[k].get_xdata()[0] for k in ("before_start", "before_end", "after_start", "after_end")] == \
        [-3.75, -1.25, 1.25, 3.75]
    view.end_drag()
    assert view.methods[3] == cs.METHOD_MANUAL and view.result.sensors[3].t_shift_s == pytest.approx(ti + 0.010)
    assert view.fit_before[3] == (-3.75, -1.25) and view.fit_after[3] == (1.25, 3.75)
    # a window edge cannot cross t_i, the body slides as one
    ti = view.result.sensors[3].t_shift_s
    view.begin_drag("before_end", -1.25)
    assert view.drag_to(ti + 0.5)["before"][1] == pytest.approx(ti)
    view.end_drag()
    assert all(w[1] == pytest.approx(ti) for w in view.fit_before)
    view.begin_drag("after", 2.0)
    assert view.drag_to(2.0 - 5.0)["after"] == (pytest.approx(ti), pytest.approx(ti + 2.5))
    view.end_drag()
    assert view.fit_after[0] == (pytest.approx(ti), pytest.approx(ti + 2.5))
    # a click without a move changes nothing
    before = list(view.fit_before)
    view.begin_drag("before_start", -3.0)
    view.end_drag()
    assert view.fit_before == before
    # t_i stays between the windows
    view.begin_drag("t_i", ti)
    assert view.drag_to(-2.0)["t_i"] == pytest.approx(view.fit_before[3][1])
    assert view.drag_to(5.0)["t_i"] == pytest.approx(view.fit_after[3][0])
    view.end_drag()


def test_mouse_events_grab_handles_and_drags_start_from_the_windows_shown(slip_app, view):
    view.set_sensor(4)
    view.canvas.draw()
    ax = view.ax_trace
    y_mid = float(np.mean(ax.get_ylim()))

    def mouse(name, x, dx_px=0.0):
        px, py = ax.transData.transform((x, y_mid))
        return MouseEvent(name, view.canvas, px + dx_px, py, button=1)

    ti = view.result.sensors[4].t_shift_s
    down = mouse("button_press_event", ti, dx_px=4)        # a few pixels off still grabs t_i
    assert down.inaxes is ax                               # not the rate axes on top of it
    view.on_press(down)
    assert view._drag["what"] == "t_i"
    view.on_motion(mouse("motion_notify_event", ti + 0.5))
    view.on_release(mouse("button_release_event", ti + 0.5))
    assert view._drag is None and view.methods[4] == cs.METHOD_MANUAL
    assert view.t_shift[4] == pytest.approx(ti + 0.5, abs=0.02)
    ti = view.t_shift[4]
    view.on_press(mouse("button_press_event", 1.25, dx_px=-3))
    assert view._drag["what"] == "after_start"
    view.on_release(mouse("button_release_event", 1.25))
    view.on_press(mouse("button_press_event", -2.0))
    assert view._drag["what"] == "before"
    view.on_release(mouse("button_release_event", -2.0))
    view.on_press(mouse("button_press_event", (ti + 1.25) / 2))   # between t_i and the after window: nothing
    assert view._drag is None
    view.toolbar.zoom()
    view.on_press(mouse("button_press_event", ti))
    assert view._drag is None
    view.toolbar.zoom()
    # a drag after a failed recompute starts from the windows shown
    view.search_var.set("abc")                            # an entry that cannot be applied
    view.begin_drag("after_start", 1.25)
    view.drag_to(2.0)
    view.end_drag()
    assert view.status_var.get().startswith("not recomputed: search half-width")
    assert view.fit_after[0] == (2.0, 3.75) and view.result.sensors[0].fit_after_s == (1.25, 3.75)
    view.begin_drag("after_end", 3.75)                    # the other edge keeps the start just dragged
    view.drag_to(4.5)
    view.end_drag()
    assert view.fit_after[0] == (2.0, 4.5) and view.handles["after_start"].get_xdata()[0] == 2.0
    enter(view, view.on_settings_edited, search_var="50")     # fixed: everything applies
    assert view.result.sensors[0].fit_after_s == (2.0, 4.5)


def test_clicks_on_profile_and_front_points_select_the_sensor(slip_app, view):
    view.canvas.draw()
    for panel, ax, k in (("profile", view.ax_profile, 5), ("front", view.ax_front, 2)):
        press(view, ax, *point(view, panel, k))
        assert view.sensor == k and view._drag is None and view.channels[k] in view.ax_trace.get_title()
    view.set_sensor(0)
    # only a left click: matplotlib picks on a scroll and on the other buttons too
    scroll = MouseEvent("scroll_event", view.canvas, *view.ax_profile.transData.transform(point(view, "profile", 5)),
                        step=1)
    view.canvas.callbacks.process("scroll_event", scroll)
    for button in (2, 3):
        press(view, view.ax_front, *point(view, "front", 3), button=button)
    assert view.sensor == 0
    press(view, view.ax_front, *point(view, "front", 3))
    assert view.sensor == 3


# ------------------------------------------------------- real Tk mouse events
def shown(view):
    view.deiconify()
    view.update()
    view.canvas.draw()
    view.update()
    if not view.canvas.get_tk_widget().winfo_ismapped():
        pytest.skip("window not mapped: Tk mouse events need a display")
    return view.canvas.get_tk_widget()


def tk_xy(view, xd, yd, ax=None):
    px, py = (ax or view.ax_trace).transData.transform((xd, yd))
    return int(round(px)), int(round(view.canvas.figure.bbox.height - py))


def tk_drag(widget, x0, y0, x1, y1):
    widget.event_generate("<Button-1>", x=x0, y=y0)
    widget.update()
    for f in (0.3, 0.6, 1.0):
        widget.event_generate("<Motion>", x=int(x0 + f * (x1 - x0)), y=int(y0 + f * (y1 - y0)), state=0x100)
        widget.update()
    widget.event_generate("<ButtonRelease-1>", x=x1, y=y1)
    widget.update()


def test_real_tk_mouse_and_scrolling_columns(slip_app, view):
    view.set_sensor(4)
    widget = shown(view)
    y = float(np.mean(view.ax_trace.get_ylim()))
    ti = view.result.sensors[4].t_shift_s
    view.after_end_var.set("3")                          # typed, then straight to the mouse
    view.schedule(view.on_window_edited)
    view.entries["after end"].focus_force()
    view.update()
    tk_drag(widget, *tk_xy(view, ti, y), *tk_xy(view, ti + 0.5, y))
    assert view.methods[4] == cs.METHOD_MANUAL and view.t_shift[4] == pytest.approx(ti + 0.5, abs=0.05)
    assert all(w == (1.25, 3.0) for w in view.fit_after)
    ti = view.t_shift[4]
    tk_drag(widget, *tk_xy(view, -3.75, y), *tk_xy(view, -3.0, y))
    assert all(w[0] == pytest.approx(-3.0, abs=0.05) and w[1] == -1.25 for w in view.fit_before)
    tk_drag(widget, *tk_xy(view, 2.0, y), *tk_xy(view, 1.5, y))
    assert all(w == (pytest.approx(0.75, abs=0.05), pytest.approx(2.5, abs=0.05)) for w in view.fit_after)
    assert view.methods[4] == cs.METHOD_MANUAL and view.t_shift[4] == ti
    # a click on a profile point picks it while the press is handled; the table selection comes back through Tk
    x, y = tk_xy(view, *point(view, "profile", 6), ax=view.ax_profile)
    widget.event_generate("<Button-1>", x=x, y=y)
    assert view.sensor == 6
    widget.event_generate("<ButtonRelease-1>", x=x, y=y)
    widget.update()
    assert view.sensor == 6 and view._drag is None
    # the wheel over a front point leaves the sensor
    x, y = tk_xy(view, *point(view, "front", 3), ax=view.ax_front)
    scrolls = []
    view.canvas.mpl_connect("scroll_event", scrolls.append)
    widget.event_generate("<MouseWheel>", x=x, y=y, rootx=widget.winfo_rootx() + x,
                          rooty=widget.winfo_rooty() + y, delta=1)
    widget.update()
    if scrolls:
        assert scrolls[0].inaxes is view.ax_front and view.sensor == 6
    else:                                                # some displays do not deliver it: say so
        warnings.warn("the wheel event did not reach the canvas; the wheel check did not run")
    # the control columns scroll in a short window
    for panel in (view.left_panel, view.fits_panel):
        assert panel.inner.winfo_children()
    view.geometry("1400x500")
    view.update()
    assert view.left_panel.overflows() and view.left_panel.scrollbar.winfo_ismapped()
    view.left_panel.scroll(5)
    assert view.left_panel.canvas.yview()[0] > 0.0
    assert view.save_button.winfo_toplevel() is view       # still reachable by scrolling
    view.geometry(f"1400x{view.left_panel.content_height() + 150}")
    view.update()
    assert not view.left_panel.scrollbar.winfo_ismapped()


def test_save_and_reopen_restore_everything(slip_app, view, tmp_path, app_from_file):
    view.set_event_time(0, 0.031)
    view.set_sensor(5)
    view.set_event_time(5, view.result.sensors[5].t_shift_s + 0.003)
    view.same_windows_var.set(False)
    view.set_windows((-3.0, -1.0), (1.0, 3.0))
    enter(view, view.on_settings_edited, snr_var="3")
    view.select_profile_model("quadratic")
    view.set_profile_use(1, False)
    view.set_front_use(6, False)
    set_front_mode(view, "unilateral")
    type_patch(view, "120", "380")
    assert view.save()
    rec = slip_app.data_manager.get_data(f"runs/[0]/events/[{EVENT}]/slip_analysis")
    assert rec is view.event["slip_analysis"]
    assert rec["version"] == 2 and rec["field"] == "slip" and rec["unit"] == "um" and rec["same_windows"] is False
    assert rec["time_method"][5] == "manual" and rec["fit_before_s"][5] == [-3.0, -1.0]
    assert rec["fit_before_s"][0] == [-3.75, -1.25] and rec["rate_snr"] == 3.0 and rec["patch_x_mm"] == [120.0, 380.0]
    assert rec["profile"]["model"] == "quadratic" and rec["profile"]["use"][1] is False
    assert rec["front"]["mode"] == "unilateral" and rec["front"]["use"][6] is False and rec["front"]["use_auto"] is False
    assert set(rec["profile"]["params"]) == {"c0", "c1", "c2"} and rec["front"]["segments"]
    assert "matches the saved record" in view.status_var.get()
    path = tmp_path / "exp.h5"
    slip_app.data_manager.save_file(path)
    for app in (slip_app, app_from_file(path)):          # in memory, and through an HDF5 file
        with opened(app) as v2:
            assert v2.methods[0] == v2.methods[5] == "manual" and v2.t_shift[0] == pytest.approx(0.031)
            assert v2.t_shift[5] == pytest.approx(rec["t_shift_s"][5])
            assert v2.fit_before[5] == (-3.0, -1.0) and v2.same_windows_var.get() is False and v2.snr_var.get() == "3"
            np.testing.assert_allclose([s.coseismic_slip for s in v2.result.sensors], rec["coseismic_slip_um"])
            assert v2.profile_model() == "quadratic" and v2.profile_use[1] is False
            assert v2.front_mode_combobox.get() == "unilateral" and v2.front_use[6] is False and not v2.front_use_auto
            assert v2.patch_x_mm == (120.0, 380.0)                 # from the record: the experiment has none
            assert "matches the saved record" in v2.status_var.get()
            if app is slip_app:
                v2.repick_all()
            else:
                v2.select_profile_model("eaz")
            assert "differs from the saved record" in v2.status_var.get()


# ------------------------------------------------- slip profile and front
def test_profile_models_tips_and_choices_carried_to_another_event(slip_app, view):
    fit = view.profile_fit                                # EAZ on this event: zero beyond -21 and 579 mm
    lo, hi = view.ax_profile.get_xlim()
    assert fit.tips_mm[0] < -12.4 and fit.tips_mm[1] > 508.4                 # outside the sensors and margins
    assert lo < fit.tips_mm[0] and fit.tips_mm[1] < hi and view.ax_front.get_xlim() == (lo, hi)
    markers = [line for line in view.ax_profile.lines if line.get_marker() == "v"]
    np.testing.assert_allclose(markers[0].get_xdata(), fit.tips_mm)
    # tips farther out than one array span beyond the sensors: a marker at the panel's edge, with the x
    for tips, texts in (((-721.0, 1279.0), ["tip at -721 mm", "tip at 1279 mm"]),
                        ((-900.0, -700.0), ["tips at -900, -700 mm"])):
        view.profile_fit = sprof.ProfileFit(model="eaz", valid=True, tips_mm=tips,
                                            params={"D": 100.0, "xc": sum(tips) / 2, "a": (tips[1] - tips[0]) / 2,
                                                    "lambda": 0.5})
        view.plot_fault()
        assert view.ax_profile.get_xlim() == (pytest.approx(-12.4), pytest.approx(508.4))
        assert [t.get_text() for t in view.ax_profile.texts] == texts
        assert not [line for line in view.ax_profile.lines if line.get_marker() == "v"]
        assert len([line for line in view.ax_profile.lines if line.get_marker() in ("<", ">")]) == len(texts)
    # the legend leaves the points visible, for every model
    for model in ("eaz", "eaz_asymmetric", "quadratic"):
        view.select_profile_model(model)
        view.canvas.draw()
        box = view.ax_profile.get_legend().get_window_extent()
        xs, ys = view.point_artists["profile"].get_data()
        assert not any(box.contains(px, py) for px, py in view.ax_profile.transData.transform(np.column_stack([xs, ys])))
    fit = view.profile_fit
    assert fit.model == "quadratic" and set(fit.params) == {"c0", "c1", "c2"} and fit.r2 > 0.999
    assert fit.peak_x_mm == pytest.approx(279.0, abs=5.0)
    view.select_profile_model("eaz_asymmetric")
    assert view.profile_fit.model == "eaz_asymmetric" and len(view.profile_fit.params) == 6
    view.weighted_var.set(True)
    view.on_profile_settings()
    assert view.profile_fit.weighted is True
    view.set_profile_use(0, False)
    view.set_profile_use(7, False)
    assert view.profile_fit.n == 6 and view.profile_use[0] is False
    assert "differs" in view.status_var.get() or "not saved" in view.status_var.get()
    # the choices carry to an event without a record; the sensors used do not
    view.weighted_var.set(False)
    view.select_profile_model("quadratic")
    set_front_mode(view, "unilateral")
    type_patch(view, "150", "400")
    add_slip_array(events(slip_app)[2], slips_um=[0.0, 0.0, 31.7, 82.9, 100.0, 82.9, 31.7, 0.0], noise_um=0.05)
    view.set_event(2)
    assert view.profile_fit.model == "quadratic" and view.front_fit.mode == "unilateral"
    assert view.profile_use == [True] * 8 and view.front_use_auto is True
    assert view.patch_x_mm == (150.0, 400.0) and view.patch_note_var.get() == "not saved to the experiment"
    # the quadratic profile is zero outside its tips: 100 um at 279 mm, zero 150 mm either side
    fit = view.profile_fit
    assert fit.valid and fit.tips_mm == (pytest.approx(129.0, abs=5.0), pytest.approx(429.0, abs=5.0))
    xx, yy = view.profile_curve.get_data()
    outside = (xx < fit.tips_mm[0]) | (xx > fit.tips_mm[1])
    assert outside.any() and np.all(yy[outside] == 0.0) and np.all(yy[~outside] >= 0.0)
    assert yy.max() == pytest.approx(100.0, abs=2.0)


def test_front_lines_and_sensors_left_out_by_their_event_time(slip_app, view):
    enter(view, view.on_window_edited, after_end_var="6")     # past the record's end: every after window is clipped
    assert all(cs.FLAG_AFTER_CLIPPED in s.flags for s in view.result.sensors)
    assert view.front_use_auto and view.front_use == [True] * 8 and view.front_fit.valid
    assert view.front_text_var.get().endswith("sensors: all, none with a doubtful t_i")
    assert "doubtful t_i" not in [t.get_text() for t in view.ax_front.get_legend().get_texts()]
    for mode in ("unilateral", "bilateral"):                  # a line and a speed per segment
        set_front_mode(view, mode)
        segments = view.front_fit.segments
        assert view.front_fit.valid and segments
        lines = [line for line in view.ax_front.lines if line.get_linestyle() == "-"]
        assert len(lines) == len(segments)
        for line, seg in zip(lines, segments):
            x = np.array([seg.x_from, seg.x_to])
            np.testing.assert_allclose(line.get_xdata(), x)
            np.testing.assert_allclose(line.get_ydata(), 1e3 * (seg.slope_s_per_mm * x + seg.intercept_s))
        assert [t.get_text() for t in view.ax_front.texts] == [
            "simultaneous" if math.isinf(s.velocity_m_s) else f"{abs(s.velocity_m_s):.3g} m/s" for s in segments]
    ev = events(slip_app)
    add_slip_array(ev[2], slips_um=[0.0, 40.0, 70.0, 90.0, 100.0, 90.0, 70.0, 0.0], noise_um=0.05)
    add_slip_array(ev[3], slips_um=[0.02] * 8, noise_um=0.05)
    add_slip_array(ev[0], slips_um=[0.02] * 4 + [100.0] + [0.02] * 3, noise_um=0.05)
    # the automatic choice follows the t_i flags until the user picks sensors
    view.set_event(2)
    doubtful = [any(f in T_I_FLAGS for f in s.flags) for s in view.result.sensors]
    assert doubtful[0] and doubtful[7]
    assert view.front_use == [not d for d in doubtful] and view.front_use_auto is True
    assert view.front_text_var.get().endswith(f"sensors: all but {sum(doubtful)} with a doubtful t_i")
    view.set_front_use(3, False)
    assert view.front_use_auto is False and view.front_use[3] is False
    set_front_mode(view, "unilateral")
    assert view.front_fit.mode == "unilateral" and view.front_fit.segments[0].n == sum(view.front_use)
    # too few sensors with a clean t_i
    view.set_event(3)
    assert view.front_use_auto and not any(view.front_use)
    for mode in ("bilateral", "unilateral"):
        set_front_mode(view, mode)
        assert view.front_text_var.get() == ("front: no sensor with a clean t_i; choose sensors under Sensors used\n"
                                             "sensors: all but 8 with a doubtful t_i")
    view.set_event(0)
    assert view.front_use == [False] * 4 + [True] + [False] * 3
    assert view.front_text_var.get() == ("front: only one sensor with a clean t_i; choose sensors under Sensors used\n"
                                         "sensors: all but 7 with a doubtful t_i")
    view.set_front_use(5, False)                          # a choice of the user's: no automatic line
    assert not view.front_use_auto and "clean t_i" not in view.front_text_var.get()
    # a front that crosses the array within one sample: the fit gets the sampling interval
    add_slip_array(ev[1], front_speed_m_s=2000.0, seed=5)
    view.front_mode_combobox.set("bilateral")
    view.set_event(1)
    segments = view.front_fit.segments
    assert segments and all(not s.resolved or math.isinf(s.velocity_m_s) for s in segments)
    dt = cs.sampling_interval(view.time)
    assert all(s.min_speed_m_s == pytest.approx((s.x_to - s.x_from) * 1e-3 / dt) for s in segments)
    assert "faster than" in view.front_text_var.get() or "simultaneous" in view.front_text_var.get()


def test_front_texts_name_the_apex_and_unresolved_speeds():
    from labquake_explorer.ui.views.slip_distribution_view import apex_text, segment_speed_label, segment_text

    def seg(x0, x1, slope, resolved=True, min_speed=float("nan")):
        return sprof.FrontSegment(x0, x1, 3, slope, 0.0, 1e-3 / slope if slope else math.inf, 0.9, resolved, min_speed)

    out = sprof.FrontFit("bilateral", True, "", [seg(31, 250, -2e-5), seg(250, 465, 2e-5)], 250.0, 0.0213)
    assert apex_text(out) == "fronts start at x = 250 mm, 21.30 ms after the event time"
    into = sprof.FrontFit("bilateral", True, "", [seg(31, 250, 2e-5), seg(250, 465, -2e-5)], 250.0, 0.0213)
    assert apex_text(into).startswith("fronts meet at x = 250 mm")
    assert apex_text(sprof.FrontFit("bilateral", True, "", [seg(31, 465, 0.0)], float("nan"), float("nan"))) is None
    assert apex_text(sprof.FrontFit("unilateral", True, "", [seg(31, 465, 2e-5)])) is None
    fast = seg(31, 279, 1e-6, resolved=False, min_speed=496.0)
    assert segment_speed_label(fast) == "> 496 m/s" and "faster than 496 m/s" in segment_text(fast)
    assert segment_speed_label(seg(31, 279, 0.0)) == "simultaneous"
    assert segment_text(seg(31, 279, 2e-5)) == "x 31 to 279 mm: 50 m/s towards +x (3 sensors, R^2 = 0.900)"


def test_unresolved_profile_parameters_are_shown_as_ranges(slip_app):
    ev = events(slip_app)
    # real t0207 run 1 event 10: no sensor inside the elliptical part, so lambda is free
    add_slip_array(ev[2], slips_um=[4.31, 0.29, 24.13, 48.47, 89.23, 64.96, 35.63, 19.4], noise_um=0.2, seed=3)
    # real run 1 event 7: three sensors slip, the rupture edges are only bracketed
    add_slip_array(ev[3], slips_um=[-0.62, -1.02, -0.31, 1.32, 3.11, 15.38, 4.8, 0.82], noise_um=0.2, seed=4)
    with opened(slip_app, 2) as v:
        assert v.profile_fit.valid and "lambda" in v.profile_fit.unresolved and "D" in v.profile_fit.ranges
        text = v.profile_text_var.get()
        assert "not resolved by the sensors:" in text and "lambda 0 to " in text and "fit equally" in text
        v.set_event(3)
        assert v.profile_fit.valid and "tips" in v.profile_fit.unresolved
        text = v.profile_text_var.get()
        assert "-x edge 217 to 279 mm" in text and "+x edge 403 to 465 mm" in text and "rupture from" not in text
        assert "rupture edge, not resolved" in [line.get_label() for line in v.ax_profile.lines]


def test_typed_profile_parameters_are_held_bounded_and_saved(slip_app, view):
    # a value typed on another tab is not fitted until that tab is chosen
    type_param(view, "quadratic", "c2", "-0.002")
    assert view.profile_model() == "eaz" and view.profile_fit.fixed == []
    view.select_profile_model("quadratic")
    assert view.profile_fit.fixed == ["c2"] and view.profile_fit.params["c2"] == -0.002
    view.select_profile_model("eaz")
    v = view.param_vars["eaz"]
    assert float(v["lambda"]["value"].get()) == pytest.approx(view.profile_fit.params["lambda"], rel=1e-5)
    type_param(view, "eaz", "lambda", "1.5")
    assert not view.profile_fit.valid and "lambda must be" in view.profile_text_var.get()
    type_param(view, "eaz", "lambda", "")                   # cleared: fitted again
    assert view.profile_fit.valid and v["lambda"]["fit"].get() is True
    type_param(view, "eaz", "lambda", "0.5")                # typed: held, the rest fitted
    fit = view.profile_fit
    assert v["lambda"]["fit"].get() is False and fit.fixed == ["lambda"] and fit.params["lambda"] == 0.5
    assert "lambda = 0.5 (held)" in view.profile_text_var.get() and fit.params["a"] > 300
    v["a"]["max"].set("abc")
    assert not view.save() and "a max" in view.status_var.get()
    type_param(view, "eaz", "a", "300", field="max")        # a bound on a free parameter
    assert view.profile_fit.params["a"] == pytest.approx(300.0) and "a" in view.profile_fit.at_bounds
    v["lambda"]["fit"].set(True)                           # ticked again: fitted
    view.on_param_fit_toggled("eaz", "lambda")
    assert view.profile_fit.fixed == []
    v["xc"]["fit"].set(False)                             # unticked: held at the value shown
    view.on_param_fit_toggled("eaz", "xc")
    assert view.profile_fit.fixed == ["xc"] and view.profile_fit.params["xc"] == pytest.approx(float(v["xc"]["value"].get()))
    assert view.save()
    rec = view.event["slip_analysis"]["profile"]
    assert rec["fixed"] == ["xc"] and rec["settings"]["eaz"]["a"]["max"] == 300.0
    assert rec["settings"]["eaz"]["xc"]["fit"] is False and math.isnan(rec["settings"]["eaz"]["D"]["value"])
    view.reset_profile_params("eaz")
    assert view.profile_fit.fixed == [] and v["a"]["max"].get() == ""
    assert "differs from the saved record" in view.status_var.get()
    view.on_close()
    with opened(slip_app) as v2:
        assert v2.param_vars["eaz"]["xc"]["fit"].get() is False and v2.param_vars["eaz"]["a"]["max"].get() == "300"
        assert v2.profile_fit.fixed == ["xc"] and "matches the saved record" in v2.status_var.get()


# ------------------------------------------------------------ patch and fault ends
def test_patch_edges_are_saved_at_the_experiment_level(slip_app, view):
    assert (view.patch_lo_var.get(), view.patch_hi_var.get()) == ("", "")
    type_patch(view, "400", "150", apply=False)           # typed, not applied: the save is refused
    assert not view.save() and "patch" in view.status_var.get() and "slip_analysis" not in view.event
    type_patch(view, "150", "400")
    assert view.patch_x_mm == (150.0, 400.0) and len(view.patch_artists) == 6      # span and edges on both panels
    assert "not saved to the experiment" in view.patch_note_var.get()
    assert view.save_patch()
    assert slip_app.data_manager.get_data("analysis/fault/patch_x_mm") == [150.0, 400.0]
    assert view.patch_note_var.get() == "saved in the experiment"
    enter(view, view.on_patch_edited, patch_hi_var="100")
    assert "patch" in view.status_var.get() and view.patch_x_mm == (150.0, 400.0)
    assert not view.save_patch()
    view.on_close()
    with opened(slip_app, 0) as v2:                           # every event of the experiment
        assert (v2.patch_lo_var.get(), v2.patch_hi_var.get()) == ("150", "400")


def test_typed_patch_edges_stay_across_events_until_saved(slip_app, view):
    dm = slip_app.data_manager
    dm.data["analysis"] = {"fault": {"patch_x_mm": [150.0, 400.0]}}
    view.set_event(2)
    assert view.patch_x_mm == (150.0, 400.0) and view.patch_note_var.get() == "saved in the experiment"
    view.patch_lo_var.set("120")                          # typed, still waiting for the keys to settle
    view.schedule(view.on_patch_edited)
    view.set_event(EVENT)                                 # applied, then kept over the experiment's edges
    assert view.patch_x_mm == (120.0, 400.0) and view.patch_lo_var.get() == "120"
    assert view.patch_note_var.get() == "not saved to the experiment"
    view.set_event(2)
    assert view.patch_x_mm == (120.0, 400.0)
    assert view.save_patch() and dm.get_data("analysis/fault/patch_x_mm") == [120.0, 400.0]
    dm.data["analysis"]["fault"]["patch_x_mm"] = [130.0, 390.0]       # changed elsewhere: no typed edges hold it back now
    view.set_event(EVENT)
    assert view.patch_x_mm == (130.0, 390.0) and view.patch_note_var.get() == "saved in the experiment"


def test_record_edges_show_until_the_experiment_has_its_own(slip_app):
    for event_idx, patch in ((1, ("120", "380")), (2, ("130", "390")), (3, None)):
        with opened(slip_app, event_idx) as v:
            if patch:
                type_patch(v, *patch)
            assert v.save()
    with opened(slip_app, 1) as v:                        # each event shows its record's edges
        assert v.patch_x_mm == (120.0, 380.0) and "matches the saved record" in v.status_var.get()
        v.set_event(2)
        assert v.patch_x_mm == (130.0, 390.0) and "matches the saved record" in v.status_var.get()
        v.set_event(1)
        assert v.patch_x_mm == (120.0, 380.0)
    slip_app.data_manager.data["analysis"] = {"fault": {"patch_x_mm": [150.0, 400.0]}}     # e.g. set later by a notebook
    for event_idx, record_patch in ((3, [math.nan, math.nan]), (1, [120.0, 380.0])):
        with opened(slip_app, event_idx) as v:            # the experiment's edges win and do not make it differ
            np.testing.assert_array_equal(v.event["slip_analysis"]["patch_x_mm"], record_patch)
            assert v.patch_x_mm == (150.0, 400.0) and v.patch_note_var.get() == "saved in the experiment"
            assert "matches the saved record" in v.status_var.get()
    with opened(slip_app, 1) as v:
        v.select_profile_model("quadratic")               # the fit settings still count
        assert "differs from the saved record" in v.status_var.get()


def test_save_to_experiment_keeps_the_edges_at_full_precision(slip_app):
    dm = slip_app.data_manager
    dm.data["analysis"] = {"fault": {"patch_x_mm": [25.4 * 6, 400.0]}}     # 6 in as a notebook converts it: 152.39999999999998
    with opened(slip_app, 2) as v:
        assert v.patch_lo_var.get() == "152.4"
        assert v.save_patch()                                # nothing typed: the edges stay as they were
        assert dm.get_data("analysis/fault/patch_x_mm") == [25.4 * 6, 400.0]
        enter(v, v.on_patch_edited, patch_hi_var="450")       # one edge typed: the other keeps all its digits
        assert v.patch_x_mm == (25.4 * 6, 450.0)
        assert v.save_patch() and dm.get_data("analysis/fault/patch_x_mm") == [25.4 * 6, 450.0]


def test_patch_saved_in_one_window_reaches_the_others(slip_app):
    views = [SlipDistributionView(slip_app, 0, e) for e in (1, 2, 3, 0)]
    v1, v2, v3, v4 = views
    try:
        type_patch(v3, "100", "300")                         # edges of its own, not saved
        type_patch(v4, "150", "400")                         # the same edges as are about to be saved
        type_patch(v1, "150", "400")
        assert v1.save_patch()
        assert v2.patch_x_mm == (150.0, 400.0) and (v2.patch_lo_var.get(), v2.patch_hi_var.get()) == ("150", "400")
        assert v2.patch_note_var.get() == "saved in the experiment" and len(v2.patch_artists) == 6
        assert v3.patch_x_mm == (100.0, 300.0) and v3.patch_note_var.get() == "not saved to the experiment"
        assert v4.patch_note_var.get() == "saved in the experiment"
        assert v3.save_patch()                               # the windows showing the old edges follow
        assert v1.patch_x_mm == v2.patch_x_mm == v4.patch_x_mm == (100.0, 300.0)
        assert {v.patch_note_var.get() for v in views} == {"saved in the experiment"}
        assert v2.save() and v2.event["slip_analysis"]["patch_x_mm"] == [100.0, 300.0]
    finally:
        for v in views:
            v.on_close()


@pytest.mark.parametrize("value, patch, note", [
    ([400.0, 150.0], (150.0, 400.0), "saved in the experiment"),
    ([150.0, float("nan")], None, "experiment patch edges unusable: [150.0, nan]"),
    ([150.0], None, "experiment patch edges unusable: [150.0]"),
    ([150.0, 150.0], None, "experiment patch edges unusable: [150.0, 150.0]"),
])
def test_experiment_patch_in_either_order_or_unusable(slip_app, value, patch, note):
    slip_app.data_manager.data["analysis"] = {"fault": {"patch_x_mm": value}}
    with opened(slip_app) as v:
        assert v.patch_x_mm == patch and v.patch_note_var.get() == note
        assert len(v.patch_artists) == (6 if patch else 0)
        type_patch(v, "120", "380")
        assert v.patch_note_var.get() == "not saved to the experiment" + ("" if patch else f"; {note}")
        assert v.save_patch() and v.patch_note_var.get() == "saved in the experiment"


@pytest.mark.parametrize("suffix", [".h5", ".npz"])
def test_experiment_patch_from_preprocessing_survives_a_file_round_trip(slip_app, tmp_path, app_from_file, suffix):
    from labquake_explorer.data.data_manager import DataManager
    from labquake_explorer.preprocessing.build import experiment
    data = slip_app.data_manager.data
    dm = DataManager()
    dm.data = experiment(data["name"], data["runs"], analysis={"fault": {"patch_x_mm": [150.0, 400.0]}})
    path = tmp_path / f"exp{suffix}"
    dm.save_file(path)
    with opened(app_from_file(path)) as v:
        assert v.patch_x_mm == (150.0, 400.0) and v.patch_note_var.get() == "saved in the experiment"
        for ax in (v.ax_profile, v.ax_front):
            span, *edges = [a for a in v.patch_artists if a.axes is ax]
            assert (span.get_x(), span.get_x() + span.get_width()) == (150.0, 400.0)
            assert [tuple(e.get_xdata()) for e in edges] == [(150.0, 150.0), (400.0, 400.0)]


def test_the_profile_is_cut_at_the_fault_ends(slip_app, view):
    enter(view, view.on_ends_edited, ends_lo_var="100", ends_hi_var="460")
    assert view.fault_ends == (100.0, 460.0) and view.ends_note_var.get() == "fault ends not saved to the experiment"
    fit = view.profile_fit
    xx, _ = view.profile_curve.get_data()
    assert xx.min() == pytest.approx(100.0) and xx.max() == pytest.approx(460.0)
    reached = view.end_slips()
    assert set(reached) == {"minus", "plus"}            # the synthetic rupture runs past both ends
    assert reached["plus"][1] == pytest.approx(float(fit.evaluate([460.0])[0]))
    text = view.profile_text_var.get()
    assert "the -x fault end (" in text and "the +x fault end (" in text
    assert all(100 <= t <= 460 for t in view.pinned_tips())
    assert "fault end" in [line.get_label() for line in view.ax_profile.lines]
    assert view.save()
    rec = view.event["slip_analysis"]
    assert rec["fault_ends_x_mm"] == [100.0, 460.0] and rec["profile"]["end_slip_um"][1] == pytest.approx(reached["plus"][1])
    assert view.save_fault() and slip_app.data_manager.get_data("analysis/fault/ends_x_mm") == [100.0, 460.0]
    assert view.ends_note_var.get() == "fault ends saved in the experiment"
    view.on_close()
    with opened(slip_app, 0) as v2:                           # another event opens with the experiment's ends
        assert v2.fault_ends == (100.0, 460.0) and v2.ends_lo_var.get() == "100"


def test_saving_keeps_a_text_fault_description(slip_app, view):
    dm = slip_app.data_manager
    dm.data["fault"] = "one-dimensional PMMA fault, 500 mm long"     # how the T0213 notebook describes the sample
    view.ends_lo_var.set("0")
    view.ends_hi_var.set("500")
    type_patch(view, "190", "315", apply=False)
    enter(view, view.on_ends_edited)
    enter(view, view.on_patch_edited)
    assert view.save_fault() and dm.data["fault"] == "one-dimensional PMMA fault, 500 mm long"
    assert dm.data["analysis"]["fault"] == {"ends_x_mm": [0.0, 500.0], "patch_x_mm": [190.0, 315.0]}
    assert "patch_x_mm" not in dm.data and "fault_ends_x_mm" not in dm.data


@pytest.mark.parametrize("layout", ["fault dict", "top level"])
def test_edges_kept_where_earlier_files_put_them_are_read_and_moved(slip_app, layout):
    dm = slip_app.data_manager
    if layout == "fault dict":
        dm.data["fault"] = {"patch_x_mm": [150.0, 400.0], "ends_x_mm": [0.0, 500.0]}
    else:
        dm.data["patch_x_mm"], dm.data["fault_ends_x_mm"] = [150.0, 400.0], [0.0, 500.0]
    with opened(slip_app) as v:
        assert v.patch_x_mm == (150.0, 400.0) and v.fault_ends == (0.0, 500.0)
        enter(v, v.on_patch_edited, patch_hi_var="410")
        assert v.save_patch() and dm.data["analysis"]["fault"]["patch_x_mm"] == [150.0, 410.0]
        assert "patch_x_mm" not in dm.data and dm.data.get("fault") != {"patch_x_mm": [150.0, 400.0], "ends_x_mm": [0.0, 500.0]}
        assert v.save_ends() and "fault" not in dm.data and "fault_ends_x_mm" not in dm.data
        assert dm.data["analysis"]["fault"] == {"patch_x_mm": [150.0, 410.0], "ends_x_mm": [0.0, 500.0]}


# ------------------------------------------------------------ fault panels
def test_fault_panel_limits_are_kept_until_another_event(slip_app, view):
    view.canvas.draw()
    ax = view.ax_front
    view.toolbar.zoom()                                   # the toolbar's zoom to a rectangle on the front panel
    for name, xd, yd in (("button_press_event", 200.0, 25.0), ("motion_notify_event", 360.0, 20.0),
                         ("button_release_event", 360.0, 20.0)):
        press(view, ax, xd, yd, name=name)
    view.toolbar.zoom()
    zoom = (ax.get_xlim(), ax.get_ylim())                 # to within a pixel: mouse events carry whole pixels
    assert zoom == ((pytest.approx(200.0, abs=3.0), pytest.approx(360.0, abs=3.0)),
                    (pytest.approx(20.0, abs=0.03), pytest.approx(25.0, abs=0.03)))
    press(view, ax, *point(view, "front", 4))
    assert view.sensor == 4 and (ax.get_xlim(), ax.get_ylim()) == zoom
    assert view.ax_profile.get_xlim() == zoom[0]          # the two panels share x
    view.step_sensor(+1)
    assert (ax.get_xlim(), ax.get_ylim()) == zoom
    enter(view, view.on_window_edited, before_start_var="-3")  # a recompute
    assert (ax.get_xlim(), ax.get_ylim()) == zoom
    view.ax_profile.set_ylim(50.0, 110.0)                 # set by hand on the other panel
    view.refit()
    assert view.ax_profile.get_ylim() == (50.0, 110.0) and (ax.get_xlim(), ax.get_ylim()) == zoom
    # another event, with a patch drawn: both panels fit their data again, from about 100 um and 20 ms
    # to almost nothing
    slip_app.data_manager.data["analysis"] = {"fault": {"patch_x_mm": [150.0, 400.0]}}
    ev = events(slip_app)
    add_slip_array(ev[2], slips_um=[0.02] * 8, noise_um=0.05)
    view.set_event(2)
    assert ax.get_xlim() != zoom[0] and ax.get_ylim() != zoom[1] and view.ax_profile.get_ylim() != (50.0, 110.0)
    lo, hi = view.ax_profile.get_ylim()
    assert len(view.patch_artists) == 6 and hi - lo < 1.0
    t = np.array([1e3 * s.t_shift_s for s in view.result.sensors])
    pad = 0.05 * (t.max() - t.min())
    np.testing.assert_allclose(ax.get_ylim(), (t.min() - pad, t.max() + pad))
    # Home after another event shows that event
    add_slip_array(ev[3], slips_um=[0.0, 2.0, 5.0, 9.0, 10.0, 8.0, 4.0, 1.0], t0_s=0.3, seed=7)
    view.canvas.draw()
    view.toolbar.push_current()                           # a toolbar zoom puts every panel's limits in its history
    view.ax_trace.set_xlim(-0.2, 0.3)
    view.set_event(3)
    view.toolbar._buttons["Home"].invoke()
    for name, panel in (("profile", view.ax_profile), ("front", view.ax_front)):
        xs, ys = view.point_artists[name].get_data()
        (x0, x1), (y0, y1) = panel.get_xlim(), panel.get_ylim()
        assert np.all((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)), name
    view.ax_front.set_ylim(250, 260)                      # a zoom on a fault panel, then Home
    view.toolbar.home()
    assert view.ax_front.get_ylim() == tuple(view._fault_auto[2])


# ------------------------------------------------------ status, t_i and filter tabs
def test_status_names_an_entry_that_was_not_applied(slip_app, view):
    assert view.save()
    enter(view, view.on_settings_edited, smoothing_var="-1")
    assert view.status_var.get() == "not recomputed: the moving average window must be >= 0"
    for change in (lambda: (view.weighted_var.set(True), view.on_profile_settings()),
                   lambda: (view.weighted_var.set(False), view.on_profile_settings()),
                   lambda: view.step_sensor(+1)):
        change()
        status = view.status_var.get()
        assert status.endswith(" | entry not applied: the moving average window must be >= 0") and "matches" not in status
    enter(view, view.on_settings_edited, smoothing_var="10")  # the applied value again
    assert view.status_var.get().endswith(" | matches the saved record")
    type_patch(view, "400", "150")
    set_front_mode(view, "unilateral")
    assert view.status_var.get().endswith(" | entry not applied: the patch must have from < to")


def test_rate_filter_tabs_are_applied_and_saved(slip_app, view):
    view.select_rate_filter("savgol")
    view.on_filter_choice()
    assert view.result.rate_filter == "savgol" and view.result.rate_smoothing_s == pytest.approx(0.020)
    enter(view, view.on_settings_edited, savgol_window_var="30", savgol_order_var="3")
    assert view.result.rate_smoothing_s == pytest.approx(0.030) and view.result.rate_savgol_order == 3
    assert view.save()
    rec = view.event["slip_analysis"]
    assert rec["rate_filter"] == "savgol" and rec["rate_smoothing_s"] == pytest.approx(0.030)
    assert rec["rate_filter_tabs"]["boxcar_window_s"] == pytest.approx(0.010)
    view.select_rate_filter("lowpass")
    view.on_filter_choice()
    assert "differs from the saved record" in view.status_var.get()
    enter(view, view.on_settings_edited, cutoff_var="5000")   # above half the sampling rate
    assert "half the sampling rate" in view.status_var.get()
    view.cutoff_var.set("50")
    view.lowpass_type_combobox.set("Bessel")
    view.on_filter_choice()
    assert view.result.rate_lowpass_type == "bessel"
    enter(view, view.on_settings_edited, lowpass_order_var="2")
    assert view.result.rate_lowpass_order == 2
    enter(view, view.on_settings_edited, lowpass_order_var="2.5")
    assert "whole number" in view.status_var.get()
    view.lowpass_order_var.set("2")
    view.select_rate_filter("savgol")
    enter(view, view.on_settings_edited, savgol_window_var="1")  # 1 ms is 1 sample at 1 kHz: too short for the order
    assert "Savitzky-Golay window must span" in view.status_var.get()
    view.on_close()
    with opened(slip_app) as v2:
        assert v2.rate_filter() == "savgol" and v2.savgol_window_var.get() == "30"
        assert v2.savgol_order_var.get() == "3" and v2.smoothing_var.get() == "10"
        assert "matches the saved record" in v2.status_var.get()


def test_event_time_tabs(slip_app, view):
    truth = slip_app.slip_truth[EVENT]
    assert view.t_i_method() == "max_rate"
    view.set_event_time(2, 0.05)                          # a sensor set by hand stays so on another pick tab
    view.select_t_i_method("onset")
    assert view.methods[2] == "manual" and all(m == "onset" for i, m in enumerate(view.methods) if i != 2)
    onset = np.array([s.t_shift_s for s in view.result.sensors])
    picks = np.asarray(truth["t_i"]) - view.t_event
    moving = np.array(truth["slips_um"]) > 5.0
    moving[2] = False
    assert np.all(onset[moving] <= picks[moving] + 0.002)  # the onset comes before the rate peak
    enter(view, view.on_settings_edited, onset_var="50")
    assert view.result.onset_fraction == 0.5
    view.select_t_i_method("step")
    assert view.result.sensors[0].method == "step" and view.method_combobox.get() == "slip step"
    view.select_t_i_method("shared")                      # every sensor at the peak of the mean rate
    ts = {s.t_shift_s for s in view.result.sensors}
    assert len(ts) == 1 and ts.pop() == pytest.approx(view.result.t_ref - view.t_event)
    assert str(view.method_combobox.cget("state")) == "disabled"
    assert view.front_fit.segments and all(math.isinf(seg.velocity_m_s) for seg in view.front_fit.segments)
    view.begin_drag("t_i", view.result.sensors[0].t_shift_s)  # dragging moves the one t_i of every sensor
    view.drag_to(0.08)
    view.end_drag()
    assert view.shared_source_var.get() == "set"
    assert all(s.t_shift_s == pytest.approx(0.08) for s in view.result.sensors)
    assert "at one t_i, +80.0 ms (set)" in view.status_var.get()
    assert view.save()
    rec = view.event["slip_analysis"]
    assert rec["shared_shift_s"] == pytest.approx(0.08) and set(rec["time_method"]) == {"shared"}
    view.repick_all()                                     # back to the peak of the mean rate
    assert view.shared_source_var.get() == "reference" and math.isnan(view.result.shared_shift_s)
    view.on_close()
    with opened(slip_app) as v2:
        assert v2.t_i_method() == "shared" and v2.shared_source_var.get() == "set" and v2.shared_shift_var.get() == "80"
        assert v2.onset_var.get() == "50" and "matches the saved record" in v2.status_var.get()


# ------------------------------------------------------------ 2 MHz waveform field
def block_2mhz(event, te, x, um_per_v):
    """A 2 MHz record with eight eddy probes (a bell-shaped step at the event) and four PZTs (a burst 0.25 ms after)."""
    from labquake_explorer.data.channels import channel_array
    fs = 2e6
    trig = te - 0.02
    t = trig - 0.02 + np.arange(int(0.06 * fs)) / fs
    names = [f"eddy_{j}" for j in range(1, 9)] + [f"pzt_{j}" for j in range(1, 5)]
    rows = []
    for xj in x:
        amp = 12.0 * np.exp(-((xj - 250.0) / 120.0) ** 2)
        rows.append(amp / um_per_v * 0.5 * (1 + np.tanh((t - te - 2e-4) / 1.5e-4))
                    + 1e-4 * np.random.default_rng(int(xj)).normal(size=t.size))
    for _ in range(4):
        rows.append(0.3 * np.exp(-((t - te - 2.5e-4) / 8e-5) ** 2) * np.sin(2 * np.pi * 40e3 * t))
    event["waveform"] = {"elsys": channel_array(np.array(rows, dtype=np.float32), names, unit="V",
                                                positions={"x": list(x) + [100, 200, 300, 400]}, time=t,
                                                sample_rate=fs, trigger_time=trig, block=3)}
    return t


def test_waveform_field_and_burst_tab(slip_app):
    from labquake_explorer.data.channels import channel_array
    from labquake_explorer.ui.views.slip_distribution_view import slip_sources, waveform_slip_array
    run = slip_app.data_manager.get_data("runs/[0]")
    event = run["events"][1]
    te = float(event["event_time"])
    x = np.array([31, 93, 155, 217, 279, 341, 403, 465], float)
    um_per_v = -95.58
    expected = [12.0 * np.exp(-((xj - 250.0) / 120.0) ** 2) for xj in x]
    block_2mhz(event, te, x, um_per_v)
    assert slip_fields(event, run) == ["slip"]                       # the run has no slip array: no source channels
    run["slip"] = channel_array(np.zeros((8, 10)), [f"slip_{j}" for j in range(1, 9)], unit="um",
                                positions={"x": list(x)}, source=[f"eddy_{j}" for j in range(1, 9)],
                                slope_mm_per_v=um_per_v / 1e3)
    assert set(slip_sources(run)) == {f"eddy_{j}" for j in range(1, 9)}
    assert slip_sources(run)["eddy_3"][1:] == ("slip_3", 155.0, um_per_v)
    assert slip_fields(event, run) == ["slip", "waveform/elsys"]
    arr = waveform_slip_array(event, run, "waveform/elsys")
    assert arr["data"].shape == (8, 120000) and arr["channels"][0] == "slip_1" and arr["positions"]["x"] == list(x)
    assert arr["pzt"].shape == (4, 120000) and arr["unit"] == "um"
    assert abs(float(np.mean(arr["data"][0][:1000]))) < 0.05          # zeroed on its first samples
    with pytest.raises(ValueError):
        waveform_slip_array(event, run, "waveform/nope")
    with opened(slip_app, 1) as view:
        assert list(view.field_combobox["values"]) == ["slip", "waveform/elsys"] and view.field == "slip"
        assert view.pzt is None
        assert view.save()
        view.field_combobox.set("waveform/elsys")
        view.on_field_selected()
        assert view.field == "waveform/elsys" and view.time.size == 120000 and view.pzt is not None
        assert "the event's record is for slip, Save replaces it" in view.status_var.get()
        assert view.start_s == pytest.approx(-0.04, abs=1e-3) and view.end_s == pytest.approx(0.02, abs=1e-3)
        assert view.result is not None and view.result.window_mode == "lines"
        view.select_t_i_method(cs.METHOD_BURST)
        r = view.result
        assert r.window_mode == "levels" and set(view.methods) == {cs.METHOD_BURST}
        assert r.t_burst - te == pytest.approx(2.5e-4, abs=2e-5) and r.burst_source == "pzt_envelope"
        assert [s.coseismic_slip for s in r.sensors] == pytest.approx(expected, abs=0.1)
        assert view.profile_fit.valid and view.profile_fit.peak == pytest.approx(12.0, rel=0.05)
        # the burst entries reach the analysis; bad ones are reported
        view.burst_halfwidth_var.set("2")
        view.on_settings_edited()
        view.flush_pending()
        s0 = view.result.sensors[0]
        assert view.result.burst_halfwidth_s == 2e-3 and s0.fit_after_s[0] == pytest.approx(s0.t_shift_s + 2e-3)
        view.level_width_var.set("0")
        view.on_settings_edited()
        view.flush_pending()
        assert "level width" in view.status_var.get()
        view.level_width_var.set("0.1")
        view.on_settings_edited()
        view.flush_pending()
        view.burst_source_var.set("peak_rate")
        view.on_burst_source()
        assert view.result.burst_source == "peak_rate" and view.result.t_burst == pytest.approx(view.result.t_ref)
        view.burst_source_var.set("pzt_envelope")
        view.on_burst_source()
        assert view.result.t_burst - te == pytest.approx(2.5e-4, abs=2e-5)
        assert view.save()
        rec = event["slip_analysis"]
        assert rec["field"] == "waveform/elsys" and rec["window_mode"] == "levels" and rec["time_method"][0] == "burst"
        assert rec["t_burst_shift_s"] == pytest.approx(2.5e-4, abs=2e-5) and rec["burst_halfwidth_s"] == 2e-3
        assert rec["coseismic_slip_um"] == pytest.approx(expected, abs=0.1)
    with opened(slip_app, 1) as v2:
        assert v2.field == "waveform/elsys" and v2.t_i_method() == cs.METHOD_BURST
        assert v2.burst_halfwidth_var.get() == "2" and v2.burst_source_var.get() == "pzt_envelope"
        assert v2.result.window_mode == "levels" and "matches the save" in v2.status_var.get()
        v2.field_combobox.set("slip")                         # back to the 2 kHz field: lines, no PZT
        v2.on_field_selected()
        assert v2.field == "slip" and v2.pzt is None
        v2.select_t_i_method(cs.METHOD_MAX_RATE)
        assert v2.result.window_mode == "lines"
