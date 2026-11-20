"""SlopeAnalyzerView, IndexPickerView, PointsSelectorView and SimplePlotView,
plus the main-window actions that open them."""
import numpy as np
import pytest

from labquake_explorer.ui.actions import actions_for
from labquake_explorer.ui.views import (
    EventPickerView, IndexPickerView, PointsSelectorView, SimplePlotView, SlopeAnalyzerView,
)


def run_menu_action(app, path, label):
    """Select ``path`` in the tree and run the context-menu action ``label`` on it."""
    item = app.find_item(path)
    assert item is not None, path
    app.data_tree.selection_set(item)
    ctx = app.context_at(item)
    action = [a for a in actions_for(ctx.kind) if a.label == label][0]
    app.run_action(action, ctx)
    return ctx


def opened(app, cls):
    return [w for w in app.child_windows if isinstance(w, cls)]


# ---------------------------------------------------------------- SlopeAnalyzerView
def test_slope_analyzer_on_a_run(app):
    truth = app.truth[0]
    view = SlopeAnalyzerView(app, item_y="runs/[0]/shear_stress")
    assert view.title() == "Slope Analyzer"
    assert view in app.child_windows
    names = list(view.data_y_combo["values"])
    assert names == list(view.data_x_combo["values"])
    for expected in ("time", "shear_stress", "displacement", "LP_displacement",
                     "LP_velocity", "normal_stress", "friction"):
        assert expected in names
    assert "name" not in names
    assert view.data_y_combo.get() == "shear_stress"
    # X defaults to the fault slip (stiffness = d(stress)/d(slip))
    assert view.data_x_combo.get() == "displacement"
    assert view.base_path == "runs/[0]"
    assert [m.get_label() for m in view.markers] == ["0", "1"]
    assert float(view.slope_textbox.get()) == view.slope and np.isfinite(view.slope)
    np.testing.assert_array_equal(view.data_x, truth.displacement)

    # loading rate: stress against time between the first two events
    view.data_x_combo.set("time")
    view.data_x_selected(None)
    assert view.item_x == "time" and view.ax.get_xlabel() == "time"
    i0, i1 = truth.event_indices[0] + 200, truth.event_indices[1] - 200
    view.picked_idx = [i0, i1]
    view.plot_picked_points()
    assert float(view.slope_textbox.get()) == pytest.approx(truth.loading_rate, rel=1e-6)
    xs, ys = view.slope_line.get_data()
    assert list(xs) == [truth.time[i0], truth.time[i1]]
    assert list(ys) == [truth.shear_stress[i0], truth.shear_stress[i1]]
    # a drag keeps the line and the readout in sync
    from matplotlib.backend_bases import MouseEvent
    target = truth.event_indices[1] - 100
    view.picker.current_artist, view.picker.currently_dragging = view.markers[1], True
    px, py = view.ax.transData.transform((truth.time[target], truth.shear_stress[target]))
    view.picker.on_motion(MouseEvent("motion_notify_event", view.canvas, px, py))
    view.picker.on_release(None)
    assert abs(view.picked_idx[1] - target) <= 50
    assert float(view.slope_textbox.get()) == pytest.approx(truth.loading_rate, rel=1e-6)
    assert view.slope_line.get_data()[0][1] == truth.time[view.picked_idx[1]]
    view.on_close()
    assert view not in app.child_windows

    view = SlopeAnalyzerView(app, item_y="runs/[0]/shear_stress", item_x="LP_displacement")
    assert view.data_x_combo.get() == "LP_displacement"
    view.picked_idx = [i0, i1]
    view.plot_picked_points()
    assert float(view.slope_textbox.get()) == pytest.approx(truth.stiffness_lp, rel=1e-6)
    view.on_close()


def test_slope_analyzer_on_an_event_switches_y(app):
    view = SlopeAnalyzerView(app, item_y="runs/[0]/events/[1]/shear_stress")
    assert view.base_path == "runs/[0]/events/[1]"
    names = list(view.data_y_combo["values"])
    assert "shear_stress" in names and "time" in names
    assert "event_time" not in names  # scalar, not an array
    n = len(app.data_manager.get_data("runs/[0]/events/[1]/shear_stress"))
    assert view.picked_idx == [int(n / 3), int(n / 3 * 2)]
    view.data_y_combo.set("displacement")
    view.data_y_selected(None)
    assert view.item_y == "displacement" and view.ax.get_ylabel() == "displacement"
    assert np.isfinite(float(view.slope_textbox.get()))
    view.on_close()


# ------------------------------------------------------------------ IndexPickerView
def test_index_picker(app):
    view = IndexPickerView(app, item_y="runs/[0]/events/[1]/shear_stress")
    assert view.title() == "Index Picker"
    assert view.data_y_combo.get() == "shear_stress"
    n = len(app.data_manager.get_data("runs/[0]/events/[1]/shear_stress"))
    assert view.picked_idx == [int(n / 3), int(n / 3 * 2)]
    assert view.index_textbox.get() == str(view.picked_idx)
    assert view.clipboard_get() == str(view.picked_idx)
    view.picked_idx = [10, 20]
    view.plot_picked_points()
    assert view.index_textbox.get() == "[10, 20]"
    assert [m.center for m in view.markers] == [(view.data_x[10], view.data_y[10]),
                                                (view.data_x[20], view.data_y[20])]
    assert view.picker.nearest_index(view.data_x[15], view.data_y[15]) == 15
    view.on_close()
    assert view not in app.child_windows

    # a drag onto a NaN gap snaps to a finite sample
    app.data_manager.get_data("runs/[0]/events/[1]")["shear_stress"][100:200] = np.nan
    view = IndexPickerView(app, item_y="runs/[0]/events/[1]/shear_stress")
    idx = view.picker.nearest_index(view.data_x[150], float(np.nanmean(view.data_y)))
    assert not 100 <= idx < 200 and np.isfinite(view.data_y[idx])
    view.on_close()

    view = IndexPickerView(app, item_y="runs/[0]/shear_stress", item_x="time")
    assert view.data_x_combo.get() == "time"
    assert np.array_equal(view.data_x, app.truth[0].time)
    view.on_close()


def test_menu_actions_open_pickers(app, no_dialogs):
    run_menu_action(app, "runs/[0]/events/[0]/displacement", "Pick Indices")
    views = opened(app, IndexPickerView)
    assert len(views) == 1
    assert views[0].item_y == "displacement" and views[0].base_path == "runs/[0]/events/[0]"
    views[0].on_close()

    run_menu_action(app, "runs/[0]/friction", "Extract Slopes")
    views = opened(app, SlopeAnalyzerView)
    assert len(views) == 1 and views[0].item_y == "friction"
    views[0].on_close()

    # Min/Max: argmax then argmin, read-only selector
    path = "runs/[0]/events/[1]/shear_stress"
    y = app.data_manager.get_data(path)
    run_menu_action(app, path, "Min/Max")
    view, = opened(app, PointsSelectorView)
    assert view.picked_idx == [int(np.argmax(y)), int(np.argmin(y))]
    assert view.add_remove_enabled is False and view.callback is None
    assert len(view.markers) == 2
    view.on_close()
    # NaN samples are ignored (np.argmin/argmax would return 0)
    y[:5] = np.nan
    run_menu_action(app, path, "Min/Max")
    view, = opened(app, PointsSelectorView)
    assert view.picked_idx == [int(np.nanargmax(y)), int(np.nanargmin(y))]
    view.on_close()
    # all-NaN: a warning, no window, no exception
    y[:] = np.nan
    before = len(app.child_windows)
    run_menu_action(app, path, "Min/Max")
    assert len(app.child_windows) == before
    assert any("no finite samples" in str(call) for call in no_dialogs)


def test_pick_events_action_opens_the_event_picker(app):
    truth_picks = sorted(int(i) for i in app.truth[0].event_indices)
    ctx = run_menu_action(app, "runs/[0]/shear_stress", "Pick Events")
    assert ctx.key == "shear_stress"
    view, = opened(app, EventPickerView)
    assert view.picks == truth_picks
    assert view.y_combo.get() == "shear_stress" and view.ax.get_ylabel() == "shear_stress"
    # saving writes a copy into the data manager; later edits stay local
    key = "runs/[0]/event_extraction/event_indices"
    view.picker.add_point(5)
    view.save_picks()
    stored = app.data_manager.get_data(key)
    assert stored == sorted([5] + truth_picks) and stored is not view.picks
    view.picker.remove_point(0)
    view.picker.add_point(7)
    assert app.data_manager.get_data(key) == stored
    view.save_picks()
    assert app.data_manager.get_data(key) == sorted([7] + truth_picks)
    view.on_close()


# ---------------------------------------------------------------- PointsSelectorView
def test_points_selector_save_sorts_and_hands_out_copies(app):
    x = np.linspace(0, 2 * np.pi, 500)
    received = []
    view = PointsSelectorView(app, x, np.sin(x), [300, 100, 200], add_remove_enabled=True,
                              callback=received.append, xlabel="x", ylabel="sin",
                              title="demo")
    assert view.title() == "Points Selector"
    assert hasattr(view, "save_button")
    assert view.ax.get_xlabel() == "x" and view.ax.get_ylabel() == "sin"
    assert view.ax.get_title() == "demo"
    view.save()
    saved = received[-1]
    assert saved == [100, 200, 300] and saved is not view.picked_idx
    # markers follow the sorted order
    assert [m.get_label() for m in view.markers] == ["0", "1", "2"]
    assert [m.center[0] for m in view.markers] == [x[100], x[200], x[300]]
    # a drag, an add and a removal in the still-open window must not reach
    # what the callback already stored; the next Save delivers a fresh copy
    view.picked_idx[0] = 5
    view.add_point(40)
    view.remove_point(1)
    assert sorted(view.picked_idx) == [5, 40, 300]
    assert saved == [100, 200, 300]
    view.save()
    assert received[-1] == [5, 40, 300]
    assert received[-1] is not saved and received[-1] is not view.picked_idx
    view.on_close()
    assert view not in app.child_windows


def test_points_selector_add_remove_and_nan_gaps(app):
    x = np.arange(100.0)
    y = x ** 2
    picks = [np.int64(10), np.int64(20), np.int64(30)]
    view = PointsSelectorView(app, x, y, picks, add_remove_enabled=True)
    assert not hasattr(view, "save_button")
    # the input list is copied into plain ints
    assert all(type(i) is int for i in view.picked_idx)
    assert view.add_point(55) == 3
    assert view.picked_idx == [10, 20, 30, 55]
    assert [m.get_label() for m in view.markers] == ["0", "1", "2", "3"]
    view.remove_point(1)
    assert view.picked_idx == [10, 30, 55] and picks[1] == 20
    # every marker still sits on the sample its label points to
    for marker in view.markers:
        idx = view.picked_idx[int(marker.get_label())]
        assert marker.center == (x[idx], y[idx])
    assert len(view.ax.patches) == 3
    assert view.nearest_index(42.2, 42.2 ** 2) == 42
    view.on_close()

    # Dragging onto a NaN gap must snap to a finite neighbour: an Ellipse
    # centred at (x, nan) is invisible and can never be picked up again.
    y = x.copy()
    y[40:50] = np.nan
    view = PointsSelectorView(app, x, y, [10, 20], add_remove_enabled=True)
    idx = view.nearest_index(45.0, 45.0)
    assert idx in (39, 50) and np.isfinite(view.y_values[idx])
    assert view.add_point(view.nearest_index(44.0, 44.0)) == 2
    assert np.isfinite(view.y_values[view.picked_idx[-1]])
    view.on_close()


# ------------------------------------------------------------------- SimplePlotView
def test_simple_plot_view(app):
    view = SimplePlotView(app)
    assert view.title() == "Simple Plot"
    assert view.ax is view.figure.axes[0] and view.toolbar is not None
    # a caller that appends the already-registered view a second time
    app.child_windows.append(view)
    assert app.child_windows.count(view) == 2
    view.on_close()
    assert view not in app.child_windows
    assert not view.winfo_exists()


def test_double_click_array_opens_simple_plot_and_close_unregisters(app):
    item = app.find_item("runs/[0]/shear_stress")
    app.data_tree.selection_set(item)
    before = list(app.child_windows)
    app.on_double_click(None)
    new = [w for w in app.child_windows if w not in before]
    view = new[0]
    assert isinstance(view, SimplePlotView)
    assert all(w is view for w in new)
    assert view.ax.get_ylabel() == "shear_stress"
    assert view.ax.get_title() == "runs[0]/shear_stress"  # main window strips '/[' -> '['
    line, = view.ax.get_lines()
    assert np.array_equal(line.get_ydata(), app.truth[0].shear_stress)
    view.on_close()
    # no stale (destroyed) entry survives, however many times the main
    # window registered the view
    assert view not in app.child_windows
    assert app.child_windows == before
