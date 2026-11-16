import numpy as np
import pytest

from labquake_explorer.ui.views.base import EventView, RunView, event_list, nearest_sample


def test_event_list_never_truth_tests_the_container():
    assert event_list({"events": [{"a": 1}, {"b": 2}]}) == [{"a": 1}, {"b": 2}]
    assert event_list({"events": ({"a": 1},)}) == [{"a": 1}]
    assert event_list({"events": []}) == []
    assert event_list({"events": np.zeros(0)}) == []          # empty list after an HDF5 round trip
    assert event_list({"events": np.zeros(3)}) == []          # not an events container
    assert event_list({"events": None}) == []
    assert event_list({}) == []
    assert event_list(None) == []


def test_nearest_sample_skips_nan_and_handles_all_nan():
    x = np.arange(10.0)
    y = x ** 2
    assert nearest_sample(x, y, 4.2, 17.0, (0, 10), (0, 100)) == 4
    y[3:6] = np.nan                                             # a gap in the signal
    assert nearest_sample(x, y, 4.0, 16.0, (0, 10), (0, 100)) in (2, 6)
    assert nearest_sample(x, y, 4.0, 16.0, (0, 10), (0, 100)) != 3   # plain argmin would give the first NaN
    x[7] = np.nan
    assert nearest_sample(x, y, 7.0, 49.0, (0, 10), (0, 100)) in (6, 8)
    assert nearest_sample(x, np.full(10, np.nan), 1.0, 1.0, (0, 10), (0, 100)) is None
    assert nearest_sample([], [], 1.0, 1.0, (0, 10), (0, 100)) is None
    # degenerate axis ranges do not divide by zero
    assert nearest_sample(np.arange(3.0), np.zeros(3), 1.2, 0.0, (0, 0), (0, 0)) == 1


class Broken(EventView):
    window_title = "Broken"

    def build_ui(self):
        raise RuntimeError("boom")


class BrokenRun(RunView):
    window_title = "BrokenRun"

    def on_run_loaded(self):
        raise KeyError("missing")


class Simple(EventView):
    window_title = "Simple"
    result_key = "simple"
    loads = 0

    def build_ui(self):
        self.build_event_selector(self, row=0, column=0)
        self.make_figure(figsize=(3, 2), figure_kwargs={"layout": "constrained"}, row=1, column=0)

    def on_event_loaded(self):
        self.loads += 1


def test_view_lifecycles(app):
    # a failed construction leaves no child window behind
    with pytest.raises(RuntimeError):
        Broken(app, 0, 0)
    assert app.child_windows == []
    with pytest.raises(KeyError):
        BrokenRun(app, 0)
    assert app.child_windows == []

    v = Simple(app, 0, 1)
    assert v.title() == "Simple - Event 1"
    assert v.loads == 1
    assert v.figure.get_layout_engine() is not None
    assert v.event_combobox.get() == "1" and len(v.event_combobox["values"]) == 4
    v.event_combobox.set("3")
    v.on_event_selected()
    assert v.event_idx == 3 and v.loads == 2 and v.title() == "Simple - Event 3"
    v.save_results({"version": 1, "x": 2})
    assert app.data_manager.get_data("runs/[0]/events/[3]/simple") == {"version": 1, "x": 2}
    assert v.load_results() == {"version": 1, "x": 2}
    v.save_field("strain/original/note", "hi")
    assert app.data_manager.get_data("runs/[0]/events/[3]/strain/original/note") == "hi"
    assert v.figure_title() == "p0001 run00 event3"
    # closing removes every registration, even a duplicate one
    app.child_windows.append(v)
    v.on_close()
    assert v not in app.child_windows

    class R(RunView):
        window_title = "R"
        result_key = "r"

    v = R(app, 0)
    assert v.title() == "R - run00"
    assert v.run is app.data_manager.get_data("runs/[0]")
    v.save_results({"version": 1})
    assert app.data_manager.get_data("runs/[0]/r") == {"version": 1}
    assert v.load_results() == {"version": 1}
    v.on_close()
    assert v not in app.child_windows
