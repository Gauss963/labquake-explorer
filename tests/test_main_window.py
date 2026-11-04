import tkinter as tk
from types import SimpleNamespace

import numpy as np

from labquake_explorer.ui import context as C
from labquake_explorer.ui.views import EventAnalyzerView


def labels(app, path):
    item = app.find_item(path)
    assert item is not None, path
    ctx = app.context_at(item)
    menu = app.build_context_menu(ctx)
    if menu is None:
        return ctx, []
    n = menu.index("end")
    try:
        return ctx, [menu.entrycget(i, "label") for i in range(n + 1)]
    finally:
        menu.destroy()


def test_tree_paths_are_slash_joined(app):
    """Tree paths are data paths (never os.path.join'ed: backslashes on Windows
    would break the context resolver, pick/extract events and delete)."""
    for path in ("runs/[0]/events/[1]/shear_stress", "runs/[0]/event_extraction", "name"):
        item = app.find_item(path)
        full, key = app.get_full_path(item)
        assert full == path and "\\" not in full
        assert key == path.rsplit("/", 1)[-1]
    assert app.context_at(app.find_item("runs/[0]/event_extraction")).parent_path == "runs/[0]"


def test_right_click_destroys_previous_menu(app, monkeypatch):
    """A fresh Menu is built per right-click; the previous one must be
    destroyed, not merely unposted, or menus accumulate under root."""
    posted = []
    monkeypatch.setattr(tk.Menu, "post", lambda self, x, y: posted.append((x, y)))  # a real post blocks
    item = app.find_item("runs/[0]/events/[1]")
    app.data_tree.selection_set(item)
    event = SimpleNamespace(x_root=0, y_root=0)
    before = sum(isinstance(w, tk.Menu) for w in app.root.winfo_children())
    app.on_right_click(event)
    first = app.active_context_menu
    assert first is not None and first.winfo_exists() and posted == [(0, 0)]
    for _ in range(5):
        app.on_right_click(event)
    assert not first.winfo_exists()
    assert app.active_context_menu is not None and app.active_context_menu is not first
    after = sum(isinstance(w, tk.Menu) for w in app.root.winfo_children())
    assert after == before + 1
    app.on_left_click(None)


def test_find_item_and_context(app):
    ctx, _ = labels(app, "runs/[0]/events/[1]/shear_stress")
    assert ctx.kind == C.EVENT_ARRAY
    assert (ctx.run_idx, ctx.event_idx, ctx.key) == (0, 1, "shear_stress")


def test_context_menus_from_registry(app):
    assert labels(app, "runs/[0]/shear_stress")[1] == ["Pick Events", "Pick Indices", "Extract Slopes"]
    assert labels(app, "runs/[0]/event_extraction")[1] == ["Pick Events"]
    event_labels = ["Analyze Event", "Slip Distribution", "Pick Arrivals", "Fit Cohesive Zone Model", "PZT Spectrum"]
    assert labels(app, "runs/[0]/events/[0]")[1] == event_labels
    assert labels(app, "runs/[0]/events/[0]/event_time")[1] == event_labels
    assert labels(app, "runs/[0]/events/[0]/shear_stress")[1] == ["Pick Indices", "Extract Slopes", "Min/Max"]
    assert labels(app, "name")[1] == ["Edit String"]
    assert labels(app, "runs/[0]/name")[1] == ["Edit String"]
    assert labels(app, "runs")[1] == []
    assert labels(app, "runs/[0]")[1] == ["Pick Events", "Plot Run Signals", "Inter-event Metrics", "Source Scaling"]


def test_run_action_opens_registered_view(app):
    ctx, menu_labels = labels(app, "runs/[0]/events/[1]")
    from labquake_explorer.ui.actions import actions_for
    action = [a for a in actions_for(ctx.kind) if a.label == "Analyze Event"][0]
    app.run_action(action, ctx)
    assert len(app.child_windows) == 1
    view = app.child_windows[0]
    assert isinstance(view, EventAnalyzerView)
    assert (view.run_idx, view.event_idx) == (0, 1)
    view.destroy()


def test_double_click_run_opens_signal_overlay(app):
    from labquake_explorer.ui.views import RunSignalsView
    item = app.find_item("runs/[0]")
    app.data_tree.selection_set(item)
    app.on_double_click(None)
    assert isinstance(app.child_windows[-1], RunSignalsView)
    assert "Plot Run Signals" in labels(app, "runs/[0]")[1]
    app.child_windows[-1].on_close()


def test_double_click_array_opens_simple_plot_once(app):
    from labquake_explorer.ui.views import SimplePlotView
    item = app.find_item("runs/[0]/shear_stress")
    app.data_tree.selection_set(item)
    app.on_double_click(None)
    views = [w for w in app.child_windows if isinstance(w, SimplePlotView)]
    assert len(views) == 1 and app.child_windows.count(views[0]) == 1
    views[0].on_close()
    assert views[0] not in app.child_windows


def test_save_as_defaults_to_hdf5(app, tmp_path, monkeypatch):
    from tkinter import filedialog
    calls = []

    def fake_save_dialog(**kw):
        calls.append(kw)
        return "" if len(calls) == 1 else str(tmp_path / "p0001.h5")
    monkeypatch.setattr(filedialog, "asksaveasfilename", fake_save_dialog)

    app.current_file_path = tmp_path / "p0001.npz"
    app.save_file()                                   # cancelled
    kw = calls[-1]
    assert kw["defaultextension"] == ".h5"
    assert kw["filetypes"][0] == ("HDF5 files", "*.h5 *.hdf5")
    assert kw["initialfile"] == "p0001.h5" and kw["initialdir"] == str(tmp_path)

    app.current_file_path = tmp_path / "p0001.hdf5"
    app.save_file()                                   # saved as p0001.h5
    assert calls[-1]["initialfile"] == "p0001.hdf5"
    assert (tmp_path / "p0001.h5").exists()
    assert app.current_file_path == tmp_path / "p0001.h5"
    assert app.data_tree.heading("#0")["text"] == "p0001.h5"
