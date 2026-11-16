"""The scrolling column of controls used by the views."""
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import pytest

from labquake_explorer.ui.views.scroll_panel import ScrollPanel, wheel_units


def test_wheel_units_on_every_platform():
    assert wheel_units(SimpleNamespace(num=4, delta=0)) == -1
    assert wheel_units(SimpleNamespace(num=5, delta=0)) == 1
    assert wheel_units(SimpleNamespace(num="??", delta=240)) == -2        # Windows: 120 per notch, up is positive
    assert wheel_units(SimpleNamespace(num="??", delta=-120)) == 1
    assert wheel_units(SimpleNamespace(num="??", delta=3)) == -3          # macOS: small counts
    assert wheel_units(SimpleNamespace(num="??", delta=0)) == 0


@pytest.fixture
def window(tk_root):
    top = tk.Toplevel(tk_root)
    top.grid_rowconfigure(0, weight=1)
    yield top
    top.destroy()


def fill(panel, rows=40):
    combo = None
    for k in range(rows):
        ttk.Label(panel.inner, text=f"row {k}").grid(row=k, column=0, pady=2)
        if k == rows - 1:
            combo = ttk.Combobox(panel.inner, values=["a", "b", "c"], state="readonly")
            combo.set("b")
            combo.grid(row=k, column=1)
    panel.bind_children()
    return combo


def test_panel_scrolls_only_when_the_window_is_shorter(window):
    panel = ScrollPanel(window)
    panel.grid(row=0, column=0, sticky="ns")
    combo = fill(panel)
    window.update()
    natural = panel.content_height()
    assert natural > 300
    assert int(panel.canvas.cget("height")) == min(natural, window.winfo_screenheight() - 160)
    window.geometry(f"400x{natural + 50}")
    window.update()
    assert not panel.overflows() and not panel.scrollbar.winfo_ismapped()
    window.geometry("400x200")
    window.update()
    if not window.winfo_ismapped():
        pytest.skip("window not mapped")
    assert panel.overflows() and panel.scrollbar.winfo_ismapped()
    assert panel.canvas.yview()[0] == 0.0
    panel._on_wheel(SimpleNamespace(num=5, delta=-120))
    assert panel.canvas.yview()[0] > 0.0
    # the wheel over a combobox scrolls the panel and leaves the value alone
    before = panel.canvas.yview()[0]
    combo.event_generate("<MouseWheel>", delta=-120)
    window.update()
    assert combo.get() == "b" and panel.canvas.yview()[0] >= before
    # focus on a widget out of sight brings it into view
    panel.canvas.yview_moveto(0.0)
    window.update()
    panel.see(combo)
    window.update()
    top = panel.canvas.canvasy(0)
    y = combo.winfo_rooty() - panel.inner.winfo_rooty()
    assert top <= y and y + combo.winfo_height() <= top + panel.canvas.winfo_height() + 1
    # tall again: the scrollbar goes and the content returns to the top
    window.geometry(f"400x{natural + 50}")
    window.update()
    assert not panel.scrollbar.winfo_ismapped() and panel.canvas.yview()[0] == 0.0
