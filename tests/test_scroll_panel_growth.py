"""ScrollPanel content that grows after the layout (added on the slip-auto-windows branch):
a status text getting longer must make the panel scroll, not be cut off."""
import tkinter as tk
from tkinter import ttk

import pytest

from labquake_explorer.ui.views.scroll_panel import ScrollPanel


@pytest.fixture
def window(tk_root):
    top = tk.Toplevel(tk_root)
    top.grid_rowconfigure(0, weight=1)
    yield top
    top.destroy()


def test_content_that_grows_later_is_reached_by_scrolling(window):
    panel = ScrollPanel(window)
    panel.grid(row=0, column=0, sticky="ns")
    for k in range(10):
        ttk.Label(panel.inner, text=f"row {k}").pack(anchor="w", pady=2)
    status = tk.StringVar(master=window, value="short")
    label = ttk.Label(panel.inner, textvariable=status, wraplength=120, justify="left")
    label.pack(anchor="w")
    panel.bind_children()
    window.update()
    window.geometry(f"300x{panel.content_height() + 10}")
    window.update()
    if not window.winfo_ismapped():
        pytest.skip("window not mapped")
    assert not panel.scrollbar.winfo_ismapped()
    status.set(" ".join(["a long status text that wraps"] * 12))         # the text grows: more lines
    window.update()
    assert panel.overflows() and panel.scrollbar.winfo_ismapped()
    region = [float(v) for v in str(panel.canvas.cget("scrollregion")).split()]
    assert region[3] == pytest.approx(panel.content_height())        # the whole content can be scrolled to
    panel.canvas.yview_moveto(1.0)
    window.update()
    bottom = label.winfo_y() + label.winfo_height()
    assert panel.canvas.canvasy(0) + panel.canvas.winfo_height() >= bottom - 1
