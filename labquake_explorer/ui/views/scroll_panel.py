"""A column of controls that scrolls vertically when the window is too short.

``ScrollPanel(master)`` is a frame holding a canvas and a vertical
scrollbar; widgets go into ``panel.inner``.  The panel asks for the width of
its content and for its full height up to the screen height less a margin,
so a window opens at its natural size on a large screen and fits a small
one.  When gridded with ``sticky="ns"`` it follows the window's height; the
scrollbar shows only while the content is taller than the panel.

The mouse wheel scrolls the panel wherever the pointer is over it, comboboxes
included (their own wheel binding would change the value as the user
scrolls past), and a widget that takes the keyboard focus is scrolled into
view.  Call :meth:`bind_children` once the content is built, and again after
adding widgets later.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

#: Height (px) kept free of the screen for the title bar and the task bar.
SCREEN_MARGIN_PX = 160
WHEEL_EVENTS = ("<MouseWheel>", "<Button-4>", "<Button-5>")


def wheel_units(event) -> int:
    """Lines to scroll for a wheel event: Windows sends multiples of 120 in
    ``delta``, macOS small counts, X11 buttons 4 (up) and 5 (down)."""
    num = getattr(event, "num", None)
    if num == 4:
        return -1
    if num == 5:
        return 1
    delta = int(getattr(event, "delta", 0) or 0)
    if delta == 0:
        return 0
    steps = delta / 120.0 if abs(delta) >= 120 else float(delta)
    return -int(steps) if abs(steps) >= 1 else (-1 if delta > 0 else 1)


class ScrollPanel(ttk.Frame):
    """A vertically scrolling frame; put widgets into ``inner``."""

    def __init__(self, master, screen_margin_px: int = SCREEN_MARGIN_PX, **kw):
        super().__init__(master, **kw)
        self.screen_margin_px = int(screen_margin_px)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0, background=self._background())
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.scrollbar.grid_remove()
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_configure, add="+")
        self.canvas.bind("<Configure>", self._on_canvas_configure, add="+")
        self._bound: set = set()
        self.bind_children()

    def _background(self) -> str:
        try:
            colour = ttk.Style(self).lookup("TFrame", "background")
            if colour:
                self.winfo_rgb(colour)              # a name this Tk knows
                return colour
        except tk.TclError:
            pass
        return self.winfo_toplevel().cget("background")

    # ------------------------------------------------------------ geometry
    def content_height(self) -> int:
        return int(self.inner.winfo_reqheight())

    def overflows(self) -> bool:
        """True while the content is taller than the panel shows."""
        return self.content_height() > max(1, int(self.canvas.winfo_height()))

    def _on_inner_configure(self, _event=None) -> None:
        width, height = self.inner.winfo_reqwidth(), self.inner.winfo_reqheight()
        cap = max(200, int(self.winfo_screenheight()) - self.screen_margin_px)
        self.canvas.configure(width=width, height=min(height, cap))
        self._layout()

    def _on_canvas_configure(self, _event=None) -> None:
        self._layout()

    def _layout(self) -> None:
        shown_w, shown_h = int(self.canvas.winfo_width()), int(self.canvas.winfo_height())
        width = max(shown_w, self.inner.winfo_reqwidth())
        height = max(shown_h, self.content_height())
        self.canvas.itemconfigure(self._window, width=width, height=height)
        self.canvas.configure(scrollregion=(0, 0, width, height))
        if self.content_height() > shown_h > 1:
            self.scrollbar.grid()
        else:
            self.scrollbar.grid_remove()
            self.canvas.yview_moveto(0.0)

    # ------------------------------------------------------------ scrolling
    def scroll(self, units: int) -> None:
        if units and self.overflows():
            self.canvas.yview_scroll(int(units), "units")

    def _on_wheel(self, event):
        self.scroll(wheel_units(event))
        return "break"                              # a combobox under the pointer keeps its value

    def see(self, widget) -> None:
        """Scroll so that ``widget`` (inside ``inner``) is shown whole."""
        if not self.overflows():
            return
        top = widget.winfo_rooty() - self.inner.winfo_rooty()
        bottom = top + widget.winfo_height()
        total = max(1, self.content_height())
        shown_top = self.canvas.canvasy(0)
        shown_bottom = shown_top + self.canvas.winfo_height()
        if top < shown_top:
            self.canvas.yview_moveto(top / total)
        elif bottom > shown_bottom:
            self.canvas.yview_moveto((bottom - self.canvas.winfo_height()) / total)

    def bind_children(self) -> None:
        """Route the wheel over every widget of the panel to the panel, and
        scroll a widget that takes the focus into view."""
        stack = [self, self.canvas, self.inner]
        while stack:
            widget = stack.pop()
            stack.extend(widget.winfo_children())
            if str(widget) in self._bound:
                continue
            self._bound.add(str(widget))
            for sequence in WHEEL_EVENTS:
                widget.bind(sequence, self._on_wheel, add="+")
            if widget not in (self, self.canvas, self.inner):
                widget.bind("<FocusIn>", lambda e, w=widget: self.see(w), add="+")


__all__ = ["ScrollPanel", "wheel_units", "SCREEN_MARGIN_PX"]
