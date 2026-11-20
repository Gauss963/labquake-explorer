"""Pick sample indices on one array plotted against a sibling array.

``ArrayPairView`` is the shared machinery: two comboboxes listing the arrays
that are siblings of ``item_y`` in the main window's tree, a figure with the
Y array plotted against the X array (or against its index), and draggable
markers snapped to samples.  ``IndexPickerView`` reports the picked indices;
``SlopeAnalyzerView`` (its own module) reports the slope between two picks.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from labquake_explorer.ui.views.base import BaseView
from labquake_explorer.ui.views.point_picker import PointPicker


def split_item_path(path: Optional[str]) -> tuple[str, Optional[str]]:
    """'runs/[0]/shear_stress' -> ('runs/[0]', 'shear_stress'); 'x' -> ('', 'x')."""
    if not path:
        return "", None
    path = path.replace("\\", "/").rstrip("/")
    base, _, name = path.rpartition("/")
    return base, name


class ArrayPairView(BaseView):
    """Y array vs sibling X array (or index) with draggable, sample-snapped markers.

    Subclasses set ``readout_label`` and implement ``update_readout()``, which
    is called whenever ``picked_idx`` changes.  ``picked_idx`` holds Python
    ints; ``data_x`` is the X array or ``arange(len(data_y))`` when no X array
    is selected.
    """

    readout_label = "Picked Index"
    marker_color = "red"

    def __init__(self, app, item_y=None, item_x=None):
        self.base_path, self.item_y = split_item_path(item_y)
        self.item_x = item_x
        self.data_x = np.array([])
        self.data_y = np.array([])
        super().__init__(app)

        self.build_ui()
        self.picker = PointPicker(self.ax, self.canvas, self.data_x, self.data_y, add_remove=False,
                                  toolbar_active=self.toolbar_active, color=self.marker_color,
                                  on_drag=lambda position, idx: self.on_dragged())
        self.init_comboboxes()
        self.plot_data()
        if len(self.data_y) > 0:
            n = len(self.data_y)
            self.picked_idx = [int(n / 3), int(n / 3 * 2)]
        self.plot_picked_points()

    @property
    def picked_idx(self) -> list[int]:
        """The picked sample indices (the picker's list: edits in place move nothing until redrawn)."""
        return self.picker.picks

    @picked_idx.setter
    def picked_idx(self, value) -> None:
        self.picker.picks = [int(i) for i in value]

    @property
    def markers(self) -> list:
        return self.picker.markers

    # ------------------------------------------------------------------ ui
    def build_ui(self):
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(2, weight=1)

        tk.Label(self, text="X Data").grid(row=0, column=0, padx=5, pady=5)
        tk.Label(self, text="Y Data").grid(row=0, column=1, padx=5, pady=5)
        tk.Label(self, text=self.readout_label).grid(row=0, column=3, padx=5, pady=5)

        self.data_x_combo = ttk.Combobox(self, state="readonly")
        self.data_x_combo.grid(row=1, column=0, padx=5, pady=5)
        self.data_y_combo = ttk.Combobox(self, state="readonly")
        self.data_y_combo.grid(row=1, column=1, padx=5, pady=5)
        self.readout_textbox = tk.Entry(self, state="readonly")
        self.readout_textbox.grid(row=1, column=3, padx=5, pady=5)
        self.data_y_combo.bind("<<ComboboxSelected>>", self.data_y_selected)
        self.data_x_combo.bind("<<ComboboxSelected>>", self.data_x_selected)

        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=3, column=0, columnspan=4, padx=0, pady=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    # --------------------------------------------------------------- paths
    def item_path(self, name: str) -> str:
        return f"{self.base_path}/{name}" if self.base_path else name

    def sibling_arrays(self) -> list[str]:
        """Names of the array-valued siblings of ``item_y`` in the main window's tree."""
        if not self.item_y:
            return []
        tree = self.app.data_tree
        item = self.app.find_item(self.item_path(self.item_y))
        if item is None:
            return []
        parent_id = tree.parent(item)
        if parent_id:
            siblings = tree.get_children(parent_id)
            self.base_path = self.app.get_full_path(parent_id)[0]
        else:
            siblings = tree.get_children("")
            self.base_path = ""
        names = []
        for sibling in siblings:
            text = tree.item(sibling)["text"]
            if ":" in text and "array" in text.split(":", 1)[1]:
                names.append(text.split(":", 1)[0].strip())
        return names

    def init_comboboxes(self):
        names = self.sibling_arrays()
        self.data_x_combo.config(values=names)
        self.data_y_combo.config(values=names)
        if self.item_y in names:
            self.data_y_combo.set(self.item_y)
        if self.item_x in names:
            self.data_x_combo.set(self.item_x)
        elif self.item_x is not None:
            print(f"Warning: '{self.item_x}' is not a sibling of '{self.item_y}'")
            self.item_x = None
        if self.item_x is None:
            self.item_x = self.default_x(names)
            if self.item_x is not None:
                self.data_x_combo.set(self.item_x)

    def default_x(self, names: list[str]) -> Optional[str]:
        """The X array to start with when none was given (None = sample index)."""
        return None

    def data_y_selected(self, event=None):
        self.item_y = self.data_y_combo.get()
        self.plot_data()
        self.plot_picked_points()

    def data_x_selected(self, event=None):
        self.item_x = self.data_x_combo.get() or None
        self.plot_data()
        self.plot_picked_points()

    # ------------------------------------------------------------- plotting
    def plot_data(self):
        if not self.item_y:
            return
        self.ax.clear()
        self.data_y = np.asarray(self.data_manager.get_data(self.item_path(self.item_y)))
        if self.item_x is None:
            self.data_x = np.arange(len(self.data_y))
            self.ax.set_xlabel("Index")
        else:
            self.data_x = np.asarray(self.data_manager.get_data(self.item_path(self.item_x)))
            if len(self.data_x) != len(self.data_y):
                print(f"Warning: '{self.item_x}' and '{self.item_y}' differ in length; plotting against index")
                self.data_x = np.arange(len(self.data_y))
                self.ax.set_xlabel("Index")
            else:
                self.ax.set_xlabel(self.item_x)
        self.ax.plot(self.data_x, self.data_y, zorder=-100)
        self.ax.set_ylabel(self.item_y)
        self.picker.set_curve(self.data_x, self.data_y)
        self.canvas.draw()

    def clamp_picked(self):
        n = len(self.data_y)
        if n == 0:
            self.picked_idx = []
            return
        self.picked_idx = [min(max(int(i), 0), n - 1) for i in self.picked_idx]

    def plot_picked_points(self):
        """(Re)draw the markers for ``picked_idx``, then refresh the readout."""
        if len(self.data_y) == 0:
            return
        self.clamp_picked()
        self.picker.draw_markers()
        self.draw_overlays()
        self.canvas.draw()
        self.update_readout()

    def draw_overlays(self):
        """Hook for subclasses to add artists that depend on the picks."""

    def update_overlays(self):
        """Hook for subclasses to move those artists while dragging."""

    def update_readout(self):
        self.set_readout(str([int(i) for i in self.picked_idx]))

    def set_readout(self, text: str):
        """Show ``text`` in the readout box and copy it to the clipboard."""
        self.readout_textbox.config(state="normal")
        self.readout_textbox.delete(0, tk.END)
        self.readout_textbox.insert(0, text)
        self.readout_textbox.config(state="readonly")
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
        except tk.TclError:
            pass

    def on_dragged(self):
        """A marker was dragged to another sample."""
        self.update_overlays()
        self.update_readout()


class IndexPickerView(ArrayPairView):
    """Two draggable markers; the picked sample indices go to the readout and clipboard."""

    window_title = "Index Picker"
    readout_label = "Picked Index"

    @property
    def index_textbox(self):
        return self.readout_textbox

    def set_index_textbox(self, text: str):
        self.set_readout(text)
