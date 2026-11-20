"""Pick three ranges on one event and read off slopes and differences."""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from labquake_explorer.analysis.event_metrics import (
    EventPicks, analyze_event, picks_from_result, picks_from_windows, windows_from_picks,
)
from labquake_explorer.data.channels import aligned_fields, get_field as resolve_field
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT
from labquake_explorer.ui.views.base import EventView
from labquake_explorer.ui.views.point_picker import PointPicker

MARKER_COLORS = ['#33CCC4', '#33CCC4', '#CC3366', '#CC3366', '#33CC66', '#33CC66']
N_PICKS = 6
RESULT_LAYOUT = (                    # label, result key, row, column
    ("Loading slope:", "loading_slope", 0, 0),
    ("Unloading slope:", "unloading_slope", 1, 0),
    ("Δx:", "delta_x", 0, 2),
    ("Δy:", "delta_y", 1, 2),
)


@register_view("Analyze Event", kinds=[EVENT], order=10)
class EventAnalyzerView(EventView):
    """Six draggable markers snapped to samples of Y against X:

      0,1  loading range    -> loading_slope   dY/dX (least squares)
      2,3  unloading range  -> unloading_slope dY/dX
      4,5  start, end       -> delta_x = X5 - X4, delta_y = Y5 - Y4

    With X = fault slip and Y = shear stress these are the stiffnesses, the
    coseismic slip and minus the stress drop; the record keeps ``x_field`` and
    ``y_field`` so a reader knows what they are.  Numbers come from
    :func:`labquake_explorer.analysis.event_metrics.analyze_event` and are
    saved under ``event['event_analysis']`` (schema version 3).
    """

    window_title = "Event Analyzer"
    result_key = "event_analysis"

    def __init__(self, app, run_idx, event_idx, item_y="shear_stress", item_x="displacement"):
        self.item_y = item_y
        self.item_x = item_x
        self.data_x = np.array([])
        self.data_y = np.array([])
        self.data_t = np.array([])
        self.picker = None
        self.loading_line = None
        self.unloading_line = None
        self.delta_span = None
        self.result = None
        super().__init__(app, run_idx, event_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(2, weight=1)

        event_selection_frame = ttk.LabelFrame(self, text="Event Selection")
        event_selection_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky="nsw")
        self.build_event_selector(event_selection_frame, row=0, column=0)
        self.save_button = ttk.Button(event_selection_frame, text="Save Event", command=self.save_event, width=15)
        self.save_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.apply_all_button = ttk.Button(event_selection_frame, text="Apply to All Events",
                                           command=self.apply_to_all_events, width=18)
        self.apply_all_button.grid(row=2, column=0, columnspan=2, padx=5, pady=2, sticky="ew")

        data_frame = ttk.LabelFrame(self, text="Data Fields")
        data_frame.grid(row=0, column=1, rowspan=2, columnspan=2, padx=5, pady=5, sticky="nsew")
        tk.Label(data_frame, text="X Data:", width=8, anchor="e").grid(row=0, column=0, padx=5, pady=3, sticky="e")
        self.data_x_combo = ttk.Combobox(data_frame, state="readonly", width=20)
        self.data_x_combo.grid(row=0, column=1, padx=5, pady=3, sticky="w")
        tk.Label(data_frame, text="Y Data:", width=8, anchor="e").grid(row=1, column=0, padx=5, pady=3, sticky="e")
        self.data_y_combo = ttk.Combobox(data_frame, state="readonly", width=20)
        self.data_y_combo.grid(row=1, column=1, padx=5, pady=3, sticky="w")
        ttk.Label(data_frame, text="Drag the markers: cyan = loading range, red = unloading range,\n"
                                   "green = the two samples differenced (Δ = end - start).",
                  justify="left").grid(row=2, column=0, columnspan=2, padx=5, pady=(6, 3), sticky="w")
        self.data_y_combo.bind("<<ComboboxSelected>>", self.data_selected)
        self.data_x_combo.bind("<<ComboboxSelected>>", self.data_selected)

        results_frame = ttk.LabelFrame(self, text="Results")
        results_frame.grid(row=0, column=3, rowspan=2, columnspan=1, padx=5, pady=5, sticky="nsew")
        self.result_entries = {}
        for text, key, row, col in RESULT_LAYOUT:
            tk.Label(results_frame, text=text, anchor="e", width=15).grid(row=row, column=col, padx=5, pady=2, sticky="e")
            entry = tk.Entry(results_frame, state="readonly", width=12, justify="right")
            entry.grid(row=row, column=col + 1, padx=5, pady=2, sticky="w")
            self.result_entries[key] = entry
        self.loading_slope_text = self.result_entries["loading_slope"]
        self.unloading_slope_text = self.result_entries["unloading_slope"]
        self.delta_x_text = self.result_entries["delta_x"]
        self.delta_y_text = self.result_entries["delta_y"]

        self.make_figure(figsize=(10, 6), row=2, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")
        self.figure.set_facecolor('#f5f5f5')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#ffffff')
        self.picker = PointPicker(self.ax, self.canvas, self.data_x, self.data_y, add_remove=False,
                                  toolbar_active=self.toolbar_active, color=MARKER_COLORS,
                                  on_drag=lambda position, idx: self.point_moved(position))

    @property
    def picked_idx(self) -> list:
        """The six picked sample indices (the picker's list)."""
        return self.picker.picks if self.picker is not None else []

    @picked_idx.setter
    def picked_idx(self, value) -> None:
        self.picker.picks = [int(i) for i in value]

    @property
    def markers(self) -> list:
        return self.picker.markers if self.picker is not None else []

    def on_event_loaded(self):
        saved = self.load_results()
        # a record remembers the fields it was analysed on: show (and re-save)
        # the same pair, not the constructor defaults (init_comboboxes falls
        # back when a remembered field no longer exists)
        if saved:
            for attr, key in (("item_x", "x_field"), ("item_y", "y_field")):
                field = saved.get(key)
                if isinstance(field, str) and field:
                    setattr(self, attr, field)
        self.init_comboboxes()
        picks = picks_from_result(saved, len(self.data_y)) if saved else None
        if picks is None:
            picks = EventPicks.defaults(len(self.data_y))
        self.picked_idx = picks.to_list()
        self.plot_picked_points()

    def _set_default_point_positions(self):
        self.picked_idx = EventPicks.defaults(len(self.data_y)).to_list()

    def init_comboboxes(self):
        """Initialize comboboxes with event fields of the same length as time"""
        if 'time' in self.event:
            time_length = len(self.event['time'])
        else:
            print("Warning: 'time' field not found in event data")
            return

        matching_fields = sorted(aligned_fields(self.event, time_length))
        self.data_x_combo.config(values=matching_fields)
        self.data_y_combo.config(values=matching_fields)

        if not (self.item_x and self.item_x in matching_fields):
            # fault slip first: 'displacement' (PSU files), else the first slip channel, else any displacement
            slip_fields = [f for f in matching_fields if f.split('/')[-1].lower().startswith('slip')]
            displacement_fields = [f for f in matching_fields if 'displacement' in f.lower()]
            if 'displacement' in matching_fields:
                self.item_x = 'displacement'
            elif slip_fields:
                self.item_x = slip_fields[0]
            elif displacement_fields:
                self.item_x = displacement_fields[0]
            elif 'time' in matching_fields:
                self.item_x = 'time'
            elif matching_fields:
                self.item_x = matching_fields[0]
        self.data_x_combo.set(self.item_x or "")

        if not (self.item_y and self.item_y in matching_fields):
            stress_fields = [f for f in matching_fields if 'shear_stress' in f.lower()]
            other_stress = [f for f in matching_fields if 'stress' in f.lower()]
            if stress_fields:
                self.item_y = stress_fields[0]
            elif other_stress:
                self.item_y = other_stress[0]
            elif matching_fields:
                self.item_y = matching_fields[0]
        self.data_y_combo.set(self.item_y or "")

        self.plot_data()

    def data_selected(self, event=None):
        """Handle data selection from comboboxes"""
        self.item_y = self.data_y_combo.get()
        self.item_x = self.data_x_combo.get()
        self.plot_data()
        n = len(self.data_y)
        if not (n > 0 and len(self.picked_idx) == N_PICKS and max(self.picked_idx) < n):
            self._set_default_point_positions()
        self.plot_picked_points()

    def get_field(self, path):
        """An event field by path: a top-level array, a nested one ('strain/time')
        or a row of a channel array ('slip/slip_3')."""
        value = resolve_field(self.event, path)
        if value is None:
            print(f"Warning: Path '{path}' not found in data")
        return value

    def plot_data(self):
        """Plot the selected data"""
        if not self.item_y:
            return
        self.ax.clear()
        data_y = self.get_field(self.item_y)
        if data_y is None:
            return
        self.data_y = np.asarray(data_y, dtype=float)
        if self.item_x:
            data_x = self.get_field(self.item_x)
            if data_x is None:
                return
            self.data_x = np.asarray(data_x, dtype=float)
            self.ax.set_xlabel(self.item_x)
        else:
            self.data_x = np.arange(len(self.data_y), dtype=float)
            self.ax.set_xlabel("Index")
        time = self.event.get('time')
        if time is not None and len(time) == len(self.data_y):
            self.data_t = np.asarray(time, dtype=float)
        else:
            self.data_t = np.arange(len(self.data_y), dtype=float)
        self.picker.set_curve(self.data_x, self.data_y)
        self.ax.plot(self.data_x, self.data_y, zorder=-100, linewidth=1.5)
        self.ax.set_ylabel(self.item_y)
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.set_title(f"{self.figure_title()}: {self.item_y} vs {self.item_x if self.item_x else 'Index'}")
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.tick_params(direction='out')
        self.canvas.draw()

    def plot_picked_points(self):
        """Plot the marker points, the two range lines and the delta span"""
        if len(self.picked_idx) != N_PICKS or len(self.data_y) == 0:
            return
        for line in (self.loading_line, self.unloading_line):
            if line is not None and line in self.ax.lines:
                line.remove()
        self._remove_span()

        if max(self.picked_idx) >= len(self.data_y):
            self._set_default_point_positions()
        self.picker.draw_markers()

        self.loading_line, = self.ax.plot(*self._segment(0, 1), '--', color=MARKER_COLORS[0],
                                          linewidth=2, zorder=-50, label='Loading')
        self.unloading_line, = self.ax.plot(*self._segment(2, 3), '--', color=MARKER_COLORS[2],
                                            linewidth=2, zorder=-50, label='Unloading')
        self._draw_span()
        self.canvas.draw()
        self.update_analysis()

    def _segment(self, i, j):
        return ([self.data_x[self.picked_idx[i]], self.data_x[self.picked_idx[j]]],
                [self.data_y[self.picked_idx[i]], self.data_y[self.picked_idx[j]]])

    def _draw_span(self):
        self.delta_span = self.ax.axvspan(self.data_x[self.picked_idx[4]], self.data_x[self.picked_idx[5]],
                                          alpha=0.15, color=MARKER_COLORS[4], zorder=-100)

    def _remove_span(self):
        if self.delta_span:
            try:
                self.delta_span.remove()
            except Exception:
                pass
            self.delta_span = None

    # ------------------------------------------------------------ analysis
    def update_analysis(self):
        """Recompute the slopes and differences from the current picks and show them."""
        if len(self.picked_idx) != N_PICKS or len(self.data_y) == 0:
            return
        picks = EventPicks.from_list(self.picked_idx)
        try:
            self.result = analyze_event(self.data_x, self.data_y, picks,
                                        x_field=self.item_x or "", y_field=self.item_y or "")
        except ValueError as e:
            print(f"Error calculating values: {e}")
            return
        for key, entry in self.result_entries.items():
            self.set_textbox(entry, self.format_value(self.result[key]))

    @staticmethod
    def format_value(value) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not np.isfinite(value) else f"{value:.6g}"

    def set_textbox(self, textbox, text):
        """Helper method to set text in a readonly textbox"""
        textbox.config(state="normal")
        textbox.delete(0, tk.END)
        textbox.insert(0, text)
        textbox.config(state="readonly")

    # ------------------------------------------------------------- dragging
    def move_point(self, point_idx: int, idx: int) -> None:
        """Move marker ``point_idx`` to sample ``idx`` and refresh lines and results."""
        idx = int(min(max(idx, 0), len(self.data_y) - 1))
        self.picked_idx[point_idx] = idx
        if point_idx < len(self.markers):
            self.markers[point_idx].set_center((self.data_x[idx], self.data_y[idx]))
        self.point_moved(point_idx)
        self.canvas.draw_idle()

    def point_moved(self, point_idx: int) -> None:
        """The lines, the span and the results after marker ``point_idx`` moved."""
        if point_idx in (0, 1):
            self.loading_line.set_data(*self._segment(0, 1))
        elif point_idx in (2, 3):
            self.unloading_line.set_data(*self._segment(2, 3))
        else:
            self._remove_span()
            self._draw_span()
        self.update_analysis()

    # ---------------------------------------------------------------- save
    def apply_to_all_events(self, confirm=True):
        """Place the current ranges, as times relative to the event, on every
        event of the run, compute and save.  Returns the number of events
        written; events whose fields are missing are skipped."""
        if len(self.picked_idx) != N_PICKS or not len(self.data_t):
            return 0
        event_time = float(self.event.get('event_time', self.data_t[0]))
        windows = windows_from_picks(self.data_t - event_time, EventPicks.from_list(self.picked_idx))
        n_events = self.n_events()
        if confirm and not messagebox.askokcancel(
                "Apply to all events",
                f"Recompute and overwrite event_analysis for all {n_events} events of run {self.run_idx} "
                f"using the current ranges?", icon=messagebox.WARNING):
            return 0
        written = 0
        for j in range(n_events):
            event = self.data_manager.get_data(f"{self.run_path}/events/[{j}]")
            try:
                t = np.asarray(event['time'], dtype=float)
                x = np.asarray(self.get_field_of(event, self.item_x), dtype=float)
                y = np.asarray(self.get_field_of(event, self.item_y), dtype=float)
                t_event = float(event['event_time'])
            except (KeyError, TypeError, ValueError):
                continue
            if not (t.size and t.size == x.size == y.size):
                continue
            picks = picks_from_windows(t - t_event, windows)
            result = analyze_event(x, y, picks, x_field=self.item_x or "", y_field=self.item_y or "")
            self.data_manager.set_data(f"{self.run_path}/events/[{j}]/event_analysis", result, True)
            event['event_analysis'] = result
            written += 1
        self.app.refresh_tree()
        self.set_event(self.event_idx)
        return written

    @staticmethod
    def get_field_of(event, path):
        value = resolve_field(event, path or "")
        if value is None:
            raise KeyError(path)
        return value

    def save_event(self):
        """Save the analysis results to the data manager"""
        if self.result is None:
            self.update_analysis()
        if self.result is None:
            return
        try:
            self.save_results(dict(self.result))
        except Exception as e:
            messagebox.showerror("Save Failed", f"Error saving results: {e}")
