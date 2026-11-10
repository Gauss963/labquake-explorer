"""Pick the events of a run and extract them, in one window.

Opened from a run, one of its arrays or ``event_extraction``.  Plots Y against X
for the whole run; picks are placed on the curve (left double-click adds one
at the nearest sample, right double-click on a pick removes it, drag moves
it) and the ``[start, end]`` window of every pick is shaded, the selected one
highlighted.  "Save picks" writes the picks; "Extract Events" writes the
picks and then the events.  Everything the form needs to reopen as it was is
kept in ``runs/[r]['event_extraction']``: the picks (``event_indices``), start
and end, the series shown and how many events the last extraction wrote.
Y can also be a channel of a converted raw-data source (``elsys/<channel>``,
e.g. the on-fault stresses read from the 2 MHz Elsys file in MPa), averaged
onto the run's samples when chosen; nothing of it is stored.
Older files with the picks at the run's top level are read, and saving moves
them into the record.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np

from labquake_explorer.data.channels import aligned_fields, get_field
from labquake_explorer.data.picks import LEGACY_INDICES_KEY, picked_indices
from labquake_explorer.data.sources import ConvertedSource, mean_on_samples
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT_EXTRACTION, RUN, RUN_ARRAY, TreeContext
from labquake_explorer.ui.views.base import RunView, event_list
from labquake_explorer.ui.views.point_picker import PointPicker

RESULT_VERSION = 1
FIELDS_REFERENCE = "reference (event_fields)"   # the waveform channels the run's reference names
FIELDS_ALL = "all channels"
DEFAULT_HALF_WINDOW_S = 5.0      # start/end default to -/+ config.DEFAULT_WINDOW_SIZE, else this
X_INDEX = "index"
ZOOM_FACTOR = 3.0                # "Zoom to event" shows this many windows around the event
STEP_S = 0.5                     # one tick of the start/end spinboxes
HELP_TEXT = ("Left double-click on the curve: add a pick at the nearest sample.   "
             "Right double-click on a pick: remove it.   Drag a pick to move it.   "
             "Save picks stores the picks; Extract Events stores the picks and then the events.")
_SOURCE_SERIES: dict = {}        # (file, channel, offset, run samples) -> source channel on the run's samples


@register_view("Pick Events", kinds=[RUN, RUN_ARRAY, EVENT_EXTRACTION], order=0)
class EventPickerView(RunView):
    window_title = "Pick Events"
    result_key = "event_extraction"
    legacy_result_key = "event_window"          # the record's name before it held the picks

    def __init__(self, app, run_idx, y_field: Optional[str] = None):
        self.y_field_hint = y_field
        self.candidates: list[str] = []
        self.source_series: dict = {}    # 'elsys/<channel>' -> (source, channel)
        self.time: np.ndarray = np.array([])
        self.picker: Optional[PointPicker] = None
        self.spans: list = []
        self.result: Optional[dict] = None
        self._shown: dict = {}
        self._plotted_window: tuple = ("", "")
        self._replot_pending = False
        super().__init__(app, run_idx)

    @classmethod
    def from_context(cls, app, ctx: TreeContext):
        if ctx.run_idx is None:
            raise ValueError(f"{cls.__name__} needs a run context, got {ctx.path!r}")
        return cls(app, ctx.run_idx, y_field=ctx.key if ctx.kind == RUN_ARRAY else None)

    @property
    def picks(self) -> list:
        """The live pick list (sample indices, kept sorted)."""
        return self.picker.picks if self.picker is not None else []

    # ------------------------------------------------------------------ ui
    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        box = ttk.LabelFrame(self, text="Events")
        box.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Label(box, text="X Data:").grid(row=0, column=0, padx=4, pady=3, sticky="e")
        self.x_combo = ttk.Combobox(box, state="readonly", width=22)
        self.x_combo.grid(row=0, column=1, padx=4, pady=3, sticky="w")
        ttk.Label(box, text="Y Data:").grid(row=0, column=2, padx=4, pady=3, sticky="e")
        self.y_combo = ttk.Combobox(box, state="readonly", width=22)
        self.y_combo.grid(row=0, column=3, padx=4, pady=3, sticky="w")
        ttk.Label(box, text="Event:").grid(row=0, column=4, padx=4, pady=3, sticky="e")
        self.event_combo = ttk.Combobox(box, state="readonly", width=6)
        self.event_combo.grid(row=0, column=5, padx=(4, 0), pady=3, sticky="w")
        self.prev_button = ttk.Button(box, text="◀", width=2, command=lambda: self.step_event(-1))
        self.prev_button.grid(row=0, column=6, padx=(2, 0), pady=3)
        self.next_button = ttk.Button(box, text="▶", width=2, command=lambda: self.step_event(+1))
        self.next_button.grid(row=0, column=7, padx=(0, 4), pady=3)
        self.save_button = ttk.Button(box, text="Save picks", command=self.save_picks)
        self.save_button.grid(row=0, column=8, padx=10, pady=3, sticky="ew")

        half = float(getattr(getattr(self.app, "config", None), "DEFAULT_WINDOW_SIZE", DEFAULT_HALF_WINDOW_S))
        ttk.Label(box, text="Start (s):").grid(row=1, column=0, padx=4, pady=3, sticky="e")
        self.start_var = tk.StringVar(master=self, value=f"{-half:g}")
        self.start_entry = ttk.Spinbox(box, textvariable=self.start_var, width=7, from_=-1e6, to=1e6,
                                       increment=STEP_S, command=self.plot)
        self.start_entry.grid(row=1, column=1, padx=4, pady=3, sticky="w")
        ttk.Label(box, text="End (s):").grid(row=1, column=2, padx=4, pady=3, sticky="e")
        self.end_var = tk.StringVar(master=self, value=f"{half:g}")
        self.end_entry = ttk.Spinbox(box, textvariable=self.end_var, width=7, from_=-1e6, to=1e6,
                                     increment=STEP_S, command=self.plot)
        self.end_entry.grid(row=1, column=3, padx=4, pady=3, sticky="w")
        ttk.Button(box, text="Zoom to event", command=self.zoom_to_event).grid(row=1, column=4, padx=4, pady=3)
        ttk.Button(box, text="Whole run", command=self.show_whole_run).grid(
            row=1, column=5, columnspan=3, padx=4, pady=3, sticky="w")
        self.extract_button = ttk.Button(box, text="Extract Events", command=self.extract)
        self.extract_button.grid(row=1, column=8, padx=10, pady=3, sticky="ew")
        ttk.Label(box, text="Waveform channels:").grid(row=4, column=0, padx=4, pady=(0, 3), sticky="e")
        self.fields_combo = ttk.Combobox(box, width=22, values=[FIELDS_REFERENCE, FIELDS_ALL])
        self.fields_combo.set(FIELDS_REFERENCE)
        self.fields_combo.grid(row=4, column=1, columnspan=2, padx=4, pady=(0, 3), sticky="w")
        ttk.Label(box, text="into waveform/<recorder>: the reference's event_fields, all, or names (comma-separated)",
                  wraplength=560, justify="left").grid(row=4, column=3, columnspan=6, padx=4, pady=(0, 3), sticky="w")
        ttk.Label(box, text=HELP_TEXT, wraplength=900, justify="left").grid(
            row=2, column=0, columnspan=9, padx=4, pady=(0, 3), sticky="w")
        self.status_var = tk.StringVar(master=self, value="")
        ttk.Label(box, textvariable=self.status_var).grid(row=3, column=0, columnspan=9, padx=4, pady=(0, 3), sticky="w")

        self.x_combo.bind("<<ComboboxSelected>>", lambda e: self.plot())
        self.y_combo.bind("<<ComboboxSelected>>", lambda e: (self.plot(), self.update_status()))
        self.event_combo.bind("<<ComboboxSelected>>", lambda e: self.on_event_selected())
        for entry in (self.start_entry, self.end_entry):
            entry.bind("<Return>", self.on_window_edited)
            entry.bind("<FocusOut>", self.on_window_edited)

        self.make_figure(figsize=(10, 5), row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.ax = self.figure.add_subplot(111)
        self.picker = PointPicker(self.ax, self.canvas, [], [], [], add_remove=True,
                                  toolbar_active=self.toolbar_active, on_change=self.on_picks_changed,
                                  on_release_hook=self.flush_replot)

    def on_run_loaded(self):
        self.time = np.asarray(self.run.get("time", []), dtype=float).ravel()
        n = self.time.size
        self.candidates = list(aligned_fields(self.run, n, recurse=False)) if n else []
        self.source_series = self.converted_channels() if n else {}
        y_candidates = [c for c in self.candidates if c != "time"] + list(self.source_series)
        self.x_combo.config(values=[X_INDEX] + self.candidates)
        self.y_combo.config(values=y_candidates)
        saved = self.load_results() or self.legacy_results()
        x = saved.get("x_field") if saved.get("x_field") in [X_INDEX] + self.candidates else (
            "time" if "time" in self.candidates else X_INDEX)
        if self.y_field_hint in y_candidates:
            y = self.y_field_hint
        elif saved.get("y_field") in y_candidates:
            y = saved["y_field"]
        else:
            y = "shear_stress" if "shear_stress" in y_candidates else (y_candidates[0] if y_candidates else "")
        self.x_combo.set(x)
        self.y_combo.set(y)
        if "start_s" in saved and "end_s" in saved:
            self.start_var.set(f"{float(saved['start_s']):g}")
            self.end_var.set(f"{float(saved['end_s']):g}")
        self.fields_combo.set(self.fields_text(saved.get("waveform_fields")))
        self.picker.picks[:] = self.saved_picks()
        self.refresh_event_list(0)
        self.plot()
        self.update_status()

    # ----------------------------------------------------------- settings
    def legacy_results(self) -> dict:
        """The record under its former name, for files saved before the rename."""
        value = self.run.get(self.legacy_result_key)
        return dict(value) if isinstance(value, dict) else {}

    def saved_picks(self) -> list:
        """The run's picks as sorted ints within the record (from the
        extraction record, or the legacy top-level ``event_indices``)."""
        if self.time.size == 0:
            return []
        return sorted({i for i in picked_indices(self.run) if 0 <= i < self.time.size})

    def window_s(self) -> tuple[float, float]:
        """``(start, end)`` in seconds relative to the event; ValueError on bad input."""
        try:
            start, end = float(self.start_var.get()), float(self.end_var.get())
        except ValueError:
            raise ValueError("start and end must be numbers (seconds)") from None
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError("start and end must be finite")
        if start >= end:
            raise ValueError("start must be before end")
        return start, end

    def x_values(self) -> tuple[np.ndarray, str]:
        choice = self.x_combo.get()
        if choice and choice != X_INDEX:
            arr = get_field(self.run, choice)
            if arr is not None and np.asarray(arr).shape == self.time.shape:
                return np.asarray(arr, dtype=float), choice
        return np.arange(self.time.size, dtype=float), "sample index"

    def converted_channels(self) -> dict:
        """``{'<recorder>/<channel>': (source, channel)}`` for every raw-data
        source read through a conversion and placed on the run clock."""
        sources = self.data_manager.event_processor.open_sources(self.run)
        return {f"{key}/{ch}": (src, ch) for key, src in sources.items()
                if isinstance(src, ConvertedSource) for ch in src.fields}

    def y_label(self, y_field: str) -> str:
        if y_field in self.source_series:
            unit = self.source_series[y_field][0].unit
            return f"{y_field} ({unit})" if unit else y_field
        return y_field

    def source_values(self, y_field: str) -> np.ndarray:
        """A converted source channel on the run's samples (read once, then cached)."""
        src, ch = self.source_series[y_field]
        key = (str(src.path), ch, src.time_offset, self.time.size)
        if key not in _SOURCE_SERIES:
            self.status_var.set(f"reading {y_field} from {src.path.name} ...")
            self.config(cursor="watch")
            self.update_idletasks()
            try:
                _SOURCE_SERIES[key] = mean_on_samples(src, ch, self.time)
            finally:
                self.config(cursor="")
        return _SOURCE_SERIES[key]

    def y_values(self) -> Optional[np.ndarray]:
        y_field = self.y_combo.get()
        if y_field in self.source_series:
            return self.source_values(y_field)
        y = get_field(self.run, y_field) if y_field else None
        if y is None or np.asarray(y).shape != self.time.shape:
            return None
        return np.asarray(y, dtype=float)

    def selected_event(self) -> Optional[int]:
        text = self.event_combo.get().strip()
        return int(text) if text.isdigit() and int(text) < len(self.picks) else None

    def index_range(self, idx: int, start: float, end: float) -> tuple[int, int]:
        """Sample range ``[beg, end]`` of the window around pick ``idx`` (the
        processor's rule: nearest samples to event_time + start / + end)."""
        t0 = self.time[idx]
        beg = int(np.argmin(np.abs(t0 + start - self.time)))
        stop = int(np.argmin(np.abs(t0 + end - self.time)))
        return beg, stop

    def refresh_event_list(self, select: Optional[int]) -> None:
        """Rebuild the event combobox for the current picks and select one."""
        n = len(self.picks)
        self.event_combo.config(values=[str(i) for i in range(n)])
        if n:
            if select is None:
                select = self.selected_event() or 0
            self.event_combo.current(min(max(select, 0), n - 1))
            self.extract_button.state(["!disabled"])
        else:
            self.event_combo.set("")
            self.extract_button.state(["disabled"])

    def dirty(self) -> bool:
        """True when the picks differ from the saved ones."""
        return self.picks != self.saved_picks()

    def update_status(self, note: str = "") -> None:
        n = len(self.picks)
        parts = [f"{n} pick{'s' if n != 1 else ''}" + (" (unsaved)" if self.dirty() else "")]
        k = self.selected_event()
        if k is not None:
            parts.append(f"event {k} at t = {self.time[self.picks[k]]:.3f} s")
        n_events = len(event_list(self.run))
        parts.append(f"{n_events} event{'s' if n_events != 1 else ''} extracted")
        if note:
            parts.append(note)
        self.status_var.set("   ".join(parts))

    # ------------------------------------------------------------- plotting
    def on_window_edited(self, event=None):
        """Replot only when the start/end text changed since the last plot.

        Bound to Return and FocusOut: clicking the canvas moves the focus
        away from a spinbox, and an unconditional replot there would rebuild
        the markers under a drag that is just starting."""
        if (self.start_var.get(), self.end_var.get()) != self._plotted_window:
            self.plot()

    def flush_replot(self):
        """Run the replot that was postponed while a marker was held."""
        if self._replot_pending:
            self.plot()

    def plot(self):
        if self.picker is not None and self.picker.current_artist is not None:
            self._replot_pending = True              # a marker is held: redraw after the release
            return
        self._replot_pending = False
        self._plotted_window = (self.start_var.get(), self.end_var.get())
        x_field, y_field = self.x_combo.get(), self.y_combo.get()
        keep_x = self._shown.get("x") == x_field
        keep_y = keep_x and self._shown.get("y") == y_field
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        self.spans = []
        self.ax.grid(True, linestyle="--", alpha=0.3)
        x, x_label = self.x_values()
        y = self.y_values()
        if y is None:
            self.ax.set_title(f"{self.figure_title()}: select a Y array")
            self.picker.set_curve(x, np.full(x.shape, np.nan))
            self.picker.draw_markers()
            self._shown = {"x": x_field, "y": y_field}
            self.canvas.draw_idle()
            return
        self.ax.plot(x, y, lw=0.7, color="C0", zorder=1)
        try:
            start, end = self.window_s()
        except ValueError as e:
            start = end = None
            self.status_var.set(str(e))
        selected = self.selected_event()
        for k, idx in enumerate(self.picks):
            self.ax.axvline(x[idx], color="k", ls=":", lw=0.8, zorder=2)
            if start is not None:
                beg, stop = self.index_range(idx, start, end)
                is_sel = k == selected
                self.spans.append(self.ax.axvspan(x[beg], x[stop], color="#33CC66" if is_sel else "0.5",
                                                  alpha=0.35 if is_sel else 0.12, zorder=0))
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(self.y_label(y_field))
        self.ax.set_title(self.figure_title())
        if keep_x:
            self.ax.set_xlim(xlim)
        if keep_y:
            self.ax.set_ylim(ylim)
        self._shown = {"x": x_field, "y": y_field}
        self.picker.set_curve(x, y)
        self.picker.draw_markers()
        self.canvas.draw_idle()

    def figure_title(self) -> str:
        return f"{self.experiment_name()} run{self.run_idx:02d}"

    def on_picks_changed(self, kind: str, idx: int) -> None:
        """A pick was added, removed or moved: keep the list sorted, follow it."""
        self.picker.sort()
        select = self.picks.index(idx) if kind in ("add", "move") and idx in self.picks else None
        self.refresh_event_list(select)
        self.plot()
        self.update_status()

    def on_event_selected(self):
        self.plot()
        self.zoom_to_event()
        self.update_status()

    def step_event(self, delta: int):
        """Select the previous (-1) or next (+1) event, staying within the list."""
        if not self.picks:
            return
        k = self.selected_event()
        k = 0 if k is None else min(max(k + delta, 0), len(self.picks) - 1)
        self.event_combo.current(k)
        self.on_event_selected()

    def zoom_to_event(self):
        k = self.selected_event()
        if k is None:
            return
        try:
            start, end = self.window_s()
        except ValueError:
            return
        x, _ = self.x_values()
        beg, stop = self.index_range(self.picks[k], ZOOM_FACTOR * start, ZOOM_FACTOR * end)
        lo, hi = sorted((float(x[beg]), float(x[stop])))
        if hi > lo:
            self.ax.set_xlim(lo, hi)
            y = self.y_values()
            if y is not None:
                seg = y[min(beg, stop):max(beg, stop) + 1]
                seg = seg[np.isfinite(seg)]
                if seg.size and seg.max() > seg.min():
                    pad = 0.05 * (seg.max() - seg.min())
                    self.ax.set_ylim(seg.min() - pad, seg.max() + pad)
            self.picker.update_marker_size()
            self.canvas.draw_idle()

    def show_whole_run(self):
        self.ax.autoscale(enable=True, axis="both")
        self.ax.relim()
        self.ax.autoscale_view()
        self.picker.update_marker_size()
        self.canvas.draw_idle()

    # ------------------------------------------------------------- saving
    def save_picks(self, refresh: bool = True) -> list:
        """Write the picks (a sorted copy) into the extraction record, whose
        other entries are kept; the legacy top-level keys are removed."""
        self.picker.sort()
        picks = list(self.picks)
        record = dict(self.load_results() or self.legacy_results())
        record.setdefault("version", RESULT_VERSION)
        record["event_indices"] = list(picks)
        self.data_manager.set_data(self.result_path, record, True)
        self.run[self.result_key] = record
        for legacy in (LEGACY_INDICES_KEY, self.legacy_result_key):
            self.run.pop(legacy, None)
        if refresh:
            self.app.refresh_tree()
            self.update_status("picks saved")
        return picks

    @staticmethod
    def fields_text(value) -> str:
        """The box text for a saved ``waveform_fields`` (None / 'reference', 'all', or a list)."""
        if value is None:
            return FIELDS_REFERENCE
        if isinstance(value, str):
            key = value.strip().lower()
            return FIELDS_REFERENCE if key in ("", "reference") else (FIELDS_ALL if key == "all" else value)
        try:
            return ", ".join(str(v.decode() if isinstance(v, bytes) else v) for v in value)
        except TypeError:
            return FIELDS_REFERENCE

    def waveform_fields(self):
        """What the Waveform channels box asks for: None (the reference's
        ``event_fields``), ``'all'`` or a list of names."""
        choice = self.fields_combo.get().strip()
        if not choice or choice == FIELDS_REFERENCE or choice.lower() == "reference":
            return None
        if choice == FIELDS_ALL or choice.lower() == "all":
            return "all"
        return [f.strip() for f in choice.replace(";", ",").split(",") if f.strip()]

    def extract(self, confirm: bool = True) -> int:
        """Save the picks, extract every one with the current window; returns the count."""
        try:
            start, end = self.window_s()
        except ValueError as e:
            self.status_var.set(str(e))
            return 0
        if not self.picks:
            self.status_var.set("no picks: double-click on the curve to add events first")
            return 0
        if confirm and event_list(self.run):
            if not messagebox.askokcancel(title="Confirmation",
                                          message=f'This will replace all data in "{self.run_path}/events".',
                                          icon=messagebox.WARNING, parent=self):
                return 0
        picks = self.save_picks(refresh=False)
        fields = self.waveform_fields()
        events = self.data_manager.event_processor.extract_events(
            self.run, picks, window=max(-start, end), pre=-start, post=end, fields=fields)
        self.data_manager.set_data(f"{self.run_path}/events", events, add_key=True)
        self.result = {
            "version": RESULT_VERSION,
            "event_indices": list(picks),
            "start_s": float(start),
            "end_s": float(end),
            "x_field": self.x_combo.get(),
            "y_field": self.y_combo.get(),
            "n_events": len(events),
            "waveform_fields": "reference" if fields is None else fields,
        }
        self.save_results(dict(self.result))      # also refreshes the tree
        self.update_status(f"{len(events)} events extracted, window {start:+g} s .. {end:+g} s")
        return len(events)
