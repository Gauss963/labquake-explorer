"""Pick rupture-front arrival times on one full-rate record of an event.

The figure stacks the four mechanical traces (which can be hidden) above the
record's waveforms (strain gauges or PZTs), each drawn at its sensor position
along the fault, on a time axis in ms from the event time.  One handle per shown channel, a
short vertical bar, marks the arrival: it is dragged along the time axis
only and snaps to a sample, or the arrival of the selected channel is typed
in ms.  A straight line through the picks of the *fitting* channels gives
the rupture speed ``Cf``.  The rupture front through the same picks is fitted
as in the Slip Distribution view (:func:`slip_profile.fit_front`): one line,
or ``bilateral``, two lines meeting at a fitted apex; each segment is drawn
with its speed, a segment the front crosses within one sample is marked
unresolved, and the table gives every shown channel's residual from it.
Each trace is drawn from its baseline (the median of the noise window) at
its sensor's position; a channel without a position (``NaN`` in the record's
``positions``, such as a pressure or LVDT channel) is hidden and not fitted,
and drawn past the last sensor when shown.

The trace filter (:mod:`labquake_explorer.analysis.trace_filter`) is off,
Savitzky-Golay, or a Butterworth low-, high- or band-pass, zero phase or
causal; the picks are made, by hand or automatically, on the traces as
drawn, and the filter is saved with them.  "Auto pick" places the shown
channels' picks (:mod:`labquake_explorer.analysis.arrival_picks`): the onset
of the largest difference from the noise level, or of the largest slope,
above a threshold in multiples of the noise, or the peak, the largest
departure from the noise level (not picked either when below the
threshold).  The noise is the record before its trigger (or, without one,
before the event time), less a tenth of that span, else its first fifth.  A
search window in ms from the event ("Use the range shown" takes the axis
limits) keeps the picks on the burst wanted when a record holds several;
the noise then ends no later than its start.  Channels it cannot pick are
flagged in the table; a pick moved by hand clears its flag.  "Fit the
patch" fits the sensors on the bare patch (``analysis['fault']['patch_x_mm']``)
and only those.  An event without picks in the set starts from the set's
last save on that kind of record, else with the channels shown and fitted
on the event shown before it.

The controls sit in a scrolling column on the left: the event and the record,
a table of the channels (name, position, arrival, whether it is shown and
whether it is fitted; a click on the show or fit cell toggles it, a click on
the row selects the channel), the view (mechanical traces, zoom), the trace
filter, Save and a status line that says whether the picks differ from the
saved ones.

A Record selector chooses which record is picked on, and the results belong
to that record (``read_arrivals``/``write_arrivals`` in
:mod:`labquake_explorer.data.sources`): ``event['waveform'][recorder]['arrivals']``
holds ``picked_idx``, ``arrival_time``, ``enabled_channels``,
``fitting_channels``, ``locations`` (``NaN`` for a channel without a position),
``filter`` (the trace filter's settings) and ``rupture_speed`` (absent when the
picks define no line), so strain and PZT picks never overwrite each other.
The PSU ``strain`` layout keeps its historical keys (``strain/original/picked_idx``,
``event['rupture_speed']``, ...).  ``rupture_speed`` keeps the sign of the
published analyses, ``-1 / slope`` of ``time = slope * location``.  The record
also holds ``front_mode`` and ``front`` (``FrontFit.as_dict()`` plus ``use``,
the channels fitted), whose segment speeds are positive towards +x as in
``slip_analysis``; the PSU layout keeps these two under ``strain/arrivals``.

A record keeps named pick sets (hand picks, onset picks, peak picks, the
picks of another burst), chosen or named in the Pick set box by Save: a set
saved with the record loads its picks, filter and settings; a new name keeps
the picks shown, and Save writes them under it.  The set saved last is the
record's current one, which the CZM fitter and PZT Spectrum read, and, as in
PZT Spectrum, the experiment's (``analysis['pick_arrivals']['current_set']``),
which a new window opens on; the window keeps its set when it steps to
another event.
Save also keeps, for the experiment, what the set was saved with on each
kind of record (``analysis['pick_arrivals']['sets'][<set>]['records'][<recorder>]``:
the filter, the auto-pick entries, the front mode, and the channels shown
and fitted by name), so that an event without picks in that set starts from
it, and choosing the set on such an event takes its controls but keeps the
picks shown.
``choose_pick_set(name)`` and ``pick_set`` drive it headlessly.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
import warnings
from typing import Optional

import numpy as np

from labquake_explorer.analysis import arrival_picks as ap
from labquake_explorer.analysis import slip_profile as sprof
from labquake_explorer.analysis import trace_filter as tf
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT
from labquake_explorer.ui.views.base import EventView
from labquake_explorer.ui.views.scroll_panel import ScrollPanel
from labquake_explorer.ui.views.slip_distribution_view import apex_text, segment_speed_label, segment_text
from labquake_explorer.data.channels import aligned_fields, get_field
from labquake_explorer.data.settings import get_setting, set_setting
from labquake_explorer.data.sources import (DEFAULT_ARRIVAL_SET, arrival_set_name, arrival_sets, current_arrival_set,
                                            pick_waveform_block, read_arrivals, waveform_block, waveform_blocks,
                                            waveform_channels, waveform_data, waveform_time, waveform_trigger_time,
                                            write_arrivals)

# Experiments from this number on use the 16-channel layout with three
# extra gauges at the ends of the fault; older ones use paired gauges.
NEW_LAYOUT_EXP_NUMBER = 5958

# Savitzky-Golay polynomial order; the window must be longer than this.
FILTER_POLYORDER = tf.SAVGOL_POLYORDER
MIN_FILTER_WINDOW = FILTER_POLYORDER + 1
DEFAULT_FILTER_WINDOW = 51            # samples: the window a record opens with, shown in ms
DEFAULT_LOW_CORNER_FRACTION = 1e-3    # Butterworth corners a record opens with, as shares of its sample rate
DEFAULT_HIGH_CORNER_FRACTION = 0.1
DEFAULT_TRACE_AMPLITUDE_MM = 12.0     # trace amplitude when the sensor spacing is unknown
ZOOM_MARGIN_FRACTION = 0.5            # "Zoom to picks": this share of the picks' spread on each side ...
ZOOM_MIN_HALFWIDTH_SAMPLES = 50       # ... and at least this many samples
TEXT_WRAP_PX = 280
HANDLE_PX = 6                         # a press this close (px) to a pick handle grabs it
HANDLE_INSET_FRACTION = 0.1           # a handle spans its trace's height less this share at each end
CHECK = "✓"
TABLE_COLUMNS = (("channel", 70), ("x (mm)", 60), ("arrival (ms)", 85), ("show", 42), ("fit", 36),
                 ("resid (\u00b5s)", 70), ("flag", 90))
FLAG_SHORT = {ap.FLAG_WEAK: "weak onset", ap.FLAG_AT_NOISE_END: "at noise end", ap.FLAG_NO_NOISE: "no noise",
              ap.FLAG_AT_SEARCH_START: "at search start"}
FRONT_COLOR = "red"
NOISE_TRIGGER_GUARD_FRACTION = 0.1    # auto pick: the noise ends this share of the pre-trigger record before it
NOISE_FRACTION = 0.2                  # ... or, without a trigger, after this share of the record
SETTING = "pick_arrivals"              # analysis['pick_arrivals']['sets'][<set>]: what a new event starts from
CURRENT_SET_SETTING = "current_set"    # analysis['pick_arrivals']['current_set']: the set saved last, where a window opens
MECHANICAL = (("shear_stress", r"$\tau$ (MPa)"), ("friction", r"$\mu$"),
              ("LP_displacement", r"$\delta_\mathrm{LP}\ \mathrm{({\mu}m)}$"), ("slip", r"$\delta\ \mathrm{({\mu}m)}$"))

_RankWarning = getattr(np, "exceptions", np).RankWarning


def _ms(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}" if value is not None and math.isfinite(value) else ""


@register_view("Pick Arrivals", kinds=[EVENT], order=20)
class DynamicStrainArrivalPickerView(EventView):
    window_title = "Pick Arrivals"
    #: whether the mechanical traces are shown; a change applies to the windows opened after it
    show_mechanical_default = True

    def __init__(self, app, run_idx, event_idx):
        self.enabled_channels = None
        self.fitting_channels = None
        self.picked_idx = None
        self.lines = []
        self.axs = None
        self.ax_wave = None
        self.fitting_markers = []
        self.not_fitting_markers = []
        self._drag = None                   # the channel whose handle is being dragged
        self.front_fit = None
        self.front_lines = []
        self.auto_flags = []                # per channel: the auto pick's flag, "" when none
        self.auto_settings = None           # the settings of the last auto pick, saved with the picks
        self.rupture_speed = None
        self.fitted_line = None
        self.xlim = None
        self.record_key = None
        self.locations = []
        self.channel = 0                    # the channel selected in the table
        self.saved_state = None             # (picks, shown, fitting) as saved for this record, or None
        self.loaded_state = None            # the same, as loaded (what an unsaved record started from)
        self._defaults = {}                 # the record's default for each entry, to tell an edited one
        self._shown_names = []              # the channel names of the record last drawn
        self.pick_set = None                # the pick set shown and saved into (chosen on the first load)
        self._filtered = None               # (key, traces, note): the filtered record, kept while it applies
        self.placed = []                    # per channel: whether the record gives its position
        self._filling_table = False
        super().__init__(app, run_idx, event_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.left_panel = ScrollPanel(self)
        self.left_panel.grid(row=0, column=0, padx=5, pady=5, sticky="nsw")
        left = self.left_panel.inner

        frame = ttk.LabelFrame(left, text="Event")
        frame.pack(fill="x", pady=(0, 4))
        self.build_event_selector(frame, row=0, column=0)
        arrows = ttk.Frame(frame)
        arrows.grid(row=0, column=2, padx=2, sticky="w")
        self.prev_event_button = ttk.Button(arrows, text="◀", width=2, command=lambda: self.step_event_asked(-1))
        self.prev_event_button.pack(side="left")
        self.next_event_button = ttk.Button(arrows, text="▶", width=2, command=lambda: self.step_event_asked(+1))
        self.next_event_button.pack(side="left")
        ttk.Label(frame, text="Record:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.record_combobox = ttk.Combobox(frame, width=16, state="readonly")
        self.record_combobox.grid(row=1, column=1, columnspan=2, padx=5, pady=2, sticky="w")
        self.record_combobox.bind("<<ComboboxSelected>>", self._record_chosen)

        frame = ttk.LabelFrame(left, text="Channels")
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text="click show or fit to toggle it; a row selects the channel",
                  foreground="#606060", wraplength=TEXT_WRAP_PX, justify="left").grid(
            row=0, column=0, columnspan=2, padx=4, sticky="w")
        self.table = ttk.Treeview(frame, columns=[c for c, _ in TABLE_COLUMNS], show="headings", height=10,
                                  selectmode="browse")
        for name, width in TABLE_COLUMNS:
            self.table.heading(name, text=name)
            self.table.column(name, width=width, minwidth=width, anchor="center", stretch=False)
        self.table.grid(row=1, column=0, padx=(4, 0), pady=2, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        scroll.grid(row=1, column=1, padx=(0, 4), pady=2, sticky="ns")
        self.table.configure(yscrollcommand=scroll.set)
        self.table.bind("<ButtonRelease-1>", self.on_table_click, add="+")
        self.table.bind("<<TreeviewSelect>>", self.on_table_select, add="+")
        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=2, padx=2, pady=2, sticky="w")
        ttk.Button(buttons, text="Show all", command=lambda: self.set_all_shown(True)).pack(side="left", padx=2)
        ttk.Button(buttons, text="Fit all shown", command=self.fit_all_shown).pack(side="left", padx=2)
        ttk.Button(buttons, text="Fit the patch", command=self.fit_the_patch).pack(side="left", padx=2)
        row = ttk.Frame(frame)
        row.grid(row=3, column=0, columnspan=2, padx=2, pady=2, sticky="w")
        self.channel_label_var = tk.StringVar(master=self, value="")
        ttk.Label(row, textvariable=self.channel_label_var, width=8).pack(side="left", padx=2)
        ttk.Button(row, text="\u25c0", width=2, command=lambda: self.step_channel(-1)).pack(side="left")
        ttk.Button(row, text="\u25b6", width=2, command=lambda: self.step_channel(+1)).pack(side="left")
        ttk.Label(row, text="arrival (ms):").pack(side="left", padx=(8, 2))
        self.arrival_var = tk.StringVar(master=self, value="")
        self.arrival_entry = ttk.Entry(row, textvariable=self.arrival_var, width=10)
        self.arrival_entry.pack(side="left", padx=2)
        for sequence in ("<Return>", "<FocusOut>"):
            self.arrival_entry.bind(sequence, self.on_arrival_typed, add="+")
        ttk.Label(frame, text="drag a pick along the time axis, or type the selected channel's arrival",
                  foreground="#606060", wraplength=TEXT_WRAP_PX, justify="left").grid(
            row=4, column=0, columnspan=2, padx=4, sticky="w")

        frame = ttk.LabelFrame(left, text="View")
        frame.pack(fill="x", pady=4)
        self.mechanical_var = tk.BooleanVar(master=self, value=type(self).show_mechanical_default)
        ttk.Checkbutton(frame, text="mechanical traces (stress, friction, slip)", variable=self.mechanical_var,
                        command=self.on_mechanical_toggled).grid(row=0, column=0, columnspan=2, padx=4, sticky="w")
        ttk.Button(frame, text="Zoom to picks", command=self.zoom_to_picks).grid(row=1, column=0, padx=4, pady=2,
                                                                              sticky="w")
        ttk.Button(frame, text="Whole record", command=self.show_whole_record).grid(row=1, column=1, padx=4,
                                                                                  pady=2, sticky="w")

        frame = ttk.LabelFrame(left, text="Trace filter")
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text="Filter:").grid(row=0, column=0, padx=4, pady=2, sticky="e")
        self.filter_kind_var = tk.StringVar(master=self, value=tf.KIND_LABELS[tf.KIND_OFF])
        self.filter_combobox = ttk.Combobox(frame, textvariable=self.filter_kind_var, state="readonly", width=22,
                                            values=[tf.KIND_LABELS[k] for k in tf.KINDS])
        self.filter_combobox.grid(row=0, column=1, padx=4, pady=2, sticky="w")
        self.filter_combobox.bind("<<ComboboxSelected>>", self.on_filter_changed, add="+")
        self.filter_rows = {}
        label = ttk.Label(frame, text="Window (ms):")
        self.filter_window_ms = tk.StringVar(master=self, value="")
        self.filter_window_entry = ttk.Entry(frame, textvariable=self.filter_window_ms, width=9)
        self.filter_rows["savgol"] = [(label, dict(row=1, column=0, padx=4, pady=2, sticky="e")),
                                      (self.filter_window_entry, dict(row=1, column=1, padx=4, pady=2, sticky="w"))]
        label = ttk.Label(frame, text="Corners (kHz):")
        corners = ttk.Frame(frame)
        self.filter_low_khz = tk.StringVar(master=self, value="")
        self.filter_high_khz = tk.StringVar(master=self, value="")
        self.filter_low_entry = ttk.Entry(corners, textvariable=self.filter_low_khz, width=8)
        self.filter_low_entry.pack(side="left")
        ttk.Label(corners, text="to").pack(side="left", padx=4)
        self.filter_high_entry = ttk.Entry(corners, textvariable=self.filter_high_khz, width=8)
        self.filter_high_entry.pack(side="left")
        order_label = ttk.Label(frame, text="Order:")
        order_row = ttk.Frame(frame)
        self.filter_order_var = tk.StringVar(master=self, value=str(tf.DEFAULT_BUTTER_ORDER))
        self.filter_order_entry = ttk.Entry(order_row, textvariable=self.filter_order_var, width=4)
        self.filter_order_entry.pack(side="left")
        self.zero_phase_var = tk.BooleanVar(master=self, value=True)
        ttk.Checkbutton(order_row, text="zero phase", variable=self.zero_phase_var,
                        command=self.on_filter_changed).pack(side="left", padx=(10, 0))
        self.filter_rows["butterworth"] = [(label, dict(row=2, column=0, padx=4, pady=2, sticky="e")),
                                           (corners, dict(row=2, column=1, padx=4, pady=2, sticky="w")),
                                           (order_label, dict(row=3, column=0, padx=4, pady=2, sticky="e")),
                                           (order_row, dict(row=3, column=1, padx=4, pady=2, sticky="w"))]
        for entry in (self.filter_window_entry, self.filter_low_entry, self.filter_high_entry,
                      self.filter_order_entry):
            for sequence in ("<Return>", "<FocusOut>"):
                entry.bind(sequence, self.on_filter_changed, add="+")
        self.filter_note_var = tk.StringVar(master=self, value="")
        ttk.Label(frame, textvariable=self.filter_note_var, foreground="#606060", wraplength=TEXT_WRAP_PX,
                  justify="left").grid(row=4, column=0, columnspan=2, padx=4, sticky="w")
        self.show_filter_rows()

        frame = ttk.LabelFrame(left, text="Auto pick")
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text="Method:").grid(row=0, column=0, padx=4, pady=2, sticky="e")
        self.auto_method_var = tk.StringVar(master=self, value=ap.METHOD_LABELS[ap.METHOD_DIFFERENCE])
        ttk.Combobox(frame, textvariable=self.auto_method_var, state="readonly", width=22,
                     values=[ap.METHOD_LABELS[m] for m in ap.METHODS]).grid(row=0, column=1, padx=4, pady=2,
                                                                             sticky="w")
        ttk.Label(frame, text="Threshold (x noise):").grid(row=1, column=0, padx=4, pady=2, sticky="e")
        self.auto_threshold_var = tk.StringVar(master=self, value=f"{ap.DEFAULT_THRESHOLD:g}")
        ttk.Entry(frame, textvariable=self.auto_threshold_var, width=8).grid(row=1, column=1, padx=4, pady=2,
                                                                            sticky="w")
        ttk.Label(frame, text="Slope smoothing (\u00b5s):").grid(row=2, column=0, padx=4, pady=2, sticky="e")
        self.auto_smoothing_var = tk.StringVar(master=self, value="")
        ttk.Entry(frame, textvariable=self.auto_smoothing_var, width=8).grid(row=2, column=1, padx=4, pady=2,
                                                                            sticky="w")
        ttk.Label(frame, text="Search (ms):").grid(row=3, column=0, padx=4, pady=2, sticky="e")
        search = ttk.Frame(frame)
        search.grid(row=3, column=1, padx=4, pady=2, sticky="w")
        self.search_from_var = tk.StringVar(master=self, value="")
        self.search_to_var = tk.StringVar(master=self, value="")
        ttk.Entry(search, textvariable=self.search_from_var, width=8).pack(side="left")
        ttk.Label(search, text="to").pack(side="left", padx=4)
        ttk.Entry(search, textvariable=self.search_to_var, width=8).pack(side="left")
        ttk.Button(frame, text="Use the range shown", command=self.use_range_shown).grid(
            row=4, column=1, padx=4, pady=2, sticky="w")
        ttk.Label(frame, text="from the event; blank: the whole record", foreground="#606060",
                  wraplength=TEXT_WRAP_PX, justify="left").grid(row=5, column=0, columnspan=2, padx=4, sticky="w")
        self.noise_note_var = tk.StringVar(master=self, value="")
        ttk.Label(frame, textvariable=self.noise_note_var, foreground="#606060", wraplength=TEXT_WRAP_PX,
                  justify="left").grid(row=6, column=0, columnspan=2, padx=4, sticky="w")
        ttk.Button(frame, text="Pick the shown channels", command=self.auto_pick).grid(
            row=7, column=0, columnspan=2, padx=4, pady=2, sticky="w")

        frame = ttk.LabelFrame(left, text="Rupture front")
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text="Fit:").grid(row=0, column=0, padx=4, pady=2, sticky="e")
        self.front_mode_var = tk.StringVar(master=self, value="unilateral")
        self.front_mode_combobox = ttk.Combobox(frame, textvariable=self.front_mode_var, state="readonly",
                                                width=10, values=list(sprof.FRONT_MODES))
        self.front_mode_combobox.grid(row=0, column=1, padx=4, pady=2, sticky="w")
        self.front_mode_combobox.bind("<<ComboboxSelected>>", lambda e: self.update_fitted_line(), add="+")
        ttk.Label(frame, text="unilateral: one line; bilateral: two lines that meet at a fitted apex",
                  foreground="#606060", wraplength=TEXT_WRAP_PX, justify="left").grid(
            row=1, column=0, columnspan=2, padx=4, sticky="w")
        self.front_text_var = tk.StringVar(master=self, value="")
        ttk.Label(frame, textvariable=self.front_text_var, wraplength=TEXT_WRAP_PX, justify="left").grid(
            row=2, column=0, columnspan=2, padx=4, pady=2, sticky="w")
        self.cf_label = ttk.Label(frame, text="Cf = n/a")
        self.cf_label.grid(row=3, column=0, columnspan=2, padx=4, pady=2, sticky="w")
        ttk.Label(frame, text="Cf: one line through the fitted picks, -1 / slope (the sign of the published "
                              "analyses; the CZM fitter takes its size)", foreground="#606060",
                  wraplength=TEXT_WRAP_PX, justify="left").grid(row=4, column=0, columnspan=2, padx=4, sticky="w")

        frame = ttk.Frame(left)
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text="Pick set:").grid(row=0, column=0, padx=4, pady=2, sticky="w")
        self.pick_set_combobox = ttk.Combobox(frame, width=16, values=[DEFAULT_ARRIVAL_SET])   # editable
        self.pick_set_combobox.grid(row=0, column=1, padx=4, pady=2, sticky="w")
        for sequence in ("<<ComboboxSelected>>", "<Return>", "<FocusOut>"):
            self.pick_set_combobox.bind(sequence, self._pick_set_chosen, add="+")
        self.save_button = ttk.Button(frame, text="Save", command=self.save)
        self.save_button.grid(row=0, column=2, padx=4, pady=2, sticky="w")
        ttk.Label(frame, text="a saved set loads its picks; a new name keeps the picks shown for Save",
                  foreground="#606060", wraplength=TEXT_WRAP_PX, justify="left").grid(
            row=1, column=0, columnspan=3, padx=4, sticky="w")
        self.status_var = tk.StringVar(master=self, value="")
        ttk.Label(left, textvariable=self.status_var, wraplength=TEXT_WRAP_PX, foreground="#a03030",
                  justify="left").pack(fill="x", pady=2)
        self.left_panel.bind_children()

        figure_frame = ttk.Frame(self)
        figure_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        figure_frame.grid_rowconfigure(0, weight=1)
        figure_frame.grid_columnconfigure(0, weight=1)
        self.make_figure(master=figure_frame, figsize=(7, 7), figure_kwargs={"layout": "constrained"},
                         row=0, column=0, sticky="nsew")
        self.fig = self.figure  # legacy alias

        self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.figure.canvas.mpl_connect("resize_event", self.on_resize)
        self.figure.canvas.mpl_connect("scroll_event", self.on_resize)

    # ------------------------------------------------------------- loading
    def exp_number(self) -> int:
        """Experiment number parsed from the name, e.g. 'p5958' -> 5958 (0 if unparsable)."""
        try:
            return int(self.experiment_name()[1:5])
        except ValueError:
            return 0

    @staticmethod
    def strain_block(event):
        """The record the view opens first: ``strain`` when present, else the
        first waveform block (``waveform/elsys``, ``waveform/ni``, ...); None without any."""
        if not isinstance(event, dict):
            return None
        key = pick_waveform_block(event, prefer_keys=("strain",))
        return waveform_block(event, key) if key else None

    def current_block(self):
        """The record selected in the Record combobox (None without any)."""
        return waveform_block(self.event, self.record_key) if self.record_key else None

    def _record_chosen(self, event=None) -> None:
        if self.record_combobox.get().strip() != self.record_key and not self.confirm_leave("open another record"):
            self.record_combobox.set(self.record_key or "")
            return
        self.on_record_selected(event)

    def _pick_set_chosen(self, event=None) -> None:
        """A saved set replaces the picks shown, so ask first; a new name keeps them."""
        typed = self.pick_set_combobox.get().strip()
        if (typed and typed != self.pick_set and typed in arrival_sets(self.current_block())
                and not self.confirm_leave(f"load pick set '{typed}'")):
            self.pick_set_combobox.set(self.pick_set)
            return
        self.on_pick_set_changed(event)

    def has_unsaved_edits(self) -> bool:
        """Picks, shown/fit choices or the front mode changed since the record was loaded or saved."""
        if self.picked_idx is None:
            return False
        reference = self.saved_state if self.saved_state is not None else self.loaded_state
        return reference is not None and self.current_state() != reference

    def on_record_selected(self, event=None):
        key = self.record_combobox.get().strip()
        if key and key != self.record_key:
            self.record_key = key
            self.load_record()

    def is_legacy_record(self) -> bool:
        block = self.current_block()
        return isinstance(block, dict) and "original" in block

    @staticmethod
    def fault_slip(event):
        """The event's fault slip trace: ``displacement`` (PSU files), else the
        first slip channel (``slip/slip_1``); None without either."""
        n = len(event.get("time", []))
        for path in ["displacement"] + [f for f in aligned_fields(event, n) if f.split("/")[-1].lower().startswith("slip")]:
            value = get_field(event, path)
            if value is not None and np.asarray(value).shape == (n,):
                return np.asarray(value, dtype=float)
        return None

    @staticmethod
    def block_locations(block, n_channels: int):
        """Sensor positions along the fault from the record's ``positions`` table
        (its ``x``, NaN for a channel without one), when it has an entry per
        channel and at least one is finite; else None."""
        positions = block.get("positions") if isinstance(block, dict) else None
        xs = positions.get("x") if isinstance(positions, dict) else None
        if xs is None or len(xs) != n_channels:
            return None
        xs = [float(v) for v in xs]
        return xs if any(math.isfinite(v) for v in xs) else None

    @classmethod
    def place_channels(cls, xs) -> tuple:
        """``(locations, placed)``: a channel without a position (NaN) is drawn
        past the last sensor, one sensor spacing apart, and ``placed`` is False for it."""
        xs = [float(v) for v in xs]
        placed = [math.isfinite(v) for v in xs]
        known = [v for v, ok in zip(xs, placed) if ok]
        if not known:
            return list(xs), placed
        spacing = cls.trace_amplitude(known)
        nxt = max(known)
        out = []
        for v, ok in zip(xs, placed):
            if not ok:
                nxt += spacing
                v = nxt
            out.append(v)
        return out, placed

    def saved_locations(self) -> list:
        """The positions as saved: NaN for the channels without one."""
        return [float(v) if ok else math.nan for v, ok in zip(self.locations, self.placed)]

    def event_has_record(self, event_idx: int) -> bool:
        try:
            candidate = self.data_manager.get_data(f"{self.run_path}/events/[{int(event_idx)}]")
        except (KeyError, IndexError, TypeError):
            return False
        return self.strain_block(candidate) is not None

    def set_event(self, event_idx: int) -> None:
        """Switch events, but refuse (in the status line) an event without a record.

        The check happens before ``EventView.set_event`` touches ``event_idx``,
        ``event``, the title or the combobox, so a refused switch leaves the
        view exactly as it was.
        """
        event_idx = int(event_idx)
        if not self.event_has_record(event_idx):
            self.refresh_event_selector()  # combobox may already show the refused index
            self.status_var.set(f"event {event_idx} has no full-rate record to pick arrivals on; "
                                f"staying on event {self.event_idx}")
            return
        super().set_event(event_idx)

    def step_event(self, step: int) -> None:
        """The previous / next event that has a full-rate record."""
        k = self.event_idx + step
        while 0 <= k < self.n_events():
            if self.event_has_record(k):
                self.set_event(k)
                return
            k += step
        self.status_var.set(f"no {'earlier' if step < 0 else 'later'} event with a full-rate record")

    def on_event_loaded(self):
        keys = list(waveform_blocks(self.event)) if isinstance(self.event, dict) else []
        if not keys:
            if self.axs is None:  # first load: nothing to show, do not leave a dead window
                self.on_close()
            raise ValueError(f"{self.event_path} has no full-rate record to pick arrivals on")
        if self.record_key not in keys:
            self.record_key = pick_waveform_block(self.event, prefer_keys=("strain",))
        self.record_combobox.config(values=keys)
        self.record_combobox.set(self.record_key)
        self.load_record()

    # ------------------------------------------------------------- template
    def recorder(self) -> str:
        """The record's short name for the template: ``elsys``, ``ni``, ``strain``."""
        return str(self.record_key).split("/")[-1]

    def template(self) -> Optional[dict]:
        """What this pick set was last saved with on this kind of record, or None."""
        value = get_setting(self.data_manager.data, SETTING, "sets", self.pick_set, "records", self.recorder())
        return value if isinstance(value, dict) else None

    def store_template(self) -> None:
        """What Save keeps for the experiment, per pick set and kind of record
        (``elsys``, ``ni``, ``strain``), so that an event without picks in that
        set starts from it: the filter, the auto-pick entries, the front mode and
        the names of the channels shown and fitted."""
        names = self.channel_names()
        entry = {"filter": self.saved_filter(), "front_mode": self.front_mode_var.get(),
                 "shown": [n for n, v in zip(names, self.enabled_channels) if v],
                 "fit": [n for n, v in zip(names, self.fitting_channels) if v]}
        try:
            auto = self.read_auto_settings()
            entry["auto_pick"] = {"method": auto["method"], "threshold": auto["threshold"],
                                  "smoothing_us": float(self.auto_smoothing_var.get()),
                                  **{k: auto[k] for k in ("search_from_s", "search_to_s") if k in auto}}
        except ValueError:
            pass
        set_setting(self.data_manager.data, entry, SETTING, "sets", self.pick_set, "records", self.recorder())

    def apply_template(self, n: int) -> bool:
        """The set's last saved controls for this kind of record, and its shown
        and fitted channels by name; False (nothing changed) without one."""
        entry = self.template()
        if entry is None:
            return False
        self.restore_filter(entry.get("filter"))
        if entry.get("front_mode") is not None and sprof.front_mode_name(entry["front_mode"]) in sprof.FRONT_MODES:
            self.front_mode_var.set(sprof.front_mode_name(entry["front_mode"]))
        auto = entry.get("auto_pick")
        if isinstance(auto, dict):
            if auto.get("method") in ap.METHODS:
                self.auto_method_var.set(ap.METHOD_LABELS[auto["method"]])
            for key, var in (("threshold", self.auto_threshold_var), ("smoothing_us", self.auto_smoothing_var)):
                if auto.get(key) is not None:
                    var.set(f"{float(auto[key]):g}")
            for key, var in (("search_from_s", self.search_from_var), ("search_to_s", self.search_to_var)):
                var.set(f"{float(auto[key]) * 1e3:.4g}" if auto.get(key) is not None else "")
        names = self.channel_names()
        for key, attr in (("shown", "enabled_channels"), ("fit", "fitting_channels")):
            if entry.get(key) is not None:
                chosen = {str(v) for v in np.asarray(entry[key], dtype=object).ravel()}
                if not chosen or chosen & set(names):               # names of another layout do not apply
                    setattr(self, attr, [name in chosen for name in names][:n])
        return True

    # ------------------------------------------------------------ pick sets
    def initial_pick_set(self, block) -> str:
        """The set a new window opens on, as in PZT Spectrum: the one last saved
        in the experiment (``analysis['pick_arrivals']['current_set']``), else
        the record's current set, else ``default``."""
        try:
            return arrival_set_name(get_setting(self.data_manager.data, SETTING, CURRENT_SET_SETTING))
        except ValueError:
            return current_arrival_set(block) or DEFAULT_ARRIVAL_SET

    def known_pick_sets(self) -> list:
        """The sets to offer: the record's, the experiment's, the current one and ``default``."""
        names = set(arrival_sets(self.current_block())) | {DEFAULT_ARRIVAL_SET, self.pick_set}
        sets = get_setting(self.data_manager.data, SETTING, "sets")
        if isinstance(sets, dict):
            names.update(str(k) for k in sets)
        return [DEFAULT_ARRIVAL_SET] + sorted(names - {DEFAULT_ARRIVAL_SET})

    def refresh_pick_sets(self) -> None:
        self.pick_set_combobox.config(values=self.known_pick_sets())
        self.pick_set_combobox.set(self.pick_set)

    def on_pick_set_changed(self, event=None) -> None:
        """A set saved with this record loads its picks, filter and settings; a
        new name keeps the picks shown, and Save writes them under it."""
        try:
            name = arrival_set_name(self.pick_set_combobox.get())
        except ValueError as e:
            self.pick_set_combobox.set(self.pick_set)
            self.update_status(str(e))
            return
        if name == self.pick_set:
            return
        self.pick_set = name
        if name in arrival_sets(self.current_block()):
            xlim = self.ax_wave.get_xlim() if self.ax_wave is not None else None
            self.load_record()
            if xlim is not None:                       # keep the zoom, to compare sets on the same stretch
                self.ax_wave.set_xlim(xlim)
                self.on_resize()
            self.update_status(f"pick set '{name}' loaded")
        else:
            self.refresh_pick_sets()
            self.saved_state = None
            if self.apply_template(len(self.picked_idx)):        # the set's controls, the picks shown kept
                self.xlim = self.ax_wave.get_xlim()
                self.plot()
                self.update_fitted_line()
                self.update_status(f"pick set '{name}' not saved for this record: its last controls, "
                                   "the picks shown; Save writes them under it")
            else:
                self.update_status(f"new pick set '{name}': Save writes the picks shown under it")

    def choose_pick_set(self, name: str) -> None:
        """Choose the pick set by name, as typing it in the Pick set box does."""
        self.pick_set_combobox.set(name)
        self.on_pick_set_changed()

    def load_record(self):
        """Show the selected record with the picks of the chosen set, or the defaults."""
        block = self.current_block()
        if self.pick_set is None:
            self.pick_set = self.initial_pick_set(block)
        self.refresh_pick_sets()
        previous = None                                   # the channel choice of what was shown before
        if self.enabled_channels is not None and self.fitting_channels is not None and self._shown_names:
            previous = (list(self._shown_names), list(self.enabled_channels), list(self.fitting_channels))
        saved = read_arrivals(block, self.event, self.pick_set)
        n = int(waveform_data(block).shape[0])

        def flags(key):
            value = saved.get(key)
            return [bool(v) for v in value] if value is not None and len(value) == n else None

        self.enabled_channels = flags("enabled_channels")
        self.fitting_channels = flags("fitting_channels")
        picks = saved.get("picked_idx")
        self.picked_idx = [int(v) for v in picks] if picks is not None and len(picks) == n else None
        self.auto_flags = [""] * n
        self.auto_settings = None
        auto = saved.get("auto_pick")
        if isinstance(auto, dict):
            flags = auto.get("flags")
            if flags is not None and len(flags) == n:
                self.auto_flags = [str(np.asarray(f).ravel()[0]) if np.size(f) else "" for f in flags]
            self.auto_settings = {k: v for k, v in auto.items() if k != "flags"}
        self.apply_record_defaults()
        if self.picked_idx is None:                       # nothing saved in this set: start from the set's last save,
            if not self.apply_template(n) and previous is not None and previous[0] == self.channel_names():
                self.enabled_channels, self.fitting_channels = previous[1], previous[2]   # else what was shown
        else:
            self.restore_filter(saved.get("filter"))
            mode = saved.get("front_mode")
            mode = sprof.front_mode_name(mode) if mode is not None and np.size(mode) else "unilateral"
            self.front_mode_var.set(mode if mode in sprof.FRONT_MODES else "unilateral")
        locations = saved.get("locations")
        if locations is not None and len(locations) == n and np.isfinite(np.asarray(locations, float)).any():
            xs = [float(v) for v in locations]
        else:
            xs = self.block_locations(block, n) or self.default_locations(n, self.exp_number())
        self.locations, self.placed = self.place_channels(xs)

        self.table.configure(height=min(max(n, 4), 10))     # a row per channel, scrolling past ten
        self.xlim = None
        self.channel = min(self.channel, n - 1)
        self.fitting_markers = []
        self.not_fitting_markers = []
        self._drag = None
        self.rupture_speed = None
        self.fitted_line = None
        self.plot()
        self.noise_note_var.set(self.noise_end()[1])
        self.saved_state = self.current_state() if self.picked_idx is not None and picks is not None else None
        self.loaded_state = self.current_state() if self.picked_idx is not None else None
        self.update_fitted_line()

    # ------------------------------------------------------------ defaults
    @staticmethod
    def default_enabled_channels(n_channels: int, exp_number: int, legacy: bool = True) -> list:
        """PSU strain layouts show a chosen subset; any other record shows every channel."""
        if not legacy:
            return [True] * n_channels
        if exp_number >= NEW_LAYOUT_EXP_NUMBER:
            return [i < 13 for i in range(n_channels)]
        return [i % 2 == 0 for i in range(n_channels)]

    @staticmethod
    def default_fitting_channels(n_channels: int, exp_number: int, legacy: bool = True) -> list:
        if not legacy:
            return [True] * n_channels
        if exp_number >= NEW_LAYOUT_EXP_NUMBER:
            base = [False] + [True] * 8 + [False] * 7
        else:
            base = [i in (6, 8, 10, 12) for i in range(16)]
        return (base + [False] * n_channels)[:n_channels]

    def reference_time(self) -> tuple:
        """``(time_s, name)``: the record's trigger time when it has one inside
        the record (the waves arrive there), else the event time when inside it,
        else ``(None, "")``."""
        block = self.current_block()
        t = waveform_time(block)
        t_trig = waveform_trigger_time(block, self.get_run(), self.record_key)
        if t_trig is not None and t[0] < t_trig <= t[-1]:
            return float(t_trig), "trigger"
        t_event = float(self.event["event_time"])
        if t[0] < t_event <= t[-1]:
            return t_event, "event time"
        return None, ""

    def default_picks(self, block, n_channels: int, n_samples: int) -> list:
        """Initial marker positions: the record's trigger time, else the event
        time, else the middle of the record."""
        t_ref, _ = self.reference_time()
        if t_ref is not None:
            idx = int(np.argmin(np.abs(waveform_time(block) - t_ref)))
        else:
            idx = n_samples // 2
        return [idx] * n_channels

    @staticmethod
    def trace_amplitude(locations) -> float:
        """Peak-to-peak height of a drawn trace in mm: the sensor spacing, so
        neighbouring traces touch but do not cross; 12 mm when unknown."""
        distinct = np.unique(np.asarray(locations, dtype=float))
        if distinct.size < 2:
            return DEFAULT_TRACE_AMPLITUDE_MM
        spacing = float(np.median(np.diff(distinct)))
        return spacing if spacing > 0 else DEFAULT_TRACE_AMPLITUDE_MM

    @staticmethod
    def default_locations(n_channels: int, exp_number: int) -> list:
        """Gauge positions along the fault in mm."""
        if exp_number >= NEW_LAYOUT_EXP_NUMBER:
            locs = [2 + 12 * i for i in range(16)]
            locs[-3:] = [2, 74, 146]
            return (locs + [locs[-1]] * n_channels)[:n_channels]
        return [10.5 + 12 * (i // 2) for i in range(n_channels)]

    def channel_names(self) -> list:
        """The record's channel names (``pzt_1``, ...), else ``ch 0``, ``ch 1``, ..."""
        block = self.current_block()
        names = waveform_channels(block) if block is not None else []
        return [f"ch {n}" if str(n).isdigit() else str(n) for n in names]

    def sample_interval(self) -> float:
        t = waveform_time(self.current_block())
        return float(np.median(np.diff(t[: min(t.size, 2000)]))) if t.size > 1 else 1.0

    def relative_time_s(self) -> np.ndarray:
        """The record's time axis in s from the event time."""
        return waveform_time(self.current_block()) - float(self.event["event_time"])

    def pick_times_s(self) -> list:
        """Every channel's pick in s from the event time."""
        t = self.relative_time_s()
        return [float(t[int(i)]) for i in self.picked_idx]

    # -------------------------------------------------------------- filter
    def record_defaults(self) -> dict:
        """The entries' defaults for the record shown, by variable name."""
        dt = self.sample_interval()
        fs = 1.0 / dt

        def g(value):
            return f"{float(f'{value:.3g}'):g}"
        return {"filter_window_ms": g(DEFAULT_FILTER_WINDOW * dt * 1e3),
                "filter_low_khz": g(DEFAULT_LOW_CORNER_FRACTION * fs / 1e3),
                "filter_high_khz": g(DEFAULT_HIGH_CORNER_FRACTION * fs / 1e3),
                "auto_smoothing_var": g(ap.DEFAULT_SMOOTHING_SAMPLES * dt * 1e6)}

    def apply_record_defaults(self) -> None:
        """Put the record's defaults in the entries that are empty or still hold
        the previous record's default; an entry the user changed keeps its value."""
        new = self.record_defaults()
        for name, value in new.items():
            var = getattr(self, name)
            if var.get().strip() in ("", self._defaults.get(name)):
                var.set(value)
        self._defaults = new

    def filter_kind(self) -> str:
        label = self.filter_kind_var.get()
        return next((k for k in tf.KINDS if tf.KIND_LABELS[k] == label), tf.KIND_OFF)

    @property
    def filtering(self) -> bool:
        return self.filter_kind() != tf.KIND_OFF

    def show_filter_rows(self) -> None:
        """Only the entries the chosen filter uses."""
        kind = self.filter_kind()
        for name, widgets in self.filter_rows.items():
            used = (name == "savgol" and kind == tf.KIND_SAVGOL) or (name == "butterworth" and kind in tf.BUTTERWORTH)
            for widget, options in widgets:
                if used:
                    widget.grid(**options)
                else:
                    widget.grid_remove()
        self.filter_low_entry.configure(state="normal" if tf.uses_low(kind) else "disabled")
        self.filter_high_entry.configure(state="normal" if tf.uses_high(kind) else "disabled")

    def filter_settings(self) -> dict:
        """The Trace filter box as settings for :func:`trace_filter.filter_traces`
        (ValueError, with the reason, when unusable)."""
        kind = self.filter_kind()
        if kind == tf.KIND_OFF:
            return {"kind": kind}
        if kind == tf.KIND_SAVGOL:
            return {"kind": kind, "window_samples": self.filter_window(), "polyorder": FILTER_POLYORDER}
        try:
            order = int(self.filter_order_var.get())
        except ValueError:
            raise ValueError(f"order {self.filter_order_var.get()!r} is not a whole number") from None
        settings = {"kind": kind, "order": order, "zero_phase": bool(self.zero_phase_var.get())}
        for key, var, used in (("low_hz", self.filter_low_khz, tf.uses_low(kind)),
                               ("high_hz", self.filter_high_khz, tf.uses_high(kind))):
            if not used:
                continue
            try:
                settings[key] = float(var.get()) * 1e3
            except ValueError:
                raise ValueError(f"corner {var.get()!r} is not a frequency in kHz") from None
        tf.check(settings, 1.0 / self.sample_interval())
        return settings

    def restore_filter(self, saved) -> None:
        """Show the filter the record's picks were saved with (nothing without one)."""
        if not isinstance(saved, dict):
            return

        def value(key, default=None):
            v = saved.get(key, default)
            if v is None:
                return default
            v = np.asarray(v).ravel()
            return v[0].decode() if isinstance(v[0], bytes) else v[0].item() if hasattr(v[0], "item") else v[0]
        kind = str(value("kind", tf.KIND_OFF))
        if kind not in tf.KINDS:
            return
        self.filter_kind_var.set(tf.KIND_LABELS[kind])
        if kind == tf.KIND_SAVGOL and value("window_samples") is not None:
            self.filter_window_ms.set(f"{float(value('window_samples')) * self.sample_interval() * 1e3:.6g}")
        if kind in tf.BUTTERWORTH:
            self.filter_order_var.set(str(int(value("order", tf.DEFAULT_BUTTER_ORDER))))
            self.zero_phase_var.set(bool(value("zero_phase", True)))
            if value("low_hz") is not None:
                self.filter_low_khz.set(f"{float(value('low_hz')) / 1e3:.6g}")
            if value("high_hz") is not None:
                self.filter_high_khz.set(f"{float(value('high_hz')) / 1e3:.6g}")
        self.show_filter_rows()

    def filtered_traces(self) -> np.ndarray:
        """The record's traces through the filter; unfiltered, with the reason
        in the filter note, when the settings are unusable."""
        block = self.current_block()
        try:
            settings = self.filter_settings()
            problem = ""
        except ValueError as e:
            settings, problem = {"kind": tf.KIND_OFF}, str(e)
        key = (id(block), self.record_key, repr(sorted(settings.items())))
        if self._filtered is None or self._filtered[0] != key:
            traces = tf.filter_traces(waveform_data(block), 1.0 / self.sample_interval(), settings)
            self._filtered = (key, traces)
        self.update_filter_note(settings, problem)
        return self._filtered[1]

    def filter_window(self) -> int:
        """Savgol window in samples from the window typed in ms; at least
        polyorder + 1, the record's default (51 samples) when unparsable."""
        try:
            ms = float(self.filter_window_ms.get())
            if not (math.isfinite(ms) and ms > 0):
                raise ValueError
            n = int(round(ms * 1e-3 / self.sample_interval()))
        except (ValueError, tk.TclError):
            n = DEFAULT_FILTER_WINDOW
        return max(n, MIN_FILTER_WINDOW)

    def on_filter_changed(self, event=None) -> None:
        """Redraw the traces through the filter now set (the picks stay)."""
        self.show_filter_rows()
        if self.ax_wave is not None:
            self.xlim = self.ax_wave.get_xlim()
        self.plot()
        self.update_fitted_line()

    def set_filter(self, kind: str, **values) -> None:
        """Choose a filter (``trace_filter.KIND_*``) and, optionally, its entries:
        ``window_ms``, ``low_khz``, ``high_khz``, ``order``, ``zero_phase``."""
        self.filter_kind_var.set(tf.KIND_LABELS[kind])
        for key, var in (("window_ms", self.filter_window_ms), ("low_khz", self.filter_low_khz),
                         ("high_khz", self.filter_high_khz), ("order", self.filter_order_var)):
            if key in values:
                var.set(f"{values[key]:g}" if isinstance(values[key], (int, float)) else str(values[key]))
        if "zero_phase" in values:
            self.zero_phase_var.set(bool(values["zero_phase"]))
        self.on_filter_changed()

    def update_filter_note(self, settings: dict, problem: str = "") -> None:
        kind = self.filter_kind()
        nyquist = f"Nyquist {0.5 / self.sample_interval() / 1e3:g} kHz"
        if problem:
            text = f"not applied: {problem}"
        elif kind == tf.KIND_SAVGOL:
            text = f"= {settings['window_samples']} samples"
        elif kind in tf.BUTTERWORTH:
            text = nyquist + ("; zero phase spreads an onset earlier" if settings.get("zero_phase", True)
                              else "; causal: onsets delayed")
        else:
            text = nyquist
        self.filter_note_var.set(text)

    # ---------------------------------------------------------------- plot
    def on_mechanical_toggled(self) -> None:
        type(self).show_mechanical_default = bool(self.mechanical_var.get())
        if self.ax_wave is not None:
            self.xlim = self.ax_wave.get_xlim()
        self.plot()
        self.update_fitted_line()

    def plot(self):
        exp_number = self.exp_number()
        strain = self.current_block()
        legacy = self.is_legacy_record()
        linestyle = ".-"

        self.figure.clear()
        self.fitting_markers = []
        self.not_fitting_markers = []
        self.fitted_line = None
        self.front_lines = []
        if self.mechanical_var.get():
            gs = self.figure.add_gridspec(5, hspace=0, height_ratios=[1, 1, 1, 1, 10])
            self.axs = list(gs.subplots(sharex=True))
            event_time = float(self.event["event_time"])
            t = (np.asarray(self.event["time"], dtype=float) - event_time) * 1e3
            for ax, (key, label) in zip(self.axs, MECHANICAL):
                ax.set_ylabel(label)
                if key == "slip":
                    values = self.fault_slip(self.event)
                else:
                    values = self.event.get(key)
                if values is None:
                    continue
                values = np.asarray(values, dtype=float)
                if key in ("LP_displacement", "slip"):
                    values = values - values[0]
                ax.plot(t, values, linestyle, color="C0")
        else:
            self.axs = [self.figure.add_subplot(1, 1, 1)]
        self.ax_wave = self.axs[-1]

        tt = self.relative_time_s() * 1e3
        y = self.filtered_traces()
        n_channels = y.shape[0]

        if len(self.locations) != n_channels or len(self.placed) != n_channels:
            xs = self.block_locations(strain, n_channels) or self.default_locations(n_channels, exp_number)
            self.locations, self.placed = self.place_channels(xs)
        if self.enabled_channels is None or len(self.enabled_channels) != n_channels:
            self.enabled_channels = self.default_enabled_channels(n_channels, exp_number, legacy)
            if not legacy:
                self.enabled_channels = list(self.placed) if any(self.placed) else self.enabled_channels
        if self.fitting_channels is None or len(self.fitting_channels) != n_channels:
            self.fitting_channels = self.default_fitting_channels(n_channels, exp_number, legacy)
            if not legacy:
                self.fitting_channels = list(self.placed) if any(self.placed) else self.fitting_channels
        amplitude = self.trace_amplitude([v for v, ok in zip(self.locations, self.placed) if ok] or self.locations)
        self._shown_names = self.channel_names()
        noise_end = self.noise_end()[0]

        self.lines = [None] * n_channels
        line_idx = 0
        for i in range(n_channels):
            if not self.enabled_channels[i]:
                continue
            loc = self.locations[i]
            self.ax_wave.plot([tt[0], tt[-1]], [loc, loc], "k:", zorder=-101)
            # from the baseline (the noise's median), so a trace on an offset (volts) sits at its sensor
            base = np.nanmedian(y[i, :max(noise_end, 1)]) if np.isfinite(y[i, :max(noise_end, 1)]).any() else 0.0
            span = np.nanmax(y[i, :]) - np.nanmin(y[i, :]) if np.isfinite(y[i, :]).any() else 0.0
            ratio = -amplitude / span if span > 0 else 0.0
            self.lines[i] = self.ax_wave.plot(tt, (y[i, :] - base) * ratio + loc, color="C%d" % line_idx,
                                              zorder=-100, lw=2.0 if i == self.channel else 1.0)[0]
            line_idx += 1
        self.ax_wave.set_ylabel("location along fault (mm)")
        self.ax_wave.set_xlabel("time from the event (ms)")
        shown = [self.locations[i] for i in range(n_channels) if self.enabled_channels[i]] or self.locations
        self.ax_wave.set_ylim(max(shown) + amplitude, min(shown) - amplitude)
        if self.xlim is None:
            self.ax_wave.set_xlim(tt[0], tt[-1])
        else:
            self.ax_wave.set_xlim(self.xlim)
        self.figure.suptitle(f"{self.figure_title()}: {self.record_key}")

        if self.picked_idx is None or len(self.picked_idx) != n_channels:
            self.picked_idx = self.default_picks(strain, n_channels, y.shape[1])
        self.draw_markers()

    def handle_extent(self, i: int) -> tuple:
        """``(x_ms, y_low, y_high)`` of channel ``i``'s pick handle: the height
        of its drawn trace, so that it never reaches a neighbouring trace."""
        x = float(self.relative_time_s()[int(self.picked_idx[i])] * 1e3)
        line = self.lines[i] if i < len(self.lines) else None
        y = np.asarray(line.get_ydata(), dtype=float) if line is not None else np.array([])
        y = y[np.isfinite(y)]
        if y.size and np.ptp(y) > 0:
            lo, hi = float(y.min()), float(y.max())
        else:
            half = 0.5 * self.trace_amplitude(self.locations)
            lo, hi = self.locations[i] - half, self.locations[i]
        inset = HANDLE_INSET_FRACTION * (hi - lo)
        return x, lo + inset, hi - inset

    def draw_markers(self):
        """One handle per shown channel: red when fitted, black when not; the
        selected channel's is wider."""
        for marker in self.fitting_markers + self.not_fitting_markers:
            try:
                marker.remove()
            except (ValueError, NotImplementedError):
                pass
        self.fitting_markers = []
        self.not_fitting_markers = []
        for i in range(len(self.picked_idx)):
            if not self.enabled_channels[i] or self.lines[i] is None:
                continue
            x, lo, hi = self.handle_extent(i)
            color = "red" if self.fitting_channels[i] else "black"
            marker = self.ax_wave.plot([x, x], [lo, hi], color=color, lw=3.5 if i == self.channel else 2.0,
                                       solid_capstyle="butt", label=str(i), zorder=5)[0]
            if self.fitting_channels[i]:
                self.fitting_markers.append(marker)
            else:
                self.not_fitting_markers.append(marker)
        self.canvas.draw()

    def marker_of(self, i: int):
        return next((m for m in self.fitting_markers + self.not_fitting_markers if m.get_label() == str(i)), None)

    def on_resize(self, event=None):
        self.canvas.draw()
        if self.axs is not None:
            self.xlim = self.ax_wave.get_xlim()

    def zoom_to_picks(self) -> None:
        """Show the shown channels' picks with a margin around them."""
        times = [self.relative_time_s()[int(self.picked_idx[i])] * 1e3
                 for i in range(len(self.picked_idx)) if self.enabled_channels[i]]
        tt = self.relative_time_s() * 1e3
        if not times:
            return
        lo, hi = min(times), max(times)
        half = max(ZOOM_MARGIN_FRACTION * (hi - lo), ZOOM_MIN_HALFWIDTH_SAMPLES * self.sample_interval() * 1e3)
        self.ax_wave.set_xlim(max(lo - half, tt[0]), min(hi + half, tt[-1]))   # never wider than the record
        self.on_resize()

    def show_whole_record(self) -> None:
        tt = self.relative_time_s() * 1e3
        self.ax_wave.set_xlim(tt[0], tt[-1])
        self.on_resize()

    # ------------------------------------------------------------- dragging
    def handle_at(self, event) -> Optional[int]:
        """The channel whose pick handle is within ``HANDLE_PX`` of a mouse event, or None."""
        if event.inaxes is not self.ax_wave or event.x is None or event.y is None:
            return None
        best, best_dx = None, HANDLE_PX + 1.0
        to_px = self.ax_wave.transData.transform
        for marker in self.fitting_markers + self.not_fitting_markers:
            i = int(marker.get_label())
            x, lo, hi = self.handle_extent(i)
            (px, py_lo), (_, py_hi) = to_px((x, lo)), to_px((x, hi))
            dx = abs(event.x - px)
            if dx < best_dx and min(py_lo, py_hi) - HANDLE_PX <= event.y <= max(py_lo, py_hi) + HANDLE_PX:
                best, best_dx = i, dx
        return best

    def channel_near(self, event) -> Optional[int]:
        """The shown channel whose trace line is nearest the event, within half a trace height."""
        if event.inaxes is not self.ax_wave or event.ydata is None:
            return None
        shown = [i for i in range(len(self.picked_idx)) if self.lines[i] is not None]
        if not shown:
            return None
        i = min(shown, key=lambda k: abs(self.locations[k] - event.ydata))
        return i if abs(self.locations[i] - event.ydata) <= 0.5 * self.trace_amplitude(self.locations) else None

    def on_press(self, event):
        """Grab a pick handle; a press elsewhere on a trace selects its channel."""
        if self.toolbar_active() or getattr(event, "button", 1) != 1:
            return
        i = self.handle_at(event)
        if i is None:
            i = self.channel_near(event)
            if i is not None and i != self.channel:
                self.select_channel(i)
            return
        self._drag = i
        if i != self.channel:
            self.select_channel(i)

    def on_motion(self, event):
        if self._drag is None or self.toolbar_active() or event.xdata is None:
            return
        self.set_pick(self._drag, self.sample_at(event.xdata), refit=True)

    def on_release(self, event):
        if self._drag is not None:
            self._drag = None
            self.update_fitted_line()
        self.on_resize()

    def sample_at(self, t_ms: float) -> int:
        """The sample nearest a time in ms from the event."""
        return int(np.argmin(np.abs(self.relative_time_s() * 1e3 - float(t_ms))))

    def set_pick(self, i: int, idx: int, refit: bool = True) -> None:
        """Put channel ``i``'s pick on sample ``idx`` and move its handle."""
        idx = int(min(max(int(idx), 0), self.relative_time_s().size - 1))
        self.picked_idx[i] = idx
        if i < len(self.auto_flags):
            self.auto_flags[i] = ""             # placed by hand now
        marker = self.marker_of(i)
        if marker is not None:
            x, _, _ = self.handle_extent(i)
            marker.set_xdata([x, x])
        if refit:
            self.update_fitted_line()

    def on_arrival_typed(self, event=None) -> None:
        """The selected channel's arrival typed in ms from the event."""
        text = self.arrival_var.get().strip()
        if not text:
            return
        try:
            t_ms = float(text)
            if not math.isfinite(t_ms):
                raise ValueError
        except ValueError:
            self.update_status(f"arrival: {text!r} is not a time in ms")
            return
        tt = self.relative_time_s() * 1e3
        if not tt[0] <= t_ms <= tt[-1]:
            self.update_status(f"arrival: {t_ms:g} ms is outside the record ({tt[0]:.4g} to {tt[-1]:.4g} ms)")
            return
        idx = self.sample_at(t_ms)
        if idx != self.picked_idx[self.channel]:
            self.set_pick(self.channel, idx)
        else:
            self.show_arrival()

    def show_arrival(self) -> None:
        """The selected channel's name and arrival in the row under the table."""
        if self.picked_idx is None:
            return
        names = self.channel_names()
        self.channel_label_var.set(names[self.channel] if self.channel < len(names) else f"ch {self.channel}")
        t = self.relative_time_s()[int(self.picked_idx[self.channel])] * 1e3
        self.arrival_var.set(_ms(float(t), 6))

    def step_channel(self, step: int) -> None:
        if self.picked_idx:
            self.select_channel((self.channel + step) % len(self.picked_idx))

    # ------------------------------------------------------------------ fit
    @staticmethod
    def fit_arrival_line(locations, times):
        """Least-squares ``time = slope * location + intercept``; None if ill-posed.

        Ill-posed means fewer than two picks, fewer than two distinct locations,
        all picked times identical (no propagation to measure: polyfit would
        return a slope of ~1e-20 rather than 0 whenever the common time is not
        exactly 0), a rank-deficient fit, or non-finite coefficients.
        """
        locations = np.asarray(locations, dtype=float)
        times = np.asarray(times, dtype=float)
        if len(times) < 2 or len(np.unique(locations)) < 2:
            return None
        if not np.all(np.isfinite(times)) or np.ptp(times) == 0:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", _RankWarning)
                coeffs = np.polyfit(locations, times, 1)
        except Exception:
            return None
        if not np.all(np.isfinite(coeffs)):
            return None
        return float(coeffs[0]), float(coeffs[1])

    def fit_channels(self) -> list:
        """The channels the line goes through: shown and fitting."""
        return [i for i in range(len(self.picked_idx))
                if self.enabled_channels[i] and self.fitting_channels[i] and self.lines[i] is not None
                and (i >= len(self.placed) or self.placed[i])]

    def update_fitted_line(self):
        """Cf from one line through the fitted picks, and the rupture front (unilateral
        or bilateral) through them, drawn with each segment's speed."""
        for artist in self.front_lines:
            try:
                artist.remove()
            except ValueError:
                pass
        self.front_lines = []
        self.fitted_line = None
        chans = self.fit_channels()
        times = self.pick_times_s()
        locs = [self.locations[i] for i in chans]
        self.rupture_speed = math.nan
        fit = self.fit_arrival_line(locs, [times[i] for i in chans])
        if fit is not None and fit[0] != 0:
            self.rupture_speed = -1e-3 / fit[0]  # mm/s -> m/s, the sign of the published analyses
        use = [i in chans for i in range(len(self.picked_idx))]
        try:
            self.front_fit = sprof.fit_front(self.locations, times, use=use, mode=self.front_mode_var.get(),
                                             dt_s=self.sample_interval())
        except ValueError as e:
            self.front_fit = sprof.FrontFit(self.front_mode_var.get(), False, str(e))
        self.draw_front()
        self.canvas.draw()
        self.update_cf_label()
        self.front_text_var.set(self.front_text())
        self.fill_table()
        self.update_status()
        self.update_idletasks()

    def draw_front(self) -> None:
        """The front's segments (time against position) with their speeds, and the apex."""
        fit = self.front_fit
        if fit is None or not fit.valid:
            return
        ax = self.ax_wave
        for seg in fit.segments:
            xx = np.array([seg.x_from, seg.x_to])
            tt = 1e3 * (seg.slope_s_per_mm * xx + seg.intercept_s)
            self.front_lines.append(ax.plot(tt, xx, "--", color=FRONT_COLOR, lw=1.5, zorder=4)[0])
            self.front_lines.append(ax.annotate(segment_speed_label(seg), (tt.mean(), xx.mean()),
                                                textcoords="offset points", xytext=(6, 0), ha="left",
                                                va="center", fontsize=8, color=FRONT_COLOR))
        if math.isfinite(fit.split_x_mm) and math.isfinite(fit.t0_s) and fit.mode == "bilateral":
            self.front_lines.append(ax.plot([1e3 * fit.t0_s], [fit.split_x_mm], "D", mfc="none", mec=FRONT_COLOR,
                                            mew=1.4, ms=8, zorder=6)[0])
        if fit.mode == "unilateral" and self.has_rupture_speed():
            self.fitted_line = self.front_lines[0]

    def front_text(self) -> str:
        fit = self.front_fit
        if fit is None:
            return ""
        if not fit.valid:
            return f"front: {fit.reason}"
        lines = [segment_text(seg) for seg in fit.segments]
        apex = apex_text(fit)
        if apex:
            lines.append(apex)
        return "\n".join(lines)

    def residuals_s(self) -> list:
        """Each channel's pick minus the front at its position (s); NaN without a front.
        A channel beyond the fitted segments is compared with the nearest one."""
        fit = self.front_fit
        n = len(self.picked_idx)
        if fit is None or not fit.valid or not fit.segments:
            return [math.nan] * n
        times = self.pick_times_s()
        out = []
        for i in range(n):
            x = self.locations[i]

            def gap(seg):
                lo, hi = min(seg.x_from, seg.x_to), max(seg.x_from, seg.x_to)
                return 0.0 if lo <= x <= hi else min(abs(x - lo), abs(x - hi))
            seg = min(fit.segments, key=gap)
            out.append(times[i] - (seg.slope_s_per_mm * x + seg.intercept_s))
        return out

    def has_rupture_speed(self) -> bool:
        return self.rupture_speed is not None and bool(np.isfinite(self.rupture_speed))

    def update_cf_label(self):
        cf = self.rupture_speed
        if not self.has_rupture_speed():
            text = "Cf = n/a"
        elif abs(cf) < 1e4:
            text = f"Cf = {cf:.2f} m/s"
        else:
            text = f"Cf = {cf:.2e} m/s"
        fit = self.front_fit
        if (self.has_rupture_speed() and fit is not None and fit.valid and len(fit.segments) == 2
                and fit.segments[0].slope_s_per_mm * fit.segments[1].slope_s_per_mm < 0):
            text += " (one line through a front that runs both ways)"
        self.cf_label.configure(text=text)

    # --------------------------------------------------------------- table
    def fill_table(self) -> None:
        """One row per channel: name, x, arrival (ms from the event), shown, fitted."""
        if self.picked_idx is None:
            return
        names = self.channel_names()
        times = self.pick_times_s()
        resid = self.residuals_s()
        self._filling_table = True
        try:
            self.table.delete(*self.table.get_children())
            for i in range(len(self.picked_idx)):
                shown = bool(self.enabled_channels[i])
                placed = i >= len(self.placed) or self.placed[i]
                self.table.insert("", "end", iid=str(i), values=(
                    names[i] if i < len(names) else f"ch {i}", _ms(self.locations[i]) if placed else "\u2013",
                    _ms(times[i] * 1e3, 5) if shown else "", CHECK if shown else "",
                    CHECK if self.fitting_channels[i] else "",
                    f"{resid[i] * 1e6:+.3g}" if shown and placed and math.isfinite(resid[i]) else "",
                    FLAG_SHORT.get(self.auto_flags[i], self.auto_flags[i]) if i < len(self.auto_flags) else ""))
            if 0 <= self.channel < len(self.picked_idx):
                self.table.selection_set(str(self.channel))
                self.table.see(str(self.channel))
        finally:
            self._filling_table = False
        self.show_arrival()

    def on_table_click(self, event) -> None:
        """A click on the show or fit cell toggles it."""
        row = self.table.identify_row(event.y)
        column = self.table.identify_column(event.x)
        if not row or not column:
            return
        name = TABLE_COLUMNS[int(column[1:]) - 1][0]
        if name == "show":
            self.set_shown(int(row), not self.enabled_channels[int(row)])
        elif name == "fit":
            self.set_fitting(int(row), not self.fitting_channels[int(row)])

    def on_table_select(self, event=None) -> None:
        if self._filling_table:
            return
        selection = self.table.selection()
        if selection and int(selection[0]) != self.channel:
            self.select_channel(int(selection[0]))

    def select_channel(self, i: int) -> None:
        """Make channel ``i`` the selected one; its trace is drawn thicker."""
        self.channel = int(i)
        for k, line in enumerate(self.lines):
            if line is not None:
                line.set_linewidth(2.0 if k == self.channel else 1.0)
        for marker in self.fitting_markers + self.not_fitting_markers:
            marker.set_linewidth(3.5 if marker.get_label() == str(self.channel) else 2.0)
        self.canvas.draw_idle()
        self.fill_table()

    def set_shown(self, i: int, value: bool) -> None:
        self.enabled_channels[i] = bool(value)
        self.xlim = self.ax_wave.get_xlim()
        self.plot()
        self.update_fitted_line()

    def set_fitting(self, i: int, value: bool) -> None:
        self.fitting_channels[i] = bool(value)
        self.draw_markers()
        self.update_fitted_line()
        if value and i < len(self.placed) and not self.placed[i]:
            self.update_status(f"{self.channel_names()[i]} has no position in the record: not fitted")

    def set_all_shown(self, value: bool = True) -> None:
        self.enabled_channels = [bool(value)] * len(self.enabled_channels)
        self.xlim = self.ax_wave.get_xlim()
        self.plot()
        self.update_fitted_line()

    def patch_edges(self) -> Optional[tuple]:
        """The bare patch's edges in mm, ``analysis['fault']['patch_x_mm']`` in
        order, or None when the experiment has no usable pair."""
        value = get_setting(self.data_manager.data, "fault", "patch_x_mm")
        try:
            lo, hi = sorted(float(v) for v in np.asarray(value, dtype=float).ravel()[:2])
        except (TypeError, ValueError):
            return None
        return (lo, hi) if math.isfinite(lo) and math.isfinite(hi) and lo < hi else None

    def fit_the_patch(self) -> None:
        """Fit the sensors on the bare patch (``analysis['fault']['patch_x_mm']``)
        and only those; they are shown, the others keep their show tick."""
        edges = self.patch_edges()
        if edges is None:
            self.update_status("no patch edges in the experiment (analysis['fault']['patch_x_mm']; "
                               "Slip Distribution saves them)")
            return
        lo, hi = edges
        inside = [bool(i >= len(self.placed) or self.placed[i]) and lo <= x <= hi for i, x in enumerate(self.locations)]
        if not any(inside):
            self.update_status(f"no sensor lies on the patch ({lo:g} to {hi:g} mm)")
            return
        self.fitting_channels = inside
        self.enabled_channels = [bool(e) or f for e, f in zip(self.enabled_channels, inside)]
        self.xlim = self.ax_wave.get_xlim()
        self.plot()
        self.update_fitted_line()
        self.update_status(f"fitting the {sum(inside)} sensors on the patch ({lo:g} to {hi:g} mm)")

    def fit_all_shown(self) -> None:
        self.fitting_channels = [bool(v) for v in self.enabled_channels]
        self.draw_markers()
        self.update_fitted_line()

    # --------------------------------------------------------------- status
    def current_state(self) -> tuple:
        return (tuple(int(i) for i in self.picked_idx), tuple(bool(v) for v in self.enabled_channels),
                tuple(bool(v) for v in self.fitting_channels), self.front_mode_var.get())

    def update_status(self, message: Optional[str] = None) -> None:
        """The status line: a message, then whether the picks are saved."""
        if self.picked_idx is None:
            return
        if self.saved_state is None:
            state = "not saved"
        elif self.current_state() == self.saved_state:
            state = "saved"
        else:
            state = "differs from the saved record"
        state = f"set '{self.pick_set}': {state}"
        parts = [message] if message else []
        if not self.has_rupture_speed():
            parts.append("Cf n/a: the fitted picks define no line (fewer than two positions, or one time)")
        parts.append(state)
        self.status_var.set(" | ".join(parts))

    # ---------------------------------------------------------------- save
    def save(self):
        """Store the picks with the record they were made on."""
        block = self.current_block()
        picked = [int(i) for i in self.picked_idx]
        arrivals = {
            "picked_idx": picked,
            "arrival_time": waveform_time(block)[picked],
            "enabled_channels": [bool(v) for v in self.enabled_channels],
            "fitting_channels": [bool(v) for v in self.fitting_channels],
            "locations": self.saved_locations(),
            "filter": self.saved_filter(),
            # a degenerate fit has no speed: no nan and no stale value for the CZM fitter to pick up as Cf
            "rupture_speed": float(self.rupture_speed) if self.has_rupture_speed() else None,
            "front_mode": self.front_mode_var.get(),
        }
        if self.auto_settings is not None:
            arrivals["auto_pick"] = {**self.auto_settings, "flags": list(self.auto_flags)}
        if self.front_fit is not None:
            front = self.front_fit.as_dict()
            front["use"] = [i in self.fit_channels() for i in range(len(self.picked_idx))]
            arrivals["front"] = front
        write_arrivals(block, arrivals, self.event, self.pick_set)
        self.store_template()
        set_setting(self.data_manager.data, self.pick_set, SETTING, CURRENT_SET_SETTING)
        self.saved_state = self.current_state()
        self.refresh_pick_sets()
        self.app.refresh_tree()
        note = "" if self.has_rupture_speed() else " without a rupture speed"
        self.update_status(f"saved to {self.record_key}{note}")

    def saved_filter(self) -> dict:
        """The filter the traces are drawn with (``kind`` off when its settings are unusable)."""
        try:
            return self.filter_settings()
        except ValueError:
            return {"kind": tf.KIND_OFF}

    # ------------------------------------------------------------ auto pick
    def noise_end(self) -> tuple:
        """``(index, description)``: the noise (for the auto pick and the traces'
        baselines) is the record before ``index``: up to its trigger, else the
        event time, less a guard; else its first fifth."""
        t = waveform_time(self.current_block())
        t_ref, name = self.reference_time()
        if t_ref is not None:
            end_t = t_ref - NOISE_TRIGGER_GUARD_FRACTION * (t_ref - t[0])
            k = int(np.searchsorted(t, end_t))
            if k >= ap.MIN_NOISE_SAMPLES:
                where = "its trigger" if name == "trigger" else "the event time"
                return k, f"noise: the record before {where}, less {NOISE_TRIGGER_GUARD_FRACTION:.0%} of it"
        return int(NOISE_FRACTION * t.size), f"noise: the first {NOISE_FRACTION:.0%} of the record"

    def read_auto_settings(self) -> dict:
        """The Auto pick box as settings (ValueError when unusable)."""
        label = self.auto_method_var.get()
        method = next((m for m in ap.METHODS if ap.METHOD_LABELS[m] == label), ap.METHOD_DIFFERENCE)
        try:
            threshold = float(self.auto_threshold_var.get())
        except ValueError:
            raise ValueError(f"threshold {self.auto_threshold_var.get()!r} is not a number") from None
        if not (math.isfinite(threshold) and threshold > 0):
            raise ValueError("the threshold must be > 0")
        try:
            us = float(self.auto_smoothing_var.get())
        except ValueError:
            raise ValueError(f"slope smoothing {self.auto_smoothing_var.get()!r} is not a number") from None
        if not (math.isfinite(us) and us >= 0):
            raise ValueError("the slope smoothing must be >= 0")
        smoothing = max(1, int(round(us * 1e-6 / self.sample_interval())))
        settings = {"method": method, "threshold": threshold, "smoothing_samples": smoothing}
        for key, var in (("search_from_s", self.search_from_var), ("search_to_s", self.search_to_var)):
            text = var.get().strip()
            if not text:
                continue
            try:
                value = float(text)
                if not math.isfinite(value):
                    raise ValueError
            except ValueError:
                raise ValueError(f"search {text!r} is not a time in ms") from None
            settings[key] = value * 1e-3
        if settings.get("search_from_s", -math.inf) >= settings.get("search_to_s", math.inf):
            raise ValueError("the search must start before it ends")
        return settings

    def use_range_shown(self) -> None:
        """The search window from the time range shown on the waveform panel."""
        lo, hi = sorted(self.ax_wave.get_xlim())
        self.search_from_var.set(f"{lo:.4g}")
        self.search_to_var.set(f"{hi:.4g}")

    def auto_pick(self, settings: Optional[dict] = None) -> Optional[list]:
        """Pick every shown channel from its trace as drawn; channels that cannot
        be picked keep their pick and are flagged.  Returns the picks (None when
        the settings are unusable)."""
        try:
            settings = self.read_auto_settings() if settings is None else dict(settings)
        except ValueError as e:
            self.update_status(f"auto pick: {e}")
            return None
        end, note = self.noise_end()
        search = None
        if settings.get("search_from_s") is not None or settings.get("search_to_s") is not None:
            tt = self.relative_time_s()
            lo = None if settings.get("search_from_s") is None else int(np.searchsorted(tt, settings["search_from_s"]))
            hi = None if settings.get("search_to_s") is None else int(np.searchsorted(tt, settings["search_to_s"],
                                                                                      side="right"))
            search = (lo, hi)
            if lo is not None and lo < end:
                end = lo
                note += "; shortened to end at the search window's start"
        shown = [i for i, line in enumerate(self.lines) if line is not None]
        # the traces as drawn, back on their own scale and sign: loc - y
        rows = [self.locations[i] - np.asarray(self.lines[i].get_ydata(), dtype=float) for i in shown]
        picks = ap.auto_pick(rows, settings["method"], end, settings["threshold"], settings["smoothing_samples"],
                             search=search) if rows else []
        for i, pick in zip(shown, picks):
            if pick.index is not None:
                self.picked_idx[i] = int(pick.index)
            self.auto_flags[i] = pick.flag
        t = waveform_time(self.current_block())
        self.auto_settings = {**settings, "noise_end_s": float(t[min(end, t.size - 1)] - float(self.event["event_time"])),
                              "filter": self.saved_filter()}
        self.auto_settings = {k: v for k, v in self.auto_settings.items() if v is not None}
        self.noise_note_var.set(note)
        self.draw_markers()
        self.update_fitted_line()
        names = self.channel_names()
        flagged = [f"{names[i]}: {p.flag}" for i, p in zip(shown, picks) if p.flag]
        self.update_status(f"auto pick ({ap.METHOD_LABELS[settings['method']]}): {len(picks)} channel(s)"
                           + (f", {len(flagged)} flagged ({'; '.join(flagged)})" if flagged else ""))
        return picks

    def pick_peaks(self):
        """Auto pick with the peak method: each shown trace's largest departure from the noise."""
        return self.auto_pick({"method": ap.METHOD_PEAK, "threshold": ap.DEFAULT_THRESHOLD,
                               "smoothing_samples": ap.DEFAULT_SMOOTHING_SAMPLES})
