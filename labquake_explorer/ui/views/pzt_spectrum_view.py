"""Displacement spectrum and omega-n (Brune) fit of one PZT / strain channel.

The view is a thin Tk shell around :mod:`labquake_explorer.analysis.spectrum`
and :mod:`labquake_explorer.analysis.source`: it reads
a full-rate record of the event (``event['waveform'][<recorder>]`` with ``time``
and ``data``, or the PSU-era ``strain`` block) and ``event['event_time']``
(or the record's own trigger time, where the recorder fired, or a picked
per-channel arrival), collects and validates the user's
parameters, and calls

    extract_window -> compute_spectrum -> fit_omega_n -> source_parameters

Nothing numerical happens here.  Invalid entries are reported in the status
line, never in a dialog.  Results are stored with the record they were
computed on, ``event['waveform'][recorder]['spectrum']`` (or
``event['strain']['spectrum']`` for the PSU layout), as
``{"version": 1, "channels": {"ch<channel>": record}}`` so that several
channels of one record can be kept side by side (see :meth:`build_record`
for the record keys); the older event-level ``event['pzt_spectrum']`` map is
still read and is moved into the record on the next save.  The ``ch`` prefix keeps the channel map from looking
like a list to loaders that turn all-digit keys into arrays.  Everything
recomputes as soon as a control changes (typing waits until the keys have
been quiet for a moment); selecting a channel with a saved record restores
its parameters, recomputes and says whether the result still matches what
was saved.  A saved record that cannot be displayed never prevents the
window from opening: the controls are left as they are and the status line
says why.

Save also keeps, for the whole experiment (``data['analysis']``, see
:mod:`labquake_explorer.data.settings`), the controls as they are, the
medium (rho, Vp, Vs) and the sensor's own values (its calibration table,
t*, distance and free-surface factor), so that a window on any other event
opens with them and nothing has to be typed again.  A calibration CSV is
read when its path is typed or browsed and then kept as a table, in the
record and with the sensor, so a saved file works without the CSV; the
frame says what is loaded.  A response in V/(N m) (an in-situ ball-drop
response, the empirical Green's function route) gives a MOMENT spectrum in
N*m*s whose plateau is the seismic moment itself; the medium, distance,
radiation and free-surface entries are then greyed out and "Compute source"
takes M0 = Omega0 (``source_parameters(..., moment_spectrum=True)``); a
"divide gain by" entry takes a constant factor out of the stored response.
Records are kept in named RESULT SETS (``labquake_explorer.data.spectrum_store``):
the "Result set" box chooses one, or names a new one to be created on Save,
so the BAC and EGF routes of one channel sit side by side; each set keeps
its own controls and sensor values at the experiment level
(``analysis['pzt_spectrum']['sets'][<name>]``) and the set last saved is
where a new window opens (``current_set``).  The signal and noise windows
are dragged on the top panel, each edge on its own or a window by its body;
a noise window left on automatic (blank entries) follows the signal window.  The
noise window lies before the signal window or after it (a recorder that
keeps little pre-trigger data); automatic noise moves after the signal when
there is no room before, and the status line says so.

Caveats reported by the analysis (``SpectralFit.warnings``: transient not
in the taper's flat region, band-limited corner, parameters at bounds, no
convergence) are shown in the ``flags`` row and in the status line; a fit
with warnings is never labelled "fit OK".  Because the amplitude convention
(``|rfft(w x)| dt`` without coherent-gain division) only holds when the
transient sits in the flat part of the taper, a pre window shorter than the
tukey ramp (``alpha/2`` of the whole window) is warned about, and source
parameters computed from a plateau the analysis flagged as attenuated carry
that warning (``source['warnings']``) rather than being refused.  A free
exponent is kept at or below the "n max" entry (``n_max`` in the record).

Typical medium values for the source section (granite-like rock):
rho = 2700 kg/m^3, Vp = 6000 m/s, Vs = 3500 m/s.  They are NOT defaults:
the entries start blank so the values used are always the user's own.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Optional

import numpy as np

from labquake_explorer.analysis import source as src
from labquake_explorer.analysis import spectrum as sp
from labquake_explorer.analysis.spectrum import N_BOUNDS
from labquake_explorer.analysis import spectrum_ensemble as se
from labquake_explorer.data.spectrum_store import DEFAULT_SET, ensure_set, set_channels, set_name, set_names
from labquake_explorer.data.settings import delete_setting, get_setting, set_setting
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT
from labquake_explorer.ui.views.base import EventView
from labquake_explorer.data.sources import (pick_waveform_block, read_arrivals, waveform_block, waveform_blocks,
                                            waveform_channels, waveform_data, waveform_time, waveform_trigger_time)

RESULT_VERSION = 1

LIVE_DELAY_MS = 400                    # recompute this long after the last keystroke
HANDLE_PICK_PX = 6                     # a press this close (pixels) to a window edge grabs it
HANDLES = ("noise_start", "noise_end", "signal_start", "signal_end")
TIE = "noise_end|signal_start"         # a press where the noise end and the signal start coincide
TIE_AFTER = "signal_end|noise_start"   # ... where the signal end and the start of a noise window after it coincide
TRIGGER_EVENT_TIME = "event_time"
TRIGGER_RECORD = "record trigger"      # where the recorder fired (the PZT trigger of an Elsys block)
TRIGGER_PICKED = "picked arrival"
TRIGGERS = (TRIGGER_EVENT_TIME, TRIGGER_RECORD, TRIGGER_PICKED)
CALIBRATION_UNITS = ("V/V", "V/m", "V/(m/s)", "V/(m/s^2)", sp.MOMENT_UNIT)

#: Tukey taper shape passed to ``compute_spectrum``; the first and last
#: ``alpha/2`` of the window are ramped, so the trigger sample (and the pulse
#: after it) must lie beyond that fraction for the plateau to be unattenuated.
TUKEY_ALPHA = 0.25

SPECTRUM_KEY = "spectrum"              # under the waveform record (or the PSU strain block)
SETTING = "pzt_spectrum"               # analysis['pzt_spectrum']: 'settings' (the controls) and 'sensors'
MEDIUM_SETTING = "medium"              # analysis['medium']: rho_kg_m3, vp_m_s, vs_m_s
SETS_SETTING = "sets"                  # analysis['pzt_spectrum']['sets'][<result set>]: 'settings' and 'sensors'
CURRENT_SET_SETTING = "current_set"    # analysis['pzt_spectrum']['current_set']: the result set last saved
#: Controls kept at the experiment level (the last saved), so a new window opens with them.
SETTINGS_KEYS = ("trigger", "pre_ms", "post_ms", "noise_start_ms", "noise_end_ms", "taper", "baseline",
                 "highpass_hz", "remove_step", "fmin_hz", "fmax_hz", "bins_per_decade", "snr_min", "n_fixed",
                 "n_max", "loss", "estimator", "ensemble_on", "ensemble_fmins_hz", "ensemble_fmaxs_hz",
                 "phase", "radiation_coefficient")
#: Per-sensor values kept at the experiment level, keyed by the channel's field name (``pzt_1``) or ``ch<k>``.
SENSOR_KEYS = ("calibration", "calibration_path", "calibration_unit", "calibration_db", "calibration_divisor",
               "t_star_s", "q_csv_path", "q_travel_time_s", "q_hold_ends", "attenuation",
               "distance_m", "free_surface_factor", "ensemble_calibration_paths")
LEGACY_SPECTRUM_KEY = "pzt_spectrum"   # the event-level map written before 2026-10-12, read only
#: Prefix of the per-channel keys in a ``spectrum['channels']`` map.
CHANNEL_KEY_PREFIX = "ch"

#: Start of the analysis warning that flags an attenuated plateau.
FLAT_REGION_WARNING_PREFIX = "transient not in the taper's flat region"

#: Default control values (strings as typed into the entries).
DEFAULTS = {
    "pre_ms": "0.5", "post_ms": "2.0", "taper": "tukey", "baseline": "pre_linear",
    "highpass_hz": "", "remove_step": False,
    "calibration_path": "", "calibration_unit": "V/V", "calibration_db": False, "calibration_divisor": "",
    "t_star_s": "", "q_csv_path": "", "q_travel_time_s": "", "q_hold_ends": False,
    "fmin_hz": "1000", "fmax_hz": "", "bins_per_decade": "30", "snr_min": "3",
    "n_fixed": True, "n_max": f"{N_BOUNDS[1]:g}", "loss": "linear", "estimator": "rms",
    "ensemble_on": False, "ensemble_fmins_hz": ", ".join(f"{v:g}" for v in se.DEFAULT_FMINS_HZ),
    "ensemble_fmaxs_hz": ", ".join(f"{v:g}" for v in se.DEFAULT_FMAXS_HZ), "ensemble_calibration_paths": "",
}
PATH_SEPARATOR = ";"                   # between the alternative calibration CSVs typed into one entry
FMAX_FRACTION_OF_FS = 0.4

_NA = "n/a"

#: Exceptions a malformed saved record may raise while being displayed.
_RECORD_ERRORS = (KeyError, TypeError, ValueError, AttributeError, IndexError)


def _parse_float(text, name: str, required: bool = True, lo: Optional[float] = None,
                 strict_lo: bool = False) -> Optional[float]:
    """Parse an entry; ``None`` for a blank optional entry.  Raises ValueError
    with a message suitable for the status line."""
    s = str(text).strip()
    if s == "":
        if required:
            raise ValueError(f"{name}: enter a number")
        return None
    try:
        v = float(s)
    except ValueError:
        raise ValueError(f"{name}: {s!r} is not a number") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    if lo is not None:
        if strict_lo and v <= lo:
            raise ValueError(f"{name} must be > {lo:g}")
        if not strict_lo and v < lo:
            raise ValueError(f"{name} must be >= {lo:g}")
    return v


def _parse_int(text, name: str, lo: int = 1) -> int:
    v = _parse_float(text, name, required=True, lo=lo)
    if v != int(v):
        raise ValueError(f"{name} must be an integer")
    return int(v)


def _text(value) -> str:
    """Entry text for a stored number (blank for None)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return ""
    return f"{v:.10g}"


def _fmt(value, fmt: str = ".4g", unit: str = "") -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _NA
    if not math.isfinite(v):
        return _NA
    s = format(v, fmt)
    return f"{s} {unit}".rstrip()


def _yes(flag) -> str:
    return "yes" if flag else "no"


def _as_list(value) -> list:
    """A plain Python list for a stored sequence (list, tuple or ndarray);
    ``[]`` for None or a scalar.  Never truth-tests an array."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(np.asarray(value, dtype=object).ravel())
    except (TypeError, ValueError):
        return []


def _band_pair(value) -> Optional[tuple[float, float]]:
    """``(lo, hi)`` floats from a stored band, or None when unusable."""
    items = _as_list(value)
    if len(items) != 2:
        return None
    try:
        lo, hi = float(items[0]), float(items[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return lo, hi


def _warning_texts(value) -> list[str]:
    return [str(w) for w in _as_list(value) if str(w).strip()]


def ensemble_stats(ensemble: dict) -> tuple:
    """``(stats, which)`` of a stored ensemble: its ``stats_resolved`` (the
    valid members whose corner is resolved inside the band) when any member
    is, else its ``stats`` over every valid member; ``which`` says which."""
    resolved = ensemble.get("stats_resolved") if isinstance(ensemble, dict) else None
    fc = resolved.get("fc") if isinstance(resolved, dict) else None
    try:
        n = int(fc.get("n", 0)) if isinstance(fc, dict) else 0
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        return resolved, "resolved"
    return (ensemble.get("stats") if isinstance(ensemble, dict) else {}) or {}, "valid"


def _parse_frequencies(text, name: str, default) -> list:
    """The grid entries: numbers in Hz (blank = the default grid)."""
    try:
        return list(se.parse_frequency_list(text, default))
    except ValueError as e:
        raise ValueError(f"{name}: {e}") from None


def _frequency_text(value) -> str:
    """Entry text for a stored frequency list (a list, or the text itself)."""
    if isinstance(value, str):
        return value
    try:
        return ", ".join(f"{float(v):g}" for v in _as_list(value))
    except (TypeError, ValueError):
        return ""


def _parse_paths(text) -> list:
    """The alternative CSVs typed into one entry, ``PATH_SEPARATOR`` between them."""
    return [p.strip() for p in str(text or "").split(PATH_SEPARATOR) if p.strip()]


def _paths_text(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return PATH_SEPARATOR.join(str(_text_value(v)) for v in _as_list(value))
    except TypeError:
        return ""


def _table_divisor(table) -> float:
    """The divisor the stored gains carry (1 when none was recorded)."""
    try:
        v = float(table.get("gain_divisor", 1.0) or 1.0)
    except (TypeError, ValueError):
        return 1.0
    return v if math.isfinite(v) and v > 0.0 else 1.0


def _text_value(value) -> str:
    """A stored text, also when an HDF5 round trip made it bytes or a 0-d array."""
    value = np.asarray(value).ravel()[0] if np.ndim(value) else value
    return value.decode() if isinstance(value, bytes) else str(value)


def channel_key(channel: int) -> str:
    """Key of ``channel`` in a ``spectrum['channels']`` map."""
    return f"{CHANNEL_KEY_PREFIX}{int(channel)}"


def parse_channel_key(key) -> Optional[int]:
    """Channel number of a stored key (``'ch3'``, or a legacy ``'3'``); None otherwise."""
    s = str(key).strip()
    if s.lower().startswith(CHANNEL_KEY_PREFIX):
        s = s[len(CHANNEL_KEY_PREFIX):]
    return int(s) if s.isdigit() else None


@register_view("PZT Spectrum", kinds=[EVENT], order=40)
class PZTSpectrumView(EventView):
    """Window, spectrum, omega-n fit and source parameters for one channel."""

    window_title = "PZT Spectrum"
    result_key = LEGACY_SPECTRUM_KEY       # the event-level map of older files; saves go under the record

    def __init__(self, app, run_idx, event_idx):
        self.strain_time: Optional[np.ndarray] = None
        self.strain_raw: Optional[np.ndarray] = None
        self.record_key: Optional[str] = None     # which full-rate record of the event is shown
        self.channel_labels: list = []
        self.window: Optional[sp.WindowResult] = None
        self.spectrum: Optional[sp.SpectrumResult] = None
        self.fit: Optional[sp.SpectralFit] = None
        self.record: Optional[dict] = None       # working record of the current channel
        self.result_set: str = DEFAULT_SET       # the result set shown and saved into
        self.calibration_table: Optional[dict] = None   # the loaded gain table (Calibration.as_dict + gain_db)
        self._first_load = True
        super().__init__(app, run_idx, event_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self) -> None:
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        left = ttk.Frame(self)
        left.grid(row=0, column=0, padx=5, pady=5, sticky="nsw")
        middle = ttk.Frame(self)
        middle.grid(row=0, column=1, padx=5, pady=5, sticky="nsw")

        self._build_selection(left)
        self._build_window_frame(left)
        self._build_calibration_frame(left)
        self._build_fit_frame(left)
        self._build_ensemble_frame(left)
        self._build_buttons(left)
        self._build_results_frame(middle)
        self._build_source_frame(middle)

        figure_frame = ttk.Frame(self)
        figure_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        figure_frame.grid_columnconfigure(0, weight=1)
        figure_frame.grid_rowconfigure(0, weight=1)
        self.make_figure(master=figure_frame, figsize=(8, 8), figure_kwargs={"layout": "constrained"},
                         row=0, column=0, sticky="nsew")
        gs = self.figure.add_gridspec(3, 1, height_ratios=[1.0, 2.0, 1.0])
        self.ax_time = self.figure.add_subplot(gs[0])
        self.ax_taper = self.ax_time.twinx()               # the taper, 0..1, on the right axis
        self.ax_spec = self.figure.add_subplot(gs[1])
        self.ax_resid = self.figure.add_subplot(gs[2])   # x limits copied from ax_spec in plot()
        self.axes = (self.ax_time, self.ax_spec, self.ax_resid)
        self.handles: dict = {}                            # window edges on the top panel
        self.signal_span = self.noise_span = None
        self._drag = None
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self._decorate_axes()
        self.canvas.draw()

    def _entry(self, master, label: str, var: tk.StringVar, row: int, width: int = 10, column: int = 0):
        ttk.Label(master, text=label).grid(row=row, column=column, padx=4, pady=2, sticky="e")
        entry = ttk.Entry(master, textvariable=var, width=width)
        entry.grid(row=row, column=column + 1, padx=4, pady=2, sticky="w")
        self._live(entry)
        if not hasattr(self, "entries"):
            self.entries = {}
        self.entries[label] = entry
        return entry

    def _live(self, widget) -> None:
        """Recompute when the widget's value changes: typing after a pause,
        Return and leaving the field at once, a combobox choice at once."""
        if isinstance(widget, ttk.Combobox):
            widget.bind("<<ComboboxSelected>>", lambda e: self.schedule_compute(now=True), add="+")
            return
        widget.bind("<KeyRelease>", lambda e: self.schedule_compute(), add="+")
        widget.bind("<Return>", lambda e: self.schedule_compute(now=True), add="+")
        widget.bind("<FocusOut>", lambda e: self.schedule_compute(now=True), add="+")

    def _build_selection(self, master) -> None:
        frame = ttk.LabelFrame(master, text="Event / channel")
        frame.pack(fill="x", pady=(0, 4))
        self.build_event_selector(frame, row=0, column=0)
        ttk.Label(frame, text="Record:").grid(row=1, column=0, padx=4, pady=2, sticky="e")
        self.record_combobox = ttk.Combobox(frame, width=14, state="readonly")
        self.record_combobox.grid(row=1, column=1, padx=4, pady=2, sticky="w")
        self.record_combobox.bind("<<ComboboxSelected>>", self.on_record_selected)
        ttk.Label(frame, text="Channel:").grid(row=2, column=0, padx=4, pady=2, sticky="e")
        self.channel_combobox = ttk.Combobox(frame, width=14, state="readonly")
        self.channel_combobox.grid(row=2, column=1, padx=4, pady=2, sticky="w")
        self.channel_combobox.bind("<<ComboboxSelected>>", self.on_channel_selected)
        ttk.Label(frame, text="Trigger:").grid(row=3, column=0, padx=4, pady=2, sticky="e")
        self.trigger_combobox = ttk.Combobox(frame, width=14, state="readonly", values=list(TRIGGERS))
        self.trigger_combobox.set(TRIGGER_EVENT_TIME)
        self.trigger_combobox.grid(row=3, column=1, padx=4, pady=2, sticky="w")
        self._live(self.trigger_combobox)
        ttk.Label(frame, text="Result set:").grid(row=4, column=0, padx=4, pady=2, sticky="e")
        self.set_combobox = ttk.Combobox(frame, width=14, values=[DEFAULT_SET])   # editable: a new name creates a set
        self.set_combobox.set(DEFAULT_SET)
        self.set_combobox.grid(row=4, column=1, padx=4, pady=2, sticky="w")
        for sequence in ("<<ComboboxSelected>>", "<Return>", "<FocusOut>"):
            self.set_combobox.bind(sequence, self.on_set_changed, add="+")
        ttk.Label(frame, text="type a new name to start another set (e.g. BAC, EGF)",
                  foreground="#606060").grid(row=5, column=0, columnspan=2, padx=4, pady=(0, 2), sticky="w")

    def _build_window_frame(self, master) -> None:
        frame = ttk.LabelFrame(master, text="Window")
        frame.pack(fill="x", pady=4)
        self.pre_var = tk.StringVar(master=self, value=DEFAULTS["pre_ms"])
        self.post_var = tk.StringVar(master=self, value=DEFAULTS["post_ms"])
        self.highpass_var = tk.StringVar(master=self, value=DEFAULTS["highpass_hz"])
        self.remove_step_var = tk.BooleanVar(master=self, value=DEFAULTS["remove_step"])
        self.noise_start_var = tk.StringVar(master=self, value="")
        self.noise_end_var = tk.StringVar(master=self, value="")
        self._entry(frame, "Pre (ms):", self.pre_var, 0)
        self._entry(frame, "Post (ms):", self.post_var, 1)
        self._entry(frame, "Noise start (ms):", self.noise_start_var, 2)
        self._entry(frame, "Noise end (ms):", self.noise_end_var, 3)
        ttk.Label(frame, text="noise: from the trigger, before the signal (end <= -pre)\nor after it "
                              "(start >= post); blank = the same length just\nbefore the signal, after it when "
                              "there is no room before.\nDrag the windows on the plot.",
                  foreground="#606060", justify="left").grid(row=4, column=0, columnspan=2, padx=4, pady=(0, 2), sticky="w")
        ttk.Label(frame, text="Taper:").grid(row=5, column=0, padx=4, pady=2, sticky="e")
        self.taper_combobox = ttk.Combobox(frame, width=10, state="readonly", values=list(sp.TAPER_KINDS))
        self.taper_combobox.set(DEFAULTS["taper"])
        self.taper_combobox.grid(row=5, column=1, padx=4, pady=2, sticky="w")
        self._live(self.taper_combobox)
        ttk.Label(frame, text="Baseline:").grid(row=6, column=0, padx=4, pady=2, sticky="e")
        self.baseline_combobox = ttk.Combobox(frame, width=10, state="readonly", values=list(sp.BASELINE_MODES))
        self.baseline_combobox.set(DEFAULTS["baseline"])
        self.baseline_combobox.grid(row=6, column=1, padx=4, pady=2, sticky="w")
        self._live(self.baseline_combobox)
        self._entry(frame, "Highpass (Hz):", self.highpass_var, 7)
        ttk.Checkbutton(frame, text="Remove coseismic step", variable=self.remove_step_var,
                        command=lambda: self.schedule_compute(now=True)).grid(
            row=8, column=0, columnspan=2, padx=4, pady=2, sticky="w")
        ttk.Label(frame, text=f"tukey ramp = {TUKEY_ALPHA / 2:.0%} of the window: keep\n"
                              f"pre >= post / {(2.0 / TUKEY_ALPHA) - 1.0:g} or Omega0 is attenuated",
                  foreground="#606060", justify="left").grid(row=9, column=0, columnspan=2, padx=4, pady=2, sticky="w")

    def _build_calibration_frame(self, master) -> None:
        frame = ttk.LabelFrame(master, text="Calibration / attenuation")
        frame.pack(fill="x", pady=4)
        self.calibration_path_var = tk.StringVar(master=self, value=DEFAULTS["calibration_path"])
        self.calibration_db_var = tk.BooleanVar(master=self, value=DEFAULTS["calibration_db"])
        self.t_star_var = tk.StringVar(master=self, value=DEFAULTS["t_star_s"])
        self.calibration_divisor_var = tk.StringVar(master=self, value=DEFAULTS["calibration_divisor"])
        self.q_path_var = tk.StringVar(master=self, value=DEFAULTS["q_csv_path"])
        self.q_travel_var = tk.StringVar(master=self, value=DEFAULTS["q_travel_time_s"])
        self.q_hold_var = tk.BooleanVar(master=self, value=DEFAULTS["q_hold_ends"])
        self.q_note_var = tk.StringVar(master=self, value="")
        self.attenuation_table: Optional[dict] = None
        ttk.Label(frame, text="CSV:").grid(row=0, column=0, padx=4, pady=2, sticky="e")
        self.calibration_entry = ttk.Entry(frame, textvariable=self.calibration_path_var, width=18)
        self.calibration_entry.grid(row=0, column=1, padx=4, pady=2, sticky="w")
        self._live(self.calibration_entry)
        self.browse_button = ttk.Button(frame, text="Browse", width=7, command=self.browse_calibration)
        self.browse_button.grid(row=0, column=2, padx=2, pady=2, sticky="w")
        ttk.Label(frame, text="Unit:").grid(row=1, column=0, padx=4, pady=2, sticky="e")
        self.unit_combobox = ttk.Combobox(frame, width=10, state="readonly", values=list(CALIBRATION_UNITS))
        self.unit_combobox.set(DEFAULTS["calibration_unit"])
        self.unit_combobox.grid(row=1, column=1, padx=4, pady=2, sticky="w")
        self._live(self.unit_combobox)
        ttk.Checkbutton(frame, text="gain in dB", variable=self.calibration_db_var,
                        command=lambda: self.schedule_compute(now=True)).grid(
            row=1, column=2, padx=2, pady=2, sticky="w")
        self.calibration_note_var = tk.StringVar(master=self, value="")
        ttk.Label(frame, textvariable=self.calibration_note_var, wraplength=250, justify="left",
                  foreground="#606060").grid(row=2, column=0, columnspan=3, padx=4, pady=(0, 2), sticky="w")
        ttk.Button(frame, text="Clear calibration", command=self.clear_calibration).grid(
            row=3, column=0, columnspan=2, padx=4, pady=2, sticky="w")
        self._entry(frame, "t* (s, blank = off):", self.t_star_var, 4)
        self._entry(frame, "divide gain by (blank = 1):", self.calibration_divisor_var, 5)
        ttk.Label(frame, text=f"{sp.MOMENT_UNIT}: a moment response (EGF route); e.g. divide\n"
                              "a Psi table by C_FM = Vp + Vs to use it as stored",
                  foreground="#606060", justify="left").grid(row=6, column=0, columnspan=3, padx=4, pady=(0, 2), sticky="w")
        ttk.Label(frame, text="Q^-1(f) CSV:").grid(row=7, column=0, padx=4, pady=2, sticky="e")
        self.q_entry = ttk.Entry(frame, textvariable=self.q_path_var, width=18)
        self.q_entry.grid(row=7, column=1, padx=4, pady=2, sticky="w")
        self._live(self.q_entry)
        ttk.Button(frame, text="Browse", width=7, command=self.browse_q).grid(row=7, column=2, padx=2, pady=2, sticky="w")
        self._entry(frame, "travel time (s):", self.q_travel_var, 8)
        ttk.Checkbutton(frame, text="hold end values outside the table", variable=self.q_hold_var,
                        command=lambda: self.schedule_compute(now=True)).grid(
            row=9, column=0, columnspan=3, padx=4, pady=2, sticky="w")
        ttk.Label(frame, textvariable=self.q_note_var, wraplength=250, justify="left",
                  foreground="#606060").grid(row=10, column=0, columnspan=3, padx=4, pady=(0, 2), sticky="w")
        ttk.Button(frame, text="Clear Q", command=self.clear_attenuation).grid(
            row=11, column=0, columnspan=2, padx=4, pady=2, sticky="w")
        self.show_calibration_note()
        self.show_q_note()

    def _build_fit_frame(self, master) -> None:
        frame = ttk.LabelFrame(master, text="Fit band / omega-n fit")
        frame.pack(fill="x", pady=4)
        self.fmin_var = tk.StringVar(master=self, value=DEFAULTS["fmin_hz"])
        self.fmax_var = tk.StringVar(master=self, value=DEFAULTS["fmax_hz"])
        self.bins_var = tk.StringVar(master=self, value=DEFAULTS["bins_per_decade"])
        self.snr_var = tk.StringVar(master=self, value=DEFAULTS["snr_min"])
        self.n_fixed_var = tk.BooleanVar(master=self, value=DEFAULTS["n_fixed"])
        self.n_max_var = tk.StringVar(master=self, value=DEFAULTS["n_max"])
        self._entry(frame, "fmin (Hz):", self.fmin_var, 0)
        self._entry(frame, "fmax (Hz):", self.fmax_var, 1)
        self._entry(frame, "bins / decade:", self.bins_var, 2)
        self._entry(frame, "SNR min (blank = off):", self.snr_var, 3)
        ttk.Checkbutton(frame, text="n fixed at 2", variable=self.n_fixed_var,
                        command=lambda: self.schedule_compute(now=True)).grid(
            row=4, column=0, columnspan=2, padx=4, pady=2, sticky="w")
        self._entry(frame, f"n max (free n, > {N_BOUNDS[0]:g}):", self.n_max_var, 5)
        ttk.Label(frame, text="Loss:").grid(row=6, column=0, padx=4, pady=2, sticky="e")
        self.loss_combobox = ttk.Combobox(frame, width=10, state="readonly", values=list(sp.LOSSES))
        self.loss_combobox.set(DEFAULTS["loss"])
        self.loss_combobox.grid(row=6, column=1, padx=4, pady=2, sticky="w")
        self._live(self.loss_combobox)
        ttk.Label(frame, text="Bin estimator:").grid(row=7, column=0, padx=4, pady=2, sticky="e")
        self.estimator_combobox = ttk.Combobox(frame, width=10, state="readonly", values=list(sp.BIN_ESTIMATORS))
        self.estimator_combobox.set(DEFAULTS["estimator"])
        self.estimator_combobox.grid(row=7, column=1, padx=4, pady=2, sticky="w")
        self._live(self.estimator_combobox)
        ttk.Label(frame, text="per log bin: rms = sqrt(mean power), logmean = geometric\nmean, median",
                  foreground="#606060", justify="left").grid(row=8, column=0, columnspan=2, padx=4, pady=(0, 2), sticky="w")

    def _build_ensemble_frame(self, master) -> None:
        frame = ttk.LabelFrame(master, text="Uncertainty ensemble")
        frame.pack(fill="x", pady=4)
        self.ensemble_var = tk.BooleanVar(master=self, value=DEFAULTS["ensemble_on"])
        self.ensemble_fmins_var = tk.StringVar(master=self, value=DEFAULTS["ensemble_fmins_hz"])
        self.ensemble_fmaxs_var = tk.StringVar(master=self, value=DEFAULTS["ensemble_fmaxs_hz"])
        self.ensemble_calibrations_var = tk.StringVar(master=self, value=DEFAULTS["ensemble_calibration_paths"])
        self.ensemble_text_var = tk.StringVar(master=self, value="")
        ttk.Checkbutton(frame, text="run on Save (refit over the band grid and the alternative tables)",
                        variable=self.ensemble_var).grid(row=0, column=0, columnspan=3, padx=4, pady=2, sticky="w")
        for row, (label, var) in enumerate((("fmin grid (Hz):", self.ensemble_fmins_var),
                                            ("fmax grid (Hz):", self.ensemble_fmaxs_var)), start=1):
            ttk.Label(frame, text=label).grid(row=row, column=0, padx=4, pady=2, sticky="e")
            entry = ttk.Entry(frame, textvariable=var, width=24)
            entry.grid(row=row, column=1, columnspan=2, padx=4, pady=2, sticky="w")
            self._live(entry)                               # validated with the other entries; the ensemble itself waits
            self.entries[label] = entry
        ttk.Label(frame, text="alt. CSVs (; separated):").grid(row=3, column=0, padx=4, pady=2, sticky="e")
        self.ensemble_calibrations_entry = ttk.Entry(frame, textvariable=self.ensemble_calibrations_var, width=18)
        self.ensemble_calibrations_entry.grid(row=3, column=1, padx=4, pady=2, sticky="w")
        self._live(self.ensemble_calibrations_entry)
        self.entries["alt. CSVs (; separated):"] = self.ensemble_calibrations_entry
        ttk.Button(frame, text="Browse", width=7, command=self.browse_ensemble_calibrations).grid(
            row=3, column=2, padx=2, pady=2, sticky="w")
        ttk.Label(frame, text="the same unit, dB flag and divisor as the calibration above;\n"
                              "the record's own table is always the first member",
                  foreground="#606060", justify="left").grid(row=4, column=0, columnspan=3, padx=4, pady=(0, 2), sticky="w")
        self.ensemble_button = ttk.Button(frame, text="Run ensemble now", command=self.compute_ensemble)
        self.ensemble_button.grid(row=5, column=0, columnspan=2, padx=4, pady=4, sticky="ew")
        ttk.Label(frame, textvariable=self.ensemble_text_var, wraplength=250, justify="left").grid(
            row=6, column=0, columnspan=3, padx=4, pady=2, sticky="w")

    def _build_buttons(self, master) -> None:
        frame = ttk.Frame(master)
        frame.pack(fill="x", pady=4)
        self.save_button = ttk.Button(frame, text="Save", command=self.save)
        self.save_button.grid(row=0, column=0, padx=4, pady=2, sticky="ew")
        ttk.Label(frame, text="recomputes as the controls change", foreground="#606060").grid(
            row=0, column=1, padx=8, pady=2, sticky="w")
        self.status_var = tk.StringVar(master=self, value="")
        ttk.Label(master, textvariable=self.status_var, wraplength=260, foreground="#a03030").pack(
            fill="x", pady=2)

    def _build_results_frame(self, master) -> None:
        frame = ttk.LabelFrame(master, text="Spectral fit")
        frame.pack(fill="x", pady=(0, 4))
        self.result_vars: dict[str, tk.StringVar] = {}
        rows = [("omega0", "Omega0:"), ("fc", "fc:"), ("n", "n:"), ("rms", "rms (ln):"),
                ("n_bins", "bins used:"), ("band", "band:"), ("taper", "taper energy:"),
                ("flags", "flags:"), ("ensemble", "ensemble 16-84 %:")]
        for i, (key, label) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=i, column=0, padx=4, pady=2, sticky="ne")
            var = tk.StringVar(master=self, value=_NA)
            ttk.Label(frame, textvariable=var, wraplength=220, justify="left").grid(
                row=i, column=1, padx=4, pady=2, sticky="w")
            self.result_vars[key] = var

    def _build_source_frame(self, master) -> None:
        frame = ttk.LabelFrame(master, text="Source parameters (SI)")
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text="Phase:").grid(row=0, column=0, padx=4, pady=2, sticky="e")
        self.phase_combobox = ttk.Combobox(frame, width=6, state="readonly", values=list(src.PHASES))
        self.phase_combobox.set("P")
        self.phase_combobox.grid(row=0, column=1, padx=4, pady=2, sticky="w")
        self.phase_combobox.bind("<<ComboboxSelected>>", self.on_phase_selected)
        self.rho_var = tk.StringVar(master=self, value="")
        self.c_var = tk.StringVar(master=self, value="")
        self.vs_var = tk.StringVar(master=self, value="")
        self.distance_var = tk.StringVar(master=self, value="")
        self.radiation_var = tk.StringVar(master=self, value=_text(src.PHASE_CONSTANTS["P"]["radiation_rms"]))
        self.radiation_hint_var = tk.StringVar(master=self, value="")
        self.free_surface_var = tk.StringVar(master=self, value="1.0")
        self.medium_entries = {"rho": self._entry(frame, "rho (kg/m^3):", self.rho_var, 1),
                               "vp": self._entry(frame, "Vp = c (m/s):", self.c_var, 2)}
        self._entry(frame, "Vs (m/s):", self.vs_var, 3)
        self.medium_entries["distance"] = self._entry(frame, "distance (m):", self.distance_var, 4)
        self.medium_entries["radiation"] = self._entry(frame, "radiation coeff.:", self.radiation_var, 5)
        ttk.Label(frame, textvariable=self.radiation_hint_var, foreground="#606060").grid(
            row=5, column=2, padx=2, pady=2, sticky="w")
        self.medium_entries["free_surface"] = self._entry(frame, "free-surface factor:", self.free_surface_var, 6)
        self.source_hint_var = tk.StringVar(master=self, value="")
        ttk.Label(frame, textvariable=self.source_hint_var,
                  foreground="#606060", justify="left").grid(row=7, column=0, columnspan=3, padx=4, pady=2, sticky="w")
        self.update_source_entry_states()
        self.source_button = ttk.Button(frame, text="Compute source", command=self.compute_source)
        self.source_button.grid(row=8, column=0, columnspan=2, padx=4, pady=4, sticky="ew")
        self.source_vars: dict[str, tk.StringVar] = {}
        rows = [("m0", "M0:"), ("mw", "Mw:"), ("radius", "radius:"), ("stress_drop", "stress drop:"),
                ("kR", "kR:"), ("warnings", "warnings:")]
        for i, (key, label) in enumerate(rows, start=9):
            ttk.Label(frame, text=label).grid(row=i, column=0, padx=4, pady=2, sticky="ne")
            var = tk.StringVar(master=self, value=_NA)
            ttk.Label(frame, textvariable=var, wraplength=220, justify="left").grid(
                row=i, column=1, columnspan=2, padx=4, pady=2, sticky="w")
            self.source_vars[key] = var
        self._update_radiation_hint()

    def _decorate_axes(self) -> None:
        # the taper axes sit behind the time axes, so the mouse lands on the trace and the
        # window edges draw on top; clear() resets this, so it is set at every redraw
        self.ax_taper.set_zorder(self.ax_time.get_zorder() - 1)
        self.ax_time.patch.set_visible(False)
        self.ax_taper.yaxis.set_label_position("right")
        self.ax_taper.yaxis.tick_right()
        self.ax_time.set_xlabel("time from trigger (ms)")
        self.ax_time.set_ylabel("signal")
        self.ax_spec.set_ylabel("amplitude spectrum")
        self.ax_resid.set_xlabel("frequency (Hz)")
        self.ax_resid.set_ylabel("ln A - ln model")
        self.ax_spec.set_xscale("log")
        self.ax_spec.set_yscale("log")
        self.ax_resid.set_xscale("log")
        for ax in self.axes:
            ax.grid(True, linestyle="--", alpha=0.3)

    # ------------------------------------------------------------- loading
    def on_event_loaded(self) -> None:
        keys = list(waveform_blocks(self.event)) if isinstance(self.event, dict) else []
        if self.record_key not in keys:
            self.record_key = pick_waveform_block(self.event, prefer_fields=("pzt",)) if keys else None
        self.record_combobox.config(values=keys)
        self.record_combobox.set(self.record_key or "")
        self._load_record()

    def on_record_selected(self, event=None) -> None:
        key = self.record_combobox.get().strip()
        if key and key != self.record_key:
            self.record_key = key
            self._load_record()

    def _load_record(self) -> None:
        """Channels of the selected record; restores the current channel's saved record."""
        self.refresh_set_choices(initial=self._first_load)
        self.strain_time, self.strain_raw = self._block_arrays(self.record_key)
        n_channels = 0 if self.strain_raw is None else int(self.strain_raw.shape[0])
        block = waveform_block(self.event, self.record_key) if (self.record_key and isinstance(self.event, dict)) else None
        self.channel_labels = waveform_channels(block) if (n_channels and block is not None) else []
        self.channel_combobox.config(values=[self._channel_label(i) for i in range(n_channels)])
        saved = self.saved_channels()
        current = self.current_channel()
        if n_channels == 0:
            self.channel_combobox.set("")
            self.status_var.set("event has no full-rate record (waveform or strain block)")
            self._clear_channel_state()
            return
        if self._first_load and saved and current not in saved:
            current = min(saved)
        if current is None or not (0 <= current < n_channels):
            current = 0
        self.channel_combobox.set(self._channel_label(current))
        if self._first_load and not saved and self.record_trigger_time() is not None:
            # the event time is a pick on the slow mechanical channels; the waves
            # arrive at the recorder's own trigger, often tens of ms earlier
            self.trigger_combobox.set(TRIGGER_RECORD)
        if self._first_load and current not in saved:
            self.apply_settings(self.experiment_settings())        # the last saved controls, if any
        if self._first_load:
            self.apply_medium(self.experiment_medium())
        self._first_load = False
        self._default_fmax()
        self.restore_channel()

    def _default_fmax(self) -> None:
        """Fill a blank fmax with a fraction of the record's sampling rate."""
        if self.fmax_var.get().strip() == "" and self.strain_time is not None:
            try:
                # 4 significant digits: the sampling rate comes from a median of time steps
                self.fmax_var.set(_text(float(f"{FMAX_FRACTION_OF_FS * sp.sampling_rate(self.strain_time):.4g}")))
            except ValueError:
                pass

    def _channel_label(self, channel) -> str:
        """Combobox text of a channel: ``'3: pzt_4'`` when the record names its fields, else ``'3'``."""
        channel = int(channel)
        if 0 <= channel < len(self.channel_labels) and self.channel_labels[channel] != str(channel):
            return f"{channel}: {self.channel_labels[channel]}"
        return str(channel)

    def channel_field(self, channel) -> Optional[str]:
        """Field name of a channel of the current record, or None when unnamed."""
        try:
            channel = int(channel)
        except (TypeError, ValueError):
            return None
        if 0 <= channel < len(self.channel_labels) and self.channel_labels[channel] != str(channel):
            return self.channel_labels[channel]
        return None

    def _block_arrays(self, key):
        block = waveform_block(self.event, key) if (key and isinstance(self.event, dict)) else None
        if block is None:
            return None, None
        try:
            time = np.asarray(waveform_time(block), dtype=float).ravel()
            raw = np.asarray(waveform_data(block), dtype=float)
        except (KeyError, TypeError, ValueError):
            return None, None
        if raw.ndim == 1:
            raw = raw[None, :]
        if raw.ndim != 2 or time.size < 2 or raw.shape[1] != time.size:
            return None, None
        return time, raw

    def current_channel(self) -> Optional[int]:
        text = self.channel_combobox.get().split(":")[0].strip()
        return int(text) if text.isdigit() else None

    def set_channel(self, channel: int) -> None:
        """Select a channel (restoring its saved record if there is one)."""
        self.channel_combobox.set(self._channel_label(channel))
        self.on_channel_selected()

    def on_channel_selected(self, event=None) -> None:
        self.restore_channel()

    # -------------------------------------------------- radiation coefficient
    def radiation_is_default(self) -> bool:
        """True when the radiation entry holds a phase RMS value (any phase),
        i.e. the view put it there and nobody typed over it."""
        try:
            value = float(self.radiation_var.get().strip())
        except ValueError:
            return False
        return any(math.isclose(value, float(c["radiation_rms"]), rel_tol=1e-8)
                   for c in src.PHASE_CONSTANTS.values())

    def _update_radiation_hint(self) -> None:
        phase = self.phase_combobox.get()
        if phase in src.PHASES:
            self.radiation_hint_var.set(f"RMS for {phase}: {src.PHASE_CONSTANTS[phase]['radiation_rms']:.3f}")
        else:
            self.radiation_hint_var.set("")

    def on_phase_selected(self, event=None) -> None:
        """Keep the radiation entry at the phase RMS while it still holds a
        default; a typed (or restored non-default) value is never overwritten.
        The hint label always shows the RMS of the selected phase."""
        phase = self.phase_combobox.get()
        if phase in src.PHASES and self.radiation_is_default():
            self.radiation_var.set(_text(src.PHASE_CONSTANTS[phase]["radiation_rms"]))
        self._update_radiation_hint()
        self.schedule_compute(now=True)

    # ----------------------------------------------------------- records
    def current_block(self):
        """The selected record (a waveform record or the PSU strain block), or None."""
        return waveform_block(self.event, self.record_key) if (self.record_key and isinstance(self.event, dict)) else None

    @staticmethod
    def _channel_entries(store) -> list:
        """``[(channel, record)]`` of a ``{"channels": ...}`` map; keys ``'ch<k>'``
        or legacy digits, or a LIST (an HDF5 loader turns contiguous digit keys
        into one, its entries being channels 0..n-1).  Only dict records count."""
        channels = store.get("channels") if isinstance(store, dict) else None
        if isinstance(channels, dict):
            items = list(channels.items())
        elif isinstance(channels, (list, tuple)):
            items = list(enumerate(channels))
        else:
            items = []
        out = []
        for key, rec in items:
            k = parse_channel_key(key)
            if k is not None and isinstance(rec, dict):
                out.append((k, rec))
        return out

    def legacy_channels(self) -> dict:
        """Records of this view's record still kept in the event-level map of older files."""
        out = {}
        for k, rec in self._channel_entries(self.load_results()):
            if rec.get("record", self.record_key) == self.record_key:
                out[k] = rec
        return out

    def saved_channels(self) -> dict:
        """``{channel: record}`` of the current result set saved with the selected
        record (``spectrum`` under the waveform record or the strain block);
        the default set also reads any channel of it still in the older
        event-level map."""
        block = self.current_block()
        channels = set_channels(block.get(SPECTRUM_KEY), self.result_set) if isinstance(block, dict) else None
        out = dict(self._channel_entries({"channels": channels}))
        if self.result_set == DEFAULT_SET:
            for k, rec in self.legacy_channels().items():
                out.setdefault(k, rec)
        return out

    # ------------------------------------------------------------- result sets
    def known_sets(self) -> list:
        """The result sets to offer: the experiment's, the selected record's and the current one."""
        names = {DEFAULT_SET, self.result_set}
        sets = get_setting(self.data_manager.data, SETTING, SETS_SETTING)
        if isinstance(sets, dict):
            names.update(str(k) for k in sets)
        block = self.current_block()
        if isinstance(block, dict):
            names.update(set_names(block.get(SPECTRUM_KEY)))
        return [DEFAULT_SET] + sorted(names - {DEFAULT_SET})

    def refresh_set_choices(self, initial: bool = False) -> None:
        """Fill the result-set box; on the first load open on the set last saved
        for the experiment (``current_set``), else the default one."""
        if initial:
            try:
                self.result_set = set_name(get_setting(self.data_manager.data, SETTING, CURRENT_SET_SETTING))
            except ValueError:
                self.result_set = DEFAULT_SET
        self.set_combobox.config(values=self.known_sets())
        self.set_combobox.set(self.result_set)

    def on_set_changed(self, event=None) -> None:
        """The chosen or typed result set: the current channel's record saved in
        it, else the set's experiment settings and sensor values, then a recompute."""
        try:
            name = set_name(self.set_combobox.get())
        except ValueError as e:
            self.status_var.set(str(e))
            self.set_combobox.set(self.result_set)
            return
        if name == self.result_set:
            return
        self.result_set = name
        self.refresh_set_choices()
        if self.current_channel() not in self.saved_channels():
            self.apply_settings(self.experiment_settings())
        self.restore_channel()

    def _clear_channel_state(self) -> None:
        self.window = self.spectrum = self.fit = None
        self.record = None
        self.show_fit(None)
        self.show_source(None)
        self.plot()

    def restore_channel(self) -> None:
        """Show the saved record of the current channel (parameters and
        numbers) if there is one; otherwise clear the results.  A record
        that cannot be displayed is reported in the status line and dropped
        (the controls keep whatever was applied), never raised."""
        self._clear_channel_state()
        self._saved_for_compare = None
        channel = self.current_channel()
        rec = self.saved_channels().get(channel) if channel is not None else None
        if not isinstance(rec, dict):
            self.apply_sensor(self.experiment_sensor(channel))      # the sensor's own stored values, if any
            self.live_compute()
            return
        try:
            self.record = copy.deepcopy(rec)
            # a record is keyed by its channel; a hand-edited / partial record
            # without the field is filed under the channel it was found at
            self.record.setdefault("channel", int(channel))
            self.apply_record(self.record)
            self._default_fmax()                         # a partial record may not carry a band
            self.show_record(self.record)
            self.show_source(self.record.get("source"))
            self.plot()
        except _RECORD_ERRORS as e:
            self.record = None
            self.show_fit(None)
            self.show_source(None)
            try:
                self.plot()
            except _RECORD_ERRORS:
                pass
            self.status_var.set(f"channel {channel}: saved record unreadable "
                                f"({type(e).__name__}: {e}) - change a control to recompute")
            return
        self._saved_for_compare = self.record
        self.live_compute()

    # ------------------------------------------------------- live updates
    def schedule_compute(self, now: bool = False) -> None:
        """Recompute at once, or once the controls have been quiet for
        ``LIVE_DELAY_MS`` (so a half-typed number is not reported as an error)."""
        pending = getattr(self, "_compute_after_id", None)
        if pending is not None:
            try:
                self.after_cancel(pending)
            except tk.TclError:
                pass
            self._compute_after_id = None
        if now:
            self.live_compute()
        else:
            self._compute_after_id = self.after(LIVE_DELAY_MS, self.live_compute)

    def live_compute(self) -> None:
        """Window, spectrum and fit for the current controls, then the source
        parameters when the medium is given; the status says whether the
        result still matches the channel's saved record."""
        self._compute_after_id = None
        if self.strain_raw is None or not self.winfo_exists():
            return
        saved = getattr(self, "_saved_for_compare", None)
        if not self.compute():
            # stale numbers must not survive a failed recompute; the status keeps the reason.  The
            # last window stays on the top panel, so the windows can be dragged back to a valid place
            reason = self.status_var.get()
            self.spectrum = self.fit = None
            self.record = None
            self.show_fit(None)
            self.show_source(None)
            self.plot(not_computed=reason)
            self.status_var.set(reason)
            return
        if isinstance(saved, dict):
            comparison = self.compare_with_saved(saved)
            self.status_var.set(self.status_var.get().replace("- not saved", "- " + comparison))
            if comparison == "matches the saved record" and self._same_ensemble_inputs(saved):
                # the same fit from the same inputs: the saved ensemble still holds
                self.record["ensemble"] = copy.deepcopy(saved["ensemble"])
                self.show_ensemble(self.record["ensemble"])
        self.auto_source()

    def _same_ensemble_inputs(self, saved: dict) -> bool:
        if not (isinstance(self.record, dict) and isinstance(saved.get("ensemble"), dict)):
            return False
        try:
            return all(list(_as_list(self.record.get(key)) or []) == list(_as_list(saved.get(key)) or [])
                       for key in ("ensemble_fmins_hz", "ensemble_fmaxs_hz")) and \
                _parse_paths(_paths_text(self.record.get("ensemble_calibration_paths"))) == \
                _parse_paths(_paths_text(saved.get("ensemble_calibration_paths")))
        except (TypeError, ValueError):
            return False

    def compare_with_saved(self, saved: dict) -> str:
        """'matches the saved record' when the fresh fit has the saved
        trigger time, Omega0 and fc, else a hint to save."""
        fresh = self.record or {}
        try:
            same = math.isclose(float(fresh["trigger_time_s"]), float(saved["trigger_time_s"]), rel_tol=1e-9, abs_tol=1e-9)
            for key in ("omega0", "fc"):
                a, b = float(fresh["fit"][key]), float(saved["fit"][key])
                same = same and (math.isclose(a, b, rel_tol=1e-6) or (math.isnan(a) and math.isnan(b)))
        except (KeyError, TypeError, ValueError):
            same = False
        return "matches the saved record" if same else "differs from the saved record (Save to update)"

    def auto_source(self) -> None:
        """Source parameters without a button: computed when the fit is valid,
        the spectrum calibrated and the medium entries filled in; otherwise the
        source section simply stays empty."""
        fitd = self.record.get("fit") if isinstance(self.record, dict) else None
        if not (isinstance(fitd, dict) and fitd.get("valid") and fitd.get("amp_units") == "m*s"):
            return
        try:
            self.read_source_inputs()
        except ValueError:
            return
        status = self.status_var.get()
        if self.compute_source():
            self.status_var.set(f"{status}; source parameters computed")
        else:
            self.status_var.set(f"{status}; source: {self.status_var.get()}")

    def apply_record(self, rec: dict) -> None:
        """Put a record's parameters into the controls."""
        g = rec.get
        self.trigger_combobox.set(g("trigger", TRIGGER_EVENT_TIME) if g("trigger") in TRIGGERS else TRIGGER_EVENT_TIME)
        self.pre_var.set(_text(g("pre_ms", DEFAULTS["pre_ms"])))
        self.post_var.set(_text(g("post_ms", DEFAULTS["post_ms"])))
        self.taper_combobox.set(g("taper") if g("taper") in sp.TAPER_KINDS else DEFAULTS["taper"])
        self.baseline_combobox.set(g("baseline") if g("baseline") in sp.BASELINE_MODES else DEFAULTS["baseline"])
        self.highpass_var.set(_text(g("highpass_hz")))
        self.noise_start_var.set(_text(g("noise_start_ms")))
        self.noise_end_var.set(_text(g("noise_end_ms")))
        self.remove_step_var.set(bool(g("remove_step", False)))
        self.calibration_path_var.set(str(g("calibration_path") or ""))
        self.unit_combobox.set(g("calibration_unit") if g("calibration_unit") in CALIBRATION_UNITS
                               else DEFAULTS["calibration_unit"])
        self.calibration_db_var.set(bool(g("calibration_db", False)))
        self.calibration_divisor_var.set(_text(g("calibration_divisor")))
        self.set_calibration_table(self._record_table(rec))
        self.t_star_var.set(_text(g("t_star_s")))
        self.q_path_var.set(str(g("q_csv_path") or ""))
        self.q_travel_var.set(_text(g("q_travel_time_s")))
        self.q_hold_var.set(bool(g("q_hold_ends", False)))
        self.set_attenuation_table(self._record_q_table(rec))
        self.fmin_var.set(_text(g("fmin_hz", DEFAULTS["fmin_hz"])))
        self.fmax_var.set(_text(g("fmax_hz")))
        self.bins_var.set(_text(g("bins_per_decade", DEFAULTS["bins_per_decade"])))
        self.snr_var.set(_text(g("snr_min")))
        self.n_fixed_var.set(bool(g("n_fixed", True)))
        self.n_max_var.set(_text(g("n_max", DEFAULTS["n_max"])) or DEFAULTS["n_max"])   # absent before 2026-11
        self.loss_combobox.set(g("loss") if g("loss") in sp.LOSSES else DEFAULTS["loss"])
        self.estimator_combobox.set(g("estimator") if g("estimator") in sp.BIN_ESTIMATORS else DEFAULTS["estimator"])
        self.ensemble_var.set(bool(g("ensemble_on", DEFAULTS["ensemble_on"])))
        for key, var in (("ensemble_fmins_hz", self.ensemble_fmins_var), ("ensemble_fmaxs_hz", self.ensemble_fmaxs_var)):
            value = g(key)
            if value is not None:
                var.set(_frequency_text(value) or DEFAULTS[key])
        if "ensemble_calibration_paths" in rec:
            self.ensemble_calibrations_var.set(_paths_text(g("ensemble_calibration_paths")))
        inputs = g("source_inputs")
        if not isinstance(inputs, dict):                 # records written before the inputs were kept
            source = g("source")
            inputs = source.get("constants") if isinstance(source, dict) else None
        if isinstance(inputs, dict):
            self.apply_source_inputs(inputs)
        self._update_radiation_hint()
        self.update_source_entry_states()

    def apply_source_inputs(self, inputs: dict, only: Optional[tuple] = None) -> None:
        """Put source entries (the keys of :meth:`source_inputs`, ``only`` those
        named when given) into the controls; a missing or None value leaves the
        entry as it is."""
        g = inputs.get
        targets = {"phase": None, "rho_kg_m3": self.rho_var, "vp_m_s": self.c_var, "vs_m_s": self.vs_var,
                   "distance_m": self.distance_var, "radiation_coefficient": self.radiation_var,
                   "free_surface_factor": self.free_surface_var}
        for key, var in targets.items():
            if only is not None and key not in only:
                continue
            value = g(key)
            if value is None:
                continue
            if key == "phase":
                value = _text_value(value)
                if value in src.PHASES:
                    self.phase_combobox.set(value)
            else:
                var.set(_text(value))

    # ------------------------------------------------- experiment-level settings
    def sensor_key(self, channel=None) -> Optional[str]:
        """Where a channel's per-sensor values live in ``analysis['pzt_spectrum']['sensors']``:
        its field name (``pzt_1``), else ``ch<k>``."""
        channel = self.current_channel() if channel is None else channel
        if channel is None:
            return None
        return self.channel_field(channel) or channel_key(channel)

    def experiment_settings(self) -> dict:
        """The current result set's last saved controls (the default set also
        reads the flat ``settings`` of files written before result sets)."""
        data = self.data_manager.data
        value = get_setting(data, SETTING, SETS_SETTING, self.result_set, "settings")
        if value is None and self.result_set == DEFAULT_SET:
            value = get_setting(data, SETTING, "settings")
        return value if isinstance(value, dict) else {}

    def experiment_medium(self) -> dict:
        value = get_setting(self.data_manager.data, MEDIUM_SETTING)
        return value if isinstance(value, dict) else {}

    def experiment_sensor(self, channel=None) -> dict:
        key = self.sensor_key(channel)
        data = self.data_manager.data
        value = get_setting(data, SETTING, SETS_SETTING, self.result_set, "sensors", key) if key else None
        if value is None and key and self.result_set == DEFAULT_SET:
            value = get_setting(data, SETTING, "sensors", key)
        return value if isinstance(value, dict) else {}

    def apply_settings(self, settings: dict) -> None:
        """The experiment's last saved controls into the view (a record's keys)."""
        if not settings:
            return
        rec = {k: settings[k] for k in SETTINGS_KEYS if k in settings}
        for key in ("trigger", "taper", "baseline", "loss"):
            if key in rec:
                rec[key] = _text_value(rec[key])
        inputs = {k: rec.pop(k) for k in ("phase", "radiation_coefficient") if k in rec}
        keep = {"calibration_path": self.calibration_path_var.get(), "calibration_unit": self.unit_combobox.get(),
                "calibration_db": bool(self.calibration_db_var.get()), "t_star_s": self.t_star_var.get(),
                "calibration_divisor": self.calibration_divisor_var.get(), "calibration": self.calibration_table,
                "ensemble_calibration_paths": self.ensemble_calibrations_var.get(),
                "q_csv_path": self.q_path_var.get(), "q_travel_time_s": self.q_travel_var.get(),
                "q_hold_ends": bool(self.q_hold_var.get()), "attenuation": self.attenuation_table}
        self.apply_record({**keep, **rec, "source_inputs": inputs})

    def apply_medium(self, medium: dict) -> None:
        self.apply_source_inputs(medium, only=("rho_kg_m3", "vp_m_s", "vs_m_s"))

    def apply_sensor(self, entry: dict) -> None:
        """A sensor's stored calibration, t*, distance and free-surface factor."""
        if not entry:
            return
        if "calibration" in entry:
            table = entry.get("calibration")
            self.set_calibration_table(table if isinstance(table, dict) else None)
            self.calibration_path_var.set(str(_text_value(entry.get("calibration_path") or "")))
        unit = _text_value(entry.get("calibration_unit", self.unit_combobox.get()))
        self.unit_combobox.set(unit if unit in CALIBRATION_UNITS else DEFAULTS["calibration_unit"])
        if "calibration_db" in entry:
            self.calibration_db_var.set(bool(entry.get("calibration_db")))
        if "calibration_divisor" in entry:
            self.calibration_divisor_var.set(_text(entry.get("calibration_divisor")))
        if "ensemble_calibration_paths" in entry:
            self.ensemble_calibrations_var.set(_paths_text(entry.get("ensemble_calibration_paths")))
        if "t_star_s" in entry:
            self.t_star_var.set(_text(entry.get("t_star_s")))
        if "attenuation" in entry:
            table = entry.get("attenuation")
            self.set_attenuation_table(table if isinstance(table, dict) else None)
            self.q_path_var.set(str(_text_value(entry.get("q_csv_path") or "")))
        if "q_travel_time_s" in entry:
            self.q_travel_var.set(_text(entry.get("q_travel_time_s")))
        if "q_hold_ends" in entry:
            self.q_hold_var.set(bool(entry.get("q_hold_ends")))
        self.apply_source_inputs(entry, only=("distance_m", "free_surface_factor"))

    def store_experiment_settings(self) -> None:
        """What Save keeps for the whole experiment: the controls, the medium
        and the current sensor's own values, so that every later window opens
        with them."""
        rec, inputs = self.record, self.source_inputs()
        data = self.data_manager.data
        name = self.result_set
        settings = {k: rec[k] for k in SETTINGS_KEYS if k in rec}
        settings.update({"phase": inputs["phase"], "radiation_coefficient": inputs["radiation_coefficient"]})
        set_setting(data, settings, SETTING, SETS_SETTING, name, "settings")
        medium = {k: inputs[k] for k in ("rho_kg_m3", "vp_m_s", "vs_m_s") if inputs[k] is not None}
        if medium:
            set_setting(data, {**self.experiment_medium(), **medium}, MEDIUM_SETTING)
        key = self.sensor_key(rec.get("channel"))
        if key:
            entry = {"calibration": copy.deepcopy(self.calibration_table), "calibration_path": rec["calibration_path"],
                     "calibration_unit": rec["calibration_unit"], "calibration_db": bool(rec["calibration_db"]),
                     "calibration_divisor": rec.get("calibration_divisor"),
                     "t_star_s": rec["t_star_s"], "distance_m": inputs["distance_m"],
                     "free_surface_factor": inputs["free_surface_factor"],
                     "ensemble_calibration_paths": list(rec.get("ensemble_calibration_paths") or []),
                     "q_csv_path": rec.get("q_csv_path"), "q_travel_time_s": rec.get("q_travel_time_s"),
                     "q_hold_ends": bool(rec.get("q_hold_ends", False)),
                     "attenuation": copy.deepcopy(self.attenuation_table)}
            set_setting(data, entry, SETTING, SETS_SETTING, name, "sensors", key)
        set_setting(data, name, SETTING, CURRENT_SET_SETTING)
        self._move_flat_settings()

    def _move_flat_settings(self) -> None:
        """The ``settings`` and ``sensors`` a file kept directly under
        ``analysis['pzt_spectrum']`` (written before result sets) belong to the
        default set; what the default set already has wins."""
        data = self.data_manager.data
        for key in ("settings", "sensors"):
            flat = get_setting(data, SETTING, key)
            if flat is None:
                continue
            if isinstance(flat, dict):
                if key == "sensors":
                    for sensor, entry in flat.items():
                        if get_setting(data, SETTING, SETS_SETTING, DEFAULT_SET, key, sensor) is None:
                            set_setting(data, entry, SETTING, SETS_SETTING, DEFAULT_SET, key, sensor)
                elif get_setting(data, SETTING, SETS_SETTING, DEFAULT_SET, key) is None:
                    set_setting(data, flat, SETTING, SETS_SETTING, DEFAULT_SET, key)
            delete_setting(data, SETTING, key)

    # ---------------------------------------------------------- parameters
    def read_parameters(self) -> dict:
        """Validated spectrum/fit parameters from the controls (ValueError on bad input)."""
        if self.strain_time is None or self.strain_raw is None:
            raise ValueError("event has no full-rate record (waveform or strain block)")
        channel = self.current_channel()
        if channel is None or not (0 <= channel < self.strain_raw.shape[0]):
            raise ValueError("select a channel")
        trigger = self.trigger_combobox.get()
        if trigger not in TRIGGERS:
            raise ValueError("select a trigger")
        p = {
            "record": self.record_key,
            "field": self.channel_field(channel),
            "channel": int(channel),
            "trigger": trigger,
            "pre_ms": _parse_float(self.pre_var.get(), "pre window (ms)", lo=0.0),
            "post_ms": _parse_float(self.post_var.get(), "post window (ms)", lo=0.0),
            "noise_start_ms": _parse_float(self.noise_start_var.get(), "noise start (ms)", required=False),
            "noise_end_ms": _parse_float(self.noise_end_var.get(), "noise end (ms)", required=False),
            "taper": self.taper_combobox.get(),
            "baseline": self.baseline_combobox.get(),
            "highpass_hz": None,
            "remove_step": bool(self.remove_step_var.get()),
            "calibration_path": self.calibration_path_var.get().strip(),
            "calibration_unit": self.unit_combobox.get(),
            "calibration_db": bool(self.calibration_db_var.get()),
            "calibration_divisor": _parse_float(self.calibration_divisor_var.get(), "divide gain by",
                                                required=False, lo=0.0, strict_lo=True),
            "t_star_s": _parse_float(self.t_star_var.get(), "t* (s)", required=False, lo=0.0),
            "q_csv_path": self.q_path_var.get().strip(),
            "q_travel_time_s": _parse_float(self.q_travel_var.get(), "travel time (s)", required=False, lo=0.0),
            "q_hold_ends": bool(self.q_hold_var.get()),
            "fmin_hz": _parse_float(self.fmin_var.get(), "fmin (Hz)", lo=0.0, strict_lo=True),
            "fmax_hz": _parse_float(self.fmax_var.get(), "fmax (Hz)", lo=0.0, strict_lo=True),
            "bins_per_decade": _parse_int(self.bins_var.get(), "bins per decade"),
            "snr_min": _parse_float(self.snr_var.get(), "SNR min", required=False, lo=0.0),
            "n_fixed": bool(self.n_fixed_var.get()),
            "n_max": _parse_float(self.n_max_var.get(), "n max", lo=N_BOUNDS[0], strict_lo=True),
            "loss": self.loss_combobox.get(),
            "estimator": self.estimator_combobox.get(),
            "ensemble_on": bool(self.ensemble_var.get()),
            "ensemble_fmins_hz": _parse_frequencies(self.ensemble_fmins_var.get(), "ensemble fmin grid", se.DEFAULT_FMINS_HZ),
            "ensemble_fmaxs_hz": _parse_frequencies(self.ensemble_fmaxs_var.get(), "ensemble fmax grid", se.DEFAULT_FMAXS_HZ),
            "ensemble_calibration_paths": _parse_paths(self.ensemble_calibrations_var.get()),
        }
        if p["pre_ms"] + p["post_ms"] <= 0:
            raise ValueError("pre + post window must be > 0")
        if (p["noise_start_ms"] is None) != (p["noise_end_ms"] is None):
            raise ValueError("give both noise start and noise end (ms from the trigger), or neither")
        if p["noise_start_ms"] is not None:
            if p["noise_start_ms"] >= p["noise_end_ms"]:
                raise ValueError("noise start must be before noise end")
            if p["noise_end_ms"] > -p["pre_ms"] + 1e-9 and p["noise_start_ms"] < p["post_ms"] - 1e-9:
                raise ValueError("the noise window must lie before the signal window (end <= -pre) "
                                 "or after it (start >= post)")
        if p["taper"] not in sp.TAPER_KINDS:
            raise ValueError("select a taper")
        if p["baseline"] not in sp.BASELINE_MODES:
            raise ValueError("select a baseline mode")
        if p["baseline"] == "highpass":
            p["highpass_hz"] = _parse_float(self.highpass_var.get(), "highpass cutoff (Hz)", lo=0.0, strict_lo=True)
        if p["calibration_unit"] not in CALIBRATION_UNITS:
            raise ValueError("select a calibration unit")
        if p["calibration_unit"] != "V/V" and not p["calibration_path"] and self.calibration_table is None:
            raise ValueError(f"calibration unit {p['calibration_unit']} needs a CSV loaded "
                             "(choose V/V for an uncalibrated spectrum)")
        if p["fmax_hz"] <= p["fmin_hz"]:
            raise ValueError("fmax must be larger than fmin")
        if p["loss"] not in sp.LOSSES:
            raise ValueError("select a loss")
        if p["estimator"] not in sp.BIN_ESTIMATORS:
            raise ValueError("select a bin estimator")
        if p["q_csv_path"] or self.attenuation_table is not None:
            if p["t_star_s"] is not None:
                raise ValueError("give either t* or a Q^-1(f) table, not both (clear one of them)")
            if p["q_travel_time_s"] is None:
                raise ValueError("the Q^-1(f) table needs the travel time (s) of this sensor")
        if not se.band_grid(p["ensemble_fmins_hz"], p["ensemble_fmaxs_hz"]):
            raise ValueError("the ensemble grid needs an fmin below an fmax")
        return p

    @staticmethod
    def parameter_warnings(p: dict) -> list[str]:
        """Caveats about a valid parameter set that do not stop the computation.

        With the tukey taper the first ``alpha/2`` of the window is ramped;
        a pre window shorter than that puts the trigger sample (and the
        pulse that follows it) on the ramp, so ``Omega0`` comes out low.
        """
        out = []
        if p.get("taper") == "tukey":
            total = float(p["pre_ms"]) + float(p["post_ms"])
            ramp_ms = 0.5 * TUKEY_ALPHA * total
            if float(p["pre_ms"]) < ramp_ms:
                out.append(f"pre window {float(p['pre_ms']):g} ms is shorter than the tukey ramp "
                           f"({ramp_ms:g} ms = {TUKEY_ALPHA / 2:.0%} of the window): the trigger sample "
                           f"is not in the taper's flat region and Omega0 is attenuated - use pre >= "
                           f"post / {(2.0 / TUKEY_ALPHA) - 1.0:g}")
        return out

    def record_trigger_time(self) -> Optional[float]:
        """The selected record's own trigger time on the run clock, or None."""
        block = self.current_block()
        try:
            run = self.get_run()
        except Exception:
            run = None
        return waveform_trigger_time(block, run, self.record_key)

    def trigger_time(self, p: dict) -> float:
        """Absolute trigger time for the parameters ``p`` (ValueError if unavailable)."""
        if p["trigger"] == TRIGGER_EVENT_TIME:
            try:
                t = float(self.event["event_time"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("event has no event_time") from None
        elif p["trigger"] == TRIGGER_RECORD:
            t_rec = self.record_trigger_time()
            if t_rec is None:
                raise ValueError(f"{self.record_key} has no trigger time of its own (continuous recorder or "
                                 "older event): use event_time or a picked arrival")
            t = t_rec
        else:
            try:
                picks = read_arrivals(waveform_block(self.event, self.record_key), self.event)
                arrivals = np.asarray(picks["arrival_time"], dtype=float).ravel()
                t = float(arrivals[p["channel"]])
            except (KeyError, TypeError, ValueError, IndexError):
                raise ValueError(f"no picked arrival for channel {p['channel']} "
                                 f"({self.record_key}: run Pick Arrivals on this record first)") from None
        if not math.isfinite(t):
            raise ValueError("trigger time is not finite")
        if not (self.strain_time[0] <= t <= self.strain_time[-1]):
            raise ValueError(f"trigger time {t:g} s lies outside the record")
        return t

    def load_calibration(self, p: dict) -> Optional[sp.Calibration]:
        """The calibration for the parameters ``p``: the CSV in the path entry is
        read (and its table kept) when it is not the table already loaded, or
        the dB flag changed; otherwise the loaded table is used, so a saved
        file works without the CSV.  None when nothing is loaded."""
        path, unit, db = p["calibration_path"], p["calibration_unit"], bool(p["calibration_db"])
        divisor = 1.0 if p.get("calibration_divisor") is None else float(p["calibration_divisor"])
        table = self.calibration_table
        same_file = table is not None and str(table.get("source") or "") == path
        same_divisor = table is not None and _table_divisor(table) == divisor
        if path and (not same_file or bool(table.get("gain_db", False)) != db or not same_divisor):
            try:
                cal = sp.load_calibration_csv(path, unit, db, gain_divisor=divisor)
            except (OSError, ValueError, KeyError, TypeError) as e:
                raise ValueError(f"calibration CSV: {e}") from None
            except Exception as e:  # pandas parser errors etc.
                raise ValueError(f"calibration CSV could not be read: {type(e).__name__}: {e}") from None
            self.set_calibration_table(dict(cal.as_dict(), gain_db=db, gain_divisor=divisor))
            return cal
        if table is None:
            return None
        if not same_divisor:                             # the stored gains carry the old divisor
            raise ValueError("the divisor changed: enter the CSV path again to re-read the table with it")
        try:
            cal = sp.Calibration.from_arrays(table["frequency_hz"], table["gain"], unit,
                                             source=str(table.get("source") or ""))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"the stored calibration table is unusable ({e}): load a CSV") from None
        if str(table.get("unit")) != unit:
            self.set_calibration_table(dict(cal.as_dict(), gain_db=db, gain_divisor=divisor))
        return cal

    def load_attenuation(self, p: dict) -> Optional[sp.Attenuation]:
        """The Q^-1(f) correction for ``p``: the CSV in the Q entry is read (and
        its table kept) when it is not the table loaded; otherwise the stored
        table is used with the travel time and hold-ends choice as entered (the
        travel time differs per sensor, the table does not).  None when no
        table is loaded."""
        path, travel, hold = p["q_csv_path"], p["q_travel_time_s"], bool(p["q_hold_ends"])
        table = self.attenuation_table
        same_file = table is not None and str(table.get("source") or "") == path
        if path and not same_file:
            try:
                att = sp.load_q_csv(path, travel, hold)
            except (OSError, ValueError, KeyError, TypeError) as e:
                raise ValueError(f"Q^-1(f) CSV: {e}") from None
            except Exception as e:
                raise ValueError(f"Q^-1(f) CSV could not be read: {type(e).__name__}: {e}") from None
            self.set_attenuation_table(att.as_dict())
            return att
        if table is None:
            return None
        try:
            att = sp.Attenuation.from_arrays(table["frequency_hz"], table["q_inv"], travel,
                                             source=str(table.get("source") or ""), hold_ends=hold)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"the stored Q^-1(f) table is unusable ({e}): load a CSV") from None
        if float(table.get("travel_time_s", float("nan")) or float("nan")) != att.travel_time_s or bool(table.get("hold_ends", False)) != hold:
            self.set_attenuation_table(att.as_dict())
        return att

    def set_attenuation_table(self, table: Optional[dict]) -> None:
        self.attenuation_table = None if not isinstance(table, dict) else {
            k: (_as_list(v) if k in ("frequency_hz", "q_inv") else v) for k, v in table.items()}
        self.show_q_note()

    def show_q_note(self) -> None:
        table = self.attenuation_table
        if table is None:
            self.q_note_var.set("no Q^-1(f) table: constant t* only (blank = no attenuation)")
            return
        source = str(table.get("source") or "")
        name = Path(source).name if source else "stored table"
        try:
            n = len(_as_list(table.get("frequency_hz")))
            span = f"{_fmt(table.get('fmin_hz'), '.3g')} - {_fmt(table.get('fmax_hz'), '.3g')} Hz"
        except TypeError:
            n, span = 0, _NA
        hold = ", end values held outside" if table.get("hold_ends") else ", no correction outside"
        self.q_note_var.set(f"Q loaded: {name}: {n} rows, {span}, T = {_fmt(table.get('travel_time_s'), '.3g')} s{hold}")

    def clear_attenuation(self) -> None:
        self.q_path_var.set("")
        self.set_attenuation_table(None)
        self.schedule_compute(now=True)

    def browse_q(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Q^-1(f) CSV",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.q_path_var.set(path)
            self.schedule_compute(now=True)

    @staticmethod
    def _record_q_table(rec: dict) -> Optional[dict]:
        """The Q^-1(f) table a record carries: its own, else the spectrum meta's."""
        table = rec.get("attenuation")
        if not isinstance(table, dict):
            spec = rec.get("spectrum")
            meta = spec.get("meta") if isinstance(spec, dict) else None
            table = meta.get("attenuation") if isinstance(meta, dict) else None
        if isinstance(table, dict) and len(_as_list(table.get("frequency_hz"))) >= 2:
            return dict(table)
        return None

    def set_calibration_table(self, table: Optional[dict]) -> None:
        """Make ``table`` (``Calibration.as_dict()`` plus ``gain_db``, or None) the
        loaded calibration and say so under the entries."""
        self.calibration_table = None if not isinstance(table, dict) else {
            k: (_as_list(v) if k in ("frequency_hz", "gain") else v) for k, v in table.items()}
        self.show_calibration_note()

    def show_calibration_note(self) -> None:
        table = self.calibration_table
        if table is None:
            self.calibration_note_var.set("no calibration loaded: the spectrum stays in V*s")
            return
        source = str(table.get("source") or "")
        name = Path(source).name if source else "stored table"
        where = "" if not source or Path(source).exists() else " (file not found; the stored table is used)"
        try:
            n = len(_as_list(table.get("frequency_hz")))
            span = f"{_fmt(table.get('fmin_hz'), '.3g')} - {_fmt(table.get('fmax_hz'), '.3g')} Hz"
        except TypeError:
            n, span = 0, _NA
        db = ", gain read in dB" if table.get("gain_db") else ""
        divisor = _table_divisor(table)
        div = f", gain / {divisor:g}" if divisor != 1.0 else ""
        self.calibration_note_var.set(f"loaded: {name}{where}: {n} rows, {span}, {table.get('unit')}{db}{div}")

    def clear_calibration(self) -> None:
        self.calibration_path_var.set("")
        self.set_calibration_table(None)
        self.schedule_compute(now=True)

    @staticmethod
    def _record_table(rec: dict) -> Optional[dict]:
        """The calibration table a record carries: its own, else the one the
        spectrum meta kept (records written before the table was stored)."""
        table = rec.get("calibration")
        if not isinstance(table, dict):
            spec = rec.get("spectrum")
            meta = spec.get("meta") if isinstance(spec, dict) else None
            table = meta.get("calibration") if isinstance(meta, dict) else None
        if isinstance(table, dict) and len(_as_list(table.get("frequency_hz"))) >= 2:
            table = dict(table)
            table.setdefault("gain_db", bool(rec.get("calibration_db", False)))
            if rec.get("calibration_divisor") is not None:
                table.setdefault("gain_divisor", rec.get("calibration_divisor"))
            return table
        return None

    # ------------------------------------------------------------- compute
    def compute(self) -> bool:
        """Window + spectrum + fit for the current channel.  Returns True on success."""
        self.update_source_entry_states()
        try:
            p = self.read_parameters()
            trigger = self.trigger_time(p)
            calibration = self.load_calibration(p)
            attenuation = self.load_attenuation(p)
            noise_window = (None if p["noise_start_ms"] is None
                            else (p["noise_start_ms"] * 1e-3, p["noise_end_ms"] * 1e-3))
            window = sp.extract_window(self.strain_time, self.strain_raw[p["channel"]], trigger,
                                       p["pre_ms"] * 1e-3, p["post_ms"] * 1e-3, baseline=p["baseline"],
                                       highpass_hz=p["highpass_hz"], remove_step=p["remove_step"],
                                       noise_window=noise_window)
            spectrum = sp.compute_spectrum(window, taper=p["taper"], alpha=TUKEY_ALPHA,
                                           calibration=calibration, t_star_s=p["t_star_s"], attenuation=attenuation,
                                           fmin=p["fmin_hz"], fmax=p["fmax_hz"],
                                           bins_per_decade=p["bins_per_decade"], estimator=p["estimator"])
        except ValueError as e:
            self.status_var.set(str(e))
            return False
        fit = sp.fit_omega_n(spectrum, snr_min=p["snr_min"], n_fixed=p["n_fixed"], loss=p["loss"],
                             n_bounds=(N_BOUNDS[0], p["n_max"]))
        self.window, self.spectrum, self.fit = window, spectrum, fit
        self.record = self.build_record(p, trigger, spectrum, fit)
        self.show_record(self.record)
        self.show_ensemble(None)
        self.show_source(None)
        self.plot()
        # automatic noise that had to move after the signal (no room before it) is said in the status
        note = "; noise taken after the signal (no room before it)" if window.noise_region == "post" else ""
        if not fit.valid:
            self.status_var.set(f"channel {p['channel']}: fit invalid - {fit.reason}{note}")
            return True
        warnings = self.record_warnings(self.record)
        if warnings:
            self.status_var.set(f"channel {p['channel']}: fit has warnings ({fit.n_bins} bins){note} - not saved: "
                                + "; ".join(warnings))
        else:
            self.status_var.set(f"channel {p['channel']}: fit OK ({fit.n_bins} bins){note} - not saved")
        return True

    def build_record(self, p: dict, trigger_time: float, spectrum: sp.SpectrumResult,
                     fit: sp.SpectralFit) -> dict:
        """The JSON-like record stored per channel.

        Keys: version, channel, trigger, trigger_time_s, fs_hz, pre_ms, post_ms,
        taper, baseline, highpass_hz, remove_step, calibration_path,
        calibration_unit, calibration_db, calibration_divisor, t_star_s, fmin_hz, fmax_hz,
        bins_per_decade, snr_min, n_fixed, n_max (the ceiling of a free n;
        the bounds used are ``fit['n_bounds']``), loss, estimator (the per-bin
        statistic), ensemble_on / ensemble_fmins_hz / ensemble_fmaxs_hz /
        ensemble_calibration_paths (the uncertainty ensemble's inputs) and
        ensemble (``Ensemble.as_dict()`` once run, else None), parameter_warnings
        (:meth:`parameter_warnings` of the entries), spectrum
        (``SpectrumResult.as_dict()``: binned arrays and the window/taper
        meta incl. ``taper_energy_fraction``), fit (``SpectralFit.as_dict()``
        incl. ``warnings``), source (``SourceParameters.as_dict()`` or None).
        """
        rec = {
            "version": RESULT_VERSION,
            "record": p.get("record"),
            "field": p.get("field"),
            "channel": int(p["channel"]),
            "trigger": p["trigger"],
            "trigger_time_s": float(trigger_time),
            "fs_hz": float(spectrum.meta.get("fs", float("nan"))),
            "pre_ms": float(p["pre_ms"]),
            "post_ms": float(p["post_ms"]),
            "noise_start_ms": p["noise_start_ms"],
            "noise_end_ms": p["noise_end_ms"],
            "taper": p["taper"],
            "baseline": p["baseline"],
            "highpass_hz": p["highpass_hz"],
            "remove_step": bool(p["remove_step"]),
            "calibration_path": p["calibration_path"],
            "calibration_unit": p["calibration_unit"],
            "calibration_db": bool(p["calibration_db"]),
            "calibration_divisor": p["calibration_divisor"],
            "calibration": copy.deepcopy(self.calibration_table) if spectrum.meta.get("calibration") else None,
            "t_star_s": p["t_star_s"],
            "q_csv_path": p["q_csv_path"],
            "q_travel_time_s": p["q_travel_time_s"],
            "q_hold_ends": bool(p["q_hold_ends"]),
            "attenuation": copy.deepcopy(self.attenuation_table) if spectrum.meta.get("attenuation") else None,
            "fmin_hz": float(p["fmin_hz"]),
            "fmax_hz": float(p["fmax_hz"]),
            "bins_per_decade": int(p["bins_per_decade"]),
            "snr_min": p["snr_min"],
            "n_fixed": bool(p["n_fixed"]),
            "n_max": float(p["n_max"]),
            "loss": p["loss"],
            "estimator": p["estimator"],
            "ensemble_on": bool(p["ensemble_on"]),
            "ensemble_fmins_hz": list(p["ensemble_fmins_hz"]),
            "ensemble_fmaxs_hz": list(p["ensemble_fmaxs_hz"]),
            "ensemble_calibration_paths": list(p["ensemble_calibration_paths"]),
            "ensemble": None,
            "parameter_warnings": self.parameter_warnings(p),
            "source_inputs": self.source_inputs(),
            "result_set": self.result_set,
            "spectrum": spectrum.as_dict(),
            "fit": fit.as_dict(),
            "source": None,
        }
        return rec

    def source_inputs(self) -> dict:
        """The source entries as typed (numbers, None when blank or not a number):
        kept with the record and the experiment whether or not the source
        parameters were computed, so that reopening needs no retyping."""
        def number(var):
            try:
                v = float(var.get().strip())
            except ValueError:
                return None
            return v if math.isfinite(v) else None
        phase = self.phase_combobox.get()
        return {"phase": phase if phase in src.PHASES else "P", "rho_kg_m3": number(self.rho_var),
                "vp_m_s": number(self.c_var), "vs_m_s": number(self.vs_var), "distance_m": number(self.distance_var),
                "radiation_coefficient": number(self.radiation_var),
                "free_surface_factor": number(self.free_surface_var)}

    @staticmethod
    def record_warnings(rec) -> list[str]:
        """All caveats of a record: the fit's own warnings followed by the
        parameter warnings (empty for anything that is not a dict)."""
        if not isinstance(rec, dict):
            return []
        fitd = rec.get("fit")
        out = _warning_texts(fitd.get("warnings")) if isinstance(fitd, dict) else []
        out.extend(_warning_texts(rec.get("parameter_warnings")))
        return out

    @staticmethod
    def plateau_attenuated(rec) -> Optional[str]:
        """The reason the record's plateau is attenuated (transient outside the
        taper's flat region, per the analysis meta or the fit warnings), or None."""
        if not isinstance(rec, dict):
            return None
        spec = rec.get("spectrum")
        meta = spec.get("meta") if isinstance(spec, dict) else None
        if isinstance(meta, dict) and meta.get("transient_in_flat_region") is False:
            frac = meta.get("taper_energy_fraction")
            frac_text = _fmt(frac, ".3f")
            return f"{FLAT_REGION_WARNING_PREFIX} (taper_energy_fraction={frac_text} < {sp.TAPER_FLAT_FRACTION})"
        for w in PZTSpectrumView.record_warnings(rec):
            if w.startswith(FLAT_REGION_WARNING_PREFIX):
                return w
        return None

    def read_source_inputs(self, moment: bool = False) -> dict:
        """The validated source entries; with ``moment`` (a moment spectrum)
        only the phase and Vs are required, the rest may be blank."""
        phase = self.phase_combobox.get()
        if phase not in src.PHASES:
            raise ValueError("select a phase")
        need = not moment
        return {
            "phase": phase,
            "rho": _parse_float(self.rho_var.get(), "rho", required=need, lo=0.0, strict_lo=True),
            "vp": _parse_float(self.c_var.get(), "Vp (c)", required=need, lo=0.0, strict_lo=True),
            "vs": _parse_float(self.vs_var.get(), "Vs", lo=0.0, strict_lo=True),
            "distance_m": _parse_float(self.distance_var.get(), "distance", required=need, lo=0.0, strict_lo=True),
            "radiation_coefficient": _parse_float(self.radiation_var.get(), "radiation coefficient", required=need),
            "free_surface_factor": _parse_float(self.free_surface_var.get(), "free-surface factor",
                                                required=need, lo=0.0, strict_lo=True),
        }

    def moment_spectrum(self) -> bool:
        """The calibration unit gives a moment spectrum (``V/(N m)``): the medium,
        distance, radiation and free-surface entries are then not used."""
        return self.unit_combobox.get() == sp.MOMENT_UNIT

    def update_source_entry_states(self) -> None:
        """Grey out the source entries a moment spectrum does not use, and say
        which moment formula applies."""
        moment = self.moment_spectrum()
        for entry in self.medium_entries.values():
            entry.state(["disabled"] if moment else ["!disabled"])
        self.source_hint_var.set("M0 = Omega0 of the moment spectrum (N*m*s); r = k Vs / fc"
                                 if moment else "M0 uses Vp for P, Vs for S; r = k Vs / fc.\n"
                                                "e.g. granite: rho 2700, Vp 6000, Vs 3500")

    def compute_source(self) -> bool:
        """Source parameters from the current record's fit.  Returns True on success.

        Refused (status message, no dialog) for an invalid fit and an
        uncalibrated spectrum (V*s).  A moment spectrum (N*m*s, from a
        V/(N m) response) takes ``M0 = Omega0`` and needs only the phase and
        Vs.  A plateau the analysis flagged as
        attenuated (transient outside the taper's flat region, as a long coda
        under a bh4 taper gives) is computed from anyway: the moment may be
        biased low by the taper, so the warning goes into
        ``source['warnings']``, the warnings row and the status.
        """
        rec = self.record
        fitd = rec.get("fit") if isinstance(rec, dict) else None
        if not isinstance(fitd, dict):
            self.status_var.set("compute a spectrum first")
            return False
        if not fitd.get("valid"):
            self.status_var.set("the spectral fit is invalid; no source parameters")
            return False
        units = fitd.get("amp_units")
        if units not in (sp.DISPLACEMENT_SPECTRUM_UNITS, sp.MOMENT_SPECTRUM_UNITS):
            self.status_var.set("spectrum is uncalibrated (V*s): load a calibration CSV with a physical "
                                "unit before computing source parameters")
            return False
        moment = units == sp.MOMENT_SPECTRUM_UNITS
        attenuated = self.plateau_attenuated(rec)
        try:
            q = self.read_source_inputs(moment)
            band = _band_pair(fitd.get("band_used"))
            result = src.source_parameters(
                fitd["omega0"], fitd["fc"], q["phase"], q["rho"], q["vp"], q["vs"], q["distance_m"],
                radiation_coefficient=q["radiation_coefficient"],
                free_surface_factor=1.0 if q["free_surface_factor"] is None else q["free_surface_factor"],
                f_plateau_hz=None if (band is None or moment) else band[0], moment_spectrum=moment)
        except (ValueError, KeyError, TypeError) as e:
            self.status_var.set(f"source parameters: {e}" if not isinstance(e, ValueError) else str(e))
            return False
        source = result.as_dict()
        if attenuated:
            source["warnings"].append(f"Omega0 may be attenuated, M0 with it ({attenuated}); a longer pre "
                                      "window puts the transient in the taper's flat region")
        rec["source"] = source
        rec["source_inputs"] = self.source_inputs()
        self.show_source(rec["source"])
        if result.valid:
            self.status_var.set(f"source parameters computed ({q['phase']}"
                                + (", M0 = Omega0 of the moment spectrum)" if moment else ")")
                                + (", Omega0 may be attenuated" if attenuated else "") + " - not saved")
        else:
            self.status_var.set(f"source parameters invalid - {result.reason}")
        return bool(result.valid)

    # ------------------------------------------------------------ ensemble
    def ensemble_spectra(self, p: dict) -> list:
        """``(label, spectrum)`` per calibration table for the ensemble: the
        record's own table first, then each alternative CSV read with the same
        unit, dB flag and divisor; every spectrum binned over the whole grid."""
        lo, hi = min(p["ensemble_fmins_hz"]), max(p["ensemble_fmaxs_hz"])
        unit, db = p["calibration_unit"], bool(p["calibration_db"])
        divisor = 1.0 if p.get("calibration_divisor") is None else float(p["calibration_divisor"])
        tables = [(self.calibration_path_var.get().strip() or ("stored table" if self.calibration_table else "uncalibrated"),
                   self.load_calibration(p))]
        for path in p["ensemble_calibration_paths"]:
            try:
                tables.append((path, sp.load_calibration_csv(path, unit, db, gain_divisor=divisor)))
            except (OSError, ValueError, KeyError, TypeError) as e:
                raise ValueError(f"alternative calibration {Path(path).name}: {e}") from None
        attenuation = self.load_attenuation(p)
        return [(label, sp.compute_spectrum(self.window, taper=p["taper"], alpha=TUKEY_ALPHA, calibration=cal,
                                            t_star_s=p["t_star_s"], attenuation=attenuation,
                                            fmin=min(lo, p["fmin_hz"]), fmax=max(hi, p["fmax_hz"]),
                                            bins_per_decade=p["bins_per_decade"], estimator=p["estimator"]))
                for label, cal in tables]

    def compute_ensemble(self) -> bool:
        """Refit the current window over the band grid and the alternative
        calibration tables; the summary goes into ``record['ensemble']`` and
        the results row.  Returns True when it ran."""
        rec = self.record
        if not isinstance(rec, dict) or self.window is None:
            self.status_var.set("compute a spectrum first")
            return False
        try:
            p = self.read_parameters()
            spectra = self.ensemble_spectra(p)
            ens = se.fit_ensemble(spectra, se.band_grid(p["ensemble_fmins_hz"], p["ensemble_fmaxs_hz"]),
                                  snr_min=p["snr_min"], n_fixed=p["n_fixed"], loss=p["loss"],
                                  n_bounds=(N_BOUNDS[0], p["n_max"]))
        except ValueError as e:
            self.status_var.set(f"ensemble: {e}")
            return False
        ens = ens.with_moment(self._moment_per_omega0(rec))
        rec["ensemble"] = ens.as_dict()
        rec.update({"ensemble_fmins_hz": list(p["ensemble_fmins_hz"]), "ensemble_fmaxs_hz": list(p["ensemble_fmaxs_hz"]),
                    "ensemble_calibration_paths": list(p["ensemble_calibration_paths"]), "ensemble_on": bool(p["ensemble_on"])})
        self.show_ensemble(rec["ensemble"])
        self.status_var.set(f"ensemble: {ens.n_valid} of {ens.n_members} fits valid over {len(ens.bands_hz)} bands x "
                            f"{len(ens.calibrations)} table(s); {self.ensemble_text_var.get()} - not saved")
        return True

    @staticmethod
    def _moment_per_omega0(rec: dict) -> float:
        """``M0 / Omega0`` of the record's source (NaN when none): the moment of
        every ensemble member follows its plateau by this factor."""
        source = rec.get("source") if isinstance(rec, dict) else None
        fitd = rec.get("fit") if isinstance(rec, dict) else None
        if not (isinstance(source, dict) and source.get("valid") and isinstance(fitd, dict)):
            return float("nan")
        try:
            return float(source["seismic_moment_nm"]) / float(fitd["omega0"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return float("nan")

    def show_ensemble(self, ensemble) -> None:
        """The ensemble row and the frame's own text, from a stored ensemble dict."""
        if not isinstance(ensemble, dict) or not isinstance(ensemble.get("stats"), dict):
            self.result_vars["ensemble"].set(_NA)
            self.ensemble_text_var.set("not run")
            return
        st, which = ensemble_stats(ensemble)

        def span(field, fmt):
            s = st.get(field) if isinstance(st.get(field), dict) else {}
            return f"{_fmt(s.get('p16'), fmt)} - {_fmt(s.get('p84'), fmt)}"

        n_valid, n_members = ensemble.get("n_valid", _NA), ensemble.get("n_members", _NA)
        count = (f"{ensemble.get('n_resolved', _NA)} resolved of {n_valid} valid, {n_members} members" if which == "resolved"
                 else f"{n_valid}/{n_members} valid, none resolved")
        self.result_vars["ensemble"].set(f"fc {span('fc', '.4g')} Hz; Omega0 {span('omega0', '.3g')}; n {span('n', '.2f')}"
                                         + (f"; M0 {span('seismic_moment_nm', '.3g')} N m" if "seismic_moment_nm" in st else "")
                                         + f" ({count})")
        self.ensemble_text_var.set(f"fc 16-84 %: {span('fc', '.4g')} Hz, range {_fmt((st.get('fc') or {}).get('min'), '.4g')} - "
                                   f"{_fmt((st.get('fc') or {}).get('max'), '.4g')} Hz ({count})")

    def browse_ensemble_calibrations(self) -> None:
        paths = filedialog.askopenfilenames(parent=self, title="Alternative calibration CSVs",
                                            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if paths:
            self.ensemble_calibrations_var.set(PATH_SEPARATOR.join(str(p) for p in paths))

    # ---------------------------------------------------------------- save
    def save(self) -> bool:
        """Merge the current channel's record into the selected record's
        ``spectrum`` map (keys ``'ch<k>'``).  Channels of this record still in
        the older event-level map move here too, legacy digit keys renamed."""
        if not isinstance(self.record, dict):
            self.status_var.set("nothing to save - the current controls give no spectrum")
            return False
        block = self.current_block()
        if not isinstance(block, dict):
            self.status_var.set("nothing saved - no record is selected")
            return False
        channel = self.record.get("channel")
        if channel is None:
            channel = self.current_channel()
        try:
            channel = int(channel)
        except (TypeError, ValueError):
            channel = None
        if channel is None:
            self.status_var.set("nothing saved - the record has no channel and none is selected")
            return False
        self.record["channel"] = channel
        self.record.setdefault("record", self.record_key)
        self.record["result_set"] = self.result_set
        if self.ensemble_var.get() and self.record.get("ensemble") is None and not self.compute_ensemble():
            self.status_var.set(f"nothing saved - {self.status_var.get()}")
            return False
        self.record["source_inputs"] = self.source_inputs()      # as the entries are now, computed or not
        store = block.get(SPECTRUM_KEY)
        if not isinstance(store, dict):
            store = {}
        legacy = self.legacy_channels()
        if legacy:                                                # the event-level map of older files joins the default set
            default = ensure_set(store, DEFAULT_SET)["channels"]
            for k, rec in legacy.items():
                default.setdefault(channel_key(k), rec)
        target = ensure_set(store, self.result_set)               # a version-1 store becomes version 2 here
        channels = {channel_key(k): rec for k, rec in sorted(self._channel_entries(target))}
        channels[channel_key(channel)] = copy.deepcopy(self.record)
        target["channels"] = channels
        block[SPECTRUM_KEY] = store
        self._drop_legacy_channels()
        self.store_experiment_settings()
        self.refresh_set_choices()
        self._saved_for_compare = self.record
        self.app.refresh_tree()
        self.status_var.set(f"channel {channel} saved under {self.record_key}, set '{self.result_set}' "
                            f"({len(channels)} channel(s) stored); the set's controls and the sensor's "
                            "calibration are saved for the experiment")
        return True

    def _drop_legacy_channels(self) -> None:
        """Remove this record's channels from the older event-level map (they
        now live with the record); an emptied map goes away."""
        legacy = self.event.get(LEGACY_SPECTRUM_KEY) if isinstance(self.event, dict) else None
        channels = legacy.get("channels") if isinstance(legacy, dict) else None
        if isinstance(channels, dict):
            for key in list(channels):
                rec = channels[key]
                if isinstance(rec, dict) and rec.get("record", self.record_key) == self.record_key:
                    del channels[key]
            if not channels:
                del self.event[LEGACY_SPECTRUM_KEY]
        elif isinstance(channels, (list, tuple)):
            keep = [rec for rec in channels if isinstance(rec, dict) and rec.get("record", self.record_key) != self.record_key]
            if keep:
                legacy["channels"] = keep
            else:
                del self.event[LEGACY_SPECTRUM_KEY]

    # ------------------------------------------------------------- display
    def show_record(self, rec: Optional[dict]) -> None:
        """Fit numbers, taper energy and all warnings of a record."""
        if not isinstance(rec, dict):
            self.show_fit(None)
            return
        spec = rec.get("spectrum")
        meta = spec.get("meta") if isinstance(spec, dict) else None
        self.show_fit(rec.get("fit"), meta=meta if isinstance(meta, dict) else None,
                      extra_warnings=_warning_texts(rec.get("parameter_warnings")))
        self.show_ensemble(rec.get("ensemble"))

    def show_fit(self, fitd: Optional[dict], meta: Optional[dict] = None, extra_warnings=()) -> None:
        """Fill the 'Spectral fit' rows.  ``meta`` (the spectrum meta) feeds
        the taper-energy row; ``extra_warnings`` are appended to the flags."""
        for var in self.result_vars.values():
            var.set(_NA)
        if isinstance(meta, dict):
            frac = meta.get("taper_energy_fraction")
            in_flat = meta.get("transient_in_flat_region")
            if frac is not None:
                self.result_vars["taper"].set(
                    f"fraction {_fmt(frac, '.3f')} (flat region: {_yes(in_flat)})")
        if not isinstance(fitd, dict) or not fitd.get("valid"):
            if isinstance(fitd, dict) and fitd.get("reason"):
                self.result_vars["flags"].set(f"invalid: {fitd['reason']}")
            return
        units = str(fitd.get("amp_units") or "")
        self.result_vars["omega0"].set(_fmt(fitd.get("omega0"), ".4g", units))
        self.result_vars["fc"].set(_fmt(fitd.get("fc"), ".4g", "Hz"))
        n_text = _fmt(fitd.get("n"), ".3g")
        self.result_vars["n"].set(n_text + (" (fixed)" if fitd.get("n_fixed") else " (free)"))
        self.result_vars["rms"].set(_fmt(fitd.get("rms"), ".3g"))
        self.result_vars["n_bins"].set(str(fitd.get("n_bins", _NA)))
        band = _band_pair(fitd.get("band_used"))
        if band is None:
            self.result_vars["band"].set(_NA)
        else:
            self.result_vars["band"].set(f"{_fmt(band[0], '.4g')} - {_fmt(band[1], '.4g')} Hz")
        at = fitd.get("at_bounds")
        at = at if isinstance(at, dict) else {}
        bound_names = [str(k) for k, v in at.items() if v] or ["none"]
        warnings = _warning_texts(fitd.get("warnings")) + list(extra_warnings)
        flags = (f"converged: {_yes(fitd.get('converged'))}; at bound: {', '.join(bound_names)}; "
                 f"band-limited: {_yes(fitd.get('band_limited'))}")
        if warnings:
            flags += "; WARNINGS: " + "; ".join(warnings)
        self.result_vars["flags"].set(flags)

    def show_source(self, sd: Optional[dict]) -> None:
        if not isinstance(sd, dict) or not sd.get("valid"):
            for var in self.source_vars.values():
                var.set(_NA)
            if isinstance(sd, dict) and sd.get("reason"):
                self.source_vars["warnings"].set(f"invalid: {sd['reason']}")
            return
        self.source_vars["m0"].set(_fmt(sd.get("seismic_moment_nm"), ".4g", "N m"))
        self.source_vars["mw"].set(_fmt(sd.get("mw"), ".3f"))
        self.source_vars["radius"].set(_fmt(sd.get("source_radius_m"), ".4g", "m"))
        stress = _fmt(sd.get("stress_drop_pa"), ".4g")
        self.source_vars["stress_drop"].set(
            _NA if stress == _NA else f"{float(sd.get('stress_drop_pa')) / 1e6:.4g} MPa")
        kR = sd.get("kR")
        ff = sd.get("far_field_ok")
        self.source_vars["kR"].set(_fmt(kR, ".3g") + ("" if ff is None else f" (far field {'ok' if ff else 'doubtful'})"))
        warnings = _warning_texts(sd.get("warnings"))
        self.source_vars["warnings"].set("; ".join(warnings) if warnings else "none")

    def plot(self, not_computed: Optional[str] = None) -> None:
        """Redraw the three axes from the live window/spectrum (when present)
        and the current record's binned spectrum and fit; ``not_computed`` is
        the reason the last recompute failed, written over the empty spectrum."""
        for ax in self.axes:
            ax.clear()
        self.ax_taper.clear()
        self.handles = {}
        self.signal_span = self.noise_span = None
        self._decorate_axes()
        w, s, rec = self.window, self.spectrum, self.record

        if w is not None:
            kind = rec.get("taper") if isinstance(rec, dict) else None
            kind = kind if kind in sp.TAPER_KINDS else self.taper_combobox.get()
            taper = sp.taper_window(w.signal.size, kind if kind in sp.TAPER_KINDS else DEFAULTS["taper"], TUKEY_ALPHA)
            t_ms = w.t_rel * 1e3
            edges = self.window_edges_ms(w)
            # the trace around both windows, so there is room to drag them
            first, last = min(edges["noise_start"], edges["signal_start"]), max(edges["noise_end"], edges["signal_end"])
            span = max(last - first, 1e-3)
            lo, hi = first - span, last + span
            t_all = (self.strain_time - w.trigger_time) * 1e3
            keep = (t_all >= lo) & (t_all <= hi)
            if np.any(keep):
                self.ax_time.plot(t_all[keep], self.strain_raw[self.current_channel()][keep],
                                  color="#bbbbbb", lw=0.6, label="trace")
            if w.noise_available:
                t_noise = (np.arange(w.i_noise0, w.i_noise1) - w.i_trigger) / w.fs * 1e3
                self.ax_time.plot(t_noise, w.noise, color="#7f7f7f", lw=0.8, label="noise window")
                self.noise_span = self.ax_time.axvspan(edges["noise_start"], edges["noise_end"],
                                                       color="#7f7f7f", alpha=0.12, zorder=0)
            self.signal_span = self.ax_time.axvspan(edges["signal_start"], edges["signal_end"],
                                                    color="#1f77b4", alpha=0.08, zorder=0)
            self.ax_time.plot(t_ms, w.signal, color="#1f77b4", lw=1.0, label="signal window")
            self.ax_time.plot(t_ms, w.signal * taper, color="#d62728", lw=1.0, label="tapered (what is transformed)")
            self.ax_time.axvline(0.0, color="k", lw=0.8, ls=":")
            for name in HANDLES:
                if name.startswith("noise") and not w.noise_available:
                    continue
                color = "#7f7f7f" if name.startswith("noise") else "#1f77b4"
                self.handles[name] = self.ax_time.axvline(edges[name], color=color, lw=1.6, zorder=5,
                                                          label="_" + name)
            self.ax_time.set_xlim(lo, hi)
            self.ax_taper.plot(t_ms, taper, color="k", lw=0.8, ls="--", label=f"{kind} taper")
            self.ax_taper.set_ylim(0.0, 1.05)
            self.ax_taper.set_ylabel("taper")
            h1, l1 = self.ax_time.get_legend_handles_labels()
            h2, l2 = self.ax_taper.get_legend_handles_labels()
            self.ax_time.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
        else:
            self.ax_taper.set_yticks([])
            self.ax_taper.set_ylabel("")

        if s is not None:
            pos = s.f > 0
            with np.errstate(invalid="ignore"):
                self.ax_spec.loglog(s.f[pos], s.amp_signal[pos], color="#1f77b4", lw=0.6, alpha=0.35, label="signal (raw)")
                if np.any(np.isfinite(s.amp_noise)):
                    self.ax_spec.loglog(s.f[pos], s.amp_noise[pos], color="#7f7f7f", lw=0.6, alpha=0.5, label="noise (raw)")

        if isinstance(rec, dict) and isinstance(rec.get("spectrum"), dict):
            spec = rec["spectrum"]
            fitd = rec.get("fit") if isinstance(rec.get("fit"), dict) else {}
            fb = np.asarray(_as_list(spec.get("f_binned")), dtype=float)
            ab = np.asarray(_as_list(spec.get("amp_binned")), dtype=float)
            nb = np.asarray(_as_list(spec.get("noise_binned")), dtype=float)
            snr = np.asarray(_as_list(spec.get("snr")), dtype=float)
            units = str(spec.get("amp_units") or "")
            self.ax_spec.set_ylabel(f"amplitude ({units})" if units else "amplitude")
            fit_valid = bool(fitd.get("valid"))
            band = _band_pair(fitd.get("band_used")) if fit_valid else None
            try:
                model_params = (float(fitd["omega0"]), float(fitd["fc"]), float(fitd["n"])) if band else None
            except (KeyError, TypeError, ValueError):
                model_params = None
            if model_params is not None and not all(math.isfinite(v) for v in model_params):
                model_params = None
            if fb.size and ab.size == fb.size:
                used = np.isfinite(ab) & (ab > 0)
                try:
                    snr_min = None if rec.get("snr_min") is None else float(rec.get("snr_min"))
                except (TypeError, ValueError):
                    snr_min = None
                if snr_min is not None and snr.size == fb.size:
                    used &= np.isfinite(snr) & (snr >= snr_min)
                if band is not None:
                    used &= (fb >= band[0]) & (fb <= band[1])
                else:
                    used = np.zeros(fb.shape, dtype=bool)
                with np.errstate(invalid="ignore"):
                    if nb.size == fb.size and np.any(np.isfinite(nb)):
                        self.ax_spec.loglog(fb, nb, color="#4d4d4d", lw=1.0, ls="--", label="noise (binned)")
                    self.ax_spec.loglog(fb[~used], ab[~used], "o", mfc="none", mec="#1f77b4", ms=4,
                                        label="binned (unused)")
                    self.ax_spec.loglog(fb[used], ab[used], "o", color="#d62728", ms=4,
                                        label=f"binned, SNR >= {snr_min:g}" if snr_min is not None else "binned (used)")
                if model_params is not None:
                    omega0, fc, n = model_params
                    f_model = np.geomspace(band[0], band[1], 200)
                    model = sp.brune_spectrum(f_model, omega0, fc, n)
                    self.ax_spec.loglog(f_model, model, color="k", lw=1.5, label="omega-n model")
                    self.ax_spec.axvline(fc, color="k", lw=0.8, ls=":")
                    self.ax_spec.text(0.02, 0.05,
                                      f"Omega0 = {omega0:.3g} {units}\n"
                                      f"fc = {fc:.4g} Hz, n = {n:.2f}",
                                      transform=self.ax_spec.transAxes, fontsize=9, va="bottom",
                                      bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8})
                    with np.errstate(invalid="ignore", divide="ignore"):
                        resid = np.log(ab[used]) - np.log(sp.brune_spectrum(fb[used], omega0, fc, n))
                    self.ax_resid.semilogx(fb[used], resid, "o-", color="#d62728", ms=3, lw=0.8)
                self.ax_resid.axhline(0.0, color="k", lw=0.8)
            self.ax_spec.legend(loc="upper right", fontsize=8)
            label = f"{rec.get('record') or ''} channel {rec.get('channel')}".strip()
            if rec.get("field"):
                label += f" ({rec['field']})"
            self.ax_spec.set_title(f"{self.figure_title()} {label}", fontsize=10)
            self.ax_resid.set_xlim(self.ax_spec.get_xlim())
        if not_computed:
            self.ax_time.set_title("windows as last computed: drag them, or fix the entries", fontsize=9)
            self.ax_spec.set_title(f"{self.figure_title()}: not recomputed - {not_computed}"[:160], fontsize=9)
        self.canvas.draw_idle()

    # ------------------------------------------------------ window dragging
    @staticmethod
    def window_edges_ms(w: sp.WindowResult) -> dict:
        """The four window edges in ms from the trigger."""
        pre_ms, post_ms = w.n_pre / w.fs * 1e3, w.n_post / w.fs * 1e3
        edges = {"signal_start": -pre_ms, "signal_end": post_ms}
        if w.noise_available:
            edges["noise_start"] = (w.i_noise0 - w.i_trigger) / w.fs * 1e3
            edges["noise_end"] = (w.i_noise1 - w.i_trigger) / w.fs * 1e3
        else:
            edges["noise_start"] = edges["noise_end"] = -pre_ms
        return edges

    def current_edges_ms(self) -> Optional[dict]:
        return self.window_edges_ms(self.window) if self.window is not None else None

    def _in_time_axes(self, event) -> bool:
        return event.inaxes in (self.ax_time, self.ax_taper) and event.xdata is not None

    def _trace_bounds_ms(self) -> tuple[float, float]:
        """The record's first and last sample, in ms from the trigger."""
        t0 = self.window.trigger_time
        return float((self.strain_time[0] - t0) * 1e3), float((self.strain_time[-1] - t0) * 1e3)

    def handle_at(self, event) -> Optional[str]:
        """The window edge within ``HANDLE_PICK_PX`` of the press, the nearest one.
        Where the noise window ends exactly at the signal window the two edges
        coincide: ``TIE`` is returned and the first motion decides which one
        moves (left: the noise end, right: the signal start); ``TIE_AFTER``
        likewise where a noise window starts exactly at the signal end (left:
        the signal end, right: the noise start)."""
        near = []
        for name, line in self.handles.items():
            px = self.ax_time.transData.transform((float(line.get_xdata()[0]), 0.0))[0]
            d = abs(px - float(event.x))
            if d <= HANDLE_PICK_PX:
                near.append((d, name))
        if not near:
            return None
        near.sort()
        nearest = {name for d, name in near if d <= near[0][0] + 0.5}
        if {"noise_end", "signal_start"} <= nearest:
            return TIE
        if {"signal_end", "noise_start"} <= nearest:
            return TIE_AFTER
        return near[0][1]

    def on_press(self, event) -> None:
        """Grab a window edge near the press, else a window by its body.  The
        edges are hit-tested here, in pixels: the taper's twin axes would get
        pick events first."""
        if self.toolbar_active() or self._drag is not None or self.window is None or not self._in_time_axes(event):
            return
        if event.button != 1:
            return
        x = float(self.ax_time.transData.inverted().transform((event.x, event.y))[0])
        name = self.handle_at(event)
        if name is not None:
            self.begin_drag(name, x)
            if name in (TIE, TIE_AFTER) and self._drag["auto_noise"]:   # the noise follows the signal anyway
                self._drag["what"] = "signal_start" if name == TIE else "signal_end"
            return
        edges = self.current_edges_ms()
        if edges["signal_start"] <= x <= edges["signal_end"]:
            self.begin_drag("signal", x)
        elif self.window.noise_available and edges["noise_start"] <= x <= edges["noise_end"]:
            self.begin_drag("noise", x)

    def begin_drag(self, what: str, x_ms: float) -> None:
        # blank noise entries: the noise window is automatic (the same length just before the
        # signal) and keeps following the signal window; dragging the noise itself makes it explicit
        auto = not (self.noise_start_var.get().strip() or self.noise_end_var.get().strip())
        edges = dict(self.current_edges_ms())
        self._drag = {"what": what, "x0": float(x_ms), "edges": edges, "moved": False,
                      "auto_noise": auto and what in ("signal_start", "signal_end", "signal", TIE, TIE_AFTER),
                      # the noise window lies after the signal: the limits and the following swap sides
                      "noise_after": bool(self.window.noise_available and edges["noise_start"] >= edges["signal_end"])}

    def on_motion(self, event) -> None:
        if self._drag is None or not self._in_time_axes(event):
            return
        self.drag_to(float(self.ax_time.transData.inverted().transform((event.x, event.y))[0]))

    def drag_to(self, x_ms: float) -> dict:
        """Move the grabbed edge, or window, to ``x_ms`` (ms from the trigger) and
        show the new values in the entries; the analysis waits for the release.
        An edge moves on its own and stops at the other window (whichever side
        of the signal the noise lies), at the trigger (the signal window keeps
        it inside) and at the record's ends; a body slides as one within the
        same limits."""
        d = self._drag
        e = dict(d["edges"])
        dx = x_ms - d["x0"]
        if d["what"] in (TIE, TIE_AFTER):                # coincident edges: the direction picks the one that can move
            if dx == 0.0:
                return e
            if d["what"] == TIE:
                d["what"] = "noise_end" if dx < 0 else "signal_start"
            else:
                d["what"] = "signal_end" if dx < 0 else "noise_start"
        what = d["what"]
        dt = 1e3 / self.window.fs
        lo_t, hi_t = self._trace_bounds_ms()
        noise = self.window.noise_available and not d["auto_noise"]
        after = d["noise_after"]
        floor = e["noise_end"] if noise and not after else lo_t      # the signal start stops here
        ceiling = e["noise_start"] if noise and after else hi_t      # the signal end stops here
        if what == "signal_start":
            e["signal_start"] = min(max(x_ms, lo_t, floor), 0.0, e["signal_end"] - dt)
        elif what == "signal_end":
            e["signal_end"] = max(min(x_ms, hi_t, ceiling), 0.0, e["signal_start"] + dt)
        elif what == "noise_start":
            e["noise_start"] = min(max(x_ms, e["signal_end"] if after else lo_t), e["noise_end"] - dt)
        elif what == "noise_end":
            e["noise_end"] = max(min(x_ms, hi_t if after else e["signal_start"]), e["noise_start"] + dt)
        elif what == "signal":
            length = e["signal_end"] - e["signal_start"]
            lo = max(lo_t, floor, -length)
            start = min(max(e["signal_start"] + dx, lo), 0.0, min(hi_t, ceiling) - length)
            e["signal_start"], e["signal_end"] = start, start + length
        elif what == "noise":
            length = e["noise_end"] - e["noise_start"]
            if after:
                start = max(min(e["noise_start"] + dx, hi_t - length), e["signal_end"])
                e["noise_start"], e["noise_end"] = start, start + length
            else:
                end = min(max(e["noise_end"] + dx, lo_t + length), e["signal_start"])
                e["noise_start"], e["noise_end"] = end - length, end
        if d["auto_noise"] and self.window.noise_available:
            length = e["noise_end"] - e["noise_start"]
            if after:
                e["noise_start"] = e["signal_end"]
                e["noise_end"] = e["noise_start"] + length
            else:
                e["noise_end"] = e["signal_start"]
                e["noise_start"] = e["noise_end"] - length
        d["moved"] = True
        self.show_edges(e, noise_entries=not d["auto_noise"])
        return e

    def show_edges(self, e: dict, noise_entries: bool = True) -> None:
        """Entries and the top panel follow the edges being dragged (no recompute
        yet); with ``noise_entries`` False the noise entries stay blank (automatic)."""
        self.pre_var.set(_text(round(abs(e["signal_start"]), 4)))     # never "-0"
        self.post_var.set(_text(round(e["signal_end"], 4) + 0.0))
        if noise_entries and self.window is not None and self.window.noise_available:
            # without a usable noise window the edges carry a stand-in (both at the signal start),
            # which must not become the entries' values
            self.noise_start_var.set(_text(round(e["noise_start"], 4) + 0.0))
            self.noise_end_var.set(_text(round(e["noise_end"], 4) + 0.0))
        for name, line in self.handles.items():
            line.set_xdata([e[name], e[name]])
        for patch, a, b in ((self.signal_span, "signal_start", "signal_end"), (self.noise_span, "noise_start", "noise_end")):
            if patch is not None:
                patch.set_x(e[a])
                patch.set_width(e[b] - e[a])
        self.canvas.draw_idle()

    def on_release(self, event) -> None:
        if self._drag is None:
            return
        self.end_drag()

    def end_drag(self) -> None:
        """The dragged windows become the settings: recompute (a click without
        a move changes nothing)."""
        d, self._drag = self._drag, None
        if d is not None and d.get("moved"):
            self.schedule_compute(now=True)

    # ------------------------------------------------------------- dialogs
    def browse_calibration(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="Calibration CSV",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.calibration_path_var.set(path)
            try:
                self.calibration_db_var.set(sp.csv_gain_is_db(path))      # the header says dB, or not
            except Exception:                                             # unreadable: compute() reports it
                pass
            self.schedule_compute(now=True)
