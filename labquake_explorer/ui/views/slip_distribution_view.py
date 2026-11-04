"""Coseismic slip of every sensor along the fault, each with its own event time.

Opened from an event that carries a slip channel array (``event['slip']``: one
row per eddy-current sensor, in micrometres, with sensor ``positions``).  Each
sensor ``i`` gets an event time ``t_i``, and its coseismic slip is the
difference at ``t_i`` of two lines fitted before and after it (see
:mod:`labquake_explorer.analysis.coseismic_slip`).  The event-time tab shown
sets every ``t_i``: the largest filtered slip rate, the slip-rate onset, the
crossing of a level of the slip step, or one shared time (the peak of the mean
slip rate, or a time typed or dragged).  On the first three a drag or a typed
``t_i`` sets that sensor by hand; on Shared it moves the shared time.

The trace panel shows the selected sensor: its slip, the two fit windows, the
lines extrapolated to ``t_i`` and, on the right axis, the filtered slip rate
used for the automatic pick.  ``t_i`` and the four window edges can be dragged
(or a window by its body); dragging ``t_i`` moves it alone, between the
windows.  The windows are times relative to the
event time, so they stay put when ``t_i`` moves.  The same values can be typed
in the entries.  With "same windows for all sensors" a window edit applies to
every sensor.
Everything recomputes as soon as a value changes.  The slip-rate filter is
the one of the tab shown (low-pass, Butterworth or Bessel of a chosen order and
cutoff; Savitzky-Golay of a chosen window and order; moving average).

The two panels on the right show the result along the fault: the coseismic
slip against ``x`` with error bars of each sensor's noise and a profile fit
(``quadratic`` with values below zero masked to zero, or the earthquake arrest
zone model of Ke et al. 2021, symmetric ``EAZ`` or ``EAZ-asymmetric``; see
:mod:`labquake_explorer.analysis.slip_profile`), and the event times ``t_i``
against ``x`` with the apparent velocity of the rupture front.  Each profile
model has a tab with its parameters: a value typed there holds the parameter,
the 'fit' tick lets the fit choose it again, and min/max replace the automatic
bounds; the tab shown is the model fitted.  A left click
on a point selects its sensor; a zoom of these two panels stays until another
event or field is loaded.  The bare PMMA patch is drawn on both panels; its
edges are typed in and saved for the whole experiment
(``analysis['fault']['patch_x_mm']``), where preprocessing can set them too.
Edges typed and not saved there stay while the view moves between events.
The fault ends (``analysis['fault']['ends_x_mm']``) are typed and saved the same
way: the profile is cut there, and where its tip lies beyond an end the text
gives the slip at that end instead of a rupture tip.

The sensors do not always pin every profile parameter (``ProfileFit.unresolved``):
an unresolved parameter is listed under "not resolved by the sensors" with the
range that fits equally well (``ProfileFit.ranges``) when it is known, and a
range of a parameter that is resolved but moves with an unresolved one (EAZ
``D`` following ``lambda``) is shown beside its value.  A rupture edge that is
not pinned is given, and drawn on the zero line, as the bracket it lies in
(``beyond`` the last sensor when it is open); markers mark only the edges
that are pinned.

The front is one line (``single``) or, by default, a hinge of two lines that
meet at a fitted apex (``bilateral``: at least four sensors; with fewer, two
lines through the earliest sensor; one line when both sides run the same way).
The text gives the apex and its time, where the fronts start (diverging) or
meet (converging), and the panel marks it.  The fit gets the event's sampling
interval: a segment that the front crosses within less than one sample is not
resolved and reads "faster than" the speed of crossing it in one sample (``>``
on the panel); a flat one (zero slope, infinite velocity) reads
"simultaneous".

Save stores ``event['slip_analysis']``: the record of
:meth:`SlipAnalysis.as_dict` plus ``field``, ``unit``, ``same_windows``,
``rate_filter_tabs``, ``patch_x_mm`` (the edges drawn), ``fault_ends_x_mm``,
``profile`` (the fit record plus ``use``, the sensors chosen for it,
``settings``, every tab's holds and bounds, and ``end_slip_um``) and ``front`` (the front fit plus ``use`` and
``use_auto``: while the front's sensor choice has not been edited it follows
the pick flags, leaving out the sensors whose ``t_i`` is doubtful: pick at
the search edge, weak or no rate peak).
"""
from __future__ import annotations

import functools
import math
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np
from matplotlib.backend_bases import MouseButton

from labquake_explorer.analysis import coseismic_slip as cs
from labquake_explorer.analysis import slip_profile as sprof
from labquake_explorer.data.channels import channel_names, is_channel_array
from labquake_explorer.data.picks import extraction_record
from labquake_explorer.data.sources import waveform_blocks, waveform_channels
from labquake_explorer.data.settings import get_setting, set_setting
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT
from labquake_explorer.ui.views.base import EventView

LIVE_DELAY_MS = 400                    # recompute this long after the last keystroke
HANDLE_PICK_PX = 6                     # a press this close (pixels) to t_i or a window edge grabs it
ZOOM_HALF_S = 0.2                      # "Zoom to t_i" shows t_i +- this
METHOD_LABELS = {cs.METHOD_MAX_RATE: "max slip rate", cs.METHOD_ONSET: "slip-rate onset", cs.METHOD_STEP: "slip step",
                 cs.METHOD_SHARED: "shared", cs.METHOD_BURST: "burst", cs.METHOD_MANUAL: "manual"}
T_I_TABS = cs.PICK_METHODS + (cs.METHOD_SHARED, cs.METHOD_BURST)   # the event-time tabs
T_I_TAB_LABELS = {cs.METHOD_MAX_RATE: "Max slip rate", cs.METHOD_ONSET: "Rate onset", cs.METHOD_STEP: "Slip step",
                  cs.METHOD_SHARED: "Shared", cs.METHOD_BURST: "Burst"}
WAVEFORM_FIELD_PREFIX = "waveform/"    # a field made of a full-rate record's eddy channels (2 MHz)
ZERO_FRACTION = 0.01                   # a waveform field's slip is zeroed on this first share of its samples
RATE_FILTER_LABELS = {"lowpass": "Low-pass", "savgol": "Savitzky-Golay", "boxcar": "Moving average"}   # the tabs
LOWPASS_LABELS = {"butterworth": "Butterworth", "bessel": "Bessel"}
DEFAULT_SAVGOL_WINDOW_S = 0.020        # the Savitzky-Golay tab's window (the moving average's: cs default)
BEFORE_COLOR, AFTER_COLOR, TI_COLOR = "#1f77b4", "#ff7f0e", "#d62728"
PROFILE_LABELS = dict(sprof.MODEL_LABELS)      # model key -> combobox text
DEFAULT_PROFILE_MODEL = "eaz"
DEFAULT_FRONT_MODE = "bilateral"
PATCH_KEY = "patch_x_mm"                       # analysis.fault.patch_x_mm, and in a slip_analysis record
ENDS_KEY = "ends_x_mm"                         # analysis.fault.ends_x_mm: where the fault ends
RECORD_ENDS_KEY = "fault_ends_x_mm"            # the ends a slip_analysis record was drawn with
FAULT_SETTING = "fault"                        # analysis['fault']
# earlier files: data['patch_x_mm'] / data['fault_ends_x_mm'], or data['fault'] = {'patch_x_mm', 'ends_x_mm'}
LEGACY_FAULT_KEY = "fault"
END_COLOR = "#606060"
T_I_FLAGS = (cs.FLAG_EDGE, cs.FLAG_WEAK, cs.FLAG_NO_PEAK)   # the pick flags that make t_i doubtful
PATCH_COLOR, POINT_COLOR, FRONT_COLOR, FLAG_COLOR = "#c08000", "#1f77b4", "#2ca02c", "#9467bd"
EDGE_COLOR = "#404040"                 # rupture edges: pinned ones as markers, free ones as their bracket
TABLE_COLUMNS = (("sensor", 70), ("x (mm)", 60), ("t_i from", 90), ("t_i - event (ms)", 100),
                 ("slip (um)", 75), ("sigma (um)", 75), ("flags", 260))
HANDLE_NAMES = ("t_i", "before_start", "before_end", "after_start", "after_end")
TEXT_WRAP_PX = 250                     # the fit texts wrap at this width
SIDES = (("minus", "-x"), ("plus", "+x"))   # the rupture edges, in the order of ProfileFit.tips_mm
NOT_RESOLVED = "not resolved by the sensors:"
# profile parameters as shown: label, unit ("u" is the slip unit), significant digits;
# the rows they are grouped in (D and xc read "D = ... at xc = ...")
PARAM_FORMAT = {"c0": ("c0", "{u}", 4), "c1": ("c1", "{u}/mm", 4), "c2": ("c2", "{u}/mm^2", 4),
                "D": ("D", "{u}", 3), "xc": ("xc", "mm", 4), "a": ("a", "mm", 4), "lambda": ("lambda", "", 3),
                "a_minus": ("a-", "mm", 4), "a_plus": ("a+", "mm", 4),
                "lambda_minus": ("lambda-", "", 3), "lambda_plus": ("lambda+", "", 3)}
PARAM_ROWS = {"quadratic": (("c0", "c1"), ("c2",)), "eaz": (("D", "xc"), ("a", "lambda")),
              "eaz_asymmetric": (("D", "xc"), ("a_minus", "a_plus"), ("lambda_minus", "lambda_plus"))}


def _text(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def _parse(text: str, name: str, scale: float = 1.0) -> float:
    try:
        value = float(str(text).strip())
    except ValueError:
        raise ValueError(f"{name}: enter a number") from None
    if not math.isfinite(value):
        raise ValueError(f"{name}: enter a finite number")
    return value * scale


def _bools(value, n: int, default: bool) -> list:
    """``value`` as ``n`` Python bools, or ``n`` times ``default`` when it does not fit."""
    try:
        flags = [bool(v) for v in np.asarray(value).ravel()]
    except (TypeError, ValueError):
        return [default] * n
    return flags if len(flags) == n and value is not None else [default] * n


def _patch_pair(value) -> Optional[tuple]:
    """Two finite, different edges (in either order) as an increasing tuple, else None."""
    try:
        lo, hi = sorted(float(v) for v in np.asarray(value, dtype=float).ravel())
    except (TypeError, ValueError):
        return None
    return (lo, hi) if math.isfinite(lo) and math.isfinite(hi) and lo < hi else None


def _value_text(value) -> str:
    """A stored value as short text, for a message."""
    try:
        return str(np.asarray(value, dtype=float).ravel().tolist())
    except (TypeError, ValueError):
        return repr(value)


def doubtful_t_i(sensor: cs.SensorSlip) -> bool:
    """True when a pick flag makes the sensor's ``t_i`` doubtful (the slip
    window flags do not)."""
    return any(flag in T_I_FLAGS for flag in sensor.flags)


def tip_brackets(fit: Optional[sprof.ProfileFit]) -> dict:
    """The rupture edges of a valid profile fit that the sensors leave free:
    ``"minus"``/``"plus"`` -> ``(lo, hi)``, the bracket the edge lies in (NaN: open
    on that end).  Empty while ``tips`` is resolved; when it is not, the sides
    without a range are the pinned ones (both brackets open if no side has one)."""
    if fit is None or not fit.valid or "tips" not in fit.unresolved:
        return {}
    out = {}
    for side, _ in SIDES:
        value = fit.ranges.get("tip_" + side)
        try:
            lo, hi = (float(v) for v in value)
        except (TypeError, ValueError):
            continue
        out[side] = (lo, hi)
    return out or {side: (math.nan, math.nan) for side, _ in SIDES}


def _range_text(lo: float, hi: float, digits: int, unit: str = "") -> str:
    """``lo to hi unit``; an open (NaN) end reads ``or more`` / ``or less``."""
    unit = f" {unit}" if unit else ""
    if math.isfinite(lo) and math.isfinite(hi):
        return f"{lo:.{digits}g} to {hi:.{digits}g}{unit}"
    if math.isfinite(lo):
        return f"{lo:.{digits}g}{unit} or more"
    if math.isfinite(hi):
        return f"{hi:.{digits}g}{unit} or less"
    return "any value"


def _edge_text(label: str, lo: float, hi: float) -> str:
    """A free rupture edge as its bracket: ``-x edge 217 to 279 mm``, ``+x edge
    beyond 465 mm`` (open above), ``-x edge below 31 mm`` (open below)."""
    if math.isfinite(lo) and math.isfinite(hi):
        return f"{label} edge {lo:.4g} to {hi:.4g} mm"
    if math.isfinite(lo):
        return f"{label} edge beyond {lo:.4g} mm"
    if math.isfinite(hi):
        return f"{label} edge below {hi:.4g} mm"
    return f"{label} edge"


def segment_speed_label(seg: sprof.FrontSegment) -> str:
    """The front panel's label of a segment: its speed, ``> N m/s`` when the front
    crosses it within one sample, ``simultaneous`` when it is flat."""
    v = seg.velocity_m_s
    if math.isinf(v):
        return "simultaneous"
    if not seg.resolved and math.isfinite(seg.min_speed_m_s):
        return f"> {seg.min_speed_m_s:.3g} m/s"
    return f"{abs(v):.3g} m/s"


def segment_text(seg: sprof.FrontSegment) -> str:
    """One line of the front text: where, the speed and direction, the sensors and R^2."""
    where = f"x {seg.x_from:.4g} to {seg.x_to:.4g} mm"
    n = f"{seg.n} sensor{'' if seg.n == 1 else 's'}"
    v, bound = seg.velocity_m_s, seg.min_speed_m_s
    if math.isinf(v):
        faster = f", faster than {bound:.3g} m/s" if math.isfinite(bound) else ""
        return f"{where}: simultaneous{faster} ({n})"
    if not seg.resolved and math.isfinite(bound):
        return f"{where}: faster than {bound:.3g} m/s (less than one sample across it; {n})"
    r2 = f", R^2 = {seg.r2:.3f}" if math.isfinite(seg.r2) else ""
    return f"{where}: {abs(v):.3g} m/s towards {'+x' if v > 0 else '-x'} ({n}{r2})"


def apex_text(fit: sprof.FrontFit) -> Optional[str]:
    """Where a bilateral front starts (two lines leaving the apex), or where its two
    fronts meet (two lines running into it), with the time there; for a front that
    runs one way, its earliest point.  None without an apex (NaN, single mode) or
    when no side moves."""
    if fit.mode != "bilateral" or not fit.segments:
        return None
    if not (math.isfinite(fit.split_x_mm) and math.isfinite(fit.t0_s)):
        return None
    t = 1e3 * fit.t0_s
    at = f"x = {fit.split_x_mm:.4g} mm, {abs(t):.2f} ms {'after' if t >= 0 else 'before'} the event time"
    left, right = fit.segments[0].slope_s_per_mm, fit.segments[-1].slope_s_per_mm
    if len(fit.segments) == 1:
        return f"one-way front, earliest at {at}" if left != 0 else None
    if left == right == 0:
        return None
    if left <= 0 <= right:
        return f"fronts start at {at}"
    if left >= 0 >= right:
        return f"fronts meet at {at}"
    return f"the two lines meet at {at}"


def _float_or_nan(value) -> float:
    """A saved number (also a 1-element array), or NaN."""
    try:
        return float(np.asarray(value, dtype=float).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return float("nan")


def _text_float(text: str) -> float:
    """The number typed in an entry, or NaN when it is blank or not a number."""
    try:
        return float(str(text).strip())
    except ValueError:
        return float("nan")


def _parse_blank(text: str, name: str) -> float:
    """A number, or NaN for a blank entry."""
    return float("nan") if not str(text).strip() else _parse(text, name)


def _parse_int(text: str, name: str) -> int:
    value = _parse(text, name)
    if value != int(value):
        raise ValueError(f"{name}: enter a whole number")
    return int(value)


def slip_fields(event, run=None) -> list:
    """Channel arrays of an event aligned with its time axis and in micrometres
    (``slip`` first), then every full-rate record (``waveform/<recorder>``)
    that holds the voltage channels the run's ``slip`` was made from: the
    fields this view can analyse."""
    if not isinstance(event, dict) or "time" not in event:
        return []
    n = len(np.asarray(event["time"]).ravel())
    out = []
    for key, value in event.items():
        if is_channel_array(value) and value["data"].shape[1] == n and str(value.get("unit", "")) == "um":
            out.append(str(key))
    out = sorted(out, key=lambda k: (k != "slip", k))
    sources = slip_sources(run)
    for key, block in waveform_blocks(event).items():
        if sources and any(c in sources for c in waveform_channels(block)):
            out.append(str(key))
    return out


def slip_sources(run) -> dict:
    """``{voltage channel: (slip row, name, x_mm, um per V)}`` of the run's
    ``slip`` array (its ``source`` channels and ``slope_mm_per_v``); empty
    when the run has none."""
    slip = run.get("slip") if isinstance(run, dict) else None
    if not is_channel_array(slip):
        return {}
    names = channel_names(slip)
    sources = slip.get("source")
    if sources is None:
        return {}
    sources = [s.decode() if isinstance(s, bytes) else str(s) for s in np.asarray(sources).ravel()]
    slope = np.asarray(slip.get("slope_mm_per_v", np.nan), dtype=float).ravel()
    if slope.size not in (1, len(names)):
        return {}
    try:
        x = np.asarray(slip["positions"]["x"], dtype=float).ravel()
    except (KeyError, TypeError, ValueError):
        x = np.full(len(names), np.nan)
    if x.size != len(names):
        x = np.full(len(names), np.nan)
    out = {}
    for j, source in enumerate(sources[:len(names)]):
        um_per_v = float(slope[j] if slope.size > 1 else slope[0]) * 1e3
        if math.isfinite(um_per_v) and um_per_v != 0.0:
            out[source] = (j, names[j], float(x[j]), um_per_v)
    return out


def waveform_slip_array(event, run, key: str) -> dict:
    """The slip channel array made from a full-rate record: its eddy channels
    (those the run's ``slip`` was made from) in micrometres on the record's own
    time axis, each zeroed on its first samples, with the sensors' positions,
    plus ``pzt``, the record's PZT traces (volts) for the burst pick."""
    block = waveform_blocks(event).get(key)
    if block is None:
        raise ValueError(f"{key}: no such full-rate record")
    sources = slip_sources(run)
    names = waveform_channels(block)
    data = np.asarray(block["data"], dtype=float)
    time = np.asarray(block["time"], dtype=float).ravel()
    if data.ndim != 2 or data.shape[1] != time.size:
        raise ValueError(f"{key}: the record's data and time do not match")
    rows = sorted((sources[c][0], k) for k, c in enumerate(names) if c in sources)
    if not rows:
        raise ValueError(f"{key}: none of the slip array's source channels ({', '.join(sources) or 'none'}) is in this record")
    head = max(1, int(ZERO_FRACTION * time.size))
    slip, channels, x = [], [], []
    for _, k in rows:
        j, name, x_mm, um_per_v = sources[names[k]]
        trace = data[k] * um_per_v
        slip.append(trace - float(np.nanmean(trace[:head])))
        channels.append(name)
        x.append(x_mm)
    pzt = [data[k] for k, c in enumerate(names) if c.lower().startswith("pzt")]
    return {"data": np.array(slip), "channels": channels, "unit": "um", "positions": {"x": x},
            "time": time, "pzt": np.array(pzt) if pzt else None,
            "sample_rate": float(block.get("sample_rate", 1.0 / cs.sampling_interval(time)))}


@register_view("Slip Distribution", kinds=[EVENT], order=15)
class SlipDistributionView(EventView):
    window_title = "Slip Distribution"
    result_key = "slip_analysis"

    def __init__(self, app, run_idx, event_idx):
        self.field: Optional[str] = None
        self.channels: list = []
        self.x_mm = np.array([])
        self.unit = "um"
        self.data: Optional[np.ndarray] = None
        self.time: Optional[np.ndarray] = None
        self.pzt: Optional[np.ndarray] = None     # the PZT traces of a waveform field (the burst pick)
        self.t_event = 0.0
        self.start_s = -1.0
        self.end_s = 1.0
        self.methods: list = []
        self.t_shift: list = []
        self.fit_before: list = []
        self.fit_after: list = []
        self.sensor = 0
        self.result: Optional[cs.SlipAnalysis] = None
        self.saved: Optional[dict] = None
        self.handles: dict = {}
        self.spans: dict = {}
        self._drag = None
        self._compute_after_id = None
        self._pending = None
        self._syncing = False
        self._shown: dict = {}         # entry variable name -> the text the view last put there
        self._stale = False            # the last recompute failed: the result lags the inputs
        self.profile_fit: Optional[sprof.ProfileFit] = None
        self.front_fit: Optional[sprof.FrontFit] = None
        self.profile_use: list = []
        self.front_use: list = []
        self.front_use_auto = True
        self.patch_x_mm: Optional[tuple] = None
        self._patch_draft = False      # patch_x_mm was typed here and not saved to the experiment
        self.fault_ends: Optional[tuple] = None
        self._ends_draft = False       # fault_ends was typed here and not saved to the experiment
        self.point_artists: dict = {}
        self.patch_artists: list = []
        self.profile_errorbar = None
        self.profile_curve = None
        self._profile_xlim = (0.0, 1.0)
        self._fault_auto = None        # the fault panels' limits as the last redraw set them
        super().__init__(app, run_idx, event_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self) -> None:
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        left = ttk.Frame(self)
        left.grid(row=0, column=0, rowspan=3, padx=5, pady=5, sticky="nsw")
        fits = ttk.Frame(self)
        fits.grid(row=0, column=1, rowspan=3, padx=5, pady=5, sticky="nsw")

        frame = ttk.LabelFrame(left, text="Event")
        frame.pack(fill="x", pady=(0, 4))
        self.build_event_selector(frame, row=0, column=0)
        ttk.Label(frame, text="Field:").grid(row=1, column=0, padx=4, pady=2, sticky="e")
        self.field_combobox = ttk.Combobox(frame, width=10, state="readonly")
        self.field_combobox.grid(row=1, column=1, padx=4, pady=2, sticky="w")
        self.field_combobox.bind("<<ComboboxSelected>>", self.on_field_selected)

        frame = ttk.LabelFrame(left, text="Sensor")
        frame.pack(fill="x", pady=4)
        row = ttk.Frame(frame)
        row.grid(row=0, column=0, columnspan=3, padx=2, pady=2, sticky="w")
        self.sensor_combobox = ttk.Combobox(row, width=18, state="readonly")
        self.sensor_combobox.pack(side="left", padx=2)
        self.sensor_combobox.bind("<<ComboboxSelected>>", self.on_sensor_selected)
        self.prev_button = ttk.Button(row, text="◀", width=2, command=lambda: self.step_sensor(-1))
        self.prev_button.pack(side="left")
        self.next_button = ttk.Button(row, text="▶", width=2, command=lambda: self.step_sensor(+1))
        self.next_button.pack(side="left")
        ttk.Label(frame, text="t_i from:").grid(row=1, column=0, padx=4, pady=2, sticky="e")
        self.method_combobox = ttk.Combobox(frame, width=13, state="readonly", values=list(METHOD_LABELS.values()))
        self.method_combobox.grid(row=1, column=1, columnspan=2, padx=4, pady=2, sticky="w")
        self.method_combobox.bind("<<ComboboxSelected>>", self.on_method_selected)
        self.t_shift_var = tk.StringVar(master=self)
        self.entries = {}
        self._entry(frame, "t_i - event time (ms):", self.t_shift_var, 2, self.on_t_shift_edited)
        self.before_start_var = tk.StringVar(master=self)
        self.before_end_var = tk.StringVar(master=self)
        self.after_start_var = tk.StringVar(master=self)
        self.after_end_var = tk.StringVar(master=self)
        ttk.Label(frame, text="Fit before (s):").grid(row=3, column=0, padx=4, pady=2, sticky="e")
        self._entry_only(frame, self.before_start_var, 3, 1, "before start", self.on_window_edited)
        self._entry_only(frame, self.before_end_var, 3, 2, "before end", self.on_window_edited)
        ttk.Label(frame, text="Fit after (s):").grid(row=4, column=0, padx=4, pady=2, sticky="e")
        self._entry_only(frame, self.after_start_var, 4, 1, "after start", self.on_window_edited)
        self._entry_only(frame, self.after_end_var, 4, 2, "after end", self.on_window_edited)
        ttk.Label(frame, text="windows are relative to the event time", foreground="#606060").grid(
            row=5, column=0, columnspan=3, padx=4, sticky="w")
        self.same_windows_var = tk.BooleanVar(master=self, value=True)
        ttk.Checkbutton(frame, text="same windows for all sensors", variable=self.same_windows_var,
                        command=self.on_same_windows_toggled).grid(row=6, column=0, columnspan=3, padx=4, pady=2, sticky="w")
        buttons = ttk.Frame(frame)
        buttons.grid(row=7, column=0, columnspan=3, padx=2, pady=2, sticky="w")
        ttk.Button(buttons, text="Zoom to t_i", command=self.zoom_to_event_time).pack(side="left", padx=2)
        ttk.Button(buttons, text="Whole window", command=self.show_whole_window).pack(side="left", padx=2)

        frame = ttk.LabelFrame(left, text="Slip rate")
        frame.pack(fill="x", pady=4)
        self.reference_var = tk.StringVar(master=self,
                                          value=_text(cs.DEFAULT_REFERENCE_HALFWIDTH_S * 1e3))
        self.smoothing_var = tk.StringVar(master=self, value=_text(cs.DEFAULT_RATE_SMOOTHING_S * 1e3))
        self.savgol_window_var = tk.StringVar(master=self, value=_text(DEFAULT_SAVGOL_WINDOW_S * 1e3))
        self.savgol_order_var = tk.StringVar(master=self, value=str(cs.DEFAULT_SAVGOL_ORDER))
        self.cutoff_var = tk.StringVar(master=self, value=_text(cs.DEFAULT_RATE_CUTOFF_HZ))
        self.lowpass_order_var = tk.StringVar(master=self, value=str(cs.DEFAULT_LOWPASS_ORDER))
        self.search_var = tk.StringVar(master=self, value=_text(cs.DEFAULT_SEARCH_HALFWIDTH_S * 1e3))
        self.snr_var = tk.StringVar(master=self, value=_text(cs.DEFAULT_RATE_SNR))
        self.onset_var = tk.StringVar(master=self, value=_text(cs.DEFAULT_ONSET_FRACTION * 100))
        self.step_var = tk.StringVar(master=self, value=_text(cs.DEFAULT_STEP_FRACTION * 100))
        self.shared_shift_var = tk.StringVar(master=self, value="")
        self.shared_source_var = tk.StringVar(master=self, value="reference")
        self.burst_source_var = tk.StringVar(master=self, value=cs.DEFAULT_BURST_SOURCE)
        self.burst_halfwidth_var = tk.StringVar(master=self, value=_text(cs.DEFAULT_BURST_HALFWIDTH_S * 1e3))
        self.level_width_var = tk.StringVar(master=self, value=_text(cs.DEFAULT_LEVEL_WIDTH_S * 1e3))
        self.burst_search_var = tk.StringVar(master=self, value=_text(cs.DEFAULT_BURST_SEARCH_S * 1e3))
        for var in (self.onset_var, self.step_var, self.shared_shift_var):
            self._shown[str(var)] = var.get()
        for var in self.settings_vars():
            self._shown[str(var)] = var.get()
        ttk.Label(frame, text="filter (the tab shown is used):", foreground="#606060").grid(
            row=0, column=0, columnspan=2, padx=4, sticky="w")
        self.rate_filter_notebook = ttk.Notebook(frame)
        self.rate_filter_notebook.grid(row=1, column=0, columnspan=2, padx=4, pady=2, sticky="ew")
        tabs = {key: ttk.Frame(self.rate_filter_notebook) for key in cs.RATE_FILTERS}
        tab = tabs["lowpass"]
        ttk.Label(tab, text="Type:").grid(row=0, column=0, padx=4, pady=2, sticky="e")
        self.lowpass_type_combobox = ttk.Combobox(tab, width=11, state="readonly",
                                                  values=[LOWPASS_LABELS[t] for t in cs.LOWPASS_TYPES])
        self.lowpass_type_combobox.set(LOWPASS_LABELS[cs.DEFAULT_LOWPASS_TYPE])
        self.lowpass_type_combobox.grid(row=0, column=1, padx=4, pady=2, sticky="w")
        self.lowpass_type_combobox.bind("<<ComboboxSelected>>", lambda e: self.on_filter_choice())
        self.cutoff_entry = self._entry(tab, "Cutoff (Hz):", self.cutoff_var, 1, self.on_settings_edited)
        self._entry(tab, "Order:", self.lowpass_order_var, 2, self.on_settings_edited, name="low-pass order")
        ttk.Label(tab, text="on the slip, forwards and backwards:\nzero phase, -6 dB at the cutoff",
                  foreground="#606060", justify="left").grid(row=3, column=0, columnspan=2, padx=4, sticky="w")
        tab = tabs["savgol"]
        self._entry(tab, "Window (ms):", self.savgol_window_var, 0, self.on_settings_edited, name="savgol window")
        self._entry(tab, "Polynomial order:", self.savgol_order_var, 1, self.on_settings_edited, name="savgol order")
        ttk.Label(tab, text="derivative of the local polynomial", foreground="#606060").grid(
            row=2, column=0, columnspan=2, padx=4, sticky="w")
        tab = tabs["boxcar"]
        self.smoothing_entry = self._entry(tab, "Window (ms):", self.smoothing_var, 0, self.on_settings_edited,
                                           name="moving average window")
        ttk.Label(tab, text="the derivative averaged over the window", foreground="#606060").grid(
            row=1, column=0, columnspan=2, padx=4, sticky="w")
        for key in cs.RATE_FILTERS:
            self.rate_filter_notebook.add(tabs[key], text=RATE_FILTER_LABELS[key])
        self.rate_filter_notebook.select(cs.RATE_FILTERS.index(cs.DEFAULT_RATE_FILTER))
        self.rate_filter_notebook.bind("<<NotebookTabChanged>>", lambda e: self.on_filter_choice())
        frame = ttk.LabelFrame(left, text="Event times t_i")
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text="the tab shown sets every sensor's t_i;\na drag or a typed t_i overrides one sensor",
                  foreground="#606060", justify="left").grid(row=0, column=0, columnspan=2, padx=4, sticky="w")
        self.t_i_notebook = ttk.Notebook(frame)
        self.t_i_notebook.grid(row=1, column=0, columnspan=2, padx=4, pady=2, sticky="ew")
        tabs = {key: ttk.Frame(self.t_i_notebook) for key in T_I_TABS}
        ttk.Label(tabs[cs.METHOD_MAX_RATE], text="each sensor at its largest filtered\nslip rate",
                  foreground="#606060", justify="left").grid(row=0, column=0, columnspan=2, padx=4, sticky="w")
        tab = tabs[cs.METHOD_ONSET]
        self._entry(tab, "Threshold (% of peak):", self.onset_var, 0, self.on_settings_edited, name="onset")
        ttk.Label(tab, text="where the rate, followed back from its\npeak, last rose through the threshold",
                  foreground="#606060", justify="left").grid(row=1, column=0, columnspan=2, padx=4, sticky="w")
        tab = tabs[cs.METHOD_STEP]
        self._entry(tab, "Level (% of the step):", self.step_var, 0, self.on_settings_edited, name="step")
        ttk.Label(tab, text="where the slip crosses that share of the\nway from the before to the after line",
                  foreground="#606060", justify="left").grid(row=1, column=0, columnspan=2, padx=4, sticky="w")
        tab = tabs[cs.METHOD_SHARED]
        ttk.Label(tab, text="one t_i for every sensor:", foreground="#606060").grid(
            row=0, column=0, columnspan=3, padx=4, sticky="w")
        ttk.Radiobutton(tab, text="peak of the mean slip rate", value="reference", variable=self.shared_source_var,
                        command=self.on_shared_source).grid(row=1, column=0, columnspan=3, padx=4, sticky="w")
        ttk.Radiobutton(tab, text="set (ms after the event):", value="set", variable=self.shared_source_var,
                        command=self.on_shared_source).grid(row=2, column=0, columnspan=2, padx=4, sticky="w")
        self._entry_only(tab, self.shared_shift_var, 2, 2, "shared t_i", self.on_shared_shift_edited, width=8)
        ttk.Label(tab, text="dragging the red line moves it", foreground="#606060").grid(
            row=3, column=0, columnspan=3, padx=4, sticky="w")
        tab = tabs[cs.METHOD_BURST]
        ttk.Label(tab, text="one t_i for every sensor at the AE burst; slip =\nlevel after - level before "
                            "(no line extrapolation):", foreground="#606060", justify="left").grid(
            row=0, column=0, columnspan=2, padx=4, sticky="w")
        ttk.Radiobutton(tab, text="peak of the PZT envelope (a waveform field)", value="pzt_envelope",
                        variable=self.burst_source_var, command=self.on_burst_source).grid(
            row=1, column=0, columnspan=2, padx=4, sticky="w")
        ttk.Radiobutton(tab, text="peak of the mean slip rate", value="peak_rate",
                        variable=self.burst_source_var, command=self.on_burst_source).grid(
            row=2, column=0, columnspan=2, padx=4, sticky="w")
        self._entry(tab, "Half-width w (ms):", self.burst_halfwidth_var, 3, self.on_settings_edited, name="burst w")
        self._entry(tab, "Level width (ms):", self.level_width_var, 4, self.on_settings_edited, name="level width")
        self._entry(tab, "Search +- (ms):", self.burst_search_var, 5, self.on_settings_edited, name="burst search")
        ttk.Label(tab, text="windows [t_i-w-d, t_i-w] and [t_i+w, t_i+w+d];\nchoose the waveform/<recorder> field "
                            "for 2 MHz", foreground="#606060", justify="left").grid(
            row=6, column=0, columnspan=2, padx=4, sticky="w")
        for key in T_I_TABS:
            self.t_i_notebook.add(tabs[key], text=T_I_TAB_LABELS[key])
        self.t_i_notebook.bind("<<NotebookTabChanged>>", lambda e: self.on_t_i_tab_changed())
        self._entry(frame, "Reference +- (ms):", self.reference_var, 2, self.on_settings_edited)
        self._entry(frame, "Search +- (ms):", self.search_var, 3, self.on_settings_edited)
        self._entry(frame, "Flag below (x rate noise):", self.snr_var, 4, self.on_settings_edited)
        ttk.Label(frame, text="reference: peak of the mean slip rate of\nall sensors near the event time; each\n"
                              "sensor's peak is searched around it.\nflag below: a weaker peak is flagged",
                  foreground="#606060", justify="left").grid(row=5, column=0, columnspan=2, padx=4, sticky="w")
        ttk.Button(frame, text="Re-pick all sensors", command=self.repick_all).grid(
            row=6, column=0, columnspan=2, padx=4, pady=2, sticky="w")

        frame = ttk.Frame(left)
        frame.pack(fill="x", pady=4)
        self.save_button = ttk.Button(frame, text="Save", command=self.save)
        self.save_button.grid(row=0, column=0, padx=4, pady=2, sticky="w")
        ttk.Label(frame, text="recomputes as the values change", foreground="#606060").grid(
            row=0, column=1, padx=6, sticky="w")
        self.status_var = tk.StringVar(master=self, value="")
        ttk.Label(left, textvariable=self.status_var, wraplength=280, foreground="#a03030", justify="left").pack(
            fill="x", pady=2)

        self._build_fit_frames(fits)
        self.make_figure(figsize=(8.5, 6), figure_kwargs={"layout": "constrained"},
                         row=0, column=2, padx=5, pady=5, sticky="nsew")
        gs = self.figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
        self.ax_trace = self.figure.add_subplot(gs[:, 0])
        self.ax_rate = self.ax_trace.twinx()
        self.ax_profile = self.figure.add_subplot(gs[0, 1])
        self.ax_front = self.figure.add_subplot(gs[1, 1], sharex=self.ax_profile)
        self.canvas.mpl_connect("pick_event", self.on_pick)                   # points of the fit panels
        toolbar = getattr(self, "toolbar", None)
        if toolbar is not None:                  # the Home button and the 'h' key go through home()
            self._toolbar_home = toolbar.home
            toolbar.home = self.home
            button = getattr(toolbar, "_buttons", {}).get("Home")
            if button is not None:
                button.config(command=self.home)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        self.table = ttk.Treeview(self, columns=[c for c, _ in TABLE_COLUMNS], show="headings", height=8,
                                  selectmode="browse")
        for name, width in TABLE_COLUMNS:
            self.table.heading(name, text=name)
            self.table.column(name, width=width, anchor="e" if name not in ("sensor", "t_i from", "flags") else "w")
        self.table.grid(row=2, column=2, padx=5, pady=(0, 5), sticky="ew")
        self.table.bind("<ButtonPress-1>", lambda e: self.flush_pending(), add="+")    # before the class binding
        self.table.bind("<<TreeviewSelect>>", self.on_table_select)

    def _build_fit_frames(self, master) -> None:
        frame = ttk.LabelFrame(master, text="Fault (experiment)")
        frame.pack(fill="x", pady=(0, 4))
        self.patch_lo_var = tk.StringVar(master=self, value="")
        self.patch_hi_var = tk.StringVar(master=self, value="")
        self.ends_lo_var = tk.StringVar(master=self, value="")
        self.ends_hi_var = tk.StringVar(master=self, value="")
        for column, text in ((1, "from x (mm)"), (2, "to x (mm)")):
            ttk.Label(frame, text=text, foreground="#606060").grid(row=0, column=column, padx=4, sticky="w")
        ttk.Label(frame, text="Fault ends:").grid(row=1, column=0, padx=4, pady=2, sticky="e")
        self._entry_only(frame, self.ends_lo_var, 1, 1, "ends from", self.on_ends_edited, width=7)
        self._entry_only(frame, self.ends_hi_var, 1, 2, "ends to", self.on_ends_edited, width=7)
        ttk.Label(frame, text="Bare PMMA patch:").grid(row=2, column=0, padx=4, pady=2, sticky="e")
        self._entry_only(frame, self.patch_lo_var, 2, 1, "patch from", self.on_patch_edited, width=7)
        self._entry_only(frame, self.patch_hi_var, 2, 2, "patch to", self.on_patch_edited, width=7)
        ttk.Button(frame, text="Save to experiment", command=self.save_fault).grid(
            row=3, column=0, columnspan=3, padx=4, pady=2, sticky="w")
        self.ends_note_var = tk.StringVar(master=self, value="")
        ttk.Label(frame, textvariable=self.ends_note_var, foreground="#606060", wraplength=TEXT_WRAP_PX,
                  justify="left").grid(row=4, column=0, columnspan=3, padx=4, sticky="w")
        self.patch_note_var = tk.StringVar(master=self, value="")
        ttk.Label(frame, textvariable=self.patch_note_var, foreground="#606060", wraplength=TEXT_WRAP_PX,
                  justify="left").grid(row=5, column=0, columnspan=3, padx=4, sticky="w")

        frame = ttk.LabelFrame(master, text="Slip profile")
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text="the tab shown is fitted; a typed value holds,\n"
                              "'fit' frees it; blank bounds are automatic", foreground="#606060",
                  justify="left").grid(row=0, column=0, columnspan=2, padx=4, sticky="w")
        self.profile_notebook = ttk.Notebook(frame)
        self.profile_notebook.grid(row=1, column=0, columnspan=2, padx=4, pady=2, sticky="ew")
        self.param_vars = {}
        for model in sprof.MODELS:
            tab = ttk.Frame(self.profile_notebook)
            for column, text in enumerate(("", "value", "fit", "min", "max")):
                ttk.Label(tab, text=text, foreground="#606060").grid(row=0, column=column, padx=2, sticky="w")
            self.param_vars[model] = {}
            row = 0
            for row, name in enumerate(sprof.PARAMETERS[model], start=1):
                label, unit, _ = self._param_format(name)
                ttk.Label(tab, text=f"{label} ({unit})" if unit else label).grid(row=row, column=0, padx=2, sticky="e")
                v = {"value": tk.StringVar(master=self, value=""), "fit": tk.BooleanVar(master=self, value=True),
                     "min": tk.StringVar(master=self, value=""), "max": tk.StringVar(master=self, value="")}
                self.param_vars[model][name] = v
                on_value = functools.partial(self.on_param_value_edited, model, name)
                on_bounds = functools.partial(self.on_param_bounds_edited, model, name)
                self._entry_only(tab, v["value"], row, 1, f"{model} {name} value", on_value, width=9)
                ttk.Checkbutton(tab, variable=v["fit"], command=functools.partial(self.on_param_fit_toggled, model,
                                                                                    name)).grid(row=row, column=2)
                self._entry_only(tab, v["min"], row, 3, f"{model} {name} min", on_bounds, width=7)
                self._entry_only(tab, v["max"], row, 4, f"{model} {name} max", on_bounds, width=7)
                for key in ("value", "min", "max"):
                    self._shown[str(v[key])] = ""
            ttk.Button(tab, text="Fit all, automatic bounds", command=functools.partial(self.reset_profile_params,
                                                                                         model)).grid(
                row=row + 1, column=0, columnspan=5, padx=2, pady=2, sticky="w")
            self.profile_notebook.add(tab, text=PROFILE_LABELS[model])
        self.profile_notebook.select(sprof.MODELS.index(DEFAULT_PROFILE_MODEL))
        self.profile_notebook.bind("<<NotebookTabChanged>>", lambda e: self.on_profile_tab_changed())
        self.weighted_var = tk.BooleanVar(master=self, value=False)
        ttk.Checkbutton(frame, text="weight by 1/sigma^2", variable=self.weighted_var,
                        command=self.on_profile_settings).grid(row=2, column=0, columnspan=2, padx=4, pady=2, sticky="w")
        self.profile_use_mb = self._use_menu(frame, 3)
        self.profile_text_var = tk.StringVar(master=self, value="")
        ttk.Label(frame, textvariable=self.profile_text_var, wraplength=TEXT_WRAP_PX, justify="left").grid(
            row=4, column=0, columnspan=2, padx=4, pady=2, sticky="w")
        try:                                             # the font the texts are packed into lines with
            self._text_font = tkfont.nametofont("TkDefaultFont")
        except tk.TclError:
            self._text_font = None

        frame = ttk.LabelFrame(master, text="Rupture front (t_i against x)")
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text="Fit:").grid(row=0, column=0, padx=4, pady=2, sticky="e")
        self.front_mode_combobox = ttk.Combobox(frame, width=15, state="readonly", values=list(sprof.FRONT_MODES))
        self.front_mode_combobox.set(DEFAULT_FRONT_MODE)
        self.front_mode_combobox.grid(row=0, column=1, padx=4, pady=2, sticky="w")
        self.front_mode_combobox.bind("<<ComboboxSelected>>", lambda e: self.on_front_settings())
        ttk.Label(frame, text="single: one line; bilateral: two lines that\nmeet at a fitted apex",
                  foreground="#606060", justify="left").grid(row=1, column=0, columnspan=2, padx=4, sticky="w")
        self.front_use_mb = self._use_menu(frame, 2)
        self.front_text_var = tk.StringVar(master=self, value="")
        ttk.Label(frame, textvariable=self.front_text_var, wraplength=TEXT_WRAP_PX, justify="left").grid(
            row=3, column=0, columnspan=2, padx=4, pady=2, sticky="w")

    def _use_menu(self, master, row) -> tk.Menubutton:
        mb = tk.Menubutton(master, text="Sensors used", relief="raised")
        mb.grid(row=row, column=0, columnspan=2, padx=4, pady=2, sticky="w")
        mb.menu = tk.Menu(mb, tearoff=0)
        mb["menu"] = mb.menu
        mb.items = []
        return mb

    def _fill_use_menu(self, mb, flags, command) -> None:
        mb.menu.delete(0, "end")
        mb.items = [tk.IntVar(master=self, value=int(bool(f))) for f in flags]
        for i, var in enumerate(mb.items):
            mb.menu.add_checkbutton(label=self.sensor_label(i), variable=var,
                                    command=lambda i=i, var=var: command(i, bool(var.get())))

    @staticmethod
    def _sync_use_menu(mb, flags) -> None:
        for var, flag in zip(mb.items, flags):
            if bool(var.get()) != bool(flag):
                var.set(int(bool(flag)))

    def _entry(self, master, label, var, row, callback, width=9, name=None):
        ttk.Label(master, text=label).grid(row=row, column=0, padx=4, pady=2, sticky="e")
        return self._entry_only(master, var, row, 1, name or label, callback, width)

    def _entry_only(self, master, var, row, column, name, callback, width=9):
        entry = ttk.Entry(master, textvariable=var, width=width)
        entry.grid(row=row, column=column, padx=4, pady=2, sticky="w")
        entry.bind("<KeyRelease>", lambda e: self.schedule(callback), add="+")
        entry.bind("<Return>", lambda e: self.schedule(callback, now=True), add="+")
        entry.bind("<FocusOut>", lambda e: self.schedule(callback, now=True), add="+")
        self.entries[name] = entry
        return entry

    def _show(self, var: tk.StringVar, text: str) -> None:
        """Put ``text`` in an entry and remember it, so that re-reading the same
        (rounded) text never counts as an edit."""
        self._shown[str(var)] = text
        var.set(text)

    def _edited(self, *variables) -> bool:
        """True when the user changed the text of any of these entries."""
        return any(var.get() != self._shown.get(str(var)) for var in variables)

    def on_close(self) -> None:
        if self._compute_after_id is not None:
            try:
                self.after_cancel(self._compute_after_id)
            except tk.TclError:
                pass
            self._compute_after_id = None
            self._pending = None
        super().on_close()

    # ------------------------------------------------------------- loading
    def set_event(self, event_idx: int) -> None:
        """Switch events, refusing (with a warning) one without a slip array."""
        self.flush_pending()
        event_idx = int(event_idx)
        try:
            candidate = self.data_manager.get_data(f"{self.run_path}/events/[{event_idx}]")
        except (KeyError, IndexError, TypeError):
            candidate = None
        if not slip_fields(candidate):
            self.refresh_event_selector()
            messagebox.showwarning("No slip data", f"Event {event_idx} has no slip channel array; "
                                                   f"staying on event {self.event_idx}.", parent=self)
            return
        super().set_event(event_idx)

    def on_event_loaded(self) -> None:
        fields = slip_fields(self.event, self.get_run())
        if not fields:
            raise ValueError(f"{self.event_path} has no slip channel array (one row per sensor, in um) "
                             "to compute the coseismic slip of")
        saved = self.load_results()
        saved_field = str(saved.get("field")) if isinstance(saved, dict) and saved.get("field") is not None else None
        if saved_field in fields:
            field = saved_field
        elif self.field in fields:
            field = self.field
        else:
            field = fields[0]
        self.field_combobox.config(values=fields)
        self.field_combobox.set(field)
        self.load_field(field, saved)

    def on_field_selected(self, event=None) -> None:
        field = self.field_combobox.get()
        if field and field != self.field:
            self.flush_pending()
            self.load_field(field, self.load_results())

    def extraction_window(self) -> tuple[float, float]:
        """The event's extraction window ``(start_s, end_s)`` around the event
        time: from the run's ``event_extraction``, else the event's time axis."""
        if not self.is_waveform_field():         # a full-rate record has its own extent
            try:
                record = extraction_record(self.get_run())
                start, end = float(record["start_s"]), float(record["end_s"])
                if start < 0.0 < end:
                    return start, end
            except (KeyError, TypeError, ValueError):
                pass
        dt = cs.sampling_interval(self.time)
        return min(float(self.time[0]) - self.t_event, -dt), max(float(self.time[-1]) - self.t_event + dt, dt)

    def is_waveform_field(self, field: Optional[str] = None) -> bool:
        field = self.field if field is None else field
        return bool(field) and str(field).startswith(WAVEFORM_FIELD_PREFIX)

    def load_field(self, field: str, saved=None) -> None:
        if self.is_waveform_field(field):
            array = waveform_slip_array(self.event, self.get_run(), field)
            self.pzt = array.get("pzt")
        else:
            array = self.event[field]
            self.pzt = None
        self.field = field
        self.channels = channel_names(array)
        self.data = np.asarray(array["data"], dtype=float)
        self.unit = str(array.get("unit", "um"))
        positions = array.get("positions")
        try:
            x = np.asarray(positions["x"], dtype=float).ravel()
        except (KeyError, TypeError, ValueError):
            x = np.full(len(self.channels), np.nan)
        self.x_mm = x if x.size == len(self.channels) else np.full(len(self.channels), np.nan)
        self.time = np.asarray(array["time"] if self.is_waveform_field(field) else self.event["time"], dtype=float).ravel()
        self.t_event = float(self.event["event_time"])
        self.start_s, self.end_s = self.extraction_window()
        n = len(self.channels)
        record = saved if isinstance(saved, dict) and str(saved.get("field", field)) == field else None
        inputs = cs.restore_inputs(record, self.channels) if record is not None else None
        if inputs is not None:
            self.methods = list(inputs["methods"])
            self.t_shift = list(inputs["t_shift_s"])
            self.fit_before = list(inputs["fit_before_s"])
            self.fit_after = list(inputs["fit_after_s"])
            self._show(self.reference_var, _text(inputs["reference_halfwidth_s"] * 1e3))
            self.restore_filter_tabs(inputs, record)
            self._show(self.search_var, _text(inputs["search_halfwidth_s"] * 1e3))
            self._show(self.snr_var, _text(inputs["rate_snr"]))
            self.restore_t_i_tab(inputs, record)
            self.same_windows_var.set(bool(np.asarray(record.get("same_windows", True)).ravel()[0]))
            self.saved = record
        else:
            before, after = cs.default_fit_windows(self.start_s, self.end_s)
            self.methods = self.methods_for(self.t_i_method(), [self.t_i_method()] * n)
            self.t_shift = [float("nan")] * n
            self.fit_before = [before] * n
            self.fit_after = [after] * n
            self.saved = None
        self.sensor_combobox.config(values=[self.sensor_label(i) for i in range(n)])
        self.sensor = min(max(self.sensor, 0), n - 1)
        self.restore_fit_settings(record if inputs is not None else None)
        self.result = None                           # never keep another event's or field's result
        if not self.recompute(reset_view=True):
            self.clear_display()

    def restore_fit_settings(self, record: Optional[dict]) -> None:
        """Profile and front settings from a saved record.  Without one the model,
        the weighting and the front mode stay as they are and every sensor is
        used.  Patch edges typed here and not saved to the experiment stay;
        otherwise they come from the experiment, else the record, else stay as
        they are."""
        n = len(self.channels)
        profile = record.get("profile") if isinstance(record, dict) else None
        front = record.get("front") if isinstance(record, dict) else None
        if isinstance(profile, dict):
            model = str(profile.get("model", DEFAULT_PROFILE_MODEL))
            self.profile_notebook.select(sprof.MODELS.index(model if model in sprof.MODELS else DEFAULT_PROFILE_MODEL))
            self.weighted_var.set(bool(np.asarray(profile.get("weighted", False)).ravel()[0]))
            self.restore_profile_params(profile.get("settings"))
        self.profile_use = _bools(profile.get("use") if isinstance(profile, dict) else None, n, True)
        if isinstance(front, dict):
            mode = str(front.get("mode", DEFAULT_FRONT_MODE))
            self.front_mode_combobox.set(mode if mode in sprof.FRONT_MODES else DEFAULT_FRONT_MODE)
            self.front_use_auto = bool(np.asarray(front.get("use_auto", True)).ravel()[0])
            self.front_use = _bools(front.get("use"), n, True)
        else:
            self.front_use_auto = True
            self.front_use = [True] * n
        patch = self.patch_x_mm                      # a draft typed here; else what stays when nothing is found
        if not self._patch_draft:
            found = self.experiment_patch()
            if found is None and isinstance(record, dict):
                found = _patch_pair(record.get(PATCH_KEY))
            if found is not None:
                patch = found
        ends = self.fault_ends
        if not self._ends_draft:
            found = self.experiment_ends()
            if found is None and isinstance(record, dict):
                found = _patch_pair(record.get(RECORD_ENDS_KEY))
            if found is not None:
                ends = found
        self._fill_use_menu(self.profile_use_mb, self.profile_use, self.set_profile_use)
        self._fill_use_menu(self.front_use_mb, self.front_use, self.set_front_use)
        self.show_patch(patch)
        self.show_ends(ends)

    def restore_profile_params(self, settings) -> None:
        """Every model tab's holds and bounds from a record's ``profile.settings``;
        what it lacks is fitted, within the automatic bounds."""
        settings = settings if isinstance(settings, dict) else {}
        for model, params in self.param_vars.items():
            saved = settings.get(model) if isinstance(settings.get(model), dict) else {}
            for name, v in params.items():
                entry = saved.get(name) if isinstance(saved.get(name), dict) else {}
                held = _float_or_nan(entry.get("value"))
                fit = bool(np.asarray(entry.get("fit", True)).ravel()[0]) or not math.isfinite(held)
                v["fit"].set(fit)
                self._show(v["value"], "" if fit else _text(held))
                for key in ("min", "max"):
                    self._show(v[key], _text(_float_or_nan(entry.get(key))))

    def profile_settings(self) -> dict:
        """Every model tab's holds and bounds, as stored in ``profile.settings``."""
        out = {}
        for model, params in self.param_vars.items():
            out[model] = {}
            for name, v in params.items():
                fit = bool(v["fit"].get())
                out[model][name] = {"fit": fit, "value": float("nan") if fit else _text_float(v["value"].get()),
                                    "min": _text_float(v["min"].get()), "max": _text_float(v["max"].get())}
        return out

    def restore_t_i_tab(self, inputs: dict, record: dict) -> None:
        """The event-time tab and its values from a record: the tab of the shared or
        the pick method the sensors use (max slip rate when all are set by hand)."""
        self._show(self.onset_var, _text(inputs["onset_fraction"] * 100))
        self._show(self.step_var, _text(inputs["step_fraction"] * 100))
        shared = inputs["shared_shift_s"]
        typed = _float_or_nan(record.get("shared_set_s"))
        self.shared_source_var.set("reference" if math.isnan(shared) else "set")
        value = shared if math.isfinite(shared) else typed
        self._show(self.shared_shift_var, _text(round(value * 1e3, 6)) if math.isfinite(value) else "")
        self._show(self.burst_halfwidth_var, _text(round(inputs["burst_halfwidth_s"] * 1e3, 6)))
        self._show(self.level_width_var, _text(round(inputs["level_width_s"] * 1e3, 6)))
        self._show(self.burst_search_var, _text(round(inputs["burst_search_s"] * 1e3, 6)))
        self.burst_source_var.set(inputs["burst_source"])
        methods = inputs["methods"]
        if cs.METHOD_BURST in methods:
            tab = cs.METHOD_BURST
        elif cs.METHOD_SHARED in methods:
            tab = cs.METHOD_SHARED
        else:
            picks = [m for m in methods if m in cs.PICK_METHODS]
            tab = max(cs.PICK_METHODS, key=picks.count) if picks else cs.METHOD_MAX_RATE
        self.t_i_notebook.select(T_I_TABS.index(tab))

    def restore_filter_tabs(self, inputs: dict, record: dict) -> None:
        """Every filter tab's values from a record: the analysis keys give the filter
        used, ``rate_filter_tabs`` the window of the other windowed filter."""
        rate_filter = inputs["rate_filter"]
        windows = record.get("rate_filter_tabs") if isinstance(record.get("rate_filter_tabs"), dict) else {}

        def window(key, default):
            try:
                value = float(np.asarray(windows.get(key, default)).ravel()[0])
            except (TypeError, ValueError, IndexError):
                value = default
            return value if math.isfinite(value) else default

        savgol = inputs["rate_smoothing_s"] if rate_filter == "savgol" else window("savgol_window_s",
                                                                                   DEFAULT_SAVGOL_WINDOW_S)
        boxcar = inputs["rate_smoothing_s"] if rate_filter == "boxcar" else window("boxcar_window_s",
                                                                                   cs.DEFAULT_RATE_SMOOTHING_S)
        self._show(self.savgol_window_var, _text(savgol * 1e3))
        self._show(self.smoothing_var, _text(boxcar * 1e3))
        self._show(self.savgol_order_var, str(inputs["rate_savgol_order"]))
        self._show(self.cutoff_var, _text(inputs["rate_cutoff_hz"]))
        self._show(self.lowpass_order_var, str(inputs["rate_lowpass_order"]))
        self.lowpass_type_combobox.set(LOWPASS_LABELS[inputs["rate_lowpass_type"]])
        self.select_rate_filter(rate_filter)

    def show_patch(self, patch: Optional[tuple]) -> None:
        """Make ``patch`` the edges drawn, put them in the entries and update the note."""
        self.patch_x_mm = patch
        self._show(self.patch_lo_var, _text(patch[0]) if patch else "")
        self._show(self.patch_hi_var, _text(patch[1]) if patch else "")
        self.update_patch_note()

    def sensor_label(self, i: int) -> str:
        x = self.x_mm[i]
        return f"{i + 1}: {self.channels[i]}" + (f" ({x:g} mm)" if math.isfinite(x) else "")

    def clear_display(self) -> None:
        """No result: empty table and axes; the status keeps the reason."""
        self.table.delete(*self.table.get_children())
        self.ax_trace.clear()
        self.ax_rate.clear()
        self.handles = {}
        self.spans = {}
        self.profile_fit = self.front_fit = None
        self.plot_fault(reset_view=True)
        self.profile_text_var.set("")
        self.front_text_var.set("")
        self.canvas.draw_idle()

    # ----------------------------------------------------------- settings
    def settings_vars(self) -> tuple:
        return (self.reference_var, self.smoothing_var, self.savgol_window_var, self.savgol_order_var,
                self.cutoff_var, self.lowpass_order_var, self.search_var, self.snr_var, self.onset_var, self.step_var,
                self.burst_halfwidth_var, self.level_width_var, self.burst_search_var)

    def rate_filter(self) -> str:
        """The filter of the tab shown."""
        try:
            return cs.RATE_FILTERS[self.rate_filter_notebook.index("current")]
        except (tk.TclError, IndexError):
            return cs.DEFAULT_RATE_FILTER

    def select_rate_filter(self, rate_filter: str) -> None:
        """Show the tab of ``rate_filter`` (the analysis follows on the next recompute)."""
        self.rate_filter_notebook.select(cs.RATE_FILTERS.index(rate_filter))

    def lowpass_type(self) -> str:
        label = self.lowpass_type_combobox.get()
        return next((k for k, v in LOWPASS_LABELS.items() if v == label), cs.DEFAULT_LOWPASS_TYPE)

    def on_filter_choice(self) -> None:
        """Another filter tab or low-pass type: recompute, unless the result already
        has it (a tab selected while restoring a record)."""
        self.flush_pending()
        r = self.result
        if r is not None and not self._stale and r.rate_filter == self.rate_filter() and \
                r.rate_lowpass_type == self.lowpass_type():
            return
        self.recompute()

    def read_settings(self) -> dict:
        reference = _parse(self.reference_var.get(), "reference half-width (ms)", 1e-3)
        boxcar = _parse(self.smoothing_var.get(), "moving average window (ms)", 1e-3)
        savgol = _parse(self.savgol_window_var.get(), "Savitzky-Golay window (ms)", 1e-3)
        savgol_order = _parse_int(self.savgol_order_var.get(), "Savitzky-Golay order")
        cutoff = _parse(self.cutoff_var.get(), "low-pass cutoff (Hz)")
        lowpass_order = _parse_int(self.lowpass_order_var.get(), "low-pass order")
        search = _parse(self.search_var.get(), "search half-width (ms)", 1e-3)
        snr = _parse(self.snr_var.get(), "flag threshold")
        if reference <= 0:
            raise ValueError("the reference half-width must be > 0")
        if boxcar < 0:
            raise ValueError("the moving average window must be >= 0")
        if savgol <= 0:
            raise ValueError("the Savitzky-Golay window must be > 0")
        if cutoff <= 0:
            raise ValueError("the low-pass cutoff must be > 0")
        for order, name in ((savgol_order, "Savitzky-Golay order"), (lowpass_order, "low-pass order")):
            if not 1 <= order <= cs.MAX_FILTER_ORDER:
                raise ValueError(f"the {name} must be from 1 to {cs.MAX_FILTER_ORDER}")
        if search <= 0:
            raise ValueError("the search half-width must be > 0")
        if snr < 0:
            raise ValueError("the flag threshold must be >= 0")
        onset = _parse(self.onset_var.get(), "onset threshold (%)") / 100.0
        step = _parse(self.step_var.get(), "step level (%)") / 100.0
        for value, name in ((onset, "onset threshold"), (step, "step level")):
            if not 0.0 < value < 1.0:
                raise ValueError(f"the {name} must be between 0 and 100 %")
        shared = float("nan")
        if self.shared_source_var.get() == "set":
            shared = _parse(self.shared_shift_var.get(), "shared t_i (ms)", 1e-3)
        burst_w = _parse(self.burst_halfwidth_var.get(), "burst half-width (ms)", 1e-3)
        level_d = _parse(self.level_width_var.get(), "level width (ms)", 1e-3)
        burst_search = _parse(self.burst_search_var.get(), "burst search half-width (ms)", 1e-3)
        if burst_w < 0:
            raise ValueError("the burst half-width must be >= 0")
        if level_d <= 0:
            raise ValueError("the level width must be > 0")
        if burst_search <= 0:
            raise ValueError("the burst search half-width must be > 0")
        burst_source = self.burst_source_var.get()
        if burst_source not in cs.BURST_SOURCES:
            raise ValueError("choose where the burst is picked")
        rate_filter = self.rate_filter()
        return {"rate_filter": rate_filter, "rate_cutoff_hz": cutoff, "rate_lowpass_order": lowpass_order,
                "rate_lowpass_type": self.lowpass_type(), "rate_savgol_order": savgol_order,
                "rate_smoothing_s": savgol if rate_filter == "savgol" else boxcar,
                "search_halfwidth_s": search, "rate_snr": snr, "reference_halfwidth_s": reference,
                "onset_fraction": onset, "step_fraction": step, "shared_shift_s": shared,
                "window_mode": "levels" if self.t_i_method() == cs.METHOD_BURST else "lines",
                "burst_halfwidth_s": burst_w, "level_width_s": level_d, "burst_search_s": burst_search,
                "burst_source": burst_source}

    def filter_windows(self) -> dict:
        """The windows of the two windowed filters, whichever tab is shown (s)."""
        return {"savgol_window_s": _parse(self.savgol_window_var.get(), "Savitzky-Golay window (ms)", 1e-3),
                "boxcar_window_s": _parse(self.smoothing_var.get(), "moving average window (ms)", 1e-3)}

    def schedule(self, callback, now: bool = False) -> None:
        """Run ``callback`` at once, or once the keys have been quiet for ``LIVE_DELAY_MS``."""
        if self._compute_after_id is not None:
            try:
                self.after_cancel(self._compute_after_id)
            except tk.TclError:
                pass
            self._compute_after_id = None
        pending, self._pending = self._pending, None
        if pending is not None and pending is not callback:
            pending()                                   # a different edit was waiting: apply it first
        if now:
            callback()
        else:
            self._pending = callback
            self._compute_after_id = self.after(LIVE_DELAY_MS, self._run_scheduled)

    def _run_scheduled(self) -> None:
        self._compute_after_id = None
        callback, self._pending = self._pending, None
        if callback is not None:
            callback()

    def flush_pending(self) -> None:
        """Apply a typed edit that is still waiting for the keys to settle."""
        if self._compute_after_id is not None:
            try:
                self.after_cancel(self._compute_after_id)
            except tk.TclError:
                pass
            self._compute_after_id = None
        callback, self._pending = self._pending, None
        if callback is not None:
            callback()

    def recompute(self, reset_view: bool = False) -> bool:
        """Analyse every sensor with the current inputs and redraw; False (and the
        reason in the status line) when an input is unusable."""
        try:
            settings = self.read_settings()
            result = cs.analyze_slip(self.time, self.data, self.t_event, self.start_s, self.end_s,
                                     channels=self.channels, x_mm=self.x_mm, methods=self.methods,
                                     t_shift_s=self.t_shift, fit_before_s=self.fit_before,
                                     fit_after_s=self.fit_after, pzt=self.pzt, **settings)
        except ValueError as e:
            self.status_var.set(f"not recomputed: {e}")      # the plot still shows the last result
            self._stale = True
            return False
        for var in self.settings_vars():
            self._shown[str(var)] = var.get()
        self._stale = False
        self.result = result
        self.methods = [s.method for s in result.sensors]
        self.t_shift = [s.t_shift_s for s in result.sensors]
        self.refit(draw=False)
        self.show_sensor()
        self.fill_table()
        self.plot(reset_view=reset_view)
        self.update_status()
        return True

    # --------------------------------------------------------------- sensor
    def current(self) -> Optional[cs.SensorSlip]:
        if self.result is None or not (0 <= self.sensor < len(self.result.sensors)):
            return None
        return self.result.sensors[self.sensor]

    def show_sensor(self) -> None:
        """Put the selected sensor's values into the controls."""
        s = self.current()
        if s is None:
            return
        self._syncing = True
        try:
            self.sensor_combobox.set(self.sensor_label(self.sensor))
            tab = self.t_i_method()
            if tab == cs.METHOD_SHARED:
                self.method_combobox.config(values=[METHOD_LABELS[cs.METHOD_SHARED]], state="disabled")
            else:
                self.method_combobox.config(values=[METHOD_LABELS[tab], METHOD_LABELS[cs.METHOD_MANUAL]],
                                            state="readonly")
            self.method_combobox.set(METHOD_LABELS[s.method])
            self._show(self.t_shift_var, _text(round(s.t_shift_s * 1e3, 6)))
            self._show(self.before_start_var, _text(s.fit_before_s[0]))
            self._show(self.before_end_var, _text(s.fit_before_s[1]))
            self._show(self.after_start_var, _text(s.fit_after_s[0]))
            self._show(self.after_end_var, _text(s.fit_after_s[1]))
            iid = str(self.sensor)
            if self.table.exists(iid) and tuple(self.table.selection()) != (iid,):
                self.table.selection_set(iid)
                self.table.see(iid)
        finally:
            self._syncing = False

    def set_sensor(self, i: int) -> None:
        if self.result is None:
            return
        self.flush_pending()
        i = min(max(int(i), 0), len(self.channels) - 1)
        if i == self.sensor and self.sensor_combobox.get() == self.sensor_label(i):
            return
        self.sensor = i
        self.show_sensor()
        self.plot(reset_view=False, new_sensor=True)
        self.update_status()

    def step_sensor(self, delta: int) -> None:
        self.set_sensor(self.sensor + delta)

    def on_sensor_selected(self, event=None) -> None:
        values = list(self.sensor_combobox["values"])
        text = self.sensor_combobox.get()
        if text in values:
            self.set_sensor(values.index(text))

    def on_table_select(self, event=None) -> None:
        if self._syncing:
            return
        selection = self.table.selection()
        if selection:
            self.set_sensor(int(selection[0]))

    def fill_table(self) -> None:
        self._syncing = True
        try:
            self.table.delete(*self.table.get_children())
            for i, s in enumerate(self.result.sensors):
                self.table.insert("", "end", iid=str(i), values=(
                    s.channel, _text(s.x_mm), METHOD_LABELS[s.method], f"{s.t_shift_s * 1e3:.2f}",
                    f"{s.coseismic_slip:.3f}" if math.isfinite(s.coseismic_slip) else "n/a",
                    f"{s.noise:.3f}" if math.isfinite(s.noise) else "n/a", "; ".join(s.flags)))
            if self.table.exists(str(self.sensor)):
                self.table.selection_set(str(self.sensor))
        finally:
            self._syncing = False

    # --------------------------------------------------------------- edits
    def on_method_selected(self, event=None) -> None:
        self.flush_pending()
        if self.result is None or self.t_i_method() == cs.METHOD_SHARED:
            return
        label = self.method_combobox.get()
        method = next((k for k, v in METHOD_LABELS.items() if v == label), None)
        if method is None or method == self.methods[self.sensor]:
            return
        self.methods[self.sensor] = method       # manual keeps the current t_i as its start
        self.recompute()

    def on_t_shift_edited(self) -> None:
        if self._syncing or self.current() is None:
            return
        if not self._edited(self.t_shift_var):
            self.update_status()                  # the applied value again: an earlier error no longer holds
            return
        try:
            shift = _parse(self.t_shift_var.get(), "t_i - event time (ms)", 1e-3)
        except ValueError as e:
            self.status_var.set(str(e))
            return
        self.set_event_time(self.sensor, shift)

    def set_event_time(self, i: int, shift_s: float) -> None:
        """Make sensor ``i`` manual at ``event_time + shift_s`` and recompute; on the
        Shared tab, set the one t_i of every sensor there instead."""
        if self.t_i_method() == cs.METHOD_SHARED:
            self.shared_source_var.set("set")
            self._show(self.shared_shift_var, _text(round(float(shift_s) * 1e3, 6)))
            self.recompute()
            return
        self.methods[i] = cs.METHOD_MANUAL
        self.t_shift[i] = float(shift_s)
        self.recompute()

    # ------------------------------------------------------------ t_i tabs
    def t_i_method(self) -> str:
        """The event-time method of the tab shown."""
        try:
            return T_I_TABS[self.t_i_notebook.index("current")]
        except (tk.TclError, IndexError):
            return cs.METHOD_MAX_RATE

    def select_t_i_method(self, method: str) -> None:
        """Show ``method``'s tab and apply it."""
        self.t_i_notebook.select(T_I_TABS.index(method))
        self.on_t_i_tab_changed()

    def methods_for(self, tab: str, methods) -> list:
        """The sensors' methods under ``tab``: all shared, or the tab's pick except
        for the sensors set by hand."""
        if tab in (cs.METHOD_SHARED, cs.METHOD_BURST):
            return [tab] * len(methods)
        return [m if m == cs.METHOD_MANUAL else tab for m in methods]

    def on_t_i_tab_changed(self) -> None:
        self.flush_pending()
        methods = self.methods_for(self.t_i_method(), self.methods)
        if methods == self.methods and self.result is not None and not self._stale:
            return                                       # a tab shown while a record was restored
        self.methods = methods
        if self.result is not None or self._stale:
            self.recompute()

    def on_shared_source(self) -> None:
        self.flush_pending()
        if self.shared_source_var.get() == "set" and not self.shared_shift_var.get().strip():
            ref = self.result.t_ref - self.t_event if self.result is not None else 0.0
            self._show(self.shared_shift_var, _text(round(ref * 1e3, 6)))   # start from the reference
        if self.t_i_method() == cs.METHOD_SHARED:
            self.recompute()

    def on_burst_source(self) -> None:
        """The burst is picked elsewhere: recompute when the Burst tab is shown."""
        self.flush_pending()
        if self.t_i_method() == cs.METHOD_BURST:
            self.recompute()
        else:
            self.update_status()

    def on_shared_shift_edited(self) -> None:
        if self._syncing or not self._edited(self.shared_shift_var):
            return
        self.shared_source_var.set("set")
        if self.t_i_method() == cs.METHOD_SHARED:
            self.recompute()
        else:
            self.update_status()

    def read_windows(self) -> tuple[tuple, tuple]:
        """The windows typed in the entries (ValueError when unusable)."""
        before = cs.check_window((_parse(self.before_start_var.get(), "fit before start (s)"),
                                  _parse(self.before_end_var.get(), "fit before end (s)")), "fit before")
        after = cs.check_window((_parse(self.after_start_var.get(), "fit after start (s)"),
                                 _parse(self.after_end_var.get(), "fit after end (s)")), "fit after")
        self.check_sides(before, after)
        return before, after

    def on_window_edited(self) -> None:
        window_vars = (self.before_start_var, self.before_end_var, self.after_start_var, self.after_end_var)
        if self._syncing or self.result is None:
            return
        if not self._edited(*window_vars):
            self.update_status()
            return
        try:
            before, after = self.read_windows()
        except ValueError as e:
            self.status_var.set(str(e))
            return
        self.set_windows(before, after)

    @staticmethod
    def check_sides(before, after) -> None:
        if before[1] > after[0]:
            raise ValueError("the before window must end at or before the after window starts")

    def set_windows(self, before, after) -> None:
        """Windows of the selected sensor, or of every sensor with "same windows"."""
        targets = range(len(self.channels)) if self.same_windows_var.get() else [self.sensor]
        for i in targets:
            self.fit_before[i] = tuple(float(v) for v in before)
            self.fit_after[i] = tuple(float(v) for v in after)
        self.recompute()

    def on_same_windows_toggled(self) -> None:
        self.flush_pending()
        if self.same_windows_var.get() and self.result is not None:
            self.set_windows(self.fit_before[self.sensor], self.fit_after[self.sensor])
        else:
            self.update_status()

    def on_settings_edited(self) -> None:
        # after a recompute that failed (no result, or one that lags the inputs) any value is worth a try
        if self.result is None or self._stale or self._edited(*self.settings_vars()):
            self.recompute()
        else:
            self.update_status()

    def repick_all(self) -> None:
        """Every sensor back to the tab's pick (on the Shared tab: the one t_i back
        to the peak of the mean slip rate)."""
        self.flush_pending()
        tab = self.t_i_method()
        if tab == cs.METHOD_SHARED:
            self.shared_source_var.set("reference")
        self.methods = [tab] * len(self.channels)
        self.recompute()

    # ------------------------------------------------------- profile and front
    def profile_model(self) -> str:
        """The model of the tab shown."""
        try:
            return sprof.MODELS[self.profile_notebook.index("current")]
        except (tk.TclError, IndexError):
            return DEFAULT_PROFILE_MODEL

    def select_profile_model(self, model: str) -> None:
        """Show ``model``'s tab and fit it."""
        self.profile_notebook.select(sprof.MODELS.index(model))
        self.refit()

    def on_profile_tab_changed(self) -> None:
        self.flush_pending()
        if self.profile_fit is None or self.profile_fit.model != self.profile_model():
            self.refit()

    def read_profile_constraints(self, model: Optional[str] = None) -> tuple:
        """``(fixed, bounds)`` of a model's tab: the held values (no 'fit' tick) and
        the bounds typed (NaN for a blank end).  ValueError when an entry is unusable."""
        fixed, bounds = {}, {}
        for name, v in self.param_vars[model or self.profile_model()].items():
            label = self._param_format(name)[0]
            if not v["fit"].get():
                fixed[name] = _parse(v["value"].get(), f"{label} value")
            lo, hi = (_parse_blank(v[key].get(), f"{label} {key}") for key in ("min", "max"))
            if math.isfinite(lo) or math.isfinite(hi):
                bounds[name] = (lo, hi)
        return fixed, bounds

    def on_param_value_edited(self, model: str, name: str) -> None:
        """A typed value holds the parameter there; clearing it lets the fit choose again."""
        v = self.param_vars[model][name]
        if self._syncing or not self._edited(v["value"]):
            return
        text = v["value"].get().strip()
        if text:
            try:
                _parse(text, f"{self._param_format(name)[0]} value")
            except ValueError as e:
                self.status_var.set(str(e))
                return
        v["fit"].set(not text)
        self._shown[str(v["value"])] = v["value"].get()
        self.on_param_changed(model)

    def on_param_fit_toggled(self, model: str, name: str) -> None:
        """Ticked: the fit chooses; unticked: the parameter holds the value shown."""
        self.flush_pending()
        v = self.param_vars[model][name]
        if not v["fit"].get():
            try:
                _parse(v["value"].get(), f"{self._param_format(name)[0]} value")
            except ValueError:
                v["fit"].set(True)
                self.status_var.set(f"type a value for {self._param_format(name)[0]} to hold it at")
                return
        self.on_param_changed(model)

    def on_param_bounds_edited(self, model: str, name: str) -> None:
        v = self.param_vars[model][name]
        if self._syncing or not self._edited(v["min"], v["max"]):
            return
        try:
            for key in ("min", "max"):
                _parse_blank(v[key].get(), f"{self._param_format(name)[0]} {key}")
        except ValueError as e:
            self.status_var.set(str(e))
            return
        for key in ("min", "max"):
            self._shown[str(v[key])] = v[key].get()
        self.on_param_changed(model)

    def reset_profile_params(self, model: str) -> None:
        """Every parameter of ``model`` fitted again, within the automatic bounds."""
        self.flush_pending()
        for v in self.param_vars[model].values():
            v["fit"].set(True)
            self._show(v["min"], "")
            self._show(v["max"], "")
        self.on_param_changed(model)

    def on_param_changed(self, model: str) -> None:
        if model == self.profile_model():
            self.refit()
        else:
            self.update_status()

    def show_profile_params(self) -> None:
        """The fitted values in the value entries of the free parameters of the tab shown."""
        fit = self.profile_fit
        if fit is None or not fit.valid:
            return
        for name, v in self.param_vars[fit.model].items():
            if v["fit"].get() and name in fit.params:
                self._show(v["value"], _text(fit.params[name]))

    def front_mode(self) -> str:
        mode = self.front_mode_combobox.get()
        return mode if mode in sprof.FRONT_MODES else DEFAULT_FRONT_MODE

    def refit(self, draw: bool = True) -> None:
        """Fit the slip profile and the rupture front to the current per-sensor result."""
        if self.result is None:
            return
        sensors = self.result.sensors
        slip = np.array([s.coseismic_slip for s in sensors])
        sigma = np.array([s.noise for s in sensors])
        model = self.profile_model()
        try:
            fixed, bounds = self.read_profile_constraints(model)
            self.profile_fit = sprof.fit_profile(self.x_mm, slip, sigma, model=model,
                                                 weighted=bool(self.weighted_var.get()), use=self.profile_use,
                                                 fixed=fixed, bounds=bounds)
        except ValueError as e:                          # a parameter entry that cannot be used
            self.profile_fit = sprof.ProfileFit(model, {name: float("nan") for name in sprof.PARAMETERS[model]},
                                                False, str(e), 0, bool(self.weighted_var.get()))
        self.show_profile_params()
        if self.front_use_auto:
            self.front_use = [not doubtful_t_i(s) for s in sensors]
            self._sync_use_menu(self.front_use_mb, self.front_use)
        # the t_i sit on the samples: a segment crossed within one of them is not resolved
        self.front_fit = sprof.fit_front(self.x_mm, [s.t_shift_s for s in sensors], use=self.front_use,
                                         mode=self.front_mode(), dt_s=cs.sampling_interval(self.time))
        self.profile_text_var.set(self.profile_text())
        self.front_text_var.set(self.front_text())
        if draw:
            self.plot_fault()
            self.canvas.draw_idle()
            self.update_status()

    def on_profile_settings(self) -> None:
        self.refit()

    def on_front_settings(self) -> None:
        self.refit()

    def set_profile_use(self, i: int, used: bool) -> None:
        self.profile_use[int(i)] = bool(used)
        self._sync_use_menu(self.profile_use_mb, self.profile_use)
        self.refit()

    def set_front_use(self, i: int, used: bool) -> None:
        """An explicit choice of the front's sensors; it stops following the flags."""
        self.front_use_auto = False
        self.front_use[int(i)] = bool(used)
        self._sync_use_menu(self.front_use_mb, self.front_use)
        self.refit()

    def _fits(self, text: str) -> bool:
        """True when ``text`` fits on one line of the fit texts."""
        if self._text_font is not None:
            try:
                return self._text_font.measure(text) <= TEXT_WRAP_PX
            except tk.TclError:
                pass
        return len(text) <= TEXT_WRAP_PX // 7

    def _pack(self, rows) -> list:
        """Lines of short texts: those of a row joined by ", " while they fit
        ``TEXT_WRAP_PX``, and a row that fits whole on the line before it joins it."""
        lines = []
        for row in rows:
            row_lines = []
            for text in row:
                if row_lines and self._fits(f"{row_lines[-1]}, {text}"):
                    row_lines[-1] = f"{row_lines[-1]}, {text}"
                else:
                    row_lines.append(text)
            if lines and len(row_lines) == 1 and self._fits(f"{lines[-1]}, {row_lines[0]}"):
                lines[-1] = f"{lines[-1]}, {row_lines[0]}"
            else:
                lines.extend(row_lines)
        return lines

    def _param_format(self, name: str) -> tuple:
        """``(label, unit, digits)`` of a profile parameter as shown."""
        label, unit, digits = PARAM_FORMAT.get(name, (name, "", 4))
        return label, unit.format(u=self.unit), digits

    def _param_text(self, fit: sprof.ProfileFit, name: str) -> str:
        """``name = value unit``, with the range that fits equally well beside it when
        the parameter has one (it moves with an unresolved one)."""
        label, unit, digits = self._param_format(name)
        text = f"{label} = {fit.params[name]:.{digits}g}" + (f" {unit}" if unit else "")
        if name in fit.fixed:
            text += " (held)"
        if name in fit.ranges:
            lo, hi = (float(v) for v in fit.ranges[name])
            text += f" ({_range_text(lo, hi, digits)} fit equally)"
        return text

    def profile_text(self) -> str:
        """The profile fit: the parameters the sensors pin and the rupture extent;
        under "not resolved by the sensors" those they leave free, as the range that
        fits equally well where it is known, and the free rupture edges as their
        brackets; then the fit quality."""
        fit = self.profile_fit
        label = PROFILE_LABELS.get(fit.model, fit.model) if fit is not None else ""
        if fit is None or not fit.valid:
            return f"{label}: {fit.reason if fit is not None else 'no fit'}"
        u = self.unit
        unresolved = [name for name in fit.unresolved if name in fit.params]
        rows = []
        for row in PARAM_ROWS.get(fit.model, (tuple(fit.params),)):
            shown = [name for name in row if name not in unresolved]
            if shown == ["D", "xc"] and "D" not in fit.ranges:
                rows.append([f"{self._param_text(fit, 'D')} at {self._param_text(fit, 'xc')}"])
            else:
                rows.append([self._param_text(fit, name) for name in shown])
        lines = self._pack(rows)
        if fit.model == "quadratic" and math.isfinite(fit.peak) and not unresolved:
            lines.append(f"peak {fit.peak:.3g} {u} at x = {fit.peak_x_mm:.4g} mm")
        brackets = tip_brackets(fit)
        reached = self.end_slips(fit)

        def edge(side, label, tip) -> str:
            if side in reached:
                x_end, value = reached[side]
                return f"the {label} fault end ({value:.3g} {u} at {x_end:.4g} mm)"
            return f"{tip:.4g} mm"

        if not brackets:
            if all(math.isfinite(v) for v in fit.tips_mm):
                (s_lo, l_lo), (s_hi, l_hi) = SIDES
                lines.append(f"rupture from {edge(s_lo, l_lo, fit.tips_mm[0])} to {edge(s_hi, l_hi, fit.tips_mm[1])}")
        else:                                            # the pinned edge of a pair with a free one
            lines.extend(f"{side_label} edge at {edge(side, side_label, tip)}"
                         for (side, side_label), tip in zip(SIDES, fit.tips_mm)
                         if side not in brackets and math.isfinite(tip))
            lines.extend(f"slip at the {side_label} fault end: {reached[side][1]:.3g} {u}"
                         for side, side_label in SIDES if side in brackets and side in reached)
        free = []
        named = [self._param_format(name)[0] for name in unresolved if name not in fit.ranges]
        if named:
            free.append(", ".join(named))
        for name in unresolved:
            if name in fit.ranges:
                plabel, unit, digits = self._param_format(name)
                lo, hi = (float(v) for v in fit.ranges[name])
                free.append(f"{plabel} {_range_text(lo, hi, digits, unit)}")
        free.extend(_edge_text(side_label, *brackets[side]) for side, side_label in SIDES if side in brackets)
        if free:
            lines += [NOT_RESOLVED] + free
        prefix = f"{label}:"
        if lines and self._fits(f"{prefix} {lines[0]}"):
            lines[0] = f"{prefix} {lines[0]}"
        else:
            lines.insert(0, prefix)
        lines.append(f"R^2 = {fit.r2:.4f}, rms = {fit.rms:.3g} {u}, {fit.n} sensors"
                     + (", weighted" if fit.weighted else ""))
        if fit.at_bounds:
            lines.append("at a bound: " + ", ".join(self._param_format(name)[0] for name in fit.at_bounds))
        return "\n".join(lines)

    def front_text(self) -> str:
        fit = self.front_fit
        if fit is None:
            return "front: no fit"
        auto = None
        if self.front_use_auto:
            n_doubtful = sum(1 for used in self.front_use if not used)
            auto = (f"sensors: all but {n_doubtful} with a doubtful t_i" if n_doubtful
                    else "sensors: all, none with a doubtful t_i")
        if not fit.valid:
            reason = fit.reason
            if self.front_use_auto:
                n = sum(1 for used in self.front_use if used)
                if n < 2:
                    reason = f"{'no sensor' if n == 0 else 'only one sensor'} with a clean t_i"
                reason += "; choose sensors under Sensors used"
            return "\n".join([f"front: {reason}"] + ([auto] if auto else []))
        lines = [segment_text(seg) for seg in fit.segments]
        apex = apex_text(fit)
        if apex:
            lines.append(apex)
        if auto:
            lines.append(auto)
        return "\n".join(lines)

    def experiment_value(self, key: str, legacy_key: str):
        """``analysis['fault'][key]`` as stored, else where earlier files kept it:
        ``data[legacy_key]`` at the top level, or in a ``data['fault']`` dict (a
        ``fault`` that is text, a description of the sample, is left alone); None
        when nowhere."""
        data = self.data_manager.data if isinstance(self.data_manager.data, dict) else {}
        value = get_setting(data, FAULT_SETTING, key)
        if value is not None:
            return value
        if data.get(legacy_key) is not None:
            return data[legacy_key]
        fault = data.get(LEGACY_FAULT_KEY)
        return fault.get(key) if isinstance(fault, dict) else None

    def store_experiment_value(self, key: str, legacy_key: str, value: list) -> None:
        """Write ``analysis['fault'][key]`` and drop the copies earlier files kept."""
        data = self.data_manager.data
        set_setting(data, value, FAULT_SETTING, key)
        data.pop(legacy_key, None)
        fault = data.get(LEGACY_FAULT_KEY)
        if isinstance(fault, dict) and key in fault:
            del fault[key]
            if not fault:
                del data[LEGACY_FAULT_KEY]

    def experiment_patch_value(self):
        """The experiment's patch edges as stored, or None when there are none."""
        return self.experiment_value(PATCH_KEY, PATCH_KEY)

    def experiment_patch(self) -> Optional[tuple]:
        """The experiment's patch edges, increasing, or None when absent or unusable."""
        return _patch_pair(self.experiment_patch_value())

    def read_patch(self) -> Optional[tuple]:
        """The patch edges in the entries: None when both are blank, ValueError
        when unusable.  An entry the user has not edited gives the edge in use at
        full precision, not its rounded text."""
        lo_text, hi_text = self.patch_lo_var.get().strip(), self.patch_hi_var.get().strip()
        if not lo_text and not hi_text:
            return None
        applied = self.patch_x_mm
        lo = applied[0] if applied and not self._edited(self.patch_lo_var) else _parse(lo_text, "patch from (mm)")
        hi = applied[1] if applied and not self._edited(self.patch_hi_var) else _parse(hi_text, "patch to (mm)")
        if lo >= hi:
            raise ValueError("the patch must have from < to")
        return lo, hi

    def on_patch_edited(self) -> None:
        if self._syncing:
            return
        if not self._edited(self.patch_lo_var, self.patch_hi_var):
            self.update_status()
            return
        try:
            patch = self.read_patch()
        except ValueError as e:
            self.status_var.set(str(e))
            return
        for var in (self.patch_lo_var, self.patch_hi_var):
            self._shown[str(var)] = var.get()
        self.patch_x_mm = patch
        self._patch_draft = patch is not None
        self.update_patch_note()
        self.plot_fault()
        self.canvas.draw_idle()
        self.update_status()

    def update_patch_note(self) -> None:
        saved, value = self.experiment_patch(), self.experiment_patch_value()
        if self.patch_x_mm is None:
            note = "no patch edges in the experiment" if saved is None else ""
        elif saved == self.patch_x_mm:
            note = "saved in the experiment"
        else:
            note = "not saved to the experiment"
        if saved is None and value is not None:
            unusable = f"experiment patch edges unusable: {_value_text(value)}"
            note = f"{note}; {unusable}" if self.patch_x_mm is not None else unusable
        self.patch_note_var.set(note)

    def save_patch(self) -> bool:
        """Store the patch edges for the whole experiment (``analysis['fault']['patch_x_mm']``)
        and show them in the other open Slip Distribution windows."""
        self.flush_pending()
        try:
            patch = self.read_patch()
        except ValueError as e:
            self.status_var.set(str(e))
            return False
        if patch is None:
            self.status_var.set("type both patch edges first")
            return False
        old = self.experiment_patch()
        self.store_experiment_value(PATCH_KEY, PATCH_KEY, [float(patch[0]), float(patch[1])])
        for var in (self.patch_lo_var, self.patch_hi_var):
            self._shown[str(var)] = var.get()
        self.patch_x_mm = patch
        self._patch_draft = False
        self.app.refresh_tree()
        self.update_patch_note()
        self.plot_fault()
        self.canvas.draw_idle()
        for view in list(getattr(self.app, "child_windows", [])):
            if view is not self and isinstance(view, SlipDistributionView) and view.winfo_exists():
                view.on_experiment_patch_saved(old)
        self.status_var.set("patch edges saved in the experiment")
        return True

    def on_experiment_patch_saved(self, old: Optional[tuple]) -> None:
        """Another window saved new patch edges in the experiment (replacing
        ``old``): show them, unless this view holds edges typed here, which stay."""
        self.flush_pending()
        if self._patch_draft and self.patch_x_mm is not None and self.patch_x_mm != old:
            self.update_patch_note()
            return
        self._patch_draft = False
        self.show_patch(self.experiment_patch())
        self.plot_fault()
        self.canvas.draw_idle()
        self.update_status()

    # ------------------------------------------------------------ fault ends
    def experiment_ends_value(self):
        """The experiment's ``fault_ends_x_mm`` as stored, or None when there is none."""
        return self.experiment_value(ENDS_KEY, RECORD_ENDS_KEY)

    def experiment_ends(self) -> Optional[tuple]:
        return _patch_pair(self.experiment_ends_value())

    def show_ends(self, ends: Optional[tuple]) -> None:
        self.fault_ends = ends
        self._show(self.ends_lo_var, _text(ends[0]) if ends else "")
        self._show(self.ends_hi_var, _text(ends[1]) if ends else "")
        self.update_ends_note()

    def read_ends(self) -> Optional[tuple]:
        """The fault ends in the entries: None when both are blank, ValueError when
        unusable; an entry not edited gives the end in use at full precision."""
        lo_text, hi_text = self.ends_lo_var.get().strip(), self.ends_hi_var.get().strip()
        if not lo_text and not hi_text:
            return None
        applied = self.fault_ends
        lo = applied[0] if applied and not self._edited(self.ends_lo_var) else _parse(lo_text, "fault end from (mm)")
        hi = applied[1] if applied and not self._edited(self.ends_hi_var) else _parse(hi_text, "fault end to (mm)")
        if lo >= hi:
            raise ValueError("the fault ends must have from < to")
        return lo, hi

    def on_ends_edited(self) -> None:
        if self._syncing:
            return
        if not self._edited(self.ends_lo_var, self.ends_hi_var):
            self.update_status()
            return
        try:
            ends = self.read_ends()
        except ValueError as e:
            self.status_var.set(str(e))
            return
        for var in (self.ends_lo_var, self.ends_hi_var):
            self._shown[str(var)] = var.get()
        self.fault_ends = ends
        self._ends_draft = ends is not None
        self.update_ends_note()
        if self.profile_fit is not None:
            self.profile_text_var.set(self.profile_text())
        self.plot_fault()
        self.canvas.draw_idle()
        self.update_status()

    def update_ends_note(self) -> None:
        saved, value = self.experiment_ends(), self.experiment_ends_value()
        if self.fault_ends is None:
            note = "no fault ends: the profile is drawn whole" if saved is None else ""
        elif saved == self.fault_ends:
            note = "fault ends saved in the experiment"
        else:
            note = "fault ends not saved to the experiment"
        if saved is None and value is not None:
            note += f"; experiment fault ends unusable: {_value_text(value)}"
        self.ends_note_var.set(note)

    def save_ends(self) -> bool:
        """Store the fault ends for the whole experiment (``analysis['fault']['ends_x_mm']``)
        and show them in the other open windows that hold none typed there."""
        try:
            ends = self.read_ends()
        except ValueError as e:
            self.status_var.set(str(e))
            return False
        if ends is None:
            self.status_var.set("type both fault ends first")
            return False
        self.store_experiment_value(ENDS_KEY, RECORD_ENDS_KEY, [float(ends[0]), float(ends[1])])
        for var in (self.ends_lo_var, self.ends_hi_var):
            self._shown[str(var)] = var.get()
        self.fault_ends = ends
        self._ends_draft = False
        self.app.refresh_tree()
        self.update_ends_note()
        self.plot_fault()
        self.canvas.draw_idle()
        for view in list(getattr(self.app, "child_windows", [])):
            if view is not self and isinstance(view, SlipDistributionView) and view.winfo_exists():
                view.on_experiment_ends_saved()
        self.status_var.set("fault ends saved in the experiment")
        return True

    def on_experiment_ends_saved(self) -> None:
        self.flush_pending()
        if self._ends_draft:
            self.update_ends_note()
            return
        self.show_ends(self.experiment_ends())
        if self.profile_fit is not None:
            self.profile_text_var.set(self.profile_text())
        self.plot_fault()
        self.canvas.draw_idle()
        self.update_status()

    def save_fault(self) -> bool:
        """The Save to experiment button: the fault ends and the patch edges typed."""
        self.flush_pending()
        typed_ends = bool(self.ends_lo_var.get().strip() or self.ends_hi_var.get().strip())
        typed_patch = bool(self.patch_lo_var.get().strip() or self.patch_hi_var.get().strip())
        if not (typed_ends or typed_patch):
            self.status_var.set("type the fault ends or the patch edges first")
            return False
        ok = (not typed_ends or self.save_ends()) and (not typed_patch or self.save_patch())
        if ok and typed_ends and typed_patch:
            self.status_var.set("fault ends and patch edges saved in the experiment")
        return ok

    def end_slips(self, fit: Optional[sprof.ProfileFit] = None) -> dict:
        """``side -> (x of the fault end, slip there)`` for the fault ends that the
        fitted profile reaches (slip > 0 there: its tip lies beyond the end)."""
        fit = fit or self.profile_fit
        if fit is None or not fit.valid or self.fault_ends is None:
            return {}
        out = {}
        for (side, _), x_end in zip(SIDES, self.fault_ends):
            value = float(fit.evaluate(np.array([x_end], dtype=float))[0])
            if value > 0:
                out[side] = (float(x_end), value)
        return out

    def _draw_ends(self, ax) -> None:
        if self.fault_ends is None:
            return
        for i, x in enumerate(self.fault_ends):
            self.patch_artists.append(ax.axvline(x, color=END_COLOR, ls="-.", lw=1.0, zorder=1,
                                                 label="fault end" if i == 0 and ax is self.ax_profile else "_end"))

    def _draw_patch(self, ax) -> None:
        if self.patch_x_mm is None:
            return
        lo, hi = self.patch_x_mm
        self.patch_artists.append(ax.axvspan(lo, hi, color=PATCH_COLOR, alpha=0.12, zorder=0))
        for x in (lo, hi):
            self.patch_artists.append(ax.axvline(x, color=PATCH_COLOR, ls="--", lw=1.0, zorder=1))

    def pinned_tips(self) -> list:
        """The finite rupture tips of a valid profile fit that the sensors pin."""
        fit = self.profile_fit
        if fit is None or not fit.valid:
            return []
        brackets = tip_brackets(fit)
        lo, hi = self.fault_ends or (-math.inf, math.inf)          # a tip beyond a fault end is not a tip
        return [float(t) for (side, _), t in zip(SIDES, fit.tips_mm)
                if side not in brackets and math.isfinite(t) and lo <= t <= hi]

    def _x_range(self) -> tuple:
        """x-limits of the fault panels: the sensors with a margin, the patch, and
        the pinned tips and the ends of the free edges' brackets of a valid profile
        fit up to one array span beyond the sensors."""
        x = self.x_mm[np.isfinite(self.x_mm)]
        if x.size == 0:
            return 0.0, 1.0
        span = float(x.max() - x.min()) or 1.0
        lo, hi = min(0.0, float(x.min()) - 0.1 * span), float(x.max()) + 0.1 * span
        if self.patch_x_mm is not None:
            lo, hi = min(lo, self.patch_x_mm[0]), max(hi, self.patch_x_mm[1])
        fault_lo, fault_hi = self.fault_ends or (-math.inf, math.inf)
        ends = self.pinned_tips() + [v for bracket in tip_brackets(self.profile_fit).values() for v in bracket]
        if self.fault_ends is not None:
            ends = [min(max(v, fault_lo), fault_hi) for v in ends if math.isfinite(v)] + list(self.fault_ends)
        for tip in ends:
            if math.isfinite(tip) and float(x.min()) - span <= tip <= float(x.max()) + span:
                lo, hi = min(lo, tip - 0.05 * span), max(hi, tip + 0.05 * span)
        return lo, hi

    def _draw_brackets(self, ax, lo: float, hi: float) -> None:
        """Each free rupture edge as a bar on the zero line over the bracket it lies
        in, with ticks at its ends; an open end runs past that edge of the panel."""
        far = 10.0 * (hi - lo)                           # an open end: well past the panel, whatever the zoom
        for i, (side, (b_lo, b_hi)) in enumerate(tip_brackets(self.profile_fit).items()):
            start = b_lo if math.isfinite(b_lo) else lo - far
            stop = b_hi if math.isfinite(b_hi) else hi + far
            if self.fault_ends is not None:              # no edge beyond the fault
                start, stop = (min(max(v, self.fault_ends[0]), self.fault_ends[1]) for v in (start, stop))
            ax.plot([start, stop], [0.0, 0.0], "-", color=EDGE_COLOR, lw=6, alpha=0.35, solid_capstyle="butt",
                    zorder=2, label="rupture edge, not resolved" if i == 0 else "_edge_" + side)
            ends = [v for v in (b_lo, b_hi) if math.isfinite(v)]
            if ends:
                ax.plot(ends, [0.0] * len(ends), "|", color=EDGE_COLOR, ms=11, mew=1.5, zorder=2,
                        label="_edge_ends_" + side)

    def plot_profile(self) -> None:
        ax = self.ax_profile
        ax.clear()
        self.patch_artists = []
        self.point_artists.pop("profile", None)
        self.profile_errorbar = self.profile_curve = None
        if self.result is None:
            return
        sensors = self.result.sensors
        x = self.x_mm
        y = np.array([s.coseismic_slip for s in sensors])
        sigma = np.array([s.noise for s in sensors])
        lo, hi = self._x_range()
        self._profile_xlim = (lo, hi)
        self._draw_patch(ax)
        self._draw_ends(ax)
        ax.axhline(0.0, color="#b0b0b0", lw=0.6, zorder=1)
        self.profile_errorbar = ax.errorbar(x, y, yerr=np.where(np.isfinite(sigma), sigma, 0.0), fmt="none",
                                            ecolor=POINT_COLOR, elinewidth=1.0, capsize=3, zorder=3)
        self.point_artists["profile"], = ax.plot(x, y, "o", color=POINT_COLOR, ms=6, picker=5, zorder=4,
                                                 label="coseismic slip, error bar = noise")
        unused = ~np.asarray(self.profile_use, dtype=bool)
        if np.any(unused):
            ax.plot(x[unused], y[unused], "o", mfc="white", mec="#9a9a9a", ms=6, zorder=5, label="not fitted")
        k = self.sensor
        ax.plot([x[k]], [y[k]], "o", mfc="none", mec=TI_COLOR, mew=1.6, ms=12, zorder=6)
        fit = self.profile_fit
        if fit is not None and fit.valid:
            c_lo, c_hi = (max(lo, self.fault_ends[0]), min(hi, self.fault_ends[1])) if self.fault_ends else (lo, hi)
            xx = np.linspace(c_lo, c_hi, 600)                # cut at the fault ends
            self.profile_curve, = ax.plot(xx, fit.evaluate(xx), "-", color="k", lw=1.4, zorder=2,
                                          label=PROFILE_LABELS.get(fit.model, fit.model))
            for side, (x_end, value) in self.end_slips(fit).items():
                if lo <= x_end <= hi:
                    ax.plot([x_end], [value], "s", mfc="white", mec="k", ms=6, zorder=6)
                    ax.annotate(f"{value:.3g} {self.unit}", (x_end, value), textcoords="offset points",
                                xytext=(4 if side == "minus" else -4, 6), ha="left" if side == "minus" else "right",
                                fontsize=7)
            self._draw_brackets(ax, lo, hi)
            tips = self.pinned_tips()                    # markers only where the sensors pin a tip
            inside = [t for t in tips if lo <= t <= hi]
            if inside:
                ax.plot(inside, [0.0] * len(inside), "v", color="k", ms=6, zorder=6)
            # a tip farther out than _x_range reaches: a marker at that edge of the panel, with its x
            edge = ax.get_yaxis_transform()              # x in axes fractions, y in data
            for beyond, at, marker, ha, dx in (([t for t in tips if t < lo], 0.0, "<", "left", 3),
                                               ([t for t in tips if t > hi], 1.0, ">", "right", -3)):
                if beyond:
                    ax.plot([0.02 if at == 0.0 else 0.98], [0.0], marker, color="k", ms=6, zorder=6,
                            transform=edge)
                    ax.annotate(f"tip{'s' if len(beyond) > 1 else ''} at "
                                f"{', '.join(f'{t:.4g}' for t in beyond)} mm", (at, 0.0), xycoords=edge,
                                textcoords="offset points", xytext=(dx, 7), ha=ha, fontsize=7)
        # data limits from what is drawn now: after clear() the patch span (no y extent) comes first
        # and would leave the previous drawing's y limits in place
        ax.relim()
        ax.set_xlim(lo, hi)
        ax.set_ylabel(f"coseismic slip ({self.unit})")
        ax.set_title("slip along the fault", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=7)
        ax.tick_params(labelbottom=False)

    def plot_front(self) -> None:
        ax = self.ax_front
        ax.clear()
        self.point_artists.pop("front", None)
        if self.result is None:
            return
        sensors = self.result.sensors
        x = self.x_mm
        t_ms = np.array([1e3 * s.t_shift_s for s in sensors])
        self._draw_patch(ax)
        self._draw_ends(ax)
        self.point_artists["front"], = ax.plot(x, t_ms, "o", color=FRONT_COLOR, ms=6, picker=5, zorder=4, label="t_i")
        doubtful = np.array([doubtful_t_i(s) for s in sensors])
        if np.any(doubtful):
            ax.plot(x[doubtful], t_ms[doubtful], "s", mfc="none", mec=FLAG_COLOR, ms=9, mew=1.2, zorder=5,
                    label="doubtful t_i")
        unused = ~np.asarray(self.front_use, dtype=bool)
        if np.any(unused):
            ax.plot(x[unused], t_ms[unused], "x", color="#7a7a7a", ms=7, zorder=6, label="not fitted")
        k = self.sensor
        ax.plot([x[k]], [t_ms[k]], "o", mfc="none", mec=TI_COLOR, mew=1.6, ms=12, zorder=6)
        fit = self.front_fit
        if fit is not None and fit.valid:
            # each segment from x_from to x_to: those of a hinge both end at the apex, at t0
            for seg in fit.segments:
                xx = np.array([seg.x_from, seg.x_to])
                ax.plot(xx, 1e3 * (seg.slope_s_per_mm * xx + seg.intercept_s), "-", color="k", lw=1.2, zorder=3)
                ax.annotate(segment_speed_label(seg),
                            (xx.mean(), 1e3 * (seg.slope_s_per_mm * xx.mean() + seg.intercept_s)),
                            textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
            if math.isfinite(fit.split_x_mm) and math.isfinite(fit.t0_s):
                ax.plot([fit.split_x_mm], [1e3 * fit.t0_s], "D", mfc="none", mec="k", mew=1.4, ms=8, zorder=6,
                        label="apex")
        ax.relim()                                       # as in plot_profile
        ax.set_xlim(*self._profile_xlim)
        ax.set_xlabel("x along the fault (mm)")
        ax.set_ylabel("t_i - event time (ms)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    def plot_fault(self, reset_view: bool = False) -> None:
        """Redraw the profile and front panels.  Limits the user changed there
        (they differ from those the last redraw set: the toolbar's zoom or pan)
        are kept, the shared x and each panel's y on its own, unless
        ``reset_view``."""
        limits = (self.ax_profile.get_xlim, self.ax_profile.get_ylim, self.ax_front.get_ylim)
        setters = (self.ax_profile.set_xlim, self.ax_profile.set_ylim, self.ax_front.set_ylim)
        now = [get() for get in limits]
        auto = self._fault_auto
        kept = [None] * 3 if reset_view or auto is None else [n if n != a else None for n, a in zip(now, auto)]
        toolbar = getattr(self, "toolbar", None)
        if reset_view and toolbar is not None:
            toolbar.update()                     # the toolbar's history holds the previous event's limits
        self.plot_profile()
        self.plot_front()
        if self.result is None:
            self._fault_auto = None
            return
        self._fault_auto = [get() for get in limits]
        for value, set_lim in zip(kept, setters):
            if value is not None:
                set_lim(value)

    def home(self) -> None:
        """The toolbar's Home: the trace back to the first view in the toolbar's
        history, the fault panels to the limits the last redraw set (the history
        may hold limits from before a refit)."""
        self._toolbar_home()
        if self._fault_auto is not None:
            for value, set_lim in zip(self._fault_auto, (self.ax_profile.set_xlim, self.ax_profile.set_ylim,
                                                         self.ax_front.set_ylim)):
                set_lim(value)
            self.canvas.draw_idle()

    # ------------------------------------------------------------- plotting
    def plot(self, reset_view: bool = False, new_sensor: bool = False) -> None:
        ax, ax_rate = self.ax_trace, self.ax_rate
        keep = not reset_view and ax.has_data()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
        ax_rate.clear()
        # the rate axis sits behind the trace, so the slip, the handles and the legend draw on top
        # and the mouse events land on the trace axes; clear() resets these, so set them every time
        ax_rate.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
        ax_rate.yaxis.set_label_position("right")
        ax_rate.yaxis.tick_right()
        self.handles = {}
        self.spans = {}
        s = self.current()
        if s is None:
            self.canvas.draw_idle()
            return
        t_rel = self.time - self.t_event
        y = self.data[self.sensor]
        ti = s.t_shift_s
        (b0, b1), (a0, a1) = s.fit_before_s, s.fit_after_s
        ax.plot(t_rel, y, color="#4d4d4d", lw=0.8, label=s.channel, zorder=2)
        self.spans["before"] = ax.axvspan(b0, b1, color=BEFORE_COLOR, alpha=0.12, zorder=0)
        self.spans["after"] = ax.axvspan(a0, a1, color=AFTER_COLOR, alpha=0.12, zorder=0)
        if s.before.valid:
            tb = np.linspace(min(b0, ti), ti, 50)
            ax.plot(tb, s.before.at(tb - ti), "--", color=BEFORE_COLOR, lw=1.4, label="fit before", zorder=3)
        if s.after.valid:
            ta = np.linspace(ti, max(a1, ti), 50)
            ax.plot(ta, s.after.at(ta - ti), "--", color=AFTER_COLOR, lw=1.4, label="fit after", zorder=3)
        self.handles["t_i"] = ax.axvline(ti, color=TI_COLOR, lw=1.8, label="t_i", zorder=5)
        for name, x, color in (("before_start", b0, BEFORE_COLOR), ("before_end", b1, BEFORE_COLOR),
                               ("after_start", a0, AFTER_COLOR), ("after_end", a1, AFTER_COLOR)):
            self.handles[name] = ax.axvline(x, color=color, lw=1.2, label="_" + name, zorder=5)
        if s.before.valid and s.after.valid:
            ax.plot([ti, ti], [s.before.value, s.after.value], "-", color="k", lw=2.2, zorder=6,
                    marker="_", markersize=14, markeredgewidth=2.2,
                    label=f"slip {s.coseismic_slip:.3g} {self.unit}")
        rate = self.result.rate[self.sensor]
        ax_rate.plot(t_rel, rate, color="#9a9a9a", lw=0.6, label="slip rate (filtered)", zorder=1)
        if s.method in cs.PICK_METHODS:
            t_ref = self.result.t_ref - self.t_event
            hw = self.result.search_halfwidth_s
            ax_rate.axvspan(t_ref - hw, t_ref + hw, color="#cccccc", alpha=0.25, zorder=0)
        if s.method == cs.METHOD_ONSET and math.isfinite(s.peak_rate):
            # the threshold the rate rose through: a share of the peak of the max-rate pick
            seg = rate[np.abs(t_rel - (self.result.t_ref - self.t_event)) <= self.result.search_halfwidth_s]
            peak = float(np.nanmax(seg)) if seg.size and np.any(np.isfinite(seg)) else float("nan")
            if math.isfinite(peak):
                ax_rate.axhline(self.result.onset_fraction * peak, color=TI_COLOR, lw=0.8, ls=":", zorder=1,
                                label=f"onset threshold ({self.result.onset_fraction:.0%} of the peak)")
        if s.method == cs.METHOD_STEP and s.before.valid and s.after.valid:
            ts = np.linspace(b1, a0, 50)
            level = s.before.at(ts - ti) + self.result.step_fraction * (s.after.at(ts - ti) - s.before.at(ts - ti))
            ax.plot(ts, level, ":", color=TI_COLOR, lw=1.0, zorder=3,
                    label=f"step level ({self.result.step_fraction:.0%})")
        ax_rate.set_ylabel(f"slip rate ({self.unit}/s)")
        ax.set_xlabel(f"time - event time {self.t_event:.4f} s (s)")
        ax.set_ylabel(f"{self.field} ({self.unit})")
        flags = f"  [{'; '.join(s.flags)}]" if s.flags else ""
        ax.set_title(f"{self.figure_title()}  {self.sensor_label(self.sensor)}  t_i from {METHOD_LABELS[s.method]}"
                     f"{flags}", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_rate.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
        if keep:
            ax.set_xlim(xlim)
            if new_sensor:
                self._fit_ylim()
            else:
                ax.set_ylim(ylim)
        else:
            ax.set_xlim(float(t_rel[0]), float(t_rel[-1]))
        self._fit_rate_ylim()
        self.plot_fault(reset_view=reset_view)
        self.canvas.draw_idle()

    def _visible(self) -> np.ndarray:
        lo, hi = self.ax_trace.get_xlim()
        t_rel = self.time - self.t_event
        return (t_rel >= lo) & (t_rel <= hi)

    def _fit_ylim(self) -> None:
        if self.result is None:
            return
        y = self.data[self.sensor][self._visible()]
        y = y[np.isfinite(y)]
        if y.size and y.max() > y.min():
            pad = 0.05 * (y.max() - y.min())
            self.ax_trace.set_ylim(y.min() - pad, y.max() + pad)

    def _fit_rate_ylim(self) -> None:
        if self.result is None:
            return
        r = self.result.rate[self.sensor][self._visible()]
        r = r[np.isfinite(r)]
        if r.size and r.max() > r.min():
            pad = 0.05 * (r.max() - r.min())
            self.ax_rate.set_ylim(r.min() - pad, r.max() + pad)

    def zoom_to_event_time(self) -> None:
        s = self.current()
        if s is None:
            return
        self.ax_trace.set_xlim(s.t_shift_s - ZOOM_HALF_S, s.t_shift_s + ZOOM_HALF_S)
        self._fit_ylim()
        self._fit_rate_ylim()
        self.canvas.draw_idle()

    def show_whole_window(self) -> None:
        if self.result is None:
            return
        t_rel = self.time - self.t_event
        self.ax_trace.set_xlim(float(t_rel[0]), float(t_rel[-1]))
        self._fit_ylim()
        self._fit_rate_ylim()
        self.canvas.draw_idle()

    # ------------------------------------------------------------- dragging
    def _in_trace(self, event) -> bool:
        return event.inaxes in (self.ax_trace, self.ax_rate) and event.xdata is not None

    def handle_at(self, event) -> Optional[str]:
        """The handle (t_i or a window edge) within ``HANDLE_PICK_PX`` of the
        press, the nearest one; t_i wins a tie."""
        best, best_d = None, HANDLE_PICK_PX + 1e-9
        for name in HANDLE_NAMES:
            line = self.handles.get(name)
            if line is None:
                continue
            x = float(line.get_xdata()[0])
            px = self.ax_trace.transData.transform((x, 0.0))[0]
            d = abs(px - float(event.x))
            if d < best_d:
                best, best_d = name, d
        return best

    def on_pick(self, event) -> None:
        """A left click on a point of the profile or front panel selects its
        sensor; the wheel or another button over a point does nothing."""
        mouse = getattr(event, "mouseevent", None)
        if mouse is None or mouse.name != "button_press_event" or mouse.button != MouseButton.LEFT:
            return
        if self.toolbar_active() or self._drag is not None or self.result is None:
            return
        ind = getattr(event, "ind", None)
        if event.artist in self.point_artists.values() and ind is not None and len(ind):
            self.set_sensor(int(ind[0]))

    def on_press(self, event) -> None:
        """Grab ``t_i`` or a window edge near the press, else a fit window by its
        body.  The handles are hit-tested here, in pixels, so that the nearest
        one wins and ``t_i`` wins a tie."""
        self.flush_pending()
        if self.toolbar_active() or self._drag is not None or self.current() is None or not self._in_trace(event):
            return
        if event.button != 1:
            return
        x_data = float(self.ax_trace.transData.inverted().transform((event.x, event.y))[0])
        name = self.handle_at(event)
        if name is not None:
            self.begin_drag(name, x_data)
            return
        s = self.current()
        if s.fit_before_s[0] <= x_data <= s.fit_before_s[1]:
            self.begin_drag("before", x_data)
        elif s.fit_after_s[0] <= x_data <= s.fit_after_s[1]:
            self.begin_drag("after", x_data)

    def begin_drag(self, what: str, x: float) -> None:
        """Start from the view's own values, the ones the handles show: after a
        recompute that failed they differ from the last result's."""
        i = self.sensor
        ti = self.t_shift[i] if math.isfinite(self.t_shift[i]) else self.current().t_shift_s
        self._drag = {"what": what, "x0": float(x), "t_i": float(ti),
                      "before": tuple(self.fit_before[i]), "after": tuple(self.fit_after[i]), "moved": False}

    def on_motion(self, event) -> None:
        if self._drag is None or not self._in_trace(event):
            return
        self.drag_to(float(self.ax_trace.transData.inverted().transform((event.x, event.y))[0]))

    def drag_to(self, x: float) -> dict:
        """Move the grabbed handle (or window) to ``x`` (s from the event time);
        the entries and the handles follow, the analysis waits for the release.
        ``t_i`` moves alone and stays between the windows; the windows stay off
        this sensor's ``t_i``."""
        d = self._drag
        dt = cs.sampling_interval(self.time)
        ti = d["t_i"]
        (b0, b1), (a0, a1) = d["before"], d["after"]
        what = d["what"]
        if what == "t_i":
            ti = min(max(x, b1), a0)
        elif what == "before_start":
            b0 = min(x, b1 - dt)
        elif what == "before_end":
            b1 = min(max(x, b0 + dt), ti)
        elif what == "after_start":
            a0 = max(min(x, a1 - dt), ti)
        elif what == "after_end":
            a1 = max(x, a0 + dt)
        elif what == "before":
            shift = min(x - d["x0"], ti - b1)
            b0, b1 = b0 + shift, b1 + shift
        elif what == "after":
            shift = max(x - d["x0"], ti - a0)
            a0, a1 = a0 + shift, a1 + shift
        state = {"t_i": ti, "before": (b0, b1), "after": (a0, a1)}
        d["state"] = state
        d["moved"] = True
        self._syncing = True
        try:
            self._show(self.t_shift_var, _text(round(ti * 1e3, 6)))
            self._show(self.before_start_var, _text(round(b0, 6)))
            self._show(self.before_end_var, _text(round(b1, 6)))
            self._show(self.after_start_var, _text(round(a0, 6)))
            self._show(self.after_end_var, _text(round(a1, 6)))
        finally:
            self._syncing = False
        positions = {"t_i": ti, "before_start": b0, "before_end": b1, "after_start": a0, "after_end": a1}
        for name, line in self.handles.items():
            line.set_xdata([positions[name], positions[name]])
        for name, (lo, hi) in (("before", (b0, b1)), ("after", (a0, a1))):
            patch = self.spans.get(name)
            if patch is not None:
                patch.set_x(lo)
                patch.set_width(hi - lo)
        self.canvas.draw_idle()
        return state

    def on_release(self, event) -> None:
        if self._drag is not None:
            self.end_drag()

    def end_drag(self) -> None:
        """The dragged values become the sensor's settings: recompute."""
        d, self._drag = self._drag, None
        if d is None or not d.get("moved"):
            return
        state = d["state"]
        if d["what"] == "t_i":
            self.set_event_time(self.sensor, state["t_i"])
        else:
            self.set_windows(state["before"], state["after"])

    # ----------------------------------------------------------------- save
    def record(self) -> Optional[dict]:
        """The ``slip_analysis`` record of the current state, or None."""
        if self.result is None:
            return None
        rec = self.result.as_dict()
        rec.update({"field": self.field, "unit": self.unit, "same_windows": bool(self.same_windows_var.get())})
        try:
            rec["rate_filter_tabs"] = self.filter_windows()
            rec["shared_set_s"] = _parse_blank(self.shared_shift_var.get(), "shared t_i (ms)") * 1e-3
        except ValueError:
            pass
        rec[PATCH_KEY] = [float(v) for v in self.patch_x_mm] if self.patch_x_mm else [float("nan")] * 2
        if self.profile_fit is not None:
            profile = self.profile_fit.as_dict()
            profile["use"] = [bool(v) for v in self.profile_use]
            profile["settings"] = self.profile_settings()
            reached = self.end_slips()
            profile["end_slip_um"] = [reached[side][1] if side in reached else float("nan") for side, _ in SIDES]
            rec["profile"] = profile
        rec[RECORD_ENDS_KEY] = [float(v) for v in self.fault_ends] if self.fault_ends else [float("nan")] * 2
        if self.front_fit is not None:
            front = self.front_fit.as_dict()
            front["use"] = [bool(v) for v in self.front_use]
            front["use_auto"] = bool(self.front_use_auto)
            rec["front"] = front
        return rec

    def entry_error(self) -> Optional[str]:
        """Why an entry holds a value that cannot be applied, or None."""
        try:
            self.read_settings()
            _parse(self.t_shift_var.get(), "t_i - event time (ms)")
            self.read_windows()
            self.read_patch()
            self.read_ends()
            self.read_profile_constraints()
        except ValueError as e:
            return str(e)
        return None

    def save(self) -> bool:
        """Store the record; refused while an entry holds a value that was not
        applied, or when there is no result for this event."""
        self.flush_pending()
        error = self.entry_error()
        if error is not None:
            self.status_var.set(f"not saved: {error}")
            return False
        rec = self.record()
        if rec is None or self.result.event_time != self.t_event:
            self.status_var.set("nothing to save - the current values give no result")
            return False
        self.save_results(rec)
        self.saved = rec
        self.update_status()
        return True

    def matches_saved(self) -> bool:
        """True when the inputs equal those of the saved record."""
        rec = self.record()
        if rec is None or not isinstance(self.saved, dict):
            return False
        inputs = cs.restore_inputs(self.saved, self.channels)
        if inputs is None or str(self.saved.get("field", self.field)) != self.field:
            return False
        same = (inputs["methods"] == rec["time_method"]
                and np.allclose(inputs["fit_before_s"], rec["fit_before_s"])
                and np.allclose(inputs["fit_after_s"], rec["fit_after_s"])
                and inputs["rate_filter"] == rec["rate_filter"]
                and math.isclose(inputs["rate_cutoff_hz"], rec["rate_cutoff_hz"])
                and inputs["rate_lowpass_order"] == rec["rate_lowpass_order"]
                and inputs["rate_lowpass_type"] == rec["rate_lowpass_type"]
                and inputs["rate_savgol_order"] == rec["rate_savgol_order"]
                and math.isclose(inputs["rate_smoothing_s"], rec["rate_smoothing_s"])
                and math.isclose(inputs["search_halfwidth_s"], rec["search_halfwidth_s"])
                and math.isclose(inputs["rate_snr"], rec["rate_snr"])
                and math.isclose(inputs["reference_halfwidth_s"], rec["reference_halfwidth_s"])
                and math.isclose(inputs["onset_fraction"], rec["onset_fraction"])
                and math.isclose(inputs["step_fraction"], rec["step_fraction"])
                and (math.isnan(inputs["shared_shift_s"]) == math.isnan(rec["shared_shift_s"]))
                and (math.isnan(rec["shared_shift_s"]) or math.isclose(inputs["shared_shift_s"], rec["shared_shift_s"]))
                and bool(np.asarray(self.saved.get("same_windows", True)).ravel()[0]) == rec["same_windows"])
        manual = [i for i, m in enumerate(rec["time_method"]) if m == cs.METHOD_MANUAL]
        same = same and all(math.isclose(inputs["t_shift_s"][i], rec["t_shift_s"][i], abs_tol=1e-12) for i in manual)
        return bool(same and self._fit_settings_match(rec))

    def _fit_settings_match(self, rec: dict) -> bool:
        """Profile model, weighting and sensors, front mode and sensors, and patch
        edges equal those of the saved record (a record without them counts as
        the defaults).  The patch edges count only while the experiment has
        none: otherwise its edges are the ones every event opens with."""
        n = len(self.channels)
        saved = self.saved
        profile = saved.get("profile") if isinstance(saved.get("profile"), dict) else {}
        front = saved.get("front") if isinstance(saved.get("front"), dict) else {}
        if str(profile.get("model", DEFAULT_PROFILE_MODEL)) != rec.get("profile", {}).get("model", DEFAULT_PROFILE_MODEL):
            return False
        if bool(np.asarray(profile.get("weighted", False)).ravel()[0]) != bool(rec["profile"]["weighted"]):
            return False
        if _bools(profile.get("use"), n, True) != rec["profile"]["use"]:
            return False
        if str(front.get("mode", DEFAULT_FRONT_MODE)) != rec["front"]["mode"]:
            return False
        saved_auto = bool(np.asarray(front.get("use_auto", True)).ravel()[0]) if front else True
        if saved_auto != rec["front"]["use_auto"]:
            return False
        if not saved_auto and _bools(front.get("use"), n, True) != rec["front"]["use"]:
            return False
        if not self._profile_settings_match(profile.get("settings"), rec["profile"]["settings"],
                                            rec["profile"]["model"]):
            return False
        if self.experiment_ends() is None and _patch_pair(saved.get(RECORD_ENDS_KEY)) != _patch_pair(rec[RECORD_ENDS_KEY]):
            return False
        if self.experiment_patch() is not None:
            return True
        return _patch_pair(saved.get(PATCH_KEY)) == _patch_pair(rec[PATCH_KEY])

    @staticmethod
    def _profile_settings_match(saved, current: dict, model: str) -> bool:
        """The holds and bounds of the model fitted equal the saved ones (none saved:
        everything fitted within the automatic bounds)."""
        saved = saved.get(model) if isinstance(saved, dict) and isinstance(saved.get(model), dict) else {}

        def same(a, b) -> bool:
            return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)

        for name, cur in current[model].items():
            old = saved.get(name) if isinstance(saved.get(name), dict) else {}
            held = _float_or_nan(old.get("value"))
            fit = bool(np.asarray(old.get("fit", True)).ravel()[0]) or not math.isfinite(held)
            if fit != cur["fit"] or (not fit and not same(held, cur["value"])):
                return False
            if not all(same(_float_or_nan(old.get(key)), cur[key]) for key in ("min", "max")):
                return False
        return True

    def update_status(self) -> None:
        if self.result is None:
            return
        flagged = [s for s in self.result.sensors if s.flags]
        manual = sum(1 for s in self.result.sensors if s.method == cs.METHOD_MANUAL)
        reference = f"reference at {1e3 * (self.result.t_ref - self.t_event):+.1f} ms"
        shared = [s for s in self.result.sensors if s.method == cs.METHOD_SHARED]
        if shared:
            how = "the reference" if math.isnan(self.result.shared_shift_s) else "set"
            parts = [f"{len(self.result.sensors)} sensors at one t_i, {1e3 * shared[0].t_shift_s:+.1f} ms ({how}); "
                     + reference]
        else:
            parts = [f"{len(self.result.sensors)} sensors, {manual} manual; " + reference]
        if flagged:
            parts.append("check: " + ", ".join(f"{s.channel} ({'; '.join(s.flags)})" for s in flagged))
        error = self.entry_error()
        if error is not None:
            parts.append(f"entry not applied: {error}")
        elif self.saved is None:
            parts.append("not saved")
        else:
            parts.append("matches the saved record" if self.matches_saved() else "differs from the saved record (Save to update)")
        self.status_var.set(" | ".join(parts))
