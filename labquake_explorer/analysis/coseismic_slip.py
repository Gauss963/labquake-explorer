"""Coseismic slip per sensor, each sensor with its own event time.

A rupture reaches the slip sensors one after another.  One event time and one
pair of fit windows for every sensor cuts the coseismic step short far from
the nucleation and mixes afterslip in near it, so every sensor ``i`` gets its
own event time ``t_i``, chosen by one of two methods:

* ``max_rate``: the time of the largest filtered slip rate.  The rate is the
  time derivative of the slip, filtered by ``rate_filter`` (:func:`slip_rate`):
  ``lowpass``, a zero-phase Butterworth or Bessel low-pass of the slip at
  ``rate_cutoff_hz`` (order ``rate_lowpass_order``) before the derivative (the
  default); ``savgol``, the Savitzky-Golay derivative (polynomial order
  ``rate_savgol_order``) over ``rate_smoothing_s``; ``boxcar``, the derivative
  averaged over ``rate_smoothing_s``.  A reference time ``t_ref`` comes
  first, at the peak of the mean rate of all sensors within
  ``event_time +- reference_halfwidth_s`` (narrow enough to keep the
  neighbouring events out); each sensor is then searched within
  ``t_ref +- search_halfwidth_s``.
* ``onset``: the slip-rate onset, where the filtered rate, followed back from
  the max-rate pick, last rose through ``onset_fraction`` of that peak
  (interpolated between samples).
* ``step``: where the slip crosses ``step_fraction`` of the way from the
  before-window line to the after-window line, between the two windows and
  nearest the max-rate pick (interpolated between samples).
* ``shared``: one time for every sensor, ``event_time + shared_shift_s``, or
  ``t_ref`` when ``shared_shift_s`` is NaN.
* ``manual``: a time set by the user, stored as ``t_shift_s`` relative to the
  event time (``t_i = event_time + t_shift_s``).

The coseismic slip of a sensor is the difference at ``t_i`` between two
straight lines: one fitted to the slip over ``fit_before_s`` and one over
``fit_after_s``, both windows given in seconds relative to the event time, so
that moving ``t_i`` leaves them in place.  The default windows come from the
event's extraction window ``[start_s, end_s]`` (``start_s < 0 < end_s``):
before from 75 % to 25 % of the negative side, after from 25 % to 75 % of the
positive side.  The noise of a sensor is the
standard deviation of the residuals of its before-window line, the noise level
of one sample (the error bar the Slip Distribution view draws).

Flags tell the user which sensors to check: ``pick at search edge`` and
``weak rate peak`` (peak below ``rate_snr`` times the rate noise, the
standard deviation of the smoothed rate in the before window) for automatic
picks, ``no rate peak`` when the search found no finite rate,
``before window clipped`` / ``after window clipped`` when a window reaches past
the record, ``before window has gaps`` / ``after window has gaps`` when NaN
samples leave less than half of a window, ``before window reaches t_i`` /
``after window reaches t_i`` when ``t_i`` lies inside a window (the line then
takes in the slip it should measure), and ``before fit invalid`` /
``after fit invalid`` when a window holds fewer than ``MIN_FIT_SAMPLES``
samples.

Everything here is pure numerics; :class:`SlipAnalysis.as_dict` gives the
JSON-like record the Slip Distribution view stores as ``event['slip_analysis']``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import bessel, butter, savgol_filter, sosfiltfilt

RESULT_VERSION = 2                    # 2: fit windows relative to the event time (1: to t_i)
METHOD_MAX_RATE = "max_rate"
METHOD_MANUAL = "manual"
METHOD_ONSET = "onset"
METHOD_STEP = "step"
METHOD_SHARED = "shared"
METHOD_BURST = "burst"                # one t_i for all: the PZT envelope peak (or the peak slip rate)
PICK_METHODS = (METHOD_MAX_RATE, METHOD_ONSET, METHOD_STEP)    # each sensor from its own record
METHODS = PICK_METHODS + (METHOD_SHARED, METHOD_BURST, METHOD_MANUAL)
WINDOW_MODES = ("lines", "levels")   # slip from the lines at t_i, or from mean levels in narrow windows
BURST_SOURCES = ("pzt_envelope", "peak_rate")
DEFAULT_BURST_SOURCE = "pzt_envelope"
DEFAULT_BURST_HALFWIDTH_S = 1e-3      # levels: the windows sit this far on either side of t_i ...
DEFAULT_LEVEL_WIDTH_S = 1e-4          # ... and are this wide
DEFAULT_BURST_SEARCH_S = 3e-3         # the envelope peak is searched within t_ref +- this
BURST_ENVELOPE_S = 2e-5               # the PZT envelope: |pzt| averaged over this
DEFAULT_ONSET_FRACTION = 0.2          # onset: the rate rises through this fraction of its peak
DEFAULT_STEP_FRACTION = 0.5           # step: the slip crosses this fraction of the step

RATE_FILTERS = ("lowpass", "savgol", "boxcar")
DEFAULT_RATE_FILTER = "lowpass"
DEFAULT_RATE_CUTOFF_HZ = 50.0         # lowpass: corner frequency of the slip's low-pass
DEFAULT_RATE_SMOOTHING_S = 0.010      # savgol and boxcar: the window
LOWPASS_TYPES = ("butterworth", "bessel")
DEFAULT_LOWPASS_TYPE = "butterworth"
DEFAULT_LOWPASS_ORDER = 4             # run forwards and backwards: zero phase, -6 dB at the cutoff
MAX_FILTER_ORDER = 10
DEFAULT_SAVGOL_ORDER = 2
LEGACY_RATE_FILTER = "boxcar"         # records without rate_filter were made with it
DEFAULT_SEARCH_HALFWIDTH_S = 0.05     # each sensor is searched within t_ref +- this
DEFAULT_RATE_SNR = 5.0                # flag a pick whose peak rate is below this x the rate noise
DEFAULT_REFERENCE_HALFWIDTH_S = 0.5   # t_ref is searched within event_time +- this
FIT_FRACTIONS = (0.25, 0.75)          # default fit windows: 75 %..25 % before, 25 %..75 % after
MIN_FIT_SAMPLES = 3
EDGE_TOLERANCE = 1e-6                 # of a sample: a window edge on a sample includes it

FLAG_EDGE = "pick at search edge"
FLAG_WEAK = "weak rate peak"
FLAG_NO_PEAK = "no rate peak"
FLAG_NO_CROSSING = "no crossing found"
FLAG_BEFORE_CLIPPED = "before window clipped"
FLAG_AFTER_CLIPPED = "after window clipped"
FLAG_BEFORE_INVALID = "before fit invalid"
FLAG_AFTER_INVALID = "after fit invalid"
FLAG_BEFORE_GAPS = "before window has gaps"
FLAG_AFTER_GAPS = "after window has gaps"
FLAG_BEFORE_OVERLAP = "before window reaches t_i"
FLAG_NO_PZT = "no PZT trace: burst at the peak slip rate"
FLAG_AFTER_OVERLAP = "after window reaches t_i"
GAP_FRACTION = 0.5                    # flag a window whose finite samples are fewer than this share


# ---------------------------------------------------------------------- helpers
def sampling_interval(time) -> float:
    """Median sample spacing of a time axis (s)."""
    t = np.asarray(time, dtype=float).ravel()
    if t.size < 2:
        raise ValueError("a time axis needs at least two samples")
    dt = float(np.median(np.diff(t)))
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError("the time axis must increase")
    return dt


def smoothing_samples(smoothing_s: float, dt: float) -> int:
    """Odd boxcar length (samples) for a smoothing time; 1 means no smoothing."""
    k = int(round(max(0.0, float(smoothing_s)) / dt))
    k = max(1, k)
    return k if k % 2 == 1 else k + 1


def _fill_gaps(t: np.ndarray, d: np.ndarray) -> np.ndarray:
    """``d`` with its NaN samples interpolated linearly along each row (a row
    without two finite samples stays as it is)."""
    out = np.array(d, dtype=float, copy=True)
    for row in out.reshape(-1, t.size):
        finite = np.isfinite(row)
        if 2 <= np.count_nonzero(finite) < row.size:
            row[~finite] = np.interp(t[~finite], t[finite], row[finite])
    return out


def _filter_order(value, low: int, name: str) -> int:
    """A filter order as an int in ``[low, MAX_FILTER_ORDER]`` (ValueError otherwise)."""
    try:
        order = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number") from None
    if order != value or not low <= order <= MAX_FILTER_ORDER:
        raise ValueError(f"{name} must be a whole number from {low} to {MAX_FILTER_ORDER}")
    return order


def slip_rate(time, data, smoothing_s: float = DEFAULT_RATE_SMOOTHING_S, rate_filter: str = "boxcar",
              cutoff_hz: float = DEFAULT_RATE_CUTOFF_HZ, lowpass_order: int = DEFAULT_LOWPASS_ORDER,
              lowpass_type: str = DEFAULT_LOWPASS_TYPE, savgol_order: int = DEFAULT_SAVGOL_ORDER) -> np.ndarray:
    """Filtered time derivative of ``data`` (one row per sensor, or one row);
    same shape as ``data``, unit/s.

    ``rate_filter``: ``boxcar`` averages the derivative over ``smoothing_s``
    (NaN samples left out of each mean, so a bad sample only blanks its
    neighbourhood); ``savgol`` is the Savitzky-Golay derivative of polynomial
    order ``savgol_order`` over ``smoothing_s``; ``lowpass`` filters the slip
    with a ``lowpass_type`` (Butterworth or Bessel, the Bessel normalised to
    -3 dB at the cutoff) low-pass of order ``lowpass_order`` at ``cutoff_hz``,
    run forwards and backwards (zero phase, -6 dB at the cutoff), and then
    differentiates.  The last two bridge NaN gaps for the filter and blank them
    again afterwards."""
    t = np.asarray(time, dtype=float).ravel()
    d = np.asarray(data, dtype=float)
    if d.shape[-1] != t.size:
        raise ValueError("data and time must have the same number of samples")
    dt = sampling_interval(t)
    if rate_filter not in RATE_FILTERS:
        raise ValueError(f"unknown rate filter {rate_filter!r}; use one of {RATE_FILTERS}")
    if rate_filter == "lowpass":
        cutoff_hz = float(cutoff_hz)
        nyquist = 0.5 / dt
        if not (math.isfinite(cutoff_hz) and 0 < cutoff_hz < nyquist):
            raise ValueError(f"the cutoff must be between 0 and half the sampling rate ({nyquist:g} Hz)")
        order = _filter_order(lowpass_order, 1, "the low-pass order")
        if lowpass_type not in LOWPASS_TYPES:
            raise ValueError(f"unknown low-pass type {lowpass_type!r}; use one of {LOWPASS_TYPES}")
        if lowpass_type == "bessel":
            sos = bessel(order, cutoff_hz, fs=1.0 / dt, output="sos", norm="mag")
        else:
            sos = butter(order, cutoff_hz, fs=1.0 / dt, output="sos")
        filled = _fill_gaps(t, d)
        finite = np.isfinite(filled).all(axis=-1, keepdims=True)
        smooth = sosfiltfilt(sos, np.where(finite, filled, 0.0), axis=-1) if t.size > 3 * (2 * order + 1) \
            else filled
        rate = np.gradient(np.where(finite, smooth, np.nan), t, axis=-1)
        return np.where(np.isfinite(d), rate, np.nan)
    if rate_filter == "savgol":
        order = _filter_order(savgol_order, 1, "the Savitzky-Golay order")
        k = smoothing_samples(smoothing_s, dt)
        if k < order + 2:
            raise ValueError(f"the Savitzky-Golay window must span at least {order + 2} samples "
                             f"({(order + 2) * dt * 1e3:g} ms) for order {order}")
        if k > t.size:
            return np.gradient(d, t, axis=-1)
        filled = _fill_gaps(t, d)
        rate = savgol_filter(filled, k, order, deriv=1, delta=dt, axis=-1, mode="interp")
        return np.where(np.isfinite(d), rate, np.nan)
    rate = np.gradient(d, t, axis=-1)
    k = smoothing_samples(smoothing_s, dt)
    if k > 1:
        # mean of the finite samples in each window: a NaN only blanks its neighbourhood, and at
        # the ends the window holds only the samples there are (padding with copies of the end
        # sample would weight its one-sided, noisiest difference up to half the window)
        finite = np.isfinite(rate)
        total = uniform_filter1d(np.where(finite, rate, 0.0), size=k, axis=-1, mode="constant")
        count = uniform_filter1d(finite.astype(float), size=k, axis=-1, mode="constant")
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(count > 0.5 / k, total / np.maximum(count, 1e-300), np.nan)
    return rate


def default_fit_windows(start_s: float, end_s: float, fractions=FIT_FRACTIONS) -> tuple[tuple, tuple]:
    """``(before, after)`` windows relative to the event time from an extraction window
    ``[start_s, end_s]``: before = (hi * start_s, lo * start_s), after =
    (lo * end_s, hi * end_s) with ``fractions = (lo, hi)``."""
    start_s, end_s = float(start_s), float(end_s)
    lo, hi = (float(f) for f in fractions)
    if not (start_s < 0.0 < end_s):
        raise ValueError("the extraction window must have start < 0 < end")
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError("fractions must satisfy 0 <= lo < hi <= 1")
    return (hi * start_s, lo * start_s), (lo * end_s, hi * end_s)


def check_window(window, name: str = "window") -> tuple[float, float]:
    """A (start, end) pair as floats with start < end (ValueError otherwise)."""
    try:
        a, b = (float(v) for v in window)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a pair of numbers") from None
    if not (math.isfinite(a) and math.isfinite(b)) or a >= b:
        raise ValueError(f"{name} must have start < end")
    return a, b


def reference_time(time, rate, event_time: float,
                   halfwidth_s: float = DEFAULT_REFERENCE_HALFWIDTH_S) -> float:
    """Time of the peak of the mean rate of all sensors within
    ``event_time +- halfwidth_s``; the event time itself when that part of the
    record holds no finite rate."""
    t = np.asarray(time, dtype=float).ravel()
    r = np.asarray(rate, dtype=float)
    if r.ndim == 1:
        mean = r
    else:
        finite = np.isfinite(r)
        count = finite.sum(axis=0)
        total = np.where(finite, r, 0.0).sum(axis=0)
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    inside = (np.abs(t - event_time) <= halfwidth_s) & np.isfinite(mean)
    if not np.any(inside):
        return float(event_time)
    idx = np.flatnonzero(inside)
    return float(t[idx[int(np.argmax(mean[idx]))]])


def burst_time(time, pzt, t_ref: float, search_s: float = DEFAULT_BURST_SEARCH_S,
               envelope_s: float = BURST_ENVELOPE_S) -> Optional[float]:
    """The peak of the PZT envelope within ``t_ref +- search_s``: the largest
    ``|pzt|`` over the channels, averaged over ``envelope_s``.  None without
    PZT samples there (``pzt`` None, empty, or all NaN in the search)."""
    if pzt is None:
        return None
    t = np.asarray(time, dtype=float).ravel()
    p = np.asarray(pzt, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    if p.ndim != 2 or p.shape[1] != t.size or p.shape[0] == 0:
        return None
    with np.errstate(invalid="ignore"):
        env = np.nanmax(np.abs(p), axis=0) if p.shape[0] > 1 else np.abs(p[0])
    env = np.where(np.isfinite(env), env, 0.0)
    n = smoothing_samples(float(envelope_s), sampling_interval(t))
    if n > 1:
        env = uniform_filter1d(env, n, mode="nearest")
    inside = (t >= t_ref - search_s) & (t <= t_ref + search_s) & (env > 0)
    if not np.any(inside):
        return None
    return float(t[np.flatnonzero(inside)[np.argmax(env[inside])]])


@dataclass(frozen=True)
class RatePick:
    time: float          # absolute time of the peak
    at_edge: bool        # the peak sits on the first or last sample searched
    peak_rate: float     # smoothed rate at the peak (unit/s)


def pick_max_rate(time, rate_row, t_ref: float, halfwidth_s: float) -> Optional[RatePick]:
    """Largest smoothed rate within ``t_ref +- halfwidth_s``; None when no finite rate."""
    t = np.asarray(time, dtype=float).ravel()
    r = np.asarray(rate_row, dtype=float).ravel()
    idx = np.flatnonzero(np.abs(t - float(t_ref)) <= float(halfwidth_s))
    if idx.size == 0:
        return None
    values = r[idx]
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    j = int(np.argmax(np.where(finite, values, -np.inf)))
    at_edge = j in (0, idx.size - 1)          # a single searched sample is at both edges
    return RatePick(float(t[idx[j]]), bool(at_edge), float(values[j]))


def _crossing(t, g, i: int, j: int) -> float:
    """The time between samples ``i`` and ``j`` where ``g`` crosses zero, linearly."""
    if g[j] == g[i]:
        return float(t[i])
    return float(t[i] + (t[j] - t[i]) * g[i] / (g[i] - g[j]))


def onset_time(time, rate_row, pick: RatePick, t_ref: float, halfwidth_s: float,
               fraction: float = DEFAULT_ONSET_FRACTION) -> Optional[float]:
    """Where the rate, followed back from the max-rate ``pick``, last rose through
    ``fraction`` of the peak, between samples; None when it stays above it back to
    the start of the search window ``t_ref +- halfwidth_s``."""
    t = np.asarray(time, dtype=float).ravel()
    r = np.asarray(rate_row, dtype=float).ravel()
    k = int(np.argmin(np.abs(t - pick.time)))
    first = int(np.searchsorted(t, float(t_ref) - float(halfwidth_s), side="left"))
    g = r - float(fraction) * pick.peak_rate
    below = np.flatnonzero((g[first:k] < 0) | ~np.isfinite(g[first:k]))
    if below.size == 0:
        return None
    j = first + int(below[-1])
    if not np.isfinite(g[j]):
        return float(t[j + 1])
    return _crossing(t, g, j, j + 1)


def step_time(time, values, before, after, t_i0: float, lo: float, hi: float,
              fraction: float = DEFAULT_STEP_FRACTION) -> Optional[float]:
    """Where the slip crosses ``fraction`` of the way from the ``before`` line to the
    ``after`` line (:class:`LineFit`, both about ``t_i0``), the crossing nearest
    ``t_i0`` within ``[lo, hi]`` (the gap between the fit windows, where the step
    is), between samples; None without one."""
    if not (before.valid and after.valid):
        return None
    t = np.asarray(time, dtype=float).ravel()
    y = np.asarray(values, dtype=float).ravel()
    sel = np.flatnonzero((t >= float(lo)) & (t <= float(hi)))
    if sel.size < 2:
        return None
    tt = t[sel]
    level = before.at(tt - t_i0) + float(fraction) * (after.at(tt - t_i0) - before.at(tt - t_i0))
    g = y[sel] - level
    ok = np.isfinite(g[:-1]) & np.isfinite(g[1:]) & (np.sign(g[:-1]) != np.sign(g[1:]))
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return None
    times = np.array([_crossing(tt, g, i, i + 1) for i in idx])
    return float(times[int(np.argmin(np.abs(times - t_i0)))])


@dataclass(frozen=True)
class LineFit:
    """``y = slope * (t - t_i) + value`` over one window.  ``window`` and ``used``
    are relative to the origin the window was given against (the event time in
    :func:`analyze_slip`); :meth:`at` takes times relative to ``t_i``."""
    slope: float                     # unit/s
    value: float                     # the line at t_i
    n: int                           # samples used
    rms: float                       # residual standard deviation (ddof = 2)
    window: tuple                    # requested window, relative to the origin
    used: tuple                      # span of the samples used, relative to the origin
    clipped: bool                    # the requested window reaches past the record
    valid: bool
    gaps: bool = False               # NaN samples leave less than GAP_FRACTION of the window

    def at(self, t_rel) -> np.ndarray:
        return self.slope * np.asarray(t_rel, dtype=float) + self.value


def window_mask(time, t_i: float, window, dt: Optional[float] = None) -> np.ndarray:
    """Samples inside ``[t_i + window[0], t_i + window[1]]``; an edge that falls on
    a sample (to ``EDGE_TOLERANCE`` of a sample) includes it."""
    t = np.asarray(time, dtype=float).ravel()
    a, b = window
    eps = EDGE_TOLERANCE * (sampling_interval(t) if dt is None else dt)
    return (t >= t_i + a - eps) & (t <= t_i + b + eps)


def fit_line(time, values, t_i: float, window, origin: Optional[float] = None) -> LineFit:
    """Least-squares line over ``[origin + window[0], origin + window[1]]``
    (``origin`` defaults to ``t_i``), as ``slope * (t - t_i) + value``."""
    a, b = check_window(window)
    origin = float(t_i if origin is None else origin)
    t = np.asarray(time, dtype=float).ravel()
    y = np.asarray(values, dtype=float).ravel()
    dt = sampling_interval(t)
    clipped = bool(origin + a < t[0] - 0.5 * dt or origin + b > t[-1] + 0.5 * dt)
    in_window = window_mask(t, origin, (a, b), dt)
    sel = in_window & np.isfinite(y)
    n = int(np.count_nonzero(sel))
    gaps = bool(n < GAP_FRACTION * np.count_nonzero(in_window))
    nan = float("nan")
    if n < MIN_FIT_SAMPLES:
        return LineFit(nan, nan, n, nan, (a, b), (nan, nan), clipped, False, gaps)
    tt = t[sel] - t_i
    yy = y[sel]
    slope, value = np.polyfit(tt, yy, 1)
    resid = yy - (slope * tt + value)
    rms = float(np.sqrt(np.sum(resid ** 2) / (n - 2))) if n > 2 else nan
    used = (float(tt[0] + t_i - origin), float(tt[-1] + t_i - origin))
    return LineFit(float(slope), float(value), n, rms, (a, b), used, clipped, True, gaps)


# ------------------------------------------------------------------ per sensor
@dataclass
class SensorSlip:
    channel: str
    x_mm: float
    method: str
    t_shift_s: float                 # t_i - event_time
    before: LineFit
    after: LineFit
    peak_rate: float                 # smoothed rate at t_i (unit/s)
    rate_noise: float                # std of the smoothed rate in the before window
    flags: list = field(default_factory=list)
    window_mode: str = "lines"       # 'levels': the slip is after_level - before_level
    before_level: float = float("nan")   # mean slip in the before window ('levels' only)
    after_level: float = float("nan")

    @property
    def coseismic_slip(self) -> float:
        if self.window_mode == "levels":
            if math.isfinite(self.before_level) and math.isfinite(self.after_level):
                return float(self.after_level - self.before_level)
            return float("nan")
        if self.before.valid and self.after.valid:
            return float(self.after.value - self.before.value)
        return float("nan")

    @property
    def noise(self) -> float:
        return float(self.before.rms)

    @property
    def fit_before_s(self) -> tuple:
        return self.before.window

    @property
    def fit_after_s(self) -> tuple:
        return self.after.window


@dataclass
class SlipAnalysis:
    event_time: float
    start_s: float
    end_s: float
    rate_smoothing_s: float
    search_halfwidth_s: float
    rate_snr: float
    reference_halfwidth_s: float
    t_ref: float                     # absolute
    sensors: list
    time: np.ndarray = field(repr=False, default=None)
    rate: np.ndarray = field(repr=False, default=None)
    rate_filter: str = DEFAULT_RATE_FILTER
    rate_cutoff_hz: float = DEFAULT_RATE_CUTOFF_HZ
    rate_lowpass_order: int = DEFAULT_LOWPASS_ORDER
    rate_lowpass_type: str = DEFAULT_LOWPASS_TYPE
    rate_savgol_order: int = DEFAULT_SAVGOL_ORDER
    onset_fraction: float = DEFAULT_ONSET_FRACTION
    step_fraction: float = DEFAULT_STEP_FRACTION
    shared_shift_s: float = float("nan")      # NaN: the shared t_i is t_ref
    window_mode: str = "lines"
    burst_halfwidth_s: float = DEFAULT_BURST_HALFWIDTH_S
    level_width_s: float = DEFAULT_LEVEL_WIDTH_S
    burst_search_s: float = DEFAULT_BURST_SEARCH_S
    burst_source: str = DEFAULT_BURST_SOURCE
    t_burst: float = float("nan")             # absolute; NaN when no sensor used the burst

    @property
    def t_i(self) -> np.ndarray:
        return np.array([self.event_time + s.t_shift_s for s in self.sensors])

    def as_dict(self) -> dict:
        """The JSON-like part stored in ``event['slip_analysis']`` (no arrays of samples)."""
        s = self.sensors
        return {
            "version": RESULT_VERSION,
            "event_time": float(self.event_time),
            "start_s": float(self.start_s),
            "end_s": float(self.end_s),
            "channels": [x.channel for x in s],
            "x_mm": [float(x.x_mm) for x in s],
            "rate_filter": self.rate_filter,
            "rate_cutoff_hz": float(self.rate_cutoff_hz),
            "rate_lowpass_order": int(self.rate_lowpass_order),
            "rate_lowpass_type": self.rate_lowpass_type,
            "rate_savgol_order": int(self.rate_savgol_order),
            "rate_smoothing_s": float(self.rate_smoothing_s),
            "search_halfwidth_s": float(self.search_halfwidth_s),
            "rate_snr": float(self.rate_snr),
            "reference_halfwidth_s": float(self.reference_halfwidth_s),
            "onset_fraction": float(self.onset_fraction),
            "step_fraction": float(self.step_fraction),
            "shared_shift_s": float(self.shared_shift_s),
            "t_ref_shift_s": float(self.t_ref - self.event_time),
            "window_mode": self.window_mode,
            "burst_halfwidth_s": float(self.burst_halfwidth_s),
            "level_width_s": float(self.level_width_s),
            "burst_search_s": float(self.burst_search_s),
            "burst_source": self.burst_source,
            "t_burst_shift_s": float(self.t_burst - self.event_time) if math.isfinite(self.t_burst) else float("nan"),
            "before_level_um": [float(x.before_level) for x in s],
            "after_level_um": [float(x.after_level) for x in s],
            "rate_noise_um_s": [float(x.rate_noise) for x in s],
            "time_method": [x.method for x in s],
            "t_shift_s": [float(x.t_shift_s) for x in s],
            "fit_before_s": [[float(v) for v in x.fit_before_s] for x in s],
            "fit_after_s": [[float(v) for v in x.fit_after_s] for x in s],
            "coseismic_slip_um": [x.coseismic_slip for x in s],
            "noise_um": [x.noise for x in s],
            "before_line": [[x.before.slope, x.before.value] for x in s],
            "after_line": [[x.after.slope, x.after.value] for x in s],
            "peak_rate_um_s": [float(x.peak_rate) for x in s],
            "flags": ["; ".join(x.flags) for x in s],
        }


def _per_sensor(value, n: int, name: str, default):
    """``value`` as a list of length ``n``: None -> default for all, one item for
    all (a pair counts as one item for windows), or one per sensor."""
    if value is None:
        return [default] * n
    if name.startswith("fit_"):
        arr = np.asarray(value, dtype=float)
        if arr.shape == (2,):
            return [tuple(arr.tolist())] * n
        if arr.shape == (n, 2):
            return [tuple(row) for row in arr.tolist()]
        raise ValueError(f"{name} must be one (start, end) pair or one pair per sensor")
    if isinstance(value, str) or np.ndim(value) == 0:
        return [value] * n
    values = list(value)
    if len(values) == 1 and n != 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"{name} needs one value per sensor ({n}), got {len(values)}")
    return values


def analyze_slip(time, data, event_time: float, start_s: float, end_s: float, *,
                 channels: Optional[Sequence[str]] = None, x_mm: Optional[Sequence[float]] = None,
                 methods: Optional[Sequence[str]] = None, t_shift_s: Optional[Sequence[float]] = None,
                 fit_before_s=None, fit_after_s=None,
                 rate_filter: str = DEFAULT_RATE_FILTER,
                 rate_cutoff_hz: float = DEFAULT_RATE_CUTOFF_HZ,
                 rate_lowpass_order: int = DEFAULT_LOWPASS_ORDER,
                 rate_lowpass_type: str = DEFAULT_LOWPASS_TYPE,
                 rate_savgol_order: int = DEFAULT_SAVGOL_ORDER,
                 rate_smoothing_s: float = DEFAULT_RATE_SMOOTHING_S,
                 search_halfwidth_s: float = DEFAULT_SEARCH_HALFWIDTH_S,
                 rate_snr: float = DEFAULT_RATE_SNR,
                 reference_halfwidth_s: float = DEFAULT_REFERENCE_HALFWIDTH_S,
                 onset_fraction: float = DEFAULT_ONSET_FRACTION,
                 step_fraction: float = DEFAULT_STEP_FRACTION,
                 shared_shift_s: float = float("nan"),
                 pzt=None, window_mode: str = "lines",
                 burst_halfwidth_s: float = DEFAULT_BURST_HALFWIDTH_S,
                 level_width_s: float = DEFAULT_LEVEL_WIDTH_S,
                 burst_search_s: float = DEFAULT_BURST_SEARCH_S,
                 burst_source: str = DEFAULT_BURST_SOURCE) -> SlipAnalysis:
    """Event time, line fits and coseismic slip of every sensor.

    A ``burst`` sensor takes one t_i for all: the peak of the PZT envelope
    (``pzt``, ``(n_pzt, n)`` on ``time``) within ``t_ref +- burst_search_s``
    when ``burst_source`` is ``'pzt_envelope'`` and PZT traces are given,
    else the peak of the mean slip rate ``t_ref`` (flagged when PZT traces
    were asked for).  ``window_mode='levels'`` replaces the fit windows by
    ``[t_i - w - d, t_i - w]`` and ``[t_i + w, t_i + w + d]`` with
    ``w = burst_halfwidth_s`` and ``d = level_width_s``, and the coseismic slip
    is the difference of the mean slip in them (no line extrapolation): the
    slip accrued on the time scale of the burst, for a full-rate record.

    ``data`` is ``(n_sensors, n)`` on ``time``.  ``methods`` (one per sensor or
    one for all, default ``max_rate``; see :data:`METHODS`) chooses how each
    ``t_i`` is found; a ``manual`` sensor takes ``t_shift_s`` (relative to
    ``event_time``) and falls back to ``max_rate`` when that shift is missing or
    not finite; an ``onset`` or ``step`` sensor without a crossing keeps its
    max-rate pick and is flagged.
    ``fit_before_s`` / ``fit_after_s`` are one pair for every sensor or one
    pair per sensor, relative to ``event_time``; by default they follow from
    ``[start_s, end_s]`` (:func:`default_fit_windows`).  ``t_ref`` is searched
    within ``event_time +- reference_halfwidth_s``.
    """
    t = np.asarray(time, dtype=float).ravel()
    sampling_interval(t)                      # at least two increasing samples
    d = np.asarray(data, dtype=float)
    if d.ndim == 1:
        d = d[None, :]
    if d.ndim != 2 or d.shape[1] != t.size:
        raise ValueError("data must be (n_sensors, n) on the time axis")
    n_ch = d.shape[0]
    event_time = float(event_time)
    if not math.isfinite(event_time):
        raise ValueError("event_time must be finite")
    rate_smoothing_s = float(rate_smoothing_s)
    search_halfwidth_s = float(search_halfwidth_s)
    rate_snr = float(rate_snr)
    reference_halfwidth_s = float(reference_halfwidth_s)
    if not (math.isfinite(reference_halfwidth_s) and reference_halfwidth_s > 0):
        raise ValueError("the reference half-width must be > 0")
    if not (math.isfinite(rate_smoothing_s) and rate_smoothing_s >= 0):
        raise ValueError("the rate filter window must be >= 0")
    rate_filter = str(rate_filter)
    rate_cutoff_hz = float(rate_cutoff_hz)
    if not (math.isfinite(search_halfwidth_s) and search_halfwidth_s > 0):
        raise ValueError("the search half-width must be > 0")
    if not math.isfinite(rate_snr) or rate_snr < 0:
        raise ValueError("the rate SNR threshold must be >= 0")
    onset_fraction, step_fraction, shared_shift_s = float(onset_fraction), float(step_fraction), float(shared_shift_s)
    if not 0.0 < onset_fraction < 1.0:
        raise ValueError("the onset threshold must be between 0 and 100 % of the peak rate")
    if not 0.0 < step_fraction < 1.0:
        raise ValueError("the step level must be between 0 and 100 % of the step")
    if math.isinf(shared_shift_s):
        raise ValueError("the shared t_i must be finite")
    window_mode, burst_source = str(window_mode), str(burst_source)
    if window_mode not in WINDOW_MODES:
        raise ValueError(f"window_mode must be one of {WINDOW_MODES}, got {window_mode!r}")
    if burst_source not in BURST_SOURCES:
        raise ValueError(f"burst_source must be one of {BURST_SOURCES}, got {burst_source!r}")
    burst_halfwidth_s, level_width_s, burst_search_s = float(burst_halfwidth_s), float(level_width_s), float(burst_search_s)
    if not (math.isfinite(burst_halfwidth_s) and burst_halfwidth_s >= 0):
        raise ValueError("the burst half-width must be >= 0")
    if not (math.isfinite(level_width_s) and level_width_s > 0):
        raise ValueError("the level width must be > 0")
    if not (math.isfinite(burst_search_s) and burst_search_s > 0):
        raise ValueError("the burst search half-width must be > 0")
    default_before, default_after = default_fit_windows(start_s, end_s)
    channels = list(channels) if channels is not None else [str(i) for i in range(n_ch)]
    x_mm = [float(v) for v in x_mm] if x_mm is not None else [float("nan")] * n_ch
    if len(channels) != n_ch or len(x_mm) != n_ch:
        raise ValueError("channels and x_mm need one entry per sensor")
    methods = [str(m) for m in _per_sensor(methods, n_ch, "methods", METHOD_MAX_RATE)]
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}; use one of {METHODS}")
    shifts = _per_sensor(t_shift_s, n_ch, "t_shift_s", float("nan"))
    befores = [check_window(w, "fit_before_s") for w in _per_sensor(fit_before_s, n_ch, "fit_before_s", default_before)]
    afters = [check_window(w, "fit_after_s") for w in _per_sensor(fit_after_s, n_ch, "fit_after_s", default_after)]

    rate = slip_rate(t, d, rate_smoothing_s, rate_filter, rate_cutoff_hz, rate_lowpass_order, rate_lowpass_type,
                     rate_savgol_order)
    t_ref = reference_time(t, rate, event_time, reference_halfwidth_s)
    t_burst, burst_flag = float("nan"), False
    if METHOD_BURST in methods:
        found = burst_time(t, pzt, t_ref, burst_search_s) if burst_source == "pzt_envelope" else None
        burst_flag = burst_source == "pzt_envelope" and found is None
        t_burst = t_ref if found is None else found
    sensors = []
    for i in range(n_ch):
        flags: list[str] = []
        method = methods[i]
        try:
            shift = float(shifts[i])
        except (TypeError, ValueError):
            shift = float("nan")
        pick = None
        if method == METHOD_MANUAL and math.isfinite(shift):
            t_i = event_time + shift
        elif method == METHOD_SHARED:
            t_i = event_time + shared_shift_s if math.isfinite(shared_shift_s) else t_ref
        elif method == METHOD_BURST:
            t_i = t_burst
            if burst_flag:
                flags.append(FLAG_NO_PZT)
        else:
            if method == METHOD_MANUAL:
                method = METHOD_MAX_RATE
            pick = pick_max_rate(t, rate[i], t_ref, search_halfwidth_s)
            if pick is None:
                t_i = t_ref
                flags.append(FLAG_NO_PEAK)
            else:
                t_i = pick.time
                if pick.at_edge:
                    flags.append(FLAG_EDGE)
                crossing = None
                if method == METHOD_ONSET:
                    crossing = onset_time(t, rate[i], pick, t_ref, search_halfwidth_s, onset_fraction)
                elif method == METHOD_STEP:
                    crossing = step_time(t, d[i], fit_line(t, d[i], t_i, befores[i], event_time),
                                         fit_line(t, d[i], t_i, afters[i], event_time), t_i,
                                         event_time + befores[i][1], event_time + afters[i][0], step_fraction)
                if crossing is not None:
                    t_i = crossing
                elif method != METHOD_MAX_RATE:
                    flags.append(FLAG_NO_CROSSING)
        before_level = after_level = float("nan")
        if window_mode == "levels":                   # narrow windows on either side of t_i, from t_i itself
            rel = t_i - event_time
            befores[i] = (rel - burst_halfwidth_s - level_width_s, rel - burst_halfwidth_s)
            afters[i] = (rel + burst_halfwidth_s, rel + burst_halfwidth_s + level_width_s)
            for window, side in ((befores[i], "before"), (afters[i], "after")):
                values = d[i][window_mask(t, event_time, window)]
                values = values[np.isfinite(values)]
                level = float(np.mean(values)) if values.size else float("nan")
                if side == "before":
                    before_level = level
                else:
                    after_level = level
        before = fit_line(t, d[i], t_i, befores[i], event_time)
        after = fit_line(t, d[i], t_i, afters[i], event_time)
        in_before = window_mask(t, event_time, befores[i]) & np.isfinite(rate[i])
        rate_noise = float(np.std(rate[i][in_before])) if np.count_nonzero(in_before) >= MIN_FIT_SAMPLES else float("nan")
        peak_rate = float(np.interp(t_i, t, rate[i])) if np.all(np.isfinite(rate[i])) else (
            pick.peak_rate if pick is not None else float("nan"))
        if pick is not None and math.isfinite(rate_noise) and rate_noise > 0 and pick.peak_rate < rate_snr * rate_noise:
            flags.append(FLAG_WEAK)
        if before.clipped:
            flags.append(FLAG_BEFORE_CLIPPED)
        if after.clipped:
            flags.append(FLAG_AFTER_CLIPPED)
        if event_time + befores[i][1] > t_i:
            flags.append(FLAG_BEFORE_OVERLAP)
        if event_time + afters[i][0] < t_i:
            flags.append(FLAG_AFTER_OVERLAP)
        if before.gaps:
            flags.append(FLAG_BEFORE_GAPS)
        if after.gaps:
            flags.append(FLAG_AFTER_GAPS)
        if not before.valid:
            flags.append(FLAG_BEFORE_INVALID)
        if not after.valid:
            flags.append(FLAG_AFTER_INVALID)
        sensors.append(SensorSlip(str(channels[i]), float(x_mm[i]), method, float(t_i - event_time),
                                  before, after, peak_rate, rate_noise, flags, window_mode,
                                  before_level, after_level))
    return SlipAnalysis(event_time, float(start_s), float(end_s), rate_smoothing_s, search_halfwidth_s,
                        rate_snr, reference_halfwidth_s, float(t_ref), sensors, time=t, rate=rate,
                        rate_filter=rate_filter, rate_cutoff_hz=rate_cutoff_hz,
                        rate_lowpass_order=int(rate_lowpass_order), rate_lowpass_type=str(rate_lowpass_type),
                        rate_savgol_order=int(rate_savgol_order), onset_fraction=onset_fraction,
                        step_fraction=step_fraction, shared_shift_s=shared_shift_s,
                        window_mode=window_mode, burst_halfwidth_s=burst_halfwidth_s, level_width_s=level_width_s,
                        burst_search_s=burst_search_s, burst_source=burst_source, t_burst=t_burst)


# --------------------------------------------------------------- saved records
def _text_value(value) -> str:
    """A saved text, also when an HDF5 round trip made it bytes or a 0-d array."""
    value = np.asarray(value).ravel()[0] if np.ndim(value) else value
    return value.decode() if isinstance(value, bytes) else str(value)


def restore_inputs(record, channels: Sequence[str]) -> Optional[dict]:
    """The analysis inputs of a saved ``slip_analysis`` record for the given
    channels, or None when the record does not match them or is unreadable.

    Keys: ``methods``, ``t_shift_s``, ``fit_before_s``, ``fit_after_s``,
    ``rate_filter`` (``boxcar`` for records made before there was a choice),
    ``rate_cutoff_hz``, ``rate_lowpass_order``, ``rate_lowpass_type``,
    ``rate_savgol_order``, ``rate_smoothing_s``, ``search_halfwidth_s``, ``rate_snr``,
    ``reference_halfwidth_s``.  Arrays that
    an HDF5 round trip turned into numpy arrays are accepted.
    """
    if not isinstance(record, dict):
        return None
    try:
        saved_channels = [c.decode() if isinstance(c, bytes) else str(c) for c in np.asarray(record["channels"]).ravel()]
        if saved_channels != [str(c) for c in channels]:
            return None
        n = len(saved_channels)
        methods = [m.decode() if isinstance(m, bytes) else str(m) for m in np.asarray(record["time_method"]).ravel()]
        shifts = [float(v) for v in np.asarray(record["t_shift_s"], dtype=float).ravel()]
        before = np.asarray(record["fit_before_s"], dtype=float).reshape(n, 2)
        after = np.asarray(record["fit_after_s"], dtype=float).reshape(n, 2)
        if len(methods) != n or len(shifts) != n or any(m not in METHODS for m in methods):
            return None
        if int(np.asarray(record.get("version", 1)).ravel()[0]) < 2:
            # version 1 gave the windows relative to t_i: move them to the event time
            shift = np.nan_to_num(np.asarray(shifts, dtype=float))[:, None]
            before, after = before + shift, after + shift
        out = {
            "methods": methods,
            "t_shift_s": shifts,
            "fit_before_s": [tuple(r) for r in before.tolist()],
            "fit_after_s": [tuple(r) for r in after.tolist()],
            "rate_filter": _text_value(record.get("rate_filter", LEGACY_RATE_FILTER)),
            "rate_cutoff_hz": float(record.get("rate_cutoff_hz", DEFAULT_RATE_CUTOFF_HZ)),
            "rate_lowpass_order": int(np.asarray(record.get("rate_lowpass_order", DEFAULT_LOWPASS_ORDER)).ravel()[0]),
            "rate_lowpass_type": _text_value(record.get("rate_lowpass_type", DEFAULT_LOWPASS_TYPE)),
            "rate_savgol_order": int(np.asarray(record.get("rate_savgol_order", DEFAULT_SAVGOL_ORDER)).ravel()[0]),
            "rate_smoothing_s": float(record.get("rate_smoothing_s", DEFAULT_RATE_SMOOTHING_S)),
            "search_halfwidth_s": float(record.get("search_halfwidth_s", DEFAULT_SEARCH_HALFWIDTH_S)),
            "rate_snr": float(record.get("rate_snr", DEFAULT_RATE_SNR)),
            "reference_halfwidth_s": float(record.get("reference_halfwidth_s", DEFAULT_REFERENCE_HALFWIDTH_S)),
            "onset_fraction": float(record.get("onset_fraction", DEFAULT_ONSET_FRACTION)),
            "step_fraction": float(record.get("step_fraction", DEFAULT_STEP_FRACTION)),
            "shared_shift_s": float(record.get("shared_shift_s", float("nan"))),
            "window_mode": _text_value(record.get("window_mode", "lines")),
            "burst_halfwidth_s": float(record.get("burst_halfwidth_s", DEFAULT_BURST_HALFWIDTH_S)),
            "level_width_s": float(record.get("level_width_s", DEFAULT_LEVEL_WIDTH_S)),
            "burst_search_s": float(record.get("burst_search_s", DEFAULT_BURST_SEARCH_S)),
            "burst_source": _text_value(record.get("burst_source", DEFAULT_BURST_SOURCE)),
        }
        for w in out["fit_before_s"] + out["fit_after_s"]:
            check_window(w)
        if out["rate_filter"] not in RATE_FILTERS or out["rate_lowpass_type"] not in LOWPASS_TYPES:
            return None
        if out["window_mode"] not in WINDOW_MODES or out["burst_source"] not in BURST_SOURCES:
            return None
        return out
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


__all__ = [
    "RESULT_VERSION", "METHODS", "PICK_METHODS", "METHOD_MAX_RATE", "METHOD_ONSET", "METHOD_STEP", "METHOD_SHARED",
    "METHOD_BURST", "WINDOW_MODES", "BURST_SOURCES", "DEFAULT_BURST_SOURCE", "DEFAULT_BURST_HALFWIDTH_S",
    "DEFAULT_LEVEL_WIDTH_S", "DEFAULT_BURST_SEARCH_S", "FLAG_NO_PZT", "burst_time",
    "METHOD_MANUAL", "DEFAULT_ONSET_FRACTION", "DEFAULT_STEP_FRACTION", "onset_time", "step_time",
    "RATE_FILTERS", "DEFAULT_RATE_FILTER", "DEFAULT_RATE_CUTOFF_HZ", "LOWPASS_TYPES", "DEFAULT_LOWPASS_TYPE",
    "DEFAULT_LOWPASS_ORDER", "DEFAULT_SAVGOL_ORDER", "MAX_FILTER_ORDER",
    "DEFAULT_RATE_SMOOTHING_S", "DEFAULT_SEARCH_HALFWIDTH_S", "DEFAULT_RATE_SNR", "DEFAULT_REFERENCE_HALFWIDTH_S",
    "FIT_FRACTIONS", "MIN_FIT_SAMPLES", "sampling_interval", "slip_rate", "default_fit_windows",
    "check_window", "window_mask", "reference_time", "RatePick", "pick_max_rate", "LineFit", "fit_line", "SensorSlip",
    "SlipAnalysis", "analyze_slip", "restore_inputs",
]
