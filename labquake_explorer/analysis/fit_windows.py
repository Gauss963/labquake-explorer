"""Automatic fit windows for the coseismic slip of :mod:`coseismic_slip`.

The coseismic slip of a sensor is the after-window line minus the
before-window line at its event time ``t_i``.  Far from the asperity the slip
keeps rising after the event (afterslip), the after line then depends on
where its window sits, and the windows had to be placed by hand, sensor by
sensor.  This module places them from the record itself, each sensor on its
own, so that the same rule gives the same windows for every event.  It only
chooses the windows: the slip is still measured by
:func:`coseismic_slip.analyze_slip`, unchanged.

Both windows are ``width_s`` long and sit as close to the coseismic rise as
the record allows:

* before: the window ends ``before_gap_s`` ahead of the onset, where the slip
  rate, followed back from ``t_i``, last rose through ``onset_fraction`` of
  the event's peak rate (the largest from ``t_i`` to ``t_i + min_start_s``,
  so the rule does not depend on the t_i method); the gap keeps the precursory slip out of the line.  It
  starts no earlier than the previous event, and is shortened (and flagged)
  when it does not fit.  The slope of its line is the sensor's background
  (pre-event) slip rate.
* after: the after line, extrapolated back to ``t_i``, is computed for window
  starts ``t_i + a``, ``a`` from ``min_start_s`` to ``max_start_s`` in steps
  of ``step_s``, every window ending before the next event and the end of the
  record.  Where that value levels off (``plateau``: it varies by no more
  than ``plateau_tol_um`` over ``plateau_span_s`` of starts, and the stretch
  ends at least another ``plateau_span_s`` before the last start), the window
  starts at the first ``a`` of the stretch: the earliest window whose line no
  longer depends on where it sits.  Without one (``decay``: the value keeps
  rising with ``a``, which is common far from the asperity) the window starts
  where the slip has settled back towards steady sliding: the filtered slip
  rate, averaged over ``rate_smoothing_s`` and followed on from ``t_i``,
  first falls to ``background_factor`` times the background rate, no earlier
  than ``t_i + min_after_s``.  The background, unlike the peak rate, does not
  grow with the size of the step, so the same factor serves sensors near and
  far from the asperity.  When the rate has not fallen that far by
  ``t_i + max_decay_s`` (or there is no positive background rate) the window
  starts there, a provisional choice, and the sensor is flagged for a check
  by hand.  A decay window that would run past the next event or the end of
  the record is moved earlier, to the latest start that fits, and flagged.

A sensor whose background rate is more than ``creep_ratio`` of its peak slip
rate creeps nearly as fast before the event as it slips during it: its
coseismic step does not stand out from the creep, the windows (whichever
rule placed them) decide much of its slip, and it is flagged for a check by
hand.  A slip that comes out negative with the placed windows (``FLAG_NEGATIVE``,
added by the caller, which measures the slip) is flagged too: the fault does
not slip backwards in an event, so it marks windows the trace does not suit.

Times are relative to the event time, as ``fit_before_s`` / ``fit_after_s``
are in a ``slip_analysis`` record.  The defaults were set against windows
placed by hand in experiment t0234 (runs 0 to 2, 352 sensors).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d

from labquake_explorer.analysis.coseismic_slip import fit_line, sampling_interval

KIND_PLATEAU = "plateau"              # the after line levels off: the earliest stable window
KIND_DECAY = "decay"                  # no plateau: the window starts where the slip rate has settled
KINDS = (KIND_PLATEAU, KIND_DECAY)

DEFAULT_WIDTH_S = 0.6                 # both windows
DEFAULT_MIN_START_S = 0.1             # after-window starts searched from t_i + this ...
DEFAULT_MAX_START_S = 1.2             # ... to t_i + this
DEFAULT_STEP_S = 0.02                 # ... in steps of this
DEFAULT_PLATEAU_TOL_UM = 0.5          # plateau: the after line at t_i varies by no more than this ...
DEFAULT_PLATEAU_SPAN_S = 0.2          # ... over this span of window starts
DEFAULT_BACKGROUND_FACTOR = 5.0       # no plateau: the rate falls to this many times the background ...
DEFAULT_RATE_SMOOTHING_S = 0.05       # ... averaged over this ...
DEFAULT_MIN_AFTER_S = 0.2             # ... the window starting no earlier than t_i + this ...
DEFAULT_MAX_DECAY_S = 0.8             # ... and no later than t_i + this (provisional, flagged)
DEFAULT_CREEP_RATIO = 0.1             # flag a background rate above this fraction of the peak rate
DEFAULT_ONSET_FRACTION = 0.1         # before: the onset, where the rate rises through this fraction
DEFAULT_BEFORE_GAP_S = 0.4            # before: the window ends this far ahead of the onset
DEFAULT_ONSET_SEARCH_S = 1.0          # before: the onset is searched back to t_i - this
DEFAULT_EVENT_MARGIN_S = 0.1          # no window comes closer than this to a neighbouring event
MIN_WIDTH_FRACTION = 0.5              # a before window shortened below this fraction of width_s is not used

FLAG_NO_DECAY = "auto: slip rate stays high, check the after window"
FLAG_NO_BACKGROUND = "auto: no background slip rate, check the after window"
FLAG_CREEP = "auto: creep close to the coseismic rate, check the windows"
FLAG_NEGATIVE = "auto: negative slip, check the windows"   # set by the caller from the slip measured
FLAG_NO_ROOM_AFTER = "auto: no room for the after window"
FLAG_AFTER_MOVED = "auto: after window moved earlier to fit, check the after window"
FLAG_NO_ONSET = "auto: no onset found"
FLAG_BEFORE_SHORT = "auto: before window shortened"
FLAG_NO_ROOM_BEFORE = "auto: no room for the before window"
FLAG_NO_RATE = "auto: no slip rate at t_i"


@dataclass
class AutoWindows:
    """The windows chosen for one sensor, relative to the event time; a window
    is None when none could be placed (the caller keeps its own)."""
    before: Optional[tuple]
    after: Optional[tuple]
    kind: str = ""                    # KIND_PLATEAU or KIND_DECAY ("" without an after window)
    onset_s: float = float("nan")     # the onset, relative to the event time
    background_rate: float = float("nan")   # slope of the before line (unit/s)
    creep_ratio: float = float("nan")       # background_rate / the event's peak slip rate (peak_rate)
    flags: list = field(default_factory=list)


def after_line_values(time, values, t_i: float, starts, width_s: float) -> np.ndarray:
    """The after line at ``t_i`` for windows ``[t_i + a, t_i + a + width_s]``,
    one per start ``a`` (NaN where the window holds too few samples)."""
    out = np.full(len(starts), np.nan)
    for j, a in enumerate(starts):
        fit = fit_line(time, values, t_i, (a, a + width_s))
        if fit.valid:
            out[j] = fit.value
    return out


def plateau_start(values_at, step_s: float, tol_um: float = DEFAULT_PLATEAU_TOL_UM,
                  span_s: float = DEFAULT_PLATEAU_SPAN_S) -> Optional[int]:
    """Index of the first start from which the after-line values stay within
    ``tol_um`` of each other over ``span_s`` of starts, the stretch ending at
    least another ``span_s`` before the last start (values that level off only
    at the end of the search are a later slip entering the windows, not a
    plateau); None without one."""
    v = np.asarray(values_at, dtype=float)
    n = max(2, int(round(float(span_s) / float(step_s))) + 1)
    for j in range(v.size - 2 * n + 1):
        part = v[j:j + n]
        if np.all(np.isfinite(part)) and float(np.ptp(part)) <= tol_um:
            return j
    return None


def settle_time(time, rate_row, t_i: float, threshold: float,
                smoothing_s: float = DEFAULT_RATE_SMOOTHING_S) -> Optional[float]:
    """Time after ``t_i`` (relative to it) where the slip rate, averaged over
    ``smoothing_s`` and followed on from ``t_i``, first falls to ``threshold``
    or below; None when it does not (NaN samples count as not below)."""
    t = np.asarray(time, dtype=float)
    r = np.asarray(rate_row, dtype=float)
    n = max(1, int(round(float(smoothing_s) / sampling_interval(t))))
    finite = np.isfinite(r)
    if n > 1:
        total = uniform_filter1d(np.where(finite, r, 0.0), n, mode="nearest")
        count = uniform_filter1d(finite.astype(float), n, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(count > 0.5, total / np.maximum(count, 1e-300), np.nan)
    k = int(np.searchsorted(t, t_i, side="left"))
    below = np.flatnonzero(r[k:] <= threshold)
    return float(t[k + below[0]] - t_i) if below.size else None


def peak_rate(time, rate_row, t_i: float, span_s: float = DEFAULT_MIN_START_S) -> float:
    """The event's peak slip rate: the largest finite rate from ``t_i`` to
    ``t_i + span_s`` (an onset or step pick sits below the peak, a max-rate
    pick on it); NaN when there is none.

    shortcut: an onset pick more than ``span_s`` (0.1 s) ahead of the peak (a
    rise slower than about 70 ms) misses the peak; widen the span if such
    events need auto windows."""
    t = np.asarray(time, dtype=float)
    r = np.asarray(rate_row, dtype=float)
    k0 = int(np.argmin(np.abs(t - t_i)))
    k1 = max(k0 + 1, int(np.searchsorted(t, t_i + span_s, side="right")))
    part = r[k0:k1]
    part = part[np.isfinite(part)]
    return float(part.max()) if part.size else float("nan")


def choose_before(time, rate_row, t_i: float, *, limit_s: float,
                  width_s: float = DEFAULT_WIDTH_S, onset_fraction: float = DEFAULT_ONSET_FRACTION,
                  before_gap_s: float = DEFAULT_BEFORE_GAP_S,
                  onset_search_s: float = DEFAULT_ONSET_SEARCH_S,
                  peak_span_s: float = DEFAULT_MIN_START_S) -> tuple:
    """``(window, onset, flags)``: the before window ``(start, end)`` and the
    onset (where the rate last rose through ``onset_fraction`` of
    :func:`peak_rate`), both relative to ``t_i``; the window is None when less
    than ``MIN_WIDTH_FRACTION`` of it fits after ``t_i + limit_s`` (``limit_s`` < 0)."""
    flags: list[str] = []
    k = int(np.argmin(np.abs(time - t_i)))
    peak = peak_rate(time, rate_row, t_i, peak_span_s)
    onset = 0.0
    if math.isfinite(peak) and peak > 0:
        first = int(np.searchsorted(time, t_i - onset_search_s, side="left"))
        below = np.flatnonzero(rate_row[first:k] < onset_fraction * peak)
        if below.size:
            onset = float(time[first + below[-1]] - t_i)
        else:
            flags.append(FLAG_NO_ONSET)
    else:
        flags.append(FLAG_NO_RATE)
    end = onset - before_gap_s
    start = max(end - width_s, limit_s)
    if end - start < MIN_WIDTH_FRACTION * width_s:
        return None, onset, flags + [FLAG_NO_ROOM_BEFORE]
    if end - start < width_s - 1e-9:
        flags.append(FLAG_BEFORE_SHORT)
    return (float(start), float(end)), onset, flags


def choose_after(time, values, rate_row, t_i: float, background_rate: float, *, limit_s: float,
                 width_s: float = DEFAULT_WIDTH_S, min_start_s: float = DEFAULT_MIN_START_S,
                 max_start_s: float = DEFAULT_MAX_START_S, step_s: float = DEFAULT_STEP_S,
                 plateau_tol_um: float = DEFAULT_PLATEAU_TOL_UM, plateau_span_s: float = DEFAULT_PLATEAU_SPAN_S,
                 background_factor: float = DEFAULT_BACKGROUND_FACTOR,
                 rate_smoothing_s: float = DEFAULT_RATE_SMOOTHING_S, min_after_s: float = DEFAULT_MIN_AFTER_S,
                 max_decay_s: float = DEFAULT_MAX_DECAY_S) -> tuple:
    """``(a, kind, flags)``: the after window starts at ``t_i + a``; ``a`` is
    None when no window fits before ``t_i + limit_s`` (``time`` and ``t_i``
    absolute, ``limit_s`` relative to ``t_i``)."""
    starts = np.arange(min_start_s, max_start_s + 0.5 * step_s, step_s)
    starts = starts[starts + width_s <= limit_s + 1e-9]
    if starts.size < 2:
        return None, "", [FLAG_NO_ROOM_AFTER]
    first = plateau_start(after_line_values(time, values, t_i, starts, width_s), step_s, plateau_tol_um,
                          plateau_span_s)
    if first is not None:
        return float(starts[first]), KIND_PLATEAU, []
    flags: list[str] = []
    a = None
    if math.isfinite(background_rate) and background_rate > 0:
        a = settle_time(time, rate_row, t_i, background_factor * background_rate, rate_smoothing_s)
        if a is None or a > max_decay_s:
            a = None
            flags.append(FLAG_NO_DECAY)
    else:
        flags.append(FLAG_NO_BACKGROUND)
    a = float(max_decay_s) if a is None else max(a, float(min_after_s))
    if a + width_s > limit_s + 1e-9:
        a = float(starts[-1])                # the latest window that still ends before the limit
        flags.append(FLAG_AFTER_MOVED)
    return a, KIND_DECAY, flags


def auto_windows(time, data, event_time: float, t_shift_s: Sequence[float], rate, *,
                 prev_event_s: float = float("nan"), next_event_s: float = float("nan"),
                 width_s: float = DEFAULT_WIDTH_S, min_start_s: float = DEFAULT_MIN_START_S,
                 max_start_s: float = DEFAULT_MAX_START_S, step_s: float = DEFAULT_STEP_S,
                 plateau_tol_um: float = DEFAULT_PLATEAU_TOL_UM, plateau_span_s: float = DEFAULT_PLATEAU_SPAN_S,
                 background_factor: float = DEFAULT_BACKGROUND_FACTOR,
                 rate_smoothing_s: float = DEFAULT_RATE_SMOOTHING_S, min_after_s: float = DEFAULT_MIN_AFTER_S,
                 max_decay_s: float = DEFAULT_MAX_DECAY_S, creep_ratio: float = DEFAULT_CREEP_RATIO,
                 onset_fraction: float = DEFAULT_ONSET_FRACTION, before_gap_s: float = DEFAULT_BEFORE_GAP_S,
                 onset_search_s: float = DEFAULT_ONSET_SEARCH_S,
                 event_margin_s: float = DEFAULT_EVENT_MARGIN_S) -> list:
    """One :class:`AutoWindows` per sensor of ``data`` (``(n_sensors, n)`` on
    ``time``), each about its own ``t_i = event_time + t_shift_s[i]``.

    ``rate`` is the filtered slip rate on ``time``, as
    :func:`coseismic_slip.slip_rate` (or ``SlipAnalysis.rate``) gives it, so
    the onset and the settling follow the same filter as the ``t_i`` picks.
    ``prev_event_s`` / ``next_event_s`` are the neighbouring events relative to
    ``event_time`` (NaN when there is none); no window comes closer to them
    than ``event_margin_s``.  The returned windows are relative to
    ``event_time``.
    """
    t = np.asarray(time, dtype=float).ravel()
    sampling_interval(t)
    d = np.asarray(data, dtype=float)
    r = np.asarray(rate, dtype=float)
    if d.ndim == 1:
        d, r = d[None, :], r.reshape(1, -1)
    if d.shape != r.shape or d.shape[-1] != t.size:
        raise ValueError("data and rate must be (n_sensors, n) on the time axis")
    if len(t_shift_s) != d.shape[0]:
        raise ValueError("t_shift_s needs one value per sensor")
    width_s = float(width_s)
    if not (math.isfinite(width_s) and width_s > 0):
        raise ValueError("the window width must be > 0")
    if not 0 < float(onset_fraction) < 1:
        raise ValueError("the onset fraction must be between 0 and 1")
    if not (float(step_s) > 0 and float(min_start_s) < float(max_start_s)):
        raise ValueError("the after-window starts need min < max and a step > 0")
    if not (float(plateau_tol_um) >= 0 and float(plateau_span_s) > 0):
        raise ValueError("the plateau tolerance and span must be positive")
    if not (float(background_factor) > 0 and float(rate_smoothing_s) >= 0):
        raise ValueError("the background factor must be > 0 and the rate smoothing >= 0")
    if not float(creep_ratio) > 0:
        raise ValueError("the creep ratio must be > 0")
    if not 0 <= float(min_after_s) <= float(max_decay_s):
        raise ValueError("the after window needs 0 <= earliest start <= latest start")
    event_time = float(event_time)
    after_limit = float(t[-1]) - event_time
    before_limit = float(t[0]) - event_time
    if math.isfinite(next_event_s):
        after_limit = min(after_limit, next_event_s - event_margin_s)
    if math.isfinite(prev_event_s):
        before_limit = max(before_limit, prev_event_s + event_margin_s)
    out = []
    for i in range(d.shape[0]):
        shift = float(t_shift_s[i])
        if not math.isfinite(shift):
            out.append(AutoWindows(None, None, flags=[FLAG_NO_RATE]))
            continue
        t_i = event_time + shift
        before, onset, flags_b = choose_before(t, r[i], t_i, limit_s=before_limit - shift, width_s=width_s,
                                               onset_fraction=onset_fraction, before_gap_s=before_gap_s,
                                               onset_search_s=onset_search_s, peak_span_s=min_start_s)
        background = float("nan")
        if before is not None:
            line = fit_line(t, d[i], t_i, before)
            if line.valid:
                background = line.slope
        a, kind, flags = choose_after(t, d[i], r[i], t_i, background, limit_s=after_limit - shift,
                                      width_s=width_s, min_start_s=min_start_s, max_start_s=max_start_s,
                                      step_s=step_s, plateau_tol_um=plateau_tol_um,
                                      plateau_span_s=plateau_span_s, background_factor=background_factor,
                                      rate_smoothing_s=rate_smoothing_s, min_after_s=min_after_s,
                                      max_decay_s=max_decay_s)
        ratio = float("nan")
        peak = peak_rate(t, r[i], t_i, min_start_s)
        if math.isfinite(background) and math.isfinite(peak) and peak > 0:
            ratio = background / peak
            if ratio > creep_ratio:
                flags = flags + [FLAG_CREEP]
        after = None if a is None else (shift + a, shift + a + width_s)
        if before is not None:
            before = (shift + before[0], shift + before[1])
        out.append(AutoWindows(before, after, kind, shift + onset, background, ratio, flags + flags_b))
    return out


__all__ = [
    "KIND_PLATEAU", "KIND_DECAY", "KINDS", "DEFAULT_WIDTH_S", "DEFAULT_MIN_START_S", "DEFAULT_MAX_START_S",
    "DEFAULT_STEP_S", "DEFAULT_PLATEAU_TOL_UM", "DEFAULT_PLATEAU_SPAN_S", "DEFAULT_BACKGROUND_FACTOR",
    "DEFAULT_RATE_SMOOTHING_S", "DEFAULT_MIN_AFTER_S", "DEFAULT_MAX_DECAY_S", "DEFAULT_CREEP_RATIO",
    "DEFAULT_ONSET_FRACTION", "DEFAULT_BEFORE_GAP_S", "DEFAULT_ONSET_SEARCH_S", "DEFAULT_EVENT_MARGIN_S",
    "MIN_WIDTH_FRACTION", "FLAG_NO_DECAY", "FLAG_NO_BACKGROUND", "FLAG_CREEP", "FLAG_NEGATIVE", "FLAG_NO_ROOM_AFTER",
    "FLAG_AFTER_MOVED",
    "FLAG_NO_ONSET", "FLAG_BEFORE_SHORT", "FLAG_NO_ROOM_BEFORE", "FLAG_NO_RATE", "AutoWindows",
    "after_line_values", "plateau_start", "peak_rate",
    "settle_time", "choose_before", "choose_after", "auto_windows",
]
