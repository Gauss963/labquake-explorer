"""Displacement amplitude spectra and omega-n (Brune) fits for one channel.

Pure numerics: arrays in, frozen dataclasses out.  Nothing here knows about
Tk, the data tree or persistence.  The pipeline is

    extract_window  ->  compute_spectrum  ->  bin_spectrum  ->  fit_omega_n

Conventions (documented once here, echoed in the result ``meta``):

* Windows are defined by SAMPLE COUNTS around a trigger sample:
  ``n_pre = round(pre_s * fs)``, ``n_post = round(post_s * fs)`` with
  ``fs = 1 / median(diff(time))``.  The trigger is the sample nearest to the
  caller's ``trigger_time`` (an event time or a picked arrival).
* The noise window is the equal-length segment immediately before the signal
  window, or immediately after it when there is no room before (a recorder
  that keeps little pre-trigger data); a caller's ``noise_window`` may lie on
  either side.  ``noise_region`` says which (``'pre'``, ``'post'``,
  ``'explicit'``).  When fewer samples exist it is shortened down to 25% of
  the signal length (and its spectrum is scaled to the signal length); below
  that the noise is NaN and ``noise_available`` is False.  It never overlaps
  the signal window.
* Amplitude spectra are one-sided ``|rfft(w * x)| * dt`` (units: input * s),
  i.e. estimates of the continuous Fourier transform of a transient that is
  fully inside the flat part of the taper - no coherent-gain division.  This
  convention only holds when the transient sits in the flat part of the
  taper: with the default tukey taper (``alpha = 0.25``) the first and last
  ``alpha/2 = 12.5%`` of the window are ramped, so ``pre_s`` must exceed
  ``(alpha/2) * (pre_s + post_s)`` plus the pulse duration.  ``meta``
  records ``taper_energy_fraction = sum(w * x**2) / sum(x**2)`` and
  ``transient_in_flat_region`` (fraction >= 0.95); ``fit_omega_n`` turns a
  False flag into a warning.  The noise spectrum is divided by
  ``sqrt(mean(w**2))`` (incoherent gain) so it is comparable to the
  transient estimate.
* Calibration converts volts to a DISPLACEMENT spectral density in m*s by
  dividing by the sensor gain and by ``(2*pi*f)**k`` (k = 0, 1, 2 for V/m,
  V/(m/s), V/(m/s^2)), or, with an in-situ response in V/(N m) (an
  empirical Green's function route: the gain already holds the path and the
  medium), to a MOMENT spectrum in N*m*s whose plateau is the seismic moment
  itself.  Outside the calibration table the spectrum is NaN (masked), never
  extrapolated.
* Log-binned amplitudes are ``sqrt(mean(power))`` per bin (``estimator =
  'rms'``, the default); noise adds in quadrature, so bins near the SNR cut
  read high (+5% at SNR = 3).  ``noise_subtract=True`` removes the mean
  noise power per bin before the square root (off by default).  The
  ``'logmean'`` estimator takes the geometric mean of the raw amplitudes per
  bin and ``'median'`` their median: both sit below the rms on a noisy
  spectrum (a log-interpolation chain without averaging is closer to them),
  and neither subtracts noise.
* The fit is ``ln A(f) = ln Omega0 - ln(1 + (f/fc)**n)`` in the parameters
  ``(ln Omega0, ln fc, n)`` with ``n`` fixed at 2 by default (a free ``n``
  stays inside ``n_bounds``, ``N_BOUNDS = (1, 4)`` unless given), weighted by
  ``sqrt(count)`` per log bin, using only bins with ``SNR >= snr_min``.
  ``fit_omega_n`` never raises on bad data; it returns ``valid=False`` with
  a reason.
"""
from __future__ import annotations

import hashlib
import math
import re
import warnings
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.optimize import least_squares

RESULT_VERSION = 1

BASELINE_MODES = ("pre_linear", "none", "highpass")
TAPER_KINDS = ("tukey", "bh4")
LOSSES = ("linear", "soft_l1")
BIN_ESTIMATORS = ("rms", "logmean", "median")   # per log bin: sqrt(mean(power)), geometric mean, median of |X|

# Calibration units the caller may declare and the derivative order k that
# turns the calibrated spectrum into a displacement spectral density.
CALIBRATION_UNITS = {"V/m": 0, "V/(m/s)": 1, "V/(m/s^2)": 2, "V/V": 0, "V/(N m)": 0}
MOMENT_UNIT = "V/(N m)"                 # a response that turns volts into a moment spectrum
DISPLACEMENT_SPECTRUM_UNITS = "m*s"
MOMENT_SPECTRUM_UNITS = "N*m*s"
_FREQ_ALIASES = ("frequency_hz", "frequency", "freq", "f", "f_hz", "freq_hz")
_GAIN_ALIASES = ("gain", "amplitude", "response")

MIN_NOISE_FRACTION = 0.25
STEP_WIDTH_FRACTION = 0.01      # logistic scale of the removed step, fraction of the window
STEP_EDGE_FRACTION = 0.10       # first/last fraction of the window used to measure the step
STEP_MIN_SCALE = 0.5            # smallest logistic scale (samples) the step fit may return
STEP_CENTRE_METHODS = ("antisymmetric", "peak", "trigger")
TAPER_FLAT_FRACTION = 0.95      # taper_energy_fraction at or above this counts as "in the flat region"
N_BOUNDS = (1.0, 4.0)           # default (n_min, n_max) of a free exponent; fit_omega_n takes others
NOISE_REGIONS = ("pre", "post", "explicit")   # where the noise window lies / came from ('' when none)
MIN_FIT_BINS = 8
CONVERGED_REL_OPTIMALITY = 1e-2  # converged: scaled gradient <= this fraction of the residual norm
CONVERGED_ABS_OPTIMALITY = 1e-6  # ... or below this absolute value (exact data, zero residual)


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowResult:
    """A signal window (and the noise window before it) cut from one trace."""
    t_rel: np.ndarray           # time of each signal sample relative to the trigger sample (s)
    signal: np.ndarray          # baseline-corrected signal window, length n_pre + n_post
    noise: np.ndarray           # noise window (length <= signal); all-NaN when unavailable
    fs: float
    i_trigger: int              # index of the trigger sample in the original trace
    n_pre: int
    n_post: int
    noise_fraction: float       # len(noise) / len(signal); 0 when unavailable
    noise_available: bool
    baseline_mode: str          # requested mode
    baseline_region: str        # 'noise', 'pre_trigger', 'none' (where the line was fitted)
    step_removed: bool
    step_amplitude: float       # amplitude of the removed step (input units); NaN if not removed
    step_centre: float          # fractional sample (in the signal window) of the removed step's half-rise
    step_scale: float           # logistic scale used for the removal (samples)
    step_fit_scale: float       # logistic scale fitted to the step's antisymmetric part (samples)
    highpass_hz: Optional[float] = None
    step_centre_method: str = ""    # 'antisymmetric' / 'peak' / 'trigger' when a step was removed
    trigger_time: float = float("nan")          # the caller's trigger_time (s, trace clock)
    step_width_fraction: Optional[float] = None  # requested width (None = fitted); None if no step
    step_residual: float = float("nan")         # rms antisymmetric residual after removal / |amplitude|
    i_noise0: int = -1                          # trace index of the first noise sample (-1 when unavailable)
    i_noise1: int = -1                          # one past the last noise sample
    noise_region: str = ""                      # 'pre' / 'post' (automatic), 'explicit' (given); '' when unavailable

    @property
    def n(self) -> int:
        return int(self.signal.size)

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def window_s(self) -> float:
        return self.n * self.dt


def sampling_rate(time) -> float:
    """``1 / median(diff(time))``; raises ValueError on a degenerate axis."""
    time = np.asarray(time, dtype=float).ravel()
    if time.size < 2:
        raise ValueError("time needs at least two samples")
    dt = float(np.median(np.diff(time)))
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError("time must be increasing with a finite sample interval")
    return 1.0 / dt


def _logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _antisymmetric(sig: np.ndarray, c: float, i: np.ndarray) -> np.ndarray:
    """``sig(c + i) - sig(c - i)`` with linear interpolation for fractional ``c``."""
    grid = np.arange(sig.size, dtype=float)
    return np.interp(c + i, grid, sig) - np.interp(c - i, grid, sig)


def _step_centre(sig: np.ndarray, step: float, half_width: int,
                 c_fixed: Optional[float] = None) -> tuple[float, float]:
    """Centre ``c`` (fractional sample) and logistic scale ``s`` (samples) of the
    step in ``sig``, from its antisymmetric part alone.

    For ``i = 1..half_width`` the antisymmetric part ``sig[c+i] - sig[c-i]`` is
    matched to ``step * (2 L(i/s) - 1)``.  A transient symmetric about the step
    (the coseismic offset develops with the pulse) has no antisymmetric part
    about the true centre, so the estimate does not depend on the
    pulse-to-offset ratio; the free scale keeps a shape mismatch from
    dragging the centre.

    The cost ``sum_i (a_c(i) - m_s(i))**2 = A(c) - 2 C_s(c) + D_s`` is
    evaluated at UNIT stride for every ``c`` in ``[half_width, n-1-half_width]``
    and every scale of a log grid: ``A(c) = sum_i a_c(i)**2`` in one pass over
    the offsets, ``C_s(c) = sum_i m_s(i) a_c(i)`` as two FFT correlations.
    (A coarse grid in ``c`` misses the basin, which is only as wide as the
    pulse.)  ``(c, s)`` are then polished continuously (fractional ``c`` by
    linear interpolation of ``sig``) with ``s >= STEP_MIN_SCALE``.

    ``c_fixed`` skips the sweep and only fits the scale at that centre.
    Returns ``(c, s)``.
    """
    n = sig.size
    h = int(half_width)
    if c_fixed is not None:
        c_fixed = float(min(max(c_fixed, 0.0), n - 1.0))
        h = min(h, int(math.floor(min(c_fixed, n - 1 - c_fixed))))
    if h < 1 or n < 2 * h + 3:
        c = float(np.argmax(np.abs(sig - sig[0]))) if c_fixed is None else c_fixed
        return c, 1.0
    i = np.arange(1, h + 1, dtype=float)
    scales = np.geomspace(STEP_MIN_SCALE, max(2.0 * STEP_MIN_SCALE, h / 2.0), 48)
    models = [step * (2.0 * _logistic(i / sc) - 1.0) for sc in scales]

    if c_fixed is None:
        c_lo, c_hi = h, n - 1 - h
        cs = np.arange(c_lo, c_hi + 1)
        a2 = np.zeros(cs.size)
        for ii in range(1, h + 1):
            d = sig[cs + ii] - sig[cs - ii]
            a2 += d * d
        best = (np.inf, c_lo, 0)
        for k, m in enumerate(models):
            fwd = sps.correlate(sig, np.concatenate([[0.0], m]), mode="valid")      # sum_i m(i) sig[c+i]
            bwd = sps.correlate(sig, np.concatenate([m[::-1], [0.0]]), mode="valid")  # sum_i m(i) sig[c-i] at c-h
            cost = a2 - 2.0 * (fwd[cs] - bwd[cs - h]) + float(np.sum(m * m))
            j = int(np.argmin(cost))
            if cost[j] < best[0]:
                best = (float(cost[j]), int(cs[j]), k)
        _, c, k = best
        c, sc = float(c), float(scales[k])
    else:
        c_lo = c_hi = c = c_fixed
        anti = _antisymmetric(sig, c, i)
        k = int(np.argmin([np.sum((anti - m) ** 2) for m in models]))
        sc = float(scales[k])

    # continuous polish of (c, s) (or s alone for a fixed centre)
    def resid(p):
        cc, ss = (p[0], p[1]) if c_fixed is None else (c, p[0])
        return _antisymmetric(sig, cc, i) - step * (2.0 * _logistic(i / ss) - 1.0)

    p0 = np.array([c, sc]) if c_fixed is None else np.array([sc])
    lb = [float(c_lo), STEP_MIN_SCALE] if c_fixed is None else [STEP_MIN_SCALE]
    ub = [float(c_hi), float(h)] if c_fixed is None else [float(h)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = least_squares(resid, p0, bounds=(lb, ub), max_nfev=200)
        if res.success and np.all(np.isfinite(res.x)) and res.cost <= float(np.sum(resid(p0) ** 2)) / 2.0:
            if c_fixed is None:
                c, sc = float(res.x[0]), float(res.x[1])
            else:
                sc = float(res.x[0])
    except Exception:  # keep the discrete estimate
        pass
    return c, sc


def _step_residual(resid_sig: np.ndarray, step: float, c: float, scale: float, fit_scale: float,
                   half_width: int) -> float:
    """Shape mismatch of the removed step: rms of the antisymmetric part of
    the signal AFTER removal, ``r[c0+i] - r[c0-i]`` about the nearest grid
    sample ``c0 = round(c)`` for ``i`` up to ``3 * max(scale, fit_scale)``
    samples, relative to ``|step|``.  A pulse symmetric about the step leaves
    no antisymmetric part, so this is ~0 when the removed logistic matched the
    step and O(0.1-1) when it did not (e.g. a one-sample step).  Evaluated on
    the grid, not by interpolation, so a sub-sample centre cannot hide an
    on-grid mismatch.  NaN when it cannot be evaluated or the step is
    negligible (< 1e-9 of the signal peak)."""
    peak = float(np.max(np.abs(resid_sig))) if resid_sig.size else 0.0
    if not (np.isfinite(step) and np.isfinite(c) and abs(step) > 1e-9 * peak and step != 0.0):
        return float("nan")     # no step worth diagnosing
    c0 = int(round(c))
    reach = int(math.ceil(3.0 * max(scale, fit_scale if np.isfinite(fit_scale) else 0.0)))
    reach = min(max(reach, 1), int(half_width), min(c0, resid_sig.size - 1 - c0))
    if reach < 1:
        return float("nan")
    i = np.arange(1, reach + 1)
    anti = resid_sig[c0 + i] - resid_sig[c0 - i]
    return float(np.sqrt(np.mean(anti ** 2)) / abs(step))


def extract_window(time, x, trigger_time: float, pre_s: float, post_s: float,
                   baseline: str = "pre_linear", highpass_hz: Optional[float] = None,
                   remove_step: bool = False,
                   step_width_fraction: Optional[float] = STEP_WIDTH_FRACTION,
                   step_centre: str = "antisymmetric",
                   noise_window: Optional[tuple] = None) -> WindowResult:
    """Cut ``[trigger - pre_s, trigger + post_s)`` out of ``x`` by sample counts.

    The noise window is the stretch of the same length just before the signal
    window (``noise_region == 'pre'``); when fewer than ``MIN_NOISE_FRACTION``
    of the signal window's samples lie before it, the stretch just after the
    signal window instead (``'post'``).  ``noise_window = (start_s, end_s)``
    relative to the trigger gives one explicitly (``'explicit'``): it must end
    at or before the signal window or start at or after it, a gap between the
    two being allowed.  Either way it has to hold at least
    ``MIN_NOISE_FRACTION`` of the signal window's samples to count.

    For the spectrum's no-coherent-gain amplitude convention the transient
    must sit in the flat part of the taper: choose ``pre_s`` larger than
    ``(alpha/2) * (pre_s + post_s)`` plus the pulse duration (12.5% of the
    window for the default tukey ``alpha = 0.25``); ``compute_spectrum``
    reports ``taper_energy_fraction`` so the choice can be checked.

    ``baseline``:
      * ``'pre_linear'`` (default): a straight line fitted over the noise window
        (or, when no noise window lies before the signal, over the pre-trigger
        samples) is subtracted from noise and signal.  With fewer than two
        such samples no baseline is removed and ``baseline_region == 'none'``.
      * ``'none'``: raw samples.
      * ``'highpass'``: zero-phase Butterworth (order 2) at ``highpass_hz``,
        run over noise + signal jointly so the join has no edge effect.
    ``remove_step``: subtract ``step * logistic((i - i_c) / (step_width_fraction * n))``
    where ``step`` is the mean of the last 10% of the signal window minus the
    mean of the first 10% and ``i_c`` the step centre; this lets strain-gauge
    channels with a permanent coseismic offset be analysed.  ``step_centre``:
      * ``'antisymmetric'`` (default): the centre about which the
        antisymmetric part of the signal best matches a logistic step (see
        ``_step_centre``).  A pulse symmetric about the step does not bias it,
        whatever the offset-to-pulse ratio.
      * ``'trigger'``: the trigger sample (``n_pre``), i.e. the caller's picked
        arrival.  Deterministic; right when the pick is the step's half-rise.
      * ``'peak'``: the sample of maximum |x|.  Only correct when the step is
        sharp and co-located with the peak; when the offset exceeds the
        pulse peak the maximum lies on the post-step plateau and the removal
        is wrong.  Kept as the simple alternative.
    ``step_width_fraction=None`` removes the step with the logistic scale
    fitted to the data (at the chosen centre) instead of 1% of the window;
    the fitted scale is never below ``STEP_MIN_SCALE`` samples.  The removal
    is only as good as its shape: a true step much sharper than the removed
    one leaves a residual that grows with frequency.  ``step_residual``
    (rms mismatch of the antisymmetric part relative to the step amplitude)
    reports how well the removed shape matched.

    Raises ValueError if the signal window is not fully inside the trace.
    """
    if baseline not in BASELINE_MODES:
        raise ValueError(f"baseline must be one of {BASELINE_MODES}, got {baseline!r}")
    if step_centre not in STEP_CENTRE_METHODS:
        raise ValueError(f"step_centre must be one of {STEP_CENTRE_METHODS}, got {step_centre!r}")
    time = np.asarray(time, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if time.shape != x.shape:
        raise ValueError("time and x must have the same length")
    if not (np.isfinite(pre_s) and np.isfinite(post_s)) or pre_s < 0 or post_s < 0:
        raise ValueError("pre_s and post_s must be finite and >= 0")
    fs = sampling_rate(time)
    dt = 1.0 / fs
    if not np.isfinite(trigger_time):
        raise ValueError("trigger_time must be finite")
    i_trigger = int(np.argmin(np.abs(time - float(trigger_time))))
    n_pre = int(round(pre_s * fs))
    n_post = int(round(post_s * fs))
    n = n_pre + n_post
    if n < 2:
        raise ValueError("window must contain at least two samples")
    i0, i1 = i_trigger - n_pre, i_trigger + n_post
    if i0 < 0 or i1 > x.size:
        raise ValueError(
            f"window [{i0}, {i1}) is not inside the trace of {x.size} samples "
            f"(trigger sample {i_trigger}, n_pre={n_pre}, n_post={n_post})")

    min_noise = max(2, int(math.ceil(MIN_NOISE_FRACTION * n)))
    if noise_window is None:
        # the same length just before the signal; just after it when there is no room before
        i_n0, i_n1, noise_region = max(0, i0 - n), i0, "pre"
        if i_n1 - i_n0 < min_noise and min(x.size, i1 + n) - i1 >= min_noise:
            i_n0, i_n1, noise_region = i1, min(x.size, i1 + n), "post"
    else:
        start_s, end_s = (float(v) for v in noise_window)
        if not (np.isfinite(start_s) and np.isfinite(end_s)) or start_s >= end_s:
            raise ValueError("noise_window must be (start, end) seconds relative to the trigger, start < end")
        noise_region = "explicit"
        if end_s <= -pre_s + 0.5 * dt:                   # before the signal window
            i_n0 = max(0, i_trigger + int(round(start_s * fs)))
            i_n1 = min(i0, i_trigger + int(round(end_s * fs)))
        elif start_s >= post_s - 0.5 * dt:               # after it
            i_n0 = max(i1, i_trigger + int(round(start_s * fs)))
            i_n1 = min(x.size, i_trigger + int(round(end_s * fs)))
        else:
            raise ValueError("the noise window must lie before the signal window (end <= -pre_s) "
                             "or after it (start >= post_s)")
    n_noise = max(0, i_n1 - i_n0)
    noise_available = n_noise >= min_noise
    if noise_available:
        noise = x[i_n0:i_n1].copy()
    else:
        n_noise = 0
        noise = np.full(n, np.nan)
        i_n0 = i_n1 = -1
        noise_region = ""
    noise_before = noise_available and i_n1 <= i0
    sig = x[i0:i1].copy()

    # --- baseline ---------------------------------------------------------
    region = "none"
    if baseline == "pre_linear":
        # sample indices relative to the start of the signal window
        if noise_before:
            k = np.arange(i_n0 - i0, i_n1 - i0, dtype=float)   # noise samples relative to the signal start
            y = noise
            region = "noise"
        elif n_pre >= 2:
            k = np.arange(0, n_pre, dtype=float)
            y = sig[:n_pre]
            region = "pre_trigger"
        else:
            k = y = None
        if k is not None and np.all(np.isfinite(y)):
            slope, intercept = np.polyfit(k, y, 1)
            sig = sig - (slope * np.arange(n) + intercept)
            if noise_available:
                noise = noise - (slope * np.arange(i_n0 - i0, i_n1 - i0) + intercept)

    # --- coseismic step ---------------------------------------------------
    step_amp = step_scale = step_fit_scale = step_resid = float("nan")
    centre_method = step_centre if remove_step else ""
    width_fraction = None
    step_centre = float("nan")
    if remove_step:
        m = max(1, int(round(STEP_EDGE_FRACTION * n)))
        lo_level, hi_level = float(np.mean(sig[:m])), float(np.mean(sig[-m:]))
        step_amp = hi_level - lo_level
        if centre_method == "antisymmetric":
            c_fit, step_fit_scale = _step_centre(sig, step_amp, m)
        else:
            c_fixed = float(np.argmax(np.abs(sig))) if centre_method == "peak" else float(n_pre)
            c_fit, step_fit_scale = _step_centre(sig, step_amp, m, c_fixed=c_fixed)
        step_centre = c_fit
        if step_width_fraction is None:
            step_scale = step_fit_scale
        else:
            width_fraction = float(step_width_fraction)
            step_scale = max(width_fraction * n, 1e-9)
        sig = sig - step_amp * _logistic((np.arange(n) - step_centre) / step_scale)
        step_resid = _step_residual(sig, step_amp, step_centre, step_scale, step_fit_scale, m)

    # --- high-pass --------------------------------------------------------
    if baseline == "highpass":
        if highpass_hz is None or not (0.0 < float(highpass_hz) < 0.5 * fs):
            raise ValueError("highpass_hz must be given and lie in (0, fs/2) for baseline='highpass'")
        sos = sps.butter(2, float(highpass_hz), btype="highpass", fs=fs, output="sos")
        if noise_available and (i_n1 == i0 or i_n0 == i1):   # adjacent: one pass, no edge at the join
            joint = np.concatenate([noise, sig] if i_n1 == i0 else [sig, noise])
            padlen = min(3 * 2 * 2, joint.size - 1)  # scipy default is small; keep it valid
            joint = sps.sosfiltfilt(sos, joint, padlen=padlen)
            if i_n1 == i0:
                noise, sig = joint[:n_noise].copy(), joint[n_noise:].copy()
            else:
                sig, noise = joint[:n].copy(), joint[n:].copy()
        elif noise_available:                        # a gap: filter the two stretches on their own
            noise = sps.sosfiltfilt(sos, noise, padlen=min(12, noise.size - 1))
            sig = sps.sosfiltfilt(sos, sig, padlen=min(12, sig.size - 1))
        else:
            sig = sps.sosfiltfilt(sos, sig, padlen=min(12, sig.size - 1))
        region = "highpass"

    t_rel = (np.arange(n) - n_pre) * dt
    return WindowResult(
        t_rel=t_rel, signal=sig, noise=noise, fs=fs, i_trigger=i_trigger,
        n_pre=n_pre, n_post=n_post,
        noise_fraction=float(n_noise / n) if noise_available else 0.0,
        noise_available=bool(noise_available), baseline_mode=baseline,
        baseline_region=region, step_removed=bool(remove_step),
        step_amplitude=step_amp, step_centre=step_centre, step_scale=step_scale,
        step_fit_scale=step_fit_scale,
        highpass_hz=float(highpass_hz) if highpass_hz is not None else None,
        step_centre_method=centre_method, trigger_time=float(trigger_time),
        step_width_fraction=width_fraction, step_residual=step_resid,
        i_noise0=int(i_n0), i_noise1=int(i_n1), noise_region=noise_region,
    )


def taper_window(n: int, kind: str = "tukey", alpha: float = 0.25) -> np.ndarray:
    """Taper of length ``n``: ``'tukey'`` (flat middle, cosine edges of total
    fraction ``alpha``) or ``'bh4'`` (4-term Blackman-Harris)."""
    n = int(n)
    if n < 1:
        raise ValueError("taper length must be >= 1")
    if kind == "tukey":
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("tukey alpha must be in [0, 1]")
        return sps.windows.tukey(n, alpha=float(alpha), sym=True)
    if kind == "bh4":
        return sps.windows.blackmanharris(n, sym=True)
    raise ValueError(f"taper must be one of {TAPER_KINDS}, got {kind!r}")


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Calibration:
    """Sensor gain table.  ``interp`` is log-log; NaN outside the table."""
    frequency_hz: np.ndarray
    gain: np.ndarray            # linear gain in ``unit`` (never dB)
    unit: str                   # one of CALIBRATION_UNITS
    k: int                      # derivative order: divide by (2*pi*f)**k for displacement
    source: str = ""

    @property
    def fmin(self) -> float:
        return float(self.frequency_hz[0])

    @property
    def fmax(self) -> float:
        return float(self.frequency_hz[-1])

    @property
    def output_units(self) -> str:
        if self.unit == "V/V":
            return "V*s"
        return MOMENT_SPECTRUM_UNITS if self.unit == MOMENT_UNIT else DISPLACEMENT_SPECTRUM_UNITS

    @property
    def sha256(self) -> str:
        """Content hash of the (frequency, gain, unit) table, 16 hex digits."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.frequency_hz, dtype=float).tobytes())
        h.update(np.ascontiguousarray(self.gain, dtype=float).tobytes())
        h.update(self.unit.encode())
        return h.hexdigest()[:16]

    def interp(self, f) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        out = np.full(f.shape, np.nan)
        inside = np.isfinite(f) & (f >= self.fmin) & (f <= self.fmax) & (f > 0)
        if np.any(inside):
            out[inside] = 10.0 ** np.interp(np.log10(f[inside]), np.log10(self.frequency_hz),
                                            np.log10(self.gain))
        return out

    @classmethod
    def from_arrays(cls, frequency_hz, gain, unit: str, gain_is_db: bool = False,
                    source: str = "", gain_divisor: float = 1.0) -> "Calibration":
        """``gain_divisor`` divides the (linear) gain, so a stored response can
        be used as it is when the unit needs a constant factor taken out
        (a moment response ``Psi`` in V/(N m) per ``C_FM = vP + vS``, say)."""
        if unit not in CALIBRATION_UNITS:
            raise ValueError(f"unit must be one of {tuple(CALIBRATION_UNITS)}, got {unit!r}")
        divisor = float(gain_divisor)
        if not (math.isfinite(divisor) and divisor > 0.0):
            raise ValueError(f"gain_divisor must be a positive number, got {gain_divisor!r}")
        f = np.asarray(frequency_hz, dtype=float).ravel()
        g = np.asarray(gain, dtype=float).ravel()
        if f.shape != g.shape:
            raise ValueError("frequency and gain columns must have the same length")
        if gain_is_db:
            g = 10.0 ** (g / 20.0)
        g = g / divisor
        keep = np.isfinite(f) & np.isfinite(g) & (f > 0) & (g > 0)
        f, g = f[keep], g[keep]
        order = np.argsort(f, kind="stable")
        f, g = f[order], g[order]
        f, idx = np.unique(f, return_index=True)
        g = g[idx]
        if f.size < 2:
            raise ValueError("calibration needs at least two rows with positive frequency and gain")
        return cls(f, g, unit, CALIBRATION_UNITS[unit], source)

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        """Rebuild a Calibration from ``as_dict()`` output (the stored table)."""
        return cls.from_arrays(d["frequency_hz"], d["gain"], d["unit"], source=str(d.get("source", "")))

    def as_dict(self) -> dict:
        """JSON-like record including the table itself (it is small), so a
        saved spectrum can be regenerated without the CSV."""
        return {"unit": self.unit, "k": self.k, "fmin_hz": self.fmin, "fmax_hz": self.fmax,
                "n_rows": int(self.frequency_hz.size), "source": self.source,
                "sha256": self.sha256,
                "frequency_hz": self.frequency_hz.tolist(), "gain": self.gain.tolist()}


def _parse_header(name) -> tuple[str, str]:
    """Normalise a CSV header into ``(base, unit)``: ``'Frequency (Hz)'`` ->
    ``('frequency', 'hz')``, ``'Mean_Inst_App_Psi'`` -> ``('mean_inst_app_psi', '')``."""
    text = str(name).strip().lower()
    unit = ""
    m = re.search(r"\(([^)]*)\)", text)
    if m:
        unit = re.sub(r"[^a-z0-9/]+", "", m.group(1).lower())
        text = text[:m.start()] + text[m.end():]
    base = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return base, unit


_FREQ_BASES = {"frequency", "freq", "f", "hz", "frequency_hz", "freq_hz", "f_hz",
               "frequency_khz", "freq_khz", "f_khz"}
_GAIN_TOKENS = ("gain", "amp", "amplitude", "response", "ratio", "mag", "db", "psi",
                "calib", "inst", "sensitivity", "tf")


def _frequency_column(columns) -> tuple[Optional[str], float]:
    """The frequency column and the factor that turns it into Hz (1e3 for kHz)."""
    for c in columns:
        base, unit = _parse_header(c)
        if base in _FREQ_BASES or base.startswith("freq"):
            khz = "khz" in unit or base.endswith("khz")
            return c, (1e3 if khz else 1.0)
    return None, 1.0


def _gain_column(columns, f_col) -> tuple[Optional[str], bool]:
    """The gain column (preferring names that look like a gain) and whether its
    header says dB."""
    others = [c for c in columns if c != f_col]
    preferred = []
    for c in others:
        base, unit = _parse_header(c)
        tokens = set(base.split("_"))
        if unit == "db" or tokens & set(_GAIN_TOKENS) or any(t in base for t in ("gain", "amp", "response", "calib", "psi")):
            preferred.append(c)
    chosen = preferred[0] if len(preferred) == 1 else (others[0] if len(others) == 1 and not preferred else None)
    if chosen is None and preferred:
        chosen = preferred[0]
    if chosen is None:
        return None, False
    base, unit = _parse_header(chosen)
    is_db = unit == "db" or "db" in base.split("_")
    return chosen, is_db


def _find_column(columns, aliases) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def load_calibration_csv(path: Union[str, Path], unit: str,
                         gain_is_db: Optional[bool] = False, gain_divisor: float = 1.0) -> Calibration:
    """Read a calibration CSV.

    The frequency column is recognised by name (``frequency``, ``freq``, ``f``,
    with or without a unit such as ``Frequency (Hz)`` or ``freq_kHz``; kHz is
    converted).  The gain column is the one whose header mentions gain,
    amplitude, response, ratio, dB, psi, calib or inst (or the only other
    column).  ``unit`` is declared by the caller (see ``CALIBRATION_UNITS``;
    ``'V/V'`` means uncalibrated).  ``gain_is_db`` says the gain column is
    20*log10; ``None`` takes it from the header (``(dB)`` or ``_dB``).
    ``gain_divisor`` divides the linear gain (see :meth:`Calibration.from_arrays`).
    """
    if unit not in CALIBRATION_UNITS:
        raise ValueError(f"unit must be one of {tuple(CALIBRATION_UNITS)}, got {unit!r}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"calibration CSV not found: {path}")
    df = pd.read_csv(path)
    f_col, f_scale = _frequency_column(df.columns)
    if f_col is None:
        raise ValueError("calibration CSV needs a frequency column (e.g. 'Frequency (Hz)', 'freq_kHz', 'f')")
    g_col, header_db = _gain_column(df.columns, f_col)
    if g_col is None:
        raise ValueError("calibration CSV needs one gain column (header mentioning gain, amplitude, "
                         "response, ratio, dB, psi or calib)")
    is_db = header_db if gain_is_db is None else bool(gain_is_db)
    return Calibration.from_arrays(pd.to_numeric(df[f_col], errors="coerce").to_numpy() * f_scale,
                                   pd.to_numeric(df[g_col], errors="coerce").to_numpy(),
                                   unit, is_db, source=str(path), gain_divisor=gain_divisor)


def csv_gain_is_db(path: Union[str, Path]) -> bool:
    """Whether the gain column header of a calibration CSV says dB."""
    df = pd.read_csv(path, nrows=1)
    f_col, _ = _frequency_column(df.columns)
    _, is_db = _gain_column(df.columns, f_col)
    return bool(is_db)


# ---------------------------------------------------------------------------
# frequency-dependent attenuation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Attenuation:
    """A ``Q^-1(f)`` table with a travel time: the spectrum is multiplied by
    ``exp(pi * f * travel_time * Q^-1(f))`` inside the table's band and left
    unchanged outside it, or, with ``hold_ends``, corrected outside the band
    with the end values of the table held (a clamped interpolation)."""
    frequency_hz: np.ndarray
    q_inv: np.ndarray
    travel_time_s: float
    source: str = ""
    hold_ends: bool = False

    @property
    def fmin(self) -> float:
        return float(self.frequency_hz[0])

    @property
    def fmax(self) -> float:
        return float(self.frequency_hz[-1])

    def q_inv_at(self, f) -> np.ndarray:
        """Log-log interpolated ``Q^-1``; NaN outside the table (the end values
        held there with ``hold_ends``)."""
        f = np.asarray(f, dtype=float)
        out = np.full(f.shape, np.nan)
        positive = np.isfinite(f) & (f > 0)
        inside = positive & (f >= self.fmin) & (f <= self.fmax) if not self.hold_ends else positive
        if np.any(inside):
            fq = np.clip(f[inside], self.fmin, self.fmax)
            out[inside] = 10.0 ** np.interp(np.log10(fq), np.log10(self.frequency_hz), np.log10(self.q_inv))
        return out

    def factor(self, f) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        q_inv = self.q_inv_at(f)
        out = np.ones(f.shape)
        inside = np.isfinite(q_inv)
        out[inside] = np.exp(np.pi * f[inside] * self.travel_time_s * q_inv[inside])
        return out

    @classmethod
    def from_arrays(cls, frequency_hz, q_inv, travel_time_s: float, source: str = "",
                    hold_ends: bool = False) -> "Attenuation":
        f = np.asarray(frequency_hz, dtype=float).ravel()
        q = np.asarray(q_inv, dtype=float).ravel()
        if f.shape != q.shape:
            raise ValueError("frequency and Q^-1 columns must have the same length")
        if not (np.isfinite(travel_time_s) and travel_time_s >= 0):
            raise ValueError("travel_time_s must be finite and >= 0")
        keep = np.isfinite(f) & np.isfinite(q) & (f > 0) & (q > 0)
        f, q = f[keep], q[keep]
        order = np.argsort(f, kind="stable")
        f, q = f[order], q[order]
        f, idx = np.unique(f, return_index=True)
        q = q[idx]
        if f.size < 2:
            raise ValueError("attenuation table needs at least two rows with positive frequency and Q^-1")
        return cls(f, q, float(travel_time_s), source, bool(hold_ends))

    def as_dict(self) -> dict:
        return {"travel_time_s": self.travel_time_s, "fmin_hz": self.fmin, "fmax_hz": self.fmax,
                "n_rows": int(self.frequency_hz.size), "source": self.source, "hold_ends": bool(self.hold_ends),
                "frequency_hz": self.frequency_hz.tolist(), "q_inv": self.q_inv.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Attenuation":
        return cls.from_arrays(d["frequency_hz"], d["q_inv"], d["travel_time_s"], str(d.get("source", "")),
                               bool(d.get("hold_ends", False)))


def load_q_csv(path: Union[str, Path], travel_time_s: float, hold_ends: bool = False) -> Attenuation:
    """Read a ``Q^-1(f)`` CSV (frequency column as in :func:`load_calibration_csv`,
    kHz converted; the other column is ``Q^-1``) with the travel time it applies
    to; ``hold_ends`` extends the table's end values outside its band."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Q CSV not found: {path}")
    df = pd.read_csv(path)
    f_col, f_scale = _frequency_column(df.columns)
    if f_col is None:
        raise ValueError("Q CSV needs a frequency column (e.g. 'freq_kHz')")
    others = [c for c in df.columns if c != f_col]
    q_cols = [c for c in others if "q" in _parse_header(c)[0].split("_")[0] or "q" in _parse_header(c)[0]]
    q_col = q_cols[0] if q_cols else (others[0] if len(others) == 1 else None)
    if q_col is None:
        raise ValueError("Q CSV needs one Q^-1 column")
    return Attenuation.from_arrays(pd.to_numeric(df[f_col], errors="coerce").to_numpy() * f_scale,
                                   pd.to_numeric(df[q_col], errors="coerce").to_numpy(),
                                   travel_time_s, source=str(path), hold_ends=hold_ends)


def t_star_from_q(distance_m: float, q: float, c: float) -> float:
    """``t* = distance / (Q * c)`` in seconds."""
    if not (q > 0 and c > 0 and distance_m >= 0):
        raise ValueError("Q and c must be > 0 and distance >= 0")
    return float(distance_m) / (float(q) * float(c))


# ---------------------------------------------------------------------------
# spectrum
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpectrumResult:
    f: np.ndarray               # rfft grid (Hz)
    amp_signal: np.ndarray      # amplitude spectral density, NaN where masked
    amp_noise: np.ndarray       # same grid; all-NaN when no noise window
    amp_units: str              # 'V*s' (uncalibrated / V/V) or 'm*s'
    f_binned: np.ndarray        # log-bin centres (geometric mean of raw f in the bin)
    amp_binned: np.ndarray      # sqrt(mean(power)) per bin
    noise_binned: np.ndarray
    snr: np.ndarray             # amp_binned / noise_binned
    counts: np.ndarray          # raw bins per log bin
    calibrated: bool
    t_star_s: Optional[float]
    meta: dict = field(default_factory=dict)

    @property
    def n_bins(self) -> int:
        return int(self.f_binned.size)

    def as_dict(self) -> dict:
        """JSON-like summary (binned arrays as lists; the raw grid is omitted)."""
        return {
            "version": RESULT_VERSION,
            "amp_units": self.amp_units,
            "calibrated": bool(self.calibrated),
            "t_star_s": self.t_star_s,
            "f_binned": self.f_binned.tolist(),
            "amp_binned": self.amp_binned.tolist(),
            "noise_binned": self.noise_binned.tolist(),
            "snr": self.snr.tolist(),
            "counts": self.counts.tolist(),
            "meta": dict(self.meta),
        }


def compute_spectrum(window: WindowResult, taper: str = "tukey", alpha: float = 0.25,
                     nfft: Optional[int] = None, calibration: Optional[Calibration] = None,
                     t_star_s: Optional[float] = None, fmin: Optional[float] = None,
                     fmax: Optional[float] = None, bins_per_decade: int = 30,
                     attenuation: Optional[Attenuation] = None,
                     noise_subtract: bool = False, estimator: str = "rms") -> SpectrumResult:
    """Amplitude spectra of the signal and noise windows plus log-binned averages.

    ``nfft >= n`` zero-pads (a smaller value is ignored).  ``calibration``
    converts to displacement (see module docstring); ``t_star_s`` applies
    ``exp(+pi*f*t*)`` to both signal and noise.  ``fmin``/``fmax`` default to
    the fundamental ``fs/nfft`` and Nyquist; ``noise_subtract`` and
    ``estimator`` (the per-bin statistic, ``BIN_ESTIMATORS``) are passed to
    ``bin_spectrum``.

    ``meta['taper_energy_fraction'] = sum(w * x**2) / sum(x**2)`` measures how
    much of the transient sits under the taper; the amplitude convention (no
    coherent-gain division) holds only when it is ~1.
    ``meta['transient_in_flat_region']`` is that fraction ``>= 0.95``.
    """
    if taper not in TAPER_KINDS:
        raise ValueError(f"taper must be one of {TAPER_KINDS}, got {taper!r}")
    x = window.signal
    n = x.size
    dt = window.dt
    nfft = n if nfft is None else max(int(nfft), n)
    w = taper_window(n, taper, alpha)
    f = np.fft.rfftfreq(nfft, dt)
    amp_sig = np.abs(np.fft.rfft(w * x, nfft)) * dt
    energy = float(np.sum(x * x))
    taper_fraction = float(np.sum(w * x * x) / energy) if energy > 0 and np.isfinite(energy) else float("nan")
    in_flat = bool(np.isfinite(taper_fraction) and taper_fraction >= TAPER_FLAT_FRACTION)

    if window.noise_available:
        noise = window.noise
        n_noise = noise.size
        w_n = taper_window(n_noise, taper, alpha)
        incoherent_gain = float(np.sqrt(np.mean(w_n ** 2)))
        amp_noise = np.abs(np.fft.rfft(w_n * noise, nfft)) * dt / incoherent_gain
        # scale a shorter noise segment to the signal length: white noise of
        # variance s^2 gives |X| dt ~ s dt sqrt(N), so multiply by sqrt(n / n_noise)
        amp_noise = amp_noise * math.sqrt(n / n_noise)
    else:
        amp_noise = np.full(f.shape, np.nan)

    units = "V*s"
    calibrated = False
    if calibration is not None:
        gain = calibration.interp(f)
        with np.errstate(divide="ignore", invalid="ignore"):
            conv = gain * (2.0 * np.pi * f) ** calibration.k
            conv = np.where(np.isfinite(conv) & (conv > 0), conv, np.nan)
            amp_sig = amp_sig / conv
            amp_noise = amp_noise / conv
        units = calibration.output_units
        calibrated = calibration.unit != "V/V"

    if t_star_s is not None and attenuation is not None:
        raise ValueError("give either t_star_s (constant) or attenuation (Q^-1 table), not both")
    if t_star_s is not None:
        if not (np.isfinite(t_star_s) and t_star_s >= 0):
            raise ValueError("t_star_s must be finite and >= 0")
        corr = np.exp(np.pi * f * float(t_star_s))
        amp_sig = amp_sig * corr
        amp_noise = amp_noise * corr
    if attenuation is not None:
        corr = attenuation.factor(f)
        amp_sig = amp_sig * corr
        amp_noise = amp_noise * corr

    meta = {
        "n": int(n), "nfft": int(nfft), "fs": float(window.fs), "dt": float(dt),
        "n_pre": window.n_pre, "n_post": window.n_post, "window_s": float(window.window_s),
        "i_trigger": int(window.i_trigger), "trigger_time": float(window.trigger_time),
        "taper": taper, "alpha": float(alpha),
        "taper_energy_fraction": taper_fraction, "transient_in_flat_region": in_flat,
        "baseline_mode": window.baseline_mode, "baseline_region": window.baseline_region,
        "highpass_hz": window.highpass_hz,
        "step_removed": window.step_removed, "step_amplitude": window.step_amplitude,
        "step_centre_sample": window.step_centre, "step_scale_samples": window.step_scale,
        "step_fit_scale_samples": window.step_fit_scale,
        "step_width_fraction": window.step_width_fraction,
        "step_centre_method": window.step_centre_method,
        "step_residual": window.step_residual,
        "noise_available": window.noise_available, "noise_fraction": window.noise_fraction,
        "noise_region": window.noise_region,
        "calibration": calibration.as_dict() if calibration is not None else None,
        "t_star_s": None if t_star_s is None else float(t_star_s),
        "attenuation": attenuation.as_dict() if attenuation is not None else None,
        "amplitude_convention": "|rfft(w*x)|*dt one-sided, no coherent-gain division; "
                                "noise / sqrt(mean(w^2)), scaled to the signal length",
    }
    spec = SpectrumResult(f=f, amp_signal=amp_sig, amp_noise=amp_noise, amp_units=units,
                          f_binned=np.array([]), amp_binned=np.array([]),
                          noise_binned=np.array([]), snr=np.array([]),
                          counts=np.array([], dtype=int), calibrated=calibrated,
                          t_star_s=None if t_star_s is None else float(t_star_s), meta=meta)
    return bin_spectrum(spec, fmin, fmax, bins_per_decade, noise_subtract, estimator)


def _bin_estimate(idx: np.ndarray, values: np.ndarray, nb: int, estimator: str) -> np.ndarray:
    """Per-bin ``'logmean'`` (geometric mean) or ``'median'`` of positive finite
    ``values`` whose bins are ``idx``; NaN for a bin without any."""
    ok = np.isfinite(values) & (values > 0)
    idx, values = idx[ok], values[ok]
    out = np.full(nb, np.nan)
    if idx.size == 0:
        return out
    if estimator == "logmean":
        n = np.bincount(idx, minlength=nb)[:nb]
        s = np.bincount(idx, weights=np.log(values), minlength=nb)[:nb]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(n > 0, np.exp(s / np.maximum(n, 1)), np.nan)
        return out
    order = np.argsort(idx, kind="stable")
    sorted_idx, sorted_values = idx[order], values[order]
    bounds = np.searchsorted(sorted_idx, np.arange(nb + 1))
    for b in range(nb):
        seg = sorted_values[bounds[b]:bounds[b + 1]]
        if seg.size:
            out[b] = float(np.median(seg))
    return out


def bin_spectrum(spectrum: SpectrumResult, fmin: Optional[float] = None,
                 fmax: Optional[float] = None, bins_per_decade: int = 30,
                 noise_subtract: bool = False, estimator: str = "rms") -> SpectrumResult:
    """Average power in log-spaced bins over ``[fmin, fmax]``.

    Per bin: ``amp = sqrt(mean(amp_signal**2))`` over finite raw bins, likewise
    for the noise, ``snr = amp / noise``, ``counts`` = raw bins used.  Empty
    bins are dropped.  ``noise_subtract=True`` uses
    ``amp = sqrt(max(mean(signal power) - mean(noise power), 0))`` instead,
    removing the quadrature bias of additive noise (bins with no noise
    estimate are left as they are; ``meta['noise_subtracted']`` says whether
    it was applied).  ``estimator`` (``BIN_ESTIMATORS``): ``'rms'`` is the
    above; ``'logmean'`` takes the geometric mean of the raw amplitudes per
    bin and ``'median'`` their median, for the signal and the noise alike,
    with no noise subtraction (``meta['estimator']`` records the choice).
    Returns a new SpectrumResult with the binned fields.
    """
    if estimator not in BIN_ESTIMATORS:
        raise ValueError(f"estimator must be one of {BIN_ESTIMATORS}, got {estimator!r}")
    f = spectrum.f
    positive = f[f > 0]
    if positive.size == 0:
        raise ValueError("spectrum has no positive frequencies")
    fmin = float(positive[0]) if fmin is None else float(fmin)
    fmax = float(f[-1]) if fmax is None else float(fmax)
    if not (fmin > 0 and fmax > fmin):
        raise ValueError("need 0 < fmin < fmax")
    bpd = int(bins_per_decade)
    if bpd < 1:
        raise ValueError("bins_per_decade must be >= 1")
    decades = math.log10(fmax) - math.log10(fmin)
    n_edges = int(math.ceil(decades * bpd)) + 1
    edges = np.logspace(math.log10(fmin), math.log10(fmax), max(n_edges, 2))
    edges[0], edges[-1] = fmin, fmax  # include both ends exactly

    inside = (f >= fmin) & (f <= fmax)
    idx = np.clip(np.digitize(f, edges) - 1, 0, edges.size - 2)
    idx = np.where(inside, idx, -1)
    sig_ok = inside & np.isfinite(spectrum.amp_signal)
    noise_ok = inside & np.isfinite(spectrum.amp_noise)
    nb = edges.size - 1

    counts = np.bincount(idx[sig_ok], minlength=nb)[:nb]
    power = np.bincount(idx[sig_ok], weights=spectrum.amp_signal[sig_ok] ** 2, minlength=nb)[:nb]
    logf = np.bincount(idx[sig_ok], weights=np.log(f[sig_ok]), minlength=nb)[:nb]
    n_counts = np.bincount(idx[noise_ok], minlength=nb)[:nb]
    n_power = np.bincount(idx[noise_ok], weights=spectrum.amp_noise[noise_ok] ** 2, minlength=nb)[:nb]

    keep = counts > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        f_b = np.exp(logf[keep] / counts[keep])
        if estimator == "rms":
            mean_power = power[keep] / counts[keep]
            has_noise = n_counts[keep] > 0
            noise_power = np.where(has_noise, n_power[keep] / np.maximum(n_counts[keep], 1), np.nan)
            subtracted = bool(noise_subtract) and bool(np.any(has_noise))
            if subtracted:
                mean_power = np.where(has_noise, np.maximum(mean_power - noise_power, 0.0), mean_power)
            amp_b = np.sqrt(mean_power)
            noise_b = np.sqrt(noise_power)
        else:
            amp_b = _bin_estimate(idx[sig_ok], spectrum.amp_signal[sig_ok], nb, estimator)[keep]
            noise_b = _bin_estimate(idx[noise_ok], spectrum.amp_noise[noise_ok], nb, estimator)[keep]
            subtracted = False
        snr = amp_b / noise_b
    meta = dict(spectrum.meta)
    meta.update({"fmin": fmin, "fmax": fmax, "bins_per_decade": bpd,
                 "noise_subtract": bool(noise_subtract), "noise_subtracted": subtracted,
                 "estimator": estimator})
    return replace(spectrum, f_binned=f_b, amp_binned=amp_b, noise_binned=noise_b, snr=snr,
                   counts=counts[keep].astype(int), meta=meta)


# ---------------------------------------------------------------------------
# omega-n fit
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpectralFit:
    omega0: float               # plateau, in the spectrum's amp_units
    fc: float                   # corner frequency (Hz)
    n: float                    # high-frequency fall-off exponent
    n_fixed: bool
    rms: float                  # weighted rms of ln residuals
    n_bins: int                 # bins used
    band_used: tuple            # (lo, hi) Hz of the used bins
    converged: bool
    at_bounds: dict             # {'ln_omega0': False, 'ln_fc': bool, 'n': bool}
    band_limited: bool          # fc > band_hi/2 or fc < 2*band_lo
    valid: bool
    reason: str = ""
    loss: str = "linear"
    snr_min: Optional[float] = None
    status: int = -1
    optimality: float = float("nan")
    amp_units: str = ""
    warnings: tuple = ()        # human-readable caveats (taper, band, bounds, convergence)
    n_bounds: tuple = N_BOUNDS  # (n_min, n_max) a free n was kept inside

    def model(self, f) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        if not self.valid:
            return np.full(f.shape, np.nan)
        return self.omega0 / (1.0 + (f / self.fc) ** self.n)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["band_used"] = list(self.band_used)
        d["warnings"] = list(self.warnings)
        d["n_bounds"] = list(self.n_bounds)
        d["version"] = RESULT_VERSION
        return d


def _invalid_fit(reason: str, n_bins: int = 0, n_fixed: bool = True, loss: str = "linear",
                 snr_min: Optional[float] = None, units: str = "", n_bounds: tuple = N_BOUNDS) -> SpectralFit:
    nan = float("nan")
    return SpectralFit(nan, nan, nan, n_fixed, nan, int(n_bins), (nan, nan), False,
                       {"ln_omega0": False, "ln_fc": False, "n": False}, False, False, reason,
                       loss, snr_min, -1, nan, units, n_bounds=n_bounds)


def _n_bounds(n_bounds) -> tuple:
    """``(n_min, n_max)`` as floats, or ValueError."""
    try:
        n_lo, n_hi = (float(v) for v in n_bounds)
    except (TypeError, ValueError):
        raise ValueError(f"n_bounds must be (n_min, n_max), got {n_bounds!r}") from None
    if not (math.isfinite(n_lo) and math.isfinite(n_hi) and 0.0 < n_lo < n_hi):
        raise ValueError(f"n_bounds must satisfy 0 < n_min < n_max, got {n_bounds!r}")
    return n_lo, n_hi


def _model_ln(q, lnf, n_fixed_value):
    if n_fixed_value is None:
        ln_o, ln_fc, n = q
    else:
        ln_o, ln_fc = q
        n = n_fixed_value
    return ln_o - np.log1p(np.exp(n * (lnf - ln_fc)))


def fit_omega_n(spectrum: SpectrumResult, fmin: Optional[float] = None, fmax: Optional[float] = None,
                snr_min: Optional[float] = 3.0, n_fixed: bool = True, n0: float = 2.0,
                loss: str = "linear", bins_per_decade: Optional[int] = None,
                n_bounds: tuple = N_BOUNDS) -> SpectralFit:
    """Fit ``ln A = ln Omega0 - ln(1 + (f/fc)**n)`` to the binned spectrum.

    Uses bins inside ``[fmin, fmax]`` (default: the binned range) with finite
    positive amplitude and ``snr >= snr_min`` (``snr_min=None`` disables the
    gate; bins with NaN SNR are dropped when it is on).  Requires at least
    ``MIN_FIT_BINS`` bins.  Parameters ``(ln Omega0, ln fc[, n])``; ``ln fc``
    is bounded to the used band, a free ``n`` to ``n_bounds = (n_min, n_max)``
    (default ``N_BOUNDS``; ``at_bounds['n']`` refers to them).  Weights are
    ``sqrt(count)``.  ``converged`` requires a gradient/cost-based
    termination and a scaled optimality at most ``CONVERGED_REL_OPTIMALITY``
    times the residual norm (or below ``CONVERGED_ABS_OPTIMALITY``).
    ``warnings`` lists caveats (transient outside the taper's flat region,
    band-limited corner, parameters at bounds, no convergence).  Never
    raises on bad data.
    """
    units = spectrum.amp_units
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")
    n_lo, n_hi = _n_bounds(n_bounds)
    bounds = (n_lo, n_hi)
    try:
        if bins_per_decade is not None:
            spectrum = bin_spectrum(spectrum, fmin, fmax, bins_per_decade,
                                    bool(spectrum.meta.get("noise_subtract", False)),
                                    str(spectrum.meta.get("estimator", "rms")))
        fb, ab, snr, counts = spectrum.f_binned, spectrum.amp_binned, spectrum.snr, spectrum.counts
        if fb.size == 0:
            return _invalid_fit("spectrum has no binned data", 0, n_fixed, loss, snr_min, units, bounds)
        lo = float(fb.min()) if fmin is None else float(fmin)
        hi = float(fb.max()) if fmax is None else float(fmax)
        use = (fb >= lo) & (fb <= hi) & np.isfinite(ab) & (ab > 0) & (counts > 0)
        if snr_min is not None:
            if not np.any(np.isfinite(snr)):
                return _invalid_fit("noise spectrum unavailable; pass snr_min=None to fit without SNR gating",
                                    0, n_fixed, loss, snr_min, units, bounds)
            use &= np.isfinite(snr) & (snr >= float(snr_min))
        m = int(np.count_nonzero(use))
        if m < MIN_FIT_BINS:
            return _invalid_fit(f"only {m} usable bins (need {MIN_FIT_BINS}) in "
                                f"[{lo:.3g}, {hi:.3g}] Hz with SNR >= {snr_min}",
                                m, n_fixed, loss, snr_min, units, bounds)
        lnf = np.log(fb[use])
        lna = np.log(ab[use])
        wts = np.sqrt(counts[use].astype(float))
        band = (float(fb[use].min()), float(fb[use].max()))
        lnf_lo, lnf_hi = math.log(band[0]), math.log(band[1])

        # initial guess
        k = max(1, int(math.ceil(0.2 * m)))
        order = np.argsort(lnf)
        ln_o0 = float(np.median(lna[order[:k]]))
        below = order[lna[order] <= ln_o0 - math.log(2.0)]
        ln_fc0 = float(lnf[below[0]]) if below.size else 0.5 * (lnf_lo + lnf_hi)
        eps = 1e-6 * (lnf_hi - lnf_lo)
        ln_fc0 = min(max(ln_fc0, lnf_lo + eps), lnf_hi - eps)
        n_start = min(max(float(n0), n_lo + 1e-6), n_hi - 1e-6)

        # Solve in centred parameters (ln(A/A0), ln(f/f_mid)) so the
        # termination tests are independent of the amplitude units and of
        # the absolute frequency scale.
        lnf_mid = 0.5 * (lnf_lo + lnf_hi)
        lnf_c = lnf - lnf_mid
        lna_c = lna - ln_o0
        if n_fixed:
            q0 = np.array([0.0, ln_fc0 - lnf_mid])
            lb = np.array([-np.inf, lnf_lo - lnf_mid])
            ub = np.array([np.inf, lnf_hi - lnf_mid])
            n_val = float(n0)
        else:
            q0 = np.array([0.0, ln_fc0 - lnf_mid, n_start])
            lb = np.array([-np.inf, lnf_lo - lnf_mid, n_lo])
            ub = np.array([np.inf, lnf_hi - lnf_mid, n_hi])
            n_val = None

        def resid(q):
            return wts * (lna_c - _model_ln(q, lnf_c, n_val))

        # xtol is disabled: on near-exact data the step test can fire one
        # iteration before the gradient test, which would read as a stall.
        with warnings.catch_warnings():   # scipy's trf warns on degenerate spectra; results are checked below
            warnings.simplefilter("ignore", RuntimeWarning)
            res = least_squares(resid, q0, bounds=(lb, ub), loss=loss, f_scale=1.0,
                                max_nfev=5000, x_scale="jac", xtol=None)
        q = res.x
        ln_o, ln_fc = float(q[0]) + ln_o0, float(q[1]) + lnf_mid
        n_hat = float(q[2]) if not n_fixed else float(n0)
        if not (np.isfinite(ln_o) and np.isfinite(ln_fc) and np.isfinite(n_hat)):
            return _invalid_fit("fit produced non-finite parameters", m, n_fixed, loss, snr_min, units, bounds)
        r = lna_c - _model_ln(q, lnf_c, n_val)
        rms = float(np.sqrt(np.sum(wts ** 2 * r ** 2) / np.sum(wts ** 2)))
        omega0, fc = math.exp(ln_o), math.exp(ln_fc)
        span = lnf_hi - lnf_lo
        at_bounds = {
            "ln_omega0": False,
            "ln_fc": bool(min(ln_fc - lnf_lo, lnf_hi - ln_fc) <= 0.01 * span),
            "n": bool((not n_fixed) and min(n_hat - n_lo, n_hi - n_hat) <= 0.01 * (n_hi - n_lo)),
        }
        # least_squares reports the infinity norm of the scaled (projected)
        # gradient, bounded by the residual norm ||r||; at an optimum the
        # residual is orthogonal to the Jacobian, so optimality / ||r|| is a
        # cosine: <= 1.6e-3 on genuine optima, >= 0.05 when stopped early.
        resid_norm = float(np.sqrt(np.sum(wts ** 2 * r ** 2)))
        converged = bool(res.status in (1, 2, 4) and np.isfinite(res.optimality)
                         and res.optimality <= max(CONVERGED_ABS_OPTIMALITY,
                                                   CONVERGED_REL_OPTIMALITY * resid_norm))
        band_limited = bool(fc > band[1] / 2.0 or fc < 2.0 * band[0])
        notes = []
        frac = spectrum.meta.get("taper_energy_fraction")
        if spectrum.meta.get("transient_in_flat_region") is False:
            notes.append(f"transient not in the taper's flat region (taper_energy_fraction="
                            f"{float(frac):.3f} < {TAPER_FLAT_FRACTION}); Omega0 is attenuated" if frac is not None
                            and np.isfinite(frac) else "transient not in the taper's flat region")
        if band_limited:
            notes.append(f"band_limited: fc={fc:.3g} Hz not resolved inside "
                            f"[{band[0]:.3g}, {band[1]:.3g}] Hz")
        for name, flag in at_bounds.items():
            if flag:
                notes.append(f"{name} at bound")
        if not converged:
            notes.append(f"not converged (status {int(res.status)}, optimality {float(res.optimality):.2g})")
        return SpectralFit(omega0, fc, n_hat, bool(n_fixed), rms, m, band, converged, at_bounds,
                           band_limited, True, "", loss, snr_min, int(res.status),
                           float(res.optimality), units, tuple(notes), bounds)
    except Exception as exc:  # never raise inside the fit on bad data
        return _invalid_fit(f"fit failed: {type(exc).__name__}: {exc}", 0, n_fixed, loss, snr_min, units, bounds)


# ---------------------------------------------------------------------------
# synthetic pulse (for tests and the view's self-test)
# ---------------------------------------------------------------------------
def brune_spectrum(f, omega0: float, fc: float, n: float = 2.0) -> np.ndarray:
    """Analytic ``Omega0 / (1 + (f/fc)**n)``."""
    f = np.asarray(f, dtype=float)
    return omega0 / (1.0 + (np.abs(f) / fc) ** n)


def brune_pulse(fs: float, n: int, omega0: float, fc: float, t0: float,
                derivative: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise ``n`` samples at ``fs`` of a displacement pulse whose
    spectrum is exactly ``Omega0 / (1 + (f/fc)**2)`` (a two-sided exponential
    ``Omega0*pi*fc*exp(-2*pi*fc*|t - t0|)``), by inverse FFT of the analytic
    spectrum with a linear phase centring it at ``t0``.  ``derivative=1``
    (velocity) or ``2`` (acceleration) multiplies by ``(2*pi*i*f)**derivative``.
    Returns ``(t, x)`` with ``t = arange(n) / fs``."""
    n = int(n)
    dt = 1.0 / float(fs)
    f = np.fft.rfftfreq(n, dt)
    spec = brune_spectrum(f, omega0, fc) * np.exp(-2j * np.pi * f * t0)
    spec = spec * (2j * np.pi * f) ** int(derivative)
    x = np.fft.irfft(spec, n) / dt
    return np.arange(n) * dt, x


__all__ = [
    "RESULT_VERSION", "BASELINE_MODES", "TAPER_KINDS", "LOSSES", "BIN_ESTIMATORS", "CALIBRATION_UNITS",
    "STEP_CENTRE_METHODS", "STEP_MIN_SCALE", "TAPER_FLAT_FRACTION", "N_BOUNDS", "NOISE_REGIONS",
    "CONVERGED_REL_OPTIMALITY", "CONVERGED_ABS_OPTIMALITY",
    "WindowResult", "Calibration", "SpectrumResult", "SpectralFit",
    "sampling_rate", "extract_window", "taper_window", "load_calibration_csv",
    "csv_gain_is_db", "Attenuation", "load_q_csv",
    "t_star_from_q", "compute_spectrum", "bin_spectrum", "fit_omega_n",
    "brune_spectrum", "brune_pulse",
]
