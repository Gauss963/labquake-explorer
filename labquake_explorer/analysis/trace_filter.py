"""Filters for the traces of a full-rate record (the Pick Arrivals view).

``filter_traces(data, sample_rate_hz, settings)`` filters each row of
``data`` by ``settings['kind']``:

* ``off``: unchanged;
* ``savgol``: Savitzky-Golay of ``window_samples`` and ``polyorder`` (2);
* ``lowpass``, ``highpass``, ``bandpass``: Butterworth of ``order`` with
  corners ``low_hz`` and/or ``high_hz``, run forwards and backwards
  (``zero_phase``, the default: no delay, but a sharp onset spreads earlier by
  about one period of the highest corner) or forwards only (causal: the onset
  never moves earlier, but is delayed by the filter's group delay).

The settings are plain values, so they are saved with the picks as they are.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

KIND_OFF = "off"
KIND_SAVGOL = "savgol"
KIND_LOWPASS = "lowpass"
KIND_HIGHPASS = "highpass"
KIND_BANDPASS = "bandpass"
KINDS = (KIND_OFF, KIND_SAVGOL, KIND_LOWPASS, KIND_HIGHPASS, KIND_BANDPASS)
KIND_LABELS = {KIND_OFF: "off", KIND_SAVGOL: "Savitzky-Golay", KIND_LOWPASS: "Butterworth low-pass",
               KIND_HIGHPASS: "Butterworth high-pass", KIND_BANDPASS: "Butterworth band-pass"}
BUTTERWORTH = (KIND_LOWPASS, KIND_HIGHPASS, KIND_BANDPASS)

SAVGOL_POLYORDER = 2
DEFAULT_BUTTER_ORDER = 4
MAX_BUTTER_ORDER = 10


def uses_low(kind: str) -> bool:
    """Whether the filter has a low corner (high-pass and band-pass)."""
    return kind in (KIND_HIGHPASS, KIND_BANDPASS)


def uses_high(kind: str) -> bool:
    """Whether the filter has a high corner (low-pass and band-pass)."""
    return kind in (KIND_LOWPASS, KIND_BANDPASS)


def check(settings: dict, sample_rate_hz: float) -> None:
    """Raise ValueError, with a reason a user can act on, when the settings
    cannot filter a record sampled at ``sample_rate_hz``."""
    kind = settings.get("kind", KIND_OFF)
    if kind not in KINDS:
        raise ValueError(f"unknown filter {kind!r}; use one of {KINDS}")
    if kind == KIND_SAVGOL:
        window = int(settings.get("window_samples", 0))
        polyorder = int(settings.get("polyorder", SAVGOL_POLYORDER))
        if window <= polyorder:
            raise ValueError(f"the Savitzky-Golay window must be longer than {polyorder} samples")
    if kind not in BUTTERWORTH:
        return
    order = int(settings.get("order", DEFAULT_BUTTER_ORDER))
    if not 1 <= order <= MAX_BUTTER_ORDER:
        raise ValueError(f"the order must be 1 to {MAX_BUTTER_ORDER}")
    nyquist = 0.5 * float(sample_rate_hz)
    corners = []
    if uses_low(kind):
        corners.append(("low corner", settings.get("low_hz")))
    if uses_high(kind):
        corners.append(("high corner", settings.get("high_hz")))
    for name, value in corners:
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"the {name} must be a frequency > 0")
        if float(value) >= nyquist:
            raise ValueError(f"the {name} must be below the Nyquist frequency, {nyquist / 1e3:g} kHz")
    if kind == KIND_BANDPASS and float(settings["low_hz"]) >= float(settings["high_hz"]):
        raise ValueError("the low corner must be below the high corner")


def filter_traces(data, sample_rate_hz: float, settings: dict) -> np.ndarray:
    """The rows of ``data`` filtered by ``settings`` (a new float array; see
    the module docstring).  Raises ValueError when the settings are unusable."""
    y = np.array(data, dtype=float, copy=True)
    kind = settings.get("kind", KIND_OFF)
    check(settings, sample_rate_hz)
    if kind == KIND_OFF or y.shape[-1] < 2:
        return y
    if kind == KIND_SAVGOL:
        window = min(int(settings["window_samples"]), y.shape[-1])
        polyorder = int(settings.get("polyorder", SAVGOL_POLYORDER))
        if window <= polyorder:
            return y
        return signal.savgol_filter(y, window, polyorder, axis=-1)
    order = int(settings.get("order", DEFAULT_BUTTER_ORDER))
    if kind == KIND_LOWPASS:
        wn = float(settings["high_hz"])
    elif kind == KIND_HIGHPASS:
        wn = float(settings["low_hz"])
    else:
        wn = [float(settings["low_hz"]), float(settings["high_hz"])]
    sos = signal.butter(order, wn, btype=kind, fs=float(sample_rate_hz), output="sos")
    finite = np.isfinite(y)
    filled = np.where(finite, y, 0.0)
    if settings.get("zero_phase", True):
        padlen = min(3 * (2 * len(sos) + 1), y.shape[-1] - 1)
        out = signal.sosfiltfilt(sos, filled, axis=-1, padlen=padlen)
    else:
        # start from the steady state of the first sample, so the record's offset does not ring
        zi = signal.sosfilt_zi(sos)[:, None, :] * filled[..., :1][None, ...] if y.ndim == 2 else \
            signal.sosfilt_zi(sos) * filled[0]
        out, _ = signal.sosfilt(sos, filled, axis=-1, zi=zi)
    return np.where(finite, out, np.nan)


def describe(settings: dict) -> str:
    """A short description, such as ``Butterworth band-pass 2-200 kHz, order 4, zero phase``."""
    kind = settings.get("kind", KIND_OFF)
    if kind == KIND_OFF:
        return "no filter"
    if kind == KIND_SAVGOL:
        return f"Savitzky-Golay, {int(settings['window_samples'])} samples, order {settings.get('polyorder', 2)}"
    if kind == KIND_LOWPASS:
        band = f"below {settings['high_hz'] / 1e3:g} kHz"
    elif kind == KIND_HIGHPASS:
        band = f"above {settings['low_hz'] / 1e3:g} kHz"
    else:
        band = f"{settings['low_hz'] / 1e3:g}-{settings['high_hz'] / 1e3:g} kHz"
    phase = "zero phase" if settings.get("zero_phase", True) else "causal"
    return f"{KIND_LABELS[kind]} {band}, order {int(settings.get('order', DEFAULT_BUTTER_ORDER))}, {phase}"


__all__ = ["KIND_OFF", "KIND_SAVGOL", "KIND_LOWPASS", "KIND_HIGHPASS", "KIND_BANDPASS", "KINDS", "KIND_LABELS",
           "BUTTERWORTH", "SAVGOL_POLYORDER", "DEFAULT_BUTTER_ORDER", "MAX_BUTTER_ORDER", "uses_low", "uses_high",
           "check", "filter_traces", "describe"]
