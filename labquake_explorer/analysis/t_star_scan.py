"""A constant attenuation ``t*`` shared by many records of one sensor, found
by a grid search.

The path from the fault to one sensor is the same in every event of a run
(and across experiments on the same sample), so one ``t*`` per sensor is a
reasonable model of the attenuation.  For each ``t*`` of the grid every
record's binned spectrum is corrected by ``exp(pi f t*)`` and refitted with
the record's own settings (band, n fixed or free, loss, SNR gate); the cost
is the summed weighted squared ln residual, ``sum(rms^2 * n_bins)`` over the
records, and the best ``t*`` is the grid point with the least cost.  A
``t*`` that has to go negative or to the end of the grid is a sign that the
data cannot separate attenuation from a steeper source spectrum.

``Q = T / t*`` with ``T`` the travel time.  The fits here correct the binned
amplitudes (``fit_omega_n`` never sees the raw grid); the PZT Spectrum view's
``t*`` corrects the raw FFT before binning, which agrees to about a percent.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from labquake_explorer.analysis.spectrum import SpectralFit, SpectrumResult, fit_omega_n

RESULT_VERSION = 1
DEFAULT_T_STAR_GRID_S = tuple(float(v) for v in np.arange(-20e-6, 150.1e-6, 5e-6))


@dataclass(frozen=True)
class RecordInput:
    """One record of the scan: its binned spectrum and fit settings."""
    spectrum: SpectrumResult
    fmin_hz: Optional[float] = None
    fmax_hz: Optional[float] = None
    snr_min: Optional[float] = None
    n_fixed: bool = True
    loss: str = "linear"
    n_bounds: tuple = (1.0, 4.0)
    label: str = ""

    def fit(self, t_star_s: float = 0.0) -> SpectralFit:
        return fit_omega_n(corrected(self.spectrum, t_star_s), fmin=self.fmin_hz, fmax=self.fmax_hz,
                           snr_min=self.snr_min, n_fixed=self.n_fixed, loss=self.loss, n_bounds=self.n_bounds)


def corrected(spectrum: SpectrumResult, t_star_s: float) -> SpectrumResult:
    """The binned spectrum with ``exp(pi f t*)`` applied to the signal and the
    noise (the SNR is unchanged); the spectrum itself for ``t* = 0``."""
    t = float(t_star_s)
    if t == 0.0:
        return spectrum
    gain = np.exp(np.pi * spectrum.f_binned * t)
    meta = dict(spectrum.meta)
    meta["t_star_s"] = (float(meta.get("t_star_s") or 0.0)) + t
    return replace(spectrum, amp_binned=spectrum.amp_binned * gain, noise_binned=spectrum.noise_binned * gain,
                   t_star_s=(spectrum.t_star_s or 0.0) + t, meta=meta)


def spectrum_from_record(record: dict) -> SpectrumResult:
    """A ``SpectrumResult`` holding the binned arrays a saved record stores
    (``record['spectrum']``), enough for ``fit_omega_n``; the raw grid is empty."""
    spec = record.get("spectrum") if isinstance(record, dict) else None
    if not isinstance(spec, dict):
        raise ValueError("the record has no spectrum")
    fb = np.asarray(spec.get("f_binned", []), dtype=float).ravel()
    ab = np.asarray(spec.get("amp_binned", []), dtype=float).ravel()
    if fb.size == 0 or ab.size != fb.size:
        raise ValueError("the record's binned spectrum is empty or inconsistent")
    nb = np.asarray(spec.get("noise_binned", []), dtype=float).ravel()
    snr = np.asarray(spec.get("snr", []), dtype=float).ravel()
    counts = np.asarray(spec.get("counts", []), dtype=float).ravel()
    meta = dict(spec.get("meta") or {})
    return SpectrumResult(f=np.array([]), amp_signal=np.array([]), amp_noise=np.array([]),
                          amp_units=str(spec.get("amp_units") or ""),
                          f_binned=fb, amp_binned=ab,
                          noise_binned=nb if nb.size == fb.size else np.full(fb.shape, np.nan),
                          snr=snr if snr.size == fb.size else np.full(fb.shape, np.nan),
                          counts=(counts if counts.size == fb.size else np.ones(fb.shape)).astype(int),
                          calibrated=bool(spec.get("calibrated", False)),
                          t_star_s=None if spec.get("t_star_s") is None else float(spec.get("t_star_s")),
                          meta=meta)


def record_input(record: dict, label: str = "") -> RecordInput:
    """The scan input of one saved PZT Spectrum record (its binned spectrum and
    its own fit settings)."""
    g = record.get
    n_max = g("n_max")
    try:
        n_hi = float(n_max) if n_max is not None else 4.0
    except (TypeError, ValueError):
        n_hi = 4.0
    snr = g("snr_min")
    try:
        snr = None if snr is None else float(snr)
    except (TypeError, ValueError):
        snr = None
    return RecordInput(spectrum_from_record(record),
                       fmin_hz=None if g("fmin_hz") is None else float(g("fmin_hz")),
                       fmax_hz=None if g("fmax_hz") is None else float(g("fmax_hz")),
                       snr_min=snr, n_fixed=bool(g("n_fixed", True)),
                       loss=str(g("loss") or "linear"), n_bounds=(1.0, max(n_hi, 1.0 + 1e-6)), label=label)


def cost_of(fit: SpectralFit) -> float:
    """The weighted squared ln residual summed over the fit's bins; NaN when invalid."""
    return float(fit.rms ** 2 * fit.n_bins) if fit.valid else float("nan")


@dataclass(frozen=True)
class TStarScan:
    t_star_grid_s: np.ndarray
    cost: np.ndarray                  # summed over the records, NaN where every fit failed
    n_valid: np.ndarray               # records with a valid fit at each t*
    best_t_star_s: float              # NaN when no grid point has a valid cost
    best_index: int
    n_records: int

    @property
    def at_grid_end(self) -> bool:
        """The minimum sits on the first or last grid point: the scan did not bracket it."""
        return self.best_index in (0, self.t_star_grid_s.size - 1) if self.best_index >= 0 else False

    def as_dict(self) -> dict:
        return {"version": RESULT_VERSION, "t_star_grid_s": self.t_star_grid_s.tolist(), "cost": self.cost.tolist(),
                "n_valid": self.n_valid.tolist(), "best_t_star_s": float(self.best_t_star_s),
                "best_index": int(self.best_index), "n_records": int(self.n_records),
                "at_grid_end": bool(self.at_grid_end)}


def scan_t_star(records: Sequence[RecordInput], t_star_grid_s: Sequence[float] = DEFAULT_T_STAR_GRID_S) -> TStarScan:
    """Grid-search the ``t*`` shared by ``records`` (one sensor's).  The cost at
    a grid point sums the records whose fit is valid there; a point where
    fewer records fit than at the best-covered point is not preferred over it
    (the costs are compared only among the points with the fullest coverage)."""
    grid = np.asarray(list(t_star_grid_s), dtype=float).ravel()
    if grid.size == 0:
        raise ValueError("the t* grid is empty")
    if not np.all(np.isfinite(grid)):
        raise ValueError("the t* grid must be finite")
    cost = np.full(grid.size, np.nan)
    n_valid = np.zeros(grid.size, dtype=int)
    for i, t in enumerate(grid):
        values = [cost_of(r.fit(float(t))) for r in records]
        finite = [v for v in values if np.isfinite(v)]
        n_valid[i] = len(finite)
        if finite:
            cost[i] = float(np.sum(finite))
    best = -1
    if np.any(n_valid > 0):
        full = n_valid == n_valid.max()
        candidates = np.where(full & np.isfinite(cost))[0]
        if candidates.size:
            best = int(candidates[np.argmin(cost[candidates])])
    return TStarScan(grid, cost, n_valid, float(grid[best]) if best >= 0 else float("nan"), best, len(records))


__all__ = ["RESULT_VERSION", "DEFAULT_T_STAR_GRID_S", "RecordInput", "corrected", "spectrum_from_record",
           "record_input", "cost_of", "TStarScan", "scan_t_star"]
