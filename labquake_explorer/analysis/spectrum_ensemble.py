"""Uncertainty ensemble of one omega-n fit: the same window refitted over a
grid of fit bands and, optionally, over alternative calibration tables (the
per-position responses of one receiver, say).

The band grid is ``fmins x fmaxs`` (every pair with ``fmin < fmax``); each
spectrum (one per calibration table) is fitted on every band with the same
settings as the record's own fit.  The members' ``omega0``, ``fc`` and ``n``
are summarised by their min, max, 16th / 84th percentiles and median over
the valid members (``stats``) and again over the valid members whose corner
was resolved inside their band (``stats_resolved``: a band whose fmin sits
at fc pins the fit at the edge and would widen the spread); the seismic
moment follows the plateau linearly for one
set of source constants, so ``with_moment`` scales it from the record's own
``M0 / Omega0``.  Pure numerics; the PZT Spectrum view stores
:meth:`Ensemble.as_dict` under ``record['ensemble']`` and Source Scaling
draws the percentiles as error bars.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from labquake_explorer.analysis.spectrum import SpectrumResult, fit_omega_n

RESULT_VERSION = 1
DEFAULT_FMINS_HZ = (1000.0, 1500.0, 2000.0, 3000.0)
DEFAULT_FMAXS_HZ = (15000.0, 22000.0, 36000.0, 50000.0)
STAT_KEYS = ("min", "max", "p16", "p84", "median")
MEMBER_FIELDS = ("omega0", "fc", "n")


def parse_frequency_list(text, default: Sequence[float] = ()) -> tuple:
    """Numbers (Hz) typed as ``'1000, 1500 2000'`` (commas, semicolons or blanks
    between them), sorted and without repeats; blank text gives ``default``.
    Raises ValueError on anything that is not a positive number."""
    parts = [p for p in str(text if text is not None else "").replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    if not parts:
        return tuple(float(v) for v in default)
    values = []
    for p in parts:
        try:
            v = float(p)
        except ValueError:
            raise ValueError(f"{p!r} is not a number") from None
        if not (np.isfinite(v) and v > 0):
            raise ValueError(f"frequencies must be positive, got {p!r}")
        values.append(v)
    return tuple(sorted(set(values)))


def band_grid(fmins: Sequence[float], fmaxs: Sequence[float]) -> list:
    """Every ``(fmin, fmax)`` of the grid with ``fmin < fmax``, fmin-major."""
    return [(float(lo), float(hi)) for lo in fmins for hi in fmaxs if float(lo) < float(hi)]


def stats(values, valid=None) -> dict:
    """``min``, ``max``, ``p16``, ``p84``, ``median`` and ``n`` of the finite
    values (those flagged ``valid`` when given); NaN with ``n = 0`` when none."""
    v = np.asarray(values, dtype=float).ravel()
    keep = np.isfinite(v)
    if valid is not None:
        keep &= np.asarray(valid, dtype=bool).ravel()
    v = v[keep]
    if v.size == 0:
        return {key: float("nan") for key in STAT_KEYS} | {"n": 0}
    return {"min": float(v.min()), "max": float(v.max()), "p16": float(np.percentile(v, 16)),
            "p84": float(np.percentile(v, 84)), "median": float(np.median(v)), "n": int(v.size)}


@dataclass(frozen=True)
class Ensemble:
    """The members of one ensemble (arrays of one entry per member)."""
    bands_hz: tuple               # the band grid, ((fmin, fmax), ...)
    calibrations: tuple           # labels of the calibration tables, the record's own first
    fmin_hz: np.ndarray
    fmax_hz: np.ndarray
    calibration: np.ndarray       # index into ``calibrations``
    omega0: np.ndarray
    fc: np.ndarray
    n: np.ndarray
    valid: np.ndarray
    band_limited: np.ndarray
    moment: Optional[np.ndarray] = None   # seismic moment per member, once ``with_moment`` was applied

    @property
    def n_members(self) -> int:
        return int(self.valid.size)

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def resolved(self) -> np.ndarray:
        """Valid members whose corner is resolved inside their band."""
        return self.valid & ~self.band_limited

    @property
    def n_resolved(self) -> int:
        return int(np.count_nonzero(self.resolved))

    def stats_of(self, field: str, resolved: bool = False) -> dict:
        """The statistics of ``field`` over the valid members, or with
        ``resolved`` over the valid members that are not band-limited."""
        values = self.moment if field == "seismic_moment_nm" else getattr(self, field)
        if values is None:
            return stats([])
        return stats(values, self.resolved if resolved else self.valid)

    def with_moment(self, m0_per_omega0: float) -> "Ensemble":
        """The members' seismic moments, ``omega0`` times the record's own
        ``M0 / Omega0`` (the moment is linear in the plateau for fixed source
        constants; 1 for a moment spectrum)."""
        scale = float(m0_per_omega0)
        moment = self.omega0 * scale if np.isfinite(scale) and scale > 0 else np.full(self.omega0.shape, np.nan)
        return Ensemble(self.bands_hz, self.calibrations, self.fmin_hz, self.fmax_hz, self.calibration,
                        self.omega0, self.fc, self.n, self.valid, self.band_limited, moment)

    def as_dict(self) -> dict:
        members = {
            "fmin_hz": self.fmin_hz.tolist(), "fmax_hz": self.fmax_hz.tolist(),
            "calibration": self.calibration.tolist(),
            "omega0": self.omega0.tolist(), "fc": self.fc.tolist(), "n": self.n.tolist(),
            "valid": self.valid.tolist(), "band_limited": self.band_limited.tolist(),
        }
        fields = list(MEMBER_FIELDS)
        if self.moment is not None:
            members["seismic_moment_nm"] = self.moment.tolist()
            fields.append("seismic_moment_nm")
        return {
            "version": RESULT_VERSION,
            "bands_hz": [list(b) for b in self.bands_hz],
            "calibrations": list(self.calibrations),
            "n_members": self.n_members, "n_valid": self.n_valid, "n_resolved": self.n_resolved,
            "members": members,
            "stats": {field: self.stats_of(field) for field in fields},
            "stats_resolved": {field: self.stats_of(field, resolved=True) for field in fields},
        }


def fit_ensemble(spectra: Sequence[tuple], bands: Sequence[tuple], **fit_kw) -> Ensemble:
    """Fit every ``(label, SpectrumResult)`` of ``spectra`` on every band of
    ``bands`` with ``fit_omega_n(spectrum, fmin, fmax, **fit_kw)``.  The
    spectra must be binned over at least the widest band."""
    labels, fmin, fmax, cal, omega0, fc, n, valid, limited = [], [], [], [], [], [], [], [], []
    for k, (label, spectrum) in enumerate(spectra):
        labels.append(str(label))
        for lo, hi in bands:
            fit = fit_omega_n(spectrum, fmin=lo, fmax=hi, **fit_kw)
            fmin.append(lo); fmax.append(hi); cal.append(k)
            omega0.append(fit.omega0 if fit.valid else float("nan"))
            fc.append(fit.fc if fit.valid else float("nan"))
            n.append(fit.n if fit.valid else float("nan"))
            valid.append(bool(fit.valid)); limited.append(bool(fit.band_limited))
    return Ensemble(tuple((float(lo), float(hi)) for lo, hi in bands), tuple(labels),
                    np.asarray(fmin, float), np.asarray(fmax, float), np.asarray(cal, int),
                    np.asarray(omega0, float), np.asarray(fc, float), np.asarray(n, float),
                    np.asarray(valid, bool), np.asarray(limited, bool))


__all__ = ["RESULT_VERSION", "DEFAULT_FMINS_HZ", "DEFAULT_FMAXS_HZ", "STAT_KEYS", "MEMBER_FIELDS",
           "parse_frequency_list", "band_grid", "stats", "Ensemble", "fit_ensemble"]
