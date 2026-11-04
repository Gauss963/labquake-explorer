"""Scaling of PZT source parameters across the events of one run.

For every event of a run the view collects the source parameters saved by
the PZT spectrum view (``spectrum['channels'][ch]['source']`` of each
waveform record, or of the strain block; the older event-level
``event['pzt_spectrum']`` map is read too:
seismic moment, Mw, source radius, stress drop; the corner frequency and its
quality flags come from the same channel record's ``fit``) together with the
mechanical results of the event analyser (``event['event_analysis']``:
``-delta_y`` and ``delta_x`` of a version 3 record run on stress against
slip, ``stress_drop`` and ``displacement`` of older records).  It plots a chosen quantity against seismic
moment on log-log axes, fits a power law with
:mod:`labquake_explorer.analysis.scaling` and, for the corner frequency,
draws the constant-stress-drop reference ``fc ~ M0^-1/3`` through the
median point.

Quality gating.  The spectrum module reports, per channel, whether the
corner frequency is band-limited (``fit['band_limited']``), sits at a bound
of the fit (``fit['at_bounds']['ln_fc']``) or comes from an unconverged
solver (``fit['converged']``), and the source module whether the far-field
assumption holds (``source['far_field_ok']``, from ``kR`` evaluated at the
LOW EDGE of the fitted band, i.e. the plateau).  A band-limited or at-bound
fc is a bound, not a measurement, and it biases the exponent toward zero, so
such records are EXCLUDED from the fit by default; they are still listed
(``flags`` column) and plotted as open symbols.  The *include flagged fits*
checkbox overrides the gating; the choice and the number of excluded events
are stored with the result.

The near-field flag is a WARNING, not a default exclusion: ``kR`` at the
plateau frequency (default fmin 1 kHz) is far below 3 for every laboratory
source-sensor distance (kR >= 3 at 1 kHz needs R >= 2.9 m for P at 6 km/s),
so gating on it would exclude every record without telling apart good and
bad ones.  Near-field records are listed with the ``kR`` code and counted in
the status line; the *exclude near-field* checkbox (off by default, stored
with the result) turns the warning into an exclusion.

Channel ``all`` combines, per event, the channels that pass the gating (all
channels when the override is on; the flagged ones only when nothing else is
available, in which case the row itself is flagged).  The combination is
done in log space on the PRIMARY quantities only -- median of ln M0 and of
ln fc over channels -- and the derived quantities are recomputed from them
(``Mw`` from M0; ``r = kVs / fc`` with ``kVs`` the median of ``r * fc`` of the
channels, i.e. their ``k * Vs``; ``stress drop = 7 M0 / (16 r^3)``) so that
the aggregated M0, r and stress drop obey the same Eshelby identity as a
single-channel record.  See :func:`combine_channels`.

Regression direction.  M0 on the x axis is derived from the spectral plateau
and carries an error comparable to that of fc, so the ordinary
least-squares exponent (y on x) is attenuated by regression dilution.  The
reduced-major-axis exponent (``PowerLawFit.exponent_rma``) is shown next to
it and stored.

Save writes the fit, the x/y pairs actually fitted, the gating choice, and
the phases and medium/model constants of the used records under
``runs/[r]/source_scaling`` (see :meth:`SourceScalingView.results`); a saved
result is restored (selections and summary) when the view is reopened.
"""
from __future__ import annotations

import copy
import json
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np

from labquake_explorer.analysis.scaling import (
    PowerLawFit, bootstrap_exponent, fit_power_law, reference_line,
)
from labquake_explorer.analysis import source as src
from labquake_explorer.analysis import t_star_scan as ts
from labquake_explorer.analysis.source import moment_magnitude, stress_drop_eshelby
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import RUN
from labquake_explorer.ui.views.base import RunView, event_list
from labquake_explorer.data.sources import waveform_blocks
from labquake_explorer.data.settings import get_setting, set_setting
from labquake_explorer.data.spectrum_store import DEFAULT_SET, ensure_set, set_channels, set_name, set_names
from labquake_explorer.ui.views.pzt_spectrum_view import (
    SETTING as PZT_SETTING, CURRENT_SET_SETTING, ensemble_stats, parse_channel_key,
)

RESULT_VERSION = 2
PZT_KEY = "pzt_spectrum"               # the older event-level map
SPECTRUM_KEY = "spectrum"              # per waveform record (or the strain block)
MECH_KEY = "event_analysis"
T_STAR_KEY = "t_star_scan"             # run-level record of the shared-t* scan
PLOT_SCALING = "scaling"
PLOT_T_STAR = "t* scan"
DEFAULT_T_STAR_GRID_US = "-20, 150, 5"  # from, to, step in microseconds
ALL_CHANNELS = "all"
REFERENCE_EXPONENT = -1.0 / 3.0
REFERENCE_LABEL = "fc ~ M0^-1/3"
N_BOOT = 1000
BOOT_SEED = 0

# Y-variable label -> record field
Y_FC = "fc (Hz)"
Y_STRESS_DROP = "stress drop (Pa)"
Y_RADIUS = "source radius (m)"
Y_MECH_STRESS_DROP = "mechanical stress drop (MPa)"
Y_SLIP = "coseismic slip (um)"
Y_VARIABLES: dict[str, str] = {
    Y_FC: "fc_hz",
    Y_STRESS_DROP: "stress_drop_pa",
    Y_RADIUS: "source_radius_m",
    Y_MECH_STRESS_DROP: "mech_stress_drop",
    Y_SLIP: "mech_slip",
}
#: The 16th / 84th percentiles of the record's uncertainty ensemble (``record['ensemble']``), NaN without one.
ERROR_FIELDS = ("fc_p16", "fc_p84", "m0_p16", "m0_p84")
PZT_FIELDS = ("seismic_moment_nm", "mw", "fc_hz", "stress_drop_pa", "source_radius_m", "kR") + ERROR_FIELDS
MECH_FIELDS = ("mech_stress_drop", "mech_slip")
FIELDS = PZT_FIELDS + MECH_FIELDS

# Quality flags of one channel record (text stored in rows/results) and their table codes.
FLAG_BAND_LIMITED = "band-limited"
FLAG_AT_BOUND = "fc at bound"
FLAG_NOT_CONVERGED = "not converged"
FLAG_NEAR_FIELD = "near-field (kR < 3 at band low edge)"
FLAG_CODES = {
    FLAG_BAND_LIMITED: "band",
    FLAG_AT_BOUND: "bound",
    FLAG_NOT_CONVERGED: "nconv",
    FLAG_NEAR_FIELD: "kR",
}
#: Flags that exclude a record from the fit by default (fit-quality flags).
#: ``FLAG_NEAR_FIELD`` joins them only when the user asks (see module docstring).
EXCLUDING_FLAGS = (FLAG_BAND_LIMITED, FLAG_AT_BOUND, FLAG_NOT_CONVERGED)
#: Medium / model constants that must agree across the records entering one fit.
#: Geometry-dependent entries (distance, radiation coefficient, free-surface
#: factor) legitimately differ per sensor and event and are not compared.
CONSTANT_KEYS = ("phase", "rho_kg_m3", "vp_m_s", "vs_m_s", "k")

TABLE_COLUMNS = (
    ("event", "event", 60),
    ("seismic_moment_nm", "M0 (N m)", 100),
    ("mw", "Mw", 70),
    ("fc_hz", "fc (Hz)", 90),
    ("stress_drop_pa", "stress drop (Pa)", 110),
    ("source_radius_m", "r (m)", 90),
    ("flags", "flags", 90),
)


# --------------------------------------------------------------------- helpers
def _finite(value) -> float:
    """``value`` as a float, NaN when it is not a finite number."""
    if isinstance(value, bool) or value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _is_false(value) -> bool:
    """True when ``value`` is an explicit False-like flag (bool, numpy bool, 0).

    ``None`` (flag unknown / not reported) is NOT treated as False.
    """
    if value is None:
        return False
    try:
        return not bool(value)
    except (TypeError, ValueError):
        return False


def _is_true(value) -> bool:
    if value is None:
        return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _float_list(value, n: Optional[int] = None) -> list[float]:
    """Stored sequence (list, tuple or ndarray) as floats, NaN where unusable.

    ``n`` forces the length (padded with NaN / truncated).  Never truth-tests
    the value: HDF5 hands stored lists back as arrays.
    """
    if value is None:
        items = []
    else:
        try:
            items = [_finite(v) for v in np.asarray(value, dtype=object).ravel()]
        except (TypeError, ValueError):
            items = []
    if n is not None:
        items = (items + [float("nan")] * n)[:n]
    return items


def _int_list(value) -> list[int]:
    """Stored sequence of indices as Python ints (unusable entries dropped)."""
    if value is None:
        return []
    try:
        items = np.asarray(value, dtype=object).ravel()
    except (TypeError, ValueError):
        return []
    out = []
    for v in items:
        f = _finite(v)
        if np.isfinite(f):
            out.append(int(f))
    return out


def excluding_flags(flags: list[str], exclude_near_field: bool = False) -> list[str]:
    """The subset of ``flags`` that excludes a record from the fit.

    Fit-quality flags (``EXCLUDING_FLAGS``) always exclude; ``FLAG_NEAR_FIELD``
    only when ``exclude_near_field``.
    """
    keep = set(EXCLUDING_FLAGS) | ({FLAG_NEAR_FIELD} if exclude_near_field else set())
    return [f for f in flags if f in keep]


def _channel_items(channels) -> list[tuple[str, dict]]:
    """``(channel name, record)`` pairs from a channels dict or list."""
    if isinstance(channels, dict):
        items = [(str(k), v) for k, v in channels.items()]
    elif isinstance(channels, (list, tuple)):
        items = [(str(i), v) for i, v in enumerate(channels)]
    else:
        return []
    return [(name, rec) for name, rec in items if isinstance(rec, dict)]


def _channel_fc(record: dict, source: dict) -> float:
    """Corner frequency of one channel record (record, fit, or source)."""
    for candidate in (record.get("fc_hz"), record.get("fc")):
        fc = _finite(candidate)
        if np.isfinite(fc):
            return fc
    fit = record.get("fit")
    if isinstance(fit, dict):
        for candidate in (fit.get("fc"), fit.get("fc_hz")):
            fc = _finite(candidate)
            if np.isfinite(fc):
                return fc
    return _finite(source.get("fc_hz"))


def channel_flags(record: dict, source: dict) -> list[str]:
    """Quality flags of one channel record (empty list = usable for scaling).

    Reads ``fit['band_limited']``, ``fit['at_bounds']['ln_fc']``,
    ``fit['converged']`` and ``source['far_field_ok']``.  A missing or
    ``None`` flag is treated as "not reported" and does not flag the record;
    only an explicit True (band-limited / at bound) or explicit False
    (converged / far_field_ok) does.
    """
    flags: list[str] = []
    fit = record.get("fit") if isinstance(record.get("fit"), dict) else {}
    if _is_true(fit.get("band_limited")):
        flags.append(FLAG_BAND_LIMITED)
    at_bounds = fit.get("at_bounds") if isinstance(fit.get("at_bounds"), dict) else {}
    if _is_true(at_bounds.get("ln_fc")):
        flags.append(FLAG_AT_BOUND)
    if "converged" in fit and _is_false(fit.get("converged")):
        flags.append(FLAG_NOT_CONVERGED)
    if "far_field_ok" in source and _is_false(source.get("far_field_ok")):
        flags.append(FLAG_NEAR_FIELD)
    return flags


def _ensemble_bounds(record: dict) -> dict:
    """``fc_p16``, ``fc_p84``, ``m0_p16``, ``m0_p84`` of a record's uncertainty
    ensemble (NaN without one, or without the moment in it): from the members
    whose corner is resolved inside the band (``stats_resolved``) when there
    are any, else from every valid member (``stats``)."""
    out = {field: float("nan") for field in ERROR_FIELDS}
    ensemble = record.get("ensemble") if isinstance(record, dict) else None
    stats, _ = ensemble_stats(ensemble) if isinstance(ensemble, dict) else ({}, "")
    if not isinstance(stats, dict) or not stats:
        return out
    for prefix, key in (("fc", "fc"), ("m0", "seismic_moment_nm")):
        entry = stats.get(key)
        if isinstance(entry, dict):
            out[f"{prefix}_p16"] = _finite(entry.get("p16"))
            out[f"{prefix}_p84"] = _finite(entry.get("p84"))
    return out


def _record_constants(source: dict) -> dict:
    """The medium/model constants (``CONSTANT_KEYS``) of one source record."""
    consts = source.get("constants") if isinstance(source.get("constants"), dict) else {}
    out = {key: consts.get(key) for key in CONSTANT_KEYS}
    if out["phase"] is None and source.get("phase") is not None:
        out["phase"] = source.get("phase")
    for key in CONSTANT_KEYS[1:]:
        if out[key] is not None:
            value = _finite(out[key])
            out[key] = value if np.isfinite(value) else None
    return out


def pzt_channel_records(event: dict, exclude_near_field: bool = False,
                        result_set: str = DEFAULT_SET) -> dict[str, dict]:
    """Per-channel PZT source values of one event: ``{channel: {field: ...}}``,
    from the result set ``result_set`` of each record's spectrum store (the
    version-1 store and the older event-level map count as ``'default'``).

    Only channels with a valid ``source`` dict are returned (``valid`` False
    or a numpy False skips the channel).  Every numeric field is a float
    (NaN when missing); in addition each record carries ``band_limited``,
    ``at_bound_fc``, ``converged``, ``far_field_ok`` (Python bools or None
    when not reported), ``flags`` (all flags, see :func:`channel_flags`),
    ``gating_flags`` (the ones that exclude it, see :func:`excluding_flags`),
    ``flagged`` (excluded from the fit by default), ``phase`` and
    ``constants`` (the ``CONSTANT_KEYS`` subset).
    """
    out: dict[str, dict] = {}
    if not isinstance(event, dict):
        return out
    stores = [block.get(SPECTRUM_KEY) for block in waveform_blocks(event).values() if isinstance(block, dict)]
    stores.append(event.get(PZT_KEY))
    for name, record in [item for store in stores if isinstance(store, dict)
                         for item in _channel_items(set_channels(store, result_set))]:
        if name in out:
            continue
        source = record.get("source")
        if not isinstance(source, dict) or _is_false(source.get("valid", True)):
            continue
        fit = record.get("fit") if isinstance(record.get("fit"), dict) else {}
        at_bounds = fit.get("at_bounds") if isinstance(fit.get("at_bounds"), dict) else {}
        flags = channel_flags(record, source)
        gating = excluding_flags(flags, exclude_near_field)
        constants = _record_constants(source)
        out[name] = {
            "seismic_moment_nm": _finite(source.get("seismic_moment_nm")),
            "mw": _finite(source.get("mw")),
            "fc_hz": _channel_fc(record, source),
            "stress_drop_pa": _finite(source.get("stress_drop_pa")),
            "source_radius_m": _finite(source.get("source_radius_m")),
            "kR": _finite(source.get("kR")),
            "band_limited": None if fit.get("band_limited") is None else _is_true(fit.get("band_limited")),
            "at_bound_fc": None if at_bounds.get("ln_fc") is None else _is_true(at_bounds.get("ln_fc")),
            "converged": None if fit.get("converged") is None else _is_true(fit.get("converged")),
            "far_field_ok": None if source.get("far_field_ok") is None else _is_true(source.get("far_field_ok")),
            "flags": flags,
            "gating_flags": gating,
            "flagged": bool(gating),
            "phase": constants.get("phase"),
            "constants": constants,
            **_ensemble_bounds(record),
        }
    return out


def mechanical_values(event: dict, trend: bool = False) -> dict[str, float]:
    """Mechanical stress drop (MPa) and slip (um) from ``event['event_analysis']``."""
    nan = float("nan")
    mech = event.get(MECH_KEY) if isinstance(event, dict) else None
    if not isinstance(mech, dict):
        return {"mech_stress_drop": nan, "mech_slip": nan}
    if "delta_y" in mech or "delta_x" in mech:          # version 3: plain differences, no trend values
        if trend:
            return {"mech_stress_drop": nan, "mech_slip": nan}
        return {"mech_stress_drop": -_finite(mech.get("delta_y")), "mech_slip": _finite(mech.get("delta_x"))}
    if trend:
        return {"mech_stress_drop": _finite(mech.get("stress_drop_trend")),
                "mech_slip": _finite(mech.get("displacement_trend"))}
    return {"mech_stress_drop": _finite(mech.get("stress_drop")),
            "mech_slip": _finite(mech.get("displacement"))}


def _nanmedian(values) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def _log_median(values) -> float:
    """Median in log space (geometric median) of the positive finite values, else NaN."""
    positive = [v for v in values if np.isfinite(v) and v > 0]
    return float(np.exp(np.median(np.log(positive)))) if positive else float("nan")


def _distinct_constants(records) -> list[dict]:
    """Distinct ``CONSTANT_KEYS`` dicts among ``records`` (order of first appearance)."""
    seen: dict[str, dict] = {}
    for rec in records:
        consts = rec.get("constants") if isinstance(rec, dict) else None
        if not isinstance(consts, dict):
            continue
        key = json.dumps(consts, sort_keys=True, default=str)
        seen.setdefault(key, dict(consts))
    return list(seen.values())


def combine_channels(records: list[dict]) -> dict:
    """Combine the channel records of one event into a self-consistent source.

    Primary quantities are combined in log space: ``M0`` and ``fc`` are the
    geometric medians over the channels with a positive finite value.  The
    derived quantities are then recomputed from the combined values rather
    than medianed separately, so the aggregate obeys the Eshelby identity:

    * ``mw = moment_magnitude(M0)``
    * ``r = kVs / fc`` with ``kVs`` the median of ``r_ch * fc_ch`` over the
      channels (each equals the channel's ``k * Vs`` by construction in
      :mod:`labquake_explorer.analysis.source`)
    * ``stress_drop = 7 M0 / (16 r^3)``
    * ``kR``: plain median (a far-field diagnostic, not a source quantity).

    Returns the ``PZT_FIELDS`` as floats (NaN when not derivable), plus
    ``kvs_m_s``, ``phases`` (sorted distinct phases), ``constants`` (distinct
    ``CONSTANT_KEYS`` dicts) and ``constants_consistent`` (a single distinct
    set of constants, or none reported).  Records whose ``k * Vs`` differ by
    more than 1e-6 relative also set ``constants_consistent`` False.
    """
    nan = float("nan")
    out = {field: nan for field in PZT_FIELDS}
    out.update({"kvs_m_s": nan, "phases": [], "constants": [], "constants_consistent": True})
    if not records:
        return out
    m0 = _log_median([r["seismic_moment_nm"] for r in records])
    fc = _log_median([r["fc_hz"] for r in records])
    kvs_values = [r["source_radius_m"] * r["fc_hz"] for r in records
                  if np.isfinite(r["source_radius_m"]) and np.isfinite(r["fc_hz"])
                  and r["source_radius_m"] > 0 and r["fc_hz"] > 0]
    kvs = float(np.median(kvs_values)) if kvs_values else nan
    out["seismic_moment_nm"] = m0
    out["fc_hz"] = fc
    out["kvs_m_s"] = kvs
    out["kR"] = _nanmedian([r["kR"] for r in records])
    for field in ERROR_FIELDS:                      # the ensembles' percentiles, combined like the values
        out[field] = _log_median([r.get(field, float("nan")) for r in records])
    if np.isfinite(m0) and m0 > 0:
        out["mw"] = float(moment_magnitude(m0))
    if np.isfinite(kvs) and kvs > 0 and np.isfinite(fc) and fc > 0:
        radius = kvs / fc
        out["source_radius_m"] = radius
        if np.isfinite(m0) and m0 > 0:
            out["stress_drop_pa"] = float(stress_drop_eshelby(m0, radius))
    out["phases"] = sorted({str(r["phase"]) for r in records if r.get("phase") is not None})
    out["constants"] = _distinct_constants(records)
    consistent = len(out["constants"]) <= 1
    if kvs_values and np.ptp(kvs_values) > 1e-6 * abs(kvs):
        consistent = False
    out["constants_consistent"] = bool(consistent)
    return out


def gather_records(run: dict, channel: str = ALL_CHANNELS, trend: bool = False,
                   include_flagged: bool = False, exclude_near_field: bool = False,
                   result_set: str = DEFAULT_SET) -> list[dict]:
    """One row per event with all ``FIELDS`` as floats (NaN when unavailable).

    ``result_set`` names the spectrum result set the records are taken from
    (:func:`available_sets`).  ``channel`` selects one PZT channel; ``ALL_CHANNELS`` combines the
    channels of each event with :func:`combine_channels`.  Quality gating:
    a single-channel row is ``flagged`` when the record carries an excluding
    flag (:func:`excluding_flags`: the fit-quality flags, plus the near-field
    flag when ``exclude_near_field``); for ``ALL_CHANNELS`` only unflagged
    channels are combined unless ``include_flagged`` is True or no unflagged
    channel has a record, in which case the flagged ones are combined and
    the row is flagged.  Flagged rows are kept (for the table and the plot)
    and left out of the fit by :func:`fit_pairs` unless ``include_flagged``.

    Each row carries ``event``, the fields, ``flags`` (all flag texts of the
    records used), ``gating_flags`` (the excluding ones), ``flagged`` (bool),
    ``n_channels`` (channels combined), ``phases``, ``constants`` (distinct
    ``CONSTANT_KEYS`` dicts) and ``constants_consistent``.  Events without
    any PZT record still appear (NaN PZT fields) so mechanical quantities
    can be listed; they never enter a fit because M0 is NaN.
    """
    rows = []
    nan = float("nan")
    for j, event in enumerate(event_list(run)):
        if not isinstance(event, dict):
            continue
        row = {"event": j, "flags": [], "gating_flags": [], "flagged": False, "n_channels": 0,
               "phases": [], "constants": [], "constants_consistent": True}
        for field in PZT_FIELDS:
            row[field] = nan
        per_channel = pzt_channel_records(event, exclude_near_field, result_set)
        if channel == ALL_CHANNELS:
            records = list(per_channel.values())
            if not include_flagged:
                clean = [r for r in records if not r["flagged"]]
                records = clean if clean else records
            combined = combine_channels(records)
            for field in PZT_FIELDS:
                row[field] = combined[field]
            row["phases"] = combined["phases"]
            row["constants"] = combined["constants"]
            row["constants_consistent"] = combined["constants_consistent"]
            row["n_channels"] = len(records)
            flags: list[str] = []
            gating: list[str] = []
            for rec in records:
                flags.extend(f for f in rec["flags"] if f not in flags)
                gating.extend(f for f in rec["gating_flags"] if f not in gating)
            row["flags"] = flags
            row["gating_flags"] = gating
            row["flagged"] = bool(gating)
        else:
            rec = per_channel.get(str(channel))
            if rec is not None:
                for field in PZT_FIELDS:
                    row[field] = rec[field]
                row["flags"] = list(rec["flags"])
                row["gating_flags"] = list(rec["gating_flags"])
                row["flagged"] = rec["flagged"]
                row["n_channels"] = 1
                row["phases"] = [str(rec["phase"])] if rec.get("phase") is not None else []
                row["constants"] = [dict(rec["constants"])]
        row.update(mechanical_values(event, trend))
        rows.append(row)
    return rows


def channel_records(run: dict, result_set: str = DEFAULT_SET) -> dict[str, list]:
    """``{channel key: [(event index, store, record), ...]}`` of every saved PZT
    record of the run in ``result_set`` (records with a binned spectrum only)."""
    out: dict[str, list] = {}
    for j, event in enumerate(event_list(run)):
        if not isinstance(event, dict):
            continue
        stores = [block.get(SPECTRUM_KEY) for block in waveform_blocks(event).values() if isinstance(block, dict)]
        stores.append(event.get(PZT_KEY))
        for store in stores:
            if not isinstance(store, dict):
                continue
            for name, record in _channel_items(set_channels(store, result_set)):
                if isinstance(record.get("spectrum"), dict):
                    out.setdefault(name, []).append((j, store, record))
    return out


def parse_t_star_grid(text) -> np.ndarray:
    """``'from, to, step'`` in microseconds -> the grid in seconds (ValueError when unusable)."""
    parts = [p.strip() for p in str(text or "").replace(";", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("t* grid: give from, to, step in microseconds")
    try:
        lo, hi, step = (float(p) for p in parts)
    except ValueError:
        raise ValueError("t* grid: from, to and step must be numbers") from None
    if not (np.isfinite([lo, hi, step]).all() and step > 0 and hi >= lo):
        raise ValueError("t* grid: need step > 0 and to >= from")
    n = int(round((hi - lo) / step)) + 1
    if n > 2000:
        raise ValueError("t* grid: more than 2000 points")
    return (lo + step * np.arange(n)) * 1e-6


def scan_run(run: dict, result_set: str, grid_s) -> dict[str, ts.TStarScan]:
    """One shared-t* scan per sensor (channel key) over the run's records in ``result_set``."""
    out = {}
    for name, items in channel_records(run, result_set).items():
        inputs = []
        for j, _, record in items:
            try:
                inputs.append(ts.record_input(record, label=f"event {j}"))
            except ValueError:
                continue
        if inputs:
            out[name] = ts.scan_t_star(inputs, grid_s)
    return out


def refit_record(record: dict, t_star_s: float) -> dict:
    """A copy of a saved record refitted with ``t*`` added to its own: the
    binned spectrum corrected, the fit redone with the record's settings, the
    source recomputed from the record's own constants (None when it had none
    or the fit failed), the ensemble dropped."""
    inp = ts.record_input(record)
    fit = inp.fit(t_star_s)
    spec = ts.corrected(inp.spectrum, t_star_s)
    new = copy.deepcopy(record)
    old = record.get("t_star_s")
    new["t_star_s"] = (0.0 if old is None else float(old)) + float(t_star_s)
    new["spectrum"] = spec.as_dict()
    new["fit"] = fit.as_dict()
    new["ensemble"] = None
    new["t_star_refit"] = {"version": 1, "from_set": record.get("result_set"), "shared_t_star_s": float(t_star_s)}
    source = record.get("source")
    new["source"] = None
    if isinstance(source, dict) and source.get("valid") and fit.valid:
        c = source.get("constants") if isinstance(source.get("constants"), dict) else {}
        moment = bool(c.get("moment_spectrum", False))
        try:
            new["source"] = src.source_parameters(
                fit.omega0, fit.fc, c.get("phase") or source.get("phase") or "P", c.get("rho_kg_m3"), c.get("vp_m_s"),
                c.get("vs_m_s"), c.get("distance_m"), radiation_coefficient=c.get("radiation_coefficient"),
                free_surface_factor=1.0 if c.get("free_surface_factor") is None else c.get("free_surface_factor"),
                k=c.get("k"), f_plateau_hz=None if moment else fit.band_used[0], moment_spectrum=moment).as_dict()
        except (ValueError, KeyError, TypeError):
            new["source"] = None
    return new


def available_sets(run: dict) -> list[str]:
    """The spectrum result sets saved in any record of the run, ``'default'``
    first (always offered), the rest sorted."""
    names: set[str] = set()
    for event in event_list(run):
        if not isinstance(event, dict):
            continue
        stores = [block.get(SPECTRUM_KEY) for block in waveform_blocks(event).values() if isinstance(block, dict)]
        stores.append(event.get(PZT_KEY))
        for store in stores:
            names.update(set_names(store))
    return [DEFAULT_SET] + sorted(names - {DEFAULT_SET})


def available_channels(run: dict, result_set: str = DEFAULT_SET) -> list[str]:
    """Channel names present in any saved PZT record of the run's result set (sorted)."""
    names: set[str] = set()
    for event in event_list(run):
        names.update(pzt_channel_records(event, result_set=result_set).keys())

    def key(name: str):
        number = parse_channel_key(name)          # 'ch3' (PZT view) or legacy '3'
        return (0, number, name) if number is not None else (1, 0, name)

    return sorted(names, key=key)


def split_pairs(rows: list[dict], y_field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(M0, y, event index, flagged)`` of the rows where both values are finite and positive."""
    x = np.array([r["seismic_moment_nm"] for r in rows], dtype=float)
    y = np.array([r[y_field] for r in rows], dtype=float)
    idx = np.array([r["event"] for r in rows], dtype=int)
    flagged = np.array([bool(r.get("flagged", False)) for r in rows], dtype=bool)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return x[keep], y[keep], idx[keep], flagged[keep]


def fit_pairs(rows: list[dict], y_field: str,
              include_flagged: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(M0, y, event index)`` of the usable pairs: finite, positive, and not flagged.

    ``include_flagged=True`` keeps the quality-flagged rows as well.
    """
    x, y, idx, flagged = split_pairs(rows, y_field)
    keep = np.ones(x.shape, dtype=bool) if include_flagged else ~flagged
    return x[keep], y[keep], idx[keep]


def excluded_pairs(rows: list[dict], y_field: str) -> list[dict]:
    """The flagged rows with a usable (finite, positive) pair: ``[{event, flags}, ...]``
    (``flags`` are the excluding ones, see :func:`excluding_flags`)."""
    _, _, idx, flagged = split_pairs(rows, y_field)
    by_event = {r["event"]: r for r in rows}
    return [{"event": int(j), "flags": list(by_event[int(j)].get("gating_flags", by_event[int(j)]["flags"]))}
            for j, f in zip(idx, flagged) if f]


def near_field_rows(rows: list[dict]) -> list[dict]:
    """Rows carrying the near-field flag that still have a finite M0."""
    return [r for r in rows if FLAG_NEAR_FIELD in r["flags"] and np.isfinite(r["seismic_moment_nm"])]


def _fmt(value: float) -> str:
    return f"{value:.4g}" if np.isfinite(value) else "n/a"


def _flag_codes(flags: list[str]) -> str:
    return ",".join(FLAG_CODES.get(f, f) for f in flags)


def _constants_text(constants: dict) -> str:
    return "/".join(str(constants.get(k)) for k in CONSTANT_KEYS)


# ------------------------------------------------------------------------ view
@register_view("Source Scaling", kinds=[RUN], order=30)
class SourceScalingView(RunView):
    """Power-law scaling of saved source parameters against seismic moment.

    Attributes tests and callers may use: ``rows`` (one dict per event, see
    :func:`gather_records`), ``fit`` (the last :class:`PowerLawFit` or None),
    ``bootstrap`` (16-84 range of the exponent), ``fit_indices`` (event
    indices used), ``excluded`` (``[{event, flags}]`` left out by the quality
    gating), ``channel_combo``, ``y_combo``, ``trend_var``,
    ``include_flagged_var``, ``exclude_near_field_var``, ``table``, ``ax``, ``scatter_artist`` (used
    pairs), ``flagged_scatter_artist`` (open symbols), ``reference_line_artist``
    (Line2D or None), ``fit_line_artist``, ``saved_result`` (the dict restored
    on open, or None) and ``saved_agrees`` (True/False/None).
    """

    window_title = "Source Scaling"
    result_key = "source_scaling"

    def __init__(self, app, run_idx: int):
        self.rows: list[dict] = []
        self.channels: list[str] = []
        self.fit: Optional[PowerLawFit] = None
        self.bootstrap: tuple[float, float] = (float("nan"), float("nan"))
        self.fit_indices: list[int] = []
        self.excluded: list[dict] = []
        self.scatter_artist = None
        self.flagged_scatter_artist = None
        self.fit_line_artist = None
        self.reference_line_artist = None
        self.saved_result: Optional[dict] = None
        self.saved_agrees: Optional[bool] = None
        self.t_star_scans: dict[str, ts.TStarScan] = {}
        self.t_star_grid_s: Optional[np.ndarray] = None
        self.error_bar_artist = None
        super().__init__(app, run_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        left = ttk.Frame(self)
        left.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky="ns")
        left.grid_rowconfigure(10, weight=1)
        left.grid_columnconfigure(1, weight=1)

        ttk.Label(left, text="Result set:").grid(row=0, column=0, padx=(0, 4), pady=2, sticky="w")
        self.set_combo = ttk.Combobox(left, state="readonly", width=16, values=[DEFAULT_SET])
        self.set_combo.set(DEFAULT_SET)
        self.set_combo.grid(row=0, column=1, pady=2, sticky="ew")
        self.set_combo.bind("<<ComboboxSelected>>", lambda e: self.on_set_selected())

        ttk.Label(left, text="Channel:").grid(row=1, column=0, padx=(0, 4), pady=2, sticky="w")
        self.channel_combo = ttk.Combobox(left, state="readonly", width=16)
        self.channel_combo.grid(row=1, column=1, pady=2, sticky="ew")
        self.channel_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh())

        ttk.Label(left, text="Y variable:").grid(row=2, column=0, padx=(0, 4), pady=2, sticky="w")
        self.y_combo = ttk.Combobox(left, state="readonly", width=26, values=list(Y_VARIABLES))
        self.y_combo.set(Y_FC)
        self.y_combo.grid(row=2, column=1, pady=2, sticky="ew")
        self.y_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh())

        self.trend_var = tk.BooleanVar(master=self, value=False)
        ttk.Checkbutton(left, text="Trend-corrected mechanical values",
                        variable=self.trend_var, command=self.refresh,
                        ).grid(row=3, column=0, columnspan=2, pady=2, sticky="w")
        self.include_flagged_var = tk.BooleanVar(master=self, value=False)
        ttk.Checkbutton(left, text="Include flagged fits (band-limited / fc at bound / "
                                   "not converged)",
                        variable=self.include_flagged_var, command=self.refresh,
                        ).grid(row=4, column=0, columnspan=2, pady=2, sticky="w")
        self.exclude_near_field_var = tk.BooleanVar(master=self, value=False)
        ttk.Checkbutton(left, text="Exclude near-field records (kR < 3 at the band's low edge)",
                        variable=self.exclude_near_field_var, command=self.refresh,
                        ).grid(row=9, column=0, columnspan=2, pady=2, sticky="w")

        buttons = ttk.Frame(left)
        buttons.grid(row=5, column=0, columnspan=2, pady=(6, 2), sticky="ew")
        buttons.grid_columnconfigure((0, 1), weight=1)
        self.fit_button = ttk.Button(buttons, text="Fit power law", command=self.fit_power_law)
        self.fit_button.grid(row=0, column=0, padx=(0, 2), sticky="ew")
        self.save_button = ttk.Button(buttons, text="Save", command=self.save)
        self.save_button.grid(row=0, column=1, padx=(2, 0), sticky="ew")

        self.fit_text = tk.StringVar(master=self, value="No fit yet")
        ttk.Label(left, textvariable=self.fit_text, wraplength=300, justify="left",
                  ).grid(row=6, column=0, columnspan=2, pady=(4, 2), sticky="w")
        self.status_var = tk.StringVar(master=self, value="")
        ttk.Label(left, textvariable=self.status_var, wraplength=300, justify="left",
                  ).grid(row=7, column=0, columnspan=2, pady=(2, 6), sticky="w")

        ttk.Label(left, text="Typical constants: rho 2650 kg/m3 (rock), Vp ~ 5000-6000 m/s "
                             "(granite), Vs ~ Vp/1.7 - set them in the PZT spectrum view. "
                             "Open symbols: quality-flagged records (excluded unless included above). "
                             "kR is evaluated at the plateau (band low edge), where every laboratory "
                             "record is near-field: a warning, excluded only on request.",
                  wraplength=300, justify="left", foreground="gray",
                  ).grid(row=8, column=0, columnspan=2, pady=(0, 6), sticky="w")

        table_frame = ttk.Frame(left)
        table_frame.grid(row=10, column=0, columnspan=2, sticky="nsew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        columns = [c[0] for c in TABLE_COLUMNS]
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)
        for name, heading, width in TABLE_COLUMNS:
            self.table.heading(name, text=heading)
            self.table.column(name, width=width, anchor="e", stretch=True)
        self.table.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.table.config(yscrollcommand=scrollbar.set)

        self.make_figure(figsize=(8, 6), row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.ax = self.figure.add_subplot(111)
        self._build_t_star_frame()

    def _build_t_star_frame(self) -> None:
        frame = ttk.LabelFrame(self, text="Shared t* per sensor (grid search over this set's records)")
        frame.grid(row=1, column=1, padx=5, pady=(0, 5), sticky="ew")
        ttk.Label(frame, text="t* grid (us) from, to, step:").grid(row=0, column=0, padx=4, pady=2, sticky="e")
        self.t_star_grid_var = tk.StringVar(master=self, value=DEFAULT_T_STAR_GRID_US)
        ttk.Entry(frame, textvariable=self.t_star_grid_var, width=14).grid(row=0, column=1, padx=4, pady=2, sticky="w")
        self.scan_button = ttk.Button(frame, text="Scan t*", command=self.scan_t_star)
        self.scan_button.grid(row=0, column=2, padx=4, pady=2, sticky="w")
        ttk.Label(frame, text="Plot:").grid(row=0, column=3, padx=(12, 4), pady=2, sticky="e")
        self.plot_combo = ttk.Combobox(frame, state="readonly", width=9, values=[PLOT_SCALING, PLOT_T_STAR])
        self.plot_combo.set(PLOT_SCALING)
        self.plot_combo.grid(row=0, column=4, padx=4, pady=2, sticky="w")
        self.plot_combo.bind("<<ComboboxSelected>>", lambda e: self.plot())
        ttk.Label(frame, text="Save refits as set:").grid(row=1, column=0, padx=4, pady=2, sticky="e")
        self.refit_set_var = tk.StringVar(master=self, value="")
        ttk.Entry(frame, textvariable=self.refit_set_var, width=14).grid(row=1, column=1, padx=4, pady=2, sticky="w")
        self.refit_button = ttk.Button(frame, text="Save refits", command=self.save_refits)
        self.refit_button.grid(row=1, column=2, padx=4, pady=2, sticky="w")
        self.t_star_text = tk.StringVar(master=self, value="no scan yet")
        ttk.Label(frame, textvariable=self.t_star_text, wraplength=640, justify="left").grid(
            row=2, column=0, columnspan=5, padx=4, pady=(2, 4), sticky="w")

    # ------------------------------------------------------------- shared t*
    def saved_t_star_scan(self) -> Optional[dict]:
        value = self.run.get(T_STAR_KEY) if isinstance(self.run, dict) else None
        return value if isinstance(value, dict) else None

    def restore_t_star_scan(self) -> None:
        """A scan saved with the run, when it was made on the set shown."""
        self.t_star_scans, self.t_star_grid_s = {}, None
        saved = self.saved_t_star_scan()
        if not saved or str(saved.get("result_set", DEFAULT_SET)) != self.result_set:
            self.t_star_text.set("no scan yet" if not saved else f"the saved scan is for set '{saved.get('result_set')}'")
            return
        grid_text = saved.get("t_star_grid_us")
        if grid_text:
            self.t_star_grid_var.set(str(grid_text))
        sensors = saved.get("sensors") if isinstance(saved.get("sensors"), dict) else {}
        for name, entry in sensors.items():
            try:
                grid = np.asarray(entry["t_star_grid_s"], dtype=float)
                self.t_star_scans[str(name)] = ts.TStarScan(
                    grid, np.asarray(entry["cost"], dtype=float), np.asarray(entry["n_valid"], dtype=int),
                    float(entry["best_t_star_s"]), int(entry["best_index"]), int(entry.get("n_records", 0)))
                self.t_star_grid_s = grid
            except (KeyError, TypeError, ValueError):
                continue
        self.show_t_star_scan()

    def show_t_star_scan(self) -> None:
        if not self.t_star_scans:
            self.t_star_text.set("no scan yet")
            return
        parts = []
        for name in sorted(self.t_star_scans, key=lambda n: (parse_channel_key(n) is None, parse_channel_key(n) or 0, n)):
            scan = self.t_star_scans[name]
            best = f"{scan.best_t_star_s * 1e6:.0f} us" if np.isfinite(scan.best_t_star_s) else "n/a"
            parts.append(f"{name}: t* = {best} ({scan.n_records} records)" + (" at grid end" if scan.at_grid_end else ""))
        self.t_star_text.set(f"best shared t* (set '{self.result_set}'): " + "; ".join(parts))

    def scan_t_star(self) -> bool:
        """Grid-search one t* per sensor over this set's records; the result
        is kept with the run (``t_star_scan``) and plotted."""
        try:
            grid = parse_t_star_grid(self.t_star_grid_var.get())
        except ValueError as e:
            self.status_var.set(str(e))
            return False
        scans = scan_run(self.run, self.result_set, grid)
        if not scans:
            self.status_var.set(f"no PZT records with a binned spectrum in set '{self.result_set}'")
            return False
        self.t_star_scans, self.t_star_grid_s = scans, grid
        record = {"version": ts.RESULT_VERSION, "result_set": self.result_set,
                  "t_star_grid_us": self.t_star_grid_var.get().strip(),
                  "sensors": {name: scan.as_dict() for name, scan in scans.items()}}
        self.data_manager.set_data(f"{self.run_path}/{T_STAR_KEY}", record, add_key=True)
        self.run[T_STAR_KEY] = record
        self.app.refresh_tree()
        self.show_t_star_scan()
        self.plot_combo.set(PLOT_T_STAR)
        self.plot()
        self.status_var.set(f"shared t* scanned over {sum(s.n_records for s in scans.values())} records of "
                            f"{len(scans)} sensor(s); kept with the run as {T_STAR_KEY}")
        return True

    def save_refits(self) -> bool:
        """Every record of this set refitted at its sensor's best t*, written
        into a new result set (with the set's controls copied and the sensors'
        t* updated), which is then shown."""
        if not self.t_star_scans:
            self.status_var.set("scan t* first")
            return False
        try:
            name = set_name(self.refit_set_var.get())
        except ValueError as e:
            self.status_var.set(str(e))
            return False
        if name == self.result_set:
            self.status_var.set("give the refits a set name different from the one scanned")
            return False
        n_written = 0
        for channel, items in channel_records(self.run, self.result_set).items():
            scan = self.t_star_scans.get(channel)
            if scan is None or not np.isfinite(scan.best_t_star_s):
                continue
            for _, store, record in items:
                try:
                    new = refit_record(record, scan.best_t_star_s)
                except ValueError:
                    continue
                new["result_set"] = name
                ensure_set(store, name)["channels"][channel] = new
                n_written += 1
        if not n_written:
            self.status_var.set("no record could be refitted")
            return False
        data = self.data_manager.data
        settings = copy.deepcopy(get_setting(data, PZT_SETTING, "sets", self.result_set))
        if isinstance(settings, dict):
            sensors = settings.get("sensors") if isinstance(settings.get("sensors"), dict) else {}
            for channel, scan in self.t_star_scans.items():
                for _, _, record in channel_records(self.run, self.result_set).get(channel, []):
                    key = str(record.get("field") or channel)
                    if key in sensors and np.isfinite(scan.best_t_star_s):
                        old = sensors[key].get("t_star_s")
                        sensors[key]["t_star_s"] = (0.0 if old is None else float(old)) + float(scan.best_t_star_s)
                    break
            set_setting(data, settings, PZT_SETTING, "sets", name)
        self.app.refresh_tree()
        self.refresh_sets()
        self.set_combo.set(name)
        self.on_set_selected()
        self.status_var.set(f"{n_written} record(s) refitted with the shared t* into set '{name}'")
        return True

    def plot_t_star(self) -> None:
        ax = self.ax
        ax.clear()
        if not self.t_star_scans:
            ax.text(0.5, 0.5, "no t* scan: press Scan t*", transform=ax.transAxes, ha="center", va="center", color="gray")
        for name in sorted(self.t_star_scans, key=lambda n: (parse_channel_key(n) is None, parse_channel_key(n) or 0, n)):
            scan = self.t_star_scans[name]
            t_us = scan.t_star_grid_s * 1e6
            line, = ax.plot(t_us, scan.cost, marker=".", label=f"{name} ({scan.n_records} records)")
            if np.isfinite(scan.best_t_star_s):
                ax.axvline(scan.best_t_star_s * 1e6, color=line.get_color(), linestyle=":", linewidth=0.8)
        ax.set_xlabel("shared t* (us)")
        ax.set_ylabel("summed weighted ln residual^2")
        if any(np.nanmin(s.cost) > 0 for s in self.t_star_scans.values() if np.isfinite(s.cost).any()):
            ax.set_yscale("log")
        ax.set_title(f"{self.experiment_name()} run{self.run_idx:02d} shared t* scan (set '{self.result_set}')".strip())
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        if self.t_star_scans:
            ax.legend(loc="best")
        self.canvas.draw_idle()

    def on_run_loaded(self) -> None:
        saved = self.load_results()
        self.refresh_sets()
        self.refresh_channels()
        if saved is not None:
            self.apply_saved_selections(saved)
        self.restore_t_star_scan()
        self.refresh()
        if saved is not None:
            self.show_saved(saved)

    # ------------------------------------------------------------- selection
    def refresh_sets(self) -> None:
        """The result sets saved in the run; a window opens on the set the PZT
        Spectrum view last saved for the experiment when the run has it."""
        previous = self.set_combo.get() if getattr(self, "_sets_loaded", False) else ""
        values = available_sets(self.run)
        self.set_combo.config(values=values)
        current = get_setting(self.data_manager.data, PZT_SETTING, CURRENT_SET_SETTING)
        for choice in (previous, str(current) if current is not None else "", DEFAULT_SET):
            if choice in values:
                self.set_combo.set(choice)
                break
        self._sets_loaded = True

    def on_set_selected(self) -> None:
        self.refresh_channels()
        self.restore_t_star_scan()
        self.refresh()

    def refresh_channels(self) -> None:
        previous = self.channel_combo.get()
        self.channels = available_channels(self.run, self.result_set)
        values = [ALL_CHANNELS] + self.channels
        self.channel_combo.config(values=values)
        self.channel_combo.set(previous if previous in values else ALL_CHANNELS)

    def apply_saved_selections(self, saved: dict) -> None:
        """Set the controls from a saved result (falling back where a choice no longer exists)."""
        result_set = str(saved.get("result_set", DEFAULT_SET))
        if result_set in list(self.set_combo["values"]) and result_set != self.set_combo.get():
            self.set_combo.set(result_set)
            self.refresh_channels()
        channel = str(saved.get("channel", ALL_CHANNELS))
        values = list(self.channel_combo["values"])
        self.channel_combo.set(channel if channel in values else ALL_CHANNELS)
        y_variable = saved.get("y_variable")
        if y_variable in Y_VARIABLES:
            self.y_combo.set(y_variable)
        self.trend_var.set(_is_true(saved.get("trend_corrected", False)))
        self.include_flagged_var.set(_is_true(saved.get("include_flagged", False)))
        self.exclude_near_field_var.set(_is_true(saved.get("exclude_near_field", False)))

    @property
    def result_set(self) -> str:
        return self.set_combo.get() or DEFAULT_SET

    @property
    def channel(self) -> str:
        return self.channel_combo.get() or ALL_CHANNELS

    @property
    def y_variable(self) -> str:
        return self.y_combo.get() or Y_FC

    @property
    def y_field(self) -> str:
        return Y_VARIABLES.get(self.y_variable, Y_VARIABLES[Y_FC])

    @property
    def include_flagged(self) -> bool:
        return bool(self.include_flagged_var.get())

    @property
    def exclude_near_field(self) -> bool:
        return bool(self.exclude_near_field_var.get())

    def set_channel(self, channel: str) -> None:
        self.channel_combo.set(str(channel))
        self.refresh()

    def set_y_variable(self, label: str) -> None:
        if label not in Y_VARIABLES:
            raise ValueError(f"unknown Y variable {label!r}")
        self.y_combo.set(label)
        self.refresh()

    def set_include_flagged(self, include: bool) -> None:
        self.include_flagged_var.set(bool(include))
        self.refresh()

    def set_exclude_near_field(self, exclude: bool) -> None:
        self.exclude_near_field_var.set(bool(exclude))
        self.refresh()

    # ----------------------------------------------------------------- data
    def refresh(self) -> None:
        """Re-gather the rows for the current selections and redraw (fit cleared)."""
        self.rows = gather_records(self.run, self.channel, bool(self.trend_var.get()),
                                   self.include_flagged, self.exclude_near_field, self.result_set)
        self.fit = None
        self.bootstrap = (float("nan"), float("nan"))
        self.fit_indices = []
        self.excluded = [] if self.include_flagged else excluded_pairs(self.rows, self.y_field)
        self.fit_text.set("No fit yet")
        self.fill_table()
        self.plot()
        self.update_status()

    def pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The pairs that enter a fit (quality-flagged rows excluded unless included)."""
        return fit_pairs(self.rows, self.y_field, self.include_flagged)

    def used_rows(self) -> list[dict]:
        _, _, idx = self.pairs()
        by_event = {r["event"]: r for r in self.rows}
        return [by_event[int(j)] for j in idx]

    def constants_summary(self) -> tuple[list[str], list[dict], bool]:
        """``(phases, distinct constants, consistent)`` over the rows entering the fit."""
        rows = self.used_rows()
        phases = sorted({p for r in rows for p in r.get("phases", [])})
        constants = _distinct_constants([{"constants": c} for r in rows for c in r.get("constants", [])])
        consistent = len(constants) <= 1 and all(r.get("constants_consistent", True) for r in rows)
        return phases, constants, bool(consistent)

    def update_status(self) -> None:
        n_pzt = sum(1 for r in self.rows if np.isfinite(r["seismic_moment_nm"]))
        if not self.channels or n_pzt == 0:
            self.status_var.set("No saved PZT results in this run "
                                "(run the PZT spectrum view and save source parameters first)")
            return
        x, _, _ = self.pairs()
        n_events = len(self.rows)
        text = (f"{n_pzt} of {n_events} events have PZT source parameters "
                f"(set '{self.result_set}', channel {self.channel}); {x.size} usable pairs for {self.y_variable}")
        n_flagged = sum(1 for r in self.rows if r["flagged"] and np.isfinite(r["seismic_moment_nm"]))
        if self.include_flagged and n_flagged:
            text += f"; {n_flagged} quality-flagged event(s) INCLUDED"
        elif self.excluded:
            text += (f"; {len(self.excluded)} event(s) excluded by quality flags "
                     f"({', '.join(sorted({f for e in self.excluded for f in e['flags']}))})")
        near = near_field_rows(self.rows)
        if near and not self.exclude_near_field and not self.include_flagged:
            n_in_fit = sum(1 for r in near if not r["flagged"])
            if n_in_fit:
                text += (f"; {n_in_fit} near-field event(s) (kR < 3 at the band's low edge) in the fit "
                         f"- tick 'Exclude near-field' to drop them")
        phases, constants, consistent = self.constants_summary()
        if not consistent:
            text += (f"; WARNING: phase/constants differ across the used records "
                     f"({'; '.join(_constants_text(c) for c in constants)}) - "
                     f"M0 and r are not comparable between them")
        self.status_var.set(text)

    def fill_table(self) -> None:
        self.table.delete(*self.table.get_children())
        for row in self.rows:
            values = [str(row["event"])]
            for name, _, _ in TABLE_COLUMNS[1:]:
                values.append(_flag_codes(row["flags"]) if name == "flags" else _fmt(row[name]))
            self.table.insert("", "end", iid=str(row["event"]), values=values)

    # ------------------------------------------------------------------ fit
    def fit_power_law(self) -> Optional[PowerLawFit]:
        """Fit ``y = c * M0**exponent`` to the usable pairs and show the result."""
        x, y, idx = self.pairs()
        fit = fit_power_law(x, y)
        self.fit = fit
        if not fit.valid:
            self.bootstrap = (float("nan"), float("nan"))
            self.fit_indices = []
            self.fit_text.set(f"Fit failed: {fit.reason} (n = {fit.n})")
            self.plot()
            return fit
        self.bootstrap = bootstrap_exponent(x, y, n_boot=N_BOOT, seed=BOOT_SEED)
        self.fit_indices = [int(i) for i in idx]
        lo, hi = self.bootstrap
        text = (f"{self.y_variable} ~ M0^b: OLS b = {fit.exponent:.4f} +- {fit.exponent_stderr:.4f}"
                f" / RMA b = {fit.exponent_rma:.4f}"
                f"\nbootstrap 16-84: [{_fmt(lo)}, {_fmt(hi)}]"
                f"\nr2 = {fit.r2:.4f}, n = {fit.n}, coefficient = {fit.coefficient:.4g}"
                f" (log10 = {fit.log10_coefficient:.4f})")
        if self.excluded:
            text += f"\n{len(self.excluded)} event(s) excluded by quality flags"
        self.fit_text.set(text)
        self.plot()
        return fit

    def show_saved(self, saved: dict) -> None:
        """Summarise a saved result and check it against a fit of the current data."""
        self.saved_result = saved
        self.saved_agrees = None
        exponent = _finite(saved.get("exponent"))
        stderr = _finite(saved.get("exponent_stderr"))
        # stored lists come back as arrays from HDF5: never truth-test them
        lo, hi = _float_list(saved.get("bootstrap_16_84"), 2)
        header = (f"Saved fit ({saved.get('y_variable', '?')}, channel {saved.get('channel', '?')}): "
                  f"b = {_fmt(exponent)} +- {_fmt(stderr)}, bootstrap 16-84: [{_fmt(lo)}, {_fmt(hi)}], "
                  f"n = {saved.get('n', '?')}")
        fit = self.fit_power_law()
        current = self.fit_text.get()
        same_selection = (str(saved.get("channel")) == self.channel
                          and saved.get("y_variable") == self.y_variable
                          and _is_true(saved.get("trend_corrected", False)) == bool(self.trend_var.get())
                          and _is_true(saved.get("include_flagged", False)) == self.include_flagged
                          and _is_true(saved.get("exclude_near_field", False)) == self.exclude_near_field)
        if not same_selection:
            self.saved_agrees = False
            verdict = ("the saved selection is no longer available - current fit uses "
                       f"channel {self.channel}; re-save to update")
        elif fit is not None and fit.valid:
            same_events = _int_list(saved.get("event_indices")) == list(self.fit_indices)
            agrees = bool(same_events and np.isfinite(exponent)
                          and abs(fit.exponent - exponent) <= 1e-6 * max(1.0, abs(exponent)))
            self.saved_agrees = agrees
            verdict = ("current data agrees with the saved fit" if agrees
                       else "current data DIFFERS from the saved fit - re-save to update")
        else:
            self.saved_agrees = False
            verdict = "current data gives no valid fit"
        self.fit_text.set(f"{header}\n{verdict}\n{current}")

    # ----------------------------------------------------------------- plot
    def plot(self) -> None:
        if getattr(self, "plot_combo", None) is not None and self.plot_combo.get() == PLOT_T_STAR:
            self.plot_t_star()
            return
        self.plot_scaling()

    def plot_scaling(self) -> None:
        ax = self.ax
        ax.clear()
        self.scatter_artist = None
        self.flagged_scatter_artist = None
        self.fit_line_artist = None
        self.reference_line_artist = None

        x_all, y_all, idx_all, flagged = split_pairs(self.rows, self.y_field)
        x, y, idx = self.pairs()
        if x_all.size:
            self.draw_error_bars(ax, idx_all)
            if x.size:
                self.scatter_artist = ax.scatter(x, y, s=30, zorder=3,
                                                 label=f"events (channel {self.channel})")
            if flagged.any():
                self.flagged_scatter_artist = ax.scatter(
                    x_all[flagged], y_all[flagged], s=36, zorder=3, facecolors="none",
                    edgecolors="C0", linewidths=1.2,
                    label="quality-flagged" + (" (in fit)" if self.include_flagged else " (excluded)"))
            for xi, yi, j in zip(x_all, y_all, idx_all):
                ax.annotate(str(j), (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)
            if x.size:
                xx = np.geomspace(x.min(), x.max(), 50) if x.min() < x.max() else np.array([x.min()])
                if self.y_variable == Y_FC:
                    ref = reference_line(xx, REFERENCE_EXPONENT, float(np.median(x)), float(np.median(y)))
                    self.reference_line_artist, = ax.plot(xx, ref, linestyle="--", color="0.4",
                                                          label=REFERENCE_LABEL)
                if self.fit is not None and self.fit.valid:
                    self.fit_line_artist, = ax.plot(
                        xx, self.fit(xx), color="C3",
                        label=f"fit: M0^{self.fit.exponent:.3f} (r2 = {self.fit.r2:.3f}, n = {self.fit.n})")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "no usable pairs", transform=ax.transAxes, ha="center", va="center",
                    color="gray")
        ax.set_xlabel("seismic moment M0 (N m)")
        ax.set_ylabel(self.y_variable)
        ax.set_title(f"{self.experiment_name()} run{self.run_idx:02d} source scaling".strip())
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self.canvas.draw_idle()

    def draw_error_bars(self, ax, events) -> None:
        """16-84 % bars from the records' uncertainty ensembles: M0 along x,
        and fc along y when fc is the Y variable (the other variables carry no
        ensemble); rows without an ensemble get none."""
        by_event = {r["event"]: r for r in self.rows}
        xs, ys, xlo, xhi, ylo, yhi = [], [], [], [], [], []
        for j in events:
            r = by_event.get(int(j))
            if r is None:
                continue
            x0, y0 = r["seismic_moment_nm"], r[self.y_field]
            dx = (max(x0 - r["m0_p16"], 0.0), max(r["m0_p84"] - x0, 0.0))
            dy = (max(y0 - r["fc_p16"], 0.0), max(r["fc_p84"] - y0, 0.0)) if self.y_field == "fc_hz" else (0.0, 0.0)
            if not (np.isfinite(dx).all() or np.isfinite(dy).all()):
                continue
            xs.append(x0); ys.append(y0)
            xlo.append(dx[0] if np.isfinite(dx[0]) else 0.0); xhi.append(dx[1] if np.isfinite(dx[1]) else 0.0)
            ylo.append(dy[0] if np.isfinite(dy[0]) else 0.0); yhi.append(dy[1] if np.isfinite(dy[1]) else 0.0)
        self.error_bar_artist = None
        if xs:
            self.error_bar_artist = ax.errorbar(xs, ys, xerr=[xlo, xhi], yerr=[ylo, yhi], fmt="none",
                                                ecolor="0.6", elinewidth=0.8, capsize=2, zorder=2,
                                                label="ensemble 16-84 %")

    # ----------------------------------------------------------------- save
    def results(self) -> Optional[dict]:
        """The result dict of the last valid fit, or None.

        The fit line is ``y = 10**log10_coefficient * M0**exponent``; ``x`` /
        ``y`` are the pairs that were fitted (in ``event_indices`` order) so
        the fit can be reproduced without the live PZT records.
        """
        if self.fit is None or not self.fit.valid:
            return None
        lo, hi = self.bootstrap
        x, y, _ = self.pairs()
        phases, constants, consistent = self.constants_summary()
        return {
            "version": RESULT_VERSION,
            "result_set": self.result_set,
            "channel": self.channel,
            "y_variable": self.y_variable,
            "y_field": self.y_field,
            "trend_corrected": bool(self.trend_var.get()),
            "include_flagged": self.include_flagged,
            "exclude_near_field": self.exclude_near_field,
            "exponent": float(self.fit.exponent),
            "exponent_stderr": float(self.fit.exponent_stderr),
            "exponent_rma": float(self.fit.exponent_rma),
            "log10_coefficient": float(self.fit.log10_coefficient),
            "coefficient": float(self.fit.coefficient),
            "r2": float(self.fit.r2),
            "n": int(self.fit.n),
            "bootstrap_16_84": [float(lo), float(hi)],
            "n_boot": N_BOOT,
            "seed": BOOT_SEED,
            "event_indices": list(self.fit_indices),
            "x": [float(v) for v in x],
            "y": [float(v) for v in y],
            "n_excluded": len(self.excluded),
            "excluded": [dict(e) for e in self.excluded],
            "phases": phases,
            "constants": constants,
            "constants_consistent": consistent,
        }

    def save(self) -> Optional[dict]:
        results = self.results()
        if results is None:
            messagebox.showwarning("Source Scaling", "Fit a power law before saving")
            self.status_var.set("Nothing saved: no valid fit")
            return None
        self.save_results(results)
        self.saved_result = results
        self.saved_agrees = True
        self.status_var.set(f"Saved {self.result_path}")
        return results
