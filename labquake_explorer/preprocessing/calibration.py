"""Volt-to-physical conversions applied to a run dict.

A :class:`Calibration` is an ordered list of steps.  Each step reads fields of
the run (volts, or fields produced by earlier steps), writes new fields, and
updates ``run['units']``.  ``Calibration.apply`` also stores
``run['calibration']``, a JSON-like description of every step, so a saved
experiment says how its physical channels were derived.

Steps provided here:

* :class:`Linear` -- ``field = factor * source + offset``; the source may be a
  top-level array or a channel of ``run['raw_data']``.
* :class:`EddySlip` -- eddy-current voltages to the ``slip`` channel array
  (micrometres, zeroed at the start of the run, sensor positions attached);
  optionally a ``displacement`` field (the views fall back to the first slip
  channel without one).
* :class:`Friction` -- ``shear_stress / normal_stress`` once both are in MPa.

A new kind of conversion is a class with ``apply(run, units)`` and
``describe()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from labquake_explorer.data.channels import (
    channel_array, channel_arrays, channel_names, find_channel, get_channel, is_channel_array,
    positions_table,
)


class Step:
    def apply(self, run: dict, units: dict) -> None:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


@dataclass
class Linear(Step):
    """``field (unit) = factor * run[source] + offset``; skipped when ``source`` is absent."""
    field: str
    source: str
    factor: float
    offset: float = 0.0
    unit: str = ""

    def apply(self, run, units):
        source = get_channel(run, self.source)
        if source is None:
            return
        run[self.field] = (self.factor * np.asarray(source, dtype=np.float64) + self.offset).astype(np.float32)
        units[self.field] = self.unit

    def describe(self):
        return {"kind": "linear", "field": self.field, "source": self.source, "factor": float(self.factor),
                "offset": float(self.offset), "unit": self.unit,
                "formula": f"{self.field} ({self.unit}) = {self.factor:.6g} * {self.source} + {self.offset:.6g}"}


@dataclass
class EddySlip(Step):
    """Slip in micrometres from eddy-current sensor voltages.

    The lab calibrates each sensor as ``d (mm) = slope * V + intercept``.  The
    intercept belongs to the calibration jig, so instead every slip channel is
    zeroed on the mean of the first ``zero_window_s`` of the run (the sensors are
    zeroed after the run-in in the experiment as well).  Sensors without their
    own slope use ``default_slope_mm_per_v`` (the mean of the known ones when
    None).

    The result is the channel array ``run[out_key]`` (rows ``slip_1``, ...,
    unit um, ``source`` naming the voltage channel of each row, ``positions``
    from ``positions`` or inherited from the recorder's ``raw_data``) plus
    ``displacement``: not written (None, the default), the mean of all slip
    channels (``'mean'``) or one of them (e.g. ``'slip_5'``).
    """
    slopes_mm_per_v: Mapping[str, float]
    default_slope_mm_per_v: Optional[float] = None
    zero_window_s: float = 0.5
    source_prefix: str = "eddy_"
    out_prefix: str = "slip_"
    out_key: str = "slip"
    displacement: Optional[str] = None
    positions: Optional[Mapping[str, Sequence[float]]] = None   # source channel -> (x, y, z)
    position_unit: str = "mm"
    position_frame: str = ""

    def slope_for(self, source: str) -> float:
        if source in self.slopes_mm_per_v:
            return float(self.slopes_mm_per_v[source])
        if self.default_slope_mm_per_v is not None:
            return float(self.default_slope_mm_per_v)
        if not self.slopes_mm_per_v:
            raise ValueError("EddySlip needs at least one slope or a default")
        return float(np.mean(list(self.slopes_mm_per_v.values())))

    def sources(self, run) -> list:
        """Voltage channels ``<prefix><k>`` found at the top level or in any channel array."""
        names = {k for k, v in run.items() if isinstance(v, np.ndarray) and v.ndim == 1}
        for _, value in channel_arrays(run):
            names.update(channel_names(value))
        return sorted((k for k in names if k.startswith(self.source_prefix) and k[len(self.source_prefix):].isdigit()),
                      key=lambda k: int(k[len(self.source_prefix):]))

    def _positions(self, run, sources, out_names) -> dict:
        if self.positions is not None:
            return positions_table(sources, self.positions, self.position_unit, self.position_frame)
        table = positions_table(sources, None, self.position_unit, self.position_frame)
        arrays = dict(channel_arrays(run))
        for i, src in enumerate(sources):
            hit = find_channel(run, src)
            if hit is None:
                continue
            pos = arrays[hit[0]].get("positions")
            if isinstance(pos, Mapping):
                for axis in ("x", "y", "z"):
                    table[axis][i] = float(np.asarray(pos[axis])[hit[1]])
                table["unit"] = str(pos.get("unit", table["unit"]))
                table["frame"] = str(pos.get("frame", table["frame"]))
        return table

    def apply(self, run, units):
        sources = self.sources(run)
        if not sources:
            return
        t = np.asarray(run["time"], dtype=np.float64)
        dt = float(np.median(np.diff(t[: min(t.size, 2000)]))) if t.size > 1 else 1.0
        n0 = max(1, int(round(self.zero_window_s / dt)))
        rows, out_names = [], []
        for src in sources:
            v = np.asarray(get_channel(run, src), dtype=np.float64)
            rows.append(1000.0 * self.slope_for(src) * (v - np.nanmean(v[:n0])))
            out_names.append(self.out_prefix + src[len(self.source_prefix):])
        data = np.vstack(rows)
        run[self.out_key] = channel_array(data, out_names, unit="um", positions=self._positions(run, sources, out_names),
                                          source=list(sources),
                                          slope_mm_per_v=[self.slope_for(s) for s in sources])
        units[self.out_key] = "um"
        if self.displacement == "mean":
            run["displacement"] = data.mean(axis=0).astype(np.float32)
            units["displacement"] = "um"
        elif self.displacement:
            if self.displacement not in out_names:
                raise KeyError(f"displacement source {self.displacement!r} is not one of {out_names}")
            run["displacement"] = np.asarray(data[out_names.index(self.displacement)], dtype=np.float32)
            units["displacement"] = "um"

    def describe(self):
        return {"kind": "eddy_slip", "slopes_mm_per_v": {k: float(v) for k, v in self.slopes_mm_per_v.items()},
                "default_slope_mm_per_v": self.slope_for("__default__") if (self.slopes_mm_per_v or self.default_slope_mm_per_v is not None) else None,
                "zero_window_s": float(self.zero_window_s), "source_prefix": self.source_prefix,
                "out_prefix": self.out_prefix, "out_key": self.out_key, "displacement": self.displacement,
                "positions_given": self.positions is not None,
                "formula": f"{self.out_prefix}k (um) = 1000 * slope_k * (V - mean(V over the first {self.zero_window_s:g} s))"}


@dataclass
class Friction(Step):
    """``friction = shear / normal`` when both carry the same non-volt unit."""
    shear: str = "shear_stress"
    normal: str = "normal_stress"
    field: str = "friction"

    def apply(self, run, units):
        if self.shear in run and self.normal in run and units.get(self.shear) == units.get(self.normal) not in (None, "", "V"):
            with np.errstate(divide="ignore", invalid="ignore"):
                run[self.field] = (np.asarray(run[self.shear], dtype=np.float64)
                                   / np.asarray(run[self.normal], dtype=np.float64)).astype(np.float32)
            units[self.field] = ""

    def describe(self):
        return {"kind": "friction", "field": self.field, "formula": f"{self.field} = {self.shear} / {self.normal}"}


class Calibration:
    """An ordered list of steps applied to a run dict."""

    def __init__(self, *steps: Step, note: str = ""):
        self.steps = list(steps)
        self.note = note

    def add(self, *steps: Step) -> "Calibration":
        self.steps.extend(steps)
        return self

    def apply(self, run: dict) -> dict:
        units = run.setdefault("units", {})
        for key, value in run.items():
            if isinstance(value, np.ndarray) and key != "time":
                units.setdefault(key, "V")
        for path, value in channel_arrays(run):
            units.setdefault(path, str(value.get("unit", "")))
        for step in self.steps:
            step.apply(run, units)
        run["calibration"] = self.describe()
        return run

    def describe(self) -> dict:
        out = {"steps": [s.describe() for s in self.steps]}
        if self.note:
            out["note"] = self.note
        return out

    def __repr__(self):
        return f"Calibration({', '.join(type(s).__name__ for s in self.steps)})"


# ---------------------------------------------------------------------------
# lab presets
# ---------------------------------------------------------------------------
# Pressure-transducer voltages to stress for the biaxial apparatus
# (Tseng 2026, NTU thesis, Appendix B.3.3 / Table B.1): V = a * sigma + b on the
# normal-stress line and V = c * tau on the shear line, by fault section.
#
# A factor is c = c_2D * (A_fault / A_2D) * (n_2D / n), the 2D constant scaled by
# the fault area and the number of jacks pressed by the line: the 2D sample
# (0.25 m^2) is loaded by nine normal and three shear jacks, a 1D sample by the
# middle three normal jacks and the middle shear jack only.  The normal factors
# of Table B.1 already count three jacks (0.739 * 0.1 * 9/3 = 0.222 for 5 cm);
# its shear factors (0.109, 0.217) scale the 2D one by area alone, i.e. assume
# the three shear jacks, and gave shear stress and friction three times too
# high.  Corrected 2026-11-02 to one jack: 1.087 * 0.1 * 3 = 0.326 (5 cm) and
# 1.087 * 0.2 * 3 = 0.652 (10 cm).  Files built with the old factors are fixed
# by dividing shear_stress and friction by three.
PRESSURE_PRESETS = {
    "1D-5cm":  {"normal_v_per_mpa": 0.222, "normal_offset_v": 0.091, "shear_v_per_mpa": 0.326},
    "1D-10cm": {"normal_v_per_mpa": 0.444, "normal_offset_v": 0.091, "shear_v_per_mpa": 0.652},
    "2D-50cm": {"normal_v_per_mpa": 0.739, "normal_offset_v": 0.091, "shear_v_per_mpa": 1.087},
}


def pressure_transducers(section: str = "1D-5cm", normal_source: str = "pressure_1",
                         shear_source: str = "pressure_2") -> list:
    """``normal_stress`` and ``shear_stress`` (MPa) from the two pressure voltages."""
    p = PRESSURE_PRESETS[section]
    return [
        Linear("normal_stress", normal_source, 1.0 / p["normal_v_per_mpa"],
               -p["normal_offset_v"] / p["normal_v_per_mpa"], "MPa"),
        Linear("shear_stress", shear_source, 1.0 / p["shear_v_per_mpa"], 0.0, "MPa"),
    ]


__all__ = ["Step", "Linear", "EddySlip", "Friction", "Calibration", "PRESSURE_PRESETS", "pressure_transducers"]
