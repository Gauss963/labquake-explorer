"""Assemble run and experiment dicts in the layout Labquake Explorer reads.

A run is built from one or more :class:`~labquake_explorer.data.sources.Source`
objects: the first one (the *time base*) provides the run's time axis and its
whole-run record; the others contribute their whole-run records interpolated
onto that axis for any field the time base does not already provide.  Every
source is stored as a reference under its own key (``run['elsys']``,
``run['ni']``) and its recorded voltages go next to it as the channel array
``run[<key>]['raw_data']`` (one row per channel, with the sensor positions when
given); the calibration then adds the physical channels at the top level.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from labquake_explorer.data.channels import channel_array, positions_table
from labquake_explorer.data.sources import NINpzSource, Source, Tpc5Source
from labquake_explorer.data.stamp import stamp
from labquake_explorer.preprocessing.alignment import offset_from_trigger
from labquake_explorer.preprocessing.calibration import Calibration

RUN_NAME = re.compile(r"^(?P<experiment>[A-Za-z]\d{3,5})_(?P<index>\d+)_(?P<sigma>[\d.]+)MPa_(?P<run>run\d+)")


def parse_run_name(stem: str) -> dict:
    """``t0211_03_12MPa_run3`` -> name, nominal normal stress, experiment."""
    m = RUN_NAME.match(str(stem))
    if not m:
        return {"name": str(stem)}
    return {"name": m["run"], "normal_stress_level": float(m["sigma"]), "experiment": m["experiment"].lower()}


def run_from_sources(sources: Mapping[str, Source], base_dir, calibration: Optional[Calibration] = None,
                     time_base: Optional[str] = None, decimation: Optional[Mapping[str, int]] = None,
                     name: Optional[str] = None, positions: Optional[Mapping] = None,
                     position_unit: str = "mm", position_frame: str = "", **metadata) -> dict:
    """Build a run dict.

    ``sources`` maps the reference key each source is stored under (``'elsys'``
    for the Elsys file, ``'ni'`` for the NI file, ...) to the source; its
    ``time_offset`` must already put it on the run clock.  ``time_base`` names
    the source whose whole-run record becomes the run's time axis (default: the
    first).  ``decimation`` gives a block-mean factor per key for the whole-run
    records.  ``positions`` maps a field to its (x, y, z) on the sample for the
    ``raw_data`` positions tables.  ``metadata`` is copied into the run (name,
    normal_stress_level, ...).
    """
    if not sources:
        raise ValueError("at least one source is needed")
    keys = list(sources)
    time_base = keys[0] if time_base is None else time_base
    if time_base not in sources:
        raise KeyError(f"time_base {time_base!r} is not one of {keys}")
    decimation = dict(decimation or {})
    base = sources[time_base]

    t_run, data = base.time_history(decimation.get(time_base, 1))
    run = {"name": name or metadata.pop("name", base.path.stem)}
    run.update(metadata)
    if isinstance(base, Tpc5Source):
        run["start_time"] = base.start_time
    run["time"] = np.asarray(t_run, dtype=np.float64)
    recorded = {time_base: ([np.asarray(c, dtype=np.float32) for c in data], list(base.fields))}
    taken = set(base.fields)
    for key in keys:
        if key == time_base:
            continue
        src = sources[key]
        t_src, d_src = src.time_history(decimation.get(key, 1))
        columns, names = [], []
        for field_name, column in zip(src.fields, d_src):
            if field_name in taken:
                continue                                   # the time base wins
            columns.append(np.interp(run["time"], t_src, np.asarray(column, dtype=np.float64),
                                     left=np.nan, right=np.nan).astype(np.float32))
            names.append(field_name)
            taken.add(field_name)
        recorded[key] = (columns, names)
    run["units"] = {}
    for key, src in sources.items():                 # each recorder's block: file reference (format, filename, ...) + raw_data
        run[key] = src.to_reference(base_dir)
        columns, names = recorded[key]
        if names:
            run[key]["raw_data"] = channel_array(
                np.vstack(columns), names, unit="V",
                positions=positions_table(names, positions, position_unit, position_frame) if positions is not None else None)
            run["units"][f"{key}/raw_data"] = "V"
    if calibration is not None:
        calibration.apply(run)
    return run


def run_from_tpc5(path, channel_map, base_dir, calibration: Optional[Calibration] = None,
                  decimation: int = 1, event_fields: Optional[Sequence[str]] = None,
                  event_window_s: Optional[Sequence[float]] = None, key: str = "elsys",
                  positions: Optional[Mapping] = None, **metadata) -> dict:
    """A run whose every channel is on one Elsys tpc5 file."""
    path = Path(path)
    src = Tpc5Source.open(path, channel_map, event_fields=event_fields, event_window_s=event_window_s)
    meta = {**parse_run_name(path.stem), **metadata}
    return run_from_sources({key: src}, base_dir, calibration, decimation={key: decimation},
                            positions=positions, **meta)


def run_from_tpc5_ni(tpc5_path, ni_path, elsys_channel_map, ni_channel_map, base_dir,
                     calibration: Optional[Calibration] = None, ni_decimation: int = 250,
                     clock_offset: Optional[float] = None, ni_meta: Optional[Mapping] = None,
                     elsys_event_fields: Optional[Sequence[str]] = None,
                     elsys_event_window_s: Optional[Sequence[float]] = None,
                     ni_event_fields: Optional[Sequence[str]] = None,
                     ni_event_window_s: Optional[Sequence[float]] = None,
                     positions: Optional[Mapping] = None, **metadata) -> dict:
    """A run with the mechanical channels on an NI logger and the dynamic channels
    on an Elsys tpc5.  The NI clock is the run clock; the Elsys clock is shifted
    by ``clock_offset`` (default: from the NI trigger sample and the first Elsys
    trigger block).  Elsys fields the NI does not provide (the PZTs) are
    interpolated from the Elsys continuous block onto the run axis.
    """
    tpc5_path, ni_path = Path(tpc5_path), Path(ni_path)
    ni = NINpzSource.open(ni_path, ni_channel_map, decimation=ni_decimation, meta=ni_meta,
                          event_fields=ni_event_fields, event_window_s=ni_event_window_s)
    elsys = Tpc5Source.open(tpc5_path, elsys_channel_map, event_fields=elsys_event_fields,
                            event_window_s=elsys_event_window_s)
    elsys.time_offset = offset_from_trigger(ni, elsys) if clock_offset is None else float(clock_offset)
    elsys.meta["clock_offset_method"] = ("NI trigger_sample_index = first Elsys trigger" if clock_offset is None
                                         else "given")
    meta = {**parse_run_name(tpc5_path.stem), **metadata}
    run = run_from_sources({"ni": ni, "elsys": elsys}, base_dir, calibration,
                           time_base="ni", decimation={"ni": ni_decimation}, positions=positions, **meta)
    run["start_time"] = elsys.start_time
    return run


def experiment(name: str, runs: Sequence[dict], **metadata) -> dict:
    """The top-level experiment dict (``name``, metadata, ``runs``), stamped
    with the package version that built it (``labquake_explorer``)."""
    exp = {"name": str(name)}
    exp.update(metadata)
    stamp(exp, created=True)
    exp["runs"] = list(runs)
    return exp


__all__ = ["RUN_NAME", "parse_run_name", "run_from_sources", "run_from_tpc5", "run_from_tpc5_ni", "experiment"]
