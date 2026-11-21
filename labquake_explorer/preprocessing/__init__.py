"""Building experiment files from raw acquisition data.

Pipeline for one run::

    source  = Tpc5Source.open(path, channel_map)              # labquake_explorer.data.sources
    run     = run_from_sources(sources, calibration, ...)     # labquake_explorer.preprocessing.build
    exp     = experiment('t0211', runs, date=..., ...)
    DataManager().save(exp)

The pieces are independent: :mod:`.calibration` turns volt fields into
physical fields and records how; :mod:`.alignment` relates two recorders'
clocks; :mod:`.build` assembles run and experiment dicts in the layout the
explorer reads.  Everything here is pure (no GUI, no global state).
"""
from labquake_explorer.preprocessing.calibration import (
    Calibration, EddySlip, Friction, Linear, pressure_transducers, PRESSURE_PRESETS, PRESSURE_CALIBRATIONS,
    PRESSURE_MPA_PER_BAR,
)
from labquake_explorer.preprocessing.alignment import offset_from_trigger, slip_step_table
from labquake_explorer.preprocessing.build import (
    experiment, parse_run_name, run_from_sources, run_from_tpc5, run_from_tpc5_ni,
)

__all__ = [
    "Calibration", "EddySlip", "Friction", "Linear", "pressure_transducers", "PRESSURE_PRESETS",
    "PRESSURE_CALIBRATIONS", "PRESSURE_MPA_PER_BAR",
    "offset_from_trigger", "slip_step_table",
    "experiment", "parse_run_name", "run_from_sources", "run_from_tpc5", "run_from_tpc5_ni",
]
