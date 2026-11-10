<div align="center">
  <img src="assets/icons/labquake_explorer.svg" width="150" height="150">
</div>

# Labquake Explorer

A GUI application for analyzing and visualizing labquake event data.

## Installation

```bash
pip install -e .
```

## Usage

Run using the installed entry point:
```bash
labquake-explorer
```

Or run directly:
```bash
python -m labquake_explorer.main
```

Right-click a node in the data tree to see the analyses that apply to it
(runs, events, arrays). Results are saved into the loaded experiment under
the node they belong to and written to disk with "Save As" (`.npz`, `.h5` or
`.hdf5`; other suffixes are refused, and a failed save leaves the previous
file untouched).

## Versions

`labquake-explorer --version` prints the package version (the window title
shows it too); it is written once, in `labquake_explorer/_version.py`, and
releases are tagged `v<version>`. Every experiment file written records,
under a top-level `labquake_explorer` dict, the version that last wrote it
and when, the git commit when run from a checkout, and the version that
built it.

## Features

- Load and analyze labquake data stored in NPZ and HDF5 formats
- Pick events on a run and extract event windows
- Event Analyzer: slopes and differences of any two event fields over
  draggable ranges (stiffness, stress drop and slip when run on stress against
  slip); apply one set of ranges to all events of a run
- Inter-event Metrics: recurrence, load-point advance, fault slip and creep
  per stick-slip cycle
- Slip Distribution: coseismic slip of every eddy-current sensor with its own
  event time (largest slip rate or dragged by hand) and fit windows before and
  after it; slip along the fault with a quadratic or earthquake arrest zone
  fit, and the rupture-front velocity from the event times
- Pick dynamic strain arrivals and estimate rupture speed
- Fit a Cohesive Zone Model to dynamic strain records
- PZT Spectrum: calibrated amplitude spectrum, omega^-n source fit and
  source parameters (moment, magnitude, radius, stress drop) per channel
- Source Scaling: power-law fits of source parameters across events
- Plot Run Signals: overlay run-level signals with event markers

## Preprocessing

`labquake_explorer.preprocessing` builds experiment files from raw
acquisition data (Elsys tpc5, NI loggers): readers, calibration steps with a
provenance record, two-recorder clock alignment. `examples/preprocessing/`
holds the notebooks that use it. Event extraction reads the full-rate windows
straight from the raw files the runs reference.

## Development

Tests use synthetic stick-slip runs with known answers and run headlessly:

```bash
python -m pytest tests -q
```

See `docs/extending.md` for how to add a view or an analysis and for the
saved result schemas.
