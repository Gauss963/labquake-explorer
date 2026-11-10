"""Check which trigger block every extracted event holds from an Elsys tpc5 file.

Run it from the command line::

    python -m labquake_explorer.data.check_blocks EXPERIMENT_FILE [--base-dir DIR] [--csv OUT.csv] [--fill] [--no-backup]

For every run with events and a tpc5 reference (``run[key]['format'] == 'tpc5'``)
the raw file the reference names is listed directly, attributes and dataset
shapes only: every measurement group and every channel, the lowest-rate block
of each channel taken as its continuous record and every other block as a
trigger record.  Each event is checked with the window ``[event_time + start_s,
event_time + end_s]`` of ``run['event_extraction']`` (the event's own time
axis when the run has no such record), built as the reader builds it:
``pre = -start_s`` and ``post = end_s`` around the event time moved onto the
file clock with the reference's ``time_offset``, so that a block touching a
window edge is counted as the reader counts it:

* original blocks: the trigger blocks overlapping the window
  (:func:`~labquake_explorer.data.sources.overlapping_trigger_blocks`);
* expected block: the one :func:`~labquake_explorer.data.sources.select_trigger_block`
  picks among them, the rule the reader follows (one block per event and recorder);
* loaded block: ``event['waveform'][key]['block']`` and the record's extent.

Status: ``OK`` (the expected block is loaded), ``NOT LOADED`` (a block should
be loaded but none is), ``WRONG BLOCK`` (another block is loaded), ``NO BLOCK``
(no trigger block overlaps the window and none is loaded), ``UNCHECKED`` (the
raw file could not be read).  Missing blocks are the overlapping blocks not
loaded, marked ``(one-block rule)`` when the rule leaves them out.  Flags:
blocks crossing the window start or end, ``pre-trigger truncated`` (the record
does not start at its block's first sample or lacks some of its samples) and
trigger blocks overlapping each other in time.  File notes: extra measurement
groups, channels whose blocks differ, and other ``.tpc5`` files whose name is
the run file's stem followed by anything but a digit (``run1_0002.tpc5``, not
``run10.tpc5``) and that no run of the experiment references (TranAX split
files), which the reader never reads.

The exit code is 2 when the experiment file cannot be read; otherwise 1 when
an event is NOT LOADED, WRONG BLOCK or pre-trigger truncated, else 2 when a
raw file could not be read, else 0.

``--base-dir`` is the folder the references' filenames are relative to
(default: the experiment file's folder).  ``--csv`` writes the rows of the last
table to a CSV file.  ``--fill`` adds the expected block's record to every NOT
LOADED event, built by the event processor exactly as an extraction builds it
(the reference's ``event_fields``, else every field), saves the experiment
file (keeping a copy ``<name>.before-fill<suffix>`` next to it unless
``--no-backup``; an existing copy is never overwritten) and checks it again.
An existing record is never replaced or removed.

An ``.h5`` experiment is read with h5py, only the values needed (event times,
each record's block number, first and last time and length), so a large file
is checked in seconds; other formats are loaded with DataManager.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import h5py
import numpy as np

from labquake_explorer.data.data_manager import DataManager
from labquake_explorer.data.event_processor import EventProcessor
from labquake_explorer.data.picks import EXTRACTION_KEY
from labquake_explorer.data.sources import (WAVEFORM, Tpc5Source, open_source, overlapping_trigger_blocks,
                                            select_trigger_block)
from labquake_explorer.utils import tpc5

OK, NOT_LOADED, WRONG_BLOCK, NO_BLOCK, UNCHECKED = "OK", "NOT LOADED", "WRONG BLOCK", "NO BLOCK", "UNCHECKED"
STATUSES = (OK, NOT_LOADED, WRONG_BLOCK, NO_BLOCK, UNCHECKED)
TRUNCATED = "pre-trigger truncated"
LEFT_OUT = "(one-block rule)"
LEGACY_WINDOW_KEY = "event_window"          # where files before event_extraction kept start_s/end_s
FORMAT = Tpc5Source.format
H5_SUFFIXES = (".h5", ".hdf5")
COLUMNS = ("run", "event", "event_time", "original blocks", "loaded blocks", "loaded block",
           "expected block", "status", "missing blocks", "flags")
NUMERIC_COLUMNS = {1, 2, 3, 4}
CSV_COLUMNS = ("run", "run_name", "recorder", "event", "event_time", "window_start_s", "window_end_s",
               "original_blocks", "loaded_blocks", "loaded_block", "loaded_first_s", "loaded_last_s",
               "loaded_samples", "expected_block", "status", "missing_blocks", "flags", "file_notes")
# what reading an experiment file that is not one raises (h5py, np.load, DataManager)
READ_ERRORS = (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile, pickle.UnpicklingError)


# ---------------------------------------------------------------------------
# the experiment: only what the check needs
# ---------------------------------------------------------------------------
@dataclass
class Record:
    """An event's record from one recorder: block number and extent (run clock)."""
    block: Optional[int]
    first: Optional[float]
    last: Optional[float]
    n: int


@dataclass
class Event:
    index: int
    event_time: float
    first: Optional[float]          # the event's own time axis
    last: Optional[float]
    records: dict                   # recorder key -> Record


@dataclass
class Reference:
    key: str
    filename: str
    time_offset: float


@dataclass
class Run:
    index: int
    name: str
    window: Optional[tuple]         # (start_s, end_s) of the extraction
    references: list                # tpc5 references
    events: list


def _value(ds):
    """A scalar dataset as a Python value (strings decoded)."""
    v = ds[()]
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return None
        v = v.ravel()[0]
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v.item() if isinstance(v, np.generic) else v


def _float(value, default=None):
    try:
        return float(np.asarray(value, dtype=float).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return default


def _ordered(group) -> list:
    keys = list(group.keys())
    return sorted(keys, key=int) if keys and all(k.isdigit() for k in keys) else keys


def _index(name: str):
    return int(name) if name.isdigit() else name


def _h5_record(g) -> Optional[Record]:
    if not isinstance(g, h5py.Group) or "time" not in g:
        return None
    block = _value(g["block"]) if "block" in g else None
    block = int(block) if block is not None else None
    t = g["time"]
    if not isinstance(t, h5py.Dataset) or t.ndim != 1 or t.shape[0] == 0:
        return Record(block, None, None, 0)
    return Record(block, float(t[0]), float(t[-1]), int(t.shape[0]))


def _h5_event(index, g) -> Event:
    t = _float(_value(g["event_time"]), float("nan")) if "event_time" in g else float("nan")
    first = last = None
    axis = g.get("time")
    if isinstance(axis, h5py.Dataset) and axis.ndim == 1 and axis.shape[0]:
        first, last = float(axis[0]), float(axis[-1])
    records = {}
    waveforms = g.get(WAVEFORM)
    if isinstance(waveforms, h5py.Group):
        for name in waveforms.keys():
            record = _h5_record(waveforms[name])
            if record is not None:
                records[DataManager._unescape_key(name)] = record
    return Event(index, t, first, last, records)


def _h5_window(g) -> Optional[tuple]:
    for key in (EXTRACTION_KEY, LEGACY_WINDOW_KEY):
        record = g.get(key)
        if isinstance(record, h5py.Group) and "start_s" in record and "end_s" in record:
            start, end = _float(_value(record["start_s"])), _float(_value(record["end_s"]))
            if start is not None and end is not None:
                return start, end
    return None


def _h5_references(g) -> list:
    refs = []
    for name in g.keys():
        ref = g[name]
        if (isinstance(ref, h5py.Group) and "format" in ref and "filename" in ref
                and _value(ref["format"]) == FORMAT):
            offset = _float(_value(ref["time_offset"]), 0.0) if "time_offset" in ref else 0.0
            refs.append(Reference(DataManager._unescape_key(name), str(_value(ref["filename"])), offset))
    return refs


def _runs_from_h5(path) -> tuple:
    runs, skipped, referenced = [], [], []
    with h5py.File(path, "r") as f:
        group = f.get("runs")
        if not isinstance(group, h5py.Group):
            return runs, skipped, referenced
        for name in _ordered(group):
            g = group[name]
            if not isinstance(g, h5py.Group):
                continue
            events, refs = g.get("events"), _h5_references(g)
            referenced += [ref.filename for ref in refs]
            if not refs or not isinstance(events, h5py.Group) or not len(events):
                skipped.append(_index(name))
                continue
            run_name = str(_value(g["name"])) if "name" in g else f"run {name}"
            runs.append(Run(_index(name), run_name, _h5_window(g), refs,
                            [_h5_event(_index(j), events[j]) for j in _ordered(events) if isinstance(events[j], h5py.Group)]))
    return runs, skipped, referenced


def _record_from_dict(rec) -> Record:
    block = rec.get("block")
    block = int(np.asarray(block).ravel()[0]) if block is not None else None
    t = np.asarray(rec["time"], dtype=float).ravel()
    return Record(block, float(t[0]) if t.size else None, float(t[-1]) if t.size else None, int(t.size))


def runs_from_data(data) -> tuple:
    """``(runs, skipped, referenced)`` from a loaded experiment dict: the runs
    with events and a tpc5 reference, the indices of the others, and the
    filename of every tpc5 reference of any run."""
    runs, skipped, referenced = [], [], []
    items = data.get("runs") if isinstance(data, Mapping) else None
    for i, run in enumerate(items if isinstance(items, (list, tuple)) else []):
        if not isinstance(run, Mapping):
            continue
        refs = [Reference(str(k), str(v.get("filename", "")), _float(v.get("time_offset", 0.0), 0.0))
                for k, v in run.items() if isinstance(v, Mapping) and v.get("format") == FORMAT]
        referenced += [ref.filename for ref in refs]
        events = run.get("events")
        if not refs or not isinstance(events, (list, tuple)) or not events:
            skipped.append(i)
            continue
        window = None
        for key in (EXTRACTION_KEY, LEGACY_WINDOW_KEY):
            record = run.get(key)
            if isinstance(record, Mapping) and record.get("start_s") is not None and record.get("end_s") is not None:
                window = (float(record["start_s"]), float(record["end_s"]))
                break
        checked = []
        for j, event in enumerate(events):
            if not isinstance(event, Mapping):
                continue
            axis = np.asarray(event["time"], dtype=float).ravel() if "time" in event else np.zeros(0)
            waveforms = event.get(WAVEFORM)
            records = {str(k): _record_from_dict(r) for k, r in waveforms.items()
                       if isinstance(r, Mapping) and "time" in r} if isinstance(waveforms, Mapping) else {}
            checked.append(Event(j, _float(event.get("event_time"), float("nan")),
                                 float(axis[0]) if axis.size else None, float(axis[-1]) if axis.size else None, records))
        runs.append(Run(i, str(run.get("name", f"run {i}")), window, refs, checked))
    return runs, skipped, referenced


def read_runs(path) -> tuple:
    """``(runs, skipped, referenced)`` of an experiment file (see
    :func:`runs_from_data`): h5py for ``.h5``/``.hdf5`` (only the values the
    check needs), DataManager for anything else."""
    path = Path(path)
    if path.suffix.lower() in H5_SUFFIXES:
        return _runs_from_h5(path)
    dm = DataManager()
    dm.load_file(path)
    return runs_from_data(dm.data)


# ---------------------------------------------------------------------------
# the raw file: every block of every measurement group and channel
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RawBlock:
    """A trigger block of the raw file (times on the file clock)."""
    measurement: str
    info: tpc5.BlockInfo

    @property
    def block(self) -> int:
        return self.info.block

    @property
    def start(self) -> float:
        return self.info.start

    @property
    def end(self) -> float:
        return self.info.end

    @property
    def trigger_time(self) -> float:
        return self.info.trigger_time

    @property
    def readable(self) -> bool:
        """In the measurement group the reader reads."""
        return self.measurement == tpc5.READER_MEASUREMENT

    @property
    def label(self) -> str:
        return f"b{self.block}" if self.readable else f"m{self.measurement.lstrip('0') or '0'}/b{self.block}"


@dataclass
class RawFile:
    path: Path
    trigger: list = field(default_factory=list)       # RawBlock, every measurement group, ordered
    continuous: set = field(default_factory=set)      # continuous block numbers in the reader's group
    channels: int = 0
    notes: list = field(default_factory=list)
    error: Optional[str] = None

    def reader_block(self, number) -> Optional[RawBlock]:
        return next((b for b in self.trigger if b.readable and b.block == number), None)


def sibling_files(path, referenced=()) -> list:
    """Other ``.tpc5`` files in the same folder whose name is the file's stem
    followed by anything but a digit (``run1_0002.tpc5`` for ``run1.tpc5``, not
    ``run10.tpc5``), leaving out the files in ``referenced`` (the raw files the
    experiment's runs read)."""
    path = Path(path)
    try:
        names = sorted(os.listdir(path.parent))
    except OSError:
        return []
    stem, read = path.stem, {os.path.abspath(p) for p in referenced}
    return [path.parent / n for n in names
            if n != path.name and n.lower().endswith(".tpc5") and n.startswith(stem)
            and len(n) > len(stem) and not n[len(stem)].isdigit()
            and os.path.abspath(path.parent / n) not in read]


def scan_raw_file(path, referenced=()) -> RawFile:
    """The trigger blocks of a tpc5 file, the union over every measurement group
    and channel of the blocks other than each channel's lowest-rate one, plus
    notes on what the reader would miss.  Only attributes and shapes are read;
    ``referenced`` are the raw files the experiment's runs read (never noted
    as split files)."""
    raw = RawFile(Path(path))
    try:
        with h5py.File(raw.path, "r") as f:
            groups = tpc5.scan_blocks(f)
    except Exception as exc:          # missing, unreadable or not a tpc5 layout
        raw.error = f"cannot read {raw.path}: {exc}"
        return raw
    names = list(groups)
    if len(names) > 1:
        raw.notes.append(f"{len(names)} measurement groups ({', '.join(names)}); the reader reads "
                         f"{tpc5.READER_MEASUREMENT} only (other groups' blocks are labelled m<group>/b<block>)")
    if tpc5.READER_MEASUREMENT not in groups:
        raw.notes.append(f"no measurement group {tpc5.READER_MEASUREMENT}: the reader cannot read this file")
    order = {name: i for i, name in enumerate(names)}
    found = {}
    for name, per_channel in groups.items():
        prefix = f"measurement {name}: " if len(names) > 1 else ""
        raw.notes += [prefix + p for p in tpc5.channel_block_problems(per_channel)]
        if name == tpc5.READER_MEASUREMENT:
            raw.channels = len(per_channel)
        for blocks in per_channel.values():
            if not blocks:
                continue
            continuous = tpc5.continuous_block(blocks)
            if name == tpc5.READER_MEASUREMENT:
                raw.continuous.add(continuous.block)
            for b in blocks:
                if b.block != continuous.block:
                    found.setdefault((name, b.block), RawBlock(name, b))
    raw.trigger = sorted(found.values(), key=lambda b: (order[b.measurement], b.block))
    siblings = sibling_files(raw.path, referenced)
    if siblings:
        raw.notes.append(f"{len(siblings)} other tpc5 file(s) start with {raw.path.stem!r} (TranAX split files?), "
                         f"not read: {', '.join(p.name for p in siblings)}")
    return raw


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------
def _span(first, last, t) -> str:
    return f"[{first - t:+.3f}, {last - t:+.3f}] s"


@dataclass
class Row:
    run: object
    run_name: str
    key: str
    event: int
    event_time: float
    window: tuple                     # (start, end), run clock
    time_offset: float = 0.0
    run_label: str = ""
    overlapping: list = field(default_factory=list)
    expected: Optional[RawBlock] = None
    record: Optional[Record] = None
    loaded: Optional[RawBlock] = None
    status: str = UNCHECKED
    missing: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    truncated: bool = False
    file_notes: list = field(default_factory=list)

    def block_span(self, b: RawBlock) -> str:
        return _span(b.start + self.time_offset, b.end + self.time_offset, self.event_time)

    @property
    def loaded_text(self) -> str:
        if self.record is None:
            return "-"
        label = self.loaded.label if self.loaded is not None else (
            f"b{self.record.block}" if self.record.block is not None else "?")
        if self.record.first is None:
            return label
        return f"{label} {_span(self.record.first, self.record.last, self.event_time)}"

    @property
    def expected_text(self) -> str:
        if self.status == UNCHECKED:
            return "?"
        return self.expected.label if self.expected is not None else "-"

    def cells(self) -> list:
        return [self.run_label or str(self.run), str(self.event), f"{self.event_time:.4f}",
                "?" if self.status == UNCHECKED else str(len(self.overlapping)), "1" if self.record else "0",
                self.loaded_text, self.expected_text, self.status, "; ".join(self.missing) or "-",
                "; ".join(self.flags) or "-"]


def check_event(run: Run, ref: Reference, raw: RawFile, event: Event) -> Row:
    """One row of the check: the event's record from ``ref`` against the raw file.

    The window is built as the reader builds it, ``pre`` and ``post`` around the
    event time moved onto the file clock, so that a block touching a window edge
    is counted the same way by both."""
    t = event.event_time
    if run.window is not None:
        pre, post = -run.window[0], run.window[1]           # what Pick Events passes to the reader
    elif event.first is not None:
        pre, post = t - event.first, event.last - t
    else:
        pre = post = 0.0
    row = Row(run.index, run.name, ref.key, event.index, t, (t - pre, t + post), ref.time_offset,
              record=event.records.get(ref.key), file_notes=list(raw.notes))
    if raw.error:
        return row
    offset = ref.time_offset
    tf = t - offset                                          # file clock, as Source.to_file_time
    lof, hif = tf - pre, tf + post
    row.overlapping = overlapping_trigger_blocks(raw.trigger, tf, pre, post)
    row.expected = select_trigger_block(row.overlapping, tf, pre, post)
    record = row.record
    if record is not None and record.block is not None:
        row.loaded = raw.reader_block(record.block)
    continuous = record is not None and row.loaded is None and record.block in raw.continuous
    if record is None:
        row.status = NO_BLOCK if row.expected is None else NOT_LOADED
    elif row.loaded is not None and row.loaded is row.expected:
        row.status = OK
    elif row.expected is None and continuous:
        row.status = NO_BLOCK                  # a window of the continuous record, as for a file without trigger blocks
    else:
        row.status = WRONG_BLOCK
    if continuous:
        row.flags.append(f"continuous record b{record.block} loaded")
    for b in row.overlapping:
        if b is row.loaded:
            continue
        row.missing.append(f"{b.label} {row.block_span(b)}" + ("" if b is row.expected else f" {LEFT_OUT}"))
    for b in row.overlapping:
        if b.start < lof <= b.end:
            row.flags.append(f"{b.label} crosses window start")
        if b.start <= hif < b.end:
            row.flags.append(f"{b.label} crosses window end")
    for i, a in enumerate(row.overlapping):
        for b in row.overlapping[i + 1:]:
            if a.start <= b.end and b.start <= a.end:
                row.flags.append(f"overlapping blocks {a.label}/{b.label}")
    if record is not None:
        if record.block is None:
            row.flags.append("record has no block number")
        elif row.loaded is not None:
            info = row.loaded.info
            first_sample = info.start + offset
            late = record.first is None or abs(record.first - first_sample) > 0.5 / info.sample_rate
            if late or record.n != info.n_samples:
                row.truncated = True
                start = "?" if record.first is None else f"{record.first - t:+.4f} s"
                row.flags.append(f"{TRUNCATED} (record: {record.n} samples from {start}; "
                                 f"block: {info.n_samples} from {first_sample - t:+.4f} s)")
        elif not continuous:
            row.flags.append(f"loaded block b{record.block} is not a trigger block of the raw file")
    return row


@dataclass
class Report:
    path: Path
    base_dir: Path
    rows: list = field(default_factory=list)
    files: list = field(default_factory=list)         # (Run, Reference, RawFile)
    skipped: list = field(default_factory=list)       # runs without events or tpc5 reference

    def counts(self) -> dict:
        out = {status: 0 for status in STATUSES}
        for row in self.rows:
            out[row.status] += 1
        return out

    @property
    def n_truncated(self) -> int:
        return sum(row.truncated for row in self.rows)

    def exit_code(self) -> int:
        """1 when an event is NOT LOADED, WRONG BLOCK or truncated, 2 when a raw file was unreadable, else 0."""
        if any(row.status in (NOT_LOADED, WRONG_BLOCK) or row.truncated for row in self.rows):
            return 1
        if any(row.status == UNCHECKED for row in self.rows):
            return 2
        return 0


def check_file(path, base_dir=None) -> Report:
    """Check every event of every run with events and a tpc5 reference."""
    path = Path(path)
    base = Path(base_dir) if base_dir else path.parent
    runs, skipped, filenames = read_runs(path)
    report = Report(path, base, skipped=list(skipped))
    referenced = [base / name for name in filenames]          # every run's raw files, checked or not
    scans: dict = {}
    for run in runs:
        for ref in run.references:
            raw_path = base / ref.filename
            if raw_path not in scans:
                scans[raw_path] = scan_raw_file(raw_path, referenced)
            raw = scans[raw_path]
            report.files.append((run, ref, raw))
            label = f"{run.index}/{ref.key}" if len(run.references) > 1 else str(run.index)
            for event in run.events:
                row = check_event(run, ref, raw, event)
                row.run_label = label
                report.rows.append(row)
    return report


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def format_table(rows) -> str:
    """The rows as an aligned text table."""
    cells = [list(COLUMNS)] + [row.cells() for row in rows]
    widths = [max(len(c[i]) for c in cells) for i in range(len(COLUMNS))]
    lines = []
    for k, c in enumerate(cells):
        parts = [(c[i].rjust(widths[i]) if i in NUMERIC_COLUMNS and k else c[i].ljust(widths[i]))
                 for i in range(len(COLUMNS) - 1)]
        lines.append("  ".join(parts + [c[-1]]).rstrip())
        if k == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def format_summary(report: Report) -> str:
    counts = report.counts()
    shown = [f"{counts[s]} {s}" for s in STATUSES if s != UNCHECKED or counts[s]]
    return f"{len(report.rows)} events: " + ", ".join(shown) + f"; {report.n_truncated} {TRUNCATED}"


def format_report(report: Report) -> str:
    lines = [f"{report.path} (raw files relative to {report.base_dir})"]
    for run, ref, raw in report.files:
        if run.window is not None:
            window = f"window {run.window[0]:+g} .. {run.window[1]:+g} s"
        else:
            window = "window: each event's time axis"
        if raw.error:
            what = raw.error
        else:
            what = f"{len(raw.trigger)} trigger blocks, {raw.channels} channels"
        lines.append(f"run {run.index} ({run.name}) {ref.key}: {ref.filename}: {what}; {window}; "
                     f"time_offset {ref.time_offset:+.6f} s; {len(run.events)} events")
    if report.skipped:
        lines.append("not checked (no events or no tpc5 reference): runs " + ", ".join(str(i) for i in report.skipped))
    if not report.rows:
        lines.append("no run has events and a tpc5 reference")
        return "\n".join(lines)
    lines += ["", format_table(report.rows)]
    notes = [(run, ref, raw) for run, ref, raw in report.files if raw.notes]
    if notes:
        lines += ["", "file notes:"]
        for run, ref, raw in notes:
            lines += [f"  run {run.index} {ref.key} ({raw.path.name}): {note}" for note in raw.notes]
    lines += ["", format_summary(report)]
    return "\n".join(lines)


def write_csv(report: Report, path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for row in report.rows:
            record, t = row.record, row.event_time
            cells = row.cells()
            writer.writerow([
                row.run, row.run_name, row.key, row.event, repr(float(t)),
                f"{row.window[0] - t:.6f}", f"{row.window[1] - t:.6f}",
                cells[3], cells[4], cells[5].split()[0] if record else "",
                "" if record is None or record.first is None else f"{record.first - t:.6f}",
                "" if record is None or record.last is None else f"{record.last - t:.6f}",
                "" if record is None else record.n,
                "" if row.expected is None else row.expected.label, row.status,
                "; ".join(row.missing), "; ".join(row.flags), "; ".join(row.file_notes),
            ])


# ---------------------------------------------------------------------------
# --fill
# ---------------------------------------------------------------------------
def fill_missing(path, report: Report, backup: bool = True, dtype=np.float32) -> tuple:
    """Add the expected block's record to every NOT LOADED event of ``report``
    and save the experiment file.  Returns ``(n_added, messages)``; an event
    that already has a record for the recorder is left as it is."""
    path = Path(path)
    todo = [row for row in report.rows if row.status == NOT_LOADED and row.expected is not None]
    if not todo:
        return 0, ["--fill: no NOT LOADED event, nothing to add"]
    dm = DataManager()
    dm.load_file(path)
    runs = dm.data.get("runs") if isinstance(dm.data, Mapping) else None
    sources, messages, added = {}, [], 0
    for row in todo:
        where = f"runs/[{row.run}]/events/[{row.event}]"
        try:
            run = runs[row.run]
            event = run["events"][row.event]
        except (KeyError, IndexError, TypeError):
            messages.append(f"{where}: not found in {path.name}, skipped")
            continue
        waveforms = event.get(WAVEFORM)
        if waveforms is not None and not isinstance(waveforms, dict):
            messages.append(f"{where}: '{WAVEFORM}' is not a dict, left as it is")
            continue
        if isinstance(waveforms, dict) and row.key in waveforms:
            messages.append(f"{where}: already has {WAVEFORM}/{row.key}, left as it is")
            continue
        if not row.expected.readable:
            messages.append(f"{where}: {row.expected.label} is in measurement group {row.expected.measurement}, "
                            "which the reader does not read; not added")
            continue
        try:
            source = sources.get((row.run, row.key))
            if source is None:
                source = sources[(row.run, row.key)] = open_source(run[row.key], report.base_dir, run)
            win = source.read_record(row.expected.block, dtype=dtype)
            record = EventProcessor.waveform_record(run, row.key, source, win, dtype)
        except Exception as exc:
            messages.append(f"{where}: cannot read {row.expected.label}: {exc}")
            continue
        if waveforms is None:
            waveforms = event[WAVEFORM] = {}
        waveforms[row.key] = record
        stale = f"{row.key}: no raw record covers"
        notes = event.get("notes")
        notes = [notes] if isinstance(notes, str) else list(notes if notes is not None else [])
        event["notes"] = [n for n in notes if not str(n).startswith(stale)] + [
            f"{row.key}: {row.expected.label} added by check_blocks --fill"]
        added += 1
        messages.append(f"{where}: added {WAVEFORM}/{row.key} from {row.expected.label} "
                        f"({win.time.size} samples, {len(win.fields)} channels)")
    if added:
        copy = path.with_name(f"{path.stem}.before-fill{path.suffix}")
        if backup and not copy.exists():
            shutil.copy2(path, copy)
            messages.append(f"kept a copy of the original as {copy.name}")
        elif backup:
            messages.append(f"{copy.name} exists already and was kept as it is")
        dm.save_file(path)
        messages.append(f"saved {path}")
    return added, messages


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def _read_report(path, base_dir) -> Optional[Report]:
    """:func:`check_file`, or None after printing why the experiment file cannot be read."""
    try:
        return check_file(path, base_dir)
    except READ_ERRORS as exc:
        print(f"cannot read {path}: {exc}")
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m labquake_explorer.data.check_blocks",
        description="Check which tpc5 trigger block every extracted event holds.")
    parser.add_argument("experiment", help="experiment file (.h5, .hdf5 or .npz)")
    parser.add_argument("--base-dir", help="folder the raw-file references are relative to "
                                           "(default: the experiment file's folder)")
    parser.add_argument("--csv", help="also write the rows to this CSV file")
    parser.add_argument("--fill", action="store_true", help="add the expected block to NOT LOADED events and save")
    parser.add_argument("--no-backup", action="store_true", help="with --fill: keep no copy of the original file")
    args = parser.parse_args(argv)
    path = Path(args.experiment)
    if not path.is_file():
        print(f"{path}: no such file")
        return 2
    report = _read_report(path, args.base_dir)
    if report is None:
        return 2
    print(format_report(report))
    if args.fill:
        added, messages = fill_missing(path, report, backup=not args.no_backup)
        print()
        for message in messages:
            print(message)
        if added:
            report = _read_report(path, args.base_dir)
            if report is None:
                return 2
            print()
            print(format_report(report))
    if args.csv:
        write_csv(report, args.csv)
        print(f"wrote {args.csv}")
    return report.exit_code()


if __name__ == "__main__":
    sys.exit(main())
