"""Raw-data sources referenced from a run.

A run dict can point at the acquisition files its time history was built from,
so that event extraction can read the full-rate records around picked event
times without copying them into the experiment file.  Each pointer is a small
JSON-like dict with a ``format`` key::

    run['strain'] = {'format': 'tpc5',   'filename': 'run1.tpc5', 'time_offset': 2.39, 'fields': [...], ...}
    run['ni']     = {'format': 'ni_npz', 'filename': 'run1.npz',  'time_offset': 0.0,  'fields': [...], ...}

``time_offset`` puts the file's clock on the run's clock
(``t_run = t_file + time_offset``); ``fields`` names the run field each file
channel feeds.  :func:`open_source` turns a pointer back into the matching
:class:`Source`, which reads windows by run time.  A new recorder is added by
subclassing :class:`Source` and decorating it with :func:`register_source`.

The PSU-era layout (``strain`` with ``filename``, ``time_offset``, ``time``,
``raw`` and no ``format``) is handled by :class:`LegacyTpc5Source`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import h5py
import numpy as np

from labquake_explorer.data.channels import channel_array, channel_names, is_channel_array
from labquake_explorer.utils import tpc5
from labquake_explorer.utils.ni_npz import NIRecord, open_ni_npz

WAVEFORM = "waveform"    # event key: each recorder's full-rate record around the event

SOURCE_FORMATS: dict = {}


def register_source(cls):
    """Class decorator: make ``cls`` the reader for references with ``format == cls.format``."""
    if not getattr(cls, "format", ""):
        raise ValueError(f"{cls.__name__} needs a non-empty 'format'")
    SOURCE_FORMATS[cls.format] = cls
    return cls


def block_mean(data: np.ndarray, factor: int) -> np.ndarray:
    """Average consecutive blocks of ``factor`` samples along the last axis."""
    factor = int(factor)
    if factor <= 1:
        return np.asarray(data)
    n = (data.shape[-1] // factor) * factor
    return np.asarray(data[..., :n], dtype=np.float64).reshape(*data.shape[:-1], n // factor, factor).mean(axis=-1)


@dataclass
class Window:
    """Samples of one source between two run times."""
    time: np.ndarray            # run clock (s)
    data: np.ndarray            # (n_fields, n)
    fields: list
    sample_rate: float
    block: Optional[int] = None  # tpc5 block number, when applicable
    trigger_time: Optional[float] = None  # the record's own trigger on the run clock (recorders with trigger blocks)

    def as_dict(self, dtype=np.float32) -> dict:
        return {"time": np.asarray(self.time, dtype=np.float64), "raw": np.asarray(self.data, dtype=dtype),
                "fields": list(self.fields), "sample_rate": float(self.sample_rate), "block": self.block}

    def as_channel_array(self, dtype=np.float32, unit: str = "V", positions: Optional[dict] = None, **meta) -> dict:
        """The samples as a channel array (:mod:`labquake_explorer.data.channels`)
        carrying ``time`` (run clock), ``sample_rate`` and, for a recorder that
        writes separate records, ``block``; ``meta`` adds further entries."""
        extra = {"time": np.asarray(self.time, dtype=np.float64), "sample_rate": float(self.sample_rate)}
        if self.block is not None:
            extra["block"] = int(self.block)
        if self.trigger_time is not None:
            extra["trigger_time"] = float(self.trigger_time)
        extra.update(meta)
        return channel_array(self.data, self.fields, unit=unit, positions=positions, dtype=dtype, **extra)


class Source:
    """A recording file plus the mapping of its clock and channels onto a run."""

    format: str = ""

    def __init__(self, path, fields: Sequence[str], time_offset: float = 0.0,
                 event_fields: Optional[Sequence[str]] = None,
                 event_window_s: Optional[Sequence[float]] = None, meta: Optional[Mapping] = None):
        self.path = Path(path)
        self.fields = [str(f) for f in fields]
        self.time_offset = float(time_offset)
        self.event_fields = [str(f) for f in event_fields] if event_fields else None
        self.event_window_s = (float(event_window_s[0]), float(event_window_s[1])) if event_window_s else None
        self.meta = dict(meta or {})

    # ------------------------------------------------------------------ clocks
    def to_file_time(self, t_run: float) -> float:
        return float(t_run) - self.time_offset

    def to_run_time(self, t_file) -> np.ndarray:
        return np.asarray(t_file, dtype=np.float64) + self.time_offset

    # ------------------------------------------------------------ channels
    def field_indices(self, fields: Optional[Sequence[str]]) -> list:
        if fields is None:
            return list(range(len(self.fields)))
        out = []
        for f in fields:
            if f not in self.fields:
                raise KeyError(f"{self.path.name}: unknown field {f!r}; known: {self.fields}")
            out.append(self.fields.index(f))
        return out

    # ------------------------------------------------------- abstract reads
    def time_history(self, decimation: int = 1):
        """``(t_run, data)`` of the whole-run record, block-mean decimated."""
        raise NotImplementedError

    def trigger_times(self) -> list:
        """Trigger times on the run clock (empty when the recorder has none)."""
        return []

    def read_window(self, t_from: float, t_to: float, fields: Optional[Sequence[str]] = None,
                    dtype=np.float32) -> Optional[Window]:
        """Samples between two run times, or None when no record covers them."""
        raise NotImplementedError

    def event_window(self, event_time: float, pre: float, post: float,
                     fields: Optional[Sequence[str]] = None, dtype=np.float32) -> Optional[Window]:
        """The window around an event, honouring the source's own ``event_window_s``
        and ``event_fields`` when they are set."""
        if self.event_window_s is not None:
            pre, post = self.event_window_s
        if fields is None:
            fields = self.event_fields
        return self.read_window(float(event_time) - float(pre), float(event_time) + float(post), fields, dtype)

    def waveform(self, event_time: float, pre: float, post: float,
                  fields: Optional[Sequence[str]] = None, dtype=np.float32) -> Optional[Window]:
        """The full-rate record copied into an event (``event['waveform'][key]``).

        A recorder that writes separate records around triggers returns one
        whole record, every sample of it, chosen by :func:`select_trigger_block`
        with the window ``[event_time - pre, event_time + post]``: the record
        containing ``event_time`` that overlaps the window most, else the
        record in the window nearest to ``event_time``; None when no record
        overlaps the window (see :class:`Tpc5Source`).  A continuous recorder
        returns the window clipped to its record, its own ``event_window_s``
        taking precedence, or None when its record does not cover the window's
        middle.  ``fields`` defaults to the source's ``event_fields``.
        """
        return self.event_window(event_time, pre, post, fields, dtype)

    # ----------------------------------------------------------- references
    def to_reference(self, base_dir) -> dict:
        """The JSON-like pointer stored in the run dict."""
        ref = {
            "format": self.format,
            "filename": os.path.relpath(self.path, Path(base_dir)),
            "time_offset": self.time_offset,
            "fields": list(self.fields),
        }
        if self.event_fields is not None:
            ref["event_fields"] = list(self.event_fields)
        if self.event_window_s is not None:
            ref["event_window_s"] = list(self.event_window_s)
        for key, value in self.meta.items():
            ref.setdefault(key, value)
        return ref

    @classmethod
    def from_reference(cls, ref: Mapping, base_dir, run: Optional[Mapping] = None) -> "Source":
        raise NotImplementedError

    @staticmethod
    def _common_kwargs(ref: Mapping) -> dict:
        return {
            "time_offset": float(ref.get("time_offset", 0.0)),
            "event_fields": list(ref["event_fields"]) if ref.get("event_fields") is not None else None,
            "event_window_s": list(ref["event_window_s"]) if ref.get("event_window_s") is not None else None,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path.name}, fields={len(self.fields)}, time_offset={self.time_offset:+.4f})"


# ---------------------------------------------------------------------------
# Elsys TranAX tpc5 (single block or ECR dual mode)
# ---------------------------------------------------------------------------
def short_label(name: str) -> str:
    """'A1 (0.A1)' -> 'A1'."""
    return str(name).split()[0] if str(name).strip() else str(name)


def _ns(seconds: float) -> float:
    """A duration rounded to the nanosecond, so that rounding noise does not break ties."""
    return round(float(seconds), 9)


def _gap(block, t: float) -> float:
    """Distance from ``t`` to a block's span ``[start, end]`` (0 inside it)."""
    if t < block.start:
        return block.start - t
    if t > block.end:
        return t - block.end
    return 0.0


def overlapping_trigger_blocks(blocks: Sequence, t: float, pre: float, post: float) -> list:
    """The blocks overlapping the window ``[t - pre, t + post]`` (a block that
    only touches it counts), in their order; the candidates of
    :func:`select_trigger_block`.

    The one overlap test: the reader and
    :mod:`labquake_explorer.data.check_blocks` both go through it with the
    same ``t``, ``pre`` and ``post``, so a block touching a window edge is
    counted the same way by both, to the last bit."""
    t = float(t)
    lo, hi = t - float(pre), t + float(post)
    return [b for b in blocks if b.start <= hi and b.end >= lo]


def select_trigger_block(blocks: Sequence, t: float, pre: float, post: float):
    """The one trigger block an event's record is copied from.

    ``blocks`` are trigger records (anything with ``block``, ``start``,
    ``end`` and ``trigger_time``, such as :class:`tpc5.BlockInfo`) and ``t``
    the event time, all on one clock; the window is ``[t - pre, t + post]``
    and the candidates are the blocks overlapping it
    (:func:`overlapping_trigger_blocks`).

    1. Candidates containing ``t``: the one overlapping the window most (the
       lower block number on a tie).
    2. Otherwise the candidate nearest to ``t``, the distance being the gap
       between ``t`` and the block's ``[start, end]``; ties go to the smaller
       ``|trigger_time - t|``, then to the lower block number.
    3. None when no block overlaps the window.

    Durations are compared to the nanosecond.  Used by :class:`Tpc5Source` to
    read events and by :mod:`labquake_explorer.data.check_blocks` to check them.
    """
    t = float(t)
    lo, hi = t - float(pre), t + float(post)
    near = overlapping_trigger_blocks(blocks, t, pre, post)
    inside = [b for b in near if b.start <= t <= b.end]
    if inside:
        return min(inside, key=lambda b: (-_ns(min(b.end, hi) - max(b.start, lo)), b.block))
    if not near:
        return None
    return min(near, key=lambda b: (_ns(_gap(b, t)), _ns(abs(b.trigger_time - t)), b.block))


@register_source
class Tpc5Source(Source):
    """A tpc5 file: block 1 is the whole-run (lowest-rate) record, any further
    blocks are high-rate records around triggers, all on one clock.  Only the
    first measurement group is read, with one block table for every channel
    (:meth:`open` refuses files where that does not hold)."""

    format = "tpc5"

    def __init__(self, path, fields, channel_numbers: Sequence[int], blocks: Sequence[tpc5.BlockInfo],
                 channel_names: Optional[Sequence[str]] = None, **kwargs):
        super().__init__(path, fields, **kwargs)
        self.channel_numbers = [int(c) for c in channel_numbers]
        self.channel_names = [str(n) for n in channel_names] if channel_names else [str(c) for c in self.channel_numbers]
        if len(self.channel_numbers) != len(self.fields):
            raise ValueError("fields and channel_numbers must have the same length")
        self.blocks = list(blocks)
        self.continuous = tpc5.continuous_block(self.blocks)
        self.trigger_blocks = tpc5.trigger_blocks(self.blocks)

    @classmethod
    def open(cls, path, channel_map: Optional[Mapping[str, str] | Callable[[str], str]] = None,
             channels: Optional[Sequence[int]] = None, **kwargs) -> "Tpc5Source":
        """Open a tpc5 file; ``channel_map`` turns a channel label ('A1') into a field name.

        Raises ValueError listing :func:`tpc5.layout_problems` when the file
        has more than one measurement group or the selected channels
        (``channels``, channel numbers; default: all) do not share one block
        table, since such a file would be read wrongly."""
        with h5py.File(path, "r") as f:
            problems = tpc5.layout_problems(f, channels)
            if problems:
                raise ValueError(f"{Path(path).name}: this tpc5 layout cannot be read:\n  - " + "\n  - ".join(problems))
            infos = tpc5.list_channels(f)
            if channels is not None:
                wanted = {int(c) for c in channels}
                infos = [c for c in infos if c["number"] in wanted]
            blocks = tpc5.list_blocks(f, infos[0]["number"])
        names = [c["name"] for c in infos]
        fields = [_map_field(short_label(n), channel_map) for n in names]
        return cls(path, fields, [c["number"] for c in infos], blocks, channel_names=names, **kwargs)

    @property
    def start_time(self) -> str:
        return self.continuous.start_time

    def time_history(self, decimation: int = 1, dtype=np.float32):
        with h5py.File(self.path, "r") as f:
            t, data = tpc5.read_block(f, self.continuous.block, self.channel_numbers, dtype=np.float64)
        if decimation > 1:
            data = block_mean(data, decimation)
            t = block_mean(t, decimation)
        return self.to_run_time(t), np.asarray(data, dtype=dtype)

    def trigger_times(self) -> list:
        return [b.trigger_time + self.time_offset for b in self.trigger_blocks]

    def record_for(self, t_run: float) -> Optional[tpc5.BlockInfo]:
        """The high-rate block containing a run time; the whole-run block when the
        file has no trigger blocks; None otherwise."""
        t = self.to_file_time(t_run)
        if self.trigger_blocks:
            return tpc5.find_block(self.trigger_blocks, t)
        return self.continuous if self.continuous.contains(t) else None

    def block_for_window(self, t_run: float, pre: float, post: float) -> Optional[tpc5.BlockInfo]:
        """The trigger block an event at ``t_run`` gets with the window
        ``[t_run - pre, t_run + post]`` (:func:`select_trigger_block`): the
        block containing ``t_run`` that overlaps the window most, else the
        block in the window nearest to ``t_run``; None when no trigger block
        overlaps the window."""
        return select_trigger_block(self.trigger_blocks, self.to_file_time(t_run), pre, post)

    def waveform(self, event_time, pre, post, fields=None, dtype=np.float32) -> Optional[Window]:
        """With trigger blocks: the whole block :meth:`block_for_window` picks
        (every sample, pre-trigger part included, never trimmed to the window),
        or None; without: the window of the continuous record (:meth:`Source.waveform`)."""
        if not self.trigger_blocks:
            return super().waveform(event_time, pre, post, fields, dtype)
        block = self.block_for_window(event_time, pre, post)
        if block is None:
            return None
        return self.read_record(block, fields, dtype)

    def read_record(self, block, fields: Optional[Sequence[str]] = None, dtype=np.float32) -> Window:
        """Every sample of one block (a :class:`tpc5.BlockInfo` or a block
        number), from its first pre-trigger sample to its last, on the run
        clock; ``fields`` defaults to the source's ``event_fields``."""
        idx = self.field_indices(self.event_fields if fields is None else fields)
        channels = [self.channel_numbers[i] for i in idx]
        info = block if isinstance(block, tpc5.BlockInfo) else next((b for b in self.blocks if b.block == int(block)), None)
        with h5py.File(self.path, "r") as f:
            if info is None:
                info = next((b for b in tpc5.list_blocks(f, channels[0]) if b.block == int(block)), None)
                if info is None:
                    raise KeyError(f"{self.path.name}: no block {int(block)}")
            t, data = tpc5.read_block(f, info.block, channels, dtype=np.float64)
        return Window(self.to_run_time(t), np.asarray(data, dtype=dtype), [self.fields[i] for i in idx],
                      info.sample_rate, info.block, trigger_time=info.trigger_time + self.time_offset)

    def read_window(self, t_from, t_to, fields=None, dtype=np.float32) -> Optional[Window]:
        block = self.record_for(0.5 * (float(t_from) + float(t_to)))
        if block is None:
            return None
        idx = self.field_indices(fields)
        lo = max(self.to_file_time(t_from), block.start)
        hi = min(self.to_file_time(t_to), block.end)
        with h5py.File(self.path, "r") as f:
            t, data = tpc5.read_window(f, block, lo, hi, [self.channel_numbers[i] for i in idx], dtype=np.float64)
        return Window(self.to_run_time(t), np.asarray(data, dtype=dtype), [self.fields[i] for i in idx],
                      block.sample_rate, block.block)

    def to_reference(self, base_dir) -> dict:
        ref = super().to_reference(base_dir)
        ref.update({
            "channel_numbers": list(self.channel_numbers),
            "channel_names": list(self.channel_names),
            "start_time": self.start_time,
            "continuous": self.continuous.as_dict(),
            "blocks": tpc5.block_table(self.trigger_blocks),
        })
        ref["blocks"]["trigger_time_run"] = self.trigger_times()
        return ref

    @classmethod
    def from_reference(cls, ref, base_dir, run=None) -> "Tpc5Source":
        cont = ref["continuous"]
        blocks = [tpc5.BlockInfo(int(cont["block"]), float(cont["sample_rate"]), int(cont["n_samples"]),
                                 int(cont["trigger_sample"]), float(cont["trigger_time"]), str(cont.get("start_time", "")))]
        blocks += tpc5.blocks_from_table(ref["blocks"], str(ref.get("start_time", "")))
        meta = {k: v for k, v in ref.items() if k not in _TPC5_KEYS}
        return cls(Path(base_dir) / str(ref["filename"]), list(ref["fields"]), ref["channel_numbers"], blocks,
                   channel_names=ref.get("channel_names"), meta=meta, **cls._common_kwargs(ref))


_TPC5_KEYS = {"format", "filename", "time_offset", "fields", "event_fields", "event_window_s",
              "channel_numbers", "channel_names", "start_time", "continuous", "blocks", "raw_data"}


def _map_field(label: str, channel_map) -> str:
    if channel_map is None:
        return label
    if callable(channel_map):
        return str(channel_map(label))
    return str(channel_map.get(label, label))


# ---------------------------------------------------------------------------
# National Instruments npz
# ---------------------------------------------------------------------------
@register_source
class NINpzSource(Source):
    """An NI ``.npz`` recording (see :mod:`labquake_explorer.utils.ni_npz`)."""

    format = "ni_npz"

    def __init__(self, path, fields, decimation: int = 1, record: Optional[NIRecord] = None, **kwargs):
        super().__init__(path, fields, **kwargs)
        self.decimation = int(decimation)
        self._record = record

    @classmethod
    def open(cls, path, channel_map: Optional[Mapping[str, str] | Callable[[str], str]] = None, **kwargs) -> "NINpzSource":
        record = open_ni_npz(path)
        fields = [_map_field(name, channel_map) for name in record.channels]
        return cls(path, fields, record=record, **kwargs)

    @property
    def record(self) -> NIRecord:
        if self._record is None:
            self._record = open_ni_npz(self.path)
        return self._record

    def time_history(self, decimation: Optional[int] = None, dtype=np.float32):
        factor = self.decimation if decimation is None else int(decimation)
        t, data = self.record.decimate(factor, dtype=dtype)
        return self.to_run_time(t), data

    def trigger_times(self) -> list:
        t = self.record.trigger_time
        return [] if t is None else [t + self.time_offset]

    def read_window(self, t_from, t_to, fields=None, dtype=np.float32) -> Optional[Window]:
        rec = self.record
        centre = self.to_file_time(0.5 * (float(t_from) + float(t_to)))
        if not rec.contains(centre):
            return None
        idx = self.field_indices(fields)
        lo = max(self.to_file_time(t_from), 0.0)
        hi = min(self.to_file_time(t_to), (rec.n_samples - 1) / rec.sample_rate)
        t, data = rec.read_window(lo, hi, idx, dtype=np.float64)
        return Window(self.to_run_time(t), np.asarray(data, dtype=dtype), [self.fields[i] for i in idx], rec.sample_rate)

    def to_reference(self, base_dir) -> dict:
        ref = super().to_reference(base_dir)
        info = self.record.as_dict()
        for key in ("format", "labels", "member_offsets"):   # the file's own channel labels and zip offsets: not needed in the run
            info.pop(key, None)
        for key, value in info.items():
            ref.setdefault(key, value)
        ref["decimation"] = self.decimation
        return ref

    @classmethod
    def from_reference(cls, ref, base_dir, run=None) -> "NINpzSource":
        meta = {k: v for k, v in ref.items() if k not in _NI_KEYS}
        return cls(Path(base_dir) / str(ref["filename"]), list(ref["fields"]),
                   decimation=int(ref.get("decimation", 1)), meta=meta, **cls._common_kwargs(ref))


_NI_KEYS = {"format", "filename", "time_offset", "fields", "event_fields", "event_window_s", "decimation",
            "sample_rate", "n_samples", "duration_s", "dtype", "channels", "labels", "trigger_sample_index",
            "trigger_time", "member_offsets", "raw_data"}


# ---------------------------------------------------------------------------
# PSU-era single-block tpc5 referenced from the legacy ``strain`` layout
# ---------------------------------------------------------------------------
@register_source
class LegacyTpc5Source(Source):
    """A single-block tpc5 whose first sample sits at ``run_t0 + time_offset``
    on the run clock (the convention of the PSU-era ``strain`` dict)."""

    format = "tpc5_legacy"

    def __init__(self, path, fields, run_t0: float, channel_numbers: Sequence[int], block: tpc5.BlockInfo, **kwargs):
        super().__init__(path, fields, **kwargs)
        self.run_t0 = float(run_t0)
        self.channel_numbers = [int(c) for c in channel_numbers]
        self.block = block

    @classmethod
    def open(cls, path, run_t0: float, **kwargs) -> "LegacyTpc5Source":
        with h5py.File(path, "r") as f:
            numbers = tpc5.channel_numbers(f)
            block = tpc5.list_blocks(f, numbers[0])[0]
        return cls(path, [f"ch{c}" for c in numbers], run_t0, numbers, block, **kwargs)

    @property
    def sample_rate(self) -> float:
        return self.block.sample_rate

    def sample_at(self, t_run: float) -> int:
        i = int(round((float(t_run) - self.run_t0 - self.time_offset) * self.block.sample_rate))
        return min(max(i, 0), self.block.n_samples - 1)

    def time_history(self, decimation: int = 1, dtype=np.float32):
        with h5py.File(self.path, "r") as f:
            _, data = tpc5.read_block(f, self.block.block, self.channel_numbers, dtype=np.float64)
        t = self.run_t0 + self.time_offset + np.arange(self.block.n_samples) / self.block.sample_rate
        if decimation > 1:
            data, t = block_mean(data, decimation), block_mean(t, decimation)
        return t, np.asarray(data, dtype=dtype)

    def read_window(self, t_from, t_to, fields=None, dtype=np.float32) -> Optional[Window]:
        i0, i1 = self.sample_at(t_from), self.sample_at(t_to) + 1
        if i1 - i0 < 2:
            return None
        idx = self.field_indices(fields)
        with h5py.File(self.path, "r") as f:
            _, data = tpc5.read_block(f, self.block.block, [self.channel_numbers[i] for i in idx], i0, i1, dtype=np.float64)
        t = self.run_t0 + self.time_offset + np.arange(i0, i1) / self.block.sample_rate
        return Window(t, np.asarray(data, dtype=dtype), [self.fields[i] for i in idx], self.block.sample_rate, self.block.block)

    @classmethod
    def from_reference(cls, ref, base_dir, run=None) -> "LegacyTpc5Source":
        if run is None or "time" not in run:
            raise ValueError("a legacy strain reference needs the run (its time axis sets the clock)")
        return cls.open(Path(base_dir) / str(ref["filename"]), float(np.asarray(run["time"])[0]),
                        time_offset=float(ref.get("time_offset", 0.0)))


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def is_legacy_strain(value) -> bool:
    return (isinstance(value, Mapping) and "format" not in value and "filename" in value
            and "time_offset" in value and "raw" in value)


def reference_format(value) -> Optional[str]:
    """The source format a run entry points to, or None if it is not a reference."""
    if not isinstance(value, Mapping):
        return None
    fmt = value.get("format")
    if fmt in SOURCE_FORMATS:
        return str(fmt)
    if is_legacy_strain(value):
        return LegacyTpc5Source.format
    return None


def run_sources(run: Mapping) -> dict:
    """``{key: reference}`` for every top-level entry of a run that points at a raw file."""
    return {key: value for key, value in run.items() if reference_format(value) is not None}


def open_source(ref: Mapping, base_dir, run: Optional[Mapping] = None) -> Source:
    fmt = reference_format(ref)
    if fmt is None:
        raise ValueError(f"not a raw-data reference: {list(ref)[:8]}")
    return SOURCE_FORMATS[fmt].from_reference(ref, base_dir, run)


# ---------------------------------------------------------------------------
# waveform blocks inside an extracted event
# ---------------------------------------------------------------------------
def _is_legacy_block(value) -> bool:
    original = value.get("original") if isinstance(value, Mapping) else None
    return isinstance(original, Mapping) and "time" in original and "raw" in original


def is_waveform_block(value) -> bool:
    """True for a full-rate record of an event: a channel array with a ``time``
    axis (``event['waveform'][recorder]``) or the PSU-era ``strain`` dict
    holding ``original: {time, raw}``."""
    if not isinstance(value, Mapping):
        return False
    return _is_legacy_block(value) or ("time" in value and is_channel_array(value))


def waveform_blocks(event: Mapping) -> dict:
    """``{key: block}`` for every full-rate record of an event, in event order:
    ``'waveform/<recorder>'`` for each recorder's record and ``'strain'`` for
    the PSU layout."""
    out = {}
    if not isinstance(event, Mapping):
        return out
    for key, value in event.items():
        if key == WAVEFORM and isinstance(value, Mapping):
            for recorder, block in value.items():
                if is_waveform_block(block):
                    out[f"{WAVEFORM}/{recorder}"] = block
        elif is_waveform_block(value):
            out[str(key)] = value
    return out


def waveform_block(event: Mapping, key: str) -> Optional[Mapping]:
    """The block a :func:`waveform_blocks` key names, or None."""
    current = event
    for part in str(key).split("/"):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current if is_waveform_block(current) else None


def waveform_time(block: Mapping) -> np.ndarray:
    """The record's time axis on the run clock."""
    source = block["original"] if _is_legacy_block(block) else block
    return np.asarray(source["time"], dtype=np.float64).ravel()


def waveform_data(block: Mapping) -> np.ndarray:
    """The record's samples, ``(n_channels, n)``."""
    data = np.asarray(block["original"]["raw"] if _is_legacy_block(block) else block["data"])
    return data[None, :] if data.ndim == 1 else data


def waveform_channels(block: Mapping) -> list:
    """Channel labels of a record: its ``channels`` (or legacy ``fields``) when
    they match the data, else indices."""
    n = int(waveform_data(block).shape[0])
    names = channel_names(block) if is_channel_array(block) else [str(f) for f in (block.get("fields") or [])]
    return names if len(names) == n else [str(i) for i in range(n)]


def waveform_store(block) -> dict:
    """Where a view keeps per-record results such as ``picked_idx`` and
    ``rupture_arrival_time``: the block itself, or ``original`` for the PSU layout."""
    if not isinstance(block, Mapping):
        return {}
    return block["original"] if _is_legacy_block(block) else block


def waveform_trigger_time(block, run: Optional[Mapping] = None, key: Optional[str] = None) -> Optional[float]:
    """The record's own trigger time on the run clock (where the recorder fired,
    the PZT trigger for an Elsys block): stored with the record, else looked up
    by block number in the run's reference for the recorder (``key`` is the
    :func:`waveform_blocks` key, ``'waveform/elsys'``); None for a continuous
    record or when nothing is known."""
    if not isinstance(block, Mapping):
        return None
    stored = block.get("trigger_time")
    if stored is not None:
        try:
            return float(np.asarray(stored).ravel()[0])
        except (TypeError, ValueError, IndexError):
            return None
    number = block.get("block")
    if number is None or run is None or not key:
        return None
    ref = run.get(str(key).split("/")[-1]) if isinstance(run, Mapping) else None
    table = ref.get("blocks") if isinstance(ref, Mapping) else None
    if not isinstance(table, Mapping) or "block" not in table or "trigger_time_run" not in table:
        return None
    try:
        numbers = [int(b) for b in np.asarray(table["block"]).ravel()]
        times = np.asarray(table["trigger_time_run"], dtype=float).ravel()
        return float(times[numbers.index(int(number))])
    except (TypeError, ValueError, IndexError):
        return None


ARRIVALS_KEY = "arrivals"
ARRIVALS_VERSION = 1
_ARRIVAL_FLAGS = ("enabled_channels", "fitting_channels", "locations")


def read_arrivals(block, event: Optional[Mapping] = None) -> dict:
    """Pick Arrivals' results for a record, normalised to ``picked_idx``,
    ``arrival_time`` (s, per channel), ``enabled_channels``, ``fitting_channels``,
    ``locations`` (mm) and ``rupture_speed`` (m/s); only the keys that exist.

    A waveform record keeps them under ``block['arrivals']``.  The PSU strain
    layout keeps them spread over ``strain/original`` (``picked_idx``,
    ``rupture_arrival_time``), ``strain`` (the flags and locations) and
    ``event['rupture_speed']``, which is why ``event`` can be passed.
    """
    out: dict = {}
    if not isinstance(block, Mapping):
        return out
    if _is_legacy_block(block):
        original = block["original"]
        if "picked_idx" in original:
            out["picked_idx"] = original["picked_idx"]
        if "rupture_arrival_time" in original:
            out["arrival_time"] = original["rupture_arrival_time"]
        for key in _ARRIVAL_FLAGS:
            if key in block:
                out[key] = block[key]
        if isinstance(event, Mapping) and event.get("rupture_speed") is not None:
            out["rupture_speed"] = event["rupture_speed"]
        return out
    stored = block.get(ARRIVALS_KEY)
    if isinstance(stored, Mapping):
        out.update({k: v for k, v in stored.items() if k != "version"})
        return out
    for key in ("picked_idx",) + _ARRIVAL_FLAGS:          # records written before 'arrivals' existed
        if key in block:
            out[key] = block[key]
    if "rupture_arrival_time" in block:
        out["arrival_time"] = block["rupture_arrival_time"]
    return out


def write_arrivals(block, arrivals: Mapping, event=None) -> None:
    """Store Pick Arrivals' results with the record (the inverse of
    :func:`read_arrivals`).  A missing or None ``rupture_speed`` means the
    picks define no line and removes any stored speed."""
    speed = arrivals.get("rupture_speed")
    if _is_legacy_block(block):
        original = block["original"]
        if "picked_idx" in arrivals:
            original["picked_idx"] = arrivals["picked_idx"]
        if "arrival_time" in arrivals:
            original["rupture_arrival_time"] = arrivals["arrival_time"]
        for key in _ARRIVAL_FLAGS:
            if key in arrivals:
                block[key] = arrivals[key]
        if isinstance(event, dict):
            if speed is None:
                event.pop("rupture_speed", None)
            else:
                event["rupture_speed"] = float(speed)
        return
    record = {"version": ARRIVALS_VERSION}
    record.update({k: v for k, v in arrivals.items() if k != "rupture_speed" and v is not None})
    if speed is not None:
        record["rupture_speed"] = float(speed)
    block[ARRIVALS_KEY] = record
    for key in ("picked_idx", "rupture_arrival_time") + _ARRIVAL_FLAGS:   # the layout before 'arrivals'
        block.pop(key, None)


def pick_waveform_block(event: Mapping, prefer_fields: Sequence[str] = (),
                        prefer_keys: Sequence[str] = ("strain",)) -> Optional[str]:
    """The key of the record a view should open first: the first block that has
    a channel starting with one of ``prefer_fields``, else the first key in
    ``prefer_keys`` that exists, else the first block; None without any."""
    blocks = waveform_blocks(event)
    if not blocks:
        return None
    for key, block in blocks.items():
        if any(c.startswith(p) for c in waveform_channels(block) for p in prefer_fields):
            return key
    for key in prefer_keys:
        if key in blocks:
            return key
    return next(iter(blocks))


def block_channel_labels(block: Mapping) -> list:
    """Alias of :func:`waveform_channels`."""
    return waveform_channels(block)


__all__ = ["SOURCE_FORMATS", "WAVEFORM", "register_source", "Source", "Window", "Tpc5Source", "select_trigger_block",
           "overlapping_trigger_blocks", "NINpzSource", "LegacyTpc5Source", "short_label", "block_mean", "is_legacy_strain", "reference_format",
           "run_sources", "open_source", "is_waveform_block", "waveform_blocks", "waveform_block", "waveform_time",
           "waveform_data", "waveform_channels", "waveform_store", "waveform_trigger_time", "pick_waveform_block",
           "block_channel_labels", "ARRIVALS_KEY", "read_arrivals", "write_arrivals"]
