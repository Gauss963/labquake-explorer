import csv
import shutil

import h5py
import numpy as np
import pytest

from labquake_explorer.data import check_blocks as C
from labquake_explorer.data.data_manager import DataManager
from labquake_explorer.data.event_processor import EventProcessor
from labquake_explorer.data.sources import Tpc5Source
from labquake_explorer.preprocessing import run_from_sources
from labquake_explorer.utils import tpc5
from tests.synthetic import write_tpc5

FS_TRIGGER, PRE, POST = 20_000.0, 0.02, 0.08              # trigger blocks: 2000 samples, 20 ms before the trigger
TRIGGERS = (0.30, 0.80, 1.10, 1.60, 2.30, 2.35)            # file clock, blocks 2..7
OFFSET = 0.25                                              # run clock = file clock + 0.25 s
WINDOW = (-0.15, 0.15)
EVENTS = (0.31,     # inside b2
          0.97,     # between b3 (crosses the window start, nearer) and b4 (crosses the window end)
          1.75,     # after b5, the only block in the window
          2.36,     # inside b6 and b7, which overlap each other: equal overlap, the lower number
          3.30)     # no block in the window
EXPECTED = ["b2", "b3", "b5", "b6", "-"]
SIGNALS = {"A1 (0.A1)": lambda t: 0.05 * np.sin(2 * np.pi * 37.0 * t), "A2 (0.A2)": lambda t: 0.1 + 0.0 * t}
CHANNEL_MAP = {"A1": "pzt_1", "A2": "pzt_2"}
RAW_NAME = "t0001_01_8MPa_run1.tpc5"


def index_at(run, t):
    return int(np.argmin(np.abs(run["time"] - t)))


def make_experiment(folder, suffix=".h5", event_fields=None):
    """A saved experiment whose events were extracted with the current reader:
    run 0 has the tpc5 reference and the events above, run 1 has neither."""
    raw = folder / RAW_NAME
    write_tpc5(raw, SIGNALS, fs_continuous=1000.0, duration=4.0, fs_trigger=FS_TRIGGER,
               trigger_times=TRIGGERS, pre=PRE, post=POST)
    source = Tpc5Source.open(raw, CHANNEL_MAP, time_offset=OFFSET, event_fields=event_fields)
    run = run_from_sources({"elsys": source}, folder, name="run1")
    path = folder / f"t0001{suffix}"
    indices = [index_at(run, t + OFFSET) for t in EVENTS]
    run["events"] = EventProcessor(path).extract_events(run, indices, window=0.15, pre=-WINDOW[0], post=WINDOW[1])
    run["event_extraction"] = {"version": 1, "event_indices": indices, "start_s": WINDOW[0], "end_s": WINDOW[1],
                               "n_events": len(indices)}
    dm = DataManager()
    dm.data = {"name": "t0001", "runs": [run, {"name": "run2", "time": np.arange(3.0)}]}
    dm.save_file(path)
    return path


def edit(path, change):
    """Load an experiment, apply ``change(data)`` and save it in place."""
    dm = DataManager()
    dm.load_file(path)
    change(dm.data)
    dm.save_file(path)


def drop_record(data, j, key="elsys"):
    """What the reader before select_trigger_block left: no record, a note instead."""
    event = data["runs"][0]["events"][j]
    del event["waveform"][key]
    if not event["waveform"]:
        del event["waveform"]
    event["notes"] = [f"{key}: no raw record covers {event['event_time']:.4f} s"]


def statuses(report):
    return [row.status for row in report.rows]


def test_events_extracted_by_the_reader_pass(tmp_path, capsys):
    path = make_experiment(tmp_path)
    report = C.check_file(path)
    assert report.skipped == [1]
    assert statuses(report) == ["OK", "OK", "OK", "OK", "NO BLOCK"]
    assert [row.expected_text for row in report.rows] == EXPECTED
    assert [len(row.overlapping) for row in report.rows] == [1, 2, 1, 2, 0]
    a, b, c, d, e = report.rows
    assert a.missing == [] and a.flags == []
    assert b.missing == ["b4 [+0.110, +0.210] s (one-block rule)"]
    assert b.flags == ["b3 crosses window start", "b4 crosses window end"]
    assert c.flags == ["b5 crosses window start"] and c.missing == []
    assert d.flags == ["overlapping blocks b6/b7"] and d.missing == ["b7 [-0.030, +0.070] s (one-block rule)"]
    assert e.record is None and e.flags == [] and report.n_truncated == 0
    # every record is the whole block, on the run clock, also for an event outside its block
    for row, start in ((a, 0.28), (b, 0.78), (c, 1.58), (d, 2.28)):
        assert row.record.n == 2000
        assert row.record.first == pytest.approx(start + OFFSET, abs=1e-9)
        assert row.record.last == pytest.approx(start + OFFSET + 1999 / FS_TRIGGER, abs=1e-9)
    assert report.exit_code() == 0
    assert C.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "5 events: 4 OK, 0 NOT LOADED, 0 WRONG BLOCK, 1 NO BLOCK; 0 pre-trigger truncated" in out
    assert "not checked (no events or no tpc5 reference): runs 1" in out
    header = next(line for line in out.splitlines() if line.startswith("run  event"))
    assert header.split("  ")[0] == "run" and "missing blocks" in header and header.rstrip().endswith("flags")


EDGE_OFFSET = 2.388939
EDGE_TRIGGERS = (0.303, 0.899, 1.533, 2.30)       # blocks 2..5 of 2001 samples: [trigger - 0.02, trigger + 0.08] s
EDGE_EVENTS = ((0.133, "end"),     # file clock; the window end touches b2's start,
               (1.129, "start"),   # the window start touches b3's end,
               (1.763, "start"),   # b4's end,
               (2.13, "end"))      # b5's start


def test_a_block_touching_the_window_is_counted_as_the_reader_counts_it(tmp_path, capsys):
    raw = tmp_path / RAW_NAME
    write_tpc5(raw, SIGNALS, fs_continuous=1000.0, duration=4.0, fs_trigger=FS_TRIGGER,
               trigger_times=EDGE_TRIGGERS, pre=PRE, post=0.08005)
    source = Tpc5Source.open(raw, CHANNEL_MAP, time_offset=EDGE_OFFSET)
    run = run_from_sources({"elsys": source}, tmp_path, name="run1")
    indices = [index_at(run, t + EDGE_OFFSET) for t, _ in EDGE_EVENTS]
    times = [float(run["time"][i]) for i in indices]
    pre, post = -WINDOW[0], WINDOW[1]
    # the premise: the touching edge falls on the other side of the block's edge when the window
    # is computed on the run clock first, (t + end_s) - time_offset, instead of (t - time_offset) + end_s
    for t, (_, edge), b in zip(times, EDGE_EVENTS, source.trigger_blocks):
        if edge == "end":
            assert (b.start <= (t - EDGE_OFFSET) + post) != (b.start <= (t + WINDOW[1]) - EDGE_OFFSET)
        else:
            assert (b.end >= (t - EDGE_OFFSET) - pre) != (b.end >= (t + WINDOW[0]) - EDGE_OFFSET)
    path = tmp_path / "t0001.h5"
    run["events"] = EventProcessor(path).extract_events(run, indices, window=0.15, pre=pre, post=post)
    run["event_extraction"] = {"version": 1, "event_indices": indices, "start_s": WINDOW[0], "end_s": WINDOW[1]}
    dm = DataManager()
    dm.data = {"name": "t0001", "runs": [run]}
    dm.save_file(path)
    assert [e.get("waveform", {}).get("elsys", {}).get("block") for e in run["events"]] == [2, 3, None, None]
    report = C.check_file(path)
    assert statuses(report) == ["OK", "OK", "NO BLOCK", "NO BLOCK"]
    for row, t in zip(report.rows, times):
        picked = source.block_for_window(t, pre, post)
        assert row.expected_text == (f"b{picked.block}" if picked else "-")
        assert len(row.overlapping) == (picked is not None)
    assert report.exit_code() == 0
    assert C.main([str(path), "--fill"]) == 0
    assert "--fill: no NOT LOADED event, nothing to add" in capsys.readouterr().out


def one_sample_block(number, t):
    """A raw trigger block of one sample at ``t`` (file clock): it starts and ends exactly at ``t``."""
    return C.RawBlock(tpc5.READER_MEASUREMENT, tpc5.BlockInfo(number, 1000.0, 1, 0, t))


def test_check_event_agrees_with_itself_and_the_reader_at_the_window_edges(tmp_path):
    """Blocks touching the window, the edge computed on the file clock or on the run clock
    first: the blocks the check counts in the window, the block it expects and the block
    the reader picks agree, also when the window comes from the event's own time axis."""
    rng = np.random.default_rng(3)
    cases = [(0.1, 0.8098), (2.388939, 0.8687), (0.1, 2.9326)]       # time_offset, event time (run clock)
    cases += [(float(o), round(float(t), 4)) for o, t in zip(rng.choice([0.1, 2.388939, -1.2345678, 13.37], 200),
                                                              rng.uniform(0.0, 400.0, 200))]
    start_s, end_s = -5.0, 5.0
    continuous = tpc5.BlockInfo(1, 1.0, 1000, 0, -500.0)
    for offset, t in cases:
        ref = C.Reference("elsys", RAW_NAME, offset)
        edges = [(t + end_s) - offset, (t - offset) + end_s, (t + start_s) - offset, (t - offset) + start_s]
        for window, first, last in (((start_s, end_s), None, None), (None, t + start_s, t + end_s)):
            run = C.Run(0, "run1", window, [ref], [])
            event = C.Event(0, t, first, last, {})
            pre, post = (-start_s, end_s) if window else (t - first, last - t)
            for k, edge in enumerate(edges):
                raw = C.RawFile(tmp_path / RAW_NAME, trigger=[one_sample_block(2 + k, edge)])
                row = C.check_event(run, ref, raw, event)
                reader = Tpc5Source(raw.path, ["pzt_1"], [1], [continuous, raw.trigger[0].info], time_offset=offset)
                picked = reader.block_for_window(t, pre, post)
                assert len(row.overlapping) == (row.expected is not None) == (picked is not None), (offset, t, k)
                assert (row.status == "NO BLOCK") == (row.cells()[3] == "0")
                if picked is not None:
                    assert row.expected.info is picked


def test_fill_adds_the_missing_records_and_keeps_a_backup(tmp_path, capsys):
    path = make_experiment(tmp_path)
    reference = DataManager()
    reference.load_file(path)
    extracted = reference.data["runs"][0]["events"]

    def old_reader(data):                       # the events outside every block got nothing
        drop_record(data, 1)
        drop_record(data, 2)
    edit(path, old_reader)
    report = C.check_file(path)
    assert statuses(report) == ["OK", "NOT LOADED", "NOT LOADED", "OK", "NO BLOCK"]
    assert report.rows[1].missing == ["b3 [-0.190, -0.090] s", "b4 [+0.110, +0.210] s (one-block rule)"]
    assert C.main([str(path)]) == 1
    capsys.readouterr()

    assert C.main([str(path), "--fill"]) == 0
    out = capsys.readouterr().out
    assert "runs/[0]/events/[1]: added waveform/elsys from b3 (2000 samples, 2 channels)" in out
    assert "runs/[0]/events/[2]: added waveform/elsys from b5" in out
    assert out.count("expected block") == 2                          # the table, then the table again
    assert out.rstrip().endswith("5 events: 4 OK, 0 NOT LOADED, 0 WRONG BLOCK, 1 NO BLOCK; 0 pre-trigger truncated")
    backup = tmp_path / "t0001.before-fill.h5"
    assert statuses(C.check_file(backup))[1:3] == ["NOT LOADED", "NOT LOADED"]   # the file as it was
    assert statuses(C.check_file(path)) == ["OK", "OK", "OK", "OK", "NO BLOCK"]
    again = DataManager()
    again.load_file(path)
    events = again.data["runs"][0]["events"]
    for j, label in ((1, "b3"), (2, "b5")):
        filled, original = events[j]["waveform"]["elsys"], extracted[j]["waveform"]["elsys"]
        assert set(filled) == set(original)
        for key in ("data", "time"):
            np.testing.assert_array_equal(filled[key], original[key])
        assert filled["data"].dtype == np.float32 and filled["channels"] == original["channels"] == ["pzt_1", "pzt_2"]
        for key in ("block", "trigger_time", "sample_rate", "filename", "unit"):
            assert filled[key] == original[key], key
        assert events[j]["notes"] == [f"elsys: {label} added by check_blocks --fill"]
    np.testing.assert_array_equal(events[0]["waveform"]["elsys"]["data"], extracted[0]["waveform"]["elsys"]["data"])

    # nothing left to add: no save, the backup stays as it was
    stamp = backup.stat().st_mtime_ns
    assert C.main([str(path), "--fill"]) == 0
    assert "nothing to add" in capsys.readouterr().out
    assert backup.stat().st_mtime_ns == stamp


def test_fill_never_replaces_a_record_nor_an_existing_backup(tmp_path, capsys):
    path = make_experiment(tmp_path, event_fields=["pzt_1"])
    source = Tpc5Source.open(tmp_path / RAW_NAME, CHANNEL_MAP, time_offset=OFFSET, event_fields=["pzt_1"])

    def wrong_and_missing(data):
        run = data["runs"][0]
        event = run["events"][0]
        event["waveform"]["elsys"] = EventProcessor.waveform_record(run, "elsys", source, source.read_record(3))
        drop_record(data, 2)
    edit(path, wrong_and_missing)
    backup = tmp_path / "t0001.before-fill.h5"
    backup.write_bytes(b"an earlier copy")
    report = C.check_file(path)
    assert statuses(report) == ["WRONG BLOCK", "OK", "NOT LOADED", "OK", "NO BLOCK"]
    assert report.rows[0].missing == ["b2 [-0.030, +0.070] s"] and report.rows[0].loaded_text.startswith("b3 ")

    assert C.main([str(path), "--fill"]) == 1                         # the wrong block is still there
    out = capsys.readouterr().out
    assert "t0001.before-fill.h5 exists already and was kept as it is" in out
    assert backup.read_bytes() == b"an earlier copy"
    report = C.check_file(path)
    assert statuses(report) == ["WRONG BLOCK", "OK", "OK", "OK", "NO BLOCK"]
    dm = DataManager()
    dm.load_file(path)
    events = dm.data["runs"][0]["events"]
    assert events[0]["waveform"]["elsys"]["block"] == 3                # left as it was
    assert events[2]["waveform"]["elsys"]["channels"] == ["pzt_1"]     # the reference's event_fields
    # a record that is present is never touched, even when asked to fill it
    row = report.rows[2]
    row.status = C.NOT_LOADED
    added, messages = C.fill_missing(path, report)
    assert added == 0 and messages == ["runs/[0]/events/[2]: already has waveform/elsys, left as it is"]


def test_fill_without_backup(tmp_path, capsys):
    path = make_experiment(tmp_path)
    edit(path, lambda data: drop_record(data, 2))
    assert C.main([str(path), "--fill", "--no-backup"]) == 0
    assert not (tmp_path / "t0001.before-fill.h5").exists()
    assert statuses(C.check_file(path))[2] == "OK"


def test_truncated_record_is_flagged(tmp_path):
    path = make_experiment(tmp_path)

    def trim(data):
        record = data["runs"][0]["events"][0]["waveform"]["elsys"]
        record["time"] = record["time"][100:]
        record["data"] = record["data"][:, 100:]
    edit(path, trim)
    report = C.check_file(path)
    row = report.rows[0]
    assert row.status == "OK" and row.truncated and report.n_truncated == 1
    assert row.flags == ["pre-trigger truncated (record: 1900 samples from -0.0250 s; block: 2000 from -0.0300 s)"]
    assert report.exit_code() == 1


def test_layout_notes_and_other_measurement_groups(tmp_path, capsys):
    path = make_experiment(tmp_path)
    raw = tmp_path / RAW_NAME
    with h5py.File(raw, "a") as f:                  # a second measurement group, 3 s later, and a channel short of a block
        f.copy("/measurements/00000001", "/measurements/00000002")
        for channel in f["/measurements/00000002/channels"].values():
            for name, block in channel["blocks"].items():
                if name != "00000001":
                    block.attrs["triggerTimeSeconds"] = block.attrs["triggerTimeSeconds"] + 3.0
        del f["/measurements/00000001/channels/00000002/blocks/00000007"]
    (tmp_path / "t0001_01_8MPa_run1_0002.tpc5").write_bytes(b"")
    (tmp_path / "t0001_01_8MPa_run2.tpc5").write_bytes(b"")   # another run, not a split file
    with pytest.raises(ValueError, match="2 measurement groups"):
        Tpc5Source.open(raw, CHANNEL_MAP)
    report = C.check_file(path)
    notes = report.rows[0].file_notes
    assert notes[0].startswith("2 measurement groups (00000001, 00000002); the reader reads 00000001 only")
    assert notes[1] == ("measurement 00000001: block numbers differ between channels: "
                        "channel 1: blocks 1-7; channel 2: blocks 1-6")
    assert notes[2] == ("1 other tpc5 file(s) start with 't0001_01_8MPa_run1' (TranAX split files?), "
                        "not read: t0001_01_8MPa_run1_0002.tpc5")
    # m2/b2 now spans the last event, which the reader cannot read: expected, not loaded, and not filled
    last = report.rows[4]
    assert last.status == "NOT LOADED" and last.expected_text == "m2/b2" and last.missing == ["m2/b2 [-0.020, +0.080] s"]
    assert statuses(report)[:4] == ["OK", "OK", "OK", "OK"]
    assert C.main([str(path), "--fill"]) == 1
    out = capsys.readouterr().out
    assert "file notes:" in out and "TranAX split files" in out
    assert ("runs/[0]/events/[4]: m2/b2 is in measurement group 00000002, which the reader does not read; not added"
            in out)
    assert not (tmp_path / "t0001.before-fill.h5").exists()


@pytest.mark.parametrize("suffix", [".h5", ".npz"])
def test_split_file_note_leaves_out_other_runs(tmp_path, suffix):
    path = make_experiment(tmp_path, suffix=suffix)
    for name in ("t0001_01_8MPa_run10.tpc5", "t0001_01_8MPa_run1_0002.tpc5", "t0001_01_8MPa_run1b.tpc5"):
        (tmp_path / name).write_bytes(b"")

    def read_run1b(data):                    # the second run (no events, so not checked) reads run1b
        data["runs"][1]["elsys"] = {"format": "tpc5", "filename": "t0001_01_8MPa_run1b.tpc5", "time_offset": 0.0}
    edit(path, read_run1b)
    report = C.check_file(path)
    assert report.skipped == [1]
    assert report.rows[0].file_notes == ["1 other tpc5 file(s) start with 't0001_01_8MPa_run1' (TranAX split files?), "
                                         "not read: t0001_01_8MPa_run1_0002.tpc5"]
    raw = tmp_path / RAW_NAME
    assert [p.name for p in C.sibling_files(raw)] == ["t0001_01_8MPa_run1_0002.tpc5", "t0001_01_8MPa_run1b.tpc5"]
    assert C.sibling_files(raw, [tmp_path / "t0001_01_8MPa_run1b.tpc5"]) == [tmp_path / "t0001_01_8MPa_run1_0002.tpc5"]
    assert C.sibling_files(tmp_path / "t0001_01_8MPa_run10.tpc5") == []


def test_npz_experiment_window_fallback_and_csv(tmp_path, capsys):
    path = make_experiment(tmp_path, suffix=".npz")
    out_csv = tmp_path / "rows.csv"
    assert C.main([str(path), "--csv", str(out_csv)]) == 0
    assert f"wrote {out_csv}" in capsys.readouterr().out
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(C.CSV_COLUMNS) and len(rows) == 5
    assert [r["status"] for r in rows] == ["OK", "OK", "OK", "OK", "NO BLOCK"]
    assert [r["expected_block"] for r in rows] == ["b2", "b3", "b5", "b6", ""]
    assert rows[1]["flags"] == "b3 crosses window start; b4 crosses window end"
    assert rows[1]["loaded_block"] == "b3" and rows[1]["loaded_samples"] == "2000" and rows[4]["loaded_blocks"] == "0"
    assert float(rows[0]["window_start_s"]) == pytest.approx(-0.15) and rows[0]["recorder"] == "elsys"

    def no_window(data):                         # without the extraction record: each event's own time axis
        del data["runs"][0]["event_extraction"]
    edit(path, no_window)
    report = C.check_file(path)
    assert statuses(report) == ["OK", "OK", "OK", "OK", "NO BLOCK"]
    assert report.rows[1].window[0] == pytest.approx(EVENTS[1] + OFFSET - 0.15, abs=1e-9)


def test_raw_files_elsewhere_and_missing(tmp_path, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = make_experiment(data_dir)
    moved = tmp_path / "copy" / path.name
    moved.parent.mkdir()
    shutil.copy2(path, moved)
    report = C.check_file(moved)                                       # the raw file is not next to the copy
    assert statuses(report) == ["UNCHECKED"] * 5 and report.exit_code() == 2
    assert C.main([str(moved)]) == 2
    assert "cannot read" in capsys.readouterr().out
    assert C.main([str(moved), "--base-dir", str(data_dir)]) == 0
    assert C.main([str(tmp_path / "nope.h5")]) == 2


def test_unreadable_experiment_file(tmp_path, capsys):
    for name, content in (("bad.h5", b"not an hdf5 file"), ("bad.npz", b"not an npz file"), ("empty.npz", b"")):
        bad = tmp_path / name
        bad.write_bytes(content)
        assert C.main([str(bad)]) == 2
        assert capsys.readouterr().out.startswith(f"cannot read {bad}: ")


def test_file_without_trigger_blocks(tmp_path):
    raw = tmp_path / RAW_NAME
    write_tpc5(raw, SIGNALS, fs_continuous=1000.0, duration=4.0, trigger_times=())
    run = run_from_sources({"elsys": Tpc5Source.open(raw, CHANNEL_MAP)}, tmp_path, name="run1")
    path = tmp_path / "t0001.h5"
    indices = [index_at(run, 1.0), index_at(run, 2.5)]
    run["events"] = EventProcessor(path).extract_events(run, indices, window=0.15)
    run["event_extraction"] = {"version": 1, "event_indices": indices, "start_s": -0.15, "end_s": 0.15}
    dm = DataManager()
    dm.data = {"name": "t0001", "runs": [run]}
    dm.save_file(path)
    report = C.check_file(path)                    # the continuous recorder's window: nothing to check against
    assert statuses(report) == ["NO BLOCK", "NO BLOCK"] and report.exit_code() == 0
    assert report.rows[0].flags == ["continuous record b1 loaded"] and report.rows[0].loaded_text.startswith("b1 ")
