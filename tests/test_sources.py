import numpy as np
import pytest

from labquake_explorer.data.data_manager import DataManager
from labquake_explorer.data import sources as S
from labquake_explorer.utils import tpc5
from tests.synthetic import write_ni_npz, write_tpc5


def pulse(t0, width=1e-4, amp=0.1):
    return lambda t: amp * np.exp(-((t - t0) / width) ** 2)


ELSYS_SIGNALS = {
    "A1 (0.A1)": lambda t: pulse(0.5)(t) + pulse(1.4)(t),
    "A5 (0.A5)": lambda t: 2.0 + 0.0 * t,
    "A7 (0.A7)": lambda t: 0.1 * t,
}
ELSYS_MAP = {"A1": "pzt_1", "A5": "pressure_1", "A7": "lvdt"}


@pytest.fixture
def elsys(tmp_path):
    path = tmp_path / "t0001_01_8MPa_run1.tpc5"
    write_tpc5(path, ELSYS_SIGNALS, trigger_times=(0.5, 1.4), volt_range=12.0)
    return path


def test_tpc5_source_fields_blocks_and_history(elsys):
    src = S.Tpc5Source.open(elsys, ELSYS_MAP, time_offset=2.0)
    assert src.fields == ["pzt_1", "pressure_1", "lvdt"]
    assert src.channel_numbers == [1, 2, 3] and src.channel_names[0] == "A1 (0.A1)"
    assert src.continuous.block == 1 and [b.block for b in src.trigger_blocks] == [2, 3]
    assert src.trigger_times() == pytest.approx([2.5, 3.4])
    t, d = src.time_history()
    assert d.shape == (3, 4000) and d.dtype == np.float32
    assert t[0] == pytest.approx(2.0) and t[-1] == pytest.approx(2.0 + 3999 / 2000)
    assert d[1].mean() == pytest.approx(2.0, abs=1e-3)
    t2, d2 = src.time_history(decimation=4)
    assert d2.shape == (3, 1000) and t2[0] == pytest.approx(2.0 + 1.5 / 2000)


def test_tpc5_read_window_on_run_clock(elsys):
    src = S.Tpc5Source.open(elsys, ELSYS_MAP, time_offset=2.0)
    win = src.read_window(2.499, 2.503)
    assert win is not None and win.block == 2 and win.sample_rate == 200_000.0
    assert win.fields == ["pzt_1", "pressure_1", "lvdt"]
    assert win.time[0] == pytest.approx(2.499, abs=1e-5) and win.time[-1] == pytest.approx(2.503, abs=1e-5)
    assert win.time[np.argmax(win.data[0])] == pytest.approx(2.5, abs=2e-5)   # pulse at the trigger, shifted
    # clipped to the block, subset of fields, and None away from every trigger block
    win = src.read_window(2.0, 3.0, ["lvdt"])
    assert win.fields == ["lvdt"] and win.time[0] == pytest.approx(2.498, abs=1e-5) and win.time[-1] == pytest.approx(2.508, abs=1e-5)
    assert src.read_window(3.0, 3.01) is None
    with pytest.raises(KeyError):
        src.read_window(2.499, 2.503, ["nope"])


def test_tpc5_event_window_uses_source_settings(elsys):
    src = S.Tpc5Source.open(elsys, ELSYS_MAP, event_fields=["pzt_1"], event_window_s=(0.001, 0.002))
    win = src.event_window(0.5, pre=5.0, post=5.0)
    assert win.fields == ["pzt_1"]
    assert win.time.size == pytest.approx(0.003 * 200_000 + 1, abs=1)
    assert win.data.dtype == np.float32


def test_tpc5_reference_round_trip(elsys, tmp_path):
    src = S.Tpc5Source.open(elsys, ELSYS_MAP, time_offset=2.389, event_fields=["pzt_1"],
                            event_window_s=(0.1, 0.5), meta={"clock_offset_method": "trigger"})
    ref = src.to_reference(tmp_path)
    assert ref["format"] == "tpc5" and ref["filename"] == elsys.name and ref["time_offset"] == 2.389
    assert ref["blocks"]["trigger_time_run"] == pytest.approx([2.889, 3.789])
    assert ref["clock_offset_method"] == "trigger"
    dm = DataManager()
    dm.data = {"name": "t0001", "runs": [{"name": "run1", "time": np.arange(3.0), "strain": ref}]}
    out = tmp_path / "exp.h5"
    dm.save_file(out)
    back = DataManager()
    back.load_file(out)
    run = back.get_data("runs/[0]")
    assert S.reference_format(run["strain"]) == "tpc5" and S.run_sources(run) == {"strain": run["strain"]}
    again = S.open_source(run["strain"], tmp_path, run)
    assert isinstance(again, S.Tpc5Source)
    assert again.fields == src.fields and again.time_offset == 2.389 and again.event_fields == ["pzt_1"]
    assert again.event_window_s == (0.1, 0.5) and again.meta["clock_offset_method"] == "trigger"
    assert [b.block for b in again.trigger_blocks] == [2, 3] and again.continuous.n_samples == 4000
    a = src.read_window(2.888, 2.892)
    b = again.read_window(2.888, 2.892)
    np.testing.assert_array_equal(a.data, b.data)
    np.testing.assert_allclose(a.time, b.time)


def test_ni_source(tmp_path):
    path = tmp_path / "T0001-raw-run1-20260101_000000.npz"
    write_ni_npz(path, {"n": lambda t: 3.5 + 0 * t, "s": lambda t: 0.7 + 0 * t, "e": lambda t: -0.5 - 2.0 * t},
                 fs=10_000.0, duration=2.0, trigger_sample_index=12_000)
    src = S.NINpzSource.open(path, {"ai0": "pressure_1", "ai1": "pressure_2", "ai2": "eddy_1"},
                             decimation=5, time_offset=0.0, meta={"input_range_v": 10.0})
    assert src.fields == ["pressure_1", "pressure_2", "eddy_1"]
    assert src.trigger_times() == pytest.approx([1.2])
    t, d = src.time_history()
    assert d.shape == (3, 4000) and t[1] - t[0] == pytest.approx(5 / 10_000)
    win = src.read_window(1.0, 1.001)
    assert win.sample_rate == 10_000.0 and win.time.size == 11 and win.fields[2] == "eddy_1"
    assert win.data[2, 0] == pytest.approx(-2.5, abs=1e-6)
    assert src.read_window(5.0, 5.1) is None
    ref = src.to_reference(tmp_path)
    assert ref["format"] == "ni_npz" and ref["decimation"] == 5 and ref["input_range_v"] == 10.0
    assert ref["trigger_sample_index"] == 12_000 and ref["sample_rate"] == 10_000.0
    again = S.open_source(ref, tmp_path)
    assert isinstance(again, S.NINpzSource) and again.fields == src.fields and again.decimation == 5
    assert again.meta["input_range_v"] == 10.0
    np.testing.assert_array_equal(again.read_window(1.0, 1.001).data, win.data)


def test_legacy_strain_source(tmp_path):
    path = tmp_path / "legacy.tpc5"
    write_tpc5(path, {"ch1": lambda t: 0.1 * t, "ch2": lambda t: 0.0 * t}, fs_continuous=1000.0,
               duration=2.0, trigger_times=())
    run = {"time": 10.0 + np.arange(0, 2.0, 0.01),
           "strain": {"filename": "legacy.tpc5", "time_offset": 0.25, "time": np.arange(0, 2.0, 0.1),
                      "raw": np.zeros((2, 20)), "filename_downsampled": ""}}
    assert S.reference_format(run["strain"]) == "tpc5_legacy"
    src = S.open_source(run["strain"], tmp_path, run)
    assert isinstance(src, S.LegacyTpc5Source) and src.fields == ["ch1", "ch2"] and src.sample_rate == 1000.0
    win = src.read_window(10.75, 10.76)       # file time 0.5 .. 0.51
    assert win.time[0] == pytest.approx(10.75) and win.time.size == 11
    assert win.data[0, 0] == pytest.approx(0.05, abs=2e-4)
    with pytest.raises(ValueError):
        S.open_source(run["strain"], tmp_path)   # needs the run for its clock


def test_register_source_and_block_mean():
    with pytest.raises(ValueError):
        S.register_source(type("Bad", (S.Source,), {}))
    x = np.arange(10.0)
    np.testing.assert_array_equal(S.block_mean(x, 3), [1.0, 4.0, 7.0])
    np.testing.assert_array_equal(S.block_mean(x, 1), x)
    assert S.block_mean(np.arange(12.0).reshape(2, 6), 2).shape == (2, 3)
    assert S.reference_format({"a": 1}) is None and S.reference_format(3) is None


def test_tpc5_waveform_picks_the_block_overlapping_the_window_most(tmp_path):
    path = tmp_path / "overlap.tpc5"
    write_tpc5(path, ELSYS_SIGNALS, trigger_times=(0.5, 0.505), volt_range=12.0)   # blocks [0.498, 0.508] and [0.503, 0.513]
    src = S.Tpc5Source.open(path, ELSYS_MAP)
    assert src.block_for_window(0.504, pre=0.01, post=0.0).block == 2
    assert src.block_for_window(0.504, pre=0.0, post=0.01).block == 3
    assert src.block_for_window(0.504, pre=0.05, post=0.05).block == 2      # equal overlap: the earlier block
    # 0.6 s lies in neither block: the block in the window nearest to it (block 3 ends later), none without overlap
    assert src.block_for_window(0.6, pre=0.1, post=0.1).block == 3
    assert src.block_for_window(0.6, pre=0.05, post=0.05) is None
    win = src.waveform(0.504, 0.0, 0.01, fields=["pzt_1"])
    assert win.block == 3 and win.fields == ["pzt_1"] and win.data.shape == (1, win.time.size)
    assert win.time[0] == pytest.approx(0.503, abs=1e-5) and win.time[-1] == pytest.approx(0.513, abs=1e-5)
    assert src.waveform(0.6, 0.05, 0.05) is None
    assert src.waveform(0.6, 0.1, 0.1).block == 3
    assert win.trigger_time == pytest.approx(0.505)
    block = win.as_channel_array(unit="V", filename="overlap.tpc5")
    assert block["channels"] == ["pzt_1"] and block["block"] == 3 and block["sample_rate"] == 200_000.0
    assert block["trigger_time"] == pytest.approx(0.505)
    assert block["filename"] == "overlap.tpc5" and block["time"].dtype == np.float64 and block["data"].dtype == np.float32
    # a file without trigger blocks answers with the window itself (and event_window_s wins when set)
    single = tmp_path / "single.tpc5"
    write_tpc5(single, ELSYS_SIGNALS, trigger_times=(), volt_range=12.0)
    win = S.Tpc5Source.open(single, ELSYS_MAP).waveform(1.0, 0.1, 0.2)
    assert win.block == 1 and win.time[0] == pytest.approx(0.9, abs=1e-3) and win.time[-1] == pytest.approx(1.2, abs=1e-3)
    assert win.trigger_time is None and "trigger_time" not in win.as_channel_array()
    assert "block" not in S.NINpzSource.__dict__ or True
    win = S.Tpc5Source.open(single, ELSYS_MAP, event_window_s=(0.01, 0.02)).waveform(1.0, 0.1, 0.2)
    assert win.time[0] == pytest.approx(0.99, abs=1e-3) and win.time[-1] == pytest.approx(1.02, abs=1e-3)


def span(block, start, end, trigger=None):
    """A trigger block spanning [start, end] s at 1 kHz with its trigger at ``trigger`` (default: start)."""
    fs = 1000.0
    trigger = start if trigger is None else trigger
    return tpc5.BlockInfo(block, fs, int(round((end - start) * fs)) + 1, int(round((trigger - start) * fs)), trigger)


def test_select_trigger_block_rule():
    b2, b3 = span(2, 1.00, 1.10), span(3, 1.05, 1.15)                   # overlap each other
    b4, b5 = span(4, 2.00, 2.10), span(5, 2.40, 2.50)
    b6, b7 = span(6, 3.00, 3.10, 3.05), span(7, 3.30, 3.40, 3.35)
    blocks = [b2, b3, b4, b5, b6, b7]
    pick = S.select_trigger_block
    # (a) blocks containing the event: the one overlapping the window most, the lower number on a tie
    assert pick(blocks, 1.02, 0.1, 0.1) is b2                                # only b2 contains it
    assert pick(blocks, 1.07, 0.1, 0.0) is b2                                # window [0.97, 1.07]: b2 overlaps more
    assert pick(blocks, 1.07, 0.0, 0.1) is b3                                # window [1.07, 1.17]: b3 overlaps more
    assert pick(blocks, 1.07, 0.5, 0.5) is b2                                # both inside the window: tie
    assert pick([b3, b2], 1.07, 0.5, 0.5) is b2                              # whatever the order
    # (a) wins over a nearer-looking block that does not contain the event
    assert pick(blocks, 1.12, 1.0, 1.0) is b3
    # (b) outside every block: the block in the window nearest to the event
    assert pick(blocks, 2.22, 0.3, 0.3) is b4                                # gaps 0.12 (b4) and 0.18 (b5)
    assert pick(blocks, 2.28, 0.3, 0.3) is b5                                # gaps 0.18 and 0.12
    assert pick(blocks, 2.25, 0.3, 0.3) is b5                                # equal gaps: b5's trigger is nearer
    assert pick(blocks, 3.20, 0.3, 0.3) is b6                                # equal gaps and trigger distances
    assert pick(blocks, 1.30, 0.18, 0.18) is b3                              # b2 ends before the window
    assert pick(blocks, 2.25, 0.1, 0.1) is None                              # (c) nothing overlaps the window
    assert pick([], 1.0, 1.0, 1.0) is None


def test_only_blocks_overlapping_the_window_are_candidates():
    b2, b3, b4 = span(2, 1.00, 1.10), span(3, 1.05, 1.15), span(4, 2.00, 2.10)
    blocks = [b2, b3, b4]
    assert S.overlapping_trigger_blocks(blocks, 1.07, 0.1, 0.1) == [b2, b3]
    assert S.overlapping_trigger_blocks([b4, b3, b2], 1.07, 0.1, 0.1) == [b3, b2]     # in their order
    assert S.overlapping_trigger_blocks(blocks, 1.5, 0.1, 0.1) == []
    # a block that only touches the window counts (one sample at 2.25 s, window edges exact in binary)
    edge = tpc5.BlockInfo(8, 1000.0, 1, 0, 2.25)
    assert S.overlapping_trigger_blocks([edge], 2.5, 0.25, 0.25) == [edge]           # touches the window start
    assert S.overlapping_trigger_blocks([edge], 2.0, 0.25, 0.25) == [edge]           # touches the window end
    assert S.overlapping_trigger_blocks([edge], 2.5, 0.25 - 2 ** -20, 0.25) == []
    # a window that leaves out the event time: a block containing the event but not overlapping the window loses
    assert S.select_trigger_block(blocks, 1.02, -0.1, 0.2) is b3                     # window [1.12, 1.22]: b2 ends before it
    assert S.select_trigger_block(blocks, 1.02, -0.5, 0.6) is None                   # window [1.52, 1.62]: no block


def test_tpc5_waveform_keeps_the_whole_block(tmp_path):
    path = tmp_path / "whole.tpc5"
    write_tpc5(path, ELSYS_SIGNALS, trigger_times=(0.5, 1.4), volt_range=12.0)   # blocks [0.498, 0.508], [1.398, 1.408]
    src = S.Tpc5Source.open(path, ELSYS_MAP, time_offset=10.0)
    block = src.trigger_blocks[0]
    for t, pre, post in ((10.505, 0.001, 0.001),      # in the post-trigger part, a window shorter than the block
                         (10.52, 0.05, 0.05),         # after the block, which is the one in the window
                         (10.47, 0.05, 0.05)):        # before it
        win = src.waveform(t, pre, post)
        assert win.block == 2 and win.time.size == block.n_samples == 2000
        assert win.time[0] == pytest.approx(10.498, abs=1e-9)                 # the first pre-trigger sample
        assert win.time[-1] == pytest.approx(10.498 + 1999 / 200_000, abs=1e-9)
        assert win.trigger_time == pytest.approx(10.5)
    assert src.waveform(10.9, 0.05, 0.05) is None
    # read_record: by number or BlockInfo, event fields by default
    narrow = S.Tpc5Source.open(path, ELSYS_MAP, event_fields=["pzt_1"])
    rec = narrow.read_record(3)
    assert rec.block == 3 and rec.fields == ["pzt_1"] and rec.data.shape == (1, 2000)
    np.testing.assert_array_equal(rec.data, narrow.read_record(narrow.trigger_blocks[1]).data)
    with pytest.raises(KeyError):
        narrow.read_record(9)


def test_tpc5_open_refuses_layouts_it_cannot_read(tmp_path):
    import h5py

    def fresh(name):
        path = tmp_path / name
        write_tpc5(path, ELSYS_SIGNALS, trigger_times=(0.5, 1.4), volt_range=12.0)
        return path

    clean = fresh("clean.tpc5")
    with h5py.File(clean, "r") as f:
        assert tpc5.layout_problems(f) == []
    two_groups = fresh("two_groups.tpc5")
    with h5py.File(two_groups, "a") as f:
        f.copy("/measurements/00000001", "/measurements/00000002")
    with pytest.raises(ValueError, match="2 measurement groups"):
        S.Tpc5Source.open(two_groups, ELSYS_MAP)
    uneven = fresh("uneven.tpc5")
    with h5py.File(uneven, "a") as f:
        del f["/measurements/00000001/channels/00000002/blocks/00000003"]
    with pytest.raises(ValueError, match=r"block numbers differ between channels: channel 1, 3: blocks 1-3; channel 2: blocks 1-2"):
        S.Tpc5Source.open(uneven, ELSYS_MAP)
    assert S.Tpc5Source.open(uneven, ELSYS_MAP, channels=[1, 3]).fields == ["pzt_1", "lvdt"]   # channel 2 not selected
    shifted = fresh("shifted.tpc5")
    with h5py.File(shifted, "a") as f:
        f["/measurements/00000001/channels/00000003/blocks/00000002"].attrs["triggerTimeSeconds"] = np.float64(0.5001)
    with pytest.raises(ValueError, match=r"block 2: triggerTimeSeconds differs between channels \(0.5 on 1-2; 0.5001 on 3\)"):
        S.Tpc5Source.open(shifted, ELSYS_MAP)
    with h5py.File(shifted, "r") as f:
        assert tpc5.layout_problems(f, channels=[1, 2]) == []


def test_waveform_helpers_cover_both_layouts():
    from labquake_explorer.data.channels import channel_array
    t = np.linspace(0, 1, 11)
    new = channel_array(np.vstack([t, 2 * t]), ["pzt_1", "pzt_2"], unit="V", time=t, sample_rate=10.0, block=2)
    legacy = {"filename": "x.tpc5", "time_offset": 0.0, "time": t, "raw": np.zeros((3, 11)),
              "original": {"time": t, "raw": np.ones((3, 11))}}
    low = channel_array(np.zeros((2, 11)), ["pzt_1", "pzt_2"])
    event = {"event_time": 0.5, "time": t, "waveform": {"elsys": new}, "strain": legacy, "elsys": {"raw_data": low}}
    assert S.is_waveform_block(new) and S.is_waveform_block(legacy)
    assert not S.is_waveform_block(low) and not S.is_waveform_block(event) and not S.is_waveform_block(None)
    assert list(S.waveform_blocks(event)) == ["waveform/elsys", "strain"]
    assert S.waveform_block(event, "waveform/elsys") is new and S.waveform_block(event, "strain") is legacy
    assert S.waveform_block(event, "elsys/raw_data") is None and S.waveform_block(event, "nope") is None
    assert S.waveform_channels(new) == ["pzt_1", "pzt_2"] and S.waveform_channels(legacy) == ["0", "1", "2"]
    np.testing.assert_array_equal(S.waveform_data(legacy), np.ones((3, 11)))
    np.testing.assert_array_equal(S.waveform_time(new), t)
    assert S.waveform_store(new) is new and S.waveform_store(legacy) is legacy["original"] and S.waveform_store(None) == {}
    assert S.pick_waveform_block(event, prefer_fields=("pzt",)) == "waveform/elsys"
    assert S.pick_waveform_block(event, prefer_keys=("strain",)) == "strain"
    assert S.pick_waveform_block({"time": t}) is None
    assert S.block_channel_labels(new) == ["pzt_1", "pzt_2"]


def test_waveform_trigger_time_stored_or_looked_up():
    from labquake_explorer.data.channels import channel_array
    t = np.linspace(0, 1, 11)
    stored = channel_array(np.zeros((1, 11)), ["pzt_1"], time=t, sample_rate=10.0, block=3, trigger_time=12.5)
    assert S.waveform_trigger_time(stored) == 12.5
    older = channel_array(np.zeros((1, 11)), ["pzt_1"], time=t, sample_rate=10.0, block=3)   # extracted before trigger_time was stored
    run = {"elsys": {"format": "tpc5", "blocks": {"block": [2, 3, 4], "trigger_time_run": [10.0, 12.5, 15.0]}}}
    assert S.waveform_trigger_time(older) is None
    assert S.waveform_trigger_time(older, run, "waveform/elsys") == 12.5
    assert S.waveform_trigger_time(older, run, "waveform/ni") is None            # no such reference
    continuous = channel_array(np.zeros((1, 11)), ["eddy_1"], time=t, sample_rate=10.0)
    assert S.waveform_trigger_time(continuous, run, "waveform/ni") is None
    assert S.waveform_trigger_time({"original": {"time": t, "raw": np.zeros((1, 11))}}, run, "strain") is None
