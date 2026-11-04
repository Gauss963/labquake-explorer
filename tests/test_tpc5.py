import h5py
import numpy as np
import pytest

from labquake_explorer.utils import tpc5
from tests.synthetic import write_tpc5


@pytest.fixture
def fake(tmp_path):
    path = tmp_path / "fake.tpc5"
    signals = {
        "A1 (0.A1)": lambda t: 0.3 * np.sin(2 * np.pi * 50.0 * t),
        "A7 (0.A7)": lambda t: 0.1 * t - 0.05,
    }
    meta = write_tpc5(path, signals, trigger_times=(0.5, 1.4))
    return path, signals, meta


def test_channels_and_scaling(fake):
    path, signals, meta = fake
    with h5py.File(path, "r") as f:
        assert tpc5.channel_numbers(f) == [1, 2]
        infos = tpc5.list_channels(f)
        assert [c["name"] for c in infos] == list(signals)
        assert infos[0]["unit"] == "V" and infos[0]["n_blocks"] == 3
        assert infos[0]["bin_to_volt_factor"] == pytest.approx(meta["factor"])
        # legacy Elsys helpers still work
        assert tpc5.getNChannels(f) == 2
        assert tpc5.getSampleRate(f, 1, 1) == 2000.0


def test_block_table_and_time_axes(fake):
    path, signals, meta = fake
    with h5py.File(path, "r") as f:
        blocks = tpc5.list_blocks(f)
    assert [b.block for b in blocks] == [1, 2, 3]
    cont = tpc5.continuous_block(blocks)
    assert cont.block == 1 and cont.sample_rate == 2000.0
    assert cont.start == pytest.approx(0.0) and cont.end == pytest.approx(2.0 - 1 / 2000)
    assert cont.start_time.startswith("2026-01-01")
    trig = tpc5.trigger_blocks(blocks)
    assert [b.block for b in trig] == [2, 3]
    assert trig[0].trigger_time == 0.5
    assert trig[0].start == pytest.approx(0.5 - 0.002)
    assert trig[0].end == pytest.approx(0.5 + 0.008 - 1 / 200_000)
    assert trig[0].duration == pytest.approx(0.01)
    t = trig[0].time()
    assert t.size == trig[0].n_samples and t[trig[0].trigger_sample] == pytest.approx(0.5)
    assert trig[0].time(10, 12).tolist() == pytest.approx([0.5 - 0.002 + 10 / 2e5, 0.5 - 0.002 + 11 / 2e5])
    assert trig[0].sample_at(0.5) == trig[0].trigger_sample
    assert trig[0].sample_at(-5.0) == 0 and trig[0].sample_at(99.0) == trig[0].n_samples - 1


def test_find_block(fake):
    path, _, _ = fake
    with h5py.File(path, "r") as f:
        blocks = tpc5.list_blocks(f)
    trig = tpc5.trigger_blocks(blocks)
    assert tpc5.find_block(trig, 0.5).block == 2
    assert tpc5.find_block(trig, 0.5 + 0.0079).block == 2
    assert tpc5.find_block(trig, 1.4 - 0.001).block == 3
    assert tpc5.find_block(trig, 1.0) is None
    # the continuous block contains every time; it wins only when asked for
    assert tpc5.find_block(blocks, 1.0).block == 1
    assert tpc5.find_block(blocks, 0.5).block == 2


def test_read_block_matches_signal_within_adc_resolution(fake):
    path, signals, meta = fake
    with h5py.File(path, "r") as f:
        t, data = tpc5.read_block(f, 1)
        assert data.shape == (2, 4000) and t.shape == (4000,)
        np.testing.assert_allclose(data[0], signals["A1 (0.A1)"](t), atol=meta["factor"])
        np.testing.assert_allclose(data[1], signals["A7 (0.A7)"](t), atol=meta["factor"])
        # one channel, a slice of a trigger block
        t2, d2 = tpc5.read_block(f, 2, channels=[2], start=100, stop=110)
        assert d2.shape == (1, 10)
        np.testing.assert_allclose(d2[0], signals["A7 (0.A7)"](t2), atol=meta["factor"])
        assert tpc5.read_channel(f, 1, 2, 0, 5).shape == (5,)
        with pytest.raises(KeyError):
            tpc5.read_block(f, 9)


def test_read_window_by_time(fake):
    path, signals, _ = fake
    with h5py.File(path, "r") as f:
        blocks = tpc5.list_blocks(f)
        block = tpc5.find_block(tpc5.trigger_blocks(blocks), 1.4)
        t, data = tpc5.read_window(f, block, 1.399, 1.401, channels=[1])
    assert t[0] == pytest.approx(1.399, abs=1e-5) and t[-1] == pytest.approx(1.401, abs=1e-5)
    assert data.shape == (1, t.size)


def test_marker_bits_are_masked(tmp_path):
    path = tmp_path / "markers.tpc5"
    meta = write_tpc5(path, {"A1": lambda t: 0.2 + 0 * t}, trigger_times=(0.5,), marker_bits=2)
    with h5py.File(path, "r") as f:
        t, data = tpc5.read_block(f, 1, channels=[1])
    np.testing.assert_allclose(data[0], 0.2, atol=meta["factor"] * 2)


def test_table_round_trip_and_read_continuous(fake):
    path, signals, _ = fake
    rec = tpc5.read_continuous(path)
    assert rec["data"].shape == (2, 4000) and rec["data"].dtype == np.float32
    assert rec["block"].block == 1
    assert [b.block for b in rec["trigger_blocks"]] == [2, 3]
    assert rec["channels"][1]["name"] == "A7 (0.A7)"
    table = tpc5.block_table(rec["trigger_blocks"])
    assert table["block"] == [2, 3] and table["trigger_time"] == [0.5, 1.4]
    back = tpc5.blocks_from_table({k: np.asarray(v) for k, v in table.items()}, rec["start_time"])
    assert back == rec["trigger_blocks"]
    sub = tpc5.read_continuous(path, channels=[2])
    assert sub["data"].shape == (1, 4000)


def test_scan_blocks_every_group_and_channel(fake):
    path, _, _ = fake
    with h5py.File(path, "a") as f:
        f.copy("/measurements/00000001", "/measurements/00000002")
        del f["/measurements/00000002/channels/00000002/blocks/00000003"]
    with h5py.File(path, "r") as f:
        assert tpc5.measurement_names(f) == ["00000001", "00000002"]
        scan = tpc5.scan_blocks(f)
        problems = tpc5.layout_problems(f)
    assert list(scan) == ["00000001", "00000002"] and list(scan["00000001"]) == [1, 2]
    assert [b.block for b in scan["00000001"][2]] == [1, 2, 3] and [b.block for b in scan["00000002"][2]] == [1, 2]
    assert scan["00000002"][1][1].trigger_time == 0.5 and scan["00000002"][1][1].n_samples == 2000
    assert tpc5.channel_block_problems(scan["00000001"]) == []
    assert tpc5.channel_block_problems(scan["00000002"]) == [
        "block numbers differ between channels: channel 1: blocks 1-3; channel 2: blocks 1-2"]
    assert problems == ["2 measurement groups under /measurements (00000001, 00000002); only 00000001 is read"]
    assert tpc5.number_ranges([5, 1, 2, 3, 9, 10]) == "1-3, 5, 9-10" and tpc5.number_ranges([]) == ""
