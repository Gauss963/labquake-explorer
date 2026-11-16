import numpy as np
import pytest

from labquake_explorer.utils.ni_npz import open_ni_npz
from tests.synthetic import write_ni_npz


SIGNALS = {
    "normal": lambda t: 3.5 + 0.0 * t,
    "shear": lambda t: 0.7 + 0.01 * np.sin(2 * np.pi * 3.0 * t),
    "eddy": lambda t: -0.5 - 2.0 * t,
}


@pytest.mark.parametrize("compressed", [False, True], ids=["stored", "compressed"])
def test_reader(tmp_path, compressed):
    path = tmp_path / "T0000-raw-run1-20260101_000000.npz"
    truth = write_ni_npz(path, SIGNALS, fs=10_000.0, duration=2.0, trigger_sample_index=12_345, compressed=compressed)
    r = open_ni_npz(path)
    # metadata
    assert r.sample_rate == 10_000.0 and r.n_samples == 20_000 and r.duration == pytest.approx(2.0)
    assert r.channels == ["ai0", "ai1", "ai2"] and r.labels == ["normal", "shear", "eddy"]
    assert r.dtype == np.float64
    assert r.trigger_sample_index == 12_345 and r.trigger_time == pytest.approx(1.2345)
    d = r.as_dict()
    assert d["format"] == "ni_npz" and d["channels"] == ["ai0", "ai1", "ai2"] and d["trigger_time"] == pytest.approx(1.2345)
    assert (d["member_offsets"]["ai0"] is None) == compressed
    # channel access is lazy (a memmap unless compressed) and exact
    ch = r.channel("ai2")
    assert isinstance(ch, np.memmap) != compressed
    np.testing.assert_array_equal(np.asarray(ch[100:110]), truth["ai2"][100:110])
    np.testing.assert_array_equal(np.asarray(r.channel(0)[:5]), truth["ai0"][:5])
    pytest.raises(KeyError, r.channel, "ai9")
    pytest.raises(IndexError, r.channel, 7)
    # mappings are reused; the compressed cache holds one loaded member at a time
    a = r.channel("ai0")
    assert r.channel(0) is a
    r.channel("ai1")
    if compressed:
        assert "ai0" not in r._cache and "ai1" in r._cache
    else:
        assert set(r._cache) == {"ai0", "ai1", "ai2"}
    # read and window
    t, d = r.read([1, "ai2"], 1000, 1010)
    assert d.shape == (2, 10) and t[0] == pytest.approx(0.1)
    np.testing.assert_array_equal(d[0], truth["ai1"][1000:1010])
    t, d = r.read_window(0.5, 0.5005)
    assert d.shape == (3, 6) and t[0] == pytest.approx(0.5) and t[-1] == pytest.approx(0.5005)
    assert r.sample_at(-1.0) == 0 and r.sample_at(99.0) == r.n_samples - 1
    assert r.contains(1.0) and not r.contains(2.5)
    assert r.read(stop=None)[1].shape == (3, 20_000)
    # decimate is a block mean
    t, d = r.decimate(100, channels=["ai1", "ai2"], chunk_samples=700)   # chunk not a multiple of n
    assert d.shape == (2, 200) and d.dtype == np.float32
    np.testing.assert_allclose(d[0], truth["ai1"][:20_000].reshape(200, 100).mean(axis=1), rtol=1e-6)
    assert t[0] == pytest.approx((99 / 2) / 10_000.0) and t[1] - t[0] == pytest.approx(0.01)
    np.testing.assert_array_equal(r.decimate(1, channels=[0])[1][0], truth["ai0"].astype(np.float32))
    pytest.raises(ValueError, r.decimate, 0)
    pytest.raises(ValueError, r.decimate, r.n_samples + 1)


def test_trigger_sentinel_missing_trigger_and_bad_metadata(tmp_path):
    for name, index in (("s.npz", -1), ("a.npz", None)):         # the -1 sentinel, or no trigger at all
        kw = {} if index is None else {"trigger_sample_index": index}
        write_ni_npz(tmp_path / name, {"x": lambda t: t}, fs=1000.0, duration=0.1, **kw)
        r = open_ni_npz(tmp_path / name)
        assert r.trigger_sample_index is None and r.trigger_time is None
    bad = {"late": dict(ai0=np.zeros(5), sample_rate=np.array(10.0), trigger_sample_index=np.array(5)),
           "labels": dict(ai0=np.zeros(5), ai1=np.zeros(5), sample_rate=np.array(10.0), channels=np.array(["only_one"])),
           "no_rate": dict(ai0=np.zeros(5)),
           "lengths": dict(ai0=np.zeros(5), ai1=np.zeros(6), sample_rate=np.array(1.0))}
    for name, arrays in bad.items():
        np.savez(tmp_path / f"{name}.npz", **arrays)
        pytest.raises(ValueError, open_ni_npz, tmp_path / f"{name}.npz")
    pytest.raises(FileNotFoundError, open_ni_npz, tmp_path / "missing.npz")
    np.savez(tmp_path / "gap.npz", ai0=np.zeros(5), ai2=np.zeros(5), sample_rate=np.array(10.0))
    g = open_ni_npz(tmp_path / "gap.npz")
    assert g.channels == ["ai0", "ai2"] and g.labels == [0, 2]
