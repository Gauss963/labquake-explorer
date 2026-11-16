import numpy as np
import pytest

from labquake_explorer.data import channels as C


@pytest.fixture
def run():
    n = 100
    t = np.arange(n) / 10.0
    raw = C.channel_array(np.vstack([np.full(n, 3.5), 0.1 * t, -0.5 - 0.01 * t]), ["pressure_1", "lvdt", "eddy_1"],
                          unit="V", positions=C.positions_table(["pressure_1", "lvdt", "eddy_1"], {"eddy_1": (120.0, 0.0, 25.0)}),
                          recorder=["elsys", "elsys", "elsys"])
    slip = C.channel_array(np.vstack([t, 2 * t]), ["slip_1", "slip_2"], unit="um")
    pzt = C.channel_array(np.vstack([np.sin(t)]), ["pzt_1"], unit="V")
    return {"time": t, "raw_data": raw, "slip": slip, "normal_stress": np.full(n, 8.0),
            "strain": {"original": {"time": t[:10], "raw": np.zeros((2, 10))}},
            "elsys": {"format": "tpc5", "fields": ["a"], "raw_data": pzt}, "events": [], "name": "run1",
            "nested": {"shear": np.ones(n), "short": np.ones(3)}}


def test_channel_array_and_positions():
    arr = C.channel_array(np.zeros((2, 5)), ["a", "b"], unit="V", source=["x", "y"])
    assert C.is_channel_array(arr) and C.channel_names(arr) == ["a", "b"] and arr["source"] == ["x", "y"]
    assert C.channel_array(np.zeros(5), ["a"])["data"].shape == (1, 5)
    with pytest.raises(ValueError):
        C.channel_array(np.zeros((2, 5)), ["a"])
    table = C.positions_table(["a", "b"], {"b": (1.0, 2.0, 3.0)}, unit="mm", frame="sample")
    assert np.isnan(table["x"][0]) and table["x"][1] == 1.0 and table["z"][1] == 3.0 and table["frame"] == "sample"
    assert not C.is_channel_array({"data": np.zeros((2, 5)), "channels": ["a"]})
    assert not C.is_channel_array({"data": np.zeros(5), "channels": ["a"]})
    assert not C.is_channel_array(3) and C.channel_names({}) == []
    # positions_for subsets a table
    raw = C.channel_array(np.zeros((3, 4)), ["a", "b", "c"],
                          positions=C.positions_table(["a", "b", "c"], {"a": (1, 2, 3), "c": (7, 8, 9)}, unit="mm", frame="f"))
    sub = C.positions_for(raw, ["c", "a", "zz"])
    assert sub["x"][:2] == [7.0, 1.0] and sub["z"][:2] == [9.0, 3.0] and np.isnan(sub["x"][2])
    assert sub["unit"] == "mm" and sub["frame"] == "f"
    assert C.positions_for(C.channel_array(np.zeros((1, 4)), ["a"]), ["a"]) is None
    assert C.positions_for(None, ["a"]) is None


def test_rows_lookup_and_aligned_fields(run):
    assert C.row(run["slip"], "slip_2")[1] == pytest.approx(0.2)
    assert C.row(run["slip"], 0)[1] == pytest.approx(0.1)
    assert C.row(run["slip"], "nope") is None and C.row(run["slip"], 5) is None and C.row(3, 0) is None
    assert C.find_channel(run, "eddy_1") == ("raw_data", 2) and C.find_channel(run, "nope") is None
    assert C.find_channel(run, "pzt_1") == ("elsys/raw_data", 0)
    np.testing.assert_array_equal(C.get_channel(run, "pzt_1"), run["elsys"]["raw_data"]["data"][0])
    assert [p for p, _ in C.channel_arrays(run)] == ["raw_data", "slip", "elsys/raw_data"]
    np.testing.assert_array_equal(C.get_channel(run, "normal_stress"), run["normal_stress"])
    np.testing.assert_array_equal(C.get_channel(run, "lvdt"), run["raw_data"]["data"][1])
    assert C.get_channel(run, "missing") is None
    # aligned fields and get_field
    fields = C.aligned_fields(run, 100)
    assert fields == ["time", "raw_data/pressure_1", "raw_data/lvdt", "raw_data/eddy_1", "slip/slip_1", "slip/slip_2",
                      "normal_stress", "elsys/raw_data/pzt_1", "nested/shear"]
    assert "elsys/raw_data/pzt_1" in C.aligned_fields(run, 100, recurse=False) and "nested/shear" not in C.aligned_fields(run, 100, recurse=False)
    np.testing.assert_array_equal(C.get_field(run, "elsys/raw_data/pzt_1"), run["elsys"]["raw_data"]["data"][0])
    np.testing.assert_array_equal(C.get_field(run, "slip/slip_2"), run["slip"]["data"][1])
    np.testing.assert_array_equal(C.get_field(run, "nested/shear"), run["nested"]["shear"])
    np.testing.assert_array_equal(C.get_field(run, "time"), run["time"])
    assert C.get_field(run, "slip/nope") is None and C.get_field(run, "slip") is None
    assert C.get_field(run, "slip/slip_1/x") is None and C.get_field(run, "name") is None
    sliced = C.slice_channel_array(run["slip"], slice(10, 20))
    assert sliced["data"].shape == (2, 10) and sliced["channels"] == ["slip_1", "slip_2"] and sliced["unit"] == "um"
