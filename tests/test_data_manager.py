import h5py
import numpy as np
import pytest

from labquake_explorer.data.data_manager import DataManager
from tests.synthetic import load_manager as loaded, make_experiment


@pytest.fixture
def dm():
    data, _ = make_experiment(recurrence=12.0)
    manager = DataManager()
    manager.data = data
    return manager


def test_paths_in_memory(dm):
    assert dm.get_data("") is dm.data and dm.get_data("name") == "p0001"
    assert dm.get_data("runs/[0]/events/[1]") is dm.data["runs"][0]["events"][1]
    assert dm.get_data("runs\\[0]\\name") == "run00"
    pytest.raises(KeyError, dm.get_data, "runs/[0]/nope")
    pytest.raises(IndexError, dm.get_data, "runs/[9]")
    # set_data with add_key creates the parents
    dm.set_data("runs/[0]/events/[1]/event_analysis", {"a": 1}, True)
    assert dm.get_data("runs/[0]/events/[1]/event_analysis") == {"a": 1}
    dm.set_data("runs/[0]/derived/deep/value", 3.0, add_key=True)
    assert dm.get_data("runs/[0]/derived/deep/value") == 3.0
    pytest.raises(KeyError, dm.set_data, "runs/[0]/missing/value", 1.0)
    pytest.raises(ValueError, dm.set_data, "", 1.0)
    # delete_data, also with backslashes and at the root
    dm.set_data("runs/[0]/tmp", 1, add_key=True)
    dm.delete_data("runs/[0]/tmp")
    pytest.raises(KeyError, dm.get_data, "runs/[0]/tmp")
    dm.set_data("runs/[0]/tmp", 1, add_key=True)
    dm.delete_data("runs\\[0]\\tmp")
    assert "tmp" not in dm.get_data("runs/[0]")
    pytest.raises(ValueError, dm.delete_data, "runs/[0]/name/x")
    dm.delete_data("")
    assert dm.data is None
    empty = DataManager()
    pytest.raises(ValueError, empty.get_data, "name")
    pytest.raises(ValueError, empty.set_data, "name", "x")


def _assert_equivalent(a, b, path=""):
    """Recursively compare loaded data with the original (arrays vs lists allowed)."""
    if isinstance(a, dict):
        assert isinstance(b, dict), path
        assert set(a) == set(b), (path, set(a) ^ set(b))
        for k in a:
            _assert_equivalent(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple, np.ndarray)) and not isinstance(a, str):
        if len(a) and isinstance(a[0], dict):
            assert isinstance(b, list) and len(a) == len(b), path
            for x, y in zip(a, b):
                _assert_equivalent(x, y, path)
        else:
            x, y = np.asarray(a), np.asarray(b)
            if x.dtype.kind in "US":
                assert [str(v) for v in np.ravel(x)] == [str(v) for v in np.ravel(y)], path
            else:
                np.testing.assert_allclose(x.astype(float), y.astype(float), equal_nan=True, err_msg=path)
    elif isinstance(a, str):
        assert a == b, path
    elif isinstance(a, (bool, np.bool_)):
        assert int(a) == int(b), path
    else:
        np.testing.assert_allclose(float(a), float(b), equal_nan=True, err_msg=path)


def test_hdf5_round_trip(tmp_path, dm):
    nan = float("nan")
    dm.set_data("runs/[0]/events/[0]/event_analysis", {
        "version": 2, "x_field": "displacement", "fit_method": "ols",
        "loading_indices": [10, 20], "post_indices": [30, 40],
        "loading_window": [-4.0, -1.0], "stress_drop": 0.4, "displacement_trend": nan,
        "loading_fit": {"slope": 1.5, "valid": True, "reason": ""},
    }, True)
    dm.set_data("runs/[0]/events/[1]/pzt_spectrum", {
        "version": 1,
        "channels": {"ch0": {"channel": 0, "pre_ms": 0.5, "n_fixed": True,
                             "spectrum": {"f_binned": [1e3, 2e3], "snr": [5.0, nan], "counts": [3, 4]},
                             "fit": {"band_used": [1e3, 2e3], "at_bounds": {"ln_fc": False}, "converged": True},
                             "warnings": ["near node", "kR < 3"], "source": None},
                     "0": {"channel": 0, "warnings": ["single warning"], "phases": ["P"],
                           "flags": [], "picks": [np.arange(3.0), np.arange(5.0)]}},
    }, True)
    dm.set_data("runs/[0]/interevent", {"version": 1, "recurrence": [nan, 12.0, 12.0], "lp_field": "LP_displacement"}, True)
    dm.set_data("runs/[0]/events/[2]/czm_parms", [1.0, 2.0, 3.0, 4.0, -0.1, 0.0, -0.2, 0.2], True)
    dm.set_data("runs/[0]/legacy", {"0": {"a": 1}, "2": {"a": 3}}, True)                 # non-contiguous digit keys
    dm.set_data("runs/[0]/units", {"elsys/raw_data": "V", "slip": "um"}, True)          # keys containing slashes
    dm.set_data("runs/[0]/events/[0]/picks", [np.arange(2.0), np.arange(4.0)], True)    # ragged: must not abort
    dm.set_data("runs/[0]/events/[0]/mixed", [1, "a", 2.5], True)
    dm.set_data("runs/[0]/events/[0]/stacked", [np.arange(3.0), np.arange(3.0) + 1], True)
    for key, value in (("note", "single"), ("one", ["single"]), ("two", ["a", "b"]), ("none", [])):
        dm.set_data(f"runs/[0]/{key}", value, True)
    original = dm.data
    dm.save_file(tmp_path / "exp.h5")
    other = loaded(tmp_path / "exp.h5")

    assert isinstance(other.get_data("runs"), list) and isinstance(other.get_data("runs/[0]/events"), list)
    loaded_analysis = other.get_data("runs/[0]/events/[0]/event_analysis")
    assert loaded_analysis["version"] == 2 and list(loaded_analysis["loading_indices"]) == [10, 20]
    assert loaded_analysis["loading_fit"]["valid"] == 1
    rec = other.get_data("runs/[0]/events/[1]/pzt_spectrum/channels/ch0")
    assert list(rec["fit"]["band_used"]) == [1e3, 2e3] and list(rec["warnings"]) == ["near node", "kR < 3"]
    assert "source" not in rec  # None cannot be stored in HDF5
    np.testing.assert_allclose(rec["spectrum"]["snr"], [5.0, nan], equal_nan=True)
    # digit keys written from a dict stay a dict (legacy channel maps), one-element
    # string lists stay lists, empty lists stay empty, ragged arrays keep their shapes
    channels = other.get_data("runs/[0]/events/[1]/pzt_spectrum/channels")
    assert isinstance(channels, dict) and set(channels) == {"ch0", "0"}
    legacy = channels["0"]
    assert legacy["warnings"] == ["single warning"] and legacy["phases"] == ["P"] and list(legacy["flags"]) == []
    assert isinstance(legacy["picks"], list) and [len(a) for a in legacy["picks"]] == [3, 5]
    assert list(other.get_data("runs/[0]/events/[2]/czm_parms")) == [1.0, 2.0, 3.0, 4.0, -0.1, 0.0, -0.2, 0.2]
    assert other.get_data("runs/[0]/legacy") == {"0": {"a": 1}, "2": {"a": 3}}
    assert other.get_data("runs/[0]/units") == {"elsys/raw_data": "V", "slip": "um"}
    event = other.get_data("runs/[0]/events/[0]")
    assert isinstance(event["picks"], list) and [list(a) for a in event["picks"]] == [[0.0, 1.0], [0.0, 1.0, 2.0, 3.0]]
    assert event["mixed"] == ["1", "a", "2.5"]
    np.testing.assert_array_equal(event["stacked"], [[0, 1, 2], [1, 2, 3]])
    run = other.get_data("runs/[0]")
    assert run["note"] == "single" and run["one"] == ["single"] and run["two"] == ["a", "b"] and list(run["none"]) == []

    # everything else matches the original
    del original["runs"][0]["events"][1]["pzt_spectrum"]["channels"]["ch0"]["source"]
    for d in (original, other.data):
        del d["runs"][0]["events"][1]["pzt_spectrum"]["channels"]["0"]["picks"]
        del d["runs"][0]["events"][0]["picks"]
    _assert_equivalent(original, other.data)
    # lists survive deletion by index after a load
    other.delete_data("runs/[0]/events/[0]")
    assert len(other.get_data("runs/[0]/events")) == len(original["runs"][0]["events"]) - 1


def test_save_file_suffixes_npz_and_failed_save(tmp_path, dm, monkeypatch):
    for name in ("exp01", "exp01.txt", "exp01.npz.bak"):
        with pytest.raises(ValueError, match="Unsupported file type"):
            dm.save_file(tmp_path / name)
        assert list(tmp_path.iterdir()) == []                   # no file, no stray temporary file either
    path = tmp_path / "exp.H5"                                  # a string path, an upper-case suffix
    dm.save_file(str(path))
    assert loaded(path).get_data("name") == "p0001"
    # a failed HDF5 save keeps the existing file as it was
    before = path.read_bytes()
    monkeypatch.setattr(DataManager, "_save_h5_item", classmethod(lambda cls, group, key, value: 1 / 0))
    pytest.raises(ZeroDivisionError, dm.save_file, path)
    assert path.read_bytes() == before and [p.name for p in tmp_path.iterdir()] == ["exp.H5"]
    assert loaded(path).get_data("name") == "p0001"
    monkeypatch.undo()
    npz = tmp_path / "exp.NPZ"
    dm.set_data("runs/[0]/events/[0]/event_analysis", {"version": 2, "stress_drop": 0.4}, True)
    dm.save_file(npz)
    assert not (tmp_path / "exp.NPZ.npz").exists()
    other = loaded(npz)
    assert other.get_data("runs/[0]/events/[0]/event_analysis")["stress_drop"] == 0.4
    np.testing.assert_array_equal(other.get_data("runs/[0]/shear_stress"), dm.get_data("runs/[0]/shear_stress"))


def test_hdf5_legacy_file_without_container_markers(tmp_path):
    # files written before the container attribute: 0..n-1 groups become lists
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("name", data=b"p0001")
        run = f.create_group("runs").create_group("0")
        run.create_dataset("shear_stress", data=np.arange(3.0))
        events = run.create_group("events")
        events.create_group("0").create_dataset("event_time", data=1.0)
        events.create_group("1").create_dataset("event_time", data=2.0)
        run.create_group("channels").create_group("0").create_dataset("warnings", data=np.array([b"one"]))
    other = loaded(path)
    assert isinstance(other.get_data("runs"), list)
    assert [e["event_time"] for e in other.get_data("runs/[0]/events")] == [1.0, 2.0]
    assert other.get_data("runs/[0]/channels/[0]/warnings") == ["one"]   # the string dataset stays a list


def test_save_safety(tmp_path, dm):
    """Nothing the user saved is dropped without an error: unreadable members
    block a save over their file, unstorable values fail the save, a failed load
    keeps the experiment and its folder, and Save As keeps raw links working."""
    import os
    from pathlib import Path
    path = tmp_path / "exp.h5"
    dm.data["runs"][0]["strain"] = {"format": "tpc5", "filename": "raw/run1.tpc5", "time_offset": 0.0,
                                    "fields": ["a"], "channel_numbers": [1], "blocks": []}
    dm.save_file(path)
    os.chmod(path, 0o644)
    with h5py.File(path, "a") as f:
        f["runs/0/broken"] = h5py.SoftLink("/nowhere")
    m = loaded(path)
    assert not m.modified() and m.load_errors and "broken" in m.load_errors[0]
    with pytest.raises(ValueError, match="could not be read"):
        m.save_file(path)
    # a value of a type the file cannot hold fails the save and keeps the old file
    other = tmp_path / "copy.h5"
    m.data["runs"][0]["bad"] = {1, 2}
    assert m.modified()
    with pytest.raises(TypeError, match="bad"):
        m.save_file(other)
    assert not other.exists()
    del m.data["runs"][0]["bad"]
    # Save As into another folder: the raw link still points at the same file; the mode stays
    sub = tmp_path / "reanalysis"
    sub.mkdir()
    m.save_file(sub / "exp.h5")
    assert m.data_path == sub / "exp.h5" and m.load_errors == [] and not m.modified()
    assert m.event_processor.base_dir == sub
    assert loaded(sub / "exp.h5").get_data("runs/[0]/strain/filename") == os.path.join("..", "raw", "run1.tpc5")
    m.save_file(path)                                 # the copy has everything memory has: saving over is fine now
    assert loaded(path).get_data("runs/[0]/strain/filename") == os.path.join("raw", "run1.tpc5")
    assert os.stat(path).st_mode & 0o777 == 0o644
    # a failed load changes neither the data nor the folder raw files are read from
    bad = sub / "broken.h5"
    bad.write_bytes(b"not hdf5")
    pytest.raises(OSError, m.load_file, bad)
    assert m.data_path == path and m.event_processor.base_dir == Path(tmp_path)
    assert m.get_data("runs/[0]/strain/filename") == os.path.join("raw", "run1.tpc5")
