import numpy as np
import pytest

from labquake_explorer.data import migrate as M
from labquake_explorer.data.channels import channel_array, positions_table
from labquake_explorer.data.data_manager import DataManager
from labquake_explorer.data.sources import read_arrivals
from tests.synthetic import make_experiment


def old_style_experiment():
    """An experiment the explorer wrote between 2026-10-04 and 2026-10-11: run
    keys since dropped, picks at the top level, raw slices and 'original'
    blocks in events, version-2 analyses, arrival picks among the data keys."""
    data, truth = make_experiment(n_runs=1, window=5.0)
    run = data["runs"][0]
    t = run["time"]
    n = t.size
    positions = positions_table(["pzt_1", "pzt_2"], {"pzt_1": (100.0, 100.0, 50.0), "pzt_2": (200.0, 100.0, 50.0)})
    run["elsys"] = {"format": "tpc5", "filename": "run1.tpc5", "time_offset": 0.0, "fields": ["pzt_1", "pzt_2"],
                    "channel_numbers": [1, 2], "continuous": {"block": 1, "sample_rate": 2000.0, "n_samples": n,
                                                                 "trigger_sample": n - 1, "trigger_time": 0.0},
                    "blocks": {"block": [2, 3], "trigger_time": [13.98, 25.98], "trigger_time_run": [13.98, 25.98]},
                    "raw_data": channel_array(np.zeros((2, n)), ["pzt_1", "pzt_2"], unit="V", positions=positions)}
    run["slip"] = channel_array(np.vstack([run["displacement"]]), ["slip_1"], unit="um")
    run["units"] = {"displacement": "um", "slip": "um"}
    picks = run.pop("event_extraction")["event_indices"]
    run["event_indices"] = list(picks)
    run["event_window"] = {"version": 1, "start_s": -5.0, "end_s": 5.0}
    run["file"], run["file_index"], run["ni_file"], run["sources"] = "run1.tpc5", 1, "run1.npz", {"elsys": "tpc5"}
    run["ni"] = {"format": "ni_npz", "filename": "run1.npz", "time_offset": 0.0, "fields": ["eddy_1"],
                 "labels": ["0"], "member_offsets": {"ai0": 185}, "sample_rate": 500e3, "n_samples": 10,
                 "raw_data": channel_array(np.zeros((1, n)), ["eddy_1"], unit="V")}
    fs = 200e3
    for j, event in enumerate(run["events"]):
        m = event["time"].size
        tt = event["event_time"] - 0.1 + np.arange(int(0.6 * fs)) / fs
        raw = np.random.default_rng(j).normal(0, 1e-3, (2, tt.size))
        event["elsys"] = {"format": "tpc5", "filename": "run1.tpc5", "fields": ["pzt_1", "pzt_2"],
                          "sample_rate": fs, "block": 2 + (j % 2),
                          "raw_data": channel_array(np.zeros((2, m)), ["pzt_1", "pzt_2"], unit="V"),
                          "original": {"time": tt, "raw": raw}}
        event["ni"] = {"format": "ni_npz", "filename": "run1.npz", "raw_data": channel_array(np.zeros((1, m)), ["eddy_1"], unit="V")}
        event["slip"] = channel_array(np.vstack([event["displacement"]]), ["slip_1"], unit="um")
        event["event_analysis"] = {"version": 2, "x_field": "displacement", "y_field": "shear_stress", "fit_method": "ols",
                                   "loading_indices": [10, 20], "unloading_indices": [30, 40],
                                   "rupture_start_index": 25, "rupture_end_index": 45, "post_indices": [50, 60],
                                   "loading_window": [-4.0, -1.0], "loading_stiffness": 0.5, "unloading_stiffness": -0.1,
                                   "stress_drop": 0.4, "displacement": 30.0, "stress_drop_trend": 0.5}
    e0 = run["events"][0]
    e0["elsys"]["original"]["picked_idx"] = [100, 200]
    e0["elsys"]["original"]["rupture_arrival_time"] = e0["elsys"]["original"]["time"][[100, 200]]
    e0["elsys"]["enabled_channels"] = [True, True]
    e0["elsys"]["fitting_channels"] = [True, False]
    e0["rupture_speed"] = 1500.0
    e0["pzt_spectrum"] = {"version": 1, "channels": {"ch0": {"version": 1, "channel": 0, "record": "elsys"}}}
    e1 = run["events"][1]
    e1["high_rate"] = {"ni": channel_array(np.zeros((1, 50)), ["eddy_1"], unit="V", time=np.arange(50) / 500e3,
                                            sample_rate=500e3, filename="run1.npz")}
    e1["low_rate"] = {"ni": channel_array(np.zeros((1, e1["time"].size)), ["eddy_1"], unit="V")}
    return data, truth


def test_upgrade_brings_runs_and_events_up_to_date():
    data, truth = old_style_experiment()
    picks = list(data["runs"][0]["event_indices"])
    notes = M.upgrade_experiment(data)
    run = data["runs"][0]
    assert notes and all(isinstance(n, str) for n in notes)
    # run level
    for key in ("file", "file_index", "ni_file", "sources", "event_indices", "event_window", "displacement"):
        assert key not in run, key
    assert "displacement" not in run["units"]
    assert run["event_extraction"] == {"version": 1, "start_s": -5.0, "end_s": 5.0, "event_indices": picks}
    assert "labels" not in run["ni"] and "member_offsets" not in run["ni"]
    # events: processed channels, waveform records, arrivals, analyses, spectrum record names
    e0, e1 = run["events"][0], run["events"][1]
    for event in run["events"]:
        assert "elsys" not in event and "ni" not in event and "displacement" not in event
        assert "high_rate" not in event and "low_rate" not in event
        w = event["waveform"]["elsys"]
        assert w["channels"] == ["pzt_1", "pzt_2"] and w["data"].shape == (2, w["time"].size) and w["data"].dtype == np.float32
        assert w["sample_rate"] == 200e3 and w["filename"] == "run1.tpc5" and w["block"] in (2, 3)
        assert w["trigger_time"] == {2: 13.98, 3: 25.98}[w["block"]]
        assert w["positions"]["x"] == [100.0, 200.0]
        a = event["event_analysis"]
        assert a == {"version": 3, "x_field": "displacement", "y_field": "shear_stress", "loading_indices": [10, 20],
                     "unloading_indices": [30, 40], "delta_indices": [25, 45], "loading_slope": 0.5,
                     "unloading_slope": -0.1, "delta_x": 30.0, "delta_y": pytest.approx(-0.4)}
    arrivals = read_arrivals(e0["waveform"]["elsys"])
    assert arrivals["picked_idx"] == [100, 200] and arrivals["fitting_channels"] == [True, False]
    assert arrivals["rupture_speed"] == 1500.0 and "rupture_speed" not in e0
    assert "picked_idx" not in e0["waveform"]["elsys"] and e0["waveform"]["elsys"]["arrivals"]["version"] == 1
    assert "pzt_spectrum" not in e0
    moved = e0["waveform"]["elsys"]["spectrum"]
    assert moved["version"] == 2 and "channels" not in moved
    assert moved["sets"]["default"]["channels"]["ch0"]["record"] == "waveform/elsys"
    assert e1["waveform"]["ni"]["data"].shape == (1, 50) and "trigger_time" not in e1["waveform"]["ni"]
    # a second pass changes nothing
    assert M.upgrade_experiment(data) == []


def test_current_layout_is_left_alone():
    data, truth = make_experiment(n_runs=1)
    assert M.upgrade_experiment(data) == []


def test_psu_strain_layout_is_kept():
    from tests.synthetic import add_strain
    data, truth = make_experiment(n_runs=1)
    run = data["runs"][0]
    run["strain"] = {"filename": "legacy.tpc5", "time_offset": 0.0, "time": np.arange(10.0), "raw": np.zeros((2, 10))}
    for event in run["events"]:
        add_strain(event, n_channels=4)
    before = {k: type(v) for k, v in run["events"][0]["strain"].items()}
    notes = M.upgrade_experiment(data)
    assert notes == [] or all("strain" not in n for n in notes)
    assert {k: type(v) for k, v in run["events"][0]["strain"].items()} == before
    assert "strain" in run


def test_upgrade_file_keeps_a_copy(tmp_path):
    data, truth = old_style_experiment()
    path = tmp_path / "old.h5"
    dm = DataManager()
    dm.data = data
    dm.save_file(path)
    notes = M.upgrade_file(path)
    assert notes
    assert (tmp_path / "old.before-upgrade.h5").exists()
    again = DataManager()
    again.load_file(path)
    run = again.get_data("runs/[0]")
    assert "file" not in run and "waveform" in run["events"][0] and "elsys" not in run["events"][0]
    assert run["events"][0]["waveform"]["elsys"]["trigger_time"] == pytest.approx(13.98)
    assert M.upgrade_file(path) == []                         # already up to date: no write, no second copy
    assert M.main([str(path)]) == 0


def test_recalibrate_lp_displacement_from_volts(tmp_path):
    data, truth = make_experiment(n_runs=1)
    run = data["runs"][0]
    volts = (run["LP_displacement"] / -5040.0).astype(np.float32)          # what an uncalibrated file holds
    run["LP_displacement"] = volts.copy()
    run["units"] = {"LP_displacement": "V", "slip": "um"}
    run["calibration"] = {"note": "pressure: x; LVDT: slope unknown, LP_displacement in volts",
                          "steps": [{"kind": "linear", "field": "LP_displacement", "source": "lvdt", "factor": 1.0,
                                     "offset": 0.0, "unit": "V", "formula": "LP_displacement (V) = 1 * lvdt + 0"}]}
    for event in run["events"]:
        event["LP_displacement"] = (event["LP_displacement"] / -5040.0).astype(np.float32)
    run["events"][0]["event_analysis"] = {"version": 3, "x_field": "LP_displacement", "y_field": "shear_stress"}
    notes = M.recalibrate_lp_displacement(data, -5040.0)
    assert len(notes) == 2 and "5040" in notes[0] and "event_analysis" in notes[1]
    np.testing.assert_allclose(run["LP_displacement"], volts * -5040.0, rtol=1e-6)
    assert run["units"]["LP_displacement"] == "um"
    step = run["calibration"]["steps"][0]
    assert step["factor"] == -5040.0 and step["unit"] == "um" and step["formula"].startswith("LP_displacement (um) = -5040")
    assert run["calibration"]["note"] == "pressure: x; LVDT: -5040 um/V"
    e = run["events"][1]
    assert e["LP_displacement"].dtype == np.float32 and abs(e["LP_displacement"]).max() > 1.0
    assert M.recalibrate_lp_displacement(data, -5040.0) == []                   # already in micrometres
    # through the file tool, with the flag, keeping the first backup only
    path = tmp_path / "exp.h5"
    dm = DataManager(); dm.data = data; dm.save_file(path)
    data2, _ = make_experiment(n_runs=1)
    data2["runs"][0]["units"] = {"LP_displacement": "V"}
    dm2 = DataManager(); dm2.data = data2; dm2.save_file(path)
    copy = tmp_path / "exp.before-upgrade.h5"
    copy.write_bytes(b"first original")
    assert M.main([str(path), "--lvdt-um-per-v=-5040"]) == 0
    assert copy.read_bytes() == b"first original"                            # never overwritten
    again = DataManager(); again.load_file(path)
    assert again.get_data("runs/[0]/units")["LP_displacement"] == "um"


def test_fault_geometry_moves_into_analysis():
    data, _ = make_experiment()
    data["patch_x_mm"] = [187.5, 312.5]
    data["fault"] = {"ends_x_mm": np.array([0.0, 500.0]), "patch_x_mm": [1.0, 2.0]}   # the top-level copy wins
    notes = M.upgrade_experiment(data)
    assert data["analysis"]["fault"] == {"patch_x_mm": [187.5, 312.5], "ends_x_mm": [0.0, 500.0]}
    assert "patch_x_mm" not in data and "fault" not in data
    assert any("patch_x_mm -> analysis/fault/patch_x_mm" in n for n in notes)
    data, _ = make_experiment()
    data["fault"] = "one-dimensional PMMA fault, 500 mm long"            # a description stays
    data["fault_ends_x_mm"] = [0.0, 500.0]
    M.upgrade_experiment(data)
    assert data["fault"] == "one-dimensional PMMA fault, 500 mm long"
    assert data["analysis"]["fault"] == {"ends_x_mm": [0.0, 500.0]}
    assert M.upgrade_experiment(data) == []                               # nothing left to do


@pytest.mark.parametrize("suffix", [".h5", ".npz"])
def test_write_fault_geometry_in_place(tmp_path, suffix):
    data, _ = make_experiment()
    data["fault"] = "one-dimensional PMMA fault, 500 mm long"
    data["patch_x_mm"] = [187.5, 312.5]
    dm = DataManager()
    dm.data = data
    path = tmp_path / f"exp{suffix}"
    dm.save_file(path)
    before = {"runs": len(data["runs"]), "events": len(data["runs"][0]["events"])}
    notes = M.write_fault_geometry(path, ends_x_mm=(0, 500))
    assert any("patch_x_mm -> analysis/fault/patch_x_mm" in n for n in notes)
    assert any("ends_x_mm = [0.0, 500.0]" in n for n in notes)
    dm2 = DataManager()
    dm2.load_file(path)
    assert np.asarray(dm2.data["analysis"]["fault"]["patch_x_mm"]).tolist() == [187.5, 312.5]
    assert np.asarray(dm2.data["analysis"]["fault"]["ends_x_mm"]).tolist() == [0.0, 500.0]
    assert "patch_x_mm" not in dm2.data and dm2.data["fault"] == "one-dimensional PMMA fault, 500 mm long"
    assert len(dm2.data["runs"]) == before["runs"] and len(dm2.data["runs"][0]["events"]) == before["events"]
    assert M.write_fault_geometry(path, patch_x_mm=[187.5, 312.5], ends_x_mm=[0, 500]) == []   # already so
    notes = M.write_fault_geometry(path, patch_x_mm=[50, 450])
    assert notes == ["analysis/fault/patch_x_mm = [50.0, 450.0]"]
    dm3 = DataManager()
    dm3.load_file(path)
    assert np.asarray(dm3.data["analysis"]["fault"]["patch_x_mm"]).tolist() == [50.0, 450.0]
    assert M.main([str(path), "--ends-x-mm=0,400"]) == 0
    dm4 = DataManager()
    dm4.load_file(path)
    assert np.asarray(dm4.data["analysis"]["fault"]["ends_x_mm"]).tolist() == [0.0, 400.0]
    with pytest.raises(ValueError, match="two numbers"):
        M.write_fault_geometry(path, patch_x_mm=[1.0])


def test_flat_pzt_settings_and_version_1_stores_upgrade_to_result_sets():
    data = {"name": "x", "runs": [], "analysis": {"pzt_spectrum": {
        "settings": {"pre_ms": 1.0}, "sensors": {"pzt_1": {"t_star_s": 1e-6}, "pzt_2": {"t_star_s": 2e-6}},
        "sets": {"default": {"sensors": {"pzt_1": {"t_star_s": 3e-6}}}}}}}
    notes = M.upgrade_experiment(data)
    assert len(notes) == 2
    pz = data["analysis"]["pzt_spectrum"]
    assert "settings" not in pz and "sensors" not in pz
    assert pz["sets"]["default"]["settings"] == {"pre_ms": 1.0}
    assert pz["sets"]["default"]["sensors"] == {"pzt_1": {"t_star_s": 3e-6}, "pzt_2": {"t_star_s": 2e-6}}   # the set's own wins
    assert M.upgrade_experiment(data) == []
    event = {"waveform": {"elsys": {"spectrum": {"version": 1, "channels": {"ch0": {"channel": 0}}}}}}
    notes = M.upgrade_event(event, {}, "e")
    assert any("version 2" in n for n in notes)
    assert event["waveform"]["elsys"]["spectrum"] == {"version": 2, "sets": {"default": {"channels": {"ch0": {"channel": 0}}}}}
    assert M.upgrade_event(event, {}, "e") == []
