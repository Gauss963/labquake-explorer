import h5py
import numpy as np
import pytest

from labquake_explorer.data import sources as S
from labquake_explorer.data.channels import channel_array
from labquake_explorer.data.data_manager import DataManager
from labquake_explorer.data.event_processor import EventProcessor
from labquake_explorer.preprocessing import Calibration, EddySlip, pressure_transducers, run_from_tpc5, run_from_tpc5_ni
from tests.synthetic import write_ni_npz, write_tpc5


def pulse(t0, width=1e-4, amp=0.1):
    return lambda t: amp * np.exp(-((t - t0) / width) ** 2)


ELSYS_MAP = {"A1": "pzt_1", "A5": "pressure_1", "A6": "pressure_2", "B1": "eddy_1"}
NI_MAP = {"ai0": "pressure_1", "ai1": "pressure_2", "ai2": "eddy_1"}


@pytest.fixture
def elsys(tmp_path):
    path = tmp_path / "t0001_01_8MPa_run1.tpc5"
    write_tpc5(path, {
        "A1 (0.A1)": lambda t: pulse(0.5)(t) + pulse(1.4)(t),
        "A5 (0.A5)": lambda t: 1.867 + 0 * t,
        "A6 (0.A6)": lambda t: 0.545 + 0 * t,
        "B1 (0.B1)": lambda t: -0.5 - 0.01 * t,
    }, trigger_times=(0.5, 1.4), volt_range=12.0)
    return path


def index_at(run, t):
    return int(np.argmin(np.abs(run["time"] - t)))


def test_extract_from_tpc5_run(elsys, tmp_path):
    run = run_from_tpc5(elsys, ELSYS_MAP, tmp_path, Calibration(*pressure_transducers(), EddySlip({"eddy_1": -0.1})))
    ep = EventProcessor(tmp_path / "exp.h5")
    events = ep.extract_events(run, [index_at(run, 0.5), index_at(run, 1.0), index_at(run, 1.4)], window=0.05)
    assert len(events) == 3
    e0 = events[0]
    assert e0["event_time"] == pytest.approx(0.5) and e0["time"][0] == pytest.approx(0.45) and e0["time"].size == 200
    for f in ("normal_stress", "shear_stress"):
        assert e0[f].shape == e0["time"].shape, f
    assert "displacement" not in e0 and "displacement" not in run        # fault slip lives in slip/slip_1
    for key in ("events", "calibration", "units", "sources", "elsys"):    # processed channels only, no raw volts
        assert key not in e0, key
    assert e0["slip"]["data"].shape == (1, 200) and e0["slip"]["unit"] == "um"
    high = e0["waveform"]["elsys"]
    assert high["filename"] == elsys.name and high["block"] == 2 and high["sample_rate"] == 200_000.0
    assert high["trigger_time"] == pytest.approx(0.5)                    # where the recorder fired, run clock
    assert high["channels"] == ["pzt_1", "pressure_1", "pressure_2", "eddy_1"] and high["unit"] == "V"
    assert high["data"].shape == (4, high["time"].size) and high["data"].dtype == np.float32
    # the whole trigger block (2 ms before, 8 ms after the trigger), not the +-50 ms window
    assert high["time"][0] == pytest.approx(0.498, abs=1e-5) and high["time"][-1] == pytest.approx(0.508, abs=1e-5)
    assert high["time"][np.argmax(high["data"][0])] == pytest.approx(0.5, abs=2e-5)
    # the event at 1.0 s lies in no trigger block: processed channels only, plus a note
    assert "waveform" not in events[1] and "elsys" not in events[1] and events[1]["slip"]["data"].shape == (1, 200)
    assert events[1]["notes"] == ["elsys: no raw record covers 1.0000 s"]
    assert events[2]["waveform"]["elsys"]["block"] == 3


def test_source_event_settings_and_no_base_dir(elsys, tmp_path, capsys):
    run = run_from_tpc5(elsys, ELSYS_MAP, tmp_path, event_fields=["pzt_1"], event_window_s=(0.001, 0.002))
    ep = EventProcessor(tmp_path / "exp.h5")
    e = ep.extract_events(run, [index_at(run, 0.5)], window=1.0)[0]
    high = e["waveform"]["elsys"]
    # event_fields restrict the channels; a recorder with trigger blocks copies its whole block
    assert high["channels"] == ["pzt_1"] and high["data"].shape == (1, high["time"].size)
    assert high["time"].size == pytest.approx(2000, abs=1)
    assert e["time"].size > 0
    # without a saved experiment path the references are skipped with a warning, not an error
    e2 = EventProcessor().extract_events(run, [index_at(run, 0.5)], window=0.05)[0]
    assert "waveform" not in e2 and "elsys" not in e2 and e2["time"].size > 0
    assert "raw-data references skipped" in capsys.readouterr().out


def test_extract_from_tpc5_ni_run(elsys, tmp_path):
    ni = tmp_path / "T0001-raw-run1-20260101_000000.npz"
    write_ni_npz(ni, {"n": lambda t: 1.867 + 0 * t, "s": lambda t: 0.545 + 0 * t,
                      "e": lambda t: -0.5 - 0.01 * t - 0.3 * (t > 1.25)}, fs=10_000.0, duration=3.0,
                 trigger_sample_index=12_000)
    run = run_from_tpc5_ni(elsys, ni, ELSYS_MAP, NI_MAP, tmp_path, ni_decimation=5,
                           elsys_event_fields=["pzt_1"], ni_event_window_s=(0.01, 0.1),
                           positions={"eddy_1": (120.0, 0.0, 25.0)})
    # save and reload the way the explorer does, then extract on the reloaded run
    dm = DataManager()
    dm.data = {"name": "t0001", "runs": [run]}
    out = tmp_path / "t0001.h5"
    dm.save_file(out)
    back = DataManager()
    back.load_file(out)
    rrun = back.get_data("runs/[0]")
    e = back.event_processor.extract_events(rrun, [index_at(rrun, 1.2)], window=0.05)[0]
    assert e["event_time"] == pytest.approx(1.2, abs=1e-3)
    assert "elsys" not in e and "ni" not in e and set(e["waveform"]) == {"elsys", "ni"}
    # NI, a continuous recorder: its own (pre, post) window at the full 10 kHz rate, all NI fields, no block
    nib = e["waveform"]["ni"]
    assert nib["sample_rate"] == 10_000.0 and "block" not in nib and "trigger_time" not in nib and nib["filename"] == ni.name
    assert nib["channels"] == ["pressure_1", "pressure_2", "eddy_1"]
    assert nib["time"][0] == pytest.approx(1.19, abs=1e-3) and nib["time"][-1] == pytest.approx(1.30, abs=1e-3)
    eddy = nib["data"][2]
    assert eddy[-1] - eddy[0] == pytest.approx(-0.3, abs=0.01)
    assert nib["positions"]["x"][2] == pytest.approx(120.0) and np.isnan(nib["positions"]["x"][0])
    # Elsys, a recorder with trigger blocks: block 2 shifted onto the run clock, PZT only
    st = e["waveform"]["elsys"]
    assert st["block"] == 2 and st["channels"] == ["pzt_1"] and st["filename"] == elsys.name
    assert st["trigger_time"] == pytest.approx(1.2, abs=1e-3)           # the Elsys trigger on the NI clock
    assert st["time"][np.argmax(st["data"][0])] == pytest.approx(1.2, abs=2e-5)
    # the new events survive a save/reload
    rrun["events"] = [e]
    dm2 = DataManager(); dm2.data = back.data
    dm2.save_file(tmp_path / "t0001_events.h5")
    again = DataManager(); again.load_file(tmp_path / "t0001_events.h5")
    e_again = again.get_data("runs/[0]/events/[0]")
    assert e_again["waveform"]["elsys"]["block"] == 2 and e_again["waveform"]["ni"]["data"].shape == nib["data"].shape


def test_legacy_strain_layout(tmp_path):
    path = tmp_path / "legacy.tpc5"
    write_tpc5(path, {"ch1": lambda t: 0.1 + 0.2 * (t > 1.0), "ch2": lambda t: 0.0 * t},
               fs_continuous=1000.0, duration=2.0, trigger_times=())
    t_run = 10.0 + np.arange(0, 2.0, 0.01)
    run = {"name": "run1", "time": t_run, "shear_stress": np.ones(t_run.size, dtype=np.float32),
           "strain": {"filename": "legacy.tpc5", "time_offset": 0.0, "time": np.arange(0, 2.0, 0.1),
                      "raw": np.tile(np.arange(20.0), (2, 1)), "filename_downsampled": ""}}
    ep = EventProcessor(tmp_path / "exp.npz")
    e = ep.extract_events(run, [index_at(run, 11.0)], window=0.1)[0]
    st = e["strain"]
    assert set(st) == {"filename_downsampled", "filename", "time", "raw", "original"}
    assert st["original"]["raw"].shape[0] == 2 and st["original"]["time"][0] == pytest.approx(10.9)
    # baseline: the first 1 % of the window is removed, so the step reads +0.2 at the end
    assert abs(st["original"]["raw"][0, 0]) < 1e-3 and st["original"]["raw"][0, -1] == pytest.approx(0.2, abs=2e-3)
    assert st["time"][0] == pytest.approx(10.9) and st["raw"].shape == (2, st["time"].size)
    assert e["shear_stress"].shape == e["time"].shape


def sample_aligned_run(tmp_path, aligned=True):
    """The layout of a run stored without a time axis or offsets: the NI is the
    time base (raw_data + sample_rate), the Elsys a continuous 20 kHz tpc5 whose
    sample 14 000 (0.7 s) is the NI's sample 12 000 (1.2 s)."""
    tpc5 = tmp_path / "run1.tpc5"
    write_tpc5(tpc5, {"A1 (0.A1)": pulse(0.7)}, fs_continuous=20_000.0, trigger_times=())
    ni = tmp_path / "run1.npz"
    v = write_ni_npz(ni, {"n": lambda t: 1.0 + 0 * t}, fs=10_000.0, duration=3.0)
    run = {"name": "run1",
           "normal_stress": v["ai0"] * 2.0,
           "ni": S.NINpzSource.open(ni, {"ai0": "pressure_1"}).to_reference(tmp_path)
                 | {"raw_data": channel_array(v["ai0"][None, :], ["pressure_1"], unit="V")},
           "elsys": S.Tpc5Source.open(tpc5, {"A1": "pzt_1"}).to_reference(tmp_path) | {"sample_rate": 20_000.0}}
    for key in ("ni", "elsys"):
        del run[key]["time_offset"]
    if aligned:
        run["ni"]["aligned_sample"], run["elsys"]["aligned_sample"] = 12_000, 14_000
    return run


def test_sample_aligned_run_time_and_offsets(tmp_path):
    run = sample_aligned_run(tmp_path)
    assert S.time_base(run) == "ni"
    t = S.run_time(run)
    assert t.size == 30_000 and t[0] == 0.0 and t[1] == pytest.approx(1e-4)
    assert S.aligned_offset(run, "ni") == 0.0 and S.aligned_offset(run, "elsys") == pytest.approx(0.5)
    assert S.open_source(run["elsys"], tmp_path, run).time_offset == pytest.approx(0.5)
    del run["elsys"]["aligned_sample"]
    assert S.aligned_offset(run, "elsys") is None
    with pytest.raises(ValueError, match="not aligned"):
        S.open_source(run["elsys"], tmp_path, run)


def test_sample_aligned_run_load_extract_and_save(tmp_path):
    dm = DataManager()
    dm.data = {"name": "t0001", "runs": [sample_aligned_run(tmp_path)]}
    dm.save_file(tmp_path / "t0001.h5")
    back = DataManager()
    back.load_file(tmp_path / "t0001.h5")
    rrun = back.get_data("runs/[0]")
    assert rrun["time"].size == 30_000 and rrun["time"][-1] == pytest.approx(2.9999)   # filled in on load
    e = back.event_processor.extract_events(rrun, [index_at(rrun, 1.2)], window=0.01)[0]
    st = e["waveform"]["elsys"]
    assert st["sample_rate"] == 20_000.0 and st["time"][np.argmax(st["data"][0])] == pytest.approx(1.2, abs=1e-4)
    # the derived time axis is not written back, and stays in memory after the save
    back.save_file(tmp_path / "again.h5")
    with h5py.File(tmp_path / "again.h5") as f:
        assert "time" not in f["runs"]["0"]
    assert "time" in rrun


def test_unaligned_recorder_is_skipped(tmp_path, capsys):
    run = sample_aligned_run(tmp_path, aligned=False)
    run["time"] = S.run_time(run)
    e = EventProcessor(tmp_path / "t0001.h5").extract_events(run, [12_000], window=0.01)[0]
    assert set(e["waveform"]) == {"ni"}
    assert "not aligned" in capsys.readouterr().out


def test_conversion_reads_elsys_in_output_units(tmp_path):
    run = sample_aligned_run(tmp_path)
    run["elsys"]["conversion"] = {"channels": ["sigma_xy", "twice"], "inputs": ["pzt_1"],
                                  "matrix": np.array([[-3.0], [2.0]]), "unit": "MPa"}
    src = S.open_source(run["elsys"], tmp_path, run)
    assert isinstance(src, S.ConvertedSource) and src.fields == ["sigma_xy", "twice"]
    assert src.time_offset == pytest.approx(0.5) and src.base.time_offset == pytest.approx(0.5)
    volts = src.base.read_window(1.19, 1.21)
    win = src.read_window(1.19, 1.21, ["twice"])
    assert win.unit == "MPa" and win.fields == ["twice"] and win.sample_rate == 20_000.0
    np.testing.assert_allclose(win.data[0], 2.0 * volts.data[0], rtol=1e-6)
    # extracted events carry the converted channels and their unit
    run["time"] = S.run_time(run)
    e = EventProcessor(tmp_path / "t0001.h5").extract_events(run, [12_000], window=0.01)[0]
    st = e["waveform"]["elsys"]
    assert st["unit"] == "MPa" and list(st["channels"]) == ["sigma_xy", "twice"]
    assert st["time"][np.argmin(st["data"][0])] == pytest.approx(1.2, abs=1e-4)      # -3 x the pulse
    np.testing.assert_allclose(st["data"][1], -2.0 / 3.0 * st["data"][0], rtol=1e-5)


def test_conversion_needs_known_inputs(tmp_path):
    run = sample_aligned_run(tmp_path)
    run["elsys"]["conversion"] = {"channels": ["s"], "inputs": ["nope"], "matrix": [[1.0]], "unit": "MPa"}
    with pytest.raises(KeyError):
        S.open_source(run["elsys"], tmp_path, run)


def test_conversion_reads_only_the_inputs_a_channel_uses(tmp_path):
    run = sample_aligned_run(tmp_path)
    run["elsys"]["conversion"] = {"channels": ["sigma_xy"], "inputs": ["pzt_1"], "matrix": [[2.0]], "unit": "MPa"}
    src = S.open_source(run["elsys"], tmp_path, run)
    src.inputs, src.matrix = ["pzt_1", "spare"], np.array([[2.0, 0.0]])   # a second input this row ignores
    win = src.read_window(1.19, 1.21, ["sigma_xy"])                       # "spare" is not a field: never read
    assert win.fields == ["sigma_xy"] and win.unit == "MPa"


def test_mean_on_samples_puts_a_converted_channel_on_the_run_samples(tmp_path):
    run = sample_aligned_run(tmp_path)
    run["elsys"]["conversion"] = {"channels": ["sigma_xy"], "inputs": ["pzt_1"], "matrix": [[-3.0]], "unit": "MPa"}
    src = S.open_source(run["elsys"], tmp_path, run)
    t = S.run_time(run)
    y = S.mean_on_samples(src, "sigma_xy", t, chunk_s=0.3)
    assert y.shape == t.shape
    covered = (t >= 0.5 + 0.5e-4) & (t <= 2.5 - 0.5e-4)            # the 2 s Elsys record starts at run 0.5 s
    assert np.isnan(y[t < 0.5 - 1e-4]).all() and np.isnan(y[t > 2.5 + 1e-4]).all() and np.isfinite(y[covered]).all()
    i = int(np.nanargmin(y))
    assert t[i] == pytest.approx(1.2, abs=1e-4)                     # the pulse, at the run's samples
    win = src.read_window(t[i] - 0.5e-4, t[i] + 0.5e-4, ["sigma_xy"], dtype=np.float64)
    assert win.data[0].min() <= y[i] <= win.data[0].max() and y[i] < -0.2   # the mean of the Elsys samples there


def test_waveform_fields_choice(elsys, tmp_path):
    run = run_from_tpc5(elsys, ELSYS_MAP, tmp_path, event_fields=["pzt_1"])
    ep = EventProcessor(tmp_path / "exp.h5")
    source = ep.open_sources(run)["elsys"]
    everything = list(source.fields)
    assert len(everything) > 1
    idx = [index_at(run, 0.5)]
    assert list(ep.extract_events(run, idx, window=0.05)[0]["waveform"]["elsys"]["channels"]) == ["pzt_1"]
    e_all = ep.extract_events(run, idx, window=0.05, fields="all")[0]
    assert list(e_all["waveform"]["elsys"]["channels"]) == everything
    assert e_all["waveform"]["elsys"]["data"].shape[0] == len(everything)
    chosen = ep.extract_events(run, idx, window=0.05, fields=[everything[-1], "nope"])[0]
    assert list(chosen["waveform"]["elsys"]["channels"]) == [everything[-1]]
    assert list(ep.extract_events(run, idx, window=0.05, fields=f"{everything[0]}, {everything[-1]}")[0]
                ["waveform"]["elsys"]["channels"]) == [everything[0], everything[-1]]
    missing = ep.extract_events(run, idx, window=0.05, fields=["nope"])[0]
    assert "waveform" not in missing and any("none of the fields" in n for n in missing["notes"])
    assert EventProcessor.waveform_fields(source, None) is None
