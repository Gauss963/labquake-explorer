import numpy as np
import pytest

from labquake_explorer.data import channels as C
from labquake_explorer.data import sources as S
from labquake_explorer.preprocessing import (
    Calibration, EddySlip, Friction, Linear, experiment, offset_from_trigger, parse_run_name,
    pressure_transducers, run_from_sources, run_from_tpc5, run_from_tpc5_ni, slip_step_table,
)
from tests.synthetic import write_ni_npz, write_tpc5


def pulse(t0, width=1e-4, amp=0.1):
    return lambda t: amp * np.exp(-((t - t0) / width) ** 2)


def synthetic_run(fs=2000.0, duration=2.0):
    t = np.arange(int(duration * fs)) / fs
    return {
        "name": "run1", "time": t,
        "pressure_1": (0.222 * 8.0 + 0.091 + 0 * t).astype(np.float32),
        "pressure_2": (0.326 * 5.0 + 0 * t).astype(np.float32),
        "eddy_1": (-0.5 - 0.01 * t).astype(np.float32),        # gap opens: V falls, slip grows
        "eddy_2": (1.0 - 0.02 * t).astype(np.float32),
        "lvdt": (4.0 + 0 * t).astype(np.float32),
    }


def test_linear_friction_and_presets():
    run = synthetic_run()
    cal = Calibration(*pressure_transducers("1D-5cm"), Friction(), note="thesis table B.1")
    cal.apply(run)
    assert run["units"]["pressure_1"] == "V" and run["units"]["normal_stress"] == "MPa"
    assert run["normal_stress"].mean() == pytest.approx(8.0, abs=1e-5)
    assert run["shear_stress"].mean() == pytest.approx(5.0, abs=1e-5)
    assert run["friction"].mean() == pytest.approx(5.0 / 8.0, abs=1e-5) and run["units"]["friction"] == ""
    desc = run["calibration"]
    assert desc["note"] == "thesis table B.1" and [s["kind"] for s in desc["steps"]] == ["linear", "linear", "friction"]
    assert "normal_stress (MPa)" in desc["steps"][0]["formula"]
    # the 1D factors are the 2D constants scaled by the fault area (0.1 and 0.2 of the 2D sample) and the jacks
    # pressed: three of nine on the normal line, one of three on the shear line (Table B.1 assumed three)
    from labquake_explorer.preprocessing import PRESSURE_PRESETS as P
    for section, area in (("1D-5cm", 0.1), ("1D-10cm", 0.2)):
        assert P[section]["normal_v_per_mpa"] == pytest.approx(P["2D-50cm"]["normal_v_per_mpa"] * area * 9 / 3, rel=2e-3)
        assert P[section]["shear_v_per_mpa"] == pytest.approx(P["2D-50cm"]["shear_v_per_mpa"] * area * 3 / 1, rel=2e-3)
    # a missing source is skipped, not an error
    run2 = {"time": np.arange(5.0), "x": np.ones(5, dtype=np.float32)}
    Calibration(Linear("y", "nope", 2.0, unit="u"), Friction()).apply(run2)
    assert "y" not in run2 and "friction" not in run2 and run2["units"] == {"x": "V"}


def test_eddy_slip_sign_zero_and_displacement():
    run = synthetic_run()
    step = EddySlip({"eddy_1": -0.0947}, zero_window_s=0.5, positions={"eddy_1": (120.0, 0.0, 25.0)}, displacement="mean")
    Calibration(step).apply(run)
    slip = run["slip"]
    assert C.is_channel_array(slip) and slip["channels"] == ["slip_1", "slip_2"] and slip["unit"] == "um"
    assert slip["source"] == ["eddy_1", "eddy_2"] and slip["slope_mm_per_v"] == pytest.approx([-0.0947, -0.0947])
    assert slip["positions"]["x"] == pytest.approx([120.0, float("nan")], nan_ok=True) and slip["positions"]["unit"] == "mm"
    s1, s2 = C.row(slip, "slip_1"), C.row(slip, "slip_2")
    assert run["units"]["slip"] == "um" and run["units"]["displacement"] == "um"
    # zeroed on the first 0.5 s: mean of the first 1000 samples is ~0
    assert abs(s1[:1000].mean()) < 1e-3
    # V falls by 0.01*2 = 0.02 V over the run -> slip rises by 0.0947 mm/V * 0.02 V = 1.894 um
    assert s1[-1] - s1[0] == pytest.approx(0.0947 * 0.02 * 1000 * (3999 / 4000), rel=1e-3)
    assert np.all(np.diff(s1) >= 0)
    # eddy_2 has no slope of its own -> mean of the known slopes (here the only one)
    assert s2[-1] - s2[0] == pytest.approx(2 * (s1[-1] - s1[0]), rel=1e-3)
    np.testing.assert_allclose(run["displacement"], (s1 + s2) / 2, atol=1e-5)
    desc = run["calibration"]["steps"][0]
    assert desc["kind"] == "eddy_slip" and desc["default_slope_mm_per_v"] == pytest.approx(-0.0947)

    run = synthetic_run()
    Calibration(EddySlip({"eddy_1": -0.1, "eddy_2": -0.05}, displacement="slip_2")).apply(run)
    np.testing.assert_array_equal(run["displacement"], C.row(run["slip"], "slip_2"))
    run = synthetic_run()
    Calibration(EddySlip({}, default_slope_mm_per_v=-0.1)).apply(run)      # no displacement by default
    assert "displacement" not in run and C.channel_names(run["slip"]) == ["slip_1", "slip_2"]
    # sources inside a recorder's raw_data channel array, positions inherited from it
    n = 100
    raw = C.channel_array(np.vstack([np.full(n, 1.0), -0.5 - 0.01 * np.arange(n)]), ["pressure_1", "eddy_1"],
                          unit="V", positions=C.positions_table(["pressure_1", "eddy_1"], {"eddy_1": (10.0, 20.0, 30.0)}, unit="cm"))
    run = {"time": np.arange(n) / 10.0, "elsys": {"format": "tpc5", "filename": "x.tpc5", "raw_data": raw}}
    Calibration(Linear("normal_stress", "pressure_1", 8.0, unit="MPa"), EddySlip({"eddy_1": -0.1})).apply(run)
    assert run["normal_stress"][0] == pytest.approx(8.0) and run["units"]["elsys/raw_data"] == "V"
    assert run["slip"]["positions"]["x"] == [10.0] and run["slip"]["positions"]["unit"] == "cm"
    with pytest.raises(KeyError):
        Calibration(EddySlip({"eddy_1": -0.1}, displacement="slip_9")).apply(synthetic_run())
    with pytest.raises(ValueError):
        Calibration(EddySlip({})).apply(synthetic_run())


def test_parse_run_name():
    assert parse_run_name("t0211_03_12MPa_run3") == {"name": "run3", "normal_stress_level": 12.0, "experiment": "t0211"}
    assert parse_run_name("T0207_04_16MPa_run5") == {"name": "run5", "normal_stress_level": 16.0, "experiment": "t0207"}
    assert parse_run_name("p5986_01_1kHz_load") == {"name": "p5986_01_1kHz_load"}


@pytest.fixture
def elsys(tmp_path):
    path = tmp_path / "t0001_01_8MPa_run1.tpc5"
    write_tpc5(path, {
        "A1 (0.A1)": lambda t: pulse(0.5)(t) + pulse(1.4)(t),
        "A5 (0.A5)": lambda t: 0.222 * 8.0 + 0.091 + 0 * t,
        "A6 (0.A6)": lambda t: 0.326 * 5.0 + 0 * t,
        "B1 (0.B1)": lambda t: -0.5 - 0.01 * t,
    }, trigger_times=(0.5, 1.4), volt_range=12.0)
    return path


ELSYS_MAP = {"A1": "pzt_1", "A5": "pressure_1", "A6": "pressure_2", "B1": "eddy_1"}


def calibration():
    return Calibration(*pressure_transducers("1D-5cm"), EddySlip({"eddy_1": -0.0947}), Friction())


def test_run_from_tpc5(elsys, tmp_path):
    run = run_from_tpc5(elsys, ELSYS_MAP, tmp_path, calibration(), event_fields=["pzt_1"],
                        event_window_s=(0.001, 0.004), operator="x")
    assert run["name"] == "run1" and run["normal_stress_level"] == 8.0 and run["operator"] == "x"
    assert run["elsys"]["filename"] == elsys.name and run["start_time"].startswith("2026-01-01")
    assert not any(k in run for k in ("file", "file_index", "ni_file", "sources"))     # the recorder block says it all
    assert run["time"].size == 4000 and run["time"][0] == 0.0
    raw = run["elsys"]["raw_data"]
    assert C.is_channel_array(raw) and raw["channels"] == ["pzt_1", "pressure_1", "pressure_2", "eddy_1"]
    assert raw["data"].shape == (4, 4000) and raw["unit"] == "V" and "raw_data" not in run
    assert "positions" not in raw
    assert "displacement" not in run
    for f in ("normal_stress", "shear_stress", "friction"):
        assert run[f].shape == (4000,), f
    assert run["slip"]["data"].shape == (1, 4000) and run["slip"]["channels"] == ["slip_1"]
    assert run["units"]["normal_stress"] == "MPa" and run["units"]["slip"] == "um" and run["units"]["elsys/raw_data"] == "V"
    assert "pzt_1" not in run and "eddy_1" not in run
    assert run["normal_stress"].mean() == pytest.approx(8.0, abs=1e-3)
    ref = run["elsys"]
    assert ref["format"] == "tpc5"
    assert ref["format"] == "tpc5" and ref["filename"] == elsys.name and ref["time_offset"] == 0.0
    assert ref["event_fields"] == ["pzt_1"] and ref["event_window_s"] == [0.001, 0.004]
    assert ref["blocks"]["trigger_time"] == pytest.approx([0.5, 1.4])
    assert run["calibration"]["steps"][0]["kind"] == "linear"


@pytest.fixture
def ni(tmp_path):
    # NI clock runs 0.7 s ahead of the Elsys clock: the first Elsys trigger (0.5 s) is NI sample 12000 (1.2 s)
    path = tmp_path / "T0001-raw-run1-20260101_000000.npz"
    write_ni_npz(path, {
        "n": lambda t: 0.222 * 8.0 + 0.091 + 0 * t,
        "s": lambda t: 0.326 * 5.0 + 0 * t,
        "e": lambda t: -0.5 - 0.01 * t - 0.3 * (t > 1.25),          # slip step 50 ms after the trigger
    }, fs=10_000.0, duration=3.0, trigger_sample_index=12_000)
    return path


NI_MAP = {"ai0": "pressure_1", "ai1": "pressure_2", "ai2": "eddy_1"}


def test_offset_and_slip_step_table(elsys, ni, tmp_path):
    ni_src = S.NINpzSource.open(ni, NI_MAP)
    elsys_src = S.Tpc5Source.open(elsys, ELSYS_MAP)
    offset = offset_from_trigger(ni_src, elsys_src)
    assert offset == pytest.approx(0.7)
    elsys_src.time_offset = offset
    rows = slip_step_table(ni_src, elsys_src.trigger_times(), "eddy_1", decimation=10, half=0.5, step_min_v=0.05)
    assert [r["slip_step"] for r in rows] == [True, False]
    assert rows[0]["lag_ms"] == pytest.approx(50.0, abs=3.0) and rows[0]["dV"] == pytest.approx(-0.3, abs=0.01)
    assert rows[1]["note"] == "" and np.isnan(rows[1]["lag_ms"])


def test_run_from_tpc5_ni(elsys, ni, tmp_path):
    run = run_from_tpc5_ni(elsys, ni, ELSYS_MAP, NI_MAP, tmp_path, calibration(), ni_decimation=5,
                           ni_meta={"input_range_v": 10.0}, elsys_event_fields=["pzt_1"])
    assert run["name"] == "run1" and run["ni"]["filename"] == ni.name and run["elsys"]["filename"] == elsys.name
    assert run["ni"]["format"] == "ni_npz" and run["elsys"]["format"] == "tpc5"
    assert not any(k in run for k in ("file", "file_index", "ni_file", "sources"))
    assert "labels" not in run["ni"] and "member_offsets" not in run["ni"]
    # NI is the time base: 3 s at 2 kHz
    assert run["time"].size == 6000 and run["time"][1] - run["time"][0] == pytest.approx(5e-4)
    assert run["ni"]["raw_data"]["channels"] == ["pressure_1", "pressure_2", "eddy_1"]
    assert run["elsys"]["raw_data"]["channels"] == ["pzt_1"]            # fields the NI already has are not duplicated
    assert run["units"]["ni/raw_data"] == "V" and run["units"]["elsys/raw_data"] == "V"
    assert run["normal_stress"].mean() == pytest.approx(8.0, abs=1e-3)
    # the Elsys PZT is interpolated onto the NI axis with the 0.7 s shift: pulse at 1.2 s, NaN beyond the Elsys record
    pz = C.get_channel(run, "pzt_1")
    assert run["time"][np.nanargmax(pz)] == pytest.approx(1.2, abs=2e-3)
    assert np.isnan(pz[-1]) and not np.isnan(pz[2000])
    assert run["elsys"]["time_offset"] == pytest.approx(0.7) and run["ni"]["time_offset"] == 0.0
    assert run["elsys"]["clock_offset_method"].startswith("NI trigger")
    assert run["elsys"]["blocks"]["trigger_time_run"] == pytest.approx([1.2, 2.1])
    assert run["ni"]["decimation"] == 5 and run["ni"]["input_range_v"] == 10.0
    assert run["slip"]["data"].dtype == np.float32 and run["units"]["slip"] == "um"
    # a given offset wins over the trigger-derived one
    run2 = run_from_tpc5_ni(elsys, ni, ELSYS_MAP, NI_MAP, tmp_path, clock_offset=0.5)
    assert run2["elsys"]["time_offset"] == 0.5 and run2["elsys"]["clock_offset_method"] == "given"


def test_run_from_sources_validation(elsys, tmp_path):
    src = S.Tpc5Source.open(elsys, ELSYS_MAP)
    with pytest.raises(ValueError):
        run_from_sources({}, tmp_path)
    with pytest.raises(KeyError):
        run_from_sources({"elsys": src}, tmp_path, time_base="ni")
    run = run_from_sources({"elsys": src}, tmp_path, name="custom", decimation={"elsys": 4},
                           positions={"pzt_1": (1.0, 2.0, 3.0)}, position_frame="x along the fault")
    assert run["name"] == "custom" and run["time"].size == 1000 and "calibration" not in run
    pos = run["elsys"]["raw_data"]["positions"]
    assert pos["x"][0] == 1.0 and np.isnan(pos["x"][1]) and pos["frame"] == "x along the fault"
    assert C.aligned_fields(run, 1000, recurse=False)[:2] == ["time", "elsys/raw_data/pzt_1"]
    exp = experiment("t0001", [run], date="2026-01-01")
    assert exp["name"] == "t0001" and exp["date"] == "2026-01-01" and exp["runs"][0] is run
