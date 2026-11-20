"""Headless-Tk tests for PZTSpectrumView on synthetic strain data."""
import contextlib
import copy
import json
import math
import os

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from labquake_explorer.analysis import source as src
from labquake_explorer.analysis import spectrum as sp
from labquake_explorer.analysis import spectrum_ensemble as se
from labquake_explorer.ui.views.pzt_spectrum_view import (
    HANDLES, PZTSpectrumView, RESULT_VERSION, TIE, TIE_AFTER, TRIGGER_PICKED, TRIGGER_EVENT_TIME, TRIGGER_RECORD,
    TUKEY_ALPHA, channel_key, ensemble_stats, parse_channel_key,
)

EVENT = 1
BRUNE_CH = 3          # channel replaced by a Brune displacement pulse
TANH_CH = 0           # untouched tanh step (arrival = event_time)
OMEGA0_V = 1.0        # plateau of the synthetic pulse in V*s
FC = 10e3             # Hz
GAIN = 2.0            # V/m -> displacement plateau = OMEGA0_V / GAIN
N_WIN = 500           # samples in the default 0.5 + 2.0 ms window at 200 kHz
RHO, VP, VS, DIST = 2700.0, 6000.0, 3500.0, 0.1
RAD_S = src.PHASE_CONSTANTS["S"]["radiation_rms"]


def write_calibration(path, gain, header="frequency_hz,gain"):
    f = np.geomspace(100.0, 100e3, 25)
    path.write_text(header + "\n" + "\n".join(f"{fi:.6g},{gain}" for fi in f) + "\n")
    return str(path)


@pytest.fixture
def brune_app(app_with_strain, tmp_path):
    """Event 1 / channel 3 replaced by a Brune pulse plus white noise 30 dB below
    the plateau, tiny noise on every other channel (so the SNR gate has a finite
    noise spectrum), and a V/m calibration CSV."""
    app = app_with_strain
    event = app.data_manager.get_data(f"runs/[0]/events/[{EVENT}]")
    original = event["strain"]["original"]
    time = np.asarray(original["time"], dtype=float)
    fs = sp.sampling_rate(time)
    rng = np.random.default_rng(7)
    raw = np.asarray(original["raw"], dtype=float) + rng.normal(0.0, 5e-5, np.shape(original["raw"]))
    _, pulse = sp.brune_pulse(fs, time.size, OMEGA0_V, FC, float(event["event_time"]) - time[0])
    raw[BRUNE_CH] = pulse + rng.normal(0.0, OMEGA0_V / math.sqrt(1000.0) * fs / math.sqrt(N_WIN), time.size)
    original["raw"] = raw
    app.calibration_csv = write_calibration(tmp_path / "sensor.csv", GAIN)
    app.fs = fs
    return app


@pytest.fixture
def view(brune_app):
    v = PZTSpectrumView(brune_app, 0, EVENT)
    yield v
    if v.winfo_exists():
        v.on_close()


@contextlib.contextmanager
def opened(app, event=EVENT):
    v = PZTSpectrumView(app, 0, event)
    try:
        yield v
    finally:
        v.on_close()


def calibrated_compute(view, channel=BRUNE_CH):
    view.set_channel(channel)
    view.calibration_path_var.set(view.app.calibration_csv)
    view.unit_combobox.set("V/m")
    assert view.compute(), view.status_var.get()
    return view.fit


def set_medium(view, phase="P", radiation=None):
    view.phase_combobox.set(phase)
    view.on_phase_selected()
    view.rho_var.set(str(RHO)); view.c_var.set(str(VP)); view.vs_var.set(str(VS))
    view.distance_var.set(str(DIST))
    if radiation is not None:
        view.radiation_var.set(str(radiation))


def stored(view, set_name="default"):
    return view.event["strain"]["spectrum"]["sets"][set_name]["channels"]


def experiment(app, set_name="default"):
    return app.data_manager.data["analysis"]["pzt_spectrum"]["sets"][set_name]


def mouse(view, name, x_ms, dx_px=0.0):
    ax = view.ax_time
    px, py = ax.transData.transform((x_ms, float(np.mean(ax.get_ylim()))))
    return MouseEvent(name, view.canvas, px + dx_px, py, button=1)


def drag(view, what, x0, *targets):
    view.begin_drag(what, x0)
    edges = [view.drag_to(x) for x in targets]
    view.end_drag()
    return edges[-1] if edges else None


def shown(view):
    view.deiconify()
    view.update()
    view.canvas.draw()
    view.update()
    if not view.canvas.get_tk_widget().winfo_ismapped():
        pytest.skip("window not mapped: Tk mouse events need a display")
    return view.canvas.get_tk_widget()


def tk_drag(view, x0_ms, x1_ms):
    widget = view.canvas.get_tk_widget()

    def xy(x_ms):
        px, py = view.ax_time.transData.transform((x_ms, float(np.mean(view.ax_time.get_ylim()))))
        return int(round(px)), int(round(view.canvas.figure.bbox.height - py))

    (x0, y0), (x1, y1) = xy(x0_ms), xy(x1_ms)
    widget.event_generate("<Button-1>", x=x0, y=y0)
    widget.update()
    for f in (0.3, 0.6, 1.0):
        widget.event_generate("<Motion>", x=int(x0 + f * (x1 - x0)), y=int(y0 + f * (y1 - y0)), state=0x100)
        widget.update()
    widget.event_generate("<ButtonRelease-1>", x=x1, y=y1)
    widget.update()


# ---------------------------------------------------------------------------
def test_opens_computed_and_fits_the_uncalibrated_channels(brune_app, view):
    # defaults, computed on opening: there is no Compute button
    assert view.event_idx == EVENT and view.title() == "PZT Spectrum - Event 1" and view in brune_app.child_windows
    assert list(view.channel_combobox["values"]) == [str(i) for i in range(16)] and view.current_channel() == 0
    assert view.trigger_combobox.get() == TRIGGER_EVENT_TIME and (view.pre_var.get(), view.post_var.get()) == ("0.5", "2.0")
    assert float(view.fmax_var.get()) == pytest.approx(0.4 * brune_app.fs) and float(view.fmin_var.get()) == 1e3
    assert view.unit_combobox.get() == "V/V" and view.calibration_path_var.get() == ""
    assert view.n_fixed_var.get() and view.snr_var.get() == "3" and view.n_max_var.get() == "4"
    assert float(view.radiation_var.get()) == pytest.approx(src.PHASE_CONSTANTS["P"]["radiation_rms"])
    assert view.radiation_hint_var.get() == "RMS for P: 0.516"
    assert view.window is not None and view.spectrum is not None and view.record is not None
    assert "fit" in view.status_var.get() and "Compute" not in view.status_var.get()
    assert not hasattr(view, "compute_button")
    # a strain block has no recorder trigger of its own
    view.trigger_combobox.set(TRIGGER_RECORD)
    assert not view.compute() and "no trigger time" in view.status_var.get()
    # picked arrival on the tanh channel: no picks yet is a status line, not a dialog
    event = view.event
    arrivals = np.asarray(event["strain_truth"]["arrival_times"], dtype=float)
    view.trigger_combobox.set(TRIGGER_PICKED)
    assert not view.compute() and "picked arrival" in view.status_var.get()
    event["strain"]["original"]["rupture_arrival_time"] = arrivals
    assert view.compute(), view.status_var.get()
    time = np.asarray(event["strain"]["original"]["time"], dtype=float)
    assert view.window.i_trigger == int(np.argmin(np.abs(time - arrivals[TANH_CH])))
    assert view.record["trigger"] == TRIGGER_PICKED and view.record["trigger_time_s"] == pytest.approx(arrivals[TANH_CH])
    assert view.record["trigger_pick_set"] == "default"
    # a pick set saved later becomes the current one; the saved record keeps reading the set it was made with
    assert view.save()
    from labquake_explorer.data.sources import write_arrivals
    write_arrivals(event["strain"], {"arrival_time": arrivals + 1e-4}, event, set_name="manual2")
    view.set_channel(BRUNE_CH)
    view.set_channel(TANH_CH)
    assert view.record["trigger_pick_set"] == "default" and "matches the saved record" in view.status_var.get()
    view.trigger_combobox.set(TRIGGER_EVENT_TIME)
    # the tanh step removed: the window ends near zero
    view.remove_step_var.set(True)
    assert view.compute(), view.status_var.get()
    assert view.window.step_removed and view.window.step_amplitude == pytest.approx(0.5, rel=0.05)
    assert view.fit.valid, view.fit.reason
    assert view.record["remove_step"] is True and view.record["spectrum"]["meta"]["step_removed"] is True
    assert abs(np.mean(view.window.signal[-50:])) < 0.02
    view.remove_step_var.set(False)
    # the Brune channel: the top panel shows the taper and the tapered signal
    view.set_channel(BRUNE_CH)
    labels = [line.get_label() for line in view.ax_time.lines]
    assert {"signal window", "noise window", "tapered (what is transformed)"} <= set(labels)
    taper = view.ax_taper.lines[0].get_ydata()
    assert taper.max() == pytest.approx(1.0) and taper.min() == pytest.approx(0.0, abs=1e-9)
    assert view.ax_taper.get_ylim() == (0.0, 1.05) and view.ax_taper.get_ylabel() == "taper"
    sig = next(l for l in view.ax_time.lines if l.get_label() == "signal window").get_ydata()
    tapered = next(l for l in view.ax_time.lines if l.get_label() == "tapered (what is transformed)").get_ydata()
    np.testing.assert_allclose(tapered, sig * taper)
    # uncalibrated fit recovers the voltage plateau
    assert view.compute(), view.status_var.get()
    fit = view.fit
    assert fit.valid and fit.amp_units == "V*s" and fit.warnings == ()
    assert fit.omega0 == pytest.approx(OMEGA0_V, rel=0.05) and fit.fc == pytest.approx(FC, rel=0.05)
    assert fit.n == 2.0 and fit.n_fixed and fit.converged and not fit.band_limited and fit.n_bounds == (1.0, 4.0)
    assert view.window.n == N_WIN and view.window.noise_available
    assert view.spectrum.meta["transient_in_flat_region"] is True
    assert view.record["channel"] == BRUNE_CH and view.record["fit"]["valid"] and view.record["parameter_warnings"] == []
    assert view.record["n_max"] == 4.0 and view.record["fit"]["n_bounds"] == [1.0, 4.0]
    assert "V*s" in view.result_vars["omega0"].get() and view.result_vars["fc"].get().endswith("Hz")
    flags = view.result_vars["flags"].get()
    assert "converged: yes" in flags and "band-limited: no" in flags and "WARNINGS" not in flags
    assert "flat region: yes" in view.result_vars["taper"].get() and view.result_vars["n"].get().startswith("2")
    assert "fit OK" in view.status_var.get()
    # all bins clear the 3x gate at 30 dB below the plateau
    assert int(view.result_vars["n_bins"].get()) == fit.n_bins >= 8
    # free n with the soft_l1 loss
    view.n_fixed_var.set(False)
    view.loss_combobox.set("soft_l1")
    assert view.compute()
    assert view.fit.valid and not view.fit.n_fixed and view.fit.n == pytest.approx(2.0, abs=0.3)
    assert view.fit.fc == pytest.approx(FC, rel=0.10)
    assert view.record["loss"] == "soft_l1" and view.record["n_fixed"] is False and "(free)" in view.result_vars["n"].get()
    # closing unregisters the window
    view.on_close()
    assert view not in brune_app.child_windows


def test_controls_reach_the_analysis_live_and_errors_go_to_status(brune_app, view):
    view.set_channel(BRUNE_CH)
    assert view.record["post_ms"] == 2.0
    # every entry recomputes on Return, on leaving the field, and after a pause while typing
    for entry in view.entries.values():
        assert {"<Key-Return>", "<FocusOut>", "<KeyRelease>"} <= set(entry.bind())
    view.post_var.set("1.5")
    view.schedule_compute(now=True)                      # what Return / FocusOut do
    assert view.record["post_ms"] == 1.5 and view.window.n_post == int(round(1.5e-3 * brune_app.fs))
    view.pre_var.set("0.6")
    view.schedule_compute()                              # a keystroke: wait for the keys to settle
    assert view.record["pre_ms"] == 0.5 and view._compute_after_id is not None
    view.schedule_compute(now=True)                      # what the timer does
    assert view.record["pre_ms"] == 0.6 and view._compute_after_id is None
    view.pre_var.set("0.55")
    view.schedule_compute()
    assert view.save() and view.record["pre_ms"] == 0.55  # Save straight after typing saves what was typed
    assert stored(view)[f"ch{BRUNE_CH}"]["pre_ms"] == 0.55
    # a combobox and a checkbox recompute at once
    view.taper_combobox.set("bh4")
    view.taper_combobox.event_generate("<<ComboboxSelected>>")
    assert view.record["taper"] == "bh4"
    # a half-typed number is reported (the stale numbers cleared); the next valid value recovers
    view.pre_var.set("0.a")
    view.schedule_compute(now=True)
    assert "pre" in view.status_var.get().lower() and view.record is None
    view.pre_var.set("0.5"); view.post_var.set("2.0"); view.taper_combobox.set("tukey")
    view.n_fixed_var.set(False)
    view.schedule_compute(now=True)
    assert view.record["pre_ms"] == 0.5 and view.record["n_fixed"] is False
    # n max bounds the free exponent
    view.n_max_var.set("8")
    view.schedule_compute(now=True)
    assert view.fit.valid and view.fit.n_bounds == (1.0, 8.0) and view.record["fit"]["n_bounds"] == [1.0, 8.0]
    assert view.record["n_max"] == 8.0 and view.fit.n == pytest.approx(2.0, abs=0.2) and not view.fit.at_bounds["n"]
    view.n_max_var.set("1.5")                       # a ceiling under the truth: n sits on it, flagged for these bounds
    view.schedule_compute(now=True)
    assert view.fit.n == pytest.approx(1.5, abs=0.02) and view.fit.at_bounds["n"]
    assert "at bound: n" in view.result_vars["flags"].get()
    view.n_max_var.set("1")                         # not above n min
    view.schedule_compute(now=True)
    assert view.record is None and "n max" in view.status_var.get()
    view.n_max_var.set("4"); view.n_fixed_var.set(True)
    # invalid entries go to the status line, not a dialog
    for var, bad, message, good in ((view.pre_var, "abc", "pre window", "0.5"),
                                    (view.fmax_var, "500", "fmax", "80000"),           # below fmin
                                    (view.post_var, "50", "not inside the trace", "2.0")):   # leaves the 20 ms trace
        var.set(bad)
        assert not view.compute() and message in view.status_var.get()
        var.set(good)
    view.baseline_combobox.set("highpass")
    view.highpass_var.set("")
    assert not view.compute() and "highpass" in view.status_var.get()
    view.highpass_var.set("500")
    assert view.compute()
    assert view.record["baseline"] == "highpass" and view.record["highpass_hz"] == 500.0
    assert view.window.baseline_mode == "highpass" and view.window.highpass_hz == 500.0
    view.baseline_combobox.set("pre_linear")
    view.unit_combobox.set("V/m")                         # a unit without a CSV
    assert not view.compute() and "needs a CSV" in view.status_var.get()
    view.calibration_path_var.set("/nonexistent/sensor.csv")
    assert not view.compute() and "calibration CSV" in view.status_var.get()
    assert view.record is not None                        # the last good record is kept
    view.unit_combobox.set("V/V"); view.calibration_path_var.set("")
    # every entry is applied, not merely stored in the record
    arrivals = np.asarray(view.event["strain_truth"]["arrival_times"], dtype=float)
    view.event["strain"]["original"]["rupture_arrival_time"] = arrivals
    view.trigger_combobox.set(TRIGGER_PICKED)
    view.pre_var.set("0.6"); view.post_var.set("1.8"); view.taper_combobox.set("tukey")
    view.baseline_combobox.set("highpass"); view.highpass_var.set("300"); view.t_star_var.set("1e-6")
    view.fmin_var.set("2000"); view.fmax_var.set("70000"); view.bins_var.set("20"); view.snr_var.set("2")
    view.n_fixed_var.set(False); view.loss_combobox.set("soft_l1")
    assert view.compute(), view.status_var.get()
    fs = brune_app.fs
    assert view.window.n_pre == round(0.6e-3 * fs) and view.window.n_post == round(1.8e-3 * fs)
    assert view.window.trigger_time == pytest.approx(arrivals[BRUNE_CH])
    assert view.window.baseline_mode == "highpass" and view.window.highpass_hz == 300.0
    assert view.window.baseline_region == "highpass" and view.spectrum.t_star_s == 1e-6
    meta = view.spectrum.meta
    assert meta["taper"] == "tukey" and meta["alpha"] == TUKEY_ALPHA and meta["highpass_hz"] == 300.0
    assert meta["fmin"] == 2000.0 and meta["fmax"] == 70000.0 and meta["bins_per_decade"] == 20
    assert view.fit.snr_min == 2.0 and view.fit.loss == "soft_l1" and not view.fit.n_fixed
    assert view.fit.band_used[0] >= 2000.0 and view.fit.band_used[1] <= 70000.0
    # t* off gives a different plateau: the correction was applied
    view.t_star_var.set("")
    omega_t = view.fit.omega0
    assert view.compute() and view.spectrum.t_star_s is None
    assert view.fit.omega0 != pytest.approx(omega_t, rel=1e-6)


def test_calibration_csv_db_flag_and_moment_unit(brune_app, view, tmp_path):
    fit = calibrated_compute(view)
    assert fit.valid and fit.amp_units == "m*s" and view.spectrum.calibrated
    assert fit.omega0 == pytest.approx(OMEGA0_V / GAIN, rel=0.05) and fit.fc == pytest.approx(FC, rel=0.05)
    assert view.record["calibration_unit"] == "V/m" and view.record["calibration_path"] == brune_app.calibration_csv
    assert view.record["spectrum"]["meta"]["calibration"]["unit"] == "V/m" and "m*s" in view.result_vars["omega0"].get()
    # a dB table read with the flag; without it the 6 dB column is a linear gain of ~6
    db = 20.0 * math.log10(GAIN)
    view.calibration_path_var.set(write_calibration(tmp_path / "sensor_db.csv", f"{db:.6f}", "Freq,Response"))
    view.calibration_db_var.set(True)
    assert view.compute(), view.status_var.get()
    assert view.fit.valid and view.fit.omega0 == pytest.approx(OMEGA0_V / GAIN, rel=0.05)
    assert view.record["calibration_db"] is True
    view.calibration_db_var.set(False)
    assert view.compute() and view.fit.omega0 == pytest.approx(OMEGA0_V / db, rel=0.05)
    # a moment unit: the gain divided, the medium entries greyed, M0 taken from the plateau
    view.calibration_path_var.set(write_calibration(tmp_path / "psi.csv", 2 * GAIN, "Frequency (Hz),Mean_Inst_App_Psi"))
    view.unit_combobox.set("V/(N m)")
    view.calibration_divisor_var.set("2")                     # Psi / C_FM, say
    assert view.compute(), view.status_var.get()
    assert view.spectrum.amp_units == "N*m*s" and view.fit.valid
    assert view.fit.omega0 == pytest.approx(OMEGA0_V / GAIN, rel=0.05)
    assert all(entry.instate(["disabled"]) for entry in view.medium_entries.values())
    assert "M0 = Omega0" in view.source_hint_var.get() and "gain / 2" in view.calibration_note_var.get()
    assert view.record["calibration_divisor"] == 2.0 and view.record["calibration"]["gain_divisor"] == 2.0
    view.rho_var.set(""); view.c_var.set(""); view.distance_var.set("")     # only the phase and Vs are needed
    view.vs_var.set(str(VS))
    assert view.compute_source(), view.status_var.get()
    source = view.record["source"]
    assert source["valid"] and source["seismic_moment_nm"] == pytest.approx(view.fit.omega0)
    assert source["source_radius_m"] == pytest.approx(src.PHASE_CONSTANTS["P"]["k"] * VS / view.fit.fc)
    assert source["constants"]["moment_spectrum"] is True and source["constants"]["rho_kg_m3"] is None
    assert math.isnan(source["kR"]) and "M0 = Omega0" in view.status_var.get()
    assert view.source_vars["m0"].get().endswith("N m")
    assert view.save()
    # back to a displacement unit: the entries come back and rho is required again
    view.unit_combobox.set("V/m")
    view.calibration_divisor_var.set("")
    assert view.compute(), view.status_var.get()
    assert view.spectrum.amp_units == "m*s"
    assert not any(entry.instate(["disabled"]) for entry in view.medium_entries.values())
    assert not view.compute_source() and "rho" in view.status_var.get()
    view.on_close()
    with opened(brune_app) as v2:                             # the unit and the divisor come back with the record
        assert v2.current_channel() == BRUNE_CH and v2.unit_combobox.get() == "V/(N m)"
        assert v2.calibration_divisor_var.get() == "2" and v2.record["spectrum"]["amp_units"] == "N*m*s"
        assert all(entry.instate(["disabled"]) for entry in v2.medium_entries.values())


def test_calibration_table_is_stored_survives_hdf5_and_serves_without_the_csv(brune_app, view, tmp_path, app_from_file):
    assert view.calibration_note_var.get().startswith("no calibration loaded")
    calibrated_compute(view)
    note = view.calibration_note_var.get()
    assert note.startswith("loaded: sensor.csv") and "25 rows" in note and "V/m" in note
    table = view.record["calibration"]
    assert len(table["frequency_hz"]) == 25 and table["unit"] == "V/m" and table["source"] == brune_app.calibration_csv
    set_medium(view, "P")
    assert view.compute_source() and view.save()
    sensors = experiment(brune_app)["sensors"]
    assert set(sensors) == {"ch3"} and sensors["ch3"]["calibration"]["unit"] == "V/m"
    assert sensors["ch3"]["calibration_path"] == brune_app.calibration_csv
    # the experiment settings survive an HDF5 round trip
    path = tmp_path / "exp.h5"
    brune_app.data_manager.save_file(path)
    app2 = app_from_file(path)
    with opened(app2) as v2:
        assert v2.record["spectrum"]["amp_units"] == "m*s" and v2.record["source"]["valid"]
        assert v2.calibration_note_var.get().startswith("loaded: sensor.csv")
        assert (v2.rho_var.get(), v2.distance_var.get()) == ("2700", "0.1")
    with opened(app2, 2) as v3:
        assert v3.rho_var.get() == "2700" and v3.phase_combobox.get() == "P"
        v3.set_channel(BRUNE_CH)
        assert v3.record["spectrum"]["amp_units"] == "m*s"
    # the CSV is gone: the stored table serves
    os.remove(brune_app.calibration_csv)
    view.on_close()
    with opened(brune_app) as v2:
        assert v2.record is not None and v2.record["spectrum"]["amp_units"] == "m*s"
        assert "file not found; the stored table is used" in v2.calibration_note_var.get()
        assert "calibration CSV" not in v2.status_var.get()
        v2.set_channel(5)                                 # another sensor without a table: the controls carry over
        assert v2.record is not None and v2.record["spectrum"]["amp_units"] == "m*s"
        v2.clear_calibration()
        assert v2.calibration_path_var.get() == "" and v2.calibration_table is None
        assert v2.unit_combobox.get() == "V/m" and "needs a CSV" in v2.status_var.get()
        v2.unit_combobox.set("V/V")
        v2.schedule_compute(now=True)
        assert v2.record["spectrum"]["amp_units"] == "V*s" and v2.record["calibration"] is None


def test_source_parameters_p_and_s_phase_and_guards(view):
    # the phase sets the default radiation coefficient
    for phase, hint in (("S", "RMS for S: 0.632"), ("P", "RMS for P: 0.516")):
        view.phase_combobox.set(phase)
        view.on_phase_selected()
        assert float(view.radiation_var.get()) == pytest.approx(src.PHASE_CONSTANTS[phase]["radiation_rms"])
        assert view.radiation_hint_var.get() == hint
    fit = calibrated_compute(view)
    view.rho_var.set("")                                  # missing inputs are refused with a status message
    assert not view.compute_source() and "rho" in view.status_var.get()
    set_medium(view, "P", 0.52)
    assert view.compute_source(), view.status_var.get()
    sd = view.record["source"]
    assert sd["valid"] and sd["phase"] == "P" and sd["constants"]["wave_speed_m_s"] == VP
    m0 = sd["seismic_moment_nm"]
    assert m0 == pytest.approx(src.seismic_moment(fit.omega0, RHO, VP, DIST, 0.52), rel=1e-12)
    assert sd["mw"] == pytest.approx(src.moment_magnitude(m0))
    assert sd["source_radius_m"] == pytest.approx(0.32 * VS / fit.fc)
    assert sd["stress_drop_pa"] == pytest.approx(7 * m0 / (16 * sd["source_radius_m"] ** 3))
    assert sd["constants"]["f_plateau_hz"] == pytest.approx(fit.band_used[0]) and math.isfinite(sd["kR"])
    assert view.source_vars["m0"].get().endswith("N m")
    assert view.source_vars["mw"].get() != "n/a" and view.source_vars["warnings"].get() != "n/a"
    # S phase: the S speed, k = 0.372, kR from Vs
    set_medium(view, "S", 0.6)
    assert view.compute_source(), view.status_var.get()
    sd = view.record["source"]
    assert sd["valid"] and sd["phase"] == "S"
    assert sd["constants"]["wave_speed_m_s"] == VS and sd["constants"]["vp_m_s"] == VP
    assert sd["seismic_moment_nm"] == pytest.approx(src.seismic_moment(fit.omega0, RHO, VS, DIST, 0.6), rel=1e-12)
    assert sd["source_radius_m"] == pytest.approx(0.372 * VS / fit.fc)
    assert sd["kR"] == pytest.approx(2 * math.pi * fit.band_used[0] * DIST / VS)
    # with the medium filled in, a recompute brings the source parameters along
    set_medium(view)
    view.schedule_compute(now=True)
    assert "source parameters computed" in view.status_var.get() and view.record["source"]["valid"]
    # a malformed fit is reported, not raised
    del view.record["fit"]["fc"]
    assert not view.compute_source() and "fc" in view.status_var.get()
    view.record["fit"]["fc"] = {"bad": 1}
    assert not view.compute_source()
    assert view.record["source"]["valid"] is False and "invalid" in view.source_vars["warnings"].get()
    # an uncalibrated spectrum is refused
    view.unit_combobox.set("V/V")
    view.calibration_path_var.set("")
    assert view.compute() and not view.compute_source() and "uncalibrated" in view.status_var.get()


def test_short_pre_window_warns_the_fit_and_the_source(brune_app, view):
    """A pre window shorter than the tukey ramp attenuates Omega0: flags, status
    and taper row say so, and the source, still computed, carries the warning."""
    view.set_channel(BRUNE_CH)
    view.calibration_path_var.set(brune_app.calibration_csv)
    view.unit_combobox.set("V/m")
    view.pre_var.set("0.1"); view.post_var.set("2.5")
    assert view.compute(), view.status_var.get()
    assert view.fit.valid and view.fit.omega0 < 0.5 * OMEGA0_V / GAIN
    assert view.spectrum.meta["transient_in_flat_region"] is False
    assert any(w.startswith("transient not in the taper's flat region") for w in view.fit.warnings)
    flags = view.result_vars["flags"].get()
    assert "WARNINGS" in flags and "attenuated" in flags and "flat region" in flags
    assert "shorter than the tukey ramp" in flags                # the parameter warning too
    assert "tukey ramp" in view.record["parameter_warnings"][0]
    assert "flat region: no" in view.result_vars["taper"].get()
    status = view.status_var.get()
    assert "fit OK" not in status and "warnings" in status and "attenuated" in status
    set_medium(view, "P", 0.52)
    assert view.compute_source(), view.status_var.get()
    assert "attenuated" in view.status_var.get() and "not saved" in view.status_var.get()
    assert any("attenuated" in w for w in view.record["source"]["warnings"])
    assert "attenuated" in view.source_vars["warnings"].get() and "attenuated" in view.result_vars["flags"].get()
    assert view.save()
    saved = stored(view)["ch3"]
    assert saved["source"]["valid"] and any("attenuated" in w for w in saved["source"]["warnings"])
    assert PZTSpectrumView.parameter_warnings({"taper": "tukey", "pre_ms": 0.3, "post_ms": 2.5})
    assert PZTSpectrumView.parameter_warnings({"taper": "tukey", "pre_ms": 0.5, "post_ms": 2.0}) == []
    # a proper pre window: no warnings, source accepted
    view.pre_var.set("0.5")
    assert view.compute() and view.fit.omega0 == pytest.approx(OMEGA0_V / GAIN, rel=0.05)
    assert view.record["parameter_warnings"] == [] and view.fit.warnings == ()
    assert "fit OK" in view.status_var.get() and "WARNINGS" not in view.result_vars["flags"].get()
    assert view.compute_source(), view.status_var.get()
    assert not any("attenuated" in w for w in view.record["source"]["warnings"])
    assert "attenuated" not in view.status_var.get()


def test_typed_radiation_and_free_surface_factor_survive_a_reopen(brune_app, view):
    view.radiation_var.set("0.9")                         # a typed value survives phase switches
    for phase in ("S", "P"):
        view.phase_combobox.set(phase)
        view.on_phase_selected()
        assert view.radiation_var.get() == "0.9"
        if phase == "S":
            assert view.radiation_hint_var.get() == "RMS for S: 0.632"
    # the free-surface factor divides M0 and is kept in the constants
    fit = calibrated_compute(view)
    set_medium(view, "S", RAD_S)
    assert view.radiation_is_default()
    view.free_surface_var.set("1.7")
    assert view.compute_source(), view.status_var.get()
    sd = view.record["source"]
    m0 = src.seismic_moment(fit.omega0, RHO, VS, DIST, RAD_S)
    assert sd["seismic_moment_nm"] == pytest.approx(src.seismic_moment(fit.omega0, RHO, VS, DIST, RAD_S, 1.7), rel=1e-12)
    assert sd["seismic_moment_nm"] == pytest.approx(m0 / 1.7, rel=1e-12)
    assert sd["constants"]["free_surface_factor"] == 1.7
    assert view.save()
    view.on_close()
    with opened(brune_app) as v2:
        assert v2.phase_combobox.get() == "S" and v2.free_surface_var.get() == "1.7"
        assert v2.record["source"]["constants"]["free_surface_factor"] == 1.7
        for bad in ("0", "-2", "nan", "abc", ""):         # a status message, not a dialog
            v2.free_surface_var.set(bad)
            assert not v2.compute_source() and "free-surface factor" in v2.status_var.get()
        v2.free_surface_var.set("1.0")
        assert v2.compute_source()                        # the radiation text keeps fewer digits than RAD_S
        assert v2.record["source"]["seismic_moment_nm"] == pytest.approx(
            src.seismic_moment(fit.omega0, RHO, VS, DIST, float(v2.radiation_var.get())), rel=1e-12)
        # the saved S value was the default: a fresh channel follows the phase ...
        v2.set_channel(5)
        v2.phase_combobox.set("P")
        v2.on_phase_selected()
        assert float(v2.radiation_var.get()) == pytest.approx(src.PHASE_CONSTANTS["P"]["radiation_rms"])
        # ... but a typed value is kept across a phase switch
        v2.set_channel(BRUNE_CH)
        v2.radiation_var.set("0.45")
        v2.phase_combobox.set("P")
        v2.on_phase_selected()
        assert v2.radiation_var.get() == "0.45"


def test_save_merges_channels_and_moves_legacy_event_level_records(brune_app, view):
    assert channel_key(3) == "ch3" and parse_channel_key("ch3") == 3
    assert parse_channel_key("3") == 3 and parse_channel_key("CH12") == 12
    assert parse_channel_key("chx") is None and parse_channel_key("source") is None
    # files written before 2026-10-12 keep their records at event['pzt_spectrum'], legacy digit keys included;
    # they are read and move under the record, renamed, on the next save
    calibrated_compute(view)
    rec = copy.deepcopy(view.record)
    view.event["pzt_spectrum"] = {"version": 1, "channels": {"5": dict(rec, channel=5), "ch7": dict(rec, channel=7)}}
    assert set(view.saved_channels()) == {5, 7}
    for channel in (5, 7):
        view.set_channel(channel)
        assert view.record is not None and "saved record" in view.status_var.get()
    view.set_channel(BRUNE_CH)
    assert view.compute() and view.save()
    channels = stored(view)
    assert set(channels) == {"ch3", "ch5", "ch7"} and "pzt_spectrum" not in view.event
    assert "under strain" in view.status_var.get()
    # the saved record is compared with a recomputation
    view.set_channel(TANH_CH)
    view.set_channel(BRUNE_CH)
    assert "matches the saved record" in view.status_var.get()
    view.post_var.set("1.5")
    view.schedule_compute(now=True)
    assert "differs from the saved record" in view.status_var.get()
    view.post_var.set("2.0")
    # a saved record carries every setting and is JSON-like
    set_medium(view, "P")
    assert view.compute_source() and view.save()
    saved = brune_app.data_manager.get_data(f"runs/[0]/events/[{EVENT}]/strain/spectrum")
    assert saved is view.event["strain"]["spectrum"]
    assert saved["version"] == 2 and list(saved["sets"]) == ["default"] and "channels" not in saved
    rec = saved["sets"]["default"]["channels"]["ch3"]
    for key in ("version", "channel", "trigger", "trigger_time_s", "fs_hz", "pre_ms", "post_ms", "taper",
                "baseline", "highpass_hz", "remove_step", "calibration_path", "calibration_unit",
                "calibration_db", "calibration", "t_star_s", "fmin_hz", "fmax_hz", "bins_per_decade", "snr_min",
                "n_fixed", "n_max", "loss", "parameter_warnings", "source_inputs", "spectrum", "fit", "source"):
        assert key in rec, key
    assert rec["result_set"] == "default" and rec["n_max"] == 4.0 and rec["fit"]["n_bounds"] == [1.0, 4.0]
    assert rec["spectrum"]["meta"]["noise_region"] == "pre" and rec["spectrum"]["amp_units"] == "m*s"
    assert rec["channel"] == BRUNE_CH and rec["fit"]["valid"] and rec["source"]["valid"]
    assert isinstance(rec["fit"]["warnings"], list) and isinstance(rec["fit"]["band_used"], list)
    json.dumps(saved)  # JSON-like (NaN allowed)
    # a second channel is merged, the first kept
    view.set_channel(TANH_CH)
    assert view.record is not None and view.record["channel"] == TANH_CH      # computed on selection
    view.remove_step_var.set(True)
    view.unit_combobox.set("V/V")
    view.calibration_path_var.set("")
    assert view.compute() and view.save()
    channels = stored(view)
    assert set(channels) == {"ch0", "ch3", "ch5", "ch7"} and channels["ch3"]["source"]["valid"]
    assert channels["ch0"]["source"] is None and channels["ch0"]["remove_step"] is True
    # nothing computable (a bad control): Save is refused
    view.set_channel(9)
    view.pre_var.set("abc")
    view.schedule_compute(now=True)
    assert view.record is None and view.result_vars["fc"].get() == "n/a"
    assert not view.save() and "nothing to save" in view.status_var.get()


def test_partial_and_list_shaped_legacy_records_reopen(brune_app, view):
    fit = calibrated_compute(view)
    rec = view.record
    # a hand-edited record without 'channel' is filed under the channel it was found at
    view.event["pzt_spectrum"] = {"version": RESULT_VERSION, "channels": {"ch2": {"pre_ms": 1.0}}}
    view.on_close()
    with opened(brune_app) as v2:
        assert v2.current_channel() == 2 and "saved record" in v2.status_var.get()   # recomputed from its parameters
        assert v2.pre_var.get() == "1" and v2.record["channel"] == 2
        assert v2.save()
        assert set(stored(v2)) == {"ch2"} and stored(v2)["ch2"]["channel"] == 2
        assert "channel 2 saved" in v2.status_var.get()
        v2.record = {"pre_ms": 1.0}                       # no channel and no selection: refused
        v2.channel_combobox.set("")
        assert not v2.save() and "no channel" in v2.status_var.get()
    # a legacy digit-keyed map comes back from an older HDF5 file as a LIST: entries are channels 0..n-1
    brune_app.data_manager.get_data(f"runs/[0]/events/[{EVENT}]")["pzt_spectrum"] = {
        "version": RESULT_VERSION, "channels": [dict(rec, channel=0), dict(rec, channel=1), "junk"]}
    with opened(brune_app) as v3:
        assert set(v3.saved_channels()) == {0, 1, 2} and v3.current_channel() == 0
        # computed on channel 3, recomputed on channel 0 it no longer matches
        assert "differs from the saved record" in v3.status_var.get()
        assert v3.result_vars["fc"].get() != f"{fit.fc:.4g} Hz"
        v3.set_channel(1)
        assert "saved record" in v3.status_var.get()
        calibrated_compute(v3)
        assert v3.save()
        channels = stored(v3)
        assert set(channels) == {"ch0", "ch1", "ch2", "ch3"} and channels["ch1"]["channel"] == 1   # kept and renamed


def test_array_valued_and_malformed_saved_records_reopen(brune_app, view, monkeypatch):
    calibrated_compute(view)
    set_medium(view, "P")
    assert view.compute_source() and view.save()
    fit_fc = view.fit.fc
    good = copy.deepcopy(stored(view)["ch3"])
    channels = stored(view)
    # a loader that hands back numpy arrays for stored lists (truth-testing an array raises)
    rec = channels["ch3"]
    rec["fit"]["band_used"] = np.asarray(rec["fit"]["band_used"])
    rec["fit"]["warnings"] = np.asarray(["band_limited: synthetic"])
    for key in ("f_binned", "amp_binned", "noise_binned", "snr", "counts"):
        rec["spectrum"][key] = np.asarray(rec["spectrum"][key])
    rec["source"]["warnings"] = np.asarray(["kR synthetic warning"])
    rec["parameter_warnings"] = np.asarray([], dtype=object)
    # fit stored as a string, at_bounds not a dict, band garbage, source constants odd
    channels["ch2"] = dict(copy.deepcopy(good), channel=2, fit="garbage",
                           source={"valid": True, "constants": "x", "warnings": 3})
    channels["ch4"] = dict(good, channel=4, pre_ms="abc",
                           fit=dict(good["fit"], at_bounds="oops", band_used="abc", omega0="?", warnings=None),
                           spectrum=dict(good["spectrum"], f_binned="nope", meta="meta"))
    view.on_close()
    with opened(brune_app) as v2:
        assert v2.current_channel() == 2
        assert v2.result_vars["fc"].get() != "n/a" and v2.pre_var.get() == "0.5"   # parameters applied, recomputed
        v2.set_channel(4)
        assert "saved record" in v2.status_var.get()     # unreadable: reported, nothing recomputed
        assert v2.result_vars["omega0"].get() == "n/a" and v2.result_vars["band"].get() == "n/a"
        assert v2.pre_var.get() == "abc"
        assert not v2.compute() and "pre window" in v2.status_var.get()
        assert not v2.compute_source()
        # the array-valued record is applied and compared without raising; the recomputation agrees
        v2.set_channel(BRUNE_CH)
        assert "matches the saved record" in v2.status_var.get()
        assert v2.result_vars["fc"].get() == f"{fit_fc:.4g} Hz" and v2.result_vars["band"].get().endswith("Hz")
        set_medium(v2, "P")
        v2.schedule_compute(now=True)
        assert "source parameters computed" in v2.status_var.get() and v2.source_vars["m0"].get().endswith("N m")
        assert v2.compute_source(), v2.status_var.get()   # band_used array -> f_plateau
        # an exception while applying a record lands in the status line, not in the caller
        monkeypatch.setattr(PZTSpectrumView, "apply_record", lambda self, rec: (_ for _ in ()).throw(KeyError("boom")))
        v2.set_channel(2)
        assert "saved record unreadable" in v2.status_var.get() and "boom" in v2.status_var.get()
        assert v2.record is None and v2.result_vars["fc"].get() == "n/a"
        assert not v2.save()
    monkeypatch.undo()
    with opened(brune_app):                               # opening again with the broken records works too
        pass


def test_reopen_restores_parameters_and_numbers(brune_app, view):
    view.set_channel(BRUNE_CH)
    view.pre_var.set("0.4"); view.post_var.set("1.5")
    view.calibration_path_var.set(brune_app.calibration_csv)
    view.unit_combobox.set("V/m")
    view.taper_combobox.set("bh4")          # a bell taper: the pulse at 21% of the window is attenuated
    assert view.compute() and "flat region: no" in view.result_vars["taper"].get()
    set_medium(view, "P", 0.6)
    assert view.compute_source() and "attenuated" in view.status_var.get()   # computed, with the warning
    view.taper_combobox.set("tukey")        # ramp 0.24 ms < pre 0.4 ms: flat region
    view.snr_var.set("2"); view.bins_var.set("25"); view.fmin_var.set("1500"); view.t_star_var.set("1e-6")
    assert view.compute(), view.status_var.get()
    fit_saved = view.fit
    assert view.compute_source() and view.save()
    rec_saved = stored(view)["ch3"]
    view.on_close()
    with opened(brune_app) as v2:
        assert v2.current_channel() == BRUNE_CH and v2.fit is not None and v2.window is not None
        assert v2.record["fit"]["fc"] == pytest.approx(fit_saved.fc)
        assert (v2.pre_var.get(), v2.post_var.get(), v2.taper_combobox.get(), v2.snr_var.get()) == ("0.4", "1.5", "tukey", "2")
        assert float(v2.t_star_var.get()) == pytest.approx(1e-6)
        assert v2.calibration_path_var.get() == brune_app.calibration_csv and v2.unit_combobox.get() == "V/m"
        assert (v2.rho_var.get(), v2.c_var.get(), v2.vs_var.get(), v2.distance_var.get()) == ("2700", "6000", "3500", "0.1")
        assert float(v2.radiation_var.get()) == 0.6
        assert v2.result_vars["fc"].get() == f"{fit_saved.fc:.4g} Hz" and v2.source_vars["m0"].get().endswith("N m")
        assert "saved record" in v2.status_var.get()
        # every parameter the controls produce equals the saved record's
        p = v2.read_parameters()
        assert all(p[k] == rec_saved[k] for k in p), {k: (p[k], rec_saved[k]) for k in p if p[k] != rec_saved[k]}
        # an unsaved channel is computed with the same controls; switching back restores
        v2.set_channel(5)
        assert v2.record["channel"] == 5 and "not saved" in v2.status_var.get() and v2.pre_var.get() == "0.4"
        v2.set_channel(BRUNE_CH)
        assert v2.record["fit"]["fc"] == pytest.approx(fit_saved.fc)
        assert v2.compute() and v2.fit.fc == pytest.approx(fit_saved.fc)    # recomputed with the restored parameters
        assert v2.window.n_pre == round(0.4e-3 * brune_app.fs) and v2.spectrum.t_star_s == 1e-6 and v2.fit.snr_min == 2.0
        meta = v2.spectrum.meta
        assert meta["bins_per_decade"] == 25 and meta["fmin"] == 1500.0 and meta["taper"] == "tukey"
        # switching event keeps the channel; event 2 has no saved record
        v2.set_event(2)
        assert v2.current_channel() == BRUNE_CH and v2.record is not None
        assert "saved record" not in v2.status_var.get() and v2.title() == "PZT Spectrum - Event 2"


def test_every_control_reopens_from_the_experiment_settings(brune_app, view):
    view.set_channel(BRUNE_CH)
    view.pre_var.set("0.4"); view.post_var.set("1.5"); view.snr_var.set("2"); view.bins_var.set("25")
    view.fmin_var.set("1500"); view.t_star_var.set("1e-6"); view.taper_combobox.set("tukey")
    view.calibration_path_var.set(brune_app.calibration_csv); view.unit_combobox.set("V/m")
    view.n_fixed_var.set(False); view.n_max_var.set("8")
    assert view.compute(), view.status_var.get()
    set_medium(view, "S", 0.7)
    view.free_surface_var.set("1.3")
    assert view.save()                                    # the source was never computed, still everything is kept
    a = brune_app.data_manager.data["analysis"]
    assert a["medium"] == {"rho_kg_m3": RHO, "vp_m_s": VP, "vs_m_s": VS}
    settings = experiment(brune_app)["settings"]
    assert settings["pre_ms"] == 0.4 and settings["phase"] == "S" and settings["radiation_coefficient"] == 0.7
    assert settings["snr_min"] == 2.0 and settings.get("t_star_s") is None and settings["n_max"] == 8.0
    sensor = experiment(brune_app)["sensors"]["ch3"]
    assert sensor["distance_m"] == DIST and sensor["free_surface_factor"] == 1.3 and sensor["t_star_s"] == 1e-6
    assert view.record["source_inputs"]["free_surface_factor"] == 1.3
    rec = copy.deepcopy(view.record)
    view.on_close()
    with opened(brune_app, 2) as v2:                      # another event, nothing saved there
        assert "saved record" not in v2.status_var.get()
        assert (v2.pre_var.get(), v2.post_var.get(), v2.snr_var.get(), v2.bins_var.get()) == ("0.4", "1.5", "2", "25")
        assert v2.fmin_var.get() == "1500" and v2.taper_combobox.get() == "tukey"
        assert v2.n_max_var.get() == "8" and not v2.n_fixed_var.get()
        assert (v2.rho_var.get(), v2.c_var.get(), v2.vs_var.get()) == ("2700", "6000", "3500")
        assert v2.phase_combobox.get() == "S" and v2.radiation_var.get() == "0.7"
        assert v2.current_channel() == 0                   # no record anywhere: channel 0, whose sensor has no entry
        v2.set_channel(BRUNE_CH)                           # this sensor's own values
        assert (v2.distance_var.get(), v2.free_surface_var.get()) == ("0.1", "1.3")
        assert float(v2.t_star_var.get()) == pytest.approx(1e-6) and v2.unit_combobox.get() == "V/m"
        assert v2.calibration_note_var.get().startswith("loaded: sensor.csv")
        assert v2.record is not None and v2.record["spectrum"]["amp_units"] == "m*s"
        rec.pop("n_max")                                   # a record written before the key existed reads as the default
        v2.apply_record(rec)
        assert v2.n_max_var.get() == "4"


def test_event_without_strain_and_the_waveform_record_trigger(app):
    """A waveform record from an Elsys trigger block carries the recorder's own
    trigger time; the view triggers on it by default because the event_time is a
    pick on the slow mechanical channels, tens of ms later."""
    from labquake_explorer.data.channels import channel_array
    with opened(app) as v:
        assert list(v.channel_combobox["values"]) == [] and "no full-rate record" in v.status_var.get()
        assert not v.compute()
    run = app.data_manager.get_data("runs/[0]")
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    fs = 200e3
    t_trig = float(event["event_time"]) - 0.02
    time = t_trig - 0.1 + np.arange(int(0.6 * fs)) / fs
    raw = np.random.default_rng(3).normal(0.0, 1e-4, (2, time.size))
    raw[0] += 0.5 * np.exp(-((time - t_trig - 2e-4) / 1e-4) ** 2) * np.sin(2 * np.pi * 50e3 * (time - t_trig))
    event["waveform"] = {"elsys": channel_array(raw, ["pzt_1", "pzt_2"], unit="V", time=time, sample_rate=fs,
                                                block=4, trigger_time=t_trig, filename="run.tpc5")}
    with opened(app) as v:
        assert v.record_key == "waveform/elsys" and v.trigger_combobox.get() == TRIGGER_RECORD
        assert v.record_trigger_time() == pytest.approx(t_trig)
        assert v.compute(), v.status_var.get()
        assert v.window.trigger_time == pytest.approx(t_trig)
        assert v.record["trigger"] == TRIGGER_RECORD and v.record["trigger_time_s"] == pytest.approx(t_trig)
        # an older record without the stored time is looked up in the run's block table
        del event["waveform"]["elsys"]["trigger_time"]
        run["elsys"] = {"format": "tpc5", "blocks": {"block": [2, 3, 4], "trigger_time_run": [1.0, 2.0, t_trig]}}
        assert v.record_trigger_time() == pytest.approx(t_trig)
        del event["waveform"]["elsys"]["block"]           # no trigger of its own: reported, not silent
        assert not v.compute() and "no trigger time" in v.status_var.get()


# ------------------------------------------------------- window dragging
def test_noise_entries_and_dragged_windows_before_the_signal(brune_app, view):
    view.set_channel(BRUNE_CH)
    w0 = view.window
    assert w0.i_noise1 == w0.i_trigger - w0.n_pre                     # default: just before the signal
    assert view.record["noise_start_ms"] is None and view.record["noise_end_ms"] is None
    e0 = view.current_edges_ms()
    assert (e0["signal_start"], e0["signal_end"]) == (pytest.approx(-0.5), pytest.approx(2.0))
    assert (e0["noise_start"], e0["noise_end"]) == (pytest.approx(-3.0), pytest.approx(-0.5))
    assert set(view.handles) == {"noise_start", "noise_end", "signal_start", "signal_end"}
    # typed noise entries reach the analysis, the record, and come back with the saved record
    view.noise_start_var.set("-4"); view.noise_end_var.set("-2.5")
    view.schedule_compute(now=True)
    w = view.window
    assert w.i_noise0 == w.i_trigger - int(round(4e-3 * w.fs)) and w.i_noise1 == w.i_trigger - int(round(2.5e-3 * w.fs))
    assert view.record["noise_start_ms"] == -4.0 and view.record["noise_end_ms"] == -2.5
    assert view.save()
    view.set_channel(TANH_CH)
    view.set_channel(BRUNE_CH)
    assert (view.noise_start_var.get(), view.noise_end_var.get()) == ("-4", "-2.5")
    view.noise_end_var.set("0.1")                                     # past the signal start
    view.schedule_compute(now=True)
    assert "noise window" in view.status_var.get()
    view.noise_end_var.set(""); view.noise_start_var.set("")
    view.schedule_compute(now=True)
    assert view.record["noise_start_ms"] is None
    # the post edge
    view.begin_drag("signal_end", e0["signal_end"])
    view.drag_to(2.5)
    assert view.post_var.get() == "2.5" and view.handles["signal_end"].get_xdata()[0] == pytest.approx(2.5)
    view.end_drag()
    assert view.record["post_ms"] == 2.5 and view.window.n_post == int(round(2.5e-3 * brune_app.fs))
    # the pre edge cannot cross the trigger; the automatic noise window (blank entries) follows it
    view.begin_drag("signal_start", -0.5)
    assert view.drag_to(0.3)["signal_start"] == 0.0 and view.pre_var.get() == "0"
    e = view.drag_to(-1.0)
    assert e["signal_start"] == pytest.approx(-1.0) and e["noise_end"] == pytest.approx(-1.0)
    assert (view.noise_start_var.get(), view.noise_end_var.get()) == ("", "")
    view.end_drag()
    assert view.record["pre_ms"] == 1.0 and view.record["noise_start_ms"] is None
    assert view.window.i_noise1 == view.window.i_trigger - view.window.n_pre
    # the whole signal window slides, keeping its length, and stops at the trigger
    e1 = view.current_edges_ms()
    view.begin_drag("signal", 0.5)
    e = view.drag_to(0.25)
    assert e["signal_end"] - e["signal_start"] == pytest.approx(e1["signal_end"] - e1["signal_start"])
    assert e["signal_start"] == pytest.approx(e1["signal_start"] - 0.25)
    assert view.drag_to(5.5)["signal_start"] == 0.0
    view.end_drag()
    assert view.record["pre_ms"] == 0.0 and view.record["post_ms"] == pytest.approx(e1["signal_end"] - e1["signal_start"])
    # a dragged noise edge makes the noise explicit; its end never passes the signal start
    view.pre_var.set("0.5"); view.post_var.set("2.0")
    view.schedule_compute(now=True)
    e2 = view.current_edges_ms()
    view.begin_drag("noise_end", e2["noise_end"])
    assert view.drag_to(1.0)["noise_end"] == pytest.approx(e2["signal_start"])
    view.drag_to(-1.0)
    view.end_drag()
    assert view.record["noise_end_ms"] == -1.0 and view.window.i_noise1 == view.window.i_trigger - int(round(1e-3 * brune_app.fs))
    assert view.record["noise_start_ms"] == pytest.approx(e2["noise_start"])      # the start stayed put
    # explicit noise: the signal start now stops at it instead of pushing it
    e = drag(view, "signal_start", -0.5, -3.0)
    assert e["signal_start"] == pytest.approx(-1.0) and e["noise_end"] == pytest.approx(-1.0)
    e3 = view.current_edges_ms()
    e = drag(view, "noise", e3["noise_start"] + 0.5, e3["noise_start"] - 0.5)
    assert e["noise_end"] - e["noise_start"] == pytest.approx(e3["noise_end"] - e3["noise_start"])
    assert e["noise_start"] == pytest.approx(e3["noise_start"] - 1.0)
    assert view.record["noise_start_ms"] == pytest.approx(e3["noise_start"] - 1.0)
    # a click without a move changes nothing
    post = view.record["post_ms"]
    drag(view, "signal_end", view.current_edges_ms()["signal_end"])
    assert view.record["post_ms"] == post
    # a noise window too short to use (under a quarter of the signal window) is not written back by a drag
    view.noise_start_var.set("-1.0"); view.noise_end_var.set("-0.6"); view.pre_var.set("0.5"); view.post_var.set("10")
    view.schedule_compute(now=True)
    assert view.window is not None and not view.window.noise_available
    assert set(view.handles) == {"signal_start", "signal_end"}
    drag(view, "signal_start", view.current_edges_ms()["signal_start"], 0.0)
    assert (view.noise_start_var.get(), view.noise_end_var.get()) == ("-1.0", "-0.6")   # left as typed
    assert view.pre_var.get() == "0" and view.record is not None
    # a recompute that fails keeps the top panel and its drags
    view.noise_end_var.set("-1.0")
    view.schedule_compute(now=True)
    assert view.record is None and "noise start must be before noise end" in view.status_var.get()
    assert view.window is not None and view.ax_time.lines and "not recomputed" in view.ax_spec.get_title()
    drag(view, "signal_end", view.current_edges_ms()["signal_end"], 5.0)
    assert view.post_var.get() == "5" and view.record is None      # the noise entries are still wrong
    view.noise_end_var.set("-0.6")
    view.schedule_compute(now=True)
    assert view.record is not None and view.record["post_ms"] == 5.0


def test_mouse_presses_grab_edges_bodies_and_coincident_edges(brune_app, view):
    view.set_channel(BRUNE_CH)
    view.canvas.draw()
    e0 = view.current_edges_ms()
    assert e0["noise_end"] == e0["signal_start"]          # automatic noise: it ends where the signal starts
    view.on_press(mouse(view, "button_press_event", e0["signal_start"]))
    assert view._drag["what"] == "signal_start"           # the noise follows anyway
    view.on_release(mouse(view, "button_release_event", e0["signal_start"]))
    press = mouse(view, "button_press_event", e0["signal_end"], dx_px=3)     # a few pixels off still grabs the edge
    assert press.inaxes is view.ax_time                   # not the taper axes on top of it
    view.on_press(press)
    assert view._drag["what"] == "signal_end"
    view.on_motion(mouse(view, "motion_notify_event", e0["signal_end"] + 1.0))
    assert float(view.post_var.get()) == pytest.approx(3.0, abs=0.02)      # matplotlib keeps whole pixels
    view.on_release(mouse(view, "button_release_event", e0["signal_end"] + 1.0))
    assert view._drag is None and view.record["post_ms"] == pytest.approx(3.0, abs=0.02)
    e1 = view.current_edges_ms()
    for x_ms, dx, what in ((e1["noise_start"], -3, "noise_start"),
                           (0.5 * (e1["signal_start"] + e1["signal_end"]), 0, "signal"),
                           (0.5 * (e1["noise_start"] + e1["noise_end"]), 0, "noise")):
        view.on_press(mouse(view, "button_press_event", x_ms, dx_px=dx))
        assert view._drag["what"] == what
        view.on_release(mouse(view, "button_release_event", x_ms))
    view.toolbar.zoom()                                   # with the toolbar's zoom active nothing is grabbed
    view.on_press(mouse(view, "button_press_event", e1["signal_end"]))
    assert view._drag is None
    view.toolbar.zoom()
    # an explicit noise window glued to the signal: the drag direction tells the edges apart
    view.post_var.set("2.0"); view.noise_start_var.set("-3"); view.noise_end_var.set("-0.5")
    view.schedule_compute(now=True)
    for target, moved, kept in ((-1.0, "noise_end", "signal_start"), (-0.2, "signal_start", "noise_end")):
        view.noise_end_var.set("-0.5")
        view.schedule_compute(now=True)
        view.canvas.draw()
        view.on_press(mouse(view, "button_press_event", -0.5))
        assert view._drag["what"] == TIE
        e = view.drag_to(target)                          # left: the noise end moves; right: the signal start
        assert e[moved] == pytest.approx(target) and e[kept] == pytest.approx(-0.5)
        view.end_drag()
    assert view.record["pre_ms"] == pytest.approx(0.2) and view.record["noise_end_ms"] == -0.5


def test_noise_window_after_the_signal_in_the_view(brune_app, view):
    view.set_channel(BRUNE_CH)
    # explicit, after the signal: entries, analysis, record, handles
    view.noise_start_var.set("2"); view.noise_end_var.set("4.5")
    view.schedule_compute(now=True)
    w = view.window
    assert w.noise_available and w.noise_region == "explicit" and w.i_noise0 == w.i_trigger + w.n_post
    assert view.fit.valid, view.fit.reason
    assert view.record["noise_start_ms"] == 2.0 and view.record["spectrum"]["meta"]["noise_region"] == "explicit"
    e = view.current_edges_ms()
    assert e["noise_start"] == pytest.approx(2.0) and e["noise_end"] == pytest.approx(4.5)
    assert set(view.handles) == set(HANDLES) and view.ax_time.get_xlim()[1] > 4.5
    # the signal end stops at the noise start, the noise start at the signal end; the noise end is free
    view.begin_drag("signal_end", e["signal_end"])
    assert view.drag_to(3.0)["signal_end"] == pytest.approx(2.0)
    view.end_drag()
    e = view.current_edges_ms()
    view.begin_drag("noise_start", e["noise_start"])
    assert view.drag_to(0.5)["noise_start"] == pytest.approx(e["signal_end"])
    view.drag_to(2.5)
    view.end_drag()
    assert view.record["noise_start_ms"] == 2.5 and view.record["noise_end_ms"] == 4.5
    drag(view, "noise_end", view.current_edges_ms()["noise_end"], 6.0)
    assert view.record["noise_end_ms"] == 6.0
    # the noise body slides after the signal, never onto it
    e = view.current_edges_ms()
    e1 = drag(view, "noise", e["noise_start"] + 1.0, e["noise_start"] - 4.0)
    assert e1["noise_start"] == pytest.approx(e["signal_end"])
    assert e1["noise_end"] - e1["noise_start"] == pytest.approx(e["noise_end"] - e["noise_start"])
    assert view.record["noise_start_ms"] == pytest.approx(e["signal_end"])
    # coincident edges after the signal: right moves the noise start, left the signal end
    end = view.current_edges_ms()["signal_end"]
    assert view.current_edges_ms()["noise_start"] == pytest.approx(end)
    for target, moved, kept in ((end + 1.0, "noise_start", "signal_end"), (end - 0.5, "signal_end", "noise_start")):
        view.noise_start_var.set(str(end))
        view.schedule_compute(now=True)
        view.canvas.draw()
        view.on_press(mouse(view, "button_press_event", end))
        assert view._drag["what"] == TIE_AFTER
        e1 = view.drag_to(target)
        assert e1[moved] == pytest.approx(target) and e1[kept] == pytest.approx(end)
        view.end_drag()
    assert view.record["post_ms"] == pytest.approx(end - 0.5)
    # automatic noise with no room before the signal is taken after it, says so, and follows the signal end
    view.noise_start_var.set(""); view.noise_end_var.set("")
    view.pre_var.set("9.5"); view.post_var.set("0.5")                # the trace starts 10 ms before the trigger
    view.schedule_compute(now=True)
    w = view.window
    assert w.noise_region == "post" and w.i_noise0 == w.i_trigger + w.n_post
    assert "after the signal" in view.status_var.get() and view.record["noise_start_ms"] is None
    view.begin_drag("signal_end", view.current_edges_ms()["signal_end"])
    e1 = view.drag_to(1.0)
    assert e1["signal_end"] == pytest.approx(1.0) and e1["noise_start"] == pytest.approx(1.0)
    assert (view.noise_start_var.get(), view.noise_end_var.get()) == ("", "")
    view.end_drag()
    assert view.window.i_noise0 == view.window.i_trigger + view.window.n_post and view.record["noise_start_ms"] is None
    # with room before again, the noise goes back before the signal
    view.pre_var.set("0.5"); view.post_var.set("2.0")
    view.schedule_compute(now=True)
    assert view.window.noise_region == "pre" and "after the signal" not in view.status_var.get()


def test_real_mouse_drags_and_scrolling_control_columns(brune_app, view):
    view.set_channel(BRUNE_CH)
    shown(view)
    e = view.current_edges_ms()
    tk_drag(view, e["signal_end"], 3.0)
    assert float(view.post_var.get()) == pytest.approx(3.0, abs=0.05)
    e = view.current_edges_ms()                            # the automatic noise window followed
    assert e["noise_start"] == pytest.approx(-4.0, abs=0.05)
    tk_drag(view, e["noise_start"], -5.0)                  # the noise becomes explicit
    assert float(view.noise_start_var.get()) == pytest.approx(-5.0, abs=0.05)
    tk_drag(view, view.current_edges_ms()["noise_end"], -1.5)
    assert float(view.noise_end_var.get()) == pytest.approx(-1.5, abs=0.05)
    tk_drag(view, view.current_edges_ms()["signal_start"], -1.0)
    assert float(view.pre_var.get()) == pytest.approx(1.0, abs=0.05)
    assert view.record["pre_ms"] == pytest.approx(1.0, abs=0.05) and view.record["noise_end_ms"] == pytest.approx(-1.5, abs=0.05)
    # the end of a noise window after the signal
    view.pre_var.set("0.5"); view.post_var.set("2.0"); view.noise_start_var.set("2"); view.noise_end_var.set("4.5")
    view.schedule_compute(now=True)
    view.canvas.draw()
    view.update()
    tk_drag(view, view.current_edges_ms()["noise_end"], 6.0)
    assert float(view.noise_end_var.get()) == pytest.approx(6.0, abs=0.05)
    assert view.record["noise_end_ms"] == pytest.approx(6.0, abs=0.05) and view.record["noise_start_ms"] == 2.0
    # the control columns scroll in a short window
    for panel in (view.left_panel, view.middle_panel):
        assert panel.inner.winfo_children()
    view.geometry("1400x400")
    view.update()
    assert view.left_panel.overflows() and view.left_panel.scrollbar.winfo_ismapped()
    view.left_panel.scroll(5)
    assert view.left_panel.canvas.yview()[0] > 0.0
    tall = view.left_panel.content_height() + 150
    if tall < view.winfo_screenheight():                  # the column is taller than many screens
        view.geometry(f"1400x{tall}")
        view.update()
        assert not view.left_panel.scrollbar.winfo_ismapped()


# ------------------------------------------------------------- result sets, ensemble, Q
def test_result_sets_keep_routes_side_by_side(brune_app, view):
    """Records in different result sets do not overwrite each other; each set keeps
    its own experiment settings; a window opens on the set last saved."""
    calibrated_compute(view)                                  # V/m: the 'default' set
    assert view.result_set == "default" and list(view.set_combobox["values"]) == ["default"]
    set_medium(view, "P")
    assert view.compute_source() and view.save()
    fc_default = view.fit.fc
    assert view.record["result_set"] == "default"
    # a new set typed into the box: the same channel through another calibration
    view.set_combobox.set("EGF")
    view.on_set_changed()
    assert view.result_set == "EGF" and list(view.set_combobox["values"]) == ["default", "EGF"]
    assert "not saved" in view.status_var.get()               # nothing saved in EGF yet: recomputed
    view.unit_combobox.set("V/(N m)")
    view.calibration_divisor_var.set("2")
    assert view.compute(), view.status_var.get()
    assert view.spectrum.amp_units == "N*m*s"
    assert view.compute_source(), view.status_var.get()
    assert view.save() and "set 'EGF'" in view.status_var.get()
    store = view.event["strain"]["spectrum"]
    assert store["version"] == 2 and set(store["sets"]) == {"default", "EGF"}
    assert stored(view)["ch3"]["calibration_unit"] == "V/m"
    assert stored(view, "EGF")["ch3"]["calibration_unit"] == "V/(N m)" and stored(view, "EGF")["ch3"]["result_set"] == "EGF"
    a = brune_app.data_manager.data["analysis"]["pzt_spectrum"]
    assert a["current_set"] == "EGF" and set(a["sets"]) == {"default", "EGF"} and "settings" not in a and "sensors" not in a
    assert a["sets"]["default"]["sensors"]["ch3"]["calibration_unit"] == "V/m"
    assert a["sets"]["EGF"]["sensors"]["ch3"]["calibration_unit"] == "V/(N m)"
    assert a["sets"]["EGF"]["sensors"]["ch3"]["calibration_divisor"] == 2.0
    # back to default: its record and its calibration come back
    view.set_combobox.set("default")
    view.on_set_changed()
    assert view.unit_combobox.get() == "V/m" and view.fit.fc == pytest.approx(fc_default)
    assert "matches the saved record" in view.status_var.get()
    # a bad name is refused and the box reverts
    view.set_combobox.set("a/b")
    view.on_set_changed()
    assert view.result_set == "default" and view.set_combobox.get() == "default" and "'/'" in view.status_var.get()
    view.on_close()
    with opened(brune_app, 2) as v2:                          # opens on the set last saved, its sensor values too
        assert v2.result_set == "EGF" and v2.set_combobox.get() == "EGF"
        assert list(v2.set_combobox["values"]) == ["default", "EGF"]
        v2.set_channel(BRUNE_CH)
        assert v2.unit_combobox.get() == "V/(N m)" and v2.calibration_divisor_var.get() == "2"
        v2.set_combobox.set("default")
        v2.on_set_changed()
        assert v2.unit_combobox.get() == "V/m" and v2.calibration_divisor_var.get() == ""


def test_version_1_store_and_flat_settings_are_read_as_the_default_set(brune_app, view):
    calibrated_compute(view)
    rec = copy.deepcopy(view.record)
    view.event["strain"]["spectrum"] = {"version": 1, "channels": {"ch3": dict(rec, channel=3), "ch5": dict(rec, channel=5)}}
    data = brune_app.data_manager.data
    data["analysis"] = {"pzt_spectrum": {"settings": {"pre_ms": 0.7, "post_ms": 1.8},
                                         "sensors": {"ch6": {"t_star_s": 2e-6}}}}
    view.on_close()
    with opened(brune_app) as v2:
        assert v2.result_set == "default" and set(v2.saved_channels()) == {3, 5}
        assert v2.current_channel() == 3 and "saved record" in v2.status_var.get()
        assert v2.experiment_settings()["pre_ms"] == 0.7                     # the flat settings
        v2.set_channel(6)
        assert float(v2.t_star_var.get()) == pytest.approx(2e-6)             # the flat sensors entry
        v2.set_channel(3)
        assert v2.save()
        store = v2.event["strain"]["spectrum"]
        assert store["version"] == 2 and "channels" not in store and set(stored(v2)) == {"ch3", "ch5"}
        a = data["analysis"]["pzt_spectrum"]
        assert "settings" not in a and "sensors" not in a and a["current_set"] == "default"
        assert a["sets"]["default"]["sensors"]["ch6"] == {"t_star_s": 2e-6}  # moved
        assert a["sets"]["default"]["sensors"]["ch3"]["calibration_unit"] == "V/m"
        assert a["sets"]["default"]["settings"]["pre_ms"] == 0.5             # the saved record's controls


def test_estimator_and_ensemble_in_the_view(brune_app, view, tmp_path):
    alt = [write_calibration(tmp_path / "alt1.csv", 1.1 * GAIN), write_calibration(tmp_path / "alt2.csv", 0.9 * GAIN)]
    calibrated_compute(view)
    assert view.estimator_combobox.get() == "rms" and view.record["estimator"] == "rms"
    assert view.record["ensemble"] is None and not view.record["ensemble_on"]
    assert view.record["ensemble_fmins_hz"] == list(se.DEFAULT_FMINS_HZ)
    assert view.record["ensemble_fmaxs_hz"] == list(se.DEFAULT_FMAXS_HZ)
    assert view.result_vars["ensemble"].get() == "n/a" and view.ensemble_text_var.get() == "not run"
    # the estimator reaches the analysis and the record
    view.estimator_combobox.set("logmean")
    view.schedule_compute(now=True)
    assert view.spectrum.meta["estimator"] == "logmean" and view.record["estimator"] == "logmean"
    assert view.record["spectrum"]["meta"]["estimator"] == "logmean"
    fc_logmean = view.fit.fc
    assert fc_logmean == pytest.approx(FC, rel=0.1)
    # the ensemble: a smaller grid and two alternative tables, run by hand
    view.ensemble_fmins_var.set("1000, 2000")
    view.ensemble_fmaxs_var.set("20000, 50000")
    view.ensemble_calibrations_var.set(";".join(alt))
    view.schedule_compute(now=True)
    set_medium(view, "P")
    assert view.compute_source()
    assert view.compute_ensemble(), view.status_var.get()
    e = view.record["ensemble"]
    assert e["n_members"] == 12 and e["n_valid"] == 12
    assert e["bands_hz"] == [[1000.0, 20000.0], [1000.0, 50000.0], [2000.0, 20000.0], [2000.0, 50000.0]]
    assert e["calibrations"][0] == brune_app.calibration_csv and e["calibrations"][1:] == alt
    st = e["stats"]
    assert st["fc"]["p16"] <= fc_logmean * 1.05 and st["fc"]["p84"] >= fc_logmean * 0.95
    m0 = view.record["source"]["seismic_moment_nm"]
    assert st["seismic_moment_nm"]["min"] == pytest.approx(m0 / 1.1, rel=0.1)      # the 1.1 x gain table
    assert st["seismic_moment_nm"]["max"] == pytest.approx(m0 / 0.9, rel=0.1)
    assert e["n_resolved"] == 6 and e["stats_resolved"]["fc"]["n"] == 6        # fmax 20 kHz < 2 fc: band-limited
    assert [m for m, lim in zip(e["members"]["fmax_hz"], e["members"]["band_limited"]) if lim] == [20000.0] * 6
    assert "fc" in view.result_vars["ensemble"].get() and "6 resolved of 12 valid" in view.result_vars["ensemble"].get()
    assert "fc 16-84 %" in view.ensemble_text_var.get() and "ensemble:" in view.status_var.get()
    assert view.record["ensemble_calibration_paths"] == alt
    # a recompute clears it; Save runs it again when the box is ticked, and stores the inputs with the record
    view.post_var.set("2.5")
    view.schedule_compute(now=True)
    assert view.record["ensemble"] is None and view.ensemble_text_var.get() == "not run"
    view.ensemble_var.set(True)
    assert view.compute_source() and view.save(), view.status_var.get()
    rec = stored(view)["ch3"]
    assert rec["ensemble"]["n_members"] == 12 and rec["ensemble_on"] is True
    assert rec["ensemble_fmins_hz"] == [1000.0, 2000.0] and rec["ensemble_calibration_paths"] == alt
    assert experiment(brune_app)["sensors"]["ch3"]["ensemble_calibration_paths"] == alt
    settings = experiment(brune_app)["settings"]
    assert settings["estimator"] == "logmean" and settings["ensemble_on"] is True
    assert settings["ensemble_fmaxs_hz"] == [20000.0, 50000.0]
    # a bad grid or table is reported, not raised; Save then refuses
    view.ensemble_fmins_var.set("abc")
    view.schedule_compute(now=True)
    assert view.record is None and "ensemble fmin grid" in view.status_var.get()
    view.ensemble_fmins_var.set("1000, 2000")
    view.ensemble_calibrations_var.set(str(tmp_path / "missing.csv"))
    view.schedule_compute(now=True)
    assert view.record is not None and not view.compute_ensemble() and "alternative calibration" in view.status_var.get()
    assert not view.save() and "nothing saved" in view.status_var.get()
    # reopening restores the controls and shows the stored ensemble
    view.ensemble_calibrations_var.set(";".join(alt))
    view.on_close()
    with opened(brune_app) as v2:
        assert v2.estimator_combobox.get() == "logmean" and v2.ensemble_var.get()
        assert v2.ensemble_fmins_var.get() == "1000, 2000" and v2.ensemble_calibrations_var.get() == ";".join(alt)
        assert "6 resolved of 12 valid" in v2.result_vars["ensemble"].get()
        ens = v2.record["ensemble"]
        assert ensemble_stats(ens)[1] == "resolved"
        assert ensemble_stats({"stats": ens["stats"], "stats_resolved": {"fc": {"n": 0}}})[1] == "valid"
        assert ensemble_stats({"stats": ens["stats"]})[1] == "valid"           # written before the resolved set existed


def test_q_table_per_sensor(brune_app, view, tmp_path):
    q = tmp_path / "q.csv"
    q.write_text("freq_kHz,Qp_inv_median\n1,0.05\n100,0.05\n")
    calibrated_compute(view)
    f_plain, amp_plain = view.spectrum.f_binned.copy(), view.spectrum.amp_binned.copy()
    assert view.record["attenuation"] is None and view.record["q_csv_path"] == "" and "no Q" in view.q_note_var.get()
    # the table needs a travel time, and excludes t*
    view.q_path_var.set(str(q))
    view.schedule_compute(now=True)
    assert view.record is None and "travel time" in view.status_var.get()
    view.q_travel_var.set("5e-5")
    view.t_star_var.set("1e-6")
    view.schedule_compute(now=True)
    assert view.record is None and "not both" in view.status_var.get()
    view.t_star_var.set("")
    view.schedule_compute(now=True)
    assert view.record is not None, view.status_var.get()
    meta = view.spectrum.meta["attenuation"]
    assert meta["travel_time_s"] == 5e-5 and meta["hold_ends"] is False and meta["n_rows"] == 2
    assert view.record["q_travel_time_s"] == 5e-5 and view.record["attenuation"]["source"] == str(q)
    # the correction lifts the spectrum by exp(pi f T / Q), applied before binning
    np.testing.assert_allclose(view.spectrum.amp_binned / amp_plain, np.exp(np.pi * f_plain * 5e-5 * 0.05), rtol=1e-2)
    assert "Q loaded: q.csv" in view.q_note_var.get() and "T = 5e-05" in view.q_note_var.get()
    # hold ends and the travel time are re-applied from the stored table, no CSV needed
    view.q_hold_var.set(True)
    view.q_travel_var.set("1e-4")
    q.unlink()
    view.schedule_compute(now=True)
    assert view.record is not None, view.status_var.get()
    assert view.spectrum.meta["attenuation"]["hold_ends"] is True and view.spectrum.meta["attenuation"]["travel_time_s"] == 1e-4
    assert "end values held" in view.q_note_var.get()
    set_medium(view, "P")
    assert view.compute_source() and view.save()
    sensor = experiment(brune_app)["sensors"]["ch3"]
    assert sensor["q_travel_time_s"] == 1e-4 and sensor["q_hold_ends"] is True and sensor["attenuation"]["n_rows"] == 2
    view.clear_attenuation()
    assert view.record is not None and view.spectrum.meta["attenuation"] is None and view.attenuation_table is None
    # reopening restores the table, travel time and flag; another channel of the sensor set too
    view.on_close()
    with opened(brune_app) as v2:
        assert v2.current_channel() == BRUNE_CH
        assert v2.q_travel_var.get() == "0.0001" and v2.q_hold_var.get() and v2.attenuation_table["n_rows"] == 2
        assert v2.spectrum.meta["attenuation"]["travel_time_s"] == 1e-4 and "matches the saved record" in v2.status_var.get()
        v2.set_channel(TANH_CH)                           # no record, no sensor entry: the loaded table carries over
        assert v2.attenuation_table["n_rows"] == 2 and v2.spectrum.meta["attenuation"]["travel_time_s"] == 1e-4
