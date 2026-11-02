"""Headless-Tk tests for PZTSpectrumView on synthetic strain data."""
import copy
import json
import math

import numpy as np
import pytest

from labquake_explorer.analysis import source as src
from labquake_explorer.analysis import spectrum as sp
from labquake_explorer.analysis import spectrum_ensemble as se
from labquake_explorer.ui.views.pzt_spectrum_view import (
    HANDLES, PZTSpectrumView, RESULT_VERSION, TIE_AFTER, TRIGGER_PICKED, TRIGGER_EVENT_TIME, TRIGGER_RECORD,
    TUKEY_ALPHA, channel_key, parse_channel_key,
)

EVENT = 1
BRUNE_CH = 3          # channel replaced by a Brune displacement pulse
TANH_CH = 0           # untouched tanh step (arrival = event_time)
OMEGA0_V = 1.0        # plateau of the synthetic pulse in V*s
FC = 10e3             # Hz
GAIN = 2.0            # V/m -> displacement plateau = OMEGA0_V / GAIN
N_WIN = 500           # samples in the default 0.5 + 2.0 ms window at 200 kHz
RHO, VP, VS, DIST = 2700.0, 6000.0, 3500.0, 0.1


@pytest.fixture
def brune_app(app_with_strain, tmp_path):
    """The strain app with event 1 / channel 3 replaced by a Brune pulse plus
    white noise 30 dB below the plateau, tiny noise on every other channel
    (so the SNR gate has a finite noise spectrum), and a V/m calibration CSV."""
    app = app_with_strain
    event = app.data_manager.get_data(f"runs/[0]/events/[{EVENT}]")
    original = event["strain"]["original"]
    time = np.asarray(original["time"], dtype=float)
    raw = np.asarray(original["raw"], dtype=float)
    fs = sp.sampling_rate(time)
    rng = np.random.default_rng(7)
    raw = raw + rng.normal(0.0, 5e-5, raw.shape)
    t0 = float(event["event_time"]) - time[0]
    _, pulse = sp.brune_pulse(fs, time.size, OMEGA0_V, FC, t0)
    sigma = OMEGA0_V / math.sqrt(1000.0) * fs / math.sqrt(N_WIN)
    raw[BRUNE_CH] = pulse + rng.normal(0.0, sigma, time.size)
    original["raw"] = raw
    csv = tmp_path / "sensor.csv"
    f = np.geomspace(100.0, 100e3, 25)
    csv.write_text("frequency_hz,gain\n" + "\n".join(f"{fi:.6g},{GAIN}" for fi in f) + "\n")
    app.calibration_csv = str(csv)
    app.fs = fs
    return app


@pytest.fixture
def view(brune_app):
    v = PZTSpectrumView(brune_app, 0, EVENT)
    yield v
    if v.winfo_exists():
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


# ---------------------------------------------------------------------------
def test_open_defaults(brune_app, view):
    assert view.event_idx == EVENT
    assert view.title() == "PZT Spectrum - Event 1"
    assert view in brune_app.child_windows
    assert list(view.channel_combobox["values"]) == [str(i) for i in range(16)]
    assert view.current_channel() == 0
    assert view.trigger_combobox.get() == TRIGGER_EVENT_TIME
    assert (view.pre_var.get(), view.post_var.get()) == ("0.5", "2.0")
    assert float(view.fmax_var.get()) == pytest.approx(0.4 * brune_app.fs)
    assert float(view.fmin_var.get()) == 1e3
    assert view.unit_combobox.get() == "V/V" and view.calibration_path_var.get() == ""
    assert view.n_fixed_var.get() and view.snr_var.get() == "3"
    assert float(view.radiation_var.get()) == pytest.approx(src.PHASE_CONSTANTS["P"]["radiation_rms"])
    assert view.radiation_hint_var.get() == "RMS for P: 0.516"
    # computed on opening: no Compute button to press
    assert view.window is not None and view.spectrum is not None and view.record is not None
    assert "fit" in view.status_var.get() and "Compute" not in view.status_var.get()
    assert not hasattr(view, "compute_button")


def test_compute_uncalibrated_brune_channel(view):
    view.set_channel(BRUNE_CH)
    assert view.compute(), view.status_var.get()
    fit = view.fit
    assert fit.valid, fit.reason
    assert fit.amp_units == "V*s"
    assert fit.omega0 == pytest.approx(OMEGA0_V, rel=0.05)
    assert fit.fc == pytest.approx(FC, rel=0.05)
    assert fit.n == 2.0 and fit.n_fixed and fit.converged and not fit.band_limited
    assert fit.warnings == ()
    assert view.window.n == N_WIN and view.window.noise_available
    assert view.spectrum.meta["transient_in_flat_region"] is True
    assert view.record["channel"] == BRUNE_CH and view.record["fit"]["valid"]
    assert view.record["parameter_warnings"] == []
    assert "V*s" in view.result_vars["omega0"].get()
    assert view.result_vars["fc"].get().endswith("Hz")
    assert "converged: yes" in view.result_vars["flags"].get()
    assert "band-limited: no" in view.result_vars["flags"].get()
    assert "WARNINGS" not in view.result_vars["flags"].get()
    assert "flat region: yes" in view.result_vars["taper"].get()
    assert view.result_vars["n"].get().startswith("2")
    assert "fit OK" in view.status_var.get()
    # the bins-used label mirrors the fit (all bins clear the 3x gate at 30 dB below the plateau)
    assert int(view.result_vars["n_bins"].get()) == fit.n_bins >= 8


def test_compute_calibrated_recovers_displacement_plateau(view):
    fit = calibrated_compute(view)
    assert fit.valid, fit.reason
    assert fit.amp_units == "m*s"
    assert fit.omega0 == pytest.approx(OMEGA0_V / GAIN, rel=0.05)
    assert fit.fc == pytest.approx(FC, rel=0.05)
    assert view.spectrum.calibrated
    assert view.record["calibration_unit"] == "V/m"
    assert view.record["calibration_path"] == view.app.calibration_csv
    assert view.record["spectrum"]["meta"]["calibration"]["unit"] == "V/m"
    assert "m*s" in view.result_vars["omega0"].get()


def test_free_n_and_soft_l1(view):
    view.set_channel(BRUNE_CH)
    view.n_fixed_var.set(False)
    view.loss_combobox.set("soft_l1")
    assert view.compute()
    assert view.fit.valid and not view.fit.n_fixed
    assert view.fit.n == pytest.approx(2.0, abs=0.3)
    assert view.fit.fc == pytest.approx(FC, rel=0.10)
    assert view.record["loss"] == "soft_l1" and view.record["n_fixed"] is False
    assert "(free)" in view.result_vars["n"].get()


def test_remove_step_on_tanh_channel(view):
    view.set_channel(TANH_CH)
    view.remove_step_var.set(True)
    assert view.compute(), view.status_var.get()
    assert view.window.step_removed
    assert view.window.step_amplitude == pytest.approx(0.5, rel=0.05)
    assert view.fit.valid, view.fit.reason
    assert view.record["remove_step"] is True
    assert view.record["spectrum"]["meta"]["step_removed"] is True
    # the step is fully removed: the window ends near zero
    assert abs(np.mean(view.window.signal[-50:])) < 0.02


def test_picked_arrival_trigger(brune_app, view):
    event = view.event
    arrivals = np.asarray(event["strain_truth"]["arrival_times"], dtype=float)
    view.set_channel(TANH_CH)
    view.trigger_combobox.set(TRIGGER_PICKED)
    # no picks yet -> status, no dialog
    assert not view.compute()
    assert "picked arrival" in view.status_var.get()
    event["strain"]["original"]["rupture_arrival_time"] = arrivals
    assert view.compute(), view.status_var.get()
    time = np.asarray(event["strain"]["original"]["time"], dtype=float)
    assert view.window.i_trigger == int(np.argmin(np.abs(time - arrivals[TANH_CH])))
    assert view.record["trigger"] == TRIGGER_PICKED
    assert view.record["trigger_time_s"] == pytest.approx(arrivals[TANH_CH])


def test_invalid_entries_go_to_status_not_dialog(view):
    view.set_channel(BRUNE_CH)
    view.pre_var.set("abc")
    assert not view.compute()
    assert "pre window" in view.status_var.get()
    view.pre_var.set("0.5")
    view.fmax_var.set("500")           # below fmin
    assert not view.compute()
    assert "fmax" in view.status_var.get()
    view.fmax_var.set("80000")
    view.post_var.set("50")            # 50 ms: window leaves the 20 ms trace
    assert not view.compute()
    assert "not inside the trace" in view.status_var.get()
    view.post_var.set("2.0")
    view.baseline_combobox.set("highpass")
    view.highpass_var.set("")
    assert not view.compute()
    assert "highpass" in view.status_var.get()
    view.highpass_var.set("500")
    assert view.compute()
    assert view.record["baseline"] == "highpass" and view.record["highpass_hz"] == 500.0
    assert view.window.baseline_mode == "highpass" and view.window.highpass_hz == 500.0
    view.baseline_combobox.set("pre_linear")
    view.unit_combobox.set("V/m")      # unit without a CSV
    view.calibration_path_var.set("")
    assert not view.compute()
    assert "needs a CSV" in view.status_var.get()
    view.calibration_path_var.set("/nonexistent/sensor.csv")
    assert not view.compute()
    assert "calibration CSV" in view.status_var.get()
    assert view.record is not None  # last good record kept


def test_parameters_reach_the_analysis_calls(brune_app, view):
    """Every entry is applied, not merely stored in the record."""
    event = view.event
    arrivals = np.asarray(event["strain_truth"]["arrival_times"], dtype=float)
    event["strain"]["original"]["rupture_arrival_time"] = arrivals
    view.set_channel(BRUNE_CH)
    view.trigger_combobox.set(TRIGGER_PICKED)
    view.pre_var.set("0.6")
    view.post_var.set("1.8")
    view.taper_combobox.set("tukey")
    view.baseline_combobox.set("highpass")
    view.highpass_var.set("300")
    view.t_star_var.set("1e-6")
    view.fmin_var.set("2000")
    view.fmax_var.set("70000")
    view.bins_var.set("20")
    view.snr_var.set("2")
    view.n_fixed_var.set(False)
    view.loss_combobox.set("soft_l1")
    assert view.compute(), view.status_var.get()
    fs = brune_app.fs
    assert view.window.n_pre == round(0.6e-3 * fs) and view.window.n_post == round(1.8e-3 * fs)
    assert view.window.trigger_time == pytest.approx(arrivals[BRUNE_CH])
    assert view.window.baseline_mode == "highpass" and view.window.highpass_hz == 300.0
    assert view.window.baseline_region == "highpass"
    assert view.spectrum.t_star_s == 1e-6
    assert view.spectrum.meta["taper"] == "tukey" and view.spectrum.meta["alpha"] == TUKEY_ALPHA
    assert view.spectrum.meta["fmin"] == 2000.0 and view.spectrum.meta["fmax"] == 70000.0
    assert view.spectrum.meta["bins_per_decade"] == 20
    assert view.spectrum.meta["highpass_hz"] == 300.0
    assert view.fit.snr_min == 2.0 and view.fit.loss == "soft_l1" and not view.fit.n_fixed
    assert view.fit.band_used[0] >= 2000.0 and view.fit.band_used[1] <= 70000.0
    # the same calls with t* off give a different plateau (the correction was applied)
    view.t_star_var.set("")
    omega_t = view.fit.omega0
    assert view.compute()
    assert view.spectrum.t_star_s is None
    assert view.fit.omega0 != pytest.approx(omega_t, rel=1e-6)


def test_short_pre_window_warning_is_visible_and_warns_the_source(brune_app, view):
    """A pre window shorter than the tukey ramp attenuates Omega0; the view
    must say so (flags, status, taper row) and the source parameters, still
    computed, carry the warning."""
    view.set_channel(BRUNE_CH)
    view.calibration_path_var.set(view.app.calibration_csv)
    view.unit_combobox.set("V/m")
    view.pre_var.set("0.1")
    view.post_var.set("2.5")
    assert view.compute(), view.status_var.get()
    assert view.fit.valid
    assert view.fit.omega0 < 0.5 * OMEGA0_V / GAIN               # attenuated well below the truth
    assert view.spectrum.meta["transient_in_flat_region"] is False
    assert any(w.startswith("transient not in the taper's flat region") for w in view.fit.warnings)
    flags = view.result_vars["flags"].get()
    assert "WARNINGS" in flags and "attenuated" in flags and "flat region" in flags
    assert "shorter than the tukey ramp" in flags                # the parameter warning too
    assert view.record["parameter_warnings"] and "tukey ramp" in view.record["parameter_warnings"][0]
    assert "flat region: no" in view.result_vars["taper"].get()
    status = view.status_var.get()
    assert "fit OK" not in status
    assert "warnings" in status and "attenuated" in status
    # the source parameters are computed anyway, the warning carried along and saved with them
    set_medium(view, "P", 0.52)
    assert view.compute_source(), view.status_var.get()
    assert "attenuated" in view.status_var.get() and "not saved" in view.status_var.get()
    source = view.record["source"]
    assert source["valid"] and any("attenuated" in w for w in source["warnings"])
    assert "attenuated" in view.source_vars["warnings"].get()
    assert "attenuated" in view.result_vars["flags"].get()
    assert view.save()
    saved = brune_app.data_manager.get_data(f"runs/[0]/events/[{EVENT}]/strain/spectrum")["sets"]["default"]["channels"]["ch3"]
    assert saved["source"]["valid"] and any("attenuated" in w for w in saved["source"]["warnings"])
    # the parameter warning fires on its own from the entries
    assert PZTSpectrumView.parameter_warnings({"taper": "tukey", "pre_ms": 0.3, "post_ms": 2.5})
    assert PZTSpectrumView.parameter_warnings({"taper": "tukey", "pre_ms": 0.5, "post_ms": 2.0}) == []
    # a proper pre window: no warnings, source accepted
    view.pre_var.set("0.5")
    assert view.compute()
    assert view.fit.omega0 == pytest.approx(OMEGA0_V / GAIN, rel=0.05)
    assert view.record["parameter_warnings"] == [] and view.fit.warnings == ()
    assert "fit OK" in view.status_var.get()
    assert "WARNINGS" not in view.result_vars["flags"].get()
    assert view.compute_source(), view.status_var.get()
    assert not any("attenuated" in w for w in view.record["source"]["warnings"])
    assert "attenuated" not in view.status_var.get()


def test_compute_source(view):
    fit = calibrated_compute(view)
    # uncalibrated / missing inputs are refused with a status message
    view.rho_var.set("")
    assert not view.compute_source()
    assert "rho" in view.status_var.get()
    set_medium(view, "P", 0.52)
    assert view.compute_source(), view.status_var.get()
    sd = view.record["source"]
    assert sd["valid"] and sd["phase"] == "P"
    m0 = sd["seismic_moment_nm"]
    assert math.isfinite(m0) and m0 > 0 and math.isfinite(sd["mw"])
    assert m0 == pytest.approx(src.seismic_moment(fit.omega0, RHO, VP, DIST, 0.52), rel=1e-12)
    assert sd["constants"]["wave_speed_m_s"] == VP
    assert sd["mw"] == pytest.approx(src.moment_magnitude(m0))
    assert sd["source_radius_m"] == pytest.approx(0.32 * VS / fit.fc)
    assert sd["stress_drop_pa"] == pytest.approx(7 * m0 / (16 * sd["source_radius_m"] ** 3))
    assert sd["constants"]["f_plateau_hz"] == pytest.approx(fit.band_used[0])
    assert math.isfinite(sd["kR"])
    assert view.source_vars["m0"].get().endswith("N m")
    assert view.source_vars["mw"].get() != "n/a"
    assert view.source_vars["warnings"].get() != "n/a"
    # an uncalibrated spectrum is refused
    view.unit_combobox.set("V/V")
    view.calibration_path_var.set("")
    assert view.compute()
    assert not view.compute_source()
    assert "uncalibrated" in view.status_var.get()


def test_free_surface_factor_reaches_the_moment(brune_app, view):
    """The typed free-surface factor divides M0, is stored in the constants and
    restored on reopen; an unusable value is a status message, not a dialog."""
    fit = calibrated_compute(view)
    set_medium(view, "P", 0.52)
    view.free_surface_var.set("1.7")
    assert view.compute_source(), view.status_var.get()
    sd = view.record["source"]
    assert sd["seismic_moment_nm"] == pytest.approx(
        src.seismic_moment(fit.omega0, RHO, VP, DIST, 0.52, 1.7), rel=1e-12)
    assert sd["seismic_moment_nm"] == pytest.approx(
        src.seismic_moment(fit.omega0, RHO, VP, DIST, 0.52) / 1.7, rel=1e-12)
    assert sd["constants"]["free_surface_factor"] == 1.7
    assert view.save()
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.free_surface_var.get() == "1.7"
        assert v2.record["source"]["constants"]["free_surface_factor"] == 1.7
        for bad in ("0", "-2", "nan", "abc", ""):
            v2.free_surface_var.set(bad)
            assert not v2.compute_source()
            assert "free-surface factor" in v2.status_var.get()
        v2.free_surface_var.set("1.0")
        assert v2.compute_source()
        assert v2.record["source"]["seismic_moment_nm"] == pytest.approx(
            src.seismic_moment(fit.omega0, RHO, VP, DIST, 0.52), rel=1e-12)
    finally:
        v2.on_close()


def test_compute_source_s_phase_uses_vs(view):
    fit = calibrated_compute(view)
    set_medium(view, "S", 0.6)
    assert view.compute_source(), view.status_var.get()
    sd = view.record["source"]
    assert sd["valid"] and sd["phase"] == "S"
    assert sd["constants"]["wave_speed_m_s"] == VS and sd["constants"]["vp_m_s"] == VP
    assert sd["seismic_moment_nm"] == pytest.approx(src.seismic_moment(fit.omega0, RHO, VS, DIST, 0.6), rel=1e-12)
    assert sd["seismic_moment_nm"] != pytest.approx(src.seismic_moment(fit.omega0, RHO, VP, DIST, 0.6), rel=1e-3)
    assert sd["source_radius_m"] == pytest.approx(0.372 * VS / fit.fc)
    assert sd["kR"] == pytest.approx(2 * math.pi * fit.band_used[0] * DIST / VS)


def test_compute_source_guards_malformed_fit(view):
    calibrated_compute(view)
    set_medium(view, "P")
    del view.record["fit"]["fc"]
    assert not view.compute_source()
    assert "fc" in view.status_var.get()
    view.record["fit"]["fc"] = {"bad": 1}
    assert not view.compute_source()
    assert view.record["source"]["valid"] is False       # the analysis reports the bad fc, no raise
    assert "invalid" in view.source_vars["warnings"].get()


def test_phase_switch_updates_default_radiation(view):
    view.phase_combobox.set("S")
    view.on_phase_selected()
    assert float(view.radiation_var.get()) == pytest.approx(src.PHASE_CONSTANTS["S"]["radiation_rms"])
    assert view.radiation_hint_var.get() == "RMS for S: 0.632"
    view.phase_combobox.set("P")
    view.on_phase_selected()
    assert float(view.radiation_var.get()) == pytest.approx(src.PHASE_CONSTANTS["P"]["radiation_rms"])


def test_typed_radiation_survives_phase_switch(brune_app, view):
    view.radiation_var.set("0.9")
    view.phase_combobox.set("S")
    view.on_phase_selected()
    assert view.radiation_var.get() == "0.9"
    assert view.radiation_hint_var.get() == "RMS for S: 0.632"
    view.phase_combobox.set("P")
    view.on_phase_selected()
    assert view.radiation_var.get() == "0.9"
    # a saved S-phase source at the S default, reopened: a fresh channel still follows the phase
    calibrated_compute(view)
    set_medium(view, "S", src.PHASE_CONSTANTS["S"]["radiation_rms"])
    assert view.radiation_is_default()
    assert view.compute_source() and view.save()
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.phase_combobox.get() == "S"
        v2.set_channel(5)
        v2.phase_combobox.set("P")
        v2.on_phase_selected()
        assert float(v2.radiation_var.get()) == pytest.approx(src.PHASE_CONSTANTS["P"]["radiation_rms"])
        # ... but a restored typed value is kept across a phase switch
        v2.set_channel(BRUNE_CH)
        v2.radiation_var.set("0.45")
        v2.phase_combobox.set("P")
        v2.on_phase_selected()
        assert v2.radiation_var.get() == "0.45"
    finally:
        v2.on_close()


def test_save_merges_channels_and_is_json_like(brune_app, view):
    calibrated_compute(view)
    set_medium(view, "P")
    assert view.compute_source()
    assert view.save()
    saved = brune_app.data_manager.get_data(f"runs/[0]/events/[{EVENT}]/strain/spectrum")
    assert saved is view.event["strain"]["spectrum"]
    assert saved["version"] == 2 and list(saved["sets"]) == ["default"] and "channels" not in saved
    channels = saved["sets"]["default"]["channels"]
    assert set(channels) == {channel_key(BRUNE_CH)} == {"ch3"}
    rec = channels["ch3"]
    assert rec["result_set"] == "default"
    for key in ("version", "channel", "trigger", "trigger_time_s", "fs_hz", "pre_ms", "post_ms", "taper",
                "baseline", "highpass_hz", "remove_step", "calibration_path", "calibration_unit",
                "calibration_db", "calibration", "t_star_s", "fmin_hz", "fmax_hz", "bins_per_decade", "snr_min",
                "n_fixed", "n_max", "loss", "parameter_warnings", "source_inputs", "spectrum", "fit", "source"):
        assert key in rec, key
    assert rec["n_max"] == 4.0 and rec["fit"]["n_bounds"] == [1.0, 4.0]
    assert rec["spectrum"]["meta"]["noise_region"] == "pre"
    assert rec["channel"] == BRUNE_CH and rec["fit"]["valid"] and rec["source"]["valid"]
    assert rec["spectrum"]["amp_units"] == "m*s"
    assert isinstance(rec["fit"]["warnings"], list) and isinstance(rec["fit"]["band_used"], list)
    json.dumps(saved)  # JSON-like (NaN allowed)

    # a second channel is merged, the first kept
    view.set_channel(TANH_CH)
    assert view.record is not None and view.record["channel"] == TANH_CH      # computed on selection
    view.remove_step_var.set(True)
    view.unit_combobox.set("V/V")
    view.calibration_path_var.set("")
    assert view.compute()
    assert view.save()
    saved = view.event["strain"]["spectrum"]
    channels = saved["sets"]["default"]["channels"]
    assert set(channels) == {"ch3", "ch0"}
    assert channels["ch3"]["source"]["valid"]
    assert channels["ch0"]["source"] is None
    assert channels["ch0"]["remove_step"] is True
    # save with nothing computable (a bad control) is refused
    view.set_channel(7)
    view.pre_var.set("abc")
    view.schedule_compute(now=True)
    assert view.record is None and view.result_vars["fc"].get() == "n/a"
    assert not view.save()
    assert "nothing to save" in view.status_var.get()


def test_channel_keys_are_not_digit_strings_and_legacy_keys_are_read(brune_app, view):
    assert channel_key(3) == "ch3" and parse_channel_key("ch3") == 3
    assert parse_channel_key("3") == 3 and parse_channel_key("CH12") == 12
    assert parse_channel_key("chx") is None and parse_channel_key("source") is None
    calibrated_compute(view)
    rec = view.record
    # a legacy all-digit key is read and renamed on the next save
    view.event["pzt_spectrum"] = {"version": RESULT_VERSION, "channels": {"5": dict(rec, channel=5)}}
    assert set(view.saved_channels()) == {5}
    view.set_channel(5)
    assert view.record is not None and "saved record" in view.status_var.get()
    view.set_channel(BRUNE_CH)
    assert view.compute() and view.save()
    channels = view.event["strain"]["spectrum"]["sets"]["default"]["channels"]
    assert set(channels) == {"ch3", "ch5"}
    assert not any(k.isdigit() for k in channels)


def test_saved_record_without_channel_field_can_be_saved(brune_app, view):
    """A partial / hand-edited record lacking ``channel`` opens as a saved
    record; Save files it under the channel it was found at instead of raising."""
    view.event["pzt_spectrum"] = {"version": RESULT_VERSION, "channels": {"ch2": {"pre_ms": 1.0}}}
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.current_channel() == 2
        assert "saved record" in v2.status_var.get()     # recomputed from the partial record's parameters
        assert v2.pre_var.get() == "1"
        assert v2.record["channel"] == 2
        assert v2.save()
        channels = v2.event["strain"]["spectrum"]["sets"]["default"]["channels"]
        assert set(channels) == {"ch2"} and channels["ch2"]["channel"] == 2
        assert "channel 2 saved" in v2.status_var.get()
        # a record with no channel and no selection is refused with a status line
        v2.record = {"pre_ms": 1.0}
        v2.channel_combobox.set("")
        assert not v2.save()
        assert "no channel" in v2.status_var.get()
    finally:
        v2.on_close()


def test_legacy_digit_keyed_channels_loaded_as_a_list_are_read(brune_app, view):
    """A legacy ``channels`` map keyed '0'..'n-1' comes back from an unmarked
    (older) HDF5 file as a LIST; its entries are channels 0..n-1 and a later
    Save keeps them (renamed) instead of silently dropping them."""
    fit = calibrated_compute(view)
    rec = view.record
    view.event["pzt_spectrum"] = {"version": RESULT_VERSION,
                                  "channels": [dict(rec, channel=0), dict(rec, channel=1), "junk"]}
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert set(v2.saved_channels()) == {0, 1}
        assert v2.current_channel() == 0
        # the record was computed on channel 3; recomputed on channel 0 it no longer matches
        assert "differs from the saved record" in v2.status_var.get()
        assert v2.result_vars["fc"].get() != f"{fit.fc:.4g} Hz"
        v2.set_channel(1)
        assert "saved record" in v2.status_var.get()
        v2.set_channel(BRUNE_CH)
        v2.calibration_path_var.set(brune_app.calibration_csv)
        v2.unit_combobox.set("V/m")
        assert v2.compute() and v2.save()
        channels = v2.event["strain"]["spectrum"]["sets"]["default"]["channels"]
        assert isinstance(channels, dict)
        assert set(channels) == {"ch0", "ch1", "ch3"}          # the legacy records are kept and renamed
        assert channels["ch1"]["channel"] == 1
    finally:
        v2.on_close()


def test_restore_accepts_array_valued_fields(brune_app, view):
    """A loader that hands back numpy arrays for stored lists must not break
    restore (truth-testing an array raises)."""
    calibrated_compute(view)
    set_medium(view, "P")
    assert view.compute_source() and view.save()
    fit_fc = view.fit.fc
    rec = view.event["strain"]["spectrum"]["sets"]["default"]["channels"]["ch3"]
    rec["fit"]["band_used"] = np.asarray(rec["fit"]["band_used"])
    rec["fit"]["warnings"] = np.asarray(["band_limited: synthetic"])
    for key in ("f_binned", "amp_binned", "noise_binned", "snr", "counts"):
        rec["spectrum"][key] = np.asarray(rec["spectrum"][key])
    rec["source"]["warnings"] = np.asarray(["kR synthetic warning"])
    rec["parameter_warnings"] = np.asarray([], dtype=object)
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.current_channel() == BRUNE_CH
        # the array-valued record was applied and compared without raising; the recomputation agrees
        assert "matches the saved record" in v2.status_var.get()
        assert v2.result_vars["fc"].get() == f"{fit_fc:.4g} Hz"
        assert v2.result_vars["band"].get().endswith("Hz")
        set_medium(v2, "P")
        v2.schedule_compute(now=True)
        assert "source parameters computed" in v2.status_var.get()
        assert v2.source_vars["m0"].get().endswith("N m")
        assert v2.compute_source(), v2.status_var.get()   # band_used array -> f_plateau
    finally:
        v2.on_close()


def test_malformed_saved_record_does_not_block_the_window(brune_app, view, monkeypatch):
    calibrated_compute(view)
    assert view.save()
    view.on_close()
    channels = brune_app.data_manager.get_data(f"runs/[0]/events/[{EVENT}]/strain/spectrum")["sets"]["default"]["channels"]
    good = channels["ch3"]
    # fit stored as a string, at_bounds not a dict, band garbage, source constants odd
    channels["ch3"] = dict(good, fit="garbage", source={"valid": True, "constants": "x", "warnings": 3})
    channels["ch4"] = dict(good, channel=4, pre_ms="abc",
                           fit=dict(good["fit"], at_bounds="oops", band_used="abc", omega0="?", warnings=None),
                           spectrum=dict(good["spectrum"], f_binned="nope", meta="meta"))
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.current_channel() == BRUNE_CH
        assert v2.result_vars["fc"].get() != "n/a"       # recomputed from the applied parameters
        assert v2.pre_var.get() == "0.5"                 # parameters were still applied
        v2.set_channel(4)
        assert "saved record" in v2.status_var.get()     # unreadable: reported, nothing recomputed
        assert v2.result_vars["omega0"].get() == "n/a" and v2.result_vars["band"].get() == "n/a"
        assert v2.pre_var.get() == "abc"
        assert not v2.compute() and "pre window" in v2.status_var.get()
        assert not v2.compute_source()
        # an exception while applying a record lands in the status line, not in the caller
        monkeypatch.setattr(PZTSpectrumView, "apply_record",
                            lambda self, rec: (_ for _ in ()).throw(KeyError("boom")))
        v2.set_channel(BRUNE_CH)
        assert "saved record unreadable" in v2.status_var.get() and "boom" in v2.status_var.get()
        assert v2.record is None and v2.result_vars["fc"].get() == "n/a"
        assert not v2.save()
    finally:
        v2.on_close()
    monkeypatch.undo()
    v3 = PZTSpectrumView(brune_app, 0, EVENT)   # opening again with the broken records works too
    v3.on_close()


def test_reopen_restores_parameters_and_numbers(brune_app, view):
    view.set_channel(BRUNE_CH)
    view.pre_var.set("0.4")
    view.post_var.set("1.5")
    view.calibration_path_var.set(brune_app.calibration_csv)
    view.unit_combobox.set("V/m")
    view.taper_combobox.set("bh4")          # a bell taper: the pulse at 21% of the window is attenuated
    assert view.compute()
    assert "flat region: no" in view.result_vars["taper"].get()
    set_medium(view, "P", 0.6)
    assert view.compute_source() and "attenuated" in view.status_var.get()   # computed, with the warning
    view.taper_combobox.set("tukey")        # ramp 0.24 ms < pre 0.4 ms: flat region
    view.snr_var.set("2")
    view.bins_var.set("25")
    view.fmin_var.set("1500")
    view.t_star_var.set("1e-6")
    assert view.compute(), view.status_var.get()
    fit_saved = view.fit
    set_medium(view, "P", 0.6)
    assert view.compute_source()
    assert view.save()
    rec_saved = view.event["strain"]["spectrum"]["sets"]["default"]["channels"]["ch3"]
    view.on_close()

    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.current_channel() == BRUNE_CH          # opens on the saved channel
        assert v2.fit is not None and v2.window is not None   # recomputed from the restored parameters
        assert v2.record is not None and v2.record["fit"]["fc"] == pytest.approx(fit_saved.fc)
        assert (v2.pre_var.get(), v2.post_var.get()) == ("0.4", "1.5")
        assert v2.taper_combobox.get() == "tukey"
        assert v2.snr_var.get() == "2"
        assert float(v2.t_star_var.get()) == pytest.approx(1e-6)
        assert v2.calibration_path_var.get() == brune_app.calibration_csv
        assert v2.unit_combobox.get() == "V/m"
        assert (v2.rho_var.get(), v2.c_var.get(), v2.vs_var.get(), v2.distance_var.get()) == (
            "2700", "6000", "3500", "0.1")
        assert float(v2.radiation_var.get()) == 0.6
        assert v2.result_vars["fc"].get() == f"{fit_saved.fc:.4g} Hz"
        assert v2.source_vars["m0"].get().endswith("N m")
        assert "saved record" in v2.status_var.get()
        # every parameter the controls produce equals the saved record's
        p = v2.read_parameters()
        assert all(p[k] == rec_saved[k] for k in p), {k: (p[k], rec_saved[k]) for k in p if p[k] != rec_saved[k]}
        # switching to an unsaved channel computes it with the same controls, switching back restores
        v2.set_channel(5)
        assert v2.record is not None and v2.record["channel"] == 5 and "not saved" in v2.status_var.get()
        assert v2.pre_var.get() == "0.4"                 # parameters are left as they were
        v2.set_channel(BRUNE_CH)
        assert v2.record["fit"]["fc"] == pytest.approx(fit_saved.fc)
        # Compute on the restored channel recomputes with the restored parameters
        assert v2.compute()
        assert v2.fit.fc == pytest.approx(fit_saved.fc)
        assert v2.window.n_pre == round(0.4e-3 * brune_app.fs)
        assert v2.spectrum.t_star_s == 1e-6 and v2.fit.snr_min == 2.0
        assert v2.spectrum.meta["bins_per_decade"] == 25 and v2.spectrum.meta["fmin"] == 1500.0
        assert v2.spectrum.meta["taper"] == "tukey"
        # switching event keeps the channel; event 2 has no saved record, so the fresh one is unsaved
        v2.set_event(2)
        assert v2.current_channel() == BRUNE_CH and v2.record is not None
        assert "saved record" not in v2.status_var.get()   # event 2 has none
        assert v2.title() == "PZT Spectrum - Event 2"
    finally:
        v2.on_close()


def test_event_without_strain(app):
    v = PZTSpectrumView(app, 0, 1)
    try:
        assert list(v.channel_combobox["values"]) == []
        assert "no full-rate record" in v.status_var.get()
        assert not v.compute()
    finally:
        v.on_close()


def test_close_unregisters(brune_app):
    v = PZTSpectrumView(brune_app, 0, EVENT)
    assert v in brune_app.child_windows
    v.on_close()
    assert v not in brune_app.child_windows


def test_db_calibration_flag_reaches_the_loader(view, tmp_path):
    csv = tmp_path / "sensor_db.csv"
    f = np.geomspace(100.0, 100e3, 25)
    db = 20.0 * math.log10(GAIN)
    csv.write_text("Freq,Response\n" + "\n".join(f"{fi:.6g},{db:.6f}" for fi in f) + "\n")
    view.set_channel(BRUNE_CH)
    view.calibration_path_var.set(str(csv))
    view.unit_combobox.set("V/m")
    view.calibration_db_var.set(True)
    assert view.compute(), view.status_var.get()
    assert view.fit.valid and view.fit.omega0 == pytest.approx(OMEGA0_V / GAIN, rel=0.05)
    assert view.record["calibration_db"] is True
    # without the flag the 6 dB column is read as a linear gain of ~6
    view.calibration_db_var.set(False)
    assert view.compute()
    assert view.fit.omega0 == pytest.approx(OMEGA0_V / db, rel=0.05)


def test_record_trigger_is_the_default_for_a_waveform_record(app):
    """A waveform record extracted from an Elsys trigger block carries the
    recorder's own trigger time; the view triggers on it by default because the
    event_time is a pick on the slow mechanical channels, tens of ms later."""
    from labquake_explorer.data.channels import channel_array
    run = app.data_manager.get_data("runs/[0]")
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    fs = 200e3
    t_trig = float(event["event_time"]) - 0.02                      # the PZT trigger, 20 ms before the pick
    time = t_trig - 0.1 + np.arange(int(0.6 * fs)) / fs
    rng = np.random.default_rng(3)
    raw = rng.normal(0.0, 1e-4, (2, time.size))
    burst = np.exp(-((time - t_trig - 2e-4) / 1e-4) ** 2) * np.sin(2 * np.pi * 50e3 * (time - t_trig))
    raw[0] += 0.5 * burst
    event["waveform"] = {"elsys": channel_array(raw, ["pzt_1", "pzt_2"], unit="V", time=time, sample_rate=fs,
                                                block=4, trigger_time=t_trig, filename="run.tpc5")}
    v = PZTSpectrumView(app, 0, 1)
    try:
        assert v.record_key == "waveform/elsys" and v.trigger_combobox.get() == TRIGGER_RECORD
        assert v.record_trigger_time() == pytest.approx(t_trig)
        assert v.compute(), v.status_var.get()
        assert v.window.trigger_time == pytest.approx(t_trig)
        assert v.record["trigger"] == TRIGGER_RECORD and v.record["trigger_time_s"] == pytest.approx(t_trig)
        # an older record without the stored time is looked up in the run's block table
        del event["waveform"]["elsys"]["trigger_time"]
        run["elsys"] = {"format": "tpc5", "blocks": {"block": [2, 3, 4], "trigger_time_run": [1.0, 2.0, t_trig]}}
        assert v.record_trigger_time() == pytest.approx(t_trig)
        # a record with no trigger of its own reports it instead of failing silently
        del event["waveform"]["elsys"]["block"]
        assert not v.compute() and "no trigger time" in v.status_var.get()
    finally:
        v.on_close()


def test_legacy_strain_keeps_event_time_as_default(view):
    assert view.trigger_combobox.get() == TRIGGER_EVENT_TIME
    view.trigger_combobox.set(TRIGGER_RECORD)
    assert not view.compute() and "no trigger time" in view.status_var.get()


# ------------------------------------------------------------ live updates
def test_controls_recompute_without_a_button(brune_app, view):
    view.set_channel(BRUNE_CH)
    assert view.record["post_ms"] == 2.0
    # every entry recomputes on Return and on leaving the field, and after a pause while typing
    for entry in view.entries.values():
        assert {"<Key-Return>", "<FocusOut>", "<KeyRelease>"} <= set(entry.bind())
    view.post_var.set("1.5")
    view.schedule_compute(now=True)                      # what Return / FocusOut do
    assert view.record["post_ms"] == 1.5 and view.window.n_post == int(round(1.5e-3 * brune_app.fs))
    view.pre_var.set("0.6")
    view.schedule_compute()                              # what a keystroke does: wait for the keys to settle
    assert view.record["pre_ms"] == 0.5 and view._compute_after_id is not None
    view.schedule_compute(now=True)                      # what the timer does
    assert view.record["pre_ms"] == 0.6 and view._compute_after_id is None
    # a combobox and a checkbox recompute at once
    view.taper_combobox.set("bh4")
    view.taper_combobox.event_generate("<<ComboboxSelected>>")
    assert view.record["taper"] == "bh4"
    view.n_fixed_var.set(False)
    view.schedule_compute(now=True)
    assert view.record["n_fixed"] is False
    # a half-typed number is reported in the status (and the stale numbers cleared); the next valid value recovers
    view.pre_var.set("0.a")
    view.schedule_compute(now=True)
    assert "pre" in view.status_var.get().lower() and view.record is None
    view.pre_var.set("0.5")
    view.schedule_compute(now=True)
    assert view.record["pre_ms"] == 0.5


def test_top_panel_shows_taper_and_tapered_signal(brune_app, view):
    view.set_channel(BRUNE_CH)
    labels = [line.get_label() for line in view.ax_time.lines]
    assert "signal window" in labels and "noise window" in labels and "tapered (what is transformed)" in labels
    taper_line = view.ax_taper.lines[0]
    assert taper_line.get_ydata().max() == pytest.approx(1.0) and taper_line.get_ydata().min() == pytest.approx(0.0, abs=1e-9)
    assert view.ax_taper.get_ylim() == (0.0, 1.05) and view.ax_taper.get_ylabel() == "taper"
    sig = next(l for l in view.ax_time.lines if l.get_label() == "signal window").get_ydata()
    tapered = next(l for l in view.ax_time.lines if l.get_label() == "tapered (what is transformed)").get_ydata()
    np.testing.assert_allclose(tapered, sig * taper_line.get_ydata())


def test_saved_record_comparison_in_status(brune_app, view):
    calibrated_compute(view)
    assert view.save()
    view.set_channel(TANH_CH)
    view.set_channel(BRUNE_CH)
    assert "matches the saved record" in view.status_var.get()
    view.post_var.set("1.5")
    view.schedule_compute(now=True)
    assert "differs from the saved record" in view.status_var.get()


def test_source_parameters_follow_automatically(brune_app, view):
    calibrated_compute(view)
    set_medium(view)
    view.schedule_compute(now=True)
    assert "source parameters computed" in view.status_var.get()
    assert view.record["source"] is not None and view.record["source"]["valid"]


# ------------------------------------------------------- window dragging
def test_noise_window_entries_reach_the_analysis_and_the_record(brune_app, view):
    view.set_channel(BRUNE_CH)
    w0 = view.window
    assert w0.i_noise1 == w0.i_trigger - w0.n_pre                     # default: just before the signal
    assert view.record["noise_start_ms"] is None and view.record["noise_end_ms"] is None
    view.noise_start_var.set("-4")
    view.noise_end_var.set("-2.5")
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
    view.noise_end_var.set("")
    view.noise_start_var.set("")
    view.schedule_compute(now=True)
    assert view.record["noise_start_ms"] is None


def test_dragging_the_windows_updates_settings_and_recomputes(brune_app, view):
    view.set_channel(BRUNE_CH)
    e0 = view.current_edges_ms()
    assert e0["signal_start"] == pytest.approx(-0.5) and e0["signal_end"] == pytest.approx(2.0)
    assert e0["noise_end"] == pytest.approx(-0.5) and e0["noise_start"] == pytest.approx(-3.0)
    assert set(view.handles) == {"noise_start", "noise_end", "signal_start", "signal_end"}
    # the post edge
    view.begin_drag("signal_end", e0["signal_end"])
    view.drag_to(2.5)
    assert view.post_var.get() == "2.5" and view.handles["signal_end"].get_xdata()[0] == pytest.approx(2.5)
    view.end_drag()
    assert view.record["post_ms"] == 2.5 and view.window.n_post == int(round(2.5e-3 * brune_app.fs))
    # the pre edge cannot cross the trigger; the automatic noise window (blank entries) follows it
    view.begin_drag("signal_start", -0.5)
    e = view.drag_to(0.3)
    assert e["signal_start"] == 0.0 and view.pre_var.get() == "0"
    e = view.drag_to(-1.0)
    assert e["signal_start"] == pytest.approx(-1.0) and e["noise_end"] == pytest.approx(-1.0)
    assert (view.noise_start_var.get(), view.noise_end_var.get()) == ("", "")
    view.end_drag()
    assert view.record["pre_ms"] == 1.0 and view.record["noise_start_ms"] is None
    assert view.window.i_noise1 == view.window.i_trigger - view.window.n_pre
    # the whole signal window slides, keeping its length and the trigger inside
    e1 = view.current_edges_ms()
    view.begin_drag("signal", 0.5)
    e = view.drag_to(0.5 - 0.25)
    assert e["signal_end"] - e["signal_start"] == pytest.approx(e1["signal_end"] - e1["signal_start"])
    assert e["signal_start"] == pytest.approx(e1["signal_start"] - 0.25)
    e = view.drag_to(0.5 + 5.0)
    assert e["signal_start"] == 0.0                                   # stops at the trigger
    view.end_drag()
    assert view.record["pre_ms"] == 0.0 and view.record["post_ms"] == pytest.approx(e1["signal_end"] - e1["signal_start"])
    # a noise edge dragged makes the noise window explicit; its end never passes the signal start
    view.pre_var.set("0.5"); view.post_var.set("2.0")
    view.schedule_compute(now=True)
    e2 = view.current_edges_ms()
    view.begin_drag("noise_end", e2["noise_end"])
    e = view.drag_to(1.0)
    assert e["noise_end"] == pytest.approx(e2["signal_start"])
    view.drag_to(-1.0)
    view.end_drag()
    assert view.record["noise_end_ms"] == -1.0 and view.window.i_noise1 == view.window.i_trigger - int(round(1e-3 * brune_app.fs))
    assert view.record["noise_start_ms"] == pytest.approx(e2["noise_start"])      # the start stayed put
    # explicit noise: the signal start now stops at it instead of pushing it
    view.begin_drag("signal_start", -0.5)
    e = view.drag_to(-3.0)
    assert e["signal_start"] == pytest.approx(-1.0) and e["noise_end"] == pytest.approx(-1.0)
    view.end_drag()
    e3 = view.current_edges_ms()
    view.begin_drag("noise", e3["noise_start"] + 0.5)
    e = view.drag_to(e3["noise_start"] + 0.5 - 1.0)
    assert e["noise_end"] - e["noise_start"] == pytest.approx(e3["noise_end"] - e3["noise_start"])
    assert e["noise_start"] == pytest.approx(e3["noise_start"] - 1.0)
    view.end_drag()
    assert view.record["noise_start_ms"] == pytest.approx(e3["noise_start"] - 1.0)
    # a click without a move changes nothing
    before = dict(view.record)
    view.begin_drag("signal_end", view.current_edges_ms()["signal_end"])
    view.end_drag()
    assert view.record["post_ms"] == before["post_ms"]


def test_mouse_events_grab_edges_and_bodies(brune_app, view):
    from matplotlib.backend_bases import MouseEvent
    view.set_channel(BRUNE_CH)
    view.canvas.draw()
    e0 = view.current_edges_ms()
    ax = view.ax_time

    def mouse(name, x_ms, dx_px=0.0):
        px, py = ax.transData.transform((x_ms, float(np.mean(ax.get_ylim()))))
        return MouseEvent(name, view.canvas, px + dx_px, py, button=1)

    press = mouse("button_press_event", e0["signal_end"], dx_px=3)     # a few pixels off still grabs the edge
    assert press.inaxes is ax                                           # not the taper axes on top of it
    view.on_press(press)
    assert view._drag["what"] == "signal_end"
    view.on_motion(mouse("motion_notify_event", e0["signal_end"] + 1.0))
    assert float(view.post_var.get()) == pytest.approx(3.0, abs=0.02)      # matplotlib keeps whole pixels
    view.on_release(mouse("button_release_event", e0["signal_end"] + 1.0))
    assert view._drag is None and view.record["post_ms"] == pytest.approx(3.0, abs=0.02)
    e1 = view.current_edges_ms()
    view.on_press(mouse("button_press_event", e1["noise_start"], dx_px=-3))
    assert view._drag["what"] == "noise_start"
    view.on_release(mouse("button_release_event", e1["noise_start"]))
    view.on_press(mouse("button_press_event", 0.5 * (e1["signal_start"] + e1["signal_end"])))
    assert view._drag["what"] == "signal"
    view.on_release(mouse("button_release_event", 0.5 * (e1["signal_start"] + e1["signal_end"])))
    view.on_press(mouse("button_press_event", 0.5 * (e1["noise_start"] + e1["noise_end"])))
    assert view._drag["what"] == "noise"
    view.on_release(mouse("button_release_event", 0.5 * (e1["noise_start"] + e1["noise_end"])))
    view.toolbar.zoom()                                                 # with the toolbar's zoom active nothing is grabbed
    view.on_press(mouse("button_press_event", e1["signal_end"]))
    assert view._drag is None
    view.toolbar.zoom()


def shown(view):
    view.deiconify()
    view.update()
    view.canvas.draw()
    view.update()
    if not view.canvas.get_tk_widget().winfo_ismapped():
        pytest.skip("window not mapped: Tk mouse events need a display")
    return view.canvas.get_tk_widget()


def tk_xy(view, x_ms):
    px, py = view.ax_time.transData.transform((x_ms, float(np.mean(view.ax_time.get_ylim()))))
    return int(round(px)), int(round(view.canvas.figure.bbox.height - py))


def tk_drag(widget, x0, y0, x1, y1):
    widget.event_generate("<Button-1>", x=x0, y=y0)
    widget.update()
    for f in (0.3, 0.6, 1.0):
        widget.event_generate("<Motion>", x=int(x0 + f * (x1 - x0)), y=int(y0 + f * (y1 - y0)), state=0x100)
        widget.update()
    widget.event_generate("<ButtonRelease-1>", x=x1, y=y1)
    widget.update()


def test_real_mouse_drags_every_window_edge(brune_app, view):
    view.set_channel(BRUNE_CH)
    widget = shown(view)
    e = view.current_edges_ms()
    tk_drag(widget, *tk_xy(view, e["signal_end"]), *tk_xy(view, 3.0))
    assert float(view.post_var.get()) == pytest.approx(3.0, abs=0.05)
    e = view.current_edges_ms()                            # the automatic noise window followed: re-read the edges
    assert e["noise_start"] == pytest.approx(-4.0, abs=0.05)
    tk_drag(widget, *tk_xy(view, e["noise_start"]), *tk_xy(view, -5.0))      # the noise becomes explicit
    assert float(view.noise_start_var.get()) == pytest.approx(-5.0, abs=0.05)
    e = view.current_edges_ms()
    tk_drag(widget, *tk_xy(view, e["noise_end"]), *tk_xy(view, -1.5))
    assert float(view.noise_end_var.get()) == pytest.approx(-1.5, abs=0.05)
    e = view.current_edges_ms()
    tk_drag(widget, *tk_xy(view, e["signal_start"]), *tk_xy(view, -1.0))
    assert float(view.pre_var.get()) == pytest.approx(1.0, abs=0.05)
    assert view.record["pre_ms"] == pytest.approx(1.0, abs=0.05) and view.record["noise_end_ms"] == pytest.approx(-1.5, abs=0.05)


def test_legacy_event_level_map_is_read_and_moved_on_save(brune_app, view):
    """Files written before 2026-10-12 keep their records at event['pzt_spectrum'];
    they are read for the matching record and move under it on the next save."""
    calibrated_compute(view)
    rec = copy.deepcopy(view.record)
    view.event["pzt_spectrum"] = {"version": 1, "channels": {"ch3": dict(rec, channel=3), "ch7": dict(rec, channel=7)}}
    assert set(view.saved_channels()) == {3, 7}
    view.set_channel(7)
    assert "saved record" in view.status_var.get()
    assert view.save()
    store = view.event["strain"]["spectrum"]["sets"]["default"]["channels"]
    assert set(store) == {"ch3", "ch7"} and "pzt_spectrum" not in view.event
    assert "under strain" in view.status_var.get()



# ------------------------------------------- calibration table and experiment settings
def test_calibration_table_is_stored_and_used_without_the_csv(brune_app, view, tmp_path):
    assert view.calibration_note_var.get().startswith("no calibration loaded")
    calibrated_compute(view)
    note = view.calibration_note_var.get()
    assert note.startswith("loaded: sensor.csv") and "25 rows" in note and "V/m" in note
    table = view.record["calibration"]
    assert len(table["frequency_hz"]) == 25 and table["unit"] == "V/m" and table["source"] == brune_app.calibration_csv
    assert view.save()
    sensors = brune_app.data_manager.data["analysis"]["pzt_spectrum"]["sets"]["default"]["sensors"]
    assert set(sensors) == {"ch3"} and sensors["ch3"]["calibration"]["unit"] == "V/m"
    assert sensors["ch3"]["calibration_path"] == brune_app.calibration_csv
    import os
    os.remove(brune_app.calibration_csv)                  # the CSV is gone: the stored table serves
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
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
    finally:
        v2.on_close()


def test_every_control_reopens_from_the_experiment_settings(brune_app, view):
    view.set_channel(BRUNE_CH)
    view.pre_var.set("0.4"); view.post_var.set("1.5"); view.snr_var.set("2"); view.bins_var.set("25")
    view.fmin_var.set("1500"); view.t_star_var.set("1e-6"); view.taper_combobox.set("tukey")
    view.calibration_path_var.set(brune_app.calibration_csv); view.unit_combobox.set("V/m")
    assert view.compute(), view.status_var.get()
    set_medium(view, "S", 0.7)
    view.free_surface_var.set("1.3")
    assert view.save()                                    # the source was never computed, still everything is kept
    a = brune_app.data_manager.data["analysis"]
    assert a["medium"] == {"rho_kg_m3": RHO, "vp_m_s": VP, "vs_m_s": VS}
    settings = a["pzt_spectrum"]["sets"]["default"]["settings"]
    assert settings["pre_ms"] == 0.4 and settings["phase"] == "S" and settings["radiation_coefficient"] == 0.7
    assert settings["snr_min"] == 2.0 and settings["t_star_s"] is None if "t_star_s" in settings else True
    sensor = a["pzt_spectrum"]["sets"]["default"]["sensors"]["ch3"]
    assert sensor["distance_m"] == DIST and sensor["free_surface_factor"] == 1.3 and sensor["t_star_s"] == 1e-6
    assert view.record["source_inputs"]["free_surface_factor"] == 1.3
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, 2)                 # another event, nothing saved there
    try:
        assert "saved record" not in v2.status_var.get()
        assert (v2.pre_var.get(), v2.post_var.get(), v2.snr_var.get(), v2.bins_var.get()) == ("0.4", "1.5", "2", "25")
        assert v2.fmin_var.get() == "1500" and v2.taper_combobox.get() == "tukey"
        assert (v2.rho_var.get(), v2.c_var.get(), v2.vs_var.get()) == ("2700", "6000", "3500")
        assert v2.phase_combobox.get() == "S" and v2.radiation_var.get() == "0.7"
        assert v2.current_channel() == 0                   # no record anywhere: channel 0, whose sensor has no entry
        v2.set_channel(BRUNE_CH)                           # this sensor's own values
        assert (v2.distance_var.get(), v2.free_surface_var.get()) == ("0.1", "1.3")
        assert float(v2.t_star_var.get()) == pytest.approx(1e-6) and v2.unit_combobox.get() == "V/m"
        assert v2.calibration_note_var.get().startswith("loaded: sensor.csv")
        assert v2.record is not None and v2.record["spectrum"]["amp_units"] == "m*s"
    finally:
        v2.on_close()


def test_experiment_settings_survive_an_hdf5_round_trip(brune_app, view, tmp_path, app_from_file):
    calibrated_compute(view)
    set_medium(view, "P")
    assert view.compute_source() and view.save()
    path = tmp_path / "exp.h5"
    brune_app.data_manager.save_file(path)
    app2 = app_from_file(path)
    v2 = PZTSpectrumView(app2, 0, EVENT)
    try:
        assert v2.record["spectrum"]["amp_units"] == "m*s" and v2.record["source"]["valid"]
        assert v2.calibration_note_var.get().startswith("loaded: sensor.csv")
        assert (v2.rho_var.get(), v2.distance_var.get()) == ("2700", "0.1")
        v3 = PZTSpectrumView(app2, 0, 2)
        try:
            assert v3.rho_var.get() == "2700" and v3.phase_combobox.get() == "P"
            v3.set_channel(BRUNE_CH)
            assert v3.record["spectrum"]["amp_units"] == "m*s"
        finally:
            v3.on_close()
    finally:
        v2.on_close()


def test_coincident_edges_are_told_apart_by_the_drag_direction(brune_app, view):
    from matplotlib.backend_bases import MouseEvent
    from labquake_explorer.ui.views.pzt_spectrum_view import TIE
    view.set_channel(BRUNE_CH)
    view.canvas.draw()
    ax = view.ax_time

    def mouse(name, x_ms):
        px, py = ax.transData.transform((x_ms, float(np.mean(ax.get_ylim()))))
        return MouseEvent(name, view.canvas, px, py, button=1)

    e0 = view.current_edges_ms()
    assert e0["noise_end"] == e0["signal_start"]           # automatic noise: it ends where the signal starts
    view.on_press(mouse("button_press_event", e0["signal_start"]))
    assert view._drag["what"] == "signal_start"            # the noise follows anyway
    view.on_release(mouse("button_release_event", e0["signal_start"]))
    view.noise_start_var.set("-3"); view.noise_end_var.set("-0.5")   # explicit, still glued to the signal
    view.schedule_compute(now=True)
    view.canvas.draw()
    view.on_press(mouse("button_press_event", -0.5))
    assert view._drag["what"] == TIE
    e = view.drag_to(-1.0)                                 # left: the noise end moves, the signal start stays
    assert e["noise_end"] == pytest.approx(-1.0) and e["signal_start"] == pytest.approx(-0.5)
    view.end_drag()
    view.noise_end_var.set("-0.5")
    view.schedule_compute(now=True)
    view.canvas.draw()
    view.on_press(mouse("button_press_event", -0.5))
    e = view.drag_to(-0.2)                                 # right: the signal start moves
    assert e["signal_start"] == pytest.approx(-0.2) and e["noise_end"] == pytest.approx(-0.5)
    view.end_drag()
    assert view.record["pre_ms"] == pytest.approx(0.2) and view.record["noise_end_ms"] == -0.5


def test_a_noise_window_too_short_to_use_does_not_poison_the_entries(brune_app, view):
    """An explicit noise window that the analysis cannot use (shorter than a
    quarter of the signal window) must not be written back as a degenerate
    pair by the next drag; and a recompute that fails leaves the windows on
    the top panel, so they can be dragged back."""
    view.set_channel(BRUNE_CH)
    view.noise_start_var.set("-1.0"); view.noise_end_var.set("-0.6")
    view.post_var.set("10")                               # the noise (0.4 ms) is now under a quarter of the window
    view.schedule_compute(now=True)
    assert view.window is not None and not view.window.noise_available
    assert set(view.handles) == {"signal_start", "signal_end"}
    e = view.current_edges_ms()
    view.begin_drag("signal_start", e["signal_start"])
    view.drag_to(0.0)                                     # to the trigger: pre 0
    view.end_drag()
    assert (view.noise_start_var.get(), view.noise_end_var.get()) == ("-1.0", "-0.6")   # left as typed
    assert view.pre_var.get() == "0" and view.record is not None
    # a recompute that fails (entries that give no window) keeps the top panel and its drags
    view.noise_end_var.set("-1.0")
    view.schedule_compute(now=True)
    assert view.record is None and "noise start must be before noise end" in view.status_var.get()
    assert view.window is not None and view.ax_time.lines and "not recomputed" in view.ax_spec.get_title()
    view.begin_drag("signal_end", view.current_edges_ms()["signal_end"])
    view.drag_to(5.0)
    view.end_drag()
    assert view.post_var.get() == "5" and view.record is None      # the noise entries are still wrong
    view.noise_end_var.set("-0.6")
    view.schedule_compute(now=True)
    assert view.record is not None and view.record["post_ms"] == 5.0


def test_n_max_bounds_the_free_exponent_and_is_kept(brune_app, view):
    view.set_channel(BRUNE_CH)
    assert view.n_max_var.get() == "4" and view.record["n_max"] == 4.0
    assert view.record["fit"]["n_bounds"] == [1.0, 4.0] and view.fit.n_bounds == (1.0, 4.0)
    view.n_fixed_var.set(False)
    view.n_max_var.set("8")
    view.schedule_compute(now=True)
    assert view.fit.valid and not view.fit.n_fixed and view.fit.n_bounds == (1.0, 8.0)
    assert view.record["n_max"] == 8.0 and view.record["fit"]["n_bounds"] == [1.0, 8.0]
    assert view.fit.n == pytest.approx(2.0, abs=0.2) and not view.fit.at_bounds["n"]
    view.n_max_var.set("1.5")                       # a ceiling under the truth: n sits on it, flagged for these bounds
    view.schedule_compute(now=True)
    assert view.fit.n == pytest.approx(1.5, abs=0.02) and view.fit.at_bounds["n"]
    assert "at bound: n" in view.result_vars["flags"].get()
    view.n_max_var.set("1")                         # not above n min
    view.schedule_compute(now=True)
    assert view.record is None and "n max" in view.status_var.get()
    view.n_max_var.set("8")
    view.schedule_compute(now=True)
    assert view.save()
    assert brune_app.data_manager.data["analysis"]["pzt_spectrum"]["sets"]["default"]["settings"]["n_max"] == 8.0
    rec = copy.deepcopy(view.record)
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, 2)           # another event: the experiment's settings
    try:
        assert v2.n_max_var.get() == "8" and not v2.n_fixed_var.get()
        rec.pop("n_max")                            # a record written before the key existed reads as the default
        v2.apply_record(rec)
        assert v2.n_max_var.get() == "4"
    finally:
        v2.on_close()


def test_noise_window_after_the_signal_in_the_view(brune_app, view):
    from matplotlib.backend_bases import MouseEvent
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
    assert set(view.handles) == set(HANDLES)
    assert view.ax_time.get_xlim()[1] > 4.5                         # the plot shows the noise window
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
    e = view.current_edges_ms()
    view.begin_drag("noise_end", e["noise_end"])
    view.drag_to(6.0)
    view.end_drag()
    assert view.record["noise_end_ms"] == 6.0
    # the noise body slides after the signal, never onto it
    e = view.current_edges_ms()
    view.begin_drag("noise", e["noise_start"] + 1.0)
    e1 = view.drag_to(e["noise_start"] + 1.0 - 5.0)
    assert e1["noise_start"] == pytest.approx(e["signal_end"])
    assert e1["noise_end"] - e1["noise_start"] == pytest.approx(e["noise_end"] - e["noise_start"])
    view.end_drag()
    assert view.record["noise_start_ms"] == pytest.approx(e["signal_end"])
    # coincident edges after the signal: the drag direction tells them apart
    view.canvas.draw()
    ax = view.ax_time

    def press(x_ms):
        px, py = ax.transData.transform((x_ms, float(np.mean(ax.get_ylim()))))
        view.on_press(MouseEvent("button_press_event", view.canvas, px, py, button=1))

    e = view.current_edges_ms()
    assert e["noise_start"] == pytest.approx(e["signal_end"])
    press(e["signal_end"])
    assert view._drag["what"] == TIE_AFTER
    e1 = view.drag_to(e["signal_end"] + 1.0)                         # right: the noise start moves
    assert e1["noise_start"] == pytest.approx(e["signal_end"] + 1.0) and e1["signal_end"] == pytest.approx(e["signal_end"])
    view.end_drag()
    view.noise_start_var.set(str(e["signal_end"]))
    view.schedule_compute(now=True)
    view.canvas.draw()
    press(e["signal_end"])
    e1 = view.drag_to(e["signal_end"] - 0.5)                         # left: the signal end moves
    assert e1["signal_end"] == pytest.approx(e["signal_end"] - 0.5) and e1["noise_start"] == pytest.approx(e["signal_end"])
    view.end_drag()
    assert view.record["post_ms"] == pytest.approx(e["signal_end"] - 0.5)
    # automatic noise with no room before the signal is taken after it, says so, and follows the signal end
    view.noise_start_var.set(""); view.noise_end_var.set("")
    view.pre_var.set("9.5"); view.post_var.set("0.5")                # the trace starts 10 ms before the trigger
    view.schedule_compute(now=True)
    w = view.window
    assert w.noise_region == "post" and w.i_noise0 == w.i_trigger + w.n_post
    assert "after the signal" in view.status_var.get() and view.record["noise_start_ms"] is None
    e = view.current_edges_ms()
    view.begin_drag("signal_end", e["signal_end"])
    e1 = view.drag_to(1.0)
    assert e1["signal_end"] == pytest.approx(1.0) and e1["noise_start"] == pytest.approx(1.0)
    assert (view.noise_start_var.get(), view.noise_end_var.get()) == ("", "")
    view.end_drag()
    assert view.window.i_noise0 == view.window.i_trigger + view.window.n_post and view.record["noise_start_ms"] is None
    # with room before again, the noise goes back before the signal
    view.pre_var.set("0.5"); view.post_var.set("2.0")
    view.schedule_compute(now=True)
    assert view.window.noise_region == "pre" and "after the signal" not in view.status_var.get()
    # a real mouse drag of the end of a noise window after the signal
    view.noise_start_var.set("2"); view.noise_end_var.set("4.5")
    view.schedule_compute(now=True)
    widget = shown(view)
    e = view.current_edges_ms()
    tk_drag(widget, *tk_xy(view, e["noise_end"]), *tk_xy(view, 6.0))
    assert float(view.noise_end_var.get()) == pytest.approx(6.0, abs=0.05)
    assert view.record["noise_end_ms"] == pytest.approx(6.0, abs=0.05) and view.record["noise_start_ms"] == 2.0


def test_result_sets_keep_routes_side_by_side(brune_app, view):
    """Records saved in different result sets do not overwrite each other; each
    set keeps its own experiment settings; a window opens on the set last saved."""
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
    assert view.save()
    assert "set 'EGF'" in view.status_var.get()
    store = view.event["strain"]["spectrum"]
    assert store["version"] == 2 and set(store["sets"]) == {"default", "EGF"}
    assert store["sets"]["default"]["channels"]["ch3"]["calibration_unit"] == "V/m"
    assert store["sets"]["EGF"]["channels"]["ch3"]["calibration_unit"] == "V/(N m)"
    assert store["sets"]["EGF"]["channels"]["ch3"]["result_set"] == "EGF"
    a = brune_app.data_manager.data["analysis"]["pzt_spectrum"]
    assert a["current_set"] == "EGF" and set(a["sets"]) == {"default", "EGF"}
    assert a["sets"]["default"]["sensors"]["ch3"]["calibration_unit"] == "V/m"
    assert a["sets"]["EGF"]["sensors"]["ch3"]["calibration_unit"] == "V/(N m)"
    assert a["sets"]["EGF"]["sensors"]["ch3"]["calibration_divisor"] == 2.0
    assert "settings" not in a and "sensors" not in a
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
    # a new window opens on the set last saved (EGF), with that set's sensor values on another event
    v2 = PZTSpectrumView(brune_app, 0, 2)
    try:
        assert v2.result_set == "EGF" and v2.set_combobox.get() == "EGF"
        assert list(v2.set_combobox["values"]) == ["default", "EGF"]
        v2.set_channel(BRUNE_CH)
        assert v2.unit_combobox.get() == "V/(N m)" and v2.calibration_divisor_var.get() == "2"
        v2.set_combobox.set("default")
        v2.on_set_changed()
        assert v2.unit_combobox.get() == "V/m" and v2.calibration_divisor_var.get() == ""
    finally:
        v2.on_close()


def test_version_1_store_and_flat_settings_are_read_as_the_default_set(brune_app, view):
    calibrated_compute(view)
    rec = copy.deepcopy(view.record)
    view.event["strain"]["spectrum"] = {"version": 1, "channels": {"ch3": dict(rec, channel=3), "ch5": dict(rec, channel=5)}}
    data = brune_app.data_manager.data
    data["analysis"] = {"pzt_spectrum": {"settings": {"pre_ms": 0.7, "post_ms": 1.8},
                                         "sensors": {"ch6": {"t_star_s": 2e-6}}}}
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.result_set == "default" and set(v2.saved_channels()) == {3, 5}
        assert v2.current_channel() == 3 and "saved record" in v2.status_var.get()
        assert v2.experiment_settings()["pre_ms"] == 0.7                     # the flat settings
        v2.set_channel(6)
        assert float(v2.t_star_var.get()) == pytest.approx(2e-6)             # the flat sensors entry
        v2.set_channel(3)
        assert v2.save()
        store = v2.event["strain"]["spectrum"]
        assert store["version"] == 2 and "channels" not in store
        assert set(store["sets"]["default"]["channels"]) == {"ch3", "ch5"}
        a = data["analysis"]["pzt_spectrum"]
        assert "settings" not in a and "sensors" not in a and a["current_set"] == "default"
        assert a["sets"]["default"]["sensors"]["ch6"] == {"t_star_s": 2e-6}  # moved
        assert a["sets"]["default"]["sensors"]["ch3"]["calibration_unit"] == "V/m"
        assert a["sets"]["default"]["settings"]["pre_ms"] == 0.5             # the saved record's controls
    finally:
        v2.on_close()


def test_moment_unit_greys_the_medium_entries_and_takes_m0_from_the_plateau(brune_app, view, tmp_path):
    view.set_channel(BRUNE_CH)
    csv = tmp_path / "psi.csv"
    f = np.geomspace(100.0, 100e3, 25)
    csv.write_text("Frequency (Hz),Mean_Inst_App_Psi\n" + "\n".join(f"{fi:.6g},{2 * GAIN}" for fi in f) + "\n")
    view.calibration_path_var.set(str(csv))
    view.unit_combobox.set("V/(N m)")
    view.calibration_divisor_var.set("2")                     # Psi / C_FM, say
    assert view.compute(), view.status_var.get()
    assert view.spectrum.amp_units == "N*m*s" and view.fit.valid
    assert view.fit.omega0 == pytest.approx(OMEGA0_V / GAIN, rel=0.05)
    assert all(entry.instate(["disabled"]) for entry in view.medium_entries.values())
    assert "M0 = Omega0" in view.source_hint_var.get()
    assert "gain / 2" in view.calibration_note_var.get()
    assert view.record["calibration_divisor"] == 2.0 and view.record["calibration"]["gain_divisor"] == 2.0
    # only the phase and Vs are needed
    view.rho_var.set(""); view.c_var.set(""); view.distance_var.set("")
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
    # the unit and the divisor come back with the saved record
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.current_channel() == BRUNE_CH
        assert v2.unit_combobox.get() == "V/(N m)" and v2.calibration_divisor_var.get() == "2"
        assert v2.record["spectrum"]["amp_units"] == "N*m*s"
        assert all(entry.instate(["disabled"]) for entry in v2.medium_entries.values())
    finally:
        v2.on_close()


def test_estimator_and_ensemble_in_the_view(brune_app, view, tmp_path):
    f = np.geomspace(100.0, 100e3, 25)
    alt = []
    for name, gain in (("alt1.csv", 1.1 * GAIN), ("alt2.csv", 0.9 * GAIN)):
        (tmp_path / name).write_text("frequency_hz,gain\n" + "\n".join(f"{fi:.6g},{gain}" for fi in f) + "\n")
        alt.append(str(tmp_path / name))
    calibrated_compute(view)
    assert view.estimator_combobox.get() == "rms" and view.record["estimator"] == "rms"
    assert view.record["ensemble"] is None and not view.record["ensemble_on"]
    assert view.record["ensemble_fmins_hz"] == list(se.DEFAULT_FMINS_HZ) and view.record["ensemble_fmaxs_hz"] == list(se.DEFAULT_FMAXS_HZ)
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
    assert e["n_members"] == 12 and e["n_valid"] == 12 and e["bands_hz"] == [[1000.0, 20000.0], [1000.0, 50000.0],
                                                                             [2000.0, 20000.0], [2000.0, 50000.0]]
    assert e["calibrations"][0] == brune_app.calibration_csv and e["calibrations"][1:] == alt
    st = e["stats"]
    assert st["fc"]["p16"] <= fc_logmean * 1.05 and st["fc"]["p84"] >= fc_logmean * 0.95
    m0 = view.record["source"]["seismic_moment_nm"]
    assert st["seismic_moment_nm"]["min"] == pytest.approx(m0 / 1.1, rel=0.1)      # the 1.1 x gain table
    assert st["seismic_moment_nm"]["max"] == pytest.approx(m0 / 0.9, rel=0.1)
    assert e["n_resolved"] == 6 and e["stats_resolved"]["fc"]["n"] == 6        # fmax 20 kHz < 2 fc: those 6 are band-limited
    assert [m for m, lim in zip(e["members"]["fmax_hz"], e["members"]["band_limited"]) if lim] == [20000.0] * 6
    assert "fc" in view.result_vars["ensemble"].get() and "6 resolved of 12 valid" in view.result_vars["ensemble"].get()
    assert "fc 16-84 %" in view.ensemble_text_var.get() and "ensemble:" in view.status_var.get()
    assert view.record["ensemble_calibration_paths"] == alt
    # a recompute clears it; Save runs it again when the box is ticked, and stores the inputs with the record
    view.post_var.set("2.5")
    view.schedule_compute(now=True)
    assert view.record["ensemble"] is None and view.ensemble_text_var.get() == "not run"
    view.ensemble_var.set(True)
    assert view.compute_source()
    assert view.save(), view.status_var.get()
    rec = view.event["strain"]["spectrum"]["sets"]["default"]["channels"]["ch3"]
    assert rec["ensemble"]["n_members"] == 12 and rec["ensemble_on"] is True
    assert rec["ensemble_fmins_hz"] == [1000.0, 2000.0] and rec["ensemble_calibration_paths"] == alt
    sensors = brune_app.data_manager.data["analysis"]["pzt_spectrum"]["sets"]["default"]["sensors"]["ch3"]
    assert sensors["ensemble_calibration_paths"] == alt
    settings = brune_app.data_manager.data["analysis"]["pzt_spectrum"]["sets"]["default"]["settings"]
    assert settings["estimator"] == "logmean" and settings["ensemble_on"] is True and settings["ensemble_fmaxs_hz"] == [20000.0, 50000.0]
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
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.estimator_combobox.get() == "logmean" and v2.ensemble_var.get()
        assert v2.ensemble_fmins_var.get() == "1000, 2000" and v2.ensemble_calibrations_var.get() == ";".join(alt)
        assert "6 resolved of 12 valid" in v2.result_vars["ensemble"].get()
        from labquake_explorer.ui.views.pzt_spectrum_view import ensemble_stats
        stored = v2.record["ensemble"]
        assert ensemble_stats(stored)[1] == "resolved"
        assert ensemble_stats({"stats": stored["stats"], "stats_resolved": {"fc": {"n": 0}}})[1] == "valid"
        assert ensemble_stats({"stats": stored["stats"]})[1] == "valid"           # written before the resolved set existed
    finally:
        v2.on_close()


def test_q_table_per_sensor(brune_app, view, tmp_path):
    q = tmp_path / "q.csv"
    q.write_text("freq_kHz,Qp_inv_median\n1,0.05\n100,0.05\n")
    calibrated_compute(view)
    f_plain, amp_plain = view.spectrum.f_binned.copy(), view.spectrum.amp_binned.copy()
    assert view.record["attenuation"] is None and view.record["q_csv_path"] == ""
    assert "no Q" in view.q_note_var.get()
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
    lifted = view.spectrum.amp_binned / amp_plain             # the correction lifts the spectrum by exp(pi f T Q^-1)
    np.testing.assert_allclose(lifted, np.exp(np.pi * f_plain * 5e-5 * 0.05), rtol=1e-2)   # applied before binning
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
    sensor = brune_app.data_manager.data["analysis"]["pzt_spectrum"]["sets"]["default"]["sensors"]["ch3"]
    assert sensor["q_travel_time_s"] == 1e-4 and sensor["q_hold_ends"] is True and sensor["attenuation"]["n_rows"] == 2
    # cleared: no correction
    view.clear_attenuation()
    assert view.record is not None and view.spectrum.meta["attenuation"] is None and view.attenuation_table is None
    # reopening restores the table, travel time and flag from the record; another channel of the sensor set too
    view.on_close()
    v2 = PZTSpectrumView(brune_app, 0, EVENT)
    try:
        assert v2.current_channel() == BRUNE_CH
        assert v2.q_travel_var.get() == "0.0001" and v2.q_hold_var.get() and v2.attenuation_table["n_rows"] == 2
        assert v2.spectrum.meta["attenuation"]["travel_time_s"] == 1e-4 and "matches the saved record" in v2.status_var.get()
        v2.set_channel(TANH_CH)                           # no record, no sensor entry: the loaded table carries over
        assert v2.attenuation_table["n_rows"] == 2 and v2.spectrum.meta["attenuation"]["travel_time_s"] == 1e-4
    finally:
        v2.on_close()
