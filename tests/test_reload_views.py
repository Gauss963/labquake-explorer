"""Reopen every result-bearing view after a real Save As .h5 / Load cycle.

The DataManager tests round-trip dicts; the view tests feed in-memory dicts
back into the views.  This is the user's actual path: results written by the
views, ``DataManager.save_file`` to HDF5, ``load_file`` into a FRESH
application, then every view reopened on the loaded data (stored lists come
back as numpy arrays, bools as ints, None entries disappear)."""
import numpy as np
import pytest

from labquake_explorer.analysis import spectrum as sp
from labquake_explorer.ui.views import (
    CZMFitterView, EventAnalyzerView, InterEventView, PZTSpectrumView, SourceScalingView,
)
from labquake_explorer.ui.views.source_scaling_view import ALL_CHANNELS, Y_FC, Y_RADIUS

PZT_CH = 3


def _brune_on_every_event(app, channel=PZT_CH):
    events = app.data_manager.get_data("runs/[0]/events")
    rng = np.random.default_rng(11)
    for j, event in enumerate(events):
        original = event["strain"]["original"]
        time = np.asarray(original["time"], dtype=float)
        raw = np.asarray(original["raw"], dtype=float) + rng.normal(0.0, 5e-5, np.shape(original["raw"]))
        fs = sp.sampling_rate(time)
        amp = 2.0 ** j
        _, pulse = sp.brune_pulse(fs, time.size, amp, 10e3 / (1.0 + 0.3 * j), float(event["event_time"]) - time[0])
        raw[channel] = pulse + rng.normal(0.0, amp / np.sqrt(1000.0) * fs / np.sqrt(500.0), time.size)
        original["raw"] = raw
    return events


@pytest.fixture
def populated(app_with_strain, tmp_path):
    """Results saved through the views themselves, plus the expected numbers."""
    app = app_with_strain
    events = _brune_on_every_event(app)
    csv = tmp_path / "sensor.csv"
    f = np.geomspace(100.0, 100e3, 25)
    csv.write_text("frequency_hz,gain\n" + "\n".join(f"{fi:.6g},2.0" for fi in f) + "\n")
    expected = {}

    # PZT spectrum + source on every event (with a typed free-surface factor)
    for j in range(len(events)):
        pv = PZTSpectrumView(app, 0, j)
        try:
            pv.set_channel(PZT_CH)
            pv.calibration_path_var.set(str(csv))
            pv.unit_combobox.set("V/m")
            pv.snr_var.set("2")
            assert pv.compute(), pv.status_var.get()
            pv.rho_var.set("2700"); pv.c_var.set("6000"); pv.vs_var.set("3500"); pv.distance_var.set("0.05")
            pv.free_surface_var.set("1.5")
            assert pv.compute_source(), pv.status_var.get()
            assert pv.save()
            expected[("pzt", j)] = (pv.fit.fc, pv.record["source"]["seismic_moment_nm"])
        finally:
            pv.on_close()

    # event analyser, applied to all events
    ea = EventAnalyzerView(app, 0, 1)
    try:
        for point, t_rel in ((0, -4.0), (1, -1.0), (4, -0.001), (5, 0.001)):
            ea.move_point(point, int(np.argmin(np.abs(ea.data_t - ea.event["event_time"] - t_rel))))
        assert ea.apply_to_all_events(confirm=False) == len(events)
        expected["picks"] = list(ea.picked_idx)
        expected["delta_y"] = ea.result["delta_y"]
    finally:
        ea.on_close()

    # CZM parameters on event 1
    cz = CZMFitterView(app, 0, 1)
    try:
        cz.Cf.set(1000.0)
        cz.move_line(1, 0.002)
        cz.save_parameters()
    finally:
        cz.on_close()

    # inter-event metrics with non-default sampling
    iv = InterEventView(app, 0)
    try:
        iv.delay_var.set("0.1")
        iv.width_var.set("0.02")
        iv.compute()
        iv.save()
        expected["creep"] = list(iv.result["creep"])
    finally:
        iv.on_close()

    # source scaling on the PZT channel, radius vs M0
    sv = SourceScalingView(app, 0)
    try:
        sv.set_channel("ch3")
        sv.set_y_variable(Y_RADIUS)
        fit = sv.fit_power_law()
        assert fit.valid, sv.fit_text.get()
        expected["scaling"] = sv.save()
    finally:
        sv.on_close()
    return app, expected


def test_views_reopen_after_hdf5_reload(populated, app_from_file, tmp_path):
    app, expected = populated
    path = tmp_path / "exp.h5"
    app.data_manager.save_file(path)
    app2 = app_from_file(path)
    assert app2.data_manager.data is not app.data_manager.data
    events = app2.data_manager.get_data("runs/[0]/events")
    n_events = len(events)

    # PZT spectrum: channel, numbers and the free-surface factor restored, recomputed to the same result
    pv = PZTSpectrumView(app2, 0, 2)
    try:
        assert pv.current_channel() == PZT_CH
        assert "matches the saved record" in pv.status_var.get()
        assert pv.fit is not None
        fc, m0 = expected[("pzt", 2)]
        assert pv.record["fit"]["fc"] == pytest.approx(fc)
        assert pv.result_vars["fc"].get() == f"{fc:.4g} Hz"
        assert pv.source_vars["m0"].get() == f"{m0:.4g} N m"
        assert pv.free_surface_var.get() == "1.5" and pv.snr_var.get() == "2"
        assert pv.distance_var.get() == "0.05" and pv.unit_combobox.get() == "V/m"
        assert pv.read_parameters()["snr_min"] == 2.0
        # recompute + save on the reloaded data works and keeps the channel map
        pv.rho_var.set("2700")
        assert pv.compute() and pv.compute_source() and pv.save()
        assert set(events[2]["strain"]["spectrum"]["sets"]["default"]["channels"]) == {"ch3"} and "pzt_spectrum" not in events[2]
    finally:
        pv.on_close()

    # event analyser: picks and fields restored
    ea = EventAnalyzerView(app2, 0, 1)
    try:
        assert ea.picked_idx == expected["picks"]
        assert ea.data_x_combo.get() == "displacement" and ea.data_y_combo.get() == "shear_stress"
        assert float(ea.delta_y_text.get()) == pytest.approx(expected["delta_y"], rel=1e-5)
        ea.save_event()
        assert events[1]["event_analysis"]["version"] == 3
    finally:
        ea.on_close()

    # CZM: dict parameters restored
    cz = CZMFitterView(app2, 0, 1)
    try:
        assert cz.Cf.get() == 1000.0
        assert [float(l.get_xdata()[0]) for l in cz.vlines][1] == pytest.approx(0.002)
        cz.save_parameters()
    finally:
        cz.on_close()

    # inter-event: sampling parameters restored, creep recomputed from the reloaded records
    iv = InterEventView(app2, 0)
    try:
        assert iv.delay_var.get() == "0.1" and iv.width_var.get() == "0.02"
        assert iv.slip_combo.get() == "displacement"
        assert iv.result["coseismic_skipped"] == {}
        assert iv.result["creep"] == pytest.approx(expected["creep"], nan_ok=True, rel=1e-9)
        iv.save()
    finally:
        iv.on_close()

    # source scaling: selections and verdict restored (stored lists are arrays now)
    stored = app2.data_manager.get_data("runs/[0]/source_scaling")
    assert np.asarray(stored["event_indices"]).shape == (n_events,)
    sv = SourceScalingView(app2, 0)
    try:
        assert sv.channel_combo.get() == "ch3" and sv.y_combo.get() == Y_RADIUS
        assert sv.include_flagged_var.get() is False and sv.exclude_near_field_var.get() is False
        assert sv.saved_agrees is True, sv.fit_text.get()
        assert sv.fit.exponent == pytest.approx(expected["scaling"]["exponent"], abs=1e-9)
        assert sv.save() is not None
        sv.set_channel(ALL_CHANNELS)
        sv.set_y_variable(Y_FC)
        assert sv.fit_power_law().valid
    finally:
        sv.on_close()

    # a second Save As from the reloaded (and re-saved) data works, and reloads again
    path2 = tmp_path / "exp2.h5"
    app2.data_manager.save_file(path2)
    app3 = app_from_file(path2)
    sv3 = SourceScalingView(app3, 0)
    try:
        assert sv3.saved_agrees is True
    finally:
        sv3.on_close()
    pv3 = PZTSpectrumView(app3, 0, 2)
    try:
        assert "saved record" in pv3.status_var.get()
    finally:
        pv3.on_close()
