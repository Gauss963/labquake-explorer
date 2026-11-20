import numpy as np
import pytest

from labquake_explorer.analysis.scaling import fit_power_law
from labquake_explorer.analysis.source import moment_magnitude
from labquake_explorer.data.channels import channel_array
from labquake_explorer.ui.actions import actions_for
from labquake_explorer.ui.views.pzt_spectrum_view import PZTSpectrumView, channel_key
from labquake_explorer.ui.views.source_scaling_view import (
    ALL_CHANNELS, FLAG_AT_BOUND, FLAG_BAND_LIMITED, FLAG_NEAR_FIELD, FLAG_NOT_CONVERGED,
    RESULT_VERSION, Y_FC, Y_MECH_STRESS_DROP, Y_RADIUS, Y_SLIP, Y_STRESS_DROP,
    SourceScalingView, available_channels, available_sets, channel_flags, combine_channels, excluding_flags,
    fit_pairs, gather_records, mechanical_values, pzt_channel_records,
)

A_FC = 1.0e4           # fc = A_FC * M0**(-1/3)
M0_VALUES = (1.0e-2, 1.0e-1, 1.0, 1.0e1)
STRESS_DROP_PA = 1.0e5
VS = 3000.0
K = 0.32
RHO = 2650.0
VP = 5500.0
THIRD = 1.0 / 3.0


def source_record(m0, fc_scale=1.0, valid=True, phase="P", k=K, fc=None, band_limited=False,
                  at_bound=False, converged=True, far_field_ok=True) -> dict:
    """One saved PZT channel record (fit + source) in the PZT spectrum view's layout."""
    fc = fc_scale * A_FC * m0 ** (-THIRD) if fc is None else fc
    r = k * VS / fc
    constants = {"phase": phase, "rho_kg_m3": RHO, "vp_m_s": VP, "vs_m_s": VS, "distance_m": 0.1, "k": k,
                 "wave_speed_m_s": VP if phase == "P" else VS, "radiation_coefficient": 0.52,
                 "radiation_rms": 0.516, "free_surface_factor": 1.0, "f_plateau_hz": 1e3}
    source = {"phase": phase, "omega0_ms": 1e-12, "fc_hz": fc, "seismic_moment_nm": m0, "mw": moment_magnitude(m0),
              "source_radius_m": r, "rupture_area_m2": np.pi * r ** 2, "stress_drop_pa": STRESS_DROP_PA,
              "kR": 5.0 if far_field_ok else 1.0, "far_field_ok": far_field_ok, "constants": constants,
              "warnings": [], "valid": valid, "reason": "", "version": 1}
    fit = {"omega0": 1e-12, "fc": fc, "n": 2.0, "n_fixed": True, "rms": 0.1, "n_bins": 40, "band_used": [1e3, 3e5],
           "converged": converged, "at_bounds": {"ln_omega0": False, "ln_fc": at_bound, "n": False},
           "band_limited": band_limited, "valid": True, "reason": "", "amp_units": "m*s", "warnings": [], "version": 1}
    return {"version": 1, "channel": 0, "fit": fit, "source": source}


def populate(app, channels=("0", "1"), n=None, mech=False, legacy_keys=False):
    """Exact power-law PZT records (and optionally event_analysis) in run 0, keyed the way
    PZTSpectrumView.save writes them ('ch<k>'), or by the student branch's digit keys."""
    events = app.data_manager.get_data("runs/[0]/events")
    for j, event in enumerate(events[:len(events) if n is None else n]):
        m0 = M0_VALUES[j % len(M0_VALUES)]
        event["pzt_spectrum"] = {"version": 1, "channels": {
            (ch if legacy_keys else channel_key(int(ch))):
                dict(source_record(m0, fc_scale=1.0 + 0.5 * i), channel=int(ch)) for i, ch in enumerate(channels)}}
        if mech:   # constant stress drop (MPa), slip ~ M0^0.5 (um)
            event["event_analysis"] = {"version": 2, "stress_drop": 0.4, "stress_drop_trend": 0.5,
                                       "displacement": 30.0 * m0 ** 0.5, "displacement_trend": 60.0 * m0 ** 0.5}
    return events


@pytest.fixture
def view(app):
    populate(app)
    v = SourceScalingView(app, 0)
    yield v
    if v in app.child_windows:
        v.on_close()


# ---------------------------------------------------------------- helpers
def test_gather_records_reads_every_layout_and_skips_invalid_sources(app):
    run = app.data_manager.get_data("runs/[0]")
    # keys as PZTSpectrumView.save writes them ('ch3'), sorted numerically ('ch2' before 'ch10')
    events = populate(app, channels=("10", "2", "3"))
    assert set(events[0]["pzt_spectrum"]["channels"]) == {"ch10", "ch2", "ch3"}
    assert available_channels(run) == ["ch2", "ch3", "ch10"]
    assert gather_records(run, "ch3")[2]["seismic_moment_nm"] == pytest.approx(1.0)
    assert all(np.isnan(r["seismic_moment_nm"]) for r in gather_records(run, "3"))   # not the same key
    # the legacy digit keys are still read
    populate(app, channels=("3", "12"), legacy_keys=True)
    assert available_channels(run) == ["3", "12"]
    rows = gather_records(run, "3")
    assert [r["event"] for r in rows] == list(range(len(events)))
    assert rows[2]["seismic_moment_nm"] == pytest.approx(1.0) and rows[2]["fc_hz"] == pytest.approx(A_FC)
    assert rows[2]["stress_drop_pa"] == pytest.approx(STRESS_DROP_PA)   # single channel: as recorded
    assert np.isnan(rows[2]["mech_stress_drop"])
    assert rows[2]["flags"] == [] and rows[2]["flagged"] is False
    assert rows[2]["phases"] == ["P"] and rows[2]["constants"][0]["k"] == K
    assert gather_records(run, "12")[2]["fc_hz"] == pytest.approx(1.5 * A_FC)
    # an unknown channel gives NaN PZT fields but keeps the row
    rows_missing = gather_records(run, "99")
    assert len(rows_missing) == len(events) and np.isnan(rows_missing[0]["seismic_moment_nm"])
    # an invalid source is skipped, also when 'valid' is a numpy bool
    events[1]["pzt_spectrum"]["channels"]["3"]["source"]["valid"] = False
    events[2]["pzt_spectrum"]["channels"]["3"]["source"]["valid"] = np.bool_(False)
    assert [np.isfinite(r["seismic_moment_nm"]) for r in gather_records(run, "3")[:3]] == [True, False, False]
    # records under the waveform records and the strain block are gathered too
    t = np.linspace(0, 1, 11)
    event = run["events"][1]
    spectrum = {"version": 1, "channels": {"ch1": dict(source_record(2.0), channel=1)}}
    event["waveform"] = {"elsys": channel_array(np.zeros((2, 11)), ["pzt_1", "pzt_2"], time=t, sample_rate=10.0,
                                                spectrum=spectrum)}
    event["strain"] = {"filename": "x", "time_offset": 0.0, "time": t, "raw": np.zeros((1, 11)),
                       "original": {"time": t, "raw": np.zeros((1, 11))},
                       "spectrum": {"version": 1, "channels": {"ch0": dict(source_record(1.0), channel=0)}}}
    del event["pzt_spectrum"]
    per_channel = pzt_channel_records(event)
    assert set(per_channel) == {"ch0", "ch1"} and per_channel["ch1"]["seismic_moment_nm"] == pytest.approx(2.0)
    assert {"ch0", "ch1"} <= set(available_channels(run))


def test_all_channels_combination(app):
    events = populate(app, channels=("0", "1"), legacy_keys=True)
    run = app.data_manager.get_data("runs/[0]")
    # 'all' combines in log space: fc scales 1.0 and 1.5 -> geometric median sqrt(1.5)
    row = gather_records(run, ALL_CHANNELS)[2]
    assert row["fc_hz"] == pytest.approx(np.sqrt(1.5) * A_FC) and row["seismic_moment_nm"] == pytest.approx(1.0)
    assert row["n_channels"] == 2
    # three channels with different M0 and fc, so per-quantity medians would be inconsistent: the
    # combined M0, r and stress drop obey 7 M0 / (16 r^3) and r = k Vs / fc
    ch = events[2]["pzt_spectrum"]["channels"]
    for name, (m0, fc) in zip(("0", "1", "2"), ((1.0, 1e4), (2.0, 3e4), (0.5, 2e4))):
        ch[name] = source_record(m0, fc=fc)
    row = gather_records(run, ALL_CHANNELS)[2]
    assert row["seismic_moment_nm"] == pytest.approx(1.0)          # geometric median of 1, 2, 0.5
    assert row["fc_hz"] == pytest.approx(2e4)                       # median of 1e4, 3e4, 2e4
    assert row["source_radius_m"] == pytest.approx(K * VS / 2e4)
    assert row["stress_drop_pa"] == pytest.approx(7.0 * 1.0 / (16.0 * (K * VS / 2e4) ** 3))
    assert row["mw"] == pytest.approx(moment_magnitude(1.0)) and row["constants_consistent"] is True
    # a channel lacking a radius: r and stress drop come from the others
    recs = list(pzt_channel_records(events[2]).values())
    recs[0]["source_radius_m"] = float("nan")
    assert combine_channels(recs)["source_radius_m"] == pytest.approx(K * VS / 2e4)
    assert np.isnan(combine_channels([])["seismic_moment_nm"])

    # unflagged channels are preferred: a clipped channel on event 2 is left out
    ch.clear()
    ch.update({"0": source_record(1.0), "1": source_record(1.0, fc=1e3, band_limited=True)})
    row = gather_records(run, ALL_CHANNELS)[2]
    assert row["n_channels"] == 1 and row["flagged"] is False and row["fc_hz"] == pytest.approx(A_FC)
    row = gather_records(run, ALL_CHANNELS, include_flagged=True)[2]
    assert row["n_channels"] == 2 and row["flagged"] is True and row["fc_hz"] == pytest.approx(np.sqrt(A_FC * 1e3))
    # when every channel is flagged the row keeps the flagged values and is itself flagged
    ch["0"] = source_record(1.0, fc=2e3, at_bound=True)
    row = gather_records(run, ALL_CHANNELS)[2]
    assert row["flagged"] is True and row["n_channels"] == 2
    assert FLAG_AT_BOUND in row["flags"] and FLAG_BAND_LIMITED in row["flags"]


def test_channel_flags():
    rec = source_record(1.0)
    assert channel_flags(rec, rec["source"]) == []
    rec = source_record(1.0, band_limited=True, at_bound=True, converged=False, far_field_ok=False)
    assert channel_flags(rec, rec["source"]) == [FLAG_BAND_LIMITED, FLAG_AT_BOUND, FLAG_NOT_CONVERGED, FLAG_NEAR_FIELD]
    # near-field is a warning (kept in flags) but excludes only on request
    assert excluding_flags([FLAG_BAND_LIMITED, FLAG_NEAR_FIELD]) == [FLAG_BAND_LIMITED]
    assert excluding_flags([FLAG_NEAR_FIELD]) == []
    assert excluding_flags([FLAG_NEAR_FIELD], exclude_near_field=True) == [FLAG_NEAR_FIELD]
    assert "band low edge" in FLAG_NEAR_FIELD                    # the frequency used is documented
    near = {"pzt_spectrum": {"channels": {"ch0": source_record(1.0, far_field_ok=False)}}}
    per = pzt_channel_records(near)["ch0"]
    assert per["flags"] == [FLAG_NEAR_FIELD] and per["gating_flags"] == [] and per["flagged"] is False
    per = pzt_channel_records(near, exclude_near_field=True)["ch0"]
    assert per["gating_flags"] == [FLAG_NEAR_FIELD] and per["flagged"] is True
    # numpy bools count; None (not reported) does not flag
    rec = source_record(1.0, far_field_ok=None)
    rec["fit"]["band_limited"] = np.bool_(True)
    assert channel_flags(rec, rec["source"]) == [FLAG_BAND_LIMITED]
    rec = source_record(1.0, far_field_ok=None)
    rec["fit"]["converged"] = None
    assert channel_flags(rec, rec["source"]) == []
    per = pzt_channel_records({"pzt_spectrum": {"channels": {"5": rec}}})["5"]
    assert per["far_field_ok"] is None and per["flagged"] is False


def test_clipped_fc_records_are_gated_out_of_the_fit():
    """fc clipped at a fit bound is a bound, not a measurement: excluded by default."""
    m0 = np.logspace(-2, 1, 8)                       # 3 decades
    fc_true = 1e5 * m0 ** (-1.0 / 3.0)
    clipped = fc_true > 1.2e5
    assert clipped.sum() == 5
    run = {"name": "synthetic", "events": [
        {"pzt_spectrum": {"version": 1, "channels": {"0": source_record(m, fc=min(fc, 1.2e5), band_limited=bool(c),
                                                                         at_bound=bool(c))}}}
        for m, fc, c in zip(m0, fc_true, clipped)]}
    rows = gather_records(run, "0")
    assert [r["flagged"] for r in rows] == list(clipped)
    x, y, idx = fit_pairs(rows, "fc_hz")
    assert list(idx) == [5, 6, 7] and fit_power_law(x, y).exponent == pytest.approx(-THIRD, abs=1e-9)
    x_all, y_all, idx_all = fit_pairs(rows, "fc_hz", include_flagged=True)
    assert list(idx_all) == list(range(8))
    assert abs(fit_power_law(x_all, y_all).exponent) < 0.25   # the clipped fits drag the exponent toward 0
    # 'all' with the same single flagged channel: the row is flagged and still excluded
    rows_all = gather_records(run, ALL_CHANNELS)
    assert [r["flagged"] for r in rows_all] == list(clipped)
    assert list(fit_pairs(rows_all, "fc_hz")[2]) == [5, 6, 7]


# ------------------------------------------------------------------- view
def test_view_opens_fits_and_saves(app, view):
    events = app.data_manager.get_data("runs/[0]/events")
    assert view.title() == "Source Scaling - run00" and view in app.child_windows
    assert list(view.channel_combo["values"]) == [ALL_CHANNELS, "ch0", "ch1"]
    assert view.channel_combo.get() == ALL_CHANNELS and view.y_combo.get() == Y_FC
    assert view.include_flagged_var.get() is False and view.exclude_near_field_var.get() is False
    assert len(view.table.get_children()) == len(events)
    first = view.table.item("0")["values"]
    assert str(first[0]) == "0" and float(first[1]) == pytest.approx(M0_VALUES[0])
    assert float(first[2]) == pytest.approx(moment_magnitude(M0_VALUES[0]), abs=1e-3) and str(first[-1]) == ""
    assert view.fit_line_artist is None and view.flagged_scatter_artist is None
    assert "events have PZT source parameters" in view.status_var.get() and "excluded" not in view.status_var.get()
    assert view.saved_result is None and view.saved_agrees is None and view.fit_text.get() == "No fit yet"
    # reference line: exponent exactly -1/3 through the median (M0, fc) of the plotted pairs
    line = view.reference_line_artist
    assert line.get_label() == "fc ~ M0^-1/3"
    xdata, ydata = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    x, y, _ = view.pairs()
    assert np.allclose(ydata, np.median(y) * (xdata / np.median(x)) ** (-THIRD), rtol=1e-9)
    assert xdata.min() == pytest.approx(x.min()) and xdata.max() == pytest.approx(x.max())

    view.set_channel("ch0")
    fit = view.fit_power_law()
    assert fit.valid and fit.n == len(events) and view.fit_line_artist is not None
    text = view.fit_text.get()
    assert "OLS b = -0.3333" in text and "RMA b = -0.3333" in text and "16-84" in text
    stored = view.save()
    assert stored is app.data_manager.get_data("runs/[0]/source_scaling") and stored is view.run["source_scaling"]
    assert {k: stored[k] for k in ("version", "channel", "y_variable", "y_field", "trend_corrected", "include_flagged",
                                   "exclude_near_field", "n_boot", "seed", "n", "event_indices", "n_excluded",
                                   "excluded", "phases", "constants_consistent", "constants")} == {
        "version": RESULT_VERSION, "channel": "ch0", "y_variable": Y_FC, "y_field": "fc_hz", "trend_corrected": False,
        "include_flagged": False, "exclude_near_field": False, "n_boot": 1000, "seed": 0, "n": len(events),
        "event_indices": list(range(len(events))), "n_excluded": 0, "excluded": [], "phases": ["P"],
        "constants_consistent": True,
        "constants": [{"phase": "P", "rho_kg_m3": RHO, "vp_m_s": VP, "vs_m_s": VS, "k": K}]}
    # an exact power law: OLS, RMA and the bootstrap all give -1/3 with no scatter
    assert [stored[k] for k in ("exponent", "exponent_rma", "exponent_stderr", "r2")] == pytest.approx(
        [-THIRD, -THIRD, 0.0, 1.0], abs=1e-6)
    assert stored["bootstrap_16_84"] == pytest.approx([-THIRD, -THIRD], abs=1e-6)
    assert view.bootstrap == pytest.approx((-THIRD, -THIRD), abs=1e-6)
    # the fit is reproducible from the saved dict alone
    x, y = np.asarray(stored["x"]), np.asarray(stored["y"])
    assert list(x) == pytest.approx(list(M0_VALUES))
    assert list(y) == pytest.approx([A_FC * m ** (-THIRD) for m in M0_VALUES])
    assert np.allclose(10.0 ** (stored["log10_coefficient"] + stored["exponent"] * np.log10(x)), y, rtol=1e-6)
    assert np.allclose(stored["coefficient"] * x ** stored["exponent"], y, rtol=1e-6)
    refit = fit_power_law(x, y)
    assert refit.exponent == pytest.approx(stored["exponent"], abs=1e-12)
    assert refit.log10_coefficient == pytest.approx(stored["log10_coefficient"], abs=1e-12)
    assert "Saved" in view.status_var.get() and all(isinstance(k, str) for k in stored)

    # constant stress drop -> exponent 0, no reference line; radius r = k Vs / fc ~ M0^(+1/3)
    view.set_y_variable(Y_STRESS_DROP)
    assert view.reference_line_artist is None and view.fit_power_law().exponent == pytest.approx(0.0, abs=1e-9)
    view.set_y_variable(Y_RADIUS)
    assert view.fit_power_law().exponent == pytest.approx(THIRD, abs=1e-6)
    # 'all' (log-space median of two exact power laws with different prefactors): the same exponent,
    # and the rederived stress drop 7 M0 / (16 r^3) is constant too
    view.set_channel(ALL_CHANNELS)
    view.set_y_variable(Y_FC)
    assert view.fit is None
    fit_all = view.fit_power_law()
    assert fit_all.exponent == pytest.approx(-THIRD, abs=1e-6)
    assert fit_all.coefficient == pytest.approx(np.sqrt(1.5) * A_FC, rel=1e-6)
    view.set_y_variable(Y_STRESS_DROP)
    fit = view.fit_power_law()
    assert fit.valid and fit.exponent == pytest.approx(0.0, abs=1e-6)


def test_saved_result_is_restored_on_open(app, view):
    events = app.data_manager.get_data("runs/[0]/events")
    view.set_channel("ch1")
    view.set_y_variable(Y_RADIUS)
    view.trend_var.set(True)
    view.refresh()
    view.fit_power_law()
    saved = view.save()
    view.on_close()
    assert view not in app.child_windows
    # the fit is re-run on open and compared with the saved one
    v = SourceScalingView(app, 0)
    assert v.channel_combo.get() == "ch1" and v.y_combo.get() == Y_RADIUS and v.trend_var.get() is True
    assert v.saved_result is saved and v.saved_agrees is True
    assert "Saved fit" in v.fit_text.get() and "agrees" in v.fit_text.get() and "0.3333" in v.fit_text.get()
    assert v.fit.valid and v.fit.exponent == pytest.approx(THIRD, abs=1e-6) and v.fit_line_artist is not None
    v.on_close()
    # a record changed since the save: the restored summary says so
    events[2]["pzt_spectrum"]["channels"]["ch1"]["source"]["seismic_moment_nm"] *= 3.0
    v = SourceScalingView(app, 0)
    assert v.saved_agrees is False and "DIFFERS" in v.fit_text.get()
    v.on_close()
    # the saved channel no longer exists: fall back to 'all', still show the saved summary
    for event in events:
        del event["pzt_spectrum"]["channels"]["ch1"]
    v = SourceScalingView(app, 0)
    assert v.channel_combo.get() == ALL_CHANNELS and v.y_combo.get() == Y_RADIUS
    assert "Saved fit" in v.fit_text.get() and v.saved_agrees is False
    v.on_close()


def test_runs_with_missing_or_unusable_records(app):
    run = app.data_manager.get_data("runs/[0]")
    n_events = len(run["events"])
    v = SourceScalingView(app, 0)                            # no records at all
    assert list(v.channel_combo["values"]) == [ALL_CHANNELS] and "no saved pzt results" in v.status_var.get().lower()
    assert len(v.table.get_children()) == n_events and v.table.item("0")["values"][1] == "n/a"
    assert not v.fit_power_law().valid and "Fit failed" in v.fit_text.get()
    assert v.save() is None and "source_scaling" not in run
    v.on_close()

    populate(app, n=2)                                       # too few events to fit
    v = SourceScalingView(app, 0)
    fit = v.fit_power_law()
    assert not fit.valid and fit.n == 2 and np.isnan(v.bootstrap[0])
    v.on_close()

    events = populate(app)                                   # a channel missing on one event
    del events[1]["pzt_spectrum"]["channels"]["ch1"]
    v = SourceScalingView(app, 0)
    v.set_channel("ch1")
    fit = v.fit_power_law()
    assert fit.valid and fit.n == n_events - 1 and 1 not in v.fit_indices
    assert fit.exponent == pytest.approx(-THIRD, abs=1e-6)
    v.on_close()

    events = populate(app, channels=("0",))                  # a negative M0 and a numpy-False 'valid'
    events[1]["pzt_spectrum"]["channels"]["ch0"]["source"]["seismic_moment_nm"] = -1.0
    events[2]["pzt_spectrum"]["channels"]["ch0"]["source"]["valid"] = np.bool_(False)
    v = SourceScalingView(app, 0)
    v.set_channel("ch0")
    assert list(v.pairs()[2]) == [0, 3]
    fit = v.fit_power_law()
    assert not fit.valid and fit.n == 2 and v.fit_indices == [] and v.table.item("2")["values"][1] == "n/a"
    v.on_close()

    run["events"] = np.zeros(0)                              # an empty list comes back from HDF5 as an ndarray
    assert available_channels(run) == [] and gather_records(run) == []
    v = SourceScalingView(app, 0)
    assert v in app.child_windows and "no saved pzt results" in v.status_var.get().lower()
    assert len(v.table.get_children()) == 0
    v.on_close()


def test_mechanical_variables(app):
    v3 = {"event_analysis": {"version": 3, "x_field": "slip/slip_1", "y_field": "shear_stress",
                             "delta_x": 30.0, "delta_y": -0.4}}
    assert mechanical_values(v3) == {"mech_stress_drop": pytest.approx(0.4), "mech_slip": 30.0}
    assert all(np.isnan(value) for value in mechanical_values(v3, trend=True).values())
    populate(app, mech=True)                                 # version 2 records
    v = SourceScalingView(app, 0)
    v.set_y_variable(Y_MECH_STRESS_DROP)
    fit = v.fit_power_law()
    assert fit.valid and fit.exponent == pytest.approx(0.0, abs=1e-9) and fit.coefficient == pytest.approx(0.4)
    v.trend_var.set(True)
    v.refresh()
    assert v.fit_power_law().coefficient == pytest.approx(0.5)
    v.set_y_variable(Y_SLIP)
    fit = v.fit_power_law()
    assert fit.exponent == pytest.approx(0.5, abs=1e-6) and fit.coefficient == pytest.approx(60.0)


def test_quality_gating_in_view(app):
    """A clipped event is listed and drawn open, excluded from the fit, recorded on save."""
    events = populate(app, channels=("0",))
    clipped_fc = 2.0e4                                  # true fc of event 0 is 4.64e4
    events[0]["pzt_spectrum"]["channels"]["ch0"] = source_record(
        M0_VALUES[0], fc=clipped_fc, band_limited=True, at_bound=True)
    excluded = [{"event": 0, "flags": [FLAG_BAND_LIMITED, FLAG_AT_BOUND]}]
    v = SourceScalingView(app, 0)
    v.set_channel("ch0")
    assert v.table.item("0")["values"][-1] == "band,bound" and v.table.item("1")["values"][-1] == ""
    assert float(v.table.item("0")["values"][3]) == pytest.approx(clipped_fc)
    assert v.excluded == excluded and "1 event(s) excluded by quality flags" in v.status_var.get()
    assert len(v.flagged_scatter_artist.get_offsets()) == 1 and "excluded" in v.flagged_scatter_artist.get_label()
    assert len(v.scatter_artist.get_offsets()) == 3
    fit = v.fit_power_law()
    assert fit.valid and fit.n == 3 and v.fit_indices == [1, 2, 3]
    assert fit.exponent == pytest.approx(-THIRD, abs=1e-6) and "1 event(s) excluded" in v.fit_text.get()
    saved = v.save()
    assert saved["include_flagged"] is False and saved["n_excluded"] == 1
    assert saved["excluded"] == excluded and saved["event_indices"] == [1, 2, 3]
    # override: the clipped record enters the fit and biases the exponent
    v.set_include_flagged(True)
    assert v.excluded == [] and "INCLUDED" in v.status_var.get()
    assert "in fit" in v.flagged_scatter_artist.get_label() and len(v.scatter_artist.get_offsets()) == 4
    fit = v.fit_power_law()
    assert fit.n == 4 and v.fit_indices == [0, 1, 2, 3] and fit.exponent > -THIRD + 0.05
    saved = v.save()
    assert saved["include_flagged"] is True and saved["n_excluded"] == 0


def test_near_field_is_a_warning_not_a_default_exclusion(app):
    """kR at the band's low edge (~fmin) is < 3 for every laboratory record, so the near-field
    flag must not empty the fit by default; it is listed, counted and excluded only on request."""
    events = populate(app, channels=("0",))
    for j, event in enumerate(events):
        event["pzt_spectrum"]["channels"]["ch0"] = source_record(M0_VALUES[j % 4], far_field_ok=(j == 3))
    v = SourceScalingView(app, 0)
    v.set_channel("ch0")
    assert [r["flags"] for r in v.rows] == [[FLAG_NEAR_FIELD]] * 3 + [[]]
    assert all(r["flagged"] is False for r in v.rows) and v.excluded == [] and v.flagged_scatter_artist is None
    assert v.table.item("0")["values"][-1] == "kR" and v.table.item("3")["values"][-1] == ""
    assert "3 near-field event(s) (kR < 3 at the band's low edge) in the fit" in v.status_var.get()
    assert "excluded" not in v.status_var.get()
    fit = v.fit_power_law()
    assert fit.valid and fit.n == 4 and v.fit_indices == [0, 1, 2, 3]
    assert fit.exponent == pytest.approx(-THIRD, abs=1e-6)
    saved = v.save()
    assert saved["exclude_near_field"] is False and saved["n_excluded"] == 0
    # on request the near-field records are gated out like the fit-quality flags
    v.set_exclude_near_field(True)
    assert [r["flagged"] for r in v.rows] == [True, True, True, False]
    assert v.excluded == [{"event": j, "flags": [FLAG_NEAR_FIELD]} for j in range(3)]
    assert "3 event(s) excluded by quality flags" in v.status_var.get()
    assert len(v.flagged_scatter_artist.get_offsets()) == 3
    fit = v.fit_power_law()
    assert not fit.valid and fit.n == 1
    # 'all' channels and the include-flagged override interact the same way
    v.set_channel(ALL_CHANNELS)
    assert [r["flagged"] for r in v.rows] == [True, True, True, False]
    v.set_include_flagged(True)
    assert v.excluded == [] and v.fit_power_law().n == 4
    v.set_include_flagged(False)
    v.set_exclude_near_field(False)
    assert all(r["flagged"] is False for r in v.rows)
    # a band-limited AND near-field record is excluded for the band limit only
    events[1]["pzt_spectrum"]["channels"]["ch0"] = source_record(
        M0_VALUES[1], fc=1e3, band_limited=True, far_field_ok=False)
    v.set_channel("ch0")
    assert v.rows[1]["flags"] == [FLAG_BAND_LIMITED, FLAG_NEAR_FIELD]
    assert v.rows[1]["gating_flags"] == [FLAG_BAND_LIMITED]
    assert v.excluded == [{"event": 1, "flags": [FLAG_BAND_LIMITED]}]
    assert "excluded by quality flags (band-limited)" in v.status_var.get()
    # the saved gating choice is restored and compared on reopen
    v.set_exclude_near_field(True)
    v.set_include_flagged(True)
    v.fit_power_law()
    v.save()
    v.on_close()
    v2 = SourceScalingView(app, 0)
    assert v2.exclude_near_field_var.get() is True and v2.include_flagged_var.get() is True and v2.saved_agrees
    v2.on_close()


def test_inconsistent_constants_are_reported(app):
    events = populate(app, channels=("0", "1"))
    for j, event in enumerate(events):
        event["pzt_spectrum"]["channels"]["ch1"] = source_record(5.0 * M0_VALUES[j % 4], phase="S", k=0.372)
    v = SourceScalingView(app, 0)
    v.set_channel("ch0")
    phases, constants, consistent = v.constants_summary()
    assert phases == ["P"] and len(constants) == 1 and consistent
    assert "WARNING" not in v.status_var.get()
    v.set_channel(ALL_CHANNELS)
    phases, constants, consistent = v.constants_summary()
    assert phases == ["P", "S"] and len(constants) == 2 and not consistent
    assert "WARNING: phase/constants differ" in v.status_var.get()
    assert all(r["constants_consistent"] is False for r in v.rows)
    v.fit_power_law()
    saved = v.save()
    assert saved["phases"] == ["P", "S"] and saved["constants_consistent"] is False
    assert sorted(c["phase"] for c in saved["constants"]) == ["P", "S"]
    assert {c["k"] for c in saved["constants"]} == {K, 0.372}


def test_saved_result_survives_hdf5_reload(app, app_from_file, tmp_path):
    """Stored lists come back from HDF5 as arrays and must not be truth-tested."""
    populate(app)
    v = SourceScalingView(app, 0)
    v.set_channel("ch1")
    v.set_y_variable(Y_RADIUS)
    v.fit_power_law()
    saved = v.save()
    v.on_close()
    path = tmp_path / "scaling.h5"
    app.data_manager.save_file(path)

    app2 = app_from_file(path)
    stored = app2.data_manager.get_data("runs/[0]/source_scaling")
    assert np.asarray(stored["bootstrap_16_84"]).shape == (2,)
    assert list(np.asarray(stored["event_indices"])) == saved["event_indices"]
    stored["bootstrap_16_84"] = np.asarray(stored["bootstrap_16_84"], dtype=float)   # force the array form
    stored["event_indices"] = np.asarray(stored["event_indices"])
    # opened the way a right-click on the run does
    ctx = app2.context_at(app2.find_item("runs/[0]"))
    app2.run_action(next(a for a in actions_for(ctx.kind) if a.label == "Source Scaling"), ctx)
    v2 = app2.child_windows[-1]
    assert isinstance(v2, SourceScalingView)
    assert v2.channel_combo.get() == "ch1" and v2.y_combo.get() == Y_RADIUS
    assert v2.include_flagged_var.get() is False and v2.exclude_near_field_var.get() is False
    assert v2.saved_agrees is True and v2.fit.exponent == pytest.approx(saved["exponent"], abs=1e-9)
    assert "Saved fit" in v2.fit_text.get() and "agrees" in v2.fit_text.get()
    assert f"[{saved['bootstrap_16_84'][0]:.4g}" in v2.fit_text.get()
    # a second save and a second file write work on the reloaded data
    assert v2.save() is not None
    app2.data_manager.save_file(tmp_path / "scaling2.h5")


def _inject_brune_pulses(app, channel, fc_hz=10e3, omega0_v=1.0):
    """Replace ``channel`` of every event's strain block by a Brune pulse whose
    plateau grows with the event index (so M0 spans a range and a fit is possible)."""
    from labquake_explorer.analysis import spectrum as sp
    events = app.data_manager.get_data("runs/[0]/events")
    rng = np.random.default_rng(3)
    for j, event in enumerate(events):
        original = event["strain"]["original"]
        time = np.asarray(original["time"], dtype=float)
        raw = np.asarray(original["raw"], dtype=float) + rng.normal(0.0, 5e-5, np.shape(original["raw"]))
        fs = sp.sampling_rate(time)
        amp = omega0_v * 2.0 ** j
        _, pulse = sp.brune_pulse(fs, time.size, amp, fc_hz / (1.0 + 0.3 * j), float(event["event_time"]) - time[0])
        sigma = amp / np.sqrt(1000.0) * fs / np.sqrt(500.0)
        raw[channel] = pulse + rng.normal(0.0, sigma, time.size)
        original["raw"] = raw
    return events


def test_pzt_view_records_feed_the_scaling_view(app_with_strain, tmp_path):
    """End to end: PZTSpectrumView compute / compute_source / save on every event (view
    defaults, laboratory distance -> kR < 3 at the band's low edge), then a scaling fit."""
    app = app_with_strain
    events = _inject_brune_pulses(app, 3)
    csv = tmp_path / "sensor.csv"
    f = np.geomspace(100.0, 100e3, 25)
    csv.write_text("frequency_hz,gain\n" + "\n".join(f"{fi:.6g},2.0" for fi in f) + "\n")
    for j in range(len(events)):
        pv = PZTSpectrumView(app, 0, j)
        pv.set_channel(3)
        pv.calibration_path_var.set(str(csv))
        pv.unit_combobox.set("V/m")
        assert pv.compute(), pv.status_var.get()
        assert pv.fit.valid and not pv.fit.band_limited, pv.fit.warnings
        pv.rho_var.set("2700"); pv.c_var.set("6000"); pv.vs_var.set("3500"); pv.distance_var.set("0.05")
        assert pv.compute_source(), pv.status_var.get()
        assert pv.record["source"]["far_field_ok"] is False        # kR ~ 0.06 at ~1.2 kHz
        assert pv.save()
    assert set(events[0]["strain"]["spectrum"]["sets"]["default"]["channels"]) == {"ch3"}      # saved with the record

    v = SourceScalingView(app, 0)
    assert list(v.channel_combo["values"]) == [ALL_CHANNELS, "ch3"]
    v.set_channel("ch3")
    assert all(np.isfinite(r["seismic_moment_nm"]) for r in v.rows)
    assert all(r["flags"] == [FLAG_NEAR_FIELD] and not r["flagged"] for r in v.rows)
    assert "near-field event(s)" in v.status_var.get() and "excluded" not in v.status_var.get()
    fit = v.fit_power_law()
    assert fit.valid and fit.n == len(events), v.fit_text.get()
    assert fit.exponent < 0                                        # fc falls as M0 grows
    saved = v.save()
    assert saved["channel"] == "ch3" and saved["phases"] == ["P"]
    assert saved["constants"] == [{"phase": "P", "rho_kg_m3": 2700.0, "vp_m_s": 6000.0,
                                   "vs_m_s": 3500.0, "k": 0.32}]
    v.set_channel(ALL_CHANNELS)                                    # 'all' sees the same single channel
    assert v.fit_power_law().n == len(events)
    v.set_exclude_near_field(True)                                 # the gating empties the fit only on request
    assert not v.fit_power_law().valid


def test_result_set_chooser_selects_the_records(app_with_strain):
    app = app_with_strain
    events = populate(app)                                    # the legacy maps are the 'default' set
    for j, event in enumerate(events):
        event["strain"]["spectrum"] = {"version": 2, "sets": {
            "EGF": {"channels": {"ch0": dict(source_record(10.0 * M0_VALUES[j % 4]), channel=0)}}}}
    assert available_sets(app.data_manager.get_data("runs/[0]")) == ["default", "EGF"]
    v = SourceScalingView(app, 0)
    assert list(v.set_combo["values"]) == ["default", "EGF"] and v.result_set == "default"
    assert list(v.channel_combo["values"]) == [ALL_CHANNELS, "ch0", "ch1"]
    m0_default = float(v.table.item("0")["values"][1])
    v.set_combo.set("EGF")
    v.on_set_selected()
    assert list(v.channel_combo["values"]) == [ALL_CHANNELS, "ch0"]
    assert float(v.table.item("0")["values"][1]) == pytest.approx(10.0 * m0_default)
    assert "set 'EGF'" in v.status_var.get()
    assert v.fit_power_law() is not None and v.save()["result_set"] == "EGF"
    v.on_close()
    v = SourceScalingView(app, 0)                             # reopens on the saved set
    assert v.result_set == "EGF" and list(v.channel_combo["values"]) == [ALL_CHANNELS, "ch0"]
    v.on_close()


def test_ensemble_percentiles_become_error_bars(app, view):
    events = app.data_manager.get_data("runs/[0]/events")
    assert all(np.isnan(r["fc_p16"]) and np.isnan(r["m0_p84"]) for r in view.rows)
    assert view.error_bar_artist is None

    def stats(value, lo, hi, n):
        return {"p16": lo * value, "p84": hi * value, "min": 0.7 * value, "max": 1.4 * value, "median": value, "n": n}
    for j, event in enumerate(events[:3]):
        for rec in event["pzt_spectrum"]["channels"].values():
            fc, m0 = rec["source"]["fc_hz"], rec["source"]["seismic_moment_nm"]
            n_resolved = 10 if j else 0
            rec["ensemble"] = {"version": 1, "n_members": 16, "n_valid": 16, "n_resolved": n_resolved,
                               "stats": {"fc": stats(fc, 0.8, 1.3, 16), "seismic_moment_nm": stats(m0, 0.9, 1.2, 16)},
                               # the pinned members left out: narrower
                               "stats_resolved": {"fc": stats(fc, 0.9, 1.1, n_resolved),
                                                  "seismic_moment_nm": stats(m0, 0.95, 1.1, n_resolved)}}
    view.refresh()
    rows = {r["event"]: r for r in view.rows}
    # none resolved: every valid member; otherwise the resolved ones
    assert rows[0]["fc_p16"] == pytest.approx(0.8 * rows[0]["fc_hz"])
    assert rows[0]["m0_p84"] == pytest.approx(1.2 * rows[0]["seismic_moment_nm"])
    assert rows[1]["fc_p16"] == pytest.approx(0.9 * rows[1]["fc_hz"])
    assert rows[1]["m0_p84"] == pytest.approx(1.1 * rows[1]["seismic_moment_nm"])
    assert np.isnan(rows[3]["fc_p16"]) and view.error_bar_artist is not None
    # a single channel reads the record's own percentiles; the other Y variables keep the M0 bars
    view.set_channel("ch0")
    assert view.error_bar_artist is not None
    rec0 = events[0]["pzt_spectrum"]["channels"]["ch0"]
    assert view.rows[0]["fc_p84"] == pytest.approx(rec0["ensemble"]["stats"]["fc"]["p84"])
    view.set_y_variable(Y_STRESS_DROP)
    assert view.error_bar_artist is not None


def test_shared_t_star_scan_and_refit_set(app):
    from dataclasses import replace
    from labquake_explorer.analysis import spectrum as sp
    from labquake_explorer.analysis import t_star_scan as ts
    from labquake_explorer.ui.views.source_scaling_view import PLOT_T_STAR, T_STAR_KEY, parse_t_star_grid
    assert list(parse_t_star_grid("-10, 50, 2") * 1e6)[:3] == pytest.approx([-10.0, -8.0, -6.0])
    for bad in ("1, 2", "a, b, c", "10, 0, 5", "0, 10, 0"):
        with pytest.raises(ValueError):
            parse_t_star_grid(bad)
    events = populate(app, channels=("3",))
    # every record gets a binned spectrum attenuated with t* = 20 us
    for k, event in enumerate(events):
        rec = event["pzt_spectrum"]["channels"]["ch3"]
        t, x = sp.brune_pulse(1e6, 8000, 1.0, 8e3 + 2e3 * k, 4e-3)
        s = sp.compute_spectrum(sp.extract_window(t, x, 4e-3, 2e-3, 2e-3), fmin=500.0, fmax=100e3)
        loss = np.exp(-np.pi * s.f_binned * 20e-6)                 # attenuated by the medium
        s = replace(s, amp_binned=s.amp_binned * loss, noise_binned=s.noise_binned * loss)
        rec.update({"spectrum": s.as_dict(), "fmin_hz": 1000.0, "fmax_hz": 50e3, "snr_min": None, "n_fixed": True,
                    "loss": "linear", "n_max": 8.0, "t_star_s": None, "result_set": "default", "field": "pzt_4"})
        if k > 0:      # the older map's records: the refit goes to the event's (PSU strain) record
            event["strain"] = {"original": {"time": np.arange(4) * 1e-6, "raw": np.zeros((1, 4))}}
    app.data_manager.data.setdefault("analysis", {})["pzt_spectrum"] = {
        "sets": {"default": {"settings": {"pre_ms": 0.5}, "sensors": {"pzt_4": {"t_star_s": None, "distance_m": 0.1}}}}}
    v = SourceScalingView(app, 0)
    assert v.t_star_text.get() == "no scan yet"
    v.t_star_grid_var.set("abc")
    assert not v.scan_t_star() and "t* grid" in v.status_var.get()
    v.t_star_grid_var.set("-10, 50, 2")
    assert v.scan_t_star(), v.status_var.get()
    scan = v.t_star_scans["ch3"]
    assert scan.best_t_star_s == pytest.approx(20e-6) and scan.n_records == len(events)
    assert "ch3: t* = 20 us" in v.t_star_text.get() and v.plot_combo.get() == PLOT_T_STAR
    assert len(v.ax.lines) >= 2 and "t* scan" in v.ax.get_title()
    saved = app.data_manager.get_data(f"runs/[0]/{T_STAR_KEY}")
    assert saved["result_set"] == "default" and saved["sensors"]["ch3"]["best_t_star_s"] == pytest.approx(20e-6)
    v.plot_combo.set("scaling")
    v.plot()
    assert "source scaling" in v.ax.get_title()
    # the refits go into a new, named set, which is then shown; the sensor's t* goes into its settings
    assert not v.save_refits() and "needs a name" in v.status_var.get()
    v.refit_set_var.set("default")
    assert not v.save_refits() and "different" in v.status_var.get()
    v.refit_set_var.set("tstar")
    assert v.save_refits(), v.status_var.get()
    assert "1 record(s) of the older event-level map" in v.status_var.get()     # event 0 has no record to hold it
    assert v.result_set == "tstar" and list(v.set_combo["values"]) == ["default", "tstar"]
    assert events[0]["pzt_spectrum"]["version"] == 1 and "sets" not in events[0]["pzt_spectrum"]   # left as it was
    new = events[1]["strain"]["spectrum"]["sets"]["tstar"]["channels"]["ch3"]
    old = events[1]["pzt_spectrum"]["channels"]["ch3"]
    assert new["t_star_s"] == pytest.approx(20e-6) and new["result_set"] == "tstar" and new["ensemble"] is None
    assert new["t_star_refit"]["from_set"] == "default"
    assert ts.record_input(old).fit(0.0).fc < 0.5 * new["fit"]["fc"]       # the uncorrected fit sits far too low
    assert new["fit"]["fc"] == pytest.approx(10e3, rel=0.05) and new["source"]["valid"]
    assert new["source"]["seismic_moment_nm"] != old["source"]["seismic_moment_nm"]
    sets = app.data_manager.data["analysis"]["pzt_spectrum"]["sets"]
    assert sets["tstar"]["sensors"]["pzt_4"]["t_star_s"] == pytest.approx(20e-6)
    assert sets["tstar"]["settings"]["pre_ms"] == 0.5 and sets["default"]["sensors"]["pzt_4"]["t_star_s"] is None
    # the scan belongs to the set it was made on, and is restored from the run
    assert "saved scan is for set 'default'" in v.t_star_text.get()
    v.on_close()
    v = SourceScalingView(app, 0)
    v.set_combo.set("default")
    v.on_set_selected()
    assert v.t_star_scans["ch3"].best_t_star_s == pytest.approx(20e-6) and "ch3: t* = 20 us" in v.t_star_text.get()
