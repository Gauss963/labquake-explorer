from types import SimpleNamespace

import numpy as np
import pytest

from labquake_explorer.analysis import arrival_picks as ap
from labquake_explorer.analysis import trace_filter as tf
from labquake_explorer.data.channels import channel_array, positions_table
from labquake_explorer.data.sources import read_arrivals
from labquake_explorer.ui.context import resolve_context
from labquake_explorer.ui.views import CZMFitterView
from labquake_explorer.ui.views import DynamicStrainArrivalPickerView as Picker
from labquake_explorer.ui.views.pzt_spectrum_view import TRIGGER_PICKED, PZTSpectrumView

N_CHANNELS = 16
OLD_LAYOUT_ENABLED = [i % 2 == 0 for i in range(N_CHANNELS)]
OLD_LAYOUT_FITTING = [i in (6, 8, 10, 12) for i in range(N_CHANNELS)]
CHECK = "✓"
EVENT = "runs/[0]/events/[1]"
XS = [50.0, 130.0, 210.0, 290.0, 370.0, 450.0]


@pytest.fixture
def view(app_with_strain):
    v = Picker(app_with_strain, 0, 1)
    yield v
    v.on_close()


def _speed_from_picks(view):
    """Cf recomputed independently from the fitting channels' picks."""
    t = view.event["strain"]["original"]["time"]
    fit = [i for i, f in enumerate(view.fitting_channels) if f]
    slope = np.polyfit([view.locations[i] for i in fit], [t[view.picked_idx[i]] for i in fit], 1)[0]
    return -1e-3 / slope


def _move_all_picks(view, idx):
    view.picked_idx = [idx] * len(view.picked_idx)
    view.draw_markers()
    view.update_fitted_line()


def _drawn(view, channel, trace):
    """A trace as the view draws it: from the noise's median, scaled to 12 mm, at its sensor."""
    base = np.median(trace[:view.noise_end()[0]])
    return (trace - base) * (-12 / np.ptp(trace)) + view.locations[channel]


def _select_record(view, key):
    view.record_combobox.set(key)
    view.on_record_selected()


def _czm_cf(app):
    czm = CZMFitterView(app, 0, 1)
    cf = czm.Cf.get()
    czm.on_close()
    return cf


def _mouse(view, x_ms, y_mm):
    """A left-button mouse event at data point ``(x_ms, y_mm)`` of the waveform panel."""
    x, y = view.ax_wave.transData.transform((x_ms, y_mm))
    return SimpleNamespace(inaxes=view.ax_wave, xdata=x_ms, ydata=y_mm, x=x, y=y, button=1)


def add_pzt_record(event, xs, t_trig=None, fs=200e3, speed=1500.0, key="elsys", x0=None, noise=0.0):
    """A PZT-like record whose onsets leave ``x0`` (default the first sensor) at ``speed`` m/s,
    both ways when sensors lie on both sides.  Returns ``(time, t_trig)``."""
    t_trig = float(event["event_time"]) - 0.02 if t_trig is None else t_trig
    x0 = xs[0] if x0 is None else x0
    time = t_trig - 0.05 + np.arange(int(0.3 * fs)) / fs
    raw = np.array([0.5 * (1.0 + np.tanh((time - t_trig - abs(x - x0) * 1e-3 / speed) / 2e-5)) for x in xs])
    if noise:
        raw = raw + noise * np.random.default_rng(0).standard_normal(raw.shape)
    names = [f"pzt_{i + 1}" for i in range(len(xs))]
    pos = positions_table(names, {n: (x, 100.0, 50.0) for n, x in zip(names, xs)})
    event.setdefault("waveform", {})[key] = channel_array(raw, names, unit="V", positions=pos, time=time,
                                                          sample_rate=fs, block=3, trigger_time=t_trig,
                                                          filename=f"{key}.tpc5")
    return time, t_trig


# ------------------------------------------------------------- strain record
def test_strain_defaults_peak_picks_save_and_degenerate_fits(app_with_strain, view):
    dm = app_with_strain.data_manager
    t = view.event["strain"]["original"]["time"]
    raw = view.event["strain"]["original"]["raw"]
    # 'p0001' -> experiment 1 -> old (paired gauge) layout
    assert view.event_idx == 1 and view.title() == "Pick Arrivals - Event 1" and view.event_combobox.get() == "1"
    assert view in app_with_strain.child_windows and view.exp_number() == 1
    assert view.enabled_channels == OLD_LAYOUT_ENABLED and view.fitting_channels == OLD_LAYOUT_FITTING
    assert view.locations == pytest.approx(list(view.event["strain_truth"]["locations_mm"]))
    assert view.record_key == "strain" and list(view.record_combobox["values"]) == ["strain"]
    assert len(view.picked_idx) == N_CHANNELS and sum(line is not None for line in view.lines) == 8
    assert len(view.fitting_markers) == 4 and len(view.not_fitting_markers) == 4
    rows = [view.table.item(r, "values") for r in view.table.get_children()]   # show and fit as check marks
    assert [r[3] == CHECK for r in rows] == OLD_LAYOUT_ENABLED and [r[4] == CHECK for r in rows] == OLD_LAYOUT_FITTING
    assert view.ax_wave is view.axs[4] and view.ax_wave.get_xlabel() == "time from the event (ms)"
    assert "not saved" in view.status_var.get()

    # every marker starts on the middle sample: no propagation, no speed; a save keeps the
    # picks but writes no rupture_speed at all (not nan)
    assert len(set(view.picked_idx)) == 1 and np.isnan(view.rupture_speed)
    assert view.cf_label.cget("text") == "Cf = n/a"
    view.save()
    assert "rupture_speed" not in dm.get_data(EVENT)
    assert dm.get_data(f"{EVENT}/strain/original/picked_idx") == view.picked_idx

    # the peak is the largest departure from the noise: a sample on the saturated plateau of
    # each step, up or down (up to floating-point ties among plateau samples), not the onset
    view.pick_peaks()
    for i in np.flatnonzero(view.enabled_channels):
        pick = view.picked_idx[i]
        assert abs(raw[i][pick]) == pytest.approx(np.abs(raw[i]).max(), rel=1e-9)
        assert pick == int(np.argmax(np.abs(view.lines[i].get_data()[1] - view.locations[i])))
    # all picks trail the true arrivals by a near-constant lag, so Cf is consistent
    truth = view.event["strain_truth"]
    arrival_idx = [int(np.argmin(np.abs(t - a))) for a in truth["arrival_times"]]
    lags = [view.picked_idx[i] - arrival_idx[i] for i in np.flatnonzero(view.fitting_channels)]
    assert min(lags) > 0 and max(lags) - min(lags) <= 2
    good = _speed_from_picks(view)
    assert view.rupture_speed == pytest.approx(good, rel=1e-9)
    # sample quantization limits the accuracy; the magnitude must still be right
    assert 0.5 * truth["rupture_speed"] < abs(good) < 2.0 * truth["rupture_speed"]
    assert view.cf_label.cget("text") == f"Cf = {view.rupture_speed:.2f} m/s" and view.fitted_line is not None
    view.save()
    assert dm.get_data(f"{EVENT}/rupture_speed") == pytest.approx(good, rel=1e-9)
    assert dm.get_data(f"{EVENT}/strain/enabled_channels") == OLD_LAYOUT_ENABLED
    assert dm.get_data(f"{EVENT}/strain/fitting_channels") == OLD_LAYOUT_FITTING
    assert dm.get_data(f"{EVENT}/strain/original/picked_idx") == view.picked_idx
    np.testing.assert_allclose(dm.get_data(f"{EVENT}/strain/original/rupture_arrival_time"), t[view.picked_idx])
    assert "enabled_channels" not in view.event and "picked_idx" not in view.event   # only Cf at the top

    # identical picks away from t = 0: polyfit gives a slope of ~1e-21 rather than 0, which must
    # still be no speed, not ~1e18 m/s
    target = 3 * len(t) // 4
    assert t[target] - view.event["event_time"] != 0
    _move_all_picks(view, target)
    assert all(m.get_xdata()[0] == pytest.approx((t[target] - view.event["event_time"]) * 1e3)
               for m in view.fitting_markers)
    assert np.isnan(view.rupture_speed) and view.cf_label.cget("text") == "Cf = n/a" and view.fitted_line is None
    # its save removes the stale speed, so the CZM fitter opens with its default Cf rather than nan
    view.save()
    assert "rupture_speed" not in dm.get_data(EVENT)
    assert dm.get_data(f"{EVENT}/strain/original/picked_idx") == view.picked_idx
    assert _czm_cf(app_with_strain) == pytest.approx(10)
    # a well-posed save afterwards restores both
    view.pick_peaks()
    view.save()
    assert dm.get_data(f"{EVENT}/rupture_speed") == pytest.approx(good)
    assert _czm_cf(app_with_strain) == pytest.approx(abs(good))


def test_fit_arrival_line_needs_two_positions_and_two_times():
    locs = [10.5, 22.5, 34.5, 46.5]
    assert Picker.fit_arrival_line(locs, [5e-3] * 4) is None
    assert Picker.fit_arrival_line(locs, [1e-3, 1e-3, 1e-3, 1e-3 + 1e-12]) is not None
    assert Picker.fit_arrival_line([10.5] * 4, [1e-3, 2e-3, 3e-3, 4e-3]) is None
    assert Picker.fit_arrival_line([10.5], [1e-3]) is None


def test_switching_events_and_unsaved_channel_edits(app_with_strain, view, no_dialogs):
    dm = app_with_strain.data_manager
    view.pick_peaks()
    view.set_fitting(0, True)
    assert view.fitting_channels[0] is True and len(view.fitting_markers) == 5
    view.save()
    picks, fitting, speed = list(view.picked_idx), list(view.fitting_channels), view.rupture_speed
    # an unsaved edit shows in the status line, not in the data
    view.set_shown(1, True)
    assert view.enabled_channels[1] is True and len(view.not_fitting_markers) == 4
    assert view.status_var.get().endswith("differs from the saved record")
    assert dm.get_data(f"{EVENT}/strain/enabled_channels")[1] is False

    # a fresh event comes up with default picks and the channels of the last save
    view.set_event(2)
    assert view.event_idx == 2 and view.title() == "Pick Arrivals - Event 2" and view.event_combobox.get() == "2"
    assert view.event is dm.get_data("runs/[0]/events/[2]")
    assert view.fitting_channels == fitting and len(view.fitting_markers) == 5
    assert len(set(view.picked_idx)) == 1 and np.isnan(view.rupture_speed)
    assert view.cf_label.cget("text") == "Cf = n/a"
    # going back restores what was saved
    view.set_event(1)
    assert view.picked_idx == picks and view.fitting_channels == fitting
    assert [view.table.item(str(i), "values")[4] == CHECK for i in range(N_CHANNELS)] == fitting
    assert view.status_var.get().endswith("saved") and view.rupture_speed == pytest.approx(speed)

    # an event without a record is refused, from code or the combobox: a status line, no dialog
    event_2 = dm.get_data("runs/[0]/events/[2]")
    event_2.pop("strain")
    view.set_event(2)
    assert view.event_idx == 1 and view.event is dm.get_data(EVENT)
    assert view.title() == "Pick Arrivals - Event 1" and view.event_combobox.get() == "1"
    assert view.picked_idx == picks and view.rupture_speed == pytest.approx(speed)
    assert "event 2 has no full-rate record" in view.status_var.get()
    view.event_combobox.set("2")
    view.on_event_selected()
    assert view.event_idx == 1 and view.event_combobox.get() == "1" and no_dialogs == []
    view.save()                                                 # still fully usable
    assert dm.get_data(f"{EVENT}/rupture_speed") == pytest.approx(speed) and "rupture_speed" not in event_2
    # the arrows skip it, and stop with a status line before the first event
    view.step_event(+1)
    assert view.event_idx == 3 and view.title() == "Pick Arrivals - Event 3"
    view.step_event(-1)
    assert view.event_idx == 1
    view.set_event(0)
    view.step_event(-1)
    assert view.event_idx == 0 and "no earlier event" in view.status_var.get()

    view.set_all_shown(True)
    assert all(view.enabled_channels) and all(line is not None for line in view.lines)
    view.fit_all_shown()
    assert all(view.fitting_channels) and len(view.fitting_markers) == N_CHANNELS


def test_open_on_event_without_strain_raises_and_unregisters(app_with_strain):
    app_with_strain.data_manager.get_data("runs/[0]/events/[0]").pop("strain")
    with pytest.raises(ValueError, match="no full-rate record"):
        Picker(app_with_strain, 0, 0)
    assert not any(isinstance(w, Picker) for w in app_with_strain.child_windows)


def test_trace_filters_redraw_the_traces_and_are_saved(app_with_strain, view, tmp_path, app_from_file):
    from scipy import signal
    raw = view.event["strain"]["original"]["raw"]
    dt = view.sample_interval()
    fs = 1.0 / dt
    ch = next(i for i, f in enumerate(view.fitting_channels) if f)

    def drawn():
        return view.lines[ch].get_data()[1]

    # Savitzky-Golay, its window typed in ms (51 samples when the record opens); the picks stay
    assert view.filter_window() == 51
    assert float(view.filter_window_ms.get()) == pytest.approx(51 * dt * 1e3, rel=1e-2)
    view.pick_peaks()
    picks = list(view.picked_idx)
    view.set_filter(tf.KIND_SAVGOL, window_ms=float(f"{50 * dt * 1e3:.6g}"))
    assert view.filtering and view.filter_kind() == tf.KIND_SAVGOL
    assert view.picked_idx == picks and np.isfinite(view.rupture_speed)
    assert view.filter_window() == 50 and "= 50 samples" in view.filter_note_var.get()
    np.testing.assert_allclose(drawn(), _drawn(view, ch, signal.savgol_filter(raw[ch], 50, 2)))
    # a window savgol cannot use is clamped; garbage falls back to 51 samples
    for typed, samples in ((f"{dt * 1e3:.6g}", 3), ("abc", 51)):
        view.filter_window_ms.set(typed)
        view.on_filter_changed()
        assert view.filter_window() == samples

    # Butterworth: only the entries the filter uses are shown; a corner it does not use is disabled
    for kind, values in ((tf.KIND_LOWPASS, dict(high_khz=2.0)), (tf.KIND_HIGHPASS, dict(low_khz=0.5)),
                         (tf.KIND_BANDPASS, dict(low_khz=0.5, high_khz=2.0, order=2, zero_phase=False))):
        view.set_filter(kind, **values)
        settings = view.filter_settings()
        assert settings["kind"] == kind and settings["order"] == values.get("order", 4)
        assert settings["zero_phase"] is values.get("zero_phase", True)
        np.testing.assert_allclose(drawn(), _drawn(view, ch, tf.filter_traces(raw[ch:ch + 1], fs, settings)[0]))
        assert view.filter_note_var.get().startswith(f"Nyquist {fs / 2e3:g} kHz")
        assert bool(view.filter_low_entry.instate(["disabled"])) is (kind == tf.KIND_LOWPASS)
        assert bool(view.filter_high_entry.instate(["disabled"])) is (kind == tf.KIND_HIGHPASS)
        assert not view.filter_window_entry.winfo_manager()

    # an unusable filter draws the raw traces, says why, and is saved as none
    view.set_filter(tf.KIND_LOWPASS, high_khz=fs / 1e3)         # twice the Nyquist frequency
    assert "not applied" in view.filter_note_var.get() and "Nyquist" in view.filter_note_var.get()
    np.testing.assert_allclose(drawn(), _drawn(view, ch, raw[ch]))
    view.set_filter(tf.KIND_BANDPASS, low_khz=2.0, high_khz=1.0)
    assert "low corner must be below the high corner" in view.filter_note_var.get()
    view.save()
    assert read_arrivals(view.event["strain"], view.event)["filter"] == {"kind": "off"}
    view.set_filter(tf.KIND_OFF)
    assert not view.filtering and view.filter_note_var.get().startswith("Nyquist")
    np.testing.assert_allclose(drawn(), _drawn(view, ch, raw[ch]))

    # the filter is saved with the picks and the auto pick, and restored from the file
    view.set_filter(tf.KIND_BANDPASS, low_khz=0.5, high_khz=2.0, order=3, zero_phase=False)
    view.auto_pick()
    view.save()
    saved = read_arrivals(view.event["strain"], view.event)["filter"]
    assert saved == {"kind": "bandpass", "order": 3, "zero_phase": False, "low_hz": 500.0, "high_hz": 2000.0}
    assert read_arrivals(view.event["strain"], view.event)["auto_pick"]["filter"] == saved
    path = tmp_path / "filter.h5"
    app_with_strain.data_manager.save_file(path)
    reopened = Picker(app_from_file(path), 0, 1)
    assert reopened.filter_kind() == tf.KIND_BANDPASS and reopened.filter_order_var.get() == "3"
    assert float(reopened.filter_low_khz.get()) == 0.5 and float(reopened.filter_high_khz.get()) == 2.0
    assert reopened.zero_phase_var.get() is False and reopened.status_var.get().endswith("saved")


def test_picks_dragged_typed_zoomed_and_mechanical_panels_hidden(app_with_strain, view, monkeypatch):
    monkeypatch.setattr(Picker, "show_mechanical_default", True)   # restored after the toggle below
    tt = view.relative_time_s() * 1e3
    # every pick on one sample: a narrow window around it, 50 samples each side
    view.zoom_to_picks()
    lo, hi = view.ax_wave.get_xlim()
    assert lo < tt[view.picked_idx[0]] < hi and hi - lo == pytest.approx(100 * (tt[1] - tt[0]), rel=1e-3)
    view.show_whole_record()
    assert view.ax_wave.get_xlim() == pytest.approx((tt[0], tt[-1]))

    view.canvas.draw()
    channel = int(view.fitting_markers[0].get_label())
    x0, loc, target = tt[view.picked_idx[channel]], view.locations[channel], len(tt) // 4
    # toolbar in pan/zoom: the press does not grab the handle
    view.toolbar.mode = "pan/zoom"
    view.on_press(_mouse(view, x0, loc))
    assert view._drag is None
    view.on_motion(_mouse(view, tt[target], loc))
    assert view.picked_idx[channel] != target
    view.toolbar.mode = ""
    # a press on the handle grabs it and selects the channel; the drag follows the time axis only
    _, lo, hi = view.handle_extent(channel)
    view.on_press(_mouse(view, x0, 0.5 * (lo + hi)))
    assert view._drag == channel and view.channel == channel
    view.on_motion(_mouse(view, tt[target], hi + 3.0))          # off the handle vertically: still this channel
    assert view.picked_idx[channel] == target and np.isfinite(view.rupture_speed)
    assert list(view.marker_of(channel).get_xdata()) == pytest.approx([tt[target]] * 2)
    view.on_release(None)
    assert view._drag is None and view.arrival_var.get() == f"{tt[target]:.6g}"
    assert view.table.item(str(channel), "values")[2] == f"{tt[target]:.5g}"
    # a press away from every handle grabs nothing; near a trace it selects that channel
    other = next(i for i, e in enumerate(view.enabled_channels) if e and i != channel)
    view.on_press(_mouse(view, tt[10], view.locations[other]))
    assert view._drag is None and view.channel == other

    # a typed arrival in ms; outside the record or not a number: the pick stays, the status says why
    view.select_channel(6)
    assert view.channel_label_var.get() == "ch 6"
    for typed, idx, message in ((f"{tt[100]:.6f}", 100, ""), ("1e6", 100, "outside the record"),
                                ("abc", 100, "not a time in ms")):
        view.arrival_var.set(typed)
        view.on_arrival_typed()
        assert view.picked_idx[6] == idx and message in view.status_var.get()
    assert list(view.marker_of(6).get_xdata()) == pytest.approx([tt[100]] * 2)
    # the arrows step through every channel and wrap around
    view.step_channel(+1)
    assert view.channel == 7 and view.channel_label_var.get() == "ch 7"
    view.select_channel(0)
    view.step_channel(-1)
    assert view.channel == N_CHANNELS - 1

    # spread picks: the zoom covers them all, never wider than the record
    view.pick_peaks()
    view.zoom_to_picks()
    lo, hi = view.ax_wave.get_xlim()
    times = [tt[view.picked_idx[i]] for i, e in enumerate(view.enabled_channels) if e]
    assert lo <= min(times) and hi >= max(times) and lo >= tt[0] and hi <= tt[-1]

    # the mechanical panels hidden: picks kept, and the choice carries over to the next window
    picks = list(view.picked_idx)
    view.mechanical_var.set(False)
    view.on_mechanical_toggled()
    assert len(view.axs) == 1 and view.ax_wave is view.axs[0]
    assert view.picked_idx == picks and np.isfinite(view.rupture_speed)
    other = Picker.from_context(app_with_strain, resolve_context("runs/[0]/events/[2]"))
    assert (other.run_idx, other.event_idx) == (0, 2) and other in app_with_strain.child_windows
    assert other.mechanical_var.get() is False and len(other.axs) == 1
    other.on_close()
    assert other not in app_with_strain.child_windows


def test_real_mouse_drag_and_table_clicks(view):
    view.deiconify()
    view.update()
    widget = view.canvas.get_tk_widget()
    if not (widget.winfo_ismapped() and view.table.winfo_ismapped()):
        pytest.skip("window not mapped: Tk mouse events need a display")
    view.canvas.draw()
    channel = int(view.fitting_markers[1].get_label())
    tt = view.relative_time_s() * 1e3
    target = len(tt) // 3
    height = view.canvas.figure.bbox.height

    def at(x_ms, y_mm):
        px, py = view.ax_wave.transData.transform((x_ms, y_mm))
        return int(round(px)), int(round(height - py))

    _, lo, hi = view.handle_extent(channel)
    x, y = at(tt[view.picked_idx[channel]], 0.5 * (lo + hi))
    widget.event_generate("<ButtonPress-1>", x=x, y=y)
    x2, y2 = at(tt[target], view.locations[channel])
    widget.event_generate("<B1-Motion>", x=x2, y=y2)
    widget.event_generate("<ButtonRelease-1>", x=x2, y=y2)
    widget.update()
    assert abs(view.picked_idx[channel] - target) <= max(2, int(len(tt) / widget.winfo_width()) + 1)

    def click(row, column):
        view.table.see(row)
        view.update()
        x, y, w, h = view.table.bbox(row, column)
        view.table.event_generate("<ButtonPress-1>", x=x + w // 2, y=y + h // 2)
        view.table.event_generate("<ButtonRelease-1>", x=x + w // 2, y=y + h // 2)
        view.update()

    # a click on the show or fit cell toggles it
    assert view.fitting_channels[6] is True
    click("6", "fit")
    assert view.fitting_channels[6] is False and len(view.fitting_markers) == 3
    assert view.table.item("6", "values")[4] == ""
    assert view.enabled_channels[1] is False
    click("1", "show")
    assert view.enabled_channels[1] is True and view.lines[1] is not None
    # a click elsewhere on a row selects the channel, its trace drawn thicker
    click("4", "channel")
    assert view.channel == 4 and view.lines[4].get_linewidth() == 2.0 and view.lines[0].get_linewidth() == 1.0


# ---------------------------------------------------------- waveform records
def test_pzt_record_defaults_per_record_save_and_readers(app):
    event = app.data_manager.get_data(EVENT)
    xs = [100.0, 200.0, 300.0, 400.0]
    time, t_trig = add_pzt_record(event, xs)
    view = Picker(app, 0, 1)
    # every channel shown and fitted, markers at the trigger, traces at the sensor positions
    assert view.record_key == "waveform/elsys" and list(view.record_combobox["values"]) == ["waveform/elsys"]
    assert view.enabled_channels == [True] * 4 and view.fitting_channels == [True] * 4
    assert view.locations == xs and view.picked_idx == [int(np.argmin(np.abs(time - t_trig)))] * 4
    assert view.trace_amplitude(xs) == 100.0 and view.axs[4].get_ylim() == pytest.approx((500.0, 0.0))
    view.pick_peaks()
    assert abs(view.rupture_speed) == pytest.approx(1500.0, rel=0.05)
    # saved under the record's own 'arrivals', no event-level rupture_speed
    view.save()
    rec = read_arrivals(event["waveform"]["elsys"])
    assert event["waveform"]["elsys"]["arrivals"]["version"] == 2 and rec["picked_idx"] == view.picked_idx and rec["locations"] == xs
    assert rec["rupture_speed"] == pytest.approx(view.rupture_speed)
    np.testing.assert_allclose(rec["arrival_time"], time[view.picked_idx])
    assert "rupture_speed" not in event and "picked_idx" not in event["waveform"]["elsys"]
    assert read_arrivals(event["waveform"]["elsys"])["rupture_speed"] == pytest.approx(view.rupture_speed)
    view.on_close()
    view = Picker(app, 0, 1)                                      # reopening restores the saved picks
    assert view.picked_idx == rec["picked_idx"] and view.rupture_speed == pytest.approx(rec["rupture_speed"])
    view.on_close()
    # the CZM fitter's Cf and the PZT view's "picked arrival" trigger come from the record's arrivals
    assert _czm_cf(app) == pytest.approx(abs(rec["rupture_speed"]))
    pv = PZTSpectrumView(app, 0, 1)
    pv.set_channel(2)
    pv.trigger_combobox.set(TRIGGER_PICKED)
    assert pv.compute(), pv.status_var.get()
    assert pv.window.trigger_time == pytest.approx(time[rec["picked_idx"][2]])


def test_records_keep_their_own_picks_and_entry_defaults(app_with_strain):
    dm = app_with_strain.data_manager
    event = dm.get_data(EVENT)
    add_pzt_record(event, [100.0, 200.0, 300.0, 400.0], fs=200e3)
    add_pzt_record(event, [100.0, 200.0], fs=2e3, key="ni")
    view = Picker(app_with_strain, 0, 1)
    assert view.record_key == "strain"
    assert list(view.record_combobox["values"]) == ["strain", "waveform/elsys", "waveform/ni"]
    view.pick_peaks()
    view.save()
    strain_speed = dm.get_data(f"{EVENT}/rupture_speed")
    strain_picks = list(event["strain"]["original"]["picked_idx"])
    _select_record(view, "waveform/elsys")
    assert view.record_key == "waveform/elsys" and len(view.picked_idx) == 4
    assert view.filter_high_khz.get() == "20" and view.filter_low_khz.get() == "0.2"   # the record's defaults
    view.pick_peaks()
    view.save()
    assert abs(read_arrivals(event["waveform"]["elsys"])["rupture_speed"]) == pytest.approx(1500.0, rel=0.05)
    # the strain results were not touched by picking on the PZTs
    assert dm.get_data(f"{EVENT}/rupture_speed") == pytest.approx(strain_speed)
    assert list(event["strain"]["original"]["picked_idx"]) == strain_picks
    # an edited entry is kept on the next record; the others follow the record
    view.filter_low_khz.set("1")
    _select_record(view, "waveform/ni")
    assert view.filter_high_khz.get() == "0.2" and view.filter_low_khz.get() == "1"
    assert view.filter_window_ms.get() == "25.5"                 # 51 samples at 2 kHz
    _select_record(view, "strain")
    assert view.picked_idx == strain_picks
    view.on_close()


def test_a_record_on_an_offset_with_unplaced_channels_and_no_trigger(app, tmp_path, app_from_file):
    """Like an NI record: eddy-current traces on a few volts of offset, pressure channels
    without a position, no trigger time."""
    event = app.data_manager.get_data(EVENT)
    t_event = float(event["event_time"])
    fs = 2000.0
    time = t_event - 0.1 + np.arange(1201) / fs
    xs = [31.0, 93.0, 155.0, 217.0]
    names = ["pressure_1", "pressure_2"] + [f"eddy_{i + 1}" for i in range(len(xs))]
    raw = np.zeros((len(names), time.size))
    raw[0], raw[1] = 3.6, 1.2
    rng = np.random.default_rng(0)
    for i, x in enumerate(xs):
        onset = t_event + 0.005 + (x - xs[0]) * 1e-3 / 50.0      # a slow front: 50 m/s
        raw[2 + i] = 5.0 + 0.3 * (1.0 + np.tanh((time - onset) / 1e-3)) + 0.002 * rng.standard_normal(time.size)
    pos = positions_table(names, {n: (x, 0.0, 0.0) for n, x in zip(names[2:], xs)})
    event.setdefault("waveform", {})["ni"] = channel_array(raw, names, unit="V", positions=pos, time=time,
                                                           sample_rate=fs, filename="ni.tdms")
    view = Picker(app, 0, 1)
    # the unplaced channels are hidden, not fitted, and parked past the last sensor
    assert view.record_key == "waveform/ni" and view.placed == [False, False, True, True, True, True]
    assert view.locations[2:] == xs and view.locations[:2] == [279.0, 341.0]
    assert view.enabled_channels == view.placed and view.fitting_channels == view.placed
    assert [view.table.item(str(i), "values")[1] for i in range(2)] == ["–", "–"]
    # no trigger: the picks start at the event time and the noise ends before it
    assert view.picked_idx[2] == int(np.argmin(np.abs(time - t_event)))
    assert "before the event time" in view.noise_note_var.get()
    lo, hi = view.ax_wave.get_ylim()
    for i in range(2, 6):                                     # each trace from its baseline, in its own band
        y = view.lines[i].get_data()[1]
        assert abs(np.median(y[:100]) - xs[i - 2]) < 1.0
        assert xs[i - 2] - 62.0 <= y.min() and y.max() <= xs[i - 2] + 62.0
        assert hi < y.min() and y.max() < lo
    view.auto_pick()
    assert abs(view.rupture_speed) == pytest.approx(50.0, rel=0.1)
    # an unplaced channel shown and asked to fit is drawn but kept out of the fit
    view.set_shown(0, True)
    view.set_fitting(0, True)
    assert view.lines[0] is not None and 0 not in view.fit_channels()
    assert "no position" in view.status_var.get()
    view.save()
    rec = read_arrivals(event["waveform"]["ni"])
    assert np.isnan(rec["locations"][:2]).all() and rec["locations"][2:] == xs
    view.on_close()
    path = tmp_path / "ni.h5"
    app.data_manager.save_file(path)
    view = Picker(app_from_file(path), 0, 1)
    _select_record(view, "waveform/ni")
    assert view.placed == [False, False, True, True, True, True] and view.locations[2:] == xs
    assert view.status_var.get().endswith("saved")


# ------------------------------------------------------------ rupture front
def test_bilateral_front_is_kept_per_record_through_hdf5(app_with_strain, tmp_path, app_from_file):
    event = app_with_strain.data_manager.get_data(EVENT)
    # onsets leave x = 230 mm both ways: off the middle of the sensors, so that the one line
    # through all of them has a slope (through a symmetric front it is level to rounding)
    time, t_trig = add_pzt_record(event, XS, x0=230.0)
    view = Picker(app_with_strain, 0, 1)
    _select_record(view, "waveform/elsys")
    for i, x in enumerate(XS):
        view.set_pick(i, int(np.argmin(np.abs(time - t_trig - abs(x - 230.0) * 1e-3 / 1500.0))), refit=False)
    view.update_fitted_line()
    # one line averages the two directions into a meaningless speed
    assert view.front_fit.mode == "unilateral" and len(view.front_fit.segments) == 1
    view.front_mode_var.set("bilateral")
    view.update_fitted_line()
    fit = view.front_fit
    assert fit.valid and len(fit.segments) == 2 and fit.split_x_mm == pytest.approx(230.0, abs=10.0)
    speeds = sorted(seg.velocity_m_s for seg in fit.segments)
    assert speeds[0] == pytest.approx(-1500.0, rel=0.05) and speeds[1] == pytest.approx(1500.0, rel=0.05)
    assert "fronts start at x = " in view.front_text_var.get() and "towards -x" in view.front_text_var.get()
    assert view.cf_label.cget("text").endswith("(one line through a front that runs both ways)")
    assert len(view.front_lines) == 5                      # two lines, two speed labels, the apex
    # residuals within a sample or two of the hinge
    assert max(abs(r) for r in view.residuals_s()) < 2.5 / 200e3 and view.table.item("0", "values")[5] != ""
    # Cf keeps its definition: one line through the same picks
    assert np.isfinite(view.rupture_speed) and view.fitted_line is None
    assert view.status_var.get().endswith("not saved")
    view.save()
    rec = read_arrivals(event["waveform"]["elsys"])
    assert rec["front_mode"] == "bilateral" and rec["front"]["valid"] is True
    assert len(rec["front"]["segments"]) == 2 and rec["front"]["use"] == [True] * 6
    assert view.status_var.get().endswith("saved")
    view.front_mode_var.set("unilateral")
    view.update_fitted_line()
    assert view.status_var.get().endswith("differs from the saved record")

    # the mode belongs to the record: the strain record comes up unilateral
    _select_record(view, "strain")
    assert view.front_mode_var.get() == "unilateral"
    view.pick_peaks()
    view.front_mode_var.set("bilateral")
    view.update_fitted_line()
    view.save()
    strain_split = view.front_fit.as_dict()["split_x_mm"]
    assert read_arrivals(event["strain"], event)["front_mode"] == "bilateral"
    assert "front_mode" not in event["strain"]["original"]       # the historical keys stay where they were
    view.on_close()

    path = tmp_path / "front.h5"
    app_with_strain.data_manager.save_file(path)
    view = Picker(app_from_file(path), 0, 1)
    assert view.record_key == "strain" and view.front_mode_var.get() == "bilateral"
    assert view.front_fit.as_dict()["split_x_mm"] == pytest.approx(strain_split)
    assert view.status_var.get().endswith("saved")
    _select_record(view, "waveform/elsys")
    assert view.front_mode_var.get() == "bilateral" and len(view.front_fit.segments) == 2
    assert view.status_var.get().endswith("saved")


# ---------------------------------------------------------------- auto pick
def test_auto_pick_methods_flags_and_hand_moves(app):
    event = app.data_manager.get_data(EVENT)
    xs = [100.0, 200.0, 300.0, 400.0]
    time, t_trig = add_pzt_record(event, xs, noise=0.01)
    onsets = [t_trig + (x - xs[0]) * 1e-3 / 1500.0 for x in xs]
    view = Picker(app, 0, 1)
    assert "before its trigger" in view.noise_note_var.get()
    for label, method in (("difference from the noise", "difference"), ("slope", "slope")):
        view.auto_method_var.set(label)
        picks = view.auto_pick()
        assert len(picks) == 4 and all(p.flag == "" for p in picks)
        for i, onset in enumerate(onsets):                 # within 0.1 ms of the tanh's midpoint
            assert abs(time[view.picked_idx[i]] - onset) < 1e-4
        assert abs(view.rupture_speed) == pytest.approx(1500.0, rel=0.1) and "auto pick (" in view.status_var.get()
        view.save()
        auto = read_arrivals(event["waveform"]["elsys"])["auto_pick"]
        assert auto["method"] == method and auto["threshold"] == 5.0 and auto["flags"] == [""] * 4
        assert auto["noise_end_s"] < -0.02                 # before the trigger, relative to the event
    view.on_close()

    event["waveform"]["elsys"]["data"][2] *= 0.0           # a dead channel: nothing to pick
    view = Picker(app, 0, 1)
    before = view.picked_idx[2]
    picks = view.auto_pick()
    assert picks[2].flag == ap.FLAG_WEAK and view.picked_idx[2] == before
    assert view.table.item("2", "values")[6] == "weak onset"
    assert "1 flagged (pzt_3: weak onset" in view.status_var.get()
    view.save()
    # reopening keeps the flag; a pick moved by hand clears it
    view.on_close()
    view = Picker(app, 0, 1)
    assert view.auto_flags[2] == ap.FLAG_WEAK
    view.set_pick(2, before + 10)
    assert view.auto_flags[2] == "" and view.table.item("2", "values")[6] == ""
    # unusable settings: the status says why and nothing moves
    picks_now = list(view.picked_idx)
    view.auto_threshold_var.set("0")
    assert view.auto_pick() is None and "threshold must be > 0" in view.status_var.get()
    view.auto_threshold_var.set("5")
    view.auto_smoothing_var.set("x")
    assert view.auto_pick() is None and "not a number" in view.status_var.get()
    assert view.picked_idx == picks_now


# --------------------------------------------------------------- pick sets
def test_named_pick_sets(app, tmp_path, app_from_file):
    from labquake_explorer.data.sources import arrival_sets, current_arrival_set
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    time, t_trig = add_pzt_record(event, [100.0, 200.0, 300.0, 400.0], noise=0.01)
    block = event["waveform"]["elsys"]
    view = Picker(app, 0, 1)
    assert view.pick_set == "default" and view.pick_set_combobox.get() == "default"
    assert "set 'default': not saved" in view.status_var.get()
    # onset picks saved as 'default'
    view.auto_method_var.set("difference from the noise")
    view.auto_pick()
    view.save()
    onsets = list(view.picked_idx)
    # a new name keeps the picks shown until saved; the peak picks go there
    view.choose_pick_set("peak")
    assert view.picked_idx == onsets and "new pick set 'peak'" in view.status_var.get()
    assert view.status_var.get().endswith("set 'peak': not saved")
    view.set_filter(tf.KIND_HIGHPASS, low_khz=1.0)
    view.pick_peaks()
    view.save()
    peaks = list(view.picked_idx)
    assert peaks != onsets
    assert arrival_sets(block) == ["default", "peak"] and current_arrival_set(block) == "peak"
    assert app.data_manager.data["analysis"]["pick_arrivals"]["current_set"] == "peak"
    assert list(view.pick_set_combobox["values"]) == ["default", "peak"]
    # the readers that know nothing of sets take the set saved last
    assert read_arrivals(block)["picked_idx"] == peaks
    assert _czm_cf(app) == pytest.approx(abs(read_arrivals(block)["rupture_speed"]))
    # choosing a saved set loads its picks and its filter; the zoom stays
    view.zoom_to_picks()
    xlim = view.ax_wave.get_xlim()
    view.choose_pick_set("default")
    assert view.picked_idx == onsets and view.filter_kind() == tf.KIND_OFF and view.ax_wave.get_xlim() == xlim
    assert view.status_var.get().endswith("set 'default': saved") and "pick set 'default' loaded" in view.status_var.get()
    view.choose_pick_set("peak")
    assert view.picked_idx == peaks and view.filter_kind() == tf.KIND_HIGHPASS
    view.choose_pick_set("a/b")                                         # refused, the set stays
    assert view.pick_set == "peak" and view.pick_set_combobox.get() == "peak" and "'/'" in view.status_var.get()
    # an event without that set opens on the record's defaults, still in the set chosen
    add_pzt_record(app.data_manager.get_data("runs/[0]/events/[2]"), [100.0, 200.0, 300.0, 400.0])
    view.set_event(2)
    assert view.pick_set == "peak" and view.status_var.get().endswith("set 'peak': not saved")
    view.on_close()
    # a file round trip; a new window opens on the experiment's set, also on an event without it
    path = tmp_path / "sets.h5"
    app.data_manager.save_file(path)
    reopened = Picker(app_from_file(path), 0, 1)
    assert reopened.pick_set == "peak" and reopened.picked_idx == peaks
    assert reopened.status_var.get().endswith("set 'peak': saved")
    reopened.choose_pick_set("default")
    assert reopened.picked_idx == onsets
    reopened.save()                                                    # 'default' is the experiment's set now
    assert Picker(reopened.app, 0, 2).pick_set == "default"
    reopened.app.data_manager.data["analysis"]["pick_arrivals"]["current_set"] = "a/b"   # unusable: the record's
    assert Picker(reopened.app, 0, 1).pick_set == "default"


def test_pick_sets_on_the_psu_strain_layout(app_with_strain, view):
    strain = view.event["strain"]
    view.pick_peaks()
    view.save()                                                         # 'default': the historical keys
    default = list(view.picked_idx)
    assert strain["original"]["picked_idx"] == default
    view.choose_pick_set("by hand")
    view.set_pick(view.fit_channels()[0], default[view.fit_channels()[0]] - 20)
    view.save()
    assert strain["original"]["picked_idx"] == default                  # left as it was
    assert read_arrivals(strain, view.event)["picked_idx"] == view.picked_idx
    assert read_arrivals(strain, view.event, "default")["rupture_speed"] == view.event["rupture_speed"]
    view.choose_pick_set("default")
    assert view.picked_idx == default


def test_an_event_without_picks_in_the_set_starts_from_its_last_save(app, tmp_path, app_from_file):
    from labquake_explorer.data.settings import get_setting
    xs = [100.0, 200.0, 300.0, 400.0]
    for j in (1, 2, 3):
        add_pzt_record(app.data_manager.get_data(f"runs/[0]/events/[{j}]"), xs, noise=0.01)
    view = Picker(app, 0, 1)
    view.set_filter(tf.KIND_BANDPASS, low_khz=1.0, high_khz=20.0, order=2)
    view.auto_method_var.set("slope")
    view.auto_threshold_var.set("7")
    view.front_mode_var.set("bilateral")
    view.set_shown(0, False)
    view.set_fitting(3, False)
    view.auto_pick()
    view.save()
    template = get_setting(app.data_manager.data, "pick_arrivals", "sets", "default", "records", "elsys")
    assert template["shown"] == ["pzt_2", "pzt_3", "pzt_4"] and template["fit"] == ["pzt_1", "pzt_2", "pzt_3"]
    assert template["auto_pick"]["method"] == "slope" and template["front_mode"] == "bilateral"
    # a fresh window on another event: the controls and the channel choice of the last save
    view.set_filter(tf.KIND_OFF)
    view.set_event(2)
    assert view.filter_kind() == tf.KIND_BANDPASS and view.filter_order_var.get() == "2"
    assert view.auto_method_var.get() == "slope" and view.auto_threshold_var.get() == "7"
    assert view.front_mode_var.get() == "bilateral"
    assert view.enabled_channels == [False, True, True, True] and view.fitting_channels == [True, True, True, False]
    assert view.status_var.get().endswith("not saved")
    # an event with picks of its own keeps them, its filter included
    view.set_filter(tf.KIND_OFF)
    view.set_shown(0, True)
    view.save()                                                        # event 2, and the template, now unfiltered
    view.set_event(1)
    assert view.filter_kind() == tf.KIND_BANDPASS and view.enabled_channels[0] is False
    # a set never saved leaves the controls as they are; choosing a saved set on an event without
    # its picks takes the set's controls and keeps the picks shown
    view.choose_pick_set("other")
    assert view.filter_kind() == tf.KIND_BANDPASS and list(view.pick_set_combobox["values"]) == ["default", "other"]
    view.set_event(3)
    picks = list(view.picked_idx)
    assert view.pick_set == "other" and view.filter_kind() == tf.KIND_BANDPASS
    view.choose_pick_set("default")
    assert view.filter_kind() == tf.KIND_OFF and view.enabled_channels == [True] * 4 and view.picked_idx == picks
    assert "its last controls, the picks shown" in view.status_var.get()
    view.on_close()
    # the template survives the file
    app.data_manager.save_file(tmp_path / "template.h5")
    reopened = Picker(app_from_file(tmp_path / "template.h5"), 0, 3)
    assert reopened.filter_kind() == tf.KIND_OFF and reopened.fitting_channels == [True, True, True, False]


def test_search_window_fit_the_patch_and_the_choice_carried_to_the_next_event(app):
    from labquake_explorer.data.settings import get_setting, set_setting
    xs = [100.0, 200.0, 300.0, 400.0, 500.0]
    for j in (2, 1):                                                   # event 1's time axis kept
        event = app.data_manager.get_data(f"runs/[0]/events/[{j}]")
        time, t_trig = add_pzt_record(event, xs, noise=0.01)
        late = 3.0 * np.exp(-0.5 * ((time - (t_trig + 0.05)) / 2e-5) ** 2)       # a larger burst 50 ms later
        event["waveform"]["elsys"]["data"] = event["waveform"]["elsys"]["data"] + late
    view = Picker(app, 0, 1)
    # D: the patch from the experiment fits only the sensors on it; no patch, a message
    view.fit_the_patch()
    assert "no patch edges" in view.status_var.get() and view.fitting_channels == [True] * 5
    set_setting(app.data_manager.data, [450.0, 150.0], "fault", "patch_x_mm")
    view.set_shown(1, False)
    view.fit_the_patch()
    assert view.fitting_channels == [False, True, True, True, False]
    assert view.enabled_channels == [True] * 5 and "fitting the 3 sensors on the patch (150 to 450 mm)" in view.status_var.get()
    # B: the whole record takes the early burst; the range shown, the arrivals
    view.auto_method_var.set("peak")
    view.auto_pick()
    t_event = float(view.event["event_time"])
    assert all(abs(view.relative_time_s()[k] - (t_trig + 0.05 - t_event)) < 1e-4 for k in view.picked_idx)
    view.ax_wave.set_xlim((t_trig - t_event - 0.002) * 1e3, (t_trig - t_event + 0.003) * 1e3)
    view.use_range_shown()
    assert float(view.search_from_var.get()) == pytest.approx((t_trig - t_event - 0.002) * 1e3, rel=1e-3)
    view.auto_method_var.set("difference from the noise")
    view.auto_pick()
    onsets = [t_trig + (x - xs[0]) * 1e-3 / 1500.0 for x in xs]
    assert all(abs(time[view.picked_idx[i]] - onsets[i]) < 1e-4 for i in range(5))
    assert abs(view.rupture_speed) == pytest.approx(1500.0, rel=0.1)
    assert "shortened to end at the search window's start" not in view.noise_note_var.get()
    view.search_from_var.set(f"{(t_trig - t_event - 0.03) * 1e3:g}")          # before the noise's end: shortened
    view.auto_pick()
    assert "shortened" in view.noise_note_var.get()
    view.search_from_var.set("x")
    assert view.auto_pick() is None and "search 'x' is not a time in ms" in view.status_var.get()
    view.search_from_var.set("5")
    view.search_to_var.set("1")
    assert view.auto_pick() is None and "start before it ends" in view.status_var.get()
    view.search_from_var.set(f"{(t_trig - t_event - 0.002) * 1e3:.4g}")
    view.search_to_var.set(f"{(t_trig - t_event + 0.003) * 1e3:.4g}")
    view.auto_pick()
    view.save()
    auto = read_arrivals(view.event["waveform"]["elsys"])["auto_pick"]
    assert auto["search_from_s"] == pytest.approx(t_trig - t_event - 0.002, abs=1e-6)
    tpl = get_setting(app.data_manager.data, "pick_arrivals", "sets", "default", "records", "elsys")["auto_pick"]
    assert tpl["search_to_s"] == pytest.approx(t_trig - t_event + 0.003, abs=1e-6)
    # the next event without picks: the set's last save (the search included); without a save, what was shown
    view.search_from_var.set("")
    view.set_event(2)
    assert view.search_from_var.get() != "" and view.fitting_channels == [False, True, True, True, False]
    view.on_close()
    del app.data_manager.data["analysis"]["pick_arrivals"]
    del app.data_manager.get_data("runs/[0]/events/[1]")["waveform"]["elsys"]["arrivals"]
    view = Picker(app, 0, 1)
    assert view.fitting_channels == [True] * 5
    view.set_fitting(0, False)
    view.set_event(2)
    assert view.fitting_channels == [False, True, True, True, True]


def test_a_record_saved_with_single_opens_unilateral(app_with_strain, view):
    view.front_mode_var.set("bilateral")
    view.pick_peaks()
    view.save()
    view.event["strain"]["arrivals"]["sets"]["default"]["front_mode"] = "single"     # as saved before 0.3.0
    view.on_close()
    reopened = Picker(app_with_strain, 0, 1)
    assert reopened.front_mode_var.get() == "unilateral" and reopened.front_fit.mode == "unilateral"


def test_clicks_that_would_drop_unsaved_picks_ask_first(app_with_strain, view, monkeypatch):
    """The event box, the arrows, the record box and a saved pick set ask before
    unsaved picks go; "No" keeps them.  The methods a script calls never ask."""
    from tkinter import messagebox
    asked = []
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: asked.append(a) or False)
    view.pick_peaks()
    view.save()
    view.event_combobox.set("2")
    view.event_combobox.event_generate("<<ComboboxSelected>>")
    assert asked == [] and view.event_idx == 2                  # nothing unsaved: no question
    view.event_combobox.set("1")
    view.event_combobox.event_generate("<<ComboboxSelected>>")
    view.set_pick(0, view.picked_idx[0] + 5)
    edited = list(view.picked_idx)
    view.event_combobox.set("2")
    view.event_combobox.event_generate("<<ComboboxSelected>>")
    view.next_event_button.invoke()
    view.save()
    view.choose_pick_set("other")                              # a new set keeps the picks shown: no question
    view.set_pick(0, view.picked_idx[0] + 5)
    view.pick_set_combobox.set("default")
    view.pick_set_combobox.event_generate("<<ComboboxSelected>>")
    assert len(asked) == 3 and view.event_idx == 1 and view.event_combobox.get() == "1"
    assert view.pick_set == "other" and view.picked_idx[0] == edited[0] + 5
    view.set_event(2)                                          # a script's call goes ahead
    assert view.event_idx == 2 and len(asked) == 3
