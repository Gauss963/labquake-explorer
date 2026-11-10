import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from labquake_explorer.ui.views import CZMFitterView
from tests.synthetic import add_strain

EXPECTED_KEYS = {"Cf", "y", "Xc", "Gc", "x_min", "x_tip", "x_max", "x_lim_min", "x_lim_max", "strain_gauge"}


@pytest.fixture
def view(app_with_strain):
    v = CZMFitterView(app_with_strain, 0, 1)
    yield v
    try:
        v.on_close()
    except Exception:
        pass


def positions(v):
    return [float(line.get_xdata()[0]) for line in v.vlines]


def mouse_event(v, name, xdata, button=1):
    """A matplotlib MouseEvent on the top axes at data x ``xdata``."""
    ax = v.axs[0]
    ydata = float(np.mean(ax.get_ylim()))
    x, y = ax.transData.transform((xdata, ydata))
    return MouseEvent(name, v.canvas, x, y, button=button)


def dispatch(v, name, xdata, button=1):
    """Send a MouseEvent through the canvas callback registry (mpl_connect path)."""
    v.canvas.callbacks.process(name, mouse_event(v, name, xdata, button=button))


def test_open_lists_gauges(app_with_strain, view):
    assert view.event_idx == 1
    assert view.title() == "Cohesive Zone Model Fitting - Event 1"
    assert view in app_with_strain.child_windows
    assert view.num_gauges == 16
    assert list(view.gauge_combobox["values"]) == [str(i) for i in range(16)]
    assert view.strain_gauge.get() == 6
    assert view.event_combobox.get() == "1"
    # three draggable marker pairs on the two shared axes
    assert len(view.vlines) == 3 and len(view.vlines_twin) == 3
    assert positions(view) == pytest.approx([-0.05, 0.0, 0.05])
    assert [float(l.get_xdata()[0]) for l in view.vlines_twin] == pytest.approx(positions(view))
    assert view.axs[0].get_shared_x_axes().joined(view.axs[0], view.axs[1])
    assert view.axs[0].get_xlim() == pytest.approx((-0.1, 0.1))
    # default Cf falls back to 10 when the event has no rupture_speed
    assert view.Cf.get() == 10
    assert view.y.get() == pytest.approx(8e-3)


def test_switch_event_reloads(app_with_strain, view):
    view.set_event(2)
    assert view.event_idx == 2
    assert view.title() == "Cohesive Zone Model Fitting - Event 2"
    assert view.event is app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    assert view.event_combobox.get() == "2"
    assert len(view.vlines) == 3
    assert view.figure._suptitle.get_text().endswith("event2")


def test_save_writes_dict(app_with_strain, view):
    view.Cf.set(1000.0)
    view.strain_gauge.set(3)
    view.save_parameters()
    saved = app_with_strain.data_manager.get_data("runs/[0]/events/[1]/czm_parms")
    assert isinstance(saved, dict)
    assert set(saved) == EXPECTED_KEYS
    assert saved is view.event["czm_parms"]
    assert saved["Cf"] == 1000.0
    assert saved["strain_gauge"] == 3
    assert saved["x_min"] < saved["x_tip"] < saved["x_max"]
    assert (saved["x_lim_min"], saved["x_lim_max"]) == pytest.approx((-0.1, 0.1))
    # the tree shows the new key
    tree = app_with_strain.data_tree
    labels = []

    def walk(item):
        for child in tree.get_children(item):
            labels.append(tree.item(child, "text"))
            walk(child)
    walk("")
    assert any(label.startswith("czm_parms") for label in labels)


def test_reopen_restores_parameters_and_lines(app_with_strain, view):
    view.Cf.set(1000.0)
    view.y.set(5e-3)
    view.Xc.set(2.5)
    view.Gc.set(3.0)
    view.move_line(0, -0.006)
    view.move_line(1, -0.001)
    view.move_line(2, 0.004)
    view.axs[0].set_xlim(-0.02, 0.02)
    view.save_parameters()
    saved = view.event["czm_parms"]
    assert saved["x_min"] == pytest.approx(-0.006)
    assert saved["x_tip"] == pytest.approx(-0.001)
    assert saved["x_max"] == pytest.approx(0.004)

    view.set_event(2)
    assert positions(view) == pytest.approx([-0.05, 0.0, 0.05])
    assert view.Cf.get() == 10

    view.set_event(1)
    assert view.Cf.get() == 1000.0
    assert view.y.get() == pytest.approx(5e-3)
    assert view.Xc.get() == pytest.approx(2.5)
    assert view.Gc.get() == pytest.approx(3.0)
    assert positions(view) == pytest.approx([-0.006, -0.001, 0.004])
    assert [float(l.get_xdata()[0]) for l in view.vlines_twin] == pytest.approx([-0.006, -0.001, 0.004])
    assert view.axs[0].get_xlim() == pytest.approx((-0.02, 0.02))

    # a fresh window on the same event also restores everything
    other = CZMFitterView(app_with_strain, 0, 1)
    try:
        assert other.Cf.get() == 1000.0
        assert positions(other) == pytest.approx([-0.006, -0.001, 0.004])
        assert other.strain_gauge.get() == saved["strain_gauge"]
    finally:
        other.on_close()


def test_legacy_list_is_read(app_with_strain, app_from_file, tmp_path):
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[3]")
    event["czm_parms"] = [500.0, 5e-3, 2.0, 3.0, -0.004, 0.0, -0.02, 0.02]
    v = CZMFitterView(app_with_strain, 0, 3)
    try:
        assert v.Cf.get() == 500.0
        assert v.y.get() == pytest.approx(5e-3)
        assert v.Xc.get() == 2.0
        assert v.Gc.get() == 3.0
        assert positions(v) == pytest.approx([-0.004, 0.0, 0.004])
        assert v.axs[0].get_xlim() == pytest.approx((-0.02, 0.02))
        assert v.strain_gauge.get() == 6
        # loading does not rewrite the stored value; saving does
        assert isinstance(event["czm_parms"], list)
    finally:
        v.on_close()

    # the legacy list comes back from an HDF5 file as a float ndarray: still read
    path = tmp_path / "legacy_czm.h5"
    app_with_strain.data_manager.save_file(path)
    app2 = app_from_file(path)
    stored = app2.data_manager.get_data("runs/[0]/events/[3]/czm_parms")
    assert isinstance(stored, np.ndarray) and stored.shape == (8,)
    v2 = CZMFitterView(app2, 0, 3)
    try:
        assert v2.Cf.get() == 500.0
        assert v2.y.get() == pytest.approx(5e-3)
        assert positions(v2) == pytest.approx([-0.004, 0.0, 0.004])
        assert v2.axs[0].get_xlim() == pytest.approx((-0.02, 0.02))
        assert isinstance(app2.data_manager.get_data("runs/[0]/events/[3]/czm_parms"), np.ndarray)
        v2.save_parameters()
        saved = app2.data_manager.get_data("runs/[0]/events/[3]/czm_parms")
        assert isinstance(saved, dict) and set(saved) == EXPECTED_KEYS
        assert saved["Cf"] == 500.0 and saved["x_max"] == pytest.approx(0.004)
    finally:
        v2.on_close()

    # in memory the list is still a list; saving rewrites it as a dict
    assert isinstance(event["czm_parms"], list)
    v = CZMFitterView(app_with_strain, 0, 3)
    try:
        v.save_parameters()
        assert isinstance(event["czm_parms"], dict)
        assert set(event["czm_parms"]) == EXPECTED_KEYS
        assert event["czm_parms"]["x_max"] == pytest.approx(0.004)
    finally:
        v.on_close()


def test_save_prints_confirmation(app_with_strain, view, capsys):
    view.Cf.set(1000.0)
    view.save_parameters()
    out = capsys.readouterr().out
    assert "Saved parameters for event 1" in out and "'Cf': 1000.0" in out


def test_switch_to_event_without_strain_is_refused(app_with_strain, view, no_dialogs):
    """Selecting an event that has no strain block warns and leaves the view
    untouched (index, title, traces, marker lines), like the arrival picker."""
    events = app_with_strain.data_manager.get_data("runs/[0]/events")
    del events[2]["strain"]
    view.move_line(1, 0.002)
    before = positions(view)
    view.event_combobox.set("2")
    view.on_event_selected()
    assert view.event_idx == 1
    assert view.title() == "Cohesive Zone Model Fitting - Event 1"
    assert view.event_combobox.get() == "1"
    assert view.event is events[1]
    assert positions(view) == pytest.approx(before)
    assert view.figure._suptitle.get_text().endswith("event1")
    assert any("No strain data" in str(call) for call in no_dialogs)
    # the view still works afterwards
    view.update_plot()
    assert view.build_fit_objective() is not None
    view.save_parameters()
    assert events[1]["czm_parms"]["x_tip"] == pytest.approx(0.002)
    # switching to a good event still works
    view.set_event(3)
    assert view.event_idx == 3 and len(view.vlines) == 3


def test_open_on_event_without_strain_fails_cleanly(app_with_strain):
    events = app_with_strain.data_manager.get_data("runs/[0]/events")
    del events[2]["strain"]
    with pytest.raises(ValueError, match="no strain data"):
        CZMFitterView(app_with_strain, 0, 2)
    assert app_with_strain.child_windows == []


def test_fewer_than_15_channels_opens_without_eyy_panel(app_with_strain):
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    add_strain(event, n_channels=8)
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        assert v.num_gauges == 8 and v.eyy_gauge() is None
        assert v.strain_gauge.get() == 6
        assert len(v.vlines) == 3 and len(v.vlines_twin) == 3
        labels = [l.get_label() for l in v.axs[0].get_lines()]
        assert "Exy" in labels and "CZM" in labels
        assert [l.get_label() for l in v.axs[1].get_lines() if l.get_label() in ("Eyy", "CZM")] == []
        assert any("no Eyy gauge" in t.get_text() for t in v.axs[1].texts)
        # the Exy fit and save still work without the Eyy panel
        v.Cf.set(1000.0)
        result = v.fit_parameters()
        assert result is not None and result.success
        v.save_parameters()
        assert set(event["czm_parms"]) == EXPECTED_KEYS
        # a full array on another event restores the Eyy panel
        v.set_event(1)
        assert v.eyy_gauge() == 14
        assert "Eyy" in [l.get_label() for l in v.axs[1].get_lines()]
    finally:
        v.on_close()


def test_filter_toggle_and_update(app_with_strain, view):
    assert view.filter_button["text"] == "Filter Off"
    view.toggle_filter()
    assert view.filtering is True
    assert view.filter_button["text"] == "Filter On"
    view.filter_window.set(50)  # even -> corrected to odd
    view.update_plot()
    assert view.filter_window.get() == 51
    view.gauge_combobox.set("2")
    view.update_plot()
    assert view.strain_gauge.get() == 2
    view.toggle_filter()
    assert view.filtering is False
    assert view.filter_button["text"] == "Filter Off"
    assert view.validate_filter_window("51") is True
    assert view.validate_filter_window("50") is False
    assert view.validate_filter_window("") is True
    assert view.validate_filter_window("abc") is False


def test_fit_improves_objective(app_with_strain, view):
    view.Cf.set(1000.0)
    initial_guess = [view.Gc.get(), view.Xc.get()]
    objective = view.build_fit_objective()
    assert objective is not None
    initial_fun = objective(initial_guess)
    assert np.isfinite(initial_fun) and initial_fun > 0

    result = view.fit_parameters()
    assert result is not None
    assert result.success, result.message
    assert result.initial_fun == pytest.approx(initial_fun)
    # the optimizer actually did something: lower objective, parameters moved
    assert result.fun < initial_fun
    assert result.fun == pytest.approx(objective(result.x))
    assert (view.Gc.get(), view.Xc.get()) != tuple(initial_guess)
    assert view.Gc.get() == pytest.approx(result.x[0])
    assert view.Xc.get() == pytest.approx(result.x[1])
    assert view.Gc.get() >= 1e-6 and view.Xc.get() >= 1e-6  # within the bounds
    # the plotted CZM curve follows the fitted parameters
    assert any(line.get_label() == "CZM" for line in view.axs[0].get_lines())

    # fitting again with the filter on also succeeds and does not get worse
    view.toggle_filter()
    result2 = view.fit_parameters()
    assert result2 is not None and result2.success
    assert result2.fun <= result2.initial_fun + 1e-9 * max(1.0, abs(result2.initial_fun))


def test_fit_without_samples_returns_none(app_with_strain, view):
    # tip and max lines beyond the end of the strain record -> no samples in window
    t_end = float(view._time()[-1])
    view.move_line(1, t_end + 1.0)
    view.move_line(2, t_end + 2.0)
    gc, xc = view.Gc.get(), view.Xc.get()
    assert view.build_fit_objective() is None
    assert view.fit_parameters() is None
    assert (view.Gc.get(), view.Xc.get()) == (gc, xc)


def test_drag_line_with_mouse_events(app_with_strain, view):
    view.canvas.draw()
    assert positions(view) == pytest.approx([-0.05, 0.0, 0.05])

    # press far from any line: nothing is grabbed, motion does nothing
    dispatch(view, "button_press_event", 0.025)
    assert view.drag_active is False and view.active_line_idx is None
    dispatch(view, "motion_notify_event", 0.03)
    assert positions(view) == pytest.approx([-0.05, 0.0, 0.05])
    dispatch(view, "button_release_event", 0.03)

    # right button does not grab either
    dispatch(view, "button_press_event", 0.0, button=3)
    assert view.drag_active is False
    dispatch(view, "button_release_event", 0.0, button=3)

    # left press on the tip line grabs it; motion drags both marker pairs
    dispatch(view, "button_press_event", 0.0)
    assert view.drag_active is True and view.active_line_idx == 1
    dispatch(view, "motion_notify_event", 0.02)
    assert positions(view) == pytest.approx([-0.05, 0.02, 0.05])
    assert [float(l.get_xdata()[0]) for l in view.vlines_twin] == pytest.approx([-0.05, 0.02, 0.05])
    assert view.line_positions == pytest.approx([-0.05, 0.02, 0.05])

    # while the toolbar is in pan mode the drag is ignored
    view.toolbar.pan()
    assert view.toolbar_active()
    dispatch(view, "motion_notify_event", 0.04)
    assert positions(view) == pytest.approx([-0.05, 0.02, 0.05])
    view.toolbar.pan()
    assert not view.toolbar_active()

    # ... and resumes once pan is switched off
    dispatch(view, "motion_notify_event", 0.03)
    assert positions(view) == pytest.approx([-0.05, 0.03, 0.05])

    # release ends the drag; further motion does nothing
    dispatch(view, "button_release_event", 0.03)
    assert view.drag_active is False and view.active_line_idx is None
    dispatch(view, "motion_notify_event", 0.06)
    assert positions(view) == pytest.approx([-0.05, 0.03, 0.05])

    # a press while pan mode is on is ignored too
    view.toolbar.pan()
    dispatch(view, "button_press_event", 0.03)
    assert view.drag_active is False
    view.toolbar.pan()

    # the dragged position is what gets saved
    view.save_parameters()
    assert view.event["czm_parms"]["x_tip"] == pytest.approx(0.03)


def test_close_unregisters(app_with_strain):
    v = CZMFitterView(app_with_strain, 0, 0)
    assert v in app_with_strain.child_windows
    v.on_close()
    assert v not in app_with_strain.child_windows


# ---------------------------------------------------------------- stress records, experiment medium
MEDIUM = {"E_Pa": 7.662e9, "nu": 0.2, "rho_kg_m3": 1148.0}      # wave speeds follow: C_s 1667.6, C_d 2723.2 m/s
TRUE = {"Cf": 200.0, "y": 1e-3, "Xc": 5e-3, "Gc": 2.0}


def add_stress_record(app, event, with_positions=True):
    """Replace the event's strain block by a waveform record in MPa computed from
    the cohesive-zone model itself (``TRUE``, tip passing at the event time) and
    give the experiment the medium it was computed with."""
    from labquake_explorer.data.channels import channel_array, positions_table
    from labquake_explorer.ui.views.czm_fitter_view import experiment_medium
    from labquake_explorer.utils.cohesive_crack import CohesiveCrack
    app.data_manager.data["analysis"] = {"medium": dict(MEDIUM)}
    m = experiment_medium(app.data_manager.data)
    t = np.arange(-2000, 2001) / 1e6
    _, sxy, syy = CohesiveCrack.delta_sigmas(-t * TRUE["Cf"], TRUE["y"], TRUE["Xc"], TRUE["Cf"],
                                             m["C_s"], m["C_d"], m["nu"], TRUE["Gc"], m["E"])
    # more channels than DEFAULT_GAUGE, so that index stays valid and must not be what the view opens on
    names = ["rosette_1_sigma_yy", "shear_1_sigma_xy", "rosette_1_sigma_xy"] + [f"spare_{i}_sigma_yy" for i in range(5)]
    record = channel_array(np.vstack([syy, sxy + 3.0e6, sxy] + [syy] * 5) / 1e6, names, unit="MPa", dtype=np.float64,
                           positions=positions_table(names, {n: (100.0, 1.0, np.nan) for n in names})
                           if with_positions else None,
                           time=float(event["event_time"]) + t, sample_rate=1e6, block=2)
    event.pop("strain", None)
    event.pop("czm_parms", None)
    event["waveform"] = {"elsys": record}
    return m


def test_experiment_medium_rules():
    from labquake_explorer.ui.views.czm_fitter_view import experiment_medium
    assert experiment_medium({}) is None and experiment_medium({"analysis": {"medium": {}}}) is None
    # the PZT view's medium (density and wave speeds only) does not define the CZM medium
    assert experiment_medium({"analysis": {"medium": {"rho_kg_m3": 2650.0, "vp_m_s": 5000.0, "vs_m_s": 3000.0}}}) is None
    assert experiment_medium({"analysis": {"medium": {"E_Pa": 7.662e9, "nu": 0.2}}}) is None   # no speeds, no density
    m = experiment_medium({"analysis": {"medium": MEDIUM}})
    assert m["E"] == 7.662e9 and m["nu"] == 0.2
    assert m["C_s"] == pytest.approx(1667.6, abs=0.1) and m["C_d"] == pytest.approx(2723.2, abs=0.1)
    given = experiment_medium({"analysis": {"medium": {"E_Pa": np.array(51e9), "nu": 0.25, "vs_m_s": 2760.0, "vp_m_s": 4790.0}}})
    assert given == {"E": 51e9, "nu": 0.25, "C_s": 2760.0, "C_d": 4790.0}


def test_volts_record_keeps_the_default_medium_and_saved_keys(app_with_strain, view):
    assert not view.stress_mode and view.medium_source == "default"
    assert (view.E, view.nu, view.C_s, view.C_d) == (51e9, 0.25, 2760, 4790)
    assert view.channel_var.get() == "" and "default" in view.medium_var.get()
    view.save_parameters()
    assert set(view.event["czm_parms"]) == EXPECTED_KEYS


def test_stress_record_is_fitted_as_stress(app_with_strain):
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    medium = add_stress_record(app_with_strain, event)
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        assert v.stress_mode and v.medium_source == "experiment"
        assert (v.E, v.nu, v.C_s, v.C_d) == (medium["E"], medium["nu"], medium["C_s"], medium["C_d"])
        # opens on the first sigma_xy channel, at the sensor's distance, on the record's own span
        assert v.strain_gauge.get() == 1 and v.channel_var.get() == "shear_1_sigma_xy"
        assert v.y.get() == pytest.approx(1e-3)
        assert v.axs[0].get_xlim() == pytest.approx((-2e-3, 2e-3))
        assert positions(v) == pytest.approx([-1e-3, 0.0, 1e-3])
        assert v.axs[0].get_ylabel() == "sigma_xy (MPa)" and v.axs[1].get_ylabel() == "sigma_yy (MPa)"
        assert v.eyy_gauge() is None and any("no sigma_yy channel" in t.get_text() for t in v.axs[1].texts)
        # the rosette's sigma_xy brings its sigma_yy into the lower panel
        v.gauge_combobox.set("2")
        v.update_plot()
        assert v.eyy_gauge() == 0 and "sigma_yy" in [l.get_label() for l in v.axs[1].get_lines()]
        v.set_event(1), v.set_event(2)                                    # coming back keeps the channel in use
        assert v.strain_gauge.get() == 2
        # with the true parameters the model lies on the record (zeroed at x_max, drawn in MPa)
        v.Cf.set(TRUE["Cf"]), v.Xc.set(TRUE["Xc"]), v.Gc.set(TRUE["Gc"])
        v.update_plot()
        measured, model = (l.get_ydata() for l in v.axs[0].get_lines() if l.get_label() in ("sigma_xy", "CZM"))
        np.testing.assert_allclose(measured, model, atol=1e-6 * np.ptp(measured))
        assert 0.05 < np.ptp(measured) < 50                               # MPa, not Pa
        assert v.build_fit_objective()([TRUE["Gc"], TRUE["Xc"]]) < 1e-9
        # from another starting point the fit comes back to them
        v.Xc.set(8e-3), v.Gc.set(1.0)
        result = v.fit_parameters()
        assert result is not None and result.success and result.fun < 1e-3 * result.initial_fun
        assert v.Gc.get() == pytest.approx(TRUE["Gc"], rel=0.05) and v.Xc.get() == pytest.approx(TRUE["Xc"], rel=0.05)
        v.save_parameters()
        saved = event["czm_parms"]
        assert set(saved) == EXPECTED_KEYS | {"channel", "unit", "E", "nu", "C_s", "C_d", "tau_p"}
        assert saved["channel"] == "rosette_1_sigma_xy" and saved["unit"] == "MPa" and saved["strain_gauge"] == 2
        assert saved["E"] == medium["E"] and saved["C_s"] == medium["C_s"] and 1e4 < saved["tau_p"] < 1e8
    finally:
        v.on_close()
    again = CZMFitterView(app_with_strain, 0, 2)                         # reopening restores channel and parameters
    try:
        assert again.strain_gauge.get() == 2 and again.Gc.get() == pytest.approx(saved["Gc"])
        assert again.Xc.get() == pytest.approx(saved["Xc"]) and positions(again) == pytest.approx([-1e-3, 0.0, 1e-3])
    finally:
        again.on_close()


def test_stress_record_without_positions_or_medium_uses_the_defaults(app_with_strain):
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    add_stress_record(app_with_strain, event, with_positions=False)
    del app_with_strain.data_manager.data["analysis"]
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        assert v.stress_mode and v.medium_source == "default" and v.E == 51e9
        assert v.y.get() == pytest.approx(8e-3)
        v.save_parameters()
        assert event["czm_parms"]["unit"] == "MPa" and event["czm_parms"]["E"] == 51e9
        v.set_event(1)                                                   # back on a volts record: strain again
        assert not v.stress_mode and v.axs[0].get_ylabel() == "Exy"
    finally:
        v.on_close()
