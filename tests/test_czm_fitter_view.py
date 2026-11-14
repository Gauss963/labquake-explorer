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
STRESS_NAMES = ["rosette_1_sigma_yy", "shear_1_sigma_xy", "rosette_1_sigma_xy", "rosette_1_sigma_xx",
                "spare_0", "spare_1", "spare_2", "spare_3"]


def add_stress_record(app, event, with_positions=True):
    """Replace the event's strain block by a waveform record in MPa computed from
    the cohesive-zone model itself (``TRUE``, tip passing at the event time) and
    give the experiment the medium it was computed with: a shear gauge (xy
    only, on a 3 MPa level) and a rosette (xx, xy, yy), plus spare channels so
    that ``DEFAULT_GAUGE`` stays a valid index the view must not open on."""
    from labquake_explorer.data.channels import channel_array, positions_table
    from labquake_explorer.ui.views.czm_fitter_view import experiment_medium
    from labquake_explorer.utils.cohesive_crack import CohesiveCrack
    app.data_manager.data["analysis"] = {"medium": dict(MEDIUM)}
    m = experiment_medium(app.data_manager.data)
    t = np.arange(-2000, 2001) / 1e6
    sxx, sxy, syy = CohesiveCrack.delta_sigmas(-t * TRUE["Cf"], TRUE["y"], TRUE["Xc"], TRUE["Cf"],
                                               m["C_s"], m["C_d"], m["nu"], TRUE["Gc"], m["E"])
    rows = [syy - 60e6, sxy + 3.0e6, sxy, sxx + 40e6] + [0 * t] * 4
    record = channel_array(np.vstack(rows) / 1e6, STRESS_NAMES, unit="MPa", dtype=np.float64,
                           positions=positions_table(STRESS_NAMES, {n: (100.0, 1.0, np.nan) for n in STRESS_NAMES})
                           if with_positions else None,
                           time=float(event["event_time"]) + t, sample_rate=1e6, block=2)
    event.pop("strain", None)
    event.pop("czm_parms", None)
    event["waveform"] = {"elsys": record}
    return m


def drawn(ax, label):
    """y data of the line with ``label`` on ``ax``."""
    return next(l.get_ydata() for l in ax.get_lines() if l.get_label() == label)


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


def test_volts_record_keeps_the_default_medium_layout_and_saved_keys(app_with_strain, view):
    assert not view.stress_mode and view.medium_source == "default"
    assert (view.E, view.nu, view.C_s, view.C_d) == (51e9, 0.25, 2760, 4790)
    assert view.channel_var.get() == "" and "default" in view.medium_var.get()
    assert len(view.axs) == 2 and view.vlines_other == [] and view.fit_components() == ["xy"]
    assert not view.fit_frame.winfo_manager()                            # the Fit boxes are for stress records
    view.save_parameters()
    assert set(view.event["czm_parms"]) == EXPECTED_KEYS


def test_stress_record_opens_by_sensor(app_with_strain):
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    medium = add_stress_record(app_with_strain, event)
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        assert v.stress_mode and v.medium_source == "experiment"
        assert (v.E, v.nu, v.C_s, v.C_d) == (medium["E"], medium["nu"], medium["C_s"], medium["C_d"])
        # the box lists the record's sensors by name (two, for eight channels); opens on the first,
        # at the sensor's distance, on the record's span
        assert list(v.gauge_combobox["values"]) == ["shear_1", "rosette_1"] and v.sensors() == {"shear_1": 1, "rosette_1": 2}
        assert v.gauge_combobox.get() == "shear_1" and v.strain_gauge.get() == 1 and v.channel_var.get() == "xy"
        assert v.y.get() == pytest.approx(1e-3)
        assert v.axs[0].get_xlim() == pytest.approx((-2e-3, 2e-3))
        assert positions(v) == pytest.approx([-1e-3, 0.0, 1e-3])
        # three panels, the three marker lines on each
        assert [ax.get_ylabel() for ax in v.axs] == ["sigma_xy (MPa)", "sigma_xx (MPa)", "sigma_yy (MPa)"]
        assert len(v.vlines) == len(v.vlines_twin) == 3 and [len(lines) for lines in v.vlines_other] == [3]
        v.move_line(2, 1.5e-3)
        assert all(float(lines[2].get_xdata()[0]) == pytest.approx(1.5e-3)
                   for lines in (v.vlines, v.vlines_twin, v.vlines_other[0]))
        v.move_line(2, 1e-3)
        # a shear gauge has xy alone: the other panels say so and only xy can be fitted
        assert v.sensor_channels() == {"xy": 1} and v.fit_components() == ["xy"]
        assert any("no sigma_xx channel" in t.get_text() for t in v.axs[1].texts)
        assert any("no sigma_yy channel" in t.get_text() for t in v.axs[2].texts)
        assert v.fit_frame.winfo_manager() and v.fit_boxes["xx"].instate(["disabled"])
        # the rosette brings its three components
        v.gauge_combobox.set("rosette_1")
        v.update_plot()
        assert v.strain_gauge.get() == 2 and v.channel_var.get() == "xy, xx, yy"
        assert v.sensor_channels() == {"xy": 2, "xx": 3, "yy": 0} and v.fit_components() == ["xy", "xx", "yy"]
        assert v.fit_boxes["xx"].instate(["!disabled"])
        v.set_event(1)                                                    # a volts record: channels by index again
        assert list(v.gauge_combobox["values"]) == [str(i) for i in range(16)] and v.gauge_combobox.get() == "2"
        v.set_event(2)                                                    # back: the same sensor
        assert v.gauge_combobox.get() == "rosette_1" and v.strain_gauge.get() == 2 and len(v.axs) == 3
    finally:
        v.on_close()


def test_rosette_components_are_fitted_together(app_with_strain):
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    medium = add_stress_record(app_with_strain, event)
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        v.gauge_combobox.set("rosette_1")
        v.Cf.set(TRUE["Cf"]), v.Xc.set(TRUE["Xc"]), v.Gc.set(TRUE["Gc"])
        v.update_plot()
        # with the true parameters the model lies on every component (background removed, in MPa); it is
        # drawn black inside the fit window (tip .. fit end) and as a thin dashed line over the whole record
        inside = (v._time() >= 0.0) & (v._time() <= 1e-3)
        for ax, component in zip(v.axs, ("xy", "xx", "yy")):
            measured, model = drawn(ax, f"sigma_{component}"), drawn(ax, "CZM")
            assert model.size == int(inside.sum())
            np.testing.assert_allclose(measured[inside], model, atol=1e-6 * np.ptp(measured))
            whole = next(l.get_ydata() for l in ax.get_lines() if l.get_linestyle() == "--" and l.get_color() == "0.35"
                         and l.get_ydata().size == measured.size)
            np.testing.assert_allclose(measured, whole, atol=1e-6 * np.ptp(measured))
            assert 0.01 < np.ptp(measured) < 50                           # MPa, not Pa, and not the 40 / 60 MPa levels
        joint = v.build_fit_objective()
        assert joint([TRUE["Gc"], TRUE["Xc"]]) < 1e-9
        # the joint objective is the sum of the single-component ones
        off = [1.3 * TRUE["Gc"], 0.7 * TRUE["Xc"]]
        single = []
        for component in ("xy", "xx", "yy"):
            for name, var in v.fit_vars.items():
                var.set(name == component)
            assert v.fit_components() == [component]
            single.append(v.build_fit_objective()(off))
        assert all(value > 0 for value in single)
        for var in v.fit_vars.values():
            var.set(True)
        assert joint(off) == pytest.approx(sum(single))
        # from another starting point the joint fit comes back to the true parameters
        v.Xc.set(8e-3), v.Gc.set(1.0)
        result = v.fit_parameters()
        assert result is not None and result.success and result.fun < 1e-3 * result.initial_fun
        assert v.Gc.get() == pytest.approx(TRUE["Gc"], rel=0.05) and v.Xc.get() == pytest.approx(TRUE["Xc"], rel=0.05)
        # the sensor on the other side of the fault: xx and yy change sign, xy does not
        xy, xx = drawn(v.axs[0], "CZM").copy(), drawn(v.axs[1], "CZM").copy()
        v.y.set(-TRUE["y"])
        v.update_plot()
        np.testing.assert_allclose(drawn(v.axs[0], "CZM"), xy, atol=1e-9)
        np.testing.assert_allclose(drawn(v.axs[1], "CZM"), -xx, atol=1e-9)
        v.y.set(TRUE["y"])
        # nothing ticked: no fit
        for var in v.fit_vars.values():
            var.set(False)
        assert v.build_fit_objective() is None and v.fit_parameters() is None
        # xy and xx ticked: what is saved says so
        v.fit_vars["xy"].set(True), v.fit_vars["xx"].set(True)
        v.update_plot()
        v.save_parameters()
        saved = event["czm_parms"]
        assert set(saved) == EXPECTED_KEYS | {"channel", "sensor", "unit", "E", "nu", "C_s", "C_d", "tau_p",
                                              "fit_components", "x_base", "model"}
        assert saved["model"] == "sub-Rayleigh"
        assert saved["sensor"] == "rosette_1" and saved["channel"] == "rosette_1_sigma_xy"
        assert saved["unit"] == "MPa" and saved["strain_gauge"] == 2
        assert saved["fit_components"] == ["xy", "xx"]
        assert saved["E"] == medium["E"] and saved["C_s"] == medium["C_s"] and 1e4 < saved["tau_p"] < 1e8
    finally:
        v.on_close()
    again = CZMFitterView(app_with_strain, 0, 2)                         # reopening restores channel, boxes, parameters
    try:
        assert again.gauge_combobox.get() == "rosette_1" and again.strain_gauge.get() == 2
        assert again.fit_components() == ["xy", "xx"]
        assert again.Gc.get() == pytest.approx(saved["Gc"]) and again.Xc.get() == pytest.approx(saved["Xc"])
        assert positions(again) == pytest.approx([-1e-3, 0.0, 1e-3])
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
        v.set_event(1)                                                   # on a volts record: strain again, two panels
        assert not v.stress_mode and len(v.axs) == 2 and v.axs[0].get_ylabel() == "Exy"
        assert list(v.gauge_combobox["values"]) == [str(i) for i in range(16)] and len(v.vlines) == 3
        assert not v.fit_frame.winfo_manager()
    finally:
        v.on_close()


def test_stress_record_with_unnamed_sensors_fits_the_selected_channel(app_with_strain):
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    add_stress_record(app_with_strain, event)
    event["waveform"]["elsys"]["channels"] = [f"gauge_{i}" for i in range(len(STRESS_NAMES))]
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        assert v.stress_mode and list(v.gauge_combobox["values"]) == [str(i) for i in range(len(STRESS_NAMES))]
        v.gauge_combobox.set("2")
        v.update_plot()
        assert v.channel_var.get() == "gauge_2" and v.sensor_channels() == {"xy": 2} and v.fit_components() == ["xy"]
        assert v.sensors() == {} and v.strain_gauge.get() == 2
        v.Cf.set(TRUE["Cf"])
        assert v.build_fit_objective()([TRUE["Gc"], TRUE["Xc"]]) < 1e-9
    finally:
        v.on_close()


# ---------------------------------------------------------------- background window, model line, Auto
def test_rayleigh_speed():
    from labquake_explorer.ui.views.czm_fitter_view import rayleigh_speed
    from labquake_explorer.utils.cohesive_crack import CohesiveCrack
    c_s, c_d = 1667.6, 2723.2
    c_r = rayleigh_speed(c_s, c_d)
    assert 0.90 * c_s < c_r < 0.93 * c_s
    assert abs(CohesiveCrack.D(CohesiveCrack.alpha_s(c_r, c_s), CohesiveCrack.alpha_d(c_r, c_d))) < 1e-9
    assert 0.90 * 2760 < rayleigh_speed(2760.0, 4790.0) < 0.93 * 2760


def test_stress_traces_start_at_zero_on_the_background_window(app_with_strain):
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    add_stress_record(app_with_strain, event)                             # levels of +3, +40 and -60 MPa
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        v.gauge_combobox.set("rosette_1")
        v.Cf.set(TRUE["Cf"]), v.Xc.set(TRUE["Xc"]), v.Gc.set(TRUE["Gc"])
        v.update_plot()
        t = v._time()
        # the window runs from the record's start to the left line, drawn as a band with a dotted line at its start
        assert v.x_base is None and v.baseline_window(t, positions(v)) == pytest.approx((-2e-3, -1e-3))
        assert len(v.base_lines) == 3 and float(v.base_lines[0].get_xdata()[0]) == pytest.approx(-2e-3)
        inside = (t >= -2e-3) & (t <= -1e-3)
        for ax, component in zip(v.axs, ("xy", "xx", "yy")):
            measured = drawn(ax, f"sigma_{component}")
            assert abs(np.mean(measured[inside])) < 1e-3                  # MPa: every trace starts at 0
            assert np.abs(measured).max() < 50                            # the 40 / 60 MPa levels are gone
        # the model: black, solid, above the data
        model = next(l for l in v.axs[0].get_lines() if l.get_label() == "CZM")
        data = next(l for l in v.axs[0].get_lines() if l.get_label() == "sigma_xy")
        assert model.get_color() == "k" and model.get_linestyle() == "-" and model.get_zorder() > data.get_zorder()
        # moving the start of the window, by hand and with the mouse
        v.move_baseline(-1.5e-3)
        assert v.x_base == pytest.approx(-1.5e-3) and v.baseline_window(t, positions(v)) == pytest.approx((-1.5e-3, -1e-3))
        dispatch(v, "button_press_event", -1.5e-3)
        assert v.drag_base and not v.drag_active
        dispatch(v, "motion_notify_event", -1.8e-3)
        dispatch(v, "button_release_event", -1.8e-3)
        assert v.x_base == pytest.approx(-1.8e-3, abs=2e-5) and not v.drag_base
        assert positions(v) == pytest.approx([-1e-3, 0.0, 1e-3])          # the three lines did not move
        v.save_parameters()
        assert event["czm_parms"]["x_base"] == pytest.approx(-1.8e-3, abs=2e-5)
    finally:
        v.on_close()
    again = CZMFitterView(app_with_strain, 0, 2)
    try:
        assert again.x_base == pytest.approx(-1.8e-3, abs=2e-5)           # restored with the event
    finally:
        again.on_close()


def test_single_noisy_samples_do_not_shift_the_curves(app_with_strain):
    """Levels come from windows, not single samples: noise on the record moves
    the drawn traces by far less than its own size."""
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    add_stress_record(app_with_strain, event)
    record = event["waveform"]["elsys"]
    clean = record["data"].copy()
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        v.gauge_combobox.set("rosette_1")
        v.Cf.set(TRUE["Cf"]), v.Xc.set(TRUE["Xc"]), v.Gc.set(TRUE["Gc"])
        v.update_plot()
        reference = drawn(v.axs[0], "CZM").copy()
        record["data"] = clean + np.random.default_rng(1).normal(0.0, 0.05, clean.shape)     # 0.05 MPa of noise
        v.update_plot()
        assert np.abs(drawn(v.axs[0], "CZM") - reference).max() < 0.02    # the model moved by a fraction of it
    finally:
        v.on_close()


def moving_front_record(app, event, speed=200.0, saved_speed=None):
    """A stress record of three shear gauges 100 mm apart that the same front
    reaches one after the other at ``speed`` (the middle one at the event time)."""
    from labquake_explorer.data.channels import channel_array, positions_table
    from labquake_explorer.ui.views.czm_fitter_view import experiment_medium
    from labquake_explorer.utils.cohesive_crack import CohesiveCrack
    app.data_manager.data["analysis"] = {"medium": dict(MEDIUM)}
    m = experiment_medium(app.data_manager.data)
    t = np.arange(-3000, 3001) / 1e6
    names, xs = ["shear_1_sigma_xy", "shear_2_sigma_xy", "shear_3_sigma_xy"], [100.0, 200.0, 300.0]
    rows = []
    for x in xs:
        delay = (x - 200.0) * 1e-3 / speed
        _, sxy, _ = CohesiveCrack.delta_sigmas(-(t - delay) * TRUE["Cf"], TRUE["y"], TRUE["Xc"], TRUE["Cf"],
                                               m["C_s"], m["C_d"], m["nu"], TRUE["Gc"], m["E"])
        rows.append(sxy + 2.0e6)
    record = channel_array(np.vstack(rows) / 1e6, names, unit="MPa", dtype=np.float64,
                           positions=positions_table(names, {n: (x, 1.0, np.nan) for n, x in zip(names, xs)}),
                           time=float(event["event_time"]) + t, sample_rate=1e6, block=2)
    if saved_speed is not None:
        record["arrivals"] = {"version": 1, "rupture_speed": saved_speed}
    event.pop("strain", None)
    event.pop("czm_parms", None)
    event["waveform"] = {"elsys": record}


def test_auto_gives_a_starting_point_and_fits(app_with_strain):
    from labquake_explorer.ui.views.czm_fitter_view import CF_LIMIT, rayleigh_speed
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    moving_front_record(app_with_strain, event, speed=200.0)
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        assert v.auto_button.winfo_manager() and v.Cf.get() == 10          # nothing saved: the fallback speed
        assert v.estimate_rupture_speed() == pytest.approx(200.0, rel=0.02)   # from the three sensors' fronts
        v.gauge_combobox.set("shear_2")
        result = v.auto_fit()
        assert result is not None and result.success
        assert v.Cf.get() == pytest.approx(200.0, rel=0.02)
        left, tip, end = positions(v)
        assert left < tip < end and abs(tip) < 5e-5                         # the tip line on the front at the event time
        assert 1e-5 < end - tip < 2e-4 and v.x_base is None
        lo, hi = v.axs[0].get_xlim()
        assert -3e-3 <= lo < left and end < hi <= 3e-3                      # zoomed on the lines, inside the record
        # a usable starting point: the fit lands within a factor of two of the parameters the record was made with
        assert 0.5 * TRUE["Gc"] < v.Gc.get() < 2 * TRUE["Gc"] and 0.5 * TRUE["Xc"] < v.Xc.get() < 2 * TRUE["Xc"]
        assert result.fun <= result.initial_fun
    finally:
        v.on_close()
    # the speed Pick Arrivals saved wins over the estimate
    moving_front_record(app_with_strain, event, speed=200.0, saved_speed=-350.0)
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        assert v.estimate_rupture_speed() == pytest.approx(350.0)
        v.auto_fit()
        assert v.Cf.get() == pytest.approx(350.0)
    finally:
        v.on_close()
    # a front faster than any wave: Cf is kept just below C_d, where the supershear model still exists
    moving_front_record(app_with_strain, event, speed=5000.0)
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        result = v.auto_fit()
        assert v.Cf.get() == pytest.approx(CF_LIMIT * v.C_d) and result is not None
        assert "supershear model" in v.result_var.get() and "above C_d" in v.result_var.get()
        v.Cf.set(0.5 * (rayleigh_speed(v.C_s, v.C_d) + v.C_s))               # between C_R and C_s: said on the plot, no fit
        v.update_plot()
        assert any("no model" in text.get_text() for text in v.axs[0].texts) and v.fit_parameters() is None
    finally:
        v.on_close()


def test_model_speed_keeps_to_where_a_steady_solution_exists(app_with_strain):
    from labquake_explorer.ui.views.czm_fitter_view import CF_LIMIT, rayleigh_speed
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    add_stress_record(app_with_strain, event)
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        c_r, c_s, c_d = rayleigh_speed(v.C_s, v.C_d), v.C_s, v.C_d
        assert v.model_speed(0.6 * c_r) == (pytest.approx(0.6 * c_r), "")             # sub-Rayleigh: as measured
        assert v.model_speed(-1.2 * c_s) == (pytest.approx(1.2 * c_s), "")            # supershear: as measured
        speed, note = v.model_speed(1.01 * c_r)                                       # just above C_R: back under it
        assert speed == pytest.approx(CF_LIMIT * c_r) and "sub-Rayleigh model" in note
        speed, note = v.model_speed(0.995 * c_s)                                      # just under C_s: over it
        assert speed == pytest.approx(c_s / CF_LIMIT) and "supershear model" in note
        speed, note = v.model_speed(2 * c_d)
        assert speed == pytest.approx(CF_LIMIT * c_d) and "above C_d" in note
    finally:
        v.on_close()


FAST = {"Xc": 0.1, "Gc": 20.0}      # a cohesive zone a supershear front takes 50 us to cross: resolved at 2 MHz


def supershear_record(app, event, speed):
    """A stress record of three shear gauges 100 mm apart crossed by a
    supershear front (the supershear cohesive-zone model itself, ``FAST``
    for Xc and Gc), the middle gauge at the event time."""
    from labquake_explorer.data.channels import channel_array, positions_table
    from labquake_explorer.ui.views.czm_fitter_view import experiment_medium
    from labquake_explorer.utils.cohesive_crack import SupershearCohesiveCrack
    app.data_manager.data["analysis"] = {"medium": dict(MEDIUM)}
    m = experiment_medium(app.data_manager.data)
    t = np.arange(-3000, 3001) / 2e6
    names, xs = ["shear_1_sigma_xy", "shear_2_sigma_xy", "shear_3_sigma_xy"], [100.0, 200.0, 300.0]
    rows = []
    for x in xs:
        delay = (x - 200.0) * 1e-3 / speed
        _, sxy, _ = SupershearCohesiveCrack.delta_sigmas(-(t - delay) * speed, TRUE["y"], FAST["Xc"], speed,
                                                         m["C_s"], m["C_d"], m["nu"], FAST["Gc"], m["E"])
        rows.append(sxy + 2.0e6)
    event.pop("strain", None)
    event.pop("czm_parms", None)
    event["waveform"] = {"elsys": channel_array(
        np.vstack(rows) / 1e6, names, unit="MPa", dtype=np.float64, time=float(event["event_time"]) + t,
        sample_rate=2e6, block=2, positions=positions_table(names, {n: (x, 1.0, np.nan) for n, x in zip(names, xs)}))}
    return m


def test_supershear_front_is_fitted_with_the_supershear_model(app_with_strain):
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    medium = supershear_record(app_with_strain, event, speed=2000.0)       # 1.2 C_s
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        assert "supershear model between C_s and C_d" in v.medium_var.get()
        v.gauge_combobox.set("shear_2")
        result = v.auto_fit()
        assert result is not None and result.success
        assert v.Cf.get() == pytest.approx(2000.0, rel=0.03) and medium["C_s"] < v.Cf.get() < medium["C_d"]
        assert "supershear model" in v.result_var.get() and "C_s" in v.result_var.get()
        # the fit lands near the parameters the record was made with
        assert 0.5 * FAST["Gc"] < v.Gc.get() < 2 * FAST["Gc"] and 0.5 * FAST["Xc"] < v.Xc.get() < 2 * FAST["Xc"]
        # with the true parameters and the tip on the front, the model lies on the record inside the fit window
        v.Cf.set(2000.0), v.Gc.set(FAST["Gc"]), v.Xc.set(FAST["Xc"])
        left, tip, end = positions(v)
        v.set_lines([left, 0.0, end])
        v.update_plot()
        measured, model = drawn(v.axs[0], "sigma_xy"), drawn(v.axs[0], "CZM")
        inside = (v._time() >= 0.0) & (v._time() <= end)
        np.testing.assert_allclose(measured[inside], model, atol=1e-6 * np.ptp(measured))
        v.save_parameters()
        assert event["czm_parms"]["model"] == "supershear" and 1e4 < event["czm_parms"]["tau_p"] < 1e8
    finally:
        v.on_close()


def test_auto_is_for_stress_records(app_with_strain, view):
    assert not view.auto_button.winfo_manager() and view.auto_fit() is None
    assert view.base_lines == [] and view.x_base is None


def test_lines_of_a_stress_record_say_what_they_do(app_with_strain):
    from labquake_explorer.ui.views.czm_fitter_view import LINE_ROLES, STRESS_HINT
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    add_stress_record(app_with_strain, event)
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        labels = {text.get_text().strip(): text for text in v.axs[0].texts}
        assert {"background start", "background end", "tip", "fit end"} <= set(labels)
        assert [float(labels[name].get_position()[0]) for name in ("background start", "background end", "tip", "fit end")] \
            == pytest.approx([-2e-3, -1e-3, 0.0, 1e-3])
        # each line in the colour of its name: grey for the background, red for the tip (solid) and fit end (dashed)
        assert [(l.get_color(), l.get_linestyle()) for l in v.vlines] == [("0.35", "--"), ("crimson", "-"), ("crimson", "--")]
        assert [l.get_color() for l in v.vlines_twin] == [l.get_color() for l in v.vlines]
        assert labels["tip"].get_color() == "crimson" and labels["background end"].get_color() == "0.35"
        assert v.hint_var.get() == STRESS_HINT and "fit window" in v.hint_var.get()
        # the names follow the positions, not the order the lines are held in: drag the first line past the others
        v.move_line(0, 1.5e-3)
        assert [name for name, _ in v.line_styles(positions(v))] == ["fit end", "background end", "tip"]
        assert v.vlines[0].get_linestyle() == "--" and v.vlines[0].get_color() == "crimson"
        assert [role for role, _ in LINE_ROLES] == ["background end", "tip", "fit end"]
        v.move_line(0, -1e-3)
        # the line under the parameters gives the misfit over the fit window
        v.Cf.set(TRUE["Cf"]), v.Xc.set(TRUE["Xc"]), v.Gc.set(TRUE["Gc"])
        v.update_plot()
        assert "fit window 1000 us, xy" in v.result_var.get() and "misfit (rms) 0.000 MPa" in v.result_var.get()
        v.Gc.set(4 * TRUE["Gc"])
        v.update_plot()
        assert "misfit (rms) 0.000 MPa" not in v.result_var.get() and "tau_p" in v.result_var.get()
        v.set_event(1)                                                   # a record in volts: plain lines, no hint
        assert all(l.get_color() == "g" and l.get_linestyle() == "--" for l in v.vlines)
        assert v.hint_var.get() == "" and v.result_var.get() == "" and len(v.axs[0].texts) == 0
    finally:
        v.on_close()


def test_auto_leaves_a_sensor_without_a_clear_front_alone(app_with_strain):
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    moving_front_record(app_with_strain, event, speed=200.0)
    record = event["waveform"]["elsys"]
    noise = np.random.default_rng(3).normal(0.0, 0.02, record["data"].shape[1])
    record["data"][2] = 2.0 + noise                                       # shear_3: connected to nothing
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        v.gauge_combobox.set("shear_3")
        v.update_plot()
        before = (positions(v), v.Cf.get(), v.Gc.get(), v.Xc.get(), v.axs[0].get_xlim())
        assert v.auto_fit() is None
        assert (positions(v), v.Cf.get(), v.Gc.get(), v.Xc.get(), v.axs[0].get_xlim()) == before
        assert "no clear front on shear_3" in v.result_var.get() and "nothing changed" in v.result_var.get()
        v.gauge_combobox.set("shear_2")                                   # a sensor that does see the front
        assert v.auto_fit() is not None and "Cf = " in v.result_var.get() and "misfit (rms)" in v.result_var.get()
    finally:
        v.on_close()


def test_result_line_says_when_the_model_explains_nothing(app_with_strain):
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    add_stress_record(app_with_strain, event)
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        v.gauge_combobox.set("rosette_1")
        v.Cf.set(TRUE["Cf"]), v.Xc.set(TRUE["Xc"]), v.Gc.set(TRUE["Gc"])
        v.update_plot()
        assert "no better than no model" not in v.result_var.get()
        v.Gc.set(1e-9)                                                    # a model of no size
        v.update_plot()
        assert "no better than no model" in v.result_var.get()
        for var in v.fit_vars.values():
            var.set(False)
        v.update_plot()
        assert v.result_var.get() == "no component ticked to fit"
    finally:
        v.on_close()


def test_auto_finds_the_side_of_the_fault_for_a_rosette(app_with_strain):
    """sigma_xx and sigma_yy change sign with the side: a rosette on the other
    side is fitted with a negative y, which Auto finds by trying both."""
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    add_stress_record(app_with_strain, event)
    record = event["waveform"]["elsys"]
    for name in ("rosette_1_sigma_xx", "rosette_1_sigma_yy"):             # the same rosette, seen from the other side
        i = STRESS_NAMES.index(name)
        record["data"][i] = 2 * record["data"][i][0] - record["data"][i]
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        v.gauge_combobox.set("rosette_1")
        v.Cf.set(TRUE["Cf"])
        v.update_plot()
        assert v.fit_components() == ["xy", "xx", "yy"]
        result = v.auto_fit()
        assert result is not None and v.y.get() == pytest.approx(-TRUE["y"])
        assert "no better than no model" not in v.result_var.get()
        assert 0.5 * TRUE["Gc"] < v.Gc.get() < 2 * TRUE["Gc"]
    finally:
        v.on_close()


def test_components_flagged_not_valid_are_left_out(app_with_strain):
    """The record says which stress components are real (``valid``, passed on by
    the conversion): the others are drawn grey and never fitted."""
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    add_stress_record(app_with_strain, event)
    record = event["waveform"]["elsys"]
    record["valid"] = [name not in ("rosette_1_sigma_xx", "rosette_1_sigma_yy") for name in STRESS_NAMES]
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        v.gauge_combobox.set("rosette_1")
        v.Cf.set(TRUE["Cf"]), v.Xc.set(TRUE["Xc"]), v.Gc.set(TRUE["Gc"])
        v.update_plot()
        assert v.channel_var.get() == "xy, xx (not valid), yy (not valid)"
        assert all(var.get() for var in v.fit_vars.values()) and v.fit_components() == ["xy"]    # ticked, yet left out
        assert v.fit_boxes["xy"].instate(["!disabled"]) and v.fit_boxes["xx"].instate(["disabled"])
        for ax, component in zip(v.axs[1:], ("xx", "yy")):
            labels = [l.get_label() for l in ax.get_lines()]
            assert f"sigma_{component} (not valid)" in labels and "CZM" not in labels
            assert any("flagged not valid" in text.get_text() for text in ax.texts)
        assert "CZM" in [l.get_label() for l in v.axs[0].get_lines()]
        assert "fit window 1000 us, xy:" in v.result_var.get() and "misfit (rms) 0.000 MPa" in v.result_var.get()
        v.save_parameters()
        assert event["czm_parms"]["fit_components"] == ["xy"]
        # a sensor with nothing valid: said so, Auto and Fit do nothing
        record["valid"] = [not name.startswith("rosette_1") for name in STRESS_NAMES]
        v.update_plot()
        assert v.fit_components() == [] and "no valid component to fit" in v.result_var.get()
        assert v.auto_fit() is None and "flagged not valid" in v.result_var.get() and v.fit_parameters() is None
        v.gauge_combobox.set("shear_1")                                    # the shear gauge is untouched by the flags
        v.update_plot()
        assert v.fit_components() == ["xy"] and v.channel_var.get() == "xy"
    finally:
        v.on_close()


def test_auto_takes_the_drop_not_the_rise_before_it(app_with_strain):
    """sigma_xy may climb before it drops (a rosette near the front): the tip
    goes on the top of that climb and the fit window on the drop."""
    event = app_with_strain.data_manager.get_data("runs/[0]/events/[2]")
    add_stress_record(app_with_strain, event)
    record = event["waveform"]["elsys"]
    t = np.asarray(record["time"]) - event["event_time"]
    bump = 0.6 * np.exp(-((t + 1.0e-4) / 4e-5) ** 2)                      # a sharp 0.6 MPa climb 0.1 ms before the tip
    record["data"][STRESS_NAMES.index("shear_1_sigma_xy")] += bump
    v = CZMFitterView(app_with_strain, 0, 2)
    try:
        v.Cf.set(TRUE["Cf"])
        found = v.front(STRESS_NAMES.index("shear_1_sigma_xy"))
        assert found["change"] < 0 and found["t_front"] > found["t_peak"] > -2e-4   # the drop, after the top of the climb
        assert v.auto_fit() is not None
        left, tip, end = positions(v)
        assert -2e-4 < tip <= 5e-5 and tip < end
    finally:
        v.on_close()

