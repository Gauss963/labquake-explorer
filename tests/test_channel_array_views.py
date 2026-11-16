"""Views address rows of channel arrays ('slip/slip_3') like plain fields."""
import numpy as np
import pytest

from labquake_explorer.data.channels import channel_array, positions_table
from labquake_explorer.ui.views import EventAnalyzerView, InterEventView, RunSignalsView


@pytest.fixture
def app_with_arrays(app):
    """The synthetic experiment with its displacement moved into a 'slip' channel
    array (two sensors) and the volt-like channels into 'raw_data'."""
    run = app.data_manager.get_data("runs/[0]")
    d = run["displacement"]
    run["slip"] = channel_array(np.vstack([d, 2.0 * d]), ["slip_1", "slip_2"], unit="um",
                                positions=positions_table(["slip_1", "slip_2"], {"slip_1": (50.0, 0.0, 0.0), "slip_2": (450.0, 0.0, 0.0)}))
    run["raw_data"] = channel_array(np.vstack([run["LP_displacement"] / 100.0]), ["lvdt"], unit="V")
    for event in run["events"]:
        n = event["time"].size
        i0 = int(np.searchsorted(run["time"], event["time"][0]))
        event["slip"] = {k: (v[:, i0:i0 + n] if k == "data" else v) for k, v in run["slip"].items()}
    app.refresh_tree()
    return app


def test_views_use_channel_rows_and_fall_back_to_slip_1(app_with_arrays):
    app = app_with_arrays
    label = app.format_tree_label("slip", app.data_manager.get_data("runs/[0]/slip"))
    assert label.startswith("slip: 2 channels x ") and label.endswith(" um")

    v = EventAnalyzerView(app, 0, 1)
    fields = list(v.data_x_combo["values"])
    assert "slip/slip_1" in fields and "slip/slip_2" in fields and "displacement" in fields
    v.data_x_combo.set("slip/slip_2")
    v.data_selected()
    assert v.item_x == "slip/slip_2" and v.data_x.shape == v.data_y.shape
    np.testing.assert_array_equal(v.data_x, v.event["slip"]["data"][1])
    v.update_analysis()
    assert v.result["x_field"] == "slip/slip_2"
    # the row is read back by path when the picks are applied to every event
    assert v.apply_to_all_events(confirm=False) == 4
    assert app.data_manager.get_data("runs/[0]/events/[0]/event_analysis")["x_field"] == "slip/slip_2"
    v.on_close()

    v = RunSignalsView(app, 0)
    assert "slip/slip_1" in v.candidates and "raw_data/lvdt" in v.candidates
    v.select_signals(["slip/slip_1", "raw_data/lvdt"])
    v.plot()
    assert len(v.signal_lines) == 2
    v.on_close()

    v = InterEventView(app, 0)
    assert "slip/slip_2" in v.slip_combo["values"]
    v.slip_combo.set("slip/slip_2")
    v.compute()
    truth = app.truth[0]
    expected = 2.0 * (truth.slip + truth.creep_fraction * truth.lp_velocity * 12.0)
    assert v.result["slip_field"] == "slip/slip_2"
    assert v.result["slip_per_cycle"][1:] == pytest.approx([expected] * (len(v.result["slip_per_cycle"]) - 1), rel=1e-6)
    v.on_close()

    # files without a 'displacement' field (fault slip is slip/slip_1) still open the analyses
    run = app.data_manager.get_data("runs/[0]")
    del run["displacement"]
    for event in run["events"]:
        del event["displacement"]
        event.pop("event_analysis")
    app.refresh_tree()
    v = EventAnalyzerView(app, 0, 1)
    assert v.item_x == "slip/slip_1" and v.data_x_combo.get() == "slip/slip_1"
    v.on_close()
    v = InterEventView(app, 0)
    assert v.slip_combo.get() == "slip/slip_1"
    v.on_close()


def test_arrival_picker_without_a_slip_trace(app_with_strain):
    from labquake_explorer.ui.views.dynamic_strain_arrival_picker_view import DynamicStrainArrivalPickerView
    app = app_with_strain
    event = app.data_manager.get_data("runs/[0]/events/[1]")
    del event["displacement"]
    v = DynamicStrainArrivalPickerView(app, 0, 1)
    assert DynamicStrainArrivalPickerView.fault_slip(v.event) is None and len(v.axs[3].lines) == 0
    assert len(v.axs[2].lines) == 1
    v.on_close()
