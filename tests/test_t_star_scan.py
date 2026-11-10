"""The shared-t* grid search and the refit of saved records."""
from dataclasses import replace

import numpy as np
import pytest

from labquake_explorer.analysis import spectrum as sp
from labquake_explorer.analysis import t_star_scan as ts

FS, N, T0 = 1e6, 8000, 4e-3


def attenuated_record(fc: float, t_star_s: float, seed: int = 0) -> dict:
    """A saved-record-like dict whose binned spectrum is a Brune pulse attenuated by ``t*``."""
    t, x = sp.brune_pulse(FS, N, 1.0, fc, T0)
    s = sp.compute_spectrum(sp.extract_window(t, x, T0, 2e-3, 2e-3), fmin=500.0, fmax=100e3)
    loss = np.exp(-np.pi * s.f_binned * t_star_s)              # the medium's attenuation, no t* recorded
    s = replace(s, amp_binned=s.amp_binned * loss, noise_binned=s.noise_binned * loss)
    return {"fmin_hz": 1000.0, "fmax_hz": 50e3, "snr_min": None, "n_fixed": True, "loss": "linear", "n_max": 8.0,
            "spectrum": s.as_dict(), "t_star_s": None, "result_set": "default",
            "source": {"valid": True, "constants": {"phase": "P", "rho_kg_m3": 2700.0, "vp_m_s": 6000.0, "vs_m_s": 3500.0,
                                                    "distance_m": 0.1, "k": 0.32, "radiation_coefficient": 0.52,
                                                    "free_surface_factor": 1.0, "moment_spectrum": False}}}


def test_corrected_and_record_round_trip():
    rec = attenuated_record(12e3, 20e-6)
    s = ts.spectrum_from_record(rec)
    assert s.n_bins > 8 and s.f.size == 0 and np.isfinite(s.amp_binned).all()
    c = ts.corrected(s, 20e-6)
    np.testing.assert_allclose(c.amp_binned, s.amp_binned * np.exp(np.pi * s.f_binned * 20e-6))
    np.testing.assert_allclose(c.snr, s.snr, equal_nan=True)
    assert c.t_star_s == pytest.approx(20e-6) and c.meta["t_star_s"] == pytest.approx(20e-6)
    assert ts.corrected(s, 0.0) is s
    inp = ts.record_input(rec, label="x")
    assert inp.fmin_hz == 1000.0 and inp.fmax_hz == 50e3 and inp.n_fixed and inp.n_bounds == (1.0, 8.0)
    assert inp.fit(0.0).valid
    with pytest.raises(ValueError):
        ts.spectrum_from_record({"spectrum": {"f_binned": [], "amp_binned": []}})
    with pytest.raises(ValueError):
        ts.spectrum_from_record({})


def test_scan_recovers_a_shared_t_star():
    recs = [ts.record_input(attenuated_record(fc, 20e-6)) for fc in (8e3, 12e3, 16e3)]
    scan = ts.scan_t_star(recs, np.arange(-20e-6, 60.1e-6, 5e-6))
    assert scan.best_t_star_s == pytest.approx(20e-6) and not scan.at_grid_end and scan.n_records == 3
    assert scan.cost[scan.best_index] == pytest.approx(0.0, abs=1e-3) and np.all(scan.n_valid == 3)
    d = scan.as_dict()
    assert d["version"] == ts.RESULT_VERSION and d["best_t_star_s"] == pytest.approx(20e-6) and d["at_grid_end"] is False
    # a grid that stops short reports the minimum at its end
    short = ts.scan_t_star(recs, np.arange(-20e-6, 10.1e-6, 5e-6))
    assert short.at_grid_end and short.best_t_star_s == pytest.approx(10e-6)
    # the cost ignores records that do not fit at a grid point, and prefers the points with the fullest coverage
    bad = ts.RecordInput(ts.spectrum_from_record(attenuated_record(12e3, 0.0)), fmin_hz=1000.0, fmax_hz=1100.0, snr_min=None)
    mixed = ts.scan_t_star(recs + [bad], np.arange(-20e-6, 60.1e-6, 5e-6))
    assert mixed.best_t_star_s == pytest.approx(20e-6) and np.all(mixed.n_valid == 3)
    with pytest.raises(ValueError):
        ts.scan_t_star(recs, [])
    assert ts.cost_of(bad.fit(0.0)) != ts.cost_of(bad.fit(0.0))        # NaN for an invalid fit
