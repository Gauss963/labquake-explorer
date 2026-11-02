"""The uncertainty ensemble: a fit grid over bands and calibration tables."""
import json

import numpy as np
import pytest

from labquake_explorer.analysis import spectrum as sp
from labquake_explorer.analysis import spectrum_ensemble as se

FS, N, T0 = 1e6, 4000, 2e-3


def spectrum(fc=20e3, calibration=None):
    t, x = sp.brune_pulse(FS, N, 1.0, fc, T0)
    return sp.compute_spectrum(sp.extract_window(t, x, T0, 1e-3, 1e-3), calibration=calibration, fmin=500.0, fmax=100e3)


def test_frequency_lists_and_band_grid():
    assert se.parse_frequency_list("1000, 1500 2000;3000") == (1000.0, 1500.0, 2000.0, 3000.0)
    assert se.parse_frequency_list("  ", se.DEFAULT_FMINS_HZ) == se.DEFAULT_FMINS_HZ
    assert se.parse_frequency_list("2000, 1000, 2000") == (1000.0, 2000.0)
    for bad in ("1000, abc", "0", "-5", "nan"):
        with pytest.raises(ValueError):
            se.parse_frequency_list(bad)
    grid = se.band_grid((1000.0, 20000.0), (15000.0, 50000.0))
    assert grid == [(1000.0, 15000.0), (1000.0, 50000.0), (20000.0, 50000.0)]   # fmin < fmax only
    assert se.band_grid(se.DEFAULT_FMINS_HZ, se.DEFAULT_FMAXS_HZ) and len(se.band_grid(se.DEFAULT_FMINS_HZ, se.DEFAULT_FMAXS_HZ)) == 16


def test_stats():
    s = se.stats([1.0, 2.0, 3.0, np.nan, 10.0], valid=[True, True, True, True, False])
    assert s["n"] == 3 and s["min"] == 1.0 and s["max"] == 3.0 and s["median"] == 2.0
    assert s["p16"] == pytest.approx(np.percentile([1, 2, 3], 16)) and s["p84"] == pytest.approx(np.percentile([1, 2, 3], 84))
    empty = se.stats([])
    assert empty["n"] == 0 and all(np.isnan(empty[k]) for k in se.STAT_KEYS)


def test_fit_ensemble_over_bands_and_tables():
    cal_a = sp.Calibration.from_arrays([100.0, 1e5], [2.0, 2.0], "V/m")
    cal_b = sp.Calibration.from_arrays([100.0, 1e5], [4.0, 4.0], "V/m")
    spectra = [("a", spectrum(calibration=cal_a)), ("b", spectrum(calibration=cal_b))]
    bands = se.band_grid((1000.0, 2000.0), (50e3, 100e3))
    ens = se.fit_ensemble(spectra, bands, snr_min=None, n_fixed=True)
    assert ens.n_members == 8 and ens.n_valid == 8 and ens.calibrations == ("a", "b")
    assert ens.bands_hz == tuple(bands)
    assert np.all(ens.calibration[:4] == 0) and np.all(ens.calibration[4:] == 1)
    assert np.allclose(ens.fc, 20e3, rtol=0.05) and np.allclose(ens.n, 2.0)
    assert np.allclose(ens.omega0[:4], 0.5, rtol=0.05) and np.allclose(ens.omega0[4:], 0.25, rtol=0.05)
    st = ens.stats_of("omega0")
    assert st["min"] == pytest.approx(0.25, rel=0.05) and st["max"] == pytest.approx(0.5, rel=0.05) and st["n"] == 8
    assert ens.moment is None and ens.stats_of("seismic_moment_nm")["n"] == 0
    # the moment follows the plateau by the record's own M0 / Omega0
    with_m0 = ens.with_moment(3.0)
    assert np.allclose(with_m0.moment, 3.0 * ens.omega0)
    d = with_m0.as_dict()
    assert d["version"] == se.RESULT_VERSION and d["n_members"] == 8 and d["n_valid"] == 8
    assert set(d["members"]) == {"fmin_hz", "fmax_hz", "calibration", "omega0", "fc", "n", "valid", "band_limited",
                                 "seismic_moment_nm"}
    assert set(d["stats"]) == {"omega0", "fc", "n", "seismic_moment_nm"}
    assert d["stats"]["seismic_moment_nm"]["median"] == pytest.approx(3.0 * np.median(ens.omega0))
    json.dumps(d)
    assert not np.isfinite(ens.with_moment(float("nan")).moment).any()
    # band-limited members count as valid but not resolved; the resolved statistics leave them out
    pinned = se.fit_ensemble(spectra[:1], [(1000.0, 50e3), (15e3, 100e3)], snr_min=None)   # fc = 20 kHz: fmin 15 kHz pins it
    assert pinned.n_valid == 2 and pinned.n_resolved == 1 and list(pinned.band_limited) == [False, True]
    assert pinned.stats_of("fc")["n"] == 2 and pinned.stats_of("fc", resolved=True)["n"] == 1
    assert pinned.stats_of("fc", resolved=True)["median"] == pytest.approx(pinned.fc[0])
    pd = pinned.as_dict()
    assert pd["n_resolved"] == 1 and pd["stats_resolved"]["fc"]["n"] == 1 and pd["stats"]["fc"]["n"] == 2
    # a band without enough bins gives an invalid member, left out of the statistics
    sparse = se.fit_ensemble(spectra[:1], [(1000.0, 1100.0), (1000.0, 50e3)], snr_min=None)
    assert sparse.n_members == 2 and sparse.n_valid == 1 and not sparse.valid[0] and np.isnan(sparse.fc[0])
    assert sparse.stats_of("fc")["n"] == 1
