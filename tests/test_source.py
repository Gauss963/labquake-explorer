import json
import math

import numpy as np
import pytest

from labquake_explorer.analysis.source import (
    PHASE_CONSTANTS, PLATEAU_UNAVAILABLE_WARNING, SourceParameters, far_field_ok,
    moment_magnitude, seismic_moment, source_parameters, source_radius, stress_drop_eshelby,
    wavenumber_distance,
)

# a "PMMA-like" medium, chosen by the TEST (the module has no defaults)
RHO, VP, VS = 1180.0, 2700.0, 1350.0
NAN = float("nan")


def test_scalar_relations():
    assert PHASE_CONSTANTS["P"]["radiation_rms"] == pytest.approx(math.sqrt(4 / 15))
    assert PHASE_CONSTANTS["S"]["radiation_rms"] == pytest.approx(math.sqrt(2 / 5))
    assert PHASE_CONSTANTS["P"]["k"] == 0.32 and PHASE_CONSTANTS["S"]["k"] == 0.372
    # M0 = 1 N m, r = 1 cm -> 7 / (16 * 1e-6) Pa = 437.5 kPa
    assert stress_drop_eshelby(1.0, 0.01) == pytest.approx(437.5e3)
    assert moment_magnitude(10 ** 9.1) == pytest.approx(0.0, abs=1e-12)
    assert moment_magnitude(10 ** 10.6) == pytest.approx(1.0)
    omega0 = 2.5e-9
    m0 = seismic_moment(omega0, RHO, VP, 0.05, 0.8, 1.5)
    assert m0 == pytest.approx(4 * math.pi * RHO * VP ** 3 * 0.05 * omega0 / (0.8 * 1.5))
    # polarity sign of the radiation coefficient does not change the moment
    assert seismic_moment(omega0, RHO, VP, 0.05, -0.8) == seismic_moment(omega0, RHO, VP, 0.05, 0.8)
    assert source_radius(50e3, VS, 0.32) == pytest.approx(0.32 * VS / 50e3)
    assert wavenumber_distance(10e3, 0.05, VP) == pytest.approx(2 * math.pi * 10e3 * 0.05 / VP)
    assert far_field_ok(3.0) and not far_field_ok(2.99) and not far_field_ok(NAN)


def test_bad_inputs_raise():
    r = lambda fn, *args, **kw: pytest.raises(ValueError, fn, *args, **kw)
    r(moment_magnitude, 0.0)
    for omega0 in (-2.5e-9, 0.0, NAN):                        # a bad plateau is rejected, never abs()-ed
        r(seismic_moment, omega0, RHO, VP, 0.05, 0.8)
    r(seismic_moment, 1e-9, RHO, VP, 0.05, 0.0)                # node
    r(seismic_moment, 1e-9, RHO, VP, 0.05, 0.8, 0.0)           # bad free-surface factor (strict low-level API)
    r(source_parameters, 1e-10, 40e3, "L", RHO, VP, VS, 0.05, 0.8)
    r(source_parameters, 1e-10, 40e3, "P", 0.0, VP, VS, 0.05, 0.8)
    r(source_parameters, 1e-10, 40e3, "P", RHO, VP, VS, -0.05, 0.8)
    r(source_parameters, 1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.8, k=-1.0)
    r(source_parameters, 1e-10, 40e3, "P", RHO, VP, VS, 0.05, NAN)
    r(source_parameters, 2.5, 4000.0, "P", None, None, None, None, moment_spectrum=True)   # Vs still required


def test_source_parameters_p_and_s_phases():
    omega0, fc, dist, rad = 3.0e-13, 40e3, 0.06, 0.9
    sp = source_parameters(omega0, fc, "P", RHO, VP, VS, dist, rad, free_surface_factor=1.0, f_plateau_hz=30e3)
    assert isinstance(sp, SourceParameters) and sp.valid and sp.reason == ""
    m0, r = 4 * math.pi * RHO * VP ** 3 * dist * omega0 / rad, 0.32 * VS / fc   # P: Vp for M0, Vs for the radius
    assert sp.seismic_moment_nm == pytest.approx(m0) and sp.mw == pytest.approx(moment_magnitude(m0))
    assert sp.source_radius_m == pytest.approx(r) and sp.rupture_area_m2 == pytest.approx(math.pi * r ** 2)
    assert sp.stress_drop_pa == pytest.approx(7 * m0 / (16 * r ** 3))
    assert sp.kR == pytest.approx(2 * math.pi * 30e3 * dist / VP) and sp.far_field_ok is True
    assert sp.constants["wave_speed_m_s"] == VP and sp.constants["k"] == 0.32
    assert sp.constants["radiation_coefficient"] == rad and sp.constants["f_plateau_hz"] == 30e3
    assert sp.warnings == []
    json.dumps(sp.as_dict())
    sp = source_parameters(3.0e-10, 40e3, "s", RHO, VP, VS, 0.06)      # lower-case accepted
    assert sp.phase == "S" and sp.constants["wave_speed_m_s"] == VS and sp.constants["k"] == 0.372
    assert sp.constants["radiation_coefficient"] == pytest.approx(math.sqrt(2 / 5))
    assert sp.seismic_moment_nm == pytest.approx(4 * math.pi * RHO * VS ** 3 * 0.06 * 3.0e-10 / math.sqrt(2 / 5))
    assert sp.source_radius_m == pytest.approx(0.372 * VS / 40e3)
    assert math.isnan(sp.kR) and sp.far_field_ok is None
    assert sp.constants["f_plateau_hz"] is None and sp.warnings == []
    sp = source_parameters(1e-10, 60e3, "P", RHO, VP, VS, 0.042, 1.0, 1.5, f_plateau_hz=10e3)
    assert sp.valid and sp.seismic_moment_nm == pytest.approx(seismic_moment(1e-10, RHO, VP, 0.042, 1.0, 1.5))


def test_source_parameters_warnings():
    # near node, near field, enormous moment -> stress drop > 1 GPa and Mw > -1, vp/vs too small
    sp = source_parameters(1e-3, 200e3, "P", RHO, 1500.0, 1300.0, 0.02, 0.05, f_plateau_hz=1e3)
    assert sp.valid and sp.far_field_ok is False
    joined = " | ".join(sp.warnings)
    for text in ("node", "kR", "1 GPa", "Mw", "vp/vs"):
        assert text in joined, text
    ok = source_parameters(1e-13, 50e3, "P", RHO, VP, VS, 0.05, 0.8, f_plateau_hz=40e3)   # a benign case
    assert 1e3 < ok.stress_drop_pa < 1e8 and ok.mw < -5 and ok.warnings == [] and ok.far_field_ok is True
    # an invalid result: the reason explains why, the warnings still describe the inputs
    sp = source_parameters(NAN, 40e3, "P", RHO, 1500.0, 1300.0, 0.05, 0.05)
    assert not sp.valid and "finite" in sp.reason
    assert "vp/vs" in " | ".join(sp.warnings) and "node" in " | ".join(sp.warnings)


def test_source_parameters_invalid_without_raising():
    for omega0, fc in [(-1e-10, 40e3), (0.0, 40e3), (NAN, 40e3), (1e-10, 0.0), (1e-10, float("inf")), ("x", 40e3)]:
        sp = source_parameters(omega0, fc, "P", RHO, VP, VS, 0.05, 0.8)
        assert not sp.valid and sp.reason and sp.far_field_ok is None
        assert math.isnan(sp.seismic_moment_nm) and math.isnan(sp.stress_drop_pa)
        json.dumps(sp.as_dict())
    node = source_parameters(1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.0)
    assert not node.valid and "node" in node.reason
    # grazing incidence: the free-surface response is exactly 0
    sp = source_parameters(1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.8, free_surface_factor=0.0)
    assert not sp.valid and "free-surface" in sp.reason and "grazing" in sp.reason
    assert math.isnan(sp.seismic_moment_nm) and sp.far_field_ok is None
    assert sp.constants["free_surface_factor"] == 0.0 and sp.warnings == []
    json.dumps(sp.as_dict())
    # vp/vs < sqrt(2): negative factor; invalid AND the vp/vs warning still fires
    neg = -0.3
    sp = source_parameters(1e-10, 40e3, "P", RHO, 1500.0, 1300.0, 0.05, 0.8, free_surface_factor=neg, f_plateau_hz=NAN)
    assert not sp.valid and "free-surface" in sp.reason and sp.constants["free_surface_factor"] == pytest.approx(neg)
    assert "vp/vs" in " | ".join(sp.warnings) and PLATEAU_UNAVAILABLE_WARNING in sp.warnings
    for bad in (NAN, float("inf"), "x", None, -1.0):          # unknown incidence -> NaN factor; non-numeric
        sp = source_parameters(1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.8, free_surface_factor=bad)
        assert not sp.valid and "free-surface" in sp.reason
        assert math.isnan(sp.seismic_moment_nm) and math.isnan(sp.stress_drop_pa)
        json.dumps(sp.as_dict())
    # an unknown incidence (NaN factor) must not raise; the documented fallback 1.0 is valid
    sp = source_parameters(1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.8, free_surface_factor=NAN)
    assert not sp.valid and "incidence" in sp.reason
    assert source_parameters(1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.8, free_surface_factor=1.0).valid


def test_source_parameters_unusable_plateau_frequency_skips_far_field_check():
    # the view passes band_used[0] of the spectral fit; a failed fit carries NaN band edges
    reference = source_parameters(1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.8)
    for f_pl in (NAN, float("inf"), -float("inf"), 0.0, -1.0, "x", None):
        sp = source_parameters(1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.8, f_plateau_hz=f_pl)
        assert sp.valid and sp.reason == ""
        assert math.isnan(sp.kR) and sp.far_field_ok is None and sp.constants["f_plateau_hz"] is None
        assert sp.seismic_moment_nm == pytest.approx(reference.seismic_moment_nm)
        assert sp.stress_drop_pa == pytest.approx(reference.stress_drop_pa)
        assert sp.warnings == ([] if f_pl is None else [PLATEAU_UNAVAILABLE_WARNING])
        json.dumps(sp.as_dict())
    good = source_parameters(1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.8, f_plateau_hz=np.float64(40e3))
    assert good.kR == pytest.approx(2 * math.pi * 40e3 * 0.05 / VP)
    assert good.far_field_ok is True and good.constants["f_plateau_hz"] == 40e3
    assert PLATEAU_UNAVAILABLE_WARNING not in good.warnings


def test_source_parameters_moment_spectrum_takes_m0_from_the_plateau():
    r = source_parameters(2.5, 4000.0, "P", None, None, 3500.0, None, moment_spectrum=True)
    assert r.valid and r.seismic_moment_nm == 2.5 and r.mw == pytest.approx(moment_magnitude(2.5))
    assert r.source_radius_m == pytest.approx(0.32 * 3500.0 / 4000.0)
    assert r.stress_drop_pa == pytest.approx(stress_drop_eshelby(2.5, r.source_radius_m))
    assert math.isnan(r.kR) and r.far_field_ok is None and r.warnings == []
    c = r.constants
    assert c["moment_spectrum"] is True and c["rho_kg_m3"] is None and c["vp_m_s"] is None and c["distance_m"] is None
    assert c["radiation_coefficient"] is None and c["free_surface_factor"] is None and c["f_plateau_hz"] is None
    # the S phase uses its own k; entries given anyway are ignored
    s = source_parameters(2.5, 4000.0, "S", 2700.0, 6000.0, 3500.0, 0.1, radiation_coefficient=0.0,
                          free_surface_factor=-1.0, f_plateau_hz=1500.0, moment_spectrum=True)
    assert s.valid and s.seismic_moment_nm == 2.5 and s.source_radius_m == pytest.approx(0.372 * 3500.0 / 4000.0)
    bad = source_parameters(-1.0, 4000.0, "P", None, None, 3500.0, None, moment_spectrum=True)
    assert not bad.valid and "positive" in bad.reason
    assert source_parameters(1e-12, 4000.0, "P", 2700, 6000, 3500, 0.1).constants["moment_spectrum"] is False
