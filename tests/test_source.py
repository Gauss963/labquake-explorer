import json
import math

import numpy as np
import pytest

from labquake_explorer.analysis.source import (
    PHASE_CONSTANTS, PLATEAU_UNAVAILABLE_WARNING, Geometry, RadiationPattern, SourceParameters,
    far_field_ok, free_surface_amplification, geometry, moment_from_magnitude,
    moment_magnitude, plateau_from_moment, radiation_coefficient, radiation_pattern,
    seismic_moment, source_parameters, source_radius, stress_drop_eshelby,
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
    assert moment_from_magnitude(moment_magnitude(3.7e2)) == pytest.approx(3.7e2)
    omega0 = 2.5e-9
    m0 = seismic_moment(omega0, RHO, VP, 0.05, 0.8, 1.5)
    assert m0 == pytest.approx(4 * math.pi * RHO * VP ** 3 * 0.05 * omega0 / (0.8 * 1.5))
    assert plateau_from_moment(m0, RHO, VP, 0.05, 0.8, 1.5) == pytest.approx(omega0)
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
    r(radiation_pattern, (0, 0, 0), (1, 0, 0), (0, 0, 1))
    r(radiation_pattern, (0, 0, 1), (1, 0), (0, 0, 1))
    r(radiation_coefficient, "Rayleigh", (0, 0, 1), (1, 0, 0), (0, 0, 1))
    for incidence in (95.0, -1.0, float("inf")):
        r(free_surface_amplification, incidence, VP, VS)
    r(free_surface_amplification, 10.0, VS, VP)                # vs >= vp
    r(geometry, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
    r(geometry, (1.0, 2.0), (0.0, 0.0, 0.0))
    r(source_parameters, 1e-10, 40e3, "L", RHO, VP, VS, 0.05, 0.8)
    r(source_parameters, 1e-10, 40e3, "P", 0.0, VP, VS, 0.05, 0.8)
    r(source_parameters, 1e-10, 40e3, "P", RHO, VP, VS, -0.05, 0.8)
    r(source_parameters, 1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.8, k=-1.0)
    r(source_parameters, 1e-10, 40e3, "P", RHO, VP, VS, 0.05, NAN)
    r(source_parameters, 2.5, 4000.0, "P", None, None, None, None, moment_spectrum=True)   # Vs still required


def test_radiation_pattern():
    n, s = (0, 0, 1), (1, 0, 0)
    pat = radiation_pattern(n, s, (0, 0, 1))                   # fault normal: P node, S maximum
    assert pat.p == pytest.approx(0.0) and pat.s == pytest.approx(1.0) and pat.p_near_node
    assert radiation_coefficient("P", n, s, (0, 0, 1)) == pytest.approx(0.0)
    assert radiation_coefficient("S", n, s, (0, 0, 1)) == pytest.approx(1.0)
    pat = radiation_pattern(n, s, (1, 0, 1))                   # 45 degrees, not normalised on purpose
    assert pat.p == pytest.approx(1.0) and pat.s == pytest.approx(0.0, abs=1e-12) and not pat.p_near_node
    # opposite quadrant: polarity flips, magnitude preserved
    assert radiation_pattern(n, s, (-1, 0, 1)).p == pytest.approx(-1.0)
    # sensor in the fault plane along the slip direction: n.g = 0 -> P node, S = |n| = 1
    pat = radiation_pattern(n, s, (1, 0, 0))
    assert pat.p == pytest.approx(0.0) and pat.s == pytest.approx(1.0)
    assert isinstance(RadiationPattern(1.0, 0.0, False).as_dict(), dict)
    # RMS over the focal sphere matches the constants
    g = np.random.default_rng(0).normal(size=(200000, 3))
    g /= np.linalg.norm(g, axis=1)[:, None]
    n, s = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    ng, sg = g @ n, g @ s
    rp = 2 * ng * sg
    rs = np.linalg.norm(ng[:, None] * s + sg[:, None] * n - (2 * ng * sg)[:, None] * g, axis=1)
    assert math.sqrt(np.mean(rp ** 2)) == pytest.approx(PHASE_CONSTANTS["P"]["radiation_rms"], rel=0.01)
    assert math.sqrt(np.mean(rs ** 2)) == pytest.approx(PHASE_CONSTANTS["S"]["radiation_rms"], rel=0.01)
    pat = radiation_pattern(n, s, g[0])                        # the vectorised check agrees with the function
    assert pat.p == pytest.approx(rp[0]) and pat.s == pytest.approx(rs[0])


# Poisson solid (vp/vs = sqrt 3), incidence 0..85 deg in steps of 5
FREE_SURFACE_TABLE = (2.00, 1.99, 1.96, 1.92, 1.86, 1.79, 1.70, 1.60, 1.49, 1.38,
                      1.26, 1.14, 1.02, 0.90, 0.79, 0.67, 0.54, 0.35)


def test_free_surface_amplification():
    vp, vs = math.sqrt(3.0), 1.0
    angles = np.arange(0, 90, 5)
    values = free_surface_amplification(angles, vp, vs)
    assert values.shape == angles.shape and np.all(np.abs(values - np.array(FREE_SURFACE_TABLE)) < 0.03)
    assert free_surface_amplification(0.0, vp, vs) == pytest.approx(2.0)
    assert free_surface_amplification(60.0, vp, vs) == pytest.approx(1.0, abs=0.03)
    assert free_surface_amplification(90.0, vp, vs) == 0.0
    assert isinstance(free_surface_amplification(30.0, vp, vs), float)
    # scale invariance: only vp/vs matters
    assert free_surface_amplification(40.0, VP, VP / math.sqrt(3)) == pytest.approx(
        free_surface_amplification(40.0, vp, vs))
    # a NaN incidence (geometry without a sensor normal) propagates instead of crashing a view loop
    out = free_surface_amplification(NAN, VP, VS)
    assert isinstance(out, float) and math.isnan(out)
    arr = free_surface_amplification(np.array([0.0, NAN, 90.0]), VP, VS)
    assert arr[0] == pytest.approx(2.0) and math.isnan(arr[1]) and arr[2] == 0.0
    # vp/vs = 1.154 < sqrt(2): the response changes sign at steep incidence; non-negative for a physical ratio
    assert free_surface_amplification(80.0, 1500.0, 1300.0) < 0.0
    assert np.all(free_surface_amplification(np.linspace(0, 90, 91), VP, VS) >= 0.0)


def test_geometry():
    g = geometry(sensor_xyz_m=(0.0, 0.0, 0.05), source_xyz_m=(0.0, 0.0, 0.0), sensor_normal=(0.0, 0.0, -1.0))
    assert isinstance(g, Geometry) and g.distance_m == pytest.approx(0.05)
    assert g.ray == pytest.approx((0.0, 0.0, 1.0)) and g.as_dict()["ray"] == [0.0, 0.0, 1.0]
    assert g.incidence_deg == pytest.approx(0.0)               # normal sign does not matter
    g = geometry((0.05, 0.0, 0.05), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert g.distance_m == pytest.approx(0.05 * math.sqrt(2)) and g.incidence_deg == pytest.approx(45.0)
    assert math.isnan(geometry((0.0, 0.0, 0.05), (0.0, 0.0, 0.0)).incidence_deg)


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
    omega0 = plateau_from_moment(0.5, RHO, VP, 0.04, 0.7)              # round trip through the plateau
    assert source_parameters(omega0, 30e3, "P", RHO, VP, VS, 0.04, 0.7).seismic_moment_nm == pytest.approx(0.5)
    # end to end: fault in the x-y plane at z=0, slip along x, sensor on the free surface at 45 deg
    geo = geometry((0.03, 0.0, 0.03), (0.0, 0.0, 0.0), sensor_normal=(0, 0, 1))
    rad = radiation_coefficient("P", (0, 0, 1), (1, 0, 0), geo.ray)
    fsf = free_surface_amplification(geo.incidence_deg, VP, VS)
    assert rad == pytest.approx(1.0) and geo.incidence_deg == pytest.approx(45.0)
    sp = source_parameters(1e-10, 60e3, "P", RHO, VP, VS, geo.distance_m, rad, fsf, f_plateau_hz=10e3)
    assert sp.valid and sp.seismic_moment_nm == pytest.approx(seismic_moment(1e-10, RHO, VP, geo.distance_m, rad, fsf))


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
    # grazing incidence: the analytic free-surface response is exactly 0
    sp = source_parameters(1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.8,
                           free_surface_factor=free_surface_amplification(90.0, VP, VS))
    assert not sp.valid and "free-surface" in sp.reason and "grazing" in sp.reason
    assert math.isnan(sp.seismic_moment_nm) and sp.far_field_ok is None
    assert sp.constants["free_surface_factor"] == 0.0 and sp.warnings == []
    json.dumps(sp.as_dict())
    # vp/vs < sqrt(2): negative factor; invalid AND the vp/vs warning still fires
    neg = free_surface_amplification(80.0, 1500.0, 1300.0)
    sp = source_parameters(1e-10, 40e3, "P", RHO, 1500.0, 1300.0, 0.05, 0.8, free_surface_factor=neg, f_plateau_hz=NAN)
    assert not sp.valid and "free-surface" in sp.reason and sp.constants["free_surface_factor"] == pytest.approx(neg)
    assert "vp/vs" in " | ".join(sp.warnings) and PLATEAU_UNAVAILABLE_WARNING in sp.warnings
    for bad in (NAN, float("inf"), "x", None, -1.0):          # unknown incidence -> NaN factor; non-numeric
        sp = source_parameters(1e-10, 40e3, "P", RHO, VP, VS, 0.05, 0.8, free_surface_factor=bad)
        assert not sp.valid and "free-surface" in sp.reason
        assert math.isnan(sp.seismic_moment_nm) and math.isnan(sp.stress_drop_pa)
        json.dumps(sp.as_dict())
    # the whole chain from geometry without a sensor normal must not raise; the documented fallback 1.0 is valid
    geo = geometry((0.0, 0.0, 0.05), (0.0, 0.0, 0.0))
    fsf = free_surface_amplification(geo.incidence_deg, VP, VS)
    sp = source_parameters(1e-10, 40e3, "P", RHO, VP, VS, geo.distance_m, 0.8, free_surface_factor=fsf)
    assert not sp.valid and "incidence" in sp.reason
    assert source_parameters(1e-10, 40e3, "P", RHO, VP, VS, geo.distance_m, 0.8, free_surface_factor=1.0).valid


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
