"""Seismic source parameters from a fitted displacement spectrum.

Inputs are the spectral plateau ``omega0`` (displacement spectral density,
m*s) and corner frequency ``fc`` (Hz) of one phase recorded at one sensor,
plus the medium (density, P and S wave speeds) and the source-receiver
geometry.  Everything is SI: metres, seconds, kg/m^3, Pa, N m.  No material
constants are built in as defaults: ``rho``, ``vp`` and ``vs`` are required
arguments so that the values used are always visible to (and chosen by) the
caller.

Formulas
--------
* Seismic moment (far-field, Brune/Aki & Richards):
  ``M0 = 4 pi rho c^3 R omega0 / (F * S)`` with ``c`` the wave speed of the
  PHASE that was fitted (Vp for P, Vs for S), ``R`` the hypocentral distance,
  ``F`` the radiation coefficient and ``S`` the free-surface factor.
* Source radius (circular crack): ``r = k Vs / fc`` with ``k`` depending on
  the phase and model -- 0.32 for P (Madariaga 1976), 0.372 for S (Brune
  1970).  The radius ALWAYS uses Vs regardless of phase.
* Eshelby stress drop: ``delta sigma = 7 M0 / (16 r^3)``.
* Moment magnitude: ``Mw = (2/3) (log10 M0 - 9.1)``.
* Far-field check: ``kR = 2 pi f R / c``; the far-field approximation is
  reasonable for ``kR >= 3``.

Radiation coefficients for a double couple with fault normal ``n``, slip
direction ``s`` and take-off (ray) direction ``g`` (all unit vectors):
``R_P = 2 (n.g)(s.g)`` (signed) and ``R_S = |(n.g) s + (s.g) n - 2 (n.g)(s.g) g|``.
``|R_P| < 0.2`` means the sensor is near a P node and the moment estimate
is poorly constrained; ``source_parameters`` warns about it.  RMS values
over the focal sphere (``PHASE_CONSTANTS[phase]['radiation_rms']``) are
sqrt(4/15) = 0.516 for P and sqrt(2/5) = 0.632 for S.

The sign of ``R_P`` encodes first-motion polarity only.  Spectral amplitudes
are positive, so ``omega0`` must be positive (a ``ValueError`` is raised,
never an ``abs()``), and the magnitude of the radiation coefficient is used
in the moment.

Error policy
------------
The elementary relations (``seismic_moment`` etc.) validate strictly and
raise ``ValueError``.  :func:`source_parameters`, the entry point a view
calls once per event, raises only for USER entries (material constants,
distance, phase, ``k``) and never for values that come out of a fit or a
geometry computation: a degenerate plateau/corner, a zero radiation
coefficient or an unusable free-surface factor give ``valid=False`` with a
``reason``; an unusable plateau frequency only skips the far-field check.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

RESULT_VERSION = 1

PHASES = ("P", "S")

#: Per-phase constants.  ``radiation_rms`` is the RMS of the double-couple
#: radiation pattern over the focal sphere; ``k`` relates source radius to
#: corner frequency via ``r = k Vs / fc``.
PHASE_CONSTANTS = {
    "P": {"radiation_rms": math.sqrt(4.0 / 15.0), "k": 0.32},    # Madariaga (1976)
    "S": {"radiation_rms": math.sqrt(2.0 / 5.0), "k": 0.372},    # Brune (1970)
}

#: |R_P| below this is treated as "near a P node".
NODE_THRESHOLD = 0.2
#: kR below this violates the far-field assumption.
FAR_FIELD_KR = 3.0
#: Warning thresholds for implausible laboratory results.
STRESS_DROP_WARN_PA = 1.0e9
MW_WARN = -1.0

#: Warning text used when the plateau frequency cannot be used for kR.
PLATEAU_UNAVAILABLE_WARNING = "plateau frequency unavailable: far-field check skipped"


# ----------------------------------------------------------------------------
# scalar helpers
# ----------------------------------------------------------------------------

def _finite(name: str, value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number, got {value!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _positive(name: str, value) -> float:
    v = _finite(name, value)
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v!r}")
    return v


def _as_float(value) -> float:
    """``float(value)``, or NaN when the value is not a number.  Never raises."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _phase(phase: str) -> str:
    p = str(phase).upper()
    if p not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    return p


def _unit(name: str, vec) -> np.ndarray:
    v = np.asarray(vec, dtype=float).ravel()
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError(f"{name} must have non-zero length")
    return v / norm


# ----------------------------------------------------------------------------
# elementary relations
# ----------------------------------------------------------------------------

def seismic_moment(omega0_ms: float, rho: float, c: float, distance_m: float,
                   radiation_coefficient: float, free_surface_factor: float = 1.0) -> float:
    """``M0 = 4 pi rho c^3 R omega0 / (|F| S)`` in N m.

    ``c`` must be the wave speed of the phase whose spectrum gave ``omega0``.
    ``omega0_ms`` must be > 0; a non-positive plateau is an upstream error
    (a signed fit) and raises rather than being silently rectified.  The sign
    of ``radiation_coefficient`` (polarity) is ignored; zero raises.
    """
    omega0 = _positive("omega0_ms", omega0_ms)
    rho = _positive("rho", rho)
    c = _positive("c", c)
    distance = _positive("distance_m", distance_m)
    f = abs(_finite("radiation_coefficient", radiation_coefficient))
    if f == 0.0:
        raise ValueError("radiation_coefficient must be non-zero (sensor at a node)")
    s = _positive("free_surface_factor", free_surface_factor)
    return 4.0 * math.pi * rho * c ** 3 * distance * omega0 / (f * s)


def plateau_from_moment(m0_nm: float, rho: float, c: float, distance_m: float,
                        radiation_coefficient: float, free_surface_factor: float = 1.0) -> float:
    """Inverse of :func:`seismic_moment`: the displacement plateau (m*s) for a given M0."""
    m0 = _positive("m0_nm", m0_nm)
    rho = _positive("rho", rho)
    c = _positive("c", c)
    distance = _positive("distance_m", distance_m)
    f = abs(_finite("radiation_coefficient", radiation_coefficient))
    if f == 0.0:
        raise ValueError("radiation_coefficient must be non-zero (sensor at a node)")
    s = _positive("free_surface_factor", free_surface_factor)
    return m0 * f * s / (4.0 * math.pi * rho * c ** 3 * distance)


def moment_magnitude(m0_nm: float) -> float:
    """``Mw = (2/3) (log10 M0 - 9.1)`` (Hanks & Kanamori 1979, IASPEI form)."""
    m0 = _positive("m0_nm", m0_nm)
    return (2.0 / 3.0) * (math.log10(m0) - 9.1)


def moment_from_magnitude(mw: float) -> float:
    """Inverse of :func:`moment_magnitude`."""
    return 10.0 ** (1.5 * _finite("mw", mw) + 9.1)


def source_radius(fc_hz: float, vs: float, k: float) -> float:
    """Circular-crack radius ``r = k Vs / fc`` (m).  Always uses the shear speed."""
    return _positive("k", k) * _positive("vs", vs) / _positive("fc_hz", fc_hz)


def stress_drop_eshelby(m0_nm: float, radius_m: float) -> float:
    """Eshelby (1957) static stress drop of a circular crack: ``7 M0 / (16 r^3)`` (Pa)."""
    return 7.0 * _positive("m0_nm", m0_nm) / (16.0 * _positive("radius_m", radius_m) ** 3)


def wavenumber_distance(f_hz: float, distance_m: float, c: float) -> float:
    """``kR = 2 pi f R / c``: distance in wavelengths (times 2 pi) at frequency ``f``."""
    return 2.0 * math.pi * _positive("f_hz", f_hz) * _positive("distance_m", distance_m) / _positive("c", c)


def far_field_ok(kR: float) -> bool:
    """True when ``kR >= FAR_FIELD_KR`` (far-field approximation acceptable)."""
    return bool(np.isfinite(kR) and kR >= FAR_FIELD_KR)


# ----------------------------------------------------------------------------
# radiation pattern, free surface, geometry
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class RadiationPattern:
    """Double-couple radiation coefficients for one take-off direction."""
    p: float               # signed: 2 (n.g)(s.g)
    s: float               # magnitude: |(n.g) s + (s.g) n - 2 (n.g)(s.g) g|
    p_near_node: bool      # |p| < NODE_THRESHOLD

    def as_dict(self) -> dict:
        return asdict(self)


def radiation_pattern(fault_normal, slip_direction, ray_direction) -> RadiationPattern:
    """P and S radiation coefficients of a double couple.

    All three arguments are 3-vectors (normalised here).  ``ray_direction``
    points from the source towards the sensor.  The P coefficient is signed
    (polarity); the S coefficient is the magnitude of the vector S pattern.
    """
    n = _unit("fault_normal", fault_normal)
    s = _unit("slip_direction", slip_direction)
    g = _unit("ray_direction", ray_direction)
    ng = float(n @ g)
    sg = float(s @ g)
    r_p = 2.0 * ng * sg
    r_s = float(np.linalg.norm(ng * s + sg * n - 2.0 * ng * sg * g))
    return RadiationPattern(float(r_p), r_s, abs(r_p) < NODE_THRESHOLD)


def radiation_coefficient(phase: str, fault_normal, slip_direction, ray_direction) -> float:
    """Radiation coefficient for ``phase``: signed ``R_P`` or ``R_S`` magnitude.

    ``|R_P| < 0.2`` means the sensor lies near a P node; the moment derived
    from such a sensor is very sensitive to the assumed geometry.
    """
    pat = radiation_pattern(fault_normal, slip_direction, ray_direction)
    return pat.p if _phase(phase) == "P" else pat.s


def free_surface_amplification(incidence_deg, vp: float, vs: float):
    """Vertical-component free-surface response to an incident plane P wave.

    Aki & Richards (2002, eq. 5.30 family): with ray parameter
    ``p = sin(i)/vp``, ``eta_a = cos(i)/vp``, ``eta_b = sqrt(1/vs^2 - p^2)``,
    ``D = (1/vs^2 - 2 p^2)^2 + 4 p^2 eta_a eta_b``, the vertical displacement
    at the surface per unit incident P amplitude is
    ``u_z = 2 vp eta_a (1/vs^2 - 2 p^2) / (vs^2 D)``.

    Equals 2.0 at normal incidence, ~1.0 at 60 degrees for a Poisson solid,
    and 0.0 at grazing incidence (90 degrees).  ``incidence_deg`` may be a
    scalar or array in [0, 90]; a scalar input gives a float.  NaN entries
    (``Geometry.incidence_deg`` when no sensor normal was given) propagate to
    NaN instead of raising; finite angles outside [0, 90] and infinities
    raise ``ValueError``.  ``vs >= vp`` raises.

    The factor is non-negative for every angle when ``vp/vs >= sqrt(2)``
    (``1/vs^2 - 2 p^2 >= 2 cos^2(i)/vp^2``); for smaller, unphysical ratios
    it becomes NEGATIVE at steep incidence.  So the return value can be 0
    (grazing), negative (bad vp/vs) or NaN (unknown incidence), none of which
    can divide a moment.  :func:`source_parameters` therefore returns
    ``valid=False`` (it never raises) for such a factor.  View guidance: pass
    ``free_surface_factor=1.0`` when the sensor is not on a free surface or
    the incidence is unknown, and treat incidence >= ~85 degrees as unusable
    (the response collapses towards 0 there and the moment blows up).
    """
    vp = _positive("vp", vp)
    vs = _positive("vs", vs)
    if vs >= vp:
        raise ValueError("vs must be smaller than vp")
    inc = np.asarray(incidence_deg, dtype=float)
    known = ~np.isnan(inc)
    if np.any(np.isinf(inc)) or np.any(inc[known] < 0.0) or np.any(inc[known] > 90.0):
        raise ValueError("incidence_deg must be within [0, 90] (NaN propagates)")
    i = np.radians(inc)
    p = np.sin(i) / vp
    eta_a = np.cos(i) / vp
    eta_b = np.sqrt(np.maximum(1.0 / vs ** 2 - p ** 2, 0.0))
    d = (1.0 / vs ** 2 - 2.0 * p ** 2) ** 2 + 4.0 * p ** 2 * eta_a * eta_b
    u_z = 2.0 * vp * eta_a * (1.0 / vs ** 2 - 2.0 * p ** 2) / (vs ** 2 * d)
    u_z = np.where(inc >= 90.0, 0.0, u_z)       # NaN >= 90 is False: NaN stays NaN
    return float(u_z) if u_z.ndim == 0 else u_z


@dataclass(frozen=True)
class Geometry:
    """Source-receiver geometry in SI units."""
    distance_m: float
    ray: tuple[float, float, float]     # unit vector from source to sensor
    incidence_deg: float                # arccos(|n_sensor . ray|); NaN without a normal

    def as_dict(self) -> dict:
        return {"distance_m": self.distance_m, "ray": list(self.ray),
                "incidence_deg": self.incidence_deg}


def geometry(sensor_xyz_m, source_xyz_m, sensor_normal=None) -> Geometry:
    """Distance, take-off ray (source -> sensor) and incidence angle at the sensor.

    Coordinates are in METRES (the view converts from cm).  The incidence
    angle is measured from the sensor's own normal, so a sensor whose normal
    points at the source has incidence 0.  Without ``sensor_normal`` the
    incidence is NaN (and :func:`free_surface_amplification` then returns
    NaN: use a free-surface factor of 1.0 in that case).  Coincident points
    raise.
    """
    sensor = np.asarray(sensor_xyz_m, dtype=float).ravel()
    source = np.asarray(source_xyz_m, dtype=float).ravel()
    if sensor.shape != (3,) or source.shape != (3,):
        raise ValueError("sensor_xyz_m and source_xyz_m must be 3-vectors")
    if not (np.all(np.isfinite(sensor)) and np.all(np.isfinite(source))):
        raise ValueError("coordinates must be finite")
    d = sensor - source
    distance = float(np.linalg.norm(d))
    if distance == 0.0:
        raise ValueError("sensor and source coincide (zero distance)")
    ray = d / distance
    if sensor_normal is None:
        incidence = float("nan")
    else:
        n = _unit("sensor_normal", sensor_normal)
        incidence = float(np.degrees(np.arccos(np.clip(abs(float(n @ ray)), 0.0, 1.0))))
    return Geometry(distance, tuple(float(v) for v in ray), incidence)


# ----------------------------------------------------------------------------
# combined source-parameter estimate
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceParameters:
    """Source parameters of one event from one phase at one sensor (SI)."""
    phase: str
    omega0_ms: float
    fc_hz: float
    seismic_moment_nm: float
    mw: float
    source_radius_m: float
    rupture_area_m2: float
    stress_drop_pa: float
    kR: float                         # NaN when no usable plateau frequency given
    far_field_ok: Optional[bool]      # None when kR is NaN
    constants: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    valid: bool = True
    reason: str = ""
    version: int = RESULT_VERSION

    def as_dict(self) -> dict:
        d = asdict(self)
        d["constants"] = dict(self.constants)
        d["warnings"] = list(self.warnings)
        return d


def _invalid(phase: str, omega0: float, fc: float, constants: dict, warnings: list,
             reason: str) -> SourceParameters:
    nan = float("nan")
    return SourceParameters(phase, omega0, fc, nan, nan, nan, nan, nan, nan, None,
                            constants, list(warnings), False, reason)


def source_parameters(omega0_ms: float, fc_hz: float, phase: str, rho: float, vp: float,
                      vs: float, distance_m: float, radiation_coefficient: Optional[float] = None,
                      free_surface_factor: float = 1.0, k: Optional[float] = None,
                      f_plateau_hz: Optional[float] = None, moment_spectrum: bool = False) -> SourceParameters:
    """Seismic moment, magnitude, radius, area and stress drop from one spectral fit.

    ``moment_spectrum``: the plateau is already a seismic moment (a spectrum
    in N*m*s from an in-situ response in V/(N m), the empirical Green's
    function route): ``M0 = Omega0``; ``vs`` and the phase's ``k`` give the
    radius, the stress drop follows as usual.  ``rho``, ``vp``, ``distance_m``,
    the radiation coefficient and the free-surface factor are not used (they
    may be None, and are None in ``constants``), ``kR`` is NaN with
    ``far_field_ok`` None, and ``constants['moment_spectrum']`` is True.

    Material constants and geometry (``rho``, ``vp``, ``vs``, ``distance_m``,
    ``k``) are required and validated (``ValueError`` on non-positive values
    or an unknown ``phase``): they are user entries the view should check
    before calling.  Everything that comes out of a FIT or a GEOMETRY
    computation never raises, so a per-event loop cannot crash on one bad
    event:

    * non-finite or non-positive ``omega0`` or ``fc``, or a zero radiation
      coefficient -> ``valid=False`` with a ``reason`` (all derived fields
      NaN, ``far_field_ok=None``);
    * a ``free_surface_factor`` that is <= 0 or not a finite number (grazing
      incidence, ``vp/vs < sqrt(2)``, or NaN from an unknown incidence) ->
      ``valid=False`` with a ``reason``;
    * a ``f_plateau_hz`` that is non-finite or non-positive (e.g. the NaN
      band edge of a failed spectral fit) -> treated as not given: ``kR`` NaN,
      ``far_field_ok=None``, ``constants['f_plateau_hz']=None`` and the
      warning ``PLATEAU_UNAVAILABLE_WARNING``.

    Warnings that depend only on the inputs (``vp/vs < sqrt(2)``, near-node
    radiation, unusable plateau frequency) are reported on invalid results
    too, so the cause of an invalid free-surface factor is visible.

    ``radiation_coefficient=None`` uses the focal-sphere RMS value for the
    phase; ``k=None`` uses the phase default (0.32 P, 0.372 S).  The moment
    uses Vp for P and Vs for S; the radius always uses Vs.  ``f_plateau_hz``
    (typically the low edge of the fitted band) enables the ``kR`` far-field
    check.  ``warnings`` lists soft problems: near-node radiation, kR < 3,
    stress drop > 1 GPa, Mw > -1, vp/vs < sqrt(2), plateau frequency unusable.

    View guidance: pass ``free_surface_factor=1.0`` unless the sensor sits on
    a free surface AND its incidence angle is known and well below grazing
    (>= ~85 degrees is unusable); ``constants['free_surface_factor']`` echoes
    the value actually received (NaN when it was not a number).
    """
    phase = _phase(phase)
    consts = PHASE_CONSTANTS[phase]
    moment_spectrum = bool(moment_spectrum)
    if moment_spectrum:                     # the plateau is the moment: only the radius needs the medium
        vs = _positive("vs", vs)
        rho = vp = distance = c = rad = None
    else:
        rho = _positive("rho", rho)
        vp = _positive("vp", vp)
        vs = _positive("vs", vs)
        distance = _positive("distance_m", distance_m)
        if radiation_coefficient is None:
            rad = float(consts["radiation_rms"])
        else:
            rad = _finite("radiation_coefficient", radiation_coefficient)
        c = vp if phase == "P" else vs
    k_used = consts["k"] if k is None else _positive("k", k)

    # derived (never-raise) inputs
    fsf = float("nan") if moment_spectrum else _as_float(free_surface_factor)
    f_pl = None if (f_plateau_hz is None or moment_spectrum) else _as_float(f_plateau_hz)
    plateau_usable = f_pl is not None and math.isfinite(f_pl) and f_pl > 0.0

    constants = {
        "phase": phase, "rho_kg_m3": rho, "vp_m_s": vp, "vs_m_s": vs,
        "wave_speed_m_s": c, "distance_m": distance, "k": k_used,
        "radiation_coefficient": rad, "radiation_rms": float(consts["radiation_rms"]),
        "free_surface_factor": None if moment_spectrum else fsf,
        "f_plateau_hz": f_pl if plateau_usable else None,
        "moment_spectrum": moment_spectrum,
    }

    # warnings that depend on the inputs only (reported for invalid results too)
    warnings: list[str] = []
    if not moment_spectrum:
        if vp / vs < math.sqrt(2.0):
            warnings.append(f"vp/vs = {vp / vs:.3f} < sqrt(2): unphysical for an isotropic solid")
        if abs(rad) < NODE_THRESHOLD:
            warnings.append(f"|radiation coefficient| = {abs(rad):.3f} < {NODE_THRESHOLD}: sensor near a node")
        if f_plateau_hz is not None and not plateau_usable:
            warnings.append(PLATEAU_UNAVAILABLE_WARNING)

    try:
        omega0 = float(omega0_ms)
        fc = float(fc_hz)
    except (TypeError, ValueError):
        return _invalid(phase, float("nan"), float("nan"), constants, warnings,
                        "omega0 and fc must be numbers")
    if not (math.isfinite(omega0) and math.isfinite(fc)):
        return _invalid(phase, omega0, fc, constants, warnings, "omega0 or fc is not finite")
    if omega0 <= 0.0:
        return _invalid(phase, omega0, fc, constants, warnings, "omega0 must be positive (signed fit?)")
    if fc <= 0.0:
        return _invalid(phase, omega0, fc, constants, warnings, "fc must be positive")
    if moment_spectrum:
        m0 = omega0
    else:
        if rad == 0.0:
            return _invalid(phase, omega0, fc, constants, warnings,
                            "radiation coefficient is zero (sensor at a node)")
        if not math.isfinite(fsf):
            return _invalid(phase, omega0, fc, constants, warnings,
                            "free-surface factor is not a finite number "
                            "(unknown incidence? use 1.0 when the sensor is not on a free surface)")
        if fsf <= 0.0:
            return _invalid(phase, omega0, fc, constants, warnings,
                            "free-surface factor <= 0 (grazing incidence or vp/vs < sqrt(2))")
        m0 = seismic_moment(omega0, rho, c, distance, rad, fsf)
    mw = moment_magnitude(m0)
    r = source_radius(fc, vs, k_used)
    area = math.pi * r ** 2
    dsigma = stress_drop_eshelby(m0, r)

    kR = float("nan")
    ff_ok: Optional[bool] = None
    if plateau_usable:
        kR = wavenumber_distance(f_pl, distance, c)
        ff_ok = far_field_ok(kR)

    if ff_ok is False:
        warnings.append(f"kR = {kR:.2f} < {FAR_FIELD_KR:g}: far-field approximation doubtful")
    if dsigma > STRESS_DROP_WARN_PA:
        warnings.append(f"stress drop {dsigma / 1e6:.3g} MPa exceeds 1 GPa")
    if mw > MW_WARN:
        warnings.append(f"Mw = {mw:.2f} > {MW_WARN:g}: implausibly large for a laboratory event")

    return SourceParameters(phase, omega0, fc, m0, mw, r, area, dsigma, kR, ff_ok,
                            constants, warnings, True, "")
