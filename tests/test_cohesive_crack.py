"""The cohesive-zone solutions: the supershear one against its own boundary
conditions and limits, and the choice between it and the sub-Rayleigh one."""
import numpy as np
import pytest

from labquake_explorer.utils.cohesive_crack import (CohesiveCrack, SupershearCohesiveCrack, cohesive_delta_sigmas,
                                                    cohesive_gamma, cohesive_tau_p, rayleigh_speed, rupture_regime)

E, NU, RHO = 7.662e9, 0.2, 1148.0
MU = E / (2 * (1 + NU))
C_S = np.sqrt(MU / RHO)                                               # 1667.6 m/s
C_D = np.sqrt(E * (1 - NU) / (RHO * (1 + NU) * (1 - 2 * NU)))         # 2723.2 m/s
X_C, GAMMA = 0.05, 2.0
SPEEDS = [1.05 * C_S, 1.2 * C_S, np.sqrt(2) * C_S, 1.5 * C_S, 0.95 * C_D]


def on_fault(xi, speed):
    return SupershearCohesiveCrack.delta_sigmas(np.asarray(xi, dtype=float), 0.0, X_C, speed, C_S, C_D, NU, GAMMA, E)


def test_rayleigh_speed_and_regimes():
    c_r = rayleigh_speed(C_S, C_D)
    assert 0.90 * C_S < c_r < 0.93 * C_S
    assert abs(CohesiveCrack.D(CohesiveCrack.alpha_s(c_r, C_S), CohesiveCrack.alpha_d(c_r, C_D))) < 1e-9
    assert rupture_regime(0.8 * c_r, C_S, C_D) == "sub-Rayleigh" and rupture_regime(-0.8 * c_r, C_S, C_D) == "sub-Rayleigh"
    assert rupture_regime(1.2 * C_S, C_S, C_D) == "supershear" and rupture_regime(-1.2 * C_S, C_S, C_D) == "supershear"
    assert rupture_regime(0.5 * (c_r + C_S), C_S, C_D) is None and rupture_regime(1.1 * C_D, C_S, C_D) is None
    assert rupture_regime(0.0, C_S, C_D) is None


@pytest.mark.parametrize("speed", SPEEDS)
def test_supershear_boundary_conditions(speed):
    tau_p = SupershearCohesiveCrack.compute_tau_p(GAMMA, X_C, speed, C_S, C_D, MU)
    xi = np.array([-3.0, -1.5, -1.0001, -0.9, -0.5, -0.1, -1e-4]) * X_C
    sxx, sxy, syy = on_fault(xi, speed)
    # on the crack the shear stress is the cohesive law: zero behind the zone, linear inside it, tau_p at the tip
    np.testing.assert_allclose(sxy, tau_p * np.clip(1 + xi / X_C, 0, None), atol=1e-6 * tau_p)
    # the fault carries no change of normal stress, behind the tip and ahead of it
    ahead = np.array([1e-4, 0.1, 1.0, 10.0]) * X_C
    assert np.abs(syy).max() < 1e-6 * tau_p and np.abs(on_fault(ahead, speed)[2]).max() < 1e-6 * tau_p
    # the stress is continuous through the tip and bounded: no singularity left
    assert on_fault([1e-7 * X_C], speed)[1][0] == pytest.approx(tau_p, rel=5e-3)
    # far ahead it decays as r^-q with the coefficient of the solution
    a_d, b, B, q, R = SupershearCohesiveCrack.constants(speed, C_S, C_D)
    far = 500 * X_C
    assert on_fault([far], speed)[1][0] == pytest.approx(
        tau_p * 2 * a_d * X_C ** q / (np.pi * R * q * (q + 1)) * far ** -q, rel=5e-3)
    assert 0 < q <= 0.5 + 1e-12


def test_supershear_at_the_eshelby_speed():
    """At sqrt(2) C_s the shear wave vanishes: the exponent is 1/2, the energy
    factor is 8 / (9 pi a_d), and nothing trails the tip off the fault."""
    speed = np.sqrt(2) * C_S
    a_d, b, B, q, R = SupershearCohesiveCrack.constants(speed, C_S, C_D)
    assert b == pytest.approx(1.0) and B == pytest.approx(0.0, abs=1e-12) and q == pytest.approx(0.5)
    assert SupershearCohesiveCrack.energy_factor(speed, C_S, C_D) == pytest.approx(8 / (9 * np.pi * a_d), rel=1e-4)
    # the same peak strength as the sub-Rayleigh relation between the far field and tau_p: K = tau_p sqrt(32 Xc / (9 pi))
    tau_p = SupershearCohesiveCrack.compute_tau_p(GAMMA, X_C, speed, C_S, C_D, MU)
    far = 2000 * X_C
    k = on_fault([far], speed)[1][0] * np.sqrt(2 * np.pi * far)
    assert k == pytest.approx(tau_p * np.sqrt(32 * X_C / (9 * np.pi)), rel=5e-3)


def test_supershear_energy_and_peak_strength():
    factors = [SupershearCohesiveCrack.energy_factor(v, C_S, C_D) for v in np.linspace(1.03, 1.6, 12) * C_S]
    assert all(f > 0 for f in factors) and all(b > a for a, b in zip(factors, factors[1:]))   # grows with speed
    for speed in SPEEDS:
        tau_p = cohesive_tau_p(GAMMA, X_C, speed, C_S, C_D, NU, E)
        assert tau_p == pytest.approx(SupershearCohesiveCrack.compute_tau_p(GAMMA, X_C, speed, C_S, C_D, MU))
        assert cohesive_gamma(tau_p, X_C, speed, C_S, C_D, NU, E) == pytest.approx(GAMMA)
        assert cohesive_tau_p(4 * GAMMA, X_C, speed, C_S, C_D, NU, E) == pytest.approx(2 * tau_p)    # as sqrt(Gamma)
    sub = 0.7 * rayleigh_speed(C_S, C_D)
    assert cohesive_gamma(cohesive_tau_p(GAMMA, X_C, sub, C_S, C_D, NU, E), X_C, sub, C_S, C_D, NU, E) == pytest.approx(GAMMA)
    assert np.isnan(cohesive_tau_p(GAMMA, X_C, 0.98 * C_S, C_S, C_D, NU, E))


def test_supershear_off_the_fault():
    speed, y = 1.2 * C_S, 1e-3
    x = np.linspace(-4, 4, 4001) * X_C
    sxx, sxy, syy = SupershearCohesiveCrack.delta_sigmas(x, y, X_C, speed, C_S, C_D, NU, GAMMA, E)
    assert np.isfinite(sxx).all() and np.isfinite(sxy).all() and np.isfinite(syy).all()
    tau_p = SupershearCohesiveCrack.compute_tau_p(GAMMA, X_C, speed, C_S, C_D, MU)
    assert sxy.max() == pytest.approx(tau_p, rel=0.1) and abs(sxy[0]) < 0.05 * tau_p      # peak near the tip, nothing far behind
    # the other side of the fault: sigma_xx and sigma_yy change sign, sigma_xy does not
    lower = SupershearCohesiveCrack.delta_sigmas(x, -y, X_C, speed, C_S, C_D, NU, GAMMA, E)
    np.testing.assert_allclose(lower[0], -sxx)
    np.testing.assert_allclose(lower[1], sxy)
    np.testing.assert_allclose(lower[2], -syy)
    # the direction of propagation does not matter to the solution
    np.testing.assert_allclose(SupershearCohesiveCrack.delta_sigmas(x, y, X_C, -speed, C_S, C_D, NU, GAMMA, E)[1], sxy)
    with pytest.raises(ValueError, match="not between"):
        SupershearCohesiveCrack.constants(0.9 * C_S, C_S, C_D)


def test_the_model_follows_the_rupture_speed():
    x = np.linspace(-2, 2, 801) * X_C
    sub = 0.7 * rayleigh_speed(C_S, C_D)
    got = cohesive_delta_sigmas(x, 1e-3, X_C, sub, C_S, C_D, NU, GAMMA, E)
    for a, b in zip(got, CohesiveCrack.delta_sigmas(x, 1e-3, X_C, sub, C_S, C_D, NU, GAMMA, E)):
        np.testing.assert_array_equal(a, b)                              # below C_R: the sub-Rayleigh solution, untouched
    fast = 1.3 * C_S
    got = cohesive_delta_sigmas(x, 1e-3, X_C, fast, C_S, C_D, NU, GAMMA, E)
    for a, b in zip(got, SupershearCohesiveCrack.delta_sigmas(x, 1e-3, X_C, fast, C_S, C_D, NU, GAMMA, E)):
        np.testing.assert_array_equal(a, b)
    for speed in (0.99 * C_S, 1.05 * C_D):                               # no steady solution: nothing to draw
        assert all(np.isnan(part).all() and part.shape == x.shape for part in
                   cohesive_delta_sigmas(x, 1e-3, X_C, speed, C_S, C_D, NU, GAMMA, E))
