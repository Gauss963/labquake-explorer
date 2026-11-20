"""Cohesive crack analytical calculations module."""
import numpy as np
import matplotlib.pyplot as plt

class CohesiveCrack:
    """Analytical calculations for cohesive crack models."""
    
    default_params = {
        'Gamma': 0.21,   # Fracture energy (J/m^2)
        'E': 51e9,       # Young's modulus (Pa)
        'nu': 0.25,      # Poisson's ratio
        'C_f': 2404,     # Rupture speed (m/s)
        'C_s': 2760,     # Shear wave speed (m/s)
        'C_d': 4790,     # Longitudinal wave speed (m/s)
        'X_c': 13.8e-3   # Cohesive zone size (m)
    }

    @classmethod
    def delta_sigmas(cls, x, y, X_c, C_f, C_s, C_d, nu, Gamma, E):
        """Calculate stress fluctuations with explicit parameters."""
        alpha_s_value = cls.alpha_s(C_f, C_s)
        alpha_d_value = cls.alpha_d(C_f, C_d)
        D_value = cls.D(alpha_s_value, alpha_d_value)
        A2 = cls.compute_A2(C_f, C_s, nu, D_value)
        K2 = cls.compute_K2(Gamma, E, nu, A2)
        tau_p = cls.compute_tau_p(K2, X_c)
        
        z_d_value = x + 1j * alpha_d_value * y
        z_s_value = x + 1j * alpha_s_value * y
        
        M_z_d = cls._M_of_z(tau_p, X_c, z_d_value)
        M_z_s = cls._M_of_z(tau_p, X_c, z_s_value)
        
        Sxx_tmp, Syy_tmp, Sxy_tmp = cls._compute_stress_components(M_z_d, M_z_s, alpha_s_value, alpha_d_value)
        Sxx, Syy, Sxy = cls._compute_stresses(Sxx_tmp, Syy_tmp, Sxy_tmp, alpha_s_value, D_value)
        
        return Sxx, Sxy, Syy

    @staticmethod
    def alpha_s(C_f, C_s):
        return np.sqrt(1 - (C_f / C_s) ** 2)

    @staticmethod
    def alpha_d(C_f, C_d):
        return np.sqrt(1 - (C_f / C_d) ** 2)

    @staticmethod
    def D(alpha_s, alpha_d):
        return 4 * alpha_s * alpha_d - (1 + alpha_s ** 2) ** 2

    @staticmethod
    def compute_A2(C_f, C_s, nu, D_value):
        alpha_s_value = CohesiveCrack.alpha_s(C_f, C_s)
        psfactor = 1 / (1 - nu)
        return (C_f ** 2 * alpha_s_value * psfactor) / (C_s ** 2 * D_value)

    @staticmethod
    def compute_K2(Gamma, E, nu, A2):
        return np.sqrt((Gamma * E) / ((1 - nu ** 2) * A2))

    @staticmethod
    def compute_tau_p(K2, X_c):
        return K2 * np.sqrt(9 * np.pi / (32 * X_c))

    @classmethod
    def plot_stress_fluctuation(cls, output_path='./Plot/example_xx.pdf'):
        """Plot stress fluctuation with configurable parameters."""
        y_values = [1e-8, 0.1e-3, 0.5e-3, 1.0e-3, 2.0e-3, 5e-3, 10e-3, 15e-3]
        x = np.linspace(-50e-3, 50e-3, 8192)

        plt.figure(figsize=(8, 6))
        for i, y in enumerate(y_values):
            Sxy, _ = cls.delta_sigmas(x, y, 
                                    cls.default_params['X_c'],
                                    cls.default_params['C_f'],
                                    cls.default_params['C_s'],
                                    cls.default_params['C_d'],
                                    cls.default_params['nu'],
                                    cls.default_params['Gamma'],
                                    cls.default_params['E'])
            plt.plot(x * 1000, Sxy / 1e5 + i * 5, '-.', label=f'y = {y * 1e3:.1f} mm')

        plt.xlabel('Rupture tip position x (mm)')
        plt.ylabel('Stress fluctuation (MPa)')
        plt.title('Stress fluctuation along the fault')
        plt.axvline(0, color='k', linestyle='--', linewidth=1)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(output_path, dpi=900)
        plt.close()

    @staticmethod
    def _M_of_z(tau_p, X_c, z):
        return (2 / np.pi) * tau_p * ((1 + z / X_c) * np.arctan(1 / np.sqrt(z / X_c)) - np.sqrt(z / X_c))

    @staticmethod
    def _compute_stress_components(M_z_d, M_z_s, alpha_s_value, alpha_d_value):
        Sxx_tmp = (1 + 2 * alpha_d_value ** 2 - alpha_s_value ** 2) * M_z_d - (1 + alpha_s_value ** 2) * M_z_s
        Syy_tmp = M_z_d - M_z_s
        Sxy_tmp = 4 * alpha_s_value * alpha_d_value * M_z_d - (1 + alpha_s_value ** 2) ** 2 * M_z_s
        return Sxx_tmp, Syy_tmp, Sxy_tmp

    @staticmethod
    def _compute_stresses(Sxx_tmp, Syy_tmp, Sxy_tmp, alpha_s_value, D_value):
        Sxx = 2 * alpha_s_value / D_value * Sxx_tmp.imag
        Syy = -2 * alpha_s_value * (1 + alpha_s_value ** 2) / D_value * Syy_tmp.imag
        Sxy = Sxy_tmp.real / D_value
        return Sxx, Syy, Sxy

def rayleigh_speed(C_s: float, C_d: float) -> float:
    """The Rayleigh wave speed: the speed below ``C_s`` at which
    ``D = 4 alpha_s alpha_d - (1 + alpha_s^2)^2`` vanishes."""
    from scipy import optimize

    def d(c):
        a_s, a_d = np.sqrt(1 - (c / C_s) ** 2), np.sqrt(1 - (c / C_d) ** 2)
        return 4 * a_s * a_d - (1 + a_s ** 2) ** 2
    return float(optimize.brentq(d, 0.3 * C_s, C_s * (1 - 1e-9)))


class SupershearCohesiveCrack:
    """Steady mode II rupture faster than the shear waves (``C_s < C_f < C_d``)
    with the same cohesive zone as :class:`CohesiveCrack`: the shear stress
    falls linearly from ``tau_p`` at the tip to the residual level over ``X_c``.

    In the frame moving with the tip (``xi = x - C_f t``, ``y > 0``) the
    displacement potentials are ``phi = Re F(xi + i a_d y)`` with
    ``a_d = sqrt(1 - C_f^2/C_d^2)`` and ``psi = g(xi + b y)`` with
    ``b = sqrt(C_f^2/C_s^2 - 1)``: the shear part is a wave trailing the tip
    (a Mach front), not a field around it.  With ``mu`` the shear modulus::

        s_xx =  mu [ (1 + 2 a_d^2 + b^2) Re F'' + 2 b g'' ]
        s_yy = -mu [ (1 - b^2) Re F'' + 2 b g'' ]
        s_xy =  mu [ -2 a_d Im F'' + (b^2 - 1) g'' ]

    ``s_yy = 0`` on the fault gives ``g''(s) = -(1 - b^2)/(2 b) Re F''(s)``
    behind the tip and nothing ahead of it.  On the crack
    ``-2 a_d Im F'' + B Re F'' = tau(xi)/mu`` with ``B = (1 - b^2)^2/(2 b)``.
    With ``tan(pi q) = 2 a_d / B`` and ``R = sqrt(4 a_d^2 + B^2)`` the solution
    bounded at the tip is ``F''(z) = -i z^(1-q) I(z) tau_p / (pi R mu)``,

        I(z) = Integral_0^Xc (1 - u/Xc) u^(q-1) / (u + z) du
             = (1 + z/Xc) Xc^q / (q z) 2F1(1, q; q+1; -Xc/z) - Xc^(q-1) / q .

    Far ahead the stress decays as ``r^-q`` (``q <= 1/2``; ``q = 1/2`` at
    ``sqrt(2) C_s``, where the shear wave vanishes).  The fracture energy is the
    work of the cohesive stress on the slip, ``Gamma = energy_factor *
    tau_p^2 X_c / mu``; at ``sqrt(2) C_s`` the factor is ``8 / (9 pi a_d)``.
    """

    @staticmethod
    def constants(C_f, C_s, C_d):
        """``(a_d, b, B, q, R)`` of the class docstring."""
        C_f = abs(float(C_f))
        if not C_s < C_f < C_d:
            raise ValueError(f"C_f = {C_f:g} m/s is not between C_s = {C_s:g} and C_d = {C_d:g} m/s")
        a_d = np.sqrt(1 - (C_f / C_d) ** 2)
        b = np.sqrt((C_f / C_s) ** 2 - 1)
        B = (1 - b ** 2) ** 2 / (2 * b)
        return a_d, b, B, np.arctan2(2 * a_d, B) / np.pi, np.hypot(2 * a_d, B)

    @staticmethod
    def _second_derivative(z, X_c, q, R):
        """``F''(z)`` for ``tau_p / mu = 1``, ``z`` in the upper half plane."""
        from scipy.special import hyp2f1
        z = np.asarray(z, dtype=complex)
        integral = (1 + z / X_c) * X_c ** q / (q * z) * hyp2f1(1, q, q + 1, -X_c / z) - X_c ** (q - 1) / q
        return -1j * z ** (1 - q) * integral / (np.pi * R)

    @classmethod
    def energy_factor(cls, C_f, C_s, C_d) -> float:
        """``Gamma mu / (tau_p^2 X_c)``: the work of the cohesive stress on the
        slip it meets, whose gradient on the fault is ``(1 + b^2) Re F''``."""
        key = (round(abs(float(C_f)), 6), round(float(C_s), 6), round(float(C_d), 6))
        if key not in _ENERGY_FACTORS:
            a_d, b, B, q, R = cls.constants(*key)
            nodes, weights = np.polynomial.legendre.leggauss(400)
            xi = -(nodes + 1) / 2                             # the cohesive zone, X_c = 1
            slip_gradient = (1 + b ** 2) * cls._second_derivative(xi + 1e-10j, 1.0, q, R).real
            _ENERGY_FACTORS[key] = float(np.sum(weights / 2 * (1 + xi) * -slip_gradient))
        return _ENERGY_FACTORS[key]

    @classmethod
    def compute_tau_p(cls, Gamma, X_c, C_f, C_s, C_d, mu):
        """Peak strength above the residual level for a fracture energy and a cohesive zone size."""
        return np.sqrt(Gamma * mu / (X_c * cls.energy_factor(C_f, C_s, C_d)))

    @classmethod
    def delta_sigmas(cls, x, y, X_c, C_f, C_s, C_d, nu, Gamma, E):
        """Stress changes ``(Sxx, Sxy, Syy)`` at ``x`` ahead of the tip and ``y``
        off the fault, with the arguments of :meth:`CohesiveCrack.delta_sigmas`.
        ``Sxx`` and ``Syy`` change sign with the side of the fault (``y < 0``)."""
        a_d, b, B, q, R = cls.constants(C_f, C_s, C_d)
        mu = E / (2 * (1 + nu))
        tau_p = cls.compute_tau_p(Gamma, X_c, C_f, C_s, C_d, mu)
        x = np.asarray(x, dtype=float)
        side = -1.0 if y < 0 else 1.0
        height = max(abs(float(y)), 1e-10 * X_c)
        f = cls._second_derivative(x + 1j * a_d * height, X_c, q, R)
        s = x + b * height                                    # behind the Mach front where s < 0
        behind = s < 0
        g = np.zeros_like(x)
        if np.any(behind):
            on_fault = cls._second_derivative(s[behind] + 1e-10j * X_c, X_c, q, R).real
            g[behind] = -(1 - b ** 2) / (2 * b) * on_fault
        Sxx = tau_p * ((1 + 2 * a_d ** 2 + b ** 2) * f.real + 2 * b * g) * side
        Syy = -tau_p * ((1 - b ** 2) * f.real + 2 * b * g) * side
        Sxy = tau_p * (-2 * a_d * f.imag + (b ** 2 - 1) * g)
        return Sxx, Sxy, Syy


_ENERGY_FACTORS: dict = {}


def rupture_regime(C_f, C_s, C_d):
    """Which steady cohesive-zone solution exists at a rupture speed:
    ``'sub-Rayleigh'`` below the Rayleigh speed, ``'supershear'`` between
    ``C_s`` and ``C_d``, None in between and above ``C_d``."""
    C_f = abs(float(C_f))
    if 0 < C_f < rayleigh_speed(C_s, C_d):
        return "sub-Rayleigh"
    if C_s < C_f < C_d:
        return "supershear"
    return None


def cohesive_delta_sigmas(x, y, X_c, C_f, C_s, C_d, nu, Gamma, E):
    """``(Sxx, Sxy, Syy)`` of the cohesive-zone solution that exists at ``C_f``
    (:func:`rupture_regime`): :class:`CohesiveCrack` or
    :class:`SupershearCohesiveCrack`; NaN where there is none."""
    regime = rupture_regime(C_f, C_s, C_d)
    if regime == "sub-Rayleigh":
        return CohesiveCrack.delta_sigmas(x, y, X_c, abs(float(C_f)), C_s, C_d, nu, Gamma, E)
    if regime == "supershear":
        return SupershearCohesiveCrack.delta_sigmas(x, y, X_c, C_f, C_s, C_d, nu, Gamma, E)
    nothing = np.full(np.shape(x), np.nan)
    return nothing, nothing.copy(), nothing.copy()


def cohesive_tau_p(Gamma, X_c, C_f, C_s, C_d, nu, E) -> float:
    """Peak strength above the residual level (Pa) that a fracture energy and
    a cohesive zone size imply at ``C_f``; NaN where no solution exists."""
    regime = rupture_regime(C_f, C_s, C_d)
    with np.errstate(all="ignore"):
        if regime == "sub-Rayleigh":
            C_f = abs(float(C_f))
            alpha_s, alpha_d = CohesiveCrack.alpha_s(C_f, C_s), CohesiveCrack.alpha_d(C_f, C_d)
            A2 = CohesiveCrack.compute_A2(C_f, C_s, nu, CohesiveCrack.D(alpha_s, alpha_d))
            return float(CohesiveCrack.compute_tau_p(CohesiveCrack.compute_K2(Gamma, E, nu, A2), X_c))
        if regime == "supershear":
            return float(SupershearCohesiveCrack.compute_tau_p(Gamma, X_c, C_f, C_s, C_d, E / (2 * (1 + nu))))
    return float("nan")


def cohesive_gamma(tau_p, X_c, C_f, C_s, C_d, nu, E) -> float:
    """The fracture energy (J/m^2) whose peak strength is ``tau_p`` (Pa) at a
    cohesive zone size and ``C_f``: the inverse of :func:`cohesive_tau_p`."""
    unit = cohesive_tau_p(1.0, X_c, C_f, C_s, C_d, nu, E)      # tau_p grows as sqrt(Gamma)
    return float((tau_p / unit) ** 2) if np.isfinite(unit) and unit > 0 else float("nan")


def main():
    """Example usage of CohesiveCrack"""
    crack = CohesiveCrack()
    crack.plot_stress_fluctuation()

if __name__ == "__main__":
    main()