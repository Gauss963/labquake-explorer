"""Slip along the fault: profile models and their fits, and the rupture front.

Profile models, with ``x`` in mm and slip in the unit of the data (um):

* ``quadratic``: ``max(0, c0 + c1 x + c2 x^2)``.  The mask is part of the
  model, so sensors outside the ruptured stretch (slip near zero) are matched
  by the zero part instead of pulling the parabola down.
* ``eaz``: the earthquake arrest zone model of Ke, McLaskey & Kammer (2021,
  Geophys. J. Int. 224, 581), eq. 2.  With ``r = |x - xc|``::

      delta(r) = D sqrt(1 - (r / (lambda a))^2)                  0 <= r <= r_joint
      delta(r) = delta_joint ((a - r) / (a - r_joint))^(3/2)     r_joint < r <= a
      delta(r) = 0                                               a < r

  where ``r_joint = a (sqrt(1 + 3 lambda^2) - 1)`` makes the profile C1
  continuous, ``delta_joint = delta(r_joint)`` and ``0 < lambda <= 1``
  (``lambda = 1`` is the elliptical crack).  Parameters ``D`` (peak slip),
  ``xc`` (its position), ``a`` (half-length) and ``lambda``.
* ``eaz_asymmetric``: eq. 3, the same on each side of ``xc`` with its own
  half-length and ``lambda``: ``D``, ``xc``, ``a_minus``, ``a_plus``,
  ``lambda_minus``, ``lambda_plus``; the rupture tips are ``xc - a_minus`` and
  ``xc + a_plus``.

:func:`fit_profile` fits a model by least squares, unweighted by default (the
eddy sensors' noise is far below the model misfit) or weighted by
``1/sigma^2``.  The zero part of every model leaves the sensors outside the
rupture without a gradient, so the cost has many local minima, and the starts
come from a coarse search: for the quadratic, the parabola through every set of
sensors that can be the positive ones (with or without a neighbour at zero) and
a grid of vertex, half-width and sign; for the arrest-zone models, a grid of
centre, tips (at the sensors, halfway between them and beyond) and lambda with
D in closed form, and for the asymmetric model also the symmetric fit.  The
best of them are polished (the arrest-zone starts briefly first, then the best
few to the end), and the best result is continued until its cost settles; a
fit whose cost still falls by more than 1 % when that budget is used up has
not converged.  A fit that is zero at every sensor, or nowhere above a
sensor's noise, is invalid.

The sensors do not always determine every parameter.  ``unresolved`` lists the
parameters they leave free, and ``tips`` when a rupture tip is not pinned;
``ranges`` gives the range that fits equally well where it is known (NaN for
an open end):

* the Jacobian at the solution, per unit of the parameter scales, is
  singular: the parameters with a component above 0.3 in its null space, and a
  tip that moves along it;
* no more slipping sensors than parameters: the extent (``a`` and ``lambda``,
  or both of each for the asymmetric model) and the tips, each between the
  outermost slipping sensor and the next sensor out;
* EAZ with no sensor within ``r_joint`` of ``xc``: the sensors see only the
  tip part, ``A ((a - r) / a)^(3/2)`` with ``A = D Q(lambda)``, so lambda can
  take any value up to the one that puts the joint at the nearest sensor, and
  D follows at fixed ``A``; with no sensor in the tip part, only ``lambda a``
  is fixed.  The asymmetric model is checked side by side.  When no side has a
  sensor within its ``r_joint``, D is free as well; the shared D then sets the
  ranges of the two lambdas, with ``xc`` moving a little.  When the other side
  has one, D is fixed there, and lambda is unresolved only if the Jacobian says
  so (the cost then rises along it, so no range is given).

:func:`fit_front` fits the event times ``t_i`` against ``x``: one line
(``single``), or a hinge of two lines that meet at a free apex
(``bilateral``, a rupture spreading both ways from the apex).  The apparent
front velocity is ``1 / slope``; its sign gives the direction (positive
towards +x).  Given the sampling interval, a segment that the front crosses
within one sample is marked as not resolved.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

MODELS = ("quadratic", "eaz", "eaz_asymmetric")
MODEL_LABELS = {"quadratic": "Quadratic", "eaz": "EAZ", "eaz_asymmetric": "EAZ-asymmetric"}
PARAMETERS = {
    "quadratic": ("c0", "c1", "c2"),
    "eaz": ("D", "xc", "a", "lambda"),
    "eaz_asymmetric": ("D", "xc", "a_minus", "a_plus", "lambda_minus", "lambda_plus"),
}
LAMBDA_BOUNDS = (0.01, 1.0)
MAX_NFEV = 150                        # per start
MAX_NFEV_FINAL = 1000                 # to continue the best start, in chunks, until it settles
RACE_NFEV = 30                        # arrest-zone fits: every start first, and the best
N_RACED = 3                           # N_RACED go on to MAX_NFEV
N_POLISHED = {"quadratic": 10, "eaz": 8, "eaz_asymmetric": 12}      # best search points polished
SINGULAR = 1e-6                       # s_min / s_max of the scaled Jacobian below which it is singular
FRONT_MODES = ("single", "bilateral")
EXTENT = {"quadratic": (), "eaz": ("a", "lambda"),                   # the parameters that set the extent
          "eaz_asymmetric": ("a_minus", "a_plus", "lambda_minus", "lambda_plus")}
# where a parameter the user holds or bounds may go: (low, high, low included)
DOMAIN = {"D": (0.0, math.inf, True), "a": (0.0, math.inf, False), "a_minus": (0.0, math.inf, False),
          "a_plus": (0.0, math.inf, False), "lambda": (0.0, 1.0, False), "lambda_minus": (0.0, 1.0, False),
          "lambda_plus": (0.0, 1.0, False)}


# ---------------------------------------------------------------------- models
def quadratic_masked(x, c0: float, c1: float, c2: float) -> np.ndarray:
    """``max(0, c0 + c1 x + c2 x^2)``."""
    x = np.asarray(x, dtype=float)
    return np.maximum(0.0, c0 + c1 * x + c2 * x * x)


def eaz_joint(a: float, lam: float) -> float:
    """Distance from the centre where the ellipse hands over to the 3/2 tip."""
    return float(a) * (math.sqrt(1.0 + 3.0 * float(lam) ** 2) - 1.0)


def eaz_half(r, D: float, a: float, lam: float) -> np.ndarray:
    """One side of the arrest zone profile at distances ``r >= 0`` from its centre."""
    r = np.abs(np.asarray(r, dtype=float))
    out = np.zeros_like(r)
    if not (a > 0 and lam > 0):
        return out
    rj = eaz_joint(a, lam)
    ellipse = lam * a
    inner = r <= rj
    out[inner] = D * np.sqrt(np.clip(1.0 - (r[inner] / ellipse) ** 2, 0.0, None))
    if a > rj:
        tip = (r > rj) & (r < a)
        if np.any(tip):
            d_joint = D * math.sqrt(max(0.0, 1.0 - (rj / ellipse) ** 2))
            out[tip] = d_joint * ((a - r[tip]) / (a - rj)) ** 1.5
    return out


def eaz_profile(x, D: float, xc: float, a: float, lam: float) -> np.ndarray:
    """Symmetric earthquake arrest zone profile (Ke et al. 2021, eq. 2)."""
    return eaz_half(np.asarray(x, dtype=float) - xc, D, a, lam)


def eaz_asymmetric_profile(x, D: float, xc: float, a_minus: float, a_plus: float,
                           lam_minus: float, lam_plus: float) -> np.ndarray:
    """Asymmetric earthquake arrest zone profile (Ke et al. 2021, eq. 3)."""
    x = np.asarray(x, dtype=float)
    out = eaz_half(x - xc, D, a_plus, lam_plus)
    left = x < xc
    if np.any(left):
        out[left] = eaz_half(xc - x[left], D, a_minus, lam_minus)
    return out


def evaluate(model: str, params: dict, x) -> np.ndarray:
    """A model with named parameters at ``x``."""
    values = [float(params[name]) for name in PARAMETERS[model]]
    if model == "quadratic":
        return quadratic_masked(x, *values)
    if model == "eaz":
        return eaz_profile(x, *values)
    if model == "eaz_asymmetric":
        return eaz_asymmetric_profile(x, *values)
    raise ValueError(f"unknown model {model!r}; use one of {MODELS}")


def _half_derivatives(r, D: float, a: float, lam: float) -> tuple:
    """:func:`eaz_half` at distances ``r >= 0`` and its derivatives with respect to
    ``D``, ``r``, ``a`` and ``lambda``.  In the tip part the profile is
    ``D Q(lambda) ((a - r) / a)^(3/2)`` with ``Q = sqrt(1 - rho^2) / (2 - k)^(3/2)``,
    ``k = sqrt(1 + 3 lambda^2)`` and ``rho = r_joint / (lambda a) = (k - 1) / lambda``."""
    r = np.abs(np.asarray(r, dtype=float))
    f, dD, dr, da, dlam = (np.zeros_like(r) for _ in range(5))
    if not (a > 0 and lam > 0):
        return f, dD, dr, da, dlam
    k = math.sqrt(1.0 + 3.0 * lam * lam)
    rj = a * (k - 1.0)
    ellipse = lam * a
    inner = r <= rj
    ri = r[inner]
    g = np.sqrt(np.clip(1.0 - (ri / ellipse) ** 2, 1e-24, None))
    f[inner] = D * g
    dD[inner] = g
    dr[inner] = -D * ri / (ellipse ** 2 * g)
    common = D * ri * ri / (ellipse ** 3 * g)
    da[inner] = common * lam
    dlam[inner] = common * a
    tip = (r > rj) & (r < a)
    if a > rj and np.any(tip):
        rho = (k - 1.0) / lam
        q = math.sqrt(max(0.0, 1.0 - rho * rho)) / (2.0 - k) ** 1.5
        rt = r[tip]
        shape = ((a - rt) / a) ** 1.5
        ft = D * q * shape
        f[tip] = ft
        dD[tip] = q * shape
        dr[tip] = -1.5 * ft / (a - rt)
        da[tip] = 1.5 * ft * rt / (a * (a - rt))
        drho = (k - 1.0) / (k * lam * lam)
        dlam[tip] = ft * (-rho * drho / (1.0 - rho * rho) + 4.5 * lam / (k * (2.0 - k)))
    return f, dD, dr, da, dlam


def _eaz_jacobian(x, p) -> np.ndarray:
    """Jacobian of :func:`eaz_profile` with respect to ``(D, xc, a, lambda)``."""
    D, xc, a, lam = p
    d = np.asarray(x, dtype=float) - xc
    _, dD, dr, da, dlam = _half_derivatives(d, D, a, lam)
    return np.column_stack([dD, -np.sign(d) * dr, da, dlam])


def _eaz_asymmetric_jacobian(x, p) -> np.ndarray:
    """Jacobian of :func:`eaz_asymmetric_profile` with respect to its six parameters."""
    D, xc, a_minus, a_plus, lam_minus, lam_plus = p
    d = np.asarray(x, dtype=float) - xc
    jac = np.zeros((d.size, 6))
    for side, a, lam, cols, sign in ((d >= 0, a_plus, lam_plus, (3, 5), -1.0),
                                     (d < 0, a_minus, lam_minus, (2, 4), 1.0)):
        if np.any(side):
            _, dD, dr, da, dlam = _half_derivatives(d[side], D, a, lam)
            jac[side, 0] = dD
            jac[side, 1] = sign * dr
            jac[side, cols[0]] = da
            jac[side, cols[1]] = dlam
    return jac


def _unit_half(r, a, lam) -> np.ndarray:
    """:func:`eaz_half` with ``D = 1`` for arrays of ``a`` and ``lambda`` broadcast against ``r``."""
    r, a, lam = np.broadcast_arrays(np.abs(np.asarray(r, dtype=float)), np.asarray(a, dtype=float),
                                    np.asarray(lam, dtype=float))
    rj = a * (np.sqrt(1.0 + 3.0 * lam * lam) - 1.0)
    ellipse = lam * a
    out = np.zeros(r.shape)
    inner = r <= rj
    out[inner] = np.sqrt(np.clip(1.0 - (r[inner] / ellipse[inner]) ** 2, 0.0, None))
    tip = (r > rj) & (r < a)
    d_joint = np.sqrt(np.clip(1.0 - (rj[tip] / ellipse[tip]) ** 2, 0.0, None))
    out[tip] = d_joint * ((a[tip] - r[tip]) / (a[tip] - rj[tip])) ** 1.5
    return out


# ------------------------------------------------------------------ profile fit
@dataclass
class ProfileFit:
    model: str
    params: dict
    valid: bool
    reason: str = ""
    n: int = 0
    weighted: bool = False
    r2: float = float("nan")
    rms: float = float("nan")
    tips_mm: tuple = (float("nan"), float("nan"))
    peak_x_mm: float = float("nan")
    peak: float = float("nan")
    at_bounds: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)      # parameters the sensors leave free, and "tips"
    ranges: dict = field(default_factory=dict)          # name -> [lo, hi] fitting equally well; NaN: open
    fixed: list = field(default_factory=list)           # parameters held at the user's values

    def evaluate(self, x) -> np.ndarray:
        if not self.valid:
            return np.full(np.shape(x), np.nan)
        return evaluate(self.model, self.params, x)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["params"] = {k: float(v) for k, v in self.params.items()}
        d["tips_mm"] = [float(v) for v in self.tips_mm]
        d["rms_um"] = d.pop("rms")
        d["peak_um"] = d.pop("peak")
        d["unresolved"] = [str(v) for v in self.unresolved]
        d["ranges"] = {k: [float(v) for v in r] for k, r in self.ranges.items()}
        return d


def _invalid(model: str, reason: str, n: int, weighted: bool) -> ProfileFit:
    return ProfileFit(model, {name: float("nan") for name in PARAMETERS[model]}, False, reason, n, weighted)


def _polish(fun, p0, bounds, x_scale, max_nfev, jac):
    try:
        res = least_squares(fun, p0, jac=jac, bounds=bounds, x_scale=x_scale, method="trf", max_nfev=max_nfev)
    except ValueError:
        return None
    return res if np.all(np.isfinite(res.x)) else None


def _best(fun, starts, bounds, x_scale, max_nfev: Optional[int] = None, jac="2-point",
          race: bool = False) -> Optional[object]:
    """Polish every start and keep the lowest cost.  With ``race`` every start first
    gets ``RACE_NFEV`` evaluations, and those of the best ``N_RACED`` that used them
    up go on to ``max_nfev``."""
    lo, hi = (np.asarray(b, dtype=float) for b in bounds)
    with np.errstate(invalid="ignore"):
        inner_lo = np.where(np.isfinite(lo), lo + 1e-12 * (np.abs(lo) + 1), lo)     # strictly inside finite bounds
        inner_hi = np.where(np.isfinite(hi), hi - 1e-12 * (np.abs(hi) + 1), hi)
    results = []
    for p0 in starts:
        p0 = np.clip(np.asarray(p0, dtype=float), inner_lo, inner_hi)
        if np.all(np.isfinite(p0)):
            results.append(_polish(fun, p0, bounds, x_scale, RACE_NFEV if race else max_nfev, jac))
    results = sorted((r for r in results if r is not None), key=lambda r: r.cost)
    if race:
        for i, res in enumerate(results[:N_RACED]):
            if res.status == 0:
                more = _polish(fun, res.x, bounds, x_scale, max_nfev - RACE_NFEV, jac)
                results[i] = more if more is not None and more.cost <= res.cost else res
    return min(results, key=lambda r: r.cost) if results else None


def _settle(fun, res, bounds, x_scale, floor: float, jac="2-point"):
    """Continue the best start, for up to ``MAX_NFEV_FINAL`` more evaluations in
    chunks, until a chunk lowers the cost by less than 1e-4 of it; keep the result
    if it is better.  A fresh start also gets past a premature stop on the cost
    tolerance, which a badly scaled (weighted) cost can cause.  Returns ``(res,
    converged)``: not converged when the cost still fell by more than 1 % (and
    ``floor``) and the whole budget was used.  A flat valley (lambda free) creeps
    with a stable cost, and counts as converged."""
    start, chunk = res, MAX_NFEV_FINAL // 10
    for _ in range(10):
        more = _polish(fun, res.x, bounds, x_scale, chunk, jac)
        if more is None or not more.cost < res.cost:
            return res, True
        gain = res.cost - more.cost
        res = more
        if gain < 1e-4 * res.cost:
            return res, True
    return res, not start.cost - res.cost > max(0.01 * start.cost, floor)


def _amplitude(shapes, y, w2, hi: float) -> np.ndarray:
    """Least-squares amplitude of each shape (the sensors on the last axis), clipped to [0, hi]."""
    num = np.sum(shapes * (w2 * y), axis=-1)
    den = np.sum(shapes * shapes * w2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        amp = np.where(den > 0, num / den, 0.0)
    return np.clip(amp, 0.0, hi)


def _picks(points, costs, keys, k: int) -> list:
    """The ``k`` lowest-cost points, one per key."""
    seen, out = set(), []
    for i in np.argsort(costs, kind="stable"):
        key = keys[i].item() if isinstance(keys, np.ndarray) else keys[i]
        if not np.isfinite(costs[i]) or key in seen:
            continue
        seen.add(key)
        out.append(np.asarray(points[i], dtype=float))
        if len(out) == k:
            break
    return out


def _parabola(u, y, w, positive, zero) -> Optional[np.ndarray]:
    """``b`` of the weighted least-squares parabola through the sensors ``positive``
    that is zero at the sensors ``zero``; None when they do not determine it."""
    A = np.column_stack([np.ones_like(u), u, u * u])
    basis = np.eye(3)
    if len(zero):
        _, s, vt = np.linalg.svd(A[zero])
        basis = vt[int(np.count_nonzero(s > 1e-12 * s[0])):].T
    coef, _, rank, _ = np.linalg.lstsq(w[positive, None] * A[positive] @ basis, w[positive] * y[positive],
                                       rcond=None)
    return basis @ coef if rank == basis.shape[1] else None


def _quadratic_starts(u, y, w, step: float) -> list:
    """Starting points ``b`` (``b0 + b1 u + b2 u^2``) for the masked quadratic.  At the
    minimum the positive sensors are a run in x (a concave parabola) or a prefix plus
    a suffix (a convex one), and a sensor at a root is next to them; the minimum is the
    weighted parabola through the positive sensors that is zero at those.  So: that
    parabola for every such set, with and without its neighbours at zero, and a coarse
    grid of vertex (the sensors +- one span), half-width (0.5 to 8 sensor spacings) and
    sign with the height in closed form.  The best of them, one per set of positive
    sensors and sign of the curvature."""
    n = u.size
    order = np.argsort(u, kind="stable")
    sets = [(order[i:j], order[i - 1:i], order[j:j + 1]) for i in range(n) for j in range(i + 1, n + 1)]
    sets += [(np.concatenate((order[:i], order[n - k:])), order[i:i + 1], order[n - k - 1:n - k])
             for i in range(1, n) for k in range(1, n - i)]
    points = []
    for positive, before, after in sets:
        for zero in {(), tuple(before), tuple(after), tuple(np.union1d(before, after))}:
            if len(positive) + len(zero) >= 3:
                b = _parabola(u, y, w, positive, list(zero))
                if b is not None:
                    points.append(b)
    vertex = np.arange(u.min() - 2.0, u.max() + 2.0 + 0.25 * step, 0.5 * step)      # a span is 2 in u
    half = step * np.geomspace(0.5, 8.0, 9)[:, None]
    du = (u[None, None, :] - vertex[None, :, None]) / half[:, :, None]
    for sign in (1.0, -1.0):
        h = sign * _amplitude(np.clip(sign * (1.0 - du * du), 0.0, None), y, w * w, np.inf)
        b = np.stack(np.broadcast_arrays(h * (1.0 - vertex ** 2 / half ** 2), 2.0 * h * vertex / half ** 2,
                                         -h / half ** 2), axis=-1)
        points.extend(b.reshape(-1, 3))
    points = np.array(points)
    values = points[:, :1] + points[:, 1:2] * u + points[:, 2:3] * u * u
    costs = np.sum((w * (np.maximum(values, 0.0) - y)) ** 2, axis=1)
    keys = [((v > 0).tobytes(), bool(p[2] > 0)) for v, p in zip(values, points)]
    return _picks(points, costs, keys, N_POLISHED["quadratic"])


def _tip_grid(x, span: float, step: float) -> np.ndarray:
    """Rupture tip positions for the search grids: at the sensors (a tip there makes
    the profile exactly zero at that sensor), halfway between them, and beyond the ends."""
    u = np.unique(x)
    beyond = np.array([0.5 * step, 0.5 * span, span, 2.0 * span, 4.0 * span, 8.0 * span])
    return np.concatenate([u.min() - beyond, u, 0.5 * (u[1:] + u[:-1]), u.max() + beyond])


def _run_keys(i: int, shapes) -> np.ndarray:
    """One key per grid position ``i`` and run of slipping sensors (sorted in x)."""
    inside = shapes > 0
    n = inside.shape[-1] + 1
    return ((i * n + np.argmax(inside, axis=-1)) * n + np.count_nonzero(inside, axis=-1)).ravel()


def _eaz_starts(x, y, w, d_hi: float, span: float, step: float) -> list:
    """The best points ``(D, xc, a, lambda)`` of a coarse grid over the symmetric
    profile with ``D`` in closed form: ``xc`` every half sensor spacing, a tip at
    every grid tip position, five values of lambda; one per position and run of
    slipping sensors."""
    order = np.argsort(x, kind="stable")
    x, y, w = x[order], y[order], w[order]
    tips = _tip_grid(x, span, step)
    lam = np.array([LAMBDA_BOUNDS[0], 0.1, 0.3, 0.6, 0.95])
    points, costs, keys = [], [], []
    for i, xc in enumerate(np.arange(x.min() - 0.5 * span, x.max() + 0.5 * span + 0.25 * step, 0.5 * step)):
        a = np.unique(np.abs(tips - xc))
        a = a[a > 0.05 * step][:, None]
        shapes = _unit_half(x - xc, a[..., None], lam[:, None])                      # (a, lambda, sensor)
        D = _amplitude(shapes, y, w * w, d_hi)
        costs.append(np.sum((w * (D[..., None] * shapes - y)) ** 2, axis=-1).ravel())
        points.append(np.stack(np.broadcast_arrays(D, xc, a, lam), axis=-1).reshape(-1, 4))
        keys.append(_run_keys(i, shapes))
    return _picks(np.concatenate(points), np.concatenate(costs), np.concatenate(keys), N_POLISHED["eaz"])


def _eaz_asymmetric_starts(x, y, w, d_hi: float, span: float, step: float) -> list:
    """The best points of a coarse grid over ``(xc, a_minus, a_plus, lambda_minus,
    lambda_plus)`` with ``D`` in closed form: ``xc`` every half sensor spacing, each
    tip at every grid tip position on its side, three values of each lambda; one per
    position and run of slipping sensors."""
    order = np.argsort(x, kind="stable")
    x, y, w = x[order], y[order], w[order]
    tips = _tip_grid(x, span, step)
    lam = np.array([LAMBDA_BOUNDS[0], 0.1, 0.5, 0.95])
    points, costs, keys = [], [], []
    for i, xc in enumerate(np.arange(x.min() - 0.25 * span, x.max() + 0.25 * span + 0.25 * step, 0.5 * step)):
        d = x - xc
        a_minus = np.repeat(xc - tips[tips < xc - 0.05 * step], lam.size)
        a_plus = np.repeat(tips[tips > xc + 0.05 * step] - xc, lam.size)
        l_minus, l_plus = np.tile(lam, a_minus.size // lam.size), np.tile(lam, a_plus.size // lam.size)
        minus = np.where(d < 0, _unit_half(d, a_minus[:, None], l_minus[:, None]), 0.0)
        plus = np.where(d >= 0, _unit_half(d, a_plus[:, None], l_plus[:, None]), 0.0)
        shapes = minus[:, None, :] + plus[None, :, :]                                 # (minus, plus, sensor)
        D = _amplitude(shapes, y, w * w, d_hi)
        costs.append(np.sum((w * (D[..., None] * shapes - y)) ** 2, axis=-1).ravel())
        points.append(np.stack(np.broadcast_arrays(D, xc, a_minus[:, None], a_plus, l_minus[:, None], l_plus),
                               axis=-1).reshape(-1, 6))
        keys.append(_run_keys(i, shapes))
    return _picks(np.concatenate(points), np.concatenate(costs), np.concatenate(keys),
                  N_POLISHED["eaz_asymmetric"])


def _arrest_zone_setup(model: str, x, y, w, y_peak: float, span: float, step: float, x_peak: float):
    """Bounds, starting points, scales, profile and Jacobian of an arrest-zone fit:
    ``(lo, hi, starts, x_scale, profile, jacobian)``."""
    lam_lo, lam_hi = LAMBDA_BOUNDS
    d_hi = 10.0 * y_peak
    xc_lo, xc_hi = float(x.min()) - 2.0 * span, float(x.max()) + 2.0 * span
    a_lo, a_hi = 1e-3 * span, 20.0 * span
    if model == "eaz":
        lo = np.array([0.0, xc_lo, a_lo, lam_lo])
        hi = np.array([d_hi, xc_hi, a_hi, lam_hi])
        starts = _eaz_starts(x, y, w, d_hi, span, step)
        x_scale = np.array([y_peak, span, span, 1.0])
        profile, jacobian = eaz_profile, _eaz_jacobian
    else:
        lo = np.array([0.0, xc_lo, a_lo, a_lo, lam_lo, lam_lo])
        hi = np.array([d_hi, xc_hi, a_hi, a_hi, lam_hi, lam_hi])
        symmetric = _fit_arrest_zone("eaz", x, y, w, y_peak, span, step, x_peak)[0]
        starts = []
        if symmetric is not None:
            D, xc, a, lam = symmetric.x
            starts.append([D, xc, a, a, lam, lam])
            starts += [[D, xc, fm * a, fp * a, lam, lam] for fm, fp in ((0.7, 1.4), (1.4, 0.7))]
        starts.append([1.1 * y_peak, x_peak, 1.5 * span, 1.5 * span, 0.5, 0.5])
        starts += _eaz_asymmetric_starts(x, y, w, d_hi, span, step)
        x_scale = np.array([y_peak, span, span, span, 1.0, 1.0])
        profile, jacobian = eaz_asymmetric_profile, _eaz_asymmetric_jacobian
    return lo, hi, starts, x_scale, profile, jacobian


def _fit_arrest_zone(model: str, x, y, w, y_peak: float, span: float, step: float, x_peak: float):
    """Bounded least squares of an arrest-zone model.  Returns ``(res, converged,
    jac, lo, hi, x_scale)``, with ``res`` None when no start could be polished."""
    lo, hi, starts, x_scale, profile, jacobian = _arrest_zone_setup(model, x, y, w, y_peak, span, step, x_peak)

    def fun(p):
        return w * (profile(x, *p) - y)

    def jac(p):
        return w[:, None] * jacobian(x, p)

    res = _best(fun, starts, (lo, hi), x_scale, MAX_NFEV, jac, race=True)
    if res is None:
        return None, False, jac, lo, hi, x_scale
    res, converged = _settle(fun, res, (lo, hi), x_scale, 1e-20 * float(np.sum((w * y) ** 2)), jac)
    return res, converged, jac, lo, hi, x_scale


def _user_constraints(model: str, fixed, bounds) -> tuple:
    """The user's held values and bounds as ``(fixed, lo, hi, reason)``: ``fixed`` a dict
    of floats, ``lo``/``hi`` arrays in ``PARAMETERS[model]`` order (NaN: automatic),
    ``reason`` the first problem found or ''.  Unknown names raise ValueError."""
    names = PARAMETERS[model]
    fixed = dict(fixed or {})
    bounds = dict(bounds or {})
    unknown = (set(fixed) | set(bounds)) - set(names)
    if unknown:
        raise ValueError(f"{model} has no parameter {sorted(unknown)[0]!r}; it has {names}")
    lo = np.full(len(names), np.nan)
    hi = np.full(len(names), np.nan)
    for name, value in list(fixed.items()):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value):
            return {}, lo, hi, f"{name} must be a number"
        low, high, closed = DOMAIN.get(name, (-math.inf, math.inf, True))
        if value < low or (value == low and not closed) or value > high:
            return {}, lo, hi, f"{name} must be {'>=' if closed else '>'} {low:g}" + \
                (f" and <= {high:g}" if math.isfinite(high) else "")
        fixed[name] = value
    for name, pair in bounds.items():
        try:
            b_lo, b_hi = (math.nan if v is None else float(v) for v in pair)
        except (TypeError, ValueError):
            raise ValueError(f"the bounds of {name} must be a (min, max) pair") from None
        if math.isfinite(b_lo) and math.isfinite(b_hi) and b_lo >= b_hi:
            return {}, lo, hi, f"{name}: the minimum must be below the maximum"
        low, high, closed = DOMAIN.get(name, (-math.inf, math.inf, True))
        if (math.isfinite(b_hi) and (b_hi < low or (b_hi == low and not closed))) or \
                (math.isfinite(b_lo) and b_lo > high):
            return {}, lo, hi, f"{name}: the bounds leave no allowed value"
        i = names.index(name)
        lo[i], hi[i] = b_lo, b_hi
    return fixed, lo, hi, ""


def _effective_bounds(model: str, auto_lo, auto_hi, user_lo, user_hi) -> tuple:
    """The automatic bounds with the user's finite ones in their place, kept inside
    each parameter's domain."""
    lo = np.where(np.isfinite(user_lo), user_lo, auto_lo).astype(float)
    hi = np.where(np.isfinite(user_hi), user_hi, auto_hi).astype(float)
    for i, name in enumerate(PARAMETERS[model]):
        low, high, closed = DOMAIN.get(name, (-math.inf, math.inf, True))
        floor = low if closed else low + 1e-9 * (1.0 if high == math.inf else high - low)
        lo[i], hi[i] = max(lo[i], floor), min(hi[i], high)
    return lo, hi


def _fit_free(fun_full, jac_full, template, free, starts, lo, hi, x_scale, floor: float, race: bool) -> tuple:
    """Least squares over the ``free`` entries of ``template`` (the others hold their
    values).  Returns ``(p, converged)``, ``p`` None when no start could be polished."""
    def full(q):
        p = template.copy()
        p[free] = q
        return p

    def fun(q):
        return fun_full(full(q))

    def jac(q):
        return jac_full(full(q))[:, free]

    bounds = (lo[free], hi[free])
    res = _best(fun, [np.asarray(p0, dtype=float)[free] for p0 in starts], bounds, x_scale[free],
                MAX_NFEV, jac, race=race)
    if res is None:
        return None, False
    res, converged = _settle(fun, res, bounds, x_scale[free], floor, jac)
    return full(res.x), converged


def _null_space(jac) -> np.ndarray:
    """Orthonormal rows spanning the null space of ``jac``: the right-singular vectors
    whose singular values are below ``SINGULAR`` times the largest."""
    s, vt = np.linalg.svd(jac)[1:]
    if not s.size or not s[0] > 0:
        return np.eye(jac.shape[1])
    return vt[np.flatnonzero(s <= SINGULAR * s[0])]


def _tip_brackets(x, fitted, centre: float) -> dict:
    """Where each tip can lie given the sensors: from the outermost slipping sensor (or
    the centre) to the next sensor out, NaN when there is none."""
    slipping = x[fitted > 0]
    edge_lo = min([centre] + slipping.tolist())
    edge_hi = max([centre] + slipping.tolist())
    below, above = x[x < edge_lo], x[x > edge_hi]
    return {"tip_minus": [float(below.max()) if below.size else float("nan"), edge_lo],
            "tip_plus": [edge_hi, float(above.min()) if above.size else float("nan")]}


def _tip_factor(lam) -> np.ndarray:
    """``Q(lambda) = (delta_joint / D) (a / (a - r_joint))^(3/2)``, so that the tip part
    of a side is ``D Q ((a - r) / a)^(3/2)``.  It depends on lambda alone and rises
    from 1 at lambda 0 to infinity at 1."""
    lam = np.asarray(lam, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.sqrt(1.0 + 3.0 * lam * lam)
        rho = np.where(lam > 0, (k - 1.0) / np.where(lam > 0, lam, 1.0), 0.0)
        q = np.sqrt(np.clip(1.0 - rho * rho, 0.0, None)) / np.clip(2.0 - k, 0.0, None) ** 1.5
        return np.where(lam < 1.0, q, np.inf)


_LAMBDA_TABLE = np.linspace(0.0, 0.9999, 20000)
_Q_TABLE = _tip_factor(_LAMBDA_TABLE)


def _lambda_of_factor(q) -> np.ndarray:
    """The lambda at which :func:`_tip_factor` is ``q`` (``q >= 1``)."""
    return np.interp(q, _Q_TABLE, _LAMBDA_TABLE)


def _lambda_max(r_min: float, a: float) -> float:
    """The largest lambda that keeps a sensor at ``r_min`` out of ``r_joint``."""
    return min(1.0, math.sqrt(max(0.0, (1.0 + r_min / a) ** 2 - 1.0) / 3.0))


def _arrest_zone_side(r, D: float, a: float, lam: float):
    """What the sensors at distances ``r`` from ``xc`` (one side, or both for the
    symmetric model) leave free.  Returns ``(kind, info)``: ``"none"`` when no sensor
    lies inside the rupture; ``"tip"`` when none lies within ``r_joint``, so that they
    see only the tip part ``A ((a - r) / a)^(3/2)`` with ``A = D Q(lambda)``: lambda
    can take any value up to ``lam_max``, where the joint reaches the nearest sensor,
    and D follows (``info``: ``A``, ``lam_max`` and the range of D); ``"ellipse"`` when
    none lies in the tip part, so that only ``lambda a`` is fixed (``info``: ranges of
    lambda and a); ``""`` otherwise."""
    rj = eaz_joint(a, lam)
    inside = r < a
    if not np.any(inside):
        return "none", {}
    at_joint = np.abs(r - rj) <= 1e-6 * a                # on the joint the two parts agree
    if not np.any((r < rj) & ~at_joint):
        lam_max = _lambda_max(float(r.min()), a)
        A = D * float(_tip_factor(lam))                  # the tip part's amplitude, which the sensors fix
        return "tip", {"A": A, "lam_max": lam_max, "D": [A / float(_tip_factor(lam_max)), A]}
    if not np.any((r > rj) & inside & ~at_joint):
        E = lam * a                                       # the ellipse's half-axis, which the sensors fix
        q = min(1.0, float(r[(r <= rj) | at_joint].max()) / E)
        lam_min = 2.0 * q / (3.0 - q * q)                 # r_joint reaches the outermost sensor
        if np.any(~inside):
            lam_min = max(lam_min, E / float(r[~inside].min()))   # the tip reaches the next sensor out
        lam_min = min(lam_min, lam)
        return "ellipse", {"lambda": [lam_min, 1.0], "a": [E, E / lam_min if lam_min > 0 else float("nan")]}
    return "", {}


def _asymmetric_tip_ranges(x, params: dict, tip_sides) -> dict:
    """Equal-cost ranges of D and of lambda on the sides of an asymmetric fit that
    have no sensor within ``r_joint``, when no side has one.  The sensors there fix
    only the tip and the amplitude ``A = D Q(lambda)`` of each side, so ``xc`` (between
    the innermost sensors of the two sides, within the fit's bounds) and D both move:
    for each ``xc``, D lies between ``A (a / a_fit)^(3/2) / Q(lam_max)`` and
    ``A (a / a_fit)^(3/2)`` on every tip side."""
    xc_fit = params["xc"]
    tips = {"minus": xc_fit - params["a_minus"], "plus": xc_fit + params["a_plus"]}
    amplitude = {s: params["D"] * float(_tip_factor(params["lambda_" + s])) / params["a_" + s] ** 1.5
                 for s in tip_sides}                     # A / a^(3/2), fixed by the sensors
    left, right = x[x < xc_fit], x[x >= xc_fit]
    span = float(np.ptp(x)) or 1.0                       # within the fit's bounds on xc and a
    lo = max([tips["minus"], x.min() - 2.0 * span, tips["plus"] - 20.0 * span] + left.tolist())
    hi = min([tips["plus"], x.max() + 2.0 * span, tips["minus"] + 20.0 * span] + right.tolist())
    xc = np.append(np.linspace(lo, hi, 202)[1:-1], xc_fit)
    a = {"minus": xc - tips["minus"], "plus": tips["plus"] - xc}
    r_min = {"minus": xc - left.max() if left.size else np.full(xc.size, np.inf),
             "plus": right.min() - xc if right.size else np.full(xc.size, np.inf)}
    top = {s: amplitude[s] * a[s] ** 1.5 for s in tip_sides}                     # D at lambda 0
    lam_max = {s: np.array([_lambda_max(r, v) for r, v in zip(r_min[s], a[s])]) for s in tip_sides}
    d_lo = np.max([top[s] / _tip_factor(lam_max[s]) for s in tip_sides], axis=0)
    d_hi = np.min([top[s] for s in tip_sides], axis=0)
    ok = d_lo <= d_hi * (1.0 + 1e-9)
    ok[-1] = True                                        # the fit itself
    d_lo, d_hi = np.minimum(d_lo, d_hi), np.maximum(d_lo, d_hi)
    out = {"D": [float(d_lo[ok].min()), float(d_hi[ok].max())]}
    for s in tip_sides:
        lam_lo = _lambda_of_factor(np.clip(top[s] / d_hi, 1.0, None))[ok]
        lam_hi = np.minimum(_lambda_of_factor(np.clip(top[s] / d_lo, 1.0, None)), lam_max[s])[ok]
        out["lambda_" + s] = [float(lam_lo.min()), float(lam_hi.max())]
    return out


def _identify(model: str, params: dict, x, fitted, jac, tips, tip_gradients, free=None) -> tuple:
    """The parameters the sensors leave free, and the ranges that fit equally well.
    ``jac`` is the Jacobian per unit of the parameter scales, ``tip_gradients`` the
    gradients of the two tips in the same units.  Only the parameters in ``free``
    (default all) were fitted; held ones are resolved by the user."""
    names = PARAMETERS[model]
    free = set(names) if free is None else set(free)
    cols = [i for i, name in enumerate(names) if name in free]
    null = np.zeros((0, len(names)))
    if cols:
        sub = _null_space(jac[:, cols])
        null = np.zeros((sub.shape[0], len(names)))
        null[:, cols] = sub
    flags = {name for name, v in zip(names, np.sqrt(np.sum(null ** 2, axis=0))) if v > 0.3}
    loose = {}                                           # the rupture tips the sensors do not pin
    for side, g in zip(("minus", "plus"), tip_gradients):
        norm = float(np.linalg.norm(g))
        loose[side] = norm > 0 and float(np.linalg.norm(null @ (g / norm))) > 0.3
    ranges = {}
    few = np.count_nonzero(fitted > 0) <= len(cols)
    if few:
        flags |= set(EXTENT[model]) & free
        loose = {side: bool(np.any(np.asarray(g)[cols] != 0)) for side, g in zip(("minus", "plus"), tip_gradients)}
    if model == "eaz":
        kind, info = _arrest_zone_side(np.abs(x - params["xc"]), params["D"], params["a"], params["lambda"])
        if kind == "tip" and {"D", "lambda"} <= free:
            flags.add("lambda")
            if not few and flags <= {"D", "lambda"}:
                flags.discard("D")                        # D follows lambda within its range
                ranges.update({"lambda": [0.0, info["lam_max"]], "D": info["D"]})
        elif kind == "ellipse" and {"a", "lambda"} <= free:
            flags |= {"a", "lambda"}
            loose = {"minus": True, "plus": True}
            if not few:
                ranges.update(info)
                xc, (a_min, a_max) = params["xc"], info["a"]
                ranges["tip_minus"] = [xc - a_max, xc - a_min]
                ranges["tip_plus"] = [xc + a_min, xc + a_max]
    elif model == "eaz_asymmetric":
        sides = {}
        for sign, suffix in ((-1.0, "minus"), (1.0, "plus")):
            on_side = x < params["xc"] if sign < 0 else x >= params["xc"]
            kind, info = _arrest_zone_side(sign * (x[on_side] - params["xc"]), params["D"],
                                           params["a_" + suffix], params["lambda_" + suffix])
            sides[suffix] = (kind, info)
            pair = {"a_" + suffix, "lambda_" + suffix}
            if kind == "none":
                flags |= pair & free
                loose[suffix] = bool({"a_" + suffix, "xc"} & free)
            elif kind == "ellipse" and pair <= free:
                flags |= pair
                loose[suffix] = True
                if not few:
                    ranges.update({f"{k}_{suffix}": v for k, v in info.items()})
                    a_min, a_max = info["a"]
                    ranges["tip_" + suffix] = sorted((params["xc"] + sign * a_min, params["xc"] + sign * a_max))
        tip_sides = [suffix for suffix, (kind, _) in sides.items() if kind == "tip"]
        if tip_sides and all(kind in ("tip", "none") for kind, _ in sides.values()) and \
                {"D"} | {"lambda_" + suffix for suffix in tip_sides} <= free:
            # no sensor within a joint fixes D: D moves, and lambda on each tip side with it
            flags |= {"D"} | {"lambda_" + suffix for suffix in tip_sides}
            if not few:
                ranges.update(_asymmetric_tip_ranges(x, params, tip_sides))
    if not np.all(np.isfinite(tips)):
        loose = {"minus": False, "plus": False}          # a convex parabola has no tips
    if any(loose.values()):
        flags.add("tips")
        centre = params["xc"] if model != "quadratic" else 0.5 * (tips[0] + tips[1])
        brackets = _tip_brackets(x, fitted, centre)
        for side in ("minus", "plus"):
            if loose[side]:
                ranges.setdefault("tip_" + side, brackets["tip_" + side])
    unresolved = [name for name in names if name in flags and name in free] + (["tips"] if "tips" in flags else [])
    ranges = {k: v for k, v in ranges.items() if k.startswith("tip_") or k in free}
    return unresolved, ranges


def _at_bounds(model: str, p, lo, hi, free) -> list:
    """The free parameters that ended on one of their bounds."""
    names = PARAMETERS[model]
    out = []
    for i in free:
        l, h = float(lo[i]), float(hi[i])
        tol = 1e-6 * (h - l) if math.isfinite(h - l) else 1e-9 * max(1.0, abs(float(p[i])))
        if (math.isfinite(l) and p[i] - l <= tol) or (math.isfinite(h) and h - p[i] <= tol):
            out.append(names[i])
    return out


def fit_profile(x_mm, slip, sigma=None, model: str = "eaz", weighted: bool = False,
                use: Optional[Sequence[bool]] = None, fixed: Optional[dict] = None,
                bounds: Optional[dict] = None) -> ProfileFit:
    """Fit a slip profile model to the slip of the sensors at ``x_mm``.

    ``use`` selects sensors (default all); points with a non-finite position or
    slip are always left out, and with ``weighted`` those without a positive
    finite ``sigma`` too, or with one at most 1e-6 of the median (a constant,
    dead or saturated, record).  Needs at least one point more than the model
    has free parameters.  The fit is invalid when the fitted profile is zero at
    every sensor, or, given ``sigma``, nowhere above a sensor's noise.

    ``fixed`` holds parameters at given values (name -> value; all of them held
    gives the profile without a fit) and ``bounds`` replaces the automatic
    bounds of free ones (name -> ``(min, max)``, NaN or None for the automatic
    end).  Without either, the global search runs as it is; with them, the
    free parameters are polished from the same starting points.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; use one of {MODELS}")
    held, user_lo, user_hi, problem = _user_constraints(model, fixed, bounds)
    x = np.asarray(x_mm, dtype=float).ravel()
    y = np.asarray(slip, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and slip must have the same length")
    s = np.ones_like(y) if sigma is None else np.asarray(sigma, dtype=float).ravel()
    if s.shape != y.shape:
        raise ValueError("sigma must have one value per sensor")
    keep = np.isfinite(x) & np.isfinite(y)
    if use is not None:
        u = np.asarray(use, dtype=bool).ravel()
        if u.shape != y.shape:
            raise ValueError("use must have one value per sensor")
        keep &= u
    if weighted:
        with np.errstate(invalid="ignore"):
            keep &= np.isfinite(s) & (s > 0)
            if np.any(keep):
                keep &= s > 1e-6 * float(np.median(s[keep]))
    n = int(np.count_nonzero(keep))
    if problem:
        return _invalid(model, problem, n, weighted)
    names = PARAMETERS[model]
    free = [i for i, name in enumerate(names) if name not in held]
    constrained = bool(held) or bool(np.any(np.isfinite(user_lo)) or np.any(np.isfinite(user_hi)))
    n_par = len(free)
    if n < n_par + 1:
        return _invalid(model, f"needs at least {n_par + 1} sensors, has {n}", n, weighted)
    xk, yk = x[keep], y[keep]
    w = 1.0 / s[keep] if weighted else np.ones(n)
    y_peak = float(np.max(yk))
    if not y_peak > 0:
        return _invalid(model, "no positive slip to fit", n, weighted)
    span = float(xk.max() - xk.min()) or 1.0
    positions = np.unique(xk)
    step = float(np.median(np.diff(positions))) if positions.size > 1 else span
    x_peak = float(xk[int(np.argmax(yk))])

    if model == "quadratic":
        xm, xs = float(np.mean(xk)), 0.5 * span
        uk = (xk - xm) / xs

        powers = np.column_stack([np.ones(n), uk, uk * uk])

        def fun(b):
            return w * (np.maximum(0.0, powers @ b) - yk)

        def jac(b):
            return (w * (powers @ b > 0))[:, None] * powers

        def to_c(b):
            b0, b1, b2 = (float(v) for v in b)
            return np.array([b0 - b1 * xm / xs + b2 * xm ** 2 / xs ** 2, b1 / xs - 2.0 * b2 * xm / xs ** 2,
                             b2 / xs ** 2])

        starts = [np.polyfit(uk, yk, 2, w=w)[::-1], [y_peak, 0.0, -y_peak], [y_peak, 0.0, -0.25 * y_peak]]
        starts += _quadratic_starts(uk, yk, w, step / xs)
        floor = 1e-20 * float(np.sum((w * yk) ** 2))
        lo, hi = _effective_bounds(model, np.full(3, -np.inf), np.full(3, np.inf), user_lo, user_hi)
        if not constrained:
            bounds = (lo, hi)
            res = _best(fun, starts, bounds, "jac", jac=jac)
            converged = res is not None
            if res is not None:
                res, converged = _settle(fun, res, bounds, "jac", floor, jac)
            if res is None or not converged:
                return _invalid(model, "the fit did not converge", n, weighted)
            c = to_c(res.x)
        else:
            powers_x = np.column_stack([np.ones(n), xk, xk * xk])
            template = np.array([held.get(name, np.nan) for name in names])
            c = template
            if free:
                c, converged = _fit_free(lambda c: w * (np.maximum(0.0, powers_x @ c) - yk),
                                         lambda c: (w * (powers_x @ c > 0))[:, None] * powers_x,
                                         template, free, [to_c(b) for b in starts], lo, hi,
                                         y_peak * np.array([1.0, 1.0 / xs, 1.0 / xs ** 2]), floor, False)
                if c is None or not converged:
                    return _invalid(model, "the fit did not converge", n, weighted)
        c0, c1, c2 = (float(v) for v in c)
        params = {"c0": c0, "c1": c1, "c2": c2}
        tips = (float("nan"), float("nan"))
        peak_x = peak = float("nan")
        if c2 < 0:
            disc = c1 * c1 - 4.0 * c2 * c0
            peak_x = -c1 / (2.0 * c2)
            peak = float(quadratic_masked(peak_x, c0, c1, c2))
            if disc > 0:
                r1, r2 = sorted(((-c1 - math.sqrt(disc)) / (2.0 * c2), (-c1 + math.sqrt(disc)) / (2.0 * c2)))
                tips = (r1, r2)
        at_bounds = _at_bounds(model, c, lo, hi, free)
        positive = (c0 + c1 * xk + c2 * xk * xk) > 0
        scale = y_peak * np.array([1.0, 1.0 / xs, 1.0 / xs ** 2])        # the Jacobian per unit of c
        jac = (w * positive)[:, None] * np.column_stack([np.ones(n), xk, xk * xk]) * scale
        # a root r of the parabola moves by -(dc0 + r dc1 + r^2 dc2) / p'(r)
        tip_gradients = [-np.array([1.0, r, r * r]) / (c1 + 2.0 * c2 * r) * scale if math.isfinite(r)
                         else np.zeros(3) for r in tips]
    else:
        if not constrained:
            res, converged, jac, lo, hi, x_scale = _fit_arrest_zone(model, xk, yk, w, y_peak, span, step, x_peak)
            if res is None or not converged:
                return _invalid(model, "the fit did not converge", n, weighted)
            p_fit = np.array(res.x, dtype=float)
        else:
            auto_lo, auto_hi, starts, x_scale, profile, jacobian = _arrest_zone_setup(
                model, xk, yk, w, y_peak, span, step, x_peak)
            lo, hi = _effective_bounds(model, auto_lo, auto_hi, user_lo, user_hi)
            p_fit = np.array([held.get(name, np.nan) for name in names])
            if free:
                p_fit, converged = _fit_free(lambda p: w * (profile(xk, *p) - yk),
                                             lambda p: w[:, None] * jacobian(xk, p), p_fit, free, starts, lo, hi,
                                             x_scale, 1e-20 * float(np.sum((w * yk) ** 2)), True)
                if p_fit is None or not converged:
                    return _invalid(model, "the fit did not converge", n, weighted)

            def jac(p):
                return w[:, None] * jacobian(xk, p)
        params = dict(zip(names, (float(v) for v in p_fit)))
        at_bounds = _at_bounds(model, p_fit, lo, hi, free)
        if model == "eaz":
            tips = (params["xc"] - params["a"], params["xc"] + params["a"])
        else:
            tips = (params["xc"] - params["a_minus"], params["xc"] + params["a_plus"])
        peak_x, peak = params["xc"], params["D"]
        jac = jac(p_fit) * x_scale
        a_minus, a_plus = (2, 2) if model == "eaz" else (2, 3)      # tips xc - a and xc + a
        tip_gradients = [np.zeros(len(x_scale)), np.zeros(len(x_scale))]
        tip_gradients[0][[1, a_minus]] = (1.0, -1.0)
        tip_gradients[1][[1, a_plus]] = (1.0, 1.0)
        tip_gradients = [g * x_scale for g in tip_gradients]

    fitted = evaluate(model, params, xk)
    if not np.any(fitted > 0):
        return _invalid(model, "the fitted profile is zero at every sensor", n, weighted)
    if sigma is not None and not np.any(fitted > np.where(np.isfinite(s[keep]), s[keep], 0.0)):
        return _invalid(model, "the fitted profile is within the noise at every sensor", n, weighted)
    unresolved, ranges = _identify(model, params, xk, fitted, jac, tips, tip_gradients,
                                   free=[names[i] for i in free])
    resid = yk - fitted
    ss_tot = float(np.sum((yk - yk.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return ProfileFit(model, params, True, "", n, bool(weighted), r2, rms,
                      (float(tips[0]), float(tips[1])), float(peak_x), float(peak), at_bounds, unresolved, ranges,
                      [name for name in names if name in held])


# ------------------------------------------------------------------- the front
@dataclass
class FrontSegment:
    x_from: float
    x_to: float
    n: int
    slope_s_per_mm: float
    intercept_s: float
    velocity_m_s: float              # 1 / slope; positive towards +x
    r2: float
    resolved: bool = True            # False when the front crosses the segment within one sample
    min_speed_m_s: float = float("nan")      # crossing it in one sample: the lower bound when unresolved


@dataclass
class FrontFit:
    mode: str
    valid: bool
    reason: str = ""
    segments: list = field(default_factory=list)
    split_x_mm: float = float("nan")         # bilateral: where the two lines meet, or where one line starts
    t0_s: float = float("nan")               # bilateral: the fitted time there, on the clock of t

    def as_dict(self) -> dict:
        return {"mode": self.mode, "valid": bool(self.valid), "reason": self.reason,
                "split_x_mm": float(self.split_x_mm), "t0_s": float(self.t0_s),
                "segments": [asdict(s) for s in self.segments]}


def _flat(slope: float, x_span: float, t_span: float) -> bool:
    """A slope that is zero to rounding: less than 1e-9 of the times' range across ``x_span``."""
    return t_span == 0.0 or abs(slope) * x_span <= 1e-9 * t_span


def _make_segment(x_from: float, x_to: float, n: int, slope: float, intercept: float, r2: float,
                  dt_s: Optional[float]) -> FrontSegment:
    velocity = 1e-3 / slope if slope != 0 else math.inf
    resolved, min_speed = True, float("nan")
    if dt_s is not None:
        resolved = abs(slope) * (x_to - x_from) >= dt_s
        min_speed = (x_to - x_from) * 1e-3 / dt_s
    return FrontSegment(float(x_from), float(x_to), int(n), float(slope), float(intercept), float(velocity),
                        float(r2), bool(resolved), float(min_speed))


def _r2(t, predicted) -> float:
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return 1.0 - float(np.sum((t - predicted) ** 2)) / ss_tot if ss_tot > 0 else float("nan")


def _segment(x, t, dt_s: Optional[float] = None) -> Optional[FrontSegment]:
    if x.size < 2 or np.unique(x).size < 2:
        return None
    slope, intercept = np.polyfit(x, t, 1)
    if _flat(slope, float(np.ptp(x)), float(np.ptp(t))):     # polyfit returns ~1e-19, not 0
        slope, intercept = 0.0, float(np.mean(t))
    return _make_segment(x.min(), x.max(), x.size, slope, intercept, _r2(t, slope * x + intercept), dt_s)


def _hinge(x, t) -> tuple:
    """Least squares of ``t0 + s_minus min(x - x0, 0) + s_plus max(x - x0, 0)`` with the
    apex ``x0`` free from the second to the second-last sensor position, so that each
    line has two sensors (one of them may sit at the apex).  Solved exactly: the apex
    at each sensor, and in each interval between sensors the two free lines when they
    meet inside it.  Returns ``(x0, t0, s_minus, s_plus)``."""
    u = np.unique(x)
    best = None
    for x0 in u[1:-1]:
        A = np.column_stack([np.ones_like(x), np.minimum(x - x0, 0.0), np.maximum(x - x0, 0.0)])
        coef = np.linalg.lstsq(A, t, rcond=None)[0]
        ss = float(np.sum((A @ coef - t) ** 2))
        if best is None or ss < best[0]:
            best = (ss, float(x0), *(float(v) for v in coef))
    for k in range(1, u.size - 2):
        left, right = x <= u[k], x >= u[k + 1]
        s_left, i_left = np.polyfit(x[left], t[left], 1)
        s_right, i_right = np.polyfit(x[right], t[right], 1)
        if s_left == s_right:
            continue
        x0 = (i_right - i_left) / (s_left - s_right)
        if not u[k] < x0 < u[k + 1]:
            continue
        ss = float(np.sum((s_left * x[left] + i_left - t[left]) ** 2)
                   + np.sum((s_right * x[right] + i_right - t[right]) ** 2))
        if ss < best[0]:
            best = (ss, float(x0), float(s_left * x0 + i_left), float(s_left), float(s_right))
    return best[1:]


def _earliest(x, t) -> int:
    """The earliest sensor; among sensors tied for it, the middle one (of two, the one
    nearer the middle of all)."""
    tied = np.flatnonzero(t - t.min() <= 1e-9 * float(np.ptp(t)))
    middle = 0.5 * (x[tied].min() + x[tied].max())
    centre = 0.5 * (x.min() + x.max())
    return int(min(tied, key=lambda i: (abs(x[i] - middle), abs(x[i] - centre), x[i])))


def _one_line(mode: str, x, t, dt_s) -> FrontFit:
    """A bilateral fit that is one line: the apex is where the line is earliest."""
    seg = _segment(x, t, dt_s)
    if seg.slope_s_per_mm == 0:
        return FrontFit(mode, True, "", [seg])
    split = seg.x_from if seg.slope_s_per_mm > 0 else seg.x_to
    return FrontFit(mode, True, "", [seg], split, seg.slope_s_per_mm * split + seg.intercept_s)


def fit_front(x_mm, t_s, use: Optional[Sequence[bool]] = None, mode: str = "single",
              dt_s: Optional[float] = None) -> FrontFit:
    """Line(s) through the event times ``t_s`` (s) against the positions ``x_mm``.

    ``bilateral`` fits a hinge (two lines meeting at a free apex) to at least four
    sensors; when both lines run the same way (or are both flat) the front is one
    line through all of them.  With fewer sensors it keeps two lines through the
    earliest one.  ``dt_s`` is the sampling interval of the times: a segment
    that the front crosses within less than one sample is not resolved.
    """
    if mode not in FRONT_MODES:
        raise ValueError(f"unknown mode {mode!r}; use one of {FRONT_MODES}")
    x = np.asarray(x_mm, dtype=float).ravel()
    t = np.asarray(t_s, dtype=float).ravel()
    if x.shape != t.shape:
        raise ValueError("x and t must have the same length")
    if dt_s is not None and not (math.isfinite(dt_s) and dt_s > 0):
        dt_s = None
    keep = np.isfinite(x) & np.isfinite(t)
    if use is not None:
        u = np.asarray(use, dtype=bool).ravel()
        if u.shape != x.shape:
            raise ValueError("use must have one value per sensor")
        keep &= u
    xk, tk = x[keep], t[keep]
    if mode == "bilateral" and xk.size == 0:
        return FrontFit(mode, False, "no sensors selected")
    if np.unique(xk).size < 2:
        return FrontFit(mode, False, "needs at least two sensors at different positions")
    if mode == "single":
        return FrontFit(mode, True, "", [_segment(xk, tk, dt_s)])
    if xk.size >= 4 and np.unique(xk).size >= 4:
        x0, t0, s_minus, s_plus = _hinge(xk, tk)
        t_span = float(np.ptp(tk))
        s_minus = 0.0 if _flat(s_minus, x0 - xk.min(), t_span) else s_minus
        s_plus = 0.0 if _flat(s_plus, xk.max() - x0, t_span) else s_plus
        if np.sign(s_minus) == np.sign(s_plus):              # no apex: the front runs one way
            return _one_line(mode, xk, tk, dt_s)
        segments = []
        for side, slope, x_from, x_to in ((xk <= x0, s_minus, xk.min(), x0), (xk >= x0, s_plus, x0, xk.max())):
            intercept = t0 - slope * x0
            segments.append(_make_segment(x_from, x_to, np.count_nonzero(side), slope, intercept,
                                          _r2(tk[side], slope * xk[side] + intercept), dt_s))
        return FrontFit(mode, True, "", segments, x0, t0)
    first = _earliest(xk, tk)
    split = float(xk[first])
    segments = [seg for seg in (_segment(xk[side], tk[side], dt_s) for side in (xk <= split, xk >= split))
                if seg is not None]
    return FrontFit(mode, True, "", segments, split, float(tk[first]))


__all__ = ["MODELS", "MODEL_LABELS", "PARAMETERS", "LAMBDA_BOUNDS", "FRONT_MODES", "quadratic_masked",
           "eaz_joint", "eaz_half", "eaz_profile", "eaz_asymmetric_profile", "evaluate", "ProfileFit",
           "fit_profile", "FrontSegment", "FrontFit", "fit_front"]
