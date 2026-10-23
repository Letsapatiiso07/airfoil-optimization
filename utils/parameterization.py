"""
Airfoil geometry: NACA 4-digit sections and the CST (Kulfan) parameterisation.

Coordinates follow the Selig order used by XFOIL and the UIUC database:
upper trailing edge -> leading edge -> lower trailing edge, chord = 1.
"""

from math import comb
from typing import Tuple

import numpy as np


def cosine_spacing(n_points: int) -> np.ndarray:
    """x/c from 0 to 1, clustered at the leading and trailing edges."""
    beta = np.linspace(0.0, np.pi, n_points)
    return 0.5 * (1.0 - np.cos(beta))


def to_selig(xu, yu, xl, yl) -> Tuple[np.ndarray, np.ndarray]:
    """Join upper and lower surfaces (both ordered LE -> TE) into one loop."""
    x = np.concatenate([xu[::-1], xl[1:]])
    y = np.concatenate([yu[::-1], yl[1:]])
    return x, y


def naca4_camber(x: np.ndarray, m: float, p: float) -> Tuple[np.ndarray, np.ndarray]:
    """NACA 4-digit mean line yc(x) and its slope dyc/dx."""
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m > 0 and p > 0:
        fwd = x < p
        yc[fwd] = m / p**2 * (2 * p * x[fwd] - x[fwd]**2)
        dyc[fwd] = 2 * m / p**2 * (p - x[fwd])
        aft = ~fwd
        yc[aft] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x[aft] - x[aft]**2)
        dyc[aft] = 2 * m / (1 - p)**2 * (p - x[aft])
    return yc, dyc


def thin_airfoil_alpha_l0(m: float, p: float) -> float:
    """Zero-lift angle (deg) of a NACA 4-digit mean line by thin-airfoil theory:
    alpha_L0 = -(1/pi) * integral_0^pi dyc/dx (cos t - 1) dt, x = (1 - cos t) / 2."""
    if m == 0 or p == 0:
        return 0.0
    t = np.linspace(0.0, np.pi, 20001)
    _, dyc = naca4_camber(0.5 * (1 - np.cos(t)), m, p)
    return float(-np.degrees(np.trapezoid(dyc * (np.cos(t) - 1), t) / np.pi))


def naca4_surfaces(code: str, n_points: int = 81, closed_te: bool = True):
    """Upper and lower surfaces of a NACA 4-digit section, each LE -> TE.

    Uses the standard definition (Abbott and von Doenhoff, 1959). The standard
    x^4 coefficient (-0.1015) leaves a 0.25% chord gap at the trailing edge;
    closed_te (the default) uses -0.1036, which closes it.
    """
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    x = cosine_spacing(n_points)
    a4 = -0.1036 if closed_te else -0.1015
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 + a4 * x**4)
    yc, dyc = naca4_camber(x, m, p)
    theta = np.arctan(dyc)
    xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)
    return xu, yu, xl, yl


def naca4(code: str, n_points: int = 81, closed_te: bool = True):
    """NACA 4-digit coordinates in Selig order."""
    return to_selig(*naca4_surfaces(code, n_points, closed_te))


class CSTAirfoil:
    """
    Class-Shape Transformation (Kulfan, 2008).

    y(x) = C(x) * S(x) +/- x * dte / 2, with class function
    C(x) = x^n1 (1 - x)^n2 and S(x) a Bernstein polynomial whose weights are
    the design variables. n1 = 0.5, n2 = 1 gives a round nose and sharp tail.
    Upper weights are usually positive, lower weights negative.
    """

    def __init__(self, au, al, dte: float = 0.0, n1: float = 0.5, n2: float = 1.0):
        self.au = np.asarray(au, dtype=float)
        self.al = np.asarray(al, dtype=float)
        self.dte = dte
        self.n1 = n1
        self.n2 = n2

    @property
    def weights(self) -> np.ndarray:
        return np.concatenate([self.au, self.al])

    @classmethod
    def from_weights(cls, w, dte: float = 0.0) -> "CSTAirfoil":
        w = np.asarray(w, dtype=float)
        half = len(w) // 2
        return cls(w[:half], w[half:], dte)

    def _basis(self, x: np.ndarray, n_weights: int) -> np.ndarray:
        """Class function times Bernstein basis, shape (len(x), n_weights)."""
        n = n_weights - 1
        c = x**self.n1 * (1 - x)**self.n2
        b = np.stack([comb(n, i) * x**i * (1 - x)**(n - i) for i in range(n + 1)], axis=1)
        return c[:, None] * b

    def surfaces(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        yu = self._basis(x, len(self.au)) @ self.au + x * self.dte / 2
        yl = self._basis(x, len(self.al)) @ self.al - x * self.dte / 2
        return yu, yl

    def coordinates(self, n_points: int = 81) -> Tuple[np.ndarray, np.ndarray]:
        x = cosine_spacing(n_points)
        yu, yl = self.surfaces(x)
        return to_selig(x, yu, x, yl)


def fit_cst(xu, yu, xl, yl, order: int = 5) -> CSTAirfoil:
    """Least-squares CST fit to given surfaces. dte is taken from the data."""
    dte = float(yu[-1] - yl[-1])
    probe = CSTAirfoil(np.zeros(order + 1), np.zeros(order + 1), dte)
    # a cambered NACA nose can put a point or two just ahead of x = 0
    mu, ml = (xu > 0) & (xu <= 1), (xl > 0) & (xl <= 1)
    au, *_ = np.linalg.lstsq(probe._basis(xu[mu], order + 1),
                             yu[mu] - xu[mu] * dte / 2, rcond=None)
    al, *_ = np.linalg.lstsq(probe._basis(xl[ml], order + 1),
                             yl[ml] + xl[ml] * dte / 2, rcond=None)
    return CSTAirfoil(au, al, dte)


def max_thickness(x: np.ndarray, y: np.ndarray) -> float:
    """Max thickness of a Selig-ordered airfoil (vertical distance)."""
    i_le = int(np.argmin(x))
    xu, yu = x[:i_le + 1][::-1], y[:i_le + 1][::-1]
    xl, yl = x[i_le:], y[i_le:]
    xs = cosine_spacing(201)
    return float(np.max(np.interp(xs, xu, yu) - np.interp(xs, xl, yl)))


def karman_trefftz(mu_x=-0.1, mu_y=0.1, te_angle_deg=10.0, n=161):
    """Karman-Trefftz airfoil at unit chord, with its exact potential-flow lift.

    The map tends to z = zeta far away, so the circulation is that of the
    circle with the Kutta condition: cl = 8 pi R sin(alpha + beta) / chord.
    """
    centre = complex(mu_x, mu_y)
    radius = abs(1 - centre)
    beta = np.arcsin(mu_y / radius)
    k = 2 - np.radians(te_angle_deg) / np.pi
    phi = np.linspace(-beta, -beta + 2 * np.pi, n)
    zeta = centre + radius * np.exp(1j * phi)
    ratio = (zeta[1:-1] - 1) / (zeta[1:-1] + 1)
    w = np.abs(ratio) ** k * np.exp(1j * k * np.unwrap(np.angle(ratio)))
    z = np.concatenate([[k], k * (1 + w) / (1 - w), [k]])
    chord = z.real.max() - z.real.min()
    z = (z - z.real.min()) / chord

    def exact_cl(alpha_deg):
        return 8 * np.pi * radius / chord * np.sin(np.radians(alpha_deg) + beta)
    return z.real, z.imag, exact_cl
