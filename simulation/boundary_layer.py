"""
Integral boundary-layer drag estimate.

This is an approximation, not a viscous solver. It takes the inviscid edge
speed from the panel method (no coupling back to the outer flow) and marches:

- laminar: Thwaites' method, H from the Cebeci-Bradshaw fit to lambda
- transition: Michel's criterion, laminar separation (lambda < -0.09) or a
  forced trip location, whichever comes first
- turbulent: Head's entrainment method with Ludwieg-Tillmann skin friction;
  separation is flagged when H exceeds h_sep
- drag: Squire-Young formula at the trailing edge

All lengths are in chords and speeds in freestream units, so nu = 1 / Re.
"""

import math
from typing import Dict

import numpy as np

H_SEPARATION = 2.4


def _h1_of_h(h: float) -> float:
    if h <= 1.6:
        return 3.3 + 0.8234 * (h - 1.1) ** -1.287
    return 3.3 + 1.5501 * (h - 0.6778) ** -3.064


def _h_of_h1(h1: float) -> float:
    h1 = max(h1, 3.3 + 1e-6)
    if h1 >= 5.3:
        return 1.1 + ((h1 - 3.3) / 0.8234) ** (-1 / 1.287)
    return 0.6778 + ((h1 - 3.3) / 1.5501) ** (-1 / 3.064)


def _thwaites_h(lam: np.ndarray) -> np.ndarray:
    lam = np.clip(lam, -0.09, 0.25)
    return np.where(lam >= 0,
                    2.61 - 3.75 * lam + 5.24 * lam**2,
                    2.088 + 0.0731 / (lam + 0.14))


def _head_rates(theta, y, ue, due, nu):
    h1 = y / (ue * theta)
    h = _h_of_h1(h1)
    re_theta = max(ue * theta / nu, 1.0)
    cf = 0.246 * 10 ** (-0.678 * h) * re_theta ** -0.268
    dtheta = 0.5 * cf - (h + 2.0) * theta / ue * due
    dy = ue * 0.0306 * (max(h1, 3.0 + 1e-6) - 3.0) ** -0.6169
    return dtheta, dy, h


def march_surface(s: np.ndarray, ue: np.ndarray, x: np.ndarray, reynolds: float,
                  x_trip: float = 1.0, h_sep: float = H_SEPARATION) -> Dict:
    """
    March one surface from the stagnation point (s[0] = 0, ue[0] = 0) to the
    trailing edge. x holds x/c of each station, used for the trip location.
    """
    nu = 1.0 / reynolds
    ue = np.maximum(ue, 1e-8)
    ue[0] = 0.0
    due = np.gradient(ue, s)

    # Thwaites: theta^2 = 0.45 nu / ue^6 * integral(ue^5 ds)
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (ue[1:]**5 + ue[:-1]**5) * np.diff(s))])
    theta2 = np.empty_like(s)
    theta2[1:] = 0.45 * nu * integral[1:] / ue[1:]**6
    theta2[0] = 0.075 * nu / max(due[0], 1e-8)
    lam = theta2 * due / nu
    re_theta = ue * np.sqrt(theta2) / nu
    re_x = np.maximum(ue * s / nu, 1.0)
    # each criterion as a margin that turns positive where it is met
    margins = [re_theta - 1.174 * (1 + 22400 / re_x) * re_x**0.46,   # Michel
               -0.09 - lam,                                         # laminar separation
               x - x_trip]                                          # trip
    met = np.any([m >= 0 for m in margins], axis=0)
    met[0] = False
    hits = np.flatnonzero(met)

    out = {'x_transition': 1.0, 'laminar_separation': False,
           'separated': False, 'x_separation': None, 'h_max': 0.0}

    if len(hits) == 0:
        # laminar all the way to the trailing edge
        theta = math.sqrt(theta2[-1])
        h = float(_thwaites_h(lam[-1:])[0])
        out.update(theta_te=theta, h_te=h, ue_te=float(ue[-1]))
        out['h_max'] = h
        return out

    # place transition between stations i0 - 1 and i0 where the first
    # criterion crosses zero, so cd varies smoothly with the shape
    i0 = int(hits[0])
    fracs = [m[i0 - 1] / (m[i0 - 1] - m[i0]) for m in margins if m[i0] >= 0 > m[i0 - 1]]
    f = float(np.clip(min(fracs), 0.0, 1.0)) if fracs else 1.0
    lerp = lambda v: float(v[i0 - 1] + f * (v[i0] - v[i0 - 1]))
    out['x_transition'] = lerp(x)
    out['laminar_separation'] = bool(margins[1][i0] >= 0 and margins[0][i0] < 0)

    st = np.concatenate([[lerp(s)], s[i0:]])
    ut = np.concatenate([[max(lerp(ue), 1e-8)], ue[i0:]])
    dut = np.concatenate([[lerp(due)], due[i0:]])
    xt = np.concatenate([[lerp(x)], x[i0:]])
    theta = math.sqrt(lerp(theta2))
    h = 1.4
    y = ut[0] * theta * _h1_of_h(h)
    h_max = h
    for i in range(len(st) - 1):
        ds = st[i + 1] - st[i]
        if ds <= 0:
            continue
        # Heun (RK2) step
        dt1, dy1, _ = _head_rates(theta, y, ut[i], dut[i], nu)
        th_p = max(theta + ds * dt1, 1e-12)
        y_p = y + ds * dy1
        dt2, dy2, _ = _head_rates(th_p, y_p, ut[i + 1], dut[i + 1], nu)
        theta = max(theta + 0.5 * ds * (dt1 + dt2), 1e-12)
        y = y + 0.5 * ds * (dy1 + dy2)
        h = _h_of_h1(y / (ut[i + 1] * theta))
        h_max = max(h_max, h)
        if h >= h_sep:
            out['separated'] = True
            out['x_separation'] = float(xt[i + 1])
            out.update(theta_te=theta, h_te=h, ue_te=float(ut[i + 1]), h_max=h_max)
            return out

    out.update(theta_te=theta, h_te=h, ue_te=float(ut[-1]), h_max=h_max)
    return out


def squire_young(theta_te: float, h_te: float, ue_te: float) -> float:
    """Profile drag of one surface from its trailing-edge momentum thickness."""
    return 2.0 * theta_te * ue_te ** ((h_te + 5.0) / 2.0)
