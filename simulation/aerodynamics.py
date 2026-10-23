"""
Airfoil aerodynamics: a linear-strength vortex panel method for lift, moment
and pressure, plus an integral boundary-layer estimate of profile drag.

The panel method is inviscid and incompressible. It has no stall: lift keeps
rising with angle of attack. The drag model (simulation/boundary_layer.py)
is an approximation that does not feed back into the pressure field.
"""

from typing import Dict, Optional

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from simulation.boundary_layer import march_surface, squire_young


class PanelMethod:
    """
    Linear-strength vortex panels with Neumann (flow tangency) boundary
    conditions and a Kutta condition, after Katz and Plotkin, Low-Speed
    Aerodynamics, 2nd ed., chapter 11.

    Input is a closed loop in Selig order (upper TE -> LE -> lower TE),
    chord 1. Unknowns are the vortex strengths gamma at the N + 1 nodes.
    The flow inside the body is at rest, so the outer surface speed is
    |gamma|. By linearity gamma(alpha) = cos(alpha) g0 + sin(alpha) g90,
    so each geometry needs one factorisation and two solves.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        # the Kutta condition below assumes a sharp trailing edge, so close
        # any small gap at its midpoint
        for v in (self.x, self.y):
            v[0] = v[-1] = 0.5 * (v[0] + v[-1])
        dx, dy = np.diff(self.x), np.diff(self.y)
        self.length = np.hypot(dx, dy)
        self.cos = dx / self.length
        self.sin = dy / self.length
        # outward normal for the counterclockwise Selig loop
        self.nx, self.ny = self.sin, -self.cos
        self.xc = 0.5 * (self.x[:-1] + self.x[1:])
        self.yc = 0.5 * (self.y[:-1] + self.y[1:])
        self.arc = np.concatenate([[0.0], np.cumsum(self.length)])

        lu = lu_factor(self._influence_matrix())
        n = len(self.xc)
        rhs = np.zeros((n + 1, 2))
        rhs[:n, 0] = -self.nx      # alpha = 0
        rhs[:n, 1] = -self.ny      # alpha = 90 deg
        self.g0, self.g90 = lu_solve(lu, rhs).T
        self.cl0 = self._lift(self.g0)
        self.cl90 = self._lift(self.g90)

    def _influence_matrix(self) -> np.ndarray:
        n = len(self.xc)
        S, c, s = self.length, self.cos, self.sin
        dx = self.xc[:, None] - self.x[None, :-1]
        dy = self.yc[:, None] - self.y[None, :-1]
        # collocation points in each panel's frame (xl along panel, zl normal)
        xl = dx * c + dy * s
        zl = -dx * s + dy * c
        np.fill_diagonal(zl, 0.0)
        r1 = np.hypot(xl, zl)
        r2 = np.hypot(xl - S, zl)
        dth = np.arctan2(zl, xl - S) - np.arctan2(zl, xl)
        log_r = np.log(r1 / r2)
        k = 0.5 / np.pi
        # velocity from unit strength at the panel's first (a) and second (b) node
        ua = k * ((1 - xl / S) * dth + zl * log_r / S)
        ub = k * (xl * dth - zl * log_r) / S
        wa = -k * ((1 - xl / S) * log_r + 1 - zl * dth / S)
        wb = -k * (xl * log_r / S - 1 + zl * dth / S)
        nx, ny = self.nx[:, None], self.ny[:, None]
        an_a = (ua * c - wa * s) * nx + (ua * s + wa * c) * ny
        an_b = (ub * c - wb * s) * nx + (ub * s + wb * c) * ny

        a = np.zeros((n + 1, n + 1))
        a[:n, :n] += an_a
        a[:n, 1:] += an_b
        a[n, 0] = a[n, n] = 1.0     # Kutta: equal speeds leaving the TE
        return a

    def _lift(self, gamma: np.ndarray) -> float:
        # Kutta-Joukowski: cl = 2 * circulation / (V c)
        return float(2.0 * np.sum(0.5 * (gamma[:-1] + gamma[1:]) * self.length))

    def gamma(self, alpha_deg: float) -> np.ndarray:
        a = np.radians(alpha_deg)
        return np.cos(a) * self.g0 + np.sin(a) * self.g90

    def cl(self, alpha_deg: float) -> float:
        a = np.radians(alpha_deg)
        return np.cos(a) * self.cl0 + np.sin(a) * self.cl90

    def alpha_for_cl(self, cl: float) -> float:
        """Angle of attack (deg) that gives the target lift coefficient."""
        r = np.hypot(self.cl0, self.cl90)
        phi = np.arctan2(self.cl90, self.cl0)
        return float(np.degrees(phi - np.arccos(np.clip(cl / r, -1.0, 1.0))))

    def loads(self, alpha_deg: float, x_ref: float = 0.25) -> Dict:
        """Pressure, and lift and quarter-chord moment from pressure integration.

        The vortex strength at the two trailing-edge nodes is not a usable
        surface speed (it does not converge with panel count, unlike cl), so
        cp there is returned as NaN.
        """
        g = self.gamma(alpha_deg)
        cp_nodes = 1.0 - g**2
        cp_nodes[[0, -1]] = np.nan
        cp_panel = 1.0 - (0.5 * (g[:-1] + g[1:]))**2
        fx = -cp_panel * self.nx * self.length
        fy = -cp_panel * self.ny * self.length
        a = np.radians(alpha_deg)
        cl_p = float(np.sum(fy) * np.cos(a) - np.sum(fx) * np.sin(a))
        # nose-up positive
        cm = float(np.sum(self.yc * fx - (self.xc - x_ref) * fy))
        return {'gamma': g, 'cp': cp_nodes, 'cl_pressure': cl_p, 'cm': cm}


class AirfoilSimulator:
    """
    Panel method plus boundary-layer drag estimate for one airfoil.

    cl and cm are inviscid. cd is the Squire-Young profile drag from the
    integral boundary layer. There is no stall model; 'separated' flags when
    the turbulent boundary layer is predicted to separate before the TE, in
    which case cd and the inviscid cl are not to be trusted.
    """

    def __init__(self, x_trip: float = 1.0, x_end: float = 0.98):
        self.x_trip = x_trip
        # The inviscid speed falls to a stagnation point at a sharp trailing
        # edge, which a real boundary layer never sees. Marching into it gives
        # false separation, so the march (and Squire-Young) stops at x_end.
        # Moving x_end from 0.98 to 0.99 changes NACA 0012 cd by about 1%.
        self.x_end = x_end
        self._panels: Optional[PanelMethod] = None
        self._key = None

    def panels(self, x: np.ndarray, y: np.ndarray) -> PanelMethod:
        key = hash((np.asarray(x, float).tobytes(), np.asarray(y, float).tobytes()))
        if self._panels is None or self._key != key:
            self._panels = PanelMethod(x, y)
            self._key = key
        return self._panels

    def analyze(self, x: np.ndarray, y: np.ndarray, alpha: float,
                reynolds: float, mach: float = 0.0) -> Dict:
        """Analyse at angle of attack alpha (deg). mach is ignored (incompressible)."""
        pm = self.panels(x, y)
        loads = pm.loads(alpha)
        g = loads['gamma']
        cl = pm.cl(alpha)
        bl = self._boundary_layer(pm, g, reynolds)
        cd = bl['cd']
        return {
            'alpha': float(alpha),
            'cl': float(cl),
            'cd': cd,
            'cm': loads['cm'],
            'ld_ratio': float(cl / cd) if cd > 0 else 0.0,
            'cp_distribution': loads['cp'],
            'x_transition_upper': bl['upper']['x_transition'],
            'x_transition_lower': bl['lower']['x_transition'],
            'separated': bl['upper']['separated'] or bl['lower']['separated'],
            'x_separation': bl['upper']['x_separation'] or bl['lower']['x_separation'],
        }

    def analyze_at_cl(self, x, y, cl: float, reynolds: float) -> Dict:
        """Analyse at the angle of attack that gives the target cl."""
        alpha = self.panels(x, y).alpha_for_cl(cl)
        return self.analyze(x, y, alpha, reynolds)

    def _boundary_layer(self, pm: PanelMethod, g: np.ndarray, reynolds: float) -> Dict:
        # stagnation point: where gamma changes sign from + (upper) to - (lower),
        # taking the change nearest the leading edge
        changes = np.flatnonzero((g[:-1] > 0) & (g[1:] <= 0))
        if len(changes) == 0:
            changes = np.array([int(np.argmin(pm.x))])
        k = int(changes[np.argmin(pm.x[changes])])
        f = g[k] / (g[k] - g[k + 1]) if g[k] != g[k + 1] else 0.0
        s_stag = pm.arc[k] + f * (pm.arc[k + 1] - pm.arc[k])
        x_stag = pm.x[k] + f * (pm.x[k + 1] - pm.x[k])

        up = np.arange(k, -1, -1)
        lo = np.arange(k + 1, len(g))
        result = {}
        for name, idx, sign in (('upper', up, 1.0), ('lower', lo, -1.0)):
            s = np.concatenate([[0.0], np.abs(pm.arc[idx] - s_stag)])
            ue = np.concatenate([[0.0], sign * g[idx]])
            xs = np.concatenate([[x_stag], pm.x[idx]])
            keep = np.concatenate([[True], (np.diff(s) > 1e-10) & (xs[1:] <= self.x_end)])
            result[name] = march_surface(s[keep], ue[keep], xs[keep], reynolds, self.x_trip)
        result['cd'] = float(sum(squire_young(result[n]['theta_te'], result[n]['h_te'],
                                              result[n]['ue_te']) for n in ('upper', 'lower')))
        return result

    def sweep_analysis(self, x: np.ndarray, y: np.ndarray, alphas: np.ndarray,
                       reynolds: float, mach: float = 0.0) -> Dict:
        """Run analyze over a range of angles; returns arrays."""
        rows = [self.analyze(x, y, a, reynolds) for a in alphas]
        out = {'alpha': np.asarray(alphas, dtype=float)}
        for key in ('cl', 'cd', 'cm', 'ld_ratio', 'separated',
                    'x_transition_upper', 'x_transition_lower'):
            out[key] = np.array([r[key] for r in rows])
        return out


class FlightConditions:
    """International Standard Atmosphere (troposphere) and chord Reynolds number."""

    def __init__(self, altitude: float = 0.0, velocity: float = 50.0, chord: float = 1.0):
        self.altitude = altitude
        self.velocity = velocity
        self.chord = chord
        self.temperature = 288.15 - 0.0065 * min(altitude, 11000.0)
        self.pressure = 101325.0 * (self.temperature / 288.15) ** 5.2559
        self.density = self.pressure / (287.05 * self.temperature)
        # Sutherland's law
        self.viscosity = (1.716e-5 * (self.temperature / 273.15) ** 1.5
                          * (273.15 + 110.4) / (self.temperature + 110.4))
        self.reynolds = self.density * velocity * chord / self.viscosity
        self.mach = velocity / np.sqrt(1.4 * 287.05 * self.temperature)
