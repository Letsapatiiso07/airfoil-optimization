"""
The design problem: CST weights around a NACA 2412 baseline, scored by L/D
at a fixed design lift coefficient, with thickness, moment and attached-flow
limits.
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import qmc

from simulation.aerodynamics import AirfoilSimulator
from utils.parameterization import CSTAirfoil, cosine_spacing, fit_cst, naca4_surfaces

CST_ORDER = 5
N_POINTS = 81          # points per surface -> 160 panels


def baseline_cst(code: str = '2412') -> CSTAirfoil:
    """Least-squares CST fit of a NACA 4-digit section (closed trailing edge)."""
    return fit_cst(*naca4_surfaces(code, n_points=201), order=CST_ORDER)


@dataclass
class DesignProblem:
    cl_design: float = 0.5
    reynolds: float = 3.8e6
    t_min: float = 0.12            # minimum max-thickness / chord
    t90_min: float = 0.02          # minimum thickness at 90% chord
    cm_min: float = -0.08          # limit on nose-down quarter-chord moment
    half_width: float = 0.05       # +/- range on each CST weight
    baseline: CSTAirfoil = field(default_factory=baseline_cst)

    def __post_init__(self):
        w0 = self.baseline.weights
        self.lower = w0 - self.half_width
        self.upper = w0 + self.half_width
        self._x = cosine_spacing(201)
        self._basis = self.baseline._basis(self._x, CST_ORDER + 1)

    @property
    def bounds(self):
        return list(zip(self.lower, self.upper))

    def airfoil(self, w) -> CSTAirfoil:
        return CSTAirfoil.from_weights(w, self.baseline.dte)

    def geometry(self, W: np.ndarray) -> Dict[str, np.ndarray]:
        """Vectorised thickness checks for a batch of weight vectors."""
        W = np.atleast_2d(W)
        n = CST_ORDER + 1
        t = (W[:, :n] - W[:, n:]) @ self._basis.T
        interior = (self._x > 0) & (self._x < 1)
        return {
            't_max': t.max(axis=1),
            't_90': np.array([np.interp(0.9, self._x, row) for row in t]),
            'crosses': np.any(t[:, interior] <= 0, axis=1),
        }

    def geometry_violation(self, W: np.ndarray) -> np.ndarray:
        g = self.geometry(W)
        return (np.maximum(0, self.t_min - g['t_max'])
                + np.maximum(0, self.t90_min - g['t_90'])
                + g['crosses'].astype(float))

    def fitness(self, W, cd, cm, separated=None, penalty: float = 1e4) -> np.ndarray:
        """L/D at the design cl minus penalties for broken constraints."""
        violation = self.geometry_violation(W) + np.maximum(0, self.cm_min - np.asarray(cm))
        if separated is not None:
            violation = violation + 0.01 * np.asarray(separated, dtype=float)
        return self.cl_design / np.asarray(cd) - penalty * violation

    def evaluate(self, w, x_trip: float = 1.0) -> Dict:
        """Panel method + boundary layer at the design cl."""
        x, y = self.airfoil(w).coordinates(N_POINTS)
        r = AirfoilSimulator(x_trip).analyze_at_cl(x, y, self.cl_design, self.reynolds)
        return {k: r[k] for k in ('alpha', 'cl', 'cd', 'cm', 'ld_ratio', 'separated',
                                  'x_transition_upper', 'x_transition_lower')}

    def evaluate_many(self, W: np.ndarray, n_jobs: int = 2) -> Dict[str, np.ndarray]:
        rows = Parallel(n_jobs=n_jobs, batch_size=32)(delayed(self.evaluate)(w) for w in W)
        return {k: np.array([r[k] for r in rows]) for k in rows[0]}

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        """Latin hypercube sample of weight vectors that pass the crossing check."""
        sampler = qmc.LatinHypercube(d=len(self.lower), seed=seed)
        W = qmc.scale(sampler.random(int(n * 1.3)), self.lower, self.upper)
        W = W[~self.geometry(W)['crosses']]
        return W[:n]
