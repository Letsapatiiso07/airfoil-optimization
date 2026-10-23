import numpy as np

from optimization.algorithms import GeneticAlgorithm, OptimizationConfig
from optimization.problem import DesignProblem


def test_ga_finds_maximum_of_simple_function():
    target = np.array([0.3, -0.2, 0.7, 0.1, -0.5])
    cfg = OptimizationConfig(bounds=[(-1, 1)] * 5, max_iterations=80,
                             population_size=40, seed=1)
    best, fit = GeneticAlgorithm(cfg).optimize(lambda P: -np.sum((P - target)**2, axis=1))
    assert np.max(np.abs(best - target)) < 0.05


def test_baseline_meets_design_limits():
    p = DesignProblem()
    w0 = p.baseline.weights
    g = p.geometry(w0)
    assert abs(g['t_max'][0] - 0.12) < 0.002
    assert not g['crosses'][0]
    r = p.evaluate(w0)
    assert abs(r['cl'] - p.cl_design) < 1e-9
    assert r['cm'] > p.cm_min and not r['separated']


def test_geometry_check_flags_crossing_surfaces():
    p = DesignProblem()
    good = np.array([0.2] * 6 + [-0.1] * 6)
    bad = np.array([0.2] * 5 + [-0.3] + [-0.1] * 6)
    crosses = p.geometry(np.vstack([good, bad]))['crosses']
    assert not crosses[0] and crosses[1]
