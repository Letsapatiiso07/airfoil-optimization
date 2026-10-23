import numpy as np

from simulation.aerodynamics import AirfoilSimulator, PanelMethod
from utils.parameterization import (fit_cst, karman_trefftz, naca4, naca4_surfaces,
                                    thin_airfoil_alpha_l0)


def test_lift_matches_exact_karman_trefftz_solution():
    x, y, exact_cl = karman_trefftz()
    pm = PanelMethod(x, y)
    for alpha in (0.0, 5.0, 10.0):
        assert abs(pm.cl(alpha) / exact_cl(alpha) - 1) < 0.002


def test_symmetric_airfoil_has_no_lift_at_zero_angle():
    pm = PanelMethod(*naca4('0012'))
    assert abs(pm.cl(0.0)) < 1e-8
    assert abs(pm.loads(0.0)['cm']) < 1e-8


def test_thin_airfoil_lift_slope_is_two_pi():
    pm = PanelMethod(*naca4('0001', n_points=121))
    slope = (pm.cl(1.0) - pm.cl(-1.0)) / np.radians(2.0)
    assert abs(slope / (2 * np.pi) - 1) < 0.02


def test_cambered_zero_lift_angle_matches_thin_airfoil_theory():
    alpha_l0_panel = PanelMethod(*naca4('2412')).alpha_for_cl(0.0)
    assert alpha_l0_panel < 0
    assert abs(alpha_l0_panel - thin_airfoil_alpha_l0(0.02, 0.4)) < 0.3


def test_pressure_integral_matches_circulation_lift():
    pm = PanelMethod(*naca4('2412'))
    assert abs(pm.loads(4.0)['cl_pressure'] - pm.cl(4.0)) < 0.01


def test_drag_estimate_is_plausible_and_trip_adds_drag():
    x, y = naca4('0012')
    free = AirfoilSimulator().analyze(x, y, 0.0, 6e6)
    tripped = AirfoilSimulator(x_trip=0.05).analyze(x, y, 0.0, 6e6)
    assert 0.004 < free['cd'] < 0.008
    assert tripped['cd'] > free['cd']
    assert not free['separated']


def test_cst_fits_naca2412():
    xu, yu, xl, yl = naca4_surfaces('2412', n_points=201)
    cst = fit_cst(xu, yu, xl, yl, order=5)
    err = np.concatenate([cst.surfaces(xu[xu > 0])[0] - yu[xu > 0],
                          cst.surfaces(xl)[1] - yl])
    assert np.sqrt(np.mean(err**2)) < 5e-4
    aft = xu > 0.05
    assert np.max(np.abs(cst.surfaces(xu[aft])[0] - yu[aft])) < 1e-3
