"""
Surrogate-assisted airfoil optimisation, end to end.

  python main_pipeline.py            # full run, a few minutes on 2 cores
  python main_pipeline.py --quick    # small smoke test

1. Sample CST shapes around NACA 2412 and score each with the panel method
   and boundary-layer drag estimate at the design lift coefficient.
2. Train surrogates on 70% of the shapes, pick one on a 15% validation set,
   report accuracy on the last 15% (shapes no model has seen).
3. Time the solver against the surrogate.
4. Run a genetic algorithm on the surrogate, then re-check its best
   candidates with the real solver.
5. Run the same GA directly on the solver, for comparison.
6. Compare baseline and optimised shapes and write docs/results.json and
   the figures in docs/.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np  # noqa: E402

from models.surrogate import TARGETS, SurrogateModel, regression_metrics  # noqa: E402
from optimization.algorithms import GeneticAlgorithm, OptimizationConfig  # noqa: E402
from optimization.problem import N_POINTS, DesignProblem  # noqa: E402
from simulation.aerodynamics import AirfoilSimulator, FlightConditions  # noqa: E402
from utils.parameterization import naca4  # noqa: E402
from visualization import style  # noqa: E402

ROOT = Path(__file__).parent
DOCS = ROOT / 'docs'
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
N_JOBS = 2


def split(n, seed, frac=(0.70, 0.15)):
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def timed(fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t


def train_surrogates(X, y, tr, va, te, seed):
    fitted, report = {}, {}
    for mt in ('ridge', 'rf', 'xgboost', 'nn'):
        m, secs = timed(SurrogateModel(mt, seed).fit, X[tr], y[tr], X[va], y[va])
        fitted[mt] = m
        report[mt] = {'train_seconds': secs}
    # the ensemble averages the three non-linear models already trained
    ens = SurrogateModel('ensemble', seed)
    ens.scaler_X, ens.scaler_y = fitted['nn'].scaler_X, fitted['nn'].scaler_y
    ens.models = {k: fitted[k].models[k] for k in ('rf', 'xgboost', 'nn')}
    fitted['ensemble'] = ens
    report['ensemble'] = {'train_seconds': sum(report[k]['train_seconds']
                                               for k in ('rf', 'xgboost', 'nn'))}
    for mt, m in fitted.items():
        report[mt]['val'] = regression_metrics(y[va], m.predict(X[va]))
        report[mt]['test'] = regression_metrics(y[te], m.predict(X[te]))
    # choose on validation error only; the test set stays untouched
    best = min(fitted, key=lambda k: report[k]['val']['cd']['mean_abs_pct'])
    return fitted, report, best


def time_calls(problem, model, W, pop_size):
    t = time.perf_counter()
    for w in W:
        problem.evaluate(w)
    solver_ms = 1000 * (time.perf_counter() - t) / len(W)
    reps = 200
    t = time.perf_counter()
    for i in range(reps):
        model.predict(W[i % len(W)][None, :])
    single_ms = 1000 * (time.perf_counter() - t) / reps
    batch = W[:pop_size]
    t = time.perf_counter()
    for _ in range(20):
        model.predict(batch)
    batch_ms = 1000 * (time.perf_counter() - t) / (20 * len(batch))
    return {'solver_ms_per_shape': solver_ms,
            'surrogate_ms_per_shape_single_call': single_ms,
            'surrogate_ms_per_shape_in_batch_of_%d' % len(batch): batch_ms,
            'speedup_single_call': solver_ms / single_ms,
            'speedup_batched': solver_ms / batch_ms}


def describe(problem, w, label):
    """True-solver results at the design point, free and tripped transition."""
    free = problem.evaluate(w)
    trip = problem.evaluate(w, x_trip=0.05)
    g = problem.geometry(w)
    return {'label': label, 'weights': [float(v) for v in w],
            'alpha_deg': free['alpha'], 'cd': free['cd'], 'cm': free['cm'],
            'ld': free['ld_ratio'], 'separated': bool(free['separated']),
            'x_transition_upper': free['x_transition_upper'],
            'x_transition_lower': free['x_transition_lower'],
            'cd_tripped_5pct': trip['cd'], 'ld_tripped_5pct': trip['ld_ratio'],
            't_max': float(g['t_max'][0]), 't_90': float(g['t_90'][0]),
            'surfaces_cross': bool(g['crosses'][0]),
            # CST: leading-edge radius of each surface is A0^2 / 2
            'le_radius_min': float(min(abs(w[0]), abs(w[len(w) // 2])) ** 2 / 2),
            'meets_constraints': bool(problem.geometry_violation(w)[0] == 0
                                      and free['cm'] >= problem.cm_min
                                      and not free['separated'])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--samples', type=int, default=2500)
    ap.add_argument('--population', type=int, default=60)
    ap.add_argument('--generations', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quick', action='store_true', help='300 shapes, 20 generations')
    args = ap.parse_args()
    if args.quick:
        args.samples, args.generations = 300, 20
    for d in (DOCS, DATA, MODELS):
        d.mkdir(exist_ok=True)

    flight = FlightConditions(altitude=1000, velocity=60, chord=1.0)
    problem = DesignProblem(cl_design=0.5, reynolds=float(round(flight.reynolds, -4)))
    w0 = problem.baseline.weights
    print(f"Design point: cl = {problem.cl_design}, Re = {problem.reynolds:.3g} "
          f"(1000 m, 60 m/s, 1 m chord), Mach {flight.mach:.2f} treated as incompressible")
    print(f"Constraints: t/c >= {problem.t_min}, thickness at 90% chord >= {problem.t90_min}, "
          f"cm >= {problem.cm_min}, no turbulent separation; weights within "
          f"+/-{problem.half_width} of the NACA 2412 fit")

    # 1. database
    print(f"\n[1] Scoring {args.samples} sampled shapes with the solver ...")
    W = problem.sample(args.samples, seed=args.seed)
    R, t_db = timed(problem.evaluate_many, W, N_JOBS)
    np.savez_compressed(DATA / 'database.npz', weights=W, **R)
    attached = ~R['separated']
    X, y = W[attached], np.column_stack([R[k] for k in TARGETS])[attached]
    print(f"    {len(W)} shapes in {t_db:.1f} s; {int((~attached).sum())} separated "
          f"shapes left out of training")

    # 2. surrogates
    print("\n[2] Training surrogates (70% train / 15% validation / 15% test, split by shape)")
    tr, va, te = split(len(X), args.seed)
    fitted, sur_report, best = train_surrogates(X, y, tr, va, te, args.seed)
    model = fitted[best]
    model.save(str(MODELS / 'surrogate_model.pkl'))
    print(f"    {'model':10s} {'train s':>8s} {'val cd err%':>12s} {'test cd err%':>13s} "
          f"{'test cd R2':>11s} {'test alpha MAE':>15s} {'test cm MAE':>12s}")
    for k, r in sur_report.items():
        print(f"    {k:10s} {r['train_seconds']:8.1f} {r['val']['cd']['mean_abs_pct']:12.2f} "
              f"{r['test']['cd']['mean_abs_pct']:13.2f} {r['test']['cd']['r2']:11.4f} "
              f"{r['test']['alpha']['mae']:15.4f} {r['test']['cm']['mae']:12.5f}")
    print(f"    chosen on validation error: {best}")

    # 3. timing
    timing = time_calls(problem, model, X[te][:100], args.population)
    print(f"\n[3] Solver {timing['solver_ms_per_shape']:.2f} ms per shape; surrogate "
          f"{timing['surrogate_ms_per_shape_single_call']:.3f} ms (one shape per call), "
          f"speedup x{timing['speedup_single_call']:.0f} single, "
          f"x{timing['speedup_batched']:.0f} batched")

    cfg = OptimizationConfig(bounds=problem.bounds, max_iterations=args.generations,
                             population_size=args.population, seed=args.seed)

    # 4. GA on the surrogate, then check the best candidates with the solver
    def surrogate_fitness(P):
        pred = model.predict(P)
        return problem.fitness(P, pred[:, 0], pred[:, 2])

    ga_s = GeneticAlgorithm(cfg)
    (w_s, f_s), t_ga_s = timed(ga_s.optimize, surrogate_fitness, initial=w0)
    order = np.argsort(ga_s.fitness)[::-1]
    _, first = np.unique(np.round(ga_s.population[order], 6), axis=0, return_index=True)
    cands = ga_s.population[order][np.sort(first)][:20]
    check, t_check = timed(problem.evaluate_many, cands, N_JOBS)
    true_fit = problem.fitness(cands, check['cd'], check['cm'], check['separated'])
    w_sur = cands[int(np.argmax(true_fit))]
    pred_sur = model.predict(w_sur[None, :])[0]
    print(f"\n[4] Surrogate GA: {ga_s.n_evaluations} surrogate calls in {t_ga_s:.1f} s, "
          f"{len(cands)} candidates re-checked with the solver in {t_check:.1f} s")

    # 5. GA directly on the solver
    def solver_fitness(P):
        r = problem.evaluate_many(P, N_JOBS)
        return problem.fitness(P, r['cd'], r['cm'], r['separated'])

    ga_d = GeneticAlgorithm(cfg)
    (w_dir, _), t_ga_d = timed(ga_d.optimize, solver_fitness, initial=w0)
    print(f"[5] Direct GA: {ga_d.n_evaluations} solver calls in {t_ga_d:.1f} s")

    # 6. compare
    xb, yb = naca4('2412', N_POINTS)
    naca = AirfoilSimulator().analyze_at_cl(xb, yb, problem.cl_design, problem.reynolds)
    rows = [describe(problem, w0, 'Baseline: NACA 2412 (CST fit)'),
            describe(problem, w_sur, 'Optimised: surrogate GA'),
            describe(problem, w_dir, 'Optimised: direct GA')]
    base = rows[0]
    for r in rows:
        r['ld_gain_pct'] = 100 * (r['ld'] / base['ld'] - 1)
        r['ld_gain_tripped_pct'] = 100 * (r['ld_tripped_5pct'] / base['ld_tripped_5pct'] - 1)
    rows[1]['surrogate_predicted_ld'] = float(problem.cl_design / pred_sur[0])

    end_to_end = t_db + sum(sur_report[k]['train_seconds'] for k in ('ridge', 'rf', 'xgboost', 'nn'))
    end_to_end += t_ga_s + t_check
    summary = {
        'design_point': {'cl': problem.cl_design, 'reynolds': problem.reynolds,
                         'mach': float(flight.mach), 'altitude_m': 1000, 'velocity_m_s': 60,
                         'chord_m': 1.0},
        'constraints': {'t_min': problem.t_min, 't90_min': problem.t90_min,
                        'cm_min': problem.cm_min, 'weight_half_width': problem.half_width},
        'settings': vars(args),
        'database': {'shapes': int(len(W)), 'separated': int((~attached).sum()),
                     'seconds': t_db, 'n_jobs': N_JOBS},
        'naca2412_exact_at_design_cl': {'cd': naca['cd'], 'ld': naca['ld_ratio'],
                                        'cm': naca['cm'], 'alpha_deg': naca['alpha']},
        'surrogates': sur_report,
        'chosen_surrogate': best,
        'timing': timing,
        'optimisation': {
            'surrogate_ga_seconds': t_ga_s, 'surrogate_ga_calls': ga_s.n_evaluations,
            'solver_checks': len(cands), 'solver_check_seconds': t_check,
            'surrogate_route_total_seconds': end_to_end,
            'direct_ga_seconds': t_ga_d, 'direct_ga_solver_calls': ga_d.n_evaluations,
        },
        'shapes': rows,
    }
    (DOCS / 'results.json').write_text(json.dumps(summary, indent=2))
    with open(DOCS / 'optimised_airfoil.dat', 'w') as f:
        f.write('Optimised airfoil (surrogate GA), Selig order\n')
        for xi, yi in zip(*problem.airfoil(w_sur).coordinates(N_POINTS)):
            f.write(f"{xi:10.6f} {yi:10.6f}\n")

    print(f"\n[6] Results at cl = {problem.cl_design}, Re = {problem.reynolds:.3g} "
          f"(true solver, free transition)")
    print(f"    NACA 2412 exact geometry: L/D {naca['ld_ratio']:.1f}, cd {naca['cd']:.5f}")
    print(f"    {'shape':32s} {'L/D':>6s} {'gain':>7s} {'cd':>8s} {'cm':>7s} {'t/c':>6s} "
          f"{'xtr up':>6s} {'L/D trip':>8s} {'gain trip':>9s} ok")
    for r in rows:
        print(f"    {r['label']:32s} {r['ld']:6.1f} {r['ld_gain_pct']:+6.1f}% {r['cd']:8.5f} "
              f"{r['cm']:7.4f} {r['t_max']:6.3f} {r['x_transition_upper']:6.2f} "
              f"{r['ld_tripped_5pct']:8.1f} {r['ld_gain_tripped_pct']:+8.1f}% "
              f"{r['meets_constraints']}")
    print(f"    surrogate predicted L/D {rows[1]['surrogate_predicted_ld']:.1f} for its pick, "
          f"solver says {rows[1]['ld']:.1f}")
    print(f"    wall time: surrogate route {end_to_end:.0f} s (database + training + GA + checks), "
          f"direct GA {t_ga_d:.0f} s")

    plot(problem, model, X, y, te, fitted, sur_report, best, rows, ga_s, ga_d)


def plot(problem, model, X, y, te, fitted, sur_report, best, rows, ga_s, ga_d):
    style.apply()
    import matplotlib.pyplot as plt

    colors = [style.ORANGE, style.BLUE, style.AQUA]
    sim = AirfoilSimulator()
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.6),
                           gridspec_kw={'height_ratios': [0.8, 1]})
    for r, c in zip(rows, colors):
        xs, ys = problem.airfoil(r['weights']).coordinates(N_POINTS)
        ax[0, 0].plot(xs, ys, color=c, label=f"{r['label']} (t/c {r['t_max']:.3f})")
        pm = sim.panels(xs, ys)
        cp = pm.loads(r['alpha_deg'])['cp']
        ax[0, 1].plot(xs, cp, color=c, label=r['label'])
        cls = np.linspace(0.0, 1.0, 21)
        lds = [sim.analyze_at_cl(xs, ys, c_, problem.reynolds)['ld_ratio'] for c_ in cls]
        ax[1, 0].plot(cls, lds, color=c, label=r['label'])
        ax[1, 0].plot([problem.cl_design], [r['ld']], marker='o', markersize=7, color=c,
                      markeredgecolor=style.SURFACE, markeredgewidth=1.5)
    ax[0, 0].set_aspect('equal')
    ax[0, 0].set(xlabel='x/c', ylabel='y/c', title='Shapes', ylim=(-0.08, 0.27))
    ax[0, 0].legend(loc='upper right', fontsize=8)
    ax[0, 1].invert_yaxis()
    ax[0, 1].set(xlabel='x/c', ylabel='Cp', title=f'Pressure at the design point, cl = {problem.cl_design}')
    ax[0, 1].legend(loc='lower right', fontsize=8)
    ax[1, 0].axvline(problem.cl_design, color=style.INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax[1, 0].set(xlabel='Lift coefficient cl', ylabel='L/D',
                 title=f'L/D, Re = {problem.reynolds:.2g}, free transition (model)')
    ax[1, 0].legend(loc='lower right', fontsize=8)
    gens = [h['generation'] for h in ga_s.history]
    ax[1, 1].plot(gens, [h['best'] for h in ga_s.history], color=style.BLUE,
                  label='Surrogate GA (predicted L/D)')
    ax[1, 1].plot([h['generation'] for h in ga_d.history], [h['best'] for h in ga_d.history],
                  color=style.AQUA, label='Direct GA (solver L/D)')
    ax[1, 1].axhline(rows[0]['ld'], color=style.ORANGE, linewidth=1.5, linestyle=(0, (4, 3)),
                     label='Baseline')
    ax[1, 1].set(xlabel='Generation', ylabel='Best fitness (L/D minus penalties)',
                 title='GA progress')
    ax[1, 1].legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / 'optimisation.png', dpi=130)
    plt.close(fig)

    nice = {'ridge': 'Ridge (linear baseline)', 'rf': 'Random forest', 'xgboost': 'XGBoost',
            'nn': 'Neural net (MLP)', 'ensemble': 'Ensemble (RF + XGB + NN)'}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    yt = y[te][:, 0]
    for name, c in (('ridge', style.ORANGE), (best, style.BLUE)):
        ax[0].plot(yt, fitted[name].predict(X[te])[:, 0], **style.marker(c),
                   label=f"{nice[name]}, {sur_report[name]['test']['cd']['mean_abs_pct']:.1f}% "
                         f"mean error")
    lim = [yt.min() * 0.95, yt.max() * 1.05]
    ax[0].plot(lim, lim, color=style.INK_2, linewidth=1)
    ax[0].set(xlim=lim, ylim=lim, xlabel='Solver cd', ylabel='Predicted cd',
              title=f'Held-out shapes (n = {len(te)})')
    ax[0].legend(loc='upper left')
    names = list(sur_report)
    errs = [sur_report[n]['test']['cd']['mean_abs_pct'] for n in names]
    ax[1].barh([nice[n] for n in names], errs, height=0.35, color=style.BLUE)
    for i, e in enumerate(errs):
        ax[1].text(e, i, f'  {e:.2f}%', va='center', fontsize=8.5, color=style.INK_2)
    ax[1].invert_yaxis()
    ax[1].set(xlabel='Mean absolute cd error on held-out shapes (%)', title='Surrogate error by model')
    ax[1].grid(axis='y', visible=False)
    ax[1].set_xlim(0, max(errs) * 1.25)
    fig.tight_layout()
    fig.savefig(DOCS / 'surrogate.png', dpi=130)
    plt.close(fig)
    print(f"\nSaved results.json, optimised_airfoil.dat, optimisation.png and surrogate.png "
          f"in {DOCS}")


if __name__ == '__main__':
    main()
