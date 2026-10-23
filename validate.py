"""
Check the aerodynamic model against theory and wind-tunnel data.

  python validate.py

Downloads four small NACA 0012 data files from the NASA Turbulence Modeling
Resource into data/validation/ (falls back to the Internet Archive copy if
the NASA site does not answer), then writes docs/validation.png and
docs/validation.json.
"""

import json
import urllib.request
from pathlib import Path

import numpy as np

from simulation.aerodynamics import AirfoilSimulator, PanelMethod
from utils.parameterization import karman_trefftz, naca4, thin_airfoil_alpha_l0
from visualization import style

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data' / 'validation'
DOCS = ROOT / 'docs'
TMR = 'https://turbmodels.larc.nasa.gov/NACA0012_validation/'
ARCHIVE = 'https://web.archive.org/web/2024id_/' + TMR
FILES = ['CLCD_Ladson_expdata.dat', '0012.abbottdata.cl.dat',
         '0012.abbottdata.cd.dat', 'CP_Gregory_expdata.dat']
RE = 6e6


def fetch(name: str) -> Path:
    path = DATA_DIR / name
    if path.exists():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for base in (TMR, ARCHIVE):
        try:
            with urllib.request.urlopen(base + name, timeout=30) as r:
                body = r.read()
        except OSError as e:
            print(f"  {base + name}: {e}")
            continue
        if b'<html' in body[:500].lower():
            # the NASA address sometimes answers with a redirect page
            print(f"  {base + name}: got a web page, not data")
            continue
        path.write_bytes(body)
        print(f"  downloaded {name} from {base}")
        return path
    raise RuntimeError(f"could not download {name}")


def read_tecplot(path: Path) -> dict:
    """Parse the small Tecplot-style files: '#' comments, 'variables', zones."""
    zones, current = {}, 'data'
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.lower().startswith('variables'):
            continue
        if line.lower().startswith('zone'):
            current = line.split('"')[1]
            continue
        zones.setdefault(current, []).append([float(v) for v in line.split()])
    return {k: np.array(v) for k, v in zones.items()}


def main():
    style.apply()
    import matplotlib.pyplot as plt

    print("Fetching validation data")
    ladson = read_tecplot(fetch(FILES[0]))['80 grit']          # alpha, cl, cd
    abbott_cl = read_tecplot(fetch(FILES[1]))['data']          # alpha, cl
    abbott_cd = read_tecplot(fetch(FILES[2]))['data']          # cl, cd
    gregory = read_tecplot(fetch(FILES[3]))                    # x/c, cp (upper)

    out = {}

    # 1. exact potential-flow solution
    conv = []
    for n in (41, 81, 161, 321):
        x, y, exact = karman_trefftz(n=n)
        conv.append({'panels': n - 1,
                     'cl_error_pct': float(100 * (PanelMethod(x, y).cl(5.0) / exact(5.0) - 1))})
    out['karman_trefftz_alpha5'] = conv

    # 2. lift slope against thin-airfoil theory, 3. symmetric and cambered zero lift
    slopes = []
    for code in ('0001', '0006', '0012', '0018'):
        pm = PanelMethod(*naca4(code, n_points=121))
        slopes.append({'airfoil': 'NACA ' + code, 't_c': int(code[2:]) / 100,
                       'slope_per_rad': float((pm.cl(1) - pm.cl(-1)) / np.radians(2))})
    out['lift_slope'] = slopes
    out['two_pi'] = 2 * np.pi
    out['naca0012_cl_at_0deg'] = float(PanelMethod(*naca4('0012')).cl(0.0))
    out['naca2412_alpha_l0_panel'] = PanelMethod(*naca4('2412')).alpha_for_cl(0.0)
    out['naca2412_alpha_l0_thin_airfoil'] = thin_airfoil_alpha_l0(0.02, 0.4)

    # 4. NACA 0012 at Re = 6 million against wind-tunnel data
    x, y = naca4('0012')
    free = AirfoilSimulator()
    trip = AirfoilSimulator(x_trip=0.05)
    pm = free.panels(x, y)
    pts = []
    for a, cl_exp, cd_exp in ladson:
        if not -0.5 < a < 10.5:
            continue
        r = trip.analyze_at_cl(x, y, cl_exp, RE)
        pts.append({'alpha': a, 'cl_ladson': cl_exp, 'cl_panel': pm.cl(a),
                    'cd_ladson': cd_exp, 'cd_model_tripped': r['cd']})
    out['ladson_points'] = pts
    abb = []
    for cl_exp, cd_exp in abbott_cd:
        if 0 <= cl_exp <= 0.9:
            abb.append({'cl': cl_exp, 'cd_abbott': cd_exp,
                        'cd_model_free': free.analyze_at_cl(x, y, cl_exp, RE)['cd']})
    out['abbott_points'] = abb
    a_fit = ladson[(ladson[:, 0] > -5) & (ladson[:, 0] < 10.5)]
    out['ladson_lift_slope_per_deg'] = float(np.polyfit(a_fit[:, 0], a_fit[:, 1], 1)[0])
    out['panel_lift_slope_per_deg'] = float(pm.cl(1) - pm.cl(0))
    err = lambda m, d: float(100 * np.mean(np.abs(np.array(m) / np.array(d) - 1)))
    out['cd_mean_abs_error_pct_tripped_vs_ladson'] = err(
        [p['cd_model_tripped'] for p in pts], [p['cd_ladson'] for p in pts])
    out['cd_mean_abs_error_pct_free_vs_abbott'] = err(
        [p['cd_model_free'] for p in abb], [p['cd_abbott'] for p in abb])

    # ---- figure
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))
    a_line = np.linspace(-4, 18, 100)
    ax[0, 0].plot(a_line, [pm.cl(a) for a in a_line], color=style.BLUE,
                  label='Panel method (inviscid, no stall)')
    ax[0, 0].plot(ladson[:, 0], ladson[:, 1], **style.marker(style.ORANGE),
                  label='Ladson 1988, tripped')
    ax[0, 0].plot(abbott_cl[:, 0], abbott_cl[:, 1], **style.marker(style.AQUA, False),
                  label='Abbott & von Doenhoff 1959')
    ax[0, 0].set(xlim=(-4.5, 18.5), ylim=(-0.6, 2.2), xlabel='Angle of attack (deg)',
                 ylabel='Lift coefficient cl', title='NACA 0012 lift, Re = 6 million')
    ax[0, 0].legend(loc='upper left')

    cl_line = np.linspace(0, 1.15, 40)
    ax[0, 1].plot([free.analyze_at_cl(x, y, c, RE)['cd'] for c in cl_line], cl_line,
                  color=style.BLUE, label='Model, free transition')
    ax[0, 1].plot([trip.analyze_at_cl(x, y, c, RE)['cd'] for c in cl_line], cl_line,
                  color=style.BLUE, linestyle=(0, (4, 2)), label='Model, tripped at 5% chord')
    keep = (abbott_cd[:, 0] >= 0) & (abbott_cd[:, 0] <= 1.15)
    ax[0, 1].plot(abbott_cd[keep, 1], abbott_cd[keep, 0], **style.marker(style.AQUA, False),
                  label='Abbott & von Doenhoff, untripped')
    keep = (ladson[:, 1] >= 0) & (ladson[:, 1] <= 1.15) & (ladson[:, 2] < 0.03)
    ax[0, 1].plot(ladson[keep, 2], ladson[keep, 1], **style.marker(style.ORANGE),
                  label='Ladson, tripped')
    ax[0, 1].set(xlim=(0.004, 0.0145), xlabel='Drag coefficient cd',
                 ylabel='Lift coefficient cl', title='NACA 0012 drag polar, Re = 6 million')
    ax[0, 1].legend(loc='lower right')

    for alpha, ls in ((0, '-'), (10, (0, (4, 2)))):
        cp = pm.loads(alpha)['cp']
        upper = slice(1, int(np.argmin(pm.x)) + 1)
        ax[1, 0].plot(pm.x[upper], cp[upper], color=style.BLUE, linestyle=ls,
                      label=f'Panel method, upper surface, {alpha} deg')
        g = gregory[f'alpha={alpha}']
        ax[1, 0].plot(g[:, 0], g[:, 1], **style.marker(style.VIOLET, alpha == 0),
                      label=f"Gregory & O'Reilly, {alpha} deg")
    ax[1, 0].invert_yaxis()
    ax[1, 0].set(xlabel='x/c', ylabel='Pressure coefficient Cp',
                 title='NACA 0012 upper-surface pressure')
    ax[1, 0].legend(loc='upper right')

    tc = [s['t_c'] for s in slopes]
    ax[1, 1].axhline(2 * np.pi, color=style.INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax[1, 1].text(0.195, 2 * np.pi - 0.05, 'thin-airfoil theory, 2π per rad',
                  color=style.INK_2, fontsize=8.5, va='top', ha='right')
    ax[1, 1].plot(tc, [s['slope_per_rad'] for s in slopes], color=style.BLUE, marker='o',
                  markersize=7, markeredgecolor=style.SURFACE, markeredgewidth=1.5)
    for s in slopes:
        ax[1, 1].annotate(s['airfoil'], (s['t_c'], s['slope_per_rad']), xytext=(-8, 8),
                          textcoords='offset points', fontsize=8.5, color=style.INK_2,
                          ha='right')
    ax[1, 1].set(xlim=(-0.005, 0.2), ylim=(6.0, 7.6), xlabel='Thickness t/c',
                 ylabel='Lift slope (per rad)', title='Panel-method lift slope vs thickness')
    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    fig.savefig(DOCS / 'validation.png', dpi=130)

    (DOCS / 'validation.json').write_text(json.dumps(out, indent=2))

    print("\nKarman-Trefftz exact solution, cl error at 5 deg:")
    for c in conv:
        print(f"  {c['panels']:4d} panels: {c['cl_error_pct']:+.3f}%")
    print("\nLift slope (per rad), 2*pi = %.3f:" % (2 * np.pi))
    for s in slopes:
        print(f"  {s['airfoil']}: {s['slope_per_rad']:.3f}")
    print(f"\nNACA 0012 cl at 0 deg: {out['naca0012_cl_at_0deg']:.1e}")
    print(f"NACA 2412 zero-lift angle: panel {out['naca2412_alpha_l0_panel']:.2f} deg, "
          f"thin-airfoil theory {out['naca2412_alpha_l0_thin_airfoil']:.2f} deg")
    print(f"\nNACA 0012, Re 6e6, lift slope per deg: panel {out['panel_lift_slope_per_deg']:.4f}, "
          f"Ladson {out['ladson_lift_slope_per_deg']:.4f}")
    print("  alpha  cl_Ladson  cl_panel  cd_Ladson  cd_model(trip 5%, same cl)")
    for p in pts:
        print(f"  {p['alpha']:5.2f}  {p['cl_ladson']:8.4f}  {p['cl_panel']:8.4f}  "
              f"{p['cd_ladson']:.5f}    {p['cd_model_tripped']:.5f}")
    print(f"  cd mean abs error, tripped model vs Ladson: "
          f"{out['cd_mean_abs_error_pct_tripped_vs_ladson']:.1f}%")
    print("  cl     cd_Abbott  cd_model(free transition)")
    for p in abb:
        print(f"  {p['cl']:6.3f}  {p['cd_abbott']:.5f}    {p['cd_model_free']:.5f}")
    print(f"  cd mean abs error, free-transition model vs Abbott: "
          f"{out['cd_mean_abs_error_pct_free_vs_abbott']:.1f}%")
    print(f"\nSaved {DOCS / 'validation.png'} and {DOCS / 'validation.json'}")


if __name__ == '__main__':
    main()
