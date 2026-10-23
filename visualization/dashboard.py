"""
Streamlit dashboard: analyse NACA 4-digit sections with the panel method and
boundary-layer drag estimate, compare shapes, and view the optimised result.

  streamlit run visualization/dashboard.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from simulation.aerodynamics import AirfoilSimulator, FlightConditions  # noqa: E402
from utils.parameterization import CSTAirfoil, max_thickness, naca4  # noqa: E402

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6']
RESULTS = ROOT / 'docs' / 'results.json'

st.set_page_config(page_title="Airfoil Analysis", layout="wide")
st.title("Airfoil analysis and optimisation")
st.caption("Linear-vortex panel method (inviscid, so no stall) with an integral "
           "boundary-layer drag estimate. Incompressible flow.")


@st.cache_data
def load_results():
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else None


def shape_coords(name, results):
    if name.startswith('NACA'):
        return naca4(name.split()[1])
    row = next(r for r in results['shapes'] if r['label'] == name)
    return CSTAirfoil.from_weights(row['weights']).coordinates(81)


@st.cache_data
def polar(x, y, reynolds, x_trip):
    alphas = np.arange(-4.0, 12.01, 0.5)
    return AirfoilSimulator(x_trip=x_trip).sweep_analysis(np.array(x), np.array(y), alphas, reynolds)


def shape_figure(items, colors=COLORS):
    fig = go.Figure()
    for (name, x, y), c in zip(items, colors):
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name, line=dict(color=c, width=2)))
    fig.update_layout(height=300, xaxis_title='x/c', yaxis_title='y/c', template='plotly_white',
                      yaxis=dict(scaleanchor='x', scaleratio=1), margin=dict(t=30, b=40),
                      legend=dict(orientation='h', y=1.15))
    return fig


def polar_figure(items):
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Lift curve', 'Drag polar', 'L/D'))
    for (name, p), c in zip(items, COLORS):
        line = dict(color=c, width=2)
        fig.add_trace(go.Scatter(x=p['alpha'], y=p['cl'], name=name, line=line), 1, 1)
        fig.add_trace(go.Scatter(x=p['cd'], y=p['cl'], name=name, line=line, showlegend=False), 1, 2)
        fig.add_trace(go.Scatter(x=p['cl'], y=p['ld_ratio'], name=name, line=line,
                                 showlegend=False), 1, 3)
    fig.update_xaxes(title_text='alpha (deg)', row=1, col=1)
    fig.update_xaxes(title_text='cd', row=1, col=2)
    fig.update_xaxes(title_text='cl', row=1, col=3)
    fig.update_yaxes(title_text='cl', row=1, col=1)
    fig.update_yaxes(title_text='cl', row=1, col=2)
    fig.update_yaxes(title_text='L/D', row=1, col=3)
    fig.update_layout(height=360, template='plotly_white', margin=dict(t=40, b=40),
                      legend=dict(orientation='h', y=1.2))
    return fig


results = load_results()
mode = st.sidebar.radio("Mode", ["Analyse a NACA section", "Compare shapes", "Optimised result"])

st.sidebar.subheader("Flight conditions")
altitude = st.sidebar.slider("Altitude (m)", 0, 10000, 1000, 250)
velocity = st.sidebar.slider("Speed (m/s)", 10, 100, 60, 5)
chord = st.sidebar.slider("Chord (m)", 0.2, 3.0, 1.0, 0.1)
transition = st.sidebar.radio("Transition", ["Free (Michel's criterion)", "Tripped at 5% chord"])
x_trip = 1.0 if transition.startswith('Free') else 0.05
flight = FlightConditions(altitude, velocity, chord)
st.sidebar.markdown(f"Reynolds number: **{flight.reynolds:,.0f}**  \nMach: **{flight.mach:.2f}**")
if flight.mach > 0.3:
    st.sidebar.warning("Mach above 0.3: the incompressible model is no longer a fair guide.")

if mode == "Analyse a NACA section":
    st.sidebar.subheader("NACA 4-digit section")
    m = st.sidebar.slider("Max camber (% chord)", 0, 9, 2)
    p = st.sidebar.slider("Camber position (tenths of chord)", 1, 9, 4)
    t = st.sidebar.slider("Thickness (% chord)", 6, 24, 12)
    alpha = st.sidebar.slider("Angle of attack (deg)", -4.0, 12.0, 4.0, 0.5)
    code = f"{m}{p if m else 0}{t:02d}"
    x, y = naca4(code)
    sim = AirfoilSimulator(x_trip=x_trip)
    r = sim.analyze(x, y, alpha, flight.reynolds)

    cols = st.columns(5)
    cols[0].metric("cl", f"{r['cl']:.3f}")
    cols[1].metric("cd (estimate)", f"{r['cd']:.5f}")
    cols[2].metric("cm (c/4)", f"{r['cm']:.4f}")
    cols[3].metric("L/D", f"{r['ld_ratio']:.1f}")
    cols[4].metric("Transition x/c (upper, lower)",
                   f"{r['x_transition_upper']:.2f}, {r['x_transition_lower']:.2f}")
    if r['separated']:
        st.warning(f"Turbulent separation predicted near x/c = {r['x_separation']:.2f}. "
                   "The drag estimate and the inviscid lift are not reliable here.")

    left, right = st.columns(2)
    with left:
        st.subheader(f"NACA {code}, t/c = {max_thickness(x, y):.3f}")
        st.plotly_chart(shape_figure([(f"NACA {code}", x, y)]), width='stretch')
    with right:
        st.subheader(f"Pressure at {alpha:.1f} deg")
        fig = go.Figure(go.Scatter(x=x, y=r['cp_distribution'], mode='lines',
                                   line=dict(color=COLORS[0], width=2)))
        fig.update_layout(height=300, xaxis_title='x/c', yaxis_title='Cp', template='plotly_white',
                          yaxis=dict(autorange='reversed'), margin=dict(t=30, b=40))
        st.plotly_chart(fig, width='stretch')
    st.subheader("Polar")
    st.plotly_chart(polar_figure([(f"NACA {code}", polar(tuple(x), tuple(y), flight.reynolds,
                                                          x_trip))]),
                    width='stretch')

elif mode == "Compare shapes":
    options = ['NACA 0012', 'NACA 2412', 'NACA 4412']
    if results:
        options += [r['label'] for r in results['shapes']]
    chosen = st.sidebar.multiselect("Shapes (up to 4)", options,
                                    default=options[1:2] + options[-2:-1] if results else options[:2],
                                    max_selections=4)
    if not chosen:
        st.info("Pick at least one shape.")
    else:
        coords = [(n, *shape_coords(n, results)) for n in chosen]
        st.plotly_chart(shape_figure(coords), width='stretch')
        polars = [(n, polar(tuple(x), tuple(y), flight.reynolds, x_trip)) for n, x, y in coords]
        st.plotly_chart(polar_figure(polars), width='stretch')
        sim = AirfoilSimulator(x_trip=x_trip)
        table = []
        for n, x, y in coords:
            r = sim.analyze_at_cl(x, y, 0.5, flight.reynolds)
            table.append({'Shape': n, 't/c': f"{max_thickness(x, y):.3f}",
                          'alpha at cl 0.5 (deg)': f"{r['alpha']:.2f}", 'cd': f"{r['cd']:.5f}",
                          'cm': f"{r['cm']:.4f}", 'L/D': f"{r['ld_ratio']:.1f}"})
        st.subheader("At cl = 0.5")
        st.table(table)

else:
    if not results:
        st.info("No results yet. Run `python main_pipeline.py` first.")
    else:
        dp = results['design_point']
        st.markdown(f"Design point: cl = {dp['cl']}, Re = {dp['reynolds']:,.0f}. "
                    "Numbers come from docs/results.json, written by main_pipeline.py.")
        st.table([{'Shape': r['label'], 'L/D': f"{r['ld']:.1f}",
                   'Change': f"{r['ld_gain_pct']:+.1f}%",
                   'L/D, tripped at 5%': f"{r['ld_tripped_5pct']:.1f}",
                   'Change, tripped': f"{r['ld_gain_tripped_pct']:+.1f}%",
                   'cd': f"{r['cd']:.5f}", 'cm': f"{r['cm']:.4f}",
                   't/c': f"{r['t_max']:.3f}", 'Meets limits': 'yes' if r['meets_constraints'] else 'no'}
                  for r in results['shapes']])
        coords = [(r['label'], *CSTAirfoil.from_weights(r['weights']).coordinates(81))
                  for r in results['shapes']]
        # same colours as docs/optimisation.png: baseline orange, optimised blue and green
        st.plotly_chart(shape_figure(coords, [COLORS[1], COLORS[0], COLORS[2]]), width='stretch')
        img = ROOT / 'docs' / 'optimisation.png'
        if img.exists():
            st.image(str(img))
