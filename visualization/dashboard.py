"""
Interactive Dashboard for Airfoil Optimization
Streamlit app for real-time design and analysis
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import json
from pathlib import Path

# Add parent directory to path for imports - Windows compatible
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from utils.parameterization import PARSECAirfoil, load_airfoil_database
from simulation.aerodynamics import AirfoilSimulator, FlightConditions
from models.surrogate import SurrogateModel

# Page config
st.set_page_config(
    page_title="Airfoil Optimization Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">✈️ Airfoil Optimization Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'simulator' not in st.session_state:
    st.session_state.simulator = AirfoilSimulator()
if 'current_airfoil' not in st.session_state:
    database = load_airfoil_database()
    st.session_state.current_airfoil = database['NACA2412']

# Sidebar - Controls
st.sidebar.title("⚙️ Configuration")

# Mode selection
mode = st.sidebar.radio("Mode", ["Design Airfoil", "Compare Designs", "Load Optimized"])

if mode == "Design Airfoil":
    st.sidebar.subheader("PARSEC Parameters")
    
    # PARSEC parameter sliders
    params = []
    param_names = [
        ('Leading Edge Radius', 0.005, 0.025, 0.015),
        ('Upper Crest X', 0.25, 0.45, 0.35),
        ('Upper Crest Y', 0.04, 0.08, 0.06),
        ('Upper Curvature', -0.5, -0.1, -0.35),
        ('Lower Crest X', 0.30, 0.50, 0.40),
        ('Lower Crest Y', -0.05, -0.02, -0.035),
        ('Lower Curvature', 0.1, 0.5, 0.3),
        ('TE Thickness', 0.001, 0.005, 0.002),
        ('TE Direction', -0.15, 0.15, 0.0),
        ('TE Wedge Angle', 0.05, 0.20, 0.10),
        ('TE Offset', -0.002, 0.002, 0.0)
    ]
    
    with st.sidebar.expander("Geometry Parameters", expanded=True):
        for name, min_val, max_val, default in param_names:
            val = st.slider(name, min_val, max_val, default, 
                          format="%.4f", key=f"param_{name}")
            params.append(val)
    
    st.session_state.current_airfoil = np.array(params)

elif mode == "Compare Designs":
    st.sidebar.subheader("Select Airfoils")
    database = load_airfoil_database()
    selected_airfoils = st.sidebar.multiselect(
        "Choose airfoils to compare",
        list(database.keys()),
        default=['NACA0012', 'NACA2412']
    )

# Flight conditions
st.sidebar.subheader("Flight Conditions")
altitude = st.sidebar.slider("Altitude (m)", 0, 10000, 1000, 100)
velocity = st.sidebar.slider("Velocity (m/s)", 20, 150, 60, 5)

# Create flight conditions
flight_conditions = FlightConditions(altitude=altitude, velocity=velocity)

# Main content
if mode == "Design Airfoil":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Airfoil Geometry")
        
        # Create airfoil
        airfoil = PARSECAirfoil(st.session_state.current_airfoil)
        x, y = airfoil.generate_coordinates(150)
        
        # Plot with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color='#1f77b4', width=3),
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            name='Airfoil'
        ))
        
        fig.update_layout(
            height=400,
            xaxis_title='x/c',
            yaxis_title='y/c',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            hovermode='closest',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Geometry metrics - FIXED
        n_half = len(y) // 2
        thickness = np.max(y[:n_half] - y[n_half:2*n_half])
        st.metric("Max Thickness", f"{thickness*100:.2f}%")
    
    with col2:
        st.subheader("📊 Performance Analysis")
        
        # Run simulation
        alphas = np.linspace(-5, 15, 41)
        results = st.session_state.simulator.sweep_analysis(
            x, y, alphas, 
            flight_conditions.reynolds,
            flight_conditions.mach
        )
        
        # Create performance plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Lift Curve', 'Drag Polar', 'L/D Ratio', 'Moment Coefficient')
        )
        
        # CL vs alpha
        fig.add_trace(
            go.Scatter(x=results['alpha'], y=results['cl'], 
                      mode='lines', name='CL', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Drag polar
        fig.add_trace(
            go.Scatter(x=results['cd'], y=results['cl'],
                      mode='lines', name='Drag Polar', line=dict(color='red')),
            row=1, col=2
        )
        
        # L/D ratio
        fig.add_trace(
            go.Scatter(x=results['alpha'], y=results['ld_ratio'],
                      mode='lines', name='L/D', line=dict(color='green')),
            row=2, col=1
        )
        
        # CM
        fig.add_trace(
            go.Scatter(x=results['alpha'], y=results['cm'],
                      mode='lines', name='CM', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="α (deg)", row=1, col=1)
        fig.update_xaxes(title_text="CD", row=1, col=2)
        fig.update_xaxes(title_text="α (deg)", row=2, col=1)
        fig.update_xaxes(title_text="α (deg)", row=2, col=2)
        
        fig.update_yaxes(title_text="CL", row=1, col=1)
        fig.update_yaxes(title_text="CL", row=1, col=2)
        fig.update_yaxes(title_text="L/D", row=2, col=1)
        fig.update_yaxes(title_text="CM", row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.subheader("🎯 Key Performance Indicators")
    
    max_ld_idx = np.argmax(results['ld_ratio'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Max L/D Ratio",
            f"{results['ld_ratio'][max_ld_idx]:.2f}",
            f"@ α={results['alpha'][max_ld_idx]:.1f}°"
        )
    
    with col2:
        st.metric(
            "CL @ Max L/D",
            f"{results['cl'][max_ld_idx]:.3f}"
        )
    
    with col3:
        st.metric(
            "CD @ Max L/D",
            f"{results['cd'][max_ld_idx]:.4f}"
        )
    
    with col4:
        st.metric(
            "CM @ Max L/D",
            f"{results['cm'][max_ld_idx]:.4f}"
        )
    
    # Flight conditions info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Flight Info")
    st.sidebar.text(f"Reynolds: {flight_conditions.reynolds:.2e}")
    st.sidebar.text(f"Mach: {flight_conditions.mach:.3f}")
    st.sidebar.text(f"Density: {flight_conditions.density:.3f} kg/m³")

elif mode == "Compare Designs":
    st.subheader("🔄 Airfoil Comparison")
    
    if len(selected_airfoils) == 0:
        st.warning("Please select at least one airfoil to compare")
    else:
        database = load_airfoil_database()
        
        # Geometry comparison
        fig_geom = go.Figure()
        
        colors = px.colors.qualitative.Plotly
        
        for i, name in enumerate(selected_airfoils):
            params = database[name]
            airfoil = PARSECAirfoil(params)
            x, y = airfoil.generate_coordinates(150)
            
            fig_geom.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig_geom.update_layout(
            height=400,
            xaxis_title='x/c',
            yaxis_title='y/c',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            template='plotly_white',
            title='Geometry Comparison'
        )
        
        st.plotly_chart(fig_geom, use_container_width=True)
        
        # Performance comparison
        st.subheader("Performance Comparison")
        
        alphas = np.linspace(-5, 15, 41)
        
        fig_perf = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Lift Curves', 'Drag Polars')
        )
        
        comparison_data = []
        
        for i, name in enumerate(selected_airfoils):
            params = database[name]
            airfoil = PARSECAirfoil(params)
            x, y = airfoil.generate_coordinates(150)
            
            results = st.session_state.simulator.sweep_analysis(
                x, y, alphas,
                flight_conditions.reynolds,
                flight_conditions.mach
            )
            
            max_ld_idx = np.argmax(results['ld_ratio'])
            
            comparison_data.append({
                'Airfoil': name,
                'Max L/D': f"{results['ld_ratio'][max_ld_idx]:.2f}",
                'CL @ Max L/D': f"{results['cl'][max_ld_idx]:.3f}",
                'CD @ Max L/D': f"{results['cd'][max_ld_idx]:.4f}",
                'α @ Max L/D': f"{results['alpha'][max_ld_idx]:.1f}°"
            })
            
            fig_perf.add_trace(
                go.Scatter(x=results['alpha'], y=results['cl'],
                          mode='lines', name=name,
                          line=dict(color=colors[i % len(colors)])),
                row=1, col=1
            )
            
            fig_perf.add_trace(
                go.Scatter(x=results['cd'], y=results['cl'],
                          mode='lines', name=name,
                          line=dict(color=colors[i % len(colors)]),
                          showlegend=False),
                row=1, col=2
            )
        
        fig_perf.update_xaxes(title_text="α (deg)", row=1, col=1)
        fig_perf.update_xaxes(title_text="CD", row=1, col=2)
        fig_perf.update_yaxes(title_text="CL", row=1, col=1)
        fig_perf.update_yaxes(title_text="CL", row=1, col=2)
        
        fig_perf.update_layout(height=500, template='plotly_white')
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Comparison table
        st.subheader("Performance Summary")
        st.table(comparison_data)

elif mode == "Load Optimized":
    st.subheader("📂 Load Optimized Design")
    
    results_dir = script_dir / 'results'
    
    if (results_dir / 'optimized_airfoil.dat').exists():
        st.success("✓ Optimized airfoil found!")
        
        # Load results
        json_files = list(results_dir.glob('optimization_*.json'))
        if json_files:
            with open(json_files[0], 'r') as f:
                opt_results = json.load(f)
            
            st.json(opt_results)
            
            # Show optimized geometry
            params = np.array(opt_results['best_params'])
            airfoil = PARSECAirfoil(params)
            x, y = airfoil.generate_coordinates(150)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color='#1f77b4', width=3),
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.2)',
                name='Optimized Airfoil'
            ))
            
            fig.update_layout(
                height=400,
                xaxis_title='x/c',
                yaxis_title='y/c',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                template='plotly_white',
                title='Optimized Airfoil Geometry'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show result image
        if (results_dir / 'optimization_results.png').exists():
            st.image(str(results_dir / 'optimization_results.png'))
    else:
        st.info("No optimized airfoil found. Run the optimization pipeline first.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Airfoil Optimization System | Powered by ML & CFD</p>
</div>
""", unsafe_allow_html=True)