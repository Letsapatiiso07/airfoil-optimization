import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Get the script's directory
script_dir = Path(__file__).parent
results_dir = script_dir / 'results'
results_dir.mkdir(exist_ok=True)

sys.path.insert(0, str(script_dir))

from utils.parameterization import PARSECAirfoil, load_airfoil_database, generate_random_parsec_params
from simulation.aerodynamics import AirfoilSimulator, FlightConditions

print("="*70)
print("AIRFOIL OPTIMIZATION SYSTEM - QUICK DEMO")
print("="*70)

# Demo 1: Airfoil Parameterization
print("\n📐 Demo 1: Airfoil Parameterization")
print("-" * 70)

database = load_airfoil_database()
print(f"\nLoaded {len(database)} standard airfoils:")
for name in database.keys():
    print(f"  • {name}")

# Create example airfoils
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Airfoil Geometry Examples', fontsize=16, fontweight='bold')

# Standard airfoils
for idx, (name, params) in enumerate(database.items()):
    row, col = idx // 2, idx % 2
    airfoil = PARSECAirfoil(params)
    x, y = airfoil.generate_coordinates(150)
    
    axes[row, col].plot(x, y, 'b-', linewidth=2)
    axes[row, col].fill_between(x, y, alpha=0.3)
    axes[row, col].set_aspect('equal')
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].set_xlabel('x/c')
    axes[row, col].set_ylabel('y/c')
    axes[row, col].set_title(f'{name} Airfoil', fontweight='bold')
    
    # Add thickness annotation
    n_half = len(y) // 2
    thickness = np.max(y[:n_half] - y[n_half:2*n_half])
    axes[row, col].text(0.5, -0.15, f't/c = {thickness*100:.1f}%',
                        ha='center', transform=axes[row, col].transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
save_path = results_dir / 'demo_geometries.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Airfoil geometries saved to {save_path}")

# Demo 2: Aerodynamic Analysis
print("\n\n📊 Demo 2: Aerodynamic Performance Analysis")
print("-" * 70)

# Setup simulation
simulator = AirfoilSimulator()
conditions = FlightConditions(altitude=1000, velocity=60)

print(f"\nFlight Conditions:")
print(f"  Altitude: {conditions.altitude} m")
print(f"  Velocity: {conditions.velocity} m/s")
print(f"  Reynolds: {conditions.reynolds:.2e}")
print(f"  Mach: {conditions.mach:.3f}")

# Analyze NACA 2412
print(f"\nAnalyzing NACA 2412...")
airfoil_2412 = PARSECAirfoil(database['NACA2412'])
x_2412, y_2412 = airfoil_2412.generate_coordinates(150)

alphas = np.linspace(-5, 15, 41)
results_2412 = simulator.sweep_analysis(x_2412, y_2412, alphas,
                                        conditions.reynolds, conditions.mach)

# Find optimal conditions
max_ld_idx = np.argmax(results_2412['ld_ratio'])

print(f"\nPerformance Metrics:")
print(f"  Max L/D: {results_2412['ld_ratio'][max_ld_idx]:.2f} @ α={results_2412['alpha'][max_ld_idx]:.1f}°")
print(f"  CL @ max L/D: {results_2412['cl'][max_ld_idx]:.3f}")
print(f"  CD @ max L/D: {results_2412['cd'][max_ld_idx]:.4f}")
print(f"  CM @ max L/D: {results_2412['cm'][max_ld_idx]:.4f}")

# Create performance plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('NACA 2412 Aerodynamic Performance', fontsize=16, fontweight='bold')

# Lift curve
axes[0, 0].plot(results_2412['alpha'], results_2412['cl'], 'b-', linewidth=2.5)
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlabel('Angle of Attack (deg)', fontsize=11)
axes[0, 0].set_ylabel('Lift Coefficient (CL)', fontsize=11)
axes[0, 0].set_title('Lift Curve', fontweight='bold')

# Drag polar
axes[0, 1].plot(results_2412['cd'], results_2412['cl'], 'r-', linewidth=2.5)
axes[0, 1].plot(results_2412['cd'][max_ld_idx], results_2412['cl'][max_ld_idx], 
                'go', markersize=10, label=f'Max L/D = {results_2412["ld_ratio"][max_ld_idx]:.1f}')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlabel('Drag Coefficient (CD)', fontsize=11)
axes[0, 1].set_ylabel('Lift Coefficient (CL)', fontsize=11)
axes[0, 1].set_title('Drag Polar', fontweight='bold')
axes[0, 1].legend()

# L/D ratio
axes[1, 0].plot(results_2412['alpha'], results_2412['ld_ratio'], 'g-', linewidth=2.5)
axes[1, 0].plot(results_2412['alpha'][max_ld_idx], results_2412['ld_ratio'][max_ld_idx],
                'ro', markersize=10, label=f'Max @ α={results_2412["alpha"][max_ld_idx]:.1f}°')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlabel('Angle of Attack (deg)', fontsize=11)
axes[1, 0].set_ylabel('L/D Ratio', fontsize=11)
axes[1, 0].set_title('Aerodynamic Efficiency', fontweight='bold')
axes[1, 0].legend()

# Moment coefficient
axes[1, 1].plot(results_2412['alpha'], results_2412['cm'], 'm-', linewidth=2.5)
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlabel('Angle of Attack (deg)', fontsize=11)
axes[1, 1].set_ylabel('Moment Coefficient (CM)', fontsize=11)
axes[1, 1].set_title('Pitching Moment', fontweight='bold')

plt.tight_layout()
save_path = results_dir / 'demo_performance.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Performance analysis saved to {save_path}")

# Demo 3: Comparison of Multiple Airfoils
print("\n\n🔄 Demo 3: Comparative Analysis")
print("-" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Airfoil Performance Comparison', fontsize=16, fontweight='bold')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
comparison_data = []

for idx, (name, params) in enumerate(database.items()):
    airfoil = PARSECAirfoil(params)
    x, y = airfoil.generate_coordinates(150)
    
    results = simulator.sweep_analysis(x, y, alphas, conditions.reynolds, conditions.mach)
    max_ld_idx = np.argmax(results['ld_ratio'])
    
    # Plot lift curves
    axes[0].plot(results['alpha'], results['cl'], color=colors[idx], 
                linewidth=2, label=name)
    
    # Plot drag polars
    axes[1].plot(results['cd'], results['cl'], color=colors[idx],
                linewidth=2, label=name)
    
    comparison_data.append({
        'name': name,
        'max_ld': results['ld_ratio'][max_ld_idx],
        'alpha_max_ld': results['alpha'][max_ld_idx],
        'cl_max_ld': results['cl'][max_ld_idx],
        'cd_max_ld': results['cd'][max_ld_idx]
    })

axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('Angle of Attack (deg)', fontsize=11)
axes[0].set_ylabel('Lift Coefficient (CL)', fontsize=11)
axes[0].set_title('Lift Curves', fontweight='bold')
axes[0].legend()

axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('Drag Coefficient (CD)', fontsize=11)
axes[1].set_ylabel('Lift Coefficient (CL)', fontsize=11)
axes[1].set_title('Drag Polars', fontweight='bold')
axes[1].legend()

plt.tight_layout()
save_path = results_dir / 'demo_comparison.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Comparison plots saved to {save_path}")

# Print comparison table
print("\n" + "="*70)
print("PERFORMANCE COMPARISON TABLE")
print("="*70)
print(f"{'Airfoil':<15} {'Max L/D':<10} {'α @ max L/D':<15} {'CL':<8} {'CD':<8}")
print("-"*70)
for data in comparison_data:
    print(f"{data['name']:<15} {data['max_ld']:<10.2f} {data['alpha_max_ld']:<15.1f} "
          f"{data['cl_max_ld']:<8.3f} {data['cd_max_ld']:<8.4f}")

print("\n\n" + "="*70)
print("DEMO SUMMARY")
print("="*70)
print(f"""
✓ Demonstrated airfoil parameterization (PARSEC method)
✓ Performed aerodynamic analysis using CFD simulation
✓ Compared multiple airfoil designs

Generated Files in {results_dir}:
  📄 demo_geometries.png - Airfoil shapes
  📄 demo_performance.png - Performance metrics
  📄 demo_comparison.png - Design comparisons

Next Steps:
  🚀 Run full optimization: python main_pipeline.py
  🖥️  Launch dashboard: streamlit run visualization/dashboard.py
  📖 Read documentation: README.md
""")

print("="*70)
print(f"Demo complete! All visualizations saved to {results_dir}")
print("="*70)