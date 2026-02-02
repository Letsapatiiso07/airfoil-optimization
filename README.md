# Airfoil Optimization System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ML](https://img.shields.io/badge/ML-PyTorch%20%7C%20XGBoost-orange.svg)](https://pytorch.org/)

> **End-to-end machine learning system for optimizing airfoil geometries using CFD simulation, ensemble ML models, and evolutionary algorithms.**

Transform baseline NACA airfoils into high-performance designs with **10-25% efficiency improvements** using automated optimization!

![Airfoil Optimization](results/demo_comparison.png)

---

## What It Does

This system automatically designs optimal airfoil shapes for specific flight conditions by:

1. **Parameterizing** airfoil geometry using PARSEC/CST methods (11 parameters)
2. **Simulating** aerodynamic performance with fast CFD analysis
3. **Learning** performance patterns with ensemble ML models (NN + XGBoost + RF)
4. **Optimizing** designs using genetic algorithms achieving 500× speedup over pure CFD

**Result**: Airfoils optimized for maximum L/D ratio, minimum drag, or custom objectives.

---

## Key Features

- **Advanced Parameterization**: PARSEC & CST methods for smooth, realistic shapes
- **Fast CFD Simulation**: Panel method with boundary layer corrections
- **Ensemble ML Models**: Neural Network + XGBoost + Random Forest (R² > 0.98)
- **Multi-Algorithm Optimization**: Genetic Algorithm, PSO, gradient-based methods
- **Interactive Dashboard**: Real-time design and analysis with Streamlit
- **Multi-Objective**: Optimize for efficiency, lift, stability, or custom goals

---

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/airfoil-optimization.git
cd airfoil-optimization

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Demo (1 minute)
```bash
python demo.py
```

Generates airfoil geometries and performance visualizations.

### Run Full Optimization (15-30 minutes)
```bash
python main_pipeline.py
```

Complete workflow: database generation → ML training → optimization → analysis

### Launch Interactive Dashboard
```bash
streamlit run visualization/dashboard.py
```

Design and analyze airfoils in real-time at `http://localhost:8501`

---

## Results

### Performance Improvements

Typical optimization results vs. baseline NACA airfoils:

| Metric | Baseline (NACA 2412) | Optimized | Improvement |
|--------|---------------------|-----------|-------------|
| **Max L/D** | 41.2 | 48.3 | **+17.2%** |
| **Drag Coefficient** | 0.0195 | 0.0174 | **-10.8%** |
| **Lift Coefficient** | 0.804 | 0.842 | **+4.7%** |

### Example Output
```
OPTIMIZATION SUMMARY
============================================================

Optimized Airfoil Performance:
  Maximum L/D: 48.32 @ α=5.2°
  CL @ max L/D: 0.842
  CD @ max L/D: 0.0174

NACA 2412 Baseline:
  Maximum L/D: 41.15 @ α=4.5°

Improvement: +17.4%
```

---

## System Architecture
```
┌─────────────────┐
│ Flight Specs    │ → Altitude, Velocity, Reynolds number
└────────┬────────┘
         ↓
┌─────────────────┐
│ Parameterization│ → PARSEC (11 params) or CST
└────────┬────────┘
         ↓
┌─────────────────┐
│ CFD Simulation  │ → Panel method + BL corrections
└────────┬────────┘
         ↓
┌─────────────────┐
│ ML Surrogate    │ → Ensemble: NN + XGB + RF (500× faster)
└────────┬────────┘
         ↓
┌─────────────────┐
│ Optimization    │ → GA / PSO / Gradient-based
└────────┬────────┘
         ↓
┌─────────────────┐
│ Optimized Design│ → Geometry + Performance analysis
└─────────────────┘
```

---

## Project Structure
```
airfoil_optimization/
├── utils/
│   └── parameterization.py    # PARSEC & CST airfoil geometry
├── simulation/
│   └── aerodynamics.py         # CFD simulation engine
├── models/
│   └── surrogate.py            # ML models (NN, XGBoost, RF)
├── optimization/
│   └── algorithms.py           # GA, PSO, gradient methods
├── visualization/
│   └── dashboard.py            # Interactive Streamlit app
├── demo.py                     # Quick demonstration
├── main_pipeline.py            # Full optimization pipeline
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## Configuration

### Customize Flight Conditions
```python
from simulation.aerodynamics import FlightConditions

conditions = FlightConditions(
    altitude=5000,    # meters
    velocity=80       # m/s
)
```

### Custom Optimization Objective
```python
def custom_objective(params):
    """Maximize L/D while maintaining CL > 0.6"""
    prediction = surrogate_model.predict(params)
    cl, cd, cm, ld = prediction[0]
    
    if cl < 0.6:
        return -1000  # Penalty
    return ld

optimizer.optimize(custom_objective)
```

### Adjust Pipeline Parameters
```python
pipeline.run_full_pipeline(
    n_database_samples=1000,     # Training data size
    optimization_algorithm='ga',  # 'ga', 'pso', or 'gradient'
    n_optimization_iters=200      # More = better results
)
```

---

## Technical Details

### PARSEC Parameterization

11 parameters define any airfoil:
- Leading edge radius
- Upper/lower crest positions & curvatures
- Trailing edge geometry

### ML Model Architecture
```
Ensemble Model:
├── Neural Network: 12 → 128 → 256 → 256 → 128 → 4
├── XGBoost: 200 trees, depth=8
└── Random Forest: 200 trees, depth=20

Input:  11 PARSEC params + angle of attack
Output: CL, CD, CM, L/D ratio
```

### Genetic Algorithm

- Population: 50 individuals
- Crossover: Simulated Binary (SBX)
- Mutation: Polynomial
- Selection: Tournament (size=3)
- Elitism: Top 10%

---

## Use Cases

- **Aircraft Design**: Wing sections for UAVs, light aircraft
- **Wind Energy**: Optimized turbine blade profiles
- **Motorsports**: Race car wings and aerodynamic surfaces
- **Education**: Teaching aerodynamics and ML optimization
- **Research**: Rapid prototyping of novel airfoil concepts

---

## Performance Benchmarks

| Task | Time | Notes |
|------|------|-------|
| Generate 1000 airfoil database | ~10 min | One-time setup |
| Train ML surrogate | ~5 min | One-time setup |
| Optimize (GA, 200 iters) | ~2 min | Uses trained model |
| Single CFD simulation | 0.5 sec | Direct analysis |
| Single ML prediction | 0.001 sec | **500× speedup** |

---

## Dependencies

- Python 3.8+
- NumPy, SciPy, Pandas
- PyTorch (Neural Networks)
- XGBoost, scikit-learn
- Matplotlib, Plotly, Seaborn
- Streamlit (Dashboard)

---

## Future Enhancements

- [ ] Integration with XFoil/SU2 for high-fidelity CFD
- [ ] 3D wing optimization (not just 2D airfoils)
- [ ] Multi-point optimization (multiple flight conditions)
- [ ] Structural constraints (thickness, manufacturability)
- [ ] Unsteady aerodynamics and dynamic stall
- [ ] Neural architecture search for better surrogates

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional parameterization methods
- Better CFD solvers
- More optimization algorithms
- Extended validation cases

---

## 📄 License

MIT License - free to use for research and commercial applications

---

## 🙏 Acknowledgments

**Methods & References:**
- PARSEC: Sobieczky (1999) - Parametric Airfoils and Wings
- CST: Kulfan (2008) - Universal Parametric Geometry
- Panel Method: Katz & Plotkin - Low-Speed Aerodynamics
- NSGA-II: Deb et al. (2002) - Multi-objective GA

---

## 📧 Contact

**Tiiso** - Letsapamyron07@gmail.com
Project Link: [https://github.com/Letsapatiiso07/airfoil-optimization](https://github.com/Letsapatiiso07/airfoil-optimization)

---

**Built with ❤️ for aerospace engineers and ML practitioners**

🚁 Designing better airfoils, one optimization at a time! ✈️