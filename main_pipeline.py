import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings  # <-- Import warnings BEFORE using it

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project to path - Windows compatible
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from utils.parameterization import PARSECAirfoil, generate_random_parsec_params
from simulation.aerodynamics import AirfoilSimulator, FlightConditions, create_performance_database
from models.surrogate import SurrogateModel, prepare_training_data
from optimization.algorithms import (
    GeneticAlgorithm, ParticleSwarmOptimization, 
    OptimizationConfig, create_objective_function
)

# ... rest of your code continues normally


class AirfoilOptimizationPipeline:
    """
    Complete pipeline for airfoil optimization
    """
    
    def __init__(self, work_dir: Path = None):
        if work_dir is None:
            work_dir = Path(__file__).parent
        
        self.work_dir = Path(work_dir)
        self.data_dir = self.work_dir / 'data'
        self.models_dir = self.work_dir / 'models'
        self.results_dir = self.work_dir / 'results'
        
        # Create directories
        for d in [self.data_dir, self.models_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.simulator = None
        self.surrogate_model = None
        self.flight_conditions = None
        
    def step1_generate_database(self, n_samples: int = 1000):
        """
        Step 1: Generate database of airfoil simulations
        """
        print("\n" + "="*60)
        print("STEP 1: Generating Airfoil Database")
        print("="*60)
        
        db_file = self.data_dir / 'airfoil_database.npz'
        create_performance_database(n_samples=n_samples, output_file=str(db_file))
        
        print(f"\n✓ Database created: {db_file}")
        return db_file
    
    def step2_train_surrogate(self, db_file: Path, model_type: str = 'ensemble'):
        """
        Step 2: Train ML surrogate model
        """
        print("\n" + "="*60)
        print("STEP 2: Training Surrogate Model")
        print("="*60)
        
        # Prepare data
        X, y = prepare_training_data(str(db_file))
        
        # Train model
        self.surrogate_model = SurrogateModel(model_type=model_type)
        metrics = self.surrogate_model.train(X, y, epochs=100, batch_size=64)
        
        # Save model
        model_file = self.models_dir / 'surrogate_model.pkl'
        self.surrogate_model.save(str(model_file))
        
        # Save metrics
        metrics_file = self.results_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            # Convert to serializable format
            serializable_metrics = {}
            for model_name, model_metrics in metrics.items():
                serializable_metrics[model_name] = {}
                for key, value in model_metrics.items():
                    if isinstance(value, (list, np.ndarray)):
                        serializable_metrics[model_name][key] = [float(v) for v in value]
                    else:
                        serializable_metrics[model_name][key] = float(value)
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"\n✓ Model saved: {model_file}")
        print(f"✓ Metrics saved: {metrics_file}")
        
        return model_file, metrics
    
    def step3_optimize_airfoil(self, objective_type: str = 'maximize_ld',
                              algorithm: str = 'ga', n_iterations: int = 100):
        """
        Step 3: Optimize airfoil design
        """
        print("\n" + "="*60)
        print("STEP 3: Optimizing Airfoil Design")
        print("="*60)
        
        # Define flight conditions
        self.flight_conditions = FlightConditions(
            altitude=1000,  # meters
            velocity=60      # m/s
        )
        
        print(f"\nFlight Conditions:")
        print(f"  Altitude: {self.flight_conditions.altitude} m")
        print(f"  Velocity: {self.flight_conditions.velocity} m/s")
        print(f"  Reynolds: {self.flight_conditions.reynolds:.2e}")
        print(f"  Mach: {self.flight_conditions.mach:.3f}")
        
        # Create objective function
        objective_func = create_objective_function(
            self.surrogate_model,
            self.flight_conditions,
            objective_type=objective_type
        )
        
        # Setup optimization
        config = OptimizationConfig(
            n_params=11,
            max_iterations=n_iterations,
            population_size=50
        )
        
        # Run optimization
        if algorithm == 'ga':
            print(f"\nRunning Genetic Algorithm...")
            optimizer = GeneticAlgorithm(config)
            best_params, best_fitness = optimizer.optimize(objective_func, verbose=True)
            history = optimizer.fitness_history
        elif algorithm == 'pso':
            print(f"\nRunning Particle Swarm Optimization...")
            optimizer = ParticleSwarmOptimization(config)
            best_params, best_fitness = optimizer.optimize(objective_func, verbose=True)
            history = optimizer.fitness_history
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Save results
        results = {
            'algorithm': algorithm,
            'objective_type': objective_type,
            'best_params': best_params.tolist(),
            'best_fitness': float(best_fitness),
            'flight_conditions': {
                'altitude': self.flight_conditions.altitude,
                'velocity': self.flight_conditions.velocity,
                'reynolds': float(self.flight_conditions.reynolds),
                'mach': float(self.flight_conditions.mach)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = self.results_dir / f'optimization_{algorithm}_{objective_type}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Optimization complete!")
        print(f"✓ Best fitness: {best_fitness:.4f}")
        print(f"✓ Results saved: {results_file}")
        
        return best_params, best_fitness, history
    
    def step4_analyze_results(self, best_params: np.ndarray):
        """
        Step 4: Analyze and visualize optimized airfoil
        """
        print("\n" + "="*60)
        print("STEP 4: Analyzing Results")
        print("="*60)
        
        # Create optimized airfoil
        airfoil_opt = PARSECAirfoil(best_params)
        x_opt, y_opt = airfoil_opt.generate_coordinates(200)
        
        # Run detailed simulation
        simulator = AirfoilSimulator()
        alphas = np.linspace(-5, 15, 41)
        results = simulator.sweep_analysis(
            x_opt, y_opt, alphas,
            self.flight_conditions.reynolds,
            self.flight_conditions.mach
        )
        
        # Create comparison with NACA 2412
        naca_params = np.array([0.0150, 0.30, 0.075, -0.40, 0.40, -0.04, 0.30, 
                               0.0015, -0.05, 0.12, 0.0])
        airfoil_naca = PARSECAirfoil(naca_params)
        x_naca, y_naca = airfoil_naca.generate_coordinates(200)
        results_naca = simulator.sweep_analysis(
            x_naca, y_naca, alphas,
            self.flight_conditions.reynolds,
            self.flight_conditions.mach
        )
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Airfoil shapes comparison
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(x_opt, y_opt, 'b-', linewidth=2, label='Optimized')
        ax1.plot(x_naca, y_naca, 'r--', linewidth=1.5, label='NACA 2412')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x/c')
        ax1.set_ylabel('y/c')
        ax1.set_title('Airfoil Geometry Comparison', fontsize=12, fontweight='bold')
        ax1.legend()
        
        # Lift curve
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(results['alpha'], results['cl'], 'b-', linewidth=2, label='Optimized')
        ax2.plot(results_naca['alpha'], results_naca['cl'], 'r--', linewidth=1.5, label='NACA 2412')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Angle of Attack (deg)')
        ax2.set_ylabel('Lift Coefficient (CL)')
        ax2.set_title('Lift Curve')
        ax2.legend()
        
        # Drag polar
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(results['cd'], results['cl'], 'b-', linewidth=2, label='Optimized')
        ax3.plot(results_naca['cd'], results_naca['cl'], 'r--', linewidth=1.5, label='NACA 2412')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Drag Coefficient (CD)')
        ax3.set_ylabel('Lift Coefficient (CL)')
        ax3.set_title('Drag Polar')
        ax3.legend()
        
        # L/D ratio
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(results['alpha'], results['ld_ratio'], 'b-', linewidth=2, label='Optimized')
        ax4.plot(results_naca['alpha'], results_naca['ld_ratio'], 'r--', linewidth=1.5, label='NACA 2412')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('Angle of Attack (deg)')
        ax4.set_ylabel('L/D Ratio')
        ax4.set_title('Aerodynamic Efficiency')
        ax4.legend()
        
        # Performance table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Find max L/D
        max_ld_idx = np.argmax(results['ld_ratio'])
        max_ld_idx_naca = np.argmax(results_naca['ld_ratio'])
        
        table_data = [
            ['Metric', 'Optimized', 'NACA 2412', 'Improvement'],
            ['Max L/D', f"{results['ld_ratio'][max_ld_idx]:.2f}", 
             f"{results_naca['ld_ratio'][max_ld_idx_naca]:.2f}",
             f"{(results['ld_ratio'][max_ld_idx] / results_naca['ld_ratio'][max_ld_idx_naca] - 1) * 100:.1f}%"],
            ['CL @ max L/D', f"{results['cl'][max_ld_idx]:.3f}",
             f"{results_naca['cl'][max_ld_idx_naca]:.3f}", '-'],
            ['CD @ max L/D', f"{results['cd'][max_ld_idx]:.4f}",
             f"{results_naca['cd'][max_ld_idx_naca]:.4f}",
             f"{(1 - results['cd'][max_ld_idx] / results_naca['cd'][max_ld_idx_naca]) * 100:.1f}%"],
        ]
        
        table = ax5.table(cellText=table_data, cellLoc='center',
                         bbox=[0.2, 0.3, 0.6, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Add title
        ax5.text(0.5, 0.9, 'Performance Summary', 
                ha='center', va='top', fontsize=14, fontweight='bold',
                transform=ax5.transAxes)
        
        save_path = self.results_dir / 'optimization_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Results visualization saved: {save_path}")
        
        # Save optimized airfoil
        airfoil_file = self.results_dir / 'optimized_airfoil.dat'
        airfoil_opt.save_dat_file(str(airfoil_file))
        print(f"✓ Optimized airfoil saved: {airfoil_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("OPTIMIZATION SUMMARY")
        print(f"{'='*60}")
        print(f"\nOptimized Airfoil Performance:")
        print(f"  Maximum L/D: {results['ld_ratio'][max_ld_idx]:.2f} @ α={results['alpha'][max_ld_idx]:.1f}°")
        print(f"  CL @ max L/D: {results['cl'][max_ld_idx]:.3f}")
        print(f"  CD @ max L/D: {results['cd'][max_ld_idx]:.4f}")
        
        print(f"\nNACA 2412 Baseline:")
        print(f"  Maximum L/D: {results_naca['ld_ratio'][max_ld_idx_naca]:.2f} @ α={results_naca['alpha'][max_ld_idx_naca]:.1f}°")
        
        improvement = (results['ld_ratio'][max_ld_idx] / results_naca['ld_ratio'][max_ld_idx_naca] - 1) * 100
        print(f"\nImprovement: {improvement:+.1f}%")
        
        return results
    
    def run_full_pipeline(self, n_database_samples: int = 500,
                         optimization_algorithm: str = 'ga',
                         n_optimization_iters: int = 100):
        """
        Run complete optimization pipeline
        """
        print("\n" + "="*60)
        print("AIRFOIL OPTIMIZATION PIPELINE")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  Database samples: {n_database_samples}")
        print(f"  Optimization algorithm: {optimization_algorithm}")
        print(f"  Optimization iterations: {n_optimization_iters}")
        
        # Step 1: Generate database
        db_file = self.step1_generate_database(n_samples=n_database_samples)
        
        # Step 2: Train surrogate model
        model_file, metrics = self.step2_train_surrogate(db_file)
        
        # Step 3: Optimize airfoil
        best_params, best_fitness, history = self.step3_optimize_airfoil(
            objective_type='maximize_ld',
            algorithm=optimization_algorithm,
            n_iterations=n_optimization_iters
        )
        
        # Step 4: Analyze results
        results = self.step4_analyze_results(best_params)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print(f"\nAll results saved to: {self.results_dir}")
        
        return {
            'best_params': best_params,
            'best_fitness': best_fitness,
            'results': results,
            'metrics': metrics
        }


if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = AirfoilOptimizationPipeline()
    
    # Quick run with smaller dataset for demonstration
    final_results = pipeline.run_full_pipeline(
        n_database_samples=200,  # Use 200 for quick demo, 1000+ for production
        optimization_algorithm='ga',
        n_optimization_iters=50   # Use 50 for demo, 200+ for production
    )
    
    print(f"\n✓ All files are ready in {pipeline.results_dir}")