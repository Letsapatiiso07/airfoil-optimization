import numpy as np
from typing import Callable, Dict, Tuple, Optional, List
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class OptimizationConfig:
    """Configuration for optimization"""
    n_params: int = 11  # PARSEC parameters
    bounds: List[Tuple[float, float]] = None
    objective: str = 'maximize_ld'  # or 'minimize_drag', 'target_cl', etc.
    constraints: Dict = None
    max_iterations: int = 100
    population_size: int = 50
    
    def __post_init__(self):
        if self.bounds is None:
            # Default PARSEC parameter bounds
            self.bounds = [
                (0.005, 0.025),   # rle
                (0.25, 0.45),     # x_up
                (0.04, 0.08),     # y_up
                (-0.5, -0.1),     # y_xxup
                (0.30, 0.50),     # x_lo
                (-0.05, -0.02),   # y_lo
                (0.1, 0.5),       # y_xxlo
                (0.001, 0.005),   # yte
                (-0.15, 0.15),    # alpha_te
                (0.05, 0.20),     # beta_te
                (-0.002, 0.002)   # dz_te
            ]


class GeneticAlgorithm:
    """
    Genetic Algorithm for airfoil optimization
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.population = None
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = -np.inf
        
    def initialize_population(self):
        """Initialize random population within bounds"""
        population = []
        for _ in range(self.config.population_size):
            individual = np.array([
                np.random.uniform(low, high) 
                for low, high in self.config.bounds
            ])
            population.append(individual)
        self.population = np.array(population)
        
    def evaluate_population(self, objective_func: Callable) -> np.ndarray:
        """Evaluate fitness of entire population"""
        fitness = np.array([objective_func(ind) for ind in self.population])
        return fitness
    
    def selection(self, fitness: np.ndarray, n_parents: int) -> np.ndarray:
        """Tournament selection"""
        parents = []
        for _ in range(n_parents):
            # Tournament
            tournament_size = 3
            indices = np.random.choice(len(self.population), tournament_size, replace=False)
            tournament_fitness = fitness[indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx])
        return np.array(parents)
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated binary crossover (SBX)"""
        eta = 20  # Distribution index
        u = np.random.random(len(parent1))
        
        beta = np.where(u <= 0.5,
                       (2 * u) ** (1 / (eta + 1)),
                       (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
        
        child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
        
        # Clip to bounds
        for i, (low, high) in enumerate(self.config.bounds):
            child1[i] = np.clip(child1[i], low, high)
            child2[i] = np.clip(child2[i], low, high)
        
        return child1, child2
    
    def mutation(self, individual: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Polynomial mutation"""
        eta = 20  # Distribution index
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() < mutation_rate:
                low, high = self.config.bounds[i]
                delta = high - low
                
                u = np.random.random()
                if u < 0.5:
                    delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                
                mutated[i] += delta_q * delta
                mutated[i] = np.clip(mutated[i], low, high)
        
        return mutated
    
    def optimize(self, objective_func: Callable, 
                verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run genetic algorithm optimization
        
        Args:
            objective_func: Function to maximize (params -> fitness)
            verbose: Print progress
            
        Returns:
            Best parameters and fitness
        """
        self.initialize_population()
        
        iterator = tqdm(range(self.config.max_iterations)) if verbose else range(self.config.max_iterations)
        
        for generation in iterator:
            # Evaluate fitness
            fitness = self.evaluate_population(objective_func)
            
            # Track best
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_individual = self.population[best_idx].copy()
            
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(fitness),
                'std_fitness': np.std(fitness)
            })
            
            if verbose:
                iterator.set_description(
                    f"Gen {generation}: Best={self.best_fitness:.4f}, "
                    f"Mean={np.mean(fitness):.4f}"
                )
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            elite_size = max(2, self.config.population_size // 10)
            elite_indices = np.argsort(fitness)[-elite_size:]
            for idx in elite_indices:
                new_population.append(self.population[idx])
            
            # Generate offspring
            while len(new_population) < self.config.population_size:
                # Selection
                parents = self.selection(fitness, 2)
                
                # Crossover
                child1, child2 = self.crossover(parents[0], parents[1])
                
                # Mutation
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            self.population = np.array(new_population[:self.config.population_size])
        
        return self.best_individual, self.best_fitness


class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization for airfoil design
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.particles = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.global_best = None
        self.global_best_fitness = -np.inf
        self.fitness_history = []
        
    def initialize_swarm(self):
        """Initialize particle swarm"""
        # Positions
        self.particles = np.array([
            np.array([np.random.uniform(low, high) for low, high in self.config.bounds])
            for _ in range(self.config.population_size)
        ])
        
        # Velocities
        velocity_bounds = np.array([(high - low) * 0.1 for low, high in self.config.bounds])
        self.velocities = np.random.uniform(-velocity_bounds, velocity_bounds, 
                                           (self.config.population_size, self.config.n_params))
        
        # Personal bests
        self.personal_best = self.particles.copy()
        self.personal_best_fitness = np.full(self.config.population_size, -np.inf)
        
    def optimize(self, objective_func: Callable, 
                w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run PSO optimization
        
        Args:
            objective_func: Function to maximize
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
            verbose: Print progress
            
        Returns:
            Best parameters and fitness
        """
        self.initialize_swarm()
        
        iterator = tqdm(range(self.config.max_iterations)) if verbose else range(self.config.max_iterations)
        
        for iteration in iterator:
            # Evaluate fitness
            fitness = np.array([objective_func(particle) for particle in self.particles])
            
            # Update personal bests
            improved = fitness > self.personal_best_fitness
            self.personal_best[improved] = self.particles[improved]
            self.personal_best_fitness[improved] = fitness[improved]
            
            # Update global best
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > self.global_best_fitness:
                self.global_best_fitness = fitness[best_idx]
                self.global_best = self.particles[best_idx].copy()
            
            self.fitness_history.append({
                'iteration': iteration,
                'best_fitness': self.global_best_fitness,
                'mean_fitness': np.mean(fitness)
            })
            
            if verbose:
                iterator.set_description(
                    f"Iter {iteration}: Best={self.global_best_fitness:.4f}"
                )
            
            # Update velocities and positions
            r1 = np.random.random((self.config.population_size, self.config.n_params))
            r2 = np.random.random((self.config.population_size, self.config.n_params))
            
            cognitive = c1 * r1 * (self.personal_best - self.particles)
            social = c2 * r2 * (self.global_best - self.particles)
            
            self.velocities = w * self.velocities + cognitive + social
            self.particles = self.particles + self.velocities
            
            # Enforce bounds
            for i, (low, high) in enumerate(self.config.bounds):
                self.particles[:, i] = np.clip(self.particles[:, i], low, high)
        
        return self.global_best, self.global_best_fitness


class GradientBasedOptimizer:
    """
    Gradient-based optimization using scipy.optimize
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.history = []
        
    def optimize(self, objective_func: Callable, 
                initial_params: Optional[np.ndarray] = None,
                method: str = 'L-BFGS-B',
                verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run gradient-based optimization
        
        Args:
            objective_func: Function to maximize
            initial_params: Starting point (random if None)
            method: Optimization method
            verbose: Print progress
            
        Returns:
            Best parameters and fitness
        """
        if initial_params is None:
            initial_params = np.array([
                np.random.uniform(low, high) 
                for low, high in self.config.bounds
            ])
        
        # Scipy minimizes, so negate the objective
        def neg_objective(params):
            return -objective_func(params)
        
        # Optimize
        result = minimize(
            neg_objective,
            initial_params,
            method=method,
            bounds=self.config.bounds,
            options={'disp': verbose, 'maxiter': self.config.max_iterations}
        )
        
        best_params = result.x
        best_fitness = -result.fun
        
        return best_params, best_fitness


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization using NSGA-II concept
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.pareto_front = []
        
    def dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (assuming maximization)"""
        return np.all(obj1 >= obj2) and np.any(obj1 > obj2)
    
    def find_pareto_front(self, population: np.ndarray, 
                         objectives: np.ndarray) -> List[int]:
        """Find indices of non-dominated solutions"""
        n = len(population)
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i != j and self.dominates(objectives[j], objectives[i]):
                    is_dominated[i] = True
                    break
        
        return np.where(~is_dominated)[0].tolist()
    
    def optimize(self, objective_funcs: List[Callable],
                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run multi-objective optimization
        
        Args:
            objective_funcs: List of objective functions
            verbose: Print progress
            
        Returns:
            Pareto optimal parameters and objectives
        """
        # Use genetic algorithm framework
        ga = GeneticAlgorithm(self.config)
        ga.initialize_population()
        
        iterator = tqdm(range(self.config.max_iterations)) if verbose else range(self.config.max_iterations)
        
        for generation in iterator:
            # Evaluate all objectives
            objectives = np.array([
                [func(ind) for func in objective_funcs]
                for ind in ga.population
            ])
            
            # Find non-dominated solutions
            pareto_indices = self.find_pareto_front(ga.population, objectives)
            
            if verbose and generation % 10 == 0:
                iterator.set_description(f"Gen {generation}: Pareto size={len(pareto_indices)}")
            
            # Standard GA evolution (using weighted sum for fitness)
            weights = np.ones(len(objective_funcs)) / len(objective_funcs)
            fitness = objectives @ weights
            
            # Evolution steps (same as GA)
            new_population = []
            
            # Elitism with pareto front
            for idx in pareto_indices[:min(10, len(pareto_indices))]:
                new_population.append(ga.population[idx])
            
            while len(new_population) < self.config.population_size:
                parents = ga.selection(fitness, 2)
                child1, child2 = ga.crossover(parents[0], parents[1])
                child1 = ga.mutation(child1)
                child2 = ga.mutation(child2)
                new_population.extend([child1, child2])
            
            ga.population = np.array(new_population[:self.config.population_size])
        
        # Final pareto front
        final_objectives = np.array([
            [func(ind) for func in objective_funcs]
            for ind in ga.population
        ])
        pareto_indices = self.find_pareto_front(ga.population, final_objectives)
        
        self.pareto_front = ga.population[pareto_indices]
        pareto_objectives = final_objectives[pareto_indices]
        
        return self.pareto_front, pareto_objectives


def create_objective_function(surrogate_model, flight_conditions, 
                              objective_type: str = 'maximize_ld',
                              target_cl: float = 0.5) -> Callable:
    """
    Create objective function for optimization
    
    Args:
        surrogate_model: Trained surrogate model
        flight_conditions: FlightConditions object
        objective_type: Type of objective
        target_cl: Target CL for constrained optimization
        
    Returns:
        Objective function
    """
    def objective(params: np.ndarray) -> float:
        # Add alpha to params (design point)
        alpha = 5.0  # Design angle of attack
        X = np.concatenate([params, [alpha]]).reshape(1, -1)
        
        # Predict performance
        prediction = surrogate_model.predict(X)[0]
        cl, cd, cm, ld = prediction
        
        if objective_type == 'maximize_ld':
            return ld
        elif objective_type == 'minimize_drag':
            return -cd
        elif objective_type == 'maximize_lift':
            return cl
        elif objective_type == 'target_cl':
            # Minimize difference from target while maximizing L/D
            cl_penalty = abs(cl - target_cl)
            return ld - 10 * cl_penalty
        else:
            return ld
    
    return objective


if __name__ == "__main__":
    print("Airfoil Optimization Module")
    print("=" * 50)
    
    # Example: Optimize a simple analytical function
    def rosenbrock(x):
        """Rosenbrock function for testing"""
        return -(100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2)
    
    config = OptimizationConfig(
        n_params=2,
        bounds=[(-5, 5), (-5, 5)],
        max_iterations=50,
        population_size=30
    )
    
    print("\nTesting Genetic Algorithm on Rosenbrock function...")
    ga = GeneticAlgorithm(config)
    best_params, best_fitness = ga.optimize(lambda x: -rosenbrock(x), verbose=True)
    print(f"\nGA Result: params={best_params}, fitness={best_fitness:.6f}")
    print(f"Expected: params=[1, 1], fitness=0.0")
    
    print("\nTesting Particle Swarm Optimization...")
    pso = ParticleSwarmOptimization(config)
    best_params, best_fitness = pso.optimize(lambda x: -rosenbrock(x), verbose=True)
    print(f"\nPSO Result: params={best_params}, fitness={best_fitness:.6f}")
    
    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # GA convergence
    ga_gens = [h['generation'] for h in ga.fitness_history]
    ga_best = [h['best_fitness'] for h in ga.fitness_history]
    ga_mean = [h['mean_fitness'] for h in ga.fitness_history]
    
    axes[0].plot(ga_gens, ga_best, 'b-', linewidth=2, label='Best')
    axes[0].plot(ga_gens, ga_mean, 'r--', linewidth=1.5, label='Mean')
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Fitness')
    axes[0].set_title('Genetic Algorithm Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PSO convergence
    pso_iters = [h['iteration'] for h in pso.fitness_history]
    pso_best = [h['best_fitness'] for h in pso.fitness_history]
    pso_mean = [h['mean_fitness'] for h in pso.fitness_history]
    
    axes[1].plot(pso_iters, pso_best, 'g-', linewidth=2, label='Best')
    axes[1].plot(pso_iters, pso_mean, 'm--', linewidth=1.5, label='Mean')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Fitness')
    axes[1].set_title('PSO Convergence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/airfoil_optimization/data/optimization_convergence.png', dpi=150)
    print("\nConvergence plots saved to data/optimization_convergence.png")