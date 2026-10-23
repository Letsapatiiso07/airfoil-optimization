"""
Real-coded genetic algorithm. It maximises a fitness function that takes a
batch of parameter vectors (n, d) and returns n values, so a surrogate can
score a whole population in one call.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class OptimizationConfig:
    bounds: List[Tuple[float, float]]
    max_iterations: int = 100
    population_size: int = 50
    seed: int = 0


class GeneticAlgorithm:
    """Real-coded GA: tournament selection, SBX crossover, polynomial
    mutation and elitism."""

    def __init__(self, config: OptimizationConfig, eta_c: float = 15.0,
                 eta_m: float = 20.0, mutation_rate: float = None):
        self.config = config
        self.low = np.array([b[0] for b in config.bounds])
        self.high = np.array([b[1] for b in config.bounds])
        self.dim = len(self.low)
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.mutation_rate = mutation_rate or 1.0 / self.dim
        self.rng = np.random.default_rng(config.seed)
        self.history = []
        self.n_evaluations = 0

    def _tournament(self, fitness, k=3):
        idx = self.rng.integers(0, len(fitness), size=k)
        return idx[np.argmax(fitness[idx])]

    def _crossover(self, p1, p2):
        u = self.rng.random(self.dim)
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (self.eta_c + 1)),
                        (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1)))
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        return np.clip(c1, self.low, self.high), np.clip(c2, self.low, self.high)

    def _mutate(self, x):
        x = x.copy()
        for i in np.flatnonzero(self.rng.random(self.dim) < self.mutation_rate):
            u = self.rng.random()
            dq = ((2 * u) ** (1 / (self.eta_m + 1)) - 1 if u < 0.5
                  else 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1)))
            x[i] += dq * (self.high[i] - self.low[i])
        return np.clip(x, self.low, self.high)

    def optimize(self, fitness_fn: Callable[[np.ndarray], np.ndarray],
                 initial: np.ndarray = None) -> Tuple[np.ndarray, float]:
        n = self.config.population_size
        pop = self.low + self.rng.random((n, self.dim)) * (self.high - self.low)
        if initial is not None:
            pop[0] = initial
        fit = fitness_fn(pop)
        self.n_evaluations += n
        n_elite = max(2, n // 10)
        for gen in range(self.config.max_iterations):
            self.history.append({'generation': gen, 'best': float(fit.max()),
                                 'mean': float(np.mean(fit))})
            children = []
            while len(children) < n - n_elite:
                p1 = pop[self._tournament(fit)]
                p2 = pop[self._tournament(fit)]
                for c in self._crossover(p1, p2):
                    children.append(self._mutate(c))
            children = np.array(children[:n - n_elite])
            child_fit = fitness_fn(children)
            self.n_evaluations += len(children)
            elite = np.argsort(fit)[-n_elite:]
            pop = np.vstack([pop[elite], children])
            fit = np.concatenate([fit[elite], child_fit])
        self.population, self.fitness = pop, fit
        best = int(np.argmax(fit))
        return pop[best].copy(), float(fit[best])
