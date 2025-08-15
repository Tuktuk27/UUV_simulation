import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Any, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
import json
from tqdm import trange
import os
from itertools import islice

logger = logging.getLogger(__name__)

# Helper for parallel evaluation with error handling
def _parallel_evaluate(factory, fitness_fn, position) -> float:
    try:
        controller = factory(position)
        return fitness_fn(controller)
    except Exception as e:
        logger.error(f"Parallel eval failed: {e}")
        return np.inf

class ControllerBase(ABC):
    @abstractmethod
    def compute_actuator_force(self, *states: Any) -> float:
        pass

class Particle:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, bounds: np.ndarray):
        self.position = position
        self.velocity = velocity
        self.bounds = bounds  # shape (n_params, 2)
        self.best_position = position.copy()
        self.best_cost = np.inf

    def clip_position(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        self.position = np.clip(self.position, low, high)

    def clip_velocity(self, iter_idx: int, num_iterations: int, max_frac: float = 0.1):
        """Dynamically reduce velocity range over time"""
        frac = max_frac * (1 - iter_idx / max(1, num_iterations))
        ranges = self.bounds[:,1] - self.bounds[:,0]
        ranges = np.where(ranges > 0, ranges, 1e-6)
        max_vel = ranges * frac
        self.velocity = np.clip(self.velocity, -max_vel, max_vel)

class PSOOptimizer:
    def __init__(
        self,
        controller_factory: Callable[[np.ndarray], ControllerBase],
        fitness_function: Callable[[ControllerBase], float],
        param_bounds: List[Tuple[float, float]],
        num_particles: int = 30,
        num_iterations: int = 100,
        inertia: float = 0.7,
        cognitive: float = 1.4,
        social: float = 1.4,
        adaptive: bool = False,
        early_stop: bool = False,
        stop_tol: float = 1e-4,
        parallel: bool = False,
        random_seed: Optional[int] = None,
        verbose: bool = True
    ):
        if random_seed is not None:
            np.random.seed(random_seed)
        self.factory = controller_factory
        self.fitness_fn = fitness_function
        self.bounds = np.array(param_bounds)
        self.dim = len(param_bounds)
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.initial_inertia = inertia
        self.initial_cognitive = cognitive
        self.initial_social = social
        self.adaptive = adaptive
        self.early_stop = early_stop
        self.stop_tol = stop_tol
        self.parallel = parallel
        self.verbose = verbose
        self.swarm: List[Particle] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_cost: float = np.inf
        self.history: List[Dict[str, Any]] = []

    def initialize_swarm(self):
        self.swarm = []
        ranges = self.bounds[:,1] - self.bounds[:,0]
        ranges = np.where(ranges>0, ranges, 1e-6)
        for _ in range(self.num_particles):
            pos = np.random.uniform(self.bounds[:,0], self.bounds[:,1])
            vel = np.random.uniform(-1,1,size=self.dim) * ranges * 0.05
            particle = Particle(pos, vel, self.bounds)
            self.swarm.append(particle)

    def _adapt_weights(self, iter_idx: int) -> Dict[str, float]:
        if not self.adaptive:
            return {'w': self.initial_inertia, 'c1': self.initial_cognitive, 'c2': self.initial_social}
        t = iter_idx / max(1, self.num_iterations - 1)
        w = max(0.3, self.initial_inertia - 0.4 * t)
        c1 = max(0.5, self.initial_cognitive * (1 - t))
        c2 = min(2.0, self.initial_social * (1 + 0.5 * t))
        return {'w': w, 'c1': c1, 'c2': c2}
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        self.initialize_swarm()
        prev_best = np.inf
        
        # Create executor once if parallel
        if self.parallel:
            executor = ProcessPoolExecutor(max_workers=os.cpu_count())
        
        for iter_idx in trange(self.num_iterations, desc="PSO iters"):
            print(f"\n\n\n\n\n{iter_idx = }\n\n\n\n\n")
            
            # Evaluate fitness at CURRENT positions
            if self.parallel:
                costs = list(executor.map(
                    _parallel_evaluate,
                    [self.factory] * self.num_particles,
                    [self.fitness_fn] * self.num_particles,
                    [p.position for p in self.swarm]
                ))
            else:
                costs = [self.fitness_fn(self.factory(p.position)) for p in self.swarm]

            # Update personal and global bests
            for p, cost in zip(self.swarm, costs):
                if cost < p.best_cost:
                    p.best_cost = cost
                    p.best_position = p.position.copy()
                if cost < self.global_best_cost:
                    self.global_best_cost = cost
                    self.global_best_position = p.position.copy()

            # Record history and diversity
            positions = np.array([p.position for p in self.swarm])
            diversity = float(np.mean(np.std(positions, axis=0)))
            self.history.append({
                'iteration': iter_idx,
                'best_cost': self.global_best_cost,
                'diversity': diversity
            })

            # Early stopping check
            if self.early_stop and iter_idx > 5:
                if abs(prev_best) > 1e-9:
                    improvement = (prev_best - self.global_best_cost) / abs(prev_best)
                else:
                    improvement = np.inf
                if improvement < self.stop_tol:
                    if self.verbose:
                        print(f"Stopping early at iter {iter_idx}, improvement {improvement:.2e} < tol")
                    break
                prev_best = self.global_best_cost

            # Update positions for NEXT iteration (except after last iteration)
            if iter_idx < self.num_iterations - 1:
                weights = self._adapt_weights(iter_idx)
                for p in self.swarm:
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    p.velocity = (weights['w'] * p.velocity +
                                weights['c1'] * r1 * (p.best_position - p.position) +
                                weights['c2'] * r2 * (self.global_best_position - p.position))
                    p.clip_velocity(iter_idx, self.num_iterations)
                    p.position += p.velocity
                    p.clip_position()

            # Log progress
            if self.verbose and iter_idx % max(1, self.num_iterations//10) == 0:
                logger.info(f"Iter {iter_idx+1}/{self.num_iterations}, best cost {self.global_best_cost:.6f}")

        # Final logging and cleanup
        if self.verbose:
            logger.info(f"Optimization completed. Best cost: {self.global_best_cost:.6f}")
            logger.info("Best parameters:")
            for i, v in enumerate(self.global_best_position):
                logger.info(f"  Param {i}: {v:.6f}")
        
        if self.parallel:
            executor.shutdown()
        
        return self.global_best_position, self.global_best_cost

    # def optimize(self) -> Tuple[np.ndarray, float]:
    #     self.initialize_swarm()
    #     prev_best = np.inf

    #     if self.parallel:    
    #         # Create your pool just once
    #         executor = ThreadPoolExecutor(max_workers=os.cpu_count())
    #     for iter_idx in trange(self.num_iterations, desc="PSO iters"):
    #         print(f"\n\n\n\n\n{iter_idx = }\n\n\n\n\n")
    #         # Evaluate fitness
    #         if self.parallel:
    #             costs = list(executor.map(
    #                 _parallel_evaluate,
    #                 [self.factory]*self.num_particles,
    #                 [self.fitness_fn]*self.num_particles,
    #                 [p.position for p in self.swarm]
    #             ))
    #         else:
    #             costs = [self.fitness_fn(self.factory(p.position)) for p in self.swarm]

    #         # Update bests
    #         for p, cost in zip(self.swarm, costs):
    #             if cost < p.best_cost:
    #                 p.best_cost = cost
    #                 p.best_position = p.position.copy()
    #             if cost < self.global_best_cost:
    #                 self.global_best_cost = cost
    #                 self.global_best_position = p.position.copy()

    #         # Record history and diversity
    #         positions = np.array([p.position for p in self.swarm])
    #         diversity = float(np.mean(np.std(positions, axis=0)))
    #         self.history.append({'iteration': iter_idx,
    #                               'best_cost': self.global_best_cost,
    #                               'diversity': diversity})

    #         # Early stopping
    #         if self.early_stop and iter_idx > 5:
    #             if abs(prev_best) > 1e-9:
    #                 improvement = (prev_best - self.global_best_cost) / abs(prev_best)
    #             else:
    #                 improvement = np.inf
    #             if improvement < self.stop_tol:
    #                 if self.verbose:
    #                     print(f"Stopping early at iter {iter_idx}, improvement {improvement:.2e} < tol")
    #                 break
    #             prev_best = self.global_best_cost

    #         # Update velocities & positions
    #         weights = self._adapt_weights(iter_idx)
    #         for p in self.swarm:
    #             r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)

    #             p.velocity = (weights['w']*p.velocity
    #                           + weights['c1']*r1*(p.best_position - p.position)
    #                           + weights['c2']*r2*(self.global_best_position - p.position))
    #             p.clip_velocity(iter_idx, self.num_iterations)
    #             p.position += p.velocity
    #             p.clip_position()

    #         if self.verbose and iter_idx % max(1, self.num_iterations//10) == 0:
    #             logger.info(f"Iter {iter_idx+1}/{self.num_iterations}, best cost {self.global_best_cost:.6f}")

    #     # Final logging
    #     if self.verbose:
    #         logger.info(f"Optimization completed. Best cost: {self.global_best_cost:.6f}")
    #         logger.info("Best parameters:")
    #         for i, v in enumerate(self.global_best_position):
    #             logger.info(f"  Param {i}: {v:.6f}")

    #     if self.parallel:
    #         executor.shutdown()
    #     return self.global_best_position, self.global_best_cost

    def plot_swarm(self, x_idx: int = 0, y_idx: int = 1):
        xs = [p.position[x_idx] for p in self.swarm]
        ys = [p.position[y_idx] for p in self.swarm]
        plt.scatter(xs, ys)
        plt.xlabel(f"Param {x_idx}")
        plt.ylabel(f"Param {y_idx}")
        plt.title("Swarm Positions")
        plt.show()

    def save_history(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)

    def save_swarm(self, filename: str):
        state = {
            'positions': [p.position.tolist() for p in self.swarm],
            'velocities': [p.velocity.tolist() for p in self.swarm],
            'best_costs': [p.best_cost for p in self.swarm],
            'global_best': self.global_best_position.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)

    def save_results(self, filename: str):
        """Save best parameters and cost to a text file."""
        with open(filename, 'w') as f:
            f.write(f"Best cost: {self.global_best_cost:.6f}")
            f.write("Best parameters:")
            for i, param in enumerate(self.global_best_position):
                f.write(f"Param {i}: {param:.6f}")

    def plot_convergence(self):
        """Plot the convergence curve based on best cost history."""
        plt.figure()
        costs = [h['best_cost'] if isinstance(h, dict) else h for h in self.history]
        plt.plot(costs)
        plt.xlabel('Iteration')
        plt.ylabel('Best Cost')
        plt.title('PSO Convergence')
        plt.grid(True)
        plt.show()

# Usage placeholder:
# optimizer = PSOOptimizer(...)
# best_params, best_cost = optimizer.optimize()
