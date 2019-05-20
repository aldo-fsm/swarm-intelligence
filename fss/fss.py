import numpy as np
import pandas as pd

class FSS:
    def __init__(self, num_particles, num_dims, initial_weight, ind_step_range, vol_step_range, step_decay_iterations=0, weight_range=[-np.inf, np.inf], constraint=None, initializer=None, keep_history=False):
        self.num_particles = num_particles
        self.num_dims = num_dims
        self.initial_weight = initial_weight
        self.ind_step_range = ind_step_range
        self.vol_step_range = vol_step_range
        self.step_decay_iterations = step_decay_iterations
        self.weight_range = weight_range
        self.iteration = -1
        self.constraint = constraint# if constraint else lambda x: True
        self.initializer = initializer
        self.keep_history = keep_history
        self.fitness_evaluations = 0

        if self.constraint:
            raise NotImplementedError('Restrições ainda não foram implementadas')

    def isInitialized(self):
        return self.iteration > -1

    def getBestSolution(self):
        return self.best_solution

    def _get_vol_step(self):
        initial, final = self.vol_step_range
        if self.step_decay_iterations > 0 and self.iteration < self.step_decay_iterations:
            return initial - self.iteration*(initial - final)/self.step_decay_iterations
        else:
            return final

    def _get_ind_step(self):
        initial, final = self.ind_step_range
        if self.step_decay_iterations > 0 and self.iteration < self.step_decay_iterations:
            return initial - self.iteration*(initial - final)/self.step_decay_iterations
        else:
            return final

    def initialize(self, fitness_function, initializer=None):
        if not initializer:
            initializer = self.initializer
        assert initializer


        self.fitness_function = self._fitnessCounter(fitness_function)
        self.iteration = 0
        self.fitness_evaluations = 0
        self.pos = initializer(self.num_particles, self.num_dims)
        self.best_solution = np.array([np.NaN]*self.num_dims), np.inf
        self.weights = np.ones((self.num_particles, 1))*self.initial_weight
        self.history = pd.DataFrame(columns=['best_fitness', 'fitness_evaluations', 'iterations'])

    def _fitnessCounter(self, fitness_func):
        def f(x):
            self.fitness_evaluations += 1
            fitness = fitness_func(x)
            if fitness < self.getBestSolution()[1]:
                self.best_solution = x, fitness
            if self.keep_history:
                self.history = self.history.append({
                    'best_fitness': self.getBestSolution()[1],
                    'fitness_evaluations': self.fitness_evaluations,
                    'iterations': self.iteration
                }, ignore_index=True)
            return fitness
        return f

    def barycenter(self):
        return np.sum(self.pos*self.weights, axis=0)/np.sum(self.weights)

    def schoolWeight(self):
        return np.sum(self.weights)

    def minimize(self):
        assert self.isInitialized()
        
        fitness_before = np.apply_along_axis(self.fitness_function, 1, self.pos)[:, None]
        new_pos = self.pos + self._get_ind_step()*np.random.uniform(-1, 1, size=self.pos.shape)
        fitness_after = np.apply_along_axis(self.fitness_function, 1, new_pos)[:, None]
        delta_fitness = -(fitness_after - fitness_before) # Minimization -> lower is better
        better = delta_fitness > 0
        new_pos = np.where(better, new_pos, self.pos)
        delta_pos = new_pos - self.pos
        self.pos = new_pos

        weight_before = self.schoolWeight()
        self.weights = self.weights + delta_fitness/np.max(delta_fitness)
        self.weights[self.weights < self.weight_range[0]] = self.weight_range[0]
        self.weights[self.weights > self.weight_range[1]] = self.weight_range[1]
        weight_after = self.schoolWeight()
        vol_signal = -1 if weight_after > weight_before else 1

        delta_fitness_sum = np.sum(np.abs(delta_fitness))
        if delta_fitness_sum != 0:
            self.pos = self.pos + np.sum(delta_pos*delta_fitness, axis=0)/delta_fitness_sum 

        B = self.barycenter()
        barycenterDistVector = self.pos - B
        rand_vec = np.random.uniform(0, 1, size=self.pos.shape)
        self.pos = self.pos + vol_signal*self._get_vol_step()*rand_vec*barycenterDistVector/np.linalg.norm(barycenterDistVector, axis=1)[:, None]

        self.iteration += 1