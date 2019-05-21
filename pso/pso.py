import numpy as np
import pandas as pd

class PSO:
    def __init__(self, num_particles, num_dims, topology, c1=2.05, c2=2.05, clerc_factor=1, initial_w=1, final_w=1, w_decay_iterations=0, constraint=None, initializer=None, keep_history=False):
        self.num_particles = num_particles
        self.num_dims = num_dims
        self.c1 = c1
        self.c2 = c2
        self.initializer = initializer
        self.iteration = -1
        self.clerc_factor = clerc_factor
        self.initial_w = initial_w
        self.final_w = final_w
        self.w_decay_iterations = w_decay_iterations
        self.topology = topology
        self.constraint = constraint if constraint else lambda x: True
        self.fitness_evaluations = 0
        self.keep_history = keep_history

    def isInitialized(self):
        return self.iteration > -1

    def _get_w(self):
        if self.w_decay_iterations > 0 and self.iteration < self.w_decay_iterations:
            return self.initial_w - self.iteration*(self.initial_w - self.final_w)/self.w_decay_iterations
        else:
            return self.final_w
    

    def _fitnessCounter(self, fitness_func):
        def f(x):
            self.fitness_evaluations += 1
            fitness = fitness_func(x)
            if fitness < self.getBestSolution()[1]:
                self.best_solution = x, fitness
            # if self.keep_history:
            #     self.history.append([
            #         self.getBestSolution()[1], # best_fitness
            #         self.fitness_evaluations,  # fitness_evaluations
            #         self.iteration             # iterations
            #     ])
            return fitness
        return f
    
    def getHistory(self):
        return pd.DataFrame(self.history, columns=['best_fitness', 'fitness_evaluations', 'iterations'])

    def getBestSolution(self):
        return self.best_solution

    def initialize(self, fitness_function, initializer=None):
        if not initializer:
            initializer = self.initializer
        assert initializer

        self.fitness_function = self._fitnessCounter(fitness_function)
        self.iteration = 0
        self.fitness_evaluations = 0
        self.pos = initializer(self.num_particles, self.num_dims)
        self.vel = np.zeros(self.pos.shape)
        self.pbest = self.pos
        self.pbest_fitness = (np.ones(len(self.pbest))*np.inf)[:, None]
        self.best_solution = np.array([np.NaN]*self.num_dims), np.inf
        self.history = []

    def minimize(self):
        assert self.isInitialized()
        fitness = np.apply_along_axis(self.fitness_function, 1, self.pos)[:, None]
        better = fitness < self.pbest_fitness
        satisfy_constraints = np.apply_along_axis(self.constraint, 1, self.pos)[:, None]
        should_update = np.logical_and(better, satisfy_constraints)
        self.pbest_fitness = np.where(should_update, fitness, self.pbest_fitness)
        self.pbest = np.where(should_update, self.pos, self.pbest)

        lbest = self.pbest[self.topology(self.pbest_fitness)]
        
        r1, r2 = [np.random.uniform(0, 1, self.num_particles)[:, None] for _ in range(2)]
        self.vel = self.clerc_factor*(self._get_w()*self.vel + self.c1*r1*(self.pbest - self.pos) + self.c2*r2*(lbest - self.pos))
        self.pos = self.pos + self.vel
        self.iteration += 1
        if self.keep_history:
            self.history.append([
                self.getBestSolution()[1], # best_fitness
                self.fitness_evaluations,  # fitness_evaluations
                self.iteration             # iterations
            ])