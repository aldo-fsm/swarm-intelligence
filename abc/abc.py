import numpy as np
from initializers import UniformInitializer

class ABC:
    def __init__(self, num_particles, num_dims, search_space, attempts_to_exhaust, constraint=None):
        self.num_particles = num_particles
        self.num_employed = int(np.ceil(num_particles/2))
        self.num_onlooker = num_particles - self.num_employed
        self.num_dims = num_dims
        self.search_space = search_space
        self.attempts_to_exhaust = attempts_to_exhaust
        self.iteration = -1
        self.constraint = constraint# if constraint else lambda x: True
        self.fitness_evaluations = 0
        if self.constraint:
            raise NotImplementedError('Restrições ainda não foram implementadas')

    def isInitialized(self):
        return self.iteration > -1

    def getBestSolution(self):
        return self.best_solution

    def findNewFoodSources(self):
        initializer = UniformInitializer(self.search_space)
        selection = self.attempts_remaining == 0
        num_selected = np.count_nonzero(selection)
        self.employed[selection] = initializer(num_selected, self.num_dims)
        self.attempts_remaining[selection] = np.ones(num_selected)*self.attempts_to_exhaust

    def initialize(self):
        self.iteration = 0
        self.fitness_evaluations = 0
        self.best_solution = np.array([np.NaN]*self.num_dims), np.inf
        self.employed = np.zeros((self.num_employed, self.num_dims))
        self.attempts_remaining = np.zeros(self.num_employed)

    def _fitnessCounter(self, fitness_func):
        def f(x):
            self.fitness_evaluations += 1
            return fitness_func(x)
        return f

    def adjust_fitness(x): 
        return x - x.max() - x.min()
    
    def adjust_fitness2(x): 
        return np.where(x > 0, 1/(1+x), 1+np.abs(x))

    def minimize(self, fitness_func):
        assert self.isInitialized()
        fitness_func = self._fitnessCounter(fitness_func)
        
        self.findNewFoodSources()

        fitness = np.apply_along_axis(fitness_func, 1, self.employed)[:, None]
        fitness_adjusted = self.adjust_fitness(fitness)
        foodsource_proba = fitness_adjusted/fitness_adjusted.sum()
        random_selector = np.random.uniform(0, 1, size=(self.num_onlooker, self.num_employed))
        num_attempts_per_foodsource = np.count_nonzero(random_selector < foodsource_proba, axis=0) + 1
        for i, num_attempts in enumerate(num_attempts_per_foodsource):
            pass


        # new_pos = self.pos + self._get_ind_step()*np.random.uniform(-1, 1, size=self.pos.shape)
        # fitness_after = np.apply_along_axis(fitness_func, 1, new_pos)[:, None]
        # delta_fitness = -(fitness_after - fitness_before) # Minimization -> lower is better
        # better = delta_fitness > 0
        # new_pos = np.where(better, new_pos, self.pos)
        # delta_pos = new_pos - self.pos
        # self.pos = new_pos

        # weight_before = self.schoolWeight()
        # self.weights = self.weights + delta_fitness/np.max(delta_fitness)
        # self.weights[self.weights < self.weight_range[0]] = self.weight_range[0]
        # self.weights[self.weights > self.weight_range[1]] = self.weight_range[1]
        # weight_after = self.schoolWeight()
        # vol_signal = -1 if weight_after > weight_before else 1

        # delta_fitness_sum = np.sum(np.abs(delta_fitness))
        # if delta_fitness_sum != 0:
        #     self.pos = self.pos + np.sum(delta_pos*delta_fitness, axis=0)/delta_fitness_sum 

        # B = self.barycenter()
        # barycenterDistVector = self.pos - B
        # rand_vec = np.random.uniform(0, 1, size=self.pos.shape)
        # self.pos = self.pos + vol_signal*self._get_vol_step()*rand_vec*barycenterDistVector/np.linalg.norm(barycenterDistVector, axis=1)[:, None]

        # fitness = np.where(better, fitness_after, fitness_before)
        # best_index = np.argmin(fitness)
        # if fitness[best_index] < self.getBestSolution()[1]:
        #     self.best_solution = self.pos[best_index], fitness[best_index]