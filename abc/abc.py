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
        self.initializer = UniformInitializer(self.search_space)
        if self.constraint:
            raise NotImplementedError('Restrições ainda não foram implementadas')

    def isInitialized(self):
        return self.iteration > -1

    def getBestSolution(self):
        return self.best_solution

    def findNewFoodSources(self):
        selection = self.attempts_remaining == 0
        num_selected = np.count_nonzero(selection)
        self.employed[selection] = self.initializer(num_selected, self.num_dims)
        self.attempts_remaining[selection] = np.ones(num_selected)*self.attempts_to_exhaust
        return selection

    def greedySearch(self, foodsources_indexes):
        foodsources = self.employed[foodsources_indexes]
        new_values = self.initializer(*foodsources.shape)
        selected_dims = np.random.choice(self.num_dims, size=foodsources.shape[0])
        foodsources[range(foodsources.shape[0]), selected_dims] = new_values[range(foodsources.shape[0]), selected_dims]
        new_fitness = np.apply_along_axis(self.fitness_function, 1, foodsources)[:, None]
        better = new_fitness < self.foodsources_fitness
        self.employed[foodsources_indexes] = np.where(better, foodsources, self.employed[foodsources_indexes])
        self.foodsources_fitness[foodsources_indexes] = np.where(better, new_fitness, self.foodsources_fitness[foodsources_indexes])
        self.attempts_remaining[foodsources_indexes] -= np.logical_not(better)

    def initialize(self, fitness_function):
        self.fitness_function = self._fitnessCounter(fitness_function)
        self.iteration = 0
        self.fitness_evaluations = 0
        self.best_solution = np.array([np.NaN]*self.num_dims), np.inf
        self.employed = np.zeros((self.num_employed, self.num_dims))
        self.foodsources_fitness = np.ones((self.num_employed, 1))*np.inf
        self.attempts_remaining = np.zeros(self.num_employed)

    def _fitnessCounter(self, fitness_func):
        def f(x):
            self.fitness_evaluations += 1
            fitness = fitness_func(x)
            if fitness < self.getBestSolution()[1]:
                self.best_solution = x, fitness
            return fitness
        return f

    def adjust_fitness(self, fitness): 
        return fitness - fitness.max() - fitness.min()
    
    def adjust_fitness2(self, fitness): 
        return np.where(fitness > 0, 1/(1+fitness), 1+np.abs(fitness))

    def minimize(self):
        assert self.isInitialized()
        
        # Scouts bees find new foodplaces
        new_foodsources_selection = self.findNewFoodSources()
        
        # Employed bees evaluate new foodsources quality and do a greedy search
        self.foodsources_fitness[new_foodsources_selection] = np.apply_along_axis(self.fitness_function, 1, self.employed[new_foodsources_selection])[:, None]
        fitness = self.foodsources_fitness
        self.greedySearch(np.arange(self.num_employed))

        # Onlooker bees choose foodsources and do a greedy search
        fitness_adjusted = self.adjust_fitness(fitness)
        foodsource_proba = fitness_adjusted/fitness_adjusted.sum()
        random_selector = np.random.uniform(0, 1, size=(self.num_onlooker, self.num_employed))
        for selection in (random_selector < foodsource_proba):
            selection = selection & (self.attempts_remaining > 0)
            selected_foodsources_indexes = np.arange(self.num_employed)[selection]
            self.greedySearch(selected_foodsources_indexes)        
        
        self.iteration += 1