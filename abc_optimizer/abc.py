import numpy as np
import pandas as pd
from initializers import UniformInitializer

class ABC:
    def __init__(self, num_particles, num_dims, attempts_to_exhaust, constraint=None, keep_history=False):
        self.num_particles = num_particles
        self.num_employed = int(np.ceil(num_particles/2))
        self.num_onlooker = num_particles - self.num_employed
        self.num_dims = num_dims
        self.attempts_to_exhaust = attempts_to_exhaust
        self.iteration = -1
        self.constraint = constraint# if constraint else lambda x: True
        self.fitness_evaluations = 0
        self.keep_history = keep_history
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
        if len(foodsources_indexes) > 0:
            foodsources = self.employed[foodsources_indexes]
            new_values = self.initializer(*foodsources.shape)
            selected_dims = np.random.choice(self.num_dims, size=foodsources.shape[0])
            foodsources[range(foodsources.shape[0]), selected_dims] = new_values[range(foodsources.shape[0]), selected_dims]
            new_fitness = np.apply_along_axis(self.fitness_function, 1, foodsources)[:, None]
            better = new_fitness < self.foodsources_fitness[foodsources_indexes]
            self.employed[foodsources_indexes] = np.where(better, foodsources, self.employed[foodsources_indexes])
            self.foodsources_fitness[foodsources_indexes] = np.where(better, new_fitness, self.foodsources_fitness[foodsources_indexes])
            self.attempts_remaining[foodsources_indexes] -= np.logical_not(better).reshape(-1)

    def initialize(self, fitness_function, search_space):
        self.fitness_function = self._fitnessCounter(fitness_function)
        self.search_space = search_space
        self.initializer = UniformInitializer(search_space)
        self.iteration = 0
        self.fitness_evaluations = 0
        self.best_solution = np.array([np.NaN]*self.num_dims), np.inf
        self.employed = np.zeros((self.num_employed, self.num_dims))
        self.foodsources_fitness = np.ones((self.num_employed, 1))*np.inf
        self.attempts_remaining = np.zeros(self.num_employed)
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

    def adjust_fitness(self, fitness): 
        return fitness - fitness.max() - fitness.min()
    
    def adjust_fitness2(self, fitness): 
        return np.where(fitness > 0, 1/(1+fitness), 1+np.abs(fitness))

    def evaluateFoodsources(self, foodsources_indexes):
        if len(foodsources_indexes) > 0:
            self.foodsources_fitness[foodsources_indexes] = np.apply_along_axis(self.fitness_function, 1, self.employed[foodsources_indexes])[:, None]

    def minimize(self):
        assert self.isInitialized()
        
        # Scouts bees find new foodplaces
        new_foodsources_selection = self.findNewFoodSources()
        
        # Employed bees evaluate new foodsources quality and do a greedy search
        self.evaluateFoodsources(np.arange(self.num_employed)[new_foodsources_selection])
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