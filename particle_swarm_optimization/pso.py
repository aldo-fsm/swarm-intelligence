import numpy as np

class PSO:
    def __init__(self, num_particles, num_dims, initializer, communication_func, c1=2.05, c2=2.05, clerc_factor=1, initial_w=0, final_w=0, w_decay_iterations=0):
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
        self.communication_func = communication_func

    def isInitialized(self):
        return self.iteration > -1

    def _get_w(self):
        if self.w_decay_iterations > 0 and self.iteration < self.w_decay_iterations:
            return self.initial_w - self.iteration*(self.initial_w - self.final_w)/self.w_decay_iterations
        else:
            return self.final_w

    def initialize(self):
        self.iteration = 0
        self.pos = self.initializer(self.num_particles, self.num_dims)
        self.vel = np.zeros(self.pos.shape)
        self.pbest = self.pos
        self.pbest_fitness = np.ones(len(self.pbest))*np.inf

    def minimize(self, fitness_func):
        fitness = np.array([fitness_func(x) for x in self.pos])
        better = fitness < self.pbest_fitness
        self.pbest_fitness = np.where(better, fitness, self.pbest_fitness)
        self.pbest = np.where(better, self.pos, self.pbest)

        lbest = self.pbest[self.communication_func(self.pbest_fitness)]

        r1, r2 = [np.random.uniform(0, 1, self.num_particles) for _ in range(2)]
        self.vel = self._get_w()*self.vel + self.c1*r1*(self.pbest - self.pos) + self.c2*r2*(lbest - self.pos)
        self.pos = self.pos + self.clerc_factor*self.vel