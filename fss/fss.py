
class FSS:
    def __init__(self, num_particles, num_dims, constraint=None, initializer=None):
        self.num_particles = num_particles
        self.num_dims = num_dims
        self.initializer = initializer
        self.iteration = -1
        self.constraint = constraint if constraint else lambda x: True
        self.fitness_evaluations = 0

    def isInitialized(self):
        return self.iteration > -1

    def getBestSolution(self):
        raise NotImplementedError()

    def initialize(self, initializer=None):
        if not initializer:
            initializer = self.initializer
        assert initializer

        self.iteration = 0
        self.fitness_evaluations = 0
        self.pos = initializer(self.num_particles, self.num_dims)

    def _fitnessCounter(self, fitness_func):
        def f(x):
            self.fitness_evaluations += 1
            return fitness_func(x)
        return f

    def minimize(self, fitness_func):
        assert self.isInitialized()
        fitness_func = self._fitnessCounter(fitness_func)
        
        # fitness = np.apply_along_axis(fitness_func, 1, self.pos)[:, None]
        # better = fitness < self.pbest_fitness
        # satisfy_constraints = np.apply_along_axis(self.constraint, 1, self.pos)[:, None]
        # should_update = np.logical_and(better, satisfy_constraints)
        # self.pbest_fitness = np.where(should_update, fitness, self.pbest_fitness)
        # self.pbest = np.where(should_update, self.pos, self.pbest)

        # lbest = self.pbest[self.topology(self.pbest_fitness)]
        
        # r1, r2 = [np.random.uniform(0, 1, self.num_particles)[:, None] for _ in range(2)]
        # self.vel = self.clerc_factor*(self._get_w()*self.vel + self.c1*r1*(self.pbest - self.pos) + self.c2*r2*(lbest - self.pos))
        # self.pos = self.pos + self.vel