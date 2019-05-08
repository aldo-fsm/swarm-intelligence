import numpy as np

class UniformInitializer:
    def __init__(self, bounds):
        self.bounds = bounds

    def __call__(self, num_particles, num_dims):
        return np.random.uniform(*zip(*self.bounds), (num_particles, num_dims))

class NormalInitializer:
    def __init__(self, mean=0, sigma=1):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, num_particles, num_dims):
        return self.mean + self.sigma*np.random.randn(num_particles, num_dims)