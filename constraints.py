import numpy as np

class BoxConstraint:
    def __init__(self, bounds):
        self.bounds = bounds

    def __call__(self, position):
        lower_bound, upper_bound = zip(*self.bounds)
        return np.logical_and(position >= lower_bound, position < upper_bound)