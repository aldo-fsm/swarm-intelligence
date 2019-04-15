import numpy as np

class GlobalTopology:
    def __call__(self, pbest_fitness):
        best_index = np.argmin(pbest_fitness)
        return np.repeat(best_index, len(pbest_fitness))

class LocalTopology:
    def __call__(self, pbest_fitness):
        pbest_fitness = np.array(pbest_fitness)
        indexes = np.arange(len(pbest_fitness))
        result = []
        for i in range(len(pbest_fitness)):
            window = indexes[np.mod(range(i-1,i+2), len(pbest_fitness))]
            result.append(window[np.argmin(pbest_fitness[window])])
        return np.array(result)

class FocalTopology:
    def __call__(self, pbest_fitness):
        focus_index = 0
        result = []
        for i in range(len(pbest_fitness)):
            if i == focus_index:
                result.append(np.argmin(pbest_fitness))
            else:
                result.append(focus_index if pbest_fitness[focus_index] < pbest_fitness[i] else i)
        return np.array(result)