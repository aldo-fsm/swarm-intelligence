import numpy as np
from pso.initializers import UniformInitializer

def test_algorithms(algorithms, objective_function, num_experiments, num_iterations, search_bounds):
    results = []
    for experiment in range(num_experiments):
        print('Experiment {}/{}'.format(experiment+1, num_experiments))
        algorithm_results = []
        for algorithm in algorithms:
            algorithm.initialize(UniformInitializer(search_bounds))
            curve = []
            for _ in range(num_iterations):
                algorithm.minimize(objective_function)
                curve.append(algorithm.getBestSolution()[1][0])
            algorithm_results.append(curve)
        results.append(algorithm_results)
    return np.array(results)