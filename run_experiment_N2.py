import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from pso import PSO
from pso.topology import LocalTopology
from fss import FSS
from abc_optimizer import ABC
from test_functions import sphereFunction, rastriginsFunction, rosenbrockFunction
from initializers import UniformInitializer
'''
PSO:

• 30 partículas;
• 30 dimensões;
• 500.000 avaliações de fitness;
• c1 = c2 = 2,05.
• Topologia local;
• w decaindo de 0,9 até 0,4 ao longo das iterações.

ABC:

• Tamanho da colônia igual a 30;
• 30 dimensões;
• 500.000 avaliações de fitness;
• Número de tentativas igual a 100.

FSS:

• 30 peixes;
• 30 dimensões;
• 500.000 avaliações de fitness;
• Step individual decaindo de 0.1 a 0.001;
• Step volitivo decaindo de 0.01 a 0.001;
• Peso mínimo igual a 1 e sem peso máximo.
'''

num_simulations = 30
num_fitness_evaluations = 500000
num_particles = 30
num_dims = 30

pso = PSO(
    num_particles=num_particles,
    num_dims=num_dims,
    c1=2.05,
    c2=2.05,
    topology=LocalTopology(),
    initial_w=0.9,
    final_w=0.4,
    w_decay_iterations=num_fitness_evaluations//num_particles, # 1 evaluation per iteration
    keep_history=True
)

abc = ABC(
    num_particles=num_particles,
    num_dims=num_dims,
    attempts_to_exhaust=100,
    keep_history=True
)

fss = FSS(
    num_particles=num_particles,
    num_dims=num_dims,
    ind_step_range=[0.1, 0.001],
    vol_step_range=[0.01, 0.001],
    step_decay_iterations=num_fitness_evaluations//(2*num_particles), # 2 evaluations per iteration
    weight_range=[1, np.inf],
    initial_weight=1,
    keep_history=True
)

from tqdm import trange
import time
def main():
    results = pd.DataFrame(columns=['best_fitness', 'fitness_evaluations', 'iterations', 'experiment', 'test_function'])
    algoritms = tqdm([
        (pso, 'PSO'),
        (abc, 'ABC'),
        (fss, 'FSS')
    ], 'Algorithms')
    functions = tqdm([
        (sphereFunction, [[-5.12, 5.12]], 'Sphere'),
        (rastriginsFunction, [[-5.12, 5.12]], 'Rastrigin'),
        (rosenbrockFunction, [[-5, 10]], 'Rosenbrock')
    ], 'Functions')
    for function, search_space, function_name in functions:
        functions.set_postfix({'Function': function_name})
        for algoritm, algorithm_name in algoritms:
            algoritms.set_postfix({'Algorithm': algorithm_name})
            for simulation in trange(num_simulations, desc='Runs'):
                if algorithm_name == 'ABC':
                    algoritm.initialize(function, search_space=search_space)
                else:
                    algoritm.initialize(function, initializer=UniformInitializer(search_space))
                with tqdm() as pgb:
                    while algoritm.fitness_evaluations <= num_fitness_evaluations:
                        algoritm.minimize()
                        progress = round(100*algoritm.fitness_evaluations/num_fitness_evaluations)
                        pgb.update(progress)

                history = algoritm.getHistory()
                history.insert(history.shape[1], 'experiment', value=simulation)
                history.insert(history.shape[1], 'test_function', value=function_name)
                results = results.append(history, sort=False, ignore_index=True)

    print('\nSaving results...')
    file_name = 'experiment_N2_results.csv'
    results.to_csv(file_name)
    print('Results saved to ' + file_name)
    
if __name__ == '__main__':
    main()