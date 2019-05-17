import numpy as np
import pandas as pd

from pso import PSO
from pso.topology import LocalTopology
from fss import FSS
from abc_optimizer import ABC

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

def main():
    pass


if __name__ == '__main__':
    main()