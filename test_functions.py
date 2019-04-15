import numpy as np

# search space: [-5.12, 5.12]
# optimal value: 0
def sphereFunction(x):
    return np.sum(x**2)

# search space: [-5.12, 5.12]
# optimal value: 0
def rastriginsFunction(x):
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)

# search space: [-5, 10]
# optimal value: 0
def rosenbrockFunction(x):
    x_i = x[:-1]
    x_ip1 = x[1:]
    return np.sum(100*(x_ip1 - x_i**2)**2 + (x_i - 1)**2)