import numpy as np

def calculateClercFactor(c1, c2):
    phi = c1+c2
    return 2/np.abs(2-phi-np.sqrt(phi**2-4*phi))