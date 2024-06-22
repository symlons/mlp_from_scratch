import numpy as np

def one_hot(x):
    result = np.zeros((x.size, x.max() + 1))
    result[np.arange(x.size), x.flatten()] = 1
    return result 
