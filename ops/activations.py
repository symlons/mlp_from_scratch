import numpy as np

def sigmoid(x, deriv=False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def relu(x, deriv=False):
    if(deriv == True):
        return np.where(x > 0, 1, 0)
    return np.where(x > 0, x, 0)

def swish(x, deriv=False):
    beta = 0.6 
    if(deriv == True):
        input_x = x *(1/1+np.exp(-beta*x))
        deriv = beta * x + sigmoid(beta * input_x)*(1 - beta * x) 
        return  deriv
    return x * sigmoid(beta * x)

