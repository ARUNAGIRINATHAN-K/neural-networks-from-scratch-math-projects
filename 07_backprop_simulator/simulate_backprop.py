import numpy as np
#activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sig_x):
    return sig_x * (1 - sig_x)

#Input/Output
