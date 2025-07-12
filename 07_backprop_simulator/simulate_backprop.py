import numpy as np
#activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sig_x):
    return sig_x * (1 - sig_x)

#Input/Output
x = np.array([[0.5, 0.8]])  
y_true = np.array([[1]]) 

#initianlize
np.random.seed(0)
W1 = np.random.randn(2, 2) 
b1 = np.random.randn(1, 2)
