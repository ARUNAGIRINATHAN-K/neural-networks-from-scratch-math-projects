import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# XOR data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Network structure
input_dim = 2
hidden_dim = 2
output_dim = 1

# Weight init
np.random.seed(0)
W1 = np.random.randn(input_dim, hidden_dim)
