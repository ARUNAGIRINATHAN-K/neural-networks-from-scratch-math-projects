import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sig):
    return sig * (1 - sig)

# XOR data
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Hyperparameters
epochs = 10000
lr = 0.1
input_size = 2
hidden_size = 4
output_size = 1
dropout_rate = 0.5

#weight initialization
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
