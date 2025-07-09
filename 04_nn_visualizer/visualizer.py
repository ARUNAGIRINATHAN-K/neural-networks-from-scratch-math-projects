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
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((1, output_dim))

# Training parameters
epochs = 10000
lr = 0.1
plot_interval = 1000

# Plot setup
fig, ax = plt.subplots()
plt.ion()

for epoch in range(epochs):
    # Forward
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    loss = np.mean((y - A2) ** 2)

    # Backward
    dA2 = 2 * (A2 - y)
    dZ2 = dA2 * sigmoid_derivative(Z2)
    