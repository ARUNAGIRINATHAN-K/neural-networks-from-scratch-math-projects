import numpy as np
from utils import sigmoid, sigmoid_derivative, mean_squared_error, mse_derivative

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
epochs = 10000