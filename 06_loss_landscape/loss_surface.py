import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Toy-data
np.random.seed(42)
X = np.linspace(0, 1, 100).reshape(-1, 1)
true_w, true_b = 2, 3
y = true_w * X + true_b + np.random.normal(0, 0.1, size=X.shape)
#Loss
w_range = np.linspace(0, 4, 100)
b_range = np.linspace(0, 6, 100)
W, B = np.meshgrid(w_range, b_range)
Loss = np.zeros_like(W)
#MSE loss
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        y_pred = W[i, j] * X + B[i, j]
        loss = np.mean((y - y_pred)**2)
        Loss[i, j] = loss
#plot
