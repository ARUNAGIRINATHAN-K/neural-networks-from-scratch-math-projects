import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Toy-data
np.random.seed(42)
X = np.linspace(0, 1, 100).reshape(-1, 1)
true_w, true_b = 2, 3
y = true_w * X + true_b + np.random.normal(0, 0.1, size=X.shape)
