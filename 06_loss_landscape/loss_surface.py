import numpy as np
import matplotlib.pyplot as plt
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
plt.figure(figsize=(10, 5))
plt.contourf(W, B, Loss, levels=50, cmap="viridis")
plt.colorbar(label="MSE Loss")
plt.xlabel("Weight (w)")
plt.ylabel("Bias (b)")
plt.title("2D Loss Contour for Linear Regression")
plt.grid(True)
plt.tight_layout()
plt.show()

#3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(W, B, Loss, cmap="viridis", edgecolor="none")
ax.set_xlabel("Weight (w)")
ax.set_ylabel("Bias (b)")
ax.set_zlabel("Loss")
ax.set_title("3D Loss Landscape")
plt.tight_layout()
plt.show()