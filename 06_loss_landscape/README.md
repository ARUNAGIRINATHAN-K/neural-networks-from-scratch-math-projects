# 📉 Loss Landscape Visualization

This project visualizes how the **loss function (MSE)** changes with different **weight and bias** combinations in a simple **linear regression** task.

---

## 🎯 Problem Statement

In deep learning, understanding how **model parameters** affect the **loss surface** is crucial. This visualization helps interpret:

- Where the **global minima** lies
- How gradient descent navigates the surface
- The shape of convex vs. non-convex loss functions

---

## 🔧 Tools & Libraries Used

- Python 3
- NumPy
- Matplotlib
- `mpl_toolkits.mplot3d` for 3D plotting

---

## 📊 What It Does

- Generates a toy linear dataset: \( y = 2x + 3 + \epsilon \)
- Computes the **Mean Squared Error (MSE)** over a grid of weight (w) and bias (b) values
- Plots:
  - A **2D Contour Plot** of the loss surface
  - A **3D Surface Plot** of the same

---

## 📈 Sample Output

| Contour Plot | 3D Surface |
|--------------|------------|
| ![contour](06_loss_landscape/output/Color-plot.png) | ![surface](06_loss_landscape/output/sub-plot.png) |


## ▶️ How to Run

```bash
pip install numpy matplotlib
python loss_surface.py
