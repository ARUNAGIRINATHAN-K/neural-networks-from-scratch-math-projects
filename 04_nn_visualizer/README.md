# 🧠 Neural Network Visualizer (XOR)

This project builds a simple neural network to solve the **XOR problem** and **visually shows how weights evolve during training** using a live bar chart.

---

## 🎯 Objective

🔍 Help users "see" the learning process of an ANN by plotting changing weights at intervals during training.

---

## 💡 Concepts Covered

- Forward & backward propagation
- Weight & bias updates (gradient descent)
- Real-time data visualization (Matplotlib)
- XOR — nonlinear separability

---

## 🛠️ Technologies

- Python 3
- NumPy
- Matplotlib (for live plot)

---

## 📊 Visualization

Every 1000 epochs, the program:
- Plots a bar chart of weights in the network
- Displays current loss
- Updates live using `plt.pause()`

---

## ▶️ Run Instructions

```bash
pip install matplotlib numpy
python visualizer.py
