# 🔁 Backpropagation Algorithm Simulator

This project walks through **step-by-step backpropagation math** for a toy neural network. It prints the gradients at each layer using the **chain rule** on a single data point.

---

## 🧠 Problem

Manually simulate how gradients flow from output to input using the:
- Chain Rule
- Matrix Derivatives
- Partial Derivatives

---

## 📐 Network Design

- Input Layer: 2 neurons  
- Hidden Layer: 2 neurons (Sigmoid)  
- Output Layer: 1 neuron (Sigmoid)  

---

## 🔧 Tools

- Python 3
- NumPy only (no frameworks)
- Pure matrix math

---

## 📈 What You See

- Values of **z1, a1, z2, a2**
- Final prediction and loss
- Gradients:  
  - dL/dW2, dL/db2 (Output Layer)  
  - dL/dW1, dL/db1 (Hidden Layer)  

---

## ▶️ Run It

```bash
python simulate_backprop.py
