# 🔀 XOR Gate Classification Using ANN

This project demonstrates how a simple **Artificial Neural Network (ANN)** can learn the XOR logic gate — a classic example of a **non-linearly separable problem**.

---

## 🧠 Problem

The XOR logic outputs:
- `0 XOR 0 = 0`
- `0 XOR 1 = 1`
- `1 XOR 0 = 1`
- `1 XOR 1 = 0`

A single-layer perceptron fails on XOR — hence the need for **non-linearity + hidden layer**.

---

## 🛠️ Architecture

- **Input Layer:** 2 neurons  
- **Hidden Layer:** 2 neurons with sigmoid  
- **Output Layer:** 1 neuron with sigmoid  
- **Loss Function:** Mean Squared Error (MSE)

---

## 🚀 Technologies Used

- Python  
- NumPy  
- Matplotlib  
- Jupyter Notebook (`xor_ann.ipynb`)

---

## 📈 Training Overview

- **Epochs:** 10,000  
- **Learning Rate:** 0.1  
- **Optimizer:** Gradient Descent (manual backpropagation)

---

## 📊 Results

```text
Epoch 0, Loss: 0.3091
Epoch 2000, Loss: 0.0248
Epoch 4000, Loss: 0.0128
Epoch 6000, Loss: 0.0085
Epoch 8000, Loss: 0.0064

Final predictions:
[[0.01]
 [0.98]
 [0.98]
 [0.01]]
