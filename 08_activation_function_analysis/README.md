# 🔄 Comparison of Activation Functions

This project compares the performance of **ReLU**, **Sigmoid**, and **Tanh** activation functions in a small neural network using the `make_moons` dataset.

---

## 🎯 Problem

Explore how different activation functions:
- Affect convergence speed
- Impact model performance
- Respond to non-linearly separable data

---

## 🧪 Dataset

- Synthetic binary classification using `make_moons`
- 2 input features
- Moderate noise for challenge

---

## 🧠 Model Architecture

- MLP with 2 hidden layers (16, 8 neurons)
- Trained separately with ReLU, Sigmoid, and Tanh
- Optimizer: Adam
- Loss: Cross Entropy (handled internally)

---

## 📊 What We Measure

- 🔻 **Loss curve**: How fast loss decreases
- ✅ **Accuracy**: Train and test accuracy after training

---

## 📈 Sample Results

```text
Final Accuracy Scores:
Logistic - Train: 0.85, Test: 0.84
Tanh     - Train: 0.88, Test: 0.87
ReLU     - Train: 0.93, Test: 0.91

```
```
pip install scikit-learn matplotlib
python compare_activations.py
```
