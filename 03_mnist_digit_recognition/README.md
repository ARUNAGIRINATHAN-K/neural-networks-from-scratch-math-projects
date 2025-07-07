# 🔢 Digit Recognition Using MNIST (ANN - MLP)

This project demonstrates how to classify handwritten digits (0 to 9) from the MNIST dataset using a **multi-layer perceptron (MLP)**, a basic form of an **Artificial Neural Network (ANN)**.

---

## 🧠 Problem

Given an image of a handwritten digit (28x28 grayscale), predict the correct digit label (0–9).

---

## 🧩 Concepts Used

- Multi-Layer Perceptron (MLP)
- Softmax Output Layer
- Cross-Entropy Loss
- ReLU Activation
- Data Normalization
- One-Hot Encoding
- Model Evaluation

---

## 🏗️ Model Architecture

| Layer       | Type         | Units | Activation |
|-------------|--------------|-------|------------|
| Input       | Flatten      | 784   | -          |
| Hidden 1    | Dense        | 128   | ReLU       |
| Hidden 2    | Dense        | 64    | ReLU       |
| Output      | Dense        | 10    | Softmax    |

---

## 🛠️ Files

- `train.ipynb`: Model building, training, evaluation, and accuracy plots  
- `model.ipynb`: Modular version with prediction and visualization  
- `mlp_weights.h5`: (optional) for loading pretrained weights

---

## 📈 Results

- Training Accuracy: ~98%  
- Test Accuracy: ~97%  
- Fast convergence with only 10 epochs

---

## 📊 Sample Output

