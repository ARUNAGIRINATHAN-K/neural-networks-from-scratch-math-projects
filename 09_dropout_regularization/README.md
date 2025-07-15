# 🚫 Dropout Regularization (Manual Implementation)

This project demonstrates how to **manually implement dropout** regularization in a basic neural network trained on the **XOR problem**.

---

## 💡 Problem

Neural networks can **overfit** on small datasets. Dropout helps by:
- Randomly "dropping" neurons during training
- Preventing co-adaptation
- Improving generalization

---

## 🔁 What This Project Does

- Builds a 2-layer ANN from scratch
- Adds **manual dropout logic** to the hidden layer
- Applies **inverted dropout** (rescale during train, nothing during test)
- Shows forward & backward pass integration

---

## 🧠 Network Structure

| Layer   | Neurons | Activation | Dropout |
|---------|---------|------------|---------|
| Input   | 2       | -          | No      |
| Hidden  | 4       | Sigmoid    | ✅ 0.5  |
| Output  | 1       | Sigmoid    | No      |

---

## 📈 Sample Output

```text
Epoch 0, Loss: 0.2806
Epoch 2000, Loss: 0.1998
Epoch 4000, Loss: 0.1387
Epoch 6000, Loss: 0.2727
Epoch 8000, Loss: 0.2769
...
Final Predictions:
[[0.251]
 [0.84 ]
 [0.429]
 [0.456]]
