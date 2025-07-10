# 🍎 Train an ANN on a Custom Dataset

This project demonstrates how to train an **Artificial Neural Network (ANN)** on a real-world inspired custom dataset (e.g., fruit classification based on physical features).

---

## 📦 Dataset: `dataset.csv`

Sample features:
- **weight:** in grams  
- **color_score:** visual color richness  
- **texture:** a value indicating firmness  
- **label:** fruit name (Apple, Mango, etc.)

---

## 🧠 Concepts Covered

- Data preprocessing
  - Normalization with `StandardScaler`
  - Label encoding
- ANN structure with Keras
- Model evaluation and predictions
- Training-validation pipeline

---

## 🧩 Model Architecture

| Layer    | Units | Activation |
|----------|-------|------------|
| Input    | 3     | -          |
| Hidden 1 | 8     | ReLU       |
| Hidden 2 | 4     | ReLU       |
| Output   | 2+    | Softmax    |

---

## ⚙️ Technologies Used

- Python  
- Pandas, NumPy  
- scikit-learn  
- TensorFlow / Keras

---

## ▶️ Run Instructions

```bash
pip install pandas scikit-learn tensorflow
python train.py
