# ⏳ ANN for Time Series Forecasting

This project demonstrates how to use a **feedforward Artificial Neural Network (ANN)** to predict future values in a time series using past observations.

---

## 📈 Problem Statement

Given the past 10 time steps, predict the value at the next time step.

---

## 🧠 Concepts Covered

- Time Series Windowing  
- Regression with ANN  
- Data Normalization  
- Overfitting prevention via validation  
- Mean Squared Error (MSE) loss

---

## 🛠️ Tools Used

- Python  
- NumPy, scikit-learn  
- TensorFlow / Keras  
- Matplotlib (for visualization)

---

## 🧩 Model Architecture

| Layer      | Type   | Units | Activation |
|------------|--------|-------|------------|
| Input      | Dense  | 64    | ReLU       |
| Hidden     | Dense  | 32    | ReLU       |
| Output     | Dense  | 1     | Linear     |

---

## 🔄 Data Processing

- Normalized using `MinMaxScaler`
- Used **sliding window** of size `10`
- Split into training and test sets

---

## ▶️ Run Instructions

```bash
pip install numpy matplotlib scikit-learn tensorflow
python forecast.py
