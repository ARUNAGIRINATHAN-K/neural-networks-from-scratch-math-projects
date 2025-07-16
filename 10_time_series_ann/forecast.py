import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Time-series
np.random.seed(0)
time = np.arange(0, 200)
series = np.sin(0.1 * time) + np.random.normal(0, 0.1, size=len(time))

plt.plot(time, series, label="Time Series")
plt.title("Synthetic Time Series Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.show()

#Normalize
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

#window size
window_size = 10
X = []
y = []

for i in range(len(series_scaled) - window_size):
    X.append(series_scaled[i:i+window_size])
    y.append(series_scaled[i+window_size])

X = np.array(X)
y = np.array(y)

#train
