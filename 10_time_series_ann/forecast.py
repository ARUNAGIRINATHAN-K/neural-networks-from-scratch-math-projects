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