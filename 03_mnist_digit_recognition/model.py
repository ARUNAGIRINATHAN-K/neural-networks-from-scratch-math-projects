import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test / 255.0

model = load_model('mlp_mnist.h5')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten