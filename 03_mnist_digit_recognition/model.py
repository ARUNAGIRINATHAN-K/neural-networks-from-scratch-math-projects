import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test / 255.0

model = load_model('mlp_mnist.h5')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('mlp_weights.h5')

sample_idx = 123
plt.imshow(x_test[sample_idx], cmap='gray')
plt.title("Actual Label: " + str(y_test[sample_idx]))
plt.axis('off')
plt.show()

prediction = np.argmax(model.predict(x_test[sample_idx].reshape(1, 28, 28)), axis=-1)
print("Predicted Digit:", prediction[0])