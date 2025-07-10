import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# Load dataset
data = pd.read_csv("/workspaces/ANN-Mathematics-Projects/05_custom_dataset_ann/dataset.csv")

# Separate features and labels
X = data.drop("label", axis=1).values
y = data["label"].values

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build ANN
model = Sequential([
    Dense(8, input_shape=(X.shape[1],), activation="relu"),
    Dense(4, activation="relu"),
    Dense(len(np.unique(y_encoded)), activation="softmax")
])

# Compile model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train
model.fit(X_train, y_train, epochs=100, batch_size=2, verbose=1)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.2f}")

# Predict on test set
preds = model.predict(X_test)
pred_labels = np.argmax(preds, axis=1)

# Show predictions
for i in range(len(pred_labels)):
    print(f"Actual: {label_encoder.inverse_transform([y_test[i]])[0]}, Predicted: {label_encoder.inverse_transform([pred_labels[i]])[0]}")
