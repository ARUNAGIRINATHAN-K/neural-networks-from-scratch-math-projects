import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

