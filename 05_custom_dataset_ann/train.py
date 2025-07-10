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
