import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from models.model import create_model  # Import model architecture
from sklearn.metrics import classification_report, f1_score


# Load metadata from RECORDS_1 and RECORDS_2 (skipping the headers)
records_1_path = os.path.join(folder_path, "RECORDS_1.csv")
records_2_path = os.path.join(folder_path, "RECORDS_2.csv")

# Read the files and skip the headers
records_1 = pd.read_csv(records_1_path, skiprows=1, header=None)
records_2 = pd.read_csv(records_2_path, skiprows=1, header=None)

# Combine both RECORDS files into a single DataFrame
metadata = pd.concat([records_1, records_2], ignore_index=True)
metadata.columns = ["file", "label","unk"]  # Rename columns for clarity

# Initialize data and labels
data = []
labels = []

# Load patient CSV files
for _, row in metadata.iterrows():
    csv_file = f"{row['file']}.csv"
    label = row['label']

    # Read patient data (skip the header row in each patient CSV)
    patient_data = pd.read_csv(os.path.join(folder_path, csv_file), skiprows=1)
    ekg = patient_data.iloc[:, 0].values  # First column (EKG)
    ppg = patient_data.iloc[:, 1].values  # Second column (PPG)

    # Combine EKG and PPG as features
    combined = np.stack((ekg, ppg), axis=1)
    data.append(combined)
    labels.append(label)

# Convert to numpy arrays
data = np.array(data, dtype=object)
labels = np.array(labels)

# Normalize the data (optional, depending on range)
data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.2, random_state=42)

# Encode labels (if not already one-hot encoded)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# One-hot encode the labels
y_train_onehot = to_categorical(y_train_encoded)
y_test_onehot = to_categorical(y_test_encoded)

# Define paths
MODEL_WEIGHTS_PATH = os.path.join("models", "baseline03.weights.h5")

# Load model
model = create_model()
model.load_weights(MODEL_WEIGHTS_PATH)  # Load weights

# Compile the model (important for evaluation)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model architecture defined, and weights loaded successfully!")

# Predict class probabilities
y_pred_prob = model.predict(X_test)

# Convert probabilities to class predictions
y_pred = y_pred_prob.argmax(axis=1)  # Get the index of the maximum probability
y_true = y_test_onehot.argmax(axis=1)  # Convert one-hot encoding back to class labels

# Generate classification report with 4 decimal places
report = classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(y_test_onehot.shape[1])], digits=4)
print(report)

# Calculate and format F1 scores per class (rounded to 4 decimal places)
f1_per_class = f1_score(y_true, y_pred, average=None)
f1_per_class = np.round(f1_per_class, 4)  # Round to 4 decimal places
print("F1 Score per Class:", f1_per_class)
