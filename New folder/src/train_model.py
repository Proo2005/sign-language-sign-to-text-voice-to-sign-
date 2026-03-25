import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import glob
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# === Let user choose sign language type ===
print("Choose sign language type:")
print("1. Indian Sign Language")
print("2. American Sign Language")
print("3. Hand Gesture")
print("4. Bengali Sign Language")
choice = input("Enter choice (1/2/3/4): ").strip()

if choice == "1":
    dataset_path = "dataset/indian_sign_language/**/*.csv"
    model_dir = "model/indian_sign_language"
elif choice == "2":
    dataset_path = "dataset/american_sign_language/**/*.csv"
    model_dir = "model/american_sign_language"
elif choice == "3":
    dataset_path = "dataset/hand gestures/**/*.csv"
    model_dir = "model/hand gestures"
elif choice == "4":
    dataset_path = "dataset/bengali_sign_language/**/*.csv"
    model_dir = "model/bengali_sign_language"
else:
    raise ValueError("❌ Invalid choice! Please select 1, 2, 3, or 4.")

print(f"📂 Loading dataset from: {dataset_path}")

# === Load all CSVs recursively ===
csv_files = glob.glob(dataset_path, recursive=True)
if not csv_files:
    raise FileNotFoundError(f"❌ No CSV files found in {dataset_path}")

df_list = []
for file in csv_files:
    try:
        df = pd.read_csv(file, header=None, encoding="utf-8")
        df_list.append(df)
    except Exception as e:
        print(f"⚠️ Skipping file {file}: {e}")

# Combine all data
df = pd.concat(df_list, ignore_index=True)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# === Separate features and labels ===
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# === Encode labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)  # one-hot for LSTM classification

# === Reshape for LSTM ===
# LSTM expects input: (samples, timesteps, features)
# Assuming each row = one sequence, treat features as timesteps=1
X = np.expand_dims(X, axis=1)  # shape: (samples, 1, features)

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# === Build LSTM Model ===
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# === Train ===
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# === Evaluate ===
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# === Save Model & Label Encoder ===
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "gesture_lstm_model.h5"))
joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))

print(f"💾 LSTM model and label encoder saved to '{model_dir}'")
