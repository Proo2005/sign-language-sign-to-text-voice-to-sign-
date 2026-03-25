import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# === Load your data here ===
# Example loading HAND GETURES
dataset_dir = r"D:\A  DRIVE\programming\hand_gesture\dataset\hand gestures"

all_dfs = []
for fname in os.listdir(dataset_dir):
    if fname.endswith(".csv"):
        path = os.path.join(dataset_dir, fname)
        df = pd.read_csv(path, header=None)
        label = os.path.splitext(fname)[0]
        df['label'] = label
        all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

# Prepare features and labels
X_df = combined_df.drop(columns=['label'])
X_df = X_df.apply(pd.to_numeric, errors='coerce')
X_df = X_df.dropna(axis=1, how='all')
X_df = X_df.dropna(axis=0, how='any')

y = combined_df.loc[X_df.index, 'label']

print(f"Feature shape: {X_df.shape}, Labels: {len(y)}")

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

# Reshape for Conv1D: (samples, steps, channels)
# Here, steps = number of features, channels = 1
X_reshaped = X_scaled[..., np.newaxis]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# One-hot encode labels for Keras
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Build CNN model (Conv1D)
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(
    X_train, y_train_cat,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
import matplotlib.pyplot as plt

# After model.fit(...) completes, plot training history:

def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Call this function after training finishes
plot_training_history(history)
history = model.fit(
    X_train, y_train_cat,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Show plots of training progress in a new window
plot_training_history(history)

# Then evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")
