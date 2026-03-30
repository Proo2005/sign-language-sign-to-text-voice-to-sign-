import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- 1. Data Loading & Preprocessing ---
dataset_dir = r"D:\PROGRAMMING\hand_gesture_main\dataset\american_sign_language"
all_dfs = []

# Verify directory exists before proceeding
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Directory not found: {dataset_dir}")

for fname in os.listdir(dataset_dir):
    if fname.endswith(".csv"):
        path = os.path.join(dataset_dir, fname)
        df = pd.read_csv(path, header=None)
        label = os.path.splitext(fname)[0]
        df['label'] = label
        all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

# Clean and validate numerical features
X_df = combined_df.drop(columns=['label']).apply(pd.to_numeric, errors='coerce')
X_df = X_df.dropna(axis=1, how='all').dropna(axis=0, how='any')
y = combined_df.loc[X_df.index, 'label']

# Encode labels and scale features
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

# --- 2. PCA Visualization (Pre-Training Insight) ---
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
unique_labels = np.unique(y)
colors = plt.cm.get_cmap('tab20', len(unique_labels))

for idx, label in enumerate(unique_labels):
    mask = y == label
    plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                label=label, alpha=0.6, s=25, color=colors(idx))

plt.title("PCA of Scaled ASL Features")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- 3. Model Preparation & Training ---
X_reshaped = X_scaled[..., np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    Dropout(0.3),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_cat, epochs=30, batch_size=32, validation_split=0.1, verbose=1)

# --- 4. Performance Metrics (Precision, Recall, F1) ---
y_pred_probs = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

print("\n" + "="*30)
print("CLASSIFICATION REPORT")
print("="*30)
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

# Confusion Matrix Visualization
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix: Predicted vs. Actual ASL Gestures')
plt.ylabel('Actual Gesture')
plt.xlabel('Predicted Gesture')
plt.show()