import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load data (same as before)
dataset_dir = r"D:\A  DRIVE\programming\hand_gesture\dataset\american_sign_language"

all_dfs = []
for fname in os.listdir(dataset_dir):
    if fname.endswith(".csv"):
        path = os.path.join(dataset_dir, fname)
        print(f"Loading {path}...")
        df = pd.read_csv(path, header=None)
        label = os.path.splitext(fname)[0]
        df['label'] = label
        all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

X_df = combined_df.drop(columns=['label'])
X_df = X_df.apply(pd.to_numeric, errors='coerce')
X_df = X_df.dropna(axis=1, how='all')
X_df = X_df.dropna(axis=0, how='any')

y = combined_df.loc[X_df.index, 'label']

print(f"Features shape after cleaning: {X_df.shape}")
print(f"Number of labels: {len(y)}")

# Dimensionality reduction with PCA (to 2D)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_df)

# Plot
plt.figure(figsize=(10, 8))

# Assign a color to each label
unique_labels = y.unique()
colors = plt.cm.get_cmap('tab20', len(unique_labels))

for idx, label in enumerate(unique_labels):
    indices = y == label
    plt.scatter(X_reduced[indices, 0], X_reduced[indices, 1], label=label, alpha=0.6, s=20, color=colors(idx))

plt.title("PCA of American Sign Language Features")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()
