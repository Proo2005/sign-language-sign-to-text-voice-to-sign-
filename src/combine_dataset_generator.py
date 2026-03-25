import os
import pandas as pd

# Base dataset folder
base_path = "dataset"

# Languages
languages = ["INDIAN_SIGN_LANGUAGE", "HAND_GESTURES", "AMERICAN_SIGN_LANGUAGE", "BENGALI_SIGN_LANGUAGE"]

dataframes = []

for lang in languages:
    folder_path = os.path.join(base_path, lang)
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        continue
    
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                df['label'] = file.replace(".csv", "")  # use file name as label
                dataframes.append(df)
                print(f"✅ Loaded: {file_path} ({df.shape[0]} rows)")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

# Merge all into one DataFrame
if dataframes:
    data = pd.concat(dataframes, ignore_index=True)
    print(f"\n✅ Dataset loaded successfully with {data.shape[0]} samples and {data.shape[1]} features.")
else:
    raise ValueError("❌ No CSV files found. Check your folder paths.")

# Save combined dataset
data.to_csv("combined_dataset.csv", index=False)
