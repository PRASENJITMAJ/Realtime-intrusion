import os
import pandas as pd
import numpy as np


FOLDER_PATH = "/Users/radha-krishna1060/Desktop/RealTime_IDS/Dataset"  # <- CHANGE THIS


csv_files = [os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH) if f.endswith('.csv')]

print(f"Found {len(csv_files)} CSV files.\n")


all_columns = []
label_counts = {}
null_counts = {}
infinite_counts = {}
row_counts = {}
duplicate_counts = {}

for file in csv_files:
    print(f"\n📁 Inspecting: {file}")
    try:
        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.strip() 

        
        all_columns.append(set(df.columns))

        # 🔹 Label distribution
        if 'Label' in df.columns:
            label_counts[file] = df['Label'].value_counts().to_dict()
        else:
            print("❌ 'Label' column not found.")

        # 🔹 Nulls and Infs
        null_counts[file] = df.isnull().sum().sum()
        infinite_counts[file] = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

        # 🔹 Rows and duplicates
        row_counts[file] = len(df)
        duplicate_counts[file] = df.duplicated().sum()

        print(f"✅ Rows: {len(df)} | Nulls: {null_counts[file]} | Infs: {infinite_counts[file]} | Duplicates: {duplicate_counts[file]}")
        print(f"🔖 Label counts: {label_counts[file] if file in label_counts else 'N/A'}")
    except Exception as e:
        print(f"❌ Error reading {file}: {e}")

# 🔍 Compare columns across files
print("\n🔁 Column Structure Comparison:")
if all(colset == all_columns[0] for colset in all_columns):
    print("✅ All files have consistent columns.")
else:
    print("⚠️ Columns differ across files!")

print("\n✅ Inspection complete.")
