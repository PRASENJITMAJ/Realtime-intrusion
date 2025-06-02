import os
import pandas as pd
import numpy as np


DATASET_FOLDER = "/Users/radha-krishna1060/Desktop/RealTime_IDS/Dataset"


csv_files = [os.path.join(DATASET_FOLDER, f) for f in os.listdir(DATASET_FOLDER) if f.endswith('.csv')]
print(f"Found {len(csv_files)} CSV files.")


df_list = []

for file in csv_files:
    print(f"\n📁 Loading: {file}")
    df = pd.read_csv(file, low_memory=False)
    
    
    df.columns = df.columns.str.strip()
    
   
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
   
    df.drop_duplicates(inplace=True)
    
    
    if 'Label' in df.columns:
        df['Label'] = df['Label'].astype(str).str.strip()
        df['Label'] = df['Label'].str.replace(r'[^\w\s]', '', regex=True)
    else:
        print("⚠️ No 'Label' column found. Skipping file.")
        continue
    
    
    df['AttackLabel'] = df['Label'].apply(lambda x: 0 if x.upper() == 'BENIGN' else 1)

    df_list.append(df)


full_df = pd.concat(df_list, ignore_index=True)


output_file = os.path.join(DATASET_FOLDER, "cleaned_dataset.csv")
full_df.to_csv(output_file, index=False)
print(f"\n✅ Cleaned dataset saved as: {output_file}")
print(f"🧮 Final shape: {full_df.shape}")
print(f"📊 Label distribution:\n{full_df['AttackLabel'].value_counts()}")
