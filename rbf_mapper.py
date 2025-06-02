import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model # type: ignore
from sklearn.kernel_approximation import RBFSampler


ROOT = "C:\\Users\\majum\\OneDrive\\Pictures\\RealTime_IDS"
DATASET = os.path.join(ROOT, "Dataset", "cleaned_dataset.csv")
MODEL_DIR = os.path.join(ROOT, "Dataset", "Phase2_Models")


df = pd.read_csv(DATASET)
df.columns = df.columns.str.strip()
X = df.drop(columns=["Label", "AttackLabel"]).apply(pd.to_numeric, errors="coerce").fillna(0)


scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_rbf.pkl"))
encoder = load_model(os.path.join(MODEL_DIR, "encoder_rbf_fast.h5"))


X_scaled = scaler.transform(X)
X_latent = encoder.predict(X_scaled, batch_size=128)


rbf_sampler = RBFSampler(gamma=1.0, n_components=500, random_state=42)
rbf_sampler.fit(X_latent)
joblib.dump(rbf_sampler, os.path.join(MODEL_DIR, "rbf_mapper.pkl"))

print("✅ rbf_mapper.pkl saved successfully.")
