import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import joblib
import tensorflow as tf


print("TensorFlow Version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))


ROOT_DIR = "/Users/radha-krishna1060/Documents/RealTime_IDS"
DATASET_PATH = os.path.join(ROOT_DIR, "Dataset", "cleaned_dataset.csv")
REPORT_DIR = os.path.join(ROOT_DIR, "Phase2_Reports")
MODEL_DIR = os.path.join(ROOT_DIR, "Phase2_Models")
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


T0 = time.time()


print("[1] Loading Dataset...")
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()
X = df.drop(columns=["Label", "AttackLabel"])
y = df["AttackLabel"]


print("[2] Preprocessing & Scaling...")
X = X.apply(pd.to_numeric, errors="coerce")
X.fillna(0, inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("[3] Splitting Train/Test Data...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
T1 = time.time()
print(f"Data Ready! Time: {T1 - T0:.2f}s\n")


print("[4] Building Autoencoder...")
input_dim = X_train.shape[1]
encoding_dim = 32

input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)
autoencoder.compile(optimizer=Adam(0.001), loss='mse')

print("[5] Training Autoencoder...")
T2_start = time.time()
autoencoder.fit(
    X_train, X_train,
    epochs=20,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test),
    verbose=1
)
T2 = time.time()
print(f"Autoencoder Training Complete! Time: {T2 - T2_start:.2f}s\n")


print("[6] Extracting Latent Features...")
T3_start = time.time()
X_train_latent = encoder.predict(X_train)
X_test_latent = encoder.predict(X_test)
T3 = time.time()
print(f"Latent Features Extracted! Time: {T3 - T3_start:.2f}s\n")


print("[7] Training Fast Approximate RBF SVM Classifier...")
T4_start = time.time()
rbf_sampler = RBFSampler(gamma=1.0, n_components=500, random_state=42)
X_train_rbf = rbf_sampler.fit_transform(X_train_latent)
X_test_rbf = rbf_sampler.transform(X_test_latent)

svm_clf = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=42)
svm_clf.fit(X_train_rbf, y_train)

y_scores = svm_clf.decision_function(X_test_rbf)
y_pred = (y_scores > 0).astype(int)
y_prob = 1 / (1 + np.exp(-y_scores))
T4 = time.time()
print(f"Fast RBF-SVM Approx Training Done in {T4 - T4_start:.2f}s\n")


print("[8] Evaluating Model...")
report = classification_report(y_test, y_pred, digits=4)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"ROC-AUC: {roc_auc:.4f}")
print("Classification Report:\n", report)

with open(os.path.join(REPORT_DIR, "ae_svm_rbf_report.txt"), "w") as f:
    f.write("AE + Approx RBF SVM Classification Report\n")
    f.write(report)
    f.write(f"\nROC-AUC: {roc_auc:.4f}\n")


print("[9] Saving Confusion Matrix + ROC Curve...")
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
plt.title("Confusion Matrix - AE + RBF SVM (Fast)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix_ae_svm_rbf.png"))
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - AE + RBF SVM (Fast)")
plt.legend(loc="lower right")
plt.savefig(os.path.join(REPORT_DIR, "roc_curve_ae_svm_rbf.png"))
plt.close()


print("[10] Saving Models + Scaler...")
autoencoder.save(os.path.join(MODEL_DIR, "autoencoder_rbf_fast.h5"))
encoder.save(os.path.join(MODEL_DIR, "encoder_rbf_fast.h5"))
joblib.dump(svm_clf, os.path.join(MODEL_DIR, "svm_rbf_fast_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_rbf.pkl"))
with open(os.path.join(MODEL_DIR, "features_rbf.txt"), "w") as f:
    f.write("\n".join(X.columns))


T5 = time.time()
print("[11] Writing Timing Logs...")
with open(os.path.join(REPORT_DIR, "timing_ae_svm_rbf.txt"), "w") as f:
    f.write(f"Preprocessing: {T1 - T0:.2f} sec\n")
    f.write(f"AE Training: {T2 - T1:.2f} sec\n")
    f.write(f"Latent Extraction: {T3 - T2:.2f} sec\n")
    f.write(f"SVM Training: {T4 - T3:.2f} sec\n")
    f.write(f"Evaluation & Saving: {T5 - T4:.2f} sec\n")
    f.write(f"TOTAL: {T5 - T0:.2f} sec\n")

print("\nAll Done! AE + Fast RBF SVM training and saving completed.")
print(f"Reports saved to: {REPORT_DIR}")
print(f"Models saved to: {MODEL_DIR}")
