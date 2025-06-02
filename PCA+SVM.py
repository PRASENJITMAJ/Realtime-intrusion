import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from scipy.special import expit  # sigmoid
import joblib


ROOT_DIR = "/Users/radha-krishna1060/Documents/RealTime_IDS"
DATASET_PATH = os.path.join(ROOT_DIR, "Dataset", "cleaned_dataset.csv")
REPORT_DIR = os.path.join(ROOT_DIR, "Phase2_Reports")
MODEL_DIR = os.path.join(ROOT_DIR, "Phase2_Models")
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Start timer
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


print("[4] Applying PCA (32 components)...")
T2_start = time.time()
pca = PCA(n_components=32, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
T2 = time.time()
print(f"PCA Complete! Time: {T2 - T2_start:.2f}s\n")


print("[5] Training Linear SVM (Fast, No Calibration)...")
T3_start = time.time()
svm_clf = LinearSVC(max_iter=10000, random_state=42)
svm_clf.fit(X_train_pca, y_train)
y_pred = svm_clf.predict(X_test_pca)


decision_scores = svm_clf.decision_function(X_test_pca)
y_prob = expit(decision_scores)
T3 = time.time()
print(f"Linear SVM Training Done in {T3 - T3_start:.2f}s\n")


print("[6] Evaluating Model...")
report = classification_report(y_test, y_pred, digits=4)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"ROC-AUC: {roc_auc:.4f}")
print("Classification Report:\n", report)

with open(os.path.join(REPORT_DIR, "pca_svm_linear_report.txt"), "w") as f:
    f.write("PCA + Linear SVM (No Calibration) Report\n")
    f.write(report)
    f.write(f"\nROC-AUC: {roc_auc:.4f}\n")


print("[7] Saving Confusion Matrix + ROC Curve...")
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
plt.title("Confusion Matrix - PCA + Linear SVM (Fast)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix_pca_linear_svm_fast.png"))
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - PCA + Linear SVM (Fast)")
plt.legend(loc="lower right")
plt.savefig(os.path.join(REPORT_DIR, "roc_curve_pca_linear_svm_fast.png"))
plt.close()


print("[8] Saving Models + Scaler + PCA...")
joblib.dump(pca, os.path.join(MODEL_DIR, "pca_model_linear_fast.pkl"))
joblib.dump(svm_clf, os.path.join(MODEL_DIR, "svm_pca_linear_fast_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_pca_fast.pkl"))
with open(os.path.join(MODEL_DIR, "features_pca_fast.txt"), "w") as f:
    f.write("\n".join(X.columns))


T4 = time.time()
print("[9] Writing Timing Logs...")
with open(os.path.join(REPORT_DIR, "timing_pca_svm_linear_fast.txt"), "w") as f:
    f.write(f"Preprocessing: {T1 - T0:.2f} sec\n")
    f.write(f"PCA Time: {T2 - T1:.2f} sec\n")
    f.write(f"SVM Training: {T3 - T2:.2f} sec\n")
    f.write(f"Evaluation & Saving: {T4 - T3:.2f} sec\n")
    f.write(f"TOTAL: {T4 - T0:.2f} sec\n")

print("\n✅ All Done! Fast PCA + Linear SVM training complete.")
print(f"📊 Reports saved to: {REPORT_DIR}")
print(f"💾 Models saved to: {MODEL_DIR}")
