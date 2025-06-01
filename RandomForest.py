import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# 🔧 Paths
ROOT_DIR = "/Users/radha-krishna1060/Desktop/RealTime_IDS/Dataset"
DATASET_PATH = os.path.join(ROOT_DIR, "cleaned_dataset.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "Phase2_Models")
REPORT_DIR = os.path.join(ROOT_DIR, "Reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# 📄 Load dataset
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()
X = df.drop(columns=["Label", "AttackLabel"])
y = df["AttackLabel"]

# 🧼 Convert all to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X.fillna(0, inplace=True)

# 📐 Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✂️ Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# 📈 Classification report
report = classification_report(y_test, y_pred, digits=4)
roc_auc = roc_auc_score(y_test, y_prob)

with open(os.path.join(REPORT_DIR, "classification_report_rf.txt"), "w") as f:
    f.write("Random Forest Classification Report\n")
    f.write(report)
    f.write(f"\nROC-AUC: {roc_auc:.4f}\n")

# 📉 ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.savefig(os.path.join(REPORT_DIR, "roc_curve_rf.png"))
plt.close()

# 📊 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix_rf.png"))
plt.close()

# 💾 Save model and scaler
joblib.dump(rf, os.path.join(MODEL_DIR, "model_rf.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "features.txt"), "w") as f:
    f.write("\n".join(X.columns))

print(f"\n✅ Training complete.")
print(f"📊 Metrics saved to: {REPORT_DIR}")
print(f"💾 Model saved to: {MODEL_DIR}")
