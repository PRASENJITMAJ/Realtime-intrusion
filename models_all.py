import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Paths
ROOT_DIR = "/Users/radha-krishna1060/Desktop/RealTime_IDS"
DATASET_PATH = os.path.join(ROOT_DIR, "Dataset/cleaned_dataset.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "Dataset/Phase2_Models")
REPORT_DIR = os.path.join(ROOT_DIR, "Dataset/Reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Load Data
print("📥 Loading dataset...")
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()

X = df.drop(columns=["Label", "AttackLabel"])
y = df["AttackLabel"]

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# Scaling
print("📐 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler + feature list
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "features.txt"), "w") as f:
    f.write("\n".join(X.columns))

# Train/test split
print("✂️ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Model definitions
models = {
    "randomforest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', verbosity=0),
    "lightgbm": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    "catboost": CatBoostClassifier(verbose=0, iterations=100, random_state=42)
}

def evaluate_and_save(model_name, model):
    print(f"\n🚀 Training {model_name.upper()}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print(f"📊 Evaluating {model_name}...")
    report = classification_report(y_test, y_pred, digits=4)
    auc = roc_auc_score(y_test, y_prob)

    with open(os.path.join(REPORT_DIR, f"report_{model_name}.txt"), "w") as f:
        f.write(f"{model_name.upper()} Classification Report\n")
        f.write(report)
        f.write(f"\nROC-AUC: {auc:.4f}\n")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name.upper()}")
    plt.legend()
    plt.savefig(os.path.join(REPORT_DIR, f"roc_{model_name}.png"))
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
    plt.title(f"Confusion Matrix - {model_name.upper()}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(REPORT_DIR, f"confusion_{model_name}.png"))
    plt.close()

    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, f"model_{model_name}.pkl"))
    print(f"✅ {model_name.upper()} saved to: model_{model_name}.pkl")

# Run all models
for name, model in models.items():
    evaluate_and_save(name, model)

print("\n🎉 All models trained and saved with their reports.")
