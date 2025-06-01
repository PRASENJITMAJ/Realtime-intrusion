import os
import re
import pandas as pd
import matplotlib.pyplot as plt


REPORT_DIR = "/Users/radha-krishna1060/Documents/RealTime_IDS/Dataset/Reports"

# === Extract metrics from report ===
def extract_metrics(filepath):
    metrics = {"Accuracy": None, "Precision (Attack)": None, "Recall (Attack)": None, "F1-score (Attack)": None, "ROC AUC": None}
    with open(filepath, 'r') as f:
        content = f.read()

    auc = re.search(r"ROC[-_ ]?AUC:?\s*([0-9.]+)", content, re.IGNORECASE)
    if auc:
        metrics["ROC AUC"] = float(auc.group(1))

    for line in content.splitlines():
        if line.strip().startswith("1") and len(line.strip().split()) >= 4:
            vals = re.findall(r"[0-9.]+", line)
            if len(vals) >= 3:
                metrics["Precision (Attack)"] = float(vals[0])
                metrics["Recall (Attack)"] = float(vals[1])
                metrics["F1-score (Attack)"] = float(vals[2])
            break

    for line in content.splitlines():
        if "accuracy" in line.lower():
            acc = re.findall(r"[0-9.]+", line)
            if acc:
                metrics["Accuracy"] = float(acc[0])
            break
    return metrics


pca_report = ae_report = None
for file in os.listdir(REPORT_DIR):
    if file.endswith(".txt") and "report" in file.lower():
        fname = file.lower()
        if "pca" in fname and "svm" in fname:
            pca_report = os.path.join(REPORT_DIR, file)
        elif "ae" in fname and "svm" in fname:
            ae_report = os.path.join(REPORT_DIR, file)

if not pca_report or not ae_report:
    raise FileNotFoundError("❌ Could not find both PCA and AE SVM report files in the given directory.")


models = {
    "PCA + Linear SVM": extract_metrics(pca_report),
    "AE + RBF SVM": extract_metrics(ae_report)
}
df = pd.DataFrame(models).T


txt_path = os.path.join(REPORT_DIR, "comparison_pca_vs_ae.txt")
with open(txt_path, "w") as f:
    f.write("PCA vs AE - SVM Model Metric Comparison\n")
    f.write("="*50 + "\n\n")
    for model in df.index:
        f.write(f"{model}:\n")
        for metric, value in df.loc[model].items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\n")
print(f"✅ Metrics saved to: {txt_path}")


fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()
metric_names = df.columns.tolist()

for i, metric in enumerate(metric_names):
    df[metric].plot(kind='bar', ax=axes[i], color=['skyblue', 'orange'])
    axes[i].set_title(metric)
    axes[i].set_ylabel("Score")
    axes[i].set_ylim(0, 1.1)
    axes[i].grid(True)
    axes[i].set_xticklabels(df.index, rotation=15)

fig.delaxes(axes[-1])
fig.suptitle("PCA vs AE - SVM Model Comparison (Bar Chart)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
bar_path = os.path.join(REPORT_DIR, "bar_chart_pca_vs_ae.png")
plt.savefig(bar_path)
plt.show()


plt.figure(figsize=(10, 6))
for model in df.index:
    plt.plot(df.columns, df.loc[model], marker='o', label=model)
plt.title("PCA vs AE - SVM Comparison (Line Chart)")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.grid(True)
plt.legend()
line_path = os.path.join(REPORT_DIR, "line_chart_pca_vs_ae.png")
plt.savefig(line_path)
plt.show()

print(f"📊 Charts saved to:\n  {bar_path}\n  {line_path}")
