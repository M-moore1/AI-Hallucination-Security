import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Load data
# =========================

csv_files = {
    "o3": "TestData_o3.csv",
    "Opus 4.6": "TestData_opus4.6.csv",
    "Gemini": "TestData_Gemini.csv"
}

dfs = []

for model_name, file_path in csv_files.items():
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["model"] = model_name
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# Ensure numeric values
data["correct"] = pd.to_numeric(data["correct"], errors="coerce")
data["confidence_score"] = pd.to_numeric(data["confidence_score"], errors="coerce")

# =========================
# Derived metrics
# =========================

data["is_correct"] = data["correct"] == 1
data["is_fail"] = data["correct"] == 0
data["is_critical_fail"] = data["correct"] == -1
data["is_high_confidence_fail"] = (data["is_fail"]) & (data["confidence_score"] >= 90)

# =========================
# 1. Overall Accuracy by Model
# =========================

accuracy = data.groupby("model")["is_correct"].mean() * 100

plt.figure(figsize=(8, 5))
accuracy.plot(kind="bar")
plt.title("Overall Accuracy by Model")
plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# =========================
# 2. Hallucination / Failure Rate by Model
# =========================

fail_rate = data.groupby("model")["is_fail"].mean() * 100

plt.figure(figsize=(8, 5))
fail_rate.plot(kind="bar")
plt.title("Hallucination / Failure Rate by Model")
plt.ylabel("Failure Rate (%)")
plt.xlabel("Model")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# =========================
# 3. Critical Failure Rate by Model
# =========================

critical_rate = data.groupby("model")["is_critical_fail"].mean() * 100

plt.figure(figsize=(8, 5))
critical_rate.plot(kind="bar")
plt.title("Critical Failure Rate by Model")
plt.ylabel("Critical Failure Rate (%)")
plt.xlabel("Model")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# =========================
# 4. Accuracy by Category and Model
# =========================

category_accuracy = (
    data.groupby(["category", "model"])["is_correct"]
    .mean()
    .mul(100)
    .unstack()
)

category_accuracy.plot(kind="bar", figsize=(10, 6))
plt.title("Accuracy by Category and Model")
plt.ylabel("Accuracy (%)")
plt.xlabel("Category")
plt.ylim(0, 100)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# =========================
# 5. Average Confidence by Model
# =========================

avg_confidence = data.groupby("model")["confidence_score"].mean()

plt.figure(figsize=(8, 5))
avg_confidence.plot(kind="bar")
plt.title("Average Confidence by Model")
plt.ylabel("Average Confidence Score")
plt.xlabel("Model")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# =========================
# 6. Average Confidence on Failures
# =========================

fail_confidence = (
    data[data["is_fail"]]
    .groupby("model")["confidence_score"]
    .mean()
)

plt.figure(figsize=(8, 5))
fail_confidence.plot(kind="bar")
plt.title("Average Confidence on Hallucinated / Wrong Answers")
plt.ylabel("Average Confidence Score")
plt.xlabel("Model")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# =========================
# 7. Percent of Failures with Confidence >= 90
# =========================

high_conf_fail_rate = (
    data[data["is_fail"]]
    .groupby("model")["is_high_confidence_fail"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(8, 5))
high_conf_fail_rate.plot(kind="bar")
plt.title("Percent of Failures with Confidence ≥ 90")
plt.ylabel("High-Confidence Failures (%)")
plt.xlabel("Model")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# =========================
# 8. Confidence Distribution by Correctness
# =========================

plt.figure(figsize=(8, 5))

for label, subset in data.groupby("correct"):
    if label == 1:
        name = "Correct"
    elif label == 0:
        name = "Hallucination / Wrong"
    elif label == -1:
        name = "Critical Fail / Refusal"
    else:
        name = str(label)

    subset["confidence_score"].dropna().plot(kind="hist", alpha=0.5, bins=10, label=name)

plt.title("Confidence Score Distribution by Outcome")
plt.xlabel("Confidence Score")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 9. Print Summary Table
# =========================

summary = data.groupby("model").agg(
    total_questions=("correct", "count"),
    accuracy_percent=("is_correct", lambda x: x.mean() * 100),
    hallucination_rate_percent=("is_fail", lambda x: x.mean() * 100),
    critical_fail_rate_percent=("is_critical_fail", lambda x: x.mean() * 100),
    avg_confidence=("confidence_score", "mean"),
    avg_confidence_on_fails=("confidence_score", lambda x: data.loc[x.index][data.loc[x.index, "is_fail"]]["confidence_score"].mean()),
    high_confidence_fail_rate=("is_high_confidence_fail", lambda x: x.mean() * 100)
)



print("\n=== Summary Metrics ===")
print(summary.round(2))

# =========================
# 10. Save Summary to CSV
# =========================

summary.to_csv("summary_metrics.csv")
print("\nSaved summary_metrics.csv")