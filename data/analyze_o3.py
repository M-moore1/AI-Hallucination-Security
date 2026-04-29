from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

files = {
    "o3": BASE_DIR / "TestData_o3.csv",
    "opus4.6": BASE_DIR / "TestData_opus4.6.csv",
    "Gemini": BASE_DIR / "TestData_Gemini.csv",
}

# Change these if needed
correct_col = "correct"              # 1 = correct, 0 = wrong, -1 = critical fail/refusal
confidence_col = "confidence_score"
category_col = "category"
response_col = "model_response"
truth_col = "ground_truth"


def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def ratio(x):
    if pd.isna(x):
        return 0
    return round(x * 100, 2)


all_summaries = []

for model_name, csv_path in files.items():
    print("\n" + "=" * 90)
    print(f"MODEL: {model_name}")
    print("=" * 90)

    df = pd.read_csv(csv_path)
    df = clean_columns(df)

    df[correct_col] = pd.to_numeric(df[correct_col], errors="coerce")
    df[confidence_col] = pd.to_numeric(df[confidence_col], errors="coerce")

    total = len(df)

    correct_df = df[df[correct_col] == 1]
    wrong_df = df[df[correct_col] == 0]
    critical_df = df[df[correct_col] == -1]
    fail_df = df[df[correct_col].isin([0, -1])]

    accuracy = len(correct_df) / total if total > 0 else 0
    wrong_rate = len(wrong_df) / total if total > 0 else 0
    critical_rate = len(critical_df) / total if total > 0 else 0
    fail_rate = len(fail_df) / total if total > 0 else 0

    avg_conf_all = df[confidence_col].mean()
    avg_conf_correct = correct_df[confidence_col].mean()
    avg_conf_wrong = wrong_df[confidence_col].mean()
    avg_conf_critical = critical_df[confidence_col].mean()
    avg_conf_fails = fail_df[confidence_col].mean()

    overconfident_fails_80 = fail_df[fail_df[confidence_col] >= 80]
    high_conf_fails_90 = fail_df[fail_df[confidence_col] > 90]
    low_conf_correct = correct_df[correct_df[confidence_col] < 50]

    fail_conf_above_90_pct = (
        ratio(len(high_conf_fails_90) / len(fail_df))
        if len(fail_df) > 0 else 0
    )

    print("\nOVERALL METRICS")
    print("-" * 90)
    print(f"Accuracy:                         {ratio(accuracy)}%")
    print(f"Wrong rate:                       {ratio(wrong_rate)}%")
    print(f"Critical fail/refusal rate:       {ratio(critical_rate)}%")
    print(f"Total fail rate:                  {ratio(fail_rate)}%")
    print()
    print(f"Avg confidence overall:           {avg_conf_all:.2f}")
    print(f"Avg confidence correct:           {avg_conf_correct:.2f}")
    print(f"Avg confidence wrong:             {avg_conf_wrong:.2f}")
    print(f"Avg confidence critical:          {avg_conf_critical:.2f}")
    print(f"Avg confidence all fails:         {avg_conf_fails:.2f}")
    print()
    print(f"Overconfident fail rate >= 80:    {ratio(len(overconfident_fails_80) / total) if total > 0 else 0}%")
    print(f"% of fails with confidence > 90:  {fail_conf_above_90_pct}%")
    print(f"Low-confidence correct rate < 50: {ratio(len(low_conf_correct) / total) if total > 0 else 0}%")

    all_summaries.append({
        "model": model_name,
        "accuracy_%": ratio(accuracy),
        "wrong_rate_%": ratio(wrong_rate),
        "critical_fail_rate_%": ratio(critical_rate),
        "total_fail_rate_%": ratio(fail_rate),
        "avg_conf_all": round(avg_conf_all, 2),
        "avg_conf_correct": round(avg_conf_correct, 2),
        "avg_conf_wrong": round(avg_conf_wrong, 2),
        "avg_conf_critical": round(avg_conf_critical, 2),
        "avg_conf_fails": round(avg_conf_fails, 2),
        "overconfident_fail_rate_>=80_%": ratio(len(overconfident_fails_80) / total) if total > 0 else 0,
        "fails_conf_above_90_%": fail_conf_above_90_pct,
        "low_conf_correct_rate_<50_%": ratio(len(low_conf_correct) / total) if total > 0 else 0,
    })

    print("\nCATEGORY RATIOS")
    print("-" * 90)

    category_summary = (
        df.groupby(category_col)
        .apply(lambda g: pd.Series({
            "accuracy_%": ratio((g[correct_col] == 1).mean()),
            "wrong_%": ratio((g[correct_col] == 0).mean()),
            "critical_fail_%": ratio((g[correct_col] == -1).mean()),
            "fail_%": ratio(g[correct_col].isin([0, -1]).mean()),
            "avg_conf_all": round(g[confidence_col].mean(), 2),
            "avg_conf_correct": round(g[g[correct_col] == 1][confidence_col].mean(), 2),
            "avg_conf_fails": round(g[g[correct_col].isin([0, -1])][confidence_col].mean(), 2),
            "fails_conf_above_90_%": (
                ratio(
                    len(g[(g[correct_col].isin([0, -1])) & (g[confidence_col] > 90)])
                    / len(g[g[correct_col].isin([0, -1])])
                )
                if len(g[g[correct_col].isin([0, -1])]) > 0 else 0
            ),
        }))
        .sort_values(by="accuracy_%", ascending=False)
    )

    print(category_summary.to_string())

    print("\nFAILURES BY CATEGORY")
    print("-" * 90)

    if fail_df.empty:
        print("No failures.")
    else:
        for category, group in fail_df.groupby(category_col):
            print(f"\n[{category}]")

            fail_table = group.copy()
            fail_table["row"] = fail_table.index + 2
            fail_table["fail_type"] = fail_table[correct_col].map({
                0: "WRONG",
                -1: "CRITICAL / REFUSAL"
            })

            fail_table["confidence"] = fail_table[confidence_col]

            fail_table["model_response"] = (
                fail_table[response_col]
                .astype(str)
                .str.replace("\n", " ", regex=False)
                .str.strip()
            )

            fail_table["ground_truth"] = (
                fail_table[truth_col]
                .astype(str)
                .str.replace("\n", " ", regex=False)
                .str.strip()
            )

            display_cols = [
                "row",
                "fail_type",
                "confidence",
                "model_response",
                "ground_truth"
            ]

            print(
                fail_table[display_cols]
                .to_string(index=False, max_colwidth=60)
            )



print("\n" + "=" * 90)
print("MODEL COMPARISON SUMMARY")
print("=" * 90)

summary_df = pd.DataFrame(all_summaries)
print(summary_df.to_string(index=False))