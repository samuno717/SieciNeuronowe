#%%
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns

from config import DATA_DIR

columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
           "thal", "num"]
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]


df = pd.read_csv(DATA_DIR / "processed.cleveland.data", names=columns, na_values="?")


#%%
# Tabela 1. Statystyki ilościowe baz danych zbioru UCI Heart Disease
temp = []
databases = []
for path in Path(DATA_DIR).glob("processed*"):
    sub_df = pd.read_csv(path, header=None, na_values="?")
    temp.append({
        "Liczba wierszy": len(sub_df),
        "Suma braków": sub_df.isna().sum().sum()
    })
    databases.append(path.name)

df_databases = pd.DataFrame(temp, index=databases)
df_databases.index.name = "Baza danych"
print(df_databases)

#%%
# Tabela 3, 4, 5: Liczebność i udział procentowy klas, średnie i odchylenia standardowe
ex1 = df["num"].value_counts().reset_index()
ex1.columns = ["Class", "Samples"]
ex1 = ex1.set_index("Class")

total = ex1["Samples"].sum()
ex1["Percentage"] = (ex1["Samples"] / total * 100).round(2).astype(str) + "%"

sick = (df["num"] > 0).astype(int)
binary_df = pd.DataFrame(
    {
        "Samples": sick.value_counts(),
        "Percentage": (sick.value_counts(normalize=True) * 100).round(2).astype(str) + "%",
    },
    index=[0, 1]
).rename_axis("Class")

df_stats = df.drop(categorical_cols, axis=1).drop("num", axis=1)
statistics = df_stats.agg(["mean", "std"], numeric_only=True).round(2)
print(statistics)

#%%
# Tabela 6. Shapiro Wilk
results = []
for col in continuous_cols:
    data = df[col].dropna()
    stat, p = stats.shapiro(data)
    results.append(
        {
            "Feature": col,
            "Shapiro W": round(stat, 3),
            "p-value": f"{p:.4e}",
            "Is strictly normal? (p > 0.05)": p > 0.05,
        }
    )

shapiro_df = pd.DataFrame(results)
print()
print(shapiro_df)

#%%
# Kod wygenerowany - wykresy kwantyl-kwantyl
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
axes = axes.flatten()

for i, col in enumerate(continuous_cols):
    ax = axes[i]
    raw_data = df[col].dropna().to_numpy(dtype=float)

    sorted_data = np.sort((raw_data - np.mean(raw_data)) / np.std(raw_data))
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))

    ax.scatter(theoretical_quantiles, sorted_data, color="#3b82f6", s=35, alpha=0.7, edgecolors="white", linewidth=0.6,
               label="Data Points",)

    min_val = min(theoretical_quantiles.min(), sorted_data.min())
    max_val = max(theoretical_quantiles.max(), sorted_data.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, alpha=0.85, label="Reference Line",)

    ax.set_title(f"Q-Q Plot: {col}", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Theoretical Quantiles", fontsize=10, fontweight="bold")
    ax.set_ylabel("Sample Quantiles", fontsize=10, fontweight="bold")

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper left", frameon=True)

fig.delaxes(axes[5])

plt.tight_layout()
plt.show()

#%%
# Tabela 7 i 8 chi-kwadrat
tests_summary = []
dist_records = []

for col in categorical_cols:
    clean_series = df[col].dropna()
    total_count = len(clean_series)
    counts = clean_series.value_counts().sort_index()
    n_categories = len(counts)

    chi2_stat, p_val = stats.chisquare(counts)

    tests_summary.append(
        {
            "Cecha": col,
            "Liczba unikalnych klas": n_categories,
            "Wynik Chi2": round(chi2_stat, 2),
            "Wartość p": f"{p_val:.4e}",
            "Czy rozkład równomierny? (p > 0.05)": p_val > 0.05,
        }
    )

    for category, count in counts.items():
        percentage = (count / total_count) * 100
        dist_records.append(
            {
                "Cecha": col,
                "Klasa": category,
                "Liczba": count,
                "Procent": f"{percentage:.2f}%",
            }
        )

uniformity_df = pd.DataFrame(tests_summary)
dist_df = pd.DataFrame(dist_records)

print(uniformity_df.to_string(index=False))
print(dist_df.to_string(index=False))

#%%
# Kod wygenerowany - wartości brakujące + heatmapa
missing_counts = df.isna().sum()
missing_pct = (df.isna().mean() * 100).round(2)
missing_table = pd.DataFrame({"Liczba brakow": missing_counts, "Procent": missing_pct})[missing_counts > 0]

print(missing_table)

missing_per_dataset = {}

for path in sorted(Path(DATA_DIR).glob("processed*")):
    df_temp = pd.read_csv(path, header=None, names=columns, na_values="?", sep=r"\s+|,",
                          engine="python").drop('num', axis=1)

    missing_pct = (df_temp.isna().mean() * 100).round(1)
    dataset_name = path.name.replace("processed.", "").replace(".data", "")
    missing_per_dataset[dataset_name] = missing_pct

missing_summary = pd.DataFrame(missing_per_dataset)

print(missing_summary.to_string())

plt.figure(figsize=(9, 7), dpi=600)
sns.heatmap(
    missing_summary,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    cbar_kws={"label": "% brakujących danych"},
)
plt.title(
    "",
    fontsize=12,
    fontweight="bold",
    pad=12,
)
plt.ylabel("Cecha")
plt.xlabel("Nazwa bazy")
plt.tight_layout()
plt.show()