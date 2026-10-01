import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, kruskal

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

df = pd.read_csv("df_cohort_cleaned.csv")

# ------------------------------------------------------------
# 2. Heart rate group
# ------------------------------------------------------------

hr_bins = [-np.inf, 59, 69, 79, 89, np.inf]
hr_labels = ["<60 bpm", "60-69 bpm", "70-79 bpm", "80-89 bpm", ">=90 bpm"]

df["heart_rate_group"] = pd.cut(
    df["pulse_rate_60"],
    bins=hr_bins,
    labels=hr_labels,
    right=True
)

# ------------------------------------------------------------
# 3. Derived variables for Table 1
# ------------------------------------------------------------

# Age group
df["age_group"] = pd.cut(
    df["age_numeric"],
    bins=[-np.inf, 39, 59, np.inf],
    labels=["<40 year", "40-59 year", ">=60 year"],
    right=True
)

# Sex
df["sex_group"] = df["sex_numeric"].map({
    1: "Male",
    2: "Female"
})

# Stress
# BP1_reversed:
# 1 = 거의 느끼지 않음
# 2 = 조금 느낌
# 3 = 많이 느낌
# 4 = 매우 많이 느낌
df["stress_status"] = df["BP1_reversed"].map({
    1: "Low",
    2: "Low",
    3: "High",
    4: "High"
})

# Physical activity
# 현재 변수는 minutes/week이므로 MET-min/week와 완전히 같지는 않음.
# 일단 선행논문 표 형식에 맞춰 600 기준으로 분류.

df["physical_activity_group"] = df["physical_activity_group"].replace("nan", np.nan)

# BMI
if "BMI_numeric" in df.columns and "BMI_group" not in df.columns:
    df["BMI_group"] = pd.cut(
        df["BMI_numeric"],
        bins=[-np.inf, 23, 25, np.inf],
        labels=["<23 kg/m2", "23-24.9 kg/m2", ">=25 kg/m2"],
        right=False
    )

# Income
df["income_group"] = df["household_income"].map({
    1: "Low",
    2: "Middle-low",
    3: "Middle-high",
    4: "High"
})

# Education
df["education_group"] = df["edu_numeric"].map({
    1: "Primary school or below",
    2: "Middle school",
    3: "High school",
    4: "College or above"
})

# Marital status
df["marital_status"] = df["marri_1_numeric"].map({
    1: "Yes",
    2: "No"
})

# Working status
if "employment_status" in df.columns:
    df["working_status"] = df["employment_status"].map({
        1: "Yes",
        2: "No"
    })

# Smoking
df["smoking_group"] = df["smoking_history"].map({
    1: "Never",
    2: "Previous",
    3: "Current"
})

# Alcohol
df["alcohol_group"] = df["BD1_11_numeric"].map({
    1: "Never",
    2: "Previous",
    3: "Current"
})

# PHQ group
df["phq_severity"] = df["PHQ_group"].astype(str)

# ------------------------------------------------------------
# 4. Formatting helpers
# ------------------------------------------------------------

def format_p(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}".lstrip("0")


def mean_sd(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return ""
    return f"{series.mean():.1f}±{series.std(ddof=1):.1f}"


def n_pct(n, denom):
    if denom == 0 or pd.isna(denom):
        return f"{n} (0.0)"
    return f"{n:,} ({n / denom * 100:.1f})"


# ------------------------------------------------------------
# 5. P-value functions
# ------------------------------------------------------------

def pvalue_continuous(data, value_col, group_col="heart_rate_group"):
    temp = data[[value_col, group_col]].dropna()
    groups = [
        temp.loc[temp[group_col] == g, value_col].dropna()
        for g in hr_labels
    ]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan
    return kruskal(*groups).pvalue


def pvalue_categorical(data, cat_col, group_col="heart_rate_group"):
    temp = data[[cat_col, group_col]].dropna()
    if temp.empty:
        return np.nan
    
    table = pd.crosstab(temp[cat_col], temp[group_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    
    return chi2_contingency(table).pvalue


# ------------------------------------------------------------
# 6. Table builders
# ------------------------------------------------------------

def add_continuous_row(rows, data, label, value_col):
    p = pvalue_continuous(data, value_col)
    
    row = {
        "Variable": label,
        "Level": "M ± SD",
        "All": mean_sd(data[value_col]),
        "p-value": format_p(p)
    }
    
    for g in hr_labels:
        row[g] = mean_sd(data.loc[data["heart_rate_group"] == g, value_col])
    
    rows.append(row)


def add_categorical_block(rows, data, label, cat_col, levels):
    p = pvalue_categorical(data, cat_col)
    
    # Header row
    rows.append({
        "Variable": label,
        "Level": "",
        "All": "",
        **{g: "" for g in hr_labels},
        "p-value": format_p(p)
    })
    
    total_n = len(data)
    
    for level in levels:
        level_data = data[data[cat_col] == level]
        level_n = len(level_data)
        
        row = {
            "Variable": "",
            "Level": level,
            "All": n_pct(level_n, total_n),
            "p-value": ""
        }
        
        # 선행논문 Table 1처럼 각 level 안에서 heart-rate group 분포 %
        for g in hr_labels:
            group_total = len(data[data["heart_rate_group"] == g])
            count = len(level_data[level_data["heart_rate_group"] == g])
            row[g] = n_pct(count, group_total)
        
        rows.append(row)
        # Missing row
    missing_n = data[cat_col].isna().sum()
    if missing_n > 0:
        missing_data = data[data[cat_col].isna()]
        
        row = {
            "Variable": "",
            "Level": "Missing",
            "All": n_pct(missing_n, total_n),
            "p-value": ""
        }
        
        # Missing row는 row percent가 아니라 column percent로 계산
        for g in hr_labels:
            group_total = len(data[data["heart_rate_group"] == g])
            count = len(missing_data[missing_data["heart_rate_group"] == g])
            row[g] = n_pct(count, group_total)
        
        rows.append(row)


def make_table1(data):
    rows = []
    
    # Column header N
    total_n = len(data)
    hr_ns = data["heart_rate_group"].value_counts(dropna=False).reindex(hr_labels).fillna(0).astype(int)
    
    rows.append({
        "Variable": "N",
        "Level": "",
        "All": f"{total_n:,}",
        **{g: f"{hr_ns[g]:,}" for g in hr_labels},
        "p-value": ""
    })
    
    # Age
    add_continuous_row(rows, data, "Age", "age_numeric")
    add_categorical_block(
        rows, data, "Age group, n (%)", "age_group",
        ["<40 year", "40-59 year", ">=60 year"]
    )
    
    # Sex
    add_categorical_block(
        rows, data, "Sex, n (%)", "sex_group",
        ["Male", "Female"]
    )
    
    # BMI
    if "BMI_numeric" in data.columns:
        add_continuous_row(rows, data, "BMI, kg/m2", "BMI_numeric")
    if "BMI_group" in data.columns:
        add_categorical_block(
            rows, data, "BMI, kg/m2, n (%)", "BMI_group",
            ["<23 kg/m2", "23-24.9 kg/m2", ">=25 kg/m2"]
        )
    
    # Smoking
    add_categorical_block(
        rows, data, "Smoking, n (%)", "smoking_group",
        ["Never", "Previous", "Current"]
    )
    
    # Alcohol
    add_categorical_block(
        rows, data, "Alcohol, n (%)", "alcohol_group",
        ["Never", "Previous", "Current"]
    )
    
    # Education
    add_categorical_block(
        rows, data, "Education, n (%)", "education_group",
        ["Primary school or below", "Middle school", "High school", "College or above"]
    )
    
    # Income
    add_categorical_block(
        rows, data, "Income, n (%)", "income_group",
        ["Low", "Middle-low", "Middle-high", "High"]
    )
    
    # Working status
    if "working_status" in data.columns:
        add_categorical_block(
            rows, data, "Working status, n (%)", "working_status",
            ["Yes", "No"]
        )
    
    # Marital status
    add_categorical_block(
        rows, data, "Marital status, n (%)", "marital_status",
        ["Yes", "No"]
    )
    
    # Physical activity
    add_categorical_block(
        rows, data, "Total physical activity, n (%)", "physical_activity_group",
        ["<600 MET-min/week", ">=600 MET-min/week"]
    )
    
    # Stress
    add_categorical_block(
        rows, data, "Stress status, n (%)", "stress_status",
        ["Low", "High"]
    )
    
    # PHQ-9 severity group
    add_categorical_block(
        rows, data, "PHQ-9 severity, n (%)", "phq_severity",
        ["Minimal", "Mild_or_greater"]
    )
    
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 7. Create tables
# ------------------------------------------------------------

table1_all = make_table1(df)

table1_men = make_table1(df[df["sex_numeric"] == 1].copy())
table1_women = make_table1(df[df["sex_numeric"] == 2].copy())

# ------------------------------------------------------------
# 8. Export
# ------------------------------------------------------------

with pd.ExcelWriter("table1_characteristics_by_heart_rate.xlsx", engine="openpyxl") as writer:
    table1_all.to_excel(writer, sheet_name="Total", index=False)
    table1_men.to_excel(writer, sheet_name="Male", index=False)
    table1_women.to_excel(writer, sheet_name="Female", index=False)

print("Saved:")
print("table1_characteristics_by_heart_rate.xlsx")

print("\nPreview:")
print(table1_all.head(30))
