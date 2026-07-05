import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_PATH = OUTPUT_DIR / "HN16_24_even_all.csv"
VALID_BP1 = [1, 2, 3, 4]

# 1. Load original data
raw_df = pd.read_csv(DATA_PATH, low_memory=False)

# 원본 보존용 cleaning dataframe
df_clean = raw_df.copy()

print("Original raw_df shape:", raw_df.shape)
print("df_clean initial shape:", df_clean.shape)


# 2. Clean PHQ-9 score variable

df_clean["PHQ_sum"] = pd.to_numeric(
    df_clean["mh_PHQ_S"].astype(str).str.strip(),
    errors="coerce"
)

invalid_phq_mask = df_clean["PHQ_sum"].notna() & ~df_clean["PHQ_sum"].between(0, 27)

print("\nPHQ_sum check:")
print("Invalid PHQ_sum outside 0–27:", invalid_phq_mask.sum())

df_clean.loc[invalid_phq_mask, "PHQ_sum"] = pd.NA

print("PHQ_sum missing:", df_clean["PHQ_sum"].isna().sum())
print(df_clean["PHQ_sum"].describe())


# 3. Clean BP1 self-rated health variable

df_clean["BP1_numeric"] = pd.to_numeric(
    df_clean["BP1"].astype(str).str.strip(),
    errors="coerce"
)

df_clean["BP1_numeric"] = df_clean["BP1_numeric"].where(
    df_clean["BP1_numeric"].isin(VALID_BP1)
)

df_clean["BP1_reversed"] = df_clean["BP1_numeric"].replace({
    1: 4,
    2: 3,
    3: 2,
    4: 1
})

print("\nBP1 reversed check:")
print(df_clean["BP1_reversed"].value_counts(dropna=False).sort_index())


# 4. Clean pulse regularity variable

df_clean["HE_rPLS_numeric"] = pd.to_numeric(
    df_clean["HE_rPLS"].astype(str).str.strip(),
    errors="coerce"
)

df_clean["HE_rPLS_numeric"] = df_clean["HE_rPLS_numeric"].where(
    df_clean["HE_rPLS_numeric"].isin([1, 2])
)

df_clean["pulse_irregular"] = df_clean["HE_rPLS_numeric"].replace({
    1: 0,
    2: 1
})

print("\nPulse irregularity check:")
print(df_clean["pulse_irregular"].value_counts(dropna=False).sort_index())


# 5. Clean physical activity variables
# 활동량 column
activity_sets = [
    ("BE3_71", "BE3_72", "BE3_73", "BE3_74"), #71=yes/no, 72=days, 73=hours, 74=minutes (vigorous work)
    ("BE3_75", "BE3_76", "BE3_77", "BE3_78"), #75=yes/no, 76=days, 77=hours, 78=minutes (vigorous leisure)
    ("BE3_81", "BE3_82", "BE3_83", "BE3_84"), #81=yes/no, 82=days, 83=hours, 84=minutes (moderate work)
    ("BE3_85", "BE3_86", "BE3_87", "BE3_88"), #85=yes/no, 86=days, 87=hours, 88=minutes (moderate leisure)
    ("BE3_91", "BE3_92", "BE3_93", "BE3_94"), #91=yes/no, 92=days, 93=hours, 94=minutes (movement between places)
]

activity_original_cols = [col for activity_set in activity_sets for col in activity_set]

def clean_activity_inputs(participation_col, days_col, hours_col, minutes_col):
    for col in [participation_col, days_col, hours_col, minutes_col]:
        df_clean[f"{col}_clean"] = pd.to_numeric(
            df_clean[col].astype(str).str.strip(),
            errors="coerce"
        )

    df_clean[f"{participation_col}_clean"] = df_clean[f"{participation_col}_clean"].where(
        df_clean[f"{participation_col}_clean"].isin([1, 2])
    )
    df_clean[f"{days_col}_clean"] = df_clean[f"{days_col}_clean"].where(
        df_clean[f"{days_col}_clean"].between(1, 7)
    )
    df_clean[f"{hours_col}_clean"] = df_clean[f"{hours_col}_clean"].where(
        df_clean[f"{hours_col}_clean"].between(0, 24)
    )
    df_clean[f"{minutes_col}_clean"] = df_clean[f"{minutes_col}_clean"].where(
        df_clean[f"{minutes_col}_clean"].between(0, 59)
    )


# Physical activity is based on activities performed for at least 10 consecutive minutes.
# No activity in a domain is coded as 0 minutes; unknown/nonresponse remains missing.
def weekly_activity_minutes(participation_col, days_col, hours_col, minutes_col):
    result = pd.Series(pd.NA, index=df_clean.index, dtype="Float64")

    participation = df_clean[participation_col]
    days = df_clean[days_col]
    hours = df_clean[hours_col]
    minutes = df_clean[minutes_col]

    result.loc[participation.eq(2)] = 0

    yes_activity = participation.eq(1)
    has_days = days.notna()
    has_time = hours.notna() & minutes.notna()
    can_calculate = yes_activity & has_days & has_time

    daily_minutes = hours.fillna(0) * 60 + minutes.fillna(0)
    result.loc[can_calculate] = days.loc[can_calculate] * daily_minutes.loc[can_calculate]

    return result


activity_component_minutes = []

for participation_col, days_col, hours_col, minutes_col in activity_sets:
    clean_activity_inputs(participation_col, days_col, hours_col, minutes_col)

    activity_component_minutes.append(
        weekly_activity_minutes(
            f"{participation_col}_clean",
            f"{days_col}_clean",
            f"{hours_col}_clean",
            f"{minutes_col}_clean",
        )
    )

df_clean["total_mvpa_min_week"] = pd.concat(activity_component_minutes, axis=1).sum(
    axis=1,
    min_count=len(activity_component_minutes),
).astype("Float64")

activity_clean_cols = [f"{col}_clean" for col in activity_original_cols]
activity_weekly_cols = [
    "total_mvpa_min_week",
]


def numeric_unique_values(col):
    values = pd.to_numeric(df_clean[col], errors="coerce").dropna()
    return sorted(values.astype(int).unique().tolist())


print("\nOriginal BE3 unique values:")
for col in activity_original_cols:
    print(f"{col}: {numeric_unique_values(col)}")

print("\nCleaned BE3 unique values:")
for col in activity_clean_cols:
    print(f"{col}: {numeric_unique_values(col)}")

print("\nPhysical activity missing counts:")
print(df_clean[activity_weekly_cols].isna().sum())

print("\nPhysical activity summary statistics:")
print(df_clean[activity_weekly_cols].describe())

print("\nPhysical activity quantiles:")
print(df_clean[activity_weekly_cols].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]))


# 6. Create current EDA charts

chart_specs = [
    ("PHQ_sum", "PHQ_sum_bar_chart.png"),
    ("BP1_numeric", "BP1_original_bar_chart.png"),
    ("BP1_reversed", "BP1_reversed_bar_chart.png"),
    ("pulse_irregular", "pulse_irregular_bar_chart.png"),
]

for variable, output_name in chart_specs:
    counts = df_clean[variable].dropna().astype(int).value_counts().sort_index()

    print(f"\nValue counts for {variable}:")
    print(counts)

    ax = counts.plot(kind="bar", figsize=(12, 6))
    ax.set_title(f"Distribution of {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Count")
    ax.bar_label(ax.containers[0], fontsize=8)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=300)
    plt.close()

    print(f"Saved chart: {output_name}")


activity_bins = [-0.5, 0.5, 29.5, 59.5, 119.5, 149.5, 299.5, 599.5, float("inf")]
activity_bin_labels = ["0", "1-29", "30-59", "60-119", "120-149", "150-299", "300-599", "600+"]
activity_chart_specs = [
    ("total_mvpa_min_week", "total_mvpa_min_week_barchart.png"),
]

for variable, output_name in activity_chart_specs:
    binned = pd.cut(
        df_clean[variable],
        bins=activity_bins,
        labels=activity_bin_labels,
        include_lowest=True,
    )
    counts = binned.value_counts().sort_index()

    print(f"\nBinned counts for {variable}:")
    print(counts)

    ax = counts.plot(kind="bar", figsize=(12, 6))
    ax.set_title(f"Binned distribution of {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Count")
    ax.bar_label(ax.containers[0], fontsize=8)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=300)
    plt.close()

    print(f"Saved chart: {output_name}")


print("Current df_clean shape:", df_clean.shape)
