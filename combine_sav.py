from pathlib import Path
import re

import pandas as pd
import pyreadstat


# Folder containing all KNHANES .sav files.
# By default, this uses the same folder where this Python file is saved.
SAV_FOLDER = Path(__file__).resolve().parent

# Output CSV file.
OUTPUT_FILE = SAV_FOLDER / "KNHANES_2016_2024_union.csv"

# Keywords to inspect in each file and in the final combined dataset.
KEYWORDS = ["PHQ", "GAD", "BP5", "mh_"]


def extract_year_from_filename(file_path: Path):
    """
    Extract a year from the filename.
    First tries a 4-digit year such as 2016, 2018, 2020, 2022, or 2024.
    If that is not found, also handles KNHANES names like HN16_all.sav.
    Returns pandas NA if no year is found.
    """
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", file_path.name)
    if match:
        return int(match.group(1))

    match = re.search(r"\bHN(\d{2})(?:\b|_)", file_path.name, flags=re.IGNORECASE)
    if match:
        year_two_digits = match.group(1)
        return int(f"20{year_two_digits}")

    return pd.NA


def find_keyword_columns(df: pd.DataFrame, keywords=KEYWORDS):
    """Return columns whose names contain any keyword, case-insensitive."""
    matched_columns = []

    for col in df.columns:
        col_text = str(col)
        col_lower = col_text.lower()

        if any(keyword.lower() in col_lower for keyword in keywords):
            matched_columns.append(col_text)

    return matched_columns


def read_sav_file(file_path: Path) -> pd.DataFrame:
    """
    Read one SPSS .sav file while preserving original numeric codes and
    SPSS user-missing values.
    """
    df, meta = pyreadstat.read_sav(
        file_path,
        apply_value_formats=False,
        user_missing=True,
    )

    # Avoid overwriting an existing year column from the source file.
    if "year" in df.columns:
        df = df.rename(columns={"year": "year_original"})

    df["year"] = extract_year_from_filename(file_path)

    keyword_columns = find_keyword_columns(df)

    print("=" * 80)
    print(f"File: {file_path.name}")
    print(f"Shape: {df.shape}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Extracted year: {df['year'].iloc[0]}")
    print(f"Keyword columns ({len(keyword_columns)}):")
    print(keyword_columns)

    return df


def main():
    sav_files = sorted(SAV_FOLDER.glob("*.sav"))

    if not sav_files:
        raise FileNotFoundError(f"No .sav files found in: {SAV_FOLDER}")

    yearly_dataframes = []

    for sav_file in sav_files:
        df = read_sav_file(sav_file)
        yearly_dataframes.append(df)

    # Outer union of columns:
    # variables that appear only in some years are preserved, and missing
    # values in years where they do not exist become NaN.
    combined_df = pd.concat(
        yearly_dataframes,
        join="outer",
        ignore_index=True,
        sort=False,
    )

    final_keyword_columns = find_keyword_columns(combined_df, keywords=["PHQ", "GAD"])

    print("=" * 80)
    print("Final combined dataset")
    print(f"Final shape: {combined_df.shape}")
    print(f"Total number of columns: {combined_df.shape[1]}")
    print(f"Final PHQ/GAD-related columns ({len(final_keyword_columns)}):")
    print(final_keyword_columns)

    combined_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print(f"Saved CSV file to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
