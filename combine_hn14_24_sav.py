from pathlib import Path

import pandas as pd
import pyreadstat


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "HN14_24_even_all.csv"

INPUT_FILES = [
    (BASE_DIR / "HN14_all.sav", 2014),
    (BASE_DIR / "HN16_all.sav", 2016),
    (BASE_DIR / "HN18_all.sav", 2018),
    (BASE_DIR / "HN20_all.sav", 2020),
    (BASE_DIR / "HN22_all.sav", 2022),
    (BASE_DIR / "HN24_ALL.sav", 2024),
]


def read_sav(path, fallback_year=None):
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    print(f"Reading {path.name}...")
    df, _ = pyreadstat.read_sav(
        str(path),
        apply_value_formats=False,
        user_missing=True,
    )

    df["year"] = fallback_year

    print(f"{path.name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    return df


frames = [read_sav(path, year) for path, year in INPUT_FILES]

df = pd.concat(frames, join="outer", sort=False, ignore_index=True)

if "year" in df.columns:
    df = df[["year"] + [col for col in df.columns if col != "year"]]

print(f"Combined: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
print(f"Writing {OUTPUT_FILE.name}...")
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print("Done.")
