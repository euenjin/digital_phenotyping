# Digital Phenotyping

This repository contains KNHANES digital phenotyping data preparation and exploratory analysis scripts.

## Files

- `combine_sav.py`: combines KNHANES `.sav` files into one union CSV while preserving year-specific columns.
- `1_data_cleaning.py`: loads `HN16_24_even_all.csv`, cleans PHQ, self-rated health, pulse irregularity, and physical activity variables, then saves EDA chart images.
- `*.png`: generated charts from the current EDA workflow.

## Data

Large raw and processed dataset files are not tracked in Git. Place the required KNHANES `.sav` and `.csv` files in the project folder before running the scripts.

## Run

```powershell
python 1_data_cleaning.py
```

