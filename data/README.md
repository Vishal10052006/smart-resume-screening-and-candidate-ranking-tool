# Data

The project keeps raw and processed datasets separate. Private candidate resumes must not be committed to this repository.

## Dataset workflow

```text
Kaggle raw dataset
      |
      v
 data/raw/
      |
      v
 ml/data_cleaning.py
      |
      +--> data/processed/cleaned_resume_data.csv
      |
      +--> data/reports/data_cleaning_report.json
      |
      v
 Feature engineering / ML training
```

## Current dataset

The primary development dataset is a Kaggle resume-ranking dataset containing resume information, job requirements, and a `matched_score` target.

For the current local experiment:

- Rows: 9,544
- Columns: 35
- Exact duplicate rows: 0
- Invalid target values: 0
- Target: `matched_score`

The raw CSV should be downloaded from its licensed Kaggle source and placed locally at:

```text
data/raw/resume_data_for_ranking.csv
```

The raw dataset is intentionally not committed to GitHub unless its license explicitly permits redistribution.

## Cleaning

Run:

```bash
python -m ml.data_cleaning --input data/raw/resume_data_for_ranking.csv
```

The cleaner:

- normalizes column names;
- normalizes text whitespace;
- removes exact duplicate rows;
- validates `matched_score` as a numeric value in `[0, 1]`;
- removes only rows with invalid target values;
- converts missing text values to empty strings;
- does not replace missing numeric values with artificial zeros;
- writes a reproducible JSON quality report.

High missingness in optional resume fields is not automatically treated as a bad record. Feature selection and imputation decisions are handled separately during feature engineering.

## Privacy

Do not commit real candidate resumes, personally identifiable information, contact details, or private recruitment records. Use synthetic, anonymized, or appropriately licensed data for experiments.
