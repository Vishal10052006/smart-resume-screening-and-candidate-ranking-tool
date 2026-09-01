# Processed Dataset

This directory contains generated datasets produced by the cleaning pipeline.

Generate the cleaned dataset with:

```bash
python -m ml.data_cleaning --input data/raw/resume_data_for_ranking.csv
```

Expected output:

```text
data/processed/cleaned_resume_data.csv
```

The processed CSV is generated from the Kaggle source and may be large, so it is kept out of GitHub. The cleaning report under `data/reports/` records the transformation and quality checks.
