#!/usr/bin/env bash
# Run the reproducible data-to-model pipeline from the project root.
set -euo pipefail

RAW_DATA="${1:-data/raw/resume_data_for_ranking.csv}"
CLEAN_DATA="data/processed/cleaned_resume_data.csv"

if [[ ! -f "$RAW_DATA" ]]; then
  echo "Raw dataset not found: $RAW_DATA"
  echo "Download the Kaggle dataset and place it at data/raw/resume_data_for_ranking.csv"
  exit 1
fi

python -m ml.data_cleaning --input "$RAW_DATA"
python -m ml.eda --data "$CLEAN_DATA"
python -m ml.train --data "$CLEAN_DATA"
python -m pytest -q

echo "Pipeline completed successfully."
