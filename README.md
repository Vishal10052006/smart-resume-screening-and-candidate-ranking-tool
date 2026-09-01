# Smart Resume Screening and Candidate Ranking Tool

A machine-learning application for comparing resumes with job requirements and producing an explainable candidate ranking.

## What it does

A recruiter enters a job description and uploads multiple resumes. The application extracts the resume text, identifies relevant skills, calculates a match score, and ranks candidates. When a trained model is available, the ranking uses supervised NLP; otherwise the application falls back to transparent TF-IDF similarity.

The system is designed for **first-pass screening assistance**. Final hiring decisions remain with the recruiter.

## Project Pipeline

```text
Kaggle Dataset
      |
      v
Data Collection
      |
      v
Data Cleaning + Validation
      |
      v
Exploratory Data Analysis
      |
      v
Feature Engineering
      |
      +--> TF-IDF unigrams/bigrams
      +--> Skill coverage
      +--> Matched-skill signals
      |
      v
Ridge Regression
      |
      v
Evaluation
      |
      +--> MAE / RMSE / R2
      +--> NDCG@10
      |
      v
FastAPI Application
      |
      v
Candidate Ranking + Skill Gaps
```

## Dataset

The development dataset is **Resume Data for Ranking** from Kaggle:

https://www.kaggle.com/datasets/thejohnwick001/resume-data-for-ranking

The local experiment contains **9,544 records and 35 columns** with resume information, job requirements, and the `matched_score` target. The raw CSV is intentionally not stored in this repository.

Place it at:

```text
data/raw/resume_data_for_ranking.csv
```

## Data Preparation

### Clean the dataset

```bash
python -m ml.data_cleaning --input data/raw/resume_data_for_ranking.csv
```

The cleaner:

- normalizes column names;
- normalizes text whitespace;
- removes exact duplicate rows;
- validates `matched_score` in `[0, 1]`;
- removes only invalid-target rows;
- fills missing text with empty strings;
- avoids artificial numeric zero imputation;
- writes a reproducible quality report.

### Run EDA

```bash
python -m ml.eda --data data/processed/cleaned_resume_data.csv
```

The current EDA reports 9,544 records, 28 job positions, no exact duplicates, no invalid targets, and no remaining missing cells after cleaning. The target mean is 0.6608 and the median is 0.6833. Detailed reports are under `data/reports/`.

## Feature Engineering

Feature construction is shared between training and API inference to avoid train/inference mismatch.

The representation contains:

- resume-side text;
- job-side text;
- TF-IDF word and bigram features;
- phrase-aware skill extraction;
- skill coverage buckets;
- matched-skill count signals.

`matched_score` is never used as an input feature.

## Model Training

Run:

```bash
python -m ml.train --data data/processed/cleaned_resume_data.csv
```

The training pipeline:

1. builds the engineered resume/job representation;
2. evaluates a random 80/20 hold-out;
3. evaluates a stricter job-position-grouped hold-out;
4. calculates MAE, RMSE, R², and NDCG@10;
5. refits the final Ridge model on the complete cleaned dataset;
6. saves `models/resume_match_ridge.joblib` and `models/evaluation.json`.

The binary model is excluded from GitHub. This keeps the repository lightweight and makes training reproducible from the documented source dataset.

### Baseline reference

An earlier TF-IDF + Ridge experiment achieved:

| Metric | Baseline |
|---|---:|
| MAE | 0.0919 |
| RMSE | 0.1195 |
| R² | 0.4840 |

These values are retained only as a baseline reference. They are **not** the evaluation of the upgraded feature pipeline. Run `ml.train` to generate the current metrics.

## Application

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### Start

```bash
uvicorn backend.app.main:app --reload
```

Open:

- `http://127.0.0.1:8000` — recruiter interface
- `http://127.0.0.1:8000/docs` — interactive API documentation
- `http://127.0.0.1:8000/health` — service health

### API

`POST /api/analyze` accepts:

- `job_description` — job description text;
- `resumes` — one or more PDF, DOCX, or TXT files.

The response contains ranked candidates, predicted score, semantic similarity, skill coverage, matched skills, missing skills, detected email, and extracted experience when available.

## Testing

Run the automated tests with:

```bash
python -m pytest -q
```

GitHub Actions runs the test suite on pushes to `main` and on pull requests.

## Docker

Build and run:

```bash
docker build -t smart-resume-screening .
docker run --rm -p 8000:8000 smart-resume-screening
```

The application can still demonstrate the screening workflow without a trained model because it has a TF-IDF similarity fallback.

## Repository Structure

```text
smart-resume-screening-and-candidate-ranking-tool/
|
├── .github/workflows/ci.yml
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── ml_predictor.py
│   └── requirements.txt
├── data/
│   ├── raw/README.md
│   ├── processed/README.md
│   └── reports/
├── docs/
│   └── PROJECT_REPORT.md
├── ml/
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── features.py
│   ├── preprocessing.py
│   └── train.py
├── models/
│   ├── README.md
│   └── evaluation.json
├── tests/
│   ├── test_api.py
│   └── test_features.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── LICENSE
└── README.md
```

## Current Limitations

This is an academic/project MVP. It does not claim production-grade hiring accuracy. Further work should include OCR for scanned resumes, stronger semantic embeddings, richer experience/education extraction, external validation, fairness testing, authentication, secure persistence, monitoring, and a larger representative dataset.

## Responsible Use

Resume data can contain personally identifiable information. Do not commit real candidate documents or private recruitment records. Use appropriately licensed, anonymized, or synthetic data for development.

A model score is a ranking signal, not a hiring decision. Human review is required before any employment action.

## Author

**Vishal Raj**  
Computer Science & Engineering (AI)

GitHub: https://github.com/Vishal10052006
