# Smart Resume Screening and Candidate Ranking Tool

A machine-learning application that compares candidate resumes with job requirements and produces an explainable candidate ranking.

## Overview

Initial resume screening is repetitive: recruiters read many documents, identify relevant skills, and compare candidates against the same requirements.

This project automates that first-pass workflow. A recruiter provides a job description and uploads multiple resumes. The system extracts text, builds resume/job representations, predicts a match score with a supervised NLP model, and shows skill-level diagnostics alongside the ranking.

The tool is intended to support screening, not replace human hiring decisions.

## Current MVP

- PDF, DOCX, and TXT resume parsing
- Multiple resume upload
- Job-description input
- Resume/job text preprocessing
- Dataset cleaning and validation pipeline
- Exploratory data analysis pipeline
- Supervised ML training pipeline
- TF-IDF feature extraction
- Ridge regression for match-score prediction
- Candidate ranking by predicted match score
- Skill extraction and skill-gap diagnostics
- Email and basic experience extraction
- Fallback TF-IDF similarity when a trained model artifact is unavailable
- FastAPI REST API
- Browser-based recruiter interface
- File-size and file-type validation
- Health-check endpoint

## Dataset

The initial training dataset is **Resume Data for Ranking** from Kaggle.

Source: https://www.kaggle.com/datasets/thejohnwick001/resume-data-for-ranking

The dataset contains **9,544 records and 35 columns**, including resume information, job requirements, and the `matched_score` target.

The raw dataset is not committed to this repository. Download it from Kaggle and place it at:

```text
data/raw/resume_data_for_ranking.csv
```

See `data/raw/README.md` for setup instructions.

## Data Preparation and EDA

The project keeps the raw source data separate from generated datasets. The cleaning pipeline:

- normalizes column names and text whitespace
- checks and removes exact duplicate records
- validates `matched_score` as a numeric value in the 0–1 range
- handles missing text without inventing numeric values
- writes a machine-readable quality report

Run cleaning with:

```bash
python -m ml.data_cleaning --input data/raw/resume_data_for_ranking.csv
```

Then run EDA on the cleaned data:

```bash
python -m ml.eda --data data/processed/cleaned_resume_data.csv
```

The EDA report examines target distribution, job-position distribution, resume/job text length, skill frequencies, and simple relationships between derived numeric features and the target.

### EDA findings

- **9,544** cleaned records and **35** columns
- **0** duplicate records
- **0** invalid target records removed
- **0** missing cells after cleaning
- **28** distinct job positions
- Mean `matched_score`: **0.6608**
- Median `matched_score`: **0.6833**
- Target range: **0.00–0.97**
- Resume-side text averages about **101 words** per record
- Job-side text averages about **64 words** per record

The job positions are represented at nearly equal frequencies. This suggests deliberate balancing or construction of the dataset, so model results should be validated on newly collected or external examples before production use.

Detailed outputs are stored in `data/reports/`.

## ML Pipeline

```text
Kaggle Dataset
      |
      v
Data Cleaning
      |
      v
Exploratory Data Analysis
      |
      v
Resume + Job Text Construction
      |
      v
TF-IDF Vectorization
      |
      v
Ridge Regression
      |
      v
Predicted Match Score
      |
      v
Candidate Ranking
```

The training representation uses candidate-side fields such as skills, education, experience, positions, responsibilities, and career objective together with job-side requirements, responsibilities, and required skills.

### Initial evaluation

The first development experiment used an 80/20 split with `random_state=42`.

| Metric | Result |
|---|---:|
| MAE | 0.0919 |
| RMSE | 0.1195 |
| R² | 0.4840 |

These are development results on a single hold-out split and should not be interpreted as production performance. Role-aware and external validation are planned improvements.

The detailed values are stored in `models/evaluation.json`.

## Training the Model

After downloading the dataset and generating the cleaned dataset:

```bash
python -m ml.data_cleaning --input data/raw/resume_data_for_ranking.csv
python -m ml.train --data data/processed/cleaned_resume_data.csv
```

This creates:

```text
models/resume_match_ridge.joblib
models/evaluation.json
```

The binary model is intentionally not committed to GitHub. This keeps the repository small and makes the training process reproducible.

## Running the Application

### 1. Clone the repository

```bash
git clone https://github.com/Vishal10052006/smart-resume-screening-and-candidate-ranking-tool.git
cd smart-resume-screening-and-candidate-ranking-tool
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Download the dataset, clean it, and train

```bash
mkdir -p data/raw
# Place resume_data_for_ranking.csv in data/raw/
python -m ml.data_cleaning --input data/raw/resume_data_for_ranking.csv
python -m ml.eda --data data/processed/cleaned_resume_data.csv
python -m ml.train --data data/processed/cleaned_resume_data.csv
```

### 5. Start the API

```bash
uvicorn backend.app.main:app --reload
```

Open `http://127.0.0.1:8000` for the interface or `http://127.0.0.1:8000/docs` for the API documentation.

## API

### `GET /health`

Returns application and model availability information.

### `POST /api/analyze`

Accepts:

- `job_description` — job description text
- `resumes` — one or more PDF, DOCX, or TXT files

Returns ranked candidates with the predicted match score and supporting diagnostics.

## Repository Structure

```text
smart-resume-screening-and-candidate-ranking-tool/
|
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── ml_predictor.py
│   ├── __init__.py
│   ├── .env.example
│   └── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── README.md
│   ├── processed/
│   ├── reports/
│   └── README.md
│
├── ml/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── preprocessing.py
│   └── train.py
│
├── models/
│   ├── README.md
│   └── evaluation.json
│
├── .gitignore
├── LICENSE
└── README.md
```

## Limitations

The current version is an academic/project MVP. It does not yet include:

- OCR for scanned/image-only resumes
- Transformer embeddings
- Advanced resume section classification
- Persistent candidate database
- Authentication and role-based access
- Comprehensive bias and fairness evaluation
- Ranking-specific validation such as NDCG@K
- Production monitoring

These are planned improvements rather than claims about the current implementation.

## Data Privacy and Responsible Use

Resumes may contain personally identifiable information. Candidate documents should not be committed to this repository. The API processes uploaded files in memory and does not intentionally persist them.

A predicted match score is a screening signal. It should not be used as the sole basis for employment decisions.

## Future Work

1. Compare the baseline model with tree-based regressors and transformer embeddings.
2. Add ranking metrics such as NDCG@K and Precision@K.
3. Improve skill/entity extraction with a maintained taxonomy or NER model.
4. Add OCR for scanned resumes.
5. Add database storage, authentication, and recruiter-specific workflows.
6. Add automated tests and CI.
7. Validate against a separately collected, unseen resume/job dataset.

## Author

**Vishal Raj**  
Computer Science & Engineering (AI)

GitHub: https://github.com/Vishal10052006

## License

See the `LICENSE` file for license information.
