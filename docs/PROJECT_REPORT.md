# Project Report

## 1. Problem Statement

Manual resume screening becomes slow and inconsistent when many candidates apply to the same role. The system provides a first-pass ranking based on resume/job relevance while keeping the recruiter in control of the final decision.

## 2. Data Collection

Development data comes from the Kaggle **Resume Data for Ranking** dataset. The repository records the source and reproducible download location but does not redistribute the raw CSV.

## 3. Data Cleaning

The cleaning stage normalizes column names and text, removes exact duplicates, validates `matched_score` in `[0, 1]`, and handles missing text without inventing numeric values.

The current cleaned experiment contains 9,544 records, 35 columns, no exact duplicates, and no invalid target values.

## 4. Exploratory Data Analysis

EDA covers target distribution, job-position balance, text length, resume skills, job skills, and basic correlations. The dataset has 28 job positions and a mean `matched_score` of 0.6608. Job positions are close to evenly represented, so role-aware validation is important.

## 5. Feature Engineering

The model representation combines:

- resume-side text;
- job-side text;
- TF-IDF unigrams and bigrams;
- explicit skill extraction;
- skill coverage buckets;
- matched-skill count tokens.

The target column is never included in the feature matrix.

## 6. Model Development

The primary model is Ridge Regression over the TF-IDF representation. It is intentionally used as an interpretable baseline before introducing heavier transformer-based approaches.

The training script evaluates both:

- random 80/20 hold-out;
- job-position-grouped hold-out.

The second split is designed to expose over-reliance on repeated role labels.

## 7. Evaluation

Regression metrics:

- MAE
- RMSE
- R²

Ranking metric:

- NDCG@10 within job-position groups

The repository retains the previous baseline metrics for comparison. Current upgraded metrics must be regenerated with `python -m ml.train` after the local cleaned dataset is available.

## 8. Application and Testing

The FastAPI service supports PDF, DOCX, and TXT resumes, validates uploads, extracts resume text, performs ranking, and exposes `/health` and `/api/analyze`. The interface also shows matched skills and skill gaps.

Automated tests cover feature extraction, skill coverage, inference representation, and API health/validation behavior. GitHub Actions runs the test suite on pushes and pull requests.

## 9. Deployment

A Dockerfile is included for container deployment. The application listens on port 8000 and can run without a trained model because a transparent TF-IDF similarity fallback is available for demonstration.

## 10. Responsible Use

The output is a screening aid, not an employment decision. Candidate documents may contain personally identifiable information and should not be committed to the repository. Production use requires privacy controls, fairness evaluation, security review, and validation on representative real-world data.
