# Demo & Viva Checklist

Use this checklist before presenting the Smart Resume Screening and Candidate Ranking Tool.

## 1. Environment

```bash
source .venv/bin/activate
python --version
```

Install dependencies if required:

```bash
python -m pip install -r backend/requirements.txt
```

## 2. Reproduce the complete ML pipeline

Place the licensed Kaggle CSV at:

```text
data/raw/resume_data_for_ranking.csv
```

Then run:

```bash
bash scripts/run_pipeline.sh
```

This performs data cleaning, EDA, model evaluation, final model fitting, and automated tests.

## 3. Start the application

```bash
uvicorn backend.app.main:app --reload
```

Open:

- `http://127.0.0.1:8000` — recruiter interface
- `http://127.0.0.1:8000/docs` — Swagger/OpenAPI documentation
- `http://127.0.0.1:8000/health` — service health

## 4. Demo sequence

1. Paste a realistic job description.
2. Upload 2–5 resumes in PDF, DOCX, or TXT format.
3. Click **Screen Resumes**.
4. Show the ranked candidates.
5. Explain the score, semantic similarity, skill coverage, matched skills, and skill gaps.
6. Show `/health` and explain whether the trained model is loaded.
7. Open `/docs` and demonstrate the API contract.
8. Show `models/evaluation.json` and explain the random and job-position-grouped evaluation.

## 5. Viva points

### Why TF-IDF?

It provides a strong, interpretable lexical baseline for resume/job matching and works well with a linear model on a medium-sized text dataset.

### Why Ridge Regression?

The target is a continuous `matched_score` in `[0, 1]`, so regression is appropriate. Ridge also handles high-dimensional sparse TF-IDF features with regularization.

### Why engineered skill tokens?

Explicit skill-overlap signals make the model more sensitive to concrete requirements instead of relying only on word-frequency similarity.

### Why NDCG@10?

The product is a ranking system, so regression error alone does not fully measure usefulness. NDCG@10 evaluates whether stronger candidates are placed near the top of each job-position group.

### Why grouped evaluation?

A random split can allow the same job-position distribution to appear in both train and test data. A job-position-grouped hold-out is a stricter robustness check.

### Why not make an automatic hiring decision?

The system is a screening-assistance tool. Human review remains necessary because resume ranking can encode dataset limitations and bias.

## 6. Important limitation to state honestly

The repository does not contain the raw dataset or generated model binary. The final model and current evaluation metrics must be regenerated locally from the documented dataset using `scripts/run_pipeline.sh`.
