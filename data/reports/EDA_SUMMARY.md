# Exploratory Data Analysis Report

## Dataset
- Records: **9,544**
- Columns: **35**
- Distinct job positions: **28**
- Duplicate rows after cleaning: **0**

## Data quality
- Missing cells in raw dataset: **101,704**
- Missing cells after cleaning: **0**
- Invalid target rows removed: **0**

Missing values were concentrated in optional resume fields. Text fields are normalized to empty strings when absent; numeric values are not blindly replaced with zero.

## Target: `matched_score`
- Mean: **0.6608**
- Median: **0.6833**
- Standard deviation: **0.1670**
- Range: **0.00–0.97**
- Unique values: **229**

### Score bands
| Score range | Records | Share |
|---|---:|---:|
| 0.00–0.20 | 48 | 0.5% |
| 0.20–0.40 | 1,110 | 11.6% |
| 0.40–0.60 | 1,512 | 15.8% |
| 0.60–0.80 | 4,593 | 48.1% |
| 0.80–1.00 | 2,281 | 23.9% |

## Text characteristics
Resume-side text averages **101.3 words** per record (median 91).

Job-side text averages **64.2 words** per record (median 59).

## Skills
Resume records contain an average of **21.6 listed skills**. The job-side `related_skils_in_job` field contains nested skill lists and averages **9.5 skills** per record (median 6) after flattening.

Top resume skills include Python, Machine Learning, SQL, Data Analysis, Deep Learning, Excel, Java, C++, and Natural Language Processing.

Top job-side skills include Machine Learning, Project Management, Sales, Troubleshooting, Data Analysis, Python, Customer Service, Marketing, and Accounting.

## Key findings
1. `matched_score` is a continuous target, so the supervised task is regression followed by candidate ranking.
2. Resume and job text are the primary NLP inputs and are suitable for TF-IDF features.
3. Skill counts and explicit resume/job skill overlap should be engineered as structured features.
4. `matched_score` must never enter the model feature matrix.
5. Job positions are almost evenly represented, so role-aware validation is recommended.
6. The near-uniform job-position distribution suggests deliberate balancing or construction. Model performance should therefore be checked on newly collected/external examples before production claims.

## Reproducibility
Run:

```bash
python -m ml.eda --data data/processed/cleaned_resume_data.csv
```

The command regenerates `eda_report.json`, `job_position_distribution.csv`, `match_score_distribution.csv`, `top_skills.csv`, and `top_job_skills.csv` in `data/reports/`.
