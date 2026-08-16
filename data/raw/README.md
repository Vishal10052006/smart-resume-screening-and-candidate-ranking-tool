# Raw Dataset

The project uses the **Resume Data for Ranking** dataset from Kaggle.

Source: https://www.kaggle.com/datasets/thejohnwick001/resume-data-for-ranking

The raw CSV is intentionally not committed to this repository. Download it from Kaggle and place it here as:

```text
data/raw/resume_data_for_ranking.csv
```

Then train the model with:

```bash
python -m ml.train --data data/raw/resume_data_for_ranking.csv
```

Do not commit real candidate resumes or personally identifiable information to the repository.
