# Model Artifacts

The training pipeline writes the trained model to:

```text
models/resume_match_ridge.joblib
```

The binary model is intentionally not committed to the repository. Generate it locally with:

```bash
python -m ml.train --data data/raw/resume_data_for_ranking.csv
```

`evaluation.json` contains the latest hold-out evaluation recorded during development.
