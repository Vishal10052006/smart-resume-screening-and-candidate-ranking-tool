# Model Artifacts

The training pipeline creates:

```text
models/resume_match_ridge.joblib
models/evaluation.json
```

The binary model is intentionally excluded from Git because it is generated from the locally downloaded Kaggle dataset.

## Generate the current model

```bash
python -m ml.train --data data/processed/cleaned_resume_data.csv
```

The current training pipeline:

- builds TF-IDF unigram and bigram features;
- adds explicit skill-coverage and matched-skill tokens;
- evaluates a standard random hold-out;
- evaluates a stricter job-position-grouped hold-out;
- calculates MAE, RMSE, R², and NDCG@10;
- refits the final model on the complete cleaned dataset.

`evaluation.json` currently retains the earlier baseline metrics as a reference. It should be regenerated after the upgraded feature pipeline is trained.
