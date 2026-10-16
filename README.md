# Health — The Forgotten Patient: ML investigation system

This is a dependency-light hospital operations dashboard built on a synthetic DATATHON dataset: 2,000 patients, 8,000 encounters, 5,000 diagnostic tests, and 12 departments. The supplied README explicitly says this is not real patient data and must not be used for clinical decisions. Original CSV files are unchanged.

## Architecture

```text
CSV records -> deterministic alert rules -> feature extraction (no IDs)
-> saved IsolationForest + preprocessing -> anomaly score (unusualness)
-> evidence confidence + impact + priority -> dashboard queue and room assignment
-> optional server-side Gemini/Grok summary (never scores or source data)
```

## Install, train, evaluate, and run

```bash
cd 01_health_forgotten_patient
python3 -m pip install -r requirements.txt
python3 train_model.py
python3 evaluate_model.py
python3 -m unittest -v
python3 app.py                         # http://localhost:8000
```

The application loads the saved artifact at startup and does not retrain on HTTP requests. If it is missing, the deterministic queue remains available but ML scores fall back to zero until `train_model.py` is run. Generated artifacts and reports are separate from source CSVs.

## ML method and features

There are no verified labels for whether an encounter was genuinely forgotten or whether an operational alert was harmful. The encounter `outcome` field is a recorded disposition, not a trustworthy target for this investigation task. The system therefore uses unsupervised `sklearn.ensemble.IsolationForest` with 250 trees, contamination `0.05`, and random seed `42`.

Features are derived from encounter and linked test records: `wait_minutes`, `service_minutes`, arrival-to-service-start duration, reported/derived wait difference, test count, urgent-test count, maximum test delay, missing timestamp count, timeline inconsistency, ordinal triage, department, and visit type. Patient IDs, encounter IDs, and test IDs are excluded. Categorical fields use `DictVectorizer`; numeric values use median imputation and standardization. The artifact records its feature version, feature names, training count, threshold, and configuration.

The raw IsolationForest decision function is converted to a 0–100 reference-relative anomaly score using the training range. A score is **not** a probability, clinical risk, or claim that a patient was forgotten. The 95th-percentile training score is a review flag. The measured run trained/scored 8,000 records; 400 (5%) were flagged, median score was 24.4, p95 was 58.7, and inference took about 0.3 seconds locally. See `reports/ml_evaluation.md`.

## Ranking and limitations

Rule-generated alerts remain the queue population: prolonged wait (>=60 minutes), delayed urgent test (>=120 minutes), incomplete timeline, and high-acuity wait (>=30 minutes). Evidence confidence remains deterministic: 40% completeness, 30% timestamp/report agreement, and 30% alert evidence strength. Impact remains separate and uses triage, recorded disposition, alert type, wait duration, and chronic-condition field.

Priority is now: `40% impact + 25% evidence confidence + 30% ML anomaly score + 5% high-impact uncertainty value`. This is a transparent review ranking, not a calibrated prediction. The measured top-10% overlap with the former rule-only ranking was 57.3%, demonstrating that ML changes ordering. No accuracy, precision, recall, calibration, causal, or clinical claims are made because no verified labels exist. Expert review and future temporal validation are needed.

## Dashboard

The dashboard has a waiting queue, room/capability/capacity map, priority groups, selected-alert evidence and model feature values, explicit assignment confirmation, active assignments, and subtle route animation. Assignments are process-local in-memory state and update occupancy; source records are not changed. Room capabilities are documented operational metadata in `app.py`, because the dataset contains capacity and service type but not alert-handling capabilities.

## Optional AI explanation provider

AI is only called after the user clicks **Generate AI explanation**. The server sends a small structured, de-identified evidence payload; it does not send the full dataset or patient profile. Model scores and source fields cannot be overwritten. Responses must contain `summary` and `next_step`; failures fall back to local evidence.

```bash
export AI_PROVIDER=gemini
export GEMINI_API_KEY='...'
export GEMINI_MODEL=gemini-2.0-flash
# alternatively: AI_PROVIDER=grok and XAI_API_KEY / GROK_MODEL
python3 app.py
```

Keys remain server-side. Provider availability and authorization must be checked separately. Never send identifiable health information externally without authorization.

## Files

- `ml_pipeline.py` — feature extraction, training, artifact loading, inference.
- `train_model.py` — explicit reproducible training command.
- `evaluate_model.py` — unsupervised evaluation and rule-ranking comparison.
- `investigation.py` — deterministic alerts, evidence/impact/confidence, ML-integrated priority.
- `app.py` — HTTP API and dashboard.
- `models/encounter_isolation_forest.joblib` — generated fitted artifact.
- `reports/ml_evaluation.md` and `.json` — measured evaluation output.
