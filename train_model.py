"""Explicitly train/retrain the local anomaly artifact."""
import json
from investigation import load_csv
from ml_pipeline import train_model, ARTIFACT

if __name__ == "__main__":
    metadata=train_model(load_csv("encounters.csv"), load_csv("diagnostic_tests.csv"))
    print(json.dumps({"artifact":str(ARTIFACT), **metadata}, indent=2))
