"""Reproducible unsupervised anomaly pipeline for encounter operations.

There are no verified labels for a "forgotten" encounter in this synthetic data,
so IsolationForest identifies unusual operational patterns only. It does not
estimate clinical risk or the probability that an alert is correct.
"""
from __future__ import annotations
import json, time
from datetime import datetime
from pathlib import Path
from typing import Iterable
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
ARTIFACT = ROOT / "models" / "encounter_isolation_forest.joblib"
SEED = 42
FEATURE_VERSION = "encounter-ops-v1"


def _num(value, default=0.0):
    try: return float(value)
    except (TypeError, ValueError): return default


def _minutes(a, b):
    try: return max(0.0, (datetime.strptime(b, "%Y-%m-%d %H:%M:%S") - datetime.strptime(a, "%Y-%m-%d %H:%M:%S")).total_seconds()/60)
    except (TypeError, ValueError): return 0.0


def extract_features(encounters: list[dict], tests: list[dict]) -> tuple[list[str], list[dict]]:
    """Extract only operational fields; IDs are retained outside feature vectors."""
    by_enc = {}
    for t in tests: by_enc.setdefault(t.get("encounter_id"), []).append(t)
    rows=[]
    triage={"Low":0.0,"Medium":1.0,"High":2.0,"Critical":3.0}
    for e in encounters:
        linked=by_enc.get(e.get("encounter_id"),[])
        delays=[_minutes(t.get("requested_time"),t.get("completion_time")) for t in linked]
        missing=sum(not e.get(k) for k in ("arrival_time","service_start_time","service_end_time","discharge_time"))
        start_wait=_minutes(e.get("arrival_time"),e.get("service_start_time"))
        reported=_num(e.get("wait_minutes"))
        row={
            "wait_minutes": reported, "service_minutes": _num(e.get("service_minutes")),
            "arrival_to_start_minutes": start_wait, "wait_difference_minutes": abs(reported-start_wait),
            "test_count": float(len(linked)), "urgent_test_count": float(sum(t.get("priority")=="Urgent" for t in linked)),
            "max_test_delay_minutes": max(delays, default=0.0), "missing_timestamp_count": float(missing),
            "timeline_inconsistent": float((start_wait < 0) or (e.get("service_start_time") and e.get("service_end_time") and e["service_end_time"] < e["service_start_time"])),
            "triage_level": triage.get(e.get("triage_level"), -1.0),
            "department": e.get("department_id", "unknown"), "visit_type": e.get("visit_type", "unknown"),
        }
        rows.append(row)
    names=list(rows[0].keys()) if rows else []
    return names, rows


def train_model(encounters: list[dict], tests: list[dict], artifact: Path = ARTIFACT) -> dict:
    names, rows=extract_features(encounters, tests)
    if len(rows)<10: raise ValueError("at least 10 encounters are required")
    vectorizer=DictVectorizer(sparse=False)
    matrix=vectorizer.fit_transform(rows)
    prep=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    X=prep.fit_transform(matrix)
    model=IsolationForest(n_estimators=250, contamination=0.05, random_state=SEED, n_jobs=-1)
    started=time.perf_counter(); model.fit(X); fit_seconds=time.perf_counter()-started
    raw=-model.decision_function(X)
    lo,hi=float(min(raw)),float(max(raw))
    # Fixed training-reference min/max makes inference deterministic and bounded.
    metadata={"feature_version":FEATURE_VERSION,"feature_names":names,"vector_features":vectorizer.get_feature_names_out().tolist(),"records":len(rows),"seed":SEED,"contamination":0.05,"raw_min":lo,"raw_max":hi,"fit_seconds":round(fit_seconds,3),"threshold_95":float(sorted(raw)[int(.95*len(raw))])}
    artifact.parent.mkdir(exist_ok=True)
    joblib.dump({"vectorizer":vectorizer,"preprocessor":prep,"model":model,"metadata":metadata},artifact)
    return metadata


def load_artifact(artifact: Path = ARTIFACT):
    if not artifact.exists(): raise FileNotFoundError(f"ML artifact missing: {artifact}. Run train_model.py")
    bundle=joblib.load(artifact)
    if not all(k in bundle for k in ("vectorizer","preprocessor","model","metadata")): raise ValueError("corrupt ML artifact")
    if bundle["metadata"].get("feature_version") != FEATURE_VERSION: raise ValueError("unsupported feature version")
    return bundle


def score_encounters(encounters: list[dict], tests: list[dict], artifact: Path = ARTIFACT) -> dict[str, dict]:
    bundle=load_artifact(artifact); _, rows=extract_features(encounters,tests)
    X=bundle["preprocessor"].transform(bundle["vectorizer"].transform(rows))
    raw=-bundle["model"].decision_function(X); meta=bundle["metadata"]; lo,hi=meta["raw_min"],meta["raw_max"]
    scores={}
    for e,r,v in zip(encounters,rows,raw):
        anomaly=round(max(0,min(100,100*(float(v)-lo)/(hi-lo or 1))),1)
        scores[e.get("encounter_id")]={"anomaly_score":anomaly,"flagged":float(v)>=meta["threshold_95"],"model_features":r}
    return scores


def model_summary(artifact: Path = ARTIFACT):
    return load_artifact(artifact)["metadata"]
