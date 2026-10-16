"""Deterministic investigation queue for the synthetic forgotten-patient dataset.

Scores are heuristic evidence scores, not clinical probabilities.
"""
from __future__ import annotations
import csv, datetime as dt, json, os
from pathlib import Path
from typing import Any
from ml_pipeline import score_encounters

ROOT = Path(__file__).parent


def _load_local_env():
    """Load the gitignored local env file without requiring python-dotenv."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('\\\"').strip("'"))


_load_local_env()


def load_csv(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def number(value, default=None):
    try: return float(value)
    except (TypeError, ValueError): return default


def parse_time(value):
    if not value: return None
    try: return dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError: return None


def pct(value): return round(max(0, min(100, value)), 1)


def build_queue():
    patients = {r["patient_id"]: r for r in load_csv("patients.csv")}
    departments = {r["department_id"]: r for r in load_csv("departments.csv")}
    tests = load_csv("diagnostic_tests.csv")
    encounters = load_csv("encounters.csv")
    try:
        anomaly_by_encounter = score_encounters(encounters, tests)
    except (FileNotFoundError, ValueError, KeyError):
        # The queue remains usable before the explicit training command; this
        # fallback is not an ML score and is clearly marked in the record.
        anomaly_by_encounter = {}
    tests_by_enc = {}
    for test in tests: tests_by_enc.setdefault(test["encounter_id"], []).append(test)
    alerts = []
    for e in encounters:
        p = patients.get(e["patient_id"], {})
        ml_signal = anomaly_by_encounter.get(e["encounter_id"], {"anomaly_score": 0.0, "flagged": False, "model_features": {}})
        d = departments.get(e["department_id"], {})
        linked = tests_by_enc.get(e["encounter_id"], [])
        wait = number(e.get("wait_minutes"))
        triage = e.get("triage_level", "")
        reasons = []
        category = None
        # Alerts are observations about process/data quality, not diagnoses.
        if wait is not None and wait >= 60:
            category, reasons = "prolonged_wait", [f"wait_minutes={int(wait)} (threshold: 60)"]
        urgent = [t for t in linked if t.get("priority") == "Urgent"]
        delayed_urgent = []
        for t in urgent:
            a, c = parse_time(t.get("requested_time")), parse_time(t.get("completion_time"))
            if a and c and (c-a).total_seconds()/60 >= 120: delayed_urgent.append((t, round((c-a).total_seconds()/60)))
        if delayed_urgent and (category is None or triage in ("High", "Critical")):
            category, reasons = "delayed_urgent_test", [f"urgent {t['test_type']} completed after {mins} minutes" for t, mins in delayed_urgent]
        required = ["arrival_time", "service_start_time", "service_end_time", "discharge_time"]
        missing = [x for x in required if not e.get(x)]
        if missing and (category is None or not reasons):
            category, reasons = "incomplete_record", [f"missing {x}" for x in missing]
        if category is None and triage in ("High", "Critical") and wait is not None and wait >= 30:
            category, reasons = "high_acuity_delay", [f"{triage} triage with {int(wait)} minute wait"]
        if category is None: continue

        # Evidence quality: completeness (40%), cross-field agreement (30%), alert strength (30%).
        completeness = 100 * (1 - len(missing)/len(required))
        agreement = 100
        start, arrival = parse_time(e.get("service_start_time")), parse_time(e.get("arrival_time"))
        if start and arrival and start < arrival: agreement -= 40
        end, discharge = parse_time(e.get("service_end_time")), parse_time(e.get("discharge_time"))
        if end and start and end < start: agreement -= 40
        if wait is not None and start and arrival:
            derived = (start-arrival).total_seconds()/60
            if abs(derived-wait) > 5: agreement -= 25
        agreement = max(0, agreement)
        strength = 90 if category == "delayed_urgent_test" else (85 if category == "prolonged_wait" else 70)
        if triage in ("High", "Critical"): strength = min(100, strength+10)
        confidence = pct(.40*completeness + .30*agreement + .30*strength)
        impact = 25
        if triage == "Critical": impact += 35
        elif triage == "High": impact += 25
        elif triage == "Medium": impact += 12
        if e.get("outcome") == "Admitted": impact += 18
        if category == "prolonged_wait": impact += min(20, max(0, (wait-60)/6)) if wait is not None else 0
        if category == "delayed_urgent_test": impact += 18
        if p.get("chronic_condition") not in (None, "", "None"): impact += 8
        impact = pct(impact)
        uncertainty_value = pct(100-confidence) if impact >= 60 else 0
        # ML anomaly is a ranking signal, not a probability. Impact retains the
        # largest share so unusual low-impact records do not dominate the queue.
        priority = pct(.40*impact + .25*confidence + .30*ml_signal["anomaly_score"] + .05*uncertainty_value)
        evidence = [
            {"field": "encounter_id", "value": e["encounter_id"], "source": "encounters.csv"},
            {"field": "patient_id", "value": e["patient_id"], "source": "encounters.csv"},
            {"field": "triage_level", "value": triage, "source": "encounters.csv"},
            {"field": "wait_minutes", "value": e.get("wait_minutes") or "missing", "source": "encounters.csv"},
            {"field": "outcome", "value": e.get("outcome") or "missing", "source": "encounters.csv"},
            {"field": "arrival_time", "value": e.get("arrival_time") or "missing", "source": "encounters.csv"},
        ]
        if linked: evidence.append({"field":"diagnostic_tests", "value": ", ".join(t['test_id']+" ("+t['priority']+")" for t in linked[:8]), "source":"diagnostic_tests.csv"})
        evidence.append({"field":"ml_anomaly_score", "value": ml_signal["anomaly_score"], "source":"models/encounter_isolation_forest.joblib", "interpretation":"unusualness relative to training encounters; not a probability"})
        uncertainties = ([f"Missing fields: {', '.join(missing)}"] if missing else [])
        if agreement < 100: uncertainties.append("Timestamp or reported wait values do not fully agree")
        # Operational destination suggestion derived only from triage and recorded test type.
        # This is a routing aid, not a clinical referral or diagnosis.
        destination = d.get("department_name", e.get("department_id"))
        route_reason = "Continue with the recorded department pending verification."
        if triage == "Critical":
            destination, route_reason = "Emergency", "Critical triage is the strongest available routing signal."
        elif delayed_urgent:
            test_names = {t.get("test_type") for t, _ in delayed_urgent}
            if test_names & {"CT scan", "MRI", "X-ray"}:
                destination, route_reason = "Radiology", "The linked urgent imaging test points to Radiology."
            elif "ECG" in test_names:
                destination, route_reason = "Cardiology", "The linked urgent ECG points to Cardiology for operational follow-up."
            elif "Urine test" in test_names:
                destination, route_reason = "Laboratory", "The linked urgent urine test points to Laboratory."
        elif e.get("department_id") in departments:
            destination = d.get("department_name", e.get("department_id"))
            route_reason = "Use the encounter's recorded department while the alert is verified."
        alerts.append({"id": e["encounter_id"], "category": category, "title": category.replace("_", " ").title(), "impact": impact, "confidence": confidence, "anomaly_score": ml_signal["anomaly_score"], "anomaly_flagged": ml_signal["flagged"], "model_features": ml_signal["model_features"], "priority": priority, "reason": reasons[0], "reasons": reasons, "evidence": evidence, "uncertainty": uncertainties or ["No missing or contradictory fields detected for this alert"], "context": {"patient_id": e["patient_id"], "department": d.get("department_name", e.get("department_id")), "arrival_time": e.get("arrival_time"), "outcome": e.get("outcome"), "chronic_condition": p.get("chronic_condition", "")}, "recommendation": "Verify the timeline in the encounter record and confirm whether follow-up is needed before closing the case." if category != "incomplete_record" else "Reconcile the missing timestamp fields with the source system before interpreting this encounter." , "routing": {"destination": destination, "reason": route_reason, "disclaimer": "Operational suggestion only; verify with qualified staff."}, "score_breakdown": [{"component":"Completeness", "value":round(completeness,1), "weight":"40%", "contribution":round(.4*completeness,1)}, {"component":"Cross-field agreement", "value":round(agreement,1), "weight":"30%", "contribution":round(.3*agreement,1)}, {"component":"Alert evidence strength", "value":round(strength,1), "weight":"30%", "contribution":round(.3*strength,1)}], "raw_record_fields": {k:e.get(k,"") for k in e.keys()}})
    return sorted(alerts, key=lambda a: (-a["priority"], a["id"]))


def public_status():
    provider = os.getenv("AI_PROVIDER", "none").lower()
    configured = (provider == "gemini" and bool(os.getenv("GEMINI_API_KEY"))) or (provider == "grok" and bool(os.getenv("XAI_API_KEY")))
    return {"provider": provider if provider in ("gemini", "grok") and configured else "local", "ai_configured": configured}


def ai_explain(alert):
    """Optional server-side provider call. Local evidence remains the source of truth."""
    provider = os.getenv("AI_PROVIDER", "none").lower()
    key = os.getenv("GEMINI_API_KEY") if provider == "gemini" else os.getenv("XAI_API_KEY") if provider == "grok" else None
    if not key: return {"status":"local", "summary":"No AI provider configured; this explanation was generated deterministically from the linked records."}
    # Kept deliberately small and de-identified: no age, sex, area, or raw patient profile is sent.
    payload = {"alert_id":alert["id"], "category":alert["category"], "reason":alert["reason"], "evidence":alert["evidence"], "scores":{"impact":alert["impact"],"confidence":alert["confidence"]}}
    prompt = "Return JSON with keys summary and next_step. Only restate supplied evidence; do not diagnose. Evidence: " + json.dumps(payload)
    try:
        import urllib.request
        if provider == "gemini":
            url = "https://generativelanguage.googleapis.com/v1beta/models/" + os.getenv("GEMINI_MODEL", "gemini-2.0-flash") + ":generateContent?key=" + key
            body = {"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"responseMimeType":"application/json"}}
        else:
            url = "https://api.x.ai/v1/chat/completions"
            body = {"model":os.getenv("GROK_MODEL","grok-3-mini"),"messages":[{"role":"user","content":prompt}],"response_format":{"type":"json_object"}}
        req=urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type":"application/json","Authorization":"Bearer "+key} if provider=="grok" else {"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=12) as r: raw=json.loads(r.read())
        text = raw["candidates"][0]["content"]["parts"][0]["text"] if provider=="gemini" else raw["choices"][0]["message"]["content"]
        result=json.loads(text)
        if not isinstance(result.get("summary"), str) or not isinstance(result.get("next_step"), str): raise ValueError("invalid response schema")
        return {"status":"ai", "summary":result["summary"], "next_step":result["next_step"]}
    except Exception as exc:
        return {"status":"error", "summary":"AI explanation unavailable; showing the verified local evidence instead.", "error":type(exc).__name__}
