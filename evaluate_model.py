"""Evaluate the unsupervised model without pretending labels exist."""
import json, statistics, time
from investigation import load_csv, build_queue
from ml_pipeline import score_encounters, model_summary

if __name__ == "__main__":
    encounters, tests = load_csv("encounters.csv"), load_csv("diagnostic_tests.csv")
    start=time.perf_counter(); scores=score_encounters(encounters, tests); elapsed=time.perf_counter()-start
    values=[x["anomaly_score"] for x in scores.values()]
    flagged=[x for x in scores.values() if x["flagged"]]
    queue=build_queue()
    ml_top={x["id"] for x in queue[:max(1,round(len(queue)*.1))]}
    # Existing rule ranking comparator, calculated from unchanged impact/confidence fields.
    old=sorted(queue,key=lambda x:(-(.55*x["impact"]+.35*x["confidence"]+.10*(100-x["confidence"] if x["impact"]>=60 else 0)),x["id"]))
    old_top={x["id"] for x in old[:max(1,round(len(queue)*.1))]}
    report={"task":"unsupervised operational anomaly detection","records_used_for_training":model_summary()["records"],"records_scored":len(scores),"features":model_summary()["feature_names"],"preprocessing":"DictVectorizer for categorical fields; median imputation; StandardScaler; IDs excluded","model":{"type":"IsolationForest","n_estimators":250,"contamination":0.05,"random_seed":42},"validation":"No ground-truth labels. Reference-fit score distribution and stability/rule overlap are reported; predictive validity is not established.","anomaly_score":{"min":round(min(values),2),"median":round(statistics.median(values),2),"p95":round(sorted(values)[int(.95*len(values))],2),"max":round(max(values),2),"flagged_count":len(flagged),"flagged_proportion":round(len(flagged)/len(values),4)},"inference_seconds":round(elapsed,3),"comparison":{"queue_records":len(queue),"top_10_percent_overlap_with_previous_rule_ranking":round(len(ml_top&old_top)/max(1,len(ml_top)),3)},"flagged_examples":[{"encounter_id":k,"anomaly_score":v["anomaly_score"],"features":v["model_features"]} for k,v in list(sorted(scores.items(),key=lambda kv:-kv[1]["anomaly_score"]))[:10]],"limitations":["Synthetic data has no verified forgotten-patient label.","Anomaly means unusual relative to this dataset, not clinically important.","The 95th percentile flag is a review threshold, not a validated operating point."]}
    with open("reports/ml_evaluation.json","w") as f: json.dump(report,f,indent=2)
    with open("reports/ml_evaluation.md","w") as f:
        f.write("# ML anomaly evaluation\n\n")
        f.write("This is an unsupervised review-signal evaluation; no accuracy, precision, recall, or clinical validity is claimed.\n\n")
        for k,v in report["anomaly_score"].items(): f.write(f"- **{k}:** {v}\n")
        f.write(f"\nTraining records: {report['records_used_for_training']}  \\ Scored: {report['records_scored']}  \\ Inference seconds: {report['inference_seconds']}\n")
        f.write(f"\nTop-10% overlap with prior rule ranking: {report['comparison']['top_10_percent_overlap_with_previous_rule_ranking']}\n\n## Top flagged examples\n")
        for x in report["flagged_examples"]: f.write(f"- `{x['encounter_id']}` anomaly `{x['anomaly_score']}`; wait `{x['features']['wait_minutes']}`, triage ordinal `{x['features']['triage_level']}`, test delay `{x['features']['max_test_delay_minutes']}`\n")
    print(json.dumps(report,indent=2))
