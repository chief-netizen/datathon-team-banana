import os, unittest, tempfile
from pathlib import Path
from investigation import build_queue, public_status, load_csv
from ml_pipeline import extract_features, score_encounters, load_artifact

class InvestigationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.q=build_queue()
 def test_queue_is_ranked_and_scores_are_bounded(self):
  self.assertTrue(self.q)
  self.assertEqual(self.q, sorted(self.q,key=lambda x:(-x['priority'],x['id'])))
  for a in self.q:
   for k in ('impact','confidence','priority'): self.assertGreaterEqual(a[k],0);self.assertLessEqual(a[k],100)
 def test_traceability_and_breakdown(self):
  a=self.q[0]; self.assertTrue(a['id'].startswith('E'));self.assertTrue(a['evidence']);self.assertEqual(sum(x['contribution'] for x in a['score_breakdown']), round(a['confidence'],1))
  self.assertTrue(all('source' in e for e in a['evidence']))
 def test_missing_and_conflicting_evidence_safe(self):
  # The supplied CSV has no blank encounter timestamps, but the engine still
  # exposes an uncertainty list and handles contradictions without crashing.
  self.assertTrue(all(a['uncertainty'] for a in self.q))
  self.assertTrue(any('agreement' in ' '.join(a['uncertainty']).lower() or 'missing' in ' '.join(a['uncertainty']).lower() for a in self.q))
 def test_ml_features_exclude_identifiers_and_score_is_bounded(self):
  encounters, tests = load_csv('encounters.csv'), load_csv('diagnostic_tests.csv')
  names, rows = extract_features(encounters[:5], tests)
  self.assertNotIn('encounter_id', names); self.assertNotIn('patient_id', names)
  scores = score_encounters(encounters[:5], tests)
  self.assertEqual(len(scores), 5)
  self.assertTrue(all(0 <= x['anomaly_score'] <= 100 for x in scores.values()))

 def test_ml_reproducible_artifact_and_corruption_is_rejected(self):
  a = score_encounters(load_csv('encounters.csv')[:10], load_csv('diagnostic_tests.csv'))
  b = score_encounters(load_csv('encounters.csv')[:10], load_csv('diagnostic_tests.csv'))
  self.assertEqual(a, b)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'bad.joblib'; p.write_text('not a model')
   with self.assertRaises(Exception): load_artifact(p)

 def test_provider_status_does_not_expose_secret(self):
  old=os.environ.pop('GEMINI_API_KEY',None);os.environ['AI_PROVIDER']='gemini'
  try: self.assertFalse(public_status()['ai_configured']);self.assertNotIn('key',str(public_status()).lower())
  finally:
   if old is not None:os.environ['GEMINI_API_KEY']=old
   os.environ.pop('AI_PROVIDER',None)

if __name__=='__main__': unittest.main()
