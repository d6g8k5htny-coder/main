"""Browser evidence records survive source-admission failures before launch."""
import json
import copy
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
ROOT=Path(__file__).resolve().parents[1]
class BrowserReportFailureTests(unittest.TestCase):
 def test_missing_and_malformed_source_write_failure_report(self):
  for raw in [None,b'{broken']:
   with self.subTest(raw=raw),tempfile.TemporaryDirectory() as tmp:
    repo=Path(tmp);(repo/'tools').mkdir();(repo/'docs/site').mkdir(parents=True)
    script=repo/'tools/current_site_browser_check.py';shutil.copyfile(ROOT/'tools/current_site_browser_check.py',script)
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','core.hooksPath=/dev/null','commit','--allow-empty','-qm','Fixture'],check=True)
    if raw is not None:(repo/'docs/site/current.json').write_bytes(raw)
    output=repo/'evidence';result=subprocess.run([sys.executable,str(script),'--output',str(output)],capture_output=True,text=True)
    self.assertNotEqual(result.returncode,0);self.assertTrue((output/'report.json').is_file(),result.stderr)
    report=json.loads((output/'report.json').read_text());self.assertTrue(report['error']);self.assertEqual(report['cases'],[]);self.assertEqual(report['scientific_effect'],'NONE')

class ResumeContractTests(unittest.TestCase):
 def setUp(self):
  spec=importlib.util.spec_from_file_location('current_site_browser_contract_target',ROOT/'tools/current_site_browser_check.py')
  self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
  self.node='math.lifetime-remainder';pin='a'*40
  review='https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206'
  self.data={'observed_at':'2026-10-13T12:00:00Z','repositories':[{'repository':'d6g8k5htny-coder/main','commit':pin}],
             'graph':{'identity':{'repository':'d6g8k5htny-coder/Math-','commit':pin,'path':'frontiers/x/GRAPH.json','sha256':'b'*64},
                      'data':{'nodes':{self.node:{'classification':'AUTHOR_SIDE_CANDIDATE','controlling':False,'review_source':review}}}}}
  record=json.loads((ROOT/'records/node_status/math.lifetime-remainder.json').read_bytes())
  self.records={self.node:record};self.now=0
  shared=[['classification','AUTHOR_SIDE_CANDIDATE','controlling','false'],['review_source',review]]
  links=[review,'https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6351719961',
         'https://github.com/d6g8k5htny-coder/main/issues/388#issuecomment-6352139396']
  unrecorded=['Next action','not recorded in this snapshot','Missing input','not recorded in this snapshot',
              'Owner evidence','not recorded in this snapshot','Retained stop','not recorded in this snapshot']
  self.static={'rows':copy.deepcopy(shared)+[unrecorded],'links':list(links)}
  note=f'Node-status view {record["status"]} as of {record["observed_at"]}. Recorded status {record["status"]}. Scientific effect NONE. Freshness is not scientific validity.'
  operational=['Next action',record['next_action']+' '+note,'Missing input','; '.join(record['missing_input']),
               'Owner evidence','No owner recorded in this node-status view.','Retained stop','; '.join(record['stops'])]
  self.interactive={'rows':copy.deepcopy(shared)+[operational],'links':list(links)}

 def test_published_overlay_and_snapshot_only_static_reading_have_separate_exact_expectations(self):
  self.assertNotEqual(self.interactive,self.static)
  self.module.assert_resume_readings(self.interactive,self.static,self.data,self.node,self.records,self.now)

 def test_wrong_shared_fields_phantom_links_and_false_operational_state_are_refused(self):
  mutations=[('shared classification',lambda a,b:(a['rows'][0].__setitem__(1,'PROVED_REVIEWED'),b['rows'][0].__setitem__(1,'PROVED_REVIEWED'))),
             ('phantom link',lambda a,b:(a['links'].append('https://github.com/d6g8k5htny-coder/main/issues/1#issuecomment-1'),b['links'].append('https://github.com/d6g8k5htny-coder/main/issues/1#issuecomment-1'))),
             ('missing continuation',lambda a,b:(a['links'].pop(),b['links'].pop())),
             ('false owner',lambda a,b:a['rows'][-1].__setitem__(5,'self-appointed owner')),
             ('unrecorded overlay',lambda a,b:a['rows'].__setitem__(-1,copy.deepcopy(b['rows'][-1]))),
             ('static operational promotion',lambda a,b:b['rows'].__setitem__(-1,copy.deepcopy(a['rows'][-1])))]
  for label,mutate in mutations:
   with self.subTest(label=label):
    live,static=copy.deepcopy(self.interactive),copy.deepcopy(self.static);mutate(live,static)
    with self.assertRaises(ValueError):self.module.assert_resume_readings(live,static,self.data,self.node,self.records,self.now)
