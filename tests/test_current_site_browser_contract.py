"""Browser evidence records survive source-admission failures before launch."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
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
