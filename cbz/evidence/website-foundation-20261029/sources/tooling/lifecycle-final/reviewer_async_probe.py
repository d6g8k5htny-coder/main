import importlib.util,unittest,io,json,warnings,inspect,hashlib,sys
from pathlib import Path
p=Path('/workspace/universal-law/website-repair-20261028/tools/cbz_mutation.py')
s=importlib.util.spec_from_file_location('runner',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rows=[]
for kind in ('async','generator','returned_coroutine'):
 events=[]
 async def deferred():
  events.append('deferred_body');raise AssertionError('must execute or refuse')
 if kind=='async':
  class Fixture(unittest.IsolatedAsyncioTestCase):
   async def test_guard(self):
    events.append('body');self.fail('must execute or refuse')
 elif kind=='generator':
  class Fixture(unittest.TestCase):
   def test_guard(self):
    events.append('body');self.fail('must execute or refuse');yield
 else:
  class Fixture(unittest.TestCase):
   def test_guard(self): return deferred()
 with warnings.catch_warnings(record=True) as warnings_seen:
  warnings.simplefilter('always');case=Fixture('test_guard')
  report=m.run_suite(unittest.TestSuite([case]),io.StringIO(),[case.id()])
 rows.append(dict(kind=kind,baseline_success=report['baseline_success'],body_kill=report['body_kill'],events=events,warnings=len(warnings_seen),invalid=report['invalid']))
print(json.dumps(dict(helper_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),optimization=sys.flags.optimize,rows=rows)))
