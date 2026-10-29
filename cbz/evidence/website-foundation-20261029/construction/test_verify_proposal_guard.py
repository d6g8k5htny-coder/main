from pathlib import Path
import contextlib,hashlib,importlib.util,io,json,tempfile,unittest
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'cbz/evidence/website-foundation-20261029'
p=OUT/'construction/verify_proposal.py';source=p.read_text();site="require(index['coordination']['all_main_obligations_complete'] is False, 'contribution identity invariant failed')"
if source.count(site)!=1:raise RuntimeError('ambiguous mutation site')
index=json.loads((OUT/'proof-index.json').read_text());index['coordination']['all_main_obligations_complete']=True
rows=[]
with tempfile.TemporaryDirectory(prefix='metadata-guard-control-') as temp:
 for label,text in [('unmutated',source),('missing_all_main_guard',source.replace(site,"require(True, 'contribution identity invariant failed')"))]:
  f=Path(temp)/label/'a/b/c/d/verifier.py';f.parent.mkdir(parents=True);f.write_text(text)
  spec=importlib.util.spec_from_file_location(label,f);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=ROOT;m.OUT=OUT
  class Guard(unittest.TestCase):
   def runTest(self):
    with self.assertRaises(AssertionError):m.validate(index,False)
  log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.TestSuite([Guard()]))
  rows.append({'label':label,'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'log':log.getvalue()})
if rows[0]['failures'] or rows[0]['errors'] or rows[0]['tests']!=1:raise RuntimeError('unmutated positive harness failed')
if rows[1]['failures']!=1 or rows[1]['errors'] or rows[1]['tests']!=1:raise RuntimeError('mutant was not killed by exactly one assertion failure')
print(json.dumps({'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'one finite source-guard mutation, not complete validator fault model or theorem certification','rows':rows},indent=2))
