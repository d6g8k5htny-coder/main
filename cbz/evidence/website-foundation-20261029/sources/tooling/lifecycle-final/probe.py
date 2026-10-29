import ast, hashlib, importlib.util, io, json, re, shutil, sys, tempfile
from pathlib import Path
ROOT=Path('/workspace/universal-law/website-repair-20261028')
OUT=Path('/workspace/cbz0/release/mutation-review')
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def names(p,cls):
 tree=ast.parse(p.read_text());c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
 return [n.name for n in c.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
def fake_test(cls,method_names,phase,audit=False):
 src='import unittest\nfrom pathlib import Path\nBODY_ENTRIES=[]\n'
 if audit: src+='import os\nTOOL=Path(os.environ["CBZ_AUDIT_UNDER_TEST"])\n'
 else: src+='TOOL=None\n'
 src+='class '+cls+'(unittest.TestCase):\n'
 src+='    def mutant(self):\n        return Path(TOOL).name != '+repr('cbz_audit.py' if audit else 'ORIGINAL')+'\n'
 if not audit: src=src.replace('Path(TOOL).name != \'ORIGINAL\'','Path(TOOL).name == \'mutant.py\'')
 if phase=='setup': src+='    def setUp(self):\n        if self.mutant(): self.fail("inert setup assertion")\n'
 if phase=='teardown': src+='    def tearDown(self):\n        if self.mutant(): self.fail("inert teardown assertion")\n'
 if phase=='cleanup': src+='    def setUp(self):\n        if self.mutant(): self.addCleanup(self.fail,"inert cleanup assertion")\n'
 for n in method_names:
  src+='    def '+n+'(self):\n        BODY_ENTRIES.append(self._testMethodName)\n'
  if phase=='body': src+='        if self.mutant(): self.fail("inert body assertion")\n'
 if audit: src+='if __name__=="__main__": unittest.main()\n'
 return src
rows=[]
for kind,cls in [('ref','RefSafetyTests'),('baseline','BaselineTests')]:
 runner=load(ROOT/'cbz/evidence/website-foundation-20261029/verification'/f'{kind}_controls.py','runner_'+kind)
 ns=names(ROOT/'tests'/runner.TEST_NAME,cls)
 for phase in ('setup','teardown','cleanup','body'):
  with tempfile.TemporaryDirectory(prefix='cbz-inert-lifecycle-') as d:
   d=Path(d);(d/'tools').mkdir();(d/'tests').mkdir();(d/'output').mkdir()
   shutil.copyfile(ROOT/'tools'/runner.TOOL_NAME,d/'tools'/runner.TOOL_NAME)
   (d/'tests'/runner.TEST_NAME).write_text(fake_test(cls,ns,phase))
   captured=[];original=runner.load
   def capture(p,n):
    m=original(p,n);captured.append(m);return m
   runner.load=capture
   try:
    positive=runner.positive(d,d/'output',runner.TEST_NAME,cls)
    mutants=runner.mutations(d,d/'output')
   finally: runner.load=original
   rows.append(dict(runner=kind,phase=phase,positive=positive,mutations=len(mutants),accepted=sum(x['detected'] for x in mutants),body_entries=[len(m.BODY_ENTRIES) for m in captured[1:]],fixture_scope='benign replacement test bodies only; not actual original tests'))
audit=load(ROOT/'tools/cbz_audit.py','auditor_review')
ns=names(ROOT/'tests/test_cbz_audit.py','AuditContract')
for phase in ('setup','teardown','cleanup','body'):
 with tempfile.TemporaryDirectory(prefix='cbz-inert-audit-') as d:
  p=Path(d)/'inert_tests.py';p.write_text(fake_test('AuditContract',ns,phase,True));r=audit.selftest(p)
  rows.append(dict(runner='audit',phase=phase,positive_exit=r['baseline']['exit_code'],mutations=r['total'],accepted=r['killed'],result=r['result'],fixture_scope='benign replacement test bodies only; not original43 assertions'))
(OUT/('probe-optimized.json' if sys.flags.optimize else 'probe-normal.json')).write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
