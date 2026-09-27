import hashlib, importlib.util, json, pathlib, tempfile, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=ROOT/'tools/formal_execution_context.py'
s=importlib.util.spec_from_file_location('ctx',p);m=importlib.util.module_from_spec(s)
if p.exists():s.loader.exec_module(m)
class ContextTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=pathlib.Path(self.tmp.name)
  self.sha='a'*40; (self.root/'build.log').write_bytes(b'actual log\n')
  self.receipt={'repository':'d6g8k5htny-coder/main','checked_commit':self.sha,'workflow_run_id':'123', 'formalization_status':'kernel-checked','scientific_effect':'NONE','logs':{'build.log':hashlib.sha256(b'actual log\n').hexdigest()}}
  self.write()
  self.kw={'source_repository':'d6g8k5htny-coder/Math-','expected_commit':self.sha,'observed_commit':self.sha,'observed_remote':'https://github.com/d6g8k5htny-coder/Math-','execution_repository':'d6g8k5htny-coder/main','execution_commit':'b'*40,'run_id':'123'}
 def write(self): (self.root/'receipt.json').write_text(json.dumps(self.receipt))
 def bind(self):return m.bind(self.root/'receipt.json',**self.kw)
 def test_distinct_source_and_execution_binding(self):
  self.assertTrue(hasattr(m,'bind'),'cross-repository context binder is missing')
  r=self.bind();self.assertEqual(r['source']['repository'],self.kw['source_repository']);self.assertEqual(r['execution']['repository'],self.kw['execution_repository'])
  self.assertNotEqual(r['source']['repository'],r['execution']['repository']);self.assertEqual(r['receipt_repository_field_means'],'execution repository, not source repository')
  self.assertEqual(r['receipt_sha256'],hashlib.sha256((self.root/'receipt.json').read_bytes()).hexdigest())
 def test_source_commit_mismatch(self):
  self.kw['observed_commit']='c'*40
  with self.assertRaises(ValueError): self.bind()
 def test_receipt_commit_mismatch(self):
  self.receipt['checked_commit']='c'*40;self.write()
  with self.assertRaises(ValueError): self.bind()
 def test_source_remote_mismatch(self):
  self.kw['observed_remote']='https://github.com/other/Math-'
  with self.assertRaises(ValueError):self.bind()
 def test_receipt_host_mismatch(self):
  self.receipt['repository']='d6g8k5htny-coder/Math-';self.write()
  with self.assertRaises(ValueError):self.bind()
 def test_run_mismatch(self):
  self.kw['run_id']='124'
  with self.assertRaises(ValueError):self.bind()
 def test_modified_log(self):
  (self.root/'build.log').write_text('modified')
  with self.assertRaises(ValueError):self.bind()
 def test_log_path_escape(self):
  self.receipt['logs']={'../build.log':'f'*64};self.write()
  with self.assertRaises(ValueError):self.bind()
 def test_kernel_status_is_required(self):
  self.receipt['formalization_status']='specified';self.write()
  with self.assertRaises(ValueError):self.bind()
 def test_git_suffix(self):
  self.kw['observed_remote']+=' .git'
  with self.assertRaises(ValueError):self.bind()
  self.kw['observed_remote']='https://github.com/d6g8k5htny-coder/Math-.git'
  self.assertEqual(self.bind()['source']['commit'],self.sha)
if __name__=='__main__': unittest.main()
