import importlib.util,io,json,os,pathlib,struct,unittest,zipfile,gzip,tempfile,subprocess,warnings
P=pathlib.Path(os.environ.get('CBZ_COVERAGE_UNDER_TEST',pathlib.Path(__file__).with_name('coverage_companion.py')))
s=importlib.util.spec_from_file_location('companion',P);c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
def archive(rows):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w') as z:
  for name,data in rows:
   with warnings.catch_warnings():
    warnings.simplefilter('ignore',UserWarning);z.writestr(name,data)
 return b.getvalue()
def npy(descr='<f8',shape=(2,),payload=b'\0'*16):
 h=(repr({'descr':descr,'fortran_order':False,'shape':shape})+'\n').encode()
 return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(h))+h+payload
class Contract(unittest.TestCase):
 def test_member_bytes_and_lean_hits_preserved(self):
  r=c.inspect_zip(archive([('a.lean',b'example := by sorry'),('b.txt',b'ordinary')]))
  self.assertEqual(len(r['members']),2);self.assertEqual(r['members'][0]['sha256'],c.sha(b'example := by sorry'))
  self.assertEqual(len(r['members'][0]['inspection']['lean_findings']),1);self.assertEqual(r['members'][0]['inspection']['lean_findings'][0]['code'],'lean.sorry')
 def test_lean_comments_are_not_hits(self):
  r=c.inspect_payload('a.lean',b'/- sorry /- axiom -/ -/ def x := "native_decide"')
  self.assertEqual(r['lean_findings'],[])
 def test_duplicate_unsafe_nested_and_symlink_members_refused(self):
  for rows in ([('../a',b'x')],[('a',b'x'),('a',b'y')],[('nested.zip',archive([('a',b'x')]))]):
   self.assertTrue(c.inspect_zip(archive(rows))['residuals'])
  b=io.BytesIO()
  with zipfile.ZipFile(b,'w') as z:
   i=zipfile.ZipInfo('link');i.create_system=3;i.external_attr=0o120777<<16;z.writestr(i,b'target')
  self.assertTrue(c.inspect_zip(b.getvalue())['residuals'])
 def test_budget_and_corruption_are_gaps(self):
  self.assertTrue(c.inspect_zip(archive([('a',b'12345')]),limit=4)['residuals'])
  self.assertTrue(c.inspect_zip(b'PKbroken')['residuals'])
 def test_npy_metadata_and_payload_exact(self):
  r=c.inspect_npy(npy());self.assertEqual(r['dtype'],'<f8');self.assertEqual(r['shape'],[2]);self.assertEqual(r['payload_bytes'],16)
 def test_npy_object_and_wrong_length_refused(self):
  for raw in (npy('|O8'),npy(payload=b'\0')):
   with self.assertRaises(ValueError):c.inspect_npy(raw)
 def test_python_float_observation_and_exact_negative(self):
  self.assertTrue(c.inspect_payload('a.py',b'x=0.1')['float_observations'])
  self.assertEqual(c.inspect_payload('a.py',b'from fractions import Fraction\nx=Fraction(1,10)')['float_observations'],[])
 def test_notebook_cells_outputs_and_image_bound(self):
  raw=json.dumps({'nbformat':4,'metadata':{},'cells':[{'cell_type':'code','source':['x=0.5\n'],'outputs':[{'output_type':'display_data','data':{'image/png':'YWJj'}}],'metadata':{}}]}).encode()
  r=c.inspect_notebook(raw);self.assertEqual(len(r['cells']),1);self.assertEqual(len(r['cells'][0]['outputs']),1);self.assertEqual(r['cells'][0]['outputs'][0]['media'][0]['decoded_sha256'],c.sha(b'abc'))
  self.assertTrue(r['cells'][0]['inspection']['float_observations'])
 def test_malformed_notebook_and_duplicate_json_refused(self):
  for raw in (b'{"cells":1}',b'{"nbformat":4,"nbformat":4,"cells":[]}'):
   with self.assertRaises(ValueError):c.inspect_notebook(raw)
 def test_gzip_bounded_and_hash_bound(self):
  raw=gzip.compress(b'{"x":1}');r=c.inspect_gzip(raw);self.assertEqual(r['members'][0]['sha256'],c.sha(b'{"x":1}'))
  self.assertTrue(c.inspect_gzip(raw,limit=3)['residuals'])
 def test_html_active_content_visible(self):
  self.assertEqual(c.inspect_payload('a.html',b'<p>plain</p>')['active_html'],[])
  self.assertTrue(c.inspect_payload('a.html',b'<script>x()</script><b onclick="x()">x</b>')['active_html'])
 def test_inventory_hash_change_and_empty_inputs_remain_residual(self):
  with tempfile.TemporaryDirectory() as folder:
   root=pathlib.Path(folder);path=next(iter(c.POLICIES));target=root/path;target.parent.mkdir(parents=True);target.write_bytes(archive([('a.txt',b'ok')]))
   g1=root/'g1.json';g1.write_text(json.dumps({'coverage_gaps':[{'path':path,'sha256':'0'*64}]}))
   r=c.inventory(root,g1);self.assertIn('G1 exact byte binding changed',r['containers'][0].get('error',''));self.assertEqual(r['coverage_result'],'INCOMPLETE')
   g1.write_text('{"coverage_gaps":[]}');self.assertEqual(c.inventory(root,g1)['coverage_result'],'INCOMPLETE')
 def test_preservation_detects_changed_and_missing_test_bytes(self):
  with tempfile.TemporaryDirectory() as folder:
   root=pathlib.Path(folder);subprocess.run(['git','init','-q',str(root)],check=True)
   for path in ('tests/test_one.py','formal/A.lean'):
    target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('pass\n')
   subprocess.run(['git','-C',str(root),'add','.'],check=True)
   subprocess.run(['git','-C',str(root),'-c','user.name=test','-c','user.email=test@example.invalid','commit','-qm','base'],check=True)
   self.assertTrue(c.preservation(root,'HEAD')['all_preserved'])
   (root/'tests/test_one.py').write_text('pass # changed\n');self.assertFalse(c.preservation(root,'HEAD')['all_preserved'])
   (root/'tests/test_one.py').unlink();self.assertFalse(c.preservation(root,'HEAD')['all_preserved'])
 def test_control_log_hash_binding_and_tamper(self):
  receipt={'negative_controls':{'sorry':'REJECTED_BY_AXIOM_GATE'},'logs':{'sorry.log':c.sha(b'sorryAx')},'alignment_status':'PENDING_INDEPENDENT_REVIEW'}
  raw=archive([('receipt/receipt.json',json.dumps(receipt).encode()),('receipt/sorry.lean',b'example := by sorry'),('receipt/sorry.log',b'sorryAx')])
  self.assertEqual(c.control_bindings(raw)['controls'][0]['source_sha256'],c.sha(b'example := by sorry'))
  bad=archive([('receipt/receipt.json',json.dumps(receipt).encode()),('receipt/sorry.lean',b'example := by sorry'),('receipt/sorry.log',b'changed')])
  with self.assertRaises(ValueError):c.control_bindings(bad)
 def test_total_budget_is_enforced_when_each_member_fits(self):
  original=c.MAX_TOTAL
  try:
   c.MAX_TOTAL=4
   self.assertTrue(c.inspect_zip(archive([('a',b'123'),('b',b'456')]),limit=10)['residuals'])
  finally:c.MAX_TOTAL=original
if __name__=='__main__':unittest.main()
