from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util,json,os,hashlib
ROOT=Path('/workspace/universal-law/website-repair-20261028');OUT=Path('/workspace/cbz0/review-tools')
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
s=module(ROOT/'tools/cbz_storage.py','storage');t=module(ROOT/'tests/test_cbz_refs.py','ref_tests');t.RefSafetyTests.setUpClass();report={}
with TemporaryDirectory(prefix='cbz-review-storage-fixed-',dir=OUT) as tmp:
 roots=[Path(tmp)/n for n in ('a','b')]
 for p in roots:p.mkdir();(p/'payload').write_bytes(b'exact immutable bytes');(p/'payload').chmod(0o644)
 os.setxattr(roots[1]/'payload',b'user.review-note',b'target-only metadata')
 try:s.make_plan([str(p) for p in roots]);raise AssertionError('metadata ambiguity accepted')
 except s.Refusal as err:report['storage_refusal']=str(err)
 report['xattr_preserved']=os.getxattr(roots[1]/'payload',b'user.review-note')==b'target-only metadata';report['files_remain_distinct']=not os.path.samefile(*(p/'payload' for p in roots))
case=t.RefSafetyTests('test_archive_has_unique_parents_exact_manifest_and_recoverable_content');case.setUp()
try:
 blob=t.git(case.remote,'hash-object','-w','--stdin',input='unique tag-only blob\n');t.git(case.remote,'update-ref','refs/tags/tag-only-blob',blob);case.snapshot();archive=case.out/'repository.git'
 report['intact_snapshot_accepted']=case.tool.verify_snapshot(case.out)['verified'];t.git(archive,'update-ref','-d','refs/tags/tag-only-blob')
 try:case.tool.verify_snapshot(case.out);raise AssertionError('missing tag accepted')
 except case.tool.SafetyError as err:report['refs_refusal']=str(err)
 report['object_still_exists_at_refusal']=t.git(archive,'cat-file','-t',blob)=='blob'
finally:case.doCleanups()
report['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['tools/cbz_refs.py','tests/test_cbz_refs.py','tools/cbz_storage.py','tests/test_cbz_storage.py']}
(OUT/'safety-successor-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
