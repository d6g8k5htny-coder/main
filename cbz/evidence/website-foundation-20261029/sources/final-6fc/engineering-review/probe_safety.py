from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import importlib.util,json,os,subprocess
ROOT=Path('/workspace/universal-law/website-repair-20261028');OUT=Path('/workspace/cbz0/review-tools')
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
s=module(ROOT/'tools/cbz_storage.py','storage');t=module(ROOT/'tests/test_cbz_refs.py','ref_tests');t.RefSafetyTests.setUpClass()
report={}
with TemporaryDirectory(prefix='cbz-review-storage-',dir=OUT) as tmp:
 roots=[Path(tmp)/n for n in ('a','b')]
 for p in roots:p.mkdir();(p/'payload').write_bytes(b'exact immutable bytes');(p/'payload').chmod(0o644)
 os.setxattr(roots[1]/'payload',b'user.review-note',b'target-only metadata')
 plan=s.make_plan([str(p) for p in roots]);before=[os.listxattr(p/'payload') for p in roots]
 result=s.apply_plan(plan);after=[os.listxattr(p/'payload') for p in roots]
 report['xattr_loss']={'planned_operations':plan['operations'],'before':before,'after':after,'tool_report':result,'bytes_preserved':[(p/'payload').read_bytes().decode() for p in roots]}
case=t.RefSafetyTests('test_archive_has_unique_parents_exact_manifest_and_recoverable_content');case.setUp()
try:
 blob=t.git(case.remote,'hash-object','-w','--stdin',input='unique tag-only blob\n')
 t.git(case.remote,'update-ref','refs/tags/tag-only-blob',blob)
 case.snapshot();archive=case.out/'repository.git'
 t.git(archive,'update-ref','-d','refs/tags/tag-only-blob')
 verified=case.tool.verify_snapshot(case.out)
 exists_before=t.git(archive,'cat-file','-t',blob)
 t.git(archive,'reflog','expire','--expire=now','--all');t.git(archive,'gc','--prune=now')
 proc=subprocess.run(['git','-C',str(archive),'cat-file','-e',blob],capture_output=True)
 report['unretained_tag_pass']={'verified_after_local_tag_deletion':verified['verified'],'tag_only_object':blob,'object_type_before_gc':exists_before,'exists_after_gc':proc.returncode==0,'original_fixture_remote_tag_intact':t.git(case.remote,'rev-parse','refs/tags/tag-only-blob')==blob}
finally:case.doCleanups()
(OUT/'safety-probe-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
