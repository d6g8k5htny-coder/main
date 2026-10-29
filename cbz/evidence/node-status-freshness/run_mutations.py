#!/usr/bin/env python3
"""Bounded, non-destructive assertion-only mutation experiment; no math certification."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
CASES=[
 ('object-drift', 'before == after', 'True', 'test_changed_blob_is_stale'),
 ('mode-drift', 'before == after', 'before[2] == after[2]', 'test_mode_change_is_stale'),
 ('truncation', 'value.get("truncated") is not False', 'False', 'test_partial_tree_unknown'),
 ('tree-identity', 'value.get("sha") != sha', 'False', 'test_wrong_response_identity_unknown'),
 ('comment-direction', 'if current_z < recorded:', 'if False:', 'test_future_recorded_comment_unknown'),
 ('schema-preflight', 'if problems:', 'if False:', 'test_strict_invalid_records_never_repaired_to_green'),
 ('strict-exit', 'return 1 if args.check and (drift or not report or any(row["status"] != "fresh" for row in report)) else 0', 'return 0', 'test_strict_real_refresh_end_to_end'),
 ('input-drift', 'drift = before != snapshot()', 'drift = False', 'test_strict_rejects_input_drift_during_observation'),
 ('directory-drift', 'before == after', 'True', 'test_directory_changed_even_unchanged_child'),
]

def execute(root, method, optimized):
    cmd=[sys.executable,'-B','-S']+(['-O'] if optimized else [])+['-m','unittest','-v','tests.test_node_status_freshness.ContentTests.'+method]
    return subprocess.run(cmd,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)

def main():
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        dest=Path(tmp)
        for directory in ('tools','tests','records/node_status','architecture','docs/site'):
            (dest/directory).mkdir(parents=True,exist_ok=True)
        for directory in ('tools','tests'):
            (dest/directory/'__init__.py').write_text('')
        paths=['tools/node_status_refresh.py','tools/validate_node_status.py','tests/test_node_status_freshness.py',
               'architecture/node-status.schema.json','docs/site/current.json']
        paths += [str(p.relative_to(ROOT)) for p in (ROOT/'records/node_status').glob('*.json')]
        for path in paths: shutil.copyfile(ROOT/path,dest/path)
        module=dest/'tools/node_status_refresh.py'; original=module.read_text()
        for optimized in (False,True):
            for name,old,new,method in CASES:
                if original.count(old)!=1: raise RuntimeError('nonunique mutation: '+name)
                module.write_text(original)
                baseline=execute(dest,method,optimized)
                if baseline.returncode != 0 or 'Ran 1 test' not in baseline.stdout or 'skipped' in baseline.stdout: raise RuntimeError('baseline failed or vacuous: '+baseline.stdout)
                module.write_text(original.replace(old,new))
                result=execute(dest,method,optimized)
                killed=(result.returncode==1 and 'AssertionError' in result.stdout
                        and 'FAILED (failures=' in result.stdout and 'errors=' not in result.stdout
                        and method in result.stdout)
                rows.append(dict(name=name,optimized=optimized,killed=killed,output=result.stdout))
                if not killed: raise RuntimeError('not an intended assertion kill: '+result.stdout)
        module.write_text(original)
    print(json.dumps({'cases':len(rows),'results':rows},indent=2))
    return 0
if __name__=='__main__': sys.exit(main())
