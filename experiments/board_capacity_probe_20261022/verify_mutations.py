"""Real isolated weakening controls. These do not execute repository software."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
source=(ROOT/'board_capacity_probe.py').read_text()
tests=(ROOT/'test_board_capacity_probe.py').read_bytes()

def replace_once(old,new):
    if source.count(old)!=1:
        raise ValueError('mutation anchor not unique')
    return source.replace(old,new,1)

start=source.index("    if (str(issue['number'])")
end=source.index("    try:\n        observed =",start)
identity_off=source[:start]+source[end:]
mutations={
 'late_prepare':replace_once('count >= prepare_at','count > prepare_at'),
 'late_cutover':replace_once('count >= cutover_at','count > cutover_at'),
 'skip_identity':identity_off,
 'accept_boolean_count':replace_once("type(issue.get('comments')) is not int", "not isinstance(issue.get('comments'), int)"),
 'ignore_staleness':replace_once('if age > max_age_seconds:', 'if False:'),
 'ignore_future':replace_once('if age < 0:', 'if False:'),
 'ignore_lock':replace_once("if issue['locked']:", 'if False:'),
 'admit_pull_request':replace_once("if 'pull_request' in issue:", 'if False:'),
 'admit_duplicate_keys':replace_once('if key in value:', 'if False:'),
 'fabricate_permission':source.replace("'posting_permission': 'NOT_ESTABLISHED'", "'posting_permission': 'ALLOWED'"),
 'quiet_cli_success':replace_once('raise SystemExit(main())','raise SystemExit(0)'),
}
out=ROOT/'evidence'/'mutations'
out.mkdir(exist_ok=False)
results=[]
for name,text in mutations.items():
    directory=out/name;directory.mkdir()
    (directory/'board_capacity_probe.py').write_text(text)
    (directory/'test_board_capacity_probe.py').write_bytes(tests)
    for mode in ('normal','optimized'):
        cmd=[sys.executable,'-B','-S']+(['-O'] if mode=='optimized' else [])+[str(directory/'test_board_capacity_probe.py')]
        before=time.monotonic()
        p=subprocess.run(cmd,env={'PATH':os.defpath,'LC_ALL':'C.UTF-8'},capture_output=True,timeout=10)
        elapsed=time.monotonic()-before
        (directory/(mode+'.stdout')).write_bytes(p.stdout)
        (directory/(mode+'.stderr')).write_bytes(p.stderr)
        log=p.stderr.decode()
        fail=re.search(r'FAILED \(failures=(\d+)\)\s*$',log)
        detected=(p.returncode==1 and fail is not None and 'Ran 21 tests' in log and 'ERROR:' not in log)
        results.append({'name':name,'mode':mode,'exit':p.returncode,
                        'assertion_failures':int(fail.group(1)) if fail else None,
                        'detected_as_assertions':detected,'elapsed_seconds':elapsed,
                        'source_sha256':hashlib.sha256(text.encode()).hexdigest(),
                        'stdout_sha256':hashlib.sha256(p.stdout).hexdigest(),
                        'stderr_sha256':hashlib.sha256(p.stderr).hexdigest()})
report={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'tests_sha256':hashlib.sha256(tests).hexdigest(), 'cases':results,
        'passed':all(row['detected_as_assertions'] for row in results)}
(out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'runs':len(results),
                 'normal_failures':{r['name']:r['assertion_failures'] for r in results if r['mode']=='normal'}},indent=2))
raise SystemExit(0 if report['passed'] else 1)
