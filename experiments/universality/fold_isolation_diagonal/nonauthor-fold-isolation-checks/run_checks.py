import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
import time

HERE=pathlib.Path(__file__).resolve().parent
SOURCE=HERE.parent
EXPECTED={
 'QUANTITATIVE_FOLD_ISOLATION_PROPOSAL.md':(19138,'827c82953845eb12005542ff6a4e1c99886b68d1ea9f1aeae16f981622ac6a30'),
 'fold_isolation_controls.py':(6308,'13fae24ace9b3e8afb26d115f1a03acaec9ea72b0953a74ac54f2f1fd1fd46f5'),
 'FOLD_ISOLATION_CONTROLS.json':(795,'ccaf141441c3f890970c8e168c478a1f98f6eec8791aba002ffacfef06333bcf')}

def inventory():
    rows=[]
    for name,(size,digest) in EXPECTED.items():
        b=(SOURCE/name).read_bytes()
        actual=hashlib.sha256(b).hexdigest()
        assert len(b)==size and actual==digest,name
        rows.append({'path':str(SOURCE/name),'bytes':len(b),'sha256':actual})
    return rows

before=inventory()
runs=[]
for label,path in [('author-controls',SOURCE/'fold_isolation_controls.py'),('independent-controls',HERE/'review_checks.py')]:
    argv=[sys.executable,'-B',str(path)]
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    tick=time.perf_counter()
    result=subprocess.run(argv,cwd=SOURCE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    elapsed=time.perf_counter()-tick
    (HERE/(label+'.stdout')).write_bytes(result.stdout)
    (HERE/(label+'.stderr')).write_bytes(result.stderr)
    record={'command_argv':argv,'cwd':str(SOURCE),'started_at_utc':start,
            'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'wall_seconds':elapsed,'exit_code':result.returncode,
            'stdout':label+'.stdout','stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),
            'stderr':label+'.stderr','stderr_sha256':hashlib.sha256(result.stderr).hexdigest()}
    (HERE/(label+'.meta.json')).write_text(json.dumps(record,indent=2)+'\n')
    assert result.returncode==0,(label,result.stderr.decode())
    payload=json.loads(result.stdout)
    if label=='author-controls':
        assert result.stdout==(SOURCE/'FOLD_ISOLATION_CONTROLS.json').read_bytes()
        record['exact_frozen_output_match']=True
    else:
        assert payload['all_checks']=='PASS'
        record['finite_cases']=payload['total_cases']
    runs.append(record)
after=inventory()
assert after==before
summary={'state':'LOCAL_REPLAY_AND_INDEPENDENT_FINITE_CONTROLS_VERIFIED_NO_PROOF_ACCEPTANCE',
         'performer':'OpenAI/Codex /root/c261a_local_runtime_20261016',
         'organization_independence_credit':0,'exposure':'frozen proposal and source read; nonauthor of reviewed bytes',
         'interpreter_version':sys.version,'frozen_inputs_before':before,'frozen_inputs_after':after,
         'runs':runs,'frozen_inputs_unchanged':True}
(HERE/'CHECK_RUN_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
