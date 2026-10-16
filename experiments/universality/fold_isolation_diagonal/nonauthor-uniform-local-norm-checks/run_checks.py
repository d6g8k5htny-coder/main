import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
import time

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parent
EXPECTED={
 'UNIFORM_LOCAL_NORM_DIAGONAL_PROPOSAL.md':(26687,'8e1eefca9888d43824a9fd9caca34c44d2de4bf8e30c668b474fa2097491c76b'),
 'uniform_local_norm_controls.py':(6486,'4b84755b771598ed34193c075718a41da447719f90a2ff5c6a17d48d0abc3dc4'),
 'UNIFORM_LOCAL_NORM_CONTROLS.json':(1790,'5d6cdb2d4925c6ba0b1b545d7a5aca1977655c1d18aa1ea29a82273f30845ece'),
 'QUANTITATIVE_FOLD_ISOLATION_PROPOSAL.md':(19138,'827c82953845eb12005542ff6a4e1c99886b68d1ea9f1aeae16f981622ac6a30'),
 'fold_isolation_controls.py':(6308,'13fae24ace9b3e8afb26d115f1a03acaec9ea72b0953a74ac54f2f1fd1fd46f5'),
 'FOLD_ISOLATION_CONTROLS.json':(795,'ccaf141441c3f890970c8e168c478a1f98f6eec8791aba002ffacfef06333bcf'),
 'FINITE_FOURIER_CRITICAL_COUNT_DIAGONAL_ROUTE.md':(25699,'621cee96854aa440498cecf5452a8d1e23f90a4458779c3286c18eba106fe138'),
 'finite_fourier_count_controls.py':(5153,'fd6f29739280a55017945526cb75fb46ebc80d86c012ff4a1ff65baa404c31de'),
 'FINITE_FOURIER_COUNT_CONTROLS.json':(849,'bbe147fdf409a3e37b450d4cbcd5d2a927140f95bd3ce0f36cafe16861123bfd'),
 'NONAUTHOR_FOLD_ISOLATION_REVIEW.md':(18409,'f06c61e3f3e60c795a1a14e824640f1689d4143dba4f6698cada7a41929eec90'),
 'NONAUTHOR_FINITE_COUNT_DIAGONAL_REVIEW.md':(21880,'16152427f7013822ac1bba81142429800334e6bf0f354a6439b23a1997bc4524'),
 'uniform-local-norm-primary-sources/SOURCE_IDENTITY.json':(443,'92bf5297b309e947f42ee1dc837487775e7341bc7c1051e493899963ff85891a'),
 'uniform-local-norm-primary-sources/simon-lecs-on-pde.pdf':(940975,'e1f1b2f51d2558aa467a8c064fe82e7b9e157ad28c9c75f01e38596cde656870'),
 'experiments/universality/finite_gaussian_contact/PROOF.md':(25367,'7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474')}

def inventory():
    rows=[]
    for name,(size,expected) in EXPECTED.items():
        b=(ROOT/name).read_bytes();digest=hashlib.sha256(b).hexdigest()
        assert len(b)==size and digest==expected,name
        rows.append({'path':name,'bytes':len(b),'sha256':digest})
    return rows

before=inventory();runs=[]
for label,script in [('author-controls',ROOT/'uniform_local_norm_controls.py'),('independent-controls',HERE/'review_checks.py')]:
    argv=[sys.executable,'-B',str(script)]
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.perf_counter()
    r=subprocess.run(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    elapsed=time.perf_counter()-tick
    (HERE/(label+'.stdout')).write_bytes(r.stdout);(HERE/(label+'.stderr')).write_bytes(r.stderr)
    record={'command_argv':argv,'cwd':str(ROOT),'started_at_utc':start,
            'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'wall_seconds':elapsed,'exit_code':r.returncode,'stdout':label+'.stdout','stderr':label+'.stderr',
            'stdout_sha256':hashlib.sha256(r.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(r.stderr).hexdigest()}
    (HERE/(label+'.meta.json')).write_text(json.dumps(record,indent=2)+'\n')
    assert r.returncode==0,(label,r.stderr.decode())
    output=json.loads(r.stdout)
    if label=='author-controls':
        assert r.stdout==(ROOT/'UNIFORM_LOCAL_NORM_CONTROLS.json').read_bytes()
        record['exact_frozen_output_match']=True
    else:
        assert output['all_checks']=='PASS'
        record['independent_finite_cases']=output['total_finite_cases']
    runs.append(record)
after=inventory();assert before==after
result={'state':'NEW_CONTROLS_REPLAY_AND_INDEPENDENT_FINITE_CHECKS_ONLY','performer':'OpenAI/Codex /root/c261a_local_runtime_20261016',
        'source_exposure':'new proposal and controls read fully; bounded Fourier/Plancherel import read; prior two exact lemmas consumed, not re-reviewed',
        'organizational_independence_credit':0,'interpreter_version':sys.version,'inputs_before':before,'inputs_after':after,
        'all_14_inputs_unchanged':True,'runs':runs,'previous_controls_rerun':False,
        'production_remote_mutations_or_formal_execution':0,'primary_source_read_only_browsing':True,
        'source_adoption_or_scientific_promotion':'NONE'}
(HERE/'CHECK_RUN_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
