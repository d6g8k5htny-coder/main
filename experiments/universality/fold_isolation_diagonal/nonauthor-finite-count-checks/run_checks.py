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
 'FINITE_FOURIER_CRITICAL_COUNT_DIAGONAL_ROUTE.md':(25699,'621cee96854aa440498cecf5452a8d1e23f90a4458779c3286c18eba106fe138'),
 'finite_fourier_count_controls.py':(5153,'fd6f29739280a55017945526cb75fb46ebc80d86c012ff4a1ff65baa404c31de'),
 'FINITE_FOURIER_COUNT_CONTROLS.json':(849,'bbe147fdf409a3e37b450d4cbcd5d2a927140f95bd3ce0f36cafe16861123bfd'),
 'QUANTITATIVE_FOLD_ISOLATION_PROPOSAL.md':(19138,'827c82953845eb12005542ff6a4e1c99886b68d1ea9f1aeae16f981622ac6a30'),
 'fold_isolation_controls.py':(6308,'13fae24ace9b3e8afb26d115f1a03acaec9ea72b0953a74ac54f2f1fd1fd46f5'),
 'FOLD_ISOLATION_CONTROLS.json':(795,'ccaf141441c3f890970c8e168c478a1f98f6eec8791aba002ffacfef06333bcf'),
 'critical-count-primary-sources/SOURCE_IDENTITIES.json':(796,'2c42d92a4732f09a58aebd9b9ef8672b51b672ab99f3bbdad61681a0dcf4f504'),
 'critical-count-primary-sources/guth-lect13.pdf':(83539,'92ed0fc0f940179871a25f3a244eb181aaa0176ee5738ac1f8f68a8bacd072a9'),
 'critical-count-primary-sources/stacks-01WC.html':(15419,'08a39f68a8159e34e436df9c4b8f5e8886f58b3c0d5163c15610e12aef302737'),
 'critical-count-primary-sources/stacks-01W6.html':(14844,'060b8c179f815d59977f8a31057b5f5f9e1f2104351d46c33f1fb6d9ed95a202'),
 'experiments/universality/finite_gaussian_contact/PROOF.md':(25367,'7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474'),
 'NONAUTHOR_FOLD_ISOLATION_REVIEW.md':(18409,'f06c61e3f3e60c795a1a14e824640f1689d4143dba4f6698cada7a41929eec90')}

def inventory():
    result=[]
    for n,(size,expected) in EXPECTED.items():
        b=(ROOT/n).read_bytes();digest=hashlib.sha256(b).hexdigest()
        assert len(b)==size and digest==expected,n
        result.append({'path':n,'bytes':len(b),'sha256':digest})
    return result

before=inventory();runs=[]
for label,script in [('author-controls',ROOT/'finite_fourier_count_controls.py'),('independent-controls',HERE/'review_checks.py')]:
    argv=[sys.executable,'-B',str(script)]
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.perf_counter()
    r=subprocess.run(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    elapsed=time.perf_counter()-tick
    (HERE/(label+'.stdout')).write_bytes(r.stdout);(HERE/(label+'.stderr')).write_bytes(r.stderr)
    data={'command_argv':argv,'cwd':str(ROOT),'started_at_utc':start,
          'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'wall_seconds':elapsed,'exit_code':r.returncode,'stdout':label+'.stdout','stderr':label+'.stderr',
          'stdout_sha256':hashlib.sha256(r.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(r.stderr).hexdigest()}
    (HERE/(label+'.meta.json')).write_text(json.dumps(data,indent=2)+'\n')
    assert r.returncode==0,(label,r.stderr.decode())
    output=json.loads(r.stdout)
    if label=='author-controls':
        assert r.stdout==(ROOT/'FINITE_FOURIER_COUNT_CONTROLS.json').read_bytes()
        data['exact_frozen_output_match']=True
    else:
        assert output['all_checks']=='PASS';data['independent_finite_cases']=output['total_cases']
    runs.append(data)
after=inventory();assert after==before
result={'state':'EXACT_CONTROL_REPLAY_AND_SEPARATE_FINITE_CHECKS_ONLY','performer':'OpenAI/Codex /root/c261a_local_runtime_20261016',
        'source_exposure':'full new frozen proposal, control source/output and named primary imports read; prior exact chart consumed',
        'organization_independence_credit':0,'interpreter_version':sys.version,'inputs_before':before,'inputs_after':after,
        'all_12_inputs_unchanged':True,'runs':runs,'production_or_remote_actions':0,'scientific_acceptance':'NONE'}
(HERE/'CHECK_RUN_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
