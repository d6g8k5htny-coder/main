from pathlib import Path
import hashlib,json,os,re,subprocess,sys,time
ROOT=Path('/workspace/universal-law/website-repair-20261028')
OUT=Path(__file__).resolve().parent
PINNED=Path('/workspace/research/audit/formal/toolchain/bin')
VENV='/workspace/cbz0/verification-venv/bin/python'
baseenv=dict(os.environ)
baseenv['PYTHONPATH']=str(ROOT/'tests')+os.pathsep+baseenv.get('PYTHONPATH','')

def run(label,args,expected,env=None,extra=None,timeout=1200):
    stdout=OUT/(label+'.stdout');stderr=OUT/(label+'.stderr')
    if stdout.exists()or stderr.exists():raise RuntimeError('refuse existing record: '+label)
    started=time.time()
    with stdout.open('w')as out,stderr.open('w')as err:
        try:p=subprocess.run(args,cwd=ROOT,env=baseenv if env is None else env,stdout=out,stderr=err,timeout=timeout);code=p.returncode;failure=None
        except subprocess.TimeoutExpired:code=None;failure='TIMEOUT'
    text=stderr.read_text()+stdout.read_text()
    matches=re.findall(r'Ran (\d+) tests? in',text)
    count=int(matches[-1])if matches else None
    row={'label':label,'argv':args,'cwd':str(ROOT),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(started)),'elapsed_seconds':time.time()-started,'exit_code':code,'runner_failure':failure,'expected_tests':expected,'observed_tests':count,'unittest_skips_observed':bool(re.search(r'\bskipped[= ]',text)),'stdout_sha256':hashlib.sha256(stdout.read_bytes()).hexdigest(),'stderr_sha256':hashlib.sha256(stderr.read_bytes()).hexdigest(),'pass':code==0 and count==expected and not re.search(r'\bskipped[= ]',text)}
    if extra:row.update(extra)
    (OUT/(label+'.json')).write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
    return row

if sys.argv[1]=='python':
    for optimized in (False,True):
        run('pytest-bytecode-'+('optimized'if optimized else'normal'),[VENV,*(['-O']if optimized else[]),'-B','-m','unittest','discover','-s','tests','-p','test_pytest_bytecode_policy.py','-v'],10)
    env=dict(baseenv,GB_MATH_CHECKOUT='/workspace/cbz0/repos/Math-')
    for optimized in (False,True):
        run('composition525-'+('optimized'if optimized else'normal'),[sys.executable,*(['-O']if optimized else[]),'-B','-m','unittest','test_graph_binding_check.Composition525Test','-v'],1,env=env,extra={'explicit_environment':{'GB_MATH_CHECKOUT':env['GB_MATH_CHECKOUT']},'pinned_main_composition_commit':'29b7d5969e2cd707debb7449c69a71bbcd39cf63','pinned_math_composition_commit':'dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde','scope':'Actual historical pinned composition; not current Math-default or full-source closure'})
elif sys.argv[1]=='lean':
    env=dict(baseenv,PATH=str(PINNED)+os.pathsep+baseenv.get('PATH',''),LEAN_NUM_THREADS='1')
    # These share a project state; execute sequentially only.
    run('lean-actual-compiled-inventory',[sys.executable,'-B','-m','unittest','test_formal_compiled_inventory.PinnedLeanInventoryTests.test_actual_clean_and_omitted_constants','-v'],1,env=env,extra={'pinned_bin':str(PINNED),'scope':'Actual current compiled inventory integration test; clean plus5 planted omission/admission controls'})
    run('lean-actual-package-gate',[sys.executable,'-B','-m','unittest','test_formal_gate.RepositoryControls.test_real_package_executes_with_receipt','-v'],1,env=env,extra={'pinned_bin':str(PINNED),'scope':'Actual current ordinary package gate integration test; writes live formal/.lake/formal-evidence, not a new shared cbz-wrapper receipt'})
elif sys.argv[1]=='sandbox':
    env=dict(baseenv)
    for key in ('DOCKER_HOST','DOCKER_CONTEXT','DOCKER_TLS','DOCKER_TLS_VERIFY','DOCKER_CERT_PATH'):env.pop(key,None)
    health=subprocess.run(['docker','--host=unix:///var/run/docker.sock','info','--format','{{.ServerVersion}}'],env=env,capture_output=True,text=True)
    (OUT/'docker-local-health.json').write_text(json.dumps({'exit_code':health.returncode,'server_version':health.stdout.strip(),'stderr':health.stderr},indent=2)+'\n')
    if health.returncode:raise RuntimeError('managed local Docker unavailable; no daemon selector bypass')
    previous=json.loads(Path('/workspace/cbz0/final-regression/sandbox-staged-result.json').read_text())
    argv=previous['argv'];argv[argv.index('--name')+1]='cbz-website-sandbox-current-skips-20261029'
    run('sandbox-five-current-test-bytes',argv,5,env=env,extra={'source_fixture_commit':'6fc021839072d8c34f41e18e77b0f618c53baa15','test_sha256':hashlib.sha256((ROOT/'tests/test_architecture_sandbox.py').read_bytes()).hexdigest(),'fixture_test_sha256':hashlib.sha256(Path('/workspace/cbz0/sandbox-source-6fc0218/tests/test_architecture_sandbox.py').read_bytes()).hexdigest(),'scope':'Five actual runtime controls using6fc source fixture with exact current test bytes; not current full-source snapshot or kernel receipt'},timeout=120)
else:raise RuntimeError('unknown bounded group')
