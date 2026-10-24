# End-to-end composition check on the composed tree (candidate three files on main b7deb198):
# produce a receipt with the candidate checker through the test harness's synthetic runtime
# (Lean/Lake mocked, real execute/run/log writer), then feed it to the CURRENT consumer
# tools/required_formal_check.bind_receipt and tools/formal_execution_context.bind.
import os, sys, json, importlib.util, pathlib, hashlib
sys.path.insert(0, '.')
os.environ['GITHUB_REPOSITORY']='d6g8k5htny-coder/main'; os.environ['GITHUB_RUN_ID']='424242'; os.environ['GITHUB_RUN_ATTEMPT']='1'
import tests.test_formal_gate as T
t=T.SyntheticControls('test_valid_runtime_preserves_both_version_observations_and_logs'); t.setUp()
final="Lean (version 4.34.1, aarch64-unknown-linux-gnu, commit "+T.SyntheticControls.LEAN_COMMIT+", Release)\n"
receipt=t.runtime_probe([T.SyntheticControls.RUNTIME_RELEASE, final])
out=t.runtime_state['out']; rp=out/'receipt.json'
print('receipt logs:', sorted(receipt['logs'])); print('lean_version:', receipt['lean_version'])
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
rfc=load('rfc','tools/required_formal_check.py')
env={'GITHUB_REPOSITORY':'d6g8k5htny-coder/main','GITHUB_SHA':receipt['checked_commit'],'GITHUB_RUN_ID':'424242','GITHUB_RUN_ATTEMPT':'1'}
r=rfc.bind_receipt(rp, t.root/'pkg/manifest.json', env, receipt['checked_commit']); print('required_formal_check.bind_receipt OK:', r)
# negative: tamper with the new preflight log -> must refuse
(out/'preflight-version.log').write_text('tampered\n')
try:
    rfc.bind_receipt(rp, t.root/'pkg/manifest.json', env, receipt['checked_commit']); print('UNEXPECTED: tamper accepted')
except ValueError as e: print('tamper refused as expected:', e)
(out/'preflight-version.log').write_text(T.SyntheticControls.RUNTIME_RELEASE)
fec=load('fec','tools/formal_execution_context.py')
try:
    r2=fec.bind(rp, source_repository='d6g8k5htny-coder/Math-', expected_commit=receipt['checked_commit'], observed_commit=receipt['checked_commit'], observed_remote='https://github.com/d6g8k5htny-coder/Math-', execution_repository='d6g8k5htny-coder/main', execution_commit='b'*40, run_id='424242')
    print('formal_execution_context.bind OK: receipt_sha256', r2['receipt_sha256'][:16])
except Exception as e: print('formal_execution_context.bind refused:', e)
t.tmp.cleanup()
