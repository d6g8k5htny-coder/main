#!/usr/bin/env python3
"""Run same-source normal/-O checks and adversarial mutations.

All executions belong to the same author lineage. This runner does not perform
independent mathematical review or certify Gaussian analytic assertions.
"""
from __future__ import annotations
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MUTATIONS=('gap_sign','denominator_power','omit_rare_mass','hessian_scale',
           'clearance','drop_palm_weight','unauthorized_promotion','duplicate_keys')

def digest(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def execute(optimized:bool,mutation:str|None=None):
    args=[sys.executable]+(['-O'] if optimized else [])+[str(ROOT/'verify_local_path.py')]
    if mutation:args+=['--mutation',mutation]
    p=subprocess.run(args,cwd=ROOT,capture_output=True,timeout=30,check=False)
    return p

def main()->int:
    normal=execute(False);optimized=execute(True)
    if normal.returncode!=0 or optimized.returncode!=0:
        raise RuntimeError('BASELINE_FAILED\n'+normal.stdout.decode()+optimized.stdout.decode())
    if normal.stdout!=optimized.stdout:raise RuntimeError('MODE_TRANSCRIPTS_DIFFER')
    if normal.stderr or optimized.stderr:raise RuntimeError('BASELINE_STDERR')
    baseline=json.loads(normal.stdout)
    (ROOT/'ALGEBRA_NORMAL.json').write_bytes(normal.stdout)
    (ROOT/'ALGEBRA_OPTIMIZED.json').write_bytes(optimized.stdout)
    mutations=[]
    for name in MUTATIONS:
        a,b=execute(False,name),execute(True,name)
        if a.returncode!=1 or b.returncode!=1:raise RuntimeError('MUTATION_NOT_REJECTED:'+name)
        if a.stdout!=b.stdout:raise RuntimeError('MUTATION_MODE_MISMATCH:'+name)
        if a.stderr or b.stderr:raise RuntimeError('MUTATION_STDERR:'+name)
        parsed=json.loads(a.stdout)
        mutations.append({'name':name,'normal_exit':a.returncode,'optimized_exit':b.returncode,
                          'byte_identical':True,'sha256':digest(a.stdout),'output':parsed})
    mb=(json.dumps({'scope':'Same-source adversarial regression, not independent review',
                    'mutations':mutations},sort_keys=True,indent=2)+'\n').encode()
    (ROOT/'MUTATION_RECEIPTS.json').write_bytes(mb)
    receipt={'artifact':'LPW-EXEC-20260912','date':'2026-09-12',
             'normal_exit':normal.returncode,'optimized_exit':optimized.returncode,
             'normal_optimized_byte_identical':True,'algebra_checks_passed':baseline['checks_passed'],
             'mutation_families_rejected':len(mutations),'mutation_executions':2*len(mutations),
             'normal_transcript_sha256':digest(normal.stdout),'optimized_transcript_sha256':digest(optimized.stdout),
             'mutation_receipts_sha256':digest(mb),
             'input_files':{name:digest((ROOT/name).read_bytes()) for name in ('verify_local_path.py','CURRENT_STATE.json','02_LOCAL_PATH_LOWER_BOUND_CANDIDATE.md','requirements.txt')},
             'limitations':['Exact algebra and regression only','No independent review',
                            'No numerical c or r0','No Gaussian probability machine certificate',
                            'No Lean compilation','No theorem promotion','No change to old verifier'],
             'result':'PASS_WITHIN_TEST_SCOPE'}
    (ROOT/'EXECUTION_RECEIPT.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps(receipt,sort_keys=True,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
