#!/usr/bin/env python3
"""Exact reproduction, with source-bound batches for constrained tool runtimes."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from verify_p15 import FAULTS
from exact_core import need

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--batch',type=int);ap.add_argument('--batch-size',type=int,default=6)
    ap.add_argument('--collect',action='store_true');a=ap.parse_args()
    need(a.batch_size>0 and (a.batch is None or a.batch>=0),'positive-batch-size')
    a.output.mkdir(parents=True,exist_ok=True)
    code=Path(__file__).resolve().parent
    source_binding={str(p.relative_to(code.parent)):digest(p) for p in sorted(code.glob('*.py'))}
    source_binding.update({str(p.relative_to(code.parent)):digest(p) for p in sorted((code.parent/'inputs').glob('PALETTE*.json'))})
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0');jobs=[]
    def job(name,args,expected,status):jobs.append({'name':name,'args':args,'expected_exit':expected,'expected_status':status})
    for mode,flag in [('normal',[]),('optimized',['-O'])]:job('baseline_'+mode,[*flag,str(code/'verify_p15.py')],0,'PASS')
    for fault in FAULTS:
        for mode,flag in [('normal',[]),('optimized',['-O'])]:job('fault_'+fault+'_'+mode,[*flag,str(code/'verify_p15.py'),'--fault',fault],1,'REJECT')
    for mode,flag in [('normal',[]),('optimized',['-O'])]:job('intentional_survivor_'+mode,[*flag,str(code/'verify_p15.py'),'--fault','survivor'],0,'UNDETECTED')
    for file,expected,status in [('PALETTE_THREE_BLOCKS.json',0,'PASS'),('PALETTE_INVALID_COMMON_COLOR.json',1,'REJECT'),('PALETTE_SCALAR_TWO_BLOCKS.json',0,'PASS')]:
        for mode,flag in [('normal',[]),('optimized',['-O'])]:job('cli_'+file.split('.')[0]+'_'+mode,[*flag,str(code/'certify_palette_plan.py'),str(code.parent/'inputs'/file)],expected,status)
    selected=jobs if a.batch is None else jobs[a.batch*a.batch_size:(a.batch+1)*a.batch_size]
    if not a.collect:
        for j in selected:
            t=time.monotonic();cp=subprocess.run([sys.executable,*j['args']],cwd=code,env=env,capture_output=True,timeout=45);elapsed=time.monotonic()-t
            (a.output/(j['name']+'.stdout')).write_bytes(cp.stdout);(a.output/(j['name']+'.stderr')).write_bytes(cp.stderr)
            try:body=json.loads(cp.stdout)
            except (ValueError,UnicodeDecodeError) as e:raise RuntimeError(j['name']+': missing structured output; not a mathematical rejection') from e
            need(cp.returncode==j['expected_exit'],j['name']+':unexpected-exit')
            need(body.get('status')==j['expected_status'],j['name']+':unexpected-status')
            need(not cp.stderr,j['name']+':stderr-not-empty')
            record={'name':j['name'],'exit':cp.returncode,'status':body['status'],'seconds':elapsed,'stdout_sha256':hashlib.sha256(cp.stdout).hexdigest(),'source_binding':source_binding}
            (a.output/(j['name']+'.receipt.json')).write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
        if a.batch is not None:
            print(json.dumps({'status':'BATCH_COMPLETE','batch':a.batch,'jobs':len(selected),'total_expected_jobs':len(jobs)},sort_keys=True));return
    records=[]
    for j in jobs:
        record=json.loads((a.output/(j['name']+'.receipt.json')).read_text());raw=(a.output/(j['name']+'.stdout')).read_bytes()
        need(record['source_binding']==source_binding,j['name']+':source-drift')
        need(record['stdout_sha256']==hashlib.sha256(raw).hexdigest(),j['name']+':output-drift')
        need(record['status']==j['expected_status'] and record['exit']==j['expected_exit'],j['name']+':receipt-verdict')
        need(json.loads(raw)['status']==j['expected_status'] and not (a.output/(j['name']+'.stderr')).read_bytes(),j['name']+':actual-output')
        records.append({k:v for k,v in record.items() if k!='source_binding'})
    normal=(a.output/'baseline_normal.stdout').read_bytes();optimized=(a.output/'baseline_optimized.stdout').read_bytes();need(normal==optimized,'baseline-mode-byte-drift')
    result={'status':'PASS','subprocesses':len(records),'checks_per_mode':json.loads(normal)['checks'],'baseline_stdout_sha256':hashlib.sha256(normal).hexdigest(),'baseline_modes':'BYTE_IDENTICAL','fault_families':len(FAULTS),'rejected_fault_runs':2*len(FAULTS),'intentional_survivors':2,'cli_tests':6,'external_reviews':0,'formal_prover_runs':0,'source_binding':source_binding,'records':records,'summed_subprocess_seconds':sum(x['seconds'] for x in records)}
    (a.output/'REPRODUCTION_RECEIPT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','source_binding')},sort_keys=True))
if __name__=='__main__':main()
