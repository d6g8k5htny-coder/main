#!/usr/bin/env python3
"""Replay selected inspected Kimi programs from temporary copies.

Receipt/output must be outside the sealed bundle. Same-source replay is not
independent proof. W8's full sweep is deliberately excluded. No network or Drive
operation is made by this runner. Verify the root manifest before execution.
"""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, os, platform
import shutil, subprocess, sys, tempfile, time

SPECS = [
 ('constant','LPW_CONSTANT/lpw_constant.py','LPW_CONSTANT/out_normal.txt'),
 ('constant','LPW_CONSTANT/falsify.py','LPW_CONSTANT/falsify_normal.txt'),
 ('review','RB_R2_part1_symbolic.py','p1_normal.out'),
 ('review','rd_r4r5_verify.py','out_normal.json'),
 ('review','rc_r3_numeric.py','rc_r3_numeric_output.txt'),
 ('review','rb_r2_part2_spectral_v2.py','rb_v2_out_normal.txt'),
 ('qc','QC_RETURN03_REVIEW/qc_return03_numeric.py','QC_RETURN03_REVIEW/OUTPUT_NORMAL.json')]
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--group', choices=['all','quick','constant','review','qc'], default='quick')
    ap.add_argument('--output', type=Path, required=True, help='New output directory OUTSIDE this bundle')
    ap.add_argument('--timeout', type=int, default=240)
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=args.output.resolve()
    if out == root or root in out.parents or out.exists():
        ap.error('Output must be a NEW directory outside the sealed bundle')
    p=subprocess.run([sys.executable, str(root/'tools/verify_bundle.py')], capture_output=True)
    if p.returncode:
        sys.stdout.buffer.write(p.stdout); sys.stderr.buffer.write(p.stderr)
        return 2
    out.mkdir(parents=True)
    env=dict(os.environ)
    for key in list(env):
        if key.startswith(('LPW_MUTATE','QC_MUTATE')): env.pop(key)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    specs=[s for s in SPECS if args.group=='all' or s[0]==args.group or
           (args.group=='quick' and (s[1].endswith('/lpw_constant.py') or s[1]=='rd_r4r5_verify.py'))]
    results=[]
    for group,script,ref in specs:
        for opt in [False,True]:
            name=group+'_'+Path(script).stem+('_O' if opt else '_normal')
            with tempfile.TemporaryDirectory(prefix='lpw-replay-') as tmp:
                dst=Path(tmp)/'source';shutil.copytree(root/'raw'/group,dst)
                target=dst/script
                cmd=[sys.executable]+(['-O'] if opt else [])+[str(target)]
                start=time.monotonic()
                record={'name':name,'source_sha256':sha(target.read_bytes()),'mode':'optimized' if opt else 'normal'}
                try:
                    run=subprocess.run(cmd,cwd=target.parent,env=env,capture_output=True,timeout=args.timeout)
                    record.update(exit=run.returncode, stdout_sha256=sha(run.stdout),stderr_sha256=sha(run.stderr),
                                  matches_delivered_stdout=(run.stdout==(root/'raw'/group/ref).read_bytes()),
                                  stderr_empty=(not run.stderr))
                    (out/(name+'.out')).write_bytes(run.stdout);(out/(name+'.err')).write_bytes(run.stderr)
                except subprocess.TimeoutExpired as exc:
                    record.update(exit=None,timeout=True,matches_delivered_stdout=False)
                    (out/(name+'.out')).write_bytes(exc.stdout or b'');(out/(name+'.err')).write_bytes(exc.stderr or b'')
                record['elapsed_seconds']=round(time.monotonic()-start,3)
                results.append(record)
                print(json.dumps(record),flush=True)
    success=all(r.get('exit')==0 and r.get('matches_delivered_stdout') and r.get('stderr_empty') for r in results)
    receipt={'scope':'SAME_SOURCE_REPLAY_ONLY', 'python':sys.version,'platform':platform.platform(),
             'versions':{k:importlib.metadata.version(k) for k in ['sympy','mpmath','numpy','scipy']},
             'results':results, 'all_match':success, 'remote_writes':0,'W8_full_sweep_executed':False}
    (out/'REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return 0 if success else 1

if __name__=='__main__':
    sys.exit(main())
