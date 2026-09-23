"""Build a bounded register preflight candidate; never update a Git ref or science status."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zlib

BASE = '6628463c1341d3c23d14913af9dad9af24822eb1'
CARRIER_HASH = '7d3d01a838d2f0a5b38837ddeb39bcbcc9445c3f5619c5b042bfbb0f927807f0'
ALLOWED = {'registers/CONSUMERS.json', 'registers/README.md',
           'tests/test_register_preflight.py', 'tests/test_register_source_selection.py',
           'tools/registers_import.py', 'tools/registers_preflight.py'}

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def blob(data: bytes) -> str:
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    args = ap.parse_args(); root = Path(args.root).resolve(); out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    def run(argv, expect=0, label=None, timeout=400):
        p = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=timeout)
        key = label or f'command-{len(rows):02d}'
        log = (p.stdout+'\nSTDERR\n'+p.stderr).encode()
        (out/(key+'.log')).write_bytes(log)
        rows.append({'command': argv, 'returncode': p.returncode, 'expected_returncode': expect,
                     'log': key+'.log', 'log_sha256': sha(log)})
        if p.returncode != expect:
            raise RuntimeError(f'{key}: expected {expect}, got {p.returncode}; see preserved log')
        return p.stdout
    report = {'scope':'bounded engineering validation, no R1 activation or mathematical acceptance',
              'source_commit':BASE, 'python':sys.version, 'scientific_status_changed':False,
              'canonical_import_completed':False, 'git_refs_modified':False, 'executions':rows}
    try:
        if run(['git','rev-parse','HEAD']).strip()!=BASE: raise RuntimeError('wrong checkout')
        carrier = b''.join((Path(__file__).parent/f'carrier.{i:02d}').read_bytes() for i in range(4))
        if len(carrier)!=10273 or sha(carrier)!=CARRIER_HASH: raise RuntimeError('carrier identity mismatch')
        obj=json.loads(zlib.decompress(carrier)); files=obj['files']
        if len(files)!=6 or {r['path'] for r in files}!=ALLOWED: raise RuntimeError('unexpected patch paths')
        for r in files:
            p=root/r['path']; before=r['before_git_blob']
            if before is None:
                if p.exists(): raise RuntimeError('new path already exists: '+r['path'])
            elif not p.is_file() or p.is_symlink() or blob(p.read_bytes())!=before:
                raise RuntimeError('preimage drift: '+r['path'])
        patch=out/'candidate.patch'; patch.write_text(obj['patch'],encoding='utf-8')
        run(['git','apply','--check',str(patch)],label='patch-check')
        run(['git','apply',str(patch)],label='patch-apply')
        def check_expected_files():
            for r in files:
                data=(root/r['path']).read_bytes()
                if len(data)!=r['bytes'] or sha(data)!=r['sha256'] or blob(data)!=r['after_git_blob']:
                    raise RuntimeError('postimage mismatch: '+r['path'])
        check_expected_files()
        for tool in ['registers_import','registers_check','claims_check','reviews_check','frozen_check','consumers_check']:
            argv=[sys.executable,'tools/'+tool+'.py']+(['--check'] if tool=='registers_import' else [])
            run(argv,label=tool)
        run([sys.executable,'-m','pytest','-q','tests/test_registers.py','tests/test_register_source_selection.py',
             'tests/test_register_preflight.py','tests/test_reviews.py','tests/test_consumers.py'],label='targeted-tests')
        for name,rc,label in [('GP-REG-032_v1.2_export_2026-09-18.xlsx',0,'current-preview'),
                              ('GP-REG-032_v1.2_export_2026-09-23_R1.xlsx',1,'r1-preview')]:
            text=run([sys.executable,'tools/registers_preflight.py','--source-name',name],expect=rc,label=label)
            preview=json.loads(text); (out/(label+'.json')).write_text(text,encoding='utf-8')
            if preview['canonical_import_completed'] or preview['scientific_status_changed']:
                raise RuntimeError('preview falsely claims activation or science change')
            if label=='r1-preview':
                report['r1_blocker_counts']={k:len(v) for k,v in preview['blockers'].items()}
                if report['r1_blocker_counts']!={'transcription':0,'register_structure':3,'stale_known_findings':0,
                        'bound_observations':0,'review_interface':3,'frozen_interface':11}:
                    raise RuntimeError('R1 observations drifted; reconcile instead of hiding')
        run(['git','diff','--check'],label='whitespace')
        changed=set(run(['git','diff','--name-only']).splitlines())
        changed.update(run(['git','ls-files','--others','--exclude-standard']).splitlines())
        if changed!=ALLOWED: raise RuntimeError('unexpected changed paths: '+repr(sorted(changed)))
        check_expected_files()
        run(['git','add','--',*sorted(ALLOWED)],label='stage-six-only')
        tree=run(['git','write-tree'],label='candidate-tree').strip()
        base_tree=run(['git','rev-parse','HEAD^{tree}']).strip()
        # Publish only content-addressed blobs/tree. No refs/commits/settings API is called.
        token=os.environ['GH_TOKEN']
        def post(endpoint,body):
            req=urllib.request.Request('https://api.github.com/repos/d6g8k5htny-coder/main/git/'+endpoint,
                data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+token,
                'Accept':'application/vnd.github+json','Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(req,timeout=60) as resp: return json.load(resp)
        elements=[]
        for r in files:
            result=post('blobs',{'encoding':'utf-8','content':(root/r['path']).read_text(encoding='utf-8')})
            if result['sha']!=r['after_git_blob']: raise RuntimeError('remote blob mismatch')
            elements.append({'path':r['path'],'mode':'100644','type':'blob','sha':result['sha']})
        result=post('trees',{'base_tree':base_tree,'tree':elements})
        if result['sha']!=tree: raise RuntimeError('remote tree differs from tested tree')
        report.update({'status':'PASS','candidate_tree':tree,'files':files,'protected_paths_unchanged':True,
                       'current_source_still_sept18':True,'carrier_sha256':CARRIER_HASH})
        print(json.dumps({k:v for k,v in report.items() if k not in ('files','executions')}))
        return 0
    except Exception as exc:
        report['status']='FAILED'; report['error']=str(exc)
        raise
    finally:
        (out/'build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': raise SystemExit(main())
