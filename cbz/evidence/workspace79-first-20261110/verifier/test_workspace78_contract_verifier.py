"""Adverse and source-mutation checks for the one-off artifact verifier."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

root=Path(__file__).resolve().parent
original=Path(sys.argv[1]).resolve()
expected=sys.argv[2]
if hashlib.sha256(original.read_bytes()).hexdigest()!=expected:
    raise ValueError('original artifact digest mismatch')
source_verifier=root/'verify_workspace78_contract.py'
code=source_verifier.read_text()
source_guard='source!=candidate or '
skip_guard="if re.search(r'(?m)\\bskipped\\b|^FAILED|^ERROR:',text,re.I):"
if code.count(source_guard)!=1 or code.count(skip_guard)!=1:
    raise ValueError('verifier mutation anchors changed')
out=Path(sys.argv[3]).resolve()
out.mkdir(parents=True,exist_ok=False)
for relative in ('native/tools/verify_federation_vector.py',
                 'proposal/tests/test_verify_federation_vector_mutations.py'):
    target=out/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/relative,target)
verifier=out/'verify_workspace78_contract.py'
verifier.write_text(code)
source_mutant=out/'verify_workspace78_contract_mutant_source.py'
skip_mutant=out/'verify_workspace78_contract_mutant_skip.py'
source_mutant.write_text(code.replace(source_guard,'',1))
skip_mutant.write_text(code.replace(skip_guard,'if False:',1))
with zipfile.ZipFile(original) as z:
    members={n:z.read(n) for n in z.namelist()}

def save(name,changes,remove=()):
    path=out/(name+'.zip')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for key,data in members.items():
            if key not in remove:z.writestr(key,changes.get(key,data))
    return path,hashlib.sha256(path.read_bytes()).hexdigest()

summaries=[n for n in members if n.endswith('/summary.json')]
if len(summaries)!=2:raise ValueError('original summary count')
changes={}
for summary in summaries:
    folder=summary.rsplit('/',1)[0];stem=folder+'/vector-set-mutant'
    altered=members[stem+'.py']+b'\n# synthetic source-integrity adverse fixture\n'
    digest=hashlib.sha256(altered).hexdigest()
    event=json.loads(members[stem+'.json']);event['source_before']=event['source_after']=digest
    rows=json.loads(members[summary]);selected=[r for r in rows if r['mutation']=='vector-set' and r['kind']=='mutant']
    if len(selected)!=1:raise ValueError('original source row mismatch')
    selected[0]['source_sha256']=digest
    changes[stem+'.py']=altered;changes[stem+'.json']=json.dumps(event).encode();changes[summary]=json.dumps(rows).encode()
forged,forged_sha=save('forged-source',changes)
skip,skip_sha=save('inserted-skip',{'discovery.log':members['discovery.log']+b"synthetic ... skipped 'adverse'\n"})
missing,missing_sha=save('missing-summary',{},(summaries[0],))
cases=[('good',verifier,original,expected,0),('missing',verifier,missing,missing_sha,1),
       ('source-refused',verifier,forged,forged_sha,1),('source-guard-mutant',source_mutant,forged,forged_sha,0),
       ('skip-refused',verifier,skip,skip_sha,1),('skip-guard-mutant',skip_mutant,skip,skip_sha,0)]
results=[]
for mode in ('normal','optimized'):
    for label,script,archive,digest,want in cases:
        target=out/(mode+'-'+label+'.json')
        p=subprocess.run([sys.executable,'-B',*(['-O'] if mode=='optimized' else []),'-S',
                          str(script),str(archive),digest,str(target)],capture_output=True,text=True,timeout=60)
        (out/(mode+'-'+label+'.stdout')).write_text(p.stdout)
        (out/(mode+'-'+label+'.stderr')).write_text(p.stderr)
        status=0 if p.returncode==0 else 1
        results.append({'mode':mode,'case':label,'actual_exit':p.returncode,'expected_class':want,
                        'met_expectation':status==want,'fixture_sha256':digest})
        print(json.dumps(results[-1]),flush=True)
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
if len(results)!=12 or any(not r['met_expectation'] for r in results):
    raise SystemExit('verifier adverse/mutation check failed')
