"""Independently check an original Workspace #78 contract artifact."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile

root=Path(__file__).resolve().parent
expected_harness='90abecc6c203ce98614938d24b0c2f21f2a8431ff6c784680c5f6ad63c10e95d'
subject=(root/'native/tools/verify_federation_vector.py').read_bytes()
harness=(root/'proposal/tests/test_verify_federation_vector_mutations.py').read_bytes()
if hashlib.sha256(harness).hexdigest()!=expected_harness:
    raise ValueError('candidate harness source mismatch')
parsed=ast.parse(harness)
decl=next(n.value for n in parsed.body if isinstance(n,ast.Assign)
          and any(isinstance(t,ast.Name) and t.id=='MUTANTS' for t in n.targets))
mutations=ast.literal_eval(decl)
if len(mutations)!=13 or len({x[0] for x in mutations})!=13:
    raise ValueError('mutation declaration mismatch')
specs={label:(before,after,test,reason) for label,before,after,test,reason in mutations}

def verify(path,expected_sha):
    raw=Path(path).read_bytes();actual=hashlib.sha256(raw).hexdigest()
    if actual!=expected_sha:raise ValueError('artifact digest mismatch')
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=212 or len(names)!=len(set(names)) or z.testzip() is not None:
            raise ValueError('artifact population, uniqueness or CRC mismatch')
        if any(n.startswith('/') or '..' in Path(n).parts for n in names):
            raise ValueError('unsafe archive path')
        for log in ('discovery.log','discovery-optimized.log'):
            text=z.read(log).decode('utf-8')
            if not re.search(r'(?m)^Ran 276 tests in [0-9.]+s$',text) or not re.search(r'(?m)^OK$',text):
                raise ValueError('full discovery count or result mismatch: '+log)
            if re.search(r'(?m)\bskipped\b|^FAILED|^ERROR:',text,re.I):
                raise ValueError('skip or failure in discovery: '+log)
        summaries=[n for n in names if n.endswith('/summary.json')]
        if len(summaries)!=2:raise ValueError('summary count mismatch')
        records={};nonces=set()
        original=subject.decode('utf-8')
        for summary in summaries:
            parts=summary.split('/');mode=parts[1] if len(parts)>1 else ''
            if mode not in ('normal','optimized'):raise ValueError('unknown mode')
            entries=json.loads(z.read(summary));folder=summary.rsplit('/',1)[0]
            if not isinstance(entries,list) or len(entries)!=26:raise ValueError('summary population')
            for row in entries:
                label,kind=row['mutation'],row['kind'];key=(mode,label,kind)
                if label not in specs or kind not in ('baseline','mutant') or key in records:
                    raise ValueError('invalid or duplicate mutation row')
                before,after,test,reason=specs[label]
                if original.count(before)!=1:raise ValueError('native mutation anchor mismatch')
                candidate=subject if kind=='baseline' else original.replace(before,after,1).encode('utf-8')
                base=folder+'/'+label+'-'+kind
                source=z.read(base+'.py');stderr=z.read(base+'.stderr');z.read(base+'.stdout')
                event=json.loads(z.read(base+'.json'));digest=hashlib.sha256(source).hexdigest()
                if source!=candidate or digest!=row['source_sha256'] or event['source_before']!=digest or event['source_after']!=digest:
                    raise ValueError('candidate source or identity mismatch')
                if hashlib.sha256(stderr).hexdigest()!=row['stderr_sha256']:
                    raise ValueError('stderr hash mismatch')
                if (row['test']!=test or event['test']!=test or type(event.get('tests_run')) is not int
                        or event['tests_run']!=1 or type(row.get('exit_code')) is not int or any(
                        type(event.get(k)) is not int or event[k]!=0
                        for k in ('errors','skipped','expected_failures','unexpected_successes'))):
                    raise ValueError('child outcome identity or count mismatch')
                nonce=event.get('nonce')
                if not isinstance(nonce,str) or not re.fullmatch('[0-9a-f]{32}',nonce) or nonce in nonces:
                    raise ValueError('invalid or reused nonce')
                nonces.add(nonce);err=stderr.decode('utf-8')
                if not re.search(r'(?m)^Ran 1 test in ',err) or 'ERROR:' in err or 'skipped' in err:
                    raise ValueError('child transcript invalid')
                if kind=='baseline':
                    if row['exit_code']!=0 or event['failures']!=[] or not err.rstrip().endswith('OK'):
                        raise ValueError('baseline did not pass')
                else:
                    failures=event['failures']
                    if (row['exit_code']!=1 or len(failures)!=1 or failures[0].get('selected_body') is not True
                            or reason not in failures[0].get('message','') or err.count('FAIL: ')!=1
                            or not err.rstrip().endswith('FAILED (failures=1)')):
                        raise ValueError('mutant outcome mismatch')
                records[key]=digest
        if len(records)!=52 or len(nonces)!=52:
            raise ValueError('campaign record count mismatch')
        for label in specs:
            for kind in ('baseline','mutant'):
                if records[('normal',label,kind)]!=records[('optimized',label,kind)]:
                    raise ValueError('cross-mode source mismatch')
        return {'artifact_sha256':actual,'members':len(names),'crc':'valid',
                'tests_per_mode':276,'skips':0,'pairs_per_mode':13,'events':52,
                'candidate_bytes_and_stderr_hashes':'matched','unique_nonces':len(nonces),
                'candidate_harness_sha256':expected_harness,
                'scope':'source-bound finite contract artifact; not final federation gates or scientific review'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive');parser.add_argument('sha256');parser.add_argument('output')
    args=parser.parse_args();result=verify(args.archive,args.sha256)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
