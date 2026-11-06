#!/usr/bin/env python3
"""G0: real isolated checker-source mutations; 100% declared mutants must be killed."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent

# Each change weakens a claimed auditor rule, rather than planting a bad proof fixture.
MUTATIONS=[
 ('empty-lean-findings','source_checks.py',"return {'findings': findings, 'imports': imports, 'declarations': declarations}","return {'findings': [], 'imports': imports, 'declarations': declarations}"),
 ('canonical-unicode-boundary-weakened','canonical_lean.py',"if ch in _EXCLUDED_LETTERS:","if False:"),
 ('disable-placeholder-category','source_checks.py',"placeholders = {'sorry', 'admit', 'sorryAx', 'admitGoal', 'mkSorry'}","placeholders = set()"),
 ('disable-axiom-category','source_checks.py',"if tail in {'axiom', 'constant', 'constants'}:","if False:"),
 ('disable-native-category','source_checks.py',"if tail in native:","if False:"),
 ('disable-macro-gap-category','source_checks.py',"if tail in expansion:","if False:"),
 ('drop-lean-dependencies','source_checks.py',"'imports': imports, 'declarations': declarations}","'imports': [], 'declarations': declarations}"),
 ('disable-float-literal-category','source_checks.py',"if isinstance(node, ast.Constant) and isinstance(node.value, (float, complex)):","if False:"),
 ('allow-python-division','source_checks.py',"if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):","if False:"),
 ('allow-negative-dynamic-powers','source_checks.py',"if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):","if False:"),
 ('allow-unknown-python-calls','source_checks.py',"if not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls:","if False:"),
 ('disable-exact-python-profile','source_checks.py',"if not mathematical:\n            continue","if True:\n            continue"),
 ('allow-reflective-attributes','source_checks.py',"if isinstance(node, ast.Attribute) and node.attr not in {'numerator','denominator'}:","if False:"),
 ('allow-protected-name-rebinding','source_checks.py',"if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in protected:","if False:"),
 ('skip-all-but-first-file','auditor.py',"for path in sorted(set(tracked)|set(files)):","for path in sorted(set(tracked)|set(files))[:1]:"),
 ('allow-incomplete-classification','auditor.py',"if set(tracked)!=set(files):","if False:"),
 ('allow-unknown-classification','auditor.py',"'external_pinned','binary_proof','documentation','data'})","'external_pinned','binary_proof','documentation','data','ignore'})"),
 ('disable-content-hash-check','auditor.py',"if digest(data)!=entry.get('sha256'):","if False:"),
 ('allow-symlink-files','auditor.py',"if any((root.joinpath(*PurePosixPath(path).parts[:i])).is_symlink() for i in range(1,len(PurePosixPath(path).parts)+1)):","if False:"),
 ('allow-unreadable-files','auditor.py',"if not info.st_mode & (stat.S_IRUSR|stat.S_IRGRP|stat.S_IROTH):","if False:"),
 ('skip-unreadable-file','auditor.py',"add('FILE_UNREADABLE','no read permissions even under privileged runner',rid,path);continue","continue"),
 ('allow-relabeled-source','auditor.py',"if kind in {'documentation','data'} and (source_like or suffix in BINARY_EXTENSIONS):","if False:"),
 ('allow-unpinned-external-source','auditor.py',"if kind=='external_pinned' or 'pin' in entry:","if False:"),
 ('skip-nonmath-review','auditor.py',"if kind=='engineering_python':","if False:"),
 ('skip-binary-proof-bindings','auditor.py',"if kind=='binary_proof' or suffix in BINARY_EXTENSIONS:","if False:"),
 ('ignore-retained-finding-counts','auditor.py',"if actual!=entry.get('expected_findings'):","if False:"),
 ('ignore-quarantine-boundary','auditor.py',"if kind=='quarantined_source':","if False:"),
 ('restore-shadowed-fractions-exemption','auditor.py'," and 'python:fractions' not in module_map:",":"),
 ('drop-python-submodule-candidates','auditor.py',"for candidate in result.get('import_candidates',[]):","for candidate in []:"),
 ('allow-nonmath-dependency-values','auditor.py',"if kind=='math_python' and records[target]['entry']['kind']=='engineering_python':","if False:"),
 ('skip-path-validation','auditor.py',"if not normalized_path(path):","if False:"),
 ('allow-quarantine-dependency','auditor.py',"if records[target]['entry']['kind'] in RETAINED:","if False:"),
 ('skip-vacuity-coverage','auditor.py',"if not isinstance(annotations,dict) or len(keys)!=len(set(keys)) or set(annotations)!=set(keys):","if False:"),
 ('ignore-contradictory-hypotheses','auditor.py',"if declaration['contradictions'] and a['status']!='vacuous':","if False:"),
 ('allow-unbound-nonvacuity','auditor.py',"if a['status']=='nonvacuous':","if False:"),
 ('skip-status-evidence','auditor.py',"if entry.get('status_evidence') is not None:","if False:"),
 ('allow-arbitrary-foundation-set','auditor.py',"if not isinstance(approved,list) or any(not isinstance(a,str) for a in approved) or len(approved)!=3 or set(approved)!=APPROVED_AXIOMS:","if False:"),
 ('empty-final-findings','auditor.py',"'retained_count':len(retained_findings),'findings':findings,","'retained_count':len(retained_findings),'findings':[],"),
 ('always-report-static-pass','auditor.py',"'result':'STATIC_PASS' if not findings else 'FAIL'","'result':'STATIC_PASS'"),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def parse_test_result(output, returncode):
    count=re.search(r'Ran (\d+) tests? in ',output)
    tests=int(count.group(1)) if count else 0
    failed=re.search(r'FAILED \(([^)]*)\)',output)
    fields=dict((k,int(v)) for k,v in re.findall(r'(failures|errors)=(\d+)',failed.group(1))) if failed else {}
    failures,errors=fields.get('failures',0),fields.get('errors',0)
    return tests,failures,errors


def run_one(name, filename, transform, output_dir):
    """Execute the unchanged behavioral suite against a temporary actual checker mutation."""
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    original=(HERE/filename).read_text(encoding='utf-8')
    try:
        changed=transform(original)
        ast.parse(changed,filename=filename)
    except (SyntaxError,ValueError,TypeError) as exc:
        return {'name':name,'file':filename,'outcome':'INVALID','detail':str(exc),'assertion_failures':0,'errors':0}
    with tempfile.TemporaryDirectory(prefix='cbz-layer-a-mutant-') as temporary:
        root=Path(temporary);(root/'tests').mkdir()
        for source in ['auditor.py','source_checks.py','canonical_lean.py']:
            (root/source).write_bytes((HERE/source).read_bytes())
        (root/'tests/test_auditor.py').write_bytes((HERE/'tests/test_auditor.py').read_bytes())
        (root/filename).write_text(changed,encoding='utf-8')
        command=[sys.executable,'-B','-S','-m','unittest','discover','-s',str(root/'tests'),'-p','test_auditor.py','-v']
        env=dict(os.environ);env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONNOUSERSITE']='1'
        try:
            p=subprocess.run(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
        except subprocess.TimeoutExpired as exc:
            return {'name':name,'file':filename,'outcome':'ERROR','detail':'behavioral suite timeout','assertion_failures':0,'errors':1}
        output=p.stdout
    log=output_dir/(name+'.log');log.write_text(output,encoding='utf-8')
    tests,failures,errors=parse_test_result(output,p.returncode)
    if p.returncode==0 and tests:
        outcome='SURVIVED'
    elif p.returncode!=0 and tests and failures and not errors:
        outcome='KILLED'
    else:
        outcome='ERROR'
    return {'name':name,'file':filename,'outcome':outcome,'returncode':p.returncode,'tests_run':tests,
            'assertion_failures':failures,'errors':errors,'original_sha256':sha(original.encode()),
            'mutated_sha256':sha(changed.encode()),'log':str(log),'log_sha256':sha(output.encode())}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='G0 receipt JSON; logs in a receipt-specific sibling directory')
    args=parser.parse_args(argv)
    started=datetime.now(timezone.utc).isoformat()
    source_paths=['auditor.py','source_checks.py','canonical_lean.py','mutation_test.py','tests/test_auditor.py','tests/test_mutation_runner.py','SPEC.md','README.md','LEXER_PROVENANCE.json']
    starting_sources={p:sha((HERE/p).read_bytes()) for p in source_paths}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    logs=args.output.parent/(args.output.stem+'-mutation-logs');logs.mkdir(exist_ok=True)
    command=[sys.executable,'-B','-S','-m','unittest','discover','-s',str(HERE/'tests'),'-v']
    baseline=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,cwd=HERE,timeout=60)
    (logs/'baseline.log').write_text(baseline.stdout,encoding='utf-8')
    tests,failures,errors=parse_test_result(baseline.stdout,baseline.returncode)
    results=[]
    if baseline.returncode==0 and tests:
        for name,filename,before,after in MUTATIONS:
            def transform(source,before=before,after=after):
                if source.count(before)!=1:
                    raise ValueError('mutation anchor is not unique/present: '+repr(before))
                return source.replace(before,after,1)
            result=run_one(name,filename,transform,logs);results.append(result)
            print(name+': '+result['outcome'],flush=True)
    killed=sum(1 for r in results if r['outcome']=='KILLED')
    ending_sources={p:sha((HERE/p).read_bytes()) for p in source_paths}
    passed=starting_sources==ending_sources and baseline.returncode==0 and tests>0 and len(results)==len(MUTATIONS) and killed==len(MUTATIONS)
    report={'schema_version':1,'gate':'G0','observed_status':'PASS' if passed else 'FAIL',
        'started_utc':started,'ended_utc':datetime.now(timezone.utc).isoformat(),
        'command':[sys.executable,'-B','-S',str(HERE/'mutation_test.py'),'--output',str(args.output)],
        'baseline':{'command':command,'returncode':baseline.returncode,'tests_run':tests,'assertion_failures':failures,'errors':errors,'log':str(logs/'baseline.log'),'log_sha256':sha(baseline.stdout.encode())},
        'threshold':{'required_killed':len(MUTATIONS),'declared_mutants':len(MUTATIONS),'killed':killed,'ratio':{'numerator':killed,'denominator':len(MUTATIONS)},'rule':'every declared valid checker-source mutant must be killed by assertions; syntax/import/runtime errors do not count'},
        'mutants':results,'source_sha256':ending_sources,'source_unchanged_during_run':starting_sources==ending_sources,
        'claim':'finite behavioral selftests and actual checker-source mutations; not a proof of universal scanner correctness'}
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print('G0 '+report['observed_status']+': '+str(tests)+' baseline tests; '+str(killed)+'/'+str(len(MUTATIONS))+' checker-source mutants killed')
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
