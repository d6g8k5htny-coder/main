#!/usr/bin/env python3
"""Bind an unchanged formal receipt to separate source and execution repositories."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_sha(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{40}', value), 'invalid exact commit')
    return value


def bind(receipt_path: Path, *, source_repository: str, expected_commit: str,
         observed_commit: str, observed_remote: str, execution_repository: str,
         execution_commit: str, run_id: str) -> dict:
    require(source_repository == 'd6g8k5htny-coder/Math-', 'unexpected primary source repository')
    require(execution_repository == 'd6g8k5htny-coder/main', 'unexpected execution repository')
    require(observed_remote in ('https://github.com/'+source_repository,
                                'https://github.com/'+source_repository+'.git'), 'source remote mismatch')
    require(exact_sha(expected_commit) == exact_sha(observed_commit), 'source commit mismatch')
    exact_sha(execution_commit)
    require(isinstance(run_id, str) and re.fullmatch(r'[0-9]+',run_id), 'invalid workflow run id')
    raw=receipt_path.read_bytes(); receipt=json.loads(raw)
    require(receipt.get('checked_commit') == observed_commit, 'receipt/source mismatch')
    require(receipt.get('repository') == execution_repository, 'receipt execution host mismatch')
    require(receipt.get('workflow_run_id') == run_id, 'receipt workflow mismatch')
    require(receipt.get('scientific_effect') == 'NONE', 'unexpected scientific effect')
    require(receipt.get('formalization_status') == 'kernel-checked', 'successful kernel receipt required')
    logs=receipt.get('logs')
    require(isinstance(logs,dict) and logs, 'missing log bindings')
    for name,digest in logs.items():
        require(isinstance(name,str) and re.fullmatch(r'[A-Za-z0-9_-]+\.log',name), 'unsafe log path')
        require(isinstance(digest,str) and re.fullmatch(r'[0-9a-f]{64}',digest), 'invalid log digest')
        path=receipt_path.parent/name
        require(not path.is_symlink(), 'symlinked log')
        require(hashlib.sha256(path.read_bytes()).hexdigest()==digest, 'log digest mismatch')
    return {'schema_version':1, 'scientific_effect':'NONE',
            'source':{'repository':source_repository,'commit':observed_commit,'remote':observed_remote},
            'execution':{'repository':execution_repository,'commit':execution_commit,'workflow_run_id':run_id},
            'receipt_sha256':hashlib.sha256(raw).hexdigest(),
            'receipt_repository_field_means':'execution repository, not source repository',
            'meaning':'Source/execution custody binding only; no independent alignment or scientific acceptance.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--operations',type=Path,required=True)
    p.add_argument('--expected-commit',required=True)
    p.add_argument('--run-id',required=True)
    a=p.parse_args()
    def git(root,*args):
        return subprocess.check_output(['git','-C',str(root),*args],text=True,timeout=30).strip()
    receipt=a.source/'formal/.lake/formal-evidence/receipt.json'
    result=bind(receipt,source_repository='d6g8k5htny-coder/Math-',expected_commit=a.expected_commit,
                observed_commit=git(a.source,'rev-parse','HEAD'),observed_remote=git(a.source,'remote','get-url','origin'),
                execution_repository='d6g8k5htny-coder/main',execution_commit=git(a.operations,'rev-parse','HEAD'),run_id=a.run_id)
    with (receipt.parent/'execution-context.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
