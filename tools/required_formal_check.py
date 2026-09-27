#!/usr/bin/env python3
"""Bind current-run formal evidence to an existing required check; no scientific authority."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPOSITORIES = {'d6g8k5htny-coder/main', 'd6g8k5htny-coder/Math-'}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact(value, pattern: str, name: str) -> str:
    require(isinstance(value, str) and re.fullmatch(pattern, value) is not None,
            'invalid '+name)
    return value


def strict_json(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON: '+value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def context(env: dict) -> dict:
    repository = env.get('GITHUB_REPOSITORY')
    require(repository in REPOSITORIES, 'unexpected execution repository')
    return {
        'checked_commit': exact(env.get('GITHUB_SHA'), r'[0-9a-f]{40}', 'commit'),
        'repository': repository,
        'run_id': exact(env.get('GITHUB_RUN_ID'), r'[1-9][0-9]*', 'run ID'),
        'run_attempt': exact(env.get('GITHUB_RUN_ATTEMPT'), r'[1-9][0-9]*', 'attempt'),
    }


def bind_receipt(receipt: Path, manifest: Path, env: dict, git_head: str) -> dict:
    ctx = context(env)
    require(git_head == ctx['checked_commit'], 'checkout/tested commit mismatch')
    require(not receipt.is_symlink() and not manifest.is_symlink(), 'symlinked evidence')
    raw = receipt.read_bytes()
    record = strict_json(raw.decode('utf-8'))
    require(isinstance(record, dict), 'receipt must be an object')
    for key, expected in [('checked_commit', ctx['checked_commit']),
                          ('repository', ctx['repository']),
                          ('workflow_run_id', ctx['run_id']),
                          ('formalization_status', 'kernel-checked'),
                          ('scientific_effect', 'NONE')]:
        require(record.get(key) == expected, 'receipt mismatch: '+key)
    require(record.get('manifest_sha256') == hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'manifest/receipt mismatch')
    logs = record.get('logs')
    require(isinstance(logs, dict) and bool(logs), 'missing receipt log bindings')
    for name, digest in logs.items():
        exact(name, r'[A-Za-z0-9_-]+\.log', 'log name')
        exact(digest, r'[0-9a-f]{64}', 'log digest')
        log = receipt.parent/name
        require(not log.is_symlink(), 'symlinked log')
        require(hashlib.sha256(log.read_bytes()).hexdigest() == digest, 'log hash mismatch: '+name)
    return {**ctx, 'receipt_sha256': hashlib.sha256(raw).hexdigest()}


def aggregate(needs, env: dict) -> dict:
    ctx = context(env)
    require(isinstance(needs, dict) and set(needs) == {'checks', 'formal'},
            'exact checks/formal dependencies required')
    for name in ('checks', 'formal'):
        job = needs[name]
        require(isinstance(job, dict) and job.get('result') == 'success',
                'dependency did not succeed: '+name)
        require(isinstance(job.get('outputs'), dict), 'missing dependency outputs: '+name)
        require(job['outputs'].get('checked_commit') == ctx['checked_commit'],
                'dependency commit mismatch: '+name)
    outputs = needs['formal']['outputs']
    for key, value in ctx.items():
        require(outputs.get(key) == value, 'formal evidence mismatch: '+key)
    digest = exact(outputs.get('receipt_sha256'), r'[0-9a-f]{64}', 'receipt digest')
    return {**ctx, 'receipt_sha256': digest, 'conclusion': 'success',
            'scientific_effect': 'NONE',
            'meaning': 'Required engineering checks and current-run formal evidence only; not statement alignment or scientific acceptance.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['receipt', 'aggregate'])
    args = parser.parse_args()
    try:
        env = dict(os.environ)
        if args.operation == 'receipt':
            folder = Path('formal/.lake/formal-evidence')
            head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, timeout=30).strip()
            result = bind_receipt(folder/'receipt.json', Path('formal/manifest.json'), env, head)
            # Never rewrite the original gate receipt.
            (folder/'required-check-binding.json').write_text(json.dumps(result, indent=2)+'\n')
            with open(env['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
                for key, value in result.items():
                    output.write(f'{key}={value}\n')
        else:
            result = aggregate(strict_json(env['REQUIRED_FORMAL_NEEDS']), env)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        print('REQUIRED_FORMAL_CHECK_FAILED: '+str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
