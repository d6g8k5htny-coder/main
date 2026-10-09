#!/usr/bin/env python3
"""Retain sequential formal workflow observations; no scientific authority.

The producing steps must have stopped before capture. This is byte custody in
an owned job workspace, not a sandbox against a concurrently hostile writer.
Gate receipts are never edited. Only the optimized directory remains canonical.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys

from required_formal_check import context, require, strict_json


SOURCE = Path('formal/.lake/formal-evidence')
PHASES = Path('formal-evidence/phases')
NAMES = ('preexisting', 'initial', 'normal', 'optimized')
OUTCOMES = ('success', 'failure', 'cancelled', 'skipped')


def safe_path(path):
    """Reject linked ancestors before any traversal or directory creation."""
    require(not path.is_absolute() and '..' not in path.parts, 'unsafe path')
    for parent in reversed(path.parents):
        require(not parent.is_symlink(), 'symlinked parent: '+str(parent))
        require(not parent.exists() or parent.is_dir(), 'non-directory parent')
    require(not path.is_symlink(), 'symlinked path: '+str(path))


def inventory(folder):
    safe_path(folder)
    if not folder.exists():
        return {}
    require(folder.is_dir(), 'evidence root must be a directory')
    result = {}
    for path in sorted(folder.rglob('*')):
        safe_path(path)
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        require(stat.S_ISREG(info.st_mode), 'nonregular evidence: '+str(path))
        require(info.st_nlink == 1, 'hard-linked evidence: '+str(path))
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                value.update(chunk)
        result[path.relative_to(folder).as_posix()] = {
            'sha256': value.hexdigest(), 'size': info.st_size}
    return result


def capture(phase, outcome, binding_outcome=None):
    ctx = context(dict(os.environ))
    destination = PHASES / phase
    safe_path(destination)
    require(not destination.exists(), 'phase destination already exists: '+phase)
    # A skipped producer never owns whatever may remain at the canonical path.
    present = False
    files = {}
    if outcome != 'skipped':
        safe_path(SOURCE)
        files = inventory(SOURCE)
        present = SOURCE.exists()
    destination.mkdir(parents=True)
    if present:
        if phase == 'optimized':
            shutil.copytree(SOURCE, destination / 'raw')
        else:
            SOURCE.rename(destination / 'raw')
        require(inventory(destination / 'raw') == files, 'capture changed evidence bytes')
    record = {
        'schema_version': 1, 'scientific_effect': 'NONE',
        'scientific_status_authority': False, 'context': ctx,
        'phase': phase, 'outcome': outcome, 'binding_outcome': binding_outcome,
        'source_present': present, 'files': files,
    }
    with (destination / 'observation.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(record, sort_keys=True, indent=2)+'\n')
    # Preserve the observation even when the claimed success has no evidence.
    require(outcome != 'success' or present, 'successful phase produced no evidence')
    return record


def verify():
    ctx = context(dict(os.environ))
    safe_path(PHASES)
    require(PHASES.is_dir(), 'missing phase observations')
    require({p.name for p in PHASES.iterdir()} == set(NAMES), 'incomplete or extra phases')
    for phase in NAMES:
        folder = PHASES / phase
        safe_path(folder)
        record_path = folder / 'observation.json'
        safe_path(record_path)
        require(record_path.is_file() and record_path.stat().st_nlink == 1,
                'missing or aliased observation')
        record = strict_json(record_path.read_text(encoding='utf-8'))
        require(isinstance(record, dict), 'observation must be an object')
        require(record.get('schema_version') == 1 and record.get('phase') == phase,
                'observation identity mismatch')
        require(record.get('context') == ctx, 'observation execution context mismatch')
        require(record.get('scientific_effect') == 'NONE' and
                record.get('scientific_status_authority') is False, 'unexpected authority')
        allowed = ('preexisting',) if phase == 'preexisting' else OUTCOMES
        require(record.get('outcome') in allowed, 'invalid phase outcome')
        require(record.get('binding_outcome') in (None, *OUTCOMES), 'invalid binding outcome')
        present = record.get('source_present')
        require(type(present) is bool, 'invalid source presence')
        raw = folder / 'raw'
        files = inventory(raw)
        require(raw.exists() == present, 'source presence mismatch')
        require(record.get('files') == files, 'snapshot inventory mismatch: '+phase)
        require({p.name for p in folder.iterdir()} ==
                ({'observation.json', 'raw'} if present else {'observation.json'}),
                'unlisted observation material')
        require(record['outcome'] != 'skipped' or not present, 'skipped phase adopted evidence')
        require(record['outcome'] != 'success' or present, 'successful phase lacks evidence')
        if phase != 'optimized':
            require(record['binding_outcome'] is None, 'binding belongs to optimized phase')
        if record['binding_outcome'] == 'success':
            require(phase == 'optimized' and record['outcome'] == 'success',
                    'binding cannot succeed for an unsuccessful phase')
            binding = strict_json((raw / 'required-check-binding.json').read_text(encoding='utf-8'))
            require(isinstance(binding, dict) and all(binding.get(k) == v for k, v in ctx.items()),
                    'retained binding context mismatch')
            require(binding.get('receipt_sha256') ==
                    hashlib.sha256((raw / 'receipt.json').read_bytes()).hexdigest(),
                    'retained binding receipt mismatch')
    return {'custody': 'verified', 'phases': list(NAMES), 'context': ctx,
            'scientific_effect': 'NONE', 'scientific_status_authority': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    commands.add_parser('prepare')
    command = commands.add_parser('capture')
    command.add_argument('phase', choices=NAMES[1:])
    command.add_argument('outcome', choices=OUTCOMES)
    command.add_argument('--binding-outcome', choices=OUTCOMES)
    commands.add_parser('verify')
    args = parser.parse_args()
    try:
        if args.operation == 'prepare':
            result = capture('preexisting', 'preexisting')
        elif args.operation == 'capture':
            require(args.phase == 'optimized' or args.binding_outcome is None,
                    'binding outcome only applies to optimized phase')
            result = capture(args.phase, args.outcome, args.binding_outcome)
        else:
            result = verify()
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError) as exc:
        print('FORMAL_EVIDENCE_CUSTODY_FAILED: '+str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
