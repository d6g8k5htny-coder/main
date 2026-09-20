#!/usr/bin/env python3
"""Replay three portable author-side mathematical candidates; no promotion."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.rn.certificate import canonical_bytes
from tools.rn_side24_check import load_report


def h3_payload_bytes(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def h3_mathematics(envelope):
    if type(envelope) is not dict or set(envelope) != {'payload', 'payload_sha256'}:
        raise ValueError('invalid H3 report envelope')
    payload = envelope['payload']
    if type(payload) is not dict or hashlib.sha256(h3_payload_bytes(payload)).hexdigest() != envelope['payload_sha256']:
        raise ValueError('H3 payload digest mismatch')
    # Original external evidence records host path and authoring-code identity.
    # Portable replay independently snapshots its actual imports. Exclude only
    # these audit identities from cross-host mathematical equality, never source
    # custody, intervals, assumptions, checks, negative controls or status flags.
    audit = {'repository_root', 'computational_sources', 'checker_sha256'}
    if not audit <= payload.keys():
        raise ValueError('H3 audit identities missing')
    return {key: value for key, value in payload.items() if key not in audit}


def compare_h3(actual, expected):
    if canonical_bytes(h3_mathematics(actual)) != canonical_bytes(h3_mathematics(expected)):
        raise ValueError('H3 mathematical reconstruction mismatch')


def replay(name, output):
    base = ROOT / 'research/parallel' / name
    if name == 'h3':
        args = [str(base/'verify_h3_floor.py'), '--repo', str(ROOT), '--output', str(output)]
    elif name == 'lpw':
        args = [str(base/'lpw_modulus.py'), '--repo', str(ROOT), '--certificate', str(base/'candidate.json')]
    elif name == 'c2':
        args = [str(base/'c2_band.py'), '--verify', str(base/'candidate.json')]
    else:
        raise ValueError('unknown mathematical target')
    result = subprocess.run([sys.executable, '-B', *args], cwd=ROOT,
                            capture_output=True, text=True, timeout=240)
    if result.returncode:
        raise ValueError(name+' replay failed: '+result.stdout[-3000:]+result.stderr[-3000:])
    if name == 'h3':
        compare_h3(load_report(output), load_report(base/'candidate.json'))
    return name+' PASS'


def main():
    try:
        with tempfile.TemporaryDirectory(prefix='parallel-math-') as directory:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                tasks = [pool.submit(replay, name, Path(directory)/(name+'.json'))
                         for name in ('h3', 'lpw', 'c2')]
                results = [task.result() for task in tasks]
        print('; '.join(results)+'; author candidates, fixed declared scopes, independence=0, no scientific promotion')
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print('Parallel mathematics: REJECTED: '+str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
