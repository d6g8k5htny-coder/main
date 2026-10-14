"""Differential control for meta-framework#30: base vs head contract_validation.

usage: python3 -I diff_mf30.py BASE_FILE HEAD_FILE
Claim checked: the only behaviour change is that path strings containing NUL
are refused with exactly ContractError('unsafe path'); every other input keeps
its exact outcome (same exception type and message, or an equal returned row).
"""
import importlib.util
import itertools
import json
import random
import sys


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = load('cv_base', sys.argv[1])
H = load('cv_head', sys.argv[2])

TOKENS = ['a', 'b', 'docs', 'x.md', '.', '..', '/', '//', '\\', ':', '\x00', '\x01', '\n', '\t', '\r', ' ',
          '\x7f', 'α', ' ', '／', '∕', 'sandbox', 'Sandbox', 'SANDBOX', 'ſandbox', '', '~', '%00', '\\0']


def outcome(fn, value):
    try:
        res = fn(value)
        return ('ok', json.dumps(res, sort_keys=True, ensure_ascii=True))
    except Exception as e:  # noqa: BLE001 - outcome comparison
        return ('err', type(e).__name__, str(e))


def ref(path):
    return {'schema_version': '1.0', 'kind': 'git', 'repository': 'd6g8k5htny-coder/Math-', 'commit': '1' * 40,
            'path': path, 'git_blob_sha1': '2' * 40, 'sha256': '3' * 64, 'size_bytes': 1,
            'role': 'source', 'visibility': 'public'}


def manifest(path):
    return {'schema_version': '1.0', 'distribution': 'synthetic', 'version': '0.1.0',
            'repository': 'd6g8k5htny-coder/Math-', 'commit': '1' * 40, 'scientific_status_authority': False,
            'files': [{'path': path, 'sha256': '2' * 64, 'bytes': 0}]}


def corpus():
    yield from (None, 1, 1.5, b'docs', ['docs'], {'p': 1}, True)
    for n in range(1, 4):
        for combo in itertools.product(TOKENS, repeat=n):
            yield ''.join(combo)
    rnd = random.Random(20261014)
    for _ in range(60000):
        yield ''.join(rnd.choice(TOKENS) for _ in range(rnd.randint(1, 7)))


stats = {'inputs': 0, 'nul': 0, 'nonnul_same': 0, 'accepted_nonnul': 0, 'refused_nonnul': 0}
for value in corpus():
    stats['inputs'] += 1
    for fname, wrap in (('_safe_path', None), ('validate_source_ref', ref), ('validate_source_manifest', manifest)):
        if wrap is None:
            ob, oh = outcome(B._safe_path, value), outcome(H._safe_path, value)
        else:
            row_b, row_h = wrap(value), wrap(value)
            snap = repr(row_h)
            ob = outcome(getattr(B, fname), row_b)
            oh = outcome(getattr(H, fname), row_h)
            assert repr(row_h) == snap, ('input mutated', fname, repr(value))
        if isinstance(value, str) and '\x00' in value:
            assert oh == ('err', 'ContractError', 'unsafe path'), ('NUL not refused exactly', fname, repr(value), oh)
        else:
            assert ob == oh, ('behaviour changed for non-NUL input', fname, repr(value), ob, oh)
    if isinstance(value, str) and '\x00' in value:
        stats['nul'] += 1
    else:
        stats['nonnul_same'] += 1
        if outcome(H._safe_path, value)[0] == 'ok':
            stats['accepted_nonnul'] += 1
        else:
            stats['refused_nonnul'] += 1
# where did base accept NUL paths? (the defect being fixed)
base_accepted_nul = sum(1 for v in ('\x00docs/proof.md', 'docs/\x00proof.md', 'docs/proof.md\x00', '\x00')
                        if outcome(B._safe_path, v)[0] == 'ok')
print(json.dumps({**stats, 'base_accepts_of_4_canonical_nul_paths': base_accepted_nul, 'python': sys.version.split()[0]}))
print('DIFFERENTIAL OK')
