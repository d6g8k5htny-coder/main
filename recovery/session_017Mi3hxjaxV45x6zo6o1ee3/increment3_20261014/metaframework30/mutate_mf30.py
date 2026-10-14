"""Mutation controls for meta-framework#30 (reviewer-private until delivery).

usage: python3 -I mutate_mf30.py WORKTREE PYTHON LOG
Each mutant edits tools/contract_validation.py (exactly one match required),
runs tests/test_source_contracts.py with the workflow flags, then restores the
file byte-for-byte and re-checks its blob.
"""
import hashlib
import subprocess
import sys

WT, PY, LOG = sys.argv[1], sys.argv[2], sys.argv[3]
PATH = WT + '/tools/contract_validation.py'
FIX = " or '\\x00' in text: raise ContractError('unsafe path')"
MUTANTS = [
    ('N1 fix removed (base behaviour)', FIX, ": raise ContractError('unsafe path')"),
    ('N2 only trailing NUL refused', FIX, " or text.endswith('\\x00'): raise ContractError('unsafe path')"),
    ('N3 only leading NUL refused', FIX, " or text.startswith('\\x00'): raise ContractError('unsafe path')"),
    ('N4 NUL refused with a different message', FIX, ": raise ContractError('unsafe path')\n    if '\\x00' in text: raise ContractError('unsafe path: NUL')"),
    ('N5 NUL silently stripped instead of refused', FIX, ": raise ContractError('unsafe path')\n    text=text.replace('\\x00','')"),
    ('N6 refusal limited to source refs (manifest path unguarded)', FIX, ": raise ContractError('unsafe path')"),
    ('N7 validate_source_ref returns the caller object', "def validate_source_ref(data: dict, *, public: bool=True) -> dict:\n    row=_copy(data)", "def validate_source_ref(data: dict, *, public: bool=True) -> dict:\n    row=data if isinstance(data,dict) else _copy(data)"),
    ('N8 validate_source_manifest mutates its input', "def validate_source_manifest(data: dict) -> dict:\n    row=_copy(data)", "def validate_source_manifest(data: dict) -> dict:\n    data.setdefault('checked',True) if isinstance(data,dict) else None\n    row=_copy(data)"),
    ('N9 NUL refused as a ValueError, not a ContractError', FIX, ": raise ContractError('unsafe path')\n    if '\\x00' in text: raise ValueError('unsafe path')"),
]
# N6 additionally guards git refs only, at the call site in validate_source_ref.
EXTRA = {'N6 refusal limited to source refs (manifest path unguarded)':
         ("        _safe_path(row.get('path'))\n",
          "        _safe_path(row.get('path'))\n        if '\\x00' in row.get('path'): raise ContractError('unsafe path')\n")}


def blob(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


original = open(PATH, 'rb').read()
text = original.decode()
out = ['original blob %s' % blob(original)]
for name, old, new in MUTANTS:
    edits = [(old, new)] + ([EXTRA[name]] if name in EXTRA else [])
    mutated = text
    bad = [o for o, _ in edits if mutated.count(o) != 1]
    if bad:
        out.append('%s: SETUP ERROR %r' % (name, bad))
        continue
    for o, n in edits:
        mutated = mutated.replace(o, n)
    open(PATH, 'w').write(mutated)
    try:
        p = subprocess.run([PY, '-B', '-S', '-m', 'unittest', 'tests.test_source_contracts', '-v'], cwd=WT,
                           capture_output=True, text=True, timeout=300)
        err = p.stderr
        tail = [l for l in err.splitlines() if l.startswith(('FAILED', 'OK', 'Ran '))]
        failing = sorted({l.split(' (')[0].replace('FAIL: ', '').replace('ERROR: ', '')
                          for l in err.splitlines() if l.startswith(('FAIL: ', 'ERROR: '))})
        out.append('%s: %s rc=%d %s | %s' % (name, 'CAUGHT' if p.returncode else 'SURVIVED', p.returncode,
                                            ' '.join(tail), ', '.join(failing)))
    finally:
        open(PATH, 'wb').write(original)
restored = open(PATH, 'rb').read()
out.append('restored blob %s %s' % (blob(restored), 'OK' if restored == original else 'MISMATCH'))
open(LOG, 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
