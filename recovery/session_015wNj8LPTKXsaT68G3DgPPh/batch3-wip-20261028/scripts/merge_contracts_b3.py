#!/usr/bin/env python3
"""Add the theorem-A planar and torus families to tools/check_mode_contracts.json.

Usage: merge_contracts_b3.py WORKTREE RUN1DIR RUN2DIR
Both characterization runs must agree byte for byte on every child. Exit status
must be 0 for each baseline and 1 for each mutant, and each baseline's stderr
must be empty. A JSON-verdict mutant must have empty stderr; a torus mutant
must have empty stdout and one "FAIL: ..." line on stderr, the same in both
modes. Theorem-A's two baselines take different arguments, so they are stored
per mode; so are the torus baselines, for one schema. The torus entry also binds
the git blobs of the three packet files its checker loads. The existing families
stay exactly as they are, and the current file must round-trip first.
"""
import hashlib
import json
import pathlib
import sys

FAMILIES = {
    'c8-theorem-a-constants-planar': ('frontiers/c8_theorem_a_constants_planar_20260930', 'theorem_a.py',
                                      ('no-delta', 'cap-half', 'no-zfloor-correction', 'no-om-term'), False, ()),
    'c8-torus-vanishing-remainder': ('frontiers/c8_torus_vanishing_remainder_20261001', 'theorem_v.py',
                                     ('torus-const-only', 'exp-radius', 'no-lin', 'no-om-term', 'coupling-half'), True,
                                     ('theorem_b.py', 'engine_e2.py', 'engine_e3.py')),
}
MODES = ('normal', 'optimized')


def dump(doc):
    return json.dumps(doc, sort_keys=True, separators=(',', ':')) + '\n'


def main():
    worktree, run1, run2 = map(pathlib.Path, sys.argv[1:4])
    path = worktree / 'tools/check_mode_contracts.json'
    before = path.read_text()
    doc = json.loads(before)
    assert dump(doc) == before, 'existing contracts do not round-trip'
    old_families = json.loads(json.dumps(doc['families']))
    for family, (directory, script, labels, stderr_rejections, loads) in FAMILIES.items():
        assert family not in doc['families'], family
        rows = {}
        for run in (run1, run2):
            records = json.loads((run / (family + '.json')).read_text())
            rows[run] = {(r['mode'], r['label']): r for r in records}
            assert len(rows[run]) == len(records), (run, family)
        assert set(rows[run1]) == set(rows[run2]), family
        keys = sorted(rows[run1], key=lambda k: (k[0], k[1] or ''))
        for key in keys:
            a, b = rows[run1][key], rows[run2][key]
            for field in ('args', 'returncode', 'stdout', 'stderr'):
                assert a[field] == b[field], (family, key, field)
            mode, label = key
            if label is None:
                assert a['returncode'] == 0 and a['stderr'] == '', (family, key)
            elif stderr_rejections:
                assert a['returncode'] == 1 and a['stdout'] == '', (family, key)
                assert a['stderr'].startswith('FAIL: ') and a['stderr'].endswith('\n') and a['stderr'].count('\n') == 1
            else:
                assert a['returncode'] == 1 and a['stderr'] == '', (family, key)
        baselines = {mode: json.loads(rows[run1][(mode, None)]['stdout']) for mode in MODES}
        mutants = {}
        for label in labels:
            seen = {rows[run1][key]['stderr' if stderr_rejections else 'stdout'] for key in keys if key[1] == label}
            assert len(seen) == 1, (family, label, seen)
            text = seen.pop()
            mutants[label] = text if stderr_rejections else json.loads(text)
        packet = worktree / directory
        data = (packet / script).read_bytes()
        results = (packet / 'RESULTS.json').read_bytes()
        doc['families'][family] = {
            'directory': directory, 'script': script,
            'script_git_blob': hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest(),
            'results_bytes': len(results), 'results_sha256': hashlib.sha256(results).hexdigest(),
            'baselines': baselines, 'mutants': mutants}
        if loads:
            blobs = {}
            for name in loads:
                content = (packet / name).read_bytes()
                blobs[name] = hashlib.sha1(b'blob %d\0' % len(content) + content).hexdigest()
            doc['families'][family]['loaded_git_blobs'] = blobs
    for family, entry in old_families.items():
        assert doc['families'][family] == entry, family
    path.write_text(dump(doc))
    canonical = json.dumps(doc['families'], sort_keys=True, separators=(',', ':')).encode()
    print('families_digest', hashlib.sha256(canonical).hexdigest())
    print('file', len(path.read_bytes()), hashlib.sha256(path.read_bytes()).hexdigest())
    for family in FAMILIES:
        print(family, json.dumps(doc['families'][family], sort_keys=True)[:1200])


if __name__ == '__main__':
    main()
