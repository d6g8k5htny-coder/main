#!/usr/bin/env python3
"""Add the lifetime-difference and d = 3 Theorem A families to tools/check_mode_contracts.json.

Usage: merge_contracts_b4.py WORKTREE RUN1DIR RUN2DIR

Both characterization runs must agree byte for byte on every child's exit status,
stdout and stderr, and each child must have run with the wrapper's arguments:
normal --check-full, optimized --check, then each mutant in normal mode, all with
--procs max(2, cpu_count) as recorded with the child. Exit status must be 0 for
each baseline (empty stderr) and 1 for each mutant. A d3 verdict is one JSON
line on stdout. A lifetime baseline is the plain line "CHECK PASSED (full)" in
normal mode and "CHECK PASSED (sample)" in optimized mode, stored as text; a
lifetime mutant has empty stdout and one "FAIL: ..." line on stderr. The
lifetime entry also binds the git blobs of the two engine files its checker
loads. The existing families stay exactly as they are, the current file must
round-trip first, and the custody digest in tests/test_check_mode_workflows.py
moves to the new families digest.
"""
import hashlib
import json
import pathlib
import re
import sys

FAMILIES = {
    'c8-lifetime-difference-constant': ('frontiers/c8_lifetime_difference_constant_20261001', 'theorem_b.py',
                                        ('no-om-term', 'no-bad-set', 'no-det', 'no-torus', 'cap-half'), True, True,
                                        ('engine_e2.py', 'engine_e3.py')),
    'c8-theorem-a-constants-d3': ('frontiers/c8_theorem_a_constants_d3_20261001', 'theorem_a3.py',
                                  ('no-delta', 'cap-half', 'no-zfloor-correction', 'no-om-term', 'no-lambda2'),
                                  False, False, ()),
}
MODES = ('normal', 'optimized')


def dump(doc):
    return json.dumps(doc, sort_keys=True, separators=(',', ':')) + '\n'


def git_blob(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def main():
    worktree, run1, run2 = map(pathlib.Path, sys.argv[1:4])
    path = worktree / 'tools/check_mode_contracts.json'
    before = path.read_text()
    doc = json.loads(before)
    assert dump(doc) == before, 'existing contracts do not round-trip'
    old_families = json.loads(json.dumps(doc['families']))
    for family, (directory, script, labels, stderr_rejections, text_baseline, loads) in FAMILIES.items():
        assert family not in doc['families'], family
        rows = {}
        for run in (run1, run2):
            records = json.loads((run / (family + '.json')).read_text())
            rows[run] = {(r['mode'], r['label']): r for r in records}
            assert len(rows[run]) == len(records) == 2 + len(labels), (run, family)
        assert set(rows[run1]) == set(rows[run2]), family
        keys = sorted(rows[run1], key=lambda k: (k[0], k[1] or ''))
        for key in keys:
            a, b = rows[run1][key], rows[run2][key]
            for field in ('args', 'returncode', 'stdout', 'stderr', 'procs'):
                assert a[field] == b[field], (family, key, field)
            mode, label = key
            procs = a['procs']
            assert procs == str(max(2, a['cpu_count'] or 2)) and int(procs) >= 2, (family, key, procs)
            if label is None:
                assert a['args'] == ['--check-full' if mode == 'normal' else '--check', '--procs', procs], (family, key)
                assert a['returncode'] == 0 and a['stderr'] == '', (family, key)
                if text_baseline:
                    assert a['stdout'] == 'CHECK PASSED (%s)\n' % ('full' if mode == 'normal' else 'sample'), (family, key)
            else:
                assert mode == 'normal' and a['args'] == ['--check', '--procs', procs, '--mutant', label], (family, key)
                if stderr_rejections:
                    assert a['returncode'] == 1 and a['stdout'] == '', (family, key)
                    assert (a['stderr'].startswith('FAIL: ') and a['stderr'].endswith('\n')
                            and a['stderr'].count('\n') == 1), (family, key)
                else:
                    assert a['returncode'] == 1 and a['stderr'] == '', (family, key)
        baselines = {}
        for mode in MODES:
            text = rows[run1][(mode, None)]['stdout']
            baselines[mode] = text if text_baseline else json.loads(text)
        mutants = {}
        for label in labels:
            text = rows[run1][('normal', label)]['stderr' if stderr_rejections else 'stdout']
            mutants[label] = text if stderr_rejections else json.loads(text)
        packet = worktree / directory
        data = (packet / script).read_bytes()
        results = (packet / 'RESULTS.json').read_bytes()
        entry = {'directory': directory, 'script': script, 'script_git_blob': git_blob(data),
                 'results_bytes': len(results), 'results_sha256': hashlib.sha256(results).hexdigest(),
                 'baselines': baselines, 'mutants': mutants}
        if loads:
            entry['loaded_git_blobs'] = {name: git_blob((packet / name).read_bytes()) for name in loads}
        doc['families'][family] = entry
    for family, entry in old_families.items():
        assert doc['families'][family] == entry, family
    path.write_text(dump(doc))
    canonical = json.dumps(doc['families'], sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    test = worktree / 'tests/test_check_mode_workflows.py'
    text = test.read_text()
    found = re.search(r"self\.assertEqual\(digest\(canonical\), '([0-9a-f]{64})'\)", text)
    assert found and text.count(found.group(1)) == 1, 'custody digest line not found once'
    test.write_text(text.replace(found.group(1), digest))
    print('families_digest', digest, '(was', found.group(1) + ')')
    print('file', len(path.read_bytes()), hashlib.sha256(path.read_bytes()).hexdigest())
    for family in FAMILIES:
        print(family, json.dumps(doc['families'][family], sort_keys=True)[:1500])


if __name__ == '__main__':
    main()
