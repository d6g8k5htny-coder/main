#!/usr/bin/env python3
"""Exercise the inventory self-mutation driver against real false kills.

The driver must refuse both a no-op mutant and a subject-origin assertion.
This checks both parent interpreter modes; each refuses on its first normal
child mode. The separately qualified node checker owns traceback provenance.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = 'cbz/evidence/inventory-origin-20261109'
DRIVER = ROOT / EVIDENCE / 'run_self_mutations.py'
CHECKER = ROOT / 'cbz/evidence/node-status-freshness/run_mutations.py'
MARKER = 'self-mutation not killed by selected oracle: '


def main():
    driver = DRIVER.read_text(encoding='utf-8')
    helper = (ROOT / 'tests/cbz_inventory_origin.py').read_bytes()
    controls = (ROOT / 'tests/test_cbz_inventory_origin.py').read_bytes()
    rows = []
    with tempfile.TemporaryDirectory(prefix='cbz-inventory-driver-adverse-') as tmp:
        root = Path(tmp)
        (root / EVIDENCE).mkdir(parents=True)
        (root / 'cbz/evidence/node-status-freshness').mkdir(parents=True)
        (root / 'tests').mkdir()
        (root / 'tests/cbz_inventory_origin.py').write_bytes(helper)
        (root / 'tests/test_cbz_inventory_origin.py').write_bytes(controls)
        (root / 'cbz/evidence/node-status-freshness/run_mutations.py').write_bytes(CHECKER.read_bytes())
        target = root / EVIDENCE / DRIVER.name
        anchor = 'changed = changed.replace(before, after)'
        write = "candidate.write_text(changed, encoding='utf-8')"
        if driver.count(anchor) != 1 or driver.count(write) != 1:
            raise RuntimeError('driver adverse anchors must be unique')
        definition = 'def classify(report, expected_ids, target, allowed_skips):\n'
        injection = definition + '    raise AssertionError("driver subject-origin probe")\n'
        subject_write = ('candidate.write_text(changed.replace(' + repr(definition) + ', ' +
                         repr(injection) + ", 1), encoding='utf-8')")
        variants = {'no_op': driver.replace(anchor, 'changed = changed'),
                    'subject_assertion': driver.replace(write, subject_write)}
        env = {key: value for key, value in os.environ.items() if not key.startswith('PYTHON')}
        for optimized in (False, True):
            for name, source in variants.items():
                target.write_text(source, encoding='utf-8')
                command = [sys.executable, '-B', *(['-O'] if optimized else []), '-S', str(target)]
                child = subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       capture_output=True, text=True, timeout=90)
                if child.returncode != 1 or MARKER not in child.stderr:
                    raise RuntimeError('adverse driver was not refused: ' + child.stderr)
                payload = json.loads(child.stderr.rsplit(MARKER, 1)[1].splitlines()[0])
                mutant = payload['mutant']
                if payload['killed'] is not False or payload['baseline']['child_exit'] != 0:
                    raise RuntimeError('invalid baseline or adverse classification')
                if name == 'no_op':
                    if mutant['child_exit'] != 0 or mutant['events']:
                        raise RuntimeError('no-op did not produce the expected passing child')
                else:
                    if (mutant['child_exit'] != 1 or not mutant['events'] or
                            any(event.get('authorized') is not False for event in mutant['events'])):
                        raise RuntimeError('subject assertion did not have refused provenance')
                rows.append({'name': name, 'parent_optimized': optimized, 'exit': child.returncode,
                             'payload': payload, 'stdout': child.stdout, 'stderr': child.stderr})
    print(json.dumps({'schema': 'cbz-inventory-driver-adverse-v1',
                      'cases': len(rows), 'results': rows}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
