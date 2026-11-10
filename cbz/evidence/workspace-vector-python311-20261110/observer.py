"""Read-only CPython 3.11 traceback-position observation for Workspace #78."""
import dis
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

EXPECTED_COMMIT = '0e4162c8f0fc9c61d13ed75c3ef91407c22c236d'
EXPECTED = {
    'tests/test_verify_federation_vector.py': 'b536cf8398d03bbe607d56e88ccd111a14c99c2c',
    'tests/test_verify_federation_vector_mutations.py': 'cc8f57010c218859ede6eb8925245b9e3c96420a',
    'tools/verify_federation_vector.py': 'bb7b974ce09481e861baf95fefce85ba5b20d394',
    'tools/check_workspace.py': '52619edbc3612ae4fadb225639c2c102d6a0facb',
    'tools/verify_pins.py': 'b05916e54c34cffc95601b313bc1c154d99ccbcc',
    'tools/verify_canonical_inputs.py': '1718c34179afb5c89526b5002bb59b94eb0b945e',
}
source_root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
if output.exists():
    raise ValueError('refusing to overwrite diagnostic output')
output.mkdir(parents=True)
if subprocess.check_output(['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip() != EXPECTED_COMMIT:
    raise ValueError('candidate commit mismatch')
for name, expected in EXPECTED.items():
    p = source_root / name
    actual = subprocess.check_output(['git', 'hash-object', str(p)], text=True).strip()
    if actual != expected:
        raise ValueError('candidate blob mismatch: ' + name)
    target = output / 'copy' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, target)

harness = output / 'copy/tests/test_verify_federation_vector_mutations.py'
code = harness.read_text()
needle = "        def addFailure(self, case, error):\n            record['failures'].append(dict(selected_body=failure_origin(error, selected), message=str(error[1])))\n"
instrumentation = '''        def addFailure(self, case, error):
            import dis
            trace = error[2]
            frames = []
            while trace:
                code = trace.tb_frame.f_code
                ins = next((i for i in dis.get_instructions(code, show_caches=True) if i.offset == trace.tb_lasti), None)
                pos = ins.positions if ins else None
                frames.append({'name': code.co_qualname, 'lasti': trace.tb_lasti,
                    'opname': ins.opname if ins else None,
                    'position': [pos.lineno, pos.end_lineno, pos.col_offset, pos.end_col_offset] if pos else None})
                trace = trace.tb_next
            record['trace_diagnostic'] = frames
            record['failures'].append(dict(selected_body=failure_origin(error, selected), message=str(error[1])))
'''
if code.count(needle) != 1:
    raise ValueError('diagnostic insertion anchor mismatch')
harness.write_text(code.replace(needle, instrumentation, 1))
original = (output / 'copy/tools/verify_federation_vector.py').read_text()
anchor = 'set(values) == set(REPOSITORIES)'
if original.count(anchor) != 1:
    raise ValueError('mutation anchor mismatch')
mutant = output / 'vector-set.py'
mutant.write_text(original.replace(anchor, 'True', 1))
selected = 'VectorSources.test_vector_omission_rejected'
results = []
for mode in ('normal', 'optimized'):
    event_path = output / (mode + '-event.json')
    env = dict(os.environ, CBZ_VECTOR_UNDER_TEST=str(mutant), CBZ_VECTOR_EVENT=str(event_path),
               CBZ_VECTOR_NONCE=secrets.token_hex(16))
    child = subprocess.run([sys.executable, '-B', *(['-O'] if mode == 'optimized' else []),
                            '-S', str(harness), '--child', selected],
                           env=env, text=True, capture_output=True, timeout=180)
    (output / (mode + '.stdout')).write_text(child.stdout)
    (output / (mode + '.stderr')).write_text(child.stderr)
    event = json.loads(event_path.read_text())
    if child.returncode != 1 or event['tests_run'] != 1 or event['errors'] or event['skipped']:
        raise ValueError('diagnostic child did not reproduce the intended failure')
    results.append({'mode': mode, 'returncode': child.returncode,
                    'failure': event['failures'], 'frames': event['trace_diagnostic']})
summary = {'python': sys.version, 'source_commit': EXPECTED_COMMIT, 'blobs': EXPECTED,
           'scope': 'instrumented diagnostic copy; no independent scientific claim', 'results': results}
(output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
