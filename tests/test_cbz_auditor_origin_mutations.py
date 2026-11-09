"""Adverse source mutations of the auditor-origin result adapter."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import unittest

from tests import cbz_inventory_origin as recorder

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tests' / 'cbz_auditor_origin.py'
CHILD = ROOT / 'tests' / 'cbz_auditor_origin_mutation_child.py'
PREFIX = 'tests.test_cbz_auditor_origin.AuditorOriginControls.'
MUTANTS = (
    ('accept any outcome', "return outcome(report, expected_test) == 'kill'", 'return True',
     'test_baselines_and_actual_selected_body_oracles'),
    ('source drift accepted', 'or before != after):', 'or False):',
     'test_missing_or_inconsistent_report_provenance_is_refused'),
    ('nonhex source digest accepted', "any(c not in '0123456789abcdef' for c in before)", 'False',
     'test_missing_or_inconsistent_report_provenance_is_refused'),
    ('wrong digest length accepted', 'len(before) != 64', 'False',
     'test_missing_or_inconsistent_report_provenance_is_refused'),
    ('inconsistent classification accepted', "report.get('classification') == actual", 'True',
     'test_missing_or_inconsistent_report_provenance_is_refused'),
)


def run(path, target, label):
    before = Path(path).read_bytes()
    env = dict(os.environ)
    env.pop('CBZ_AUDITOR_EVIDENCE_DIR', None)
    command = [sys.executable, '-B', *(['-O'] if sys.flags.optimize else []), '-S',
               str(CHILD), str(path), target]
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=180)
    after = Path(path).read_bytes()
    try:
        report = json.loads(process.stdout)
        report.update(protocol_error=False, child_exit=process.returncode, selected_id=target)
        classification = recorder.classify(report, [target], target, {})
    except (ValueError, TypeError, AttributeError):
        report, classification = {}, 'invalid'
    if before != after:
        classification = 'invalid'
    directory = os.environ.get('CBZ_AUDITOR_ORIGIN_MUTATION_EVIDENCE_DIR')
    if directory:
        nonce = secrets.token_hex(16)
        dest = Path(directory)
        for suffix, data in (('.source.py', before), ('.stdout', process.stdout),
                             ('.stderr', process.stderr)):
            with (dest / (nonce + suffix)).open('xb') as stream:
                stream.write(data)
        entry = {'nonce': nonce, 'label': label, 'target': target,
                 'optimize': sys.flags.optimize, 'command': command,
                 'exit_code': process.returncode, 'classification': classification,
                 'source_before': hashlib.sha256(before).hexdigest(),
                 'source_after': hashlib.sha256(after).hexdigest(),
                 'stdout_sha256': hashlib.sha256(process.stdout).hexdigest(),
                 'stderr_sha256': hashlib.sha256(process.stderr).hexdigest()}
        with (dest / (nonce + '.json')).open('x', encoding='utf-8') as stream:
            json.dump(entry, stream, indent=2, sort_keys=True)
            stream.write('\n')
    return classification, report, process.stderr.decode(errors='replace')


class AuditorOriginMutations(unittest.TestCase):
    def test_actual_mutants_have_clean_baselines_and_authorized_failures(self):
        source = SOURCE.read_text()
        with tempfile.TemporaryDirectory() as directory:
            subject = Path(directory) / 'origin_adapter.py'
            for name, old, new, test in MUTANTS:
                with self.subTest(mutant=name):
                    self.assertEqual(source.count(old), 1, name)
                    target = PREFIX + test
                    subject.write_text(source)
                    baseline = run(subject, target, name + ':baseline')
                    self.assertEqual(baseline[0], 'baseline', baseline)
                    subject.write_text(source.replace(old, new))
                    mutant = run(subject, target, name + ':mutant')
                    self.assertEqual(mutant[0], 'kill', mutant)


if __name__ == '__main__':
    unittest.main()
