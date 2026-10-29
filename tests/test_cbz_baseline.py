"""Portable baseline validator controls; local fixture inputs and processes only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[1] / 'tools/cbz_baseline.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = None
        if TOOL.exists():
            spec = importlib.util.spec_from_file_location('cbz_baseline', TOOL)
            cls.tool = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.tool)

    def setUp(self):
        self.assertIsNotNone(self.tool, 'baseline validator is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'repo'
        self.root.mkdir()
        subprocess.run(['git', 'init', '-q', '-b', 'main', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), 'config', 'user.name', 'Fixture'], check=True)
        subprocess.run(['git', '-C', str(self.root), 'config', 'user.email', 'fixture@invalid'], check=True)
        (self.root / 'tools').mkdir()
        self.gate = self.root / 'tools/check.py'
        self.gate.write_text('print("observed fixture execution")\n')
        proof = self.root / 'proof.lean'
        proof.write_text('theorem fixture : True := True.intro\n')
        self.proof = {'schema': 'cbz.proof-index.v1', 'records': [
            {'id': 'fixture', 'status': 'QUARANTINED', 'source_hashes': [{'path': 'proof.lean', 'sha256': sha(proof)}],
             'dependencies': [], 'exact_boundary': 'fixture declaration', 'scientific_closure_limits': 'No kernel claim',
             'reason': 'not kernel audited', 'attempts': ['fixture inspection'], 'isolation': 'No CLOSED dependents', 'owner': 'none'}]}
        write(self.root / 'cbz/proof-index.json', self.proof)
        write(self.root / 'cbz/tool-registry.json', {'schema': 'cbz.tools.v1', 'tools': [
            {'path': 'tools/check.py', 'self_test': [sys.executable, 'tools/check.py']}]})
        statuses = {f'G{i}': 'INCOMPLETE' for i in range(12)}
        self.doc = '# Baseline state\n# Scientific boundaries\n# Cold-start commands\npython tools/check.py\n# Evidence and invariants\n# Unmet criteria\n'
        self.doc += '<!-- cbz-gates: ' + json.dumps(statuses) + ' -->\n'
        (self.root / 'STATUS.md').write_text(self.doc)
        self.config = {'schema': 'cbz.baseline.v1', 'proof_index': 'cbz/proof-index.json',
                       'tool_registry': 'cbz/tool-registry.json', 'docs': ['STATUS.md'],
                       'allowed_axioms': ['propext', 'Classical.choice', 'Quot.sound'], 'pins': [],
                       'cold_start': {'status': 'STATUS.md', 'required_answers': [
                           'Baseline state', 'Scientific boundaries', 'Cold-start commands',
                           'Evidence and invariants', 'Unmet criteria'], 'commands': [['python', 'tools/check.py']]},
                       'gates': [{'id': f'G{i}', 'argv': [sys.executable, 'tools/check.py'],
                                  'sources': ['tools/check.py'], 'timeout_seconds': 3} for i in range(12)]}
        write(self.root / self.tool.DEFAULT_CONFIG, self.config)
        self.commit()
        self.out = self.base / 'receipts'

    def commit(self):
        subprocess.run(['git', '-C', str(self.root), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(self.root), 'commit', '-qm', 'fixture'], check=True)

    def inspect(self, receipts=None):
        return self.tool.inspect(self.root, receipts=receipts)

    def test_missing_receipts_are_incomplete_never_green(self):
        report = self.inspect()
        self.assertEqual(report['status'], 'INCOMPLETE')
        self.assertEqual(report['gate_statuses'], {f'G{i}': 'INCOMPLETE' for i in range(12)})
        self.assertFalse(report['certifies_cbz0'])

    def test_live_proof_tamper_fails(self):
        (self.root / 'proof.lean').write_text('axiom injected : False\n')
        self.assertEqual(self.inspect()['status'], 'FAIL')

    def test_closed_cannot_depend_transitively_on_quarantine(self):
        record = dict(self.proof['records'][0], id='closed', status='CLOSED', dependencies=['fixture'],
                      axioms=[], kernel_evidence=[])
        self.proof['records'].append(record)
        write(self.root / 'cbz/proof-index.json', self.proof)
        report = self.inspect()
        self.assertTrue(any('quarant' in x.lower() for x in report['failures']))

    def test_custom_axiom_cannot_be_closed(self):
        record = self.proof['records'][0]
        record.update(status='CLOSED', axioms=['Custom.fake'], kernel_evidence=[])
        write(self.root / 'cbz/proof-index.json', self.proof)
        self.assertTrue(any('axiom' in x.lower() for x in self.inspect()['failures']))

    def test_unknown_proof_status_refused(self):
        self.proof['records'][0]['status'] = 'PASS'
        write(self.root / 'cbz/proof-index.json', self.proof)
        self.assertEqual(self.inspect()['status'], 'FAIL')

    def test_unlisted_mission_tool_refused(self):
        (self.root / 'tools/cbz_hidden.py').write_text('print("hidden")\n')
        self.assertTrue(any('unlisted' in x.lower() for x in self.inspect()['failures']))

    def test_receipts_bind_actual_execution_and_stale_source_is_refused(self):
        run = self.tool.run(self.root, self.out)
        self.assertTrue((self.out / 'receipts.json').exists())
        self.assertEqual([r['status'] for r in run['gates']], ['PASS'] * 12)
        observed = self.inspect(self.out / 'receipts.json')
        self.assertEqual(set(observed['gate_statuses'].values()), {'PASS'})
        self.assertEqual(observed['status'], 'FAIL')  # docs still say INCOMPLETE
        self.gate.write_text('print("changed source")\n')
        stale = self.inspect(self.out / 'receipts.json')
        self.assertEqual(set(stale['gate_statuses'].values()), {'INCOMPLETE'})
        self.assertTrue(any('stale' in x.lower() for x in stale['unmet']))

    def test_forged_pass_without_process_logs_is_refused(self):
        fake = self.base / 'forged.json'
        write(fake, {'schema': 'cbz.receipts.v1', 'config_sha256': sha(self.root / self.tool.DEFAULT_CONFIG),
                     'gates': [{'id': f'G{i}', 'status': 'PASS'} for i in range(12)]})
        report = self.inspect(fake)
        self.assertEqual(set(report['gate_statuses'].values()), {'INCOMPLETE'})
        self.assertNotEqual(report['status'], 'PASS')

    def test_changed_command_invalidates_receipt(self):
        self.tool.run(self.root, self.out)
        self.config['gates'][0]['argv'].append('new-argument')
        write(self.root / self.tool.DEFAULT_CONFIG, self.config)
        self.assertEqual(set(self.inspect(self.out / 'receipts.json')['gate_statuses'].values()), {'INCOMPLETE'})

    def test_output_inside_source_root_refused(self):
        with self.assertRaises(self.tool.ValidationError):
            self.tool.run(self.root, self.root / 'out')

    def test_timeout_is_recorded_incomplete(self):
        self.gate.write_text('import time\ntime.sleep(10)\n')
        self.config['gates'] = [dict(self.config['gates'][0], timeout_seconds=1)]
        write(self.root / self.tool.DEFAULT_CONFIG, self.config)
        result = self.tool.run(self.root, self.out)
        self.assertEqual(result['gates'][0]['status'], 'INCOMPLETE')
        self.assertTrue(result['gates'][0]['timed_out'])

    def test_offline_pin_object_and_manifest_checked(self):
        head = subprocess.check_output(['git', '-C', str(self.root), 'rev-parse', 'HEAD'], text=True).strip()
        write(self.root / 'lake-manifest.json', {'packages': [{'name': 'Math', 'rev': head}]})
        self.config['pins'] = [{'repo': '.', 'commit': head, 'object_path': 'proof.lean',
                               'sha256': sha(self.root / 'proof.lean'), 'pin_file': 'lake-manifest.json', 'package': 'Math'}]
        write(self.root / self.tool.DEFAULT_CONFIG, self.config)
        report = self.inspect()
        self.assertEqual(report['pins'][0]['status'], 'PASS')
        write(self.root / 'lake-manifest.json', {'packages': [{'name': 'Math', 'rev': 'f' * 40}]})
        self.assertEqual(self.inspect()['pins'][0]['status'], 'FAIL')

    def test_missing_offline_dependency_is_incomplete(self):
        self.config['pins'] = [{'repo': '../missing', 'commit': 'f' * 40, 'object_path': 'x', 'sha256': '0' * 64}]
        write(self.root / self.tool.DEFAULT_CONFIG, self.config)
        self.assertEqual(self.inspect()['pins'][0]['status'], 'INCOMPLETE')

    def test_cold_start_checks_executable_entrypoints_without_theorem_claim(self):
        result = self.tool.cold_start(self.root)
        self.assertEqual(result['status'], 'PASS')
        self.assertFalse(result['certifies_theorems'])
        (self.root / 'STATUS.md').write_text('PASS\n')
        self.assertEqual(self.tool.cold_start(self.root)['status'], 'FAIL')

    def test_exit_two_is_incomplete_rather_than_an_observed_failure(self):
        self.gate.write_text('import sys\nsys.exit(2)\n')
        self.config['gates'] = [self.config['gates'][0]]
        write(self.root / self.tool.DEFAULT_CONFIG, self.config)
        self.assertEqual(self.tool.run(self.root, self.out)['gates'][0]['status'], 'INCOMPLETE')

    def test_receipt_log_tampering_is_refused(self):
        self.tool.run(self.root, self.out)
        (self.out / 'G0.stdout').write_text('forged PASS')
        self.assertEqual(self.inspect(self.out / 'receipts.json')['gate_statuses']['G0'], 'INCOMPLETE')

    def test_declared_allowed_axioms_cannot_admit_custom_axioms(self):
        self.config['allowed_axioms'].append('Custom.fake')
        write(self.root / self.tool.DEFAULT_CONFIG, self.config)
        with self.assertRaises(self.tool.ValidationError):
            self.inspect()

    def test_duplicate_json_key_is_refused(self):
        (self.root / self.tool.DEFAULT_CONFIG).write_text('{"schema":"cbz.baseline.v1","schema":"forged"}')
        with self.assertRaises(self.tool.ValidationError):
            self.inspect()


if __name__ == '__main__':
    unittest.main()
