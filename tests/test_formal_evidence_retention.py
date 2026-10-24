"""Custody controls for three real formal executions; stdlib, no Lean fixtures.

The workflow-order regression runs on the test-only RED baseline. CLI controls
explicitly skip until the retention implementation exists. These fixtures are
for hosted execution; writing this source is not an execution claim.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/formal_evidence_history.py'
WORKFLOW = ROOT / '.github/workflows/formal-verification.yml'
PHASES = ('initial', 'normal', 'optimized')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def workflow_commands(text):
    """Read actual run scalars in the repository's existing step indentation."""
    commands = []
    for block in re.split(r'(?=^      - )', text, flags=re.MULTILINE):
        if not block.startswith('      - '):
            continue
        match = re.search(r'^        run: (.*)$', block, re.MULTILINE)
        if match is None:
            continue
        value = match.group(1)
        if value in ('|', '|-', '>', '>-'):
            lines = []
            for line in block[match.end():].splitlines()[1:]:
                if line and not line.startswith('          '):
                    break
                lines.append(line[10:])
            value = '\n'.join(lines)
        commands.append((value, block))
    return commands


class WorkflowRetentionOrderTests(unittest.TestCase):
    def test_every_real_producer_is_retained_before_its_evidence_can_be_reused(self):
        commands = workflow_commands(WORKFLOW.read_text(encoding='utf-8'))

        def locate(pattern):
            found = [(i, block) for i, (body, block) in enumerate(commands)
                     if re.search(pattern, body, re.MULTILINE)]
            self.assertEqual(len(found), 1, 'expected one executable workflow command: '+pattern)
            return found[0]

        prepare = locate(r'tools/formal_evidence_history\.py\s+prepare\b')
        initial = locate(r'tools/formal_gate_check\.py\s+--run-lean\b')
        normal = locate(r'python\S*\s+-B\s+-S\s+-m\s+unittest\s+tests\.test_formal_gate\b')
        optimized = locate(r'python\S*\s+-B\s+-O\s+-S\s+-m\s+unittest\s+tests\.test_formal_gate\b')
        binder = locate(r'tools/required_formal_check\.py\s+receipt\b')
        captures = {phase: locate(r'tools/formal_evidence_history\.py\s+capture\s+'+phase+r'\b')
                    for phase in PHASES}
        verify = locate(r'tools/formal_evidence_history\.py\s+verify\b')
        order = [prepare[0], initial[0], captures['initial'][0], normal[0],
                 captures['normal'][0], optimized[0], binder[0],
                 captures['optimized'][0], verify[0]]
        self.assertTrue(all(left < right for left, right in zip(order, order[1:])),
                        'capture initial/normal before the next producer; retain optimized after binding')
        for phase, producer in zip(PHASES, (initial, normal, optimized)):
            with self.subTest(phase=phase):
                block = captures[phase][1]
                self.assertRegex(block, r'(?m)^        if:\s*(?:\$\{\{\s*)?always\(\)')
                producer_id = re.search(r'^        id:\s*(\S+)\s*$', producer[1], re.MULTILINE)
                self.assertIsNotNone(producer_id, 'capture needs the actual producing step outcome')
                self.assertIn('steps.'+producer_id.group(1)+'.outcome', block)
        self.assertIn('--binding-outcome', captures['optimized'][1])
        self.assertIn('steps.required-binding.outcome', captures['optimized'][1])


@unittest.skipUnless(SCRIPT.is_file(), 'implementation absent in RED baseline')
class FormalEvidenceRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'formal/.lake/formal-evidence'
        self.phases = self.root / 'formal-evidence/phases'
        self.manifest = self.root / 'formal/manifest.json'
        self.manifest.parent.mkdir()
        self.manifest.write_bytes(b'{"fixture": "custody only, not a Lean manifest"}\n')
        self.env = dict(os.environ)
        for name in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR'):
            self.env.pop(name, None)
        self.git('init', '-q')
        self.git('-c', 'user.name=Formal custody fixture',
                 '-c', 'user.email=formal-fixture@example.invalid',
                 '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null',
                 'commit', '--allow-empty', '-qm', 'disposable custody fixture')
        self.head = self.git('rev-parse', 'HEAD').stdout.strip()
        self.env.update(GITHUB_SHA=self.head, GITHUB_REPOSITORY='d6g8k5htny-coder/main',
                        GITHUB_RUN_ID='226001', GITHUB_RUN_ATTEMPT='2')
        self.context = {'checked_commit': self.head, 'repository': 'd6g8k5htny-coder/main',
                        'run_id': '226001', 'run_attempt': '2'}

    def git(self, *args):
        result = subprocess.run(['git', *args], cwd=self.root, env=self.env,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        return result

    def cli(self, *args, succeeds=True):
        mode = ['-O'] if sys.flags.optimize else []
        result = subprocess.run([sys.executable, '-B', *mode, '-S', str(SCRIPT), *args],
                                cwd=self.root, env=self.env, capture_output=True,
                                text=True, timeout=20)
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, 'unsafe custody operation succeeded')
        return result

    def payload(self, label, receipt=True):
        files = {'build.log': (label+' build\n').encode(),
                 'version.log': (label+' version observation\n').encode(),
                 'Audit.lean': b'-- inert custody fixture, never compiled\n'}
        if receipt:
            record = {'schema_version': 1, 'scientific_effect': 'NONE',
                      'scientific_status_authority': False, 'checked_commit': self.head,
                      'repository': self.context['repository'],
                      'workflow_run_id': self.context['run_id'],
                      'formalization_status': 'kernel-checked',
                      'alignment_status': 'PENDING_INDEPENDENT_REVIEW',
                      'manifest_sha256': digest(self.manifest.read_bytes()),
                      'logs': {name: digest(raw) for name, raw in files.items()
                               if name.endswith('.log')}}
            files['receipt.json'] = (json.dumps(record, sort_keys=True)+'\n').encode()
        self.source.mkdir(parents=True, exist_ok=True)
        for name, raw in files.items():
            (self.source/name).write_bytes(raw)
        return files

    def observation(self, phase):
        return json.loads((self.phases/phase/'observation.json').read_text(encoding='utf-8'))

    def assert_snapshot(self, phase, expected, outcome=None, source_present=True):
        observed = self.observation(phase)
        self.assertEqual(observed['phase'], phase)
        self.assertEqual(observed['context'], self.context)
        self.assertIs(observed['source_present'], source_present)
        self.assertIn('binding_outcome', observed)
        if outcome is not None:
            self.assertEqual(observed['outcome'], outcome)
        self.assertEqual(observed['files'], {name: {'sha256': digest(raw), 'size': len(raw)}
                                             for name, raw in expected.items()})
        raw_dir = self.phases/phase/'raw'
        actual = {p.relative_to(raw_dir).as_posix(): p.read_bytes()
                  for p in raw_dir.rglob('*') if p.is_file()}
        self.assertEqual(actual, expected)
        return observed

    def finish_skipped(self, *phases):
        for phase in phases:
            self.cli('capture', phase, 'skipped')

    def complete_fixture(self):
        self.cli('prepare')
        expected = {}
        for phase in PHASES:
            expected[phase] = self.payload(phase)
            self.cli('capture', phase, 'success')
        self.cli('verify')
        return expected

    def test_prepare_preserves_preexisting_bytes_and_removes_them_from_next_phase(self):
        old = self.payload('old unrelated execution')
        self.cli('prepare')
        self.assert_snapshot('preexisting', old, 'preexisting')
        self.assertFalse(self.source.exists())
        self.finish_skipped(*PHASES)
        self.cli('verify')

    def test_three_distinct_executions_keep_exact_bytes_and_optimized_stays_canonical(self):
        expected = self.complete_fixture()
        for phase in PHASES:
            self.assert_snapshot(phase, expected[phase], 'success')
        self.assertEqual(len({expected[p]['receipt.json'] for p in PHASES}), 3)
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, expected['optimized'])

    def test_failed_missing_next_phase_cannot_inherit_previous_success(self):
        self.cli('prepare')
        initial = self.payload('initial')
        self.cli('capture', 'initial', 'success')
        self.assertFalse(self.source.exists())
        self.cli('capture', 'normal', 'failure')
        self.assert_snapshot('normal', {}, 'failure', source_present=False)
        self.assert_snapshot('initial', initial, 'success')
        self.finish_skipped('optimized')
        self.cli('verify')

    def test_success_without_new_evidence_is_recorded_but_refused(self):
        self.cli('prepare')
        self.cli('capture', 'initial', 'success', succeeds=False)
        self.assert_snapshot('initial', {}, 'success', source_present=False)

    def test_skipped_phase_records_no_source_even_when_stale_canonical_bytes_exist(self):
        self.cli('prepare')
        stale = self.payload('unconsumed old source')
        self.cli('capture', 'initial', 'skipped')
        self.assert_snapshot('initial', {}, 'skipped', source_present=False)
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, stale)
        self.finish_skipped('normal', 'optimized')
        self.cli('verify')

    def test_partial_failed_execution_is_preserved_without_manufacturing_receipt(self):
        self.cli('prepare')
        partial = self.payload('interrupted build', receipt=False)
        self.cli('capture', 'initial', 'failure')
        self.assert_snapshot('initial', partial, 'failure')
        self.assertFalse(self.source.exists())
        self.assertNotIn('receipt.json', self.observation('initial')['files'])
        self.finish_skipped('normal', 'optimized')
        self.cli('verify')

    def test_gate_receipt_survives_later_suite_failure_without_relabeling_the_phase(self):
        self.cli('prepare')
        self.finish_skipped('initial')
        succeeded_gate = self.payload('gate succeeded before suite assertion failed')
        self.cli('capture', 'normal', 'failure')
        self.assert_snapshot('normal', succeeded_gate, 'failure')
        self.finish_skipped('optimized')
        self.cli('verify')

    def test_actual_required_binder_consumes_only_optimized_logs_and_receipt(self):
        spec = importlib.util.spec_from_file_location('custody_required_check', ROOT/'tools/required_formal_check.py')
        binder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(binder)
        self.cli('prepare')
        initial = self.payload('initial')
        self.cli('capture', 'initial', 'success')
        self.payload('normal')
        self.cli('capture', 'normal', 'success')
        optimized = self.payload('optimized')
        result = binder.bind_receipt(self.source/'receipt.json', self.manifest, self.env, self.head)
        self.assertEqual(result['receipt_sha256'], digest(optimized['receipt.json']))
        self.assertNotEqual(result['receipt_sha256'], digest(initial['receipt.json']))
        binding_raw = (json.dumps(result, sort_keys=True)+'\n').encode()
        (self.source/'required-check-binding.json').write_bytes(binding_raw)
        optimized['required-check-binding.json'] = binding_raw
        self.cli('capture', 'optimized', 'success', '--binding-outcome', 'success')
        observed = self.assert_snapshot('optimized', optimized, 'success')
        self.assertEqual(observed['binding_outcome'], 'success')
        self.assertEqual((self.source/'receipt.json').read_bytes(), optimized['receipt.json'])
        self.cli('verify')
        (self.source/'receipt.json').write_bytes(initial['receipt.json'])
        with self.assertRaises(ValueError):
            binder.bind_receipt(self.source/'receipt.json', self.manifest, self.env, self.head)

    def test_duplicate_capture_refuses_to_overwrite_archive_or_consume_new_source(self):
        self.cli('prepare')
        first = self.payload('first')
        self.cli('capture', 'initial', 'success')
        observation_raw = (self.phases/'initial/observation.json').read_bytes()
        second = self.payload('second')
        self.cli('capture', 'initial', 'success', succeeds=False)
        self.assert_snapshot('initial', first, 'success')
        self.assertEqual((self.phases/'initial/observation.json').read_bytes(), observation_raw)
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, second)

    def test_prepare_refuses_duplicate_preexisting_archive(self):
        self.cli('prepare')
        raw = (self.phases/'preexisting/observation.json').read_bytes()
        old = self.payload('would overwrite prior observation')
        self.cli('prepare', succeeds=False)
        self.assertEqual((self.phases/'preexisting/observation.json').read_bytes(), raw)
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, old)

    def test_source_symlink_is_refused_without_reading_or_moving_its_target(self):
        self.cli('prepare')
        outside = self.root/'outside'; outside.mkdir()
        (outside/'secret.log').write_bytes(b'outside sentinel\n')
        self.source.parent.mkdir(parents=True, exist_ok=True)
        self.source.symlink_to(outside, target_is_directory=True)
        self.cli('capture', 'initial', 'success', succeeds=False)
        self.assertTrue(self.source.is_symlink())
        self.assertEqual((outside/'secret.log').read_bytes(), b'outside sentinel\n')
        self.assertFalse((self.phases/'initial/raw/secret.log').exists())

    def test_linked_archive_parent_is_refused(self):
        outside = self.root/'outside'; outside.mkdir()
        (self.root/'formal-evidence').symlink_to(outside, target_is_directory=True)
        self.cli('prepare', succeeds=False)
        self.assertEqual(list(outside.iterdir()), [])

    def test_linked_source_parent_is_refused(self):
        outside = self.root/'outside'; outside.mkdir()
        (self.root/'formal/.lake').symlink_to(outside, target_is_directory=True)
        self.cli('prepare', succeeds=False)
        self.assertEqual(list(outside.iterdir()), [])

    def test_payload_symlink_is_refused_without_consuming_originals(self):
        self.cli('prepare')
        self.source.mkdir(parents=True)
        outside = self.root/'sentinel'; outside.write_bytes(b'external bytes\n')
        link = self.source/'linked.log'; link.symlink_to(outside)
        self.cli('capture', 'initial', 'failure', succeeds=False)
        self.assertTrue(link.is_symlink())
        self.assertEqual(outside.read_bytes(), b'external bytes\n')

    def test_fifo_is_refused_without_consuming_source(self):
        if not hasattr(os, 'mkfifo'):
            self.skipTest('FIFO control requires hosted POSIX filesystem')
        self.cli('prepare')
        self.source.mkdir(parents=True)
        fifo = self.source/'pipe.log'; os.mkfifo(fifo)
        self.cli('capture', 'initial', 'failure', succeeds=False)
        self.assertTrue(fifo.exists())

    def test_verify_rejects_missing_phase_observation(self):
        self.cli('prepare')
        self.finish_skipped('initial', 'normal')
        self.cli('verify', succeeds=False)

    def test_verify_rejects_changed_missing_and_unlisted_payloads(self):
        expected = self.complete_fixture()
        log = self.phases/'initial/raw/build.log'
        original = expected['initial']['build.log']
        for mutation in ('changed', 'missing', 'unlisted'):
            with self.subTest(mutation=mutation):
                extra = self.phases/'initial/raw/not-in-inventory.log'
                if mutation == 'changed':
                    log.write_bytes(b'forged earlier output\n')
                elif mutation == 'missing':
                    log.unlink()
                else:
                    extra.write_bytes(b'unbound extra output\n')
                self.cli('verify', succeeds=False)
                log.write_bytes(original)
                if extra.exists():
                    extra.unlink()
                self.cli('verify')

    def test_verify_rejects_inventory_digest_size_and_path_substitution(self):
        self.complete_fixture()
        path = self.phases/'normal/observation.json'
        original = path.read_bytes()
        for mutation in ('digest', 'size', 'path'):
            with self.subTest(mutation=mutation):
                record = json.loads(original)
                if mutation == 'digest':
                    record['files']['build.log']['sha256'] = '0'*64
                elif mutation == 'size':
                    record['files']['build.log']['size'] += 1
                else:
                    record['files']['../outside.log'] = record['files'].pop('build.log')
                path.write_text(json.dumps(record), encoding='utf-8')
                self.cli('verify', succeeds=False)
                path.write_bytes(original)
                self.cli('verify')

    def test_verify_rejects_replaced_snapshot_link_even_when_target_bytes_match(self):
        expected = self.complete_fixture()
        log = self.phases/'normal/raw/build.log'
        outside = self.root/'same-bytes.log'; outside.write_bytes(expected['normal']['build.log'])
        log.unlink(); log.symlink_to(outside)
        self.cli('verify', succeeds=False)


if __name__ == '__main__':
    unittest.main()
