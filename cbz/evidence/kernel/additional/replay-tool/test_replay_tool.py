"""Fail-closed tests for the bounded CBZ replay generator and transcript reader."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

MODULE = Path(os.environ.get('REPLAY_TOOL_MODULE', Path(__file__).with_name('replay_tool.py')))
spec = importlib.util.spec_from_file_location('tested_replay_tool', MODULE)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

COLLECTOR = ('structure _inv.St where\n  visited : NameSet := {}\n  axioms : NameSet := {}\n'
             'partial def _inv.collect (env : Environment) (c : Name) : StateM _inv.St Unit := pure ()\n\n')


def expected():
    return {'module_counts': {'Alpha': 2, 'Beta': 1},
            'module_declarations': {'Alpha': ['A', 'B'], 'Beta': ['C']}}


def transcript():
    return ('KERNEL_CLOSURE_MODULE Alpha COUNT 2\n'
            'KERNEL_CLOSURE_SEED Alpha A\n'
            'KERNEL_CLOSURE_SEED Alpha B\n'
            'KERNEL_CLOSURE_MODULE Beta COUNT 1\n'
            'KERNEL_CLOSURE_SEED Beta C\n'
            'KERNEL_CLOSURE_REPLAY_BEGIN SEEDS 3 CLOSURE 9\n'
            'KERNEL_CLOSURE_STANDARD_ONLY_PASS SEEDS 3 CLOSURE 9 AXIOMS [Classical.choice, Quot.sound, propext]\n')


class TranscriptTests(unittest.TestCase):
    def check(self, text, census=None):
        return tool.validate_transcript(text, expected() if census is None else census, 'a' * 64, exit_code=0, stderr='')

    def test_positive_exact_census(self):
        result = self.check(transcript())
        self.assertEqual(result['owned_constants'], 3)
        self.assertEqual(result['transitive_type_value_closure_constants'], 9)
        self.assertEqual(result['module_declarations'], expected()['module_declarations'])

    def test_empty_axiom_list_is_allowed(self):
        result = self.check(transcript().replace('Classical.choice, Quot.sound, propext', ''))
        self.assertEqual(result['axioms'], [])

    def test_raw_stream_digest_preserves_line_endings(self):
        raw = transcript().replace('\n', '\r\n')
        self.assertEqual(self.check(raw)['stdout_sha256'], hashlib.sha256(raw.encode()).hexdigest())

    def test_execution_failure_or_stderr_cannot_qualify(self):
        for code, stderr in ((1, ''), (0, 'warning'), (None, ''), (False, '')):
            with self.subTest(code=code, stderr=stderr), self.assertRaises(ValueError):
                tool.validate_transcript(transcript(), expected(), 'a' * 64, exit_code=code, stderr=stderr)

    def test_wrong_owned_name(self):
        with self.assertRaisesRegex(ValueError, 'owned names differ'):
            self.check(transcript().replace('SEED Alpha B', 'SEED Alpha D'))

    def test_wrong_module_assignment(self):
        text = transcript().replace('SEED Alpha B', 'SEED Alpha C').replace('SEED Beta C', 'SEED Beta B')
        with self.assertRaisesRegex(ValueError, 'module assignment differs'):
            self.check(text)

    def test_forbidden_axiom(self):
        with self.assertRaisesRegex(ValueError, 'forbidden axiom'):
            self.check(transcript().replace('propext]', 'sorryAx]'))

    def test_duplicate_axiom(self):
        with self.assertRaisesRegex(ValueError, 'duplicate axiom'):
            self.check(transcript().replace('propext]', 'propext, propext]'))

    def test_malformed_markers_and_noise(self):
        for line in ('KERNEL_CLOSURE_UNKNOWN X', 'KERNEL_CLOSURE_SEED Alpha', 'warning: fake', 'KERNEL_CLOSURE_REPLAY_BEGIN SEEDS -3 CLOSURE 9'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                self.check(transcript() + line + '\n')

    def test_missing_or_repeated_pass(self):
        last = transcript().splitlines()[-1] + '\n'
        for text in (transcript().replace(last, ''), transcript() + last):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.check(text)

    def test_missing_or_repeated_begin(self):
        begin = 'KERNEL_CLOSURE_REPLAY_BEGIN SEEDS 3 CLOSURE 9\n'
        for text in (transcript().replace(begin, ''), transcript().replace(begin, begin + begin)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.check(text)

    def test_begin_and_completion_counts_agree(self):
        for old, new in (('BEGIN SEEDS 3', 'BEGIN SEEDS 2'), ('BEGIN SEEDS 3 CLOSURE 9', 'BEGIN SEEDS 3 CLOSURE 8'), ('PASS SEEDS 3', 'PASS SEEDS 4')):
            with self.subTest(new=new), self.assertRaises(ValueError):
                self.check(transcript().replace(old, new))

    def test_missing_extra_or_duplicate_seed(self):
        line = 'KERNEL_CLOSURE_SEED Alpha B\n'
        for text in (transcript().replace(line, ''), transcript().replace(line, line + line), transcript().replace(line, line + 'KERNEL_CLOSURE_SEED Alpha D\n')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.check(text)

    def test_bad_module_records(self):
        for text in (transcript().replace('MODULE Alpha COUNT 2', 'MODULE Alpha COUNT 0'), transcript().replace('MODULE Beta COUNT 1', 'MODULE Alpha COUNT 1'), transcript().replace('SEED Alpha A', 'SEED Beta A')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.check(text)

    def test_empty_or_inconsistent_expected_census(self):
        for census in ({}, {'module_counts': {}, 'module_declarations': {}}, {'module_counts': {'Alpha': 1}, 'module_declarations': {'Alpha': ['A', 'A']}}, {'module_counts': {'Alpha': 2}, 'module_declarations': {'Alpha': ['A']}}, {'module_counts': {'Alpha': True}, 'module_declarations': {'Alpha': ['A']}}):
            with self.subTest(census=census), self.assertRaises(ValueError):
                self.check(transcript(), census)

    def test_caller_cannot_expand_axiom_allowlist(self):
        census = expected()
        census['allowed_axioms'] = ['sorryAx']
        with self.assertRaises(ValueError):
            self.check(transcript(), census)

    def test_malformed_allowlist_or_nontext_input(self):
        for raw in ([{}], [1], 'propext', None, ['propext', 'propext']):
            census = expected()
            census['allowed_axioms'] = raw
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.check(transcript(), census)
        for raw in (None, b'bytes'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.check(raw)

    def test_no_records_after_completion(self):
        with self.assertRaises(ValueError):
            self.check(transcript() + 'KERNEL_CLOSURE_SEED Beta D\n')

    def test_no_pass_before_begin(self):
        lines = transcript().splitlines()
        lines[-2], lines[-1] = lines[-1], lines[-2]
        with self.assertRaises(ValueError):
            self.check('\n'.join(lines) + '\n')


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.helper = self.root / 'helper.py'
        self.helper.write_text('def inventory_source():\n    return ' + repr('prefix\n' + COLLECTOR + 'def _inv.main (arg) := pure ()\n') + '\n')
        self.digest = hashlib.sha256(self.helper.read_bytes()).hexdigest()

    def generate(self, **kw):
        args = {'collector_path': self.helper, 'expected_collector_sha256': self.digest,
                'modules': ['Alpha', 'Beta'], 'imports': ['Beta']}
        args.update(kw)
        return tool.generate_program(**args)

    def test_positive_generator_and_verbatim_collector(self):
        source, metadata = self.generate()
        self.assertIn(COLLECTOR, source)
        self.assertIn('let fresh ← mkEmptyEnvironment 0\n', source)
        self.assertIn('let checked ← fresh.toKernelEnv.replay selected\n', source)
        self.assertIn('if ci.isUnsafe || ci.isPartial then', source)
        self.assertIn('kernel replay omitted candidate constant', source)
        self.assertEqual(metadata['seed_modules'], ['Alpha', 'Beta'])

    def test_nonempty_unique_simple_modules_and_imports(self):
        for modules, imports in (([], ['Beta']), (['Alpha', 'Alpha'], ['Beta']), (['Alpha; bad'], ['Beta']), (['Alpha'], []), (['Alpha'], ['Beta', 'Beta']), (['Alpha'], ['../Beta'])):
            with self.subTest(modules=modules, imports=imports), self.assertRaises(ValueError):
                self.generate(modules=modules, imports=imports)

    def test_collector_hash_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'collector hash mismatch'):
            self.generate(expected_collector_sha256='0' * 64)

    def test_no_execution_of_collector_python(self):
        marker = self.root / 'executed'
        self.helper.write_text(self.helper.read_text() + '\nopen(' + repr(str(marker)) + ', "w").write("bad")\n')
        self.digest = hashlib.sha256(self.helper.read_bytes()).hexdigest()
        self.generate()
        self.assertFalse(marker.exists())

    def test_missing_or_ambiguous_collector_literal(self):
        for source in ('def inventory_source(): return "nothing"\n', self.helper.read_text() + self.helper.read_text()):
            self.helper.write_text(source)
            self.digest = hashlib.sha256(self.helper.read_bytes()).hexdigest()
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.generate()

    def test_exclusive_output_preserves_existing_file(self):
        destination = self.root / 'existing'
        destination.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            tool.write_outputs_exclusive({destination: b'replace'})
        self.assertEqual(destination.read_bytes(), b'preserve')

    def test_output_alias_is_rejected(self):
        destination = self.root / 'out'
        with self.assertRaises(ValueError):
            tool.ensure_distinct_paths([destination, self.root / '.' / 'out'])

    def test_existing_second_output_creates_no_first_output(self):
        first, second = self.root / 'first', self.root / 'second'
        second.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            tool.write_outputs_exclusive({first: b'new', second: b'overwrite'})
        self.assertFalse(first.exists())
        self.assertEqual(second.read_bytes(), b'preserve')
        self.assertFalse(list(self.root.glob('.cbz-replay-*')))

    def test_symlink_output_alias_is_rejected(self):
        link = self.root / 'link'
        link.symlink_to(self.helper)
        with self.assertRaises(ValueError):
            tool.ensure_distinct_paths([self.helper, link])

    def test_duplicate_and_nonfinite_json_are_rejected(self):
        path = self.root / 'data.json'
        for source in ('{"a":1,"a":2}', '{"x":{"a":1,"a":2}}', '{"x":NaN}'):
            path.write_text(source)
            with self.subTest(source=source), self.assertRaises(ValueError):
                tool.load_json_unique(path)

    def test_cli_validation_round_trip_and_no_overwrite(self):
        out, err = self.root / 'out.log', self.root / 'err.log'
        census, program, receipt = self.root / 'census.json', self.root / 'proof.lean', self.root / 'receipt.json'
        raw = transcript().replace('\n', '\r\n').encode()
        out.write_bytes(raw)
        err.write_bytes(b'')
        census.write_text(json.dumps(expected()))
        program.write_text('generated program identity fixture')
        command = [sys.executable, '-B', '-S', str(MODULE), 'validate', '--transcript', str(out), '--stderr', str(err), '--exit-code', '0', '--expected-census', str(census), '--program', str(program), '--output', str(receipt)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = tool.load_json_unique(receipt)
        self.assertEqual(payload['stdout_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertFalse(payload['execution_authentication'])
        before = receipt.read_bytes()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(receipt.read_bytes(), before)

    def test_outputs_cannot_replace_input(self):
        with self.assertRaises(ValueError):
            tool.ensure_distinct_paths([self.helper, self.helper.resolve()])


if __name__ == '__main__':
    unittest.main()
