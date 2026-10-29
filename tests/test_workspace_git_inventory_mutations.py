"""CLI and false-success controls for the census mutation self-test runner."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MutationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        (self.source / 'tools').mkdir(parents=True)
        (self.source / 'tests').mkdir()
        for path in ['tools/workspace_git_inventory.py', 'tools/workspace_git_inventory_mutations.py', 'tests/test_workspace_git_inventory.py']:
            shutil.copyfile(ROOT / path, self.source / path)
        self.runner = self.source / 'tools/workspace_git_inventory_mutations.py'
        self.output = self.root / 'evidence'

    def run_cli(self, *, override=None, env=None, output=True):
        interpreter = [sys.executable, '-S', '-B']
        if sys.flags.optimize:
            interpreter.append('-' + 'O' * sys.flags.optimize)
        if override is None:
            command = [*interpreter, str(self.runner)]
        else:
            driver = (
                'import importlib.util,sys\n'
                'p=sys.argv[1];s=importlib.util.spec_from_file_location("runner",p)\n'
                'm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n'
                + override + '\n'
                'sys.argv=[p]+sys.argv[2:];raise SystemExit(m.main())\n')
            command = [*interpreter, '-c', driver, str(self.runner)]
        if output:
            command += ['--output', str(self.output)]
        return subprocess.run(command, cwd=self.root, env=env, capture_output=True, text=True, timeout=90)

    def report(self):
        return json.loads((self.output / 'results.json').read_bytes())

    def test_requires_new_output_and_keeps_private_success_evidence(self):
        missing = self.run_cli(output=False)
        self.assertEqual(missing.returncode, 2)
        self.assertFalse(self.output.exists())
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        report = self.report()
        self.assertTrue(report['complete'])
        self.assertTrue(report['baseline']['passed'])
        self.assertEqual(report['baseline']['tests_run'], 11)
        self.assertEqual(len(report['mutations']), 4)
        self.assertTrue(all(row['detected_by_assertion'] for row in report['mutations']))
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o700)
        for path in self.output.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.output.iterdir()}
        repeated = self.run_cli()
        self.assertEqual(repeated.returncode, 2)
        self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.output.iterdir()})

    def test_failed_baseline_never_runs_mutations(self):
        fake = self.root / 'bin'
        fake.mkdir()
        git = fake / 'git'
        git.write_text('#!/bin/sh\nexit 7\n')
        git.chmod(0o700)
        env = os.environ.copy()
        env['PATH'] = str(fake) + os.pathsep + env['PATH']
        result = self.run_cli(env=env)
        self.assertEqual(result.returncode, 1, result.stderr)
        report = self.report()
        self.assertFalse(report['baseline']['passed'])
        self.assertEqual(report['mutations'], [])
        self.assertFalse(report['complete'])

    def test_surviving_mutant_is_failure(self):
        result = self.run_cli(override='n,o,_new,t=m.CASES[0];m.CASES[0]=(n,o,o,t)')
        self.assertEqual(result.returncode, 1, result.stderr)
        report = self.report()
        self.assertTrue(report['baseline']['passed'])
        self.assertEqual(report['mutations'][0]['returncode'], 0)
        self.assertFalse(report['mutations'][0]['detected_by_assertion'])
        self.assertFalse(report['all_detected'])
        self.assertFalse(report['complete'])

    def test_test_error_cannot_count_as_mutation_detection(self):
        result = self.run_cli(override='n,o,new,_t=m.CASES[0];m.CASES[0]=(n,o,new,"nonexistent_test")')
        self.assertEqual(result.returncode, 1, result.stderr)
        report = self.report()
        self.assertTrue(report['baseline']['passed'])
        self.assertEqual(report['mutations'][0]['errors'], 1)
        self.assertFalse(report['mutations'][0]['detected_by_assertion'])
        self.assertFalse(report['all_detected'])
        self.assertFalse(report['complete'])


if __name__ == '__main__':
    unittest.main()
