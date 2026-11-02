"""Acceptance tests for the bounded exact-module adapter.

Changes that suppress manifest validation, failure propagation, time limits,
sequential waiting, or binding checks must make these tests fail.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LEAN = Path(os.environ.get('CBZ_TEST_LEAN', '/home/agent/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean'))


class WrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('kernel_replay', ROOT / 'kernel_replay.py')
        cls.kr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.kr)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'evidence')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / 'Toy.lean'
        self.source.write_text('theorem toy : True := True.intro\n')
        (self.base / 'Toy.olean').write_bytes(b'opaque bytes for wrapper test')

    def manifest(self, names=('Toy',)):
        for name in names:
            (self.base / (name + '.lean')).write_text('theorem toy : True := True.intro\n')
            (self.base / (name + '.olean')).write_bytes(b'opaque bytes for wrapper test')
        return self.kr.prepare_manifest(str(LEAN), [str(self.base)],
            [f'{n}={self.base / (n + ".lean")}' for n in names], [])

    def fake(self, code):
        path = self.base / 'fake-lean'
        path.write_text('#!/usr/bin/env python3\n' + code)
        path.chmod(0o700)
        return path

    def run_fake(self, code, names=('Toy',), timeout=4):
        manifest = self.manifest(names)
        fake = self.fake(code)
        # Tests replace only executable binding, retaining the real sysroot:
        manifest['lean']['path'] = str(fake)
        manifest['lean']['sha256'] = self.kr.sha256(fake)
        return self.kr.run_manifest(manifest, 256, timeout)

    def test_empty_coverage_rejected(self):
        with self.assertRaises(ValueError):
            self.kr.prepare_manifest(str(LEAN), [str(self.base)], [], [])

    def test_duplicate_modules_rejected(self):
        with self.assertRaises(ValueError):
            self.manifest(('Toy', 'Toy'))

    def test_missing_module_rejected(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.kr.prepare_manifest(str(LEAN), [str(self.base)], [f'Missing={self.source}'], [])

    def test_individual_olean_symlink_rejected(self):
        linked = self.base / 'linked'
        linked.mkdir()
        (linked / 'Toy.olean').symlink_to(self.base / 'Toy.olean')
        # Upstream selects lexical siblings. Canonicalizing a file symlink
        # could bind destination private parts while replaying other parts.
        with self.assertRaises(ValueError):
            self.kr.prepare_manifest(str(LEAN), [str(linked)], [f'Toy={self.source}'], [])

    def test_search_root_and_part_symlinks_rejected(self):
        linked = self.base / 'linked-root'
        linked.symlink_to(self.base, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.kr.prepare_manifest(str(LEAN), [str(linked)], [f'Toy={self.source}'], [])
        part = self.base / 'Toy.olean.server'
        part.symlink_to(self.base / 'Toy.olean')
        with self.assertRaises(ValueError):
            self.manifest()

    def test_expected_coverage_mismatch_rejected(self):
        m = self.manifest()
        m['expected_modules'].append('Missing')
        self.assertFalse(self.kr.run_manifest(m, 256, 4)['ok'])

    def test_source_change_rejected(self):
        m = self.manifest()
        self.source.write_text('changed')
        r = self.kr.run_manifest(m, 256, 4)
        self.assertFalse(r['ok'])
        self.assertIn('binding changed', r.get('error', ''))

    def test_part_change_rejected(self):
        m = self.manifest()
        (self.base / 'Toy.olean').write_bytes(b'changed')
        r = self.kr.run_manifest(m, 256, 4)
        self.assertFalse(r['ok'])
        self.assertIn('compiled parts changed', r.get('error', ''))

    def test_new_private_part_rejected(self):
        m = self.manifest()
        (self.base / 'Toy.olean.server').write_bytes(b'added')
        r = self.kr.run_manifest(m, 256, 4)
        self.assertFalse(r['ok'])
        self.assertIn('compiled parts changed', r.get('error', ''))

    def test_tool_binding_change_rejected(self):
        m = self.manifest()
        m['tool'][0]['sha256'] = '0' * 64
        r = self.kr.run_manifest(m, 256, 4)
        self.assertFalse(r['ok'])
        self.assertIn('tool binding changed', r.get('error', ''))

    def test_nonzero_exit_fatal_even_with_marker(self):
        r = self.run_fake("import os\nprint('CBZ_KERNEL_REPLAY_OK ' + os.environ['CBZ_KERNEL_REPLAY_MODULE'])\nraise SystemExit(7)\n")
        self.assertFalse(r['ok'])
        self.assertEqual(r['results'][0]['returncode'], 7)

    def test_missing_or_wrong_marker_fatal(self):
        for code in ("print('no replay')\n", "print('CBZ_KERNEL_REPLAY_OK Other')\n"):
            with self.subTest(code=code):
                self.assertFalse(self.run_fake(code)['ok'])

    def test_timeout_fatal(self):
        r = self.run_fake('import time\ntime.sleep(5)\n', timeout=0.1)
        self.assertFalse(r['ok'])
        self.assertEqual(r['results'][0]['status'], 'TIMEOUT')

    def test_memory_limit_applied(self):
        r = self.run_fake("import os, resource\nassert resource.getrlimit(resource.RLIMIT_AS)[0] == 256*1024*1024\nprint('CBZ_KERNEL_REPLAY_OK ' + os.environ['CBZ_KERNEL_REPLAY_MODULE'])\n")
        self.assertTrue(r['ok'], r)

    def test_one_child_at_a_time_and_exact_coverage(self):
        lock = self.base / 'active.lock'
        code = ("import os, time\nfrom pathlib import Path\n"
            + f"p = Path({str(lock)!r})\n"
            + "fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)\n"
            + "time.sleep(0.05)\nos.close(fd)\np.unlink()\n"
            + "print('CBZ_KERNEL_REPLAY_OK ' + os.environ['CBZ_KERNEL_REPLAY_MODULE'])\n")
        r = self.run_fake(code, ('Toy', 'Other'))
        self.assertTrue(r['ok'], r)
        self.assertEqual([x['module'] for x in r['results']], ['Toy', 'Other'])
        self.assertEqual(r['completed_count'], 2)

    def test_strict_cli(self):
        for args in (['--unknown'], ['run', '--help', '--bogus'], ['one', 'Toy', 'Other']):
            p = subprocess.run([sys.executable, str(ROOT / 'kernel_replay.py'), *args],
                capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_module_names_and_unknown_fields_rejected(self):
        for name in ('', '-v', 'Toy.*', 'Toy..Other', 'Toy/Other'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.kr.prepare_manifest(str(LEAN), [str(self.base)], [f'{name}={self.source}'], [])
        m = self.manifest()
        m['ignored_flag'] = True
        self.assertFalse(self.kr.run_manifest(m, 256, 4)['ok'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
