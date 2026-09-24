"""Synthetic transport tests, distinct from the capsule's 36 math controls."""
from __future__ import annotations
import hashlib
from pathlib import Path
import stat
import tempfile
import unittest
import warnings
import zipfile
import replay_capsule as replay


class TestCapsuleTransport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, entries):
        path = self.root / 'input.zip'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(path, 'w') as archive:
                for name, body in entries:
                    archive.writestr(name, body)
        raw = path.read_bytes()
        return path, {'expected_sha': hashlib.sha256(raw).hexdigest(),
                      'expected_bytes': len(raw)}

    def test_safe_nested_files(self):
        source, identity = self.fixture([('a.txt', 'a'), ('nested/b.txt', 'b')])
        target = self.root / 'output'
        self.assertEqual(replay.stage_capsule(source, target, **identity), 2)
        self.assertEqual((target / 'nested/b.txt').read_text(), 'b')

    def test_modified_bytes_refused_before_staging(self):
        source, identity = self.fixture([('a.txt', 'a')])
        raw = bytearray(source.read_bytes()); raw[-1] ^= 1; source.write_bytes(raw)
        target = self.root / 'output'
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            replay.stage_capsule(source, target, **identity)
        self.assertFalse(target.exists())

    def test_wrong_size_refused(self):
        source, identity = self.fixture([('a.txt', 'a')])
        identity['expected_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'byte-count'):
            replay.stage_capsule(source, self.root / 'output', **identity)

    def test_unsafe_paths_refused(self):
        for name in ('../escape', '/absolute', 'a\\b', 'C:drive'):
            source, identity = self.fixture([(name, 'x')])
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                replay.stage_capsule(source, self.root / 'output', **identity)
            self.assertFalse((self.root / 'output').exists())

    def test_duplicate_and_file_parent_refused(self):
        for entries in ([('a', 'x'), ('a', 'y')], [('a', 'x'), ('a/b', 'y')]):
            source, identity = self.fixture(entries)
            with self.assertRaises(ValueError):
                replay.stage_capsule(source, self.root / 'output', **identity)
            self.assertFalse((self.root / 'output').exists())

    def test_symlink_member_refused(self):
        info = zipfile.ZipInfo('link')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        source, identity = self.fixture([(info, '/outside')])
        with self.assertRaisesRegex(ValueError, 'regular file'):
            replay.stage_capsule(source, self.root / 'output', **identity)

    def test_inflated_limit_refused(self):
        source, identity = self.fixture([('a', '123456789')])
        with self.assertRaisesRegex(ValueError, 'ceiling'):
            replay.stage_capsule(source, self.root / 'output', max_uncompressed=8, **identity)

    def test_existing_output_refused(self):
        source, identity = self.fixture([('a', 'x')])
        target = self.root / 'output'; target.mkdir()
        (target / 'sentinel').write_text('keep')
        with self.assertRaises(FileExistsError):
            replay.stage_capsule(source, target, **identity)
        self.assertEqual((target / 'sentinel').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main(verbosity=2)
