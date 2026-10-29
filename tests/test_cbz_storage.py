"""Real-filesystem adversarial tests for immutable byte sharing only."""
import importlib.util
from pathlib import Path
import os
import tempfile
import subprocess
import sys
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('cbz_storage', Path(__file__).resolve().parents[1] / 'tools/cbz_storage.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.roots = [Path(self.tmp.name) / name for name in ('a', 'b')]
        for root in self.roots:
            root.mkdir()
            (root / 'file').write_bytes(b'immutable data' * 1024)
            (root / 'file').chmod(0o644)
        self.args = list(map(str, self.roots))

    def test_real_apply_preserves_bytes_modes_and_idempotence(self):
        plan = s.make_plan(self.args)
        self.assertEqual(len(plan['operations']), 1)
        report = s.apply_plan(plan)
        self.assertEqual(report['applied'], 1)
        self.assertTrue(report['content_paths_modes_verified'])
        self.assertTrue(os.path.samefile(*(r / 'file' for r in self.roots)))
        self.assertEqual(s.make_plan(self.args)['operations'], [])
        self.assertEqual(s.apply_plan(s.make_plan(self.args))['applied'], 0)

    def test_changed_input_refused_before_link(self):
        plan = s.make_plan(self.args)
        (self.roots[1] / 'file').write_bytes(b'changed')
        with self.assertRaises(s.Refusal):
            s.apply_plan(plan)
        self.assertFalse(os.path.samefile(*(r / 'file' for r in self.roots)))

    def test_mode_mismatch_excluded(self):
        (self.roots[1] / 'file').chmod(0o600)
        self.assertEqual(s.make_plan(self.args)['operations'], [])

    def test_symlinks_excluded_and_ancestor_refused(self):
        (self.roots[0] / 'link').symlink_to(self.roots[1] / 'file')
        plan = s.make_plan(self.args)
        self.assertEqual(len(plan['operations']), 1)
        s.apply_plan(plan)
        self.assertTrue((self.roots[0] / 'link').is_symlink())
        alias = Path(self.tmp.name) / 'alias'
        alias.symlink_to(self.roots[0])
        with self.assertRaises(s.Refusal):
            s.make_plan([str(alias)])

    def test_forged_escape_and_operations_refused(self):
        plan = s.make_plan(self.args)
        plan['entries'][0]['path'] = '../../outside'
        with self.assertRaises(s.Refusal):
            s.apply_plan(plan)
        with self.assertRaises(s.Refusal):
            s.safe_path(self.roots[0], '../outside')
        plan = s.make_plan(self.args)
        plan['operations'] = []
        with self.assertRaises(s.Refusal):
            s.apply_plan(plan)

    def test_race_before_replace_refused_and_temp_removed(self):
        plan = s.make_plan(self.args)
        def mutate(src, dst):
            dst.write_bytes(b'race')
        with self.assertRaises(s.Refusal):
            s.apply_plan(plan, before_replace=mutate)
        self.assertEqual((self.roots[0] / 'file').read_bytes(), b'immutable data' * 1024)
        self.assertFalse(any(Path(self.tmp.name).rglob('.cbz-hardlink-*')))

    def test_race_symlink_refused(self):
        plan = s.make_plan(self.args)
        def mutate(src, dst):
            dst.unlink()
            dst.symlink_to(src)
        with self.assertRaises(s.Refusal):
            s.apply_plan(plan, before_replace=mutate)
        self.assertFalse(any(Path(self.tmp.name).rglob('.cbz-hardlink-*')))

    def test_distinct_xattr_same_bytes_modes_owners_refused(self):
        a, b = (r / 'file' for r in self.roots)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        self.assertEqual(a.stat().st_mode, b.stat().st_mode)
        self.assertEqual(a.stat().st_uid, b.stat().st_uid)
        self.assertEqual(a.stat().st_gid, b.stat().st_gid)
        os.setxattr(b, 'user.cbz_test', b'nonempty metadata')
        with self.assertRaises(s.Refusal):
            s.make_plan(self.args)
        self.assertFalse(os.path.samefile(a, b))
        self.assertEqual(os.getxattr(b, 'user.cbz_test'), b'nonempty metadata')

    def test_xattr_enumeration_failure_refused(self):
        with mock.patch.object(s.os, 'listxattr', side_effect=PermissionError('denied')):
            with self.assertRaises(s.Refusal):
                s.make_plan(self.args)

    def test_xattr_added_before_replace_refused(self):
        plan = s.make_plan(self.args)
        def mutate(src, dst):
            os.setxattr(dst, 'user.cbz_race', b'new metadata')
        with self.assertRaises(s.Refusal):
            s.apply_plan(plan, before_replace=mutate)
        self.assertFalse(os.path.samefile(*(r / 'file' for r in self.roots)))
        self.assertFalse(any(Path(self.tmp.name).rglob('.cbz-hardlink-*')))

    def test_archive_selection_excludes_mutable_small_symlink_and_caches(self):
        payload = b'x' * (1024 * 1024 + 1)
        for root in self.roots:
            (root / 'kept.tar.gz').write_bytes(payload)
            (root / 'mutable.lean').write_bytes(payload)
            (root / 'small.zip').write_bytes(b'x')
            (root / 'linked.zip').symlink_to(root / 'kept.tar.gz')
            for dirname in ('.git', '.lake', 'active-cache'):
                (root / dirname).mkdir()
                (root / dirname / 'skip.zip').write_bytes(payload)
        plan = s.make_plan(self.args, archives_only=True)
        self.assertEqual(len(plan['entries']), 2)
        self.assertEqual(len(plan['operations']), 1)
        self.assertTrue(all(e['path'] == 'kept.tar.gz' for e in plan['entries']))
        self.assertEqual(s.apply_plan(plan)['applied'], 1)
        self.assertEqual(s.make_plan(self.args, archives_only=True)['operations'], [])

    def test_cli_tampered_sha_refused_without_mutation(self):
        plan_path = Path(self.tmp.name) / 'plan.json'
        report_path = Path(self.tmp.name) / 'report.json'
        plan_path.write_bytes(s.encoded(s.make_plan(self.args)))
        result = subprocess.run([sys.executable, s.__file__, 'apply',
            '--plan', str(plan_path), '--sha256', '0' * 64,
            '--report', str(report_path)], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'plan SHA mismatch', result.stderr)
        self.assertFalse(report_path.exists())
        self.assertFalse(os.path.samefile(*(r / 'file' for r in self.roots)))

    def test_already_shared_inodes_not_planned(self):
        a, b = (r / 'file' for r in self.roots)
        b.unlink()
        os.link(a, b)
        self.assertEqual(s.make_plan(self.args)['operations'], [])


if __name__ == '__main__':
    unittest.main()
