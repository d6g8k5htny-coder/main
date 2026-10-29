"""Disposable local bare remotes only; no federation repositories are mutated."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOL = Path(__file__).resolve().parents[1] / 'tools' / 'cbz_refs.py'


def git(repo, *args, input=None, check=True):
    p = subprocess.run(['git', '-C', str(repo), *args], input=input, text=True,
                       capture_output=True, check=check)
    return p.stdout.strip()


class RefSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TOOL.exists():
            cls.tool = None
            return
        spec = importlib.util.spec_from_file_location('cbz_refs', TOOL)
        cls.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tool)

    def setUp(self):
        self.assertIsNotNone(self.tool, 'cbz_refs snapshot/prune tool is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src, self.remote, self.out = [self.root / n for n in ('work', 'remote.git', 'snapshot')]
        subprocess.run(['git', 'init', '-q', '-b', 'main', str(self.src)], check=True)
        git(self.src, 'config', 'user.name', 'Fixture')
        git(self.src, 'config', 'user.email', 'fixture@invalid')
        (self.src / 'base').write_text('base\n')
        git(self.src, 'add', 'base')
        git(self.src, 'commit', '-qm', 'base')
        self.base = git(self.src, 'rev-parse', 'HEAD')
        git(self.src, 'checkout', '-qb', 'unmerged')
        (self.src / 'valuable').write_text('unique unmerged work\n')
        git(self.src, 'add', 'valuable')
        git(self.src, 'commit', '-qm', 'valuable')
        self.tip = git(self.src, 'rev-parse', 'HEAD')
        git(self.src, 'branch', 'duplicate')
        git(self.src, 'branch', 'protected')
        git(self.src, 'branch', 'active')
        git(self.src, 'tag', '-a', 'original-tag', '-m', 'keep exact tag object')
        git(self.src, 'checkout', '-q', 'main')
        subprocess.run(['git', 'clone', '-q', '--bare', str(self.src), str(self.remote)], check=True)
        self.tag = 'pre-cbz-fixture'

    def snapshot(self):
        return self.tool.snapshot(str(self.remote), self.out, self.tag, execute=True,
                                  protected=['refs/heads/protected'], active=['refs/heads/active'])

    def test_default_is_read_only_machine_plan(self):
        before = git(self.remote, 'show-ref')
        result = self.tool.snapshot(str(self.remote), self.out, self.tag)
        self.assertEqual(result['mode'], 'dry-run')
        self.assertEqual(result['manifest']['default_ref'], 'refs/heads/main')
        self.assertFalse(self.out.exists())
        self.assertEqual(git(self.remote, 'show-ref'), before)

    def test_archive_has_unique_parents_exact_manifest_and_recoverable_content(self):
        result = self.snapshot()
        archive = self.out / 'repository.git'
        parents = git(archive, 'show', '-s', '--format=%P', self.tag).split()
        self.assertCountEqual(parents, [self.base, self.tip])
        self.assertEqual(git(archive, 'show', f'{self.tip}:valuable'), 'unique unmerged work')
        self.assertEqual(git(archive, 'show', f'{self.tag}:base'), 'base')
        embedded = json.loads(git(archive, 'show', f'{self.tag}:.cbz/refs-manifest.json'))
        self.assertEqual(embedded, result['manifest'])
        self.assertEqual(embedded['refs']['refs/tags/original-tag'],
                         git(self.remote, 'rev-parse', 'refs/tags/original-tag'))
        for tip in embedded['refs'].values():
            git(archive, 'cat-file', '-e', tip)
        self.assertTrue(self.tool.verify_snapshot(self.out)['verified'])

    def test_finished_archive_retains_no_original_branch_refs(self):
        result = self.snapshot()
        archive = self.out / 'repository.git'
        self.assertEqual(git(archive, 'for-each-ref', '--format=%(refname)', 'refs/heads'), '')
        self.assertEqual(git(archive, 'rev-parse', 'HEAD'), result['archive_commit'])
        self.assertEqual(git(archive, 'show', f'{self.tip}:valuable'), 'unique unmerged work')

    def test_missing_original_annotated_tag_ref_is_rejected_before_gc(self):
        self.snapshot()
        archive = self.out / 'repository.git'
        original = git(archive, 'rev-parse', 'refs/tags/original-tag')
        git(archive, 'update-ref', '-d', 'refs/tags/original-tag')
        self.assertEqual(git(archive, 'cat-file', '-t', original), 'tag')
        with self.assertRaisesRegex(self.tool.SafetyError, 'ref|tag'):
            self.tool.verify_snapshot(self.out)

    def test_tag_only_blob_and_tree_refs_are_required_even_before_gc(self):
        blob = git(self.remote, 'hash-object', '-w', '--stdin', input='tag-only valuable content\n')
        tree = git(self.remote, 'mktree', input=f'100644 blob {blob}\tvaluable\n')
        for name, oid in (('tag-only-blob', blob), ('tag-only-tree', tree)):
            git(self.remote, 'update-ref', 'refs/tags/' + name, oid)
        self.snapshot()
        archive = self.out / 'repository.git'
        for name, oid in (('tag-only-blob', blob), ('tag-only-tree', tree)):
            with self.subTest(name=name):
                git(archive, 'update-ref', '-d', 'refs/tags/' + name)
                git(archive, 'cat-file', '-e', oid)
                with self.assertRaises(self.tool.SafetyError):
                    self.tool.verify_snapshot(self.out)
                git(archive, 'update-ref', 'refs/tags/' + name, oid)
        self.assertTrue(self.tool.verify_snapshot(self.out)['verified'])

    def test_unexpected_archive_head_or_tag_is_rejected(self):
        self.snapshot()
        archive = self.out / 'repository.git'
        for ref in ('refs/heads/extra', 'refs/tags/extra'):
            git(archive, 'update-ref', ref, self.base)
            with self.assertRaises(self.tool.SafetyError):
                self.tool.verify_snapshot(self.out)
            git(archive, 'update-ref', '-d', ref)

    def test_archive_head_must_be_detached_at_archive_commit(self):
        self.snapshot()
        archive = self.out / 'repository.git'
        git(archive, 'update-ref', '--no-deref', 'HEAD', self.base)
        with self.assertRaises(self.tool.SafetyError):
            self.tool.verify_snapshot(self.out)

    def test_source_index_and_worktrees_untouched(self):
        (self.src / 'base').write_text('staged unrelated work\n')
        git(self.src, 'add', 'base')
        (self.src / 'loose').write_text('untracked')
        index = (self.src / '.git/index').read_bytes()
        status = git(self.src, 'status', '--porcelain=v1')
        # Snapshot the dirty worktree repository itself as the read-only source.
        self.tool.snapshot(str(self.src), self.out, self.tag, execute=True)
        self.assertEqual((self.src / '.git/index').read_bytes(), index)
        self.assertEqual(git(self.src, 'status', '--porcelain=v1'), status)
        self.assertEqual((self.src / 'loose').read_text(), 'untracked')

    def test_publish_then_atomic_prune_preserves_default_policy_and_tag(self):
        self.snapshot()
        self.assertEqual(self.tool.publish(self.out)['mode'], 'dry-run')
        self.assertNotIn(f'refs/tags/{self.tag}', self.tool.remote_state(str(self.remote))['refs'])
        self.tool.publish(self.out, execute=True)
        plan = self.tool.prune(self.out)
        self.assertEqual(set(plan['delete_refs']), {'refs/heads/duplicate', 'refs/heads/unmerged'})
        result = self.tool.prune(self.out, execute=True)
        self.assertTrue(result['verified'])
        state = self.tool.remote_state(str(self.remote))
        self.assertEqual(set(k for k in state['refs'] if k.startswith('refs/heads/')),
                         {'refs/heads/main', 'refs/heads/protected', 'refs/heads/active'})
        self.assertIn(f'refs/tags/{self.tag}', state['refs'])
        # Force object collection after all original local heads were removed: archive tag alone retains tips.
        archive = self.out / 'repository.git'
        for name in ('active', 'duplicate', 'protected', 'unmerged'):
            git(archive, 'update-ref', '-d', f'refs/heads/{name}')
        git(archive, 'update-ref', '-d', 'refs/tags/original-tag')
        git(archive, 'reflog', 'expire', '--expire=now', '--all')
        git(archive, 'gc', '--prune=now')
        self.assertEqual(git(archive, 'show', f'{self.tip}:valuable'), 'unique unmerged work')

    def test_prune_requires_actual_remote_archive_readback(self):
        self.snapshot()
        with self.assertRaisesRegex(self.tool.SafetyError, 'archive|remote'):
            self.tool.prune(self.out, execute=True)
        self.assertEqual(git(self.remote, 'rev-parse', 'unmerged'), self.tip)

    def test_moved_original_branch_refuses_all_deletions(self):
        self.snapshot()
        self.tool.publish(self.out, execute=True)
        git(self.remote, 'update-ref', 'refs/heads/unmerged', self.base)
        before = git(self.remote, 'show-ref')
        with self.assertRaisesRegex(self.tool.SafetyError, 'changed|mismatch'):
            self.tool.prune(self.out, execute=True)
        self.assertEqual(git(self.remote, 'show-ref'), before)

    def test_unexpected_new_ref_refuses_all_deletions(self):
        self.snapshot()
        self.tool.publish(self.out, execute=True)
        git(self.remote, 'update-ref', 'refs/heads/new-work', self.tip)
        before = git(self.remote, 'show-ref')
        with self.assertRaises(self.tool.SafetyError):
            self.tool.prune(self.out, execute=True)
        self.assertEqual(git(self.remote, 'show-ref'), before)

    def test_server_rejection_is_atomic_and_never_silent_success(self):
        self.snapshot()
        self.tool.publish(self.out, execute=True)
        hook = self.remote / 'hooks/update'
        hook.write_text('#!/bin/sh\ncase "$1" in refs/heads/unmerged) exit 1;; esac\nexit 0\n')
        hook.chmod(0o755)
        before = git(self.remote, 'show-ref')
        with self.assertRaises(self.tool.SafetyError):
            self.tool.prune(self.out, execute=True)
        self.assertEqual(git(self.remote, 'show-ref'), before)

    def test_tampered_manifest_fails_before_remote_mutation(self):
        self.snapshot()
        path = self.out / 'manifest.json'
        data = json.loads(path.read_text())
        data['refs']['refs/heads/main'] = self.tip
        path.write_text(json.dumps(data))
        before = git(self.remote, 'show-ref')
        with self.assertRaises(self.tool.SafetyError):
            self.tool.publish(self.out, execute=True)
        self.assertEqual(git(self.remote, 'show-ref'), before)

    def test_lease_rejects_branch_changed_after_final_preflight_atomically(self):
        self.snapshot()
        self.tool.publish(self.out, execute=True)
        original_git = self.tool._git
        raced = False
        def racing_git(repo, *args, **kwargs):
            nonlocal raced
            if args and args[0] == 'push' and ':refs/heads/unmerged' in args:
                raced = True
                git(self.remote, 'update-ref', 'refs/heads/unmerged', self.base)
            return original_git(repo, *args, **kwargs)
        with patch.object(self.tool, '_git', racing_git):
            with self.assertRaises(self.tool.SafetyError):
                self.tool.prune(self.out, execute=True)
        self.assertTrue(raced)
        self.assertEqual(git(self.remote, 'rev-parse', 'refs/heads/duplicate'), self.tip)
        self.assertEqual(git(self.remote, 'rev-parse', 'refs/heads/unmerged'), self.base)

    def test_actual_tag_fetch_rejects_forged_advertised_readback(self):
        result = self.snapshot()
        self.tool.publish(self.out, execute=True)
        expected = self.tool.remote_state(str(self.remote))
        git(self.remote, 'update-ref', f'refs/tags/{self.tag}', self.base)
        with patch.object(self.tool, 'remote_state', return_value=expected):
            with self.assertRaisesRegex(self.tool.SafetyError, 'readback'):
                self.tool.prune(self.out, execute=True)
        self.assertEqual(git(self.remote, 'rev-parse', 'refs/heads/unmerged', check=False), self.tip)

    def test_fetch_race_removes_partial_output_and_keeps_remote_work(self):
        original_git = self.tool._git
        def racing_git(repo, *args, **kwargs):
            result = original_git(repo, *args, **kwargs)
            if args and args[0] == 'fetch':
                git(self.remote, 'update-ref', 'refs/heads/unmerged', self.base)
            return result
        with patch.object(self.tool, '_git', racing_git):
            with self.assertRaisesRegex(self.tool.SafetyError, 'changed|mismatch'):
                self.snapshot()
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob('.cbz-snapshot-*')), [])
        self.assertEqual(git(self.remote, 'rev-parse', 'refs/heads/duplicate'), self.tip)

    def test_default_branch_change_refuses_prune(self):
        self.snapshot()
        self.tool.publish(self.out, execute=True)
        git(self.remote, 'symbolic-ref', 'HEAD', 'refs/heads/duplicate')
        before = git(self.remote, 'show-ref')
        with self.assertRaises(self.tool.SafetyError):
            self.tool.prune(self.out, execute=True)
        self.assertEqual(git(self.remote, 'show-ref'), before)

    def test_existing_output_and_tag_are_never_overwritten(self):
        self.snapshot()
        manifest = (self.out / 'manifest.json').read_bytes()
        with self.assertRaisesRegex(self.tool.SafetyError, 'exists'):
            self.snapshot()
        self.assertEqual((self.out / 'manifest.json').read_bytes(), manifest)
        self.tool.publish(self.out, execute=True)
        with self.assertRaisesRegex(self.tool.SafetyError, 'exists'):
            self.tool.snapshot(str(self.remote), self.root / 'other', self.tag, execute=True)

    def test_local_archive_tree_tampering_is_detected(self):
        self.snapshot()
        archive = self.out / 'repository.git'
        git(archive, 'update-ref', f'refs/tags/{self.tag}', self.base)
        with self.assertRaises(self.tool.SafetyError):
            self.tool.verify_snapshot(self.out)

    def test_credential_urls_rejected_without_echoing_secret(self):
        secret = 'secret-token-abc'
        with self.assertRaises(self.tool.SafetyError) as ctx:
            self.tool.snapshot(f'https://user:{secret}@example.invalid/repo', self.out, self.tag)
        self.assertNotIn(secret, str(ctx.exception))

    def test_cli_default_json_does_not_create_snapshot(self):
        result = subprocess.run([sys.executable, '-B', str(TOOL), 'snapshot',
                                 '--remote', str(self.remote), '--output', str(self.out),
                                 '--archive-tag', self.tag], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['mode'], 'dry-run')
        self.assertFalse(self.out.exists())

    def test_receipt_hash_tampering_is_detected(self):
        self.snapshot()
        path = self.out / 'receipt.json'
        receipt = json.loads(path.read_text())
        receipt['manifest_sha256'] = '0' * 64
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(self.tool.SafetyError, 'manifest hash mismatch'):
            self.tool.verify_snapshot(self.out)

    def test_remote_inventory_does_not_spawn_one_git_process_per_ref(self):
        # Thousands of refs must not multiply remote-inventory subprocesses.
        with patch.object(subprocess, 'run', wraps=subprocess.run) as called:
            state = self.tool.remote_state(str(self.remote))
        self.assertEqual(state['refs']['refs/heads/unmerged'], self.tip)
        self.assertLessEqual(called.call_count, 2)

    def test_invalid_ref_names_are_refused(self):
        for name in ('refs/heads/a..b', 'refs/heads/.hidden', 'refs/heads/a.lock',
                     'refs/heads/a b', 'refs/heads/a@{b', 'refs/heads/a\\b',
                     'refs/heads/a//b', 'refs/heads/trailing.', 'refs/heads/foo\nbar',
                     'refs/heads/a:b', 'refs/heads/a?b', 'refs/heads/a[b'):
            with self.subTest(name=name):
                with self.assertRaises(self.tool.SafetyError):
                    self.tool._ref(name)

    def test_final_readback_rejects_changes_instead_of_partial_success(self):
        self.snapshot()
        self.tool.publish(self.out, execute=True)
        real = self.tool.remote_state
        count = 0
        def changed(remote):
            nonlocal count
            count += 1
            state = real(remote)
            if count >= 2:
                state['refs']['refs/heads/concurrent'] = self.tip
            return state
        with patch.object(self.tool, 'remote_state', changed):
            with self.assertRaisesRegex(self.tool.SafetyError, 'final|changed|mismatch'):
                self.tool.prune(self.out, execute=True)


if __name__ == '__main__':
    unittest.main()
