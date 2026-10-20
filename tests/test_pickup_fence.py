"""Which writes a stale worker can and cannot make to a shared branch.

Standard library only. The test builds a bare repository in a temporary directory, clones it
twice (worker A and worker B), lets B move the branch, and then records which of A's pushes
succeed. It is a demonstration of what the transport actually prevents, as opposed to what a
claim comment only discourages; it is not a model of GitHub's merge API, whose
``expected_head_sha`` guard is documented separately.

Run from the repository root:  python3 -m unittest tests.test_pickup_fence
"""
import os
import subprocess
import tempfile
import unittest


def git(*args, cwd, check=True):
    env = dict(os.environ)
    env.update({
        'GIT_AUTHOR_NAME': 'test', 'GIT_AUTHOR_EMAIL': 'test@example.invalid',
        'GIT_COMMITTER_NAME': 'test', 'GIT_COMMITTER_EMAIL': 'test@example.invalid',
        'GIT_CONFIG_NOSYSTEM': '1', 'HOME': cwd,
    })
    r = subprocess.run(['git', *args], cwd=cwd, env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise AssertionError('git %s failed:\n%s%s' % (' '.join(args), r.stdout, r.stderr))
    return r


class PickupFenceTest(unittest.TestCase):
    """Worker A holds a stale view of ``work``; worker B has pushed a newer commit."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.remote = os.path.join(root, 'remote.git')
        git('init', '--bare', '-q', '-b', 'main', self.remote, cwd=root)
        seed = os.path.join(root, 'seed')
        git('clone', '-q', self.remote, seed, cwd=root)
        with open(os.path.join(seed, 'f.txt'), 'w') as f:
            f.write('base\n')
        git('add', 'f.txt', cwd=seed)
        git('commit', '-q', '-m', 'base', cwd=seed)
        git('push', '-q', 'origin', 'HEAD:refs/heads/work', cwd=seed)
        self.a = os.path.join(root, 'a')
        self.b = os.path.join(root, 'b')
        for d in (self.a, self.b):
            git('clone', '-q', '-b', 'work', self.remote, d, cwd=root)
        self.old_head = git('rev-parse', 'HEAD', cwd=self.a).stdout.strip()
        # B advances the branch; A does not fetch.
        with open(os.path.join(self.b, 'f.txt'), 'a') as f:
            f.write('b\n')
        git('commit', '-q', '-am', 'b moves work', cwd=self.b)
        git('push', '-q', 'origin', 'work', cwd=self.b)
        self.new_head = git('rev-parse', 'HEAD', cwd=self.b).stdout.strip()
        # A commits on its stale view.
        with open(os.path.join(self.a, 'f.txt'), 'a') as f:
            f.write('a\n')
        git('commit', '-q', '-am', 'a, stale', cwd=self.a)

    def tearDown(self):
        self.tmp.cleanup()

    def remote_head(self):
        return git('rev-parse', 'refs/heads/work', cwd=self.remote).stdout.strip()

    def test_plain_push_from_stale_view_is_prevented(self):
        r = git('push', 'origin', 'work', cwd=self.a, check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('[rejected]', r.stderr + r.stdout)
        self.assertEqual(self.remote_head(), self.new_head)

    def test_force_with_lease_on_stale_expected_head_is_prevented(self):
        r = git('push', '--force-with-lease=refs/heads/work:' + self.old_head, 'origin', 'work', cwd=self.a, check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('stale info', r.stderr + r.stdout)
        self.assertEqual(self.remote_head(), self.new_head)

    def test_force_with_lease_on_current_expected_head_is_allowed(self):
        git('fetch', '-q', 'origin', cwd=self.a)
        git('reset', '-q', '--hard', 'origin/work', cwd=self.a)
        with open(os.path.join(self.a, 'f.txt'), 'a') as f:
            f.write('a, fresh\n')
        git('commit', '-q', '-am', 'a, fresh', cwd=self.a)
        mine = git('rev-parse', 'HEAD', cwd=self.a).stdout.strip()
        git('push', '--force-with-lease=refs/heads/work:' + self.new_head, 'origin', 'work', cwd=self.a)
        self.assertEqual(self.remote_head(), mine)

    def test_plain_force_is_not_prevented(self):
        """A bare ``--force`` overwrites B's commit: this write is only discouraged by the workflow."""
        git('push', '--force', '-q', 'origin', 'work', cwd=self.a)
        self.assertEqual(self.remote_head(), git('rev-parse', 'HEAD', cwd=self.a).stdout.strip())
        self.assertNotEqual(self.remote_head(), self.new_head)

    def test_a_claim_record_prevents_nothing_by_itself(self):
        """A claim written beside the branch (here, a note in the remote) does not stop a push."""
        git('fetch', '-q', 'origin', cwd=self.a)
        git('reset', '-q', '--hard', 'origin/work', cwd=self.a)
        with open(os.path.join(self.a, 'CLAIM.txt'), 'w') as f:
            f.write('claimed by B until 23:59Z\n')
        git('add', 'CLAIM.txt', cwd=self.a)
        git('commit', '-q', '-m', 'a ignores the claim', cwd=self.a)
        git('push', '-q', 'origin', 'work', cwd=self.a)
        self.assertEqual(self.remote_head(), git('rev-parse', 'HEAD', cwd=self.a).stdout.strip())


if __name__ == '__main__':
    unittest.main()
