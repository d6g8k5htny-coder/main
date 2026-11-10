import unittest
import rerun_all_main811 as target


def valid_run():
    return {'id': 42778804824, 'head_sha': 'b285dd7a31ad7b297abc9f2aa93267300111a4e9',
            'run_attempt': 2, 'status': 'completed', 'conclusion': 'failure', 'event': 'push',
            'path': '.github/workflows/workspace-landing.yml',
            'repository': {'full_name': 'd6g8k5htny-coder/main'}}


class ExactRunGuardTests(unittest.TestCase):
    def test_exact_failed_run_sends_once(self):
        calls = []
        target.guarded_rerun(valid_run(), lambda: calls.append('sent'))
        self.assertEqual(calls, ['sent'])

    def test_mismatches_and_missing_fields_refuse_without_sending(self):
        for name, changed in [('id', 1), ('head_sha', '0'*40), ('run_attempt', 1),
                              ('run_attempt', True), ('status', 'in_progress'),
                              ('conclusion', 'success'), ('event', 'workflow_dispatch'),
                              ('path', '.github/workflows/other.yml'),
                              ('repository', {'full_name': 'someone/else'}),
                              ('repository', None)]:
            with self.subTest(name=name, changed=changed):
                run = valid_run(); run[name] = changed; calls = []
                with self.assertRaises(ValueError): target.guarded_rerun(run, lambda: calls.append('sent'))
                self.assertEqual(calls, [])
        for run in ({}, None, [], {'id': 42778804824}):
            with self.subTest(run=run):
                calls = []
                with self.assertRaises(ValueError): target.guarded_rerun(run, lambda: calls.append('sent'))
                self.assertEqual(calls, [])

    def test_only_expected_runner_context_is_admitted(self):
        good = {'GITHUB_REPOSITORY': 'd6g8k5htny-coder/main',
                'GITHUB_REF': 'refs/heads/codex/cbz-architecture-observation-20261110',
                'GITHUB_EVENT_NAME': 'push', 'GITHUB_API_URL': 'https://api.github.com',
                'GITHUB_TOKEN': 'test-token'}
        self.assertEqual(target.guard_environment(good), 'test-token')
        for name, value in [('GITHUB_REPOSITORY', 'other/repo'), ('GITHUB_REF', 'refs/heads/main'),
                            ('GITHUB_EVENT_NAME', 'pull_request'), ('GITHUB_API_URL', 'https://example.org'),
                            ('GITHUB_TOKEN', '')]:
            with self.subTest(name=name):
                changed = dict(good); changed[name] = value
                with self.assertRaises(ValueError): target.guard_environment(changed)

    def test_sender_error_is_not_success(self):
        def fail(): raise RuntimeError('transport refused')
        with self.assertRaisesRegex(RuntimeError, 'transport refused'):
            target.guarded_rerun(valid_run(), fail)


if __name__ == '__main__':
    unittest.main(verbosity=2)
