"""Refutation probes for TB-F4 (trust-boundary lens). No network.

Each probe PASSES against the current checker and demonstrates fail-closed
behaviour, i.e. the absence of the defect the finding gestures at.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_public_intake import IntakeTests, FakeAPI, intake, REPO, HEAD  # noqa: E402

CHECKER = Path(__file__).resolve().parents[1] / 'tools/public_intake_check.py'


class RateLimitedAPI(FakeAPI):
    """Every request answers as GitHub does when GITHUB_TOKEN's hourly budget is spent."""
    def get(self, route):
        self.calls.append(route)
        raise urllib.error.HTTPError('https://api.github.com' + route, 403, 'rate limit exceeded', {}, None)


class ProbeTBF4(IntakeTests):

    # (b) checker side: a fork that ships a same-named workflow is refused, and the
    # trusted checker refuses to run at all under the fork workflow's own event.
    def test_b_fork_adding_same_named_workflow_is_refused(self):
        change = self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'][0]
        change['filename'] = '.github/workflows/public-intake.yml'
        self.reject('unsafe path|incoming-only')  # refused at safe_path (dot component) before the lane check

    def test_b_fork_adding_workflow_beside_valid_package_is_refused(self):
        rows = self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1']
        rows.append({'filename': '.github/workflows/public-intake.yml', 'status': 'added', 'sha': 'e' * 40})
        self.pr['changed_files'] = len(rows)
        self.reject('unsafe path|incoming-only')  # refused at safe_path (dot component) before the lane check

    def test_b_main_refuses_pull_request_event_via_cli(self):
        # Even if a fork's pull_request workflow copied and ran this very checker,
        # it exits 1: GITHUB_EVENT_NAME is 'pull_request', not 'pull_request_target'.
        with tempfile.TemporaryDirectory() as tmp:
            event = Path(tmp) / 'event.json'
            event.write_text(json.dumps(self.event))
            env = dict(os.environ, GITHUB_EVENT_NAME='pull_request', GH_TOKEN='x', GITHUB_REPOSITORY=REPO,
                       GITHUB_EVENT_PATH=str(event))
            run = subprocess.run([sys.executable, '-B', '-I', str(CHECKER), '--event', str(event)],
                                 env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        out = json.loads(run.stdout)
        self.assertEqual(out['result'], 'REJECTED')
        self.assertIn('pull_request_target', out['reason'])

    # (c) exhausted token budget: fail-closed, exit 1, never a green.
    def test_c_rate_limited_api_is_rejected_not_green(self):
        with mock_api(RateLimitedAPI()):
            code, out = run_main_with_event(self.event)
        self.assertEqual(code, 1)
        self.assertEqual(out['result'], 'REJECTED')
        self.assertEqual(out['scientific_effect'], 'NONE')
        self.assertIn('403', out['reason'])

    def test_c_request_count_minimal_package(self):
        self.run_check()
        # 3-file package, 1 source: the whole run's request count.
        self.assertLessEqual(len(self.api.calls), 20, self.api.calls)
        self.request_count_minimal = len(self.api.calls)


import contextlib
from unittest import mock


@contextlib.contextmanager
def mock_api(api):
    with mock.patch.object(intake, 'API', lambda token: api):
        yield


def run_main_with_event(event):
    import io
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'event.json'
        path.write_text(json.dumps(event))
        env = {'GITHUB_EVENT_NAME': 'pull_request_target', 'GH_TOKEN': 'x', 'GITHUB_REPOSITORY': REPO}
        with mock.patch.dict(os.environ, env):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = intake.main(['--event', str(path)])
    return code, json.loads(buffer.getvalue())


if __name__ == '__main__':
    unittest.main()
