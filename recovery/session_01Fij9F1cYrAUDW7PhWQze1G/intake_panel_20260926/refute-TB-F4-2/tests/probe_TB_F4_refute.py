"""Skeptic probe for finding TB-F4 (trust-boundary, refutation lens). No network.

TB-F4 supplies no probe. These controls pin the checker-side facts the finding
leans on, against a COPY of tools/public_intake_check.py from origin/main:
  1. a fork PR that adds a workflow file whose job is named 'public-intake' is
     REJECTED by the trusted checker (exit 1); the checker never emits a green
     for it, so any same-named green check must come from outside the checker;
  2. an API rate-limit (HTTP 403/429) is fail-closed: exit 1, REJECTED, not green;
  3. the request budget of a maximal valid package (50 files, 8 sources on the
     default branch) is 95 calls, and the maintainer-engineering lane costs 3.
"""
import base64
import contextlib
import io
import json
import os
import tempfile
import urllib.error
from pathlib import Path
from unittest import mock
import unittest

from test_public_intake import IntakeTests, FakeAPI, REPO, HEAD, BASE, blob, sha, intake


class ProbeTBF4(IntakeTests):
    def cli(self):
        with tempfile.TemporaryDirectory() as folder:
            event_file = Path(folder) / 'event.json'
            event_file.write_text(json.dumps(self.event))
            env = {'GITHUB_EVENT_NAME': 'pull_request_target', 'GITHUB_REPOSITORY': REPO, 'GH_TOKEN': 'test-only'}
            with mock.patch.dict(os.environ, env), mock.patch.object(intake, 'API', return_value=self.api), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                code = intake.main(['--event', str(event_file)])
            return code, json.loads(output.getvalue())

    def test_fork_pr_adding_same_named_workflow_is_refused_by_checker(self):
        # Outsider (head repo visitor/main) adds .github/workflows/x.yml with a job named public-intake.
        wf = b"on: pull_request\njobs:\n  public-intake:\n    runs-on: ubuntu-latest\n    steps: [{run: 'true'}]\n"
        route = f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'
        self.api.routes[route] = [{'filename': '.github/workflows/public-intake.yml', 'status': 'added', 'sha': blob(wf)}]
        self.pr['changed_files'] = 1
        code, out = self.cli()
        self.assertEqual(code, 1)
        self.assertEqual(out['result'], 'REJECTED')
        # safe_path refuses the leading-dot component before the incoming-only rule is reached.
        self.assertIn(out['reason'], ('unsafe path', 'incoming-only edits required'))
        # Also when mixed into an otherwise valid package: still one refusal, exit 1.
        self.api.routes[route] = self.api.routes[route] + [
            {'filename': self.package + n, 'status': 'added', 'sha': blob(r)} for n, r in self.files.items()]
        self.pr['changed_files'] = len(self.api.routes[route])
        code, out = self.cli()
        self.assertEqual((code, out['result']), (1, 'REJECTED'))

    def test_rate_limited_api_is_fail_closed_not_green(self):
        # GITHUB_TOKEN budget exhausted: GitHub answers 403 (or 429) with rate-limit headers.
        for status in (403, 429):
            for route in (f'/repos/{REPO}/pulls/7', f'/repos/d6g8k5htny-coder/Math-/git/blobs/{blob(self.source)}'):
                err = urllib.error.HTTPError('https://api.github.com' + route, status, 'rate limited',
                                             {'x-ratelimit-remaining': '0'}, io.BytesIO(b'{"message":"API rate limit exceeded"}'))
                saved = self.api.routes[route]
                self.api.routes[route] = err
                code, out = self.cli()
                self.assertEqual(code, 1, (status, route))
                self.assertEqual(out['result'], 'REJECTED')
                self.assertNotIn('rate limit exceeded', json.dumps(out))  # body never printed
                self.api.routes[route] = saved
        self.assertEqual(self.cli()[0], 0)  # restored routes still pass: the probe's fixture is sound

    def test_request_budget_of_maximal_package_and_engineering_lane(self):
        # 50 files (48 artifacts + RESULT.md + IDENTITY.json) and 8 sources all reachable from main.
        self.raw = {'RESULT.md': b'# R\nscientific effect NONE\n'}
        for i in range(48):
            self.raw[f'a{i:02d}.txt'] = f'artifact {i}\n'.encode()
        sources = []
        self.source = b'public source 0\n'  # save() re-installs source 0 from self.source
        for i in range(8):
            commit = f'{i + 1:040x}'  # sha40 refuses the all-zero SHA
            payload = f'public source {i}\n'.encode()
            sources.append({'repository': 'd6g8k5htny-coder/Math-', 'path': f'src/{i}.json', 'commit': commit, 'sha256': sha(payload)})
            self.install_source('d6g8k5htny-coder/Math-', commit, f'src/{i}.json', payload)
        self.manifest['sources'] = sources
        self.save()
        result = self.run_check()
        self.assertEqual((result['artifact_files'], result['verified_sources']), (49, 8))
        self.assertEqual(len(self.api.calls), 1 + 1 + 2 + 50 + 8 * 5 + 1)  # == 95
        # Maintainer-engineering lane (same repo, no label, no incoming/): 3 calls.
        self.api.calls.clear()
        self.pr['head']['repo']['full_name'] = REPO
        self.api.routes[f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'] = [
            {'filename': 'tools/x.py', 'status': 'modified', 'sha': 'd' * 40}]
        self.pr['changed_files'] = 1
        self.assertEqual(self.run_check()['lane'], 'maintainer-engineering')
        self.assertEqual(len(self.api.calls), 3)


if __name__ == '__main__':
    unittest.main()
