"""Refutation probe for TB-F4 (Skeptic #3, trust-boundary). No network.

TB-F4 is an INFO observation about the operating envelope (ruleset binding,
same-named checks from a fork's pull_request workflow, 'edited' re-runs and the
GITHUB_TOKEN hourly budget). It offers no probe. This file pins the parts that
ARE checker-observable so the verdict rests on executed controls:

  (b) checker side: a fork PR that adds .github/workflows/<x>.yml (declaring a job
      named public-intake) is refused, alone or beside a valid package.
  (c) a rate-limit HTTPError (403 / 429) at ANY API call produces exit 1 and a
      REJECTED line -- fail-closed denial, never a green.
  (c') request-count measurement: the finding's "~90-request job" figure versus
      the actual worst case with reachable()'s branch scan.
"""
import contextlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from test_public_intake import IntakeTests, FakeAPI, REPO, MATH, HEAD, BASE, SOURCE, sha, blob, compare, intake, png


def rate_limited(code=403):
    return urllib.error.HTTPError('https://api.github.com/x', code, 'rate limit exceeded', {'X-RateLimit-Remaining': '0'}, None)


class ProbeTBF4(IntakeTests):

    def files_route(self):
        return f'/repos/{REPO}/pulls/7/files?per_page=100&page=1'

    # ---- (b) checker side -------------------------------------------------

    def test_b_fork_adding_same_named_workflow_is_refused(self):
        # An outsider's fork adds a workflow whose job is named public-intake. The
        # trusted pull_request_target run refuses the PR before any blob is read.
        self.api.routes[self.files_route()] = [
            {'filename': '.github/workflows/public-intake.yml', 'status': 'added', 'sha': 'd' * 40}]
        self.pr['changed_files'] = 1
        self.reject('unsafe path|incoming-only')  # dot-leading component is refused by safe_path before the incoming test
        self.assertFalse(any('/git/blobs/' in c or '/git/trees/' in c for c in self.api.calls))

    def test_b_fork_workflow_beside_valid_package_is_refused(self):
        rows = self.api.routes[self.files_route()]
        rows.append({'filename': '.github/workflows/public-intake.yml', 'status': 'added', 'sha': 'd' * 40})
        self.pr['changed_files'] = len(rows)
        self.reject('unsafe path|incoming-only')  # dot-leading component is refused by safe_path before the incoming test

    def test_b_same_repo_workflow_edit_without_label_is_engineering_lane(self):
        # Requires write access to the pillar (cannot be created by an outsider).
        self.pr['head']['repo']['full_name'] = REPO
        self.api.routes[self.files_route()] = [
            {'filename': '.github/workflows/public-intake.yml', 'status': 'modified', 'sha': 'd' * 40}]
        self.pr['changed_files'] = 1
        self.assertEqual(self.run_check()['lane'], 'maintainer-engineering')

    # ---- (c) rate limit is fail-closed at every call ----------------------

    def cli(self):
        event_file = Path(self.folder) / 'event.json'
        event_file.write_text(json.dumps(self.event))
        env = {'GITHUB_EVENT_NAME': 'pull_request_target', 'GITHUB_REPOSITORY': REPO, 'GH_TOKEN': 'test-only'}
        with mock.patch.dict(os.environ, env), mock.patch.object(intake, 'API', return_value=self.api), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            code = intake.main(['--event', str(event_file)])
        return code, output.getvalue()

    def test_c_rate_limit_at_any_call_is_exit_1_never_green(self):
        with tempfile.TemporaryDirectory() as self.folder:
            code, out = self.cli()
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out)['lane'], 'incoming')
            sequence = list(self.api.calls)  # the exact routes a green run touches
            self.assertGreater(len(sequence), 5)
            for index, route in enumerate(sequence):
                for status in (403, 429):
                    with self.subTest(index=index, route=route, status=status):
                        original = self.api.routes[route]
                        self.api.routes[route] = rate_limited(status)
                        code, out = self.cli()
                        self.api.routes[route] = original
                        self.assertEqual(code, 1)
                        parsed = json.loads(out)
                        self.assertEqual(parsed['result'], 'REJECTED')
                        self.assertEqual(parsed['scientific_effect'], 'NONE')
                        self.assertIn('rate limit', parsed['reason'])

    # ---- (c') request budget: the finding's "~90" versus measured ----------

    def test_cprime_request_count_minimal_and_worst_case(self):
        self.api.calls.clear()
        self.run_check()
        minimal = len(self.api.calls)
        # Maximal legitimate package: 50 files, 8 distinct sources, every source
        # reachable only from the last of 300 listed branches of the pillar.
        for i in range(50 - len(self.files)):
            self.raw[f'extra{i}.txt'] = b'x'
        names = ['main'] + [f'topic-{i}' for i in range(1, 300)]
        self.manifest['sources'] = []
        for k in range(8):
            commit = format(k + 1, 'x') * 40  # never the null SHA, which sha40 refuses
            raw = b'src' + bytes([k])
            self.manifest['sources'].append({'repository': MATH, 'path': f'p{k}.json', 'commit': commit, 'sha256': sha(raw)})
        self.save()  # install() re-registers sources[0] with default bytes; register ours afterwards so they win
        for k in range(8):
            commit = format(k + 1, 'x') * 40
            raw = b'src' + bytes([k])
            self.install_source(MATH, commit, f'p{k}.json', raw, branches=(), reachable_from=())
            for page in range(1, 4):
                self.api.routes[f'/repos/{MATH}/branches?per_page=100&page={page}'] = [{'name': n} for n in names[(page - 1) * 100:page * 100]]
            for n in names:
                self.api.routes[f'/repos/{MATH}/compare/{n}...{commit}?per_page=1'] = (
                    compare(commit) if n == names[-1] else compare(commit, 'diverged', 2, BASE))
        self.api.calls.clear()
        result = self.run_check()
        self.assertEqual(result['verified_sources'], 8)
        self.assertEqual(result['artifact_files'], 49)
        worst = len(self.api.calls)
        print(f'\n[probe] requests per run: minimal={minimal} worst_case_legitimate={worst} '
              f'(GITHUB_TOKEN budget is 1000/hour/repository)')
        self.assertLess(minimal, 90)
        self.assertGreater(worst, 1000)


if __name__ == '__main__':
    unittest.main()
