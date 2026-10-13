"""Refutation probes for TB-F2 (skeptic lens: trust-boundary). No network.

Run with INTAKE_CHECKER=<copy of tools/public_intake_check.py>. Each test states
what it establishes; all are expected to PASS on the current checker.
"""
import contextlib, http.client, importlib.util, io, json, os, subprocess, sys, tempfile
from pathlib import Path
from unittest import mock
import unittest

CHECKER = Path(os.environ['INTAKE_CHECKER'])
spec = importlib.util.spec_from_file_location('intake', CHECKER)
intake = importlib.util.module_from_spec(spec); spec.loader.exec_module(intake)
REPO = 'd6g8k5htny-coder/main'


class FakeAPI:
    def __init__(self, routes): self.routes = routes
    def get(self, route):
        r = self.routes[route]
        if isinstance(r, Exception): raise r
        return r


def run_cli(api):
    with tempfile.TemporaryDirectory() as folder:
        ev = Path(folder) / 'event.json'
        ev.write_text(json.dumps({'number': 7, 'repository': {'full_name': REPO}, 'pull_request': {'head': {'sha': 'a' * 40}}}))
        env = {'GITHUB_EVENT_NAME': 'pull_request_target', 'GITHUB_REPOSITORY': REPO, 'GH_TOKEN': 'test-only'}
        with mock.patch.dict(os.environ, env), mock.patch.object(intake, 'API', return_value=api), contextlib.redirect_stdout(io.StringIO()) as out:
            code = intake.main(['--event', str(ev)])
        return code, out.getvalue()


class ContentLengthTruncation(unittest.TestCase):
    """Observed 2026-09-26 (one GET each, through the session proxy's TLS
    tunnel): api.github.com framed /repos/{r}, /git/trees/{sha}?recursive=1 and
    /pulls/{n}/files with Content-Length; /pulls/{n} with Transfer-Encoding:
    chunked. urllib's HTTPResponse.read(amt) on a cut-off Content-Length body
    returns a SHORT READ and does not raise IncompleteRead (http.client comment:
    'it might break compatibility'); only a cut-off chunked body raises
    IncompleteRead. So for the Content-Length-framed endpoints the checker already
    hits strict_json -> JSONDecodeError (a ValueError) -> the documented JSON
    REJECTED verdict, exit 1. The chunked /pulls/{n} read is the one endpoint
    where the finder's IncompleteRead escape is reachable on real transport; it
    is a GitHub/runner-side fault, not something a PR author controls."""
    def test_truncated_content_length_body_yields_json_rejected_verdict(self):
        api = intake.API('t')
        class Resp:
            def read(self, n=-1): return b'{"number": 7, "state": "op'   # cut mid-body
            def __enter__(self): return self
            def __exit__(self, *a): return False
        class Opener:
            def open(self, request, timeout=None): return Resp()
        api.opener = Opener()
        with self.assertRaises(ValueError):          # JSONDecodeError is a ValueError
            api.get(f'/repos/{REPO}/pulls/7')
        with tempfile.TemporaryDirectory() as folder:
            ev = Path(folder) / 'event.json'
            ev.write_text(json.dumps({'number': 7, 'repository': {'full_name': REPO}, 'pull_request': {'head': {'sha': 'a' * 40}}}))
            env = {'GITHUB_EVENT_NAME': 'pull_request_target', 'GITHUB_REPOSITORY': REPO, 'GH_TOKEN': 'test-only'}
            with mock.patch.dict(os.environ, env), mock.patch.object(intake, 'API', return_value=api), contextlib.redirect_stdout(io.StringIO()) as out:
                code = intake.main(['--event', str(ev)])
        self.assertEqual(code, 1)
        verdict = json.loads(out.getvalue())
        self.assertEqual(verdict['result'], 'REJECTED')
        self.assertEqual(verdict['scientific_effect'], 'NONE')


class EscapeIsStillExitOne(unittest.TestCase):
    """The finder's two escapes reproduce, and neither can produce a zero exit
    or a printed pass. Checked at the process level exactly as the workflow runs
    it (python3 -B -I ... ; raise SystemExit(main()))."""
    def _subprocess(self, kind):
        driver = f'''
import importlib.util, json, os, sys, tempfile, http.client
from pathlib import Path
spec = importlib.util.spec_from_file_location('intake', {str(CHECKER)!r}); intake = importlib.util.module_from_spec(spec); spec.loader.exec_module(intake)
class FakeAPI:
    def __init__(self, token): pass
    def get(self, route):
        if {kind!r} == 'list': return []
        raise http.client.IncompleteRead(b'', 10)
intake.API = FakeAPI
d = tempfile.mkdtemp(); ev = Path(d)/'event.json'
ev.write_text(json.dumps({{'number': 7, 'repository': {{'full_name': {REPO!r}}}, 'pull_request': {{'head': {{'sha': 'a'*40}}}}}}))
os.environ.update({{'GITHUB_EVENT_NAME': 'pull_request_target', 'GITHUB_REPOSITORY': {REPO!r}, 'GH_TOKEN': 'test-only'}})
raise SystemExit(intake.main(['--event', str(ev)]))
'''
        return subprocess.run([sys.executable, '-B', '-I', '-c', driver], capture_output=True, text=True)

    def test_non_dict_pr_body_exits_1_prints_nothing_on_stdout_traceback_has_no_body(self):
        p = self._subprocess('list')
        self.assertEqual(p.returncode, 1)
        self.assertEqual(p.stdout, '')                      # no pass line, no verdict line
        self.assertIn('AttributeError', p.stderr)
        self.assertNotIn('test-only', p.stderr)             # token never appears

    def test_incomplete_read_exits_1_prints_nothing_on_stdout(self):
        p = self._subprocess('incomplete')
        self.assertEqual(p.returncode, 1)
        self.assertEqual(p.stdout, '')
        self.assertIn('IncompleteRead', p.stderr)
        self.assertNotIn('test-only', p.stderr)

    def test_in_process_escape_matches_finder(self):
        with self.assertRaises(AttributeError):
            run_cli(FakeAPI({f'/repos/{REPO}/pulls/7': []}))
        with self.assertRaises(http.client.HTTPException):
            run_cli(FakeAPI({f'/repos/{REPO}/pulls/7': http.client.IncompleteRead(b'', 10)}))


if __name__ == '__main__':
    unittest.main()
