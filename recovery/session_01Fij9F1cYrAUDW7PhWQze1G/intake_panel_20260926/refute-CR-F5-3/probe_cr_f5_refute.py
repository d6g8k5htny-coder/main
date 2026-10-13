"""Skeptic probe for CR-F5 (bidi-override / terminal-escape bytes in text artifacts).

INTAKE_TREE selects which checker copy is exercised: current (origin/main dbb9dcf),
base_e495 (the finder's stated base e495837) or patched (origin/main + isolated F5 patch).
INTAKE_EXPECT is 'green' for an unpatched copy and 'refused' for the patched copy.

The class subclasses the house IntakeTests fixture directly (the finder's probe copied
five methods by name and so errors on the HEAD fixture, which added install_source).
No network: the FakeAPI serves a dict of routes exactly as tests/test_public_intake.py does.
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TREE = Path(__file__).resolve().parent / os.environ.get('INTAKE_TREE', 'current')
EXPECT = os.environ.get('INTAKE_EXPECT', 'green')
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec)
spec.loader.exec_module(house)
intake = house.intake
assert Path(intake.__file__).resolve() == (TREE / 'tools/public_intake_check.py').resolve()

RLO, PDF, RLM = '‮', '‬', '‏'
ESC = b'\x1b'


class F5Probe(house.IntakeTests):
    def green_or_refused(self, message):
        if EXPECT == 'green':
            result = self.run_check()
            self.assertEqual(result['lane'], 'incoming')
            self.assertEqual(result['scientific_effect'], 'NONE')
            return result
        self.reject(message)

    # ---- Reproduction of the finder's F5a / F5b ------------------------------------------
    def test_F5a_bidi_override_in_RESULT_md(self):
        self.raw['RESULT.md'] = f'Scientific effect: {RLO}TPECCA{PDF} NONE.\n'.encode()
        self.save()
        self.green_or_refused('control or bidi')

    def test_F5b_terminal_escape_sequences_in_txt(self):
        self.raw['log.txt'] = ESC + b'[2J' + ESC + b'[H' + ESC + b']0;pwned\x07plain looking log\n'
        self.save()
        self.green_or_refused('control or bidi')

    # ---- What the checker itself decides is not display-dependent (all trees) -------------
    def test_bidi_cannot_forge_a_machine_checked_status_field(self):
        for field in ('scientific_effect', 'review_status'):
            with self.subTest(field=field):
                good = self.manifest[field]
                self.manifest[field] = RLO + good[::-1] + PDF  # renders as the accepted word
                self.save()
                self.reject('intake is review-only; no status promotion')
                self.manifest[field] = good

    def test_bidi_in_manifest_artifact_path_or_package_path_is_refused(self):
        self.save()
        self.manifest['artifacts'][0]['path'] = RLO + 'dm.TLUSER' + PDF
        self.files['IDENTITY.json'] = json.dumps(self.manifest).encode()
        self.install()
        self.reject('unsafe path')
        self.setUp()
        self.raw['notes' + RLO + '.md'] = b'x\n'
        self.save()
        self.reject('unsafe path')

    def test_escape_bytes_in_changed_filename_refused_before_package_checks(self):
        self.save()
        self.files['log\x1b[2J.txt'] = b'x\n'
        self.install()
        self.reject('invalid changed path')

    def test_checker_stdout_is_pure_ascii_and_never_echoes_submitted_bytes(self):
        # The only surface the checker itself writes is its one JSON line; CONTRIBUTING assigns
        # content inspection to the reviewer. Submitted bytes never reach the CI log.
        self.raw['RESULT.md'] = f'Scientific effect: {RLO}TPECCA{PDF} NONE.\n'.encode()
        self.raw['log.txt'] = ESC + b'[2J' + ESC + b']0;pwned\x07log\n'
        self.save()
        with tempfile.TemporaryDirectory() as folder:
            event_file = Path(folder) / 'event.json'
            event_file.write_text(json.dumps(self.event))
            env = {'GITHUB_EVENT_NAME': 'pull_request_target', 'GITHUB_REPOSITORY': house.REPO, 'GH_TOKEN': 'test-only'}
            with mock.patch.dict(os.environ, env), mock.patch.object(intake, 'API', return_value=self.api), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                code = intake.main(['--event', str(event_file)])
        text = output.getvalue()
        self.assertEqual(code, 0 if EXPECT == 'green' else 1)
        self.assertTrue(text.isascii(), 'checker output must be ASCII')
        self.assertNotIn('\x1b', text)
        self.assertNotIn('pwned', text)
        self.assertNotIn('TPECCA', text)
        json.loads(text)

    # ---- Legitimate content the proposed patch refuses (wrong rejection) -------------------
    def test_patch_refuses_rtl_prose_with_right_to_left_mark(self):
        # U+200F is the standard way to keep punctuation on the correct side of a Hebrew or
        # Arabic name in otherwise-LTR prose; it is not an override.
        self.raw['RESULT.md'] = f'# Result\nReviewer: אורי{RLM} (Uri). Scientific effect NONE.\n'.encode('utf-8')
        self.save()
        self.green_or_refused('control or bidi')

    def test_patch_refuses_ansi_colored_test_log_in_txt(self):
        # `pytest --color=yes | tee regeneration/test_run.txt` produces exactly this.
        self.raw['regeneration/test_run.txt'] = b'tests/test_x.py ' + ESC + b'[32mPASSED' + ESC + b'[0m\n' + ESC + b'[32m2 passed' + ESC + b'[0m in 0.1s\n'
        self.save()
        self.green_or_refused('control or bidi')

    def test_patch_refuses_form_feed_page_break_in_txt(self):
        self.raw['notes.txt'] = b'page one\n\x0cpage two\n'
        self.save()
        self.green_or_refused('control or bidi')

    def test_patch_refuses_the_repository_own_mojibake_export_bytes(self):
        # history/2025/body.original.txt in this repository holds U+0081/U+008F/U+009D from a
        # double-encoded Drive export; a contributor re-exporting the same way is refused.
        self.raw['excerpt.txt'] = 'Stelle ratio Î³â‚\u0081/Î³â‚‚ = âˆ’ 1/2\n'.encode('utf-8')
        self.save()
        self.green_or_refused('control or bidi')

    # ---- Unicode prose is accepted in every tree (control) ----------------------------------
    def test_unicode_prose_control(self):
        self.raw['RESULT.md'] = 'Résumé: α ≤ β, 数学, \U0001f642, tabs\tand CRLF\r\nscientific effect NONE\n'.encode()
        self.save()
        self.assertEqual(self.run_check()['lane'], 'incoming')


if __name__ == '__main__':
    print(f'[probe] tree={TREE.name} expect={EXPECT} checker={intake.__file__}', file=sys.stderr)
    unittest.main()
