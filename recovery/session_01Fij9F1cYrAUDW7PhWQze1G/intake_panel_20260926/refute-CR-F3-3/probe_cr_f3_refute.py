"""Skeptic re-run of CR-F3 against a fresh origin/main copy of tools/public_intake_check.py.

INTAKE_MODE=current -> assert what the current checker does (green for case twins / nested
                       look-alikes; the decoy bytes never influence the result dict).
INTAKE_MODE=patched -> the finder's F3 hunk (folded set + RESERVED_NAMES) applied to the same
                       fresh copy; assert refusal, and ALSO show the wrong rejection it introduces.
Fixture reused verbatim from the house tests; no network.
"""
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

MODE = os.environ.get('INTAKE_MODE', 'current')
TREE = Path(__file__).resolve().parent / MODE
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec)
spec.loader.exec_module(house)
intake = house.intake
assert Path(intake.__file__).resolve() == (TREE / 'tools/public_intake_check.py').resolve()


class Probes(unittest.TestCase):
    setUp = house.IntakeTests.setUp
    save = house.IntakeTests.save
    install = house.IntakeTests.install
    install_source = house.IntakeTests.install_source
    run_check = house.IntakeTests.run_check
    reject = house.IntakeTests.reject

    def green(self):
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        self.assertEqual(result['scientific_effect'], 'NONE')
        return result

    def wrong_green_now_refused(self, message):
        if MODE == 'current':
            return self.green()
        self.reject(message)

    def test_F3a_lowercase_identity_json_beside_manifest(self):
        decoy = dict(self.manifest, review_status='ACCEPTED', scientific_effect='ACCEPT')
        self.raw['identity.json'] = json.dumps(decoy).encode()
        self.save()
        result = self.wrong_green_now_refused('case-insensitive|look-alike')
        if MODE == 'current':
            # The decoy is inert: the result dict carries the same NONE/lane and the
            # schema check ran against the canonical IDENTITY.json (REVIEW_REQUIRED).
            self.assertEqual(result['artifact_files'], 3)
            canonical = json.loads(self.files['IDENTITY.json'])
            self.assertEqual(canonical['review_status'], 'REVIEW_REQUIRED')

    def test_F3b_result_md_case_twin(self):
        self.raw['Result.md'] = b'# Result\nThis result is ACCEPTED.\n'
        self.save()
        self.wrong_green_now_refused('case-insensitive|look-alike')

    def test_F3c_nested_identity_json(self):
        self.raw['data/IDENTITY.json'] = json.dumps(dict(self.manifest, review_status='ACCEPTED')).encode()
        self.save()
        self.wrong_green_now_refused('look-alike')

    def test_F3_distinct_names_control(self):
        self.raw['notes.md'] = b'notes\n'
        self.raw['data/values.json'] = b'[1, 2]\n'
        self.save()
        self.green()

    def test_F3_decoy_cannot_stand_in_for_missing_canonical_manifest(self):
        # Both modes: identity.json alone is NOT accepted as the manifest.
        self.files = dict(self.raw, **{'identity.json': json.dumps(self.manifest).encode()})
        self.install()
        self.reject('RESULT.md and IDENTITY.json required|look-alike')

    def test_F3_decoy_still_digest_bound_in_manifest(self):
        # Both modes: an unlisted or mis-digested identity.json is refused like any artifact.
        self.raw['identity.json'] = b'{"review_status": "ACCEPTED"}\n'
        self.save()
        self.manifest['artifacts'] = [a for a in self.manifest['artifacts'] if a['path'] != 'identity.json']
        self.files['IDENTITY.json'] = json.dumps(self.manifest).encode()
        self.install()
        self.reject('exactly cover|look-alike')

    def test_F3_patch_wrong_rejection_of_vendored_prior_manifest(self):
        # A successor package (CONTRIBUTING step 2: 'use a new ID and link the previous result')
        # that vendors the previous package's manifest and result for the reviewer's convenience.
        prior = dict(self.manifest)
        self.raw['prior/side24-identity-replay-20260926/IDENTITY.json'] = json.dumps(prior).encode()
        self.raw['prior/side24-identity-replay-20260926/RESULT.md'] = b'# Prior result\nscientific effect NONE\n'
        self.save()
        if MODE == 'current':
            self.green()
        else:
            self.reject('look-alike')  # the finder's patch refuses this legitimate package

    def test_F3_landed_packages_in_tree_have_no_case_twins(self):
        # Observation only: the two landed packages use distinct, case-unique names.
        root = Path('/home/user/main-review-refute-CR-F3-3/incoming')
        names = [str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()]
        self.assertEqual(len(names), len({n.lower() for n in names}))
        self.assertIn('side24-chart-claude-20260926/identity_detail.json', names)


if __name__ == '__main__':
    print(f'[probe] mode={MODE} checker={intake.__file__}', file=sys.stderr)
    unittest.main()
