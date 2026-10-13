"""Skeptic probe for CR-F3. INTAKE_ROOT holds tools/public_intake_check.py and tests/test_public_intake.py;
the house fixture (fake API routes, no network) is reused verbatim.
INTAKE_PATCHED: '' = unpatched checker, 'finder' = the finder's combined patch, 'collision' = a
collision-only hardening (refuse case-folded path twins, nothing else).

Part A reproduces the finder's F3a/F3b/F3c acceptances (or the patched refusals).
Part B models the case-folding checkout the finding relies on: git checkout unlinks and recreates a
colliding path, so one on-disk entry per folded name holds the LATER entry's bytes and any case
variant of the name reads it. It records what a reviewer's exact-bytes step observes there, including
the strongest decoy (policy-conformant, differing only in its source pin).
Part C checks a patch for wrong rejections of ordinary packages.
"""
import hashlib, importlib.util, json, os, sys, unittest
from pathlib import Path

ROOT = Path(os.environ['INTAKE_ROOT']).resolve()
spec = importlib.util.spec_from_file_location('house_tests', ROOT / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec)
spec.loader.exec_module(house)
intake = house.intake
assert Path(intake.__file__).resolve() == (ROOT / 'tools/public_intake_check.py').resolve(), intake.__file__
PATCHED = os.environ.get('INTAKE_PATCHED', '')
SOURCE2 = 'e' * 40


class Disk:
    """A case-insensitive, case-preserving directory after a git checkout of `files`."""
    def __init__(self, files):
        self.entries = {}
        for path in sorted(files):  # git index order: bytewise
            self.entries[path.lower()] = files[path]  # unlink + create: later entry wins

    def read(self, name):
        return self.entries[name.lower()]

    def __len__(self):
        return len(self.entries)

    def __contains__(self, name):
        return name.lower() in self.entries


class Probes(unittest.TestCase):
    setUp = house.IntakeTests.setUp
    save = house.IntakeTests.save
    install = house.IntakeTests.install
    run_check = house.IntakeTests.run_check
    reject = house.IntakeTests.reject
    if hasattr(house.IntakeTests, 'install_source'):
        install_source = house.IntakeTests.install_source

    def green(self):
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        self.assertEqual(result['scientific_effect'], 'NONE')
        return result

    def decoy(self, **changes):
        return json.dumps(dict(self.manifest, **changes)).encode()

    # ---- Part A ----
    def test_A_F3a_lowercase_identity_json(self):
        self.raw['identity.json'] = self.decoy(review_status='ACCEPTED', scientific_effect='ACCEPT')
        self.save()
        if PATCHED:
            self.reject('collision|look-alike')
        else:
            self.assertEqual(self.green()['artifact_files'], 3)  # the decoy is an ordinary counted artifact

    def test_A_F3b_result_md_case_twin(self):
        self.raw['Result.md'] = b'# Result\nThis result is ACCEPTED.\n'
        self.save()
        if PATCHED:
            self.reject('collision|look-alike')
        else:
            self.green()

    def test_A_F3c_nested_identity_json(self):
        self.raw['data/IDENTITY.json'] = self.decoy(review_status='ACCEPTED')
        self.save()
        if PATCHED == 'finder':
            self.reject('look-alike')
        else:
            self.green()  # not a collision on any filesystem; an ordinary nested JSON artifact

    # ---- Part B: the checkout the finding relies on ----
    def test_B_F3b_twin_breaks_the_manifest_row_a_reviewer_verifies(self):
        self.raw['Result.md'] = b'# Result\nThis result is ACCEPTED.\n'
        self.save()
        disk = Disk(self.files)
        self.assertEqual(len(disk), len(self.files) - 1)
        rows = {r['path']: r['sha256'] for r in json.loads(disk.read('IDENTITY.json'))['artifacts']}
        on_disk = hashlib.sha256(disk.read('RESULT.md')).hexdigest()
        self.assertNotEqual(on_disk, rows['RESULT.md'])  # the exact-bytes step fails on this checkout
        self.assertEqual(on_disk, rows['Result.md'])     # ... and names a second entry the disk lacks

    def test_B_F3a_naive_decoy_contradicts_policy_on_its_face(self):
        self.raw['identity.json'] = self.decoy(review_status='ACCEPTED', scientific_effect='ACCEPT')
        self.save()
        seen = json.loads(Disk(self.files).read('IDENTITY.json'))
        self.assertEqual((seen['review_status'], seen['scientific_effect']), ('ACCEPTED', 'ACCEPT'))
        with self.assertRaisesRegex(ValueError, 'review-only'):  # the rule CONTRIBUTING states, re-applied
            intake.require(seen['scientific_effect'] == 'NONE' and seen['review_status'] == 'REVIEW_REQUIRED',
                           'intake is review-only; no status promotion')

    def test_B_F3a_best_decoy_is_self_consistent_on_a_case_folding_checkout(self):
        # Honest strongest case: the decoy keeps NONE/REVIEW_REQUIRED and only re-pins the source.
        # On the folding checkout it is self-consistent; the twin is visible only in the PR file list,
        # the CI result (artifact_files counts it), the tree API and any case-sensitive checkout.
        other = dict(self.manifest['sources'][0], commit=SOURCE2)
        self.raw['identity.json'] = self.decoy(sources=[other])
        self.save()
        if PATCHED:
            self.reject('collision|look-alike'); return
        result = self.green()
        self.assertEqual(result['artifact_files'], 3)
        seen = json.loads(Disk(self.files).read('IDENTITY.json'))
        self.assertEqual(seen['sources'][0]['commit'], SOURCE2)          # reviewer sees the unverified pin
        real = json.loads(self.files['IDENTITY.json'])
        self.assertEqual(real['sources'][0]['commit'], house.SOURCE)      # the landed exact-case file pins the verified one
        rows = {r['path']: r['sha256'] for r in seen['artifacts']}
        disk = Disk(self.files)
        self.assertTrue(all(hashlib.sha256(disk.read(p)).hexdigest() == d for p, d in rows.items()))
        self.assertNotIn('identity.json', rows)  # the decoy does not list itself; the real manifest does
        self.assertIn('identity.json', {r['path'] for r in real['artifacts']})

    def test_B_F3c_nested_identity_never_collides(self):
        self.raw['data/IDENTITY.json'] = self.decoy(review_status='ACCEPTED')
        self.save()
        disk = Disk(self.files)
        self.assertEqual(len(disk), len(self.files))
        self.assertEqual(json.loads(disk.read('IDENTITY.json'))['review_status'], 'REVIEW_REQUIRED')

    # ---- Part C: cost of the patches ----
    def test_C_nested_lowercase_result_md_is_an_ordinary_artifact(self):
        self.raw['experiments/result.md'] = b'# sub-result\nNONE\n'
        self.save()
        if PATCHED == 'finder':
            self.reject('look-alike')  # the finder's look-alike rule refuses a legitimate nested name
        else:
            self.green()

    def test_C_plain_package_stays_green(self):
        self.green()


if __name__ == '__main__':
    print(f'[refute-F3] root={ROOT} patched={PATCHED!r}', file=sys.stderr)
    unittest.main()
