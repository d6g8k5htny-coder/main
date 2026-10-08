"""Proposed W8b control (Grok Bot agent 2): tools/public_intake_check.py must bind every
IDENTITY.json artifact row to BOTH its declared byte count and its sha256. A declared
byte count that differs from the submitted bytes must be refused even when the digest
is right. Reuses the no-network harness of tests/test_public_intake.py without
re-running its 71 tests. Positive case: a valid package is accepted, at the function and
CLI level. Negatives assert the exact reason 'artifact identity mismatch', at the
function level and as main()'s handled REJECTED result with exit 1."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import test_public_intake as base  # module object only; its TestCase class is not collected here

intake = base.intake


class ArtifactIdentityControl(unittest.TestCase):
    setUp = base.IntakeTests.setUp
    save = base.IntakeTests.save
    install = base.IntakeTests.install
    install_source = base.IntakeTests.install_source
    run_check = base.IntakeTests.run_check

    def declare(self, index, **fields):
        self.manifest['artifacts'][index].update(fields)
        self.files['IDENTITY.json'] = json.dumps(self.manifest).encode()
        self.install()

    def refused(self):
        with self.assertRaises(ValueError) as caught:
            self.run_check()
        self.assertEqual(str(caught.exception), 'artifact identity mismatch')

    def run_cli(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        event = Path(temp.name) / 'event.json'
        event.write_text(json.dumps(self.event))
        out = io.StringIO()
        env = {'GITHUB_EVENT_NAME': 'pull_request_target', 'GITHUB_REPOSITORY': base.REPO, 'GH_TOKEN': 'unused'}
        with mock.patch.dict(os.environ, env), mock.patch.object(intake, 'API', lambda token: self.api), \
                contextlib.redirect_stdout(out):
            code = intake.main(['--event', str(event)])
        return code, json.loads(out.getvalue())

    # --- positive -----------------------------------------------------------------
    def test_exact_artifact_rows_are_accepted(self):
        result = self.run_check()
        self.assertEqual((result['lane'], result['package'], result['artifact_files'], result['verified_sources']),
                         ('incoming', 'incoming/example', 2, 1))
        code, printed = self.run_cli()
        self.assertEqual(code, 0)
        self.assertEqual(printed['package'], 'incoming/example')

    # --- negatives: the exact handled refusal ---------------------------------------
    def test_overstated_byte_count_with_the_right_digest_is_refused(self):
        name = self.manifest['artifacts'][0]['path']
        self.declare(0, bytes=len(self.raw[name]) + 1)
        self.refused()

    def test_understated_byte_count_with_the_right_digest_is_refused_by_the_cli(self):
        name = self.manifest['artifacts'][1]['path']
        self.declare(1, bytes=len(self.raw[name]) - 1)
        code, printed = self.run_cli()
        self.assertEqual((code, printed), (1, {'result': 'REJECTED', 'reason': 'artifact identity mismatch',
                                                 'scientific_effect': 'NONE'}))

    def test_same_length_wrong_digest_is_refused(self):
        name = self.manifest['artifacts'][0]['path']
        self.declare(0, sha256=hashlib.sha256(self.raw[name].upper()).hexdigest())
        self.refused()


if __name__ == '__main__':
    unittest.main()
