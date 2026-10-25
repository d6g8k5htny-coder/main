"""Retained protocol replay is isolated from the current live diagnostic worker."""
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

try:
    compat = importlib.import_module('retained_worker_compat')
except ImportError:
    compat = None

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = next((ROOT / 'examples/change_cycle').rglob('receipt.json'))


class RetainedProtocolCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(compat, 'retained protocol compatibility dispatch is missing')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_original_retained_receipt_replays_with_pinned_original_checker(self):
        before = hashlib.sha256(RECEIPT.read_bytes()).hexdigest()
        result = compat.verify_retained_receipt(RECEIPT, root=ROOT, replay=True)
        self.assertTrue(result['valid'])
        self.assertTrue(result['replayed'])
        self.assertEqual(result['result'], 'diagnostic_completed')
        self.assertEqual(result['formal_certification'], 'not_certified')
        self.assertEqual(result['verification_runtime']['protocol'], 'UL-worker-v17.1')
        self.assertEqual(hashlib.sha256(RECEIPT.read_bytes()).hexdigest(), before)

    def test_tampered_legacy_artifact_fails_after_relocation(self):
        attempt = self.root / 'attempt'
        shutil.copytree(RECEIPT.parent, attempt)
        (attempt / 'diagnostic.json').write_bytes(b'{"result":"diagnostic_completed"}')
        with self.assertRaisesRegex(ValueError, 'artifact hash'):
            compat.verify_retained_receipt(attempt / 'receipt.json', root=ROOT, replay=False)

    def test_unknown_protocol_never_falls_back_to_legacy(self):
        receipt = json.loads(RECEIPT.read_bytes())
        receipt['protocol_version'] = 'unknown-protocol'
        target = self.root / 'receipt.json'
        target.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'unsupported retained worker receipt protocol'):
            compat.verify_retained_receipt(target, root=ROOT)

    def test_modified_sealed_archive_is_rejected_before_execution(self):
        archive = self.root / 'sealed.zip'
        archive.write_bytes(b'forged archive')
        with patch.object(compat, '_sealed_archive', return_value=archive):
            with self.assertRaisesRegex(ValueError, 'sealed v17 archive hash mismatch'):
                compat.verify_retained_receipt(RECEIPT, root=ROOT)

    def test_current_protocol_dispatches_to_current_verifier(self):
        target = self.root / 'receipt.json'
        target.write_text(json.dumps({'protocol_version':'research-flow-worker-1'}))
        # Failure injection proves no fallback silently accepts the receipt.
        with patch.object(compat.current_workers, 'verify_receipt', side_effect=ValueError('current-verifier-rejected')):
            with self.assertRaisesRegex(ValueError, 'current-verifier-rejected'):
                compat.verify_retained_receipt(target, root=ROOT)

    def test_legacy_receipt_cannot_self_award_mathematical_certificate(self):
        attempt = self.root / 'attempt'
        shutil.copytree(RECEIPT.parent, attempt)
        receipt = json.loads((attempt / 'receipt.json').read_bytes())
        receipt['theorem_certificates_awarded'] = 1
        body = {k:v for k,v in receipt.items() if k != 'receipt_sha256'}
        receipt['receipt_sha256'] = hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',', ':'),ensure_ascii=False).encode()).hexdigest()
        (attempt / 'receipt.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'falsely claims theorem certification'):
            compat.verify_retained_receipt(attempt / 'receipt.json', root=ROOT)


if __name__ == '__main__':
    unittest.main()
