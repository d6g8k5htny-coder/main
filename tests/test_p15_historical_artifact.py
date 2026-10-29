import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import p15_historical_artifact as subject

class HistoricalArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root=Path(self.temporary.name)
        self.root=root
        self.archive=root/"artifact.zip"
        self.manifest=root/"manifest.json"
        self.commit="a"*40
        self.target="UniversalLaw.P15.example"
        self.log=b"elaborated fixture type\n"
        self.source=root/"formal/fixture.lean"
        self.source.parent.mkdir()
        self.source.write_text("theorem fixture : True := True.intro\n")
        self.manifest.write_text(json.dumps({"targets":[{"name":self.target}],"files":{"formal/fixture.lean":subject.digest(self.source.read_bytes())}}))
        self.manifest_hash=subject.digest(self.manifest.read_bytes())
        self.receipt={"schema_version":1,"checked_commit":self.commit,"workflow_run_id":"7","repository":"d6g8k5htny-coder/main","manifest_sha256":self.manifest_hash,
                      "formalization_status":"kernel-checked","axioms":{self.target:[]},
                      "logs":{"elaborated-types.log":subject.digest(self.log)}}
        self.write_zip()

    def write_zip(self,receipt=None,log=None,extra=()):
        self.receipt_bytes=json.dumps(receipt if receipt is not None else self.receipt).encode()
        with zipfile.ZipFile(self.archive,"w") as z:
            z.writestr("receipt/receipt.json",self.receipt_bytes)
            z.writestr("receipt/elaborated-types.log",self.log if log is None else log)
            for name,value in extra:z.writestr(name,value)
        self.zip_hash=subject.digest(self.archive.read_bytes())
        self.receipt_hash=subject.digest(self.receipt_bytes)

    def check(self,**kw):
        return subject.verify(self.archive,self.manifest,self.root,kw.get("zip",self.zip_hash),
                              kw.get("receipt",self.receipt_hash),kw.get("manifest",self.manifest_hash),
                              kw.get("commit",self.commit),kw.get("total",1),kw.get("p15",1),"7")

    def test_positive_retains_literal_identity(self):
        x=self.check()
        self.assertEqual(x["p15_names"],[self.target])
        self.assertEqual(x["logs"]["elaborated-types.log"],subject.digest(self.log))

    def test_archive_hash_refuses_changed_zip(self):
        with self.assertRaisesRegex(ValueError,"archive SHA256 mismatch"):
            self.check(zip="0"*64)

    def test_manifest_hash_refuses_changed_source(self):
        self.manifest.write_bytes(self.manifest.read_bytes()+b" ")
        with self.assertRaisesRegex(ValueError,"manifest SHA256 mismatch"):
            self.check()

    def test_receipt_hash_refuses_changed_receipt(self):
        with self.assertRaisesRegex(ValueError,"receipt SHA256 mismatch"):
            self.check(receipt="0"*64)

    def test_p15_target_count_refuses_other_count(self):
        with self.assertRaisesRegex(ValueError,"P15 target count mismatch"):
            self.check(p15=2)

    def test_nonempty_axiom_list_is_refused(self):
        self.receipt["axioms"][self.target]=["Injected.axiom"]
        self.write_zip()
        with self.assertRaisesRegex(ValueError,"unexpected theorem axiom"):
            self.check()

    def test_source_blob_mismatch_is_refused(self):
        self.source.write_text("theorem changed : True := True.intro\n")
        with self.assertRaisesRegex(ValueError,"source file SHA256 mismatch"):
            self.check()

    def test_wrong_log_bytes_are_refused(self):
        self.write_zip(log=b"altered")
        with self.assertRaisesRegex(ValueError,"receipt log SHA256 mismatch"):
            self.check()

    def test_wrong_commit_is_refused(self):
        with self.assertRaisesRegex(ValueError,"receipt source identity mismatch"):
            self.check(commit="b"*40)

    def test_missing_log_and_unlisted_log_are_refused(self):
        self.receipt["logs"]={"missing.log":"0"*64}
        self.write_zip()
        with self.assertRaisesRegex(ValueError,"receipt log inventory mismatch"):
            self.check()

    def test_unsafe_path_is_refused(self):
        self.write_zip(extra=[("../outside",b"bad")])
        with self.assertRaisesRegex(ValueError,"unsafe member path"):
            self.check()

    def test_duplicate_member_is_refused(self):
        self.write_zip(extra=[("receipt/elaborated-types.log",b"duplicate")])
        with self.assertRaisesRegex(ValueError,"duplicate archive entry"):
            self.check()

    def test_target_inventory_must_match_axioms(self):
        self.receipt["axioms"]={}
        self.write_zip()
        with self.assertRaisesRegex(ValueError,"receipt theorem inventory mismatch"):
            self.check()

class MutationHarnessTests(unittest.TestCase):
    def test_detects_exact_expected_failure(self):
        from p15_historical_mutations import detected_failure
        log="Ran 1 test\nAssertionError: ValueError not raised\nFAILED (failures=1)\n"
        self.assertTrue(detected_failure(1,log))

    def test_false_kill_rejection_controls(self):
        from p15_historical_mutations import detected_failure
        expected="Ran 1 test\nAssertionError: ValueError not raised\nFAILED (failures=1)\n"
        self.assertFalse(detected_failure(86,"silent crash"))
        self.assertFalse(detected_failure(0,expected))
        self.assertFalse(detected_failure(1,"Ran 1 test\nAssertionError: wrong reason\nFAILED (failures=1)\n"))
        self.assertFalse(detected_failure(1,expected+"ERROR: fixture\n"))

if __name__=="__main__":
    unittest.main()
