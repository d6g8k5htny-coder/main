"""Real-Git identity and finite negative controls for reconstruction inputs."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

PATH=Path(__file__).resolve().with_name("regenerate.py")
spec=importlib.util.spec_from_file_location("reconstruct", PATH)
subject=importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)

class FamilyValidation(unittest.TestCase):
    def setUp(self):
        self.commit="a"*40
        self.entry={"id":"K-1","status":"CLOSED","repository":"Math-",
                    "source_commit":self.commit,"declaration":"Demo.thm",
                    "path":"Proof.lean","source_sha256":"b"*64,
                    "axioms":["propext"]}
        self.manifest={"schema":"cbz.kernel-targets.v1","source_repository":"Math-",
                       "checked_commit":self.commit,"files":{"Proof.lean":"b"*64},
                       "targets":[{"name":"Demo.thm","path":"Proof.lean"}]}
        self.receipt={"schema":"cbz.kernel-replay.receipt.v1",
                      "formalization_status":"kernel-checked","checked_repository":"Math-",
                      "checked_commit":self.commit,"source_sha256":{"Proof.lean":"b"*64},
                      "axioms":{"Demo.thm":["propext"]}}
    def validate(self):
        return subject.validate_family([self.entry],self.manifest,self.receipt,1,self.commit)
    def test_valid_family(self):
        self.assertIsNone(self.validate())
    def test_unapproved_axiom_refused(self):
        self.entry["axioms"]=["Forged.axiom"]
        self.receipt["axioms"]["Demo.thm"]=["Forged.axiom"]
        with self.assertRaisesRegex(ValueError,"literal axiom mismatch"):
            self.validate()
    def test_wrong_checked_commit_refused(self):
        self.entry["source_commit"]="d"*40
        with self.assertRaisesRegex(ValueError,"literal source binding mismatch"):
            self.validate()
    def test_altered_target_path_refused(self):
        self.manifest["targets"][0]["path"]="Other.lean"
        with self.assertRaisesRegex(ValueError,"literal source binding mismatch"):
            self.validate()
    def test_receipt_source_table_drift_refused(self):
        self.receipt["source_sha256"]["Proof.lean"]="f"*64
        with self.assertRaisesRegex(ValueError,"receipt identity mismatch"):
            self.validate()
    def test_missing_axiom_target_refused(self):
        self.receipt["axioms"]={}
        with self.assertRaisesRegex(ValueError,"receipt axiom target set differs"):
            self.validate()
    def test_duplicate_ids_refused(self):
        with self.assertRaisesRegex(ValueError,"duplicate id"):
            subject.validate_family([self.entry,copy.deepcopy(self.entry)],
                                    self.manifest,self.receipt,2,self.commit)
    def test_duplicate_axioms_refused(self):
        self.entry["axioms"]=["propext","propext"]
        self.receipt["axioms"]["Demo.thm"]=["propext","propext"]
        with self.assertRaisesRegex(ValueError,"literal axiom mismatch"):
            self.validate()

class GitBinding(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo=Path(self.temporary.name)
        def g(*args):
            return subprocess.check_output(["git","-C",str(self.repo),*args],stderr=subprocess.STDOUT)
        self.git=g
        g("init","-q")
        g("config","user.name","CBZ fixture")
        g("config","user.email","cbz@example.invalid")
        (self.repo/"source.lean").write_text("theorem test : True := True.intro\n")
        g("add","source.lean")
        g("commit","-qm","source")
        self.commit=g("rev-parse","HEAD").decode().strip()
    def test_real_git_blob_and_digest(self):
        row=subject.bound(self.repo,"Math-",self.commit,"source.lean")
        self.assertEqual(row["git_object"],self.git("rev-parse","HEAD:source.lean").decode().strip())
        self.assertEqual(row["sha256"],subject.digest((self.repo/"source.lean").read_bytes()))
    def test_wrong_digest_rejected(self):
        with self.assertRaisesRegex(ValueError,"source/evidence digest mismatch"):
            subject.bound(self.repo,"Math-",self.commit,"source.lean","0"*64)
    def test_path_escape_rejected(self):
        with self.assertRaisesRegex(ValueError,"unsafe binding path"):
            subject.bound(self.repo,"Math-",self.commit,"../source.lean")

if __name__=="__main__": unittest.main()
