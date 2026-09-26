"""F1 loss-only contract: real status changes must affect CLI exit status.

Uses synthetic data and temporary Git commits. No scientific status is written.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("f1_adapter_under_test", ROOT / "tools/claims_gate_adapter.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load subject adapter")
CGA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CGA)


def fixture():
    return {"as_of": "2026-09-25", "premises": {
        "P": {"status_register_note": "OPEN", "source": {"path": "proof.md"}}},
        "claims": {"T": {"grade": "OPEN", "controlling": False, "depends_on": ["P"],
                           "source": {"path": "theorem.md"}},
                   "U": {"grade": "OPEN", "source": {"path": "unrelated.md"}}}}


def promote(g, node="T"):
    g["claims"][node].update(grade="LIVE_ROOT_THEOREM", controlling=True)


def demote(g, node="T"):
    g["claims"][node].update(grade="RETRACTED_TO_CANDIDATE", controlling=False)


def changed_pair():
    old = fixture()
    promote(old)
    new = copy.deepcopy(old)
    new["premises"]["P"]["statement"] = "Corrected lower lemma"
    return old, new


class EnforcementLibraryTests(unittest.TestCase):
    def check_blocked(self, old, new, node="T"):
        frozen_old, frozen_new = copy.deepcopy(old), copy.deepcopy(new)
        report = CGA.compare_claims_files(old, new)
        self.assertIs(report.get("transition_ok"), False, report)
        self.assertIn(node, [row.get("node") for row in report.get("transition_errors", [])])
        self.assertFalse(report["promotion_permission"])
        self.assertEqual(old, frozen_old)
        self.assertEqual(new, frozen_new)
        return report

    def test_promotion_over_refuted_required_premise_is_refused(self):
        old=fixture(); old["premises"]["P"]["status_register_note"]="REFUTED"
        new=copy.deepcopy(old); promote(new)
        report=self.check_blocked(old,new)
        self.assertEqual(report["hold_proposals"]["T"]["refuted_required"],["P"])

    def test_promotion_over_open_required_premise_is_refused(self):
        old=fixture();new=copy.deepcopy(old);promote(new)
        self.check_blocked(old,new)

    def test_new_controlling_leaf_does_not_invent_acceptance(self):
        old=fixture();new=copy.deepcopy(old)
        new["claims"]["N"]={"grade":"LIVE_ROOT_THEOREM","controlling":True,"depends_on":[]}
        self.check_blocked(old,new,"N")

    def test_changed_premise_cannot_leave_consumer_controlling(self):
        self.check_blocked(*changed_pair())

    def test_required_edge_deletion_does_not_erase_controlling_impact(self):
        old=fixture();promote(old);new=copy.deepcopy(old)
        new["claims"]["T"]["depends_on"]=[]
        self.check_blocked(old,new)

    def test_grade_alone_still_exposes_controlling_use(self):
        old,new=changed_pair()
        old["claims"]["T"].pop("controlling");new["claims"]["T"].pop("controlling")
        self.check_blocked(old,new)

    def test_false_flag_cannot_hide_controlling_grade(self):
        old,new=changed_pair();new["claims"]["T"]["controlling"]=False
        self.check_blocked(old,new)

    def test_status_note_cannot_hide_behind_other_status_field(self):
        old=fixture();old["premises"]["P"]["status_frozen_v2_2"]="OPEN"
        new=copy.deepcopy(old);new["premises"]["P"]["status_register_note"]="CONTROLLING"
        self.check_blocked(old,new,"P")

    def test_spelling_normalization_cannot_hide_controlling_grade(self):
        old=fixture();new=copy.deepcopy(old)
        new["claims"]["T"]["grade"]=" live_root_theorem "
        self.check_blocked(old,new)

    def test_review_metadata_is_not_automatic_promotion_authority(self):
        old=fixture();new=copy.deepcopy(old);promote(new)
        new["claims"]["T"]["review_records"]=[{"disposition":"ACCEPTED","independent":True}]
        self.check_blocked(old,new)

    def test_explicit_consistent_demotion_permits_correction(self):
        old,new=changed_pair();demote(new)
        result=CGA.compare_claims_files(old,new)
        self.assertIs(result.get("transition_ok"),True,result)
        self.assertEqual(result.get("transition_errors"),[])
        self.assertIn("T",result["hold_proposals"])
        self.assertFalse(result["promotion_permission"])

    def test_all_transitive_consumers_must_be_demoted(self):
        old,new=changed_pair()
        for g in (old,new):
            g["claims"]["V"]={"grade":"LIVE_ROOT_THEOREM","controlling":True,"depends_on":["T"]}
        demote(new)
        self.check_blocked(old,new,"V")
        demote(new,"V")
        self.assertIs(CGA.compare_claims_files(old,new).get("transition_ok"),True)

    def test_unchanged_legacy_status_is_not_retroactive_transition(self):
        old=fixture();promote(old);new=copy.deepcopy(old)
        new["claims"]["U"]["statement"]="Unrelated documentation correction"
        report=CGA.compare_claims_files(old,new)
        self.assertIs(report.get("transition_ok"),True,report)
        self.assertIn("T",report["hold_proposals"])
        self.assertFalse(report["promotion_permission"])

    def test_noncontrolling_changed_source_still_produces_impact(self):
        old=fixture();new=copy.deepcopy(old);new["premises"]["P"]["statement"]="Amended"
        report=CGA.compare_claims_files(old,new)
        self.assertIs(report.get("transition_ok"),True,report)
        self.assertEqual(set(report["reverse_impact"]["impacted"]),{"P","T"})

    def test_refuted_classification_is_not_rewritten_as_acceptance(self):
        old=fixture();new=copy.deepcopy(old);new["premises"]["P"]["status_register_note"]="REFUTED"
        report=CGA.compare_claims_files(old,new)
        self.assertIs(report.get("transition_ok"),True,report)
        self.assertEqual(report["reverse_impact"]["graph"]["nodes"]["P"]["classification"],"REFUTED")


class EnforcementCLITests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="f1-status-test-")
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/"repo";self.root.mkdir()
        self.git("init","-q");self.git("config","user.email","review@example.invalid")
        self.git("config","user.name","Synthetic source fixture")
        self.write("architecture/scientific_state/v1/ID_CROSSWALK.json",{"rows":[]})
        self.write("architecture/scientific_state/v1/AUTHORITY_MAP.json",{"authorities":{}})
        for name in ("proof.md","theorem.md","unrelated.md"):
            (self.root/name).write_text("Synthetic initial source\n")

    def git(self,*args):
        p=subprocess.run(["git","-C",str(self.root),*args],capture_output=True,text=True,timeout=20)
        self.assertEqual(p.returncode,0,p.stderr)
        return p.stdout.strip()

    def write(self,name,obj):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(obj)+"\n")

    def commit(self,obj):
        self.write("claims/graph.json",obj);self.git("add",".")
        self.git("commit","--allow-empty","-qm","synthetic snapshot")
        return self.git("rev-parse","HEAD")

    def cli(self,command,old,new,source_change=False):
        before=self.commit(old)
        if source_change:(self.root/"proof.md").write_text("Changed load-bearing source\n")
        after=self.commit(new)
        env=os.environ.copy();env["CLAIMS_GATE_BEFORE_REF"]=before;env["CLAIMS_GATE_AFTER_REF"]=after
        argv=[sys.executable,*(["-O"] if sys.flags.optimize else []),"-B","-S",
              str(ROOT/"tools/claims_gate_adapter.py"),command]
        report_path=Path(self.temp.name)/"impact.json"
        if command=="event-compare":
            argv.extend(["--repo-root",str(self.root)])
        elif command=="compare-refs":
            argv.extend(["--repo-root",str(self.root),"--before-ref",before,"--after-ref",after])
        else:
            op=Path(self.temp.name)/"before.json";np=Path(self.temp.name)/"after.json"
            op.write_text(json.dumps(old));np.write_text(json.dumps(new))
            argv.extend(["--before",str(op),"--after",str(np)])
        argv.extend(["--write-report",str(report_path)])
        p=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=30)
        self.assertTrue(p.stdout.strip(),p.stderr)
        result=json.loads(p.stdout)
        self.assertTrue(report_path.is_file(),p.stderr)
        self.assertEqual(json.loads(report_path.read_text()),result)
        self.assertFalse(result["promotion_permission"])
        return p.returncode,result

    def test_real_event_promotion_exits_nonzero_and_persists_hold(self):
        old=fixture();old["premises"]["P"]["status_register_note"]="REFUTED"
        new=copy.deepcopy(old);promote(new)
        rc,r=self.cli("event-compare",old,new)
        self.assertNotEqual(rc,0,r)
        self.assertIs(r.get("transition_ok"),False)
        self.assertIn("T",r["hold_proposals"])

    def test_real_event_source_only_edit_refuses_controlling_consumer(self):
        old=fixture();promote(old);new=copy.deepcopy(old)
        rc,r=self.cli("event-compare",old,new,source_change=True)
        self.assertNotEqual(rc,0,r)
        self.assertIn("P",r["reverse_impact"]["source_byte_seeds"])

    def test_real_event_source_change_and_demotion_succeeds(self):
        old=fixture();promote(old);new=copy.deepcopy(old);demote(new)
        rc,r=self.cli("event-compare",old,new,source_change=True)
        self.assertEqual(rc,0,r)
        self.assertIs(r.get("transition_ok"),True)

    def test_compare_refs_cannot_bypass_enforcement(self):
        old=fixture();new=copy.deepcopy(old);promote(new)
        rc,r=self.cli("compare-refs",old,new)
        self.assertNotEqual(rc,0,r)

    def test_path_compare_cannot_bypass_enforcement(self):
        old=fixture();new=copy.deepcopy(old);promote(new)
        rc,r=self.cli("compare",old,new)
        self.assertNotEqual(rc,0,r)

    def test_refuted_premise_edit_is_allowed_when_consumer_noncontrolling(self):
        old=fixture();new=copy.deepcopy(old)
        new["premises"]["P"]["status_register_note"]="REFUTED"
        rc,r=self.cli("event-compare",old,new)
        self.assertEqual(rc,0,r)
        self.assertEqual(r["hold_proposals"]["T"]["refuted_required"],["P"])


if __name__=="__main__":unittest.main()
