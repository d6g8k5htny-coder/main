"""Negative controls for architecture/scientific_state (#95 schema pilot).

Scientific effect: NONE. These tests only refuse malformed architecture payloads.
They do not promote mathematics or alter claims/Math- authorities.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "scientific_state_check", ROOT / "tools" / "scientific_state_check.py"
)
assert SPEC and SPEC.loader
SSC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SSC)


def _clone_pkg(tmp: Path) -> Path:
    src = ROOT / "architecture" / "scientific_state"
    dst = tmp / "architecture" / "scientific_state"
    shutil.copytree(src, dst)
    claims_src = ROOT / "claims" / "graph.json"
    claims_dst = tmp / "claims"
    claims_dst.mkdir(parents=True)
    shutil.copy2(claims_src, claims_dst / "graph.json")
    return tmp


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class ScientificStatePilotTests(unittest.TestCase):
    def test_repository_architecture_package_passes(self):
        report = SSC.audit(ROOT)
        self.assertEqual(report["problems"], [], report)
        self.assertEqual(report["scientific_effect"], "NONE")
        self.assertGreaterEqual(report["crosswalk_rows"], 8)

    def test_unknown_main_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = _clone_pkg(Path(raw))
            path = root / "architecture" / "scientific_state" / "v1" / "ID_CROSSWALK.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["rows"].append(
                {
                    "row_id": "xwalk.invented",
                    "main_claim_or_premise_id": "NOT-A-REAL-CLAIM-ID",
                    "main_bucket": "claims",
                    "math_gate_id": "math.invented",
                    "authority": "claims_firewall",
                    "notes": "should fail",
                }
            )
            _write(path, doc)
            report = SSC.audit(root)
            self.assertTrue(
                any("unknown claim" in p for p in report["problems"]),
                report["problems"],
            )

    def test_smuggled_status_in_crosswalk_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = _clone_pkg(Path(raw))
            path = root / "architecture" / "scientific_state" / "v1" / "ID_CROSSWALK.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["rows"][0]["status"] = "PROVED_REVIEWED"
            _write(path, doc)
            report = SSC.audit(root)
            self.assertTrue(
                any("smuggles owned field 'status'" in p for p in report["problems"]),
                report["problems"],
            )

    def test_smuggled_controlling_in_crosswalk_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = _clone_pkg(Path(raw))
            path = root / "architecture" / "scientific_state" / "v1" / "ID_CROSSWALK.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["rows"][0]["controlling"] = True
            _write(path, doc)
            report = SSC.audit(root)
            self.assertTrue(
                any("controlling" in p for p in report["problems"]),
                report["problems"],
            )

    def test_missing_authority_owner_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = _clone_pkg(Path(raw))
            path = root / "architecture" / "scientific_state" / "v1" / "AUTHORITY_MAP.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            del doc["authorities"]["math_downstream_gate"]
            _write(path, doc)
            report = SSC.audit(root)
            self.assertTrue(
                any("math_downstream_gate" in p for p in report["problems"]),
                report["problems"],
            )

    def test_verification_level_as_acceptance_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = _clone_pkg(Path(raw))
            path = root / "architecture" / "scientific_state" / "v1" / "SCHEMA.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["node_fields"]["verification_level"]["role"] = "acceptance"
            _write(path, doc)
            report = SSC.audit(root)
            self.assertTrue(
                any("evidence_metadata" in p for p in report["problems"]),
                report["problems"],
            )

    def test_unknown_authority_on_row_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = _clone_pkg(Path(raw))
            path = root / "architecture" / "scientific_state" / "v1" / "ID_CROSSWALK.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["rows"][0]["authority"] = "invented_second_status_db"
            _write(path, doc)
            report = SSC.audit(root)
            self.assertTrue(
                any("unknown authority" in p for p in report["problems"]),
                report["problems"],
            )

    def test_self_contradicting_never_writes_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = _clone_pkg(Path(raw))
            path = root / "architecture" / "scientific_state" / "v1" / "AUTHORITY_MAP.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["this_package"]["never_writes"] = ["status"]  # incomplete
            _write(path, doc)
            report = SSC.audit(root)
            self.assertTrue(
                any("never_writes" in p for p in report["problems"]),
                report["problems"],
            )

    def test_scientific_effect_must_remain_none(self):
        with tempfile.TemporaryDirectory() as raw:
            root = _clone_pkg(Path(raw))
            path = root / "architecture" / "scientific_state" / "v1" / "SCHEMA.json"
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["scientific_effect"] = "PROMOTES"
            _write(path, doc)
            report = SSC.audit(root)
            self.assertTrue(
                any("scientific_effect must be 'NONE'" in p for p in report["problems"]),
                report["problems"],
            )


if __name__ == "__main__":
    unittest.main()
