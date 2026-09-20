#!/usr/bin/env python3
"""Check custody/consistency of one pinned Lean recovery; never execute Lean.

Recorded compilation is not a fresh replay, authenticated time, mathematical
promotion, or organizationally independent review. No source files are written.
"""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "research/formal/candidates/LEAN_RECOVERY_VERIFICATION_20260920_v1"
BUNDLE_FILES = tuple(BASE + suffix for suffix in (".zip", ".manifest.json", ".receipt.json", ".md"))
ZIP_SHA = "431028664d37aa4e1d76ecb2e4f69bde45c8a1e90cbc6f52f16cbd59b64f74c0"
MANIFEST_SHA = "686fa57c380d07bbb30646d6322ffd97d51a8072ebee0fd75352e6cc4f44c653"
ORIGINAL_SHA = "a6440511fc259706457b18227734013966251105d538fdeabe633bb73f168631"
MATHLIB_REV = "c44e0c8ee63ca166450922a373c7409c5d26b00b"
LIMIT = 131072
LEAN_FILES = ("ResearchFormalCoreR1.lean", "ResearchFormalCoreR1/Algebra.lean",
              "ResearchFormalCoreR1/ProbabilityCompanions.lean", "ResearchFormalCoreR1/Status.lean")
STEPS = ("version", "build", "elaborate_ResearchFormalCoreR1", "elaborate_Algebra",
         "elaborate_ProbabilityCompanions", "elaborate_Status", "axioms",
         "counterexamples", "reject_wrong_fold_gap", "reject_wrong_quotient")
AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("non-finite JSON constant")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def archive(data, *, directories=False):
    """Bounded in-memory inspection, with no extraction or payload execution."""
    require(len(data) <= LIMIT, "archive input exceeds byte budget")
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        infos = bundle.infolist()
        require(len(infos) <= 64, "archive member count exceeds budget")
        require(sum(item.file_size for item in infos) <= 1024 * 1024, "archive expanded bytes exceed budget")
        files, names = {}, set()
        for item in infos:
            name = item.filename
            parts = PurePosixPath(name).parts
            require(name and item.orig_filename == name and name not in names,
                    "duplicate or malformed archive name")
            require(not name.startswith("/") and "\\" not in name and ":" not in name
                    and "\0" not in name and not any(p in (".", "..") for p in name.rstrip("/").split("/"))
                    and all(parts) and "//" not in name, "unsafe archive path")
            names.add(name)
            kind = stat.S_IFMT(item.external_attr >> 16)
            require(kind in (0, stat.S_IFREG, stat.S_IFDIR) and not item.flag_bits & 1,
                    "symlink/special/encrypted archive member refused")
            require(item.file_size <= LIMIT and item.compress_size <= LIMIT, "oversized archive member")
            require(item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED), "unsupported archive compression")
            if item.is_dir():
                require(directories and item.file_size == 0, "unexpected archive directory")
                continue
            require(kind != stat.S_IFDIR, "archive file/directory type mismatch")
            with bundle.open(item) as stream:
                content = stream.read(LIMIT + 1)
            require(len(content) == item.file_size and len(content) <= LIMIT, "archive member size mismatch")
            files[name] = content
        return files


def contents_check(files, receipt):
    """Check the frozen report against its logged inputs; this does not compile."""
    original = files["original/GP_FOR_192.zip"]
    require(identity(original) == {"bytes": 10644, "sha256": ORIGINAL_SHA}, "original carrier identity mismatch")
    originals = archive(original, directories=True)
    prefix = "research-formal-core-r2/"
    require({name[len(prefix):] for name in originals if name.endswith(".lean")} == set(LEAN_FILES),
            "original Lean source set mismatch")
    require({name[len("recovered/"):] for name in files if name.startswith("recovered/") and name.endswith(".lean")} == set(LEAN_FILES),
            "recovered Lean source set mismatch")
    declarations, comparisons = [], {}
    for name in LEAN_FILES:
        before, after = originals[prefix + name], files["recovered/" + name]
        expected = before
        if name.endswith("/Algebra.lean"):
            target = b"\ndef foldPotential "
            require(before.count(target) == 1, "original fold declaration target mismatch")
            expected = before.replace(target, b"\nnoncomputable def foldPotential ")
        require(after == expected, "Lean body changed beyond the permitted noncomputable declaration: " + name)
        found = re.findall(r"^theorem\s+(\w+)(.*?)\s*:= by", after.decode(), re.M | re.S)
        declarations.extend({"name": "ResearchFormalCoreR1." + theorem, "source": name,
                             "statement_text_sha256": identity(statement.encode())["sha256"]}
                            for theorem, statement in found)
        comparisons[name] = {"path": name, "original_bytes": len(before), "original_sha256": identity(before)["sha256"],
                             "current_bytes": len(after), "current_sha256": identity(after)["sha256"],
                             "theorem_statements_identical": True, "theorem_count": len(found)}
    require(len(declarations) == 13 and len({d["name"] for d in declarations}) == 13, "theorem inventory mismatch")
    comparison = strict_json(files["audit/source_comparison.json"])
    require(comparison["schema"] == "recovered-lean-source-comparison/v1"
            and comparison["original_carrier_sha256"] == ORIGINAL_SHA
            and comparison["original_drive_id"] == "1hTeWNKxLcmXEB2i5enAdmzXFknUxwTL6"
            and len(comparison["files"]) == 4
            and {entry["path"]: entry for entry in comparison["files"]} == comparisons,
            "source comparison report mismatch")
    report = strict_json(files["verification/report.json"])
    payload = {k: v for k, v in report.items() if k != "payload_sha256"}
    payload_hash = identity(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())["sha256"]
    require(report["payload_sha256"] == receipt["verification_payload_sha256"] == payload_hash, "verification payload hash mismatch")
    require(report["schema"] == "recovered-lean-verification/v1"
            and report["status"] == "PASS_AT_RECOVERED_ALGEBRA_SCOPE"
            and report["source_inputs_unchanged"] is True, "recorded verification scope/status mismatch")
    for obj in (receipt, report):
        require(type(obj["scientific_promotions"]) is int and obj["scientific_promotions"] == 0
                and type(obj["independence_credit"]) is int and obj["independence_credit"] == 0,
                "promotion/independence boundary violated")
    require(sorted(report["declarations"], key=lambda d: d["name"]) == sorted(declarations, key=lambda d: d["name"]),
            "recorded theorem declarations mismatch")
    recovered = {name[len("recovered/"):]: identity(data) for name, data in files.items() if name.startswith("recovered/")}
    require(report["source_inputs"] == recovered, "recorded source inputs mismatch")
    require(files["recovered/lean-toolchain"].strip() == b"leanprover/lean4:v4.19.0", "Lean toolchain mismatch")
    lock = strict_json(files["recovered/lake-manifest.json"])
    mathlib = [p for p in lock["packages"] if p["name"] == "mathlib"]
    require(len(mathlib) == 1 and report["mathlib"] == mathlib[0] and mathlib[0]["rev"] == MATHLIB_REV,
            "Mathlib lock/report mismatch")
    require([step["name"] for step in report["steps"]] == list(STEPS), "verification step set/order mismatch")
    for step in report["steps"]:
        negative = step["name"].startswith("reject_")
        require(step["passed"] is True and step["timed_out"] is False
                and type(step["returncode"]) is int and step["returncode"] == (1 if negative else 0)
                and step["expected"] == ("REJECT_FALSE_STATEMENT" if negative else "SUCCESS"),
                "recorded step did not meet its expected outcome: " + step["name"])
        require(step["log"] == step["name"] + ".log", "step log selector mismatch")
        log = files["verification/" + step["log"]]
        require(step["log_identity"] == identity(log), "step log identity mismatch")
        if negative:
            require(b"error:" in log and (not step["name"].endswith("fold_gap") or b"unsolved goals" in log),
                    "negative control lacks expected compiler diagnostic")
    require(b"version 4.19.0" in files["verification/version.log"], "recorded compiler version mismatch")
    require(b"Build completed successfully." in files["verification/build.log"], "recorded build success missing")
    axiom_text = files["verification/axioms.log"].decode()
    rows = re.findall(r"^'([^']+)' depends on axioms: \[([^\]]*)\]$", axiom_text, re.M)
    require(len(rows) == 13 and {name for name, _ in rows} == {d["name"] for d in declarations}
            and all({x.strip() for x in values.split(",")} == AXIOMS for _, values in rows)
            and "sorryAx" not in axiom_text and report["axioms_observed"] == sorted(AXIOMS),
            "recorded axiom audit mismatch")
    for field, member in (("axiom_audit_source", "AxiomAudit.lean"), ("counterexample_source", "Counterexamples.lean")):
        require(report[field] == identity(files["verification/" + member]), "verification source identity mismatch")
    require(report["negative_control_sources"] == {name: identity(files["verification/" + name])
                                                   for name in ("WrongFoldGap.lean", "WrongQuotient.lean")},
            "negative-control source identities mismatch")
    require(files["verification/WrongFoldGap.lean"] == files["recovered/ResearchFormalCoreR1/Algebra.lean"].replace(
        b"(2 * s) ^ 3 / 6", b"(2 * s) ^ 3 / 7"), "wrong fold control changed")
    require(files["verification/WrongQuotient.lean"] == files["recovered/ResearchFormalCoreR1/ProbabilityCompanions.lean"].replace(
        b"n / z \xe2\x89\xa4 (Real.sqrt cW / cZ) * Real.sqrt q := by",
        b"n / z \xe2\x89\xa4 (Real.sqrt cW / (2*cZ)) * Real.sqrt q := by"), "wrong quotient control changed")
    return report


def manifest_check(files, manifest, manifest_bytes, readme):
    require(set(manifest) == {"schema", "members"} and manifest["schema"] == "lean-recovery-bundle/v1", "invalid member manifest")
    require(len(files) == 31 and set(files) == {"MANIFEST.json", *manifest["members"]}, "bundle member set mismatch")
    require(files["MANIFEST.json"] == manifest_bytes and files["README.md"] == readme, "external/internal manifest or README mismatch")
    for name, expected in manifest["members"].items():
        require(identity(files[name]) == expected, "member identity mismatch: " + name)


def verify(root=ROOT):
    root = Path(root).resolve()
    data = {}
    for relative in BUNDLE_FILES:
        path = root / relative
        require(path.resolve().is_relative_to(root) and not path.is_symlink()
                and all(not p.is_symlink() for p in path.parents), "bundle path symlink/escape refused")
        with path.open("rb") as stream:
            data[relative] = stream.read(LIMIT + 1)
        require(len(data[relative]) <= LIMIT, "bundle file exceeds byte budget")
    zipped, manifest_bytes, receipt_bytes, readme = (data[path] for path in BUNDLE_FILES)
    require(identity(zipped) == {"bytes": 36369, "sha256": ZIP_SHA}, "pinned Lean bundle identity mismatch")
    require(identity(manifest_bytes)["sha256"] == MANIFEST_SHA, "pinned member manifest identity mismatch")
    manifest, receipt = strict_json(manifest_bytes), strict_json(receipt_bytes)
    files = archive(zipped)
    manifest_check(files, manifest, manifest_bytes, readme)
    expected_receipt = {"schema": "lean-recovery-delivery/v1", "artifact": Path(BUNDLE_FILES[0]).name,
                        "bytes": 36369, "sha256": ZIP_SHA, "member_manifest_sha256": MANIFEST_SHA,
                        "members": 31, "theorems": 13, "verification_steps": 10,
                        "verification_status": "PASS_AT_RECOVERED_ALGEBRA_SCOPE",
                        "original_lean_statements_and_proof_tactics_unchanged": True}
    require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k, v in expected_receipt.items()), "delivery receipt mismatch")
    report = contents_check(files, receipt)
    return {"status": "RECORDED_BUILD_EVIDENCE_CUSTODY_CHECKED", "replayed_here": False,
            "scientific_promotions": 0, "independence_credit": 0, "authority": "NONE",
            "recorded_status": report["status"], "recorded_theorems": 13, "recorded_steps": 10,
            "recorded_negative_controls": 2, "recorded_axioms": sorted(AXIOMS),
            "original_carrier_sha256": ORIGINAL_SHA, "source_change": "noncomputable def foldPotential only",
            "bundle_files": {path: identity(content) for path, content in data.items()},
            "scope": report["scope"],
            "does_not_establish": "Custody and consistency of recorded compilation evidence only. Lean was not run here; no current compilation, authenticated timestamp, analytic/probability premise, mathematical promotion or independent review is established."}


def main():
    try:
        result = verify()
        print("lean_receipt: " + result["status"] + "; recorded_theorems=13 recorded_steps=10 replayed_here=false authority=NONE")
        return 0
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
        print("lean_receipt: REFUSED: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
