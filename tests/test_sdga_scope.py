"""Task 1 contracts only. Initial missing module is absence, never control credit."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "sdga_scope"
sys.path.insert(0, str(ROOT / "tools"))
import sdga_scope as api  # Deliberately absent in the immutable RED draft.

H = hashlib.sha256(b"fixture-object").hexdigest()
EMPTY = hashlib.sha256(b"").hexdigest()
BLOCKERS = {
    "EXACT_TYPE_SEMANTICS_UNRESOLVED",
    "FULL_DECLARATION_ROLES_UNRESOLVED",
    "EXECUTION_NOT_OBSERVED",
    "KERNEL_COVERAGE_UNESTABLISHED",
}


def encoded(*components):
    return "/".join("n" + str(x) if type(x) is int else "s" + x.encode("utf-8").hex()
                    for x in components)


ALPHA, BETA = encoded("Alpha"), encoded("Beta")
PUBLIC, DATA = encoded("Public", "p"), encoded("Public", "d")
PRIVATE = encoded("_private", "Alpha", 0, "hidden")
FOREIGN = encoded("Foreign", "generated")


def blob_id(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def rebound(profile, raw_inputs):
    result = copy.deepcopy(profile)
    for name, raw in raw_inputs.items():
        result["inputs"][name].update(
            sha256=hashlib.sha256(raw).hexdigest(), git_blob=blob_id(raw), bytes=len(raw))
    return result


def profile_and_projection():
    profile = json.loads((FIXTURES / "scope-f532.json").read_text(encoding="utf-8"))
    raw = {name: (FIXTURES / name).read_bytes()
           for name in ("lakefile.toml", "lake-manifest.json", "lean-toolchain")}
    raw["SOURCE_MANIFEST.json"] = (FIXTURES / "source-scope-projection.json").read_bytes()
    return rebound(profile, raw), raw


def fixture_scope():
    return {
        "scope_id": "fixture-sdga",
        "modules": [
            {"module": "Alpha", "path": "Alpha.lean", "role": "production", "source_sha256": H},
            {"module": "Beta", "path": "Beta.lean", "role": "production", "source_sha256": H},
        ],
        "targets": ["Public.p", "Public.d"],
        "expected_counts": {"public_targets": 2, "proposition_targets": 1, "data_targets": 1},
    }


def part(level, names):
    return {"level": level, "sha256": H, "const_names": names, "constant_names": names[:]}


def declaration(module, name, kind="theorem", prop=True, private=False):
    return {"record": "declaration", "module": module, "name": name, "kind": kind,
            "is_prop": prop, "is_private": private, "unsafe": False, "partial": False,
            "type_display": "True" if prop else "Nat", "object_part_sha256": H, "axioms": []}


def records():
    return [
        {"record": "header", "schema": 1, "scope_id": "fixture-sdga"},
        {"record": "module", "name": ALPHA, "const_count": 2,
         "object_parts": [part("exported", [PUBLIC]), part("server", [PUBLIC]),
                          part("private", [PUBLIC, PRIVATE])], "extra_const_names": []},
        {"record": "module", "name": BETA, "const_count": 2,
         "object_parts": [part("exported", [DATA, FOREIGN])], "extra_const_names": []},
        declaration(ALPHA, PUBLIC),
        declaration(ALPHA, PRIVATE, private=True),
        declaration(BETA, DATA, "definition", False),
        declaration(BETA, FOREIGN, "definition", False),
        {"record": "end", "module_count": 2, "declaration_count": 4, "axioms": []},
    ]


def transcript(rows):
    return "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                     for row in rows) + "\n"


def step(name):
    return {"name": name, "exit_code": 0, "signal": None, "timed_out": False,
            "stdout_sha256": EMPTY, "stderr_sha256": EMPTY,
            "started_utc": "2026-11-09T00:00:00Z", "ended_utc": "2026-11-09T00:00:01Z",
            "duration_ns": 1000000000, "max_rss_kib": 1024}


def receipt():
    return {
        "schema": 1, "scope_id": "fixture-sdga", "scientific_effect": "NONE",
        "source": {"repository": "fixture/Math-", "commit": "1" * 40, "tree": "2" * 40},
        "adapter_sha256": H, "profile_sha256": H,
        "run": {"repository": "fixture/main", "id": 10, "attempt": 1, "job_id": 20},
        "toolchain": {"commit": "5045d0056413266e57c625dcd7c365b10e377c52",
                      "lean_sha256": H, "lake_sha256": H},
        "inputs_before_sha256": H, "inputs_after_sha256": H,
        "cache_before_sha256": H, "cache_after_sha256": H,
        "owned_objects_sha256": H, "module_coverage": [ALPHA, BETA],
        "unsafe_skipped": [], "partial_skipped": [],
        "steps": [step("build"), step("inventory"), step("kernel:" + ALPHA), step("kernel:" + BETA)],
    }


class ContractCase(unittest.TestCase):
    def assert_blocked(self, result):
        self.assertEqual(result["scientific_effect"], "NONE")
        self.assertEqual(result["qualification"], "BLOCKED")
        self.assertEqual(result["classification_status"], "BLOCKED")

    def scope_error(self, code, raw, profile, observed=None):
        with self.assertRaises(api.ScopeError) as caught:
            api.extract_scope(raw, profile, observed or profile["source"])
        self.assertEqual(caught.exception.code, code)

    def invalid_receipt(self, changed, expected, code):
        result = api.classify_receipt(changed, expected)
        self.assert_blocked(result)
        self.assertEqual(result["binding_status"], "INVALID")
        self.assertEqual({x["code"] for x in result["findings"]}, {code})


class ScopeContracts(ContractCase):
    def test_frozen_scope_matches_literal_vectors_and_all_99_bindings(self):
        profile, raw = profile_and_projection()
        scope = api.extract_scope(raw, profile, profile["source"])
        self.assert_blocked(scope)
        self.assertEqual(scope["source"], profile["source"])
        self.assertEqual(scope["modules"], profile["modules"])
        self.assertEqual(scope["targets"], profile["targets"])
        self.assertEqual(len(scope["modules"]), 99)
        self.assertEqual(sum(x["role"] == "production" for x in scope["modules"]), 67)
        self.assertEqual(sum(x["role"] == "input" for x in scope["modules"]), 32)
        self.assertEqual(len(scope["targets"]), 1137)
        digest = hashlib.sha256(("\n".join(scope["targets"]) + "\n").encode("utf-8")).hexdigest()
        self.assertEqual(digest, "2c22d5ba939ed6cf74669a99a23ad8337ec9a54abcafb3bf43ef5799e5536cae")
        matrix = json.dumps(scope["modules"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.assertEqual(hashlib.sha256(matrix).hexdigest(), "4446aeee839a661d2452fa8f1c0992ee2f581d7623b94e490511b047c5b75111")

    def test_raw_input_and_identity_mismatch_refused_before_parsing(self):
        profile, raw = profile_and_projection()
        changed = dict(raw, **{"lakefile.toml": raw["lakefile.toml"] + b"\n"})
        self.scope_error("INPUT_BINDING", changed, profile)
        wrong = dict(profile["source"], commit="0" * 40)
        self.scope_error("SOURCE_IDENTITY", raw, profile, wrong)

    def test_duplicate_missing_or_changed_target_vector_refused(self):
        profile, raw = profile_and_projection()
        original = json.loads(raw["SOURCE_MANIFEST.json"])
        for targets in (original["targets"] + [original["targets"][0]], original["targets"][:-1],
                        ["I4SourceSmoothParameterSpace.weightedCarrier_memlp"] + original["targets"][1:]):
            with self.subTest(target_count=len(targets)):
                data = dict(original, targets=targets)
                changed = dict(raw, **{"SOURCE_MANIFEST.json": json.dumps(data).encode()})
                self.scope_error("TARGET_SCOPE", changed, rebound(profile, changed))

    def test_missing_module_duplicate_root_and_production_input_swap_refused(self):
        profile, raw = profile_and_projection()
        original = json.loads(raw["SOURCE_MANIFEST.json"])
        data = dict(original, source_modules=original["source_modules"][:-1])
        changed = dict(raw, **{"SOURCE_MANIFEST.json": json.dumps(data).encode()})
        self.scope_error("MODULE_SCOPE", changed, rebound(profile, changed))
        text = raw["lakefile.toml"].decode()
        for replacement in (
            text.replace('"GaussianSeriesFoundation", "SpectralWeights"',
                         '"GaussianSeriesFoundation", "GaussianSeriesFoundation"', 1),
            text.replace('name = "SourceDerivativeInputs"\nsrcDir = "inputs"',
                         'name = "SourceDerivativeInputs"\nsrcDir = "."', 1),
        ):
            with self.subTest(replacement=replacement[:30]):
                changed = dict(raw, **{"lakefile.toml": replacement.encode()})
                self.scope_error("MODULE_SCOPE", changed, rebound(profile, changed))

    def test_pin_toolchain_and_selected_source_hash_changes_refused(self):
        profile, raw = profile_and_projection()
        changed = dict(raw, **{"lean-toolchain": b"leanprover/lean4:v4.33.0\n"})
        self.scope_error("TOOLCHAIN_SCOPE", changed, rebound(profile, changed))
        lock = json.loads(raw["lake-manifest.json"])
        lock["packages"][0]["rev"] = "0" * 40
        changed = dict(raw, **{"lake-manifest.json": json.dumps(lock).encode()})
        self.scope_error("DEPENDENCY_SCOPE", changed, rebound(profile, changed))
        data = json.loads(raw["SOURCE_MANIFEST.json"])
        data["files"][profile["modules"][0]["path"]] = "0" * 64
        changed = dict(raw, **{"SOURCE_MANIFEST.json": json.dumps(data).encode()})
        self.scope_error("SOURCE_BINDING", changed, rebound(profile, changed))

    def test_historical_status_strings_cannot_confer_authority(self):
        profile, raw = profile_and_projection()
        data = json.loads(raw["SOURCE_MANIFEST.json"])
        data.update(scientific_effect="PROMOTED", goal_status="CLOSED",
                    qualification_role="PASS_ALL", compiled_inventory_admitted=True)
        changed = dict(raw, **{"SOURCE_MANIFEST.json": json.dumps(data).encode()})
        scope = api.extract_scope(changed, rebound(profile, changed), profile["source"])
        self.assert_blocked(scope)
        self.assertEqual(scope["targets"], profile["targets"])

    def test_source_files_exact_set_and_raw_hashes(self):
        raw = {"Alpha.lean": b"alpha", "Beta.lean": b"beta"}
        scope = {"modules": [{"path": p, "source_sha256": hashlib.sha256(b).hexdigest()}
                             for p, b in raw.items()]}
        self.assertIsNone(api.verify_source_files(scope, raw))
        for changed in ({}, dict(raw, **{"Alpha.lean": b"changed"}),
                        dict(raw, **{"Unexpected.lean": b"extra"})):
            with self.subTest(keys=list(changed)):
                with self.assertRaises(api.ScopeError):
                    api.verify_source_files(scope, changed)

    def test_missing_extra_input_duplicate_json_key_and_escaping_path_refused(self):
        profile, raw = profile_and_projection()
        missing = dict(raw); del missing["lean-toolchain"]
        self.scope_error("INPUT_BINDING", missing, profile)
        self.scope_error("INPUT_BINDING", dict(raw, unexpected=b"x"), profile)
        text = raw["SOURCE_MANIFEST.json"].decode()
        changed = dict(raw, **{"SOURCE_MANIFEST.json":
                              text.replace('"inherited_input_count": 32',
                                           '"inherited_input_count": 32, "inherited_input_count": 32', 1).encode()})
        self.scope_error("JSON_PROTOCOL", changed, rebound(profile, changed))
        for path in ("../escape.lean", "/absolute.lean"):
            data = json.loads(raw["SOURCE_MANIFEST.json"])
            data["source_modules"][0] = path
            changed = dict(raw, **{"SOURCE_MANIFEST.json": json.dumps(data).encode()})
            self.scope_error("PATH_SCOPE", changed, rebound(profile, changed))


class NameContracts(ContractCase):
    def test_lossless_unicode_quoted_numeric_private_and_empty_components(self):
        vectors = [
            (("I4SourceSmoothParameterSpace", "weightedCarrier_memℓp"),
             encoded("I4SourceSmoothParameterSpace", "weightedCarrier_memℓp")),
            (("a.b", 0, "x"), "s612e62/n0/s78"),
            (("a", "b", "0"), "s61/s62/s30"),
            (("_private", ".lake", "Alpha", 0, "hidden"), PRIVATE.replace("/s416c706861/", "/s2e6c616b65/s416c706861/")),
            (("",), "s"), ((), ""),
        ]
        for name, wire in vectors:
            with self.subTest(name=name):
                self.assertEqual(api.encode_name(name), wire)
                self.assertEqual(api.decode_name(wire), name)
        distinct = [encoded("ℓ"), encoded("l"), encoded("é"), encoded("e\u0301"),
                    encoded("a.b"), encoded("a", "b"), encoded(0), encoded("0")]
        self.assertEqual(len({api.decode_name(x) for x in distinct}), len(distinct))

    def test_malformed_or_noncanonical_name_encodings_refused(self):
        for wire in ("s0", "sGG", "sff", "sC3A9", "n", "n01", "n-1", "s61//s62", "/s61", "s61/"):
            with self.subTest(wire=wire):
                with self.assertRaises(api.NameProtocolError):
                    api.decode_name(wire)
        for name in ((True,), (-1,), (1.5,)):
            with self.subTest(name=name):
                with self.assertRaises(api.NameProtocolError):
                    api.encode_name(name)


class InventoryContracts(ContractCase):
    def test_complete_private_parts_foreign_namespace_and_data_are_retained(self):
        inventory = api.parse_inventory(transcript(records()))
        self.assertEqual(len(inventory["modules"]), 2)
        self.assertEqual({x["name"] for x in inventory["declarations"]},
                         {PUBLIC, PRIVATE, DATA, FOREIGN})
        self.assertEqual(inventory["axioms"], [])
        result = api.classify_inventory(inventory, fixture_scope())
        self.assert_blocked(result)
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["counts"], {"owned_declarations": 4, "public_targets": 2,
                                           "proposition_targets": 1, "data_targets": 1})
        self.assertTrue(BLOCKERS.issubset(set(result["blockers"])))

    def test_truncation_duplicate_keys_trailing_records_and_noise_refused(self):
        good = records()
        malformed = [
            transcript(good[:-1]), transcript(good + [good[0]]),
            transcript([good[0], good[0]] + good[1:]),
            transcript(good).replace('"schema":1', '"schema":1,"schema":1', 1),
            transcript(good).replace('"const_count":2', '"const_count":true', 1),
            transcript(good).replace('"const_count":2', '"const_count":NaN', 1),
            "noise\n" + transcript(good), transcript(good) + "\n",
        ]
        for text in malformed:
            with self.subTest(text=text[:70]):
                with self.assertRaises(api.InventoryProtocolError):
                    api.parse_inventory(text)

    def test_module_part_length_identity_duplicates_visibility_and_rows_reconcile(self):
        mutations = []
        x = records(); x[1]["object_parts"][-1]["constant_names"] = [PUBLIC]; mutations.append(x)
        x = records(); x[1]["object_parts"][-1]["constant_names"] = [PRIVATE, PUBLIC]; mutations.append(x)
        x = records(); x[1]["object_parts"][-1] = part("private", [PUBLIC, PUBLIC]); mutations.append(x)
        x = records(); x[1]["object_parts"] = list(reversed(x[1]["object_parts"])); mutations.append(x)
        x = records(); x[1]["const_count"] = 1; mutations.append(x)
        x = records(); del x[4]; x[-1]["declaration_count"] = 3; mutations.append(x)
        x = records(); x[4] = copy.deepcopy(x[3]); mutations.append(x)
        x = records(); x[3]["object_part_sha256"] = "0" * 64; mutations.append(x)
        for rows in mutations:
            with self.subTest(rows=rows[1]):
                with self.assertRaises(api.InventoryProtocolError):
                    api.parse_inventory(transcript(rows))

    def test_scope_target_prop_data_extras_and_skip_findings_are_explicit(self):
        for change, code in (("target", "TARGET_SCOPE"), ("data", "TARGET_KIND_COUNTS"),
                             ("extras", "UNSUPPORTED_EXTRAS"), ("unsafe", "UNSAFE_REPLAY_SKIP"),
                             ("partial", "PARTIAL_REPLAY_SKIP")):
            rows, scope = records(), fixture_scope()
            if change == "target": scope["targets"] = ["Public.absent", "Public.d"]
            elif change == "data": rows[5]["is_prop"] = True
            elif change == "extras": rows[1]["extra_const_names"] = [encoded("extra")]
            else: rows[4][change] = True
            result = api.classify_inventory(api.parse_inventory(transcript(rows)), scope)
            self.assert_blocked(result)
            self.assertIn(code, {x["code"] for x in result["findings"]})

    def test_private_forbidden_axiom_and_owned_axiom_never_admitted(self):
        rows = records()
        rows[4]["axioms"] = [encoded("sorryAx")]
        rows[-1]["axioms"] = [encoded("sorryAx")]
        result = api.classify_inventory(api.parse_inventory(transcript(rows)), fixture_scope())
        self.assert_blocked(result)
        self.assertIn({"code": "FORBIDDEN_AXIOM", "name": encoded("sorryAx")}, result["findings"])
        rows = records()
        rows[4]["kind"] = "axiom"
        rows[4]["axioms"] = [encoded("propext")]
        rows[-1]["axioms"] = [encoded("propext")]
        result = api.classify_inventory(api.parse_inventory(transcript(rows)), fixture_scope())
        self.assertIn({"code": "OWNED_AXIOM", "name": PRIVATE}, result["findings"])
        rows[-1]["axioms"] = []
        with self.assertRaises(api.InventoryProtocolError):
            api.parse_inventory(transcript(rows))

    def test_missing_scope_module_and_input_owned_target_are_reported(self):
        rows = records()
        rows = [row for row in rows if not
                (row["record"] == "module" and row["name"] == BETA or
                 row["record"] == "declaration" and row["module"] == BETA)]
        rows[-1].update(module_count=1, declaration_count=2)
        result = api.classify_inventory(api.parse_inventory(transcript(rows)), fixture_scope())
        self.assertIn("MODULE_SCOPE", {x["code"] for x in result["findings"]})
        scope = fixture_scope()
        scope["modules"][1]["role"] = "input"
        result = api.classify_inventory(api.parse_inventory(transcript(records())), scope)
        self.assertIn("TARGET_OWNER", {x["code"] for x in result["findings"]})
        self.assert_blocked(result)


class ReceiptContracts(ContractCase):
    def test_positive_fixture_validates_declared_fields_without_execution_credit(self):
        value = receipt()
        result = api.classify_receipt(value, copy.deepcopy(value))
        self.assert_blocked(result)
        self.assertEqual(result["binding_status"], "DECLARED_FIELDS_VALID")
        self.assertEqual(result["findings"], [])
        self.assertFalse(result["execution_authenticated"])
        self.assertTrue(BLOCKERS.issubset(set(result["blockers"])))

    def test_source_tool_profile_object_log_and_attempt_binding_mismatches(self):
        expected = receipt()
        for path in (
            ("source", "commit"), ("source", "tree"), ("source", "repository"),
            ("toolchain", "lean_sha256"), ("adapter_sha256",), ("profile_sha256",),
            ("owned_objects_sha256",), ("run", "attempt"), ("run", "id"), ("run", "job_id"),
        ):
            changed = copy.deepcopy(expected)
            cursor = changed
            for key in path[:-1]: cursor = cursor[key]
            old = cursor[path[-1]]
            cursor[path[-1]] = (999 if type(old) is int else "0" * len(old)
                                  if len(old) in (40, 64) else "other/Repository")
            with self.subTest(path=path):
                self.invalid_receipt(changed, expected, "RECEIPT_BINDING")
        changed = copy.deepcopy(expected); changed["steps"][0]["stdout_sha256"] = H
        self.invalid_receipt(changed, expected, "RECEIPT_BINDING")

    def test_missing_fields_drift_coverage_skips_and_step_failure_are_not_pass(self):
        expected = receipt()
        for change, code in (
            ("missing_signal", "RECEIPT_SCHEMA"), ("missing_timeout", "RECEIPT_SCHEMA"),
            ("bool_exit", "RECEIPT_SCHEMA"), ("nonzero", "STEP_FAILED"),
            ("signal", "STEP_FAILED"), ("timeout", "STEP_FAILED"),
            ("drift", "SOURCE_DRIFT"), ("cache", "CACHE_DRIFT"),
            ("coverage", "MODULE_COVERAGE"), ("unsafe", "REPLAY_SKIPPED"),
            ("partial", "REPLAY_SKIPPED"), ("missing_step", "STEP_COVERAGE"),
            ("duplicate_step", "STEP_COVERAGE"), ("negative_resource", "RECEIPT_SCHEMA"),
        ):
            changed = copy.deepcopy(expected)
            if change == "missing_signal": del changed["steps"][0]["signal"]
            elif change == "missing_timeout": del changed["steps"][0]["timed_out"]
            elif change == "bool_exit": changed["steps"][0]["exit_code"] = True
            elif change == "nonzero": changed["steps"][0]["exit_code"] = 1
            elif change == "signal": changed["steps"][0]["signal"] = 9
            elif change == "timeout": changed["steps"][0]["timed_out"] = True
            elif change == "drift": changed["inputs_after_sha256"] = "0" * 64
            elif change == "cache": changed["cache_after_sha256"] = "0" * 64
            elif change == "coverage": changed["module_coverage"] = [ALPHA]
            elif change == "unsafe": changed["unsafe_skipped"] = [PRIVATE]
            elif change == "partial": changed["partial_skipped"] = [PRIVATE]
            elif change == "missing_step": changed["steps"].pop()
            elif change == "duplicate_step": changed["steps"].append(copy.deepcopy(changed["steps"][0]))
            else: changed["steps"][0]["max_rss_kib"] = -1
            with self.subTest(change=change):
                self.invalid_receipt(changed, expected, code)

    def test_missing_whole_identity_coverage_or_log_field_is_schema_failure(self):
        expected = receipt()
        for field in ("source", "toolchain", "profile_sha256", "run", "module_coverage",
                      "unsafe_skipped", "partial_skipped", "owned_objects_sha256"):
            changed = copy.deepcopy(expected); del changed[field]
            with self.subTest(field=field):
                self.invalid_receipt(changed, expected, "RECEIPT_SCHEMA")
        changed = copy.deepcopy(expected); del changed["steps"][0]["stdout_sha256"]
        self.invalid_receipt(changed, expected, "RECEIPT_SCHEMA")


    def test_date_only_basic_or_incomplete_utc_rejected_when_bindings_match(self):
        for timestamp in ("2026-11-09Z", "20261109T000000Z",
                          "2026-11-09T00Z", "2026-11-09T00:00Z"):
            value = receipt()
            for item in value["steps"]:
                item["started_utc"] = timestamp
                item["ended_utc"] = timestamp
            with self.subTest(timestamp=timestamp):
                self.invalid_receipt(value, copy.deepcopy(value), "RECEIPT_SCHEMA")

    def test_empty_module_coverage_rejected_when_both_bindings_match(self):
        value = receipt()
        value["module_coverage"] = []
        value["steps"] = [step("build"), step("inventory")]
        self.invalid_receipt(value, copy.deepcopy(value), "MODULE_COVERAGE")



if __name__ == "__main__":
    unittest.main()
