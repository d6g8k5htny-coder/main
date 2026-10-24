"""Frozen independent first-core contract controls; execution is hosted only.

The accepted conformance data are synthetic byte oracles. The separate proposed
internal collection profile carries no native, mapping, review or custody authority.
No working implementation was used to derive these expected values.
"""
import copy
import base64
import re
import hashlib
import json
from pathlib import Path
import unittest

from architecture.formal_regression import (
    decode_request, decode_policy, request_key, content_inventory, inventory_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "4a469c57ba103d8937795c709b59765a689ae24f92b6ee1d55ddc74037b849e6"


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


class ContractControls(unittest.TestCase):
    def setUp(self):
        self.vectors = json.loads((ROOT / "tests/fixtures/formal_regression/inventory-conformance.json").read_bytes())
        self.profile = json.loads((ROOT / "tests/fixtures/formal_regression/control-plan.json").read_bytes())["synthetic_test_boundary_profile"]

    def core(self):
        return copy.deepcopy(self.vectors["cases"][0]["core"])

    def request(self):
        return copy.deepcopy(self.profile["request"])

    def policy(self):
        return copy.deepcopy(self.profile["diagnostic_policy"])

    def reject_core(self, path, value=None, delete=False):
        core = self.core()
        self.assertEqual(inventory_sha256(core), BASELINE)
        at = core
        for part in path[:-1]:
            at = at[part]
        if delete:
            del at[path[-1]]
        else:
            at[path[-1]] = value
        with self.assertRaises(ValueError):
            inventory_sha256(core)

    def collection(self, mutation=None):
        profile = copy.deepcopy(self.profile)
        self.assert_held_profile(profile)
        if mutation is not None:
            mutation(profile)
        return content_inventory(profile["records"], profile["diagnostic_policy"],
                                 profile["membership"], profile["acceptances"])

    def collection_refusal(self, mutation, reason):
        self.assert_collection_baseline(self.collection())
        result = self.collection(mutation)
        self.assertIsNone(result["formal_inventory_sha256"])
        self.assertIn(reason, result["issues"])
        self.assertEqual(result["issues"], sorted(set(result["issues"])))

    def assert_wire_shape(self, value, shape):
        """Test-only fixture integrity; never supplies native qualification."""
        types = self.profile["wire_contract"]["types"]
        if isinstance(shape, str) and shape in types:
            return self.assert_wire_shape(value, types[shape])
        if isinstance(shape, str):
            if shape == "null":
                self.assertIsNone(value)
            elif shape in ["uint", "positive_int"]:
                self.assertIs(type(value), int)
                self.assertGreaterEqual(value, 1 if shape == "positive_int" else 0)
            else:
                self.assertIs(type(value), str)
                if shape != "string":
                    self.assertNotEqual(value, "")
                patterns = {"hex40": r"[0-9a-f]{40}", "sha256": r"[0-9a-f]{64}",
                            "repository": r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
                            "name": r"[A-Za-z_][A-Za-z0-9_.']*"}
                if shape in patterns:
                    self.assertIsNotNone(re.fullmatch(patterns[shape], value))
                if shape == "path":
                    self.assertFalse(value.startswith("/"))
                    self.assertNotIn("\\", value)
                    self.assertNotIn(":", value)
                    self.assertTrue(all(part not in ["", ".", ".."] for part in value.split("/")))
                    self.assertTrue(all(ord(c) >= 32 and ord(c) != 127 for c in value))
                if shape == "base64":
                    self.assertEqual(base64.b64encode(base64.b64decode(value, validate=True)).decode("ascii"), value)
            return
        kind = shape["kind"]
        if kind == "object":
            self.assertIs(type(value), dict)
            self.assertEqual(set(value), set(shape["fields"]))
            for key, child in shape["fields"].items():
                self.assert_wire_shape(value[key], child)
        elif kind == "array":
            self.assertIs(type(value), list)
            self.assertGreaterEqual(len(value), shape["min"])
            if "max" in shape:
                self.assertLessEqual(len(value), shape["max"])
            for item in value:
                self.assert_wire_shape(item, shape["item"])
            if "unique_by" in shape:
                keys = []
                for item in value:
                    pieces = []
                    for path in shape["unique_by"]:
                        at = item
                        for part in path.split("."):
                            at = at[part]
                        pieces.append(at)
                    keys.append(wire(pieces if shape["unique_by"] else item))
                self.assertEqual(len(keys), len(set(keys)))
        elif kind == "map":
            self.assertIs(type(value), dict)
            for key, child in value.items():
                self.assert_wire_shape(key, shape["key"])
                self.assert_wire_shape(child, shape["value"])
        elif kind == "enum":
            self.assertTrue(any(type(value) is type(item) and value == item for item in shape["values"]))
        elif kind == "nullable":
            if value is not None:
                self.assert_wire_shape(value, shape["item"])
        elif kind == "union":
            branches = []
            for candidate in shape["items"]:
                try:
                    self.assert_wire_shape(value, candidate)
                except AssertionError:
                    branches.append(False)
                else:
                    branches.append(True)
            self.assertEqual(branches.count(True), 1)
        else:
            self.fail("unknown fixture schema kind")

    def assert_held_profile(self, profile):
        """Check every raw fixture prerequisite before an isolated mutation."""
        for key, typename in [("membership", "Membership"), ("acceptances", "Acceptances"), ("diagnostic_policy", "DiagnosticPolicy")]:
            self.assert_wire_shape(profile[key], typename)
        membership = profile["membership"]
        self.assertEqual(profile["request"]["policy_sha256"], hashlib.sha256(wire(profile["diagnostic_policy"])).hexdigest())
        selected = profile["request"]["base" if membership["side"] == "base" else "checked_merge"]
        self.assertEqual(membership["snapshot"], selected)
        rows = {}
        for row in membership["sources"]:
            ref = row["ref"]
            raw = base64.b64decode(row["raw_base64"], validate=True)
            self.assertEqual(len(raw), ref["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), ref["sha256"])
            self.assertEqual(hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest(), ref["git_blob"])
            identity = (ref["repository"], ref["path"])
            self.assertNotIn(identity, rows)
            rows[identity] = (ref, raw, row)
            if row["side"] != "origin":
                self.assertEqual(row["side"], membership["side"])
                self.assertEqual((ref["repository"], ref["commit"]), (selected["repository"], selected["commit"]))
        def resolved(ref):
            original, raw, row = rows[(ref["repository"], ref["path"])]
            self.assertEqual(ref, original)
            return raw
        def refs(value):
            if isinstance(value, dict):
                if set(value) == {"repository", "commit", "path", "git_blob", "sha256", "bytes"}:
                    resolved(value)
                else:
                    for child in value.values():
                        refs(child)
            elif isinstance(value, list):
                for child in value:
                    refs(child)
        for value in [membership["role_inputs"], membership["target_inputs"], membership["producer_observation"], membership["producer_observation_ref"], profile["records"], profile["diagnostic_policy"]]:
            refs(value)
        native_row = next(row for row in membership["sources"] if row["role"] == "native_manifest")
        native = json.loads(resolved(native_row["ref"]))
        self.assert_wire_shape(native, "NativeManifest")
        self.assertEqual(membership["package"], {"declared_name": native["package"], "manifest_path": native_row["ref"]["path"]})
        names = [target["name"] for target in native["targets"]]
        self.assertEqual(sorted(names), sorted(membership["native_targets"]))
        self.assertEqual(sorted(names), sorted(record["target"] for record in profile["records"]))
        self.assertEqual(sorted(names), sorted(item["target"] for item in membership["target_inputs"]))
        candidate = {path: (ref, raw) for (repo, path), (ref, raw, row) in rows.items() if row["side"] == membership["side"]}
        self.assertEqual(set(candidate), set(native["files"]) | set(native["unbound_files"]))
        for path, digest in native["files"].items():
            self.assertEqual(candidate[path][0]["sha256"], digest)
        self.assertEqual(candidate["formal/lean-toolchain"][1].decode("ascii").strip(), native["lean_toolchain"])
        root_imports = candidate[native["package_root"] + "/" + native["root_module"] + ".lean"][1].decode("utf-8").splitlines()
        self.assertEqual(root_imports, ["import " + module[:-5].replace("/", ".") for module in native["source_modules"]])
        lock = json.loads(resolved(profile["records"][0]["compiler_context"]["dependency_lock_ref"]))
        self.assert_wire_shape(lock, "LakeLock")
        dependencies = {item["name"]: item["rev"] for item in lock["packages"]}
        self.assertEqual(dependencies, native["dependency_revisions"])
        self.assertEqual(dependencies, {item["name"]: item["revision"] for item in membership["native_dependencies"]})
        observation = membership["producer_observation"]
        self.assertEqual(json.loads(resolved(membership["producer_observation_ref"])), observation)
        policy = profile["diagnostic_policy"]["native_producer_policies"][0]
        self.assertEqual((policy["workflow_source_ref"]["repository"], policy["workflow_source_ref"]["commit"]),
                         (profile["request"]["control"]["repository"], profile["request"]["control"]["commit"]))
        self.assertEqual(policy["toolchain"], native["lean_toolchain"])
        self.assertEqual(policy["print_options"], observation["print_options"])
        self.assertEqual(membership["print_options"], observation["print_options"])
        self.assertEqual(policy["runtime_identity"], observation["runtime_identity"])
        self.assertEqual(policy["source_control_files"], dict(membership["role_inputs"], target_inputs=membership["target_inputs"]))
        self.assertEqual(policy["dependency_policy"], {"lock_ref": observation["dependency_lock_ref"], "revisions": observation["dependencies"]})
        producer_keys = lambda values: sorted(wire(value) for value in values)
        self.assertEqual(producer_keys(membership["role_inputs"]["producer_sources"]), producer_keys([policy["workflow_source_ref"]] + policy["supervisor_source_refs"]))
        for ref, observed in [(observation["compiler_executable_ref"], "executable"), (observation["compiler_version_ref"], "version_output")]:
            resolved(ref)
            self.assertEqual({key: ref[key] for key in ["bytes", "sha256"]}, profile["records"][0]["compiler_context"]["compiler_identity"][observed])
        for item in observation["runtime_executable_refs"]:
            resolved(item["ref"])
            executable = next(row for row in observation["runtime_identity"]["executables"] if row["name"] == item["name"])
            self.assertEqual(executable["content"], {key: item["ref"][key] for key in ["bytes", "sha256"]})
        module_paths = {module[:-5].replace("/", "."): native["package_root"] + "/" + module for module in native["source_modules"]}
        def imports(path, seen):
            for line in candidate[path][1].decode("utf-8").splitlines():
                if line.startswith("import "):
                    imported = module_paths[line[7:]]
                    if imported not in seen:
                        seen.add(imported)
                        imports(imported, seen)
            return seen
        for record in profile["records"]:
            self.assert_wire_shape(record, "TargetStatement")
            target = next(item for item in native["targets"] if item["name"] == record["target"])
            self.assertEqual(record["manifest_ref"], native_row["ref"])
            self.assertEqual(record["module_ref"]["path"], native["package_root"] + "/" + target["module"])
            self.assertEqual(record["scope_ref"]["path"], native["package_root"] + "/SCOPE.md")
            origin = next(item for item in native["sources"] if item["id"] == target["source"])
            self.assertEqual({key: record["origin_ref"][key] for key in ["repository", "commit", "path", "bytes", "sha256"]}, {key: origin[key] for key in ["repository", "commit", "path", "bytes", "sha256"]})
            self.assertEqual(record["checked_local_copy_ref"]["path"], origin["local_copy"])
            self.assertEqual(resolved(record["origin_ref"]), resolved(record["checked_local_copy_ref"]))
            self.assertIn(target["informal_anchor"].encode("utf-8"), resolved(record["checked_local_copy_ref"]))
            context = record["compiler_context"]
            self.assertEqual(context["toolchain"], observation["toolchain"])
            self.assertEqual(context["options"], observation["options"])
            self.assertEqual(context["runtime_identity"], observation["runtime_identity"])
            self.assertEqual(context["dependency_revisions"], observation["dependencies"])
            self.assertEqual(producer_keys(context["producer_source_refs"]), producer_keys(membership["role_inputs"]["producer_sources"]))
            expected_imports = imports(record["module_ref"]["path"], set())
            target_input = next(item for item in membership["target_inputs"] if item["target"] == record["target"])
            self.assertEqual({ref["path"] for ref in target_input["imports"]}, expected_imports)
        for link in profile["acceptances"]["links"]:
            record = next(item for item in profile["records"] if item["target"] == link["target"])
            for field in ["formal_statement_selector", "formal_proof_selector"]:
                selected_slice = link[field]
                if selected_slice is not None:
                    ref = record["module_ref"]
                    self.assertEqual(selected_slice["source"], dict(kind="snapshot", **{key: value for key, value in ref.items() if key != "commit"}))
                    raw = resolved(ref)
                    start, end = selected_slice["start_byte"], selected_slice["end_byte"]
                    self.assertLess(start, end)
                    self.assertLessEqual(end, len(raw))
                    self.assertEqual(hashlib.sha256(raw[start:end]).hexdigest(), selected_slice["sha256"])
        self.assertEqual(wire(profile["expected_diagnostic_core"]), profile["expected_diagnostic_ascii_preimage"].encode("ascii"))
        self.assertEqual(hashlib.sha256(wire(profile["expected_diagnostic_core"])).hexdigest(), profile["expected_diagnostic_sha256"])

    def assert_collection_baseline(self, result):
        self.assertIsNone(result["formal_inventory_sha256"])
        self.assertEqual(result["core"], self.profile["expected_diagnostic_core"])
        self.assertEqual(result["diagnostic_inventory_sha256"], self.profile["expected_diagnostic_sha256"])
        self.assertEqual(result["issues"], self.profile["expected_issues"])

    def mutate_path(self, profile, path, value=None, delete=False):
        at = profile
        for part in path[:-1]:
            at = at[part]
        if delete:
            del at[path[-1]]
        else:
            at[path[-1]] = value

    def rewrite_observation_raw(self, profile):
        """Test-only alternate raw observation bytes, with all full refs retained."""
        membership = profile["membership"]
        row = next(item for item in membership["sources"] if item["role"] == "producer_observation")
        raw = wire(membership["producer_observation"])
        ref = dict(row["ref"], bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                   git_blob=hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest())
        row["raw_base64"] = base64.b64encode(raw).decode("ascii")
        row["ref"] = ref
        membership["producer_observation_ref"] = copy.deepcopy(ref)

    def mutate_native_manifest_wire(self, profile, path, value):
        """One malformed raw native wire condition, with full raw-ref joins held."""
        row = next(item for item in profile["membership"]["sources"] if item["role"] == "native_manifest")
        native = json.loads(base64.b64decode(row["raw_base64"], validate=True))
        self.mutate_path(native, path, copy.deepcopy(value))
        raw = wire(native)
        ref = dict(row["ref"], bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                   git_blob=hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest())
        row["raw_base64"] = base64.b64encode(raw).decode("ascii")
        row["ref"] = ref
        for record in profile["records"]:
            record["manifest_ref"] = copy.deepcopy(ref)

    def mutate_raw_observation_wire(self, profile):
        row = next(item for item in profile["membership"]["sources"] if item["role"] == "producer_observation")
        value = json.loads(base64.b64decode(row["raw_base64"], validate=True))
        value["unadmitted"] = "forged-extra"
        raw = wire(value)
        ref = dict(row["ref"], bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                   git_blob=hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest())
        row["raw_base64"] = base64.b64encode(raw).decode("ascii")
        row["ref"] = ref
        profile["membership"]["producer_observation_ref"] = copy.deepcopy(ref)

    def permute_unordered_inputs(self, profile):
        membership = profile["membership"]
        profile["records"].reverse()
        profile["acceptances"]["links"].reverse()
        for field in ["sources", "native_targets", "native_dependencies", "target_inputs"]:
            membership[field].reverse()
        for values in membership["role_inputs"].values():
            values.reverse()
        for target in membership["target_inputs"]:
            target["imports"].reverse()
        observation = membership["producer_observation"]
        observation["dependencies"].reverse()
        observation["runtime_identity"]["executables"].reverse()
        observation["runtime_executable_refs"].reverse()
        for record in profile["records"]:
            record["compiler_context"]["dependency_revisions"].reverse()
            record["compiler_context"]["producer_source_refs"].reverse()
            record["compiler_context"]["runtime_identity"]["executables"].reverse()
        producer = profile["diagnostic_policy"]["native_producer_policies"][0]
        producer["runtime_identity"]["executables"].reverse()
        producer["dependency_policy"]["revisions"].reverse()
        for field in ["build_config", "producer_sources", "scope_files", "import_files", "target_inputs"]:
            producer["source_control_files"][field].reverse()
        for target in producer["source_control_files"]["target_inputs"]:
            target["imports"].reverse()
        self.rewrite_observation_raw(profile)

    def reverse_effective_options(self, profile, field):
        membership = profile["membership"]
        membership["producer_observation"][field].reverse()
        if field == "options":
            for record in profile["records"]:
                record["compiler_context"]["options"].reverse()
        else:
            membership["print_options"].reverse()
            profile["diagnostic_policy"]["native_producer_policies"][0]["print_options"].reverse()
        self.rewrite_observation_raw(profile)


    def test_inventory_baseline_hash(self):
        core = self.core()
        self.assertEqual(inventory_sha256(core), '4a469c57ba103d8937795c709b59765a689ae24f92b6ee1d55ddc74037b849e6')

    def test_recollection_keeps_content_digest(self):
        first_core = self.vectors["cases"][0]["core"]
        second_core = self.vectors["cases"][1]["core"]
        self.assertEqual(inventory_sha256(first_core), inventory_sha256(second_core))
        self.assertEqual(inventory_sha256(first_core), BASELINE)

    def test_seven_content_deltas(self):
        baseline_hash = inventory_sha256(self.core())
        changed_hashes = [inventory_sha256(case["core"]) for case in self.vectors["cases"][2:9]]
        self.assertEqual(len(set(changed_hashes)), 7)
        self.assertNotIn(baseline_hash, changed_hashes)

    def test_unknown_required_type_is_diagnostic(self):
        result = self.collection()
        self.assertIsNone(result["formal_inventory_sha256"])
        self.assertIsNotNone(result["diagnostic_inventory_sha256"])
        self.assertEqual(result["core"]["targets"][0]["unresolved_fields"], ["type_content_bytes", "type_content_sha256"])

        self.assert_collection_baseline(result)
        self.assertEqual([target["target"] for target in result["core"]["targets"]], ["Demo.t", "Demo.unlinked"])
        for target in result["core"]["targets"]:
            self.assertEqual(target["unresolved_fields"], ["type_content_bytes", "type_content_sha256"])
        self.assertEqual(result["core"]["targets"][1]["accepted_formal_statement_selector"], [])
        self.assertEqual(result["core"]["targets"][1]["accepted_formal_proof_selector"], [])

    def test_duplicate_identity_refuses(self):
        duplicate_core = self.core()
        duplicate_core["targets"].append(copy.deepcopy(duplicate_core["targets"][0]))
        with self.assertRaises(ValueError):
            inventory_sha256(duplicate_core)

    def test_canonical_vector_0(self):
        case = self.vectors["cases"][0]
        self.assertEqual(inventory_sha256(case["core"]), '4a469c57ba103d8937795c709b59765a689ae24f92b6ee1d55ddc74037b849e6')

    def test_canonical_vector_1(self):
        case = self.vectors["cases"][1]
        self.assertEqual(inventory_sha256(case["core"]), '4a469c57ba103d8937795c709b59765a689ae24f92b6ee1d55ddc74037b849e6')

    def test_canonical_vector_2(self):
        case = self.vectors["cases"][2]
        self.assertEqual(inventory_sha256(case["core"]), 'c507727b90f939d5aad3a4eff41868ab84c8cb6bcb972b3c76be3ea00186fa8a')

    def test_canonical_vector_3(self):
        case = self.vectors["cases"][3]
        self.assertEqual(inventory_sha256(case["core"]), '5ff478abe3a20e67de8ec879e8a91930256dab1a68c8ea04263bc1296c59c4f3')

    def test_canonical_vector_4(self):
        case = self.vectors["cases"][4]
        self.assertEqual(inventory_sha256(case["core"]), 'ed156dc4afac2484f25ef5ec797258d72295a787f9fb642b913fa7854a34152e')

    def test_canonical_vector_5(self):
        case = self.vectors["cases"][5]
        self.assertEqual(inventory_sha256(case["core"]), '01127aa13ad6f9e145489e5ceaff15426926637de8245e98b11e8953de6820dd')

    def test_canonical_vector_6(self):
        case = self.vectors["cases"][6]
        self.assertEqual(inventory_sha256(case["core"]), '1e79cef0d148509119876ed49ab75d86bee2e0640b31be80e74ca2ef38a4cfc6')

    def test_canonical_vector_7(self):
        case = self.vectors["cases"][7]
        self.assertEqual(inventory_sha256(case["core"]), '364dc7d33c64bc68faf197b77e1abbdd8cd4ea0472c44f67712d5dd58a9dadb1')

    def test_canonical_vector_8(self):
        case = self.vectors["cases"][8]
        self.assertEqual(inventory_sha256(case["core"]), 'ebf1d7f9ed021c8724d19f4320f2f271fbd77bc490f68720fb3ab11817e8fbed')

    def test_fixture_preimages_are_independent(self):
        self.assertEqual(len(self.vectors["cases"]), 9)
        for case in self.vectors["cases"]:
            with self.subTest(case=case["id"]):
                raw = case["canonical_ascii_preimage"].encode("ascii")
                self.assertEqual(wire(case["core"]), raw)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), case["sha256"])

        self.assert_held_profile(self.profile)
        oracles = self.profile["origin_commit_oracles"]
        self.assertNotEqual(oracles[0]["commit"], oracles[1]["commit"])
        self.assertNotEqual(oracles[0]["canonical_ascii_preimage"], oracles[1]["canonical_ascii_preimage"])
        for oracle in oracles:
            raw = oracle["canonical_ascii_preimage"].encode("ascii")
            self.assertEqual(len(raw), oracle["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), oracle["sha256"])

    def test_request_decode_roundtrip(self):
        expected = self.request()
        self.assertEqual(decode_request(wire(expected)), expected)

    def test_request_key_exact_domain_formula(self):
        self.assertEqual(request_key(self.request()), 'ac1cffe262a84ab7fb9ee3eece0d93b2a3e0b89b00a7ff1c9d747b9aa5abbd8d')

    def test_request_key_changes_each_source_identity(self):
        baseline = request_key(self.request())
        self.assertEqual(baseline, 'ac1cffe262a84ab7fb9ee3eece0d93b2a3e0b89b00a7ff1c9d747b9aa5abbd8d')
        for field in ["base", "head", "checked_merge", "control"]:
            changed = self.request()
            changed[field]["commit"] = "e" * 40
            if field == "base":
                changed["checked_merge_parents"][0] = "e" * 40
            if field == "head":
                changed["checked_merge_parents"][1] = "e" * 40
            with self.subTest(field=field):
                self.assertNotEqual(request_key(changed), baseline)
        changed = self.request()
        changed["policy_sha256"] = "f" * 64
        self.assertNotEqual(request_key(changed), baseline)

    def test_request_key_refuses_acquisition_fields(self):
        context = self.request()
        context["acquisition_id"] = "fixture-acquisition"
        with self.assertRaises(ValueError):
            request_key(context)

    def test_request_refuses_duplicate_top_key(self):
        raw = wire(self.request()).replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1')
        with self.assertRaisesRegex(ValueError, "(?i)duplicat"):
            decode_request(raw)

    def test_request_refuses_duplicate_nested_key(self):
        raw = wire(self.request()).replace(b'"base":{', b'"base":{"commit":"'+b'1'*40+b'",', 1)
        with self.assertRaisesRegex(ValueError, '(?i)duplicat'):
            decode_request(raw)

    def test_request_refuses_trailing_value(self):
        raw = wire(self.request()) + b' {}'
        with self.assertRaisesRegex(ValueError, '(?i)(trailing|extra data)'):
            decode_request(raw)

    def test_request_refuses_nonfinite_nan(self):
        raw = wire(self.request()).replace(b'"schema_version":1', b'"schema_version":NaN')
        with self.assertRaisesRegex(ValueError, '(?i)(finite|nan|infinity)'):
            decode_request(raw)

    def test_request_refuses_nonfinite_infinity(self):
        raw = wire(self.request()).replace(b'"schema_version":1', b'"schema_version":Infinity')
        with self.assertRaisesRegex(ValueError, '(?i)(finite|nan|infinity)'):
            decode_request(raw)

        overflow = wire(self.request()).replace(b'"schema_version":1', b'"schema_version":1e999')
        with self.assertRaisesRegex(ValueError, '(?i)(finite|nan|infinity)'):
            decode_request(overflow)

    def test_request_refuses_nonfinite_negative_infinity(self):
        raw = wire(self.request()).replace(b'"schema_version":1', b'"schema_version":-Infinity')
        with self.assertRaisesRegex(ValueError, '(?i)(finite|nan|infinity)'):
            decode_request(raw)

    def test_request_refuses_invalid_utf8(self):
        raw = b'\xff'
        with self.assertRaises(ValueError):
            decode_request(raw)

    def test_request_refuses_bom(self):
        raw = b'\xef\xbb\xbf' + wire(self.request())
        with self.assertRaises(ValueError):
            decode_request(raw)

    def test_request_refuses_depth_over_64(self):
        raw = b'[' * 65 + b'0' + b']' * 65
        with self.assertRaisesRegex(ValueError, '(?i)depth'):
            decode_request(raw)

    def test_request_refuses_oversize(self):
        raw = wire(self.request()) + b' ' * (16 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(ValueError, '(?i)(size|limit|large)'):
            decode_request(raw)

    def test_request_refuses_wrong_root_type(self):
        raw = b'[]'
        with self.assertRaises(ValueError):
            decode_request(raw)

    def test_request_refuses_raw_str(self):
        raw = '{}'
        with self.assertRaises(ValueError):
            decode_request(raw)

    def test_request_refuses_raw_bytearray(self):
        raw = bytearray(b'{}')
        with self.assertRaises(ValueError):
            decode_request(raw)

    def test_request_refuses_boolean_version(self):
        context = self.request()
        context['schema_version'] = True
        with self.assertRaises(ValueError):
            decode_request(wire(context))

    def test_request_refuses_boolean_native_id(self):
        context = self.request()
        context['pull_request'] = True
        with self.assertRaises(ValueError):
            decode_request(wire(context))

    def test_request_refuses_integer_native_id(self):
        context = self.request()
        context['pull_request'] = 17
        with self.assertRaises(ValueError):
            decode_request(wire(context))

    def test_request_refuses_noncanonical_native_id(self):
        context = self.request()
        context['pull_request'] = '017'
        with self.assertRaises(ValueError):
            decode_request(wire(context))

    def test_request_refuses_uppercase_digest(self):
        context = self.request()
        context['policy_sha256'] = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
        with self.assertRaises(ValueError):
            decode_request(wire(context))

    def test_request_refuses_head_fallback(self):
        context = self.request()
        context['merge_mode'] = 'head_only'
        with self.assertRaises(ValueError):
            decode_request(wire(context))

    def test_request_refuses_supplied_url(self):
        context = self.request()
        context['repository'] = 'https://fixture.invalid/repo'
        with self.assertRaises(ValueError):
            decode_request(wire(context))

    def test_request_refuses_missing_and_extra_fields(self):
        for field in self.request():
            context = self.request()
            del context[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                decode_request(wire(context))
        context = self.request()
        context["unknown"] = None
        with self.assertRaises(ValueError):
            decode_request(wire(context))

    def test_request_refuses_wrong_merge_parents(self):
        context = self.request()
        context["checked_merge_parents"] = [context["head"]["commit"], context["base"]["commit"]]
        with self.assertRaises(ValueError):
            decode_request(wire(context))

    def test_request_commit_refs_are_closed(self):
        for field in ["base", "head", "checked_merge", "control"]:
            for key in ["repository", "commit", "tree"]:
                context = self.request()
                del context[field][key]
                with self.subTest(field=field, key=key), self.assertRaises(ValueError):
                    decode_request(wire(context))

    def test_policy_decode_uses_exact_raw_pin(self):
        expected = self.policy()
        raw = wire(expected)
        self.assertEqual(decode_policy(raw, hashlib.sha256(raw).hexdigest()), expected)

    def test_policy_wrong_raw_pin_refuses(self):
        raw = wire(self.policy())
        with self.assertRaises(ValueError):
            decode_policy(raw, "0" * 64)

    def test_policy_whitespace_is_part_of_raw_pin(self):
        raw = wire(self.policy())
        with self.assertRaises(ValueError):
            decode_policy(raw + b"\n", hashlib.sha256(raw).hexdigest())

    def test_policy_closed_field_set(self):
        for field in self.policy():
            policy = self.policy()
            del policy[field]
            raw = wire(policy)
            with self.subTest(field=field), self.assertRaises(ValueError):
                decode_policy(raw, hashlib.sha256(raw).hexdigest())

        for example in self.profile["policy_closed_object_examples"]:
            original = self.policy()
            at = original
            for part in example["path"]:
                at = at[part]
            for field in list(at):
                broken = copy.deepcopy(original)
                node = broken
                for part in example["path"]:
                    node = node[part]
                del node[field]
                raw = wire(broken)
                with self.subTest(type=example["type"], missing=field), self.assertRaises(ValueError):
                    decode_policy(raw, hashlib.sha256(raw).hexdigest())
            at["unadmitted"] = None
            raw = wire(original)
            with self.subTest(type=example["type"], extra=True), self.assertRaises(ValueError):
                decode_policy(raw, hashlib.sha256(raw).hexdigest())

    def test_policy_refuses_duplicate(self):
        raw = wire(self.policy()).replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1')
        with self.assertRaisesRegex(ValueError, "(?i)duplicat"):
            decode_policy(raw, hashlib.sha256(raw).hexdigest())

    def test_policy_refuses_trailing(self):
        raw = wire(self.policy()) + b' {}'
        with self.assertRaisesRegex(ValueError, '(?i)(trailing|extra data)'):
            decode_policy(raw, hashlib.sha256(raw).hexdigest())

    def test_policy_refuses_nonfinite(self):
        raw = wire(self.policy()).replace(b'"schema_version":1', b'"schema_version":NaN')
        with self.assertRaisesRegex(ValueError, '(?i)(finite|nan|infinity)'):
            decode_policy(raw, hashlib.sha256(raw).hexdigest())

    def test_policy_refuses_depth(self):
        raw = b'[' * 65 + b'0' + b']' * 65
        with self.assertRaisesRegex(ValueError, '(?i)depth'):
            decode_policy(raw, hashlib.sha256(raw).hexdigest())

    def test_policy_refuses_oversize(self):
        raw = wire(self.policy()) + b' ' * (16 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(ValueError, '(?i)(size|limit|large)'):
            decode_policy(raw, hashlib.sha256(raw).hexdigest())

    def test_policy_boolean_version_refuses(self):
        policy = self.policy()
        policy["schema_version"] = True
        raw = wire(policy)
        with self.assertRaises(ValueError):
            decode_policy(raw, hashlib.sha256(raw).hexdigest())

    def test_inventory_root_is_closed(self):
        for field in ["domain", "domain_id", "targets"]:
            self.reject_core([field], delete=True)
        self.reject_core(["extra"], "value")

    def test_inventory_root_types_and_domain(self):
        for path, value in [(["domain"], "wrong/v1"), (["domain_id"], ""), (["targets"], []), (["targets"], {}), (["targets"], True)]:
            with self.subTest(path=path, value=value):
                self.reject_core(path, value)

    def test_all_thirteen_target_fields_are_required(self):
        fields = ['format', 'native_repository', 'package', 'target', 'module_content_identity', 'type_content_sha256', 'type_content_bytes', 'compiler_dependency_options_identity', 'accepted_formal_statement_selector', 'accepted_formal_proof_selector', 'manifest_contract_sha256', 'scope_content_identity', 'unresolved_fields']
        self.assertEqual(len(fields), 13)
        for field in fields:
            with self.subTest(field=field):
                self.reject_core(["targets", 0, field], delete=True)

    def test_target_unknown_field_refuses(self):
        self.reject_core(["targets", 0, "native_context"], {"run_id": "101"})

    def test_exact_integer_excludes_boolean(self):
        for path in [["targets", 0, "type_content_bytes"], ["targets", 0, "module_content_identity", "bytes"], ["targets", 0, "accepted_formal_proof_selector", 0, "selector", "start_byte"], ["targets", 0, "compiler_dependency_options_identity", "compiler_identity", "executable", "bytes"]]:
            with self.subTest(path=path):
                self.reject_core(path, True)

    def test_negative_or_fractional_sizes_refuse(self):
        for value in [-1, 1.5, "14"]:
            with self.subTest(value=value):
                self.reject_core(["targets", 0, "type_content_bytes"], value)

    def test_package_identity_closed_and_safe(self):
        self.reject_core(["targets", 0, "package", "manifest_path"], "../manifest.json")
        self.reject_core(["targets", 0, "package", "extra"], None)
        self.reject_core(["targets", 0, "package", "declared_name"], delete=True)

    def test_native_format_is_supported(self):
        self.reject_core(["targets", 0, "format"], "unknown-native/v99")

    def test_snapshot_source_has_exact_six_content_fields(self):
        for field in ["kind", "repository", "path", "git_blob", "sha256", "bytes"]:
            with self.subTest(field=field):
                self.reject_core(["targets", 0, "module_content_identity", field], delete=True)
        self.reject_core(["targets", 0, "module_content_identity", "commit"], "1" * 40)

    def test_source_paths_refuse_escape_absolute_and_url(self):
        for value in ["../Demo.lean", "/tmp/Demo.lean", "formal//Demo.lean", "formal/./Demo.lean", "https://fixture.invalid/Demo.lean", "formal\\Demo.lean"]:
            with self.subTest(path=value):
                self.reject_core(["targets", 0, "module_content_identity", "path"], value)

    def test_source_hashes_are_canonical(self):
        for field, value in [("git_blob", "A" * 40), ("git_blob", "1" * 39), ("sha256", "A" * 64), ("sha256", "1" * 63)]:
            with self.subTest(field=field, value=value):
                self.reject_core(["targets", 0, "module_content_identity", field], value)

    def test_origin_requires_all_six_full_ref_fields(self):
        core = self.core()
        snapshot = core["targets"][0]["module_content_identity"]
        ref = {key: value for key, value in snapshot.items() if key != "kind"}
        ref["commit"] = "1" * 40
        for field in ["repository", "commit", "path", "git_blob", "sha256", "bytes"]:
            broken = copy.deepcopy(core)
            origin = {"kind": "origin", "ref": copy.deepcopy(ref)}
            del origin["ref"][field]
            broken["targets"][0]["module_content_identity"] = origin
            with self.subTest(field=field), self.assertRaises(ValueError):
                inventory_sha256(broken)

    def test_compiler_dependency_content_closed(self):
        fields = ["toolchain", "compiler_identity", "options", "print_options", "dependency_lock", "dependencies", "producer_sources", "build_config", "runtime_identity"]
        for field in fields:
            with self.subTest(field=field):
                self.reject_core(["targets", 0, "compiler_dependency_options_identity", field], delete=True)
        self.reject_core(["targets", 0, "compiler_dependency_options_identity", "extra"], None)

    def test_compiler_binary_and_version_content_closed(self):
        for role in ["executable", "version_output"]:
            for field in ["bytes", "sha256"]:
                with self.subTest(role=role, field=field):
                    self.reject_core(["targets", 0, "compiler_dependency_options_identity", "compiler_identity", role, field], delete=True)

    def test_options_remain_ordered_strings(self):
        for field in ["options", "print_options"]:
            self.reject_core(["targets", 0, "compiler_dependency_options_identity", field], [True])
            self.reject_core(["targets", 0, "compiler_dependency_options_identity", field], "pp.explicit=true")

    def test_dependency_names_unique_and_revisions_full(self):
        dependency = copy.deepcopy(self.core()["targets"][0]["compiler_dependency_options_identity"]["dependencies"][0])
        self.reject_core(["targets", 0, "compiler_dependency_options_identity", "dependencies"], [dependency, dependency])
        self.reject_core(["targets", 0, "compiler_dependency_options_identity", "dependencies", 0, "revision"], "a" * 39)
        self.reject_core(["targets", 0, "compiler_dependency_options_identity", "dependencies", 0, "extra"], None)

    def test_runtime_identity_closed(self):
        for field in ["platform", "image_manifest", "executables"]:
            with self.subTest(field=field):
                self.reject_core(["targets", 0, "compiler_dependency_options_identity", "runtime_identity", field], delete=True)
        self.reject_core(["targets", 0, "compiler_dependency_options_identity", "runtime_identity", "executables"], [])

    def test_runtime_executable_names_unique(self):
        executables = self.core()["targets"][0]["compiler_dependency_options_identity"]["runtime_identity"]["executables"]
        self.reject_core(["targets", 0, "compiler_dependency_options_identity", "runtime_identity", "executables"], executables * 2)

    def test_scope_and_import_sets_closed(self):
        self.reject_core(["targets", 0, "scope_content_identity", "scope_files"], [])
        self.reject_core(["targets", 0, "scope_content_identity", "import_files"], delete=True)
        self.reject_core(["targets", 0, "scope_content_identity", "scope_files"], delete=True)
        self.reject_core(["targets", 0, "scope_content_identity", "extra"], None)

    def test_duplicate_source_role_paths_refuse(self):
        scope = self.core()["targets"][0]["scope_content_identity"]["scope_files"]
        self.reject_core(["targets", 0, "scope_content_identity", "scope_files"], scope * 2)

    def test_selector_entry_and_slice_closed(self):
        for array in ["accepted_formal_statement_selector", "accepted_formal_proof_selector"]:
            for field in ["link_id", "selector"]:
                with self.subTest(array=array, field=field):
                    self.reject_core(["targets", 0, array, 0, field], delete=True)
            for field in ["source", "start_byte", "end_byte", "sha256"]:
                with self.subTest(array=array, field=field):
                    self.reject_core(["targets", 0, array, 0, "selector", field], delete=True)

    def test_selector_bounds_are_exact(self):
        for field, value in [("start_byte", -1), ("start_byte", 35), ("end_byte", 25), ("end_byte", 37), ("end_byte", True)]:
            with self.subTest(field=field, value=value):
                self.reject_core(["targets", 0, "accepted_formal_proof_selector", 0, "selector", field], value)

    def test_selector_link_sets_complete_and_unique(self):
        self.reject_core(["targets", 0, "accepted_formal_statement_selector"], [])
        entry = self.core()["targets"][0]["accepted_formal_proof_selector"][0]
        self.reject_core(["targets", 0, "accepted_formal_proof_selector"], [entry, entry])
        self.reject_core(["targets", 0, "accepted_formal_proof_selector", 0, "link_id"], "foreign-link")

    def test_unresolved_names_are_exact_sorted_unique(self):
        for value in [["not_a_field"], ["type_content_sha256", "type_content_bytes"], ["type_content_bytes", "type_content_bytes"]]:
            with self.subTest(value=value):
                self.reject_core(["targets", 0, "unresolved_fields"], value)

    def test_required_type_null_pair_cannot_be_half_known(self):
        self.reject_core(["targets", 0, "type_content_sha256"], None)
        self.reject_core(["targets", 0, "type_content_bytes"], None)

    def test_required_null_needs_exact_unresolved_name(self):
        self.reject_core(["targets", 0, "scope_content_identity"], None)
        self.reject_core(["targets", 0, "compiler_dependency_options_identity"], None)

    def test_collection_refuses_unknown_enumeration(self):
        self.collection_refusal(lambda p: p["membership"].update(enumeration="unknown"), 'NATIVE_ENUMERATION_UNRESOLVED')

    def test_collection_refuses_ambiguous_enumeration(self):
        self.collection_refusal(lambda p: p["membership"].update(enumeration="ambiguous"), 'NATIVE_ENUMERATION_UNRESOLVED')

    def test_collection_refuses_missing_enumeration(self):
        self.collection_refusal(lambda p: p["membership"].pop("native_targets"), 'NATIVE_ENUMERATION_UNRESOLVED')

    def test_collection_refuses_unsupported_native_format(self):
        self.collection_refusal(lambda p: p["membership"].update(native_format="unknown/v99"), 'NATIVE_FORMAT_UNSUPPORTED')

        for example in self.profile["wire_negative_examples"]:
            with self.subTest(example=example["id"]):
                self.collection_refusal(lambda p, e=example: self.mutate_path(p, e["path"], copy.deepcopy(e["value"])), example["reason"])
        for example in self.profile["native_wire_negative_examples"]:
            with self.subTest(raw_native=example["id"]):
                self.collection_refusal(lambda p, e=example: self.mutate_native_manifest_wire(p, e["path"], e["value"]), "WIRE_PROFILE_INVALID")
        self.collection_refusal(self.mutate_raw_observation_wire, "WIRE_PROFILE_INVALID")

    def test_collection_refuses_omitted_native_target(self):
        self.collection_refusal(lambda p: p["membership"]["native_targets"].append("Demo.uncollected"), 'NATIVE_TARGET_SET_MISMATCH')

        self.collection_refusal(lambda p: p["records"].pop(), "NATIVE_TARGET_SET_MISMATCH")
        self.collection_refusal(lambda p: p["membership"]["native_targets"].pop(), "NATIVE_TARGET_SET_MISMATCH")

    def test_collection_refuses_foreign_native_target(self):
        self.collection_refusal(lambda p: p["records"][0].update(target="Foreign.t"), 'NATIVE_TARGET_SET_MISMATCH')

    def test_collection_refuses_duplicate_native_record(self):
        self.collection_refusal(lambda p: p["records"].append(copy.deepcopy(p["records"][0])), 'DUPLICATE_TARGET_IDENTITY')

    def test_collection_refuses_omitted_dependency(self):
        self.collection_refusal(lambda p: p["records"][0]["compiler_context"].update(dependency_revisions=[]), 'DEPENDENCY_SET_MISMATCH')

        self.collection_refusal(lambda p: p["membership"].update(native_dependencies=[]), "DEPENDENCY_SET_MISMATCH")

    def test_collection_refuses_omitted_scope_role(self):
        self.collection_refusal(lambda p: p["membership"]["role_inputs"].pop("scope_files"), 'SOURCE_ROLE_ENUMERATION_UNRESOLVED')

    def test_collection_refuses_omitted_import_role(self):
        self.collection_refusal(lambda p: p["membership"]["role_inputs"].pop("import_files"), 'SOURCE_ROLE_ENUMERATION_UNRESOLVED')

    def test_collection_refuses_omitted_build_role(self):
        self.collection_refusal(lambda p: p["membership"]["role_inputs"].pop("build_config"), 'SOURCE_ROLE_ENUMERATION_UNRESOLVED')

    def test_collection_refuses_omitted_producer_role(self):
        self.collection_refusal(lambda p: p["membership"]["role_inputs"].pop("producer_sources"), 'SOURCE_ROLE_ENUMERATION_UNRESOLVED')

    def test_collection_refuses_unknown_selector_acceptance(self):
        self.collection_refusal(lambda p: p["acceptances"].pop("links"), 'SELECTOR_ACCEPTANCE_UNRESOLVED')

    def test_collection_refuses_foreign_selector_link(self):
        self.collection_refusal(lambda p: p["acceptances"]["links"][0].update(target="Foreign.t"), 'SELECTOR_LINK_SET_MISMATCH')

    def test_collection_refuses_origin_relabelled_snapshot(self):
        self.collection_refusal(lambda p: next(row for row in p["membership"]["sources"] if row["role"] == "informal_origin").update(side="candidate"), 'SOURCE_ROLE_MISMATCH')

    def test_collection_refuses_snapshot_source_wrong_commit(self):
        self.collection_refusal(lambda p: p["records"][0]["module_ref"].update(commit="9" * 40), 'SOURCE_MEMBERSHIP_MISMATCH')

        self.collection_refusal(lambda p: p["membership"]["role_inputs"]["scope_files"][0].update(commit=p["request"]["base"]["commit"]), "SOURCE_MEMBERSHIP_MISMATCH")
        def crossed_snapshot(p):
            p["membership"]["snapshot"] = copy.deepcopy(p["request"]["base"])
        self.collection_refusal(crossed_snapshot, "SOURCE_MEMBERSHIP_MISMATCH")

    def test_collection_refuses_raw_source_digest_mismatch(self):
        self.collection_refusal(lambda p: p["membership"]["sources"][0].update(raw_base64="Zm9yZ2Vk"), 'SOURCE_BYTES_MISMATCH')

        self.assert_collection_baseline(self.collection())
        def two_failures(p):
            p["membership"]["sources"][0]["raw_base64"] = "Zm9yZ2Vk"
            p["acceptances"]["links"][0]["formal_proof_selector"]["sha256"] = "0" * 64
        result = self.collection(two_failures)
        self.assertIsNone(result["formal_inventory_sha256"])
        self.assertEqual(result["issues"], sorted(set(result["issues"])))
        for reason in ["SOURCE_BYTES_MISMATCH", "SELECTOR_BYTES_MISMATCH", "SYNTHETIC_AUTHORITY_UNQUALIFIED", "TYPE_RECORD_UNAVAILABLE"]:
            self.assertIn(reason, result["issues"])

    def test_collection_unknown_enumeration_has_no_core(self):
        result = self.collection(lambda p: p["membership"].update(enumeration="unknown"))
        self.assertIsNone(result["core"])
        self.assertIsNone(result["formal_inventory_sha256"])
        self.assertIsNone(result["diagnostic_inventory_sha256"])
        self.assertIn("NATIVE_ENUMERATION_UNRESOLVED", result["issues"])


    def test_nested_compiler_runtime_and_selector_extra_fields_refuse(self):
        for path in [["compiler_dependency_options_identity", "compiler_identity"], ["compiler_dependency_options_identity", "compiler_identity", "executable"], ["compiler_dependency_options_identity", "compiler_identity", "version_output"], ["compiler_dependency_options_identity", "runtime_identity"], ["compiler_dependency_options_identity", "runtime_identity", "executables", 0], ["compiler_dependency_options_identity", "runtime_identity", "executables", 0, "content"], ["accepted_formal_statement_selector", 0], ["accepted_formal_statement_selector", 0, "selector"], ["accepted_formal_proof_selector", 0, "selector"]]:
            with self.subTest(path=path):
                self.reject_core(["targets", 0] + path + ["extra"], "unadmitted")


    def test_source_role_lists_have_unique_canonical_identity(self):
        source = self.core()["targets"][0]["module_content_identity"]
        for path in [["compiler_dependency_options_identity", "build_config"], ["compiler_dependency_options_identity", "producer_sources"], ["scope_content_identity", "import_files"]]:
            with self.subTest(path=path):
                self.reject_core(["targets", 0] + path, [source, source])


    def test_repeated_source_roles_cannot_disagree(self):
        source = copy.deepcopy(self.core()["targets"][0]["module_content_identity"])
        source["sha256"] = "0" * 64
        self.reject_core(["targets", 0, "scope_content_identity", "import_files"], [source])


    def test_import_set_excludes_target_module(self):
        source = copy.deepcopy(self.core()["targets"][0]["module_content_identity"])
        self.reject_core(["targets", 0, "scope_content_identity", "import_files"], [source])


    def test_semantic_options_order_changes_digest(self):
        core = self.core()
        core["targets"][0]['compiler_dependency_options_identity']['options'] = ['first=1', 'second=2']
        self.assertEqual(inventory_sha256(core), '219b9a536edc8447353206a5b4ccece855746c741e2f65dfecf070718854e4e1')
        core["targets"][0]['compiler_dependency_options_identity']['options'] = ['second=2', 'first=1']
        self.assertEqual(inventory_sha256(core), '94a083534fc885a6d94ca22880a5fd760c5e28533ef6e864d13e7bb445f6c88d')

        original = self.collection()
        self.assert_collection_baseline(original)
        changed = self.collection(lambda p: self.reverse_effective_options(p, "options"))
        self.assertEqual(changed["core"]["targets"][0]["compiler_dependency_options_identity"]["options"], list(reversed(self.profile["membership"]["producer_observation"]["options"])))
        self.assertNotEqual(changed["diagnostic_inventory_sha256"], original["diagnostic_inventory_sha256"])
        self.assertEqual(changed["diagnostic_inventory_sha256"], self.profile["projection_order_oracles"]["options"]["sha256"])
        self.assertEqual(wire(changed["core"]), self.profile["projection_order_oracles"]["options"]["canonical_ascii_preimage"].encode("ascii"))
        self.assertIsNone(changed["formal_inventory_sha256"])


    def test_print_options_order_changes_digest(self):
        core = self.core()
        core["targets"][0]['compiler_dependency_options_identity']['print_options'] = ['pp.explicit=true', 'pp.universes=true']
        self.assertEqual(inventory_sha256(core), '949c0173ab4ac0fef66a8f04de8c2323cafa4453d8dbbfd7ffd512119ddda5c1')
        core["targets"][0]['compiler_dependency_options_identity']['print_options'] = ['pp.universes=true', 'pp.explicit=true']
        self.assertEqual(inventory_sha256(core), '952e41e8e233ea47b2d4414346ecac97cac1fa8a48485ccef8e7ae7966278249')

        original = self.collection()
        self.assert_collection_baseline(original)
        changed = self.collection(lambda p: self.reverse_effective_options(p, "print_options"))
        self.assertEqual(changed["core"]["targets"][0]["compiler_dependency_options_identity"]["print_options"], list(reversed(self.profile["membership"]["producer_observation"]["print_options"])))
        self.assertNotEqual(changed["diagnostic_inventory_sha256"], original["diagnostic_inventory_sha256"])
        self.assertEqual(changed["diagnostic_inventory_sha256"], self.profile["projection_order_oracles"]["print_options"]["sha256"])
        self.assertEqual(wire(changed["core"]), self.profile["projection_order_oracles"]["print_options"]["canonical_ascii_preimage"].encode("ascii"))
        self.assertIsNone(changed["formal_inventory_sha256"])


    def test_origin_commit_remains_in_content_digest(self):
        core = self.core()
        core["targets"][0]["compiler_dependency_options_identity"]["producer_sources"] = [{'kind': 'origin', 'ref': {'repository': 'fixture/control', 'commit': '7777777777777777777777777777777777777777', 'path': 'control/producer.py', 'git_blob': '6666666666666666666666666666666666666666', 'sha256': '7777777777777777777777777777777777777777777777777777777777777777', 'bytes': 1}}]
        self.assertEqual(inventory_sha256(core), 'ec5c0464eb3b3ac437c8309d5a9b182530eae67547689a5c05579281268d136d')
        first_core = copy.deepcopy(core)
        core["targets"][0]["compiler_dependency_options_identity"]["producer_sources"][0]["ref"]["commit"] = "8" * 40
        self.assertNotEqual(first_core, core)
        self.assertNotEqual(wire(first_core), wire(core))
        self.assertEqual(first_core["targets"][0]["compiler_dependency_options_identity"]["producer_sources"][0]["ref"]["commit"], "7" * 40)
        self.assertEqual(core["targets"][0]["compiler_dependency_options_identity"]["producer_sources"][0]["ref"]["commit"], "8" * 40)
        self.assertEqual(inventory_sha256(core), 'c0df40e646762921112000930d9dae597ef08bd8e03f361dd9f0a6de5b34fe99')


    def test_null_and_empty_selectors_are_distinct(self):
        known_absent = self.core()
        known_absent["targets"][0]["accepted_formal_proof_selector"][0]["selector"] = None
        self.assertEqual(inventory_sha256(self.core()), BASELINE)
        self.assertNotEqual(inventory_sha256(known_absent), BASELINE)
        self.reject_core(["targets", 0, "accepted_formal_proof_selector"], [])


    def test_conflicting_duplicate_target_identity_refuses(self):
        core = self.core()
        other = copy.deepcopy(core["targets"][0])
        other["type_content_sha256"] = "0" * 64
        core["targets"].append(other)
        with self.assertRaises(ValueError):
            inventory_sha256(core)


    def test_native_identity_cannot_be_unresolved(self):
        for field, value in [("format", None), ("native_repository", ""), ("target", ""), ("package", None)]:
            with self.subTest(field=field):
                self.reject_core(["targets", 0, field], value)


    def test_runtime_executable_is_complete_nonempty_named_content(self):
        for path, value in [(["executables", 0, "name"], ""), (["executables", 0, "content"], None), (["platform"], "")]:
            with self.subTest(path=path):
                self.reject_core(["targets", 0, "compiler_dependency_options_identity", "runtime_identity"] + path, value)


    def test_scope_and_import_role_values_are_arrays(self):
        for field in ["scope_files", "import_files"]:
            for value in [None, {}, True]:
                with self.subTest(field=field, value=value):
                    self.reject_core(["targets", 0, "scope_content_identity", field], value)


    def test_policy_refuses_extra_field(self):
        policy = self.policy()
        policy["unknown"] = None
        raw = wire(policy)
        with self.assertRaises(ValueError):
            decode_policy(raw, hashlib.sha256(raw).hexdigest())


    def test_request_commit_and_tree_are_full_lowercase_hashes(self):
        for field in ["base", "head", "checked_merge", "control"]:
            for key in ["commit", "tree"]:
                for value in ["A" * 40, "1" * 39, True]:
                    context = self.request()
                    context[field][key] = value
                    with self.subTest(field=field, key=key, value=value), self.assertRaises(ValueError):
                        decode_request(wire(context))


    def test_collection_refuses_selector_raw_slice_mismatch(self):
        self.collection_refusal(lambda p: p["acceptances"]["links"][0]["formal_proof_selector"].update(sha256="0" * 64), "SELECTOR_BYTES_MISMATCH")


    def test_collection_nonimage_null_does_not_admit_image_runtime(self):
        self.collection_refusal(lambda p: p["membership"].update(runtime_kind="image"), "RUNTIME_IDENTITY_UNRESOLVED")


    def test_collection_zero_links_is_known_absence_diagnostic(self):
        result = self.collection(lambda p: p["acceptances"].update(links=[]))
        self.assertIsNone(result["formal_inventory_sha256"])
        self.assertIsNotNone(result["diagnostic_inventory_sha256"])
        self.assertEqual(result["core"]["targets"][0]["accepted_formal_statement_selector"], [])
        self.assertEqual(result["core"]["targets"][0]["accepted_formal_proof_selector"], [])

        self.assert_collection_baseline(self.collection())
        self.assertEqual([target["target"] for target in result["core"]["targets"]], ["Demo.t", "Demo.unlinked"])
        for target in result["core"]["targets"]:
            self.assertEqual(target["accepted_formal_statement_selector"], [])
            self.assertEqual(target["accepted_formal_proof_selector"], [])
            self.assertEqual(target["unresolved_fields"], ["type_content_bytes", "type_content_sha256"])


    def test_dependency_and_runtime_arrays_require_canonical_order(self):
        deps = [{"name": "z", "revision": "a" * 40}, {"name": "a", "revision": "b" * 40}]
        self.reject_core(["targets", 0, "compiler_dependency_options_identity", "dependencies"], deps)
        exe = self.core()["targets"][0]["compiler_dependency_options_identity"]["runtime_identity"]["executables"][0]
        other = copy.deepcopy(exe)
        other["name"] = "a-compiler"
        self.reject_core(["targets", 0, "compiler_dependency_options_identity", "runtime_identity", "executables"], [exe, other])

        self.assert_collection_baseline(self.collection())
        projected = self.collection(self.permute_unordered_inputs)
        self.assert_collection_baseline(projected)


    def test_source_role_arrays_require_canonical_order(self):
        one = copy.deepcopy(self.core()["targets"][0]["module_content_identity"])
        two = copy.deepcopy(one)
        one["path"] = "z/input"
        two["path"] = "a/input"
        self.reject_core(["targets", 0, "compiler_dependency_options_identity", "build_config"], [one, two])

        projected = self.collection(self.permute_unordered_inputs)
        self.assertEqual(projected["core"], self.profile["expected_diagnostic_core"])
        self.assertEqual(projected["diagnostic_inventory_sha256"], self.profile["expected_diagnostic_sha256"])


    def test_target_array_requires_canonical_identity_order(self):
        core = self.core()
        other = copy.deepcopy(core["targets"][0])
        other["target"] = "A.before"
        core["targets"].append(other)
        with self.assertRaises(ValueError):
            inventory_sha256(core)

        projected = self.collection(self.permute_unordered_inputs)
        self.assertEqual([target["target"] for target in projected["core"]["targets"]], ["Demo.t", "Demo.unlinked"])


if __name__ == "__main__":
    unittest.main()
