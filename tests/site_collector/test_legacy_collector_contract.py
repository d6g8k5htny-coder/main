"""Prospective fixture-only tests. Authored as Source; no local execution claim.

The two target function bodies are unchanged historical excerpts. adapter_core
was not supplied: LegacyRefusal, record, require and ORIGIN below are an explicit
synthetic helper contract, not a claim about the complete historical adapter.
Only the frozen scanner's transform/without_release are called. No browser,
network, Git, inventory, asset graph, CLI, or real observation is exercised.
"""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import unittest

SOURCE_COMMIT = "ef25e477faad3eba8b75aea1247e7d68c6059df8"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PINS = {
    "legacy-cut-source.py.txt": (5935, "99fa3bffd90a94d7fb69c522f4246ed645c9f20db56fe4f139106fd77313c28c"),
    "verify_maintained_html.py.txt": (479, "1302c8235a787b7198972a351102bf2f88b709967e4b39272cc46e2978fa57ec"),
    "compare_material_assets.py.txt": (1116, "228bf3ad75c29adc602bf1f22621b44c27dae0193db541c30c5d027e5aff8523"),
    "site_asset_release.py": (26305, "390ea41ad0b4426913757c186e341dfcc1d1abcdafce505f51607a3dd31c0ffd"),
    "current-case-membership-static.json": (5955, "9bea71d83dedf55c977966ba43a6ed5d36690a7ceef2c504e7bb1cd724d53da6"),
    "cases.json": (3478, "78680312079114ad0e1709ade2c1f99dc069c05b6ada9a2b4758c175a14be4e0"),
}
MAINTAINED_PAGES = (
    "cite.html", "dependencies.html", "explore.html", "formal.html",
    "index.html", "measure.html", "museum.html", "reproduce.html",
    "research.html", "workspace.html",
)
LEGACY_PAGES = ("index.html", "research.html", "workspace.html", "dependencies.html")
RESUME_NODES = {
    "math.d5-component.punctured-pin-proof", "math.lifetime-remainder",
    "math.uniform-matrix-cap-lifetime", "hist.CH-LIFT",
}
EXPECTED_MARKERS = {
    "test_wrong_imported_js_key_is_refused": "LEGACY_COLLECTOR_ACCEPTED_WRONG_IMPORTED_JS_KEY",
    "test_wrong_imported_css_key_is_refused": "LEGACY_COLLECTOR_ACCEPTED_WRONG_IMPORTED_CSS_KEY",
    "test_explore_wrong_digest_is_refused": "LEGACY_COLLECTOR_ACCEPTED_WRONG_EXPLORE_DIGEST",
    "test_formal_wrong_size_is_refused": "LEGACY_COLLECTOR_ACCEPTED_WRONG_FORMAL_SIZE",
    "test_all_ten_maintained_pages_are_checked": "LEGACY_COLLECTOR_CHECKED_FOUR_OF_TEN_PAGES",
}


class LegacyRefusal(ValueError):
    """Declared synthetic require() refusal; not inferred from adapter_core."""


def _fixture_bytes(name):
    directory = FIXTURES.lstat()
    if not stat.S_ISDIR(directory.st_mode) or stat.S_ISLNK(directory.st_mode):
        raise RuntimeError("Fixture directory must be an ordinary directory")
    path = FIXTURES / name
    before = path.lstat()
    expected_size, expected_sha = PINS[name]
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != expected_size:
        raise RuntimeError("Fixture geometry mismatch: " + name)
    raw = path.read_bytes()
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise RuntimeError("Fixture identity mismatch: " + name)
    return raw


def _record(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _require(condition, message):
    if not condition:
        raise LegacyRefusal(message)


def _function_bytes(full_source, name):
    tree = ast.parse(full_source.decode("utf-8"), filename="frozen-legacy-source")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise RuntimeError("Historical function not unique: " + name)
    return (ast.get_source_segment(full_source.decode("utf-8"), functions[0]) + "\n").encode("utf-8")


def _load_fixture_contract():
    # This function is for a separately authorized future hosted test only.
    raw = {name: _fixture_bytes(name) for name in PINS}
    cases = json.loads(raw["cases.json"])
    if cases.get("fixture_only") is not True or cases.get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("Synthetic fixture/source qualification missing")
    namespace = {"Path": Path, "re": re, "record": _record, "require": _require,
                 "ORIGIN": "https://fixture.invalid/site/"}
    for name in ("verify_maintained_html", "compare_material_assets"):
        excerpt = raw[name + ".py.txt"]
        if _function_bytes(raw["legacy-cut-source.py.txt"], name) != excerpt:
            raise RuntimeError("Historical excerpt differs from full source: " + name)
        tree = ast.parse(excerpt.decode("utf-8"), filename=name + ".py.txt")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != name:
            raise RuntimeError("Unexpected executable excerpt shape: " + name)
        exec(compile(tree, name + ".py.txt", "exec"), namespace)
    scanner_path = FIXTURES / "site_asset_release.py"
    spec = importlib.util.spec_from_file_location("frozen_fixture_site_asset_release", scanner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load frozen scanner fixture")
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)

    def normalize(text, suffix):
        return scanner.transform(text, suffix, scanner.without_release)

    return raw, cases, namespace, normalize


class FixtureContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.cases, cls.target, normalize = _load_fixture_contract()
        cls.normalize = staticmethod(normalize)

    def compare_asset(self, kind, variant):
        asset = self.cases["assets"][kind]
        return self.target["compare_material_assets"](
            {asset["path"]: asset[variant].encode("utf-8")},
            {asset["path"]: asset["source_utf8"].encode("utf-8")}, self.normalize,
        )

    def html_bodies(self, changes=None):
        payload = self.cases["payload_utf8"].encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        bodies = {}
        for name in MAINTAINED_PAGES:
            item_digest, item_size = (changes or {}).get(name, (digest, len(payload)))
            bodies[name] = (
                '<!doctype html><html data-current-digest="' + item_digest +
                '" data-current-bytes="' + str(item_size) + '"><title>Synthetic ' +
                name + '</title></html>\n'
            ).encode("utf-8")
        return payload, bodies


class BaselineSetupControls(FixtureContract):
    def test_all_six_fixture_pins(self):
        self.assertEqual(set(self.raw), set(PINS))
        for name, (size, digest) in PINS.items():
            self.assertEqual(len(self.raw[name]), size)
            self.assertEqual(hashlib.sha256(self.raw[name]).hexdigest(), digest)

    def test_unchanged_function_extractions(self):
        for name in ("verify_maintained_html", "compare_material_assets"):
            self.assertEqual(_function_bytes(self.raw["legacy-cut-source.py.txt"], name),
                             self.raw[name + ".py.txt"])

    def test_static_35_node_aware_membership(self):
        membership = json.loads(self.raw["current-case-membership-static.json"])
        fields = ["case", "width", "color_scheme", "node"]
        self.assertEqual(membership["source_commit"], SOURCE_COMMIT)
        self.assertEqual(membership["identity_fields"], fields)
        self.assertEqual(membership["count"], 35)
        self.assertEqual(len(membership["cases"]), 35)
        identities = [tuple(item["identity"].get(key) for key in fields) for item in membership["cases"]]
        self.assertEqual(len(set(identities)), 35)
        resume = [item["identity"] for item in membership["cases"]
                  if item["identity"].get("case") == "resume-node-interactive-static-parity"]
        self.assertEqual(len(resume), 4)
        self.assertEqual({item["node"] for item in resume}, RESUME_NODES)
        self.assertEqual(membership["old_frozen_count"], 31)  # historical metadata only

    def test_normalizer_recognizes_imports_and_preserves_non_url_literal(self):
        for asset in self.cases["assets"].values():
            suffix = Path(asset["path"]).suffix
            correct = asset["served_correct_utf8"]
            wrong = asset["served_wrong_key_utf8"]
            self.assertEqual(correct.replace(asset["expected_leaf_key"], asset["wrong_leaf_key"]), wrong)
            self.assertNotEqual(correct, wrong)
            self.assertEqual(self.normalize(correct, suffix), asset["source_utf8"])
            self.assertEqual(self.normalize(wrong, suffix), asset["source_utf8"])
        literal = 'const note = "site-release=' + "a" * 64 + '";\n'
        self.assertEqual(self.normalize(literal, ".mjs"), literal)

    def test_correct_imported_js_is_accepted(self):
        self.assertEqual(len(self.compare_asset("js", "served_correct_utf8")), 1)

    def test_correct_imported_css_is_accepted(self):
        self.assertEqual(len(self.compare_asset("css", "served_correct_utf8")), 1)

    def test_changed_material_bodies_are_refused(self):
        for kind, asset in self.cases["assets"].items():
            self.assertNotEqual(self.normalize(asset["served_changed_body_utf8"], Path(asset["path"]).suffix),
                                asset["source_utf8"])
            with self.assertRaises(LegacyRefusal):
                self.compare_asset(kind, "served_changed_body_utf8")

    def test_valid_historical_four_page_control(self):
        self.assertEqual(tuple(self.cases["maintained_pages"]), MAINTAINED_PAGES)
        self.assertEqual(tuple(self.cases["legacy_pages"]), LEGACY_PAGES)
        payload, bodies = self.html_bodies()
        historical = {name: bodies[name] for name in LEGACY_PAGES}
        self.assertEqual(self.target["verify_maintained_html"](payload, historical), 4)

    def test_wrong_historical_index_digest_is_refused(self):
        payload = self.cases["payload_utf8"].encode("utf-8")
        _, bodies = self.html_bodies({"index.html": ("0" * 64, len(payload))})
        with self.assertRaises(LegacyRefusal):
            self.target["verify_maintained_html"](payload, bodies)

    def test_independent_literal_leaf_key_oracles(self):
        expected_names = {"js": "site/dep.mjs", "css": "site/font.woff2"}
        for kind, asset in self.cases["assets"].items():
            self.assertEqual(asset["leaf_key_name"], expected_names[kind])
            name = asset["leaf_key_name"].encode("utf-8")
            body = asset["leaf_source_utf8"].encode("utf-8")
            # Independent explicit leaf byte formula, not scanner.asset_keys().
            material = (b"public-site-asset-key-v1\0" + len(name).to_bytes(8, "big") + name +
                        len(body).to_bytes(8, "big") + body)
            self.assertEqual(hashlib.sha256(material).hexdigest(), asset["expected_leaf_key"])
            self.assertRegex(asset["wrong_leaf_key"], r"^[0-9a-f]{64}$")
            self.assertNotEqual(asset["wrong_leaf_key"], asset["expected_leaf_key"])


class CollectorRegressionTests(FixtureContract):
    def test_wrong_imported_js_key_is_refused(self):
        try:
            self.compare_asset("js", "served_wrong_key_utf8")
        except LegacyRefusal:
            return
        self.fail(EXPECTED_MARKERS[self._testMethodName])

    def test_wrong_imported_css_key_is_refused(self):
        try:
            self.compare_asset("css", "served_wrong_key_utf8")
        except LegacyRefusal:
            return
        self.fail(EXPECTED_MARKERS[self._testMethodName])

    def test_explore_wrong_digest_is_refused(self):
        payload = self.cases["payload_utf8"].encode("utf-8")
        _, bodies = self.html_bodies({"explore.html": ("0" * 64, len(payload))})
        try:
            self.target["verify_maintained_html"](payload, bodies)
        except LegacyRefusal:
            return
        self.fail(EXPECTED_MARKERS[self._testMethodName])

    def test_formal_wrong_size_is_refused(self):
        payload = self.cases["payload_utf8"].encode("utf-8")
        _, bodies = self.html_bodies({"formal.html": (hashlib.sha256(payload).hexdigest(), len(payload) + 1)})
        try:
            self.target["verify_maintained_html"](payload, bodies)
        except LegacyRefusal:
            return
        self.fail(EXPECTED_MARKERS[self._testMethodName])

    def test_all_ten_maintained_pages_are_checked(self):
        payload, bodies = self.html_bodies()
        count = self.target["verify_maintained_html"](payload, bodies)
        if count == 10:
            return
        if count != 4:
            raise RuntimeError("Unexpected historical census: " + repr(count))
        self.fail(EXPECTED_MARKERS[self._testMethodName])
