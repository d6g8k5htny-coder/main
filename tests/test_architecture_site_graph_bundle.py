"""External controls for the native-archive SSG bundle and real builder call.

All archives/logs made here are SYNTHETIC_ONLY. They exercise real admission;
they authenticate no producer and assert no scientific or hosted result.
No new production module is imported before each method's missing-feature
assertion. Project execution belongs exclusively to hosted ephemeral runners.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
from html.parser import HTMLParser
from importlib import import_module, util
import itertools
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import unittest
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "architecture/site_graph_bundle.py"
BUILD_REPOSITORY = "synthetic-build-owner/synthetic-build"
BUILD_COMMIT = "d" * 40
CURRENT_BYTES = b'{"synthetic_current_context_only":true}\n'
MIB = 1024 * 1024
MISSING = "native graph bundle implementation is missing"
GRAPH_SOURCE = {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "7858329974e28be79f29b22644370084ff43da4f",
    "captured_at": "2026-10-03T15:07:03Z",
    "graph_sha256": "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09",
    "gate_sha256": "a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8",
}
NATIVE_ARGUMENTS = (
    "native_graph_archive", "native_graph_expected", "native_graph_source",
    "build_repository", "build_commit",
)
_MODULES = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def fixture_module(name):
    """Load the existing reviewed fixture source only during a guarded test."""
    if name not in _MODULES:
        path = ROOT / "tests" / (name + ".py")
        spec = util.spec_from_file_location("_bundle_fixture_" + name, path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return _MODULES[name]


def sample(**kwargs):
    return fixture_module("test_site_graph_projection").sample(**kwargs)


def api():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return import_module("architecture.site_graph_bundle").build_native_graph_bundle


def project(value):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    projector = import_module("architecture.site_graph_projection").project_site_graph
    return projector(value["archive"], value["pins"], value["source"])


def bundle(value, *, current=CURRENT_BYTES, repository=BUILD_REPOSITORY,
           commit=BUILD_COMMIT):
    return api()(value["archive"], value["pins"], value["source"],
                 build_repository=repository, build_commit=commit,
                 current_payload_bytes=current)


def native_paths(value):
    prefix = "native-graph/" + value["pins"]["artifact_sha256"] + "/"
    return (prefix + "graph.json", prefix + "architecture-run-receipt.json",
            prefix + "projection.json")


def expected_paths(value):
    return {*native_paths(value), "proof-tree.html", "current-build-manifest.json"}


def descriptor(path, raw):
    return {"path": path, "bytes": len(raw), "sha256": sha(raw)}


def snapshot(site):
    return {p.relative_to(site).as_posix(): p.read_bytes()
            for p in site.rglob("*") if p.is_file()}


def site_state(site):
    """Include empty directories and links when proving refusal before writes."""
    result = {}
    for path in site.rglob("*"):
        name = path.relative_to(site).as_posix()
        if path.is_symlink():
            result[name] = ("symlink", str(path.readlink()))
        elif path.is_dir():
            result[name] = ("directory",)
        elif path.is_file():
            result[name] = ("file", path.read_bytes())
        else:
            result[name] = ("other",)
    return result


def prepare_site(root):
    """Real temporary site and existing legacy payload fixture, without mocks."""
    legacy = fixture_module("test_current_site_build")
    site = root / "site"
    site.mkdir()
    for name in ("index.html", "research.html", "dependencies.html"):
        (site / name).write_bytes(
            b'<!doctype html><html lang="en"><head><meta charset="utf-8"></head>'
            b'<body><main><p>Historical evidence 2026-09-25.</p>'
            b'<!-- CURRENT-SITE:START -->old panel<!-- CURRENT-SITE:END -->'
            b'<pre>Exact history.\r\n</pre></main></body></html>\n')
    for name in ("brand.css", "style.css", "current.css"):
        (site / name).write_bytes(b"body { color: navy; }\n")
    (site / "historic.json").write_bytes(b'{"exact":"17/31"}\r\n')
    payload = root / "payload.json"
    payload.write_bytes((json.dumps(legacy.fixture(), ensure_ascii=False,
                                   indent=2) + "\n").encode("utf-8"))
    return site, payload


def builder():
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    return import_module("current_site_build").build


def build_arguments(value):
    return {
        "native_graph_archive": value["archive"],
        "native_graph_expected": copy.deepcopy(value["pins"]),
        "native_graph_source": value["source"],
        "build_repository": BUILD_REPOSITORY,
        "build_commit": BUILD_COMMIT,
    }


class Element:
    def __init__(self, tag, attrs=None, parent=None):
        self.tag, self.attrs, self.parent = tag, dict(attrs or ()), parent
        self.children, self.data = [], []

    def text(self):
        return " ".join([*self.data, *(child.text() for child in self.children)])

    def descendants(self):
        yield self
        for child in self.children:
            yield from child.descendants()

    def links(self):
        return [(e.attrs.get("href", e.attrs.get("xlink:href", "")), e.text().strip())
                for e in self.descendants() if e.tag == "a"]


class Page(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, raw):
        super().__init__(convert_charrefs=True)
        self.root = Element("document")
        self.stack = [self.root]
        self.declarations = []
        self.feed(raw.decode("utf-8"))
        self.close()

    def handle_decl(self, declaration):
        self.declarations.append(declaration)

    def handle_starttag(self, tag, attrs):
        item = Element(tag, attrs, self.stack[-1])
        self.stack[-1].children.append(item)
        if tag not in self.VOID:
            self.stack.append(item)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Element(tag, attrs, self.stack[-1]))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].data.append(data)

    def elements(self):
        return list(self.root.descendants())


def structurally_visible(element):
    """Check explicit hiding in parsed source; no browser visibility claim."""
    while element is not None:
        if ("hidden" in element.attrs or element.attrs.get("aria-hidden", "").lower() == "true"
                or re.search(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)",
                             element.attrs.get("style", ""), re.IGNORECASE)):
            return False
        element = element.parent
    return True


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate rendered node-record key")
        result[key] = value
    return result


class NativeGraphBundleContract(unittest.TestCase):
    def positive(self, value=None, **kwargs):
        value = sample() if value is None else value
        outputs = bundle(value, **kwargs)
        self.assertIs(type(outputs), dict)
        self.assertEqual(set(outputs), expected_paths(value))
        self.assertTrue(all(type(raw) is bytes for raw in outputs.values()))
        graph, receipt, projection = native_paths(value)
        self.assertEqual(outputs[graph], value["members"]["generated/graph.json"])
        self.assertEqual(outputs[receipt], value["members"]["architecture-run-receipt.json"])
        self.assertEqual(outputs[projection], project(value).projection_bytes)
        return value, outputs

    def test_admitted_positive_preserves_three_exact_buffers_and_five_paths(self):
        """Break: reserialize original members, omit output, or weaken 89-ID floor."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, outputs = self.positive()
        fixtures = fixture_module("test_site_graph_projection")
        ids = fixtures.CONTROL_IDS
        self.assertEqual(len(ids), 89)
        self.assertEqual(sha(("\n".join(sorted(ids)) + "\n").encode()),
                         "b51eaf6cfaede42105ad377c066c4b18789e320ccf542b6817a6850c350fc9ef")
        self.assertEqual(len(json.loads(outputs[native_paths(value)[0]])["nodes"]), 49)
        self.assertEqual(len(json.loads(outputs[native_paths(value)[0]])["edges"]), 55)

    def test_manifest_is_closed_canonical_and_binds_exact_four_files_without_cycle(self):
        """Break: omit/forge descriptor, alter native/build context, or hash itself."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, outputs = self.positive()
        manifest = json.loads(outputs["current-build-manifest.json"])
        self.assertEqual(manifest, {
            "schema_version": 1, "scientific_effect": "NONE",
            "scientific_status_authority": False, "custody": "unknown",
            "verification_scope": "declared-build-context-and-admitted-native-archive",
            "build": {"repository": BUILD_REPOSITORY, "commit": BUILD_COMMIT,
                      "current_payload": descriptor("current.json", CURRENT_BYTES)},
            "native": {"binding": value["pins"], "graph_source": GRAPH_SOURCE},
            "files": [descriptor(path, outputs[path]) for path in sorted(outputs)
                      if path != "current-build-manifest.json"],
        })
        self.assertIs(type(manifest["schema_version"]), int)
        self.assertIs(manifest["scientific_status_authority"], False)
        self.assertEqual(outputs["current-build-manifest.json"], canonical(manifest))
        self.assertIn("current-build-manifest.json", dict(Page(outputs["proof-tree.html"]).root.links()))

    def test_repeated_bundle_is_deterministic_and_does_not_mutate_inputs(self):
        """Break: wall clock/random output, mutate expected pins, or retain mutable state."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value = sample()
        before = copy.deepcopy(value)
        _, first = self.positive(value)
        self.assertEqual(first, bundle(value))
        self.assertEqual(value, before)
        value["pins"]["artifact_id"] = "99999"
        self.assertEqual(json.loads(first["current-build-manifest.json"])["native"]["binding"],
                         before["pins"])

    def test_page_preserves_distinct_build_producer_math_contexts_and_evidence_qualifiers(self):
        """Break: use build source as Math source or promote absent formal dimensions."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, outputs = self.positive()
        page = Page(outputs["proof-tree.html"])
        text = page.root.text()
        for literal in (BUILD_REPOSITORY, BUILD_COMMIT, value["pins"]["repository"],
                        value["pins"]["checked_commit"], value["pins"]["run_id"],
                        value["pins"]["run_attempt"], value["pins"]["artifact_id"],
                        value["pins"]["artifact_sha256"], value["pins"]["receipt_sha256"],
                        sha(value["members"]["generated/graph.json"]),
                        GRAPH_SOURCE["repository"], GRAPH_SOURCE["commit"],
                        GRAPH_SOURCE["graph_sha256"], "NONE", "unknown", "not_recorded"):
            self.assertIn(literal, text)
        self.assertNotEqual(BUILD_COMMIT, value["pins"]["checked_commit"])
        self.assertNotEqual(BUILD_COMMIT, GRAPH_SOURCE["commit"])
        self.assertTrue(any("doctype html" == d.lower() for d in page.declarations))
        self.assertTrue(any(e.tag == "html" for e in page.elements()))
        links = {href for href, label in page.root.links()}
        self.assertIn("https://github.com/" + GRAPH_SOURCE["repository"] + "/tree/"
                      + GRAPH_SOURCE["commit"], links)
        for path in native_paths(value):
            self.assertIn(path, links)
        self.assertFalse(any(e.tag in {"script", "iframe", "object", "embed", "base", "form"}
                             for e in page.elements()))
        for item in page.elements():
            self.assertFalse(any(name.lower().startswith("on") for name in item.attrs))
            for name in ("href", "xlink:href", "src"):
                if name in item.attrs:
                    url = urlsplit(item.attrs[name])
                    self.assertNotIn(url.scheme.lower(), {"javascript", "data", "file"})
                    if item.tag != "a":
                        self.assertFalse(url.scheme or url.netloc,
                                         "native page must load no external resource")

    def test_svg_keyboard_links_resolve_to_all_49_distinct_node_cards_with_five_axes(self):
        """Break: omit/misattribute node records or axes, reverse edge roles, or lose edge requirement state."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, outputs = self.positive()
        graph = json.loads(value["members"]["generated/graph.json"])
        page = Page(outputs["proof-tree.html"])
        elements = page.elements()
        ids = [e.attrs["id"] for e in elements if "id" in e.attrs]
        self.assertEqual(len(ids), len(set(ids)))
        targets = {e.attrs["id"]: e for e in elements if "id" in e.attrs}
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", name)
                            for name in targets), "safe deterministic fragment identifiers are required")
        svgs = [e for e in elements if e.tag == "svg"]
        self.assertTrue(svgs, "an accessible source-bound SVG overview is required")
        overview = [item for svg in svgs for item in svg.descendants() if item.tag == "a"]
        node_links = {}
        for identity in graph["nodes"]:
            matches = []
            for anchor in overview:
                href = anchor.attrs.get("href", anchor.attrs.get("xlink:href", ""))
                labels = [anchor.text().strip(), anchor.attrs.get("aria-label", "")]
                labels.extend(child.text().strip() for child in anchor.descendants()
                              if child.tag == "title")
                if identity in labels and href.startswith("#"):
                    matches.append(href)
            self.assertTrue(matches, "missing keyboard overview link for " + identity)
            self.assertTrue(all(href[1:] in targets for href in matches))
            node_links[identity] = matches[0]
            target = targets[matches[0][1:]]
            self.assertEqual(target.attrs.get("data-node-id"), identity)
            self.assertTrue(structurally_visible(target))
            card_text = target.text()
            self.assertIn(identity, card_text)
            records = [item for item in target.descendants()
                       if item.tag == "pre" and "data-node-record" in item.attrs]
            self.assertEqual(len(records), 1, identity + ": one attributed original node record")
            self.assertTrue(structurally_visible(records[0]))
            record = json.loads(records[0].text(), object_pairs_hook=unique_json_object)
            self.assertEqual(canonical(record), canonical(graph["nodes"][identity]),
                             identity + ": original typed node metadata was omitted or misattributed")
            recorded_metadata = {key: field for key, field in graph["nodes"][identity].items()
                                 if key == "source" or key.startswith("review_")}
            actual_metadata = {key: field for key, field in record.items()
                               if key == "source" or key.startswith("review_")}
            self.assertEqual(canonical(actual_metadata), canonical(recorded_metadata))
            guessed_blob_prefix = ("https://github.com/" + GRAPH_SOURCE["repository"]
                                   + "/blob/" + GRAPH_SOURCE["commit"] + "/")
            self.assertFalse(any(href.startswith(guessed_blob_prefix)
                                 for href, label in target.links()),
                             "unacquired source paths remain literal metadata, not guessed links")
            for axis, state in graph["dimensions"][identity].items():
                self.assertIn(axis, card_text)
                attributed = any(item.attrs.get("data-axis") == axis
                                 and item.attrs.get("data-state") == state
                                 for item in target.descendants())
                paired_text = re.search(r"(?<![A-Za-z0-9_])" + re.escape(axis)
                                        + r"[^A-Za-z0-9_]{1,40}" + re.escape(state)
                                        + r"(?![A-Za-z0-9_])", card_text, re.IGNORECASE)
                self.assertTrue(attributed or paired_text,
                                identity + ": missing exact axis/state pair " + axis + "=" + state)
        self.assertEqual(len(set(node_links.values())), 49)
        # Explicit semantic roles preserve direction and required/context state;
        # an unordered pair of matching links cannot satisfy this contract.
        rows = [item for item in elements if all(name in item.attrs for name in
                ("data-edge-from", "data-edge-to", "data-edge-relation", "data-edge-required"))]
        self.assertEqual(len(rows), 55)
        observed = []
        for row in rows:
            self.assertTrue(structurally_visible(row))
            source = row.attrs["data-edge-from"]
            target = row.attrs["data-edge-to"]
            relation = row.attrs["data-edge-relation"]
            required_text = row.attrs["data-edge-required"]
            self.assertIn(required_text, ("true", "false"))
            required = required_text == "true"
            self.assertIn(source, node_links)
            self.assertIn(target, node_links)
            fields = [item for item in row.descendants() if "data-edge-role" in item.attrs]
            self.assertEqual(Counter(item.attrs["data-edge-role"] for item in fields),
                             Counter({"source": 1, "target": 1, "relation": 1, "requirement": 1}))
            roles = {item.attrs["data-edge-role"]: item for item in fields}
            anchors = [item for item in row.descendants()
                       if item.tag == "a" and item.attrs.get("data-edge-role") in ("source", "target")]
            self.assertEqual([item.attrs["data-edge-role"] for item in anchors], ["source", "target"])
            for role, identity in (("source", source), ("target", target)):
                self.assertEqual(roles[role].tag, "a")
                self.assertEqual(roles[role].attrs.get("href", roles[role].attrs.get("xlink:href")),
                                 node_links[identity])
                self.assertEqual(roles[role].text().strip(), identity)
            self.assertEqual(roles["relation"].text().strip(), relation)
            self.assertEqual(roles["requirement"].text().strip(), "required" if required else "context")
            self.assertTrue(all(structurally_visible(item) for item in fields))
            observed.append((source, target, relation, required))
        self.assertEqual(Counter(observed), Counter((edge["from"], edge["to"], edge["relation"], edge["required"])
                                                   for edge in graph["edges"]))

    def test_fixed_source_literals_are_escaped_and_forged_hostile_records_are_rejected(self):
        """Break: interpolate real '<' text as markup, or admit changed fixed nodes."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, outputs = self.positive()
        page = outputs["proof-tree.html"]
        # This literal actually exists in the pinned Math graph; it is not an
        # injected synthetic positive that would bypass strict source equality.
        self.assertNotIn(b"r<<|x|<<rho", page)
        self.assertIn("r<<|x|<<rho", Page(page).root.text())
        hostile = '<script src="https://evil.invalid/x"></script><img onerror="boom">'
        changed = sample(graph_edit=lambda graph: graph["nodes"][next(iter(graph["nodes"]))]
                         .__setitem__("notes", hostile))
        with self.assertRaises(ValueError):
            bundle(changed)
        self.assertNotIn(hostile.encode(), page)

    def test_each_comparable_native_pin_mismatch_refuses_after_valid_positive(self):
        """Break: trust internal hashes/context or omit one externally comparable pin."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        replacements = {"checked_commit": "e" * 40, "repository": "other-owner/other-producer",
                        "run_id": "91002", "run_attempt": "11", "receipt_sha256": "7" * 64,
                        "artifact_sha256": "8" * 64,
                        "artifact_name": "research-architecture-91001-11"}
        for field, replacement in replacements.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                changed = copy.deepcopy(value)
                changed["pins"][field] = replacement
                bundle(changed)

    def test_closed_native_fields_types_and_fixed_source_are_required(self):
        """Break: coerce native identity types, allow extra trust flags, or replace source."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        for field in value["pins"]:
            for replacement in (None, 1, True, ""):
                with self.subTest(field=field, replacement=replacement), self.assertRaises(ValueError):
                    changed = copy.deepcopy(value)
                    changed["pins"][field] = replacement
                    bundle(changed)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                changed = copy.deepcopy(value)
                del changed["pins"][field]
                bundle(changed)
        changed = copy.deepcopy(value)
        changed["pins"]["custody"] = "accepted"
        with self.assertRaises(ValueError):
            bundle(changed)
        for source in (b"{}", value["source"] + b"\n"):
            with self.subTest(source_bytes=len(source)), self.assertRaises(ValueError):
                api()(value["archive"], value["pins"], source,
                      build_repository=BUILD_REPOSITORY, build_commit=BUILD_COMMIT,
                      current_payload_bytes=CURRENT_BYTES)

    def test_truncated_archive_is_refused_even_with_matching_new_archive_digest(self):
        """Break: select graph member without complete original archive admission."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        for length in (0, 21, 22, len(value["archive"]) - 1):
            changed = copy.deepcopy(value)
            changed["archive"] = changed["archive"][:length]
            changed["pins"]["artifact_sha256"] = sha(changed["archive"])
            with self.subTest(length=length), self.assertRaises(ValueError):
                bundle(changed)

    def test_constructed_projection_or_dataclass_cannot_replace_raw_archive_admission(self):
        """Break: accept a caller's projection object instead of strict raw ZIP admission."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        real = project(value)
        projection_type = type(real)
        forged = projection_type(real.original_graph_bytes, real.original_receipt_bytes,
                                 real.projection_bytes)
        for raw in (forged, json.loads(real.projection_bytes), real.projection_bytes,
                    bytearray(value["archive"]), memoryview(value["archive"])):
            with self.subTest(raw_type=type(raw).__name__), self.assertRaises(ValueError):
                api()(raw, value["pins"], value["source"], build_repository=BUILD_REPOSITORY,
                      build_commit=BUILD_COMMIT, current_payload_bytes=CURRENT_BYTES)

    def test_strict_build_context_accepts_256_char_repository_and_refuses_invalid_values(self):
        """Break: coerce build source, allow unsafe repo/commit, or impose wrong repo cap."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        exact = "o/" + "r" * 254
        self.assertEqual(len(exact), 256)
        _, outputs = self.positive(value, repository=exact)
        self.assertEqual(json.loads(outputs["current-build-manifest.json"])["build"]["repository"], exact)
        for repository in (exact + "r", "./repo", "owner/..", "/repo", "owner/", "owner/repo/extra",
                           "owner/répo", "owner/repo\n", "javascript:evil/repo", None, 7, True):
            with self.subTest(repository=repository), self.assertRaises(ValueError):
                bundle(value, repository=repository)
        for commit in ("e" * 39, "e" * 41, "E" * 40, "g" * 40, None, 7, True):
            with self.subTest(commit=commit), self.assertRaises(ValueError):
                bundle(value, commit=commit)

    def test_current_payload_exact_16_mib_passes_and_one_over_or_wrong_type_refuses(self):
        """Break: use wrong declared-payload cap, coerce bytes, or omit its digest binding."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        exact = b"x" * (16 * MIB)
        _, outputs = self.positive(value, current=exact)
        self.assertEqual(json.loads(outputs["current-build-manifest.json"])["build"]["current_payload"],
                         descriptor("current.json", exact))
        for raw in (exact + b"x", None, "{}", bytearray(b"{}"), memoryview(b"{}"), 7):
            with self.subTest(raw_type=type(raw).__name__), self.assertRaises(ValueError):
                bundle(value, current=raw)

    def manifest_boundary(self, current=CURRENT_BYTES):
        """Return exact/+1 real-admitted fixtures using physical byte calibration."""
        # Native repositories intentionally have no 256-char build-context cap.
        # A safe physical-size calibration chooses the input, while the allowed
        # manifest limit remains the independently fixed literal 65,536 bytes.
        repository = "synthetic-native-owner/" + "r" * 60000
        initial = sample(repository=repository)
        _, safe = self.positive(initial, current=current)
        initial_size = len(safe["current-build-manifest.json"])
        self.assertLess(initial_size, 65536)
        repository += "r" * (65536 - initial_size)
        exact = sample(repository=repository)
        admitted = project(exact)
        self.assertLess(len(admitted.projection_bytes), 16 * MIB)
        _, outputs = self.positive(exact, current=current)
        self.assertEqual(len(outputs["current-build-manifest.json"]), 65536,
                         "physical descriptor digit transition needs a documented fixture correction")
        self.assertLess(len(outputs["proof-tree.html"]) + 4 * len(repository), 4 * MIB,
                        "page headroom must not mask the manifest guard")
        self.assertEqual(outputs["current-build-manifest.json"],
                         canonical(json.loads(outputs["current-build-manifest.json"])))
        over = sample(repository=repository + "r")
        over_admitted = project(over)
        self.assertLess(len(over_admitted.projection_bytes), 16 * MIB)
        self.assertLess(len(over_admitted.original_receipt_bytes), 4 * MIB)
        # The mandatory repository field adds exactly one ASCII manifest byte.
        # Independently reconstruct the closed descriptor schema for the next
        # real projector result, with a same-width page-size/hash placeholder.
        # No acceptance/rejection search defines the fixed allowed limit.
        predicted = json.loads(outputs["current-build-manifest.json"])
        predicted["native"]["binding"] = over["pins"]
        exact_page = outputs["proof-tree.html"]
        page_size = len(exact_page) + exact_page.count(repository.encode("ascii"))
        self.assertEqual(len(str(page_size)), len(str(len(exact_page))))
        over_buffers = dict(zip(native_paths(over), (
            over_admitted.original_graph_bytes, over_admitted.original_receipt_bytes,
            over_admitted.projection_bytes)))
        predicted["files"] = sorted(
            [descriptor(path, raw) for path, raw in over_buffers.items()]
            + [{"path": "proof-tree.html", "bytes": page_size,
                "sha256": sha(exact_page)}], key=lambda row: row["path"])
        self.assertEqual(len(canonical(predicted)), 65537)
        for path in native_paths(over):
            self.assertEqual(len(path.split("/")[1]), 64)
        return exact, over, outputs

    def test_manifest_actual_64_kib_boundary_and_one_over_use_otherwise_admitted_native_context(self):
        """Break: remove/raise/lower actual manifest cap; calibrate bytes, never rejection search."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        exact, over, outputs = self.manifest_boundary()
        with self.assertRaises(ValueError):
            bundle(over)

    def test_every_emitted_page_manifest_and_aggregate_obeys_actual_public_caps(self):
        """Break: return over-budget actual outputs; no impossible aggregate-one-over fiction."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        for value in (sample(), sample(repository="long-native/" + "r" * 60000)):
            _, outputs = self.positive(value)
            self.assertLessEqual(len(outputs["proof-tree.html"]), 4 * MIB)
            self.assertLessEqual(len(outputs["current-build-manifest.json"]), 65536)
            self.assertLessEqual(sum(map(len, outputs.values())), 32 * MIB)
        # All component maxima together cannot reach aggregate one-over:
        # 4MiB graph + 4MiB receipt + 16MiB projection + 4MiB page + 64KiB.
        self.assertEqual(4 * MIB + 4 * MIB + 16 * MIB + 4 * MIB + 65536, 29425664)
        self.assertLess(29425664, 33554432)

    def test_private_archive_and_diagnostic_members_never_enter_any_public_output(self):
        """Break: copy raw ZIP, diagnostic bodies, or source filesystem paths to public files."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, outputs = self.positive()
        marker = fixture_module("test_site_graph_projection").CONSOLE_MARKER.strip()
        for path, raw in outputs.items():
            self.assertNotIn(marker, raw, path)
            self.assertNotIn(value["archive"], raw, path)
            self.assertNotIn(str(ROOT).encode(), raw, path)
            self.assertFalse(path.endswith((".zip", ".log", ".stdout", ".stderr")))

    def test_builder_all_none_options_preserve_exact_legacy_behavior(self):
        """Break: native connector changes default current/history/page output bytes."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        self.positive()
        with TemporaryDirectory() as temporary:
            site, payload = prepare_site(Path(temporary))
            first = builder()(site, payload)
            before = snapshot(site)
            second = builder()(site, payload, **dict.fromkeys(NATIVE_ARGUMENTS))
            self.assertEqual(second, first)
            self.assertEqual(snapshot(site), before)
            self.assertFalse((site / "current-build-manifest.json").exists())

    def test_real_builder_emits_exact_bundle_and_preserves_payload_and_history(self):
        """Break: unused bundle API, wrong payload bytes, or omitted/altered builder outputs."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        with TemporaryDirectory() as temporary:
            site, payload = prepare_site(Path(temporary))
            original_payload = payload.read_bytes()
            builder()(site, payload)
            legacy = snapshot(site)
            expected = bundle(value, current=original_payload)
            builder()(site, payload, **build_arguments(value))
            actual = snapshot(site)
            for path, raw in expected.items():
                self.assertEqual(actual[path], raw, path)
            for path, raw in legacy.items():
                self.assertEqual(actual[path], raw, path)
            self.assertEqual(payload.read_bytes(), original_payload)
            self.assertEqual(actual["current.json"], original_payload)

    def test_real_builder_refuses_bad_native_pin_and_truncation_before_any_site_write(self):
        """Break: write legacy/generated files before native admission has completed."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        bad_pin = build_arguments(value)
        bad_pin["native_graph_expected"]["receipt_sha256"] = "7" * 64
        truncated = build_arguments(value)
        truncated["native_graph_archive"] = value["archive"][:-1]
        truncated["native_graph_expected"]["artifact_sha256"] = sha(truncated["native_graph_archive"])
        for arguments in (bad_pin, truncated):
            with self.subTest(kind="bad-pin" if arguments is bad_pin else "truncated"):
                with TemporaryDirectory() as temporary:
                    site, payload = prepare_site(Path(temporary))
                    before = site_state(site)
                    with self.assertRaises(ValueError):
                        builder()(site, payload, **arguments)
                    self.assertEqual(site_state(site), before)

    def test_real_builder_refuses_every_partial_native_input_without_legacy_fallback(self):
        """Break: treat incomplete native request as optional/default and publish legacy output."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        complete = build_arguments(value)
        with TemporaryDirectory() as temporary:
            site, payload = prepare_site(Path(temporary))
            before = site_state(site)
            for count in range(1, len(NATIVE_ARGUMENTS)):
                for selected in itertools.combinations(NATIVE_ARGUMENTS, count):
                    arguments = {name: complete[name] for name in selected}
                    with self.subTest(selected=selected), self.assertRaises(ValueError):
                        builder()(site, payload, **arguments)
                    self.assertEqual(site_state(site), before)

    def test_real_builder_refuses_wrong_native_types_before_writing(self):
        """Break: coerce partial caller inputs or defer their validation until after writes."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        wrong = {
            "native_graph_archive": bytearray(value["archive"]),
            "native_graph_expected": list(value["pins"].items()),
            "native_graph_source": bytearray(value["source"]),
            "build_repository": 7, "build_commit": True,
        }
        with TemporaryDirectory() as temporary:
            site, payload = prepare_site(Path(temporary))
            before = site_state(site)
            for name, replacement in wrong.items():
                arguments = build_arguments(value)
                arguments[name] = replacement
                with self.subTest(field=name), self.assertRaises(ValueError):
                    builder()(site, payload, **arguments)
                self.assertEqual(site_state(site), before)

    def test_repeated_complete_native_build_preserves_every_emitted_site_byte(self):
        """Break: generated proof page fails next-build marker admission or changes on repeat."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        with TemporaryDirectory() as temporary:
            site, payload = prepare_site(Path(temporary))
            first = builder()(site, payload, **build_arguments(value))
            before = snapshot(site)
            second = builder()(site, payload, **build_arguments(value))
            self.assertEqual(second, first)
            self.assertEqual(snapshot(site), before)

    def test_changed_native_binding_preserves_current_json_and_all_math_snapshots_and_aliases(self):
        """Break: native run metadata churns mathematical versions or rewrites legacy/history bytes."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        first_value, _ = self.positive()
        second_value = sample(run="91002", checked_commit="e" * 40)
        self.positive(second_value)
        self.assertNotEqual(first_value["pins"]["artifact_sha256"],
                            second_value["pins"]["artifact_sha256"])
        with TemporaryDirectory() as temporary:
            site, payload = prepare_site(Path(temporary))
            builder()(site, payload)
            legacy = snapshot(site)
            builder()(site, payload, **build_arguments(first_value))
            first_native = snapshot(site)
            expected_second = bundle(second_value, current=payload.read_bytes())
            builder()(site, payload, **build_arguments(second_value))
            second_native = snapshot(site)
            for path, content in expected_second.items():
                self.assertEqual(second_native[path], content, path)
            current_page = Page(second_native["proof-tree.html"])
            for literal in (second_value["pins"]["repository"], second_value["pins"]["checked_commit"],
                            second_value["pins"]["run_id"], second_value["pins"]["run_attempt"]):
                self.assertIn(literal, current_page.root.text())
            second_manifest = json.loads(second_native["current-build-manifest.json"])
            self.assertEqual(second_manifest["native"]["binding"], second_value["pins"])
            page_rows = [row for row in second_manifest["files"] if row["path"] == "proof-tree.html"]
            self.assertEqual(page_rows, [descriptor("proof-tree.html", second_native["proof-tree.html"])])
            for path, raw in legacy.items():
                self.assertEqual(second_native[path], raw, path)
            for path in native_paths(first_value):
                self.assertEqual(second_native[path], first_native[path], path)
            self.assertNotEqual(first_native["current-build-manifest.json"],
                                second_native["current-build-manifest.json"])
            self.assertEqual({p: b for p, b in first_native.items() if p.startswith("snapshots/")},
                             {p: b for p, b in second_native.items() if p.startswith("snapshots/")})

    def test_native_generated_destination_conflict_refuses_before_existing_bytes_change(self):
        """Break: start writes before checking newly added native asset/page destinations."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        prefix = "native-graph/" + value["pins"]["artifact_sha256"]
        conflicts = [("directory", name) for name in
                     ("proof-tree.html", "current-build-manifest.json", native_paths(value)[0])]
        conflicts += [("file", "native-graph"), ("file", prefix)]
        for kind, conflict in conflicts:
            with self.subTest(kind=kind, conflict=conflict), TemporaryDirectory() as temporary:
                site, payload = prepare_site(Path(temporary))
                target = site / conflict
                target.parent.mkdir(parents=True, exist_ok=True)
                if kind == "directory":
                    target.mkdir()
                else:
                    target.write_bytes(b"non-directory native ancestor\n")
                before = site_state(site)
                with self.assertRaises(ValueError):
                    builder()(site, payload, **build_arguments(value))
                self.assertEqual(site_state(site), before)
                self.assertTrue(target.is_dir() if kind == "directory" else target.is_file())

    def test_native_to_legacy_fallback_refuses_before_any_write(self):
        """Break: legacy-only rebuild leaves a stale native manifest or rewrites its page."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        with TemporaryDirectory() as temporary:
            site, payload = prepare_site(Path(temporary))
            builder()(site, payload, **build_arguments(value))
            before = site_state(site)
            for arguments in ({}, dict.fromkeys(NATIVE_ARGUMENTS)):
                with self.subTest(explicit_none=bool(arguments)), self.assertRaises(ValueError):
                    builder()(site, payload, **arguments)
                self.assertEqual(site_state(site), before)

    def test_partial_or_broken_reserved_native_state_forbids_legacy_mode(self):
        """Break: check only a complete/exists-true native bundle before legacy fallback."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        self.positive()
        for reserved in ("proof-tree.html", "current-build-manifest.json", "native-graph",
                         "broken-native-graph-link", "broken-manifest-link"):
            with self.subTest(reserved=reserved), TemporaryDirectory() as temporary:
                site, payload = prepare_site(Path(temporary))
                if reserved == "native-graph":
                    (site / reserved).mkdir()
                elif reserved == "broken-native-graph-link":
                    (site / "native-graph").symlink_to(Path(temporary) / "absent")
                elif reserved == "broken-manifest-link":
                    (site / "current-build-manifest.json").symlink_to(Path(temporary) / "absent")
                else:
                    (site / reserved).write_bytes(b"partial native state\n")
                before = site_state(site)
                with self.assertRaises(ValueError):
                    builder()(site, payload)
                self.assertEqual(site_state(site), before)

    def test_changed_current_payload_regenerates_consistent_native_aliases_without_math_churn(self):
        """Break: preserve stale native aliases after current payload bytes change."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        with TemporaryDirectory() as temporary:
            site, payload = prepare_site(Path(temporary))
            builder()(site, payload, **build_arguments(value))
            before = snapshot(site)
            prior_index = json.loads(before["snapshots/index.json"])
            changed = json.loads(payload.read_bytes())
            changed["observed_at"] = "2026-10-13T12:00:01Z"
            raw = (json.dumps(changed, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            payload.write_bytes(raw)
            expected = bundle(value, current=raw)
            builder()(site, payload, **build_arguments(value))
            after = snapshot(site)
            for path, content in expected.items():
                self.assertEqual(after[path], content, path)
            self.assertEqual(after["current.json"], raw)
            self.assertNotEqual(after["current-build-manifest.json"],
                                before["current-build-manifest.json"])
            self.assertEqual(json.loads(after["snapshots/index.json"])["archives"], prior_index["archives"])
            for path, content in before.items():
                if path.startswith("snapshots/") and path != "snapshots/index.json":
                    self.assertEqual(after[path], content, path)

    def test_same_artifact_existing_native_bytes_conflict_refuses_without_repair(self):
        """Break: overwrite an archive-addressed graph/receipt/projection with conflicting bytes."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        for member in native_paths(value):
            with self.subTest(member=member), TemporaryDirectory() as temporary:
                site, payload = prepare_site(Path(temporary))
                builder()(site, payload, **build_arguments(value))
                (site / member).write_bytes((site / member).read_bytes() + b"tampered\n")
                before = site_state(site)
                with self.assertRaises(ValueError):
                    builder()(site, payload, **build_arguments(value))
                self.assertEqual(site_state(site), before)

    def test_native_namespace_and_asset_symlinks_refuse_before_any_write(self):
        """Break: follow a native namespace/archive-directory/asset symlink during build."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, outputs = self.positive()
        prefix = "native-graph/" + value["pins"]["artifact_sha256"]
        for relative in ("native-graph", prefix, *native_paths(value),
                         "proof-tree.html", "current-build-manifest.json"):
            with self.subTest(relative=relative), TemporaryDirectory() as temporary:
                root = Path(temporary)
                site, payload = prepare_site(root)
                outside = root / "outside"
                if relative in ("native-graph", prefix):
                    outside.mkdir()
                else:
                    outside.write_bytes(outputs.get(relative, b"outside sentinel\n"))
                target = site / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(outside)
                before = site_state(site)
                outside_before = snapshot(outside) if outside.is_dir() else outside.read_bytes()
                with self.assertRaises(ValueError):
                    builder()(site, payload, **build_arguments(value))
                self.assertEqual(site_state(site), before)
                self.assertEqual(snapshot(outside) if outside.is_dir() else outside.read_bytes(),
                                 outside_before)

    def test_native_mode_still_refuses_other_root_page_marker_defects(self):
        """Break: skip all root marker checks when exempting only the generated proof tree."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        with TemporaryDirectory() as temporary:
            site, payload = prepare_site(Path(temporary))
            builder()(site, payload, **build_arguments(value))
            (site / "index.html").write_bytes(b"<html><body>No markers</body></html>\n")
            before = site_state(site)
            with self.assertRaises(ValueError):
                builder()(site, payload, **build_arguments(value))
            self.assertEqual(site_state(site), before)

    def test_actual_builder_manifest_capacity_refusal_preserves_every_existing_site_byte(self):
        """Break: write a new current observation before admitted native manifest capacity refusal."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        with TemporaryDirectory() as temporary:
            site, payload = prepare_site(Path(temporary))
            exact, over, outputs = self.manifest_boundary(current=payload.read_bytes())
            builder()(site, payload, **build_arguments(exact))
            initial = snapshot(site)
            for path, content in outputs.items():
                self.assertEqual(initial[path], content, path)
            # A changed valid observation makes an early legacy/current write
            # observable. Its width is unchanged; the same actual manifest cap
            # is recalibrated against these exact new current-payload bytes.
            changed = json.loads(payload.read_bytes())
            changed["observed_at"] = "2026-10-13T12:00:01Z"
            changed_raw = (json.dumps(changed, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            payload.write_bytes(changed_raw)
            exact, over, outputs = self.manifest_boundary(current=changed_raw)
            self.assertEqual(len(outputs["current-build-manifest.json"]), 65536)
            # Demonstrate that the actual already-native builder accepts this
            # new observation and exact-cap context, not merely a fresh site.
            builder()(site, payload, **build_arguments(exact))
            safe_state = snapshot(site)
            for path, content in outputs.items():
                self.assertEqual(safe_state[path], content, path)
            changed["observed_at"] = "2026-10-13T12:00:02Z"
            next_raw = (json.dumps(changed, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            payload.write_bytes(next_raw)
            exact, over, outputs = self.manifest_boundary(current=next_raw)
            self.assertEqual(len(outputs["current-build-manifest.json"]), 65536)
            admitted = project(over)
            self.assertLess(len(admitted.projection_bytes), 16 * MIB)
            before = site_state(site)
            with self.assertRaises(ValueError):
                builder()(site, payload, **build_arguments(over))
            self.assertEqual(site_state(site), before)

    def test_native_builder_malformed_or_unserializable_payload_refuses_before_any_write(self):
        """Break: native mode writes files before invalid payload parse/render serialization refuses."""
        self.assertTrue(BUNDLE.is_file(), MISSING)
        value, _ = self.positive()
        with TemporaryDirectory() as temporary:
            site, payload = prepare_site(Path(temporary))
            builder()(site, payload, **build_arguments(value))
            valid_data = json.loads(payload.read_bytes())
            invalid_unicode = copy.deepcopy(valid_data)
            invalid_unicode["graph"]["data"]["nodes"]["alpha"]["notes"] = "\ud800"
            nonserializable = (json.dumps(invalid_unicode, ensure_ascii=True, indent=2) + "\n").encode("ascii")
            cases = (b'{"schema_version":', b'{"bad":NaN}\n', nonserializable)
            for raw in cases:
                with self.subTest(raw_sha256=sha(raw)):
                    payload.write_bytes(raw)
                    before = site_state(site)
                    # All five native inputs are present, well typed, and the
                    # archive was genuinely admitted by the positive control.
                    with self.assertRaises(ValueError):
                        builder()(site, payload, **build_arguments(value))
                    self.assertEqual(site_state(site), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
