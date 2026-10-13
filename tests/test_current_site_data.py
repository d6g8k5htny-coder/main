"""Offline controls for current-source generation, using real Git tree fixtures."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
OWNER = "d6g8k5htny-coder"
MAIN = f"{OWNER}/main"
MATH = f"{OWNER}/Math-"
GRAPH_PATH = "frontiers/downstream_gate_20260925/GRAPH.json"
PROOFS = [f"experiments/universality/{name}/PROOF.md" for name in (
    "fold_structural_universality", "periodized_gaussian_h0", "iid_short_bar_process")]
NOW = "2026-10-13T18:00:00Z"
FORMULA = b"# Source proof\n\n    nu(ell) ~ C ell^(-1/3),\n      ell down to 0.\n\nScope remains conditional.\n"
GRAPH = {"schema_version": 1, "object": "fixture-only", "nodes": {
    "a": {"classification": "OPEN_ACTIVE"},
    "b": {"classification": "BLOCKED_ABSENT", "fingerprint": "ABSENT"}},
    "edges": [{"from": "a", "to": "b", "required": True, "relation": "imports"},
              {"from": "b", "to": "a", "required": False, "relation": "context"}],
    "non_discharge_tokens": ["TEST_PASS"]}


def git_fixture(repository, files):
    """Git itself supplies objects; no production hashing helper makes the fixture."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        def git(*args, input=None):
            return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", *args],
                                           cwd=root, input=input, stderr=subprocess.DEVNULL)
        git("init", "--quiet")
        for path, content in files.items():
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        git("add", "--all")
        tree = git("write-tree").decode().strip()
        commit = git("-c", "user.name=Current source fixture",
                     "-c", "user.email=fixture@example.invalid", "commit-tree", tree,
                     "-m", "Offline fixture").decode().strip()
        entries = []
        for item in git("ls-tree", "-r", "-t", "-l", "-z", tree).split(b"\0"):
            if not item:
                continue
            fields, path = item.split(b"\t", 1)
            mode, kind, sha, size = fields.decode().split()
            row = {"path": path.decode(), "mode": mode, "type": kind, "sha": sha}
            if size != "-":
                row["size"] = int(size)
            entries.append(row)
    base = f"https://api.github.com/repos/{repository}"
    metadata = {"sha": commit, "commit": {"tree": {"sha": tree}}}
    return {
        "commit": commit, "tree": tree,
        "json": {base + "/commits/HEAD": metadata,
                 base + "/commits/" + commit: metadata,
                 base + "/git/trees/" + tree + "?recursive=1":
                     {"sha": tree, "truncated": False, "tree": entries}},
        "bytes": {f"https://raw.githubusercontent.com/{repository}/{commit}/{path}": raw
                  for path, raw in files.items()}}


class FixtureReader:
    def __init__(self, fixtures):
        self.json = {}
        self.bytes = {}
        self.calls = []
        for fixture in fixtures:
            self.json.update(copy.deepcopy(fixture["json"]))
            self.bytes.update(fixture["bytes"])

    def get_json(self, url):
        self.calls.append(url)
        if url not in self.json:
            raise ValueError("fixture has no pinned JSON: " + url)
        return copy.deepcopy(self.json[url])

    def get_bytes(self, url):
        self.calls.append(url)
        if url not in self.bytes:
            raise ValueError("fixture has no pinned bytes: " + url)
        return self.bytes[url]


class CurrentSiteDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_files = {"STATUS.md": b"# Status\nAMEND remains AMEND.\n",
                          "docs/RESEARCH_INDEX.md": b"# Research index\n",
                          **{path: FORMULA for path in PROOFS},
                          "experiments/new_work/PROOF.md": b"# New public source\n",
                          "legacy/old.md": b"old", "quarantine/a.md": b"held",
                          "docs/public-math/recovery-20260927/a.md": b"recovery",
                          "reviews/W3Scratch/a.lean": b"scratch",
                          "audits/dropbox_import/private-source.txt": b"private",
                          "drive_sources/a.md": b"private",
                          "operational/a.md": b"private"}
        cls.math_files = {"PROOF_INDEX.md": b"# Proof index\nNo inferred acceptance.\n",
                          GRAPH_PATH: json.dumps(GRAPH).encode(),
                          "frontiers/current/PROOF.md": b"# Public theorem source\n"}
        cls.fixtures = [git_fixture(MAIN, cls.main_files), git_fixture(MATH, cls.math_files)]

    def setUp(self):
        path = ROOT / "tools/current_site_data.py"
        self.assertTrue(path.is_file(), "current-source generator has not been implemented")
        spec = importlib.util.spec_from_file_location("current_site_data", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.reader = FixtureReader(self.fixtures)

    def generate(self, **kwargs):
        return self.module.generate(reader=self.reader, observed_at=NOW, **kwargs)

    def tree(self, index=0):
        fixture = self.fixtures[index]
        return self.reader.json[f"https://api.github.com/repos/{(MAIN, MATH)[index]}"
                                f"/git/trees/{fixture['tree']}?recursive=1"]

    def replace_math(self, files):
        self.reader = FixtureReader([self.fixtures[0], git_fixture(MATH, files)])

    def test_complete_observation_preserves_graph_and_exact_equation_lines(self):
        data = self.generate()
        self.assertEqual(set(data), {"schema_version", "observed_at", "scientific_effect",
                                     "repositories", "sources", "graph", "equations", "references"})
        self.assertEqual((data["schema_version"], data["observed_at"], data["scientific_effect"]),
                         (1, NOW, "NONE"))
        self.assertEqual(data["graph"]["data"], GRAPH)
        self.assertEqual(data["graph"]["identity"]["path"], GRAPH_PATH)
        equation = data["equations"][0]
        self.assertEqual((equation["text"], equation["line_start"], equation["line_end"]),
                         ("    nu(ell) ~ C ell^(-1/3),\n      ell down to 0.\n", 3, 4))
        self.assertEqual(equation["sha256"], hashlib.sha256(FORMULA.split(b"\n\n")[1] + b"\n").hexdigest())
        self.assertEqual(len(data["equations"]), 3)
        self.assertEqual({ref["path"] for ref in data["references"]},
                         {"STATUS.md", "docs/RESEARCH_INDEX.md", "PROOF_INDEX.md"})
        self.assertEqual(data["repositories"], [
            {"repository": MAIN, "commit": self.fixtures[0]["commit"], "tree": self.fixtures[0]["tree"]},
            {"repository": MATH, "commit": self.fixtures[1]["commit"], "tree": self.fixtures[1]["tree"]}])

    def test_inventory_includes_new_public_sources_and_omits_operational_copies(self):
        paths = {(row["repository"], row["path"]) for row in self.generate()["sources"]}
        self.assertIn((MAIN, "experiments/new_work/PROOF.md"), paths)
        self.assertIn((MATH, "frontiers/current/PROOF.md"), paths)
        for path in self.main_files:
            if path not in {"STATUS.md", "docs/RESEARCH_INDEX.md", "experiments/new_work/PROOF.md", *PROOFS}:
                self.assertNotIn((MAIN, path), paths)

    def test_default_ref_is_resolved_once_and_all_reads_are_immutable(self):
        self.generate()
        for repository, fixture in zip((MAIN, MATH), self.fixtures):
            base = "https://api.github.com/repos/" + repository
            self.assertEqual(self.reader.calls.count(base + "/commits/HEAD"), 1)
            self.assertEqual(self.reader.calls.count(base + "/git/trees/" + fixture["tree"] + "?recursive=1"), 1)
        raw_calls = [url for url in self.reader.calls if "raw.githubusercontent.com" in url]
        self.assertEqual(len(raw_calls), 7)
        self.assertTrue(all("/main/" + self.fixtures[0]["commit"] + "/" in url or
                            "/Math-/" + self.fixtures[1]["commit"] + "/" in url for url in raw_calls))

    def test_explicit_pins_do_not_resolve_a_moving_default_ref(self):
        self.generate(main_commit=self.fixtures[0]["commit"], math_commit=self.fixtures[1]["commit"])
        self.assertFalse(any(url.endswith("/HEAD") for url in self.reader.calls))

    def test_truncated_tree_is_refused(self):
        self.tree()["truncated"] = True
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.generate()

    def test_missing_truncation_marker_is_refused(self):
        del self.tree()["truncated"]
        with self.assertRaises(ValueError):
            self.generate()

    def test_tree_identity_mismatch_is_refused(self):
        self.tree()["sha"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "tree"):
            self.generate()

    def test_omitted_tree_entry_is_refused_even_when_not_marked_truncated(self):
        self.tree()["tree"] = [row for row in self.tree()["tree"] if row["path"] != "experiments/new_work/PROOF.md"]
        with self.assertRaisesRegex(ValueError, "tree"):
            self.generate()

    def test_wrong_source_size_is_refused(self):
        url = next(url for url in self.reader.bytes if url.endswith("/STATUS.md"))
        self.reader.bytes[url] += b"Promoted."
        with self.assertRaisesRegex(ValueError, "(blob|bytes|size)"):
            self.generate()

    def test_wrong_blob_bytes_at_the_correct_size_are_refused(self):
        url = next(url for url in self.reader.bytes if url.endswith("/STATUS.md"))
        self.reader.bytes[url] = self.reader.bytes[url].replace(b"AMEND", b"CLOSE")
        with self.assertRaisesRegex(ValueError, "blob integrity"):
            self.generate()

    def test_invalid_explicit_commit_is_refused_without_reads(self):
        for pin in ("main", "../HEAD", "0" * 39, True):
            with self.subTest(pin=pin), self.assertRaises(ValueError):
                self.generate(main_commit=pin)
        self.assertEqual(self.reader.calls, [])

    def test_explicit_commit_response_must_match_the_requested_pin(self):
        self.reader.json[f"https://api.github.com/repos/{MAIN}/commits/{self.fixtures[0]['commit']}"]["sha"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "commit"):
            self.generate(main_commit=self.fixtures[0]["commit"])

    def test_unsafe_tree_paths_are_refused(self):
        for path in ("../outside.md", "/absolute.md", "a/../b.md", "a//b.md", "a\\b.md", "a/%2e%2e/b.md", "a\nsecret.md"):
            with self.subTest(path=path):
                self.reader = FixtureReader(self.fixtures)
                self.tree()["tree"].append({"path": path, "mode": "100644", "type": "blob", "sha": "0" * 40, "size": 1})
                with self.assertRaisesRegex(ValueError, "path"):
                    self.generate()

    def test_duplicate_tree_paths_are_refused(self):
        self.tree()["tree"].append(copy.deepcopy(self.tree()["tree"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.generate()

    def test_required_source_missing_from_a_valid_tree_is_refused(self):
        for missing in (GRAPH_PATH, "PROOF_INDEX.md"):
            with self.subTest(missing=missing):
                self.replace_math({path: raw for path, raw in self.math_files.items() if path != missing})
                with self.assertRaisesRegex(ValueError, "missing"):
                    self.generate()

    def test_required_equation_source_missing_from_a_valid_tree_is_refused(self):
        files = {path: raw for path, raw in self.main_files.items() if path != PROOFS[0]}
        self.reader = FixtureReader([git_fixture(MAIN, files), self.fixtures[1]])
        with self.assertRaisesRegex(ValueError, "missing"):
            self.generate()

    def test_malformed_graph_is_refused(self):
        for raw in (b"not json", b'{"nodes":{},"nodes":{},"edges":[]}', b'{"nodes":{}}'):
            with self.subTest(raw=raw):
                self.replace_math(dict(self.math_files, **{GRAPH_PATH: raw}))
                with self.assertRaisesRegex(ValueError, "(JSON|graph)"):
                    self.generate()

    def test_missing_graph_fields_are_refused_without_inference(self):
        for field in ("schema_version", "nodes", "edges"):
            graph = copy.deepcopy(GRAPH)
            del graph[field]
            self.replace_math(dict(self.math_files, **{GRAPH_PATH: json.dumps(graph).encode()}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "graph"):
                self.generate()

    def test_dangling_contextual_edge_is_refused(self):
        graph = copy.deepcopy(GRAPH)
        graph["edges"][1]["to"] = "absent"
        self.replace_math(dict(self.math_files, **{GRAPH_PATH: json.dumps(graph).encode()}))
        with self.assertRaisesRegex(ValueError, "(absent|dangling)"):
            self.generate()

    def test_required_cycle_is_refused_but_contextual_cycle_is_preserved(self):
        graph = copy.deepcopy(GRAPH)
        graph["edges"][1]["required"] = True
        self.replace_math(dict(self.math_files, **{GRAPH_PATH: json.dumps(graph).encode()}))
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.generate()

    def test_missing_or_nonboolean_required_field_is_refused(self):
        for value in (None, "true", 1):
            graph = copy.deepcopy(GRAPH)
            graph["edges"][0]["required"] = value
            self.replace_math(dict(self.math_files, **{GRAPH_PATH: json.dumps(graph).encode()}))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "required"):
                self.generate()

    def test_hostile_formula_text_is_kept_inert_and_byte_exact(self):
        hostile = b'# Proof\r\n\r\n    <script>alert("inert")</script> = 1\r\n'
        files = dict(self.main_files, **{PROOFS[0]: hostile})
        self.reader = FixtureReader([git_fixture(MAIN, files), self.fixtures[1]])
        equation = self.generate()["equations"][0]
        self.assertEqual(equation["text"], '    <script>alert("inert")</script> = 1\r\n')
        self.assertEqual(equation["sha256"], hashlib.sha256(hostile.split(b"\r\n\r\n")[1]).hexdigest())

    def test_absent_equations_stay_absent_instead_of_being_invented(self):
        files = dict(self.main_files, **{path: b"# Proof\nNo displayed blocks recorded.\n" for path in PROOFS})
        self.reader = FixtureReader([git_fixture(MAIN, files), self.fixtures[1]])
        self.assertEqual(self.generate()["equations"], [])

    def test_multiline_equations_retain_internal_blank_lines_and_final_line(self):
        source = b"# Proof\n\n    a=b\n\n    c=d"
        files = dict(self.main_files, **{PROOFS[0]: source})
        self.reader = FixtureReader([git_fixture(MAIN, files), self.fixtures[1]])
        equation = self.generate()["equations"][0]
        self.assertEqual((equation["text"], equation["line_start"], equation["line_end"]),
                         ("    a=b\n\n    c=d", 3, 5))

    def test_invalid_observation_time_is_refused_before_network_reads(self):
        for observation in ("2026-10-13", "2026-13-01T00:00:00Z", "2026-10-13T00:00:00-05:00", None):
            if observation is None:
                continue
            with self.subTest(observation=observation), self.assertRaisesRegex(ValueError, "UTC"):
                self.module.generate(reader=self.reader, observed_at=observation)
        self.assertEqual(self.reader.calls, [])

    def test_failed_refresh_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "current.json"
            output.write_bytes(b"previous deployment\n")
            self.tree()["truncated"] = True
            with self.assertRaises(ValueError):
                self.module.write_current(output, reader=self.reader, observed_at=NOW)
            self.assertEqual(output.read_bytes(), b"previous deployment\n")
            self.assertEqual([path.name for path in Path(tmp).iterdir()], ["current.json"])

    def test_successful_refresh_writes_valid_json_with_no_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "current.json"
            self.module.write_current(output, reader=self.reader, observed_at=NOW)
            self.assertEqual(json.loads(output.read_bytes())["graph"]["data"], GRAPH)
            self.assertEqual([path.name for path in Path(tmp).iterdir()], ["current.json"])

    def test_network_reader_rejects_oversize_and_duplicate_json(self):
        class Transport:
            def __init__(self, raw):
                self.raw = raw
                self.requests = []
            def open(self, request, timeout):
                self.requests.append((request, timeout))
                return io.BytesIO(self.raw)
        for raw in (b"x" * 33, b'{"a":1,"a":2}'):
            transport = Transport(raw)
            reader = self.module.GitHubReader(opener=transport, max_bytes=32)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                reader.get_json(f"https://api.github.com/repos/{MAIN}/commits/HEAD")

    def test_network_reader_bounds_requests_and_keeps_token_out_of_raw_reads(self):
        class Transport:
            def __init__(self):
                self.requests = []
            def open(self, request, timeout):
                self.requests.append((request, timeout))
                return io.BytesIO(b"{}")
        transport = Transport()
        reader = self.module.GitHubReader(opener=transport, token="fixture-secret", max_requests=2)
        reader.get_json(f"https://api.github.com/repos/{MAIN}/commits/HEAD")
        reader.get_bytes(f"https://raw.githubusercontent.com/{MAIN}/{self.fixtures[0]['commit']}/STATUS.md")
        with self.assertRaisesRegex(ValueError, "(request|fetch)"):
            reader.get_json(f"https://api.github.com/repos/{MATH}/commits/HEAD")
        self.assertEqual(transport.requests[0][0].get_header("Authorization"), "Bearer fixture-secret")
        self.assertIsNone(transport.requests[1][0].get_header("Authorization"))
        self.assertTrue(all(0 < timeout <= 30 for _, timeout in transport.requests))

    def test_network_reader_rejects_nonpublic_hosts_and_repository_paths(self):
        reader = self.module.GitHubReader(token="fixture-secret")
        for url in ("http://api.github.com/repos/" + MAIN + "/commits/HEAD",
                    "https://api.github.com.example.invalid/repos/" + MAIN + "/commits/HEAD",
                    "https://api.github.com/repos/another/private/commits/HEAD"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "nonpublic"):
                reader.get_json(url)
        self.assertEqual(reader.requests, 0)


if __name__ == "__main__":
    unittest.main()
