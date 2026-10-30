"""Tests for tools/cbz/reference_crawler.py, on local multi-repository Git fixtures.

The fixture federation, under owner "acme":
  math       two commits, A1 and A2 (pinned). A2 changes one Lean statement, adds a theorem,
             changes a Python signature and changes a JSON shape. A notes file whose name
             needs percent-encoding is unchanged.
  registry   registry.json, with a "repositories" map and artifacts that cite math at A1.
  main       sources.json and docs/LINKS.md, citing math at A1 by blob, by excerpt and by URL;
             tests/fixtures/refs.json holds a synthetic reference to a commit that does not exist.
  workspace  pins.json and workspace/repositories.json; its canonical_inputs name the registry.
Every check runs offline against these repositories. No network is used.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "cbz"))
import reference_crawler as rc  # noqa: E402

LEAN_A1 = """namespace Demo
/- a block /- nested -/ comment -/
theorem foo : 1 + 1 = 2 := by rfl
-- theorem fake : False := sorry
theorem bar (n : Nat) :
    n + 0 = n := by simp
end Demo
"""
LEAN_A2 = """namespace Demo
theorem foo : 1 + 1 = 2 := by rfl
theorem bar (n : Nat) : 0 + n = n := by simp
theorem baz : True := trivial
end Demo
"""
PY_A1 = "def f(x, y=1):\n    return x + y\n\nclass K:\n    def m(self):\n        return 1\n"
PY_A2 = "def f(x, y=2, *, z=0):\n    return x + y + z\n\nclass K:\n    def m(self):\n        return 1\n"
JSON_A1 = {"a": 1, "b": {"c": "x"}}
JSON_A2 = {"a": 1, "b": {"c": 2}, "d": []}
NOTES_PATH = "notes/Open Problems \u2014 list.md"
NOTES_URL_PATH = "notes/Open%20Problems%20%E2%80%94%20list.md"
NOTES = "# Open problems\n\nfirst line\nsecond line\nthird line\n"
EXCERPT = "first line\nsecond line\n"  # lines 3-4 of NOTES


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()


def init(path):
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "t@example.invalid")
    git(path, "config", "user.name", "t")
    # Fixtures are hermetic: a host's commit-signing setup must never be invoked by the suite.
    git(path, "config", "commit.gpgsign", "false")
    return path


def commit(repo, files, message):
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            p.unlink()
        elif isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content if isinstance(content, str) else json.dumps(content, indent=1))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def blob_at(repo, sha, path):
    return git(repo, "rev-parse", f"{sha}:{path}")


class Federation:
    """Builds the fixture; options let each test break exactly one thing."""

    def __init__(self, tmp, **opt):
        self.root = Path(tmp)
        self.math = init(self.root / "math")
        self.a1 = commit(self.math, {"Thm.lean": LEAN_A1, "tool.py": PY_A1, "data.json": JSON_A1,
                                     "README.md": "math\n", NOTES_PATH: NOTES}, "A1")
        self.a2 = commit(self.math, {"Thm.lean": LEAN_A2, "tool.py": PY_A2, "data.json": JSON_A2}, "A2")
        if opt.get("math_pin_old"):
            git(self.math, "checkout", "-q", "-b", "keep", self.a2)
        readme = b"math\n"
        self.registry = init(self.root / "registry")
        artifact = {"repository": "math", "commit": self.a1, "path": "README.md",
                    opt.get("artifact_sha_key", "sha256"): opt.get("artifact_sha256", hashlib.sha256(readme).hexdigest()),
                    opt.get("artifact_bytes_key", "bytes"): opt.get("artifact_bytes", len(readme)),
                    "byte_identical_at_refreshed_cut": self.a2}  # true: README.md is unchanged at A2
        repos = {f"acme/{opt.get('registry_math_name', opt.get('math_name', 'math'))}": {},
                 "acme/registry": {}, "acme/main": {}, "acme/workspace": {}}
        repos.update({f"acme/{n}": {} for n in opt.get("registry_extra", [])})
        self.reg_old = commit(self.registry, {"registry.json": {"repositories": repos, "artifacts": []}}, "R1")
        reg_doc = {"repositories": repos, "artifacts": [artifact]}
        self.reg = commit(self.registry, {"registry.json": opt.get("registry_edit", lambda d: d)(reg_doc)}, "R2")
        self.main = init(self.root / "main")
        source = {"repository": opt.get("source_repo", "math"),
                  opt.get("source_commit_key", "commit"): opt.get("source_commit", self.a1),
                  "path": opt.get("source_path", "Thm.lean"),
                  opt.get("source_blob_key", "blob"): opt.get("source_blob") or blob_at(self.math, self.a1, "Thm.lean")}
        source.update(opt.get("source_extra", {}))
        if opt.get("source_also_at"):
            source["byte_identical_at_refreshed_cut"] = self.a2  # false: Thm.lean changed at A2
        excerpt = {"repository": "math", "commit": self.a1, "path": NOTES_PATH, "line_start": 3, "line_end": 4,
                   "text": EXCERPT, "sha256": hashlib.sha256(EXCERPT.encode()).hexdigest()}
        excerpt.update(opt.get("excerpt", {}))
        foreign = {"repository": "other/math", "commit": "e" * 40, "path": "x.lean"}  # same short name, other owner
        files = {"sources.json": {"sources": [source, excerpt, foreign]},
                 "docs/LINKS.md": (f"See https://github.com/acme/math/blob/{self.a1}/tool.py and\n"
                                   f"https://github.com/acme/math/blob/{self.a1}/data.json.\n"
                                   f"Notes: https://github.com/acme/math/blob/{self.a1}/{NOTES_URL_PATH}\n"
                                   f"Foreign: https://github.com/other/math/blob/{'f' * 40}/nope.py\n"),
                 "tests/fixtures/refs.json": [{"repository": "math", "commit": "a" * 40, "path": "Thm.lean"}]}
        if opt.get("bad_json"):
            files["broken.json"] = "{not json"
        for rel, content in opt.get("main_extra", {}).items():
            files[rel] = content.format(a1=self.a1, a1s=self.a1[:8]) if isinstance(content, str) else content
        self.main_sha = commit(self.main, files, "M1")
        self.extra = None
        if opt.get("add_repo"):
            self.extra = init(self.root / "extra")
            self.extra_sha = commit(self.extra, {"refs.json": [
                {"repository": "acme/math", "commit": self.a1, "path": "missing.txt"}]}, "E1")
        self.pruned = None
        if opt.get("prune_repo"):
            self.pruned = init(self.root / "pruned")
            commit(self.pruned, {"refs.json": [{"repository": "math", "commit": "e" * 40, "path": "x"}]}, "P1")
        self.ws = init(self.root / "workspace")
        math_name = opt.get("math_name", "math")
        pins = [{"name": math_name, "url": "https://github.com/acme/math.git",
                 "pinned_sha": self.a1 if opt.get("math_pin_old") else self.a2},
                {"name": "registry", "url": "https://github.com/acme/registry.git", "pinned_sha": self.reg},
                {"name": "main", "url": "https://github.com/acme/main.git", "pinned_sha": self.main_sha}]
        if self.extra is not None:
            pins.append({"name": "extra", "url": "https://github.com/acme/extra.git", "pinned_sha": self.extra_sha})
        self.overrides = {math_name: self.math, "registry": self.registry, "main": self.main}
        if self.extra is not None:
            self.overrides["extra"] = self.extra
        if self.pruned is not None:
            self.overrides["pruned"] = self.pruned
        observed = opt.get("observed_blob") or blob_at(self.registry, self.reg, "registry.json")
        registered = [{"full_name": f"acme/{n}"} for n in ("workspace", math_name, "registry", "main")]
        if self.pruned is not None:
            registered.append({"full_name": "acme/pruned"})
        ws_doc = opt.get("ws_edit", lambda d: d)({
            "workspace": "acme/workspace",
            "canonical_inputs": {"repository_registry": {"repository": "acme/registry", "path": "registry.json",
                                                         "observed_blob": observed}},
            "repositories": registered})
        ws_files = {"pins.json": opt.get("pins_doc") or {"schema": "ulw.pins.v1", "repos": pins}}
        if ws_doc is not None:
            ws_files["workspace/repositories.json"] = ws_doc
        commit(self.ws, ws_files, "W1")

    def crawl(self, *flags, overrides=None):
        args = ["--workspace", str(self.ws), "--offline"] + list(flags)
        for name, path in (self.overrides if overrides is None else overrides).items():
            args += ["--repo", f"{name}={path}"]
        out = self.root / "report.json"
        code = rc.main(args + ["--json", str(out)])
        report = json.loads(out.read_text()) if out.exists() else None
        if out.exists():
            out.unlink()
        return code, report


def classes(report):
    return [f["class"] for f in report["findings"]]


def silence(fn):
    """Run fn with stdout and stderr captured, so test output stays readable."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn()


class Extractors(unittest.TestCase):
    def test_nested_block_and_line_comments_are_removed_but_strings_kept(self):
        text = 'a /- x /- y -/ z -/ b -- c\n"--not a comment" d'
        out = rc.strip_lean_comments(text)
        self.assertNotIn("x", out)
        self.assertNotIn("z", out)
        self.assertTrue(out.splitlines()[0].rstrip().endswith("b"))
        self.assertIn('"--not a comment"', out)
        self.assertEqual(out.count("\n"), text.count("\n"))

    def test_lean_statements_with_namespace_multiline_and_attributes(self):
        stmts = rc.lean_statements(LEAN_A1 + "@[simp] private theorem q : 2 = 2 := rfl\nexample : 3 = 3 := rfl\n")
        self.assertEqual(stmts["Demo.foo"], "theorem : 1 + 1 = 2")
        self.assertEqual(stmts["Demo.bar"], "theorem (n : Nat) : n + 0 = n")
        self.assertNotIn("Demo.fake", stmts)
        self.assertEqual(stmts["q"], "theorem : 2 = 2")
        self.assertIn("<example#1>", stmts)

    def test_python_signatures(self):
        sigs = rc.python_signatures(PY_A2 + "async def g(a) -> int:\n    return a\n")
        self.assertEqual(sigs["f"], "def f(x, y=2, *, z=0)")
        self.assertEqual(sigs["K.m"], "def m(self)")
        self.assertEqual(sigs["K"], "class K()")
        self.assertEqual(sigs["g"], "async def g(a) -> int")

    def test_byte_identical_values_are_validated(self):
        out, findings = [], []
        rc.json_references([{"repository": "math", "path": "p", "byte_identical_at_x": "zzz"},
                            {"repository": "math", "path": "p", "blob": "b" * 40, "byte_identical_at_x": "a" * 40,
                             "byte_identical_at_y": "nope"},
                            {"repository": "math", "path": "p", "commit": "c" * 40, "byte_identical_at_x": "a" * 40}],
                           "s", "f.json", out, findings)
        self.assertEqual([(r.commit, r.blobs) for r in out], [(None, ("b" * 40,)), ("a" * 40, ("b" * 40,)), ("c" * 40, ())])
        self.assertEqual([f["detail"] for f in findings],
                         ["byte_identical_at_x = 'zzz' is not a valid value",
                          "byte_identical_at_y = 'nope' is not a valid value",
                          "byte_identical_at_x claims byte identity, but there is no blob, sha256 or bytes to compare"])

    def test_unterminated_comment_is_unparseable_and_its_tail_scanned(self):
        c = "c" * 40
        text = '{"a": 1}\n/* unterminated https://github.com/acme/r/blob/' + c + '/tail.md\n'
        refs, findings = [], []
        rc.file_references(text, "cfg.json", "s", {"acme"}, refs, findings)
        self.assertEqual(([r.path for r in refs], [f["class"] for f in findings]), (["tail.md"], ["UNPARSEABLE"]))
        with self.assertRaises(ValueError):
            rc.strip_json_comments('{"a": 1} /* open')

    def test_json_shape_walks_every_array_element(self):
        self.assertEqual(rc.json_shape([1] * 60 + ["x"])["[]"], "number|string")

    def test_json_shape_and_diff(self):
        old, new = rc.json_shape(JSON_A1), rc.json_shape(JSON_A2)
        self.assertEqual(old["b.c"], "string")
        self.assertEqual(new["b.c"], "number")
        self.assertEqual(rc.diff_maps(old, new), {"removed": [], "added": ["d"], "changed": ["b.c"]})

    def test_records_with_an_identity_are_references_and_malformed_values_are_reported(self):
        out, findings = [], []
        rc.json_references({"x": [{"repository": "m", "path": "p", "commit": "a" * 40},
                                  {"repository": "m", "path": "p"},
                                  {"repository": "m", "path": "p", "bytes": True},
                                  {"repo": "m", "path": "q", "sha256": "b" * 64, "bytes": 3},
                                  {"repository": "m", "path": ""},
                                  {"name": "m", "path": "r"}]}, "s", "f.json", out, findings)
        self.assertEqual([(r.path, r.commit, r.sha256s, r.sizes) for r in out],
                         [("p", "a" * 40, (), ()), ("q", None, ("b" * 64,), (3,))])
        self.assertEqual([(f["class"], f["detail"]) for f in findings],
                         [("MALFORMED_IDENTITY", "bytes = 'True' is not a valid value")])

    def test_record_aliases_commits_and_refs(self):
        out, findings = [], []
        rc.json_references([{"repository": "acme/math", "repo": "registry", "path": "p", "commit": "a" * 40},
                            {"repository": "math", "path": "p", "commit": "a" * 40, "head": "b" * 40},
                            {"repository": "math", "path": "p", "commit": "abcdef12"},
                            {"repository": "math", "path": "p", "ref": "main"},
                            {"repository": "math", "path": "p", "ref": "not a ref!"}], "s", "f.json", out, findings)
        self.assertEqual([(r.path, r.commit) for r in out],
                         [("p", "a" * 40), ("p", "b" * 40), ("p", "abcdef12"), ("p", None)])
        self.assertEqual([f["class"] for f in findings], ["ALIAS_CONFLICT", "MALFORMED_IDENTITY"])

    def test_json_with_comments_and_trailing_commas_is_read(self):
        c = "c" * 40
        text = ('{\n // a comment, "quoted", with https://github.com/acme/r/blob/' + c + '/in-comment.md\n'
                ' "url": "https://github.com/acme/r/blob/' + c + '/a//b.md", /* block */\n "list": [1, 2,],\n}\n')
        refs, findings = [], []
        rc.file_references(text, "cfg.json", "s", {"acme"}, refs, findings)
        self.assertEqual(([r.path for r in refs], [f["class"] for f in findings]), (["a//b.md"], ["JSON_WITH_COMMENTS"]))
        self.assertEqual(json.loads(rc.strip_json_comments('{"a": "x // y, ]", "b": [1,\n ],}')), {"a": "x // y, ]", "b": [1]})

    def test_percent_encoded_url_paths_are_decoded(self):
        out = []
        rc.text_references(f"https://github.com/acme/r/blob/{'c' * 40}/a%20b%E2%80%94c.md", "s", "f.md", {"acme"}, out)
        self.assertEqual([r.path for r in out], ["a b\u2014c.md"])

    def test_url_paths_keep_balanced_parentheses_and_stop_at_backslashes(self):
        out, c = [], "c" * 40
        rc.text_references(f"[a](https://github.com/acme/r/blob/{c}/x%20(1)/y.md) and "
                           f"https://github.com/acme/r/blob/{c}/p.md\\n", "s", "f.md", {"acme"}, out)
        self.assertEqual([r.path for r in out], ["x (1)/y.md", "p.md"])

    def test_json_urls_are_matched_in_decoded_strings(self):
        c = "c" * 40
        refs, findings = [], []
        escaped = json.dumps({"log": f"GET https://github.com/acme/r/blob/{c}/a/b.md\n"}).replace("/", "\\/")
        rc.file_references(escaped, "f.json", "s", {"acme"}, refs, findings)
        self.assertEqual(([r.path for r in refs], findings), (["a/b.md"], []))
        refs, findings = [], []
        rc.file_references(f"{{broken https://github.com/acme/r/blob/{c}/z.md", "g.json", "s", {"acme"}, refs, findings)
        self.assertEqual(([r.path for r in refs], [f["class"] for f in findings]), (["z.md"], ["UNPARSEABLE"]))

    def test_owner_and_repository_names_match_case_insensitively(self):
        out = []
        rc.text_references(f"https://github.com/ACME/r/blob/{'c' * 40}/a.md", "s", "f.md", {"acme"}, out)
        self.assertEqual([r.repo for r in out], ["ACME/r"])
        node = rc.Node("math", "acme/math", "", "c" * 40, True, True)
        self.assertEqual(rc.resolve(rc.Reference("s", "f", "Acme/MATH", "p"), {"math": node}, {"acme"}), (node, False))
        self.assertEqual(rc.resolve(rc.Reference("s", "f", "other/MATH", "p"), {"math": node}, {"acme"}), (None, True))

    def test_fixture_paths_are_directory_segments_only(self):
        self.assertTrue(rc.is_fixture("tests/fixtures/a.json"))
        self.assertTrue(rc.is_fixture("pkg/examples/b.md"))
        self.assertFalse(rc.is_fixture("tests.json"))
        self.assertFalse(rc.is_fixture("docs/examples"))
        self.assertFalse(rc.is_fixture("docs/testing/c.md"))

    def test_text_references_filter_owner_and_strip_punctuation(self):
        out = []
        rc.text_references(f"x https://github.com/acme/r/blob/{'c' * 40}/a/b.md. and "
                           f"https://github.com/zzz/r/blob/{'c' * 40}/n.md", "s", "f.md", {"acme"}, out)
        self.assertEqual([(r.repo, r.path) for r in out], [("acme/r", "a/b.md")])


class Crawl(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def fed(self, **opt):
        return Federation(self.tmp.name, **opt)

    def test_clean_federation_reports_interface_drift_without_blocking(self):
        fed = self.fed()
        code, report = silence(fed.crawl)
        self.assertEqual(code, 0)
        self.assertEqual(report["blocking"], 0)
        changed = {f["path"]: f["interface"] for f in report["findings"] if f["class"] == "FROZEN_CHANGED"}
        self.assertEqual(set(changed), {"Thm.lean", "tool.py", "data.json"})
        self.assertEqual(changed["Thm.lean"]["changed"], ["Demo.bar"])
        self.assertEqual(changed["Thm.lean"]["added"], ["Demo.baz"])
        self.assertEqual(changed["tool.py"]["changed"], ["f"])
        self.assertEqual(changed["data.json"]["changed"], ["b.c"])
        self.assertNotIn("UNKNOWN_REPOSITORY", classes(report))  # the foreign-owner URL is ignored
        # The foreign record is counted, and never checked against acme/math, which shares its short name.
        self.assertEqual([f["repo"] for f in report["findings"] if f["class"] == "FOREIGN_REPOSITORY"], ["other/math"])

    def test_strict_blocks_on_interface_drift(self):
        code, report = silence(lambda: self.fed().crawl("--strict"))
        self.assertEqual(code, 1)
        self.assertEqual(report["blocking"], 3)

    def test_identity_failures_block(self):
        cases = {"source_blob": "0" * 40, "artifact_sha256": "0" * 64, "artifact_bytes": 999}
        for key, value in cases.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                code, report = silence(lambda: Federation(tmp, **{key: value}).crawl())
                self.assertEqual(code, 1)
                self.assertIn("IDENTITY_MISMATCH", classes(report))

    def test_missing_path_commit_and_repository_block(self):
        cases = {"source_path": "MISSING_PATH", "source_commit": "MISSING_COMMIT", "source_repo": "UNKNOWN_REPOSITORY"}
        values = {"source_path": "Nope.lean", "source_commit": "f" * 40, "source_repo": "acme/ghost"}
        for key, cls in cases.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                opt = {key: values[key]}
                if key != "source_repo":
                    opt["source_blob"] = "1" * 40
                code, report = silence(lambda: Federation(tmp, **opt).crawl())
                self.assertEqual(code, 1)
                self.assertIn(cls, classes(report))

    def test_excerpt_references_are_checked_against_their_lines(self):
        tail = "first line\nsecond line\nthird line\n"
        cases = {"sha256": ({"sha256": "0" * 64}, "sha256 in lines 3-4 differs"),
                 "text": ({"text": "first line\nsecond lime\n"}, "text of lines 3-4 differs"),
                 "bytes": ({"bytes": 5}, "bytes 5 recorded, 23 found in lines 3-4"),
                 "range": ({"line_end": 9, "text": tail, "sha256": hashlib.sha256(tail.encode()).hexdigest()},
                           "lines 3-9 outside 1-5")}
        for name, (change, detail) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                code, report = silence(lambda: Federation(tmp, excerpt=change).crawl())
                self.assertEqual(code, 1)
                rows = [f for f in report["findings"] if f["class"] == "IDENTITY_MISMATCH"]
                self.assertEqual([r["path"] for r in rows], [NOTES_PATH])
                self.assertIn(detail, rows[0]["detail"])

    def test_every_recorded_identity_key_is_checked(self):
        cases = [{"source_blob_key": k, "source_blob": "0" * 40}
                 for k in ("blob_sha", "blob_sha1", "git_blob_sha", "git_blob_sha1")]
        cases += [{"source_extra": {"blob_sha": "0" * 40}},  # a second, wrong value beside the correct blob
                  {"artifact_sha_key": "raw_sha256", "artifact_sha256": "0" * 64},
                  {"artifact_bytes_key": "byte_length", "artifact_bytes": 999}]
        for opt in cases:
            with self.subTest(opt=opt), tempfile.TemporaryDirectory() as tmp:
                code, report = silence(lambda: Federation(tmp, **opt).crawl())
                self.assertEqual(code, 1)
                self.assertIn("IDENTITY_MISMATCH", classes(report))

    def test_every_commit_key_pins_the_reference(self):
        for key in ("cut", "head", "at_commit"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                fed = Federation(tmp, source_commit_key=key, source_commit="f" * 40, source_blob="1" * 40)
                code, report = silence(fed.crawl)
                self.assertEqual(code, 1)
                self.assertIn("MISSING_COMMIT", classes(report))

    def test_byte_identical_claims_are_checked_at_their_commit(self):
        fed = self.fed(source_also_at=True)
        code, report = silence(fed.crawl)
        self.assertEqual(code, 1)
        rows = [f for f in report["findings"] if f["class"] == "IDENTITY_MISMATCH"]
        self.assertEqual([(r["path"], r["commit"]) for r in rows], [("Thm.lean", fed.a2)])

    def test_every_text_file_is_read_whatever_its_extension(self):
        extra = {"docs/page.html": '<a href="https://github.com/acme/math/blob/{a1}/gone-html.md">x</a>\n',
                 "notes/README": "see https://github.com/acme/math/blob/{a1}/gone-plain.md\n",
                 "logs/records.jsonl": ('{{"repository": "math", "commit": "{a1}", "path": "gone-jsonl.md"}}\n'
                                        '{{not json https://github.com/acme/math/blob/{a1}/gone-raw.md\n')}
        code, report = silence(lambda: self.fed(main_extra=extra).crawl())
        self.assertEqual(code, 1)
        missing = sorted((f["source"], f["path"]) for f in report["findings"] if f["class"] == "MISSING_PATH")
        self.assertEqual(missing, [("main:docs/page.html", "gone-html.md"), ("main:logs/records.jsonl", "gone-jsonl.md"),
                                   ("main:logs/records.jsonl", "gone-raw.md"), ("main:notes/README", "gone-plain.md")])
        self.assertEqual([f["path"] for f in report["findings"] if f["class"] == "UNPARSEABLE"], ["logs/records.jsonl"])

    def test_binary_files_are_counted_and_oversize_files_reported(self):
        fed = self.fed(main_extra={"img/logo.png": b"\x89PNG\r\n\x1a\n\xff\xfe\x00"})
        code, report = silence(fed.crawl)
        self.assertEqual((code, report["binary_files_not_read"]), (0, {"main": 1}))
        self.assertEqual(report["binary_paths"], {"main": ["img/logo.png"]})
        self.assertIn("Binary files not read: 1.", rc.render(report))
        limit = rc.MAX_TEXT_BYTES
        rc.MAX_TEXT_BYTES = 40
        try:
            code, report = silence(fed.crawl)
        finally:
            rc.MAX_TEXT_BYTES = limit
        unread = {(f["repo"], f["path"]) for f in report["findings"] if f["class"] == "UNREAD"}
        self.assertEqual(code, 1)  # an unread file makes the census incomplete
        self.assertIn(("math", "Thm.lean"), unread)
        self.assertNotIn(("math", "README.md"), unread)  # 5 bytes, under the limit

    def test_fixture_files_are_skipped_and_counted_unless_included(self):
        fed = self.fed()
        code, report = silence(fed.crawl)
        self.assertEqual((code, report["skipped_fixture_files"]), (0, {"main": 1}))
        self.assertEqual(report["skipped_fixture_paths"], {"main": ["tests/fixtures/refs.json"]})
        self.assertIn("Fixture files skipped: 1 ", rc.render(report))
        code, report = silence(lambda: fed.crawl("--include-fixtures"))
        self.assertEqual((code, report["skipped_fixture_files"]), (1, {}))
        self.assertEqual([f["source"] for f in report["findings"] if f["class"] == "MISSING_COMMIT"],
                         ["main:tests/fixtures/refs.json"])

    def test_live_reference_drift_blocks(self):
        fed = self.fed()
        old = blob_at(fed.registry, fed.reg_old, "registry.json")
        with tempfile.TemporaryDirectory() as tmp:
            code, report = silence(lambda: Federation(tmp, observed_blob=old).crawl())
        self.assertEqual(code, 1)
        self.assertIn("LIVE_DRIFT", classes(report))

    def test_rename_adapts_traversal_and_reports_stale_names(self):
        code, report = silence(lambda: self.fed(math_name="maths").crawl())
        self.assertEqual(code, 1)  # references still using the old name "math" no longer resolve
        self.assertIn("UNKNOWN_REPOSITORY", classes(report))
        self.assertIn("maths", [n["name"] for n in report["topology"]])

    def test_stale_registry_entry_is_reported_without_halting(self):
        # The pins rename math to maths, but the repository registry still lists acme/math.
        code, report = silence(lambda: self.fed(math_name="maths", registry_math_name="math").crawl())
        self.assertEqual(code, 1)
        self.assertIn("UNREACHABLE_REPOSITORY", classes(report))
        # The crawl continued: references from both main and the registry were still collected.
        sources = {f.get("source", "").split(":", 1)[0] for f in report["findings"] if f["class"] == "TARGET_UNREACHABLE"}
        self.assertEqual(sources, {"main", "registry"})
        self.assertTrue(any(n["name"] == "maths" for n in report["topology"]))

    def test_added_repository_is_crawled(self):
        code, report = silence(lambda: self.fed(add_repo=True).crawl())
        self.assertEqual(code, 1)
        self.assertTrue(any(f["class"] == "MISSING_PATH" and f.get("source", "").startswith("extra:")
                            for f in report["findings"]))

    def test_registered_but_unpinned_repository_is_crawled_and_reported(self):
        code, report = silence(lambda: self.fed(prune_repo=True).crawl())
        self.assertIn("TOPOLOGY_UNPINNED", classes(report))
        self.assertTrue(any(f["class"] == "MISSING_COMMIT" and f.get("source", "").startswith("pruned:")
                            for f in report["findings"]))
        self.assertEqual(code, 1)

    def test_unparseable_json_blocks_as_an_incomplete_census(self):
        code, report = silence(lambda: self.fed(bad_json=True).crawl())
        self.assertEqual((code, report["blocking"]), (1, 1))
        self.assertIn("UNPARSEABLE", classes(report))

    def test_named_refs_and_abbreviated_commits_are_checked_and_path_only_records_are_not_references(self):
        extra = {"refs/live.json": ('[{{"repository": "math", "path": "Thm.lean", "ref": "main"}},'
                                    ' {{"repository": "math", "path": "gone-named.md", "ref": "main"}},'
                                    ' {{"repo": "math", "path": "cards/local-to-main.json", "card": "C-1"}},'
                                    ' {{"repository": "math", "path": "Thm.lean", "commit": "{a1s}"}}]')}
        fed = self.fed(main_extra=extra)
        code, report = silence(fed.crawl)
        rows = [(f["class"], f["path"]) for f in report["findings"] if f.get("source") == "main:refs/live.json"]
        # The card names the repository it concerns; its path is local, so it is not checked against math.
        self.assertEqual(rows, [("LIVE_DRIFT", "gone-named.md"), ("FROZEN_CHANGED", "Thm.lean")])
        self.assertEqual(code, 1)

    def test_file_identity_on_a_directory_is_a_mismatch(self):
        extra = {"refs/dir.json": '[{{"repository": "math", "commit": "{a1}", "path": "notes", "sha256": "' + "c" * 64 + '"}}]'}
        code, report = silence(lambda: self.fed(main_extra=extra).crawl())
        rows = [f for f in report["findings"] if f.get("source") == "main:refs/dir.json"]
        self.assertEqual([(f["class"], f["detail"]) for f in rows],
                         [("IDENTITY_MISMATCH", "a file identity is recorded, but the path is a tree")])
        self.assertEqual(code, 1)

    def test_template_files_are_synthetic(self):
        extra = {"docs/TEMPLATE_review.json": '{{"repository": "math", "path": "Thm.lean", "sha256": "REPLACE_ME"}}'}
        fed = self.fed(main_extra=extra)
        code, report = silence(fed.crawl)
        self.assertEqual((code, report["skipped_fixture_paths"]["main"]),
                         (0, ["docs/TEMPLATE_review.json", "tests/fixtures/refs.json"]))
        code, report = silence(lambda: fed.crawl("--include-fixtures"))
        self.assertIn("MALFORMED_IDENTITY", classes(report))

    def test_registered_unpinned_repository_resolves_through_the_cache(self):
        fed = self.fed(prune_repo=True)
        cache = Path(self.tmp.name) / "cache"
        cache.mkdir()
        overrides = {k: v for k, v in fed.overrides.items() if k != "pruned"}
        # With no mirror, the clone URL derived from the registered name is what --offline refuses.
        code, report = silence(lambda: fed.crawl("--cache", str(cache), overrides=overrides))
        rows = [f["detail"] for f in report["findings"] if f["class"] == "UNREACHABLE_REPOSITORY" and f["repo"] == "pruned"]
        self.assertEqual(len(rows), 1)
        self.assertIn("--offline forbids cloning", rows[0])
        subprocess.run(["git", "clone", "-q", "--bare", str(fed.pruned), str(cache / "pruned.git")], check=True)
        code, report = silence(lambda: fed.crawl("--cache", str(cache), overrides=overrides))
        self.assertIn("TOPOLOGY_UNPINNED", classes(report))
        self.assertTrue(any(f["class"] == "MISSING_COMMIT" and f.get("source", "").startswith("pruned:")
                            for f in report["findings"]))

    def test_pin_skew_reported_and_optionally_blocking(self):
        fed = self.fed(math_pin_old=True)
        code, report = silence(fed.crawl)
        self.assertIn("PIN_SKEW", classes(report))
        self.assertEqual(code, 0)
        code, report = silence(lambda: fed.crawl("--require-current-pins"))
        self.assertEqual(code, 1)

    def test_unreachable_pinned_repository_blocks_but_does_not_halt(self):
        fed = self.fed()
        overrides = dict(fed.overrides)
        del overrides["main"]
        code, report = silence(lambda: fed.crawl(overrides=overrides))
        self.assertEqual(code, 1)
        self.assertIn("UNREACHABLE_REPOSITORY", classes(report))
        # The registry and the workspace were still crawled: the registry's artifact reference resolved.
        self.assertGreater(report["references"], 0)
        self.assertFalse(any(f.get("source", "").startswith("main:") for f in report["findings"]))

    def test_missing_pinned_commit_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            fed = Federation(tmp)
            pins = json.loads(git(fed.ws, "show", "HEAD:pins.json"))
            pins["repos"][0]["pinned_sha"] = "d" * 40
            commit(fed.ws, {"pins.json": pins}, "bad pin")
            code, report = silence(fed.crawl)
        self.assertEqual(code, 1)
        self.assertIn("MISSING_PIN", classes(report))

    def test_candidate_head_is_crawled_in_place_of_the_pin(self):
        fed = self.fed()
        cand = commit(fed.main, {"cand.json": [{"repository": "math", "commit": fed.a1, "path": "gone-cand.md"}]}, "M2")
        commit(fed.main, {"later.md": "a later commit, so the tip is neither the pin nor the candidate\n"}, "M3")
        code, report = silence(fed.crawl)
        self.assertEqual(code, 0)  # main is crawled at its pin, where cand.json does not exist
        self.assertIn("PIN_SKEW", classes(report))
        code, report = silence(lambda: fed.crawl("--at", f"main={cand}"))
        self.assertEqual(code, 1)
        self.assertEqual([f["source"] for f in report["findings"] if f["class"] == "MISSING_PATH"], ["main:cand.json"])
        self.assertIn("CANDIDATE", classes(report))
        self.assertNotIn("PIN_SKEW", classes(report))
        self.assertTrue([n for n in report["topology"] if n["name"] == "main" and n["commit"] == cand and n["candidate"]])
        # A candidate that changes a file another repository depends on live is caught too.
        reg = commit(fed.registry, {"registry.json": {"repositories": {}, "artifacts": [], "changed": True}}, "R3")
        code, report = silence(lambda: fed.crawl("--at", f"registry={reg}"))
        self.assertEqual(code, 1)
        self.assertTrue(any(f["class"] == "LIVE_DRIFT" and f["source"].startswith("workspace:") for f in report["findings"]))

    def test_candidate_revision_must_exist_and_name_a_crawled_repository(self):
        fed = self.fed()
        code, report = silence(lambda: fed.crawl("--at", "main=no-such-branch"))
        self.assertEqual(code, 1)
        self.assertEqual([f["detail"] for f in report["findings"] if f["class"] == "MISSING_PIN"],
                         ["candidate revision 'no-such-branch' not found"])
        self.assertEqual(silence(lambda: fed.crawl("--at", "ghost=HEAD")), (2, None))

    def test_incomplete_topology_inputs_block(self):
        def without(key):
            return lambda d: {k: v for k, v in d.items() if k != key}

        def canonical(**change):
            def edit(d):
                d["canonical_inputs"]["repository_registry"].update(change)
                return d
            return edit

        def extra_entry(d):
            d["repositories"].append({"name": "no-full-name"})
            return d

        cases = {"workspace registry missing": ({"ws_edit": lambda d: None}, "the workspace registry is missing"),
                 "workspace registry not an object": ({"ws_edit": lambda d: [d]}, "is not a JSON object"),
                 "no repositories list": ({"ws_edit": without("repositories")}, "has no repositories list"),
                 "entry without full_name": ({"ws_edit": extra_entry}, "has no full_name"),
                 "canonical input absent": ({"ws_edit": without("canonical_inputs")}, "does not name a repository and a path"),
                 "registry host not in topology": ({"ws_edit": canonical(repository="acme/ghost")}, "is not in the topology"),
                 "registry document missing": ({"ws_edit": canonical(path="nope.json")}, "the repository registry is missing at"),
                 "registry map missing": ({"registry_edit": without("repositories")}, "has no repositories map")}
        for name, (opt, detail) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                code, report = silence(lambda: Federation(tmp, **opt).crawl())
                rows = [f["detail"] for f in report["findings"] if f["class"] == "TOPOLOGY_INCOMPLETE"]
                self.assertEqual(code, 1)
                self.assertTrue(any(detail in row for row in rows), rows)
        with self.subTest(case="registry host unreachable"), tempfile.TemporaryDirectory() as tmp:
            fed = Federation(tmp)
            code, report = silence(lambda: fed.crawl(overrides={k: v for k, v in fed.overrides.items() if k != "registry"}))
            self.assertIn("the registry host registry is unreachable",
                          [f["detail"] for f in report["findings"] if f["class"] == "TOPOLOGY_INCOMPLETE"])
        with self.subTest(case="registry host commit unavailable"), tempfile.TemporaryDirectory() as tmp:
            fed = Federation(tmp)
            pins = json.loads(git(fed.ws, "show", "HEAD:pins.json"))
            pins["repos"][1]["pinned_sha"] = "d" * 40
            commit(fed.ws, {"pins.json": pins}, "bad registry pin")
            code, report = silence(fed.crawl)
            self.assertIn(f"the registry host's commit {'d' * 40} is not available",
                          [f["detail"] for f in report["findings"] if f["class"] == "TOPOLOGY_INCOMPLETE"])

    def test_unreadable_workspace_pins_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Federation(tmp, pins_doc={"schema": "wrong", "repos": []})
            self.assertEqual(silence(bad.crawl), (2, None))


if __name__ == "__main__":
    unittest.main()
