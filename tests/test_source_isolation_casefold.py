"""External CLI controls on a real, writable casefold filesystem.

Run only in the disposable hosted casefold lane. The lane must provide
SOURCE_ISOLATION_CASEFOLD_ROOT on its prepared casefold filesystem. Missing
preparation, read-only fixtures, or absent real directory/file aliases are
failures, never skips or successful product refusals. Only the three pinned
source files are copied into disposable TEST fixtures; no project code is
imported by this module. Fixture results confer no scientific status.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "tools/proof_graph_export.py"
PINNED_SOURCE = ROOT / "docs/site/dependency-source"
CASEFOLD_ENV = "SOURCE_ISOLATION_CASEFOLD_ROOT"
SOURCE_SHA256 = {
    "GRAPH.json": "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09",
    "hard_gate.py": "a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8",
    "PROVENANCE.json": "c54cd92d93b2cbc650e024f30a8762fbc277d847580245053bf0c415265872c5",
}
ISOLATION_STDERR = (
    "graph export refused: output directory must be outside the immutable source directory\n"
)
NOFOLLOW_REASON = "graph export refused: symlink or invalid directory component in path "
FILE_LIMIT = 1024 * 1024


# Observe real write attempts without replacing filesystem APIs or the CLI.
# The audit descriptor is opened before the hook; recording via os.write does
# not reopen files or recursively record an audited create/write open.
AUDIT_CLI = r"""
import json
import os
import runpy
import sys

audit_path, tool = sys.argv[1:3]
sys.argv = sys.argv[2:]
audit = os.open(audit_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)

def observe(event, arguments):
    is_write = event in ('os.mkdir', 'os.rename', 'os.remove', 'os.rmdir')
    if event == 'open':
        flags = arguments[2]
        is_write = type(flags) is int and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
    if is_write:
        record = {'event': event, 'arguments': repr(arguments)}
        os.write(audit, (json.dumps(record, sort_keys=True) + '\n').encode('utf-8'))

sys.addaudithook(observe)
try:
    runpy.run_path(tool, run_name='__main__')
finally:
    os.close(audit)
"""


@contextmanager
def anchored_directory(path):
    """Keep '..' visible and refuse symlinks in every actual component."""
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            following = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        yield descriptor
    finally:
        os.close(descriptor)


def identity(info):
    return info.st_dev, info.st_ino


def directory_identity(path):
    with anchored_directory(path) as descriptor:
        return identity(os.fstat(descriptor))


def read_regular(descriptor, name):
    leaf = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor,
    )
    with os.fdopen(leaf, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > FILE_LIMIT:
            raise ValueError("TEST source member must be a bounded regular file: " + name)
        raw = stream.read(FILE_LIMIT + 1)
        if len(raw) != info.st_size or len(raw) > FILE_LIMIT:
            raise ValueError("TEST source member changed length: " + name)
        return identity(info), raw


def directory_inventory(descriptor, relative="."):
    entries = tuple(sorted(os.listdir(descriptor)))
    inventory = {relative: {"identity": identity(os.fstat(descriptor)), "entries": entries}}
    for name in entries:
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            try:
                child_relative = name if relative == "." else relative + "/" + name
                inventory.update(directory_inventory(child, child_relative))
            finally:
                os.close(child)
    return inventory


class SourceIsolationCasefoldContract(unittest.TestCase):
    def setUp(self):
        self.assertEqual(sys.platform, "linux", "casefold lane requires its hosted Linux filesystem")
        self.assertTrue(EXPORTER.is_file(), "required exporter CLI tools/proof_graph_export.py is absent")
        configured = os.environ.get(CASEFOLD_ENV)
        self.assertTrue(configured, "mandatory casefold fixture environment is absent: " + CASEFOLD_ENV)
        casefold_root = Path(configured)
        self.assertTrue(casefold_root.is_absolute(), "casefold fixture root must be an absolute path")
        # An unavailable or read-only mount fails here, before any exporter call.
        with anchored_directory(casefold_root):
            pass
        temporary = tempfile.TemporaryDirectory(prefix="TEST-source-isolation-", dir=casefold_root)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "InputSource"
        self.source.mkdir()
        for name, expected_digest in SOURCE_SHA256.items():
            raw = (PINNED_SOURCE / name).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_digest, "pinned fixture identity: " + name)
            shutil.copyfile(PINNED_SOURCE / name, self.source / name)
        self.source_alias = self.root / self.source.name.swapcase()
        self.assert_casefold_fixture()
        with anchored_directory(self.source) as descriptor:
            self.assertEqual(tuple(sorted(os.listdir(descriptor))), tuple(sorted(SOURCE_SHA256)))

    def assert_casefold_fixture(self):
        self.assertNotEqual(str(self.source), str(self.source_alias))
        self.assertEqual(
            directory_identity(self.source), directory_identity(self.source_alias),
            "source and case alias must name the same actual directory before the CLI runs",
        )
        with anchored_directory(self.source) as descriptor:
            original_identity, original = read_regular(descriptor, "GRAPH.json")
            alias_identity, alias = read_regular(descriptor, "graph.json")
        self.assertEqual(original_identity, alias_identity, "GRAPH.json/graph.json must be actual file aliases")
        self.assertEqual(original, alias)
        self.assertEqual(hashlib.sha256(original).hexdigest(), SOURCE_SHA256["GRAPH.json"])

    def source_state(self):
        with anchored_directory(self.source) as descriptor:
            members = {name: read_regular(descriptor, name) for name in SOURCE_SHA256}
            return {
                "directories": directory_inventory(descriptor),
                "file_identities": {name: value[0] for name, value in members.items()},
                "raw_bytes": {name: value[1] for name, value in members.items()},
            }

    def assert_source_preserved(self, before):
        after = self.source_state()
        # Separate subtests retain evidence for bytes, identities and entries
        # even if an accepted bad export violates more than one invariant.
        for field in ("directories", "file_identities", "raw_bytes"):
            with self.subTest(source_invariant=field):
                self.assertEqual(after[field], before[field], "export changed immutable TEST source " + field)

    def run_export(self, *, source=None, output, audit=False):
        self.assertTrue(EXPORTER.is_file(), "required exporter CLI tools/proof_graph_export.py is absent")
        self.assert_casefold_fixture()
        source = self.source if source is None else source
        flags = ["-B", "-S"]
        if sys.flags.optimize:
            flags.append("-" + "O" * sys.flags.optimize)
        env = os.environ.copy()
        env.pop("PYTHONOPTIMIZE", None)
        env.update({"PATH": "", "PYTHONDONTWRITEBYTECODE": "1"})
        arguments = [str(EXPORTER), "--source-dir", str(source), "--output", str(output)]
        if audit:
            self.audit_path = self.root / "TEST-cli-write-audit.jsonl"
            command = [sys.executable, *flags, "-c", AUDIT_CLI, str(self.audit_path), *arguments]
        else:
            command = [sys.executable, *flags, *arguments]
        try:
            return subprocess.run(
                command,
                cwd=self.root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=20, check=False,
            )
        except subprocess.TimeoutExpired:
            self.fail("exporter CLI did not finish within 20 seconds")

    def assert_isolation_refused(self, output):
        before = self.source_state()
        result = self.run_export(output=output, audit=True)
        with self.subTest(cli="exact immutable-source refusal"):
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, ISOLATION_STDERR)
        with self.subTest(cli="refusal before every write attempt"):
            self.assertTrue(self.audit_path.is_file(), "CLI audit did not run")
            events = [json.loads(line) for line in self.audit_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events, [], "source isolation refusal attempted mkdir, temporary output, or promotion")
        self.assert_source_preserved(before)

    def assert_nofollow_refused(self, *, source=None, output):
        before = self.source_state()
        result = self.run_export(source=source, output=output)
        with self.subTest(cli="nofollow refusal"):
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertTrue(result.stderr.startswith(NOFOLLOW_REASON), result.stderr)
            self.assertNotIn("Read-only file system", result.stderr)
        self.assert_source_preserved(before)

    def test_case_alias_same_source_output_is_refused_without_source_mutation(self):
        self.assert_isolation_refused(self.source_alias)

    def test_case_alias_existing_child_is_refused_before_temporary_output_or_promotion(self):
        child = self.source / "ExistingChild"
        child.mkdir()
        alias = self.source_alias / child.name.swapcase()
        self.assertEqual(directory_identity(child), directory_identity(alias))
        self.assert_isolation_refused(alias)
        self.assertEqual(tuple(child.iterdir()), ())

    def test_case_alias_new_child_is_refused_before_any_directory_creation(self):
        child = self.source / "NewChild"
        output = self.source_alias / child.name.swapcase() / "NestedExport"
        self.assertFalse(child.exists())
        self.assert_isolation_refused(output)
        self.assertFalse(child.exists(), "refusal created a source child directory")

    def test_exact_source_output_is_refused_without_source_mutation(self):
        self.assert_isolation_refused(self.source)

    def test_literal_existing_child_is_refused_without_source_mutation(self):
        child = self.source / "LiteralChild"
        child.mkdir()
        self.assert_isolation_refused(child)
        self.assertEqual(tuple(child.iterdir()), ())

    def test_literal_new_child_is_refused_before_any_directory_creation(self):
        child = self.source / "LiteralNewChild"
        self.assert_isolation_refused(child / "NestedExport")
        self.assertFalse(child.exists(), "refusal created a literal source child directory")

    def test_distinct_casefold_sibling_exports_pinned_records_and_preserves_source(self):
        output = self.root / "SiblingExport"
        output.mkdir()
        output_alias = self.root / output.name.swapcase()
        self.assertEqual(directory_identity(output), directory_identity(output_alias))
        self.assertNotEqual(directory_identity(self.source), directory_identity(output_alias))
        before = self.source_state()
        result = self.run_export(output=output_alias, audit=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout, str(output_alias / "graph.json") + "\n")
        self.assertTrue(self.audit_path.is_file(), "CLI audit did not run for the success control")
        events = [json.loads(line) for line in self.audit_path.read_text(encoding="utf-8").splitlines()]
        self.assertIn("open", [event["event"] for event in events], "audit missed the real temporary write open")
        self.assertIn("os.rename", [event["event"] for event in events], "audit missed the real output promotion")
        original = json.loads(before["raw_bytes"]["GRAPH.json"])
        provenance = json.loads(before["raw_bytes"]["PROVENANCE.json"])
        with anchored_directory(output) as descriptor:
            _, raw = read_regular(descriptor, "graph.json")
            self.assertEqual(tuple(sorted(os.listdir(descriptor))), ("graph.json",))
        graph = json.loads(raw)
        self.assertEqual(len(original["nodes"]), 49)
        self.assertEqual(len(original["edges"]), 55)
        self.assertEqual(set(graph), {"schema_version", "source", "nodes", "edges", "dimensions",
                                     "scientific_effect", "scientific_status_authority"})
        self.assertIs(type(graph["schema_version"]), int)
        self.assertEqual(graph["schema_version"], 1)
        for field in ("nodes", "edges"):
            self.assertEqual(json.dumps(graph[field], sort_keys=True), json.dumps(original[field], sort_keys=True))
        self.assertEqual(graph["source"], {
            "repository": provenance["repository"], "commit": provenance["commit"],
            "captured_at": provenance["captured_at"], "graph_sha256": SOURCE_SHA256["GRAPH.json"],
            "gate_sha256": SOURCE_SHA256["hard_gate.py"],
        })
        self.assertEqual(set(graph["dimensions"]), set(original["nodes"]))
        for dimensions in graph["dimensions"].values():
            self.assertEqual(dimensions["kernel"], "not_recorded")
            self.assertEqual(dimensions["computation"], "not_recorded")
            self.assertEqual(dimensions["alignment"], "not_recorded")
        self.assertEqual(graph["scientific_effect"], "NONE")
        self.assertIs(graph["scientific_status_authority"], False)
        self.assert_source_preserved(before)

    def test_source_directory_symlink_is_refused_without_source_mutation(self):
        link = self.root / "SourceLink"
        link.symlink_to(self.source, target_is_directory=True)
        self.assertTrue(stat.S_ISLNK(os.lstat(link).st_mode))
        self.assertEqual(identity(os.stat(link)), directory_identity(self.source))
        output = self.root / "SourceLinkExport"
        self.assert_nofollow_refused(source=link, output=output)
        self.assertFalse(output.exists(), "source symlink refusal created an output directory")

    def test_output_directory_symlink_is_refused_without_source_mutation(self):
        target = self.root / "OutputTarget"
        target.mkdir()
        link = self.root / "OutputLink"
        link.symlink_to(target, target_is_directory=True)
        self.assertTrue(stat.S_ISLNK(os.lstat(link).st_mode))
        self.assertEqual(identity(os.stat(link)), directory_identity(target))
        self.assert_nofollow_refused(output=link)
        self.assertEqual(tuple(target.iterdir()), (), "output symlink refusal wrote to its target")
        self.assertTrue(stat.S_ISLNK(os.lstat(link).st_mode))

    def test_earlier_source_symlink_then_parent_component_is_refused_without_source_mutation(self):
        link = self.root / "EarlierSourceLink"
        link.symlink_to(self.source, target_is_directory=True)
        path = link / ".." / self.source.name
        self.assertIn("..", path.parts)
        self.assertTrue(stat.S_ISLNK(os.lstat(link).st_mode))
        # Following this real path reaches the source; refusing the earlier link
        # is therefore the behavior under test, not a missing-source failure.
        self.assertEqual(identity(os.stat(path)), directory_identity(self.source))
        with self.assertRaises(OSError):
            directory_identity(path)
        output = self.root / "EarlierLinkExport"
        self.assert_nofollow_refused(source=path, output=output)
        self.assertFalse(output.exists(), "earlier source symlink refusal created an output directory")


if __name__ == "__main__":
    unittest.main()
