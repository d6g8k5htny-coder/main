"""External synthetic Task1 contract; execute only in the hosted isolated lane.

Author: OpenAI/Codex architecture_memory_regression_inventory, source exposed.
Original Task1 RED/GREEN remains historical at its exact source. This successor
adds Ready-review controls; its outcomes are NOT_RUN.
Frozen-identity successor author: OpenAI/Codex rag_frozen_identity_control_author_1017.
This successor probes the real existing admitted_corpus API through the captured
CLI loader. Its original API RED is qualified at its pinned predecessor; this
controls-first admission repair successor adds prospective strict CLI/report
compatibility controls whose behavioral outcomes remain NOT_RUN. Every source/fact/oracle
is synthetic. Only the captured CLI loader and benchmark subject are loaded as
project code; no additional project helper is loaded. Same provider and
shared principal, organizational-independence credit0, scientific effect NONE.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import runpy
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools/semantic_memory_experiment.py"
FIXTURES = ROOT / "experiments/semantic_memory/fixtures"
STRATA = ("paraphrase", "deep", "confusable_revision", "exact_ref", "no_answer")
SOURCE_KEYS = {"source_id", "revision_id", "source_sha256", "extraction_sha256",
               "extractor_version", "text_sha256", "text_bytes", "content_ref",
               "observed_at", "policy_epoch", "access_state", "exclusion_state",
               "supersession_state"}
REQUEST_KEYS = {"query", "exact_source_id", "exact_ref", "task_scope", "profile"}
SENTINEL = "EVALUATOR_ONLY_6205877540_NEVER_INDEX_OR_PROMPT"

# Literal frozen expectations copied from original manifest SHA256 fc4fb7dd...
# before these findings; never derived from the product or mutated copy.
FROZEN_GENERATION_CASE_IDS = ["P01","P02","P03","P04","D01","D02","D03","D04","V01","V02","V03","V04","X01","X02","X03","X04","N01","N02","N03","N04"]
FROZEN_QUERY_ORDER = [
    {"arms":["S","L","E"],"case_id":"P01","profiles":["W","C"]},
    {"arms":["L","E","S"],"case_id":"P05","profiles":["C","W"]},
    {"arms":["S","E","L"],"case_id":"N03","profiles":["W","C"]},
    {"arms":["S","E","L"],"case_id":"X07","profiles":["C","W"]},
    {"arms":["L","S","E"],"case_id":"X02","profiles":["W","C"]},
    {"arms":["E","L","S"],"case_id":"X05","profiles":["C","W"]},
    {"arms":["S","L","E"],"case_id":"V03","profiles":["W","C"]},
    {"arms":["L","E","S"],"case_id":"V08","profiles":["C","W"]},
    {"arms":["S","L","E"],"case_id":"X03","profiles":["W","C"]},
    {"arms":["L","S","E"],"case_id":"V04","profiles":["C","W"]},
    {"arms":["L","S","E"],"case_id":"D05","profiles":["W","C"]},
    {"arms":["E","L","S"],"case_id":"N01","profiles":["C","W"]},
    {"arms":["L","S","E"],"case_id":"D02","profiles":["W","C"]},
    {"arms":["L","E","S"],"case_id":"V07","profiles":["C","W"]},
    {"arms":["S","E","L"],"case_id":"V06","profiles":["W","C"]},
    {"arms":["S","L","E"],"case_id":"X06","profiles":["C","W"]},
    {"arms":["L","S","E"],"case_id":"X04","profiles":["W","C"]},
    {"arms":["E","L","S"],"case_id":"V01","profiles":["C","W"]},
    {"arms":["S","E","L"],"case_id":"D08","profiles":["W","C"]},
    {"arms":["E","S","L"],"case_id":"N04","profiles":["C","W"]},
    {"arms":["S","E","L"],"case_id":"N05","profiles":["W","C"]},
    {"arms":["E","L","S"],"case_id":"D01","profiles":["C","W"]},
    {"arms":["L","S","E"],"case_id":"N07","profiles":["W","C"]},
    {"arms":["L","E","S"],"case_id":"P08","profiles":["C","W"]},
    {"arms":["E","S","L"],"case_id":"X08","profiles":["W","C"]},
    {"arms":["S","E","L"],"case_id":"D04","profiles":["C","W"]},
    {"arms":["L","S","E"],"case_id":"X01","profiles":["W","C"]},
    {"arms":["L","E","S"],"case_id":"D07","profiles":["C","W"]},
    {"arms":["S","E","L"],"case_id":"P07","profiles":["W","C"]},
    {"arms":["E","S","L"],"case_id":"P06","profiles":["C","W"]},
    {"arms":["L","E","S"],"case_id":"V02","profiles":["W","C"]},
    {"arms":["E","L","S"],"case_id":"P04","profiles":["C","W"]},
    {"arms":["S","L","E"],"case_id":"N06","profiles":["W","C"]},
    {"arms":["S","E","L"],"case_id":"D06","profiles":["C","W"]},
    {"arms":["E","L","S"],"case_id":"P02","profiles":["W","C"]},
    {"arms":["L","S","E"],"case_id":"N02","profiles":["C","W"]},
    {"arms":["S","L","E"],"case_id":"N08","profiles":["W","C"]},
    {"arms":["L","S","E"],"case_id":"V05","profiles":["C","W"]},
    {"arms":["S","E","L"],"case_id":"D03","profiles":["W","C"]},
    {"arms":["E","S","L"],"case_id":"P03","profiles":["C","W"]}
]

# Independent literal expected source slices, never obtained from product/oracle.
EXPECTED = {
    "DEV01": b"Development calibration P01 has color teal_01.\n",
    "DEV02": b"Development calibration P02 has color teal_02.\n",
    "DEV03": b"Development calibration P03 has color teal_03.\n",
    "DEV04": b"Development calibration P04 has color teal_04.\n",
    "P01": b"For fixture P01, radius r lies in [0, 1/8] and the coefficient is 7/11.\n",
    "P02": b"For fixture P02, radius r lies in [0, 1/8] and the coefficient is 8/12.\n",
    "P03": b"For fixture P03, radius r lies in [0, 1/8] and the coefficient is 9/13.\n",
    "P04": b"For fixture P04, radius r lies in [0, 1/8] and the coefficient is 10/14.\n",
    "P05": b"For fixture P05, radius r lies in [0, 1/8] and the coefficient is 11/15.\n",
    "P06": b"For fixture P06, radius r lies in [0, 1/8] and the coefficient is 12/16.\n",
    "P07": b"For fixture P07, radius r lies in [0, 1/8] and the coefficient is 13/17.\n",
    "P08": b"For fixture P08, radius r lies in [0, 1/8] and the coefficient is 14/18.\n",
    "D01": b"Deep fixture D01: the elder-death selector is omega_01, and its marker is K401.\n",
    "D02": b"Deep fixture D02: the elder-death selector is omega_02, and its marker is K402.\n",
    "D03": b"Deep fixture D03: the elder-death selector is omega_03, and its marker is K403.\n",
    "D04": b"Deep fixture D04: the elder-death selector is omega_04, and its marker is K404.\n",
    "D05": b"Deep fixture D05: the elder-death selector is omega_05, and its marker is K405.\n",
    "D06": b"Deep fixture D06: the elder-death selector is omega_06, and its marker is K406.\n",
    "D07": b"Deep fixture D07: the elder-death selector is omega_07, and its marker is K407.\n",
    "D08": b"Deep fixture D08: the elder-death selector is omega_08, and its marker is K408.\n",
    "V01": b"Revision family R01 permits p <= 3/7 and permits equality.\r\n",
    "V02": b"Revision family R02 permits p <= 4/7 and permits equality.\r\n",
    "V03": b"Revision family R03 permits p <= 5/7 and permits equality.\r\n",
    "V04": b"Revision family R04 permits p <= 6/7 and permits equality.\r\n",
    "V05": b"Revision family R05 permits p <= 7/7 and permits equality.\r\n",
    "V06": b"Revision family R06 permits p <= 8/7 and permits equality.\r\n",
    "V07": b"Revision family R07 permits p <= 9/7 and permits equality.\r\n",
    "V08": b"Revision family R08 permits p <= 10/7 and permits equality.\r\n",
    "X01": b"Exact-reference note R01: current revision r2 retains marker NOW_01.\r\n",
    "X02": b"Exact-reference note R02: current revision r2 retains marker NOW_02.\r\n",
    "X03": b"Exact-reference note R03: current revision r2 retains marker NOW_03.\r\n",
    "X04": b"Exact-reference note R04: current revision r2 retains marker NOW_04.\r\n",
    "X05": b"Exact-reference note R05: current revision r2 retains marker NOW_05.\r\n",
    "X06": b"Exact-reference note R06: current revision r2 retains marker NOW_06.\r\n",
    "X07": b"Exact-reference note R07: current revision r2 retains marker NOW_07.\r\n",
    "X08": b"Exact-reference note R08: current revision r2 retains marker NOW_08.\r\n",
}

# Trusted test launcher, same interpreter as the real CLI. Both optional
# diagnostic destinations come only from the hosted harness and are removed
# before runpy invokes product code; product gains no environment writer.

def import_context(kind, entry_source):
    library = (Path(sys.base_prefix) / "lib").resolve(strict=True)
    if library.is_relative_to("/source") or library.is_relative_to("/tmp"):
        raise RuntimeError("stdlib image prefix is unsafe")
    version = "python" + str(sys.version_info.major) + "." + str(sys.version_info.minor)
    origins = {}
    for name, module, relative in (("copy", copy, Path("copy.py")),
                                   ("json", json, Path("json/__init__.py"))):
        origin = Path(module.__file__).resolve(strict=True)
        expected = (library / version / relative).resolve(strict=True)
        if origin != expected or not origin.is_relative_to(library):
            raise RuntimeError("stdlib origin is outside the pinned image library")
        origins[name] = {"path": str(origin), "sha256": hashlib.sha256(origin.read_bytes()).hexdigest()}
    if not all(type(item) is str for item in sys.path):
        raise RuntimeError("non-text interpreter import path")
    return {"schema_version": 1, "kind": kind, "optimize": sys.flags.optimize,
            "isolated": sys.flags.isolated,
            "ignore_environment": sys.flags.ignore_environment,
            "safe_path": sys.flags.safe_path, "sys_path": list(sys.path),
            "stdlib_origins": origins,
            "entry_source": str(Path(entry_source).resolve(strict=True))}


def write_import_context(destination, kind, entry_source):
    record = import_context(kind, entry_source)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND
                 | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(fd, "ab") as stream:
        original = os.fstat(stream.fileno())
        if not stat.S_ISREG(original.st_mode) or original.st_nlink != 1:
            raise RuntimeError("import-context output is not a regular trusted file")
        stream.write((json.dumps(record, sort_keys=True, ensure_ascii=True,
                                 allow_nan=False) + "\n").encode("ascii"))


LAUNCHER = r"""
import copy, hashlib, json, os, runpy, stat, sys
from pathlib import Path

def import_context(kind, entry_source):
    library = (Path(sys.base_prefix) / "lib").resolve(strict=True)
    if library.is_relative_to("/source") or library.is_relative_to("/tmp"):
        raise RuntimeError("stdlib image prefix is unsafe")
    version = "python" + str(sys.version_info.major) + "." + str(sys.version_info.minor)
    origins = {}
    for name, module, relative in (("copy", copy, Path("copy.py")),
                                   ("json", json, Path("json/__init__.py"))):
        origin = Path(module.__file__).resolve(strict=True)
        expected = (library / version / relative).resolve(strict=True)
        if origin != expected or not origin.is_relative_to(library):
            raise RuntimeError("stdlib origin is outside the pinned image library")
        origins[name] = {"path": str(origin), "sha256": hashlib.sha256(origin.read_bytes()).hexdigest()}
    if not all(type(item) is str for item in sys.path):
        raise RuntimeError("non-text interpreter import path")
    return {"schema_version": 1, "kind": kind, "optimize": sys.flags.optimize,
            "isolated": sys.flags.isolated,
            "ignore_environment": sys.flags.ignore_environment,
            "safe_path": sys.flags.safe_path, "sys_path": list(sys.path),
            "stdlib_origins": origins,
            "entry_source": str(Path(entry_source).resolve(strict=True))}


def write_import_context(destination, kind, entry_source):
    record = import_context(kind, entry_source)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND
                 | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(fd, "ab") as stream:
        original = os.fstat(stream.fileno())
        if not stat.S_ISREG(original.st_mode) or original.st_nlink != 1:
            raise RuntimeError("import-context output is not a regular trusted file")
        stream.write((json.dumps(record, sort_keys=True, ensure_ascii=True,
                                 allow_nan=False) + "\n").encode("ascii"))


destination = os.environ.pop('SEMANTIC_MEMORY_CHILD_MODE_LOG', None)
import_destination = os.environ.pop('SEMANTIC_MEMORY_IMPORT_CONTEXT', None)
if destination:
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a', encoding='ascii') as stream:
        stream.write(json.dumps({'schema_version': 1, 'optimize': sys.flags.optimize}, sort_keys=True) + '\n')
if import_destination:
    write_import_context(import_destination, 'CLI_PROCESS_IMPORT_CONTEXT', sys.argv[1])
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


class SemanticMemoryFixtureContract(unittest.TestCase):
    """Three exact plan methods; each mutation targets a consumer behavior."""

    @classmethod
    def setUpClass(cls):
        cls.deadline = time.monotonic() + 120
        cls.import_context_destination = os.environ.pop("SEMANTIC_MEMORY_IMPORT_CONTEXT", None)
        if cls.import_context_destination:
            write_import_context(cls.import_context_destination, "TEST_PROCESS_IMPORT_CONTEXT", __file__)

    def setUp(self):
        self.assertTrue(CLI.is_file(), "required Task1 fixture validator CLI tools/semantic_memory_experiment.py is absent")
        self.temp = tempfile.TemporaryDirectory(prefix="semantic-task1-")
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.serial = 0

    def fresh(self):
        self.serial += 1
        path = self.home / ("fixtures-" + str(self.serial))
        shutil.copytree(FIXTURES, path, copy_function=shutil.copyfile)
        # copytree preserves read-only source directory modes; only this fresh copy is mutable.
        for directory in (path, path / "corpus"):
            directory.chmod(0o700)
        return path

    def load(self, path, name):
        return json.loads((path / name).read_bytes())

    def save(self, path, name, value):
        (path / name).write_bytes(encoded(value))

    def manifest(self, path, change):
        value = self.load(path, "corpus-manifest.json")
        change(value)
        self.save(path, "corpus-manifest.json", value)

    def casefile(self, path, name, change):
        value = self.load(path, name)
        change(value)
        self.save(path, name, value)
        raw = (path / name).read_bytes()
        self.manifest(path, lambda m: m["case_files"].update({name: {"bytes": len(raw), "sha256": digest(raw)}}))

    def rebind_text(self, path, filename, raw, revision=None):
        m = self.load(path, "corpus-manifest.json")
        row = next(s for s in m["sources"] if s["content_ref"] == filename)
        sid, old = row["source_id"], row["revision_id"]
        new = revision or old
        (path / filename).write_bytes(raw)
        row.update(revision_id=new, source_sha256=digest(raw), extraction_sha256=digest(raw),
                   text_sha256=digest(raw), text_bytes=len(raw))
        self.save(path, "corpus-manifest.json", m)
        for name in ("development-cases.json", "heldout-cases.json"):
            def update_cases(v):
                for case in v["cases"]:
                    if case["exact_source_id"] == sid and case["exact_ref"] == old:
                        case["exact_ref"] = new
            self.casefile(path, name, update_cases)
        def update_gold(v):
            for record in v["records"] + v["development_records"]:
                for span in record["spans"]:
                    if span["source_id"] == sid and span["revision_id"] == old:
                        before = (FIXTURES / filename).read_bytes()
                        piece = before[span["start_byte"]:span["end_byte"]]
                        offset = raw.find(piece)
                        self.assertGreaterEqual(offset, 0, "mutation must preserve its declared gold piece")
                        span.update(revision_id=new, text_sha256=digest(raw), start_byte=offset,
                                    end_byte=offset + len(piece), sha256=digest(piece))
        self.casefile(path, "heldout-oracle.json", update_gold)

    def invoke(self, path, mode="prepare-report", report=None, timeout=20):
        remaining = self.deadline - time.monotonic()
        self.assertGreater(remaining, 0, "finite120s per-mode test budget exhausted")
        args = [sys.executable, "-I", "-B"] + (["-O"] if sys.flags.optimize else []) + ["-S", "-c", LAUNCHER, str(CLI), mode, "--fixtures", str(path)]
        if report is not None:
            args += ["--report", str(report)]
        environment = os.environ.copy()
        if self.import_context_destination:
            environment["SEMANTIC_MEMORY_IMPORT_CONTEXT"] = self.import_context_destination
        try:
            return subprocess.run(args, cwd=self.home, env=environment, capture_output=True,
                                  timeout=min(timeout, remaining), check=False)
        except subprocess.TimeoutExpired:
            self.fail("real CLI blocked or exceeded finite child/mode deadline")

    def accepted(self, path):
        result = self.invoke(path)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertEqual(result.stderr, b"")
        return json.loads(result.stdout), result.stdout

    def refused(self, path, timeout=20):
        result = self.invoke(path, timeout=timeout)
        self.assertEqual(result.returncode, 1, result.stdout.decode("utf-8", "replace"))
        self.assertEqual(result.stdout, b"")
        self.assertTrue(result.stderr.startswith(b"semantic memory fixtures refused:"), result.stderr[:300])
        self.assertNotIn(b"Traceback", result.stderr)

    def mutation(self, name, change, timeout=20):
        with self.subTest(mutation=name):
            path = self.fresh()
            self.accepted(path)  # Valid positive before every mutated copy.
            change(path)
            self.refused(path, timeout)

    def admission_corpus(self, path):
        """Call the actual existing API in this hosted parent process only."""
        self.assertGreater(self.deadline - time.monotonic(), 0,
                           "finite120s per-mode test budget exhausted")
        # Diagnostic paths belong to the trusted harness, never product API data.
        # Restore both exact existence and value even on loader/API exceptions.
        missing = object()
        names = ("SEMANTIC_MEMORY_CHILD_MODE_LOG", "SEMANTIC_MEMORY_IMPORT_CONTEXT")
        saved = {name: os.environ.pop(name, missing) for name in names}
        try:
            namespace = runpy.run_path(str(CLI), run_name="_semantic_memory_admission_probe")
            product = namespace["_product"]()
            return product.admitted_corpus(path)
        finally:
            for name, value in saved.items():
                if value is missing:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def original_admission(self, path):
        # Expectations come from the independent input fixture, never a report,
        # product helper, API projection, or coherently mutated copy.
        sources = self.load(FIXTURES, "corpus-manifest.json")["sources"]
        self.assertEqual(len(sources), 32)
        actual = self.admission_corpus(path)
        self.assertIs(type(actual), list)
        self.assertEqual(len(actual), 32)
        for row, source in zip(actual, sources):
            self.assertIs(type(row), dict)
            self.assertEqual(set(row), SOURCE_KEYS | {"text"})
            expected = dict(source, text=(FIXTURES / source["content_ref"]).read_bytes().decode("utf-8"))
            self.assertEqual(row, expected)
        self.assertGreater(self.deadline - time.monotonic(), 0,
                           "finite120s per-mode test budget exhausted")

    def strict_accepted(self, path):
        result = self.invoke(path, "admit-benchmark")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertEqual(result.stderr, b"")
        report = json.loads(result.stdout)
        self.assertEqual(result.stdout, encoded(report))
        self.assertIs(type(report["schema_version"]), int)
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["benchmark_admission"], {
            "admitted": True,
            "benchmark_id": "semantic-memory-synthetic-20261008-v1",
            "semantic_sha256": "8d28c546b7871916adac363f56dc58204785093686ba919f14f5a4c79cefd04f"})
        self.assertIs(report["benchmark_admission"]["admitted"], True)
        # Independently derive only the expected original corpus projection,
        # never the expected admission digest or flag from candidate output.
        sources = self.load(FIXTURES, "corpus-manifest.json")["sources"]
        self.assertEqual(len(sources), 32)
        actual = report["corpus_projection"]
        self.assertIs(type(actual), list)
        self.assertEqual(len(actual), 32)
        for row, source in zip(actual, sources):
            self.assertIs(type(row), dict)
            self.assertEqual(set(row), SOURCE_KEYS | {"text"})
            expected = dict(source, text=(FIXTURES / source["content_ref"]).read_bytes().decode("utf-8"))
            self.assertEqual(row, expected)
        return report

    def strict_refused(self, path):
        result = self.invoke(path, "admit-benchmark")
        self.assertEqual(result.returncode, 1, result.stdout.decode("utf-8", "replace"))
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"semantic memory fixtures refused: invalid input or output\n")

    def frozen_admission_mutation(self, name, change):
        with self.subTest(frozen_admission=name):
            path = self.fresh()
            self.original_admission(path)  # Real API positive before each mutation.
            change(path)
            self.accepted(path)  # Real existing CLI proves structural validity.
            # Missing API/loader/setup errors cannot satisfy this rejection.
            # The predecessor returns normally, giving ValueError-not-raised FAIL.
            with self.assertRaises(ValueError, msg="frozen benchmark identity must refuse " + name):
                self.admission_corpus(path)
            self.strict_refused(path)  # Same structurally valid mutant, real strict CLI.

    def test_frozen_case_inventory_and_hashes(self):
        # Catches omitted identity, schema, cardinality, budget or nofollow checks.
        path = self.fresh()
        report, first = self.accepted(path)
        self.assertEqual(first, encoded(report))
        second, second_bytes = self.accepted(path)
        self.assertEqual(first, second_bytes)
        self.assertEqual(report, second)
        self.assertEqual(set(report), {"schema_version", "scientific_effect", "scientific_status_authority",
            "fixture_manifest_sha256", "files", "counts", "seed", "query_order", "baseline_definitions",
            "gates", "corpus_projection", "request_projections", "validation", "pending", "benchmark_admission"})
        self.assertIs(type(report["schema_version"]), int)
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["benchmark_admission"], {"admitted": False, "reason": "STRUCTURAL_VALIDATION_ONLY"})
        self.assertIs(report["benchmark_admission"]["admitted"], False)
        self.assertEqual(report["counts"], {"documents": 32, "text_bytes": 107886, "development": 8,
            "heldout": 40, "by_stratum": {s: 8 for s in STRATA}, "generation": 20})
        self.assertIs(report["scientific_status_authority"], False)
        self.assertEqual(report["scientific_effect"], "NONE")
        self.assertEqual(report["seed"], 20261008)
        self.assertEqual(report["pending"], {"tokenizer_parity": "NOT_EVALUATED", "chunk_count": None,
            "retrieval": "NOT_EVALUATED", "quality_cost_fairness": "NOT_EVALUATED", "installed_consumers": "NOT_EVALUATED"})
        self.assertEqual(report["validation"], {"inventory": "passed", "gold": "passed", "deep": "passed", "representation": "passed"})
        expected_files = {p.relative_to(path).as_posix(): {"bytes": len(p.read_bytes()), "sha256": digest(p.read_bytes())}
                          for p in path.rglob("*") if p.is_file()}
        self.assertEqual(report["files"], expected_files)
        self.assertEqual(len(expected_files), 36)
        self.assertEqual(report["fixture_manifest_sha256"], digest((path / "corpus-manifest.json").read_bytes()))
        m = self.load(path, "corpus-manifest.json")
        self.assertNotIn("sha256", m)
        for key in ("query_order", "baseline_definitions", "gates"):
            self.assertEqual(report[key], m[key])
        self.assertEqual(len({r["case_id"] for r in m["query_order"]}), 40)
        for row in m["query_order"]:
            self.assertEqual(sorted(row["arms"]), ["E", "L", "S"])
            self.assertEqual(sorted(row["profiles"]), ["C", "W"])
        self.assertEqual(m["generation_case_ids"], FROZEN_GENERATION_CASE_IDS)
        self.assertEqual(m["query_order"], FROZEN_QUERY_ORDER)
        self.assertEqual(report["query_order"], FROZEN_QUERY_ORDER)
        self.assertEqual(len(m["generation_case_ids"]), 20)
        self.assertEqual({prefix: sum(x.startswith(prefix) for x in m["generation_case_ids"])
                          for prefix in "PDVXN"}, {p: 4 for p in "PDVXN"})
        development = self.load(path, "development-cases.json")["cases"]
        self.assertTrue(all(c["exact_source_id"] is None and c["exact_ref"] is None for c in development))
        development_gold = self.load(path, "heldout-oracle.json")["development_records"]
        self.assertEqual([c["case_id"] for c in development_gold if c["no_answer"]], ["DEV05", "DEV06", "DEV07", "DEV08"])
        saved = self.home / "report.json"
        saved.write_bytes(first)
        verified = self.invoke(path, "verify-report", saved)
        self.assertEqual((verified.returncode, verified.stdout, verified.stderr), (0, first, b""))
        bad_report = copy.deepcopy(report); bad_report["counts"]["documents"] = 31
        saved.write_bytes(encoded(bad_report))
        invalid = self.invoke(path, "verify-report", saved)
        self.assertEqual(invalid.returncode, 1)
        self.assertEqual(invalid.stdout, b"")
        self.assertTrue(invalid.stderr.startswith(b"semantic memory fixtures refused:"))

        tampered_admission = copy.deepcopy(report)
        tampered_admission["benchmark_admission"]["admitted"] = True
        saved.write_bytes(encoded(tampered_admission))
        invalid_admission = self.invoke(path, "verify-report", saved)
        self.assertEqual(invalid_admission.returncode, 1)
        self.assertEqual(invalid_admission.stdout, b"")
        self.assertEqual(invalid_admission.stderr,
                         b"semantic memory fixtures refused: invalid input or output\n")

        # Strict admission positives: literal reference identity, not transport bytes.
        strict_original = self.fresh()
        self.strict_accepted(strict_original)

        formatted = self.fresh()
        for name in ("development-cases.json", "heldout-cases.json", "heldout-oracle.json"):
            value = self.load(formatted, name)
            # Reverse top-level key order and change whitespace in every case doc.
            reordered = dict(reversed(list(value.items())))
            raw = (json.dumps(reordered, sort_keys=False, ensure_ascii=False,
                              indent=1, allow_nan=False) + "\n\n").encode("utf-8")
            self.assertNotEqual(raw, (FIXTURES / name).read_bytes())
            self.assertEqual(json.loads(raw), self.load(FIXTURES, name))
            self.assertEqual(list(json.loads(raw)), list(reordered))
            (formatted / name).write_bytes(raw)
            self.manifest(formatted, lambda m: m["case_files"].update(
                {name: {"bytes": len(raw), "sha256": digest(raw)}}))
        manifest = self.load(formatted, "corpus-manifest.json")
        original_manifest = self.load(FIXTURES, "corpus-manifest.json")
        self.assertEqual({k: v for k, v in manifest.items() if k != "case_files"},
                         {k: v for k, v in original_manifest.items() if k != "case_files"})
        reordered = dict(reversed(list(manifest.items())))
        raw = (json.dumps(reordered, sort_keys=False, ensure_ascii=False,
                          indent=1, allow_nan=False) + "\n\n").encode("utf-8")
        self.assertNotEqual(raw, (FIXTURES / "corpus-manifest.json").read_bytes())
        self.assertEqual(json.loads(raw), manifest)
        self.assertEqual(list(json.loads(raw)), list(reordered))
        (formatted / "corpus-manifest.json").write_bytes(raw)
        self.strict_accepted(formatted)
        self.original_admission(formatted)

        sentinel_only = self.fresh()
        marker = "EVALUATOR_ONLY_ADMISSION_POSITIVE_20261017_428"
        self.assertNotEqual(marker, SENTINEL)
        independent_public = b"\n".join((FIXTURES / s["content_ref"]).read_bytes()
            for s in original_manifest["sources"])
        independent_public += (FIXTURES / "development-cases.json").read_bytes()
        independent_public += (FIXTURES / "heldout-cases.json").read_bytes()
        self.assertNotIn(marker.encode("utf-8"), independent_public)
        self.casefile(sentinel_only, "heldout-oracle.json",
                      lambda v: v.update(oracle_only_sentinel=marker))
        changed_oracle = self.load(sentinel_only, "heldout-oracle.json")
        original_oracle = self.load(FIXTURES, "heldout-oracle.json")
        self.assertEqual({k: v for k, v in changed_oracle.items() if k != "oracle_only_sentinel"},
                         {k: v for k, v in original_oracle.items() if k != "oracle_only_sentinel"})
        self.strict_accepted(sentinel_only)
        self.original_admission(sentinel_only)

        self.mutation("changed_bytes", lambda p: (p / "corpus/P01.txt").write_bytes(b"changed"))
        self.mutation("missing_file", lambda p: (p / "corpus/P01.txt").unlink())
        self.mutation("undeclared_file", lambda p: (p / "corpus/UNDECLARED.txt").write_bytes(b"extra"))
        self.mutation("duplicate_source", lambda p: self.manifest(p, lambda m: m["sources"].__setitem__(1, copy.deepcopy(m["sources"][0]))))
        self.mutation("missing_source", lambda p: self.manifest(p, lambda m: m["sources"].pop()))
        self.mutation("wrong_hash", lambda p: self.manifest(p, lambda m: m["sources"][0].update(text_sha256="0" * 64)))
        self.mutation("identity_source_hash", lambda p: self.manifest(p, lambda m: m["sources"][0].update(source_sha256="0" * 64)))
        self.mutation("identity_extraction_hash", lambda p: self.manifest(p, lambda m: m["sources"][0].update(extraction_sha256="0" * 64)))
        self.mutation("wrong_bytes", lambda p: self.manifest(p, lambda m: m["sources"][0].update(text_bytes=1)))
        self.mutation("boolean_integer", lambda p: self.manifest(p, lambda m: m["sources"][0].update(policy_epoch=True)))
        self.mutation("unknown_manifest_field", lambda p: self.manifest(p, lambda m: m.update(unexpected=1)))
        self.mutation("source_extra_field", lambda p: self.manifest(p, lambda m: m["sources"][0].update(notes="extra")))
        self.mutation("wrong_seed", lambda p: self.manifest(p, lambda m: m.update(seed=20261009)))
        self.mutation("weakened_gate", lambda p: self.manifest(p, lambda m: m["gates"].update(unsupported_answer_max=1)))
        self.mutation("semantic_arm_relabelled_lexical", lambda p: self.manifest(p, lambda m: m["baseline_definitions"].update(S="literal_only")))
        self.mutation("schedule_duplicate", lambda p: self.manifest(p, lambda m: m["query_order"].__setitem__(1, copy.deepcopy(m["query_order"][0]))))
        self.mutation("schedule_arm_duplicate", lambda p: self.manifest(p, lambda m: m["query_order"][0].update(arms=["E", "E", "S"])))

        # Ready review P1s: valid alternative selections/permutations must refuse.
        self.mutation("generation_case_ids_substitute_05_08", lambda p: self.manifest(p, lambda m: m.update(
            generation_case_ids=[prefix + format(number, "02d") for prefix in "PDVXN" for number in range(5, 9)])))
        self.mutation("generation_case_ids_reordered", lambda p: self.manifest(p, lambda m: m["generation_case_ids"].reverse()))
        self.mutation("query_order_reversed", lambda p: self.manifest(p, lambda m: m["query_order"].reverse()))
        self.mutation("query_order_adjacent_swap", lambda p: self.manifest(p, lambda m: m["query_order"].__setitem__(
            slice(0, 2), [m["query_order"][1], m["query_order"][0]])))
        self.mutation("query_order_first_arms_reverse", lambda p: self.manifest(p, lambda m: m["query_order"][0]["arms"].reverse()))
        self.mutation("query_order_last_arms_reverse", lambda p: self.manifest(p, lambda m: m["query_order"][-1]["arms"].reverse()))
        self.mutation("query_order_first_profiles_swap", lambda p: self.manifest(p, lambda m: m["query_order"][0]["profiles"].reverse()))
        self.mutation("query_order_last_profiles_swap", lambda p: self.manifest(p, lambda m: m["query_order"][-1]["profiles"].reverse()))
        self.mutation("generation_19", lambda p: self.manifest(p, lambda m: m["generation_case_ids"].pop()))
        self.mutation("heldout_duplicate", lambda p: self.casefile(p, "heldout-cases.json", lambda v: v["cases"].__setitem__(1, copy.deepcopy(v["cases"][0]))))
        self.mutation("development_exact_shortcircuit", lambda p: self.casefile(p, "development-cases.json", lambda v: v["cases"][0].update(exact_source_id="SYN.P01", exact_ref="r1")))
        self.mutation("stratum_skew", lambda p: self.casefile(p, "heldout-cases.json", lambda v: v["cases"][0].update(stratum="no_answer")))
        self.mutation("query_byte_budget", lambda p: self.casefile(p, "heldout-cases.json", lambda v: v["cases"][0].update(query="x" * 2049)))
        self.mutation("traversal", lambda p: self.manifest(p, lambda m: m["sources"][0].update(content_ref="../corpus/D01.txt")))
        self.mutation("absolute_path", lambda p: self.manifest(p, lambda m: m["sources"][0].update(content_ref="/etc/passwd")))
        self.mutation("path_byte_budget", lambda p: self.manifest(p, lambda m: m["sources"][0].update(content_ref="corpus/" + "x" * 1025 + ".txt")))
        def duplicate_key(p):
            file = p / "corpus-manifest.json"
            file.write_bytes(file.read_bytes().replace(b'"schema_version": 1', b'"schema_version": 1, "schema_version": 1', 1))
        self.mutation("duplicate_json_key", duplicate_key)
        def nonfinite(p):
            file = p / "corpus-manifest.json"
            file.write_bytes(file.read_bytes().replace(b'"seed": 20261008', b'"seed": NaN', 1))
        self.mutation("nonfinite_json", nonfinite)

        # Retained mutation-audit negatives (main#428/6244680213).
        # Rebind case-file transport bytes when testing JSON semantics; keep
        # the original three-method runner contract and frozen fixtures intact.
        def nonfinite_case_token(p):
            name = "heldout-cases.json"
            value = self.load(p, name)
            case = next(c for c in value["cases"] if c["case_id"] == "P01")
            self.assertIsNone(case["exact_source_id"])
            case["exact_source_id"] = float("nan")
            raw = (json.dumps(value, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":"), allow_nan=True) + "\n").encode("utf-8")
            self.assertIn(b'"exact_source_id":NaN', raw)
            (p / name).write_bytes(raw)
            self.manifest(p, lambda m: m["case_files"].update(
                {name: {"bytes": len(raw), "sha256": digest(raw)}}))
        self.mutation("nonfinite_case_token_with_rebound_descriptor", nonfinite_case_token)
        self.mutation("stale_case_file_descriptor", lambda p: self.manifest(
            p, lambda m: m["case_files"]["heldout-cases.json"].update(sha256="0" * 64)))
        self.mutation("whitespace_only_oracle_sentinel", lambda p: self.casefile(
            p, "heldout-oracle.json", lambda v: v.update(oracle_only_sentinel="\u3000")))
        def invalid_json_utf8(p):
            file = p / "corpus-manifest.json"
            file.write_bytes(file.read_bytes() + b"\xff")
        self.mutation("metadata_invalid_UTF8", invalid_json_utf8)
        def invalid_text_utf8(p):
            self.rebind_text(p, "corpus/P01.txt", (p / "corpus/P01.txt").read_bytes() + b"\xff", "r2-invalid")
        self.mutation("text_invalid_UTF8", invalid_text_utf8)
        def oversized(p):
            name = "heldout-cases.json"; file = p / name
            raw = file.read_bytes() + b" " * (4 * 2**20)
            file.write_bytes(raw)
            self.manifest(p, lambda m: m["case_files"].update({name: {"bytes": len(raw), "sha256": digest(raw)}}))
        self.mutation("metadata_over4MiB", oversized)
        boundary = self.fresh()
        name = "heldout-cases.json"; file = boundary / name
        raw = file.read_bytes(); raw += b" " * (4 * 2**20 - len(raw))
        file.write_bytes(raw)
        self.manifest(boundary, lambda m: m["case_files"].update({name: {"bytes": len(raw), "sha256": digest(raw)}}))
        self.accepted(boundary)
        def deep_json(p):
            name = "heldout-cases.json"; raw = (b'{"schema_version":1,"cases":' + b"[" * 66 + b"0" + b"]" * 66 + b"}")
            (p / name).write_bytes(raw)
            self.manifest(p, lambda m: m["case_files"].update({name: {"bytes": len(raw), "sha256": digest(raw)}}))
        self.mutation("container_depth", deep_json)
        def large_corpus(p):
            self.rebind_text(p, "corpus/P01.txt", (p / "corpus/P01.txt").read_bytes() + b"z" * 393216, "r2-large")
        self.mutation("corpus_over384KiB", large_corpus)
        corpus_boundary = self.fresh()
        original = (corpus_boundary / "corpus/P01.txt").read_bytes()
        self.rebind_text(corpus_boundary, "corpus/P01.txt", original + b"z" * (393216 - 107886), "r2-boundary")
        boundary_report, _ = self.accepted(corpus_boundary)
        self.assertEqual(boundary_report["counts"]["text_bytes"], 393216)
        def link(p):
            target = self.home / ("outside-" + str(self.serial)); source = p / "corpus/P01.txt"
            target.write_bytes(source.read_bytes()); source.unlink(); source.symlink_to(target)
        self.mutation("file_symlink", link)
        def linkdir(p):
            source = p / "corpus"; target = self.home / ("outside-dir-" + str(self.serial))
            source.rename(target); source.symlink_to(target, target_is_directory=True)
        self.mutation("directory_symlink", linkdir)
        def directory(p):
            source = p / "corpus/P01.txt"; source.unlink(); source.mkdir()
        self.mutation("nonregular_directory", directory)
        def fifo(p):
            source = p / "corpus/P01.txt"; source.unlink(); os.mkfifo(source)
        self.mutation("fifo_no_block", fifo, timeout=2)

        # Frozen-admission controls: structural reporting remains permissive.
        # These are in-process API subtests, not new CLI negative controls.
        # Keep the original 67 CLI mutations and exactly three test methods.
        def change_query(p, filename, identifier, query):
            self.casefile(p, filename, lambda v: next(c for c in v["cases"]
                if c["case_id"] == identifier).update(query=query))

        def change_answer(p, collection, identifier, answer):
            self.casefile(p, "heldout-oracle.json", lambda v: next(r for r in v[collection]
                if r["case_id"] == identifier).update(expected_answer=answer))

        def retarget_support(p, collection, receiver, donor):
            # Support is copied from immutable external input, never product output.
            document = self.load(FIXTURES, "heldout-oracle.json")
            support = copy.deepcopy(next(r for r in document[collection]
                if r["case_id"] == donor)["spans"])
            self.assertEqual(len(support), 1)
            span = support[0]
            source = next(s for s in self.load(FIXTURES, "corpus-manifest.json")["sources"]
                if (s["source_id"], s["revision_id"]) == (span["source_id"], span["revision_id"]))
            source_bytes = (FIXTURES / source["content_ref"]).read_bytes()
            self.assertEqual(source_bytes[span["start_byte"]:span["end_byte"]], EXPECTED[donor])
            self.assertEqual(span["sha256"], digest(EXPECTED[donor]))
            self.casefile(p, "heldout-oracle.json", lambda v: next(r for r in v[collection]
                if r["case_id"] == receiver).update(spans=support))

        self.frozen_admission_mutation("P01_query_answer_leak", lambda p: change_query(
            p, "heldout-cases.json", "P01", "For fixture P01, the coefficient is 7/11. Which coefficient is recorded?"))
        self.frozen_admission_mutation("N01_query_copied_P01_fact", lambda p: change_query(
            p, "heldout-cases.json", "N01", "For fixture P01, radius r lies in [0, 1/8] and the coefficient is 7/11."))
        self.frozen_admission_mutation("DEV01_query_meaning_changed", lambda p: change_query(
            p, "development-cases.json", "DEV01", "Which calibration color is recorded for P02?"))
        self.frozen_admission_mutation("P01_expected_answer_changed", lambda p: change_answer(
            p, "records", "P01", "8/12"))
        self.frozen_admission_mutation("DEV01_expected_answer_changed", lambda p: change_answer(
            p, "development_records", "DEV01", "teal_02"))
        self.frozen_admission_mutation("corpus_source_order_reversed", lambda p: self.manifest(
            p, lambda m: m["sources"].reverse()))
        self.frozen_admission_mutation("corpus_source_order_adjacent_swap", lambda p: self.manifest(
            p, lambda m: m["sources"].__setitem__(slice(0, 2), [m["sources"][1], m["sources"][0]])))
        self.frozen_admission_mutation("P01_support_retargeted_P02", lambda p: retarget_support(
            p, "records", "P01", "P02"))
        self.frozen_admission_mutation("D01_support_retargeted_D02", lambda p: retarget_support(
            p, "records", "D01", "D02"))
        self.frozen_admission_mutation("V01_support_retargeted_V02", lambda p: retarget_support(
            p, "records", "V01", "V02"))

        def paired_exact_retarget(p):
            retarget_support(p, "records", "X01", "X02")
            self.casefile(p, "heldout-cases.json", lambda v: next(c for c in v["cases"]
                if c["case_id"] == "X01").update(exact_source_id="SYN.R02", exact_ref="r2"))
        self.frozen_admission_mutation("X01_support_and_selector_retargeted_X02", paired_exact_retarget)

        def duplicate_exact_occurrence(p):
            filename = "corpus/R01-current.txt"
            original = (p / filename).read_bytes()
            piece = EXPECTED["X01"]
            first = original.index(piece)
            self.assertEqual(original.count(piece), 1)
            duplicate_start = len(original) + 2
            raw = original + b"\r\n" + piece
            self.rebind_text(p, filename, raw)  # Rebind every affected source/case/span hash.
            self.assertNotEqual(first, duplicate_start)
            self.assertEqual(raw[duplicate_start:duplicate_start + len(piece)], piece)
            self.casefile(p, "heldout-oracle.json", lambda v: next(r for r in v["records"]
                if r["case_id"] == "X01")["spans"][0].update(
                    start_byte=duplicate_start, end_byte=duplicate_start + len(piece), sha256=digest(piece)))
        self.frozen_admission_mutation("X01_exact_span_duplicate_occurrence", duplicate_exact_occurrence)

        def coherent_filler_extension(p):
            filename = "corpus/P01.txt"
            self.rebind_text(p, filename, (p / filename).read_bytes()
                + b"\nFrozen identity control: neutral filler extension.\n")
        self.frozen_admission_mutation("corpus_coherent_filler_extension", coherent_filler_extension)
        self.frozen_admission_mutation("N02_query_meaning_changed_without_cue", lambda p: change_query(
            p, "heldout-cases.json", "N02", "For revision family R02, which marker identifies the current revision?"))
        self.frozen_admission_mutation("DEV01_support_retargeted_DEV02", lambda p: retarget_support(
            p, "development_records", "DEV01", "DEV02"))
        self.frozen_admission_mutation("X01_expected_answer_changed", lambda p: change_answer(
            p, "records", "X01", "NOW_02"))

    def test_gold_never_enters_index_or_prompt(self):
        # Catches copying evaluator annotations into indexed/request data or eval.
        path = self.fresh(); report, _ = self.accepted(path)
        sources = self.load(path, "corpus-manifest.json")["sources"]
        self.assertEqual(len(report["corpus_projection"]), 32)
        for projected, source in zip(report["corpus_projection"], sources):
            self.assertEqual(set(projected), SOURCE_KEYS | {"text"})
            self.assertEqual({k: projected[k] for k in SOURCE_KEYS}, source)
            self.assertEqual(projected["text"].encode("utf-8"), (path / source["content_ref"]).read_bytes())
        cases = self.load(path, "development-cases.json")["cases"] + self.load(path, "heldout-cases.json")["cases"]
        self.assertEqual(set(report["request_projections"]), {c["case_id"] for c in cases})
        for case in cases:
            self.assertEqual(set(report["request_projections"][case["case_id"]]), {"W", "C"})
            for profile in ("W", "C"):
                actual = report["request_projections"][case["case_id"]][profile]
                self.assertEqual(set(actual), REQUEST_KEYS)
                self.assertEqual(actual, {k: case[k] for k in REQUEST_KEYS - {"profile"}} | {"profile": profile})
                if case["case_id"].startswith("N") or case["case_id"] in {"DEV05", "DEV06", "DEV07", "DEV08"}:
                    for cue in ("absent", "no-answer", "unanswerable", case["case_id"].casefold()):
                        self.assertNotIn(cue, actual["query"].casefold())
        self.assertEqual(self.load(path, "heldout-oracle.json")["oracle_only_sentinel"], SENTINEL)
        projection = json.dumps([report["corpus_projection"], report["request_projections"]], ensure_ascii=False)
        self.assertNotIn(SENTINEL, projection)
        self.assertIn("claim ACCEPTED", projection)  # Legitimate source data is retained.
        original = (report["corpus_projection"], report["request_projections"])
        self.casefile(path, "heldout-oracle.json", lambda v: v.update(oracle_only_sentinel="OTHER_EVALUATOR_ONLY_MARKER"))
        changed, _ = self.accepted(path)
        self.assertEqual(original, (changed["corpus_projection"], changed["request_projections"]))
        self.mutation("case_oracle_field", lambda p: self.casefile(p, "heldout-cases.json", lambda v: v["cases"][0].update(expected_answer=SENTINEL)))
        self.mutation("source_oracle_field", lambda p: self.manifest(p, lambda m: m["sources"][0].update(oracle=SENTINEL)))
        self.mutation("manifest_oracle_field", lambda p: self.manifest(p, lambda m: m.update(expected_answer=SENTINEL)))

        # Coherent canary leaks: closed projections can still copy evaluator data.
        # Keep ordinary answer facts/source instructions and oracle-only changes valid.
        def append_source_marker(p, marker, revision):
            raw = (p / "corpus/P01.txt").read_bytes() + ("\n" + marker + "\n").encode("utf-8")
            self.rebind_text(p, "corpus/P01.txt", raw, revision)

        def append_query_marker(p, name, identifier, marker):
            def change(value):
                case = next(c for c in value["cases"] if c["case_id"] == identifier)
                case["query"] += " " + marker
            self.casefile(p, name, change)

        self.mutation("sentinel_in_coherent_corpus", lambda p: append_source_marker(p, SENTINEL, "r2-sentinel"))
        self.mutation("sentinel_in_answerable_heldout_query", lambda p: append_query_marker(
            p, "heldout-cases.json", "P01", SENTINEL))
        self.mutation("sentinel_in_answerable_development_query", lambda p: append_query_marker(
            p, "development-cases.json", "DEV01", SENTINEL))

        def dynamic_corpus_marker(p):
            marker = "EVALUATOR_ONLY_READY_DYNAMIC_CORPUS_20261016"
            self.casefile(p, "heldout-oracle.json", lambda value: value.update(oracle_only_sentinel=marker))
            self.accepted(p)  # Changed oracle alone remains valid before the leak.
            append_source_marker(p, marker, "r2-dynamic-sentinel")
        self.mutation("changed_sentinel_in_coherent_corpus", dynamic_corpus_marker)

        def dynamic_heldout_marker(p):
            marker = "EVALUATOR_ONLY_READY_DYNAMIC_HELDOUT_20261016"
            self.casefile(p, "heldout-oracle.json", lambda value: value.update(oracle_only_sentinel=marker))
            self.accepted(p)
            append_query_marker(p, "heldout-cases.json", "P01", marker)
        self.mutation("changed_sentinel_in_answerable_heldout_query", dynamic_heldout_marker)

        def dynamic_development_marker(p):
            marker = "EVALUATOR_ONLY_READY_DYNAMIC_DEVELOPMENT_20261016"
            self.casefile(p, "heldout-oracle.json", lambda value: value.update(oracle_only_sentinel=marker))
            self.accepted(p)
            append_query_marker(p, "development-cases.json", "DEV01", marker)
        self.mutation("changed_sentinel_in_answerable_development_query", dynamic_development_marker)

        def escaped_unicode_marker(p):
            marker = "EVALUATOR_ONLY_READY_\u00e9_\u03bb_20261016"
            self.casefile(p, "heldout-oracle.json", lambda value: value.update(oracle_only_sentinel=marker))
            self.accepted(p)
            append_query_marker(p, "heldout-cases.json", "P01", marker)
            name = "heldout-cases.json"
            raw = (json.dumps(self.load(p, name), sort_keys=True, ensure_ascii=True,
                              indent=2, allow_nan=False) + "\n").encode("ascii")
            self.assertIn(b"\\u00e9", raw)
            (p / name).write_bytes(raw)
            self.manifest(p, lambda m: m["case_files"].update({name: {"bytes": len(raw), "sha256": digest(raw)}}))
        self.mutation("changed_unicode_sentinel_escaped_query", escaped_unicode_marker)
        safe = self.fresh(); canary = self.home / "UNTRUSTED_CODE_EXECUTED"
        raw = (safe / "corpus/P01.txt").read_bytes() + ("\n__import__('pathlib').Path(" + repr(str(canary)) + ").write_text('executed')\n").encode()
        self.rebind_text(safe, "corpus/P01.txt", raw, "r2-inert")
        inert, _ = self.accepted(safe)
        self.assertFalse(canary.exists())
        projected = next(s for s in inert["corpus_projection"] if s["source_id"] == "SYN.P01")
        self.assertEqual(projected["text"].encode(), raw)

    def test_deep_and_confusable_spans_are_real(self):
        # Catches character-offset substitution, version drift and text relabels.
        path = self.fresh(); report, _ = self.accepted(path)
        m = self.load(path, "corpus-manifest.json")
        sources = {(s["source_id"], s["revision_id"]): s for s in m["sources"]}
        gold_document = self.load(path, "heldout-oracle.json")
        self.assertEqual(len(gold_document["records"]), 40)
        self.assertEqual(len(gold_document["development_records"]), 8)
        records = gold_document["records"] + gold_document["development_records"]
        whole = b"\n".join((path / s["content_ref"]).read_bytes() for s in m["sources"])
        self.assertNotIn("certified external horizon".encode(), whole)
        for record in records:
            cid = record["case_id"]
            if cid.startswith("N") or cid in {"DEV05", "DEV06", "DEV07", "DEV08"}:
                self.assertIs(record["no_answer"], True)
                self.assertIsNone(record["expected_answer"])
                self.assertEqual(record["spans"], [])
                continue
            self.assertIs(record["no_answer"], False)
            self.assertEqual(len(record["spans"]), 1)
            span = record["spans"][0]; source = sources[(span["source_id"], span["revision_id"])]
            raw = (path / source["content_ref"]).read_bytes(); piece = EXPECTED[cid]
            self.assertEqual(raw[span["start_byte"]:span["end_byte"]], piece)
            self.assertEqual(span["sha256"], digest(piece))
            self.assertEqual(span["text_sha256"], digest(raw))
            raw[:span["start_byte"]].decode("utf-8"); piece.decode("utf-8")
            if cid.startswith("D") and not cid.startswith("DEV"):
                position = len(raw[:span["start_byte"]].decode("utf-8"))
                self.assertGreater(position, 12000)
                self.assertNotEqual(position, span["start_byte"])
                self.assertNotIn(piece.strip(), raw.decode("utf-8")[:12000].encode())
            if cid.startswith(("V", "X")):
                self.assertEqual(span["revision_id"], "r2")
                self.assertIn(b"\r\n", piece)
                old = (path / source["content_ref"].replace("-current", "-old")).read_bytes()
                self.assertNotEqual(old, raw)
                self.assertIn(b"forbids equality", old)
                self.assertIn(b"permits equality", raw)
        self.assertIn("e\u0301", report["corpus_projection"][8]["text"] + (path / "corpus/P01.txt").read_text())
        calibration = self.load(path, "heldout-oracle.json")["representation_controls"][0]
        original = '{"number":7,"title":"Example α","body":"Evidence β"}'.encode("utf-8")
        derived = "Evidence β\nExample α".encode("utf-8")
        self.assertEqual((len(original), digest(original)), (54, "4bfae15708ccf9ab9b95cb2074a88d908079347c795bbb3d99646f20b81af008"))
        self.assertEqual((len(derived), digest(derived)), (22, "64fac08f4a87d4d4a6c963e801f97271a03693db55901fb24d62660bb4e5937e"))
        self.assertEqual(calibration["text_sha256"], digest(derived))
        self.assertNotEqual(digest(original), digest(derived))
        def gold(p, cid, mutate):
            self.casefile(p, "heldout-oracle.json", lambda v: mutate(next(r for r in v["records"] if r["case_id"] == cid)))
        self.mutation("character_not_byte_offset", lambda p: gold(p, "D01", lambda r: r["spans"][0].update(start_byte=len((p / "corpus/D01.txt").read_bytes()[:r["spans"][0]["start_byte"]].decode()))))
        self.mutation("off_by_one_end", lambda p: gold(p, "P01", lambda r: r["spans"][0].update(end_byte=r["spans"][0]["end_byte"] - 1)))
        self.mutation("boolean_span_start", lambda p: gold(p, "P01", lambda r: r["spans"][0].update(start_byte=True)))
        self.mutation("wrong_span_text_hash", lambda p: gold(p, "P01", lambda r: r["spans"][0].update(text_sha256="0" * 64)))
        def split_utf8(p):
            raw = (p / "corpus/P01.txt").read_bytes(); start = raw.index("μ".encode()) + 1
            gold(p, "P01", lambda r: r["spans"][0].update(start_byte=start, end_byte=start+1, sha256=digest(raw[start:start+1])))
        self.mutation("split_multibyte", split_utf8)
        def old_revision(p):
            raw = (p / "corpus/R01-old.txt").read_bytes()
            piece = b"Revision family R01 permits p <= 2/7 and forbids equality.\r\n"
            start = raw.index(piece)
            gold(p, "V01", lambda r: r["spans"][0].update(revision_id="r1", text_sha256=digest(raw), start_byte=start, end_byte=start+len(piece), sha256=digest(piece)))
        self.mutation("coherent_wrong_current_revision", old_revision)
        def exact_mismatch(p):
            self.casefile(p, "heldout-cases.json", lambda v: next(c for c in v["cases"] if c["case_id"] == "X01").update(exact_ref="r1"))
        self.mutation("exact_revision_mismatch", exact_mismatch)
        self.mutation("absent_exact_revision_not_redirected", lambda p: self.casefile(p, "heldout-cases.json", lambda v: next(c for c in v["cases"] if c["case_id"] == "X01").update(exact_ref="missing-r9")))
        def no_answer_support(p):
            support = copy.deepcopy(self.load(p, "heldout-oracle.json")["records"][0]["spans"][0])
            gold(p, "N01", lambda r: r.update(spans=[support]))
        self.mutation("no_answer_with_support", no_answer_support)
        self.mutation("positive_without_support", lambda p: gold(p, "P01", lambda r: r.update(spans=[])))
        self.mutation("representation_hash_relabelled", lambda p: self.casefile(p, "heldout-oracle.json", lambda v: v["representation_controls"][0].update(text_sha256=v["representation_controls"][0]["source_sha256"])))
        self.mutation("representation_transform_mismatch", lambda p: self.casefile(p, "heldout-oracle.json", lambda v: v["representation_controls"][0].update(extracted_text="different", text_sha256=digest(b"different"))))
        def early(p):
            raw = (p / "corpus/D01.txt").read_bytes(); piece = EXPECTED["D01"]
            self.rebind_text(p, "corpus/D01.txt", piece + raw.replace(piece, b"", 1), "r2-early")
        self.mutation("coherent_deep_evidence_moved_early", early)


if __name__ == "__main__":
    unittest.main(verbosity=2)
