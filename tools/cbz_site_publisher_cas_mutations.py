"""Bounded atomic-publication mutations with selected unittest-origin checks."""
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "tools/current_site_publication.py"
TEST = ROOT / "tests/test_current_site_publication.py"
DATA = ROOT / "tools/current_site_data.py"
RECORDER = ROOT / "tools/cbz/reference_crawler_mutation_check.py"
RECORDER_SHA256 = "14b770828aa566b529be64214a0d99e5cfa4b762be792f07af04b39985ce0481"

MUTANTS = [
    ("before-head", "test_atomic_ref_protocol_binds_repo_ref_heads_and_nonforce",
     "'beforeOid': before", "'beforeOid': after", 1),
    ("force", "test_atomic_ref_protocol_binds_repo_ref_heads_and_nonforce",
     "'afterOid': after, 'force': False", "'afterOid': after, 'force': True", 1),
    ("repository", "test_atomic_ref_protocol_binds_repo_ref_heads_and_nonforce",
     "'repositoryId': self.REPOSITORY_ID", "'repositoryId': 'wrong-repository'", 1),
    ("ref-name", "test_atomic_ref_protocol_binds_repo_ref_heads_and_nonforce",
     "'name': REF_NAME", "'name': 'refs/heads/main'", 1),
    ("response-errors", "test_atomic_ref_protocol_refuses_errors_and_wrong_response_identity",
     "'errors' in obj", "False", 1),
    ("response-origin", "test_atomic_ref_protocol_refuses_errors_and_wrong_response_identity",
     "response.geturl() != self.ORIGIN", "False", 1),
    ("response-status", "test_atomic_ref_protocol_refuses_errors_and_wrong_response_identity",
     "response.status != 200", "False", 1),
    ("zero-head", "test_atomic_ref_protocol_rejects_invalid_heads_without_network",
     "before == '0' * 40", "False", 1),
    ("rollback-bypass", "test_rollback_between_read_and_ref_update_requires_exact_head",
     "api.advance_ref(old_head, commit)", 'api.request("PATCH", REF_WRITE, {"sha": commit, "force": False})', 1),
    ("raw-rest-allowed", "test_raw_rest_ref_patch_is_refused_without_network",
     '    raise ValueError("nonmetadata GitHub route refused")',
     '    if method == "PATCH" and path == REF_WRITE:\n        return\n    raise ValueError("nonmetadata GitHub route refused")', 1),
    ("request-bound", "test_atomic_advance_obeys_request_bound_without_network",
     "if self.requests >= 64:", "if False:", 1),
]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_occurrence(source, old, new, number):
    parts = source.split(old)
    require(0 < number < len(parts), "mutation anchor missing")
    return old.join(parts[:number]) + new + old.join(parts[number:])


def load_recorder():
    require(digest(RECORDER.read_bytes()) == RECORDER_SHA256, "origin recorder changed")
    spec = importlib.util.spec_from_file_location("publisher_cas_origin", RECORDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute(recorder, source, tests, name, directory):
    with tempfile.TemporaryDirectory(prefix="publisher-cas-mutation-") as temporary:
        work = Path(temporary)
        (work / "tools").mkdir()
        (work / "tests").mkdir()
        (work / "tests/__init__.py").write_bytes(b"")
        subject = work / "tools/current_site_publication.py"
        test = work / "tests/test_current_site_publication.py"
        subject.write_text(source)
        test.write_text(tests)
        shutil.copyfile(DATA, work / "tools/current_site_data.py")
        outcome, detail, failed, errored = recorder.run_suite(
            work, [name], selector="tests.test_current_site_publication.PublicationTests." + name,
            copy_test=False)
        report = recorder.LAST_ORIGIN_REPORT
        require(isinstance(report, dict), "missing origin report")
        require(subject.read_text() == source and test.read_text() == tests, "child changed source")
        (directory / "origin.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return outcome, detail, failed, errored, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    recorder = load_recorder()
    source, tests = SUBJECT.read_text(), TEST.read_text()
    results = []
    for label, name, old, new, occurrence in MUTANTS:
        directory = out / label
        directory.mkdir()
        clean = directory / "baseline"
        clean.mkdir()
        outcome = execute(recorder, source, tests, name, clean)
        require(outcome[0] == "pass" and not outcome[2] and not outcome[3],
                "baseline failed " + label)
        mutant = replace_occurrence(source, old, new, occurrence)
        require(mutant != source, "unchanged mutant " + label)
        (directory / "mutant.py").write_text(mutant)
        altered = directory / "mutant"
        altered.mkdir()
        outcome = execute(recorder, mutant, tests, name, altered)
        require(outcome[0] == "fail" and outcome[2] == {name} and not outcome[3],
                "selected oracle did not reject mutant " + label)
        results.append({"mutation": label, "selected": name, "sha256": digest(mutant.encode()),
                        "baseline": "pass", "mutant": "authorized-selected-failure"})

    adverse = [
        ("subject", "test_atomic_ref_protocol_binds_repo_ref_heads_and_nonforce",
         "    def advance(self, before, after):\n",
         '    def advance(self, before, after):\n        raise AssertionError("subject-origin probe")\n',
         "source"),
        ("helper", "test_rollback_between_read_and_ref_update_requires_exact_head",
         "    def advance_ref(self, before, after):\n",
         '    def advance_ref(self, before, after):\n        raise AssertionError("helper-origin probe")\n',
         "test"),
        ("lifecycle", "test_atomic_ref_protocol_binds_repo_ref_heads_and_nonforce",
         '        self.assertTrue(TOOL.is_file(), "publication receipt tool has not been implemented")\n',
         '        raise AssertionError("lifecycle-origin probe")\n',
         "test"),
    ]
    for label, selected, old, new, location in adverse:
        directory = out / ("origin-" + label)
        directory.mkdir()
        changed_source = replace_occurrence(source, old, new, 1) if location == "source" else source
        changed_tests = replace_occurrence(tests, old, new, 1) if location == "test" else tests
        outcome = execute(recorder, changed_source, changed_tests, selected, directory)
        report = outcome[4]
        require(outcome[0] == "invalid" and report["failure_count"] == 1
                and report["error_count"] == 0 and report["unauthorized_failures"] == [selected]
                and not report["authorized_failures"], "assertion origin was not rejected " + label)
        results.append({"mutation": "origin-" + label, "selected": selected,
                        "mutant": "rejected-unauthorized-origin"})

    directory = out / "error-only"
    directory.mkdir()
    broken = replace_occurrence(source, "    def advance(self, before, after):\n",
        '    def advance(self, before, after):\n        raise RuntimeError("error-only probe")\n', 1)
    outcome = execute(recorder, broken, tests,
                      "test_atomic_ref_protocol_binds_repo_ref_heads_and_nonforce", directory)
    require(outcome[0] == "invalid" and outcome[4]["error_count"] == 1,
            "runtime error was counted as a kill")
    results.append({"mutation": "error-only", "mutant": "rejected-error-origin"})
    require(len(results) == len(MUTANTS) + len(adverse) + 1, "incomplete campaign")
    (out / "results.json").write_text(json.dumps({
        "optimize": sys.flags.optimize, "source_sha256": digest(source.encode()),
        "test_sha256": digest(tests.encode()), "recorder_sha256": RECORDER_SHA256,
        "results": results}, indent=2) + "\n")
    print(len(MUTANTS), "clean baselines and selected-oracle mutations;",
          len(adverse), "unauthorized assertion origins and one runtime error rejected; optimize",
          sys.flags.optimize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
