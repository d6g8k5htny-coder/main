"""Q-9006 offline contract tests; synthetic observations never earn live credit.

The real-record/live companion is intentionally absent in the RED publication.
These tests exercise its future real helper, with only the external observation
boundary replaced. They never perform HTTP, change real records, or activate
the existing dormant real-record test.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "tools" / "node_status_live_check.py"
RECORD_NAMES = (
    "eng.d0-packet-allowlist-fix.json",
    "eng.hard-gate.json",
    "math.lifetime-remainder.json",
    "math.side24-coefficient.json",
)
FRESH_NODES = [
    "eng.d0-packet-allowlist-fix",
    "math.lifetime-remainder",
    "math.side24-coefficient",
]
HEAD = "f532c3e2d91a0c7679a37c733741660917a9b21b"
COMMENT = "https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6351719961"
COMMENT_TIME = "2026-10-22T21:28:51Z"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class SyntheticSourceObserver:
    """Complete production-shaped external replies; not real live observations."""
    def __init__(self, *, changed=None, unavailable=None, hook=None,
                 omit_observation=False, wrong_observation_url=False,
                 missing_head=False):
        self.heads = {} if missing_head else {"Math-": HEAD}
        self.observations = []
        self.changed = changed
        self.unavailable = unavailable
        self.hook = hook
        self.omit_observation = omit_observation
        self.wrong_observation_url = wrong_observation_url
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if self.hook:
            hook, self.hook = self.hook, None
            hook()
        if url == self.unavailable:
            raise OSError("SYNTHETIC_UNAVAILABLE_SOURCE")
        if "/blob/" in url or "/tree/" in url:
            kind = "tree" if "/tree/" in url else "blob"
            mode = "040000" if kind == "tree" else "100644"
            old = (kind, mode, "a" * 40)
            new = (kind, mode, ("b" if url == self.changed else "a") * 40)
            observation = {
                "url": url, "head": HEAD,
                "pinned_object": old, "current_object": new,
            }
            answer = {"object_equal": old == new}
        else:
            stamp = "2026-10-22T21:28:52Z" if url == self.changed else COMMENT_TIME
            observation = {"url": url, "updated_at": stamp}
            answer = stamp
        if self.wrong_observation_url:
            observation["url"] = "https://example.invalid/not-the-requested-source"
        if not self.omit_observation:
            self.observations.append(observation)
        return answer


class NodeStatusLiveContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="q9006-contract-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for relative in ("architecture/node-status.schema.json", "docs/site/current.json"):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        self.records = self.root / "records" / "node_status"
        self.records.mkdir(parents=True)
        for name in RECORD_NAMES:
            shutil.copyfile(ROOT / "records" / "node_status" / name, self.records / name)
        self.identity = {
            "repository": "d6g8k5htny-coder/main",
            "checked_commit": "1902aac113761974c4569f78acb6a8baadabb7f1",
            "run_id": "123456",
            "run_attempt": "1",
            "python_optimize": sys.flags.optimize,
        }

    def subject(self):
        self.assertTrue(SUBJECT.is_file(), "Q-9006 live companion implementation is absent")
        spec = importlib.util.spec_from_file_location("q9006_contract_subject", SUBJECT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        sys.path.insert(0, str(ROOT / "tools"))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        self.assertTrue(callable(getattr(module, "evaluate_fresh_records", None)))
        self.assertTrue(callable(getattr(module, "run_live", None)))
        self.assertTrue(issubclass(module.LiveCheckError, ValueError))
        return module

    def snapshot(self):
        return {p.name: p.read_bytes() for p in sorted(self.records.glob("*.json"))}

    def edit(self, name, change):
        path = self.records / name
        record = json.loads(path.read_text())
        change(record)
        path.write_text(json.dumps(record, indent=2) + "\n")

    def evaluate(self, module, observer=None, identity=None):
        return module.evaluate_fresh_records(
            self.root,
            observer=SyntheticSourceObserver() if observer is None else observer,
            identity=self.identity if identity is None else identity,
        )

    def test_complete_real_input_cohort_is_checked_without_rewriting_pins(self):
        # Catches dropping a real record, checking only one fresh node, or rebasing pins.
        module = self.subject()
        before = self.snapshot()
        result = self.evaluate(module)
        self.assertEqual(result["records_total"], 4)
        self.assertEqual(result["fresh_candidates"], 3)
        self.assertEqual(result["fresh_checked"], 3)
        self.assertEqual(result["fresh_nodes"], FRESH_NODES)
        self.assertEqual(result["nonfresh_records"], [
            {"node_id": "eng.hard-gate", "declared_status": "stale"}
        ])
        self.assertEqual(result["input_sha256"], {name: sha(raw) for name, raw in before.items()})
        self.assertEqual(result["identity"], self.identity)
        self.assertEqual(result["observed_heads"], {"Math-": HEAD})
        self.assertEqual(result["source_observation_count"], 10)
        self.assertEqual(len(result["source_observations"]), 10)
        self.assertEqual(result["scientific_effect"], "NONE")
        self.assertIs(result["cbz_green"], False)
        self.assertEqual(self.snapshot(), before)

    def test_missing_any_of_the_four_real_record_files_refuses(self):
        # Catches accepting a partial directory, including omission of the newer fourth record.
        module = self.subject()
        for name in RECORD_NAMES:
            with self.subTest(name=name):
                path = self.records / name
                raw = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaises(module.LiveCheckError):
                        self.evaluate(module)
                finally:
                    path.write_bytes(raw)

    def test_invalid_schema_refuses_before_source_observation(self):
        # Catches treating syntactically readable but unadopted records as valid inputs.
        module = self.subject()
        self.edit("eng.hard-gate.json", lambda record: record.update(artifacts=["invented.txt"]))
        observer = SyntheticSourceObserver()
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module, observer)
        self.assertEqual(observer.calls, [])

    def test_missing_schema_refuses(self):
        module = self.subject()
        (self.root / "architecture/node-status.schema.json").unlink()
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module)

    def test_duplicate_record_json_keys_refuse(self):
        module = self.subject()
        path = self.records / "eng.hard-gate.json"
        raw = path.read_text()
        path.write_text(raw.replace('"owner": null', '"owner": null, "owner": null', 1))
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module)

    def test_unexpected_record_file_is_not_ignored(self):
        module = self.subject()
        shutil.copyfile(self.records / "eng.hard-gate.json", self.records / "unexpected.json")
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module)

    def test_schema_byte_drift_refuses(self):
        module = self.subject()
        path = self.root / "architecture/node-status.schema.json"
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module, SyntheticSourceObserver(
                hook=lambda: path.write_bytes(path.read_bytes() + b"\n")))

    def test_graph_byte_drift_refuses(self):
        module = self.subject()
        path = self.root / "docs/site/current.json"
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module, SyntheticSourceObserver(
                hook=lambda: path.write_bytes(path.read_bytes() + b"\n")))

    def test_no_declared_fresh_record_is_not_an_empty_success(self):
        module = self.subject()
        for name in RECORD_NAMES:
            self.edit(name, lambda record: record.update(status="stale"))
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module)

    def test_changed_cited_git_object_refuses(self):
        module = self.subject()
        url = "https://github.com/d6g8k5htny-coder/Math-/blob/dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde/coefficients/side24_v1/PROOF.md"
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module, SyntheticSourceObserver(changed=url))

    def test_changed_cited_discussion_timestamp_refuses(self):
        module = self.subject()
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module, SyntheticSourceObserver(changed=COMMENT))

    def test_unavailable_source_cannot_become_fresh(self):
        module = self.subject()
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module, SyntheticSourceObserver(unavailable=COMMENT))

    def test_record_byte_drift_even_with_unchanged_json_value_refuses(self):
        module = self.subject()
        path = self.records / "eng.d0-packet-allowlist-fix.json"
        observer = SyntheticSourceObserver(hook=lambda: path.write_bytes(path.read_bytes() + b"\n"))
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module, observer)

    def test_record_set_drift_refuses(self):
        module = self.subject()
        path = self.records / "math.lifetime-remainder.json"
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module, SyntheticSourceObserver(hook=path.unlink))

    def test_success_without_observation_counter_coverage_refuses(self):
        # The helper must check evidence completeness, not just a returned freshness label.
        module = self.subject()
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module, SyntheticSourceObserver(omit_observation=True))

    def test_correct_count_with_wrong_source_observation_urls_refuses(self):
        module = self.subject()
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module, SyntheticSourceObserver(wrong_observation_url=True))

    def test_observation_without_captured_default_head_refuses(self):
        module = self.subject()
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module, SyntheticSourceObserver(missing_head=True))

    def test_observation_head_must_match_the_captured_repository_cut(self):
        module = self.subject()
        class InconsistentHeadObserver(SyntheticSourceObserver):
            def __call__(self, url):
                answer = super().__call__(url)
                if "head" in self.observations[-1]:
                    self.observations[-1]["head"] = "e" * 40
                return answer
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module, InconsistentHeadObserver())

    def test_same_count_and_url_set_with_wrong_multiplicities_refuses(self):
        module = self.subject()
        graph_url = "https://github.com/d6g8k5htny-coder/Math-/blob/dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde/frontiers/downstream_gate_20260925/GRAPH.json"
        other_url = "https://github.com/d6g8k5htny-coder/Math-/blob/dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde/coefficients/side24_v1/PROOF.md"
        class WrongMultiplicityObserver(SyntheticSourceObserver):
            replaced = False
            def __call__(self, url):
                answer = super().__call__(url)
                if url == graph_url and not self.replaced:
                    self.observations[-1]["url"] = other_url
                    self.replaced = True
                return answer
        with self.assertRaises(module.LiveCheckError):
            self.evaluate(module, WrongMultiplicityObserver())

    def test_malformed_run_identity_refuses(self):
        module = self.subject()
        for field, bad in (
            ("repository", "other/repository"), ("checked_commit", "main"),
            ("run_id", "0"), ("run_attempt", "0"), ("python_optimize", True),
        ):
            with self.subTest(field=field):
                identity = dict(self.identity, **{field: bad})
                with self.assertRaises(module.LiveCheckError):
                    self.evaluate(module, identity=identity)

    def test_live_entry_refuses_missing_authentication_before_constructing_observer(self):
        module = self.subject()
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(module, "ContentObserver", side_effect=AssertionError("network boundary entered")):
                with self.assertRaises(module.LiveCheckError):
                    module.run_live(self.root, identity=self.identity)

    def test_live_entry_uses_one_default_observer_for_the_whole_mode(self):
        # Only the external production observer constructor is replaced; the helper stays real.
        module = self.subject()
        observer = SyntheticSourceObserver()
        with patch.dict(os.environ, {"GITHUB_TOKEN": "SYNTHETIC_NOT_A_CREDENTIAL"}):
            with patch.object(module, "ContentObserver", return_value=observer) as factory:
                result = module.run_live(self.root, identity=self.identity)
        self.assertEqual(result["fresh_checked"], 3)
        self.assertEqual(result["source_observation_count"], 10)
        factory.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
