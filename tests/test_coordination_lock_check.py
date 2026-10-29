"""CLI regressions using real pinned Git objects; no mathematical code runs."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/coordination_lock_check.py"
PATH = "docs/site/node-status.json"
ORIGIN = "https://github.com/d6g8k5htny-coder/main.git"
NOW = "2026-10-22T22:00:00Z"
EVIDENCE = "https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6353027897"
RELEASE = "https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6353027898"


def record():
    return {
        "schema_version": 0, "node_id": "eng.audit", "owner": {"id": "alice", "evidence_url": EVIDENCE},
        "next_action": "Refresh the native task discussion.", "missing_input": [], "stops": [],
        "sources": [{"url": EVIDENCE, "rev": "2026-10-22T21:59:00Z"}],
        "source_rev": "2026-10-22T21:59:00Z", "observed_at": "2026-10-22T21:59:00Z",
        "status": "fresh", "scientific_effect": "NONE",
    }


class CoordinationLockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("remote", "add", "origin", ORIGIN)
        self.value = record()
        self.commit = self.pin({"eng.audit": self.value})

    def git(self, *args):
        result = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, check=True)
        return result.stdout.decode().strip()

    def pin(self, data=None, raw=None):
        path = self.root / PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            path.unlink()
        path.write_bytes(raw if raw is not None else json.dumps(data, ensure_ascii=True).encode())
        self.git("add", "--", PATH)
        self.git("commit", "-qm", "fixture", "--allow-empty")
        self.raw = path.read_bytes()
        return self.git("rev-parse", "HEAD")

    def run_report(self, *, claimant="alice", commit=None, now=NOW, node="eng.audit", env=None):
        args = [sys.executable, "-B", "-S", str(SCRIPT), "report", "--commit", commit or self.commit, "--node", node]
        if claimant is not None:
            args += ["--claimant", claimant]
        if now is not None:
            args += ["--now", now]
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        process = subprocess.run(args, cwd=self.root, capture_output=True, env=run_env, timeout=10)
        self.assertIn(process.returncode, (0, 2), process.stderr.decode())
        self.assertEqual(len(process.stdout.splitlines()), 1, process.stderr.decode())
        self.assertTrue(process.stdout.isascii())
        result = json.loads(process.stdout)
        self.assertIs(result["authority_granted"], False)
        self.assertIs(result["universal_lock"], False)
        self.assertIs(result["advisory_only"], True)
        self.assertIs(result["native_reconciliation_required"], True)
        self.assertEqual(result["scientific_effect"], "NONE")
        return process, result

    def update(self, **changes):
        self.value.update(changes)
        self.commit = self.pin({"eng.audit": self.value})

    def hold(self, **args):
        process, result = self.run_report(**args)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(result["disposition"], "HOLD")
        self.assertTrue(result["blockers"])
        return result

    def test_matching_owner_reports_exact_pinned_source_and_age_without_authority(self):
        process, result = self.run_report()
        self.assertEqual(process.returncode, 0)
        self.assertEqual(result["disposition"], "OBSERVED")
        self.assertEqual(result["owner"], self.value["owner"])
        self.assertEqual(result["status"], "fresh")
        self.assertEqual(result["observed_age_seconds"], 60)
        self.assertEqual(result["sources"], self.value["sources"])
        self.assertEqual(result["source_rev"], self.value["source_rev"])
        self.assertEqual(result["source"]["commit"], self.commit)
        self.assertEqual(result["source"]["blob"], self.git("rev-parse", self.commit + ":" + PATH))
        self.assertEqual(result["source"]["sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(result["source"]["url"], "https://github.com/d6g8k5htny-coder/main/blob/" + self.commit + "/" + PATH)

    def test_dirty_tree_replacement_is_ignored(self):
        (self.root / PATH).write_text('{"eng.audit":{"owner":null}}')
        process, result = self.run_report()
        self.assertEqual(process.returncode, 0)
        self.assertEqual(result["owner"]["id"], "alice")
        self.assertEqual(result["source"]["sha256"], hashlib.sha256(self.raw).hexdigest())

    def test_different_owner_is_conflict(self):
        process, result = self.run_report(claimant="bob")
        self.assertEqual(process.returncode, 2)
        self.assertEqual(result["disposition"], "CONFLICT")
        self.assertEqual(result["owner"]["id"], "alice")

    def test_release_url_preserves_owner_and_requires_explicit_handoff(self):
        owner = dict(self.value["owner"], released_url=RELEASE)
        self.update(owner=owner)
        process, result = self.run_report()
        self.assertEqual(process.returncode, 2)
        self.assertEqual(result["disposition"], "EXPLICIT_HANDOFF_REQUIRED")
        self.assertEqual(result["owner"], owner)
        _, other = self.run_report(claimant="bob")
        self.assertEqual(other["disposition"], "CONFLICT")

    def test_missing_or_blank_claimant_is_observe_only(self):
        for claimant in (None, ""):
            with self.subTest(claimant=claimant):
                process, result = self.run_report(claimant=claimant)
                self.assertEqual(process.returncode, 0)
                self.assertIsNone(result["claimant_id"])

    def test_null_owner_is_hold(self):
        self.update(owner=None)
        result = self.hold()
        self.assertIsNone(result["owner"])

    def test_missing_record_is_hold(self):
        self.hold(node="eng.missing")

    def test_missing_input_and_stops_are_preserved_and_hold(self):
        self.update(stops=["Permission HOLD remains.\n::error::untrusted"], missing_input=["Fresh native task reconciliation."])
        result = self.hold()
        self.assertEqual(result["stops"], self.value["stops"])
        self.assertEqual(result["missing_input"], self.value["missing_input"])

    def test_stale_age_holds_without_expiring_owner(self):
        self.update(observed_at="2026-10-22T20:59:59Z")
        result = self.hold()
        self.assertEqual(result["status"], "stale")
        self.assertEqual(result["observed_age_seconds"], 3601)
        self.assertEqual(result["owner"]["id"], "alice")

    def test_freshness_window_boundary_is_observation_only(self):
        self.update(observed_at="2026-10-22T21:00:00Z")
        process, result = self.run_report()
        self.assertEqual(process.returncode, 0)
        self.assertEqual(result["observed_age_seconds"], 3600)

    def test_recorded_stale_and_unknown_hold_even_when_recent(self):
        for status in ("stale", "unknown"):
            with self.subTest(status=status):
                self.update(status=status)
                result = self.hold()
                self.assertEqual(result["status"], status)

    def test_future_observation_is_unknown_and_hold(self):
        self.update(observed_at="2026-10-22T22:00:01Z")
        result = self.hold()
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["observed_age_seconds"], -1)

    def test_times_require_real_strict_utc(self):
        for value in ("2026-10-22T21:59:00+00:00", "2026-10-22T21:59:00", "2026-02-30T21:59:00Z"):
            with self.subTest(value=value):
                self.update(observed_at=value)
                self.hold()
                self.hold(now=value)

    def test_unknown_schema_and_extra_fields_hold(self):
        for change in ({"schema_version": 1}, {"schema_version": True}, {"unrecognized": "value"}):
            with self.subTest(change=change):
                self.value = record()
                self.update(**change)
                self.hold()

    def test_duplicate_keys_and_invalid_json_hold(self):
        for raw in (b'{"eng.audit":{},"eng.audit":{}}', b'{"eng.audit":{"owner":null,"owner":{}}}', b'{', b'NaN', b'{} trailing'):
            with self.subTest(raw=raw):
                self.commit = self.pin(raw=raw)
                self.hold()

    def test_source_revision_identity_mismatch_holds(self):
        self.update(source_rev="2026-10-22T21:58:00Z")
        self.hold()
        self.value = record()
        self.update(sources=[{"url": "https://github.com/d6g8k5htny-coder/main/blob/" + "a" * 40 + "/record.json", "rev": "b" * 40}], source_rev="b" * 40)
        self.hold()

    def test_numeric_parse_anomalies_are_blocked_before_raw_fields_are_reported(self):
        for raw_value in (b"1e9999", b"-1e9999", b"1" * 10000):
            with self.subTest(raw_value=raw_value[:20]):
                self.commit = self.pin(raw=b'{"eng.audit":{"owner":' + raw_value + b'}}')
                self.hold()

    def test_wrong_remote_holds_before_reading_record(self):
        self.git("remote", "set-url", "origin", "https://github.com/foreign/private.git")
        result = self.hold()
        self.assertIsNone(result["owner"])

    def test_commit_must_be_exact_full_existing_commit(self):
        for value in ("HEAD", self.commit[:12], "f" * 40, "--help"):
            with self.subTest(value=value):
                self.hold(commit=value)

    def test_record_node_identity_must_match_map_key(self):
        self.update(node_id="eng.other")
        self.hold()

    def test_nonregular_blob_is_refused(self):
        path = self.root / PATH
        path.unlink()
        path.symlink_to("some-other-file")
        self.git("add", "--", PATH)
        self.git("commit", "-qm", "symlink")
        self.commit = self.git("rev-parse", "HEAD")
        self.hold()

    def test_missing_blob_is_hold(self):
        self.git("rm", "-q", "--", PATH)
        self.git("commit", "-qm", "missing")
        self.commit = self.git("rev-parse", "HEAD")
        self.hold()

    def test_repeated_matching_pickups_never_manufacture_authority_or_write(self):
        before = self.git("status", "--porcelain=v1", "--untracked-files=all")
        first = self.run_report()[1]
        second = self.run_report()[1]
        self.assertEqual(first, second)
        self.assertEqual(self.git("status", "--porcelain=v1", "--untracked-files=all"), before)

    def test_oversize_blob_is_hold(self):
        self.commit = self.pin(raw=b" " * 1_048_577)
        self.hold()

    def test_git_environment_cannot_redirect_source(self):
        process, result = self.run_report(env={"GIT_DIR": "/nonexistent", "GIT_WORK_TREE": "/nonexistent", "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "remote.origin.url", "GIT_CONFIG_VALUE_0": "https://foreign.invalid/repo"})
        self.assertEqual(process.returncode, 0)
        self.assertEqual(result["owner"]["id"], "alice")

    def test_git_blob_replacement_cannot_substitute_null_owner(self):
        original = self.git("rev-parse", self.commit + ":" + PATH)
        replacement = self.root / "replacement.json"
        replacement.write_text('{"eng.audit":{"owner":null}}')
        replacement_blob = self.git("hash-object", "-w", str(replacement))
        self.git("replace", original, replacement_blob)
        process, result = self.run_report()
        self.assertEqual(process.returncode, 0)
        self.assertEqual(result["owner"]["id"], "alice")
        self.assertEqual(result["source"]["blob"], original)
        self.assertEqual(result["source"]["sha256"], hashlib.sha256(self.raw).hexdigest())

    def test_runtime_clock_is_current_utc_when_now_is_omitted(self):
        observed = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=60)
        self.update(observed_at=observed.strftime("%Y-%m-%dT%H:%M:%SZ"))
        process, result = self.run_report(now=None)
        self.assertEqual(process.returncode, 0)
        self.assertLess(abs(result["observed_age_seconds"] - 60), 10)
        self.assertRegex(result["evaluated_at"], r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")

    def test_malformed_record_fields_and_refresh_errors_hold(self):
        for change in ({"owner": {}}, {"stops": "hold"}, {"sources": []},
                       {"status": {}}, {"scientific_effect": "PROVED"},
                       {"refresh_error": "native refresh failed"},
                       {"carried_forward": [{"id": "finding", "text": "Unresolved",
                           "source_url": EVIDENCE, "status": "open"}]}):
            with self.subTest(change=change):
                self.value = record()
                self.update(**change)
                self.hold()


if __name__ == "__main__":
    unittest.main()
