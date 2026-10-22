"""Offline tests for tools/node_status_refresh.py (LANES-20261022 Lane 3). No network."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import node_status_refresh as nsr  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "node_status_refresh"
NOW = "2026-10-22T17:00:00Z"
REVS = json.loads((FIX / "current_revs.json").read_text())


def offline_fetch(url):
    if url not in REVS:
        raise nsr.SourceError("unreachable (offline fixture)")
    return REVS[url]


def load(name):
    return json.loads((FIX / name).read_text())


class RefreshTests(unittest.TestCase):
    def setUp(self):
        def no_net(*a, **k):
            raise AssertionError("network used in tests")
        self._orig = nsr._api_get
        nsr._api_get = no_net

    def tearDown(self):
        nsr._api_get = self._orig

    def test_unchanged_is_fresh(self):
        out = nsr.refresh_record(load("fresh.json"), offline_fetch, NOW)
        self.assertEqual(out["status"], "fresh")
        self.assertEqual(out["source_rev"], "aaa111")
        self.assertEqual(out["observed_at"], NOW)
        self.assertNotIn("refresh_error", out)

    def test_changed_rev_is_stale(self):
        out = nsr.refresh_record(load("stale.json"), offline_fetch, NOW)
        self.assertEqual(out["status"], "stale")
        self.assertEqual(out["source_rev"], "ccc333")

    def test_unreachable_is_unknown_with_error(self):
        rec = load("unknown.json")
        out = nsr.refresh_record(rec, offline_fetch, NOW)
        self.assertEqual(out["status"], "unknown")
        self.assertIn("unreachable", out["refresh_error"])
        self.assertEqual(out["source_rev"], rec["source_rev"])  # not advanced on failure

    def test_failure_never_fresh_even_if_other_source_ok(self):
        rec = load("fresh.json")
        rec["sources"].append({"url": "https://github.com/example/repo/issues/9", "rev": "x"})
        self.assertEqual(nsr.refresh_record(rec, offline_fetch, NOW)["status"], "unknown")

    def test_no_sources_or_bad_schema_is_unknown(self):
        rec = load("fresh.json")
        rec["sources"] = []
        self.assertEqual(nsr.refresh_record(rec, offline_fetch, NOW)["status"], "unknown")
        rec = load("fresh.json")
        rec["schema_version"] = 1
        self.assertEqual(nsr.refresh_record(rec, offline_fetch, NOW)["status"], "unknown")

    def test_owner_and_scientific_effect_untouched(self):
        rec = load("stale.json")
        rec["owner"] = {"id": "agent", "evidence_url": "https://github.com/example/repo/issues/2#issuecomment-1"}
        out = nsr.refresh_record(rec, offline_fetch, NOW)
        self.assertEqual(out["owner"], rec["owner"])
        self.assertEqual(out["scientific_effect"], "NONE")

    def test_dir_report_only_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n in ("fresh.json", "stale.json", "unknown.json"):
                shutil.copy(FIX / n, tmp)
            before = {p.name: p.read_bytes() for p in Path(tmp).glob("*.json")}
            report = nsr.refresh_dir(Path(tmp), offline_fetch, write=False, now=NOW)
            self.assertEqual({r["file"]: r["status"] for r in report},
                             {"fresh.json": "fresh", "stale.json": "stale", "unknown.json": "unknown"})
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(tmp).glob("*.json")})

    def test_dir_write_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(FIX / "stale.json", tmp)
            nsr.refresh_dir(Path(tmp), offline_fetch, write=True, now=NOW)
            self.assertEqual(json.loads((Path(tmp) / "stale.json").read_text())["status"], "stale")

    def test_unsupported_url_raises_source_error(self):
        with self.assertRaises(nsr.SourceError):
            nsr.github_fetch("https://example.org/x")

    def test_scientific_effect_forced_none_not_passed_through(self):
        rec = load("fresh.json")
        rec["scientific_effect"] = "ACCEPTED"
        out = nsr.refresh_record(rec, offline_fetch, NOW)
        self.assertEqual(out["status"], "fresh")
        self.assertEqual(out["scientific_effect"], "NONE")
        self.assertNotEqual(out["scientific_effect"], "ACCEPTED")

        rec = load("unknown.json")
        rec["scientific_effect"] = "ACCEPTED"
        out = nsr.refresh_record(rec, offline_fetch, NOW)
        self.assertEqual(out["status"], "unknown")
        self.assertEqual(out["scientific_effect"], "NONE")

    def test_successful_refresh_clears_stale_refresh_error(self):
        rec = load("fresh.json")
        rec["refresh_error"] = "leftover from an earlier run"
        out = nsr.refresh_record(rec, offline_fetch, NOW)
        self.assertEqual(out["status"], "fresh")
        self.assertNotIn("refresh_error", out)

        rec = load("stale.json")
        rec["refresh_error"] = "leftover from an earlier run"
        out = nsr.refresh_record(rec, offline_fetch, NOW)
        self.assertEqual(out["status"], "stale")
        self.assertNotIn("refresh_error", out)

    def test_empty_or_missing_source_revision_never_fresh(self):
        rec = load("fresh.json")

        def empty_fetch(url):
            return ""

        out = nsr.refresh_record(rec, empty_fetch, NOW)
        self.assertEqual(out["status"], "unknown")
        self.assertIn("refresh_error", out)
        self.assertNotEqual(out["status"], "fresh")

        def missing_fetch(url):
            return None

        out = nsr.refresh_record(load("fresh.json"), missing_fetch, NOW)
        self.assertEqual(out["status"], "unknown")
        self.assertNotEqual(out["status"], "fresh")

        missing = load("fresh.json")
        del missing["sources"][0]["rev"]
        out = nsr.refresh_record(missing, offline_fetch, NOW)
        self.assertEqual(out["status"], "unknown")
        self.assertIn("refresh_error", out)
        self.assertNotEqual(out["status"], "fresh")

        blank = load("fresh.json")
        blank["sources"][0]["rev"] = ""
        out = nsr.refresh_record(blank, offline_fetch, NOW)
        self.assertEqual(out["status"], "unknown")
        self.assertNotEqual(out["status"], "fresh")


if __name__ == "__main__":
    unittest.main()
