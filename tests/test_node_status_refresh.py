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
        self.assertEqual(out["source_rev"], "1111111111111111111111111111111111111111")
        self.assertNotIn(",", out["source_rev"])
        self.assertEqual(out["observed_at"], NOW)
        self.assertIsNone(out["refresh_error"])

    def test_changed_rev_is_stale(self):
        rec = load("stale.json")
        out = nsr.refresh_record(rec, offline_fetch, NOW)
        self.assertEqual(out["status"], "stale")
        self.assertEqual(out["source_rev"], rec["sources"][0]["rev"])
        self.assertEqual(out["source_rev"], "2222222222222222222222222222222222222222")
        self.assertNotIn("3333333333333333333333333333333333333333", json.dumps(out))

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
        self.assertIsNone(out["refresh_error"])

        rec = load("stale.json")
        rec["refresh_error"] = "leftover from an earlier run"
        out = nsr.refresh_record(rec, offline_fetch, NOW)
        self.assertEqual(out["status"], "stale")
        self.assertIsNone(out["refresh_error"])

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

    def test_source_rev_is_primary_and_moving_ref_is_not_stored(self):
        primary = "5" * 40
        other = "6" * 40
        moved = "7" * 40
        rec = load("fresh.json")
        rec["source_rev"] = moved
        rec["sources"] = [
            {"url": "https://github.com/d6g8k5htny-coder/main/blob/" + primary + "/old.md", "rev": primary},
            {"url": "https://github.com/d6g8k5htny-coder/Math-/tree/" + other + "/new", "rev": other},
        ]

        def fetch(url):
            if url.endswith("/old.md"):
                return primary
            return moved

        out = nsr.refresh_record(rec, fetch, NOW)
        self.assertEqual(out["status"], "stale")
        self.assertEqual(out["source_rev"], primary)
        self.assertEqual(out["sources"][0]["rev"], primary)
        self.assertEqual(out["sources"][1]["rev"], other)
        self.assertNotIn(moved, json.dumps(out))
        self.assertIsNone(out["refresh_error"])

    def test_pinned_sha_accepted_and_other_rev_forms_unknown(self):
        math_sha = "a" * 40
        rec = load("fresh.json")
        rec["sources"] = [{
            "url": "https://github.com/d6g8k5htny-coder/Math-/tree/" + math_sha + "/frontiers/downstream_gate_20260925",
            "rev": math_sha,
        }]
        out = nsr.refresh_record(rec, lambda url: math_sha, NOW)
        self.assertEqual(out["status"], "fresh")
        self.assertEqual(out["source_rev"], math_sha)
        self.assertIsNone(out["refresh_error"])

        other = load("fresh.json")
        other["sources"][0]["url"] = "https://github.com/example/repo/blob/" + ("1" * 40) + "/a.md"
        other["sources"][0]["rev"] = "1" * 40
        out = nsr.refresh_record(other, offline_fetch, NOW)
        self.assertEqual(out["status"], "unknown")
        self.assertIsInstance(out["refresh_error"], str)

        mismatch = load("fresh.json")
        mismatch["sources"][0]["rev"] = "2" * 40
        out = nsr.refresh_record(mismatch, offline_fetch, NOW)
        self.assertEqual(out["status"], "unknown")
        self.assertIn("40-hex sha in its url", out["refresh_error"])

        upper = load("fresh.json")
        upper_sha = "A" * 40
        upper["sources"] = [{
            "url": "https://github.com/d6g8k5htny-coder/main/blob/" + upper_sha + "/a.md",
            "rev": upper_sha,
        }]
        self.assertEqual(nsr.refresh_record(upper, offline_fetch, NOW)["status"], "unknown")

    def test_discussion_updated_at_newer_is_stale_and_unfetched_is_unknown(self):
        issue = nsr.refresh_record(load("issue.json"), offline_fetch, NOW)
        self.assertEqual(issue["status"], "stale")
        self.assertEqual(issue["source_rev"], "2026-10-22T17:00:00Z")
        self.assertNotIn("2026-10-22T18:00:00Z", json.dumps(issue))
        self.assertIsNone(issue["refresh_error"])

        pull = load("fresh.json")
        pull["sources"] = [{
            "url": "https://github.com/d6g8k5htny-coder/main/pull/490",
            "rev": "2026-10-22T12:00:00Z",
        }]
        pull["source_rev"] = "9999999999999999999999999999999999999999"
        same = nsr.refresh_record(pull, offline_fetch, NOW)
        self.assertEqual(same["status"], "fresh")
        self.assertEqual(same["source_rev"], "2026-10-22T12:00:00Z")

        comment_url = "https://github.com/d6g8k5htny-coder/Math-/pull/463#issuecomment-6198606495"
        comment = load("fresh.json")
        comment["sources"] = [{"url": comment_url, "rev": "2026-10-15T18:00:00Z"}]
        older = nsr.refresh_record(comment, offline_fetch, NOW)
        self.assertEqual(older["status"], "fresh")
        self.assertEqual(older["source_rev"], "2026-10-15T18:00:00Z")

        missing = load("fresh.json")
        missing["sources"] = [{
            "url": "https://github.com/d6g8k5htny-coder/main/issues/1",
            "rev": "2026-10-22T17:00:00Z",
        }]
        missing["source_rev"] = "2026-10-22T17:00:00Z"
        unknown = nsr.refresh_record(missing, offline_fetch, NOW)
        self.assertEqual(unknown["status"], "unknown")
        self.assertIsInstance(unknown["refresh_error"], str)
        self.assertIn("unreachable", unknown["refresh_error"])
        self.assertEqual(unknown["source_rev"], "2026-10-22T17:00:00Z")

        bad_rev = load("issue.json")
        bad_rev["sources"][0]["rev"] = "not-a-timestamp"
        bad = nsr.refresh_record(bad_rev, offline_fetch, NOW)
        self.assertEqual(bad["status"], "unknown")
        self.assertIn("UTC Z", bad["refresh_error"])

    def test_refresh_error_is_string_or_null(self):
        fresh = nsr.refresh_record(load("fresh.json"), offline_fetch, NOW)
        self.assertIn("refresh_error", fresh)
        self.assertIsNone(fresh["refresh_error"])
        unknown = nsr.refresh_record(load("unknown.json"), offline_fetch, NOW)
        self.assertIsInstance(unknown["refresh_error"], str)
        self.assertTrue(unknown["refresh_error"])
        self.assertIn(type(fresh["refresh_error"]), (str, type(None)))
        self.assertIn(type(unknown["refresh_error"]), (str, type(None)))

    def test_release_comment_sets_released_url_only(self):
        pickup = "https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6346922771"
        release = "https://github.com/d6g8k5htny-coder/main/pull/495#issuecomment-6347046676"
        body = "**RELEASE: nonauthor review of #495**\n\n- Releases PICKUP 6346922771.\n"
        rec = load("fresh.json")
        rec["owner"] = {"id": "agent-9", "evidence_url": pickup, "lane": "2"}

        def comments(url):
            self.assertEqual(url, pickup)
            return [
                {"url": "https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6346922770",
                 "body": "**RELEASE: too early**\n\nPICKUP 6346922771\n"},
                {"url": "https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6347000000",
                 "body": "**RELEASE: other**\n\nPICKUP 111\n"},
                {"url": release, "body": body},
            ]

        out = nsr.refresh_record(rec, offline_fetch, NOW, comments)
        self.assertEqual(out["status"], "fresh")
        self.assertEqual(out["owner"]["id"], "agent-9")
        self.assertEqual(out["owner"]["evidence_url"], pickup)
        self.assertEqual(out["owner"]["lane"], "2")
        self.assertEqual(out["owner"]["released_url"], release)
        self.assertEqual(set(out["owner"]), {"id", "evidence_url", "lane", "released_url"})

        quiet = nsr.refresh_record(rec, offline_fetch, NOW, lambda url: [])
        self.assertEqual(quiet["owner"], rec["owner"])
        self.assertNotIn("released_url", quiet["owner"])

        def boom(url):
            raise nsr.SourceError("comment listing incomplete")

        failed = nsr.refresh_record(rec, offline_fetch, NOW, boom)
        self.assertEqual(failed["status"], "unknown")
        self.assertIsInstance(failed["refresh_error"], str)
        self.assertEqual(failed["owner"], rec["owner"])

    def test_release_sets_released_url_only_for_matching_pickup_id(self):
        pickup = "https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6346922771"
        other = "https://github.com/d6g8k5htny-coder/main/pull/495#issuecomment-6347042124"
        match = "https://github.com/d6g8k5htny-coder/main/pull/495#issuecomment-6347046676"
        rec = load("fresh.json")
        rec["owner"] = {"id": "agent-9", "evidence_url": pickup}

        def refresh(body, url):
            return nsr.refresh_record(rec, offline_fetch, NOW, lambda evidence: [{"url": url, "body": body}])

        different = refresh("**RELEASE: other pickup**\n\n- Releases PICKUP 111.\n", other)
        self.assertNotIn("released_url", different["owner"])
        self.assertEqual(different["owner"], rec["owner"])

        unnamed = refresh("**RELEASE: no pickup id**\n\nThe review is finished.\n", other)
        self.assertNotIn("released_url", unnamed["owner"])
        self.assertEqual(unnamed["owner"], rec["owner"])

        matched = refresh("**RELEASE: this pickup**\n\n- Releases PICKUP 6346922771.\n", match)
        self.assertEqual(matched["owner"]["released_url"], match)
        self.assertEqual(matched["owner"]["id"], "agent-9")
        self.assertEqual(matched["owner"]["evidence_url"], pickup)

    def test_github_fetch_uses_moving_ref_for_pinned_url(self):
        seen = []

        def fake(path):
            seen.append(path)
            if "/commits?" in path:
                return [{"sha": "ab" * 20, "commit": {"committer": {"date": "2026-10-22T18:00:00+00:00"}}}]
            return {"updated_at": "2026-10-22T17:30:25+00:00"}

        nsr._api_get = fake
        pin = "1" * 40
        url = "https://github.com/d6g8k5htny-coder/Math-/blob/" + pin + "/frontiers/downstream_gate_20260925/GRAPH.json"
        self.assertEqual(nsr.github_fetch(url), "ab" * 20)
        self.assertEqual(len(seen), 1)
        self.assertIn("/repos/d6g8k5htny-coder/Math-/commits?", seen[0])
        self.assertNotIn("sha=", seen[0])
        self.assertNotIn(pin, seen[0])
        self.assertEqual(nsr.github_fetch("https://github.com/d6g8k5htny-coder/main/issues/475"), "2026-10-22T17:30:25Z")
        self.assertIn("/repos/d6g8k5htny-coder/main/issues/475", seen[-1])
        self.assertEqual(nsr.github_fetch("https://github.com/d6g8k5htny-coder/main/pull/495"), "2026-10-22T17:30:25Z")
        self.assertIn("/repos/d6g8k5htny-coder/main/pulls/495", seen[-1])
        self.assertEqual(
            nsr.github_fetch("https://github.com/d6g8k5htny-coder/main/pull/495#issuecomment-6347046676"),
            "2026-10-22T17:30:25Z")
        self.assertIn("/repos/d6g8k5htny-coder/main/issues/comments/6347046676", seen[-1])
        with self.assertRaises(nsr.SourceError):
            nsr.github_fetch("https://example.org/x")
        with self.assertRaises(nsr.SourceError):
            nsr.github_fetch("https://github.com/example/repo/blob/" + pin + "/a.md")


if __name__ == "__main__":
    unittest.main()
