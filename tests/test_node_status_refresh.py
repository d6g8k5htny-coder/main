"""Offline tests for tools/node_status_refresh.py (LANES-20261022 Lane 3). No network."""
import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

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

    def test_pinned_url_trailing_newline_is_not_pinned(self):
        sha = "b" * 40
        url = "https://github.com/d6g8k5htny-coder/main/blob/" + sha + "/a.md"
        seen = []

        def fetch(got):
            seen.append(got)
            return sha

        good = load("fresh.json")
        good["sources"] = [{"url": url, "rev": sha}]
        out = nsr.refresh_record(good, fetch, NOW)
        self.assertEqual(nsr._classify(url)[0], "sha")
        self.assertEqual(out["status"], "fresh")
        self.assertEqual(out["source_rev"], sha)
        self.assertEqual(seen, [url])

        seen.clear()
        newline = url + "\n"
        self.assertIsNone(nsr._classify(newline)[0])
        bad = load("fresh.json")
        bad["sources"] = [{"url": newline, "rev": sha}]
        rejected = nsr.refresh_record(bad, fetch, NOW)
        self.assertEqual(rejected["status"], "unknown")
        self.assertIn("not a pinned", rejected["refresh_error"])
        self.assertEqual(seen, [])

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

    def test_report_only_includes_discovered_released_url(self):
        pickup = "https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6346922771"
        release = "https://github.com/d6g8k5htny-coder/main/pull/495#issuecomment-6347046676"
        body = "**RELEASE: nonauthor review**\n\n- Releases PICKUP 6346922771.\n"
        rec = load("fresh.json")
        rec["owner"] = {"id": "agent-9", "evidence_url": pickup}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fresh.json"
            path.write_text(json.dumps(rec), encoding="utf-8")
            before = path.read_bytes()
            report = nsr.refresh_dir(
                Path(tmp), offline_fetch, write=False, now=NOW,
                comments=lambda url: [{"url": release, "body": body}])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(report[0]["status"], "fresh")
            self.assertEqual(report[0].get("released_url"), release)
            quiet = nsr.refresh_dir(
                Path(tmp), offline_fetch, write=False, now=NOW, comments=lambda url: [])
            self.assertNotIn("released_url", quiet[0])
            self.assertEqual(path.read_bytes(), before)

    def test_duplicate_json_key_is_unknown_and_not_rewritten(self):
        text = json.dumps(load("fresh.json"))
        owner = text.replace('"owner": null', '"owner": {"id": "unbacked"}', 1)
        raw_owner = owner[:-1] + ',"owner": null}'
        item = ('{"id": "hold-1", "text": "Unresolved synthetic item", "status": "open", '
                '"source_url": "https://github.com/example/repo/issues/1"}')
        raw_carried = text[:-1] + ',"carried_forward": [' + item + '], "carried_forward": []}'
        calls = []

        def fetch(url):
            calls.append(url)
            return offline_fetch(url)

        for label, raw, key in (
            ("owner", raw_owner, "owner"),
            ("carried_forward", raw_carried, "carried_forward"),
        ):
            with self.subTest(label=label):
                calls.clear()
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "fresh.json"
                    path.write_text(raw, encoding="utf-8")
                    before = path.read_bytes()
                    for write in (False, True):
                        report = nsr.refresh_dir(Path(tmp), fetch, write=write, now=NOW)
                        self.assertEqual(len(report), 1)
                        self.assertEqual(report[0]["status"], "unknown")
                        self.assertIn("duplicate JSON key: " + key, report[0]["refresh_error"])
                        self.assertEqual(path.read_bytes(), before)
                        self.assertEqual(calls, [])

    def test_github_fetch_preserves_discussion_timestamp_endpoints(self):
        seen = []

        def fake(path):
            seen.append(path)
            return {"updated_at": "2026-10-22T17:30:25+00:00"}

        nsr._api_get = fake
        pin = "1" * 40
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


class GitObjectFreshnessTests(unittest.TestCase):
    """Exercise the real refresher with fixed, hand-derived GitHub API objects."""

    pin, tip, older = "1" * 40, "2" * 40, "9" * 40
    pin_tree, tip_tree = "3" * 40, "4" * 40
    pin_dir, tip_dir = "5" * 40, "6" * 40
    blob, other_blob = "7" * 40, "8" * 40

    def source_record(self, kind="blob", path="docs/proof.md"):
        url = f"https://github.com/d6g8k5htny-coder/Math-/{kind}/{self.pin}/{path}"
        rec = load("fresh.json")
        rec["sources"] = [{"url": url, "rev": self.pin}]
        rec["source_rev"] = self.pin
        return rec

    def api(self, changed=False):
        base = "/repos/d6g8k5htny-coder/Math-"
        def tree(sha, rows):
            return {"sha": sha, "tree": rows, "truncated": False}
        def entry(path, kind, sha, mode):
            return {"path": path, "type": kind, "sha": sha, "mode": mode}
        directory = self.tip_dir if changed else self.pin_dir
        responses = {
            base + "/commits?per_page=1": [{"sha": self.tip, "commit": {"tree": {"sha": self.tip_tree}}}],
            base + "/git/commits/" + self.pin: {"sha": self.pin, "tree": {"sha": self.pin_tree}},
            base + "/git/trees/" + self.pin_tree: tree(self.pin_tree, [entry("docs", "tree", self.pin_dir, "040000")]),
            base + "/git/trees/" + self.tip_tree: tree(self.tip_tree, [entry("docs", "tree", directory, "040000")]),
            base + "/git/trees/" + self.pin_dir: tree(self.pin_dir, [entry("proof.md", "blob", self.blob, "100644")]),
            base + "/git/trees/" + self.tip_dir: tree(self.tip_dir, [entry("proof.md", "blob", self.other_blob, "100644")]),
            # What the old implementation asks: older path-modification commit,
            # even though the referenced object is identical at pin and tip.
            base + "/commits?path=docs/proof.md&per_page=1": [{"sha": self.older}],
            base + "/commits?path=docs&per_page=1": [{"sha": self.older}],
        }
        seen = []
        def fetch(path):
            seen.append(path)
            if path not in responses:
                raise nsr.SourceError("unexpected or unavailable API path: " + path)
            return copy.deepcopy(responses[path])
        return base, responses, seen, fetch

    def refresh(self, rec, api):
        original = copy.deepcopy(rec)
        with patch.object(nsr, "_api_get", api):
            result = nsr.refresh_record(rec, nsr.github_fetch, NOW)
        self.assertEqual(rec, original)
        self.assertEqual(result["source_rev"], self.pin)
        self.assertEqual(result["sources"], original["sources"])
        self.assertEqual(result["scientific_effect"], "NONE")
        return result

    def test_snapshot_after_last_path_edit_is_fresh_for_same_blob(self):
        _, _, seen, api = self.api()
        out = self.refresh(self.source_record(), api)
        self.assertEqual(out["status"], "fresh")
        self.assertIsNone(out["refresh_error"])
        self.assertNotIn(self.tip, json.dumps(out))
        self.assertEqual(sum(path.endswith("/commits?per_page=1") for path in seen), 1)

    def test_unchanged_directory_uses_tree_identity(self):
        _, _, _, api = self.api()
        self.assertEqual(self.refresh(self.source_record("tree", "docs"), api)["status"], "fresh")

    def test_changed_blob_or_directory_is_stale(self):
        for kind, path in [("blob", "docs/proof.md"), ("tree", "docs")]:
            with self.subTest(kind=kind):
                _, _, _, api = self.api(changed=True)
                self.assertEqual(self.refresh(self.source_record(kind, path), api)["status"], "stale")

    def test_unavailable_or_untrustworthy_object_is_unknown(self):
        base = "/repos/d6g8k5htny-coder/Math-"
        tip_tree_path = base + "/git/trees/" + self.tip_tree
        cases = [
            (tip_tree_path, None),
            (tip_tree_path, {"sha": self.tip_tree, "tree": [], "truncated": True}),
            (tip_tree_path, {"sha": self.tip_tree, "tree": [], "truncated": False}),
            (base + "/git/commits/" + self.pin, {"sha": self.tip, "tree": {"sha": self.pin_tree}}),
            (base + "/commits?per_page=1", [{"sha": "bad", "commit": {"tree": {"sha": self.tip_tree}}}]),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                _, responses, _, api = self.api()
                responses[key] = value
                out = self.refresh(self.source_record(), api)
                self.assertEqual(out["status"], "unknown")
                self.assertTrue(out["refresh_error"])

    def test_duplicate_or_unsupported_target_is_unknown(self):
        for mode, kind, duplicate in [("040000", "tree", True), ("160000", "commit", False), ("120000", "blob", False)]:
            with self.subTest(mode=mode, duplicate=duplicate):
                base, responses, _, api = self.api()
                row = responses[base + "/git/trees/" + self.tip_tree]["tree"][0]
                row.update(mode=mode, type=kind)
                if duplicate:
                    responses[base + "/git/trees/" + self.tip_tree]["tree"].append(copy.deepcopy(row))
                out = self.refresh(self.source_record(), api)
                self.assertEqual(out["status"], "unknown")
                self.assertTrue(out["refresh_error"])

    def test_object_type_change_is_stale_even_when_sha_text_matches(self):
        base, responses, _, api = self.api(changed=True)
        target = responses[base + "/git/trees/" + self.tip_dir]["tree"][0]
        target.update(type="tree", mode="040000", sha=self.blob)
        self.assertEqual(self.refresh(self.source_record(), api)["status"], "stale")

    def test_observation_must_bind_exact_url_pin_type_and_valid_hashes(self):
        rec = self.source_record()
        _, _, _, api = self.api()
        with patch.object(nsr, "_api_get", api):
            observation = nsr.github_fetch(rec["sources"][0]["url"])
        self.assertEqual(observation.current_revision, self.tip)
        for mutation in [
            {"url": rec["sources"][0]["url"] + "-other"},
            {"pinned_revision": self.tip}, {"pinned_type": "tree"},
            {"current_revision": "bad"}, {"pinned_sha": "bad"},
            {"current_sha": "bad"}, {"current_type": "commit"},
        ]:
            with self.subTest(mutation=mutation):
                changed = observation._replace(**mutation)
                out = nsr.refresh_record(rec, lambda url: changed, NOW)
                self.assertEqual(out["status"], "unknown")
                self.assertTrue(out["refresh_error"])

    def test_url_kind_must_match_the_pinned_object(self):
        _, _, _, api = self.api()
        out = self.refresh(self.source_record("tree", "docs/proof.md"), api)
        self.assertEqual(out["status"], "unknown")

    def test_source_path_cannot_cross_a_file_or_use_dot_segments(self):
        for path in ["docs/proof.md/nested", "docs/../proof.md", "docs//proof.md"]:
            with self.subTest(path=path):
                _, _, _, api = self.api()
                self.assertEqual(self.refresh(self.source_record(path=path), api)["status"], "unknown")

    def test_report_only_keeps_entire_original_record_bytes(self):
        _, _, _, api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text(json.dumps(self.source_record()) + "\n")
            before = path.read_bytes()
            with patch.object(nsr, "_api_get", api):
                report = nsr.refresh_dir(Path(tmp), nsr.github_fetch, False, NOW)
            self.assertEqual(report[0]["status"], "fresh")
            self.assertEqual(path.read_bytes(), before)

    def test_raw_api_duplicate_object_keys_are_unknown(self):
        base, responses, _, _ = self.api()
        current = base + "/git/trees/" + self.tip_tree
        raw = json.dumps(responses[current])
        rows = json.dumps(responses[current]["tree"])
        variants = [
            '{"sha":"' + self.tip_tree + '","tree":[],"tree":' + rows + ',"truncated":false}',
            raw.replace('"sha": "' + self.pin_dir + '"',
                        '"sha":"' + self.tip_dir + '","sha":"' + self.pin_dir + '"'),
            raw.replace('"truncated": false', '"truncated": true,"truncated": false'),
        ]
        for ambiguous in variants:
            with self.subTest(response=ambiguous):
                wires = {path: json.dumps(value).encode() for path, value in responses.items()}
                wires[current] = ambiguous.encode()
                def opening(request, timeout):
                    return io.BytesIO(wires[request.full_url.removeprefix(nsr.API)])
                with patch.object(nsr.urllib.request, "urlopen", opening):
                    out = nsr.refresh_record(self.source_record(), nsr.github_fetch, NOW)
                self.assertEqual(out["status"], "unknown")
                self.assertIn("duplicate JSON key", out["refresh_error"])

    def test_raw_api_valid_object_controls(self):
        for changed, expected in [(False, "fresh"), (True, "stale")]:
            with self.subTest(changed=changed):
                _, responses, _, _ = self.api(changed=changed)
                wires = {path: json.dumps(value).encode() for path, value in responses.items()}
                def opening(request, timeout):
                    return io.BytesIO(wires[request.full_url.removeprefix(nsr.API)])
                with patch.object(nsr.urllib.request, "urlopen", opening):
                    out = nsr.refresh_record(self.source_record(), nsr.github_fetch, NOW)
                self.assertEqual(out["status"], expected)
                self.assertIsNone(out["refresh_error"])

    def test_raw_discussion_duplicate_timestamp_is_unknown(self):
        rec = load("issue.json")
        raw = b'{"updated_at":"2026-10-23T17:00:00Z","updated_at":"2026-10-22T17:00:00Z"}'
        with patch.object(nsr.urllib.request, "urlopen", lambda request, timeout: io.BytesIO(raw)):
            out = nsr.refresh_record(rec, nsr.github_fetch, NOW)
        self.assertEqual(out["status"], "unknown")
        self.assertIn("duplicate JSON key", out["refresh_error"])


# FU-3: bounded complete comment listings, using only synthetic API pages.
M = nsr
PREFIX = "https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-"
PICKUP = 100000
PIN = "a" * 40
PAGINATION_NOW = "2026-10-23T03:20:00Z"


def record():
    return {"schema_version": 0, "node_id": "TEST-FU3", "status": "unknown",
            "observed_at": "2026-10-22T00:00:00Z", "scientific_effect": "NONE",
            "sources": [{"url": "https://github.com/d6g8k5htny-coder/main/blob/" + PIN + "/AGENTS.md", "rev": PIN}],
            "source_rev": PIN, "owner": {"name": "TEST-NOT-A-REAL-OWNER", "evidence_url": PREFIX + str(PICKUP)}}


def entries(count, release=False):
    rows = [{"html_url": PREFIX + str(PICKUP + i), "body": "TEST ordinary comment"} for i in range(count)]
    if rows:
        rows[0]["body"] = "PICKUP TEST-FU3"
    if release and count > 1:
        rows[-1]["body"] = "RELEASE\nPICKUP " + str(PICKUP)
    return rows


class Paging:
    def __init__(self, rows, fault_page=None, fault=None):
        self.rows, self.fault_page, self.fault = rows, fault_page, fault
        self.calls = []

    def __call__(self, path):
        parsed = urlsplit(path)
        if parsed.path != "/repos/d6g8k5htny-coder/main/issues/419/comments":
            raise AssertionError("unexpected endpoint: " + path)
        q = parse_qs(parsed.query)
        if set(q) != {"page", "per_page"} or q["per_page"] != ["100"]:
            raise AssertionError("unexpected query: " + path)
        page = int(q["page"][0])
        if page != len(self.calls) + 1:
            raise AssertionError("repeated or out-of-order page")
        self.calls.append(page)
        if page > 27:
            raise AssertionError("test safety ceiling: unbounded product loop")
        if page == self.fault_page:
            if isinstance(self.fault, Exception):
                raise self.fault
            return self.fault
        return self.rows[(page - 1) * 100:page * 100]


class PaginationTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def read(self, paging):
        try:
            with patch.object(M, "_api_get", side_effect=paging):
                return M.github_comments(PREFIX + str(PICKUP)), None
        except M.SourceError as exc:
            return None, str(exc)

    def refresh(self, paging, input_record=None, source_value=PIN):
        incoming = record() if input_record is None else input_record
        before = copy.deepcopy(incoming)
        with patch.object(M, "_api_get", side_effect=paging):
            result = M.refresh_record(incoming, lambda _: source_value, now=PAGINATION_NOW)
        self.assertEqual(incoming, before, "caller-owned input mutated")
        self.assertEqual(result["source_rev"], PIN)
        self.assertEqual(result["scientific_effect"], "NONE")
        self.assertEqual(result["owner"]["name"], before["owner"]["name"])
        return result

    def test_full_2600_requires_unknown(self):
        p = Paging(entries(2600, True))
        data, error = self.read(p)
        self.assertIsNone(data)
        self.assertEqual(error, "comment listing incomplete")
        self.assertEqual(p.calls, list(range(1, 27)))

    def test_incomplete_does_not_project_early_release(self):
        rows = entries(2600)
        rows[1]["body"] = "RELEASE\nPICKUP " + str(PICKUP)
        p = Paging(rows)
        result = self.refresh(p)
        self.assertEqual(result["status"], "unknown")
        self.assertIn("comment listing incomplete", result["refresh_error"])
        self.assertNotIn("released_url", result["owner"])

    def test_wrong_release_reference_is_not_accepted(self):
        rows = entries(2500, True)
        rows[-1]["body"] = "RELEASE\nPICKUP 999999"
        result = self.refresh(Paging(rows))
        self.assertEqual(result["status"], "fresh")
        self.assertNotIn("released_url", result["owner"])

    def test_full_read_with_no_release_does_not_infer_one(self):
        result = self.refresh(Paging(entries(2500)))
        self.assertEqual(result["status"], "fresh")
        self.assertNotIn("released_url", result["owner"])

    def test_stale_source_remains_stale(self):
        result = self.refresh(Paging(entries(2500, True)), source_value="b" * 40)
        self.assertEqual(result["status"], "stale")
        self.assertEqual(result["owner"].get("released_url"), PREFIX + str(PICKUP + 2499))

    def test_bad_source_remains_unknown(self):
        result = self.refresh(Paging(entries(2500, True)), source_value="")
        self.assertEqual(result["status"], "unknown")
        self.assertIn("moving ref is not a 40-hex sha", result["refresh_error"])

    def test_unsupported_evidence_never_fetches(self):
        p = Paging([])
        with patch.object(M, "_api_get", side_effect=p):
            with self.assertRaisesRegex(M.SourceError, "unsupported owner evidence url"):
                M.github_comments("https://example.invalid/pickup")
        self.assertEqual(p.calls, [])

    def test_report_only_preserves_file(self):
        raw = (json.dumps(record()) + "\n").encode()
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "record.json"
            f.write_bytes(raw)
            with patch.object(M, "_api_get", side_effect=Paging(entries(2500, True))):
                report = M.refresh_dir(Path(td), lambda _: PIN, False, PAGINATION_NOW)
            self.assertEqual(f.read_bytes(), raw)
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["status"], "fresh")
        self.assertEqual(report[0].get("released_url"), PREFIX + str(PICKUP + 2499))


def complete_case(count):
    def test(self):
        p = Paging(entries(count))
        found, error = self.read(p)
        self.assertIsNone(error, "complete board incorrectly refused: " + str(error))
        self.assertEqual(found, [{"url": row["html_url"], "body": row["body"]} for row in p.rows])
        self.assertEqual(p.calls, list(range(1, count // 100 + 2)))
    return test


for COUNT in (0, 1, 99, 100, 101, 999, 1000, 1001, 1500, 2499, 2500, 2599):
    setattr(PaginationTests, f"test_complete_{COUNT:04d}", complete_case(COUNT))


def failure_case(page, kind):
    def test(self):
        fault = {"not_list": {}, "not_object": [None], "transport": M.SourceError("TEST HTTP 403 stop")}[kind]
        expected = {"not_list": "comment listing is not a list", "not_object": "comment is not an object", "transport": "TEST HTTP 403 stop"}[kind]
        p = Paging(entries(2500, True), page, fault)
        result = self.refresh(p)
        self.assertEqual(result["status"], "unknown")
        self.assertIn(expected, result["refresh_error"])
        self.assertNotIn("released_url", result["owner"])
        self.assertEqual(p.calls, list(range(1, page + 1)))
    return test


for PAGE in (2, 26):
    for KIND in ("not_list", "not_object", "transport"):
        setattr(PaginationTests, f"test_failure_{PAGE:02d}_{KIND}", failure_case(PAGE, KIND))


def release_case(count):
    def test(self):
        result = self.refresh(Paging(entries(count, True)))
        self.assertEqual(result["status"], "fresh")
        self.assertIsNone(result["refresh_error"])
        self.assertEqual(result["owner"].get("released_url"), PREFIX + str(PICKUP + count - 1))
    return test


for COUNT in (1000, 1001, 2500):
    setattr(PaginationTests, f"test_explicit_release_{COUNT}", release_case(COUNT))


if __name__ == "__main__":
    unittest.main()
