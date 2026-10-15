"""Live generated node pages load style.css before brand.css (F2: brand.css dark/reduced-motion rules must win)."""
import copy
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SITE = ROOT / "docs/site"
LINK = re.compile(r'<link rel="stylesheet" href="(?:\.\./)*([a-z]+\.css)')


class StylesheetOrder(unittest.TestCase):
    def order(self, path):
        return LINK.findall(path.read_text(encoding="utf-8"))

    def test_live_node_pages_load_style_before_brand(self):
        pages = sorted((SITE / "nodes").glob("*.html"))
        self.assertTrue(pages)
        for page in pages:
            names = self.order(page)
            with self.subTest(page=page.name):
                self.assertIn("style.css", names)
                self.assertIn("brand.css", names)
                self.assertLess(names.index("style.css"), names.index("brand.css"))

    def test_generators_emit_style_before_brand(self):
        for tool, marker in (("current_site_build.py", "def static_page"), ("current_site_history.py", "def _live_v2_static_page")):
            text = (ROOT / "tools" / tool).read_text(encoding="utf-8")
            body = text[text.index(marker):]
            body = body[:body.index("</head>")]
            with self.subTest(tool=tool):
                self.assertLess(body.index("style.css"), body.index("brand.css"))

    def test_removed_node_live_route_loads_style_before_brand_and_archive_stays(self):
        # A later observation that drops a v2 node republishes nodes/<id>.html
        # through render_static_v2(..., archived=False). That convenience route
        # must follow the live order. The immutable snapshot keeps brand.css first.
        import test_current_site_build as build_controls
        import current_site_history as history

        controls = build_controls.CurrentSiteBuildTests()
        with TemporaryDirectory() as tmp:
            site, payload = controls.prepare(Path(tmp))
            first = controls.run_build(site, payload)
            self.assertEqual(first.returncode, 0, first.stderr)
            digest = history.mathematical_digest(build_controls.fixture())
            archive = site / "snapshots" / digest
            archived_before = {p.relative_to(archive).as_posix(): p.read_bytes()
                               for p in archive.rglob("*") if p.is_file()}
            self.assertEqual(self.order(archive / "nodes/beta.html"), ["brand.css", "style.css", "current.css"])
            data = build_controls.fixture()
            data["observed_at"] = "2026-10-13T13:00:00Z"
            data["graph"]["data"]["nodes"].pop("beta")
            data["graph"]["data"]["edges"] = [
                edge for edge in data["graph"]["data"]["edges"]
                if edge["from"] != "beta" and edge["to"] != "beta"]
            payload.write_bytes(json.dumps(data, ensure_ascii=False, indent=2).encode() + b"\n")
            second = controls.run_build(site, payload)
            self.assertEqual(second.returncode, 0, second.stderr)
            live = site / "nodes/beta.html"
            self.assertIn("Historical removed-node static route", live.read_text(encoding="utf-8"))
            self.assertEqual(self.order(live), ["style.css", "brand.css", "current.css"])
            self.assertEqual(self.order(archive / "nodes/beta.html"), ["brand.css", "style.css", "current.css"])
            archived_after = {p.relative_to(archive).as_posix(): p.read_bytes()
                              for p in archive.rglob("*") if p.is_file()}
            self.assertEqual(archived_before, archived_after)

    def test_restore_keeps_predecessor_removed_node_order_and_live_pages_stay_new(self):
        # A v2 archive alias published brand-first must restore. Swapping only
        # the stylesheet links keeps every other byte, so the published
        # descriptor still identifies that exact predecessor page. A further
        # non-stylesheet edit must still be refused. Live pages stay style-first.
        import test_current_site_build as build_controls
        import test_current_site_history as history_controls
        import current_site_history as history

        harness = history_controls.CurrentSiteHistoryTests()
        harness.history = history
        harness.build = build_controls.CurrentSiteBuildTests()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "public").mkdir()
            (root / "runner").mkdir()
            public, payload = harness.prepare(root / "public")
            harness.publish(public)
            data = build_controls.fixture()
            data["observed_at"] = "2026-10-13T13:00:00Z"
            data["graph"]["data"]["nodes"].pop("beta")
            data["graph"]["data"]["edges"] = [
                edge for edge in data["graph"]["data"]["edges"]
                if edge["from"] != "beta" and edge["to"] != "beta"]
            payload.write_bytes(json.dumps(data, ensure_ascii=False).encode())
            self.assertEqual(harness.build.run_build(public, payload).returncode, 0)
            manifest = harness.publish(public)
            alias = next(row for row in manifest["aliases"] if row["path"] == "nodes/beta.html")
            self.assertEqual(alias["binding"], "archive")
            self.assertEqual(self.order(public / "nodes/beta.html"), ["style.css", "brand.css", "current.css"])
            self.assertEqual(self.order(public / "nodes/alpha.html"), ["style.css", "brand.css", "current.css"])
            current = (public / "nodes/beta.html").read_bytes()
            predecessor = current.replace(
                b'<link rel="stylesheet" href="../style.css"><link rel="stylesheet" href="../brand.css">',
                b'<link rel="stylesheet" href="../brand.css"><link rel="stylesheet" href="../style.css">',
                1)
            self.assertEqual(len(predecessor), len(current))
            self.assertNotEqual(predecessor, current)
            self.assertIn(b"Historical removed-node static route", predecessor)
            files = harness.public_files(public)
            published = json.loads(files["snapshots/index.json"])
            rewritten = copy.deepcopy(published)
            for row in rewritten["aliases"]:
                if row["path"] == "nodes/beta.html":
                    row["sha256"] = hashlib.sha256(predecessor).hexdigest()
            pred_files = dict(files)
            pred_files["nodes/beta.html"] = predecessor
            harness.rebind_manifest(pred_files, rewritten)
            snapshot_id = alias["snapshot"]
            archived = {name: blob for name, blob in pred_files.items() if name.startswith(f"snapshots/{snapshot_id}/")}
            runner, _ = harness.prepare(root / "runner")
            history.restore_history(runner, history_controls.PublicFixtureReader(pred_files))
            restored = (runner / "nodes/beta.html").read_bytes()
            self.assertEqual(restored, predecessor)
            self.assertEqual(self.order(runner / "nodes/beta.html"), ["brand.css", "style.css", "current.css"])
            self.assertEqual(self.order(runner / "nodes/alpha.html"), ["style.css", "brand.css", "current.css"])
            for name, blob in archived.items():
                self.assertEqual((runner / name).read_bytes(), blob, name)
            tampered = predecessor.replace(b"<code>HOLD</code>", b"<code>OPEN</code>", 1)
            self.assertEqual(len(tampered), len(predecessor))
            self.assertNotEqual(tampered, predecessor)
            forged = copy.deepcopy(rewritten)
            for row in forged["aliases"]:
                if row["path"] == "nodes/beta.html":
                    row["sha256"] = hashlib.sha256(tampered).hexdigest()
            bad_files = dict(pred_files)
            bad_files["nodes/beta.html"] = tampered
            harness.rebind_manifest(bad_files, forged)
            (root / "runner-tampered").mkdir()
            refused, _ = harness.prepare(root / "runner-tampered")
            with self.assertRaises(ValueError) as caught:
                history.restore_history(refused, history_controls.PublicFixtureReader(bad_files))
            self.assertIn("published alias differs", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
