"""Shared acquisition contract: exact joins, no partial badges or scientific inference."""
import copy
import json
from pathlib import Path
import sys
import unittest
from html.parser import HTMLParser
import hashlib
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    from current_inventory_acquisition import project_current_acquisitions
except ImportError:
    project_current_acquisitions = None
FIXTURES = json.loads((ROOT / "tests/fixtures/current_inventory_acquisition_cases.json").read_text())


class AcquisitionConformance(unittest.TestCase):
    def test_shared_projection_cases(self):
        self.assertIsNotNone(project_current_acquisitions, "current acquisition projection is missing")
        for case in FIXTURES["cases"]:
            with self.subTest(case=case["name"]):
                source = copy.deepcopy(case["snapshot"])
                self.assertEqual(project_current_acquisitions(source), case["expected"])
                self.assertEqual(source, case["snapshot"], "projection mutated source evidence")


class InventoryPresentation(unittest.TestCase):
    def setUp(self):
        import current_site_build as build
        import current_site_history as history
        self.build, self.history = build, history
        self.data = copy.deepcopy(FIXTURES["cases"][0]["snapshot"])
        self.meta = {"sha256": "f" * 64, "bytes": 100, "archive": "e" * 64}

    def test_current_rows_explain_content_scope_and_escape_source_labels(self):
        raw = self.build.current_section(self.data, self.meta)
        self.assertIn('aria-describedby="current-inventory-legend"', raw)
        for text in ["Metadata only", "Parsed graph", "Reference identity", "Included excerpts (2)",
                     "Mathematical review: unknown in this inventory", "Acceptance: unknown in this inventory"]:
            self.assertIn(text, raw)
        self.assertIn("&lt;b&gt;", raw)
        self.assertNotIn("<script>", raw)
        self.assertIn("#L2-L3", raw)
        self.assertIn("0fbb9ed52bcd56db320444384900aa6c7b17aacefec4c65c0808cd5b56ff32fd", raw)
        self.assertIn("producer-recorded complete-file SHA-256", raw)

    def test_live_static_index_uses_same_annotations_and_keeps_unresolved_node(self):
        pages = self.history.render_live_current_v2(self.data, self.meta)
        raw = pages["nodes/index.html"].decode()
        self.assertIn('aria-describedby="current-inventory-legend"', raw)
        for text in ["Metadata only", "Parsed graph", "Included excerpts (2)", "Reference identity"]:
            self.assertIn(text, raw)
        node = pages["nodes/n.html"].decode()
        self.assertIn("Graph-to-artifact binding: unresolved; artifact roles: unknown; custody: unknown", node)
        self.assertIn("No exact file", node)
        self.assertNotIn("Acquisition evidence: Parsed graph", node)

    def test_any_bad_association_removes_every_positive_label_in_current_output(self):
        self.data["graph"]["identity"]["bytes"] += 1
        for raw in [self.build.current_section(self.data, self.meta),
                    self.history.render_live_current_v2(self.data, self.meta)["nodes/index.html"].decode()]:
            self.assertIn("Acquisition associations could not be qualified", raw)
            self.assertIn("acquisition_bytes_conflict", raw)
            self.assertIn("Acquisition evidence: Unavailable", raw)
            self.assertNotIn("Acquisition evidence: Parsed graph", raw)
            self.assertNotIn("Acquisition evidence: Included excerpts", raw)

    def test_date_objects_have_separate_meanings_and_no_invented_classification_date(self):
        raw = self.build.current_section(self.data, self.meta)
        self.assertIn('id="current-date-explanation"', raw)
        self.assertIn('datetime="2026-10-13T12:00:00Z"', raw)
        self.assertIn("classification date not recorded separately", raw)
        self.assertIn("Graph classification source", raw)
        self.assertIn("Historical records", raw)
        self.assertIn("Curated Research stories", raw)
        self.assertIn(self.data["graph"]["identity"]["commit"], raw)
        self.data["observed_at"] = "2026-10-20T11:00:00Z"
        later = self.history.render_live_current_v2(self.data, self.meta)["nodes/index.html"].decode()
        self.assertIn('datetime="2026-10-20T11:00:00Z"', later)
        self.assertIn("classification date not recorded separately", later)
        self.assertIn(self.data["graph"]["identity"]["commit"], later)

    def test_predecessor_live_output_stays_byte_exact_for_both_style_orders(self):
        expected = json.loads((ROOT / "tests/fixtures/current_inventory_legacy_live_hashes.json").read_text())
        for style in (True, False):
            pages = self.history.render_live_current_v2(self.data, self.meta, annotations=False, style_first=style)
            self.assertEqual({k: hashlib.sha256(v).hexdigest() for k, v in pages.items()}, expected[str(style).lower()])

    def test_exact_predecessor_maps_restore_but_one_changed_descriptor_refuses_all_writes(self):
        import test_current_site_history as controls
        worker = controls.CurrentSiteHistoryTests()
        worker.setUp()
        original = self.history.render_live_current_v2
        for annotations, style in ((True, True), (True, False), (False, True), (False, False)):
            with self.subTest(annotations=annotations, style_first=style), TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "public").mkdir()
                (root / "runner").mkdir()
                public, payload = worker.prepare(root / "public")
                data = json.loads(payload.read_bytes())
                meta = json.loads((public / "current-meta.json").read_bytes())
                predecessor = original(data, meta, annotations=annotations, style_first=style)
                for name, raw in predecessor.items():
                    (public / name).write_bytes(raw)
                def legacy(*args, **kwargs):
                    return original(*args, **{**kwargs, "annotations": annotations, "style_first": style})
                with patch.object(self.history, "render_live_current_v2", side_effect=legacy):
                    worker.publish(public)
                files = worker.public_files(public)
                runner, _ = worker.prepare(root / "runner")
                self.history.restore_history(runner, controls.PublicFixtureReader(files))
                for name, raw in predecessor.items():
                    self.assertEqual((runner / name).read_bytes(), raw)
                forged = copy.deepcopy(json.loads(files["snapshots/index.json"]))
                forged["current_observation"]["files"][0]["sha256"] = "1" * 64
                worker.rebind_manifest(files, forged)
                before = worker.public_files(runner)
                with self.assertRaises(ValueError):
                    self.history.restore_history(runner, controls.PublicFixtureReader(files))
                self.assertEqual(worker.public_files(runner), before)
                # Each replacement file is valid under another admitted presentation,
                # but a hybrid map must never be admitted one file at a time.
                files = worker.public_files(public)
                hybrid = copy.deepcopy(json.loads(files["snapshots/index.json"]))
                alternate = original(data, meta, annotations=not annotations, style_first=style)
                name = sorted(predecessor)[0]
                self.assertNotEqual(predecessor[name], alternate[name])
                files[name] = alternate[name]
                descriptor = self.history.descriptor(name, alternate[name])
                for row in hybrid["current_observation"]["files"]:
                    if row["path"] == name:
                        row.update(descriptor)
                for alias in hybrid["aliases"]:
                    if alias["path"] == name:
                        alias.update(descriptor)
                hybrid["total_bytes"] = self.history.history_size(hybrid["archives"], hybrid["aliases"], hybrid["current_observation"])
                hybrid["capacity"] = self.history.capacity_record(hybrid["archives"], hybrid["total_bytes"])
                worker.rebind_manifest(files, hybrid)
                with self.assertRaisesRegex(ValueError, "live static reading does not match"):
                    self.history.restore_history(runner, controls.PublicFixtureReader(files))
                self.assertEqual(worker.public_files(runner), before)


if __name__ == "__main__":
    unittest.main()
