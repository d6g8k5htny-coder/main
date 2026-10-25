"""Public Pages history admission: real ZIPs, payloads and filesystem effects."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import urllib.error
import zipfile
import warnings

try:
    import test_current_site_build as build_controls
except ModuleNotFoundError:
    from tests import test_current_site_build as build_controls
fixture = build_controls.fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


class PublicFixtureReader:
    def __init__(self, files, failures=None):
        self.files = files
        self.failures = failures or {}
        self.calls = []

    def get_bytes(self, path):
        self.calls.append(path)
        if path in self.failures:
            raise self.failures[path]
        if path not in self.files:
            raise urllib.error.HTTPError("https://fixture.invalid/" + path, 404, "not found", {}, None)
        return self.files[path]


class CurrentSiteHistoryTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "tools/current_site_history.py"
        self.assertTrue(path.is_file(), "public history custody has not been implemented")
        import current_site_history
        self.history = current_site_history
        self.build = build_controls.CurrentSiteBuildTests()

    def prepare(self, root, data=None):
        site, payload = self.build.prepare(root, data)
        result = self.build.run_build(site, payload)
        self.assertEqual(result.returncode, 0, result.stderr)
        return site, payload

    def public_files(self, site):
        return {p.relative_to(site).as_posix(): p.read_bytes()
                for p in site.rglob("*") if p.is_file()}

    def publish(self, site):
        return self.history.publish_history(site)

    def replace_bundle(self, files, manifest, edit):
        entry = manifest["archives"][0]
        path = entry["bundle"]["path"]
        old = zipfile.ZipFile(io.BytesIO(files[path]))
        members = [(info.filename, old.read(info.filename)) for info in old.infolist()]
        stream = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
                edit(z, members)
        raw = stream.getvalue()
        files[path] = raw
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            unique = {i.filename: z.read(i) for i in z.infolist()}
        entry["files"] = [{"path": n, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                          for n, b in sorted(unique.items()) if not n.startswith("aliases/")]
        entry["alias_files"] = [{"path": n[len("aliases/"):], "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                                for n, b in sorted(unique.items()) if n.startswith("aliases/")]
        entry["bundle"].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        try:
            manifest["total_bytes"] = self.history.history_size(manifest["archives"], manifest["aliases"])
        except ValueError:
            pass
        self.rebind_manifest(files, manifest)

    def rebind_manifest(self, files, manifest):
        raw = self.history.manifest_bytes(manifest)
        files["snapshots/index.json"] = raw
        meta = json.loads(files["current-meta.json"])
        meta["history_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        files["current-meta.json"] = json.dumps(meta).encode()

    def prepare_v1(self, root):
        # Produce the exact earlier public schema/renderer without invoking today's builder.
        site, payload = self.build.prepare(root)
        raw = payload.read_bytes(); data = json.loads(raw); snapshot = hashlib.sha256(raw).hexdigest()
        meta = {"sha256": snapshot, "bytes": len(raw), "observed_at": data["observed_at"]}
        archived = {"current.json": raw, "current-meta.json": self.history.manifest_bytes(meta),
                    **self.history.render_static_v1(data, meta, archived=True)}
        entry, (members, _, _), bundle = self.history.archive_record(snapshot, archived)
        aliases = [{"path": name[len("aliases/"):], "snapshot": snapshot} for name in sorted(members) if name.startswith("aliases/")]
        manifest = {"schema_version": 1, "origin": self.history.ORIGIN, "scientific_effect": "NONE",
                    "representation": "asset-release-normalized-html-v1", "current": snapshot,
                    "archives": [entry], "aliases": aliases, "total_bytes": self.history.history_size([entry], aliases)}
        self.history.validate_manifest(manifest)
        outputs = {f"snapshots/{snapshot}/{name}": content for name, content in archived.items()}
        outputs.update({name[len("aliases/"):]: content for name, content in members.items() if name.startswith("aliases/")})
        manifest_raw = self.history.manifest_bytes(manifest)
        outputs.update({entry["bundle"]["path"]: bundle, "snapshots/index.json": manifest_raw,
                        "current.json": raw, "current-meta.json": self.history.manifest_bytes({**meta, "archive": snapshot,
                        "history_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest()})})
        self.history.atomic_outputs(site, outputs)
        return site, payload, manifest

    def test_v1_archives_remain_byte_exact_through_v2_refresh_and_fresh_restore(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "public").mkdir(); (root / "runner").mkdir()
            public, payload, v1 = self.prepare_v1(root / "public")
            old = {n: b for n, b in self.public_files(public).items() if n.startswith(f'snapshots/{v1["current"]}/')}
            moved = fixture(); moved["observed_at"] = "2026-10-13T13:30:00Z"
            payload.write_text(json.dumps(moved)); self.assertEqual(self.build.run_build(public, payload).returncode, 0)
            mixed = self.publish(public)
            self.assertEqual(mixed["schema_version"], 2)
            self.assertEqual(mixed["archives"], v1["archives"])
            for name, raw in old.items(): self.assertEqual((public / name).read_bytes(), raw)
            changed = copy.deepcopy(moved); changed["graph"]["data"]["nodes"].pop("beta")
            changed["graph"]["data"]["edges"] = []; changed["observed_at"] = "2026-10-13T14:00:00Z"
            payload.write_text(json.dumps(changed)); self.assertEqual(self.build.run_build(public, payload).returncode, 0)
            final = self.publish(public)
            self.assertEqual(len(final["archives"]), 2)
            self.assertIn(v1["archives"][0], final["archives"])
            for name, raw in old.items(): self.assertEqual((public / name).read_bytes(), raw)
            runner, _ = self.prepare(root / "runner")
            raw_manifest = (public / "snapshots/index.json").read_bytes()
            self.history.restore_history(runner, PublicFixtureReader(self.public_files(public)),
                                         expected_manifest_sha=hashlib.sha256(raw_manifest).hexdigest())
            for name, raw in old.items(): self.assertEqual((runner / name).read_bytes(), raw)
            self.assertTrue((runner / "nodes/beta.html").is_file())
            self.assertIn(f'../snapshots/{v1["current"]}/current.json', (runner / "nodes/beta.html").read_text())

    def test_v1_publication_can_restore_into_a_reviewed_v2_candidate_checkout(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "public").mkdir(); (root / "runner").mkdir()
            public, _, old = self.prepare_v1(root / "public")
            data = fixture(); data["graph"]["data"]["nodes"]["candidate-only"] = {"kind": "predicate", "classification": "OPEN"}
            runner, payload = self.prepare(root / "runner", data); self.publish(runner)
            self.history.restore_history(runner, PublicFixtureReader(self.public_files(public)))
            self.assertTrue((runner / "nodes/candidate-only.html").is_file())
            self.assertEqual(self.build.run_build(runner, payload).returncode, 0)
            mixed = self.publish(runner)
            self.assertIn(old["archives"][0], mixed["archives"])
            self.assertEqual(len(mixed["archives"]), 2)

    def test_capacity_warning_is_visible_at_75_percent_and_full_capacity_refuses_writes(self):
        import current_site_build
        with TemporaryDirectory() as tmp, patch.object(self.history, "MAX_SNAPSHOTS", 4):
            site, payload = self.build.prepare(Path(tmp))
            for number in range(4):
                data = fixture(); data["graph"]["data"]["nodes"]["alpha"]["scope"] = str(number)
                payload.write_text(json.dumps(data))
                current_site_build.build(site, payload)
                manifest = self.publish(site)
                self.assertEqual(manifest["capacity"]["warning"], number >= 2)
                if number >= 2: self.assertIn("Capacity warning: at least 75%", (site / "index.html").read_text())
            before = self.public_files(site)
            data["graph"]["data"]["nodes"]["alpha"]["scope"] = "fifth"
            payload.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "capacity"): current_site_build.build(site, payload)
            self.assertEqual(self.public_files(site), before)

    def test_live_alias_digest_and_visible_material_must_match_fresh_current_payload(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "public").mkdir(); (root / "runner").mkdir()
            public, _ = self.prepare(root / "public"); manifest = self.publish(public)
            for mutate in (lambda m: m["current_observation"]["files"][0].update(sha256="e" * 64),
                           lambda m: m["aliases"][0].update(binding="archive")):
                files = self.public_files(public); forged = copy.deepcopy(manifest); mutate(forged)
                self.rebind_manifest(files, forged)
                runner, _ = self.prepare(root / "runner") if not (root / "runner/site").exists() else (root / "runner/site", None)
                before = self.public_files(runner)
                with self.assertRaises(ValueError): self.history.restore_history(runner, PublicFixtureReader(files))
                self.assertEqual(self.public_files(runner), before)
            page = public / "nodes/alpha.html"
            page.write_bytes(page.read_bytes().replace(b"PROVED_REVIEWED", b"PASS"))
            before = self.public_files(public)
            with self.assertRaisesRegex(ValueError, "exact live observation"): self.publish(public)
            self.assertEqual(self.public_files(public), before)

    def test_local_unknown_metadata_cannot_publish_an_unreadable_public_observation(self):
        with TemporaryDirectory() as tmp:
            site, _ = self.prepare(Path(tmp)); self.publish(site)
            path = site / "current-meta.json"; meta = json.loads(path.read_bytes())
            meta["unknown_metadata"] = "must not enter publication"
            path.write_text(json.dumps(meta)); before = self.public_files(site)
            with self.assertRaisesRegex(ValueError, "metadata.*schema"): self.publish(site)
            self.assertEqual(self.public_files(site), before)

    def test_v2_archive_admission_survives_later_current_validator_and_renderer_changes(self):
        import current_site_build
        with TemporaryDirectory() as tmp:
            site, _ = self.prepare(Path(tmp)); manifest = self.publish(site)
            entry = manifest["archives"][0]; raw = (site / entry["bundle"]["path"]).read_bytes()
            with patch.object(current_site_build, "validate_payload", side_effect=RuntimeError("future current schema")), \
                    patch.object(current_site_build, "ensure_render_budget", side_effect=RuntimeError("future current UI")), \
                    patch.object(current_site_build, "render_nodes", side_effect=RuntimeError("future current renderer")):
                members, data, meta = self.history.admit_bundle(entry, raw)
            self.assertEqual(data["coverage"], "mathematical-material-only")
            self.assertEqual(meta["archive_version"], 2)
            self.assertFalse(any(name.startswith("aliases/") for name in members))

    def test_publication_and_restore_retain_removed_aliases_on_a_fresh_runner(self):
        # Restore must preserve an intermediate published B-only node when checkout contains only A.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "public").mkdir(); (root / "runner").mkdir()
            public, payload = self.prepare(root / "public")
            first = self.publish(public)
            runner, _ = self.prepare(root / "runner")
            data2 = fixture()
            data2["observed_at"] = "2026-10-13T13:00:00Z"
            data2["graph"]["data"]["nodes"]["only-in-B"] = {"kind": "predicate", "classification": "HOLD"}
            payload.write_text(json.dumps(data2))
            self.assertEqual(self.build.run_build(public, payload).returncode, 0)
            second = self.publish(public)
            self.assertEqual(len(second["archives"]), 2)
            files = self.public_files(public)
            restored = self.history.restore_history(runner, PublicFixtureReader(files))
            self.assertFalse(restored["bootstrap"])
            self.assertEqual((runner / "nodes/only-in-B.html").read_bytes(),
                             self.history.canonical_bytes("nodes/only-in-B.html", files["nodes/only-in-B.html"]))
            data3 = fixture()
            data3["observed_at"] = "2026-10-13T14:00:00Z"
            data3["graph"]["data"]["nodes"]["new-C"] = {"kind": "predicate", "classification": "OPEN"}
            next_payload = root / "runner/payload.json"
            next_payload.write_text(json.dumps(data3))
            self.assertEqual(self.build.run_build(runner, next_payload).returncode, 0)
            third = self.publish(runner)
            self.assertEqual(len(third["archives"]), 3)
            self.assertTrue((runner / "nodes/only-in-B.html").is_file())
            b_id = second["current"]
            self.assertTrue((runner / f"snapshots/{b_id}/nodes/only-in-B.html").is_file())

    def test_equivalent_sources_reuse_the_exact_prior_archive_and_refresh_current_observation(self):
        # Treating observation time as source content would create an archive on every scheduled run.
        with TemporaryDirectory() as tmp:
            site, payload = self.prepare(Path(tmp))
            first = self.publish(site)
            self.assertGreaterEqual(first["total_bytes"], self.history.MAX_FILE_BYTES + len((site / "current.json").read_bytes()))
            archive_id = first["current"]
            exact_old = (site / f"snapshots/{archive_id}/current.json").read_bytes()
            data = fixture(); data["observed_at"] = "2026-10-13T13:30:00Z"
            payload.write_text(json.dumps(data, indent=4))
            result = self.build.run_build(site, payload)
            self.assertEqual(result.returncode, 0, result.stderr)
            second = self.publish(site)
            self.assertEqual(second["current"], archive_id)
            self.assertEqual(len(second["archives"]), 1)
            self.assertEqual((site / f"snapshots/{archive_id}/current.json").read_bytes(), exact_old)
            self.assertEqual(json.loads((site / "current-meta.json").read_bytes())["observed_at"], data["observed_at"])
            self.assertEqual(json.loads(exact_old)["observed_at"], "2026-10-13T12:00:00Z")

    def test_commit_and_unrelated_inventory_movement_do_not_create_mathematical_versions(self):
        # One thousand observations of unchanged retained material must consume zero new archives.
        data = fixture()
        expected = self.history.mathematical_digest(data)
        for number in range(1000):
            moved = copy.deepcopy(data)
            commit = f"{number + 1:040x}"
            for repo in moved["repositories"]:
                repo.update(commit=commit, tree=f"{number + 2:040x}")
            for row in moved["sources"] + moved["references"] + [moved["graph"]["identity"]] + moved["equations"]:
                row["commit"] = commit
            moved["sources"].append(build_controls.identity(build_controls.MAIN, f"unrelated-{number}.md"))
            moved["sources"][-1]["commit"] = commit
            self.assertEqual(self.history.mathematical_digest(moved), expected)
        with TemporaryDirectory() as tmp:
            site, payload = self.prepare(Path(tmp), data)
            first = self.publish(site)
            original = self.public_files(site)
            moved["observed_at"] = "2026-10-13T13:30:00Z"
            payload.write_text(json.dumps(moved))
            result = self.build.run_build(site, payload)
            self.assertEqual(result.returncode, 0, result.stderr)
            second = self.publish(site)
            self.assertEqual(second["schema_version"], 2)
            self.assertEqual(len(second["archives"]), 1)
            self.assertEqual(first["current"], second["current"])
            for name, raw in original.items():
                if name.startswith(f'snapshots/{first["current"]}/'):
                    self.assertEqual((site / name).read_bytes(), raw)
            live = (site / "nodes/index.html").read_text()
            self.assertIn(commit, live)
            self.assertIn("unrelated-999.md", live)
            self.assertIn(moved["observed_at"], live)
            self.assertIn("Complete current public source inventory", live)

    def test_retained_material_changes_create_versions_and_projection_is_limited(self):
        data = fixture()
        projection = self.history.mathematical_projection(data)
        self.assertEqual(projection["schema_version"], 2)
        self.assertEqual(projection["coverage"], "mathematical-material-only")
        with self.assertRaises(ValueError): self.history.validate_projection({**projection, "schema_version": 2.0})
        self.assertLess(len(projection["sources"]), len(data["sources"]))
        baseline = self.history.mathematical_digest(data)
        for mutate in (lambda d: d["graph"]["data"]["nodes"]["alpha"].update(scope="changed"),
                       lambda d: d["graph"]["identity"].update(sha256="e" * 64),
                       lambda d: d["sources"][0].update(blob="e" * 40),
                       lambda d: d["references"][0].update(sha256="e" * 64)):
            changed = copy.deepcopy(data); mutate(changed)
            self.assertNotEqual(self.history.mathematical_digest(changed), baseline)
        equation = {"id": "eq", "repository": build_controls.MATH, "commit": build_controls.COMMIT,
                    "path": "proof/PROOF.md", "text": "x = 1\n", "line_start": 1, "line_end": 1,
                    "sha256": hashlib.sha256(b"x = 1\n").hexdigest()}
        data["equations"] = [equation]
        changed = copy.deepcopy(data); changed["equations"][0].update(text="x = 2\n", sha256=hashlib.sha256(b"x = 2\n").hexdigest())
        self.assertNotEqual(self.history.mathematical_digest(changed), self.history.mathematical_digest(data))

    def test_expected_manifest_pin_refuses_cached_self_consistent_history_before_any_write(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "public").mkdir(); (root / "runner").mkdir()
            public, payload = self.prepare(root / "public"); self.publish(public)
            cached = self.public_files(public)
            changed = fixture(); changed["graph"]["data"]["nodes"]["alpha"]["scope"] = "B"
            payload.write_text(json.dumps(changed)); self.assertEqual(self.build.run_build(public, payload).returncode, 0)
            self.publish(public)
            expected = hashlib.sha256((public / "snapshots/index.json").read_bytes()).hexdigest()
            runner, _ = self.prepare(root / "runner"); before = self.public_files(runner)
            reader = PublicFixtureReader(cached)
            with self.assertRaisesRegex(ValueError, "manifest.*pin"):
                self.history.restore_history(runner, reader, expected_manifest_sha=expected)
            self.assertEqual(reader.calls, ["snapshots/index.json"])
            self.assertEqual(self.public_files(runner), before)
            with self.assertRaisesRegex(ValueError, "manifest.*pin"):
                self.history.restore_history(runner, PublicFixtureReader({}), expected_manifest_sha=expected)
            self.assertEqual(self.public_files(runner), before)

    def test_production_reader_allows_exact_live_payload_with_bounded_request_count(self):
        class Response:
            status = 200
            headers = {"Content-Length": "2"}
            def geturl(self): return self.url
            def read(self, limit): return b"{}"
            def __enter__(self): return self
            def __exit__(self, *args): pass
        class Opener:
            def open(self, request, timeout):
                response = Response(); response.url = request.full_url; return response
        reader = self.history.HistoryReader(opener=Opener())
        self.assertEqual(reader.get_bytes("current.json"), b"{}")
        reader.requests = self.history.MAX_SNAPSHOTS + 2
        self.assertEqual(reader.get_bytes("current.json"), b"{}")
        with self.assertRaises(ValueError): reader.get_bytes("current.json")

    def test_failed_late_download_leaves_the_entire_runner_unchanged(self):
        # Successful early requests must never commit a partially restored history.
        with TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "public").mkdir(); (root / "runner").mkdir()
            public, payload = self.prepare(root / "public")
            self.publish(public)
            data = fixture(); data["graph"]["data"]["nodes"]["alpha"]["scope"] = "later scope"
            payload.write_text(json.dumps(data))
            self.assertEqual(self.build.run_build(public, payload).returncode, 0)
            manifest = self.publish(public)
            runner, _ = self.prepare(root / "runner")
            before = self.public_files(runner)
            bad = manifest["archives"][-1]["bundle"]["path"]
            reader = PublicFixtureReader(self.public_files(public), {bad: TimeoutError("transient fixture timeout")})
            with self.assertRaises((ValueError, TimeoutError)):
                self.history.restore_history(runner, reader)
            self.assertEqual(self.public_files(runner), before)

    def test_bootstrap_uses_remote_legacy_absence_and_never_ignores_a_published_pointer(self):
        # Local candidate archives are reviewed Git inputs; a remote history pointer forbids bootstrap.
        with TemporaryDirectory() as tmp:
            site, _ = self.prepare(Path(tmp))
            self.publish(site)
            before = self.public_files(site)
            result = self.history.restore_history(site, PublicFixtureReader({}))
            self.assertTrue(result["bootstrap"])
            self.assertEqual(self.public_files(site), before)
            for remote_meta in ({"archive": "a" * 64}, {"history_manifest_sha256": "b" * 64}):
                with self.subTest(meta=remote_meta), self.assertRaises(ValueError):
                    self.history.restore_history(site, PublicFixtureReader({"current-meta.json": json.dumps(remote_meta).encode()}))
            with self.assertRaises(ValueError):
                self.history.restore_history(site, PublicFixtureReader({"current-meta.json": b"not JSON"}))

    def test_restored_ledger_forbids_bootstrap_and_stale_manifest_history_loss(self):
        # A fresh-looking but older public response must not discard already admitted B history.
        with TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "public").mkdir(); (root / "runner").mkdir()
            public, payload = self.prepare(root / "public")
            self.publish(public)
            old_files = self.public_files(public)
            data = fixture(); data["graph"]["data"]["nodes"]["alpha"]["scope"] = "B"
            payload.write_text(json.dumps(data))
            self.assertEqual(self.build.run_build(public, payload).returncode, 0)
            self.publish(public)
            runner, _ = self.prepare(root / "runner")
            self.history.restore_history(runner, PublicFixtureReader(self.public_files(public)))
            before = self.public_files(runner)
            for remote in ({}, old_files):
                with self.subTest(remote=bool(remote)), self.assertRaises(ValueError):
                    self.history.restore_history(runner, PublicFixtureReader(remote))
                self.assertEqual(self.public_files(runner), before)

    def test_zip_path_duplicates_active_markup_missing_members_and_symlinks_are_refused(self):
        # A valid outer bundle hash is insufficient when the declared public archive is unsafe or incomplete.
        def symlink(z, rows):
            for number, (name, raw) in enumerate(rows):
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16 if number == 0 else (stat.S_IFREG | 0o644) << 16
                z.writestr(info, raw)
        edits = {
            "path escape": lambda z, rows: [z.writestr("../../outside.html", b"x")],
            "extra member": lambda z, rows: ([z.writestr(n, b) for n, b in rows], z.writestr("extra.html", b"x")),
            "duplicate": lambda z, rows: ([z.writestr(n, b) for n, b in rows], z.writestr(rows[0][0], rows[0][1])),
            "missing": lambda z, rows: [z.writestr(n, b) for n, b in rows[1:]],
            "active markup": lambda z, rows: [z.writestr(n, b + b"<script>alert(1)</script>" if n.endswith("alpha.html") else b) for n, b in rows],
            "active metadata": lambda z, rows: [z.writestr(n, b.replace(b"</head>", b'<meta http-equiv="refresh" content="0; url=https://evil.invalid/"></head>') if n.endswith("alpha.html") else b) for n, b in rows],
            "escaped local link": lambda z, rows: [z.writestr(n, b.replace(b'href="beta.html"', b'href="../../../../../../escape.html"') if n == "nodes/alpha.html" else b) for n, b in rows],
            "changed raw evidence": lambda z, rows: [z.writestr(n, b.replace(b"FAIL", b"PASS") if n.endswith("alpha.html") else b) for n, b in rows],
            "changed visible classification": lambda z, rows: [z.writestr(n, b.replace(b"<code>PROVED_REVIEWED</code>", b"<code>OPEN</code>", 1) if n.endswith("alpha.html") else b) for n, b in rows],
            "changed visible axis": lambda z, rows: [z.writestr(n, b.replace(b"<code>not_recorded</code>", b"<code>recorded</code>", 1) if n.endswith("alpha.html") else b) for n, b in rows],
            "symlink": symlink,
        }
        for label, edit in edits.items():
            with self.subTest(label=label), TemporaryDirectory() as tmp:
                root = Path(tmp); (root / "public").mkdir(); (root / "runner").mkdir()
                public, _ = self.prepare(root / "public"); manifest = self.publish(public)
                files = self.public_files(public)
                self.replace_bundle(files, manifest, edit)
                runner_data = fixture(); runner_data["graph"]["data"]["nodes"]["alpha"]["scope"] = "different reviewed checkout"
                runner, _ = self.prepare(root / "runner", runner_data); before = self.public_files(runner)
                with self.assertRaises(ValueError):
                    self.history.restore_history(runner, PublicFixtureReader(files))
                self.assertEqual(self.public_files(runner), before)

    def test_union_of_retained_aliases_cannot_be_omitted_from_a_fresh_restore(self):
        # Keeping a bundle while dropping its disappeared node alias still loses the requested old route.
        with TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "public").mkdir(); (root / "runner").mkdir()
            public, payload = self.prepare(root / "public"); self.publish(public)
            data = fixture(); data["graph"]["data"]["nodes"]["only-in-B"] = {"kind": "predicate", "classification": "HOLD"}
            payload.write_text(json.dumps(data)); self.assertEqual(self.build.run_build(public, payload).returncode, 0)
            self.publish(public)
            payload.write_text(json.dumps(fixture())); self.assertEqual(self.build.run_build(public, payload).returncode, 0)
            manifest = self.publish(public)
            manifest["aliases"] = [a for a in manifest["aliases"] if a["path"] != "nodes/only-in-B.html"]
            manifest["total_bytes"] = self.history.history_size(manifest["archives"], manifest["aliases"])
            files = self.public_files(public); self.rebind_manifest(files, manifest)
            runner, _ = self.prepare(root / "runner"); before = self.public_files(runner)
            with self.assertRaises(ValueError): self.history.restore_history(runner, PublicFixtureReader(files))
            self.assertEqual(self.public_files(runner), before)

    def test_reviewed_local_candidate_history_is_merged_with_published_history(self):
        # A fresh public A restore must keep unpublished, reviewed candidate B-only routes from the checkout.
        with TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "public").mkdir(); (root / "runner").mkdir()
            public, _ = self.prepare(root / "public"); self.publish(public)
            data = fixture(); data["graph"]["data"]["nodes"]["candidate-only"] = {"kind": "candidate", "classification": "OPEN"}
            runner, payload = self.prepare(root / "runner", data); candidate = self.publish(runner)
            candidate_bytes = (runner / "nodes/candidate-only.html").read_bytes()
            restored = self.history.restore_history(runner, PublicFixtureReader(self.public_files(public)))
            self.assertFalse(restored["bootstrap"])
            historical = (runner / "nodes/candidate-only.html").read_text()
            self.assertIn("Historical removed-node static route", historical)
            self.assertIn(data["observed_at"], historical)
            self.assertIn("OPEN", historical)
            self.assertTrue((runner / f'snapshots/{candidate["current"]}/nodes/candidate-only.html').is_file())
            self.assertEqual(self.build.run_build(runner, payload).returncode, 0)
            final = self.publish(runner)
            self.assertEqual(len(final["archives"]), 2)
            self.assertEqual(final["current"], candidate["current"])

    def test_closed_manifest_and_capacity_controls_refuse_before_download_or_write(self):
        # Extra private fields, forged descriptor targets and capacity overruns must stop admission early.
        with TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "public").mkdir(); (root / "runner").mkdir()
            public, _ = self.prepare(root / "public"); good = self.publish(public)
            runner, _ = self.prepare(root / "runner"); before = self.public_files(runner)
            mutations = [lambda m: m.update(private_token="no ingress"),
                         lambda m: m.update(origin="https://evil.invalid/"),
                         lambda m: m["archives"][0]["bundle"].update(path="../outside.zip"),
                         lambda m: m["aliases"][0].update(path="nodes/../outside.html"),
                         lambda m: m.update(total_bytes=768 * 1024 * 1024 + 1),
                         lambda m: m.update(archives=m["archives"] * 513)]
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    manifest = copy.deepcopy(good); mutation(manifest)
                    files = self.public_files(public); self.rebind_manifest(files, manifest)
                    with self.assertRaises(ValueError):
                        self.history.restore_history(runner, PublicFixtureReader(files))
                    self.assertEqual(self.public_files(runner), before)
            with patch.object(self.history, "MAX_HISTORY_BYTES", 1):
                with self.assertRaises(ValueError):
                    self.publish(public)
            self.assertEqual(self.public_files(runner), before)

    def test_reader_only_requests_fixed_origin_without_credentials_redirects_or_unbounded_reads(self):
        class Response:
            status = 200
            headers = {"Content-Length": "3"}
            def geturl(self): return "https://d6g8k5htny-coder.github.io/main/site/snapshots/index.json"
            def read(self, limit): self.limit = limit; return b"abc"
            def __enter__(self): return self
            def __exit__(self, *args): pass
        class Opener:
            def __init__(self): self.requests = []
            def open(self, req, timeout): self.requests.append(req); return Response()
        opener = Opener()
        reader = self.history.HistoryReader(opener=opener)
        self.assertEqual(reader.get_bytes("snapshots/index.json"), b"abc")
        self.assertNotIn("Authorization", opener.requests[0].headers)
        for path in ("https://evil.invalid/a", "../private", "snapshots/%2e%2e/x", "//evil.invalid"):
            with self.subTest(path=path), self.assertRaises(ValueError): reader.get_bytes(path)
        response = Response()
        class ControlledOpener:
            def open(self, request, timeout): return response
        for mutate in (lambda r: setattr(r, "geturl", lambda: "https://evil.invalid/"),
                       lambda r: setattr(r, "headers", {"Content-Encoding": "gzip"}),
                       lambda r: setattr(r, "headers", {"Content-Length": str(16 * 1024 * 1024 + 1)}),
                       lambda r: setattr(r, "headers", {"Content-Length": "4"}),
                       lambda r: setattr(r, "status", 302)):
            response = Response(); mutate(response)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.history.HistoryReader(opener=ControlledOpener()).get_bytes("snapshots/index.json")
        response = Response(); response.headers = {}
        with patch.object(self.history, "MAX_FILE_BYTES", 2), self.assertRaises(ValueError):
            self.history.HistoryReader(opener=ControlledOpener()).get_bytes("snapshots/index.json")

    def test_history_reader_retries_transient_503_then_200(self):
        history = self.history
        self.assertEqual(history.GET_RETRY_ATTEMPTS, 4)
        self.assertEqual(history.GET_RETRY_BUDGET_SECONDS, 30)
        self.assertLessEqual(
            sum(history.GET_RETRY_BACKOFF_SECONDS) + history.GET_RETRY_JITTER_SECONDS * (history.GET_RETRY_ATTEMPTS - 1),
            history.GET_RETRY_BUDGET_SECONDS)
        archive = "snapshots/" + ("ab" * 32) + "/archive.zip"

        class Response(io.BytesIO):
            def __init__(self, raw, url, status=200, headers=None):
                super().__init__(raw)
                self.status = status
                self.headers = {} if headers is None else headers
                self._url = url

            def geturl(self):
                return self._url

        class Opener:
            def __init__(self, script):
                self.script = list(script)
                self.opens = []

            def open(self, request, timeout):
                self.opens.append((request.full_url, timeout))
                action = self.script.pop(0)
                if callable(action):
                    action = action(request)
                if isinstance(action, BaseException):
                    raise action
                status, body = action
                return Response(body, request.full_url, status)

        def reader(script, **kwargs):
            opener = Opener(script)
            sleeps = []
            client = history.HistoryReader(opener=opener, pause=sleeps.append, jitter=lambda: 0, **kwargs)
            return client, opener, sleeps

        def http_error(request, code, reason, headers=None):
            return urllib.error.HTTPError(request.full_url, code, reason, {} if headers is None else headers, None)

        client, opener, sleeps = reader([
            lambda request: http_error(request, 503, "Service Unavailable"),
            (200, b"{}"),
        ])
        self.assertEqual(client.get_bytes("current.json"), b"{}")
        self.assertEqual(client.requests, 1)
        self.assertEqual(len(opener.opens), 2)
        self.assertEqual(opener.opens[0][1], 20)
        self.assertEqual(sleeps, [2])
        self.assertTrue(all(url.startswith(history.ORIGIN) for url, _ in opener.opens))

        client, opener, sleeps = reader([(503, b""), (200, b"{}")])
        self.assertEqual(client.get_bytes("snapshots/index.json"), b"{}")
        self.assertEqual(len(opener.opens), 2)
        self.assertEqual(client.requests, 1)

        client, opener, sleeps = reader([
            lambda request: http_error(request, 503, "Service Unavailable"),
            (200, b"{}"),
        ])
        client.requests = history.MAX_SNAPSHOTS + 2
        self.assertEqual(client.get_bytes("current.json"), b"{}")
        self.assertEqual(client.requests, history.MAX_SNAPSHOTS + 3)
        self.assertEqual(len(opener.opens), 2)
        with self.assertRaises(ValueError):
            client.get_bytes("current.json")
        self.assertEqual(len(opener.opens), 2)

        client, opener, sleeps = reader([lambda request: http_error(request, 503, "Service Unavailable")] * 4)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            client.get_bytes(archive)
        self.assertEqual(caught.exception.code, 503)
        self.assertIn(history.ORIGIN + archive, str(caught.exception))
        self.assertEqual(len(opener.opens), 4)
        self.assertEqual(sleeps, [2, 4, 8])
        self.assertEqual(client.requests, 1)
        self.assertLess(sum(sleeps), 30)

        client, opener, sleeps = reader([
            lambda request: http_error(request, 404, "Not Found"),
            (200, b"{}"),
        ])
        with self.assertRaises(urllib.error.HTTPError) as caught:
            client.get_bytes("snapshots/index.json")
        self.assertEqual(caught.exception.code, 404)
        self.assertIn(history.ORIGIN + "snapshots/index.json", str(caught.exception))
        self.assertEqual(len(opener.opens), 1)
        self.assertEqual(sleeps, [])

        client, opener, sleeps = reader([
            lambda request: http_error(request, 429, "Too Many Requests", {"Retry-After": "3"}),
            (200, b"{}"),
        ])
        self.assertEqual(client.get_bytes("current-meta.json"), b"{}")
        self.assertEqual(sleeps, [3])
        self.assertEqual(len(opener.opens), 2)

        client, opener, sleeps = reader([
            lambda request: urllib.error.URLError("timed out"),
            (200, b"{}"),
        ])
        self.assertEqual(client.get_bytes("current.json"), b"{}")
        self.assertEqual(len(opener.opens), 2)
        self.assertTrue(opener.opens[0][0].startswith(history.ORIGIN))

        class Clock:
            def __init__(self):
                self.value = 0.0

            def __call__(self):
                return self.value

        clock = Clock()
        recorded = []

        class BudgetOpener:
            def __init__(self):
                self.opens = []

            def open(self, request, timeout):
                self.opens.append((request.full_url, timeout))
                clock.value += timeout
                raise urllib.error.URLError("timed out")

        def pause(delay):
            recorded.append(delay)
            clock.value += delay

        budget = BudgetOpener()
        client = history.HistoryReader(opener=budget, pause=pause, clock=clock, jitter=lambda: 0)
        with self.assertRaises(urllib.error.URLError) as caught:
            client.get_bytes(archive)
        self.assertIn(history.ORIGIN + archive, str(caught.exception))
        self.assertEqual([timeout for _, timeout in budget.opens], [20, 8])
        self.assertEqual(recorded, [2])
        self.assertLessEqual(clock.value, 30)
        self.assertEqual(client.requests, 1)


if __name__ == "__main__":
    unittest.main()
