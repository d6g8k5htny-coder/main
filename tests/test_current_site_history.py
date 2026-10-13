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
            self.assertEqual((runner / "nodes/candidate-only.html").read_bytes(), candidate_bytes)
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


if __name__ == "__main__":
    unittest.main()
