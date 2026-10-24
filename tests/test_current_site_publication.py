"""Offline publication-anchor controls with an immutable Git object API fixture."""
import copy
import base64
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
import tempfile
import unittest
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/current_site_publication.py"
sys.path.insert(0, str(ROOT / "tools"))


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


class GitObjectAPI:
    """Simulates only fixed Git-data routes, computing real Git object hashes."""
    def __init__(self, module):
        self.module = module
        self.head = None
        self.objects = {}
        self.calls = []
        self.race = False
        self.unknown_patch = False

    def request(self, method, path, body=None):
        self.calls.append((method, path, copy.deepcopy(body)))
        m = self.module
        if method == "GET" and path == m.REF_READ:
            if self.head is None:
                raise urllib.error.HTTPError(m.API_ORIGIN + path, 404, "Not Found", {}, None)
            return {"ref": m.REF_NAME, "object": {"type": "commit", "sha": self.head}}
        if method == "GET":
            return copy.deepcopy(self.objects[path.rsplit("/", 1)[1]])
        if path == "/git/blobs":
            import base64
            raw = base64.b64decode(body["content"], validate=True)
            sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            self.objects[sha] = {"sha": sha, "size": len(raw), "encoding": "base64", "content": body["content"]}
            return {"sha": sha}
        if path == "/git/trees":
            blob = body["tree"][0]["sha"]
            raw = b"100644 latest.json\0" + bytes.fromhex(blob)
            sha = hashlib.sha1(b"tree " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            self.objects[sha] = {"sha": sha, "truncated": False, "tree": copy.deepcopy(body["tree"])}
            return {"sha": sha}
        if path == "/git/commits":
            raw = m.commit_bytes(body["tree"], body["parents"], body["message"], body["author"], body["committer"])
            sha = hashlib.sha1(b"commit " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            self.objects[sha] = {"sha": sha, "tree": {"sha": body["tree"]}, "parents": [{"sha": p} for p in body["parents"]],
                                 "message": body["message"], "author": body["author"], "committer": body["committer"]}
            return {"sha": sha}
        if method == "POST" and path == "/git/refs":
            if self.head is not None:
                raise ValueError("branch already exists")
            self.head = body["sha"]
        elif method == "PATCH" and path == m.REF_WRITE:
            if body["force"] is not False:
                raise AssertionError("forced update reached fixture")
            parents = self.objects[body["sha"]]["parents"]
            if self.race or parents != [{"sha": self.head}]:
                raise ValueError("non-fast-forward branch race")
            self.head = body["sha"]
            if self.unknown_patch:
                raise OSError("unknown outcome after reference update")
        else:
            raise AssertionError("unexpected mutation route")
        return {"ref": m.REF_NAME, "object": {"sha": self.head, "type": "commit"}}


class PublicReader:
    def __init__(self, raw):
        self.raw = raw
        self.calls = []

    def get_bytes(self, path):
        self.calls.append(path)
        raw = self.raw.get(path)
        if raw is None:
            raise urllib.error.HTTPError("https://d6g8k5htny-coder.github.io/main/site/" + path, 404, "Not Found", {}, None)
        return raw


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.is_file(), "publication receipt tool has not been implemented")
        spec = importlib.util.spec_from_file_location("current_site_publication", TOOL)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        m = self.module
        self.context = {"repository": m.REPOSITORY, "ref": "refs/heads/main", "commit": "a" * 40, "run_id": 42, "run_attempt": 1}
        self.candidate = {"schema_version": 1, "scientific_effect": "NONE", "repository": m.REPOSITORY,
                          "receipt_branch": m.RECEIPT_BRANCH,
                          "source": {"commit": "a" * 40, "ref": "refs/heads/main", "workflow": ".github/workflows/current-site.yml"},
                          "run": {"id": 42, "attempt": 1}, "artifact": {"sha256": "c" * 64, "files": 8, "bytes": 12345},
                          "manifest": {}, "metadata": {}, "payload": {}, "snapshot_ids": ["d" * 64]}
        payload = encoded({"scientific_effect": "NONE", "fixture": "current"})
        manifest = encoded({"schema_version": 2, "current": "d" * 64, "archives": [{"id": "d" * 64}]})
        metadata = encoded({"archive": "d" * 64, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                            "history_manifest_sha256": hashlib.sha256(manifest).hexdigest()})
        self.raw = {"snapshots/index.json": manifest, "current-meta.json": metadata, "current.json": payload}
        for field, path in (("manifest", "snapshots/index.json"), ("metadata", "current-meta.json"), ("payload", "current.json")):
            self.candidate[field] = {"bytes": len(self.raw[path]), "sha256": hashlib.sha256(self.raw[path]).hexdigest()}
        self.api = GitObjectAPI(m)
        self.reader = PublicReader(self.raw)

    def prepare(self, candidate=None, **options):
        options.setdefault("reader", PublicReader({}))
        return self.module.prepare_publication(candidate or self.candidate, api=self.api, context=self.context,
                                               allow_bootstrap=True, now="2026-10-13T15:00:00Z", **options)

    def test_prepared_anchor_is_persistent_and_only_latest_json_is_written(self):
        state = self.prepare()
        anchor = self.module.read_anchor(self.api)
        self.assertEqual(anchor, state)
        self.assertEqual(anchor["receipt"]["phase"], "prepared")
        self.assertEqual(anchor["receipt"]["scientific_effect"], "NONE")
        tree = self.api.objects[self.api.objects[self.api.head]["tree"]["sha"]]["tree"]
        self.assertEqual([(r["path"], r["type"]) for r in tree], [("latest.json", "blob")])
        self.assertFalse(any("main" in path for _, path, _ in self.api.calls))

    def test_observed_requires_exact_public_manifest_metadata_and_payload(self):
        prepared = self.prepare()
        observed = self.module.observe_publication(self.candidate, prepared, api=self.api, reader=self.reader,
                                                  context=self.context, now="2026-10-13T15:01:00Z", attempts=1)
        self.assertEqual(observed["receipt"]["phase"], "observed")
        self.assertEqual(observed["receipt"]["previous_receipt_commit"], prepared["head"])
        self.assertEqual(self.api.objects[observed["head"]]["parents"], [{"sha": prepared["head"]}])
        self.assertEqual(set(self.reader.calls), set(self.raw))

    def test_unknown_postdeploy_failure_keeps_prepared_anchor_and_refuses_old_public_pair(self):
        prepared = self.prepare()
        old_payload = encoded({"scientific_effect": "NONE", "fixture": "older"})
        old_manifest = encoded({"schema_version": 2, "current": "e" * 64, "archives": [{"id": "e" * 64}]})
        old_meta = encoded({"archive": "e" * 64, "sha256": hashlib.sha256(old_payload).hexdigest(), "bytes": len(old_payload),
                            "history_manifest_sha256": hashlib.sha256(old_manifest).hexdigest()})
        older = {"snapshots/index.json": old_manifest, "current-meta.json": old_meta, "current.json": old_payload}
        self.assertEqual(json.loads(old_meta)["history_manifest_sha256"], hashlib.sha256(old_manifest).hexdigest())
        self.assertEqual(json.loads(old_meta)["sha256"], hashlib.sha256(old_payload).hexdigest())
        with self.assertRaisesRegex(ValueError, "(recovery|readback|digest)"):
            self.module.public_readback(prepared["receipt"], reader=PublicReader(older), attempts=1)
        self.assertEqual(self.module.read_anchor(self.api), prepared)
        with self.assertRaisesRegex(ValueError, "(recovery|readback|digest)"):
            self.module.restore_publication("unused", api=self.api, reader=PublicReader(older), allow_bootstrap=True, attempts=1)

    def test_failed_observation_does_not_write_observed_receipt(self):
        prepared = self.prepare()
        wrong = dict(self.raw, **{"current.json": b"old"})
        with self.assertRaises(ValueError):
            self.module.observe_publication(self.candidate, prepared, api=self.api, reader=PublicReader(wrong),
                                            context=self.context, attempts=1)
        self.assertEqual(self.api.head, prepared["head"])

    def test_branch_race_is_rejected_without_forced_ref_update(self):
        prepared = self.prepare()
        self.api.race = True
        with self.assertRaisesRegex(ValueError, "race"):
            self.module.observe_publication(self.candidate, prepared, api=self.api, reader=self.reader,
                                            context=self.context, attempts=1)
        self.assertEqual(self.api.head, prepared["head"])
        self.assertTrue(all(body["force"] is False for method, _, body in self.api.calls if method == "PATCH"))

    def test_unknown_reference_update_can_be_recovered_idempotently(self):
        prepared = self.prepare()
        self.api.unknown_patch = True
        with self.assertRaises(OSError):
            self.module.observe_publication(self.candidate, prepared, api=self.api, reader=self.reader,
                                            context=self.context, attempts=1)
        self.api.unknown_patch = False
        recovered = self.module.observe_publication(self.candidate, prepared, api=self.api, reader=self.reader,
                                                   context=self.context, attempts=1)
        self.assertEqual(recovered["receipt"]["phase"], "observed")
        same = self.prepare()
        self.assertEqual(same["receipt"]["phase"], "observed")

    def test_missing_receipt_branch_needs_explicit_bootstrap_and_actual_remote_absence(self):
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            self.module.read_anchor(self.api)
        self.assertIsNone(self.module.read_anchor(self.api, allow_bootstrap=True))
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            self.module.restore_publication("unused", api=self.api, reader=self.reader, allow_bootstrap=True, attempts=1)
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            self.module.prepare_publication(self.candidate, api=self.api, context=self.context, reader=self.reader, allow_bootstrap=True)
        self.assertEqual(self.api.head, None)

    def test_unavailable_branch_is_not_treated_as_initial_absence(self):
        class Unavailable:
            def request(self, method, path, body=None):
                raise urllib.error.HTTPError("https://api.github.com", 503, "Unavailable", {}, None)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self.module.read_anchor(Unavailable(), allow_bootstrap=True)

    def test_source_run_and_unknown_receipt_fields_are_refused_before_mutation(self):
        for field, value in (("repository", "evil/private"), ("receipt_branch", "main"), ("scientific_effect", "PASS"), ("unexpected", "value")):
            wrong = dict(self.candidate, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.prepare(wrong)
        wrong = copy.deepcopy(self.candidate)
        wrong["source"]["commit"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "source"):
            self.prepare(wrong)
        self.assertEqual(self.api.head, None)

    def test_snapshot_regression_is_refused_against_persistent_anchor(self):
        prepared = self.prepare()
        self.module.observe_publication(self.candidate, prepared, api=self.api, reader=self.reader, context=self.context, attempts=1)
        newer = copy.deepcopy(self.candidate)
        newer["snapshot_ids"] = ["e" * 64]
        newer["run"]["attempt"] = 2
        self.context["run_attempt"] = 2
        with self.assertRaisesRegex(ValueError, "(regress|snapshot)"):
            self.prepare(newer, reader=self.reader)

    def test_corrupt_anchor_blob_or_tree_is_refused(self):
        self.prepare()
        tree = self.api.objects[self.api.objects[self.api.head]["tree"]["sha"]]
        tree["tree"][0]["path"] = "main.py"
        with self.assertRaisesRegex(ValueError, "(tree|latest)"):
            self.module.read_anchor(self.api)

    def test_corrupt_anchor_blob_bytes_are_refused(self):
        import base64
        self.prepare()
        tree = self.api.objects[self.api.objects[self.api.head]["tree"]["sha"]]
        blob = self.api.objects[tree["tree"][0]["sha"]]
        raw = base64.b64decode(blob["content"]).replace(b"prepared", b"observed")
        blob["content"] = base64.b64encode(raw).decode()
        blob["size"] = len(raw)
        with self.assertRaisesRegex(ValueError, "blob integrity"):
            self.module.read_anchor(self.api)

    def test_readback_retries_are_bounded_and_do_not_mutate_receipts(self):
        prepared = self.prepare()
        before = len([call for call in self.api.calls if call[0] != "GET"])
        stale = PublicReader({})
        with self.assertRaises(ValueError):
            self.module.public_readback(prepared["receipt"], reader=stale, attempts=3, pause=lambda _: None)
        self.assertEqual(len(stale.calls), 3)
        self.assertEqual(len([call for call in self.api.calls if call[0] != "GET"]), before)

    def test_readback_cannot_record_success_after_its_wall_clock_deadline(self):
        prepared = self.prepare()
        moment = [0]
        class ExpiredReader(PublicReader):
            def get_bytes(self, path):
                raw = super().get_bytes(path); moment[0] = 661; return raw
        reader = ExpiredReader(self.raw)
        with self.assertRaisesRegex(ValueError, "recovery"):
            self.module.public_readback(prepared["receipt"], reader=reader, attempts=67, clock=lambda: moment[0], pause=lambda _: None)
        self.assertEqual(len(reader.calls), 1)
        self.assertEqual(self.api.head, prepared["head"])

    def test_github_client_enforces_fixed_routes_and_write_body_guards(self):
        client = self.module.GitHubAPI(token="fixture-token", opener=None)
        for method, path, body in (("PATCH", "/git/refs/heads/main", {"sha": "a" * 40, "force": False}),
                                   ("DELETE", self.module.REF_WRITE, None),
                                   ("PATCH", self.module.REF_WRITE, {"sha": "a" * 40, "force": True}),
                                   ("POST", "/git/trees", {"tree": [{"path": "code.py", "mode": "100644", "type": "blob", "sha": "a" * 40}]}),
                                   ("POST", "/git/refs", {"ref": "refs/heads/main", "sha": "a" * 40})):
            with self.subTest(path=path), self.assertRaises(ValueError):
                client.request(method, path, body)
        self.assertEqual(client.requests, 0)

    def test_workflow_has_pr_specific_concurrency_and_separate_metadata_handoff(self):
        raw = (ROOT / ".github/workflows/current-site.yml").read_text()
        self.assertIn("current-research-pages-${{ github.event_name == 'pull_request' && github.event.pull_request.number || 'publish' }}", raw)
        self.assertIn("actions/download-artifact@634f93cb2916e3fdff6788551b99b062d0335ce0", raw)
        deploy = raw.split("  deploy:\n", 1)[1]
        self.assertIn("contents: write", deploy)
        self.assertNotIn("pip install", deploy)
        self.assertLess(deploy.index(" prepare "), deploy.index("actions/deploy-pages@"))
        self.assertGreater(deploy.index(" observe "), deploy.index("actions/deploy-pages@"))
        build_checkout = raw.split("  build:\n", 1)[1].split("  deploy:\n", 1)[0].split("      - name:", 1)[0]
        self.assertIn("fetch-depth: 0", build_checkout)

    def test_http_clients_bound_responses_and_keep_credentials_in_fixed_api_headers(self):
        class Response(io.BytesIO):
            status = 200
            headers = {}
            def __init__(self, raw, url):
                super().__init__(raw); self.url = url
            def geturl(self):
                return self.url
        class Transport:
            def __init__(self, raw):
                self.raw = raw; self.requests = []
            def open(self, request, timeout):
                self.requests.append((request, timeout))
                return Response(self.raw, request.full_url)
        api_transport = Transport(b"{}")
        client = self.module.GitHubAPI(token="fixture-only-secret", opener=api_transport)
        client.request("GET", self.module.REF_READ)
        request, timeout = api_transport.requests[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer fixture-only-secret")
        self.assertTrue(request.full_url.startswith(self.module.API_ORIGIN))
        self.assertEqual(timeout, 20)
        public_transport = Transport(b"public")
        public = self.module.PublicReader(opener=public_transport)
        self.assertEqual(public.get_bytes("current.json"), b"public")
        self.assertIsNone(public_transport.requests[0][0].get_header("Authorization"))
        with self.assertRaises(ValueError):
            public.get_bytes("../private")
        for raw in (b"x" * (self.module.MAX_RECEIPT_BYTES + 1), b'{"sha":1,"sha":2}'):
            client = self.module.GitHubAPI(opener=Transport(raw))
            with self.subTest(raw_length=len(raw)), self.assertRaises(ValueError):
                client.request("GET", self.module.REF_READ)
        client = self.module.GitHubAPI(opener=Transport(b"{}"))
        for _ in range(64):
            client.request("GET", self.module.REF_READ)
        with self.assertRaisesRegex(ValueError, "request bound"):
            client.request("GET", self.module.REF_READ)

    def test_qualified_metadata_package_binds_real_artifact_and_browser_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assembled = root / "assembled"; assembled.mkdir()
            site = assembled / "site"; site.mkdir()
            for path in self.module.PATHS.values():
                destination = site / path; destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((ROOT / "docs/site" / path).read_bytes())
            artifact = self.module.artifact_fingerprint(site.parent)
            payload = (site / "current.json").read_bytes()
            report = {"scientific_effect": "NONE", "checked_commit": self.context["commit"],
                      "source_payload_sha256": hashlib.sha256(payload).hexdigest(), "cases": [{"passed": True}]}
            inputs = {"scientific_effect": "NONE", "checked_commit": self.context["commit"],
                      "source_payload_sha256": hashlib.sha256(payload).hexdigest(), "assembled_docs_sha256": artifact["sha256"],
                      "assembled_docs_files": artifact["files"], "assembled_docs_bytes": artifact["bytes"]}
            # Place evidence outside the assembled artifact namespace.
            evidence = root / "evidence"; evidence.mkdir()
            (evidence / "input.json").write_bytes(encoded(inputs)); (evidence / "report.json").write_bytes(encoded(report))
            output = root / "metadata"
            candidate = self.module.package_metadata(site, evidence / "input.json", evidence / "report.json", output, context=self.context)
            self.assertEqual(self.module.load_metadata(output), candidate)
            (site / "unexpected.txt").write_text("changed after browser admission")
            with self.assertRaisesRegex(ValueError, "qualified browser input"):
                self.module.package_metadata(site, evidence / "input.json", evidence / "report.json", root / "rejected", context=self.context)

    def test_later_deploy_observes_visible_pending_receipt_before_preparing_successor(self):
        first = self.prepare()
        second = copy.deepcopy(self.candidate)
        second["run"]["attempt"] = 2
        self.context["run_attempt"] = 2
        prepared = self.prepare(second, reader=self.reader, attempts=1)
        parent = prepared["receipt"]["previous_receipt_commit"]
        old_head = self.api.head; self.api.head = parent
        observed = self.module.read_anchor(self.api); self.api.head = old_head
        self.assertEqual(observed["receipt"]["phase"], "observed")
        self.assertEqual(observed["receipt"]["previous_receipt_commit"], first["head"])
        self.assertEqual(observed["receipt"]["run"]["attempt"], 1)
        self.assertEqual(observed["receipt"]["recorded_by"]["run_attempt"], 2)
        self.assertEqual(prepared["receipt"]["phase"], "prepared")

    def test_differing_candidate_cannot_replace_pending_receipt_without_exact_public_match(self):
        first = self.prepare()
        second = copy.deepcopy(self.candidate); second["run"]["attempt"] = 2
        self.context["run_attempt"] = 2
        with self.assertRaisesRegex(ValueError, "recovery"):
            self.prepare(second, reader=PublicReader({}), attempts=1)
        self.assertEqual(self.api.head, first["head"])

    def real_publication(self, *, predecessor=False):
        site = ROOT / "docs/site"
        files = {path: (site / path).read_bytes() for path in self.module.PATHS.values()}
        frozen = {}
        if predecessor:
            fixture = json.loads((ROOT / "tests/fixtures/current_site_publication/pre_r1_resume.json").read_bytes())
            self.assertEqual(fixture["source_commit"], "f946932c1ee19859d8778b0a7af359211e472d73")
            self.assertEqual(set(fixture["files"]), {"current-meta.json", "snapshots/index.json", "nodes/math.lifetime-remainder.html"})
            for path, row in fixture["files"].items():
                raw = base64.b64decode(row["base64"], validate=True)
                self.assertEqual(len(raw), row["bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])
                frozen[path] = raw
            files.update(frozen)
        manifest = json.loads(files["snapshots/index.json"])
        for entry in manifest["archives"]:
            files[entry["bundle"]["path"]] = (site / entry["bundle"]["path"]).read_bytes()
        for row in manifest.get("current_observation", {}).get("files", []):
            files[row["path"]] = frozen.get(row["path"], (site / row["path"]).read_bytes())
        candidate = copy.deepcopy(self.candidate)
        candidate["snapshot_ids"] = sorted(entry["id"] for entry in manifest["archives"])
        for field, path in self.module.PATHS.items():
            candidate[field] = {"bytes": len(files[path]), "sha256": hashlib.sha256(files[path]).hexdigest()}
        return manifest, files, candidate

    def test_public_anchor_pins_real_history_closure_on_a_fresh_runner(self):
        manifest, files, candidate = self.real_publication()
        self.prepare(candidate)
        reader = PublicReader(files)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "fresh"; target.mkdir()
            result = self.module.restore_publication(target, api=self.api, reader=reader, history_reader=reader, attempts=1)
            self.assertFalse(result["bootstrap"])
            self.assertEqual(result["snapshots"], len(candidate["snapshot_ids"]))
            ledger = json.loads((target / ".current-site-restored-history.json").read_bytes())
            self.assertEqual(ledger["manifest_sha256"], candidate["manifest"]["sha256"])

    def test_retained_resume_table_restores_exact_bytes_without_changing_current_default(self):
        import current_site_history as history
        manifest, files, candidate = self.real_publication(predecessor=True)
        self.prepare(candidate)
        reader = PublicReader(files)
        lifetime = "nodes/math.lifetime-remainder.html"
        closeout = b"https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6351719961"
        data = json.loads(files["current.json"])
        meta = json.loads(files["current-meta.json"])
        current = history.render_live_current_v2(data, meta)[lifetime]
        self.assertIn(closeout, current)
        self.assertNotIn(closeout, files[lifetime])
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "fresh"; target.mkdir()
            self.module.restore_publication(target, api=self.api, reader=reader, history_reader=reader, attempts=1)
            for row in manifest["current_observation"]["files"]:
                self.assertEqual(history.descriptor(row["path"], (target / row["path"]).read_bytes()), row)
            self.assertEqual(history.canonical_bytes(lifetime, (target / lifetime).read_bytes()),
                             history.canonical_bytes(lifetime, files[lifetime]))
            self.assertNotIn(closeout, (target / lifetime).read_bytes())
        self.assertEqual(history.render_live_current_v2(data, meta)[lifetime], current)

    def test_unknown_resume_presentation_is_not_admitted_by_a_self_consistent_receipt(self):
        import current_site_history as history
        manifest, files, candidate = self.real_publication(predecessor=True)
        lifetime = "nodes/math.lifetime-remainder.html"
        unknown = files[lifetime] + b"\n<!-- unrecognized presentation -->\n"
        changed = history.descriptor(lifetime, unknown)
        for row in manifest["current_observation"]["files"]:
            if row["path"] == lifetime:
                row.update(changed)
        for row in manifest["aliases"]:
            if row["path"] == lifetime:
                row.update(changed)
        total = history.history_size(manifest["archives"], manifest["aliases"], manifest["current_observation"])
        manifest["total_bytes"] = total
        manifest["capacity"] = history.capacity_record(manifest["archives"], total)
        files[lifetime] = unknown
        files["snapshots/index.json"] = encoded(manifest)
        meta = json.loads(files["current-meta.json"])
        meta["history_manifest_sha256"] = hashlib.sha256(files["snapshots/index.json"]).hexdigest()
        files["current-meta.json"] = encoded(meta)
        for field, path in self.module.PATHS.items():
            candidate[field] = {"bytes": len(files[path]), "sha256": hashlib.sha256(files[path]).hexdigest()}
        self.prepare(candidate)
        reader = PublicReader(files)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "fresh"; target.mkdir()
            with self.assertRaisesRegex(ValueError, "live static reading does not match the exact observation"):
                self.module.restore_publication(target, api=self.api, reader=reader, history_reader=reader, attempts=1)
            self.assertEqual(list(target.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
