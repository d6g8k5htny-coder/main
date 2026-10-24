"""Offline reader build controls: real payloads, files and parsed HTML."""
import copy
import hashlib
import importlib.util
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
TOOL = ROOT / "tools/current_site_build.py"
MAIN = "d6g8k5htny-coder/main"
MATH = "d6g8k5htny-coder/Math-"
COMMIT = "a" * 40
MATH_COMMIT = "b" * 40
START = "<!-- CURRENT-SITE:START -->"
END = "<!-- CURRENT-SITE:END -->"


class Page(HTMLParser):
    def __init__(self, raw):
        super().__init__()
        self.tags = []
        self.links = []
        self.text = []
        self.feed(raw)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if tag == "a":
            self.links.append(attrs.get("href", ""))

    def handle_data(self, data):
        self.text.append(data)


def identity(repository, path):
    return {"repository": repository, "commit": COMMIT if repository == MAIN else MATH_COMMIT,
            "path": path, "blob": "c" * 40, "bytes": 123}


def fixture():
    nodes = {
        "alpha": {"kind": "analytic_proof", "classification": "PROVED_REVIEWED",
                  "source": "proof/PROOF.md", "review_disposition": "FAIL",
                  "review_source": "https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-123",
                  "notes": "<script>alert('label')</script>", "scope": "a < b & c"},
        "beta": {"kind": "axiom", "classification": "HOLD", "source": "missing/PROOF.md"},
        "gamma": {"kind": "computation", "classification": "OPEN_ACTIVE",
                  "review_source": "javascript:alert(1)", "source": "../../private/secret"},
    }
    nodes.update({f"node-{i}": {"kind": "predicate", "classification": "OPEN_ACTIVE"}
                  for i in range(46)})
    return {
        "schema_version": 1, "observed_at": "2026-10-13T12:00:00Z", "scientific_effect": "NONE",
        "repositories": [{"repository": MAIN, "commit": COMMIT, "tree": "d" * 40},
                         {"repository": MATH, "commit": MATH_COMMIT, "tree": "e" * 40}],
        "sources": [identity(MATH, "proof/PROOF.md"), identity(MAIN, "proof/PROOF.md")],
        "graph": {"identity": {**identity(MATH, "frontiers/gate/GRAPH.json"), "sha256": "f" * 64},
                  "data": {"nodes": nodes, "edges": [
                      {"from": "alpha", "to": "beta", "required": True, "relation": "uses <premise>"},
                      {"from": "gamma", "to": "alpha", "required": False, "relation": "counterexample"}]}},
        "equations": [],
        "references": [{**identity(MAIN, "STATUS.md"), "sha256": "f" * 64},
                       {**identity(MAIN, "docs/RESEARCH_INDEX.md"), "sha256": "f" * 64},
                       {**identity(MATH, "PROOF_INDEX.md"), "sha256": "f" * 64}],
    }


class CurrentSiteBuildTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.is_file(), "the offline current-site renderer has not been implemented")

    def prepare(self, root, data=None):
        site = root / "site"
        site.mkdir()
        for name in ("index.html", "research.html", "dependencies.html"):
            prefix = '<!doctype html><meta charset="utf-8">'
            if name == "research.html":
                prefix += '<meta http-equiv="Content-Security-Policy" content="script-src \'self\'">'
            (site / name).write_bytes((prefix + "\n<main><p>Frozen cut 2026-09-25 &amp; identity.</p>\n"
                                      + START + "old panel" + END
                                      + "\n<pre>Historical evidence stays exact.\r\n</pre></main>").encode())
        (site / "historic.json").write_bytes(b'{"date":"2026-09-25","exact":"17/31"}\r\n')
        for name in ("brand.css", "style.css", "current.css"):
            (site / name).write_text("body { color: navy; }\n")
        (site / "current.mjs").write_text("// Shared loader fixture; these Python tests do not execute JavaScript.\n")
        payload = root / "payload.json"
        payload.write_bytes(json.dumps(data or fixture(), ensure_ascii=False, indent=2).encode() + b"\n")
        return site, payload

    def run_build(self, site, payload):
        return subprocess.run([sys.executable, "-B", "-S", str(TOOL), "--site", str(site),
                               "--payload", str(payload)], capture_output=True, text=True)

    def snapshot(self, site):
        return {p.relative_to(site).as_posix(): p.read_bytes()
                for p in site.rglob("*") if p.is_file()}

    def test_two_offline_builds_are_identical_and_preserve_historical_bytes(self):
        # A wall-clock timestamp or a rewrite outside the marked section breaks this contract.
        with TemporaryDirectory() as tmp:
            site, payload = self.prepare(Path(tmp))
            before = self.snapshot(site)
            first = self.run_build(site, payload)
            self.assertEqual(first.returncode, 0, first.stderr)
            built = self.snapshot(site)
            second = self.run_build(site, payload)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(built, self.snapshot(site))
            self.assertEqual(first.stdout, second.stdout)
            self.assertEqual(built["current.json"], payload.read_bytes())
            self.assertEqual(built["historic.json"], before["historic.json"])
            for name in ("index.html", "research.html", "dependencies.html"):
                old_prefix, old_suffix = before[name].split(START.encode())[0], before[name].split(END.encode())[1]
                self.assertTrue(built[name].startswith(old_prefix + START.encode()))
                self.assertTrue(built[name].endswith(END.encode() + old_suffix))
            meta = json.loads(built["current-meta.json"])
            self.assertEqual(meta["sha256"], hashlib.sha256(payload.read_bytes()).hexdigest())
            self.assertEqual(meta["bytes"], len(payload.read_bytes()))
            self.assertEqual(meta["observed_at"], "2026-10-13T12:00:00Z")

    def test_every_node_has_static_evidence_metadata_and_typed_neighbor_navigation(self):
        # Omitting isolated/absent nodes or changing edge direction breaks no-JS parity.
        with TemporaryDirectory() as tmp:
            site, payload = self.prepare(Path(tmp))
            result = self.run_build(site, payload)
            self.assertEqual(result.returncode, 0, result.stderr)
            pages = list((site / "nodes").glob("*.html"))
            self.assertEqual(len(pages), 50)
            index = Page((site / "nodes/index.html").read_text())
            for node in fixture()["graph"]["data"]["nodes"]:
                self.assertIn(node + ".html", index.links)
                parsed = Page((site / "nodes" / (node + ".html")).read_text())
                self.assertFalse(any(tag == "script" for tag, attrs in parsed.tags))
                for axis in ("source", "review", "kernel", "computation", "alignment"):
                    self.assertTrue(any(attrs.get("data-axis") == axis for tag, attrs in parsed.tags))
                self.assertIn("artifact roles: unknown", " ".join(parsed.text))
                self.assertIn("custody: unknown", " ".join(parsed.text))
            alpha = Page((site / "nodes/alpha.html").read_text())
            digest = hashlib.sha256(payload.read_bytes()).hexdigest()
            self.assertIn("beta.html", alpha.links)
            self.assertIn("gamma.html", alpha.links)
            self.assertIn("../current.json", alpha.links)
            text = " ".join(alpha.text)
            self.assertIn("Outgoing required", text)
            self.assertIn("Incoming context", text)
            self.assertIn("uses <premise>", text)
            self.assertIn('"review_disposition": "FAIL"', text)
            self.assertIn("PROVED_REVIEWED", text)
            axes = {a["data-axis"]: a["data-state"] for t, a in alpha.tags if "data-axis" in a}
            self.assertEqual(axes, {"source": "recorded", "review": "recorded", "kernel": "not_recorded",
                                    "computation": "not_recorded", "alignment": "not_recorded"})

    def test_live_node_pages_resume_from_recorded_review_without_inventing_task_state(self):
        # Promoting a review issue, lookalike metadata or a shared hash to task state fails here.
        pin = "math.d5-component.punctured-pin-proof"
        with TemporaryDirectory() as tmp:
            data = fixture()
            nodes = data["graph"]["data"]["nodes"]
            nodes["node-0"].update({"review_issue": 67, "review_disposition": "<script>ACCEPT</script>",
                                    "next_action": "merge now", "owner": "Claude S", "status": "complete"})
            nodes["node-1"] = {"kind": "proof", "classification": "PROVED_REVIEWED", "source": "proof/PROOF.md", "fingerprint": "f" * 64}
            nodes[pin] = dict(nodes.pop("node-2"), source="proof/PROOF.md", fingerprint="f" * 64)
            site, payload = self.prepare(Path(tmp), data)
            result = self.run_build(site, payload)
            self.assertEqual(result.returncode, 0, result.stderr)
            task = "https://github.com/d6g8k5htny-coder/main/pull/327#issuecomment-6321244764"
            sections = {}
            for node in nodes:
                raw = (site / "nodes" / (node + ".html")).read_text()
                self.assertEqual(raw.count('<section id="resume-node"'), 1)
                section = raw[raw.index('<section id="resume-node"'):].split("</section>", 1)[0]
                self.assertLess(raw.index('<section id="resume-node"'), raw.index("Independent evidence metadata axes"))
                parsed = Page(section)
                text = " ".join(" ".join(parsed.text).split())
                sections[node] = (parsed, text)
                self.assertFalse(any(tag == "script" for tag, attrs in parsed.tags))
                self.assertIn("Resume this node", text)
                self.assertIn("2026-10-13T12:00:00Z", text)
                self.assertIn("before any ownership- or head-dependent action", text)
                self.assertIn("this static page cannot evaluate freshness", text)
                self.assertIn(f"commit {MATH_COMMIT}", text)
                self.assertIn("f" * 64, text)
                for label in ("Next action", "Missing input", "Owner evidence", "Retained stop"):
                    self.assertIn(f"{label} not recorded in this snapshot", text)
                self.assertEqual(task in parsed.links, node == pin)
            parsed, text = sections["alpha"]
            self.assertIn("https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-123", parsed.links)
            self.assertIn("not the current task, an owner claim or a completion record", text)
            self.assertIn("No explicitly identified task record for this node", text)
            parsed, text = sections["node-0"]
            self.assertIn("<script>ACCEPT</script>", text)
            self.assertIn("67 Issue number recorded without a repository in this snapshot; no link is inferred.", text)
            self.assertFalse(any("/issues/67" in href for href in parsed.links))
            self.assertNotIn("merge now", text)
            self.assertNotIn("Claude S", text)
            self.assertNotIn("complete", text.replace("completion record", ""))
            parsed, text = sections["gamma"]
            self.assertFalse(any("javascript:" in href for href in parsed.links))
            parsed, text = sections["node-1"]
            self.assertIn("Review metadata: not recorded in this snapshot", text)
            parsed, text = sections[pin]
            self.assertIn("https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6321679618", parsed.links)
            self.assertIn("not scientific acceptance, source-binding admission, a registration or an owner claim", text)
            self.assertIn("Site-maintained pointer, not part of the graph snapshot", text)
            archived = list((site / "snapshots").rglob("*.html"))
            self.assertTrue(archived)
            self.assertFalse(any(b"resume-node" in page.read_bytes() for page in archived))

    def test_source_links_only_resolve_the_exact_math_inventory_and_external_links_stay_separate(self):
        # A fabricated blob link for a missing/hostile path or a wrong-repository join fails here.
        with TemporaryDirectory() as tmp:
            data = fixture()
            data["sources"] = [identity(MAIN, "proof/PROOF.md")]
            site, payload = self.prepare(Path(tmp), data)
            result = self.run_build(site, payload)
            self.assertEqual(result.returncode, 0, result.stderr)
            alpha = Page((site / "nodes/alpha.html").read_text())
            text = " ".join(alpha.text)
            self.assertIn("Source unavailable", text)
            self.assertIn("Recorded external review reference (mutable)", text)
            self.assertIn("https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-123", alpha.links)
            self.assertFalse(any("/blob/" in href and href.endswith("/proof/PROOF.md") for href in alpha.links))
            for name in ("beta", "gamma"):
                page = Page((site / "nodes" / (name + ".html")).read_text())
                self.assertIn("Source unavailable", " ".join(page.text))
                self.assertFalse(any("javascript:" in href or "secret" in href or "missing/PROOF" in href
                                     for href in page.links))

    def test_hostile_metadata_is_inert_and_research_uses_only_the_shared_loader(self):
        # Inserting raw labels into markup rather than escaped text would add a script tag.
        with TemporaryDirectory() as tmp:
            data = fixture()
            data["graph"]["data"]["nodes"]["alpha"]["classification"] = '\"><img src=x onerror=alert(1)>'
            site, payload = self.prepare(Path(tmp), data)
            result = self.run_build(site, payload)
            self.assertEqual(result.returncode, 0, result.stderr)
            alpha = Page((site / "nodes/alpha.html").read_text())
            self.assertFalse(any(tag in ("script", "img", "iframe") for tag, attrs in alpha.tags))
            self.assertIn("<script>alert('label')</script>", " ".join(alpha.text))
            research = Page((site / "research.html").read_text())
            scripts = [attrs for tag, attrs in research.tags if tag == "script"]
            self.assertEqual(len(scripts), 1)
            self.assertEqual(scripts[0].get("type"), "module")
            source = urlsplit(scripts[0].get("src", ""))
            self.assertEqual((source.scheme, source.netloc, source.path), ("", "", "current.mjs"))
            self.assertFalse(any(set(attrs.get("class", "").split()) &
                                 {"current-search", "current-results", "current-more"}
                                 for tag, attrs in research.tags))
            self.assertTrue(any(attrs.get("content") == "script-src 'self'" for tag, attrs in research.tags))
            section = [attrs for tag, attrs in research.tags if attrs.get("id") == "current-research"]
            self.assertEqual(len(section), 1)
            self.assertEqual(section[0]["data-current-digest"], hashlib.sha256(payload.read_bytes()).hexdigest())
            self.assertEqual(int(section[0]["data-current-bytes"]), len(payload.read_bytes()))
            self.assertTrue(any(attrs.get("class") == "current-state" for tag, attrs in research.tags))
            home = Page((site / "index.html").read_text())
            section = [attrs for tag, attrs in home.tags if attrs.get("id") == "current-research"][0]
            self.assertEqual(section["data-current-digest"], hashlib.sha256(payload.read_bytes()).hexdigest())
            self.assertEqual(int(section["data-current-bytes"]), len(payload.read_bytes()))
            self.assertTrue(any(attrs.get("class") == "current-search" for tag, attrs in home.tags))
            for reference in ("STATUS.md", "docs/RESEARCH_INDEX.md", "PROOF_INDEX.md"):
                self.assertTrue(any(href.endswith("/" + reference) for href in research.links))

    def test_research_template_allows_the_same_origin_current_module(self):
        research = Page((ROOT / "docs/site/research.html").read_text())
        policies = [attrs["content"] for tag, attrs in research.tags if tag == "meta"
                    and attrs.get("http-equiv", "").lower() == "content-security-policy"]
        self.assertEqual(len(policies), 1)
        directives = {parts[0]: parts[1:] for directive in policies[0].split(";")
                      if (parts := directive.split())}
        effective = directives.get("script-src-elem", directives.get("script-src", directives.get("default-src", [])))
        self.assertEqual(effective, ["'self'"])
        self.assertEqual(directives.get("script-src"), ["'self'"])

    def test_formal_remains_script_free_when_research_has_age_updates(self):
        with TemporaryDirectory() as tmp:
            site, payload = self.prepare(Path(tmp))
            formal = (site / "research.html").read_bytes().replace(b"script-src 'self'", b"script-src 'none'")
            (site / "formal.html").write_bytes(formal)
            result = self.run_build(site, payload)
            self.assertEqual(result.returncode, 0, result.stderr)
            parsed = Page((site / "formal.html").read_text())
            self.assertFalse(any(tag == "script" or attrs.get("class") == "current-search" for tag, attrs in parsed.tags))
            self.assertTrue(any(attrs.get("content") == "script-src 'none'" for tag, attrs in parsed.tags))
            built = (site / "formal.html").read_bytes()
            self.assertTrue(built.startswith(formal.split(START.encode())[0] + START.encode()))
            self.assertTrue(built.endswith(END.encode() + formal.split(END.encode())[1]))

    def test_script_blocked_research_refuses_before_any_site_write(self):
        with TemporaryDirectory() as tmp:
            site, payload = self.prepare(Path(tmp))
            research = site / "research.html"
            research.write_bytes(research.read_bytes().replace(b"script-src 'self'", b"script-src 'none'"))
            before = self.snapshot(site)
            result = self.run_build(site, payload)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("current freshness loader requires a script-enabled CSP", result.stderr)
            self.assertEqual(before, self.snapshot(site))

    def test_invalid_payload_or_missing_marker_fails_before_any_site_write(self):
        # Admission must finish before mutations, preserving the last deployable tree on refusal.
        mutations = [lambda d: d["repositories"][0].update(commit="main"),
                     lambda d: d["repositories"][0].update(private_notes="no ingress"),
                     lambda d: d["graph"].update(private_notes="no ingress"),
                     lambda d: d["sources"][0].update(path="../private.md"),
                     lambda d: d.update(private_credentials="must not enter the public payload"),
                     lambda d: d["sources"][0].update(private_notes="must not enter public inventory"),
                     lambda d: d["sources"][0].update(path="operational/public-looking.md"),
                     lambda d: d["graph"]["data"]["edges"][0].update(to="missing"),
                     lambda d: d["graph"]["data"]["edges"].append(
                         {"from": "beta", "to": "alpha", "required": True, "relation": "cycle"})]
        for mutate in mutations:
            with self.subTest(mutate=mutate), TemporaryDirectory() as tmp:
                data = copy.deepcopy(fixture())
                mutate(data)
                site, payload = self.prepare(Path(tmp), data)
                before = self.snapshot(site)
                result = self.run_build(site, payload)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.snapshot(site), before)
        with TemporaryDirectory() as tmp:
            site, payload = self.prepare(Path(tmp))
            (site / "research.html").write_text("<p>Missing generated section.</p>")
            before = self.snapshot(site)
            self.assertNotEqual(self.run_build(site, payload).returncode, 0)
            self.assertEqual(self.snapshot(site), before)

    def test_non_directory_node_destination_is_refused_before_any_write(self):
        # Preflight must detect an unavailable output parent before replacing the source panel.
        with TemporaryDirectory() as tmp:
            site, payload = self.prepare(Path(tmp))
            (site / "nodes").write_bytes(b"existing non-directory content")
            before = self.snapshot(site)
            result = self.run_build(site, payload)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(self.snapshot(site), before)

    def test_equation_digest_binds_the_exact_text_instead_of_only_digest_format(self):
        # A forged well-formed digest must not allow altered source equations into publication.
        with TemporaryDirectory() as tmp:
            data = fixture()
            data["equations"] = [{"id": "equation-1", "repository": MAIN, "commit": COMMIT,
                                  "path": "proof/PROOF.md", "text": "x = 1\n", "line_start": 1,
                                  "line_end": 1, "sha256": "f" * 64}]
            site, payload = self.prepare(Path(tmp), data)
            before = self.snapshot(site)
            refused = self.run_build(site, payload)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("equation text digest", refused.stderr)
            self.assertEqual(self.snapshot(site), before)
            data["equations"][0]["sha256"] = hashlib.sha256(b"x = 1\n").hexdigest()
            payload.write_text(json.dumps(data))
            self.assertEqual(self.run_build(site, payload).returncode, 0)

    def test_repeated_large_source_references_are_refused_before_static_rendering(self):
        # A small source payload must not expand into gigabytes of repeated reference HTML.
        spec = importlib.util.spec_from_file_location("build_budget", TOOL)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertTrue(callable(getattr(module, "ensure_render_budget", None)), "pre-render output budget is missing")
        data = fixture()
        data["references"] += [{**identity(MAIN, f"proof-{i}/" + "長" * 4000 + ".md"), "sha256": "f" * 64} for i in range(13)]
        data["graph"]["data"]["nodes"].update({f"large-{i}": {"kind": "predicate", "classification": "OPEN"} for i in range(1000)})
        meta = {"sha256": "f" * 64, "bytes": 123, "observed_at": data["observed_at"]}
        with self.assertRaisesRegex(ValueError, "pre-render.*capacity"):
            module.ensure_render_budget(data, meta)
        with TemporaryDirectory() as tmp:
            site, payload = self.prepare(Path(tmp), data)
            before = self.snapshot(site)
            refused = self.run_build(site, payload)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("pre-render", refused.stderr)
            self.assertEqual(self.snapshot(site), before)

    def test_removed_and_renamed_nodes_keep_their_snapshot_and_source_bound_aliases(self):
        # Deleting old aliases or rebinding their neighbor/payload links to a new snapshot loses evidence.
        with TemporaryDirectory() as tmp:
            site, payload = self.prepare(Path(tmp))
            raw1 = payload.read_bytes()
            import current_site_history as history
            digest1 = history.mathematical_digest(fixture())
            first = self.run_build(site, payload)
            self.assertEqual(first.returncode, 0, first.stderr)
            before = self.snapshot(site)
            archive1 = site / "snapshots" / digest1
            self.assertTrue((archive1 / "nodes/alpha.html").is_file())
            projected = json.loads((archive1 / "current.json").read_bytes())
            self.assertEqual(projected["coverage"], "mathematical-material-only")
            self.assertEqual(projected["graph"], fixture()["graph"])
            self.assertEqual(projected["observed_at"], fixture()["observed_at"])
            self.assertEqual(len(list((archive1 / "nodes").glob("*.html"))), 50)
            old_mtimes = {p.relative_to(archive1).as_posix(): p.stat().st_mtime_ns
                          for p in archive1.rglob("*") if p.is_file()}
            data2 = fixture()
            data2["observed_at"] = "2026-10-13T13:00:00Z"
            nodes = data2["graph"]["data"]["nodes"]
            nodes["delta"] = nodes.pop("alpha")
            nodes.pop("beta")
            data2["graph"]["data"]["edges"] = [
                {"from": "gamma", "to": "delta", "required": False, "relation": "new context"}]
            payload.write_bytes(json.dumps(data2, ensure_ascii=False, indent=2).encode() + b"\n")
            digest2 = history.mathematical_digest(data2)
            second = self.run_build(site, payload)
            self.assertEqual(second.returncode, 0, second.stderr)
            after = self.snapshot(site)
            for name, content in before.items():
                if name.startswith(f"snapshots/{digest1}/"):
                    self.assertEqual(after[name], content, name)
            self.assertEqual(old_mtimes, {p.relative_to(archive1).as_posix(): p.stat().st_mtime_ns
                                         for p in archive1.rglob("*") if p.is_file()})
            self.assertTrue((site / "snapshots" / digest2 / "nodes/delta.html").is_file())
            self.assertFalse((site / "snapshots" / digest2 / "nodes/beta.html").exists())
            old = Page((site / "nodes/alpha.html").read_text())
            self.assertIn(f"../snapshots/{digest1}/current.json", old.links)
            self.assertIn(f"../snapshots/{digest1}/nodes/beta.html", old.links)
            self.assertIn("2026-10-13T12:00:00Z", " ".join(old.text))
            self.assertIn("Historical removed-node static route", " ".join(old.text))
            self.assertIn("PROVED_REVIEWED", " ".join(old.text))
            self.assertIn("FAIL", " ".join(old.text))
            archived = Page((archive1 / "nodes/alpha.html").read_text())
            for target in ("../current.json", "index.html", "beta.html", "../../../index.html"):
                self.assertIn(target, archived.links)
            self.assertFalse(any(tag == "script" for tag, attrs in archived.tags))
            for tag, attrs in archived.tags:
                if tag == "link" and attrs.get("rel") == "stylesheet":
                    self.assertTrue(attrs["href"].startswith("../../../"))
                    self.assertTrue((archive1 / "nodes" / urlsplit(attrs["href"]).path).resolve().is_file())

    def test_known_snapshot_is_not_rewritten_after_asset_release_and_conflicts_refuse_before_writes(self):
        # Release-key normalization must admit intact archives without repairing or overwriting damaged evidence.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            site, payload = self.prepare(root)
            self.assertEqual(self.run_build(site, payload).returncode, 0)
            subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "add", "site"], check=True, capture_output=True)
            released = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "tools/site_asset_release.py"),
                                       "--site", str(site)], text=True, capture_output=True)
            self.assertEqual(released.returncode, 0, released.stderr)
            digest = json.loads((site / "current-meta.json").read_bytes())["archive"]
            archive = site / "snapshots" / digest
            self.assertTrue(archive.is_dir(), "known snapshots need a preserved immutable directory")
            before = {p.relative_to(archive).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in archive.rglob("*") if p.is_file()}
            rebuilt = self.run_build(site, payload)
            self.assertEqual(rebuilt.returncode, 0, rebuilt.stderr)
            self.assertEqual(before, {p.relative_to(archive).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
                                     for p in archive.rglob("*") if p.is_file()})
            page = archive / "nodes/alpha.html"
            page.write_bytes(page.read_bytes().replace(b"PROVED_REVIEWED", b"altered evidence"))
            before_conflict = self.snapshot(site)
            refused = self.run_build(site, payload)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("historical renderer", refused.stderr)
            self.assertEqual(self.snapshot(site), before_conflict)

    def test_incomplete_existing_snapshot_is_refused_without_filling_missing_pages(self):
        # Repairing a truncated known archive silently would hide custody damage.
        with TemporaryDirectory() as tmp:
            site, payload = self.prepare(Path(tmp))
            self.assertEqual(self.run_build(site, payload).returncode, 0)
            digest = json.loads((site / "current-meta.json").read_bytes())["archive"]
            missing = site / "snapshots" / digest / "nodes/beta.html"
            self.assertTrue(missing.is_file(), "the initial snapshot node page must exist")
            missing.unlink()
            before = self.snapshot(site)
            refused = self.run_build(site, payload)
            self.assertNotEqual(refused.returncode, 0)
            self.assertFalse(missing.exists())
            self.assertEqual(self.snapshot(site), before)

    def test_unknown_nested_zip_cannot_hide_in_an_unregistered_snapshot(self):
        # The one root archive.zip allowance must not skip any similarly named nested file.
        with TemporaryDirectory() as tmp:
            site, payload = self.prepare(Path(tmp))
            self.assertEqual(self.run_build(site, payload).returncode, 0)
            archive_id = json.loads((site / "current-meta.json").read_bytes())["archive"]
            (site / "snapshots" / archive_id / "nodes/archive.zip").write_bytes(b"unadmitted content")
            before = self.snapshot(site)
            refused = self.run_build(site, payload)
            self.assertNotEqual(refused.returncode, 0)
            self.assertEqual(self.snapshot(site), before)


if __name__ == "__main__":
    unittest.main()
