"""Additional CBZ intent cases against the delivered drift crawler.

Real Git repositories, local remotes and public behavior; no production mocks.
These cases neither execute Lean nor establish mathematical closure.
"""
import json
import pathlib
import shutil
import subprocess
import tempfile

from test_cbz_drift import ENV, FedCase, LEAN_A, codes, crawl, sh


class IntentGraphClosureTests(FedCase):
    def add_lake_edge(self, name, url, rev):
        manifest = {"packages": [{"type": "git", "url": url, "rev": rev, "name": name}]}
        self.fed.pins["b"] = self.fed.commit(
            "b", {"lake-manifest.json": json.dumps(manifest)}
        )
        self.fed.write_seed()

    def test_unheld_reachable_lake_dependency_is_discovered_and_cloned(self):
        self.fed.build()
        sha = self.fed.commit("c", {
            "lakefile.toml": 'name = "c"\n',
            "lean-toolchain": "v4.0.0\n",
            "C.lean": LEAN_A.replace("def f : Nat := 3", "def f : Nat := 7"),
        })
        with tempfile.TemporaryDirectory() as remote_base:
            remote = pathlib.Path(remote_base) / "fed" / "c"
            remote.parent.mkdir()
            shutil.move(str(self.fed.repo("c")), str(remote))
            self.add_lake_edge("c", remote.as_uri(), sha)
            clones = pathlib.Path(remote_base) / "clones"
            clones.mkdir()
            report = self.fed.crawl(clone_into=str(clones))
            resolved = {r["repository"] for r in report["repositories"] if r["resolved"]}
            self.assertIn("fed/c", resolved)
            self.assertIn("LEAN-DRIFT", codes(report, "BLOCK"))

    def test_newly_opened_local_lake_dependency_is_actually_scanned(self):
        self.fed.build()
        sha = self.fed.commit("c", {
            "lakefile.toml": 'name = "c"\n',
            "lean-toolchain": "v4.0.0\n",
            "C.lean": LEAN_A.replace("def f : Nat := 3", "def f : Nat := 7"),
        })
        self.add_lake_edge("c", "https://github.com/fed/c.git", sha)
        report = self.fed.crawl()
        self.assertIn("fed/c", {r["repository"] for r in report["repositories"] if r["resolved"]})
        self.assertIn("LEAN-DRIFT", codes(report, "BLOCK"))

    def test_child_workspace_pins_extend_the_graph(self):
        self.fed.build()
        sha = self.fed.commit("c", {
            "lakefile.toml": 'name = "c"\n',
            "C.lean": LEAN_A.replace("def f : Nat := 3", "def f : Nat := 7"),
        })
        child = self.fed.repo("a")
        sh(child, "update-index", "--add", "--cacheinfo", "160000,%s,repos/c" % sha)
        self.fed.pins["a"] = self.fed.commit("a", {
            ".gitmodules": '[submodule "c"]\npath = repos/c\nurl = https://github.com/fed/c.git\n',
            "pins.json": json.dumps({"repos": [{
                "url": "https://github.com/fed/c.git", "path": "repos/c", "pinned_sha": sha,
            }]}),
            "workspace/repositories.json": json.dumps({
                "repositories": [{"full_name": "fed/a"}, {"full_name": "fed/c"}],
            }),
        })
        self.fed.write_seed()
        report = self.fed.crawl()
        self.assertIn("fed/c", {r["repository"] for r in report["repositories"] if r["resolved"]})
        self.assertIn("LEAN-DRIFT", codes(report, "BLOCK"))

    def test_absent_declared_canonical_registry_blocks_instead_of_zero_coverage(self):
        self.fed.build()
        registry = {
            "repositories": [{"full_name": "fed/" + n} for n in ("ws", "a", "b", "reg")],
            "canonical_inputs": {"repository_registry": {
                "repository": "fed/reg", "path": "absent-registry.json",
                "observed_blob": "0" * 40,
            }},
        }
        self.fed.write_seed(extra={"workspace/repositories.json": json.dumps(registry)})
        report = self.fed.crawl()
        self.assertGreater(report["blocking"], 0, "Declared registry disappeared without blocking the gate")

    def test_missing_lake_revision_blocks_pin_synchronization(self):
        self.fed.build()
        self.add_lake_edge("a", "https://github.com/fed/a.git", "2" * 40)
        report = self.fed.crawl()
        self.assertIn("PIN-MISSING", codes(report, "BLOCK"))


class CurrentTipContractTests(FedCase):
    def test_fetch_tips_contract_observes_advanced_local_remote(self):
        self.fed.build()
        with tempfile.TemporaryDirectory() as cache_base:
            cache = pathlib.Path(cache_base) / "cached"
            subprocess.run(
                ["git", "clone", "-q", self.fed.repo("b").as_uri(), str(cache)],
                check=True, env=ENV, capture_output=True,
            )
            advanced = self.fed.commit("b", {"current-tip-witness": "new commit\n"})
            repository = crawl.Repo("fed/b", cache, fetch=True)
            self.assertEqual(
                repository.default_tip(), advanced,
                "A missing-object-only fetch is insufficient to certify current tips",
            )
