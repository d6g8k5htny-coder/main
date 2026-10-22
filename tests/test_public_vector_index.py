"""Behavioral controls for the bounded public lexical index; all Git data is local."""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    import public_vector_index as indexer
except ImportError:
    indexer = None


class PublicVectorIndexTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(indexer, "bounded public vector index implementation is missing")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # macOS /var and /tmp are symlinks; output deliberately requires a direct path.
        self.base = Path(self.tmp.name).resolve()
        self.repo = self.base / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Local test fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("remote", "add", "origin", "https://github.com/d6g8k5htny-coder/main.git")
        self.source = b"alpha alpha beta\npublic workflow\n"
        (self.repo / "AGENTS.md").write_bytes(self.source)
        (self.repo / "private-record.txt").write_text("excluded private sentinel", encoding="utf-8")
        self.commit = self.save()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().strip()

    def save(self):
        self.git("add", "-A")
        self.git("commit", "-qm", "fixture snapshot")
        return self.git("rev-parse", "HEAD")

    def build(self, commit=None):
        return indexer.build_index(self.repo, commit or self.commit)

    def test_pinned_git_bytes_exclude_dirty_untracked_and_nonallowlisted_records(self):
        # Reading the working tree or broadening the path selection leaks the sentinels.
        (self.repo / "AGENTS.md").write_text("dirty sentinel", encoding="utf-8")
        (self.repo / "untracked.txt").write_text("untracked sentinel", encoding="utf-8")
        raw = self.build()
        data = json.loads(raw)
        self.assertEqual(data["source"], {"repository": "d6g8k5htny-coder/main", "commit": self.commit})
        self.assertEqual(data["algorithm"], {"name": "lexical-feature-hashing", "version": 1,
                                             "dimensions": 256, "tokenizer": "ascii-alnum-lower"})
        self.assertEqual([d["path"] for d in data["documents"]], ["AGENTS.md"])
        doc = data["documents"][0]
        self.assertEqual(doc["sha256"], hashlib.sha256(self.source).hexdigest())
        expected_blob = hashlib.sha1(b"blob 33\0" + self.source).hexdigest()
        self.assertEqual(doc["blob_sha1"], expected_blob)
        self.assertEqual(doc["bytes"], 33)
        self.assertEqual(data["chunks"][0]["text"], self.source.decode())
        self.assertIn("/blob/" + self.commit + "/AGENTS.md#L1-L2", data["chunks"][0]["source_url"])
        for sentinel in (b"dirty sentinel", b"untracked sentinel", b"excluded private sentinel"):
            self.assertNotIn(sentinel, raw)

    def test_repeated_build_is_byte_exact_ascii_json_without_run_metadata(self):
        raw = self.build()
        self.assertEqual(raw, self.build())
        raw.decode("ascii")
        data = json.loads(raw)
        self.assertEqual(set(data), {"schema_version", "source", "algorithm", "documents", "chunks"})
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(indexer.check_index(self.repo, raw, self.commit), data)

    def test_full_rebuild_drops_deleted_documents_without_carryover(self):
        (self.repo / "docs").mkdir()
        (self.repo / "docs/WORKSPACE.md").write_text("workspace", encoding="utf-8")
        first = self.save()
        (self.repo / "AGENTS.md").unlink()
        later = self.save()
        self.assertEqual(len(json.loads(self.build(first))["documents"]), 2)
        current = json.loads(self.build(later))
        self.assertEqual([d["path"] for d in current["documents"]], ["docs/WORKSPACE.md"])
        self.assertNotIn("AGENTS.md", {c["path"] for c in current["chunks"]})

    def test_no_remaining_allowlisted_records_produces_empty_index(self):
        (self.repo / "AGENTS.md").unlink()
        data = json.loads(self.build(self.save()))
        self.assertEqual(data["documents"], [])
        self.assertEqual(data["chunks"], [])

    def test_source_commit_requires_full_lowercase_sha1_and_expected_commit(self):
        for commit in ("HEAD", self.commit[:12], self.commit.upper(), "f" * 40):
            with self.subTest(commit=commit), self.assertRaises(ValueError):
                self.build(commit)
        raw = self.build()
        (self.repo / "AGENTS.md").write_text("new public source", encoding="utf-8")
        later = self.save()
        with self.assertRaisesRegex(ValueError, "stale|match"):
            indexer.check_index(self.repo, raw, later)

    def test_other_repository_origin_is_refused(self):
        self.git("remote", "set-url", "origin", "https://github.com/other/private.git")
        with self.assertRaisesRegex(ValueError, "repository|origin"):
            self.build()

    def test_symlink_executable_and_submodule_allowlist_entries_are_refused(self):
        for mode in ("120000", "100755", "160000"):
            with self.subTest(mode=mode):
                oid = self.commit if mode == "160000" else self.git("rev-parse", self.commit + ":AGENTS.md")
                self.git("update-index", "--cacheinfo", mode + "," + oid + ",AGENTS.md")
                self.git("commit", "-qm", "nonregular allowlisted fixture")
                with self.assertRaisesRegex(ValueError, "regular|mode"):
                    self.build(self.git("rev-parse", "HEAD"))

    def test_nonregular_parent_of_allowlisted_path_is_refused(self):
        os.symlink("elsewhere", self.repo / "docs")
        with self.assertRaisesRegex(ValueError, "tree|mode|regular"):
            self.build(self.save())

    def test_source_byte_and_utf8_limits_fail_closed(self):
        for source in (b"x" * 32769, b"\xff\xfe"):
            with self.subTest(length=len(source)):
                (self.repo / "AGENTS.md").write_bytes(source)
                with self.assertRaises(ValueError):
                    self.build(self.save())

    def test_combined_source_limit_fails_instead_of_partial_index(self):
        (self.repo / "AGENTS.md").write_bytes(b"a" * 24000)
        (self.repo / "docs").mkdir()
        (self.repo / "docs/WORKSPACE.md").write_bytes(b"b" * 24000)
        (self.repo / "governance").mkdir()
        (self.repo / "governance/OP-WORKFLOW-20260930.md").write_bytes(b"c" * 24000)
        with self.assertRaisesRegex(ValueError, "total|combined"):
            self.build(self.save())

    def test_ascii_json_expansion_cannot_exceed_payload_limit(self):
        (self.repo / "AGENTS.md").write_text("\u00e9" * 16384, encoding="utf-8")
        (self.repo / "docs").mkdir()
        (self.repo / "docs/WORKSPACE.md").write_text("\u00e9" * 16384, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "payload|index.*limit"):
            self.build(self.save())

    def test_corrupt_git_object_cannot_become_trusted_source(self):
        oid = self.git("rev-parse", self.commit + ":AGENTS.md")
        path = self.repo / ".git/objects" / oid[:2] / oid[2:]
        path.chmod(0o600)
        path.write_bytes(zlib.compress(b"blob 33\0" + b"z" * 33))
        with self.assertRaises(ValueError):
            self.build()

    def test_git_replace_cannot_substitute_source_bytes(self):
        original = self.git("rev-parse", self.commit + ":AGENTS.md")
        (self.repo / "replacement.txt").write_text("replacement sentinel", encoding="utf-8")
        replacement = self.git("hash-object", "-w", "replacement.txt")
        self.git("replace", original, replacement)
        self.assertEqual(json.loads(self.build())["chunks"][0]["text"], self.source.decode())

    def test_chunk_ranges_cover_source_once_and_identity_survives_unrelated_commit(self):
        text = "alpha\n" * 500
        (self.repo / "AGENTS.md").write_text(text, encoding="utf-8")
        first = self.save()
        data = json.loads(self.build(first))
        chunks = data["chunks"]
        self.assertEqual([(c["start_char"], c["end_char"]) for c in chunks], [(0, 2048), (2048, 3000)])
        self.assertEqual("".join(c["text"] for c in chunks), text)
        self.assertEqual((chunks[0]["start_line"], chunks[0]["end_line"]), (1, 342))
        (self.repo / "private-record.txt").write_text("unrelated change", encoding="utf-8")
        later = json.loads(self.build(self.save()))
        self.assertEqual([c["id"] for c in chunks], [c["id"] for c in later["chunks"]])

    def test_sparse_vectors_count_lexical_tokens_and_search_returns_pinned_source(self):
        raw = self.build()
        data = indexer.check_index(self.repo, raw, self.commit)
        # SHA256('alpha') -> bin173; beta->231; public->117; workflow->159.
        self.assertEqual(data["chunks"][0]["vector"], [[117, 1], [159, 1], [173, 2], [231, 1]])
        results = indexer.search_index(data, "ALPHA", limit=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["path"], "AGENTS.md")
        self.assertAlmostEqual(results[0]["score"], 2 / (7 ** 0.5))
        self.assertEqual(indexer.search_index(data, "!!!"), [])
        with self.assertRaises(ValueError):
            indexer.search_index(data, "x" * 4097)
        with self.assertRaises(ValueError):
            indexer.search_index(data, "alpha", limit=0)

    def test_public_node_record_is_retrievable_with_original_owner_stops_and_links(self):
        (self.repo / "docs/site").mkdir(parents=True)
        node = {"eng.hard-gate": {"schema_version": 0, "owner": None,
                                 "stops": ["source review remains open"],
                                 "scientific_effect": "NONE",
                                 "sources": [{"url": "https://github.com/d6g8k5htny-coder/main/issues/333"}]}}
        text = json.dumps(node)
        (self.repo / "docs/site/node-status.json").write_text(text, encoding="utf-8")
        data = json.loads(self.build(self.save()))
        result = indexer.search_index(data, "eng.hard-gate")[0]
        self.assertEqual(result["path"], "docs/site/node-status.json")
        self.assertEqual(result["text"], text)
        self.assertIn('"owner": null', result["text"])
        self.assertIn('"source review remains open"', result["text"])
        self.assertIn('/issues/333', result["text"])

    def test_search_ties_are_sorted_by_path_and_character_range(self):
        (self.repo / "docs").mkdir()
        (self.repo / "docs/WORKSPACE.md").write_bytes(self.source)
        data = json.loads(self.build(self.save()))
        self.assertEqual([r["path"] for r in indexer.search_index(data, "alpha")],
                         ["AGENTS.md", "docs/WORKSPACE.md"])

    def test_checker_refuses_modified_metadata_text_vectors_or_noncanonical_json(self):
        data = json.loads(self.build())
        mutations = [lambda d: d["documents"][0].update(sha256="0" * 64),
                     lambda d: d["chunks"][0].update(text="forged"),
                     lambda d: d["chunks"][0].update(vector=[[0, 1]]),
                     lambda d: d.update(extra="unapproved")]
        for mutate in mutations:
            forged = copy.deepcopy(data)
            mutate(forged)
            with self.assertRaises(ValueError):
                raw = (json.dumps(forged, ensure_ascii=True, sort_keys=True,
                                  separators=(",", ":")) + "\n").encode("ascii")
                indexer.check_index(self.repo, raw, self.commit)
        with self.assertRaises(ValueError):
            indexer.check_index(self.repo, json.dumps(data, indent=2).encode(), self.commit)

    def test_output_write_is_external_atomic_and_refuses_symlinks_or_checkout(self):
        raw = self.build()
        output = self.base / "index.json"
        indexer.write_output(self.repo, output, raw)
        self.assertEqual(output.read_bytes(), raw)
        indexer.write_output(self.repo, output, raw)
        output.unlink()
        victim = self.base / "victim"
        victim.write_text("keep", encoding="utf-8")
        output.symlink_to(victim)
        with self.assertRaises(ValueError):
            indexer.write_output(self.repo, output, raw)
        self.assertEqual(victim.read_text(), "keep")
        with self.assertRaises(ValueError):
            indexer.write_output(self.repo, self.repo / "generated-index.json", raw)
        self.assertFalse((self.repo / "generated-index.json").exists())

    def test_output_refuses_symlink_parent_directory(self):
        real = self.base / "real"
        real.mkdir()
        alias = self.base / "alias"
        alias.symlink_to(real, target_is_directory=True)
        with self.assertRaises(ValueError):
            indexer.write_output(self.repo, alias / "index.json", self.build())
        self.assertEqual(list(real.iterdir()), [])

    def test_temporary_filename_collision_preserves_preexisting_file(self):
        collision = self.base / ".public-vector-index-collision"
        collision.write_bytes(b"preexisting unrelated bytes")
        output = self.base / "index.json"
        with patch.object(indexer.secrets, "token_hex", return_value="collision"):
            with self.assertRaises(FileExistsError):
                indexer.write_output(self.repo, output, self.build())
        self.assertTrue(collision.exists(), "failed exclusive create deleted a file it did not own")
        self.assertEqual(collision.read_bytes(), b"preexisting unrelated bytes")
        self.assertFalse(output.exists())

    def test_git_timeout_covers_a_child_that_produces_no_stdout_or_eof(self):
        fake_bin = self.base / "fake-bin"
        fake_bin.mkdir()
        fake_git = fake_bin / "git"
        fake_git.write_text("#!" + sys.executable + "\nimport time\ntime.sleep(3)\n", encoding="utf-8")
        fake_git.chmod(0o755)
        started = time.monotonic()
        with patch.dict(os.environ, {"PATH": str(fake_bin) + os.pathsep + os.defpath}):
            with patch.object(indexer, "GIT_TIMEOUT_SECONDS", 0.1, create=True):
                with self.assertRaisesRegex(ValueError, "timed out"):
                    indexer._git(self.repo, ["cat-file", "-t", self.commit], 100)
        self.assertLess(time.monotonic() - started, 2)

    def test_log_roundtrip_accepts_timestamped_github_lines_and_checks_raw_digest(self):
        raw = self.build()
        log = indexer.frame_index(raw)
        self.assertEqual(indexer.reconstruct_index(log), raw)
        timestamped = "unrelated job line\n" + "\n".join(
            "2026-10-22T23:00:00.1234567Z " + line for line in log.splitlines()) + "\n"
        self.assertEqual(indexer.reconstruct_index(timestamped), raw)
        self.assertIn(hashlib.sha256(raw).hexdigest(), log)
        self.assertIn(str(len(raw)), log.splitlines()[0])

    def test_log_missing_duplicate_reordered_or_corrupt_chunks_fail_closed(self):
        raw = self.build()
        lines = indexer.frame_index(raw).splitlines()
        self.assertGreaterEqual(len(lines), 4)
        candidates = [lines[:-1], lines[1:], lines[:1] + lines[2:],
                      lines[:2] + [lines[1]] + lines[2:],
                      [lines[0], lines[2], lines[1]] + lines[3:],
                      lines + lines]
        altered = list(lines)
        fields = altered[1].split(" ")
        payload = bytearray(base64.b64decode(fields[-1]))
        payload[0] ^= 1
        fields[-1] = base64.b64encode(payload).decode()
        altered[1] = " ".join(fields)
        candidates.append(altered)
        for candidate in candidates:
            with self.subTest(lines=len(candidate)), self.assertRaises(ValueError):
                indexer.reconstruct_index("\n".join(candidate) + "\n")

    def test_log_oversize_declared_or_actual_payload_is_refused(self):
        with self.assertRaises(ValueError):
            indexer.frame_index(b"x" * 131073)
        log = indexer.frame_index(self.build())
        first, rest = log.split("\n", 1)
        fields = first.split(" ")
        fields[2] = "131073"
        with self.assertRaises(ValueError):
            indexer.reconstruct_index(" ".join(fields) + "\n" + rest)
        with self.assertRaises(ValueError):
            indexer.reconstruct_index("x" * 262145)

    def test_cli_build_check_log_reconstruct_search_and_stale_failure(self):
        tool = ROOT / "tools/public_vector_index.py"
        output = self.base / "index.json"
        command = [sys.executable, "-B", "-S", str(tool)]
        def run(*args):
            return subprocess.run(command + list(args), cwd=self.repo,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        built = run("build", "--commit", self.commit, "--output", str(output))
        self.assertEqual(built.returncode, 0, built.stderr.decode())
        checked = run("check", "--index", str(output), "--commit", self.commit)
        self.assertEqual(checked.returncode, 0, checked.stderr.decode())
        logged = run("log", "--index", str(output))
        self.assertEqual(logged.returncode, 0, logged.stderr.decode())
        self.assertEqual(indexer.reconstruct_index(logged.stdout.decode()), output.read_bytes())
        logpath = self.base / "job.log"
        logpath.write_bytes(logged.stdout)
        restored = self.base / "restored.json"
        reconstructed = run("reconstruct", "--log", str(logpath), "--commit", self.commit,
                            "--output", str(restored))
        self.assertEqual(reconstructed.returncode, 0, reconstructed.stderr.decode())
        self.assertEqual(restored.read_bytes(), output.read_bytes())
        searched = run("search", "--index", str(output), "--commit", self.commit, "--query", "alpha")
        self.assertEqual(searched.returncode, 0, searched.stderr.decode())
        self.assertEqual(json.loads(searched.stdout)[0]["path"], "AGENTS.md")
        stale = run("check", "--index", str(output), "--commit", "f" * 40)
        self.assertNotEqual(stale.returncode, 0)

    def test_cli_log_cannot_export_forged_local_payload(self):
        path = self.base / "forged.json"
        data = json.loads(self.build())
        data["chunks"][0]["text"] = "private forged sentinel"
        path.write_text(json.dumps(data, ensure_ascii=True, sort_keys=True,
                                   separators=(",", ":")) + "\n", encoding="ascii")
        result = subprocess.run([sys.executable, "-B", "-S", str(ROOT / "tools/public_vector_index.py"),
                                 "log", "--index", str(path)], cwd=self.repo,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")

    def test_cli_search_refuses_stale_or_canonically_serialized_tampering(self):
        path = self.base / "index.json"
        original = self.build()
        path.write_bytes(original)
        (self.repo / "AGENTS.md").write_text("successor public source", encoding="utf-8")
        later = self.save()
        forged = json.loads(original)
        forged["chunks"][0]["vector"] = [[0, 999]]
        forged_raw = (json.dumps(forged, ensure_ascii=True, sort_keys=True,
                                 separators=(",", ":")) + "\n").encode("ascii")
        for expected_commit, raw in ((later, original), (self.commit, forged_raw)):
            with self.subTest(expected_commit=expected_commit):
                path.write_bytes(raw)
                result = subprocess.run([sys.executable, "-B", "-S",
                    str(ROOT / "tools/public_vector_index.py"), "search", "--index", str(path),
                    "--commit", expected_commit, "--query", "alpha"], cwd=self.repo,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b"")


if __name__ == "__main__":
    unittest.main()
