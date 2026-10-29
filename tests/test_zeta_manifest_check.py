"""Manifest CLI contract, using the repository utility on disposable synthetic files.

Run through existing unittest discovery; no packet calculation or network is used.
The child follows the parent Python optimization mode. Run normally and with -O.
"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / "experiments" /
          "zeta_fold_exploration" / "make_manifest.py")


class ZetaManifestCheck(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="zeta_manifest_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tool = self.root / "make_manifest.py"
        self.tool.write_bytes(SOURCE.read_bytes())
        self.payload = self.root / "payload.txt"
        self.payload.write_bytes(b"data\n")
        self.manifest = self.root / "MANIFEST.json"
        self.command = [sys.executable, "-I", "-B", "-S"]
        if sys.flags.optimize:
            self.command.append("-" + "O" * sys.flags.optimize)
        self.command.append(str(self.tool))
        self.generate()
        self.baseline = self.read_manifest()

    def run_tool(self, *args):
        return subprocess.run(self.command + list(args), cwd=self.root,
                              capture_output=True, timeout=10, check=False)

    def read_manifest(self):
        return json.loads(self.manifest.read_text(encoding="utf-8"))

    def write_manifest(self, data):
        self.manifest.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")

    def snapshot(self):
        return {p.relative_to(self.root).as_posix(): p.read_bytes()
                for p in self.root.rglob("*") if p.is_file()}

    def generate(self):
        run = self.run_tool()
        self.assertEqual(run.returncode, 0, run)
        self.assertEqual(run.stderr, b"")
        data = self.read_manifest()
        expected = (f"{len(data['committed_files'])} committed files, "
                    f"{data['committed_total_bytes']} bytes; "
                    f"{len(data['raw_files'])} raw entries\n").encode()
        self.assertEqual(run.stdout, expected)
        return data

    def check_report(self, required=(), optional=()):
        """Require a real protocol result, never merely an arbitrary failure."""
        before = self.snapshot()
        run = self.run_tool("--check")
        self.assertEqual(self.snapshot(), before, "--check changed the fixture")
        self.assertEqual(run.stderr, b"")
        if not required:
            self.assertEqual((run.returncode, run.stdout), (0, b"ok\n"))
            return
        self.assertEqual(run.returncode, 1, run)
        self.assertTrue(run.stdout.endswith(b"\n"), run.stdout)
        lines = run.stdout.decode("utf-8").splitlines()
        self.assertGreaterEqual(len(lines), 2, lines)
        self.assertEqual(run.stdout, ("\n".join(lines) + "\n").encode("utf-8"))
        details = lines[:-1]
        expected = {"MISMATCH " + name for name in required}
        allowed = expected | {"MISMATCH " + name for name in optional}
        self.assertTrue(expected <= set(details), (expected, details))
        self.assertTrue(set(details) <= allowed, (allowed, details))
        self.assertEqual(len(details), len(set(details)), "duplicate diagnostic")
        self.assertEqual(lines[-1], f"{len(details)} mismatches")

    def test_valid_generated_manifest_passes_without_writing(self):
        rows = self.baseline["committed_files"]
        self.assertEqual([r["path"] for r in rows], ["make_manifest.py", "payload.txt"])
        for row in rows:
            raw = (self.root / row["path"]).read_bytes()
            self.assertEqual(row["bytes"], len(raw))
            self.assertEqual(row["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(self.baseline["committed_total_bytes"],
                         sum(r["bytes"] for r in rows))
        self.check_report()

    def test_wrong_missing_or_noninteger_total_is_rejected(self):
        total = self.baseline["committed_total_bytes"]
        variants = [total + 1, str(total), float(total), True, False,
                    -1, None, 2 ** 70, [total], {"value": total}]
        for value in variants:
            with self.subTest(total=value):
                data = copy.deepcopy(self.baseline)
                data["committed_total_bytes"] = value
                self.write_manifest(data)
                self.check_report(("committed_total_bytes",))
        with self.subTest(total="MISSING"):
            data = copy.deepcopy(self.baseline)
            del data["committed_total_bytes"]
            self.write_manifest(data)
            self.check_report(("committed_total_bytes",))

    def test_duplicate_paths_reject_regardless_of_row_order(self):
        for kind in ("identical", "wrong_first", "wrong_last", "tool_itself"):
            with self.subTest(kind=kind):
                data = copy.deepcopy(self.baseline)
                row = copy.deepcopy(data["committed_files"][0 if kind == "tool_itself" else 1])
                if kind.startswith("wrong_"):
                    row["sha256"] = "0" * 64
                if kind == "wrong_first":
                    data["committed_files"].insert(0, row)
                else:
                    data["committed_files"].append(row)
                self.write_manifest(data)
                # First- versus last-row bookkeeping is not the safety contract.
                self.check_report(("duplicate committed path: " + row["path"],),
                                  optional=(row["path"],))

    def test_wrong_hash_is_rejected(self):
        data = copy.deepcopy(self.baseline)
        data["committed_files"][1]["sha256"] = "0" * 64
        self.write_manifest(data)
        self.check_report(("payload.txt",))

    def test_same_size_payload_corruption_is_rejected(self):
        self.payload.write_bytes(b"xxxx\n")
        self.check_report(("payload.txt",))

    def test_changed_size_uses_measured_total(self):
        self.payload.write_bytes(b"longer\n")
        self.check_report(("payload.txt", "committed_total_bytes"))

    def test_forged_row_size_and_matching_forged_total_reject(self):
        data = copy.deepcopy(self.baseline)
        data["committed_files"][1]["bytes"] += 1
        data["committed_total_bytes"] += 1
        self.write_manifest(data)
        self.check_report(("payload.txt", "committed_total_bytes"))

    def test_missing_member_is_rejected(self):
        data = copy.deepcopy(self.baseline)
        data["committed_files"].pop()
        self.write_manifest(data)
        self.check_report(("payload.txt",))

    def test_extra_member_is_rejected(self):
        data = copy.deepcopy(self.baseline)
        row = dict(data["committed_files"][1], path="ghost.txt")
        data["committed_files"].append(row)
        self.write_manifest(data)
        self.check_report(("ghost.txt",))

    def test_duplicate_and_total_are_both_reported(self):
        data = copy.deepcopy(self.baseline)
        data["committed_files"].append(copy.deepcopy(data["committed_files"][1]))
        data["committed_total_bytes"] += 1
        self.write_manifest(data)
        self.check_report(("duplicate committed path: payload.txt",
                           "committed_total_bytes"))

    def test_missing_manifest_is_rejected_without_creating_one(self):
        self.manifest.unlink()
        self.check_report(("make_manifest.py", "payload.txt", "committed_total_bytes"))
        self.assertFalse(self.manifest.exists())

    def test_generation_is_idempotent_and_preserves_historical_raw_row(self):
        data = copy.deepcopy(self.baseline)
        raw = dict(path="results/raw/gauss.npz", bytes=19, sha256="f" * 64,
                   regenerate="python3 stats/gauss_proc.py 500 4")
        data["raw_files"] = [raw]
        data["raw_total_bytes"] = 19
        self.write_manifest(data)
        generated = self.generate()
        self.assertEqual(generated["raw_files"], [raw])
        self.assertEqual(generated["raw_total_bytes"], 19)
        self.assertEqual(generated["committed_files"], self.baseline["committed_files"])
        first = self.manifest.read_bytes()
        self.generate()
        self.assertEqual(self.manifest.read_bytes(), first)
        self.check_report()

    def test_generation_preserves_existing_exclusion_rules(self):
        files = {"nested/kept.txt": b"kept\n", "results/raw/.gitignore": b"*\n",
                 "results/raw/sample.npz": b"not a real sample",
                 "__pycache__/cached.pyc": b"not bytecode", "loose.pyc": b"ignored"}
        for name, raw in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        generated = self.generate()
        self.assertEqual([r["path"] for r in generated["committed_files"]],
                         ["make_manifest.py", "nested/kept.txt", "payload.txt",
                          "results/raw/.gitignore"])
        self.check_report()

    def test_check_does_not_claim_to_validate_raw_metadata(self):
        data = copy.deepcopy(self.baseline)
        data["raw_files"] = [{"path": "results/raw/unverified.npz", "bytes": 999,
                              "sha256": "f" * 64}]
        data["raw_total_bytes"] = "unverified historical value"
        self.write_manifest(data)
        self.check_report()


if __name__ == "__main__":
    unittest.main()
