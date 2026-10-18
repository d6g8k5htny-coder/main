"""Single-point validator mutants for main#427, with closure.json re-pinned so the
closure self-hash cannot mask what the PR's own tests and harness detect."""
import hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path

SRC = Path(sys.argv[1]).resolve()          # repo checkout at the PR head
PKG = "experiments/reactor_v0"
WF = ".github/workflows/reactor-v0-controls.yml"
ONLY = set(sys.argv[2:])

M = [
 ("S1", "status: accept anything but ACCEPTED", 'document["scientific_status_effect"] == "NONE"', 'document["scientific_status_effect"] != "ACCEPTED"'),
 ("S2", "control schema_version: drop int type check", 'document["schema_version"] == 1 and type(document["schema_version"]) is int,\n             "unsupported control schema"', 'document["schema_version"] == 1,\n             "unsupported control schema"'),
 ("S3", "scope check disabled", '_require(document["scope"] ==', '_require(True or document["scope"] =='),
 ("S4", "campaign_status check disabled", '_require(document["campaign_status"] ==', '_require(True or document["campaign_status"] =='),
 ("S5", "local filename: allow slash", '"/" not in document[key] and ', ''),
 ("S5b", "local filename: allow . and ..", 'document[key] not in ("", ".", "..") and', 'document[key] and'),
 ("S6", "rational: allow zero denominator", 'document["denominator"] > 0', 'document["denominator"] >= 0'),
 ("S7", "rational: drop numerator int check", 'type(document["numerator"]) is int and\n', ''),
 ("S8", "duplicate JSON keys allowed", ', object_pairs_hook=_object', ''),
 ("S9", "strict keys: drop dict type check", 'type(document) is dict and set(document) == expected', 'set(document) == expected'),
 ("D1", "alpha may be zero", '_require(alpha > 0,', '_require(alpha >= 0,'),
 ("D2", "amplitude_scale may be zero", '_require(amplitude_scale > 0,', '_require(amplitude_scale >= 0,'),
 ("D3", "threshold sign unchecked", '_require(threshold >= 0,', '_require(True,'),
 ("D4", "expected_control sign unchecked", '_require(expected_control >= 0,', '_require(True,'),
 ("F1", "fixture grid_side check disabled", '_require(type(side) is int and side == 4,', '_require(True or side == 4,'),
 ("F2", "fixture values int check disabled", 'all(type(value) is int for value in values)', 'True'),
 ("F3", "fixture exclusions check disabled", '_require(document["exclusions"] ==', '_require(True or document["exclusions"] =='),
 ("P1", "source repository check disabled", '_require(source["repository"] ==', '_require(True or source["repository"] =='),
 ("P2", "commit hex check dropped", 'len(source["commit"]) == 40 and\n                 LOWER_HEX.fullmatch(source["commit"]) is not None', 'len(source["commit"]) == 40'),
 ("P3", "raw_sha256 hex check dropped", 'len(source["raw_sha256"]) == 64 and\n                 LOWER_HEX.fullmatch(source["raw_sha256"]) is not None', 'len(source["raw_sha256"]) == 64'),
 ("P4", "blob hex check dropped", 'len(source["git_blob_sha"]) == 40 and\n                 LOWER_HEX.fullmatch(source["git_blob_sha"]) is not None', 'len(source["git_blob_sha"]) == 40'),
 ("P6", "theorem flag: any non-True accepted", 'source["theorem_conclusion_consumed"] is False', 'source["theorem_conclusion_consumed"] is not True'),
 ("P7", "source_status check disabled", '_require(source["source_status"] ==', '_require(True or source["source_status"] =='),
 ("P8", "source count widened", 'len(sources) == 2', 'len(sources) >= 1'),
 ("P9", "PR->commit/path pin check disabled", '_require((source["commit"], source["path"]) ==', '_require(True or (source["commit"], source["path"]) =='),
 ("P10", "PR set equality dropped", 'set(pull_requests) == set(SOURCE_PINS) and', 'True and'),
 ("P11", "PR uniqueness dropped", 'len(set(pull_requests)) == len(pull_requests)', 'True'),
 ("P12", "commit uniqueness dropped", 'len(set(commits)) == len(commits)', 'True'),
 ("H1", "LOWER_HEX accepts anything", 'r"[0-9a-f]+\\Z"', 'r".+\\Z"'),
 ("V1", "source length compare disabled", 'len(raw) == source["byte_length"]', 'True'),
 ("V2", "source sha256 compare disabled", 'hashlib.sha256(raw).hexdigest() == source["raw_sha256"]', 'True'),
 ("V3", "source blob compare disabled", 'blob == source["git_blob_sha"]', 'True'),
 ("V4", "load never verifies sources even with env", 'os.environ.get("REACTOR_VERIFY_SOURCES") == "1"', 'False'),
 ("C1", "closure hash comparison disabled", '_require(actual == digest,', '_require(True or actual == digest,'),
 ("K1", "package file-set check disabled", 'actual_package_files == expected_package_files', 'True'),
 ("K2", "package entry symlink check disabled", 'all(not path.is_symlink() for path in package_entries)', 'True'),
 ("K3", "closure member symlink check disabled", '_require(not path.is_symlink(), "closure member symlink', '_require(True, "closure member symlink'),
 ("K4", "verify_closure package-dir symlink check disabled", 'not package_directory.is_symlink()', 'True'),
 ("K5", "loader package-dir symlink check disabled", 'not supplied_directory.is_symlink()', 'True'),
 ("L1", "load: closure never verified", '    verify_closure(directory, closure)\n', '    pass\n'),
 ("L2", "load: sources never validated", 'sources = _validate_sources(source_manifest)', 'sources = source_manifest["sources"]'),
 ("L3", "load: fixture never validated", '    _validate_fixture(fixture)\n', '    pass\n'),
]

def repin(root):
    closure_path = root / PKG / "closure.json"
    closure = json.loads(closure_path.read_text())
    new = {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in closure}
    closure_path.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n")

def run(root, args):
    r = subprocess.run([sys.executable, "-B"] + args, cwd=root, capture_output=True, text=True, timeout=300)
    tail = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith(("Ran ", "OK", "FAILED")) or "survived" in l or "not unique" in l]
    return r.returncode, " ".join(tail[-3:])[:160]

orig = (SRC / PKG / "validator.py").read_text()
rows = []
for mid, desc, old, new in [("BASE", "unmodified", None, None)] + M:
    if ONLY and mid not in ONLY and mid != "BASE":
        continue
    with tempfile.TemporaryDirectory(prefix="r427-") as t:
        root = Path(t)
        shutil.copytree(SRC / PKG, root / PKG)
        (root / WF).parent.mkdir(parents=True)
        shutil.copy2(SRC / WF, root / WF)
        if old is not None:
            n = orig.count(old)
            if n != 1:
                rows.append((mid, desc, "BAD-PATTERN n=%d" % n, "", "")); print(rows[-1], flush=True); continue
            (root / PKG / "validator.py").write_text(orig.replace(old, new))
            repin(root)
        tn = run(root, ["-m", "unittest", "discover", "-s", PKG, "-p", "test_*.py"])
        to = run(root, ["-O", "-m", "unittest", "discover", "-s", PKG, "-p", "test_*.py"])
        hm = run(root, [PKG + "/mutation_harness.py"])
        tests = "KILLED" if tn[0] and to[0] else ("SURVIVED" if not tn[0] and not to[0] else "MIXED")
        harness = "harness-fail" if hm[0] else "harness-ok"
        rows.append((mid, desc, tests, harness, tn[1] + " | -O " + to[1] + " | H " + hm[1]))
        print(rows[-1], flush=True)
out = Path(__file__).resolve().parent / "results_c60.tsv"
out.write_text("".join("\t".join(r) + "\n" for r in rows))
