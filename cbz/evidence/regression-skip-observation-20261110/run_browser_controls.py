"""Run two previously skipped Atlas browser controls against recovered Git blobs."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

root = pathlib.Path(__file__).resolve().parent
source = root / "source"
manifest = json.loads((root / "source-manifest.json").read_text())
evidence = root / "evidence"
evidence.mkdir(exist_ok=False)


def verify_source():
    for row in manifest["files"]:
        path = source / row["path"]
        raw = path.read_bytes()
        git = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if len(raw) != row["size"] or git != row["sha"]:
            raise ValueError("source drift: " + row["path"])


results = []
for mode in ("normal", "optimized"):
    verify_source()
    dest = evidence / mode
    dest.mkdir()
    env = os.environ.copy()
    env["ATLAS_BROWSER_EXECUTABLE"] = "/usr/bin/chromium"
    env["ATLAS_FOCUS_EVIDENCE"] = str(dest / "focus")
    env["ATLAS_READER_EVIDENCE"] = str(dest / "reader")
    cmd = [sys.executable, "-B"]
    if mode == "optimized":
        cmd.append("-O")
    cmd += ["-m", "unittest", "discover", "-s", "tests", "-p", "test_atlas_*.py", "-v"]
    start = time.time()
    try:
        p = subprocess.run(cmd, cwd=source, env=env, capture_output=True, timeout=120)
        code, out, err, timed_out = p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as exc:
        code, out, err, timed_out = None, exc.stdout or b"", exc.stderr or b"", True
    (dest / "stdout.bin").write_bytes(out)
    (dest / "stderr.bin").write_bytes(err)
    verify_source()
    text = err.decode("utf-8", errors="replace")
    success = code == 0 and "Ran 2 tests" in text and "\nOK\n" in text and " ... skipped " not in text
    results.append({"mode": mode, "command": cmd, "exit": code, "timed_out": timed_out,
                    "duration_seconds": time.time() - start, "success": success,
                    "stdout_sha256": hashlib.sha256(out).hexdigest(),
                    "stderr_sha256": hashlib.sha256(err).hexdigest(),
                    "stderr_tail": text.splitlines()[-25:]})
(evidence / "observation.json").write_text(json.dumps({
    "source_commit": manifest["commit"], "python": sys.version,
    "source_blobs": len(manifest["files"]), "source_before_after": "matched",
    "scope": "two actual browser controls only; no full G5 or deployed-site claim",
    "results": results}, indent=2) + "\n")
print(json.dumps({"results": results}, indent=2), flush=True)
if not all(x["success"] for x in results):
    raise SystemExit(1)
