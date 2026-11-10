"""Exercise the prepared-artifact observer on originals and adverse copies."""
import json
import pathlib
import subprocess
import sys
import tempfile

root = pathlib.Path(__file__).resolve().parent
observer = root / "verify_prepared_artifact.py"
receipt = root / "prepared-receipt-original.json"
pages = root / "13728153243.zip"
metadata = root / "13728342519.zip"
results = []
with tempfile.TemporaryDirectory(prefix="cbz-pages-observer-adverse-") as tmp:
    tmp = pathlib.Path(tmp)
    bad_receipt = tmp / "bad-receipt.json"
    raw = receipt.read_bytes()
    old = b"429f8a20f456160225ec0a533e5334d6f7e9915ac41cf3d4884987e07442e24d"
    if raw.count(old) != 1:
        raise ValueError("receipt mutation site differs")
    bad_receipt.write_bytes(raw.replace(old, b"0" * 64))
    bad_pages = tmp / "bad-pages.zip"
    bad_pages.write_bytes(pages.read_bytes()[:1024])
    bad_metadata = tmp / "bad-metadata.zip"
    bad_metadata.write_bytes(metadata.read_bytes()[:1024])
    mutant = tmp / "mutant.py"
    source = observer.read_text()
    old_domain = "current-site-generated-browser-input-v1"
    if source.count(old_domain) != 1:
        raise ValueError("fingerprint mutation site differs")
    mutant.write_text(source.replace(old_domain, "current-site-generated-browser-input-v2"))
    for mode in ("normal", "optimized"):
        args = [sys.executable, "-B", "-S"]
        if mode == "optimized":
            args.insert(2, "-O")
        cases = [
            ("original", observer, receipt, pages, metadata, 0, ""),
            ("altered receipt", observer, bad_receipt, pages, metadata, 1,
             "prepared receipt Git blob differs"),
            ("truncated Pages ZIP", observer, receipt, bad_pages, metadata, 1,
             "original Pages ZIP differs"),
            ("truncated metadata ZIP", observer, receipt, pages, bad_metadata, 1,
             "original metadata ZIP differs"),
            ("mutated fingerprint domain", mutant, receipt, pages, metadata, 1,
             "prepared fingerprint differs"),
        ]
        for name, script, rec, page, meta, expected_exit, expected_error in cases:
            proc = subprocess.run(args + [str(script), str(rec), str(page), str(meta)],
                                  capture_output=True, text=True, timeout=120)
            if proc.returncode != expected_exit or (expected_error and
                    expected_error not in proc.stderr):
                raise ValueError(f"unexpected {mode} {name}: {proc.returncode}\n{proc.stderr}")
            if name == "original":
                record = json.loads(proc.stdout)
                if record["artifact"]["files"] != 3363:
                    raise ValueError("original file count differs")
            results.append({"mode": mode, "case": name, "exit": proc.returncode,
                            "stdout": proc.stdout, "stderr": proc.stderr})
(root / "prepared-verifier-adverse.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({"cases": len(results), "modes": ["normal", "optimized"],
                  "adverse_per_mode": 4, "result": "PASS"}))
