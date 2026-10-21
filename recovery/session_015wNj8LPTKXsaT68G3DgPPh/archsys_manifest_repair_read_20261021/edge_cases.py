"""Extra adverse cases for the manifest --check contract, outside the delivered harness.
Usage: python3 -I -B -S edge_cases.py ORIGINAL CANDIDATE  (synthetic files only)"""
import json, subprocess, sys, tempfile, copy
from pathlib import Path
def run(src, mutate, keep_manifest=True):
    with tempfile.TemporaryDirectory(prefix="edge_") as t:
        root = Path(t); s = root / "make_manifest.py"; s.write_bytes(Path(src).read_bytes())
        (root / "payload.txt").write_bytes(b"data\n")
        subprocess.run([sys.executable, "-I", "-B", "-S", str(s)], capture_output=True, check=True, timeout=10)
        m = root / "MANIFEST.json"; data = json.loads(m.read_text())
        res = mutate(data)
        if res == "DELETE": m.unlink()
        else: m.write_text(json.dumps(data, indent=1) + "\n")
        r = subprocess.run([sys.executable, "-I", "-B", "-S", str(s), "--check"], capture_output=True, timeout=10)
        last = r.stderr.decode().strip().splitlines()[-1] if r.stderr else ""
        return r.returncode, r.stdout.decode().strip().replace("\n", " | "), last
cases = {
 "E1 row missing 'path' key": lambda d: d["committed_files"][1].pop("path"),
 "E2 committed_files is a dict": lambda d: d.__setitem__("committed_files", {"x": d["committed_files"][0]}),
 "E3 MANIFEST.json absent": lambda d: "DELETE",
 "E4 total = 2**70": lambda d: d.__setitem__("committed_total_bytes", 2**70),
 "E5 duplicate of the tool's own row": lambda d: d["committed_files"].append(copy.deepcopy(d["committed_files"][0])),
 "E6 row bytes field tampered, hash intact": lambda d: d["committed_files"][1].__setitem__("bytes", 6),
 "E7 './payload.txt' alias row added": lambda d: d["committed_files"].append(dict(d["committed_files"][1], path="./payload.txt")),
 "E8 total null": lambda d: d.__setitem__("committed_total_bytes", None),
}
for name, fn in cases.items():
    for label, src in (("orig", sys.argv[1]), ("cand", sys.argv[2])):
        rc, out, err = run(src, fn)
        print(f"{name:42s} {label}: exit {rc} | {out[:110]}" + (f" | stderr: {err[:80]}" if err else ""))
