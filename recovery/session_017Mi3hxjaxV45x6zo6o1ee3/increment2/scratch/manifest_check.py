# usage: manifest_check.py <wt> <record_dir> <workflow>: refresh the manifest file entries, then run the workflow block's
# manifest and pin verification (the part before the script replay).
import json, pathlib, subprocess, sys, hashlib, textwrap, re
wt, rd, wfp = sys.argv[1:4]
WT = pathlib.Path(wt); R = WT / rd
mp = R / "SOURCE_FILES.json"; raw = mp.read_text(); man = json.loads(raw)
fmt = None
for kw in (dict(indent=2, ensure_ascii=False), dict(indent=2, ensure_ascii=False, sort_keys=True), dict(indent=2), dict(indent=2, sort_keys=True)):
    if json.dumps(man, **kw) + "\n" == raw: fmt = kw
assert fmt, "manifest format"
order = [e["path"] for e in man["files"]]
present = sorted(p.name for p in R.iterdir() if p.name != "SOURCE_FILES.json")
order = [x for x in order if x in present] + sorted(set(present) - set(order))
files = []
for name in order:
    data = (R / name).read_bytes()
    blob = subprocess.check_output(["git", "hash-object", "--stdin"], input=data, cwd=WT).decode().strip()
    files.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "git_blob": blob})
man["files"] = files
mp.write_text(json.dumps(man, **fmt) + "\n")
yml = (WT / wfp).read_text()
block = textwrap.dedent(re.search(r"python -B -S - <<'PY'\n(.*?)\n\s*PY\n", yml, re.S).group(1))
block = block[:block.index("script = str(root")] + "print('manifest and pins verified')\n"
r = subprocess.run([sys.executable, "-B", "-S", "-"], input=block.encode(), cwd=WT, capture_output=True)
print(r.returncode, r.stdout.decode().strip(), r.stderr.decode().strip()[-800:])
sys.exit(r.returncode)
