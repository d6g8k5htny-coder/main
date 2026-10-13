# usage: finish.py <wt> <record_dir> <checker> <workflow>
import json, pathlib, subprocess, sys, hashlib, textwrap, re, shutil
wt, rd, checker, wfp = sys.argv[1:5]
WT = pathlib.Path(wt); R = WT / rd
for p in WT.rglob("__pycache__"): shutil.rmtree(p, ignore_errors=True)
def run(flags, extra=()):
    return subprocess.run([sys.executable, *flags, str(R / checker), *extra], cwd=WT, capture_output=True, timeout=600)
a = run(["-B", "-S"]); b = run(["-B", "-O", "-S"])
print("baseline rc", a.returncode, b.returncode, "identical" if a.stdout == b.stdout else "DIFFER")
if a.returncode or b.returncode or a.stdout != b.stdout:
    print(a.stderr.decode()[-3000:]); print(json.loads(a.stdout or b"{}").get("checks")); sys.exit(2)
(R / "RESULTS.json").write_bytes(a.stdout)
print(json.dumps(json.loads(a.stdout)["checks"], sort_keys=True))
src = (R / checker).read_text()
muts = re.search(r"MUTANTS = \((.*?)\)\n", src, re.S).group(1)
muts = re.findall(r'"([a-z0-9-]+)"', muts)
bad = [m for m in muts for fl in (["-B","-S"],["-B","-O","-S"]) if run(fl, ["--mutant", m]).returncode != 1]
print("mutants", len(muts), "all rejected" if not bad else ("NOT REJECTED " + str(bad)))
# manifest
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
# workflow replay
yml = (WT / wfp).read_text()
block = re.search(r"python -B -S - <<'PY'\n(.*?)\n\s*PY\n", yml, re.S).group(1)
block = textwrap.dedent(block)
for p in WT.rglob("__pycache__"): shutil.rmtree(p, ignore_errors=True)
r = subprocess.run([sys.executable, "-B", "-S", "-"], input=block.encode(), cwd=WT, capture_output=True, timeout=900)
print("replay rc", r.returncode, r.stdout.decode().strip()[-200:], r.stderr.decode().strip()[-1500:])
for p in WT.rglob("__pycache__"): shutil.rmtree(p, ignore_errors=True)
sys.exit(0 if (r.returncode == 0 and not bad) else 3)
