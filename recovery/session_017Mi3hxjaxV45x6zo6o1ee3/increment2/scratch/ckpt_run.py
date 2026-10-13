# checkpointed finishing driver: baseline (both modes), mutants (4 workers, per-mutant checkpoint files), manifest, workflow replay
import json, pathlib, subprocess, sys, hashlib, textwrap, re, shutil, concurrent.futures, time
wt, rd, checker, wfp, ck = sys.argv[1:6]
WT = pathlib.Path(wt); R = WT / rd; CK = pathlib.Path(ck); CK.mkdir(parents=True, exist_ok=True)
for p in WT.rglob("__pycache__"): shutil.rmtree(p, ignore_errors=True)
def run(flags, extra=()):
    return subprocess.run([sys.executable, *flags, str(R / checker), *extra], cwd=WT, capture_output=True, timeout=3600)
t0 = time.time()
base = CK / "baseline.json"
if not base.exists():
    a = run(["-B", "-S"]); b = run(["-B", "-O", "-S"])
    print("baseline rc", a.returncode, b.returncode, "identical" if a.stdout == b.stdout else "DIFFER", round(time.time() - t0), "s", flush=True)
    if a.returncode or b.returncode or a.stdout != b.stdout:
        print(a.stderr.decode()[-3000:]); sys.exit(2)
    base.write_bytes(a.stdout)
(R / "RESULTS.json").write_bytes(base.read_bytes())
print(json.dumps(json.loads(base.read_bytes())["checks"], sort_keys=True), flush=True)
src = (R / checker).read_text()
muts = re.findall(r'"([a-z0-9-]+)"', re.search(r"MUTANTS = \((.*?)\)\n", src, re.S).group(1))
jobs = [(m, fl) for m in muts for fl in (["-B", "-S"], ["-B", "-O", "-S"])]
def one(j):
    m, fl = j
    f = CK / ("%s_%s.rc" % (m, "O" if "-O" in fl else "S"))
    if f.exists():
        return m, fl, int(f.read_text())
    rc = run(fl, ["--mutant", m]).returncode
    f.write_text(str(rc))
    print("  mutant", m, fl, "rc", rc, round(time.time() - t0), "s", flush=True)
    return m, fl, rc
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    res = list(pool.map(one, jobs))
bad = [(m, fl) for m, fl, rc in res if rc != 1]
print("mutants", len(muts), "all rejected" if not bad else ("NOT REJECTED " + str(bad)), flush=True)
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
print("manifest written", flush=True)
if bad:
    sys.exit(3)
if (CK / "replay.ok").exists():
    print("replay already ok"); sys.exit(0)
yml = (WT / wfp).read_text()
block = textwrap.dedent(re.search(r"python -B -S - <<'PY'\n(.*?)\n\s*PY\n", yml, re.S).group(1))
for p in WT.rglob("__pycache__"): shutil.rmtree(p, ignore_errors=True)
r = subprocess.run([sys.executable, "-B", "-S", "-"], input=block.encode(), cwd=WT, capture_output=True, timeout=7200)
print("replay rc", r.returncode, r.stdout.decode().strip()[-200:], r.stderr.decode().strip()[-1500:], round(time.time() - t0), "s", flush=True)
for p in WT.rglob("__pycache__"): shutil.rmtree(p, ignore_errors=True)
if r.returncode == 0:
    (CK / "replay.ok").write_text("ok")
sys.exit(0 if r.returncode == 0 else 3)
