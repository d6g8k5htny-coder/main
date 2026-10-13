import json, subprocess, sys, re, concurrent.futures, pathlib
script = pathlib.Path("<SCRATCH>/wtw7/frontiers/c8_window_coefficient_d7_20261001/window_coefficient_d7.py")
src = script.read_text()
muts = re.findall(r'"([a-z0-9-]+)"', re.search(r"MUTANTS = \((.*?)\)\n", src, re.S).group(1))
def one(m):
    r = subprocess.run([sys.executable, "-B", "-S", str(script), "--mutant", m], capture_output=True, timeout=1800)
    if r.stdout.strip():
        d = json.loads(r.stdout)
        return m, r.returncode, sorted(k for k, v in d["checks"].items() if not v)
    return m, r.returncode, "EXC: " + r.stderr.decode().strip().splitlines()[-1]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    res = list(pool.map(one, muts))
out = {m: {"rc": rc, "failed": f} for m, rc, f in res}
pathlib.Path("<SCRATCH>/rw7/mutrules.json").write_text(json.dumps(out, indent=1))
for m, rc, f in res:
    print(m, rc, f)
