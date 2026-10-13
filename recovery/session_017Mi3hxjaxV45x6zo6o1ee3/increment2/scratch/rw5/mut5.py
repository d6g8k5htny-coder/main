import subprocess, sys, json, re, concurrent.futures, pathlib
P = pathlib.Path("<SCRATCH>/wtw5/frontiers/c8_window_coefficient_d5_20261001/window_coefficient_d5.py")
src = P.read_text()
muts = re.findall(r'"([a-z0-9-]+)"', re.search(r"MUTANTS = \((.*?)\)\n", src, re.S).group(1))
jobs = [(None, fl) for fl in (("-B", "-S"), ("-B", "-O", "-S"))] + [(m, fl) for m in muts for fl in (("-B", "-S"), ("-B", "-O", "-S"))]
def one(j):
    m, fl = j
    r = subprocess.run([sys.executable, *fl, str(P)] + (["--mutant", m] if m else []), capture_output=True, timeout=3600)
    fails = None
    try:
        fails = [k for k, v in json.loads(r.stdout)["checks"].items() if not v]
    except Exception:
        fails = "crash: " + r.stderr.decode()[-300:]
    return m, fl, r.returncode, fails, r.stdout
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    res = list(pool.map(one, jobs))
outs = {}
for m, fl, rc, fails, so in res:
    print(m, " ".join(fl), rc, fails, flush=True)
    if m is None: outs[fl] = so
print("baseline identical", len(set(outs.values())) == 1)
