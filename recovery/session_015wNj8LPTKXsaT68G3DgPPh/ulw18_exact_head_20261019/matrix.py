"""C8's malformed-input matrix for ULW#18 (standard library only).

Usage: python3 -I matrix.py <checkout> <base_verifier.py> <head_verifier.py> <probe_dir>
Writes each probe manifest to <probe_dir>, runs both verifiers offline (and
with --network where flagged) against <checkout>, and prints one row per run.
"""
import copy, json, os, subprocess, sys

checkout, base_v, head_v, pdir = sys.argv[1:5]
valid = json.load(open(os.path.join(checkout, "pins.json")))
sha0 = valid["repos"][0]["pinned_sha"]

def mut(f):
    def g():
        d = copy.deepcopy(valid); r = f(d); return json.dumps(d if r is None else r, indent=1)
    return g
def setk(k, v):
    def f(d): d["repos"][0][k] = v
    return f
def delk(k):
    def f(d): del d["repos"][0][k]
    return f
raw_trunc = lambda: '{"schema": "ulw.pins.v1", "repos": ['
raw_dupschema = lambda: json.dumps(valid, indent=1).replace('"schema": "ulw.pins.v1"', '"schema": "ulw.pins.v1", "schema": "ulw.pins.v1"', 1)
def raw_dupentrykey():
    t = json.dumps(valid, indent=1)
    p = '"path": %s' % json.dumps(valid["repos"][0]["path"])
    return t.replace(p, p + ", " + p, 1)

N = True  # also run --network
probes = [
 ("V0 valid (control)", mut(lambda d: None), N),
 ("R1 root array", lambda: "[]", False),
 ("R2 root null", lambda: "null", False),
 ("R3 root string", lambda: '"pins"', False),
 ("R4 root number", lambda: "7", False),
 ("R5 truncated JSON", raw_trunc, False),
 ("R6 duplicate root key", raw_dupschema, False),
 ("R7 wrong schema", mut(lambda d: d.__setitem__("schema", "ulw.pins.v0")), False),
 ("R8 repos missing", mut(lambda d: (d.pop("repos"), None)[1]), False),
 ("R9 repos object", mut(lambda d: d.__setitem__("repos", {})), False),
 ("R10 repos empty", mut(lambda d: d.__setitem__("repos", [])), False),
 ("R11 repos string", mut(lambda d: d.__setitem__("repos", "x")), False),
 ("E1 entry array", mut(lambda d: d["repos"].__setitem__(0, [])), False),
 ("E2 entry null", mut(lambda d: d["repos"].__setitem__(0, None)), False),
 ("E3 entry string", mut(lambda d: d["repos"].__setitem__(0, "x")), False),
 ("E4 entry missing url", mut(delk("url")), False),
 ("E5 path array", mut(setk("path", [])), False),
 ("E6 path object", mut(setk("path", {})), False),
 ("E7 path empty", mut(setk("path", "")), False),
 ("E8 path number", mut(setk("path", 5)), False),
 ("E9 pinned_sha number", mut(setk("pinned_sha", 5)), False),
 ("E10 pinned_sha short", mut(setk("pinned_sha", sha0[:12])), False),
 ("E11 pinned_sha uppercase", mut(setk("pinned_sha", sha0.upper())), False),
 ("E12 public string", mut(setk("public", "yes")), False),
 ("E13 opt_in null", mut(setk("opt_in", None)), False),
 ("E14 url number (N1)", mut(setk("url", 123)), N),
 ("E15 url list", mut(setk("url", ["x"])), N),
 ("E16 default_branch number", mut(setk("default_branch", 7)), N),
 ("E17 name number", mut(setk("name", 9)), N),
 ("E18 note list", mut(setk("note", [])), N),
 ("E19 duplicate path", mut(lambda d: d["repos"].append(copy.deepcopy(d["repos"][0]))), False),
 ("E20 duplicate key in entry", raw_dupentrykey, False),
 ("C1 pin mismatch (valid form)", mut(setk("pinned_sha", "0" * 40)), N),
 ("C2 url mismatch (string)", mut(setk("url", "https://github.com/d6g8k5htny-coder/no-such-repo-c8")), N),
 ("C3 gitlink not listed", mut(lambda d: (d["repos"].pop(0), None)[1]), False),
 ("C4 extra entry not in tree", mut(lambda d: d["repos"].append(dict(copy.deepcopy(d["repos"][0]), path="repos/no-such-c8", name="x"))), False),
 ("C5 opt_in flipped", mut(lambda d: d["repos"][0].__setitem__("opt_in", not d["repos"][0]["opt_in"])), False),
]

def run(ver, pins, net):
    cmd = [sys.executable, "-B", ver, "--repo", checkout, "--pins", pins] + (["--network"] if net else [])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", False, "")
    tb = "Traceback (most recent call last)" in p.stderr
    last = (p.stdout.strip().splitlines() or [""])[-1]
    exc = ""
    if tb:
        exc = [l for l in p.stderr.strip().splitlines() if l.strip()][-1]
    return (p.returncode, tb, last if not tb else exc)

os.makedirs(pdir, exist_ok=True)
print("probe | mode | base rc / traceback | head rc / traceback | head last line")
for i, (name, gen, net) in enumerate(probes):
    path = os.path.join(pdir, "p%02d.json" % i)
    with open(path, "w") as fh: fh.write(gen())
    for mode in (["offline", "network"] if net else ["offline"]):
        b = run(base_v, path, mode == "network"); h = run(head_v, path, mode == "network")
        print("%s | %s | %s / %s | %s / %s | %s" % (name, mode, b[0], "TB" if b[1] else "-", h[0], "TB" if h[1] else "-", h[2][:150]))
