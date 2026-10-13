# Verifier probe for Q1-9 (LANDING_CLAIMS.json pinned, verified, never consumed; conflicting axis dropped).
# Read-only against cd32303 bytes. No network: urlopen is replaced by a raiser. Math- sources come from the
# byte-verified fixture built by probes/substitution/setup.sh; main sources via `git show 71400b94:...`
# (read-only). README.md@d6628da is not available offline, so README's descriptor is swapped for the
# local README@10e1f191 identity IN ALL RUNS (it only feeds replay-source descriptors).
import ast, copy, hashlib, importlib.util, json, pathlib, shutil, sys, tempfile
sys.dont_write_bytecode = True
M = pathlib.Path("<SCRATCH>/museum")
TREE = M / "cd32303"; SUB = M / "probes/substitution"; MAIN = pathlib.Path("/home/user/main")
spec = importlib.util.spec_from_file_location("md_cd32303", TREE / "tools/museum_data.py")
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
def blocked(*a, **k): raise RuntimeError("NETWORK BLOCKED")
md.urlopen = blocked
def ident(raw, path, repo="d6g8k5htny-coder/Math-", commit=md.MATH_COMMIT):
    return {"repository": repo, "commit": commit, "path": path, "bytes": len(raw),
            "blob": hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest(), "sha256": hashlib.sha256(raw).hexdigest()}
LC = "claims/LANDING_CLAIMS.json"

# 1. Static: where does the module mention LANDING_CLAIMS, and does any code (not the SOURCES literal) name it?
src = (TREE / "tools/museum_data.py").read_text()
lines = [n for n, l in enumerate(src.splitlines(), 1) if "LANDING_CLAIMS" in l]
tree = ast.parse(src)
sources_assign = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SOURCES")
code_refs = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and "LANDING" in n.value and not (sources_assign.lineno <= n.lineno <= sources_assign.end_lineno)]
print(f"[static] lines mentioning LANDING_CLAIMS: {lines}; SOURCES literal spans {sources_assign.lineno}-{sources_assign.end_lineno}; string refs outside it: {code_refs}")

# 2. Build fixtures.
work = pathlib.Path(tempfile.mkdtemp(prefix="q1_9_", dir=M / "probes/verify_substitution_0/q1_9"))
base_math = work / "math_base"; shutil.copytree(SUB / "fixture/Math-", base_math)
readme = (SUB / "readme_10e1f191.md").read_bytes(); (base_math / "README.md").write_bytes(readme)
md.SOURCES["README.md"] = ident(readme, "README.md")
ORIG_SOURCES = copy.deepcopy(md.SOURCES)
def run(math_root, sources):
    md.SOURCES.clear(); md.SOURCES.update(sources)
    try: return md.dump(md.build(MAIN, math_root))
    finally: md.SOURCES.clear(); md.SOURCES.update(ORIG_SOURCES)
baseline = run(base_math, ORIG_SOURCES)

# 2a. Baseline reproduces the shipped museum.json except README-descriptor fields.
shipped = json.loads((TREE / "docs/site/museum.json").read_text()); built = json.loads(baseline)
def flat(o, p=""):
    if isinstance(o, dict): return {k2: v2 for k, v in o.items() for k2, v2 in flat(v, f"{p}/{k}").items()}
    if isinstance(o, list): return {k2: v2 for i, v in enumerate(o) for k2, v2 in flat(v, f"{p}/{i}").items()}
    return {p: o}
fs, fb = flat(shipped), flat(built)
diff = sorted(k for k in fs.keys() | fb.keys() if fs.get(k) != fb.get(k))
def at(o, k):
    for part in k.strip("/").split("/")[:-1]: o = o[int(part)] if isinstance(o, list) else o[part]
    return o
readme_only = all(at(built, k).get("path") == "README.md" and at(shipped, k).get("path") == "README.md" for k in diff)
print(f"[baseline] build() vs shipped museum.json: {len(diff)} differing leaves; all README-descriptor fields: {readme_only}; {diff}")

# 2b. Mutate LANDING_CLAIMS: every disposition -> 'MUTATED_<x>', meaning -> 'theorem acceptance'; re-pin identity.
mut_math = work / "math_mut"; shutil.copytree(base_math, mut_math)
lc = json.loads((mut_math / LC).read_bytes())
for c in lc["claims"]: c["disposition"] = "MUTATED_" + c["disposition"]; c["disposition_reason"] = "mutated"
lc["meaning"] = "theorem acceptance (mutated)"
mraw = json.dumps(lc).encode(); (mut_math / LC).write_bytes(mraw)
mut_sources = copy.deepcopy(ORIG_SOURCES); mut_sources[LC] = ident(mraw, LC)
mutated = run(mut_math, mut_sources)
print(f"[mutate] LANDING_CLAIMS dispositions all rewritten (sha256 {mut_sources[LC]['sha256'][:12]}): museum.json identical to baseline: {mutated == baseline}")

# 2c. Remove LANDING_CLAIMS from SOURCES and from disk.
rm_math = work / "math_rm"; shutil.copytree(base_math, rm_math); (rm_math / LC).unlink()
rm_sources = copy.deepcopy(ORIG_SOURCES); del rm_sources[LC]
removed = run(rm_math, rm_sources)
print(f"[remove] LANDING_CLAIMS dropped from SOURCES and fixture: build succeeds, museum.json identical to baseline: {removed == baseline}")

# 2d. Fail-closed cost: file pinned but absent -> build() refuses (so the unused pin is a live failure surface).
try: run(rm_math, ORIG_SOURCES); print("[absent-but-pinned] build succeeded")
except Exception as e: print(f"[absent-but-pinned] build refuses: {type(e).__name__}: {e}")

# 3. Same proof blobs: museum label vs LANDING_CLAIMS disposition.
orig_lc = json.loads((base_math / LC).read_bytes()); by_blob = {c["source"]["blob"]: c for c in orig_lc["claims"]}
for c in built["claims"]:
    l = by_blob.get(c["proof"]["blob"])
    if l: print(f"[axis] {c['id']:30} label={c['source_label']:20} status_quote={'yes' if c['status_quote'] else 'no ':3} | LANDING {l['claim_id']}: {l['disposition']}")
print(f"[axis] LANDING_CLAIMS meaning: {orig_lc['meaning']!r}; audited_default_commit {orig_lc['audited_default_commit'][:8]} (pinned at {md.MATH_COMMIT[:8]})")
print(f"[axis] PROOF_INDEX D6 bullet says: {'leaves the landing disposition untouched' in built['claims'][10]['scope_quote']!r} ('leaves the landing disposition untouched')")
shutil.rmtree(work)
