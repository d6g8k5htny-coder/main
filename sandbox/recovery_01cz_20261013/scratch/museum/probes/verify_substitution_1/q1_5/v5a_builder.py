# Verifier 2 / Q1-5 (builder half), independent of the author probe's assertions.
# Loads cd32303 tools/museum_data.py by path; network blocked; never writes into the cd32303 tree.
import copy, json, sys
sys.dont_write_bytecode = True
sys.path.insert(0, "<SCRATCH>/museum/probes/verify_substitution_1/q1_5")
from builder import museum, INDEX, STATUS, TREE

def run(label, mutate):
    specs = copy.deepcopy(museum.CLAIM_SPECS); mutate(specs)
    saved = museum.CLAIM_SPECS; museum.CLAIM_SPECS = specs
    try:
        claims = museum.project(INDEX, STATUS); out = "ACCEPTED"
    except Exception as e:
        claims = None; out = f"REFUSED ({type(e).__name__}: {e})"
    finally:
        museum.CLAIM_SPECS = saved
    print(f"{label}: {out}")
    return claims

base = run("0 baseline CLAIM_SPECS", lambda s: None)
# 1 the builder never compares a command with any source: an invented command for D2 is accepted
c = run("1 D2 command = 'python invented-solver.py' (in no source)", lambda s: s[0].__setitem__(4, "python invented-solver.py"))
print("   D2 replay:", c[0]["replay"]["command"], "| source", c[0]["replay"]["source"]["path"])
# 2 the D2 <-> D6 swap
def swap(s): s[0][4], s[10][4] = s[10][4], s[0][4]
c = run("2 swap D2 <-> D6 commands", swap)
for i in (0, 10): print(f"   {c[i]['id']:22} class={c[i]['class']:14} command={c[i]['replay']['command']!r}")
# 3 asymmetry baseline: the same builder DOES bind the hard-coded review pointer to the pinned bullet
run("3 D5-all-height review pointer -> pr16 REVIEW.md (shipped negative control)",
    lambda s: s[3].__setitem__(2, "reviews/pr16_fixed_transverse_nonauthor_20260925/REVIEW.md"))
# 4 refinement of the finding: the builder cannot give the P15 counterexample a command at all
run("4 P15 counterexample command = D6 full_price.py", lambda s: s[9].__setitem__(4, s[10][4]))
# 5 the only CI gate on museum.json is equality with the builder: a swapped CLAIM_SPECS + regenerated
#    museum.json differs from the shipped export only in the two replay commands
shipped = json.loads((TREE / "docs/site/museum.json").read_text())
specs = copy.deepcopy(museum.CLAIM_SPECS); swap(specs)
saved = museum.CLAIM_SPECS; museum.CLAIM_SPECS = specs
regen = museum.project(INDEX, STATUS); museum.CLAIM_SPECS = saved
diffs = [(a["id"], k) for a, b in zip(shipped["claims"], regen) for k in a if a[k] != b[k]]
inner = [(a["id"], k) for a, b in zip(shipped["claims"], regen) for k in a["replay"] if a["replay"][k] != b["replay"].get(k)]
print("5 regenerated claims vs shipped museum.json: differing top-level fields", diffs, "| differing replay fields", inner)
