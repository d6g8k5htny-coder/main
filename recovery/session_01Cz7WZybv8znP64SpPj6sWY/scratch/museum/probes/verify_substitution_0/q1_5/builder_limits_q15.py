# Verifier probe Q1-5: limits of the builder path (cd32303 tools/museum_data.py loaded by path, no network).
import copy, sys
sys.path.insert(0, "<SCRATCH>/museum/probes/verify_substitution_0/q1_5")
from builder import museum, INDEX, STATUS, HERE
def run(label, edit):
    specs = copy.deepcopy(museum.CLAIM_SPECS); edit(specs)
    saved = museum.CLAIM_SPECS; museum.CLAIM_SPECS = specs
    try: claims = museum.project(INDEX, STATUS); print(f"{label}: builder ACCEPTED"); return claims
    except Exception as e: print(f"{label}: builder REFUSED {type(e).__name__}: {e}")
    finally: museum.CLAIM_SPECS = saved
run("P15 counterexample given the D6 command via CLAIM_SPECS",
    lambda s: s.__setitem__(9, s[9][:4] + ["python -B -S frontiers/full_price_20260924/full_price.py"]))
# D5 claims use their own review file as replay source; is a cross-claim swap caught by the frontend line test?
fx = HERE / "fixture/Math-"
specs = museum.CLAIM_SPECS
for i, j in ((3, 4), (4, 3), (3, 7), (7, 4)):
    src_text = (fx / specs[i][2]).read_text()
    cmd = specs[j][4]
    print(f"D5 replay source for {specs[i][0]} ({specs[i][2]}) contains {specs[j][0]}'s command: {any(l.strip()==cmd for l in src_text.splitlines())}")
for i in (3, 4, 7):
    src_text = (fx / specs[i][2]).read_text()
    print(f"control: {specs[i][0]}'s own command is a line of its own review: {any(l.strip()==specs[i][4] for l in src_text.splitlines())}")
