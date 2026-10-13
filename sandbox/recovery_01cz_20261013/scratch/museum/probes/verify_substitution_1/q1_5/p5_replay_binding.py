# Q1/P5 (builder half): replay commands are hard-coded in CLAIM_SPECS and never compared with any
# source by the builder. For D2/D4/D6 the "source" is the shared Math- README, so nothing ties a
# command to its claim: a swap, or another package's README command, is accepted.
import copy, sys
sys.path.insert(0, "<SCRATCH>/museum/probes/verify_substitution_1/q1_5")
from builder import museum, INDEX, STATUS
specs = copy.deepcopy(museum.CLAIM_SPECS)
specs[0][4], specs[10][4] = specs[10][4], specs[0][4]                     # swap D2 <-> D6
specs[2][4] = "python -B -S frontiers/downstream_gate_20260925/hard_gate.py"  # D4 gets the D0 hard-gate script
saved = museum.CLAIM_SPECS; museum.CLAIM_SPECS = specs
claims = museum.project(INDEX, STATUS); museum.CLAIM_SPECS = saved
for i in (0, 2, 10):
    r = claims[i]["replay"]
    print(f"ACCEPTED {claims[i]['id']:24} command={r['command']!r} source={r['source']['path']}@{r['source']['commit'][:8]}")
