import copy, json, sys
from pathlib import Path
sys.path.insert(0, 'reviews/register_execution_readiness_20260930')
import test_witness_states as T
W = T.witness
P = T.PROPOSAL
# 1. field value types in proposed nodes / edges
types = {}
for n in P["proposed_graph_nodes"]:
    for k, v in n.items():
        types.setdefault(k, set()).add(type(v).__name__)
print("node field types:", {k: sorted(v) for k, v in types.items()})
print("edge keys:", sorted({k for e in P["proposed_graph_edges"] for k in e}))
def st(g): return T.state(g)
base = T.graph_fixture("baseline"); inst = T.graph_fixture("installed"); rev = T.graph_fixture("reviewed")
print("controls:", st(base), st(inst), st(rev))
wit = [e for e in P["proposed_graph_edges"] if e["from"] == W.WITNESS]
def probe(label, g, want="partial"):
    s = st(g); print(("OK  " if s == want else "MISS"), label, "->", s)
# finding 3 original and variants
for stage, g0 in (("installed", inst), ("reviewed", rev)):
    for e in wit:
        g = copy.deepcopy(g0)
        if stage == "installed":
            g["edges"].append(dict(T.edge_fields(e), required=not e["required"]))
            probe(f"{stage}: add witness edge with flipped required -> {e['to'][-30:]}", g)
            g = copy.deepcopy(g0); g["edges"].append(dict(T.edge_fields(e), required=int(e["required"])))
            probe(f"{stage}: add witness edge with int required -> {e['to'][-30:]}", g)
        else:
            for x in g["edges"]:
                if T.edge_fields(x) == T.edge_fields(e): x["required"] = int(e["required"])
            probe(f"{stage}: reviewed edge required as int -> {e['to'][-30:]}", g)
# duplicate exact edge plus a wrong-relation twin on a proposed non-witness pair
other = [e for e in P["proposed_graph_edges"] if e["from"] != W.WITNESS]
if other:
    g = copy.deepcopy(inst); g["edges"].append(dict(T.edge_fields(other[0]), relation="supports"))
    probe("installed: wrong-relation twin on proposed non-witness pair", g)
    g = copy.deepcopy(inst); g["edges"].append(T.edge_fields(other[0]))
    probe("installed: exact duplicate of proposed non-witness edge (hard gate's job)", g, want=st(g))
# finding 5 variants on node fields
for nid in [n["id"] for n in P["proposed_graph_nodes"]]:
    g = copy.deepcopy(inst); g["nodes"][nid]["controlling"] = 0
    probe(f"installed: node controlling=0 {nid[-34:]}", g)
g = copy.deepcopy(inst); g["nodes"][W.WITNESS]["controlling"] = 0
probe("installed: witness controlling=0", g)
g = copy.deepcopy(base); g["nodes"][W.WITNESS]["controlling"] = True
probe("baseline: witness controlling=True", g)
# extra non-proposed witness edge to an unrelated target (contract silent; report only)
g = copy.deepcopy(inst)
tgt = next(k for k in g["nodes"] if k not in {e["to"] for e in wit} and k != W.WITNESS)
g["edges"].append({"from": W.WITNESS, "to": tgt, "required": True, "relation": "requires_evidence"})
print("INFO installed + extra non-proposed witness edge ->", st(g))
