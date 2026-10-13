import copy, sys
sys.path.insert(0, 'reviews/register_execution_readiness_20260930')
import test_witness_states as T
W = T.witness
inst = T.graph_fixture("installed")
nid = "math.c6-component.palm-proof"
for val in (1, "true", False):
    g = copy.deepcopy(inst)
    g["nodes"][nid]["review_basis"][0]["source_exposed"] = val
    print("review_basis[0].source_exposed =", repr(val), "->", T.state(g))
g = copy.deepcopy(inst); g["nodes"][nid]["controlling"] = 0
print("top-level controlling = 0 ->", T.state(g))
