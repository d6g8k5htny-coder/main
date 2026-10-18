"""One-quantum narrowing mutations at every rational leaf; list which the checker accepts."""
import sys, json, copy
from fractions import Fraction as Q
sys.path.insert(0, sys.argv[1])
from moment_kernel import certify
from verify_certificate import verify
def qs(q): return f"{q.numerator}/{q.denominator}"
def leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items(): yield from leaves(v, path+(k,))
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from leaves(v, path+(i,))
    elif isinstance(o, str) and "/" in o:
        yield path
def get(o, p):
    for k in p: o = o[k]
    return o
def setp(o, p, v):
    for k in p[:-1]: o = o[k]
    o[p[-1]] = v
reqs = [dict(L_lo="1/1", L_hi="1/1", cutoff=2, input_role="TEST_ONLY", width_goal="1/1000000000000"),
        dict(L_lo="1/1", L_hi="2/1", cutoff=2, input_role="TEST_ONLY", width_goal="1/1000000000000"),
        dict(L_lo="10/1", L_hi="12/1", cutoff=6, bits=24, pi_terms=3, exp_pairs=2, input_role="TEST_ONLY")]
for req in reqs:
    base = json.loads(json.dumps(certify(req)))
    verify(copy.deepcopy(base))
    bits = base["request"]["bits"]; qd = Q(1, 2**bits)
    acc = []; tot = 0
    for p in leaves(base):
        if p[0] in ("request",) or p[0] == "error_accounting": continue
        if p[-1] in ("x", "y"): continue
        v = Q(get(base, p))
        if isinstance(p[-1], int) and len(get(base, p[:-1])) == 2:
            nv = v + qd if p[-1] == 0 else v - qd
        elif p[0] == "normalizer_input_spread_lower":
            nv = v + qd
        else:
            nv = v - qd
        if nv < 0: continue
        c = copy.deepcopy(base); setp(c, p, qs(nv))
        # keep 'result' consistent with last level when mutating levels
        if "levels" in p:
            w = get(c, p[:p.index("levels")])
            w["result"] = copy.deepcopy(w["levels"][-1])
        if p[0] == "finite":
            c["error_accounting"]["finite_widths"][p[1]] = qs(Q(c["finite"][p[1]][1]) - Q(c["finite"][p[1]][0]))
        tot += 1
        try:
            verify(c); acc.append((p, get(base, p), qs(nv)))
        except ValueError as e:
            pass
    print(req["L_lo"], req["L_hi"], "mutations", tot, "accepted", len(acc))
    for a in acc: print("   ACCEPTED", a)
