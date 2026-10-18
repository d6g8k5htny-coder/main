"""Build, from scratch (no producer import), the TIGHTEST certificate the checker's
containment rules allow for a given request skeleton: every recorded interval is the
exact forward image demanded by the checker, rounded outward only to its 2^-bits grid.
If any checker inequality had a wrong direction or missing term, such a boundary
certificate would be accepted yet fail mpmath containment.  (python3 -I)
argv: workdir base_cert.json out.json [bits] [pi_terms] [exp_pairs]"""
import sys, json, copy
from fractions import Fraction as F
work = sys.argv[1]
sys.path.insert(0, work)
import verify_certificate as vc
base = json.load(open(sys.argv[2]))
c = copy.deepcopy(base)
r = c["request"]
if len(sys.argv) > 4: r["bits"] = int(sys.argv[4])
if len(sys.argv) > 5: r["pi_terms"] = int(sys.argv[5]); c["pi"]["terms"] = r["pi_terms"]
if len(sys.argv) > 6: r["exp_pairs"] = int(sys.argv[6])
bits = r["bits"]; D = 1 << bits; p = r["exp_pairs"]; n = r["pi_terms"]; M = r["cutoff"]
lo, hi = F(r["L_lo"]), F(r["L_hi"])
def dn(q): return F((q.numerator * D) // q.denominator, D)
def up(q): return F(-((-q.numerator * D) // q.denominator), D)
def t(q): return "%d/%d" % (q.numerator, q.denominator)
def iv(a, b): return [t(dn(a)), t(up(b))]
def atan(x, n):
    s = F(0)
    for j in range(n): s += (-1) ** j * x ** (2 * j + 1) / (2 * j + 1)
    u = s + (-1) ** n * x ** (2 * n + 1) / (2 * n + 1)
    return min(s, u), max(s, u)
a5, a239 = atan(F(1, 5), n), atan(F(1, 239), n)
pi = (dn(16 * a5[0] - 4 * a239[1]), up(16 * a5[1] - 4 * a239[0]))
c["pi"]["interval"] = [t(pi[0]), t(pi[1])]
h = (dn(2 * pi[0] / hi), up(2 * pi[1] / lo)); c["h"] = [t(h[0]), t(h[1])]
al = (dn(h[0] ** 2 / 2), up(h[1] ** 2 / 2)); c["alpha"] = [t(al[0]), t(al[1])]
assert al[0] * M * M >= 3
def expw(x):
    s = 0; y = x
    while y > 1: y /= 2; s += 1
    term = F(1); tot = F(1); even = None
    for j in range(1, 2 * p + 2):
        term = term * (-y) / j; tot += term
        if j == 2 * p: even = tot
    a, b = dn(max(F(0), tot)), up(min(F(1), even))
    lev = [[t(a), t(b)]]
    for _ in range(s):
        a, b = dn(a * a), up(b * b); lev.append([t(a), t(b)])
    return {"x": t(x), "y": t(y), "squarings": s, "pairs": p, "levels": lev, "result": copy.deepcopy(lev[-1])}, (a, b)
sums = {k: [F(1), F(1)] if k == "0" else [F(0), F(0)] for k in ("0", "2", "4", "6")}
modes = []
for j in range(1, M + 1):
    wl, el = expw(al[1] * j * j); wu, eu = expw(al[0] * j * j)
    w = (el[0], eu[1])
    contrib = {}
    for k in ("0", "2", "4", "6"):
        d = int(k)
        a, b = dn(2 * (h[0] * j) ** d * w[0]), up(2 * (h[1] * j) ** d * w[1])
        contrib[k] = [t(a), t(b)]; sums[k][0] += a; sums[k][1] += b
    modes.append({"m": j, "exp_at_hi": wl, "exp_at_lo": wu, "weight": [t(w[0]), t(w[1])], "contribution": contrib})
c["modes"] = modes
c["finite"] = {k: [t(v[0]), t(v[1])] for k, v in sums.items()}
tw, te = expw(al[0] * M * M); c["tail_exp"] = tw; e = te[1]
I = [up(e / (2 * al[0] * M))]
for k in range(1, 4): I.append(up((M ** (2 * k - 1) * e + (2 * k - 1) * I[-1]) / (2 * al[0])))
c["tail_integral_upper"] = [t(v) for v in I]
tails = {kk: up(2 * h[1] ** (2 * k) * I[k]) for k, kk in enumerate(("0", "2", "4", "6"))}
c["tail_upper"] = {k: t(v) for k, v in tails.items()}
raw = {k: (sums[k][0], sums[k][1] + tails[k]) for k in sums}
c["raw"] = {k: [t(a), t(b)] for k, (a, b) in raw.items()}
z = raw["0"]
c["bounds"] = {"z": [t(z[0]), t(z[1])]}
for k in ("2", "4", "6"):
    c["bounds"]["mu" + k] = iv(raw[k][0] / z[1], raw[k][1] / z[0])
if lo != hi:
    alo = (dn(2 * pi[0] ** 2 / lo ** 2), up(2 * pi[1] ** 2 / lo ** 2))
    ahi = (dn(2 * pi[0] ** 2 / hi ** 2), up(2 * pi[1] ** 2 / hi ** 2))
    w1, e1 = expw(ahi[1]); w2, e2 = expw(alo[0])
    c["endpoint_spread"] = {"alpha_Llo": [t(alo[0]), t(alo[1])], "alpha_Lhi": [t(ahi[0]), t(ahi[1])],
                            "exp_Lhi_at_alpha_hi": w1, "exp_Llo_at_alpha_lo": w2}
    c["normalizer_input_spread_lower"] = t(dn(max(F(0), 2 * (e1[0] - e2[1]))))
else:
    c["endpoint_spread"] = None; c["normalizer_input_spread_lower"] = "0/1"
goal = None if r["width_goal"] is None else F(r["width_goal"])
widths = [F(b) - F(a) for a, b in c["bounds"].values()]
spread = F(c["normalizer_input_spread_lower"])
c["outcome"] = "BOUNDS_ONLY" if goal is None else ("TOLERANCE_MET" if all(x <= goal for x in widths) else "HOLD_INPUT_UNCERTAINTY" if spread > goal else "HOLD_TARGET_UNMET")
c["error_accounting"]["producer_round_quantum"] = t(F(1, D))
c["error_accounting"]["finite_widths"] = {k: t(v[1] - v[0]) for k, v in sums.items()}
try:
    vc.verify(json.loads(json.dumps(c))); v = "ACCEPT"
except ValueError as ex:
    v = "REJECT:" + str(ex)
json.dump(c, open(sys.argv[3], "w"), sort_keys=True)
ob = base["bounds"]
print(v, "tightened vs producer widths:", {k: "%.3g->%.3g" % (float(F(ob[k][1]) - F(ob[k][0])), float(F(c["bounds"][k][1]) - F(c["bounds"][k][0]))) for k in ob})
