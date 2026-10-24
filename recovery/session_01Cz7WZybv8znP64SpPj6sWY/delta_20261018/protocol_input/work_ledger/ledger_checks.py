"""Ledger/contract composition checks for PROSPECTIVE_CORRECTIVE_PROTOCOL_V1 sections 7-9.
Pure exact-rational algebra on synthetic symbols; no field/driver/sampler/experiment."""
from fractions import Fraction as F
import random, itertools

random.seed(20261018)
def rnd(lo, hi, den=997):
    return F(random.randint(int(lo*den), int(hi*den)), den)

ok = {}
def chk(name, cond):
    ok.setdefault(name, True)
    if not cond:
        ok[name] = False

# 1. P20 vs symmetric D_j ledger and P19 width.
for _ in range(20000):
    V = F(576); Q = F(random.randint(1, 400))
    Lb = rnd(0, float(Q)); Ub = rnd(float(Lb), float(Q))
    r = rnd(0.001, 50); H = rnd(0, 0.05); qbV = rnd(0, 0.01); t = rnd(0, 0.05)
    B = H + qbV
    Am = max(F(0), max(F(0), Lb - r)/V - B)
    Ap = min(Q, Ub + r)/V + B + t
    yhat = (Lb + Ub)/(2*V); g = (Ub - Lb)/(2*V)
    D = H + qbV + t + g + r/V
    chk("P20_subset_yhat_pm_D", yhat - D <= Am and Ap <= yhat + D)
    chk("P19_width_bound", Ap - Am <= (Ub - Lb)/V + 2*r/V + 2*B + t)
    # any nu in P20 satisfies |nu-yhat|<=D
    nu = Am + (Ap - Am)*rnd(0, 1)
    chk("cover_implies_abs_le_D", abs(nu - yhat) <= D)

# 2. P22 => P21 strictly inside the eta tube (theory alone).
for _ in range(20000):
    Cm = rnd(0.01, 2); Cp = Cm + rnd(0, 0.2); w = rnd(0.01, 5); ER = rnd(0, 0.1)
    C0 = (Cm + Cp)/2; uC = (Cp - Cm)/2; T = uC*w + ER
    D = rnd(1e-6, 1)  # D_j >= r/V > 0
    eta = rnd(0, 1)
    if T + 2*D <= eta*C0*w:
        lo = max(F(0), Cm*w - ER); hi = Cp*w + ER
        chk("P22_implies_P21_in_open_tube", (1-eta)*C0*w < lo and hi < (1+eta)*C0*w)

# 3. Jensen infeasibility of P22 with only the sure width bound g<=Q/(2V).
#    t_j >= T_Q = E(N-Q)_+/V >= (E N - Q)_+/V, so 2(g+t) >= nu.
for _ in range(5000):
    V = F(576); Q = random.randint(1, 60)
    support = list(range(0, 120)); wts = [random.random()**3 for _ in support]
    s = sum(wts); probs = [F(x).limit_denominator(10**6) for x in wts]
    tot = sum(probs); probs = [p/tot for p in probs]
    EN = sum(p*k for p, k in zip(probs, support))
    TQ = sum(p*max(0, k-Q) for p, k in zip(probs, support))
    nu = EN/V; g_sure = F(Q)/(2*V); tQ = TQ/V
    chk("jensen_TQ_ge_meanexcess", TQ >= max(F(0), EN - Q))
    chk("two_D_ge_nu_with_sure_width", 2*(g_sure + tQ) >= nu)
# Consequence: under null nu >= C0 w - T, so T+2D >= C0 w > eta C0 w for eta<1.

# 4. Two-tube arithmetic and dyadic mass ratio.
q = F(21, 19)
chk("q6_lt_2", q**6 < 2)
chk("ints", 21**6 == 85766121 and 2*19**6 == 94091762)
q10 = F(11, 9); chk("q10", q10**3 < 2 and q10**6 > 2)

# 5. Decision-vocabulary overlap witness: one bin disjoint from P21 while another bin
#    has every row unresolved [0,Q] (INCOMPLETE condition) -> both labels apply.
V = F(576); Q = F(50); r = F(2); B = F(0); t = F(0)
# bin 0: all rows unresolved
Am0 = max(F(0), max(F(0), F(0) - r)/V - B); Ap0 = min(Q, Q + r)/V + B + t
# bin 1: resolved rows far above prediction
Lb1 = Ub1 = F(40); Am1 = max(F(0), (Lb1 - r)/V - B); Ap1 = min(Q, Ub1 + r)/V + B + t
P21_1 = (F(1, 1000), F(2, 1000))
chk("overlap_discrepancy", Am1 > P21_1[1])
chk("overlap_vacuous_bin0", Am0 == 0 and Ap0 == Q/V)

for k, v in ok.items():
    print(f"{k}: {'PASS' if v else 'FAIL'}")
