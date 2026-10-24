"""Section 7 checks: random finite diagrams with explicit delta-matchings.
Pure combinatorics/exact rationals. No field, sampler, driver or experiment."""
from fractions import Fraction as F
import random, itertools

random.seed(20261018)

def cnt(lifes, lo, hi):
    if lo >= hi:
        return 0
    return sum(1 for x in lifes if lo <= x < hi)

def rand_pair(delta, nmatch, nshort_f, nshort_p, grid=40, top=F(3)):
    """Build two finite diagrams (birth,death) with an explicit delta-matching:
    matched pairs move each coordinate by <= delta (L-infinity), unmatched points
    have L-inf distance to diagonal <= delta, i.e. lifetime <= 2*delta.
    Coordinates on a rational grid so that boundary atoms/ties occur often."""
    Fd, Pd = [], []
    step = delta / 2
    for _ in range(nmatch):
        d = F(random.randint(-grid, grid), 10)
        life = step * random.randint(1, int(top / step))
        b = d + life
        db = step * random.randint(-2, 2)  # |db|<=delta
        dd = step * random.randint(-2, 2)
        b2, d2 = b + db, d + dd
        if b2 - d2 <= 0:
            continue  # would cross diagonal; skip (P-side point must be finite positive)
        Fd.append((b, d)); Pd.append((b2, d2))
    for _ in range(nshort_f):
        d = F(random.randint(-grid, grid), 10)
        Fd.append((d + step * random.randint(1, 4), d))  # lifetime <= 2delta
    for _ in range(nshort_p):
        d = F(random.randint(-grid, grid), 10)
        Pd.append((d + step * random.randint(1, 4), d))
    return [b - d for b, d in Fd], [b - d for b, d in Pd]

viol = {"P8_lower": 0, "P8_upper": 0, "rev_lower": 0, "rev_upper": 0, "P9": 0}
trials = 0
for _ in range(4000):
    delta = F(random.choice([1, 2, 3, 5]), 20)
    fl, pl = rand_pair(delta, random.randint(0, 25), random.randint(0, 6), random.randint(0, 6))
    # bins aligned to the grid so endpoint atoms happen
    a = 2 * delta + F(random.randint(1, 30), 40)  # a > 2 delta strictly
    b = a + F(random.randint(1, 60), 40)
    lo, hi = a - 2 * delta, b + 2 * delta
    clo, chi = a + 2 * delta, b - 2 * delta
    NFm, NFp = cnt(fl, clo, chi), cnt(fl, lo, hi)
    NPm, NPp = cnt(pl, clo, chi), cnt(pl, lo, hi)
    M, N = cnt(pl, a, b), cnt(fl, a, b)
    trials += 1
    viol["P8_lower"] += NFm > M
    viol["P8_upper"] += M > NFp
    viol["rev_lower"] += NPm > N
    viol["rev_upper"] += N > NPp
    for Q in [1, 2, 3, 5, 8]:
        cq = lambda x: min(x, Q)
        S = NFp - NFm  # I^- subset I^+, so N_F(S) = difference (also when I^- empty)
        if abs(cq(M) - cq(N)) > S:
            viol["P9"] += 1
print("random delta-matching trials:", trials, "violations:", viol)

# Counterexample 1: equality a = 2 delta (protocol's own guard)
delta = F(1, 10)
pl, fl = [2 * delta], []
print("a=2delta: N_P([a,1))=", cnt(pl, 2 * delta, F(1)), " N_F(I^+)=", cnt(fl, 0, F(1) + 2 * delta))

# Counterexample 2: mixing conventions, a > delta_radius only (candidate 'a>delta' read with protocol delta)
a = F(3, 20)  # delta < a <= 2 delta
pl, fl = [F(3, 20)], []
# single P point at lifetime 3/20 matched to diagonal costs 3/40 <= 1/10
print("a in (delta,2delta]: bottleneck to empty =", F(3, 40), "<= delta", F(3, 40) <= delta,
      "; N_P([a,1))=", cnt(pl, a, F(1)), " N_F([a-2delta,1+2delta) cap (0,inf))=", cnt(fl, F(0), F(1) + 2 * delta))

# Data-side (reversed orientation) expected bridge on an explicit finite probability space.
# Outcomes: (prob, F lifetimes, P lifetimes, good?) ; good outcomes carry a delta-matching.
delta = F(1, 10); a, b = F(1, 2), F(1); Q = 3
Im, Ip = (a + 2 * delta, b - 2 * delta), (a - 2 * delta, b + 2 * delta)
space = []
for _ in range(200):
    fl, pl = rand_pair(delta, random.randint(0, 8), random.randint(0, 3), random.randint(0, 3))
    space.append((F(1, 1000), fl, pl, True))
# bad outcomes: no matching relation at all, arbitrary counts
for k in range(8):
    space.append((F(1, 100), [F(3, 4)] * (k * 7), [F(3, 5)] * (k % 3), False))
tot = sum(p for p, *_ in space)
space = [(p / tot, fl, pl, g) for p, fl, pl, g in space]
beta = sum(p for p, fl, pl, g in space if not g)
cq = lambda x: min(x, Q)
EcN = sum(p * cq(cnt(fl, a, b)) for p, fl, pl, g in space)
EN = sum(p * cnt(fl, a, b) for p, fl, pl, g in space)
muPm = sum(p * cq(cnt(pl, *Im)) for p, fl, pl, g in space)
muPp = sum(p * cq(cnt(pl, *Ip)) for p, fl, pl, g in space)
T = EN - EcN
lowerOK = EN >= muPm - Q * beta
upperOK = EN <= muPp + Q * beta + T
print("data-side bridge: beta=", beta, " E N_F(I)=", EN, " [muP(I-)-Q beta, muP(I+)+Q beta+T]=",
      muPm - Q * beta, muPp + Q * beta + T, " holds:", lowerOK and upperOK)
# protocol orientation on same space for comparison
muP = sum(p * cq(cnt(pl, a, b)) for p, fl, pl, g in space)
H = sum(p * (cnt(fl, *Ip) - cnt(fl, *Im)) for p, fl, pl, g in space)
print("protocol P12: [muP-B, muP+B+T]=", max(0, muP - H - Q * beta), muP + H + Q * beta + T,
      " holds:", max(0, muP - H - Q * beta) <= EN <= muP + H + Q * beta + T)

# P13 notation: (E N)^2 in place of E[N^2] is invalid
p, k, Q = F(1, 10), 2, 1
lhs = p * max(0, k - Q)
print("P13 misread: E(N-Q)_+ =", lhs, " (EN)^2/(4Q) =", (p * k) ** 2 / (4 * Q), " E[N^2]/(4Q) =", p * k * k / (4 * Q))
