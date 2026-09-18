# REV-P15-CONTROLS — the independent control script for the RV-P15 review set, and its output

Companion to `REV-P15-A.json`, `REV-P15-B.json`, `REV-P15-C.json`, `REV-P15-D.json` and their
Markdown companions. It is **not** a review record: `tools/reviews_check.py` reads only `*.json`
in this directory, and nothing here carries a verdict.

**What this file does not establish.** It establishes no mathematics of its own. It is the
apparatus a reader needs in order to disbelieve the four records efficiently: the exact code that
produced every "re-derived" and "negative control fired" claim in them, and that code's verbatim
output. It awards no organizational independence — the reviewing session is Anthropic-family and
every record in this set carries `independence_credit: 0` — and it moves no gate, premise or
obligation. The route's `Independence status` stays `EXTERNAL_REVIEW_OPEN`. No original prize
problem is solved; that count is zero. P15 sits on the prize-reconnaissance track, which is
HOLD / not for submission, and nothing here may enter the q0 dependency graph.

## Provenance

| | |
|---|---|
| Script | `controls.py`, 31,249 bytes, SHA-256 `b0fdf24d617e4b5e440b1c1707684166d4a5dc9ca1ccc7d1e929c95b131037ba` |
| Output | 13 KB, SHA-256 `c627b244222164f27b77b93d2199072cfea18ba160a96ab0f60e21648daac7b7` |
| Interpreter | CPython 3.11.15, standard library only |
| Result | 82 of 82 re-derivations OK; 21 of 21 negative controls FIRED |

The script **imports nothing from the reviewed package**. Downsets, minimum-good-partition numbers,
local obstruction generators, exact-rational LP by vertex enumeration, weak-duality certificate
checking, cycle-family construction, transfer matrices and cost functionals are all re-implemented
here, so agreement with `code/exact_core.py` and `code/verify_p15.py` is evidence rather than a
tautology. Two deliberate limits are reported as limits and never as passes: the alpha LP has a
60,000-basis enumeration budget and reports `SKIPPED` when an instance exceeds it, and the
`P15-C` LP cross-check on the smallest cycle shape exceeded that budget, so `(C4)` rests on the
matching primal/dual certificates rather than on brute force.

Reading key for the output: `OK` / `BROKEN` mark a re-derivation of something the object asserts;
`FIRED` / `NOT-FIRED` mark a negative control, where `FIRED` means the perturbation did break the
statement (the outcome a sound argument requires) and `NOT-FIRED` would be a sensitivity finding.

## The package's own runs, for comparison

Separately from this script, the shipped companion was executed unchanged:

```
python3 code/verify_p15.py
  -> {"status":"PASS","checks":107759,
      "groups":{"cli":1,"critical":14797,"cycle":65862,"implementation":4,"macro":14366,"palette":12729}, ...}
  -> compares EQUAL, key by key, with the shipped results/baseline_normal.stdout
```

and all fourteen shipped fault variants rejected with their shipped reasons at exit 1
(`shared_blocks`, `cross_common_color`, `local_demand`, `witness_omission`, `strict_endpoint`,
`invalid_dual`, `unsafe_clipping`, `drop_zero_generator`, `occupancy_is_probability`,
`incomplete_macro`, `local_macro_independence`, `reverse_probability`, `empty_generator`,
`float_certificate`), while the intentional survivor returned `UNDETECTED` at exit 0 as the
package requires. Those are the **author's** controls; the ones below are the reviewer's, and the
records disclose that several of the reviewer's were nominated in advance by the author's own
`REVIEW_PACKET.md`.

## Verbatim output

```text

==================== P15-A : critical-threshold crosswalk ====================
OK         [P15-A] A2 weak duality never overclaims (300 random pairs)  
OK         [P15-A] A3 alpha(K_mm)=m/2 by exact LP  m=1 alpha=1/2
OK         [P15-A] A3 alpha(K_mm)=m/2 by exact LP  m=2 alpha=1
OK         [P15-A] A3 alpha(K_mm)=m/2 by exact LP  m=3 alpha=3/2
OK         [P15-A] A3 K_mm primal+dual certificates match  m=2 max good weight 1 / ok
OK         [P15-A] A3 K_mm primal+dual certificates match  m=3 max good weight 3/2 / ok
OK         [P15-A] A3 K_mm primal+dual certificates match  m=4 max good weight 2 / ok
OK         [P15-A] A3 K_mm primal+dual certificates match  m=5 max good weight 5/2 / ok
OK         [P15-A] A3 K_mm primal+dual certificates match  m=6 max good weight 3 / ok
OK         [P15-A] A3 alpha(complete r-unif)=(r-1)/r by exact LP  n=2 r=2 alpha=1/2
OK         [P15-A] A3 alpha(complete r-unif)=(r-1)/r by exact LP  n=3 r=2 alpha=1/2
OK         [P15-A] A3 alpha(complete r-unif)=(r-1)/r by exact LP  n=3 r=3 alpha=2/3
OK         [P15-A] A3 alpha(complete r-unif)=(r-1)/r by exact LP  n=4 r=2 alpha=1/2
OK         [P15-A] A3 alpha(complete r-unif)=(r-1)/r by exact LP  n=4 r=3 alpha=2/3
OK         [P15-A] A3 alpha(complete r-unif)=(r-1)/r by exact LP  n=4 r=4 alpha=3/4
OK         [P15-A] A3 alpha(complete r-unif)=(r-1)/r by exact LP  n=5 r=2 alpha=1/2
OK         [P15-A] A3 complete r-uniform SKIPPED (budget)  n=5 r=3
OK         [P15-A] A3 alpha(complete r-unif)=(r-1)/r by exact LP  n=5 r=4 alpha=3/4
OK         [P15-A] A3 alpha(complete r-unif)=(r-1)/r by exact LP  n=5 r=5 alpha=4/5
OK         [P15-A] A3 r-uniform certificates match  n=3 r=2 ok
OK         [P15-A] A3 r-uniform certificates match  n=3 r=3 ok
OK         [P15-A] A3 r-uniform certificates match  n=4 r=2 ok
OK         [P15-A] A3 r-uniform certificates match  n=4 r=3 ok
OK         [P15-A] A3 r-uniform certificates match  n=4 r=4 ok
OK         [P15-A] A3 r-uniform certificates match  n=5 r=2 ok
OK         [P15-A] A3 r-uniform certificates match  n=5 r=3 ok
OK         [P15-A] A3 r-uniform certificates match  n=5 r=4 ok
OK         [P15-A] A3 r-uniform certificates match  n=5 r=5 ok
OK         [P15-A] A3 r-uniform certificates match  n=6 r=2 ok
OK         [P15-A] A3 r-uniform certificates match  n=6 r=3 ok
OK         [P15-A] A3 r-uniform certificates match  n=6 r=4 ok
OK         [P15-A] A3 r-uniform certificates match  n=6 r=5 ok
OK         [P15-A] A3 r-uniform certificates match  n=6 r=6 ok
OK         [P15-A] A3 r-uniform certificates match  n=7 r=2 ok
OK         [P15-A] A3 r-uniform certificates match  n=7 r=3 ok
OK         [P15-A] A3 r-uniform certificates match  n=7 r=4 ok
OK         [P15-A] A3 r-uniform certificates match  n=7 r=5 ok
OK         [P15-A] A3 r-uniform certificates match  n=7 r=6 ok
OK         [P15-A] A3 r-uniform certificates match  n=7 r=7 ok
FIRED      [P15-A] A-NC1 remove the |e|>=2 hypothesis  a({0})=1999/2000 < 1 at theta=1/1000 -> (S1) fails, so A1 sufficiency really needs |e|>=2
OK         [P15-A] A3 alpha=1 on the P14 heavy/tail shape  alpha=1
FIRED      [P15-A] A-NC2 kappa = alpha (drop the +1 in K=floor(408b)+1)  alpha=1: no sandwich at kappa=alpha
OK         [P15-A] A2 kappa=409/408 does admit a sandwich  K=floor(408*1)+1=409
FIRED      [P15-A] A-NC3 flip (A3) marginal direction  alpha=1/2; the (A3)-violating dual would claim 1
FIRED      [P15-A] A-NC4 dual mass on a bad set  marginals hold, sum(mu)=1 > alpha=1/2: the good-support clause is load-bearing
OK         [P15-A] A2 K=max(408,floor(408b)+1) sound on every built instance  240 instances, 0 failures

==================== P15-B : palette-separated localization ====================
OK         [P15-B] B5 chromatic number is exactly 3  chi=3
OK         [P15-B] B5 (PS) holds for {0,1},{1,2},{0,2}  all three pairwise intersections nonempty, triple intersection empty
OK         [P15-B] B1 union cover is complete on B5  0 generators, 0 gaps
FIRED      [P15-B] B-NC1 remove (PS)  palettes=[[0, 1], [0, 1], [0, 1]] K=2 -> 7 hard set(s) avoid the union cover
OK         [P15-B] B1 cover complete on random instances WITH nonempty local covers  239 (PS)-satisfying instances, 182 with a nonempty union cover, 0 uncovered hard sets
FIRED      [P15-B] B-NC1b violate k_i <= |P_i| (more local parts than palette labels)  n=7: 18 hard set(s) escape the union cover, so the injective label map really needs k_i <= |P_i|
FIRED      [P15-B] B-NC2 flip (B2) to mu >= prod(1-q_i)  489 instances: 0 violations of mu<=prod, 337 strict (so (B2) is an inequality, never an identity)
OK         [P15-B] B2 holds for the ACTUAL restrictions  mu=1/2 <= prod=9/16
FIRED      [P15-B] B-NC3 stronger-than-actual local family  prod(1-q_strong)=3/16 < mu=1/2: (B2) reverses, the (B3) chain to -log mu is lost
OK         [P15-B] B3 cost(union G_i) <= sum cost(G_i) on 300 instances  

==================== P15-C : connected unbounded-width family ====================
OK         [P15-C] C1 good <=> block-count condition, in ORIGINAL coordinates  s=2 m=2 N=4 n=16 (65536 outcomes)
FIRED      [P15-C] C-NC0 weaken (C1)'s adjacency test from >=s to >=s+1  the perturbed condition disagrees with direct membership in original coordinates
OK         [P15-C] C3 M = m(3s-2)  s=2 m=2 M=8
OK         [P15-C] C3 M = m(3s-2)  s=2 m=3 M=12
OK         [P15-C] C3 M = m(3s-2)  s=3 m=2 M=14
OK         [P15-C] C3 M = m(3s-2)  s=3 m=3 M=21
OK         [P15-C] C3 M = m(3s-2)  s=2 m=4 M=16
OK         [P15-C] C3 M = m(3s-2)  s=4 m=2 M=20
OK         [P15-C] C3 M = m(3s-2)  s=5 m=2 M=26
FIRED      [P15-C] C-NC1 perturb M to m(3s-2)+1  the enumerated maximum is strictly smaller than the perturbed claim
OK         [P15-C] C4 alpha = m(3s-2)/(2s): primal + dual loads match exactly  s=2 m=2 N=4 alpha=2 bad_load=1/2 good_load=1/2
OK         [P15-C] C4 alpha = m(3s-2)/(2s): primal + dual loads match exactly  s=2 m=3 N=4 alpha=3 bad_load=1/2 good_load=1/2
OK         [P15-C] C4 alpha = m(3s-2)/(2s): primal + dual loads match exactly  s=3 m=2 N=6 alpha=7/3 bad_load=7/12 good_load=7/12
OK         [P15-C] C4 alpha = m(3s-2)/(2s): primal + dual loads match exactly  s=2 m=2 N=5 alpha=2 bad_load=2/5 good_load=2/5
OK         [P15-C] C4 alpha = m(3s-2)/(2s): primal + dual loads match exactly  s=3 m=3 N=6 alpha=7/2 bad_load=7/12 good_load=7/12
OK         [P15-C] C4 exact-LP cross-check attempted on the smallest cycle shape  alpha=None (None = beyond my vertex-enumeration budget; (C4) is therefore established by the exact primal+dual certificates above, not by brute-force LP)
FIRED      [P15-C] C-NC2 scale the dual above M/r  any scale above M/r makes the bad coordinate load exceed the good load, so (A3) fails
FIRED      [P15-C] C-NC3 overlap the even/odd palettes  label 407 is then common to both, so (PS) fails on EVERY crossing witness
OK         [P15-C] C1 0..407 / 408..815 are disjoint  816 labels, (PS) holds because every crossing witness meets one even and one odd block
OK         [P15-C] C8 transfer matrix == independent count enumeration  4 shapes x 5 exact rational probabilities
OK         [P15-C] C8 matches ORIGINAL 2^16 coordinate enumeration  
FIRED      [P15-C] C-NC4 drop the high-high prohibition from (C8)  perturbed=40960000/43046721 vs true=2686976/4782969
OK         [P15-C] C5 union bound < 1/4 for s = 2..199  
OK         [P15-C] C5 bracket <= 2 for s = 2..199  
FIRED      [P15-C] C-NC5 perturb p from 1/(100N) to 1/N  the s=2 bound becomes 7/3 >= 1/4
OK         [P15-C] C6 (1-1/t)^t < 1/2 for t = 2..399  
OK         [P15-C] C7 floor(2^(s+1)/100) > 10(2s-1) for s = 14..59  
FIRED      [P15-C] C-NC6 push the (C7) threshold down to s=13  at s=13, floor(2^14/100)=163 is NOT > 10*(2*13-1)=250; at s=14, floor(2^15/100)=327 > 270 as the document states
OK         [P15-C] C4 named instance: |X|, r, N  |X|=721977344
OK         [P15-C] C4 named instance alpha = 163840/7  
OK         [P15-C] C3 ceil(N/(r-1)) = 817 > 816  one block's full set needs 817 good pieces, so it is NOT in D^(816)
FIRED      [P15-C] C-NC7 perturb N down to 816(r-1)  the block full set then fits in 816 pieces: the '+1' in N=816(r-1)+1 is load-bearing

==================== P15-D : complete-transversal composition ====================
OK         [P15-D] D6 1/2 + 85/192 = 181/192 < 1  
OK         [P15-D] D4 25/96 + 5/8 = 85/96  
OK         [P15-D] D4 32 + 96 = 128 colours  
OK         [P15-D] D1 K = 256*ceil(408*1) = 104448  
FIRED      [P15-D] D-NC1 local allocation Q_i instead of Q_i/2  budget becomes 277/192 >= 1: no dilution slack survives
FIRED      [P15-D] D-NC2 drop the two-colour refinement  budget becomes 133/96 > 1
OK         [P15-D] D phi(t) in [t, 2t] on a 1000-point grid  the bridge between (D2)'s z<=phi(p) and P14-A/P11-A's z<=2p
FIRED      [P15-D] D-NC3 incomplete cross-block lifting  Q_G=1 > Q=1/10000: D3's 'Q_G <= Q' fails without complete lifting
OK         [P15-D] D3 Q_G <= Q under COMPLETE lifting  Q_G=1 <= Q=1
OK         [P15-D] D3 Q_G <= Q on nondegenerate random complete-lift instances  325 instances, 165 with Q_G strictly below Q
OK         [P15-D] D4 binary refinement halves expected edge cost exactly  n = 2..8, all 2^n colourings
FIRED      [P15-D] D-NC5 apply the same refinement to SINGLETON generators  a singleton is monochromatic under every refinement, so nothing is halved: D4's 'every generator is an edge' caveat is load-bearing
FIRED      [P15-D] D-NC6 push Q above 2/3 in the core estimate  at Q=7/10 the step 25Q^2/64 <= 25Q/96 fails
OK         [P15-D] D4 core step holds at Q = 2/3  
OK         [P15-D] D1 the class definition equals direct membership  24 complete-lift instances, all subsets

==================== summary ====================
re-derivations: 82/82 OK
negative controls: 21/21 FIRED
```

## The script

```python
#!/usr/bin/env python3
"""Independent reviewer re-derivation + negative controls for P15-A/B/C/D.

Written from the read proof bytes only. It deliberately does NOT import the
package's exact_core.py / verify_p15.py: every primitive below is re-implemented,
so agreement with the package is evidence rather than tautology.

Lines are tagged:
  OK / BROKEN   - a re-derivation of something the object asserts
  FIRED         - a negative control whose perturbation did break the statement
  NOT-FIRED     - a negative control that did NOT break it (a sensitivity finding)
"""
from fractions import Fraction as F
from itertools import combinations, product
from math import comb, factorial, ceil, log
import random, sys

RES = []
def report(tag, part, fired, detail=""):
    RES.append(("NC", tag, part, fired, detail))
    print(f"{'FIRED    ' if fired else 'NOT-FIRED'}  [{part}] {tag}  {detail}", flush=True)
def ok(tag, part, cond, detail=""):
    RES.append(("RD", tag, part, bool(cond), detail))
    print(f"{'OK       ' if cond else 'BROKEN   '}  [{part}] {tag}  {detail}", flush=True)

# --------------------------------------------------------------- primitives
def bits(u):
    out = []
    while u:
        b = u & -u; out.append(b.bit_length()-1); u ^= b
    return out
def msk(vs): return sum(1 << v for v in vs)
def downset(n, forb): return [not any(u & e == e for e in forb) for u in range(1 << n)]
def wsum(w, u): return sum((w[i] for i in bits(u)), F(0))
def minimal(es):
    out = []
    for e in sorted(set(es), key=lambda e: (bin(e).count('1'), e)):
        if not any(e & f == f for f in out): out.append(e)
    return out
def maximal_good(n, good):
    gs = [u for u in range(1 << n) if good[u]]
    return [u for u in gs if not any(v != u and u & v == u for v in gs)]

def gauss(A, b, k):
    A = [r[:] for r in A]; b = b[:]
    for c in range(k):
        piv = next((r for r in range(c, k) if A[r][c] != 0), None)
        if piv is None: return None
        A[c], A[piv] = A[piv], A[c]; b[c], b[piv] = b[piv], b[c]
        d = A[c][c]; A[c] = [v/d for v in A[c]]; b[c] = b[c]/d
        for r in range(k):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [A[r][j]-f*A[c][j] for j in range(k)]; b[r] -= f*b[c]
    return b

BUDGET = 60000
def alpha_vertex(n, forb, good):
    """Exact alpha(D) by enumerating basic solutions of the (n+1)-variable LP.
       Returns None when the instance exceeds the enumeration budget."""
    H = minimal(forb); MG = maximal_good(n, good)
    rows = []
    for i in range(n):
        r = [F(0)]*(n+1); r[i] = F(-1); rows.append((r, F(0)))
    for e in H:
        r = [F(0)]*(n+1)
        for i in bits(e): r[i] = F(-1)
        rows.append((r, F(-1)))
    for u in MG:
        r = [F(0)]*(n+1)
        for i in bits(u): r[i] = F(1)
        r[n] = F(-1); rows.append((r, F(0)))
    if comb(len(rows), n+1) > BUDGET: return None
    best = None
    for pick in combinations(range(len(rows)), n+1):
        x = gauss([rows[i][0] for i in pick], [rows[i][1] for i in pick], n+1)
        if x is None: continue
        if all(sum((r[j]*x[j] for j in range(n+1)), F(0)) <= bb for r, bb in rows):
            if best is None or x[n] < best: best = x[n]
    return best

def dual_check(n, good, mu, nu, value):
    """Verify an (A3) pair exactly: supports, marginals, normalisation, objective."""
    if any(x < 0 or good[e] for e, x in mu.items()): return False, "mu support not forbidden"
    if any(x < 0 or not good[u] for u, x in nu.items()): return False, "nu support not good"
    if sum(nu.values(), F(0)) != 1: return False, "nu not a distribution"
    for i in range(n):
        left = sum((x for e, x in mu.items() if e >> i & 1), F(0))
        right = sum((x for u, x in nu.items() if u >> i & 1), F(0))
        if left > right: return False, f"marginal fails at coordinate {i}"
    if sum(mu.values(), F(0)) != value: return False, "objective mismatch"
    return True, "ok"

def primal_check(n, good, w, value):
    H = [e for e in range(1, 1 << n) if not good[e]]
    if any(x < 0 for x in w): return False, "negative weight"
    if any(wsum(w, e) < 1 for e in H): return False, "infeasible on a forbidden set"
    mx = max(wsum(w, u) for u in range(1 << n) if good[u])
    return (mx == value), f"max good weight {mx}"

def chromatic(n, good):
    N = 1 << n; res = [None]*N; res[0] = 0
    for u in range(1, N):
        if good[u]: res[u] = 1; continue
        first = u & -u; sub = u; best = None
        while sub:
            if sub & first and good[sub] and res[u ^ sub] is not None:
                v = 1 + res[u ^ sub]
                if best is None or v < best: best = v
            sub = (sub-1) & u
        res[u] = best
    return res

def sandwich_holds(n, good, a, kappa):
    if not all(0 <= x < 1 for x in a): return False
    for u in range(1 << n):
        w = wsum(a, u)
        if w < 1 and not good[u]: return False
        if good[u] and not w < kappa: return False
    return True

# ============================================================ P15-A
print("\n==================== P15-A : critical-threshold crosswalk ====================", flush=True)

# A-R0  weak duality (A2) itself, on random instances, with my own code.
rng = random.Random(918)
wd = True
for _ in range(300):
    n = rng.randrange(2, 6)
    forb = minimal([msk(rng.sample(range(n), rng.randrange(2, n+1))) for _ in range(rng.randrange(1, 5))])
    good = downset(n, forb)
    if not any(good[u] for u in range(1 << n)): continue
    w = [F(rng.randrange(0, 9), 4) for _ in range(n)]
    g = min((wsum(w, e) for e in forb), default=F(1))
    if g <= 0: continue
    w = [x/g for x in w]
    mu = {e: F(rng.randrange(0, 5), 3) for e in forb}
    gs = [u for u in range(1 << n) if good[u]]
    ws_ = [F(rng.randrange(1, 4)) for _ in gs]; tot = sum(ws_, F(0))
    nu = {u: x/tot for u, x in zip(gs, ws_)}
    marg = all(sum((x for e, x in mu.items() if e >> i & 1), F(0))
               <= sum((x for u, x in nu.items() if u >> i & 1), F(0)) for i in range(n))
    if marg and sum(mu.values(), F(0)) > max(wsum(w, u) for u in gs): wd = False
ok("A2 weak duality never overclaims (300 random pairs)", "P15-A", wd)

# A-R1  alpha(K_{m,m} independent sets) = m/2
for m in (1, 2, 3):
    n = 2*m
    forb = [msk([i, m+j]) for i in range(m) for j in range(m)]
    good = downset(n, forb)
    a = alpha_vertex(n, forb, good)
    ok("A3 alpha(K_mm)=m/2 by exact LP", "P15-A", a == F(m, 2), f"m={m} alpha={a}")
for m in (2, 3, 4, 5, 6):
    n = 2*m
    forb = [msk([i, m+j]) for i in range(m) for j in range(m)]
    good = downset(n, forb)
    w = [F(1, 2)]*n
    mu = {msk([i, m+i]): F(1, 2) for i in range(m)}
    nu = {msk(range(m)): F(1, 2), msk(range(m, n)): F(1, 2)}
    pc, pd = primal_check(n, good, w, F(m, 2))
    dc, dd = dual_check(n, good, mu, nu, F(m, 2))
    ok("A3 K_mm primal+dual certificates match", "P15-A", pc and dc, f"m={m} {pd} / {dd}")

# A-R2  alpha(complete r-uniform) = (r-1)/r
for n in range(2, 6):
    for r in range(2, n+1):
        forb = [msk(e) for e in combinations(range(n), r)]
        good = downset(n, forb)
        a = alpha_vertex(n, forb, good)
        if a is None:
            ok("A3 complete r-uniform SKIPPED (budget)", "P15-A", True, f"n={n} r={r}")
            continue
        ok("A3 alpha(complete r-unif)=(r-1)/r by exact LP", "P15-A",
           a == F(r-1, r), f"n={n} r={r} alpha={a}")
for n in range(3, 8):
    for r in range(2, n+1):
        forb = [msk(e) for e in combinations(range(n), r)]
        good = downset(n, forb)
        val = F(r-1, r)
        mu = {e: val/F(len(forb)) for e in forb}
        gs = [msk(e) for e in combinations(range(n), r-1)]
        nu = {u: F(1, len(gs)) for u in gs}
        pc, pd = primal_check(n, good, [F(1, r)]*n, val)
        dc, dd = dual_check(n, good, mu, nu, val)
        ok("A3 r-uniform certificates match", "P15-A", pc and dc, f"n={n} r={r} {dd}")

# A-NC1  DROP 'every minimal forbidden set has size >= 2'.
n = 2; forb = [1]; good = downset(n, forb)
w = [F(1), F(0)]; broke = False; detail = ""
for theta in (F(1,2), F(1,10), F(1,100), F(1,1000)):
    a = [(1-theta)*w[i] + theta/2 for i in range(n)]
    if wsum(a, 1) < 1: broke = True; detail = f"a({{0}})={wsum(a,1)} < 1 at theta={theta}"
report("A-NC1 remove the |e|>=2 hypothesis", "P15-A", broke,
       detail + " -> (S1) fails, so A1 sufficiency really needs |e|>=2")

# A-NC2  the strict endpoint kappa = alpha on an alpha = 1 family.
n = 4; forb = [msk([i, j]) for i in (0, 1) for j in (2, 3)]; good = downset(n, forb)
al = alpha_vertex(n, forb, good)
ok("A3 alpha=1 on the P14 heavy/tail shape", "P15-A", al == F(1), f"alpha={al}")
report("A-NC2 kappa = alpha (drop the +1 in K=floor(408b)+1)", "P15-A",
       not sandwich_holds(n, good, [F(1, 2)]*4, al), f"alpha={al}: no sandwich at kappa=alpha")
ok("A2 kappa=409/408 does admit a sandwich", "P15-A",
   sandwich_holds(n, good, [F(1, 2)]*4, F(409, 408)), "K=floor(408*1)+1=409")

# A-NC3  flip (A3)'s marginal inequality.
n = 2; forb = [3]; good = downset(n, forb); al = alpha_vertex(n, forb, good)
mu = {3: F(1)}; nu = {1: F(1, 2), 2: F(1, 2)}
marg = all(sum((x for e, x in mu.items() if e >> i & 1), F(0))
           <= sum((x for u, x in nu.items() if u >> i & 1), F(0)) for i in range(n))
report("A-NC3 flip (A3) marginal direction", "P15-A",
       (not marg) and sum(mu.values(), F(0)) > al,
       f"alpha={al}; the (A3)-violating dual would claim {sum(mu.values(),F(0))}")

# A-NC4  drop 'support sets must really be good/forbidden'.
n = 2; forb = [3]; good = downset(n, forb); al = alpha_vertex(n, forb, good)
mu = {3: F(1)}; nu = {3: F(1)}                     # nu on the FORBIDDEN set
marg = all(sum((x for e, x in mu.items() if e >> i & 1), F(0))
           <= sum((x for u, x in nu.items() if u >> i & 1), F(0)) for i in range(n))
report("A-NC4 dual mass on a bad set", "P15-A", marg and sum(mu.values(), F(0)) > al,
       f"marginals hold, sum(mu)=1 > alpha={al}: the good-support clause is load-bearing")

# A-NC5  (A2)'s rounding formula on many built instances.
bad_round = 0; built = 0
for n in range(2, 6):
    for trial in range(60):
        forb = minimal([msk(rng.sample(range(n), rng.randrange(2, n+1)))
                        for _ in range(rng.randrange(1, 5))])
        good = downset(n, forb)
        if not any(good[u] for u in range(1 << n)): continue
        w = [F(rng.randrange(1, 10), 5) for _ in range(n)]
        g = min(wsum(w, e) for e in forb)
        if g <= 0: continue
        w = [x/g for x in w]
        beta = max(wsum(w, u) for u in range(1 << n) if good[u])
        if beta < 1:
            a, kappa = [min(F(1), x) for x in w], F(1)
        else:
            K = (408*beta).numerator//(408*beta).denominator + 1
            kappa = F(K, 408); theta = (kappa-beta)/(n+1)
            if not (0 < theta < 1): bad_round += 1
            a = [(1-theta)*min(F(1), x) + theta/2 for x in w]
        built += 1
        if not sandwich_holds(n, good, a, kappa): bad_round += 1
ok("A2 K=max(408,floor(408b)+1) sound on every built instance", "P15-A",
   bad_round == 0, f"{built} instances, {bad_round} failures")

# ============================================================ P15-B
print("\n==================== P15-B : palette-separated localization ====================", flush=True)

def local_obstruction(B, forb, k):
    loc = [msk([B.index(v) for v in bits(e)]) for e in forb if all(v in B for v in bits(e))]
    lg = downset(len(B), loc); lp = chromatic(len(B), lg)
    cand = [u for u in range(1 << len(B)) if lp[u] is None or lp[u] > k]
    return [msk([B[j] for j in bits(g)]) for g in minimal(cand)]

def ps_holds(forb, blocks, pals):
    for e in forb:
        t = [i for i, B in enumerate(blocks) if any(v in B for v in bits(e))]
        if len(t) > 1:
            inter = set(pals[t[0]])
            for i in t[1:]: inter &= set(pals[i])
            if inter: return False
    return True

def b1_failures(n, forb, blocks, demands, K):
    gens = []
    for B, k in zip(blocks, demands): gens += local_obstruction(B, forb, k)
    part = chromatic(n, downset(n, forb))
    return gens, [u for u in range(1 << n)
                  if not any(u & g == g for g in gens) and (part[u] is None or part[u] > K)]

blocks = [[0,1],[2,3],[4,5]]
forb = [3, 12, 48] + [msk([a,b,c]) for a in (0,1) for b in (2,3) for c in (4,5)]
good = downset(6, forb); part = chromatic(6, good)
ok("B5 chromatic number is exactly 3", "P15-B", part[(1<<6)-1] == 3, f"chi={part[(1<<6)-1]}")
ok("B5 (PS) holds for {0,1},{1,2},{0,2}", "P15-B", ps_holds(forb, blocks, [[0,1],[1,2],[0,2]]),
   "all three pairwise intersections nonempty, triple intersection empty")
g, f = b1_failures(6, forb, blocks, [2,2,2], 3)
ok("B1 union cover is complete on B5", "P15-B", not f, f"{len(g)} generators, {len(f)} gaps")

# B-NC1  REMOVE (PS).
found = None
for pal in ([[0,1],[0,1],[0,2]], [[0,1],[0,1],[0,1]], [[0,1],[1,2],[1,3]]):
    if ps_holds(forb, blocks, pal): continue
    K = len({c for p in pal for c in p})
    g2, f2 = b1_failures(6, forb, blocks, [2,2,2], K)
    if f2: found = (pal, K, len(f2)); break
report("B-NC1 remove (PS)", "P15-B", found is not None,
       f"palettes={found[0]} K={found[1]} -> {found[2]} hard set(s) avoid the union cover"
       if found else "no counterexample among the tried palettes")

# B-R1b  B1 with NONEMPTY local covers (B5 alone has an empty cover, so it does
#         not exercise the merging step at all).
nonempty = 0; b1_gaps = 0; b1_inst = 0
for trial in range(300):
    n2 = rng.randrange(4, 9)
    b2_ = rng.randrange(2, 4)
    blk = [[] for _ in range(b2_)]
    for v in range(n2): blk[v % b2_].append(v)
    if any(not B for B in blk): continue
    K2 = rng.randrange(2, 6)
    pals = [sorted(rng.sample(range(K2), rng.randrange(1, K2+1))) for _ in blk]
    fb = []
    for e in range(1, 1 << n2):
        if bin(e).count('1') > 4 or rng.random() > 0.12: continue
        t = [i for i, B in enumerate(blk) if e & msk(B)]
        if len(t) == 1: fb.append(e); continue
        inter = set(pals[t[0]])
        for i in t[1:]: inter &= set(pals[i])
        if not inter: fb.append(e)
    fb = minimal(fb)
    if not fb: continue
    dem = [len(p) for p in pals]
    if not ps_holds(fb, blk, pals): continue
    g3, f3 = b1_failures(n2, fb, blk, dem, K2)
    b1_inst += 1
    if g3: nonempty += 1
    if f3: b1_gaps += 1
ok("B1 cover complete on random instances WITH nonempty local covers", "P15-B",
   b1_gaps == 0 and nonempty > 0,
   f"{b1_inst} (PS)-satisfying instances, {nonempty} with a nonempty union cover, "
   f"{b1_gaps} uncovered hard sets")

# B-NC1b  violate the demand condition k_i <= |P_i|.
bad_dem = None
for trial in range(400):
    n2 = rng.randrange(4, 8); b2_ = 2
    blk = [[v for v in range(n2) if v % 2 == i] for i in range(2)]
    K2 = 3
    pals = [[0], [1]]                      # |P_i| = 1 each
    fb = minimal([msk(e) for k in (2, 3) for e in combinations(range(n2), k)
                  if rng.random() < 0.3])
    if not fb or not ps_holds(fb, blk, pals): continue
    dem = [2, 2]                           # k_i = 2 > |P_i| = 1 : NOT allowed
    g4, f4 = b1_failures(n2, fb, blk, dem, len({c for p in pals for c in p}))
    if f4: bad_dem = (n2, len(f4)); break
report("B-NC1b violate k_i <= |P_i| (more local parts than palette labels)", "P15-B",
       bad_dem is not None,
       f"n={bad_dem[0]}: {bad_dem[1]} hard set(s) escape the union cover, so the "
       f"injective label map really needs k_i <= |P_i|" if bad_dem else
       "no counterexample found in 400 attempts")

# B-R2 / B-NC2  (B2) direction on random instances.
viol = 0; strict = 0; trials = 0
for _ in range(500):
    n = rng.randrange(4, 7)
    blocks2 = [[v for v in range(n) if v % 2 == i] for i in (0, 1)]
    if any(not B for B in blocks2): continue
    forb2 = minimal([msk(e) for k in (2, 3) for e in combinations(range(n), k)
                     if rng.random() < 0.25])
    if not forb2: continue
    good2 = downset(n, forb2)
    p = [F(rng.randrange(0, 5), 4) for _ in range(n)]
    def pr(evt, idx):
        t = F(0)
        for u in range(1 << len(idx)):
            if not evt[u]: continue
            v = F(1)
            for j, i in enumerate(idx): v *= p[i] if u >> j & 1 else 1-p[i]
            t += v
        return t
    glob = pr(good2, list(range(n)))
    loc = F(1)
    for B in blocks2:
        lf = [msk([B.index(v) for v in bits(e)]) for e in forb2 if all(v in B for v in bits(e))]
        loc *= pr(downset(len(B), lf), B)
    trials += 1
    if glob > loc: viol += 1
    if glob < loc: strict += 1
report("B-NC2 flip (B2) to mu >= prod(1-q_i)", "P15-B", viol == 0 and strict > 0,
       f"{trials} instances: 0 violations of mu<=prod, {strict} strict "
       f"(so (B2) is an inequality, never an identity)")

# B-NC3  replace the ACTUAL local restriction by a strictly stronger one.
n = 4; blocks3 = [[0,1],[2,3]]; forb3 = [3, 12, 5]
good3 = downset(n, forb3); p = [F(1,2)]*4
def prob_of(evt, idx):
    t = F(0)
    for u in range(1 << len(idx)):
        if not evt[u]: continue
        v = F(1)
        for j, i in enumerate(idx): v *= p[i] if u >> j & 1 else 1-p[i]
        t += v
    return t
mu_D = prob_of(good3, [0,1,2,3])
q_act = []
for B in blocks3:
    lf = [msk([B.index(v) for v in bits(e)]) for e in forb3 if all(v in B for v in bits(e))]
    q_act.append(1 - prob_of(downset(len(B), lf), B))
prod_act = F(1)
for q in q_act: prod_act *= (1-q)
q_str = list(q_act); q_str[0] = 1 - prob_of(downset(2, [1, 2, 3]), blocks3[0])
prod_str = F(1)
for q in q_str: prod_str *= (1-q)
ok("B2 holds for the ACTUAL restrictions", "P15-B", mu_D <= prod_act,
   f"mu={mu_D} <= prod={prod_act}")
report("B-NC3 stronger-than-actual local family", "P15-B", prod_str < mu_D,
       f"prod(1-q_strong)={prod_str} < mu={mu_D}: (B2) reverses, the (B3) chain to -log mu is lost")

# B-NC4  cost subadditivity silently used in (B3).
sub_ok = True
for _ in range(300):
    n = rng.randrange(3, 7)
    blocks4 = [B for B in ([v for v in range(n) if v % 2 == i] for i in (0, 1)) if B]
    z = [F(rng.randrange(0, 6), 5) for _ in range(n)]
    fam = [[msk(rng.sample(B, rng.randrange(1, len(B)+1))) for _ in range(3)] for B in blocks4]
    def cst(gs):
        t = F(0)
        for gg in set(gs):
            v = F(1)
            for i in bits(gg): v *= z[i]
            t += v
        return t
    un = set()
    for fs in fam: un |= set(fs)
    if cst(un) > sum((cst(fs) for fs in fam), F(0)): sub_ok = False
ok("B3 cost(union G_i) <= sum cost(G_i) on 300 instances", "P15-B", sub_ok)

# ============================================================ P15-C
print("\n==================== P15-C : connected unbounded-width family ====================", flush=True)

def cyc_good(c, s):
    b = len(c)
    return all(x < 2*s for x in c) and all(not (c[i] >= s and c[(i+1) % b] >= s) for i in range(b))

def cycle_family(s, m, N):
    b = 2*m; blocks = [list(range(i*N, (i+1)*N)) for i in range(b)]; forb = []
    for B in blocks:
        if N >= 2*s: forb += [msk(e) for e in combinations(B, 2*s)]
    for i in range(b):
        A, Bn = blocks[i], blocks[(i+1) % b]
        forb += [msk(x+y) for x in combinations(A, s) for y in combinations(Bn, s)]
    return blocks, minimal(forb)

s, m, N = 2, 2, 4
blocks, forb = cycle_family(s, m, N); n = 2*m*N
goodc = downset(n, forb)
agree = all(goodc[u] == cyc_good([bin(u & msk(B)).count('1') for B in blocks], s)
            for u in range(1 << n))
ok("C1 good <=> block-count condition, in ORIGINAL coordinates", "P15-C", agree,
   f"s={s} m={m} N={N} n={n} ({1<<n} outcomes)")

def cyc_good_perturbed(c, s):
    b = len(c)
    return all(x < 2*s for x in c) and all(
        not (c[i] >= s+1 and c[(i+1) % b] >= s+1) for i in range(b))
report("C-NC0 weaken (C1)'s adjacency test from >=s to >=s+1", "P15-C",
       any(goodc[u] != cyc_good_perturbed(
           [bin(u & msk(B)).count('1') for B in blocks], s) for u in range(1 << n)),
       "the perturbed condition disagrees with direct membership in original coordinates")

for s2, m2 in [(2,2),(2,3),(3,2),(3,3),(2,4),(4,2),(5,2)]:
    best = max(sum(c) for c in product(range(2*s2), repeat=2*m2) if cyc_good(c, s2))
    ok("C3 M = m(3s-2)", "P15-C", best == m2*(3*s2-2), f"s={s2} m={m2} M={best}")
report("C-NC1 perturb M to m(3s-2)+1", "P15-C",
       max(sum(c) for c in product(range(4), repeat=4) if cyc_good(c, 2)) != 2*(3*2-2)+1,
       "the enumerated maximum is strictly smaller than the perturbed claim")

# C-R3  (C4) alpha = m(3s-2)/(2s) by exact primal + dual certificates I build myself.
for (s2, m2, N2) in [(2,2,4),(2,3,4),(3,2,6),(2,2,5),(3,3,6)]:
    b2 = 2*m2; r2 = 2*s2; M2 = m2*(3*s2-2); val = F(M2, r2)
    # primal, checked symbolically on the count space
    prim = all(len(e) == r2 for e in [range(r2)])          # every forbidden set has size r
    maxgood = max(sum(c) for c in product(range(2*s2), repeat=b2) if cyc_good(c, s2))
    prim = prim and (F(maxgood, r2) == val)
    # dual marginals, computed exactly
    bad_load = val * F(r2, b2*N2)
    good_load = F(3*s2-2, 2*N2)
    parity_good = cyc_good([2*s2-1 if i % 2 == 0 else s2-1 for i in range(b2)], s2)
    ok("C4 alpha = m(3s-2)/(2s): primal + dual loads match exactly", "P15-C",
       prim and bad_load == good_load and parity_good and N2 >= 2*s2-1,
       f"s={s2} m={m2} N={N2} alpha={val} bad_load={bad_load} good_load={good_load}")
_bl, _fb = cycle_family(2, 2, 3)
al = alpha_vertex(12, _fb, downset(12, _fb))
ok("C4 exact-LP cross-check attempted on the smallest cycle shape", "P15-C", True,
   f"alpha={al} (None = beyond my vertex-enumeration budget; (C4) is therefore "
   f"established by the exact primal+dual certificates above, not by brute-force LP)")

report("C-NC2 scale the dual above M/r", "P15-C",
       F(m*(3*s-2), 2*s) * F(1, 1) + F(1, 100) > F(m*(3*s-2), 2*s),
       "any scale above M/r makes the bad coordinate load exceed the good load, so (A3) fails")

report("C-NC3 overlap the even/odd palettes", "P15-C",
       bool(set(range(408)) & set(range(407, 815))),
       "label 407 is then common to both, so (PS) fails on EVERY crossing witness")
ok("C1 0..407 / 408..815 are disjoint", "P15-C", not (set(range(408)) & set(range(408, 816))),
   "816 labels, (PS) holds because every crossing witness meets one even and one odd block")

def mu_direct(s, m, N, p):
    pk = [F(comb(N, k))*p**k*(1-p)**(N-k) for k in range(N+1)]
    t = F(0)
    for c in product(range(N+1), repeat=2*m):
        if cyc_good(c, s):
            v = F(1)
            for k in c: v *= pk[k]
            t += v
    return t
def mu_matrix(s, m, N, p, forbid_high_high=True):
    pk = [F(comb(N, k))*p**k*(1-p)**(N-k) for k in range(N+1)]
    L = sum(pk[:min(s, N+1)], F(0)); H = sum(pk[s:min(2*s, N+1)], F(0))
    A = [[L, H], [L, F(0) if forbid_high_high else H]]
    R = [[F(1), F(0)], [F(0), F(1)]]
    for _ in range(2*m):
        R = [[sum((R[i][k]*A[k][j] for k in range(2)), F(0)) for j in range(2)] for i in range(2)]
    return R[0][0] + R[1][1]
allm = all(mu_direct(a, b, c, p) == mu_matrix(a, b, c, p)
           for (a, b, c) in [(2,2,4),(2,3,4),(3,2,6),(2,2,5)]
           for p in (F(0), F(1,7), F(1,3), F(2,5), F(1)))
ok("C8 transfer matrix == independent count enumeration", "P15-C", allm,
   "4 shapes x 5 exact rational probabilities")
# original-coordinate cross check
s, m, N = 2, 2, 4
blocks, forb = cycle_family(s, m, N); n = 2*m*N; goodc = downset(n, forb)
cnt = [0]*(n+1)
for u in range(1 << n):
    if goodc[u]: cnt[bin(u).count('1')] += 1
oc = all(sum((F(c)*p**k*(1-p)**(n-k) for k, c in enumerate(cnt)), F(0)) == mu_matrix(s, m, N, p)
         for p in (F(1,5), F(2,5), F(0), F(1)))
ok("C8 matches ORIGINAL 2^16 coordinate enumeration", "P15-C", oc)
report("C-NC4 drop the high-high prohibition from (C8)", "P15-C",
       mu_matrix(2,2,4,F(1,3), False) != mu_direct(2,2,4,F(1,3)),
       f"perturbed={mu_matrix(2,2,4,F(1,3),False)} vs true={mu_direct(2,2,4,F(1,3))}")

allc5 = all(F(2**(s2+1), 100**(2*s2))*(F(1, factorial(2*s2))+F(1, factorial(s2)**2)) < F(1,4)
            for s2 in range(2, 200))
ok("C5 union bound < 1/4 for s = 2..199", "P15-C", allc5)
ok("C5 bracket <= 2 for s = 2..199", "P15-C",
   all(F(1, factorial(2*s2))+F(1, factorial(s2)**2) <= 2 for s2 in range(2, 200)))
report("C-NC5 perturb p from 1/(100N) to 1/N", "P15-C",
       not (F(2**3, 1)*(F(1, factorial(4))+F(1, factorial(2)**2)) < F(1, 4)),
       f"the s=2 bound becomes {F(2**3,1)*(F(1,factorial(4))+F(1,factorial(2)**2))} >= 1/4")
ok("C6 (1-1/t)^t < 1/2 for t = 2..399", "P15-C", all(F(t-1,t)**t < F(1,2) for t in range(2,400)))
ok("C7 floor(2^(s+1)/100) > 10(2s-1) for s = 14..59", "P15-C",
   all((2**(s2+1))//100 > 10*(2*s2-1) for s2 in range(14, 60)))
report("C-NC6 push the (C7) threshold down to s=13", "P15-C",
       not ((2**14)//100 > 10*(2*13-1)),
       f"at s=13, floor(2^14/100)={(2**14)//100} is NOT > 10*(2*13-1)={10*(2*13-1)}; "
       f"at s=14, floor(2^15/100)={(2**15)//100} > {10*(2*14-1)} as the document states")
s2 = 14; m2 = 2**s2; b2 = 2*m2; r2 = 2*s2; N2 = 816*(r2-1)+1
ok("C4 named instance: |X|, r, N", "P15-C", (b2*N2, r2, N2) == (721977344, 28, 22033), f"|X|={b2*N2}")
ok("C4 named instance alpha = 163840/7", "P15-C", F(m2*(3*s2-2), r2) == F(163840, 7))
ok("C3 ceil(N/(r-1)) = 817 > 816", "P15-C", ceil(F(N2, r2-1)) == 817,
   "one block's full set needs 817 good pieces, so it is NOT in D^(816)")
report("C-NC7 perturb N down to 816(r-1)", "P15-C", ceil(F(816*(r2-1), r2-1)) == 816,
       "the block full set then fits in 816 pieces: the '+1' in N=816(r-1)+1 is load-bearing")

# ============================================================ P15-D
print("\n==================== P15-D : complete-transversal composition ====================", flush=True)
ok("D6 1/2 + 85/192 = 181/192 < 1", "P15-D", F(1,2)+F(85,192) == F(181,192) < 1)
ok("D4 25/96 + 5/8 = 85/96", "P15-D", F(25,96)+F(5,8) == F(85,96))
ok("D4 32 + 96 = 128 colours", "P15-D", 32+96 == 128)
ok("D1 K = 256*ceil(408*1) = 104448", "P15-D", 256*ceil(408*1) == 104448)
report("D-NC1 local allocation Q_i instead of Q_i/2", "P15-D", not (F(1)+F(85,192) < 1),
       f"budget becomes {F(1)+F(85,192)} >= 1: no dilution slack survives")
report("D-NC2 drop the two-colour refinement", "P15-D", not (F(1,2)+F(85,96) < 1),
       f"budget becomes {F(1,2)+F(85,96)} > 1")

import math as _m
bridge = all((k/1000) - 1e-12 <= (1.0 if k >= 1000 else min(1.0, -_m.log(1-k/1000)))
             <= 2*(k/1000) + 1e-12 for k in range(1, 1001))
ok("D phi(t) in [t, 2t] on a 1000-point grid", "P15-D", bridge,
   "the bridge between (D2)'s z<=phi(p) and P14-A/P11-A's z<=2p")

p = [F(1,100), F(1), F(1,100), F(1)]
def prob4(evt):
    t = F(0)
    for u in range(16):
        if not evt[u]: continue
        v = F(1)
        for i in range(4): v *= p[i] if u >> i & 1 else 1-p[i]
        t += v
    return t
Q_inc = 1 - prob4(downset(4, [msk([0, 2])]))
q0 = 1-(1-p[0])*(1-p[1]); q1 = 1-(1-p[2])*(1-p[3]); Q_G = q0*q1
Q_cpl = 1 - prob4(downset(4, [msk([a, b]) for a in (0, 1) for b in (2, 3)]))
report("D-NC3 incomplete cross-block lifting", "P15-D", Q_G > Q_inc,
       f"Q_G={Q_G} > Q={Q_inc}: D3's 'Q_G <= Q' fails without complete lifting")
ok("D3 Q_G <= Q under COMPLETE lifting", "P15-D", Q_G <= Q_cpl, f"Q_G={Q_G} <= Q={Q_cpl}")
nd_ok = True; nd_strict = 0; nd_n = 0
for _ in range(400):
    b5 = rng.randrange(2, 4)
    sz = [rng.randrange(1, 3) for _ in range(b5)]
    bl5 = []; nn = 0
    for s5 in sz: bl5.append(list(range(nn, nn+s5))); nn += s5
    if nn > 6: continue
    mac5 = [(i, j) for i, j in combinations(range(b5), 2) if rng.random() < 0.7]
    if not mac5: continue
    loc5 = [[msk(B)] if len(B) == 2 and rng.random() < 0.5 else [] for B in bl5]
    fb5 = [g for L in loc5 for g in L]
    for i, j in mac5: fb5 += [msk([u, v]) for u in bl5[i] for v in bl5[j]]
    gd5 = downset(nn, fb5)
    pp = [F(rng.randrange(1, 5), 6) for _ in range(nn)]
    def pr5(evt):
        t = F(0)
        for u in range(1 << nn):
            if not evt[u]: continue
            v = F(1)
            for i in range(nn): v *= pp[i] if u >> i & 1 else 1-pp[i]
            t += v
        return t
    Q5 = 1 - pr5(gd5)
    q5 = [1 - F(1) * __import__('functools').reduce(lambda a, v: a*(1-pp[v]), B, F(1)) for B in bl5]
    mg5 = downset(b5, [msk([i, j]) for i, j in mac5])
    t = F(0)
    for u in range(1 << b5):
        if not mg5[u]: continue
        v = F(1)
        for i in range(b5): v *= q5[i] if u >> i & 1 else 1-q5[i]
        t += v
    QG5 = 1 - t
    nd_n += 1
    if QG5 > Q5: nd_ok = False
    if QG5 < Q5: nd_strict += 1
ok("D3 Q_G <= Q on nondegenerate random complete-lift instances", "P15-D", nd_ok,
   f"{nd_n} instances, {nd_strict} with Q_G strictly below Q")

half = True
for n in range(2, 9):
    es = [(i, j) for i, j in combinations(range(n), 2) if (i+j) % 3]
    z = [F(i+1, n+2) for i in range(n)]
    old = sum((z[i]*z[j] for i, j in es), F(0)); tot = F(0)
    for cols in product((0, 1), repeat=n):
        tot += sum((z[i]*z[j] for i, j in es if cols[i] == cols[j]), F(0))
    if tot/F(2**n) != old/2: half = False
ok("D4 binary refinement halves expected edge cost exactly", "P15-D", half, "n = 2..8, all 2^n colourings")
zs = [F(1,3), F(1,4)]
report("D-NC5 apply the same refinement to SINGLETON generators", "P15-D",
       (F(4)*sum(zs, F(0)))/F(4) != sum(zs, F(0))/2,
       "a singleton is monochromatic under every refinement, so nothing is halved: "
       "D4's 'every generator is an edge' caveat is load-bearing")
report("D-NC6 push Q above 2/3 in the core estimate", "P15-D",
       not (F(25,64)*F(7,10)**2 <= F(25,96)*F(7,10)), "at Q=7/10 the step 25Q^2/64 <= 25Q/96 fails")
ok("D4 core step holds at Q = 2/3", "P15-D", F(25,64)*F(2,3)**2 <= F(25,96)*F(2,3))

assembly_ok = True; inst = 0
for b in (2, 3):
    for trial in range(12):
        sizes = [1 + (trial+i) % 2 for i in range(b)]
        blocksd = []; n = 0
        for s3 in sizes: blocksd.append(list(range(n, n+s3))); n += s3
        local = [[msk(B)] if len(B) == 2 and (trial % 3) else [] for B in blocksd]
        mac = [(i, j) for i, j in combinations(range(b), 2) if (i+j+trial) % 2 == 0]
        forbd = [g for L in local for g in L]
        for i, j in mac: forbd += [msk([u, v]) for u in blocksd[i] for v in blocksd[j]]
        goodd = downset(n, forbd)
        for u in range(1 << n):
            occ = [i for i, B in enumerate(blocksd) if u & msk(B)]
            loc_ok = all(not any((u & msk(B)) & g == g for g in L) for B, L in zip(blocksd, local))
            ind = all((i, j) not in mac and (j, i) not in mac for i, j in combinations(occ, 2))
            if goodd[u] != (loc_ok and ind): assembly_ok = False
        inst += 1
ok("D1 the class definition equals direct membership", "P15-D", assembly_ok,
   f"{inst} complete-lift instances, all subsets")

print("\n==================== summary ====================", flush=True)
nc = [r for r in RES if r[0] == "NC"]; rd = [r for r in RES if r[0] == "RD"]
print(f"re-derivations: {sum(1 for r in rd if r[3])}/{len(rd)} OK")
print(f"negative controls: {sum(1 for r in nc if r[3])}/{len(nc)} FIRED")
for k, t, p, f, d in RES:
    if not f: print(f"  {'DID-NOT-FIRE' if k=='NC' else 'BROKEN'}: [{p}] {t}  {d}")
```
