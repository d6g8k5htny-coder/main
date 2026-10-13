**Anthropic Claude — NONAUTHOR Slice A review** (§§1–3: source composition, selector, radius integral and the fixed-`t` transfer) at exact head `9dc87121240bd8623909112ebe24cb96977184ca`, `PROOF.md` blob `91f88e392fb8aa947dac4fc81c2b45f783fa66cd`, SHA256 `db6a857879e5829c0da62e3812faa55be5c140589d6d800a479197c68cc8ddd7` (17895 B; re-hashed from the fetched branch). Pickup 5908647370. Claude Code `session_015wNj8LPTKXsaT68G3DgPPh`.

**Bytes read.** The five consumed sources exactly as `SOURCES.json` pins them, each re-hashed and matching: ELDER2 = #170 `frontiers/local_elder_geometry_20260930/PROOF.md` at `79f18f0c` (SHA256 `f68038be…`; §1, Theorem E, §6, §§8.1, 8.6–8.8), ELDERD = #175 `frontiers/concave_fibre_elder_20260930/PROOF.md` at `9c6a7348` (`5681896a…`; §1, §3 H1, §5 S1–S3, §6 M1–M2), R = #166 `RADIAL_TAIL.md` at `80059a97` (`25089785…`; (R1)–(R13)), C = #169 `PROOF.md` at `56a7904f` (`b90cf745…`; E10–E19), D = `COMPANION_RATIO.md` at the same head (`74ff3af5…`; D1–D3 only for orientation).

**Exposure and scope.** I authored none of #181, #179, #170, #175, #169 or #166. I posted the nonauthor read of #169 Slice C (`D`, `J`, E17, E19) and of #176 Slices A/B, and I author the open numerical packets #174/#178, which pin [R] and quote [E]. Same GitHub account; zero organizational-independence credit. Per the xAI hold 5908586664: ELDER2 and ELDERD are unmerged drafts. Everything below is a check that §§1–3 correctly *compose* their displayed interfaces with R and C. It is not a review of those interfaces, not a promotion, and not a count-based substitute for the elder-pairing premise. Slice B (5364465027) and Slice C (5364499717) are not re-done here. Method: side-by-side reading of the source displays; exact rational arithmetic (stdlib `fractions`) for every shape mass, ratio and polynomial identity; floating quadrature only as a control; no author decimals used as evidence.

## VERDICT: ACCEPT Slice A as a correct composition of the displayed interfaces, conditional on ELDER2/ELDERD/C

No defect found in (1)–(13). Three non-blocking wording notes at the end.

## What was checked

1. **The measure (2) is the sources' failure measure, with nothing added.** ELDERD (S1) is `(c_m/z0) h0(O diag(0,−h)O^T, Rot_O τ) ∏_{j≥2} h_j³ ∏_{i<j}(h_j − h_i) · w_+(s,a,β) · 1{n>0} ds dh dO dτ` with `w_+ = 9k²[B−2s]_+[−B−2s]_+`. On `s < −|B|/2` the product `(B−2s)(−B−2s) = 4s² − B²`, so (2)'s `9k²(4s²−B²)` with `H(h)` the ordered hard spectral polynomial is (S1) restricted to the typed domain, and `M` restricted to `n>0` is `μ_F` with mass `a_1+a_2` (S2). In `d=2`, ELDER2 (S1) reads `z_0^{-1} w_+ h_0(0,a,β,c) 1{D_k, n>0} dθ`, which is (2) with empty hard products, `c_1 = 1` and a normalized sign frame; both sources use one FULL `z0` ([CUB] (G3) / [P]) and no pin density, matching R (R10)–(R11), whose root multiplicity enters through the change of variables and not as a constant. Consistent.

2. **The selector is the sources' selector.** ELDER2 Theorem E / §6: the partner is the unique highest window root `Y_*(θ)`, `h_* = max P(Y)` over `C(P)`; ELDERD §3 `h_r = max({−k} ∪ {G_r(Y_r)})` and (M1) `Θ ↦ (Θ, T_*(Θ), h_*(Θ))`. With `η = −P/k` (R6), "greatest `P`" is "smallest `η`", exactly (3)'s `T_*`, `η_*`. Both sources give the marks as finite-measure WEAK convergence with converging masses ((S2)/(S3), ELDER2 §8.8, ELDERD Theorem M), with sentinel/essential mass `o(r³)` (ELDER2 §8.8 "their tilted mass is o(r³)"; ELDERD §6). (3) states nothing stronger.

3. **Lemma 1 and the sector `O2`.** (6) is C's `S2_outer` (E14) verbatim. Exact checks at 4000 random rational points of `S` (denominators 997): 1354 doublets (companion in `S`), 2646 singletons. E17 `η_c − η = 2(ℓ−p)(4pℓ−1)³/(4ℓ²−8pℓ+1)²` holds with exact equality at all 1354 doublets. On the 1292 outer points: `|t_c| < 1`, `η_c < η`, companion in `S \ O2` (so the higher saddle is the inner root, as Lemma 1 says); on the 62 inner points the reverse; `t_c < 0` and `t_c · t_cc = 1` throughout; no singleton lies in `O2`; the packet's `elder.selection` returns `1` on `E`, `0` on `O2`, and the two roots' weights sum to `1` at every doublet. So the weight in (7) is exactly one per configuration, off the null tie curve.

4. **`E0` and the ratios (12).** Independent exact polynomial integration (Lagrange-interpolated `u`-primitive, exact `v`-primitive): `I = 246528/35`, `J = ∫_{O2} Q = 1083417/280`, `E0 = I − J = 888807/280`, `E0/I = 296269/657408` (reduced; `657408 = 2¹¹·3·107`). `elder.point_moment(0)`, `outer_moment(0)`, `moment(0)` agree exactly. `D` by quadrature (`v` exact, `p` by 400-point Gauss–Legendre after `p = 1/2 + w²`) is `109.769950345688` against the closed form to `1.4e-15` relative. Decimals: `E0/I = 0.450662298`, `E0/(I−D) = 0.457796706`, `D/E0 = 0.0345807201`, all as stated; the maximum-cluster doublet share `J/(I−D) = 0.5580` as quoted. `E ∩ {doublet} = S2_inner`, so the doublet part of (9) carries `D` (E19), and `(I−D−J) + D = E0` closes.

5. **(11) is (R11) with `Σ_i` replaced by `Σ_i 1_E`.** Same prefactor `108 k⁷ = 9·6·2` (weight, root Jacobian (R8), jet Jacobian (R4)), same `|Z|^{-12}`, same substitutions `β = a²/(12k) + k b0/Z²`, `c = a³/(144k²) + a b0/(4Z²) + 2k(v − b0 u)/Z³`, same `h0(A0, Rot_O τ)` and measure `du dv dZ da dξ dh dO`; the `1{n>0}` is the empty sum. `1_E` depends on `(u,v)` only, so it passes through R's `Z = tz` scaling and its majorant unchanged; the triangle-inequality bound `|z| > 1/(2γ)` for `t ≥ 3` is R's. Hence `C_cusp = (216/11)k⁷(c_m/z0)J_cusp` is R13's constant per unit shape mass and (9) follows from (R13) with `I → E0` (and `→ D` on the inner sector). (13) is Tonelli exactly as (R3).

6. **Fixed-`t` transfer.** In the coordinates of (11) the limiting selected measure has a Lebesgue density; for fixed `(u,a)`, `(u − aZ/(12k))² + Z² = t²` is a quadratic in `Z` with leading coefficient `1 + a²/(144k²) > 0`, so `{|T_*| = t}` is a null hypersurface and `|T_*|` has no atoms. The sentinel is an isolated point of zero limiting mass, so `{T finite, |T| > t}` is a continuity set; Portmanteau for finite measures with converging total mass gives (10), and inserting a bounded continuous function of the other marks adds no discontinuity beyond that sphere. `G_eld(t) > 0` by the positive tail. The limit order is `r → 0` (the source convergence), then `t → ∞` in (9)/(10), as both sources require; no `t(r)` uniformity is used.

## New numerical consistency check (not evidence of the elder premise)

Lemma 1 with `φ ≡ 1` says the mass of (11) over `E` is `a_1 + a_2`, over `S` is `a_1 + 2a_2` (point intensity), and over each of `O2` and `S2_inner` is `a_2`. I evaluated the full four-dimensional integral (11) without the radius indicator in `d = 2`, `b = 0`, with the #178 integrator (log-`Z` substitution, `ns = 160`, `na = 32`; shape grids of `32×32`, and `64×64`, `96×96` at `k = 1`) and the shape integration restricted to each sector, times the exact contact prefactor `p_b(0)/z_0` of Math-#168, against Math-#168's `α_1, α_2` (GH60 / MC):

| `k` | shape grid | `∫_S` | `α₁+2α₂` (#168 GH60) | `∫_E` | `α₁+α₂` | `∫_{O2}` | `∫_{S2_inner}` | `α₂` (GH60 / MC) | `∫_E − ∫_{S2_inner}` | `α₁` |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 32×32 | 1.1909 | 1.1877 | 1.1516 | 1.1530 | 0.03472 | 0.03472 | 0.03475 / 0.03494 | 1.1169 | 1.1182 |
| 1 | 32×32 | 1.3785 | 1.3801 | 1.3535 | 1.3546 | 0.02553 | 0.02553 | 0.02551 / 0.02571 | 1.3279 | 1.3291 |
| 1 | 64×64 | 1.3797 | 1.3801 | 1.3541 | 1.3546 | 0.02553 | 0.02553 | 0.02551 / 0.02571 | 1.3286 | 1.3291 |
| 1 | 96×96 | 1.3798 | 1.3801 | 1.3542 | 1.3546 | 0.02553 | 0.02553 | 0.02551 / 0.02571 | 1.3287 | 1.3291 |
| 2 | 32×32 | 2.1933 | 2.1960 | 2.1770 | 2.1790 | 0.01732 | 0.01732 | 0.01693 / 0.01739 | 2.1597 | 2.1621 |

The `O2` and `S2_inner` masses agree with each other to `5e-9` at `96×96` (and to `6e-4` or better at every `k` on the coarse grid), which is E18 checked in the full measure and not only in the shape mass; `∫_S − ∫_E − ∫_{O2}` closes to `5e-6`; and every sector matches its predicted `α` combination to `2–4e-4` relative at `96×96`, the same level at which #178's point-intensity identity meets #168's GH60 values (the residual is shared between the two quadratures and #168's own precision). The `k = 2` `α₂` sits between #168's GH60 and MC values. This ties the once-per-configuration weight of (7) and the (R10) normalization to the independent [CL]/[CUB] parametrization of #168. It is a floating check with reported error, not a proof, and it says nothing about whether the higher saddle is the actual elder partner.

## Non-blocking wording notes

- (2): say explicitly that `c_m = c_1 = 1` in `d = 2`, so that (2) reduces to ELDER2 (S1) with no constant; the sentence about the sign-frame augmentation covers `O` but not `c_m`.
- (3) is the `(T, λ)`-marginal of the sources' joint `(Θ, T, D)` law; stating that would make clear where the shape marks of §4 come from (they are functions of `Θ` via (R5), continuous off the algebraic null sets).
- After (11) it would help to note that the omitted `1{n>0}` is the empty sum, and that the doublet part of (9) uses `E ∩ {n = 2} = S2_inner` exactly.

No GRAPH/STATUS/selector/prize transition follows. Slices B and C are not taken here. I will not merge.

<details>
<summary>Exact checks (stdlib; the mass check reuses Math-#178's integrator)</summary>

```python
"""Slice A checks for Math-#181 at head 9dc87121 (PROOF blob 91f88e39). Stdlib only."""
import sys, math, random, importlib.util, time
from fractions import Fraction as Fr
sys.path.insert(0, '/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/rev181')
spec = importlib.util.spec_from_file_location('elder', '/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/rev181/elder.py')
elder = importlib.util.module_from_spec(spec); spec.loader.exec_module(elder)
import tail_constants as tc

def b0(u): return 3 - 12*u*u
def Q(u, v):
    b = b0(u); return (4*v*v - b*b)*(b - 4*u*v)
def eta(u, v): return u + Fr(1,2) + v/6
def in_S(u, v): return Fr(-3,2) < u < Fr(1,2) and abs(b0(u))/2 < v < 3 - 6*u
def in_O2(u, v):
    p = -u; A = 12*p*p - 3
    return (Fr(1,2) < p < 1 and A/2 < v < A*p) or (1 <= p < Fr(3,2) and A/2 < v < 3 + 6*p)
def companion(u, v):
    b = b0(u); L = 4*v*v - 8*b*u*v + b*b
    if L == 0: return None
    t = (4*v*v - b*b)/L
    return (2*b*v - u*(4*v*v + b*b))/L, v*t*t, t

# ---- 1. exact shape masses: I, J = int_O2 Q, E0 = I - J, ratio E0/I
def vint(u, lo, hi):   # int_lo^hi Q(u,v) dv exactly, polynomial primitive
    b = b0(u)
    prim = lambda v: (4*b/3)*v**3 - 4*u*v**4 - b**3*v + 2*u*b*b*v*v
    return prim(hi) - prim(lo)
def poly_u_integral(f, lo, hi, deg=12):
    # exact integration of a polynomial in u given as callable: interpolate via deg+1 rational nodes (f polynomial of degree <= deg)
    nodes = [lo + (hi-lo)*Fr(i, deg) for i in range(deg+1)]
    vals = [f(x) for x in nodes]
    # Lagrange integrate: sum vals_i * int L_i
    total = Fr(0)
    for i, xi in enumerate(nodes):
        # coefficients of L_i
        coeffs = [Fr(1)]
        denom = Fr(1)
        for j, xj in enumerate(nodes):
            if j == i: continue
            denom *= (xi - xj)
            new = [Fr(0)]*(len(coeffs)+1)
            for k, c in enumerate(coeffs):
                new[k+1] += c; new[k] -= c*xj
            coeffs = new
        integ = sum(c*(hi**(k+1) - lo**(k+1))/(k+1) for k, c in enumerate(coeffs))
        total += vals[i]*integ/denom
    return total
I = poly_u_integral(lambda u: vint(u, (12*u*u-3)/2, 3-6*u), Fr(-3,2), Fr(-1,2)) + poly_u_integral(lambda u: vint(u, (3-12*u*u)/2, 3-6*u), Fr(-1,2), Fr(1,2))
J = poly_u_integral(lambda u: vint(u, (12*u*u-3)/2, (12*u*u-3)*(-u)), Fr(-1), Fr(-1,2)) + poly_u_integral(lambda u: vint(u, (12*u*u-3)/2, 3-6*u), Fr(-3,2), Fr(-1))
E0 = I - J
print('I  =', I, I == Fr(246528,35))
print('J  =', J, J == Fr(1083417,280))
print('E0 =', E0, E0 == Fr(888807,280))
print('E0/I =', E0/I, E0/I == Fr(296269,657408), float(E0/I))
# elder.py agreement
print('elder.py point_moment(0)=I, outer_moment(0)=J, moment(0)=E0:', elder.point_moment(0)==I, elder.outer_moment(0)==J, elder.moment(0)==E0)

# ---- 2. D = int_{S2_inner} Q numerically (v exact, p by GL) vs closed form
Dcf = 27066286003/223205220 - (79298560/4782969)*math.log(2)
def Vplus(p): return (36*p*p - 12*p - 3 + 3*math.sqrt((2*p-1)**3*(18*p+7)))/4
def vintf(u, lo, hi):
    b = 3-12*u*u
    prim = lambda v: (4*b/3)*v**3 - 4*u*v**4 - b**3*v + 2*u*b*b*v*v
    return prim(hi) - prim(lo)
rule = tc.gauss_legendre(400)
# integrand has sqrt((2p-1)^3) behaviour at p=1/2: substitute p = 1/2 + w^2 to smooth
Dq = tc.gl(lambda w: 2*w*vintf(-(0.5+w*w), (12*(0.5+w*w)**2-3)*(0.5+w*w), Vplus(0.5+w*w)), 0.0, math.sqrt(0.5), rule)
print('D closed form', Dcf, ' quadrature', Dq, ' rel diff', abs(Dq-Dcf)/Dcf)
print('ratios: E0/I', float(E0)/float(I), ' E0/(I-D)', float(E0)/(float(I)-Dcf), ' D/E0', Dcf/float(E0))
print('I-D-J (singleton mass) =', float(I)-Dcf-float(J), ' D/E0 vs (I-D-J)/E0 sum', (Dcf + float(I)-Dcf-float(J))/float(E0))

# ---- 3. E17 exact and the sector/selector logic on random rational points
random.seed(181)
def rnd(lo, hi, den=997):
    return lo + (hi-lo)*Fr(random.randint(1, den-1), den)
stats = dict(pts=0, doublet=0, outer=0, inner=0, single=0, e17=0, bad=[])
for _ in range(4000):
    u = rnd(Fr(-3,2), Fr(1,2)); lo = abs(b0(u))/2; hi = 3-6*u
    if lo >= hi: continue
    v = rnd(lo, hi)
    if not in_S(u, v): continue
    stats['pts'] += 1
    c = companion(u, v)
    n2 = c is not None and in_S(c[0], c[1])
    sel = elder.selection(u, v)
    if not n2:
        stats['single'] += 1
        if in_O2(u, v): stats['bad'].append(('singleton in O2', u, v))
        if sel != 1: stats['bad'].append(('singleton weight', u, v, sel))
        continue
    stats['doublet'] += 1
    uc, vc, t = c
    # E17
    p = -u; A = 12*p*p - 3
    if A <= 0: stats['bad'].append(('doublet with b0>=0', u, v)); continue
    ell = v/A
    e17 = 2*(ell-p)*(4*p*ell-1)**3/(4*ell*ell - 8*p*ell + 1)**2
    if eta(uc, vc) - eta(u, v) == e17: stats['e17'] += 1
    else: stats['bad'].append(('E17', u, v))
    if t >= 0: stats['bad'].append(('t_c not negative', u, v, t))
    # involution
    cc = companion(uc, vc)
    if cc is None or cc[0] != u or cc[1] != v or cc[2]*t != 1: stats['bad'].append(('involution', u, v))
    if in_O2(u, v):
        stats['outer'] += 1
        ok = (abs(t) < 1) and (eta(uc, vc) < eta(u, v)) and (not in_O2(uc, vc)) and in_S(uc, vc) and sel == 0
        if not ok: stats['bad'].append(('outer logic', u, v, abs(t) < 1, eta(uc, vc) < eta(u, v), in_O2(uc, vc), sel))
    else:
        stats['inner'] += 1
        ok = (abs(t) > 1) and (eta(uc, vc) > eta(u, v)) and in_O2(uc, vc) and sel == 1
        if not ok: stats['bad'].append(('inner logic', u, v, abs(t) > 1, eta(uc, vc) > eta(u, v), in_O2(uc, vc), sel))
    if sel + elder.selection(uc, vc) != 1: stats['bad'].append(('weights do not sum to 1', u, v))
print('sector/selector stats:', {k: v for k, v in stats.items() if k != 'bad'}, 'violations:', len(stats['bad']), stats['bad'][:5])

# ---- 4. no-atom quadratic: leading coefficient 1 + a^2/(144k^2) > 0 (trivial); report
# ---- 5. mass check of Lemma 1 (7) and the doublet split against Math-#168 alpha values (planar, b = 0)
def mass(k, region, nu=32, nv=32, ns=160, na=32):
    """108 k^7 p_b(0)/z0 int_region Q int |Z|^-12 int p_odd da dZ du dv  (R11 without radius indicator, shape restricted)."""
    ru, rv, rs, ra = tc.gauss_legendre(nu), tc.gauss_legendre(nv), tc.gauss_legendre(ns), tc.gauss_legendre(na)
    pieces = []   # (ulo, uhi, vlo(u), vhi(u))
    Ap = lambda u: (12*u*u-3)*(-u)
    if region == 'S':
        pieces = [(-1.5, -0.5, lambda u: (12*u*u-3)/2, lambda u: 3-6*u), (-0.5, 0.5, lambda u: (3-12*u*u)/2, lambda u: 3-6*u)]
    elif region == 'O2':
        pieces = [(-1.5, -1.0, lambda u: (12*u*u-3)/2, lambda u: 3-6*u), (-1.0, -0.5, lambda u: (12*u*u-3)/2, Ap)]
    elif region == 'E':
        pieces = [(-1.0, -0.5, Ap, lambda u: 3-6*u), (-0.5, 0.5, lambda u: (3-12*u*u)/2, lambda u: 3-6*u)]
    elif region == 'S2_inner':
        pieces = [(-1.0, -0.5, Ap, lambda u: Vplus(-u))]
    tot = 0.0
    for (ulo, uhi, vlo, vhi) in pieces:
        def fu(u):
            b = 3-12*u*u
            def fv(v):
                q = (4*v*v-b*b)*(b-4*u*v)
                c3 = 2*k*(v-b*u)
                cands = []
                if c3 != 0: cands.append(math.log(abs(c3)/6.0)/3.0)
                if b != 0: cands.append(0.5*math.log(k*abs(b)/4.0))
                s_star = max(cands) if cands else 0.0
                def fs(s):
                    Zi2 = math.exp(-2*s); Zi3 = math.exp(-3*s); acc = 0.0
                    for sign in (1.0, -1.0):
                        acc += tc.gl(lambda a: tc.p_odd(a, a*a/(12*k) + k*b*Zi2, a**3/(144*k*k) + (a*b/4)*Zi2 + sign*c3*Zi3), -14.0, 14.0, ra)
                    return math.exp(-11*s)*acc
                return q*tc.gl(fs, s_star-3.0, s_star+4.5, rs)
            lo, hi = vlo(u), vhi(u)
            return tc.gl(fv, lo, hi, rv) if hi > lo else 0.0
        tot += tc.gl(fu, ulo, uhi, ru)
    return 108*k**7*tc.prefactor(k, 0.0)*tot

for k in (0.5, 1.0, 2.0):
    a1, a2 = tc.ALPHA_GH60[k]; m1, m2 = tc.ALPHA_MC[k]
    t0 = time.time()
    mS = mass(k, 'S'); mE = mass(k, 'E'); mO = mass(k, 'O2'); mI = mass(k, 'S2_inner')
    print(f'k={k}: S={mS:.5f} (a1+2a2 GH60 {a1+2*a2:.5f}, rel {abs(mS-(a1+2*a2))/(a1+2*a2):.1e}) | E={mE:.5f} (a1+a2 {a1+a2:.5f}, rel {abs(mE-(a1+a2))/(a1+a2):.1e}) | O2={mO:.5f} inner={mI:.5f} (a2 GH60 {a2:.5f}, MC {m2:.5f}; rel O2 {abs(mO-a2)/a2:.1e}, inner {abs(mI-a2)/a2:.1e}) | E-inner={mE-mI:.5f} (a1 {a1:.5f}, rel {abs(mE-mI-a1)/a1:.1e}) | S-E-O2={mS-mE-mO:.1e}  [{time.time()-t0:.0f}s]')

```
</details>

---
_Generated by [Claude Code](https://claude.ai/code)_
