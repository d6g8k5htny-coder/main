# Quantitative fold isolation in arbitrary pair directions

16 October 2026 UTC. Local conventional-mathematics proposal by OpenAI/Codex `/root/core_theorem_audit_20261015`, independently reconstructing and attacking the sketch supplied by OpenAI/Codex `/root`. Organizational-independence credit: **0**. This document has not yet received a nonauthor review. It changes no production source, native record, claim, graph node or scientific status. In particular, it is not a Lean theorem, a factorial Rice bound, or a spatial point-process limit.

## 1. Concrete advance and source binding

An explicit Lyapunov–Schmidt chart isolates **at most two ambient critical points**, including at most one maximum, around a maximum with a nonvanishing soft cubic and a uniformly negative hard eigenvalue. Its radius is bounded below by a constant times the inverse square of a finite Gaussian coefficient norm. A literal positive cubic-gap mark can imply the required eigenframe cubic, but only with the derivative errors displayed below. Combining these statements gives an arbitrary-angle **occurrence-probability** estimate `C exp(-c/eta)` for two nearby short pairs on fixed positive transverse/cubic mark windows. It does not bound their factorial counting kernel.

The source consumed for the pair convention and finite Gaussian law is Main `85b928b326ff0586715a56ae91f51d3fbd89557a`, [experiments/universality/finite_gaussian_contact/PROOF.md](https://github.com/d6g8k5htny-coder/main/blob/85b928b326ff0586715a56ae91f51d3fbd89557a/experiments/universality/finite_gaussian_contact/PROOF.md). Git blob `5f345201309cfa1781e92d6292f542de4bcb2999`, 25,367 bytes, SHA-256 `7fea2656e11d986e1a5d6bfb0a05c8421bd93262d28051b8cfef05320595f474`. Its complete local body was checked against this tree identity. Its lines 41–57 define the actual finite source, 68–76 define the pins and radius, 145–186 give the divided differences, and 253–307 specify physical Hessian weights. The source is pinned, rather than asserted to be the latest moving head.

The previous frozen [COALESCING_CONTACT_CALCULATION.md](COALESCING_CONTACT_CALCULATION.md), [NONAUTHOR_COALESCENCE_REVIEW.md](NONAUTHOR_COALESCENCE_REVIEW.md), and [AUTHOR_ALIGNMENT_ADDENDUM.md](AUTHOR_ALIGNMENT_ADDENDUM.md) were read. This proposal is a separate arbitrary-angle calculation. Their review verdict does not extend to this document. Their collinear `exp(-c/eta^4)` Rice-kernel estimate is **not** the estimate derived here. Their four-site height-covariance rank obstructions remain in force.

## 2. Deterministic chart, including domain and constants

Use Euclidean coordinates in a Hessian eigenframe at an anchored critical point, moved to the origin. Let `F` be `C^4` on an open neighborhood of the closed square `Q_R0=[-R0,R0]^2`, with `R0>0`. Suppose

```text
grad F(0)=0,     F_xy(0)=0,
F_xx(0)<0,      F_yy(0)<=-a<0,
|F_xxx(0)|>=tau>0.
```

Let `B>=sup_Q_R0 ||D^3 F||op` and `C>=sup_Q_R0 ||D^4 F||op`. These are multilinear operator norms with unit vectors in each slot, so the bounds survive frame changes. In particular, each coordinate partial is bounded by the corresponding constant. Necessarily `B>0`. If only the maximum absolute coordinate partials `M3,M4` in an initial orthonormal frame are available, the conservative substitutions `B=2 sqrt(2) M3`, `C=4 M4` suffice by `||v||_1<=sqrt(2)||v||_2` in each slot.

Set

```text
D = C+B^2/a > 0,
R = min(R0, a/(32B), tau/(32D)).                       (1)
```

The following statements hold on the **closed** square `Q_R`. Classifications refer to ambient critical points of `F`, not constrained extrema of the square.

For each `|x|<=R`, there is a unique `h(x)` in `(-R,R)` with `F_y(x,h(x))=0`. The ridge has `C^3` regularity (locally extending at the two endpoints), and

```text
h(0)=h'(0)=0,
|h(x)|<=B x^2/a<=R/32,
|h'(x)|<=4BR/a<=1/8.                                  (2)
```

For `f(x)=F(x,h(x))`,

```text
|f'''(x)-F_xxx(0)|<=2CR+16B^2 R/a<=tau/2.              (3)
```

Thus `f'''` has the fixed sign of `F_xxx(0)` and magnitude at least `tau/2`. There are at most two critical points of `F` in `Q_R`. The anchored maximum is its only local maximum. If a second critical point exists, including on `x=+/-R`, it is nondegenerate and has Morse index one. The faces `y=+/-R` contain no ambient critical points. This is an isolation result containing a possible maximum–saddle pair, not an ordinary quadratic Morse chart about the maximum.

### Proof of the chart and its vertical barrier

On the square, integration along coordinate segments gives

```text
|F_yy(x,y)-F_yy(0)|<=B(|x|+|y|)<=2BR,
F_yy(x,y)<=-a/2.
```

Since `F_y(0)=F_xy(0)=0`, the one-dimensional Taylor integral on `y=0` gives `|F_y(x,0)|<=B x^2/2`. Hence

```text
F_y(x,+R)<=B R^2/2-aR/2<=-31 aR/64,
F_y(x,-R)>=-B R^2/2+aR/2>=+31 aR/64.                  (4)
```

The intermediate value theorem and strict decrease in `y` give the unique root. The implicit function theorem gives its stated regularity; the strict face margins permit local continuation at the endpoints. Subtracting `F_y(x,0)` from `F_y(x,h)=0` bounds `|h|<=B x^2/a`. Also

```text
h'=-F_xy/F_yy,
|F_xy(x,h)|<=B(|x|+|h|)<=2BR,
|h'|<=4BR/a.
```

All derivatives in the next display are evaluated at `(x,h(x))`. Exact differentiation, using the first and second derivatives of `F_y(x,h(x))=0`, yields

```text
f'=F_x,
f''=F_xx+F_xy h'=F_xx-F_xy^2/F_yy,
f'''=F_xxx+3F_xxy h'+3F_xyy(h')^2+F_yyy(h')^3.          (5)
```

For example, differentiating `f''=F_xx+F_xy h'` initially produces `F_xy h''`; the second derivative of `F_y=0` and `F_xy=-F_yy h'` convert this into the remaining mixed terms in (5). Omitting that cancellation would be an error.

The first term differs from `F_xxx(0)` by at most `2CR`. For `p=|h'|<=1/8`, `3p+3p^2+p^3<=4p`. The remaining terms have total absolute value at most `4Bp<=16B^2R/a`, proving (3). Here `2C+16B^2/a<=16(C+B^2/a)` and (1) give the final `tau/2` bound.

Put `g=f'`. Fixed-sign `g''=f'''` makes `g` strictly convex or strictly concave, so it has at most two zeros. Every critical point of `F` in the square lies on `h`, and its `x` coordinate is a zero of `g`. At the anchored zero, `g'(0)=f''(0)=F_xx(0)<0`. If a second zero exists, strict monotonicity of `g'`, or Rolle's theorem between the two roots, implies its derivative is strictly positive. Thus the second root is simple and `f''>0`. At a critical point,

```text
det Hess F = F_yy f''.
```

The second point has a negative determinant and is an index-one saddle. This also proves that a second degenerate root cannot coexist with the anchored nondegenerate maximum. Without that anchor hypothesis, a double root `g=g'=0` is possible; the two-root bound by itself would not prove nondegeneracy. No critical point outside the square or constrained boundary extremum is covered.

Finally, integrating `F_yy<=-a/2` twice from the ridge gives the useful **vertical** height barrier

```text
F(x,y)<=f(x)-(a/4)(y-h(x))^2,
F(x,+/-R)<=f(x)-aR^2/8.                               (6)
```

The last bound follows from `|h|<=R/32` and `(31/32)^2/4>=1/8`. It is relative to `f(x)`, rather than to the height at the anchored maximum. In particular, a distinct maximum at Euclidean distance strictly less than `R` from the origin is excluded, regardless of its own eigenframe or pair direction.

## 3. Falsification attempts that must remain attached

**A height drop on the entire boundary relative to `F(0)` is false.** For

```text
F(x,y)=-mu x^2/2+tau x^3/6-a y^2/2,    mu,a,tau>0,
```

the origin is a nondegenerate maximum with the required transverse curvature and cubic. But `F(R,0)>F(0)` whenever `mu<tau R/3`. The radius (1) need not shrink with `mu`, so this remains a counterexample at the proposed chart radius by taking `mu` sufficiently small. This prevents inferring an enclosing superlevel barrier, trapping all paths, actual elder selection, or global persistence pairing from (6).

**The `B^2/a` term in (1) cannot be discarded just because `D^4F=0`.** Take

```text
F(x,y)=-mu x^2/2+tau x^3/6+(b/2)x^2 y-a y^2/2,
h(x)=b x^2/(2a),
f(x)=-mu x^2/2+tau x^3/6+b^2 x^4/(8a),
f'''(x)=tau+3b^2 x/a.                                 (7)
```

Every fourth derivative of `F` is zero, but the reduced cubic changes sign at `x=-a tau/(3b^2)`. Thus controlling the variation of `F_xxx` alone does not control the reduced cubic. This example falsifies that shortcut; it is not itself a construction of two maxima in the claimed isolation square.

**No separation radius independent of derivative bounds is possible.** For `delta,tau,a>0`, let

```text
F(x,y)=-A x^2(x-delta)^2-a y^2/2,    A=tau/(12delta).
```

There are nondegenerate maxima at `x=0,delta`, both with `|F_xxx|=tau`; but `|F_xxxx|=2tau/delta`. Their distance can vanish as the fourth-derivative bound diverges. They share a local saddle; they are not being asserted to represent two actual finite persistence bars.

These polynomial examples are deterministic local controls. They do not assert positive-probability occurrences within a particular finite Fourier law.

## 4. Literal endpoint distance and conversion of marks to the eigenframe

The pinned source uses `M=z-r u/2`, `S=z+r u/2`, so **`r=|S-M|` is the endpoint distance**, and `ell=F(M)-F(S)=k r^3`. It is not the half-distance. Assume the whole endpoint segment is in the domain on which the operator bounds `B,C` hold, and both endpoints are critical. The source's fourth divided-difference row becomes exactly

```text
12k = integral_0^r [6s(r-s)/r^3] F_uuu(M+s u) ds.        (8)
```

The nonnegative kernel has mass one and mean `s=r/2`. Consequently,

```text
|F_uuu(M)-12k|<=C r/2.                                (9)
```

This is an endpoint estimate. It must not be replaced by a midpoint estimate without moving the evaluation point.

Let `H=Hess F(M)`. Vector Taylor expansion of `grad F(S)-grad F(M)=0` gives

```text
H u = -(1/r) integral_0^r (r-s) D^3F(M+s u)[u,u,.] ds,
||H u||<=B r/2.                                       (10)
```

Suppose `M` is a nondegenerate maximum and its most negative eigenvalue is at most `-a`. For example the source transverse condition `F_vv(M)<=-a` for `v` perpendicular to the **pair** direction implies this by the Rayleigh quotient; it does not assert that the pair frame diagonalizes `H`. Assume `rB<=a`. The soft eigenvalue has absolute value at most `Br/2<=a/2`, since the smallest singular value of `H` is at most `||Hu||`. Thus the hard and soft eigenspaces are distinct. Let `v_h` be a unit hard eigenvector and `e` a unit soft eigenvector oriented by `e.u>=0`. Projecting (10) onto the hard eigenspace gives

```text
|u.v_h|<=Br/(2a)<=1/2,
||u-e||<=sqrt(2)|u.v_h|,
|F_eee(M)-F_uuu(M)|<=3B||e-u||<=3B^2 r/a.              (11)
```

The last constant is deliberately conservative. Combining (9) and (11),

```text
F_eee(M)>=12k-r(C/2+3B^2/a).                           (12)
```

Therefore `k>=k_->0`, `rB<=a`, and

```text
r(C/2+3B^2/a)<=6k_-                                   (13)
```

imply `F_eee(M)>=tau:=6k_-`. Choose the isolation chart's `x` direction to be this `e` and its `y` direction the hard eigenvector. The original pair direction can have any angle; only the estimate (11) is used. An upper mark bound is not needed for this implication. The derivative conditions are retained explicitly.

The normalization check requested by the sketch author is exact:

```text
F=x^3/6-delta^2 x/2-a y^2/2,
M=-delta, S=+delta, r=2delta,
ell=2delta^3/3, k=1/12, F_xxx=1=12k.                  (14)
```

Conversely, positive `k` alone does **not** give a nonzero cubic at the maximum. The polynomial

```text
F=(k/r)x^4-2kr x^2-a y^2/2
```

has a nondegenerate maximum at `M=0`, a saddle at `S=r`, and gap `k r^3`, but `F_xxx(M)=0` and `F_xxxx=24k/r`. The error `Cr/2=12k` in (9) is exactly saturated. This falsifies a uniform mark-to-endpoint-cubic inference without controlling `rC`. It asserts no global elder relation.

## 5. Consequence for the actual fixed finite Gaussian source

Keep exactly the pinned source law: fixed `L>0`, `omega=2pi/L`, square support `S_K={-K,...,K}^2`, fixed positive symmetric weights `q_n` summing to one, and its `N=(2K+1)^2` independent standard real coefficients `Z`. No covariance isotropy or independent-copy replacement is made. For `j=3,4`, put

```text
beta_j = [sum_(n in S_K) q_n (omega|n|)^(2j)]^(1/2)>0,
W=1+||Z||_2.
```

For arbitrary unit vectors in the `j` derivative slots, the squared Euclidean norm of the coefficient row for that directional derivative is at most `beta_j^2`: within each sine/cosine pair the squares add to `2q_n` times the squared product of the frequency projections. Cauchy–Schwarz then gives, uniformly in location and frame,

```text
sup_X ||D^jF||op <= beta_j ||Z||_2 <= beta_j W.          (15)
```

Fix an embedded square scale `R0>0` small enough for all translated/rotated charts being used, for example `R0<=L/16`, and use global derivative bounds `B=beta_3 W`, `C=beta_4 W`. For a maximum satisfying the direct transverse/cubic hypotheses with fixed `a,tau>0`, define

```text
kappa=min(R0, a/(32beta_3),
          tau/[32(beta_4+beta_3^2/a)])>0.
```

Because `W>=1`, each member of the minimum in (1) is at least the corresponding constant divided by `W^2`. Thus its isolation radius obeys

```text
R>=kappa/W^2.                                        (16)
```

If another distinct maximum lies at Euclidean chart distance `delta`, (16) implies `W^2>=kappa/delta`. In particular, for any measurable occurrence event `E_delta` of two such maxima at positive distance at most `delta`, and `delta<=kappa/4`,

```text
P(E_delta)<=2^(N/2) exp[-kappa/(16delta)].              (17)
```

Indeed `W>=sqrt(kappa/delta)` forces `||Z||>=sqrt(kappa/delta)-1>=sqrt(kappa/delta)/2`, and `E exp(||Z||^2/4)=2^(N/2)`. Markov's inequality yields (17). The same statement applies to outer probability if an occurrence set's measurability has not been separately supplied. It does not require a first/second-jet covariance inverse.

### Pair marks without silently assuming their errors small

Here is a precise norm dichotomy that incorporates the errors (13). Fix `a,k_->0`, `0<rho<1`, and a small embedded scale `eta0>0`. Suppose two candidate pairs `(M_i,S_i)`, with distinct maxima `M_1!=M_2`, satisfy

```text
grad F(M_i)=grad F(S_i)=0,
M_i is a nondegenerate local maximum,
u_i=(S_i-M_i)/r_i,     r_i=|S_i-M_i|>0,
F(M_i)-F(S_i)=k_i r_i^3,  k_i>=k_-,
F_(v_i v_i)(M_i)<=-a,     v_i perpendicular to u_i,
0<eta=|z_2-z_1|<=eta0,    z_i=(M_i+S_i)/2,
r_i<=rho eta.
```

Use common short lifts so every endpoint, pair segment and displacement appearing here is represented in one embedded Euclidean patch. For instance take `eta0<=R0/4` and `R0<=L/16`. The angles `u_1,u_2` are arbitrary. Saddle type at `S_i`, an upper mark bound, actual elder selection, and disjointness of the saddles are not needed for this obstruction. Distinctness of the two maxima is essential. By the triangle inequality,

```text
delta=|M_2-M_1|<=(1+rho)eta.
```

Set `tau=6k_-`, use its `kappa` above, and put

```text
D0=beta_4/2+3beta_3^2/a,
c_*=min(a^2/(rho^2 beta_3^2 eta0),
        6k_-/(rho D0),
        kappa/(1+rho))>0.                              (18)
```

Then these two pairs force

```text
W^2>=c_*/eta.                                        (19)
```

For otherwise `W^2<c_*/eta`, the first pair satisfies

```text
r_1 B <=rho beta_3 sqrt(c_* eta)<=a,
r_1(C/2+3B^2/a)<=rho eta D0 W^2<rho D0 c_*<=6k_-.
```

Equations (12)–(13) supply its eigenframe cubic lower bound `tau`. Its chart radius satisfies `R>=kappa/W^2>kappa eta/c_*>=(1+rho)eta>=delta`, placing the other maximum strictly inside the isolation square, a contradiction. Global derivative bounds justify using the same `B,C` along the first pair segment and throughout this chart.

For an occurrence event `E_eta` of such configurations with center distance at most `eta<=min(eta0,c_*/4)`, (19) gives

```text
P(E_eta)<=2^(N/2) exp[-c_*/(16eta)].                   (20)
```

For varying center distances `s<=eta`, apply (19) with `s` and obtain the stronger lower bound `W^2>=c_*/s>=c_*/eta`. Thus this is a bound on the whole close-pair occurrence event, not just one fixed configuration. It is fixed-law and fixed-mark-window; no uniformity in vanishing `a` or `k_-`, unbounded `K`, varying `L`, or varying spectra is claimed.

## 6. What this does and does not change

The next finite mathematical task can use (19) as an arbitrary-angle collision obstruction on positive transverse/cubic windows. This removes a collinearity restriction from this **occurrence** bound and avoids demanding a nonexistent uniform twelve-row height density on the K2/K3 coordinate stratum. It does not repair that density, calculate a conditional four-site covariance determinant, or supply conditional determinant/selection-weight estimates. In particular, an exponentially small probability of some collision does not by itself bound the expectation of a possibly large factorial count. A separate weighted counting argument, with its actual conditioning list and finite-support rank control, remains necessary.

Small transverse curvature, small cubic marks, pairs whose radii are not bounded by a fixed fraction of center separation, shared-maximum incidences, and other collision geometries are outside (18)–(20). The height barrier (6) proves no actual local selection probability. Global H0 max–saddle pairing, interval decomposition, essential-class handling, higher homology, leading complement estimates, same-field expanding-domain control, process convergence, and the literal lifetime density `ell^(-1/3)` rather than merely cumulative `t^(2/3)` remain separate obligations. No theorem closure is inferred from this local normal form.

## 7. Finite algebra controls and review custody

[fold_isolation_controls.py](fold_isolation_controls.py) is a standard-library exact-rational polynomial checker. [FOLD_ISOLATION_CONTROLS.json](FOLD_ISOLATION_CONTROLS.json) is its output. It checks the full mixed reduced-third-derivative identity on a prescribed nontrivial stationary ridge, the endpoint-distance factor 12, the positive-mark/zero-endpoint-cubic counterexample, the mixed-cubic sign-loss example with `D^4F=0`, the false full-height-barrier example, the two-maxima derivative-scale example, and the displayed rational chart margins. Every check passed using Python `Fraction`; no floating arithmetic, field draws, sampling, persistence experiment or process test occurred.

These controls catch finite algebra and normalization mistakes. They do not mechanically verify the analytic proof, geometric selection, measure-theoretic counting, or a Lean statement. The current review lineage is root sketch -> this agent's independent conventional derivation plus falsification controls -> pending separate nonauthor review. All are OpenAI/Codex in the same parent thread; there is no provider independence or human review. The report and its control source/output are to be frozen by SHA-256 for that next review. Any correction after freezing should be an append-only addendum.
