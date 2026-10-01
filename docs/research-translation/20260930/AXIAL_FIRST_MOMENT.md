# An axial path turns selector failure into one window saddle

The [persistence bridge](PERSISTENCE_BRIDGE.md#5-a-collision-shortcut-rejected-by-an-actual-window-count)
rules out charging every rejected candidate to **two** additional window
critical points. A different comparison does work. A fourth-derivative bound
along the line of the two pins supplies a path to an older point. On that
event, rejection forces the actual death saddle into the open between-pin
height window. Its **first** moment then bounds the rejection probability.

This companion proves that deterministic comparison and its weighted
consequences. The Gaussian window first-moment estimates are explicit imported
inputs, not new continuum proofs. The result recovers the existing compact-mark
`O(r⁵)` weighted loss through a different dependency chain; it claims no improved
exponent. It also isolates deaths below the proposed saddle, including the
essential case, in an exception smaller than every prescribed power of r.

Exact source identities, actual reading scope and coauthor exposure are in
[AXIAL_SOURCES.json](AXIAL_SOURCES.json). P, IW, PP, CP and RM below refer to that
record. The parent P is read with the full identity set and corrections required
by REC §1, including E1 and W1 for its normalizer. The present argument does not
consume P's cap-failure estimate or global selector theorem as a premise.

## 1. Model, original law and the imported estimates

Fix the flat torus `X=ℝ²/(24ℤ²)` and its centered, variance-one Gaussian field
with exact covariance

```text
K(z) = Σ[n∈ℤ²] exp(−|z+24n|²/2) / Σ[n∈ℤ²] exp(−|24n|²/2).
```

Fix compact birth marks B and gap marks `K₀=[k_−,k_+]⊂(0,∞)`. Locations range
over X and orthonormal frames over O(2). Translate the midpoint to zero and
let u be the axial unit vector. The prescribed pins and values are

```text
M=−ru/2, S=ru/2,   f(M)=b, f(S)=s=b−kr³,
∇f(M)=∇f(S)=0,     b∈B, k∈K₀.
```

Q_r is canonical whole-field Gaussian regression on **these six observations**.
For a symmetric Hessian H, let `F_j(H)=|det H|` when H is nonsingular with j
negative eigenvalues, and zero otherwise. Retain

```text
W_r=F₂(H_M)F₁(H_S),  Z_r=E_Qr W_r,  dQ_r^W=(W_r/Z_r)dQ_r.       (A1)
```

There is no adjacency condition, pin density or spatial Jacobian in Z_r. The
tilted law Q_r^W is not Gaussian. All constants below may depend on the fixed
model and mark compacts; they are uniform in the stated locations, frames,
marks and sufficiently small positive r. No numerical constant is supplied.

The proof consumes the following exact source interfaces.

| Input | Statement and source |
|---|---|
| Derivative moments | `sup E_Qr(1+‖f‖_(C^q(X)))^p<∞` for every fixed finite q,p; P (4.1). |
| Full normalizer | `z_*r²≤Z_r≤z^*r²`, with `z_*>0`; P (5.5), read with E1 and REC W1. |
| Genericity and selector | Each fixed Q_r is almost surely Morse with distinct critical values; the global maximin mark is Borel and agrees with ordinary superlevel elder death; P §8 and the persistence bridge §§2.5,3. |
| Global window first moment | `E_Qr^W N_(r,j)(X\{M,S})≤Cr³`; IW (I5), with its declared PP/CP/RM inputs. |
| Intermediate window first moment | `E_Qr^W N_(r,j)({A₀r≤dist(X,0)≤ρ})≤Cr³(A₀^(−2)+ρ²)` for `A₀≥4`, `A₀r<ρ≤s₀`; IW (I4). |

Here `N_(r,j)(D)` counts additional critical points of index j in the Borel
set D with height in the **open** interval `(s,b)`, excluding M and S. The
window restriction in the last two rows is essential. We use only j=1.

IW (I1)–(I2) specifies exactly (A1) and the same six pins. Its formula (I33)
retains W_r and one witness determinant, with one division by Z_r. Its §10
combines the punctured pin balls, a fixed scaled collar, the intermediate
window shells and a fixed remote region under that same law. PP and CP supply
local all-height counts, whose window subsets are smaller; the intermediate
and remote counts retain the shrinking height window. RM explicitly states
that it does not consume the cap or global elder-selection conclusion. This
is a count-based dependency route, not an invocation of the desired selector
bound under another name. The present source-interface check is not a fresh
full analytic review of those continuum count proofs.

Take the common radius cutoff below 1, all imported positive cutoffs, and
`24/(4√2)`. The last restriction is a convenient sufficient embedding bound
already used by the parent; in particular it embeds the axial segment below
for every frame. No rotational invariance of the periodized covariance is
assumed.

## 2. A deterministic path with an explicit older endpoint

This lemma requires only a C⁴ function near the indicated line segment and
the two axial derivative pins. It does not require transverse concavity.
Put

```text
a=−r/2, c=r/2, d=3r/2,  g(x)=f(xu),
L_(4,r)=sup[a≤x≤d] |g''''(x)|,
A_r={r L_(4,r)≤3k}.                                      (A2)
```

**Axial lemma.** Suppose `g(a)=b`, `g(c)=b−kr³` and
`g'(a)=g'(c)=0`, where r,k are positive. On A_r, the straight path from M
to `du` has minimum exactly s, and

```text
g(d)≥b+(3/2)kr³>b.                                      (A3)
```

To prove it, integration by parts twice gives

```text
g(c)−g(a)=−(1/2)∫[a,c](x−a)(c−x)g'''(x) dx.
```

The nonnegative kernel integrates to `r³/6`. Since the left side is `−kr³`,
the weighted mean of g''' is 12k. Continuity gives an `x₀∈[a,c]` with
`g'''(x₀)=12k`. Every point of `[a,d]` is at distance at most 2r from x₀,
so (A2) yields

```text
g'''(x)≥12k−2r L_(4,r)≥6k>0,  a≤x≤d.                    (A4)
```

Write `F=g'`. We have `F(a)=F(c)=0` and `F''≥6k`. Strict convexity makes
F negative on `(a,c)` and positive on `(c,d]`: the secant slope between
its two zeros is zero, its derivative at c is positive, and its derivative
keeps increasing to the right. Thus g decreases from b to s, then increases.

For the quantitative rise, `H(x)=F(x)−3k(x−a)(x−c)` is convex and vanishes
at a,c. A convex function is nonnegative to the right of its two zeros,
by comparison of secant slopes. Hence

```text
g(d)−g(c) ≥ 3k∫[c,d](x−a)(x−c) dx
           =3k∫[0,r](t+r)t dt = (5/2)kr³.
```

This proves (A3), and the derivative signs show that the minimum of the
whole path is exactly g(c)=s. The path need not be a gradient trajectory or
a saddle separatrix. Both its bottleneck and its strictly older endpoint
have been proved directly.

## 3. The actual death saddle is the witness

For a maximum M of a compact Morse function with distinct critical values,
define

```text
d_f(M)=sup[γ(0)=M, f(γ(1))>f(M)] min[t∈[0,1]] f(γ(t)),
```

with value `−∞` when there is no older endpoint. At a regular level below
birth, the superlevel component containing M has met an older component
exactly when it contains a point higher than b. These regular-level
components are path connected. The maximal connection level in this
definition is therefore the ordinary elder death level. A finite death is
realized by a merging index-one saddle; distinct critical values identify
that saddle uniquely. This is the maximin interpretation already derived
in the persistence bridge §2.5.

A nondegenerate maximum also has `d_f(M)<b`: choose a small closed ball
around M where every other point has value below b. Its boundary has a
strictly smaller maximum. Every path to an older point exits that ball and
crosses its boundary. The essential value `−∞` satisfies the same strict
inequality.

On A_r, the axial lemma gives an actual older endpoint, so M is
nonessential and `d_f(M)≥s`. Let `e=1{S is the actual elder death of M}`.
If e=0 on A_r, equality `d_f(M)=s` is impossible: the unique critical point
at value s is S. Therefore

```text
A_r∩{e=0} ⊂ {s<d_f(M)<b}.                              (A5)
```

The actual killing saddle T on the right side is neither pin and lies in
the open window. In particular, on the Morse typed support,

```text
1−e ≤ N_(r,1)(X\{M,S}) + 1{A_r fails}.                 (A6)
```

For spatial statements, let τ_f(M) be the unique finite killing saddle,
and assign a separate cemetery value when M is essential. Define the
topological witness on a Borel spatial set D by

```text
T_r(D)=Σ[T∈D\{M,S}, ∇f(T)=0, index H_T=1, s<f(T)<b]
                       1{d_f(M)=f(T)}.                 (A7)
```

Then

```text
0≤T_r(D)≤N_(r,1)(D),   T_r(X)≤1,
T_r(X)=1−e on A_r.                                    (A8)
```

These are Borel marks. One can see this without presuming continuity of
the elder indicator: near each Morse function with distinct critical
values, its finitely many critical points continue continuously in C²,
and no extra critical points appear. In each such neighborhood, (A7) is
a finite sum of Borel spatial/type/value tests and the Borel maximin
equality. The Morse locus is open and C²(X) is separable, so a countable
cover by these neighborhoods proves Borelness on the locus. Extend the
marks by zero outside it. Pinned genericity gives that complement Q_r
and Q_r^W measure zero for each fixed pinned law; no common probability-one
set for uncountably many pin parameters is needed.

The definition (A7) identifies the real topological witness. Its estimate
does not assume a bound on that witness: (A8) compares it to the independently
estimated raw index-one count, and (A5) proves that every good-event failure
has such a witness.

## 4. The derivative exception with the original determinant weight

The moment required below follows from the endpoint gradient pins and the
uniform derivative moments; no cap-selection conclusion enters it. To
make the factor r² explicit, take Euclidean operator norms of derivative
tensors and let `K₃=1+‖f‖_(C³(X))`, with fixed norm-equivalence constants.
Along the segment from M to S,

```text
∫[a,c] H_f(xu)u dx = ∇f(S)−∇f(M)=0.
```

Subtracting this vector average at either endpoint and integrating its
Lipschitz bound gives `|H_i u|≤Cr K₃` for `i=M,S`. In an orthonormal
basis whose first vector is u, Hadamard's inequality in two dimensions
gives `|det H_i|≤|H_i u|‖H_i‖≤Cr K₃²`. Thus

```text
0≤W_r/r²≤C K₃⁴.
```

Since `L_(4,r)≤C‖f‖_(C⁴(X))` uniformly in the frame, P (4.1) gives for
each finite positive integer m

```text
sup E_Qr[(W_r/r²)L_(4,r)^m] ≤ C_m < ∞.                 (A9)
```

This is a joint weighted moment. Neither Hessian/derivative independence
nor Gaussianity of Q_r^W is asserted. Pointwise Markov followed by (A9)
and the full normalizer floor proves

```text
E_Qr[W_r; A_r fails]
 ≤ (r/(3k_−))^m E_Qr[W_r L_(4,r)^m]
 ≤ C_m r^(m+2),

Q_r^W(A_r fails) ≤ C_m r^m.                            (A10)
```

For every prescribed power one may choose the corresponding finite
moment; the constants can depend on that power. This is the sense of
“smaller than every prescribed power” here, not an exponential estimate.

Equation (A5) also gives the useful one-sided statement

```text
Q_r^W{d_f(M)<s} ≤ C_m r^m,                             (A11)
```

including `d_f(M)=−∞`. On the generic typed locus, e=0 is the disjoint
union of `d_f(M)>s` and `d_f(M)<s`. Its second part is therefore
superpolynomial in this precise sense. A death at s is the selected
candidate, rather than an omitted boundary case.

## 5. First-moment loss and the regional consequence

Multiply (A6) by W_r and integrate. Writing `p_r=E_Qr[W_r e]/Z_r`, use
the same-law IW (I5) and `Z_r≤z^*r²` to obtain

```text
0≤Z_r(1−p_r)=E_Qr[W_r(1−e)]
 ≤ Z_r E_Qr^W N_(r,1)(X\{M,S}) + E_Qr[W_r; A_r fails]
 ≤ Cr⁵+C_m r^(m+2).                                   (A12)
```

For example m=4 gives `E_Qr[W_r(1−e)]≤Cr⁵+C₄r⁶` and
`1−p_r≤Cr³+C₄r⁴`. There is exactly one endpoint determinant product and
one full normalizer. The imported count theorem already uses Q_r^W;
an unweighted count would not establish (A12).

For a Borel spatial set D, (A7)–(A10) give

```text
Q_r^W{e=0, τ_f(M)∈D}
 ≤ E_Qr^W N_(r,1)(D)+C_m r^m.                         (A13)
```

The cemetery value is outside every spatial D. On A_r the event in
(A13) is counted by T_r(D); off A_r its indicator is at most one.
Apply IW (I4) with `A₀=a/r`. Uniformly when
`4r≤a<ρ≤s₀`, this proves

```text
Q_r^W{e=0, a≤dist(τ_f(M),0)≤ρ}
 ≤ Cr³[(r/a)²+ρ²]+C_m r^m.                            (A14)
```

Here 0 is the pin midpoint, not M. For families with
`a_r/r→∞`, `ρ_r→0` and `4r≤a_r<ρ_r≤s₀`, divide (A14) by r³ and take
m>3 to see that the probability is `o(r³)`. This is a bound for the
location of the **actual** rejecting saddle in an intermediate annulus.
It does not exclude a fixed-remote contribution or describe two additional
witnesses approaching each other.

On fixed compact marks, the existing pair-measure Jacobian converts the
`O(r⁵)` numerator in (A12) into `O(r⁴)dr` radial loss, then into
`O(ℓ^(2/3))` lifetime-density loss under the exact substitution `ℓ=kr³`.
Its cumulative loss is `O(t^(5/3))`. These are the existing compact
orders in the [completion audit](COMPLETION_AUDIT.md#4-error-conversion-with-the-jacobian-retained),
now reached by the first-moment comparison. All-mark integration requires
its separate estimates; (A12) is not an unrestricted density theorem.

## 6. Scope and failure cases

The cap route proves e=1 from both a global exit barrier and a path to an
older point. This route constructs the latter only. It allows e=0 on A_r
and controls those preemptions by a window first moment. It has a heavier
count-estimate dependency than the cap route and does not replace that
route in the existing lifetime theorem.

The retained C39/C46 example in the persistence bridge has e=0 and exactly
one additional open-window critical point. It rules out the unconditional
inclusion `{e=0}⊂{N_r≥2}`. This proof supplies the one-witness inclusion
(A5), not a two-witness implication on A_r; substituting a second factorial
count would require a separate deterministic argument. A positive raw count
also need not identify the death saddle: its points can be unrelated to the
component born at M. Equations (A6) and (A8) are one-way comparisons, not an
equality with the raw count or a Poisson assertion.

The conditions do work at identifiable places. The exact gradient pins
give the two zeros of g' and the determinant's r² factor. The positive
gap supplies the forced positive third derivative. The segment must
embed, and the derivative bound must cover its full extension through
`3ru/2`, not just the interval between the pins. Without the older
endpoint proof, an essential birth or a death below s would escape the
open-window count. Without distinct critical values, equality at s
would not identify S uniquely. These exceptional possibilities are
handled by the stated deterministic hypotheses, pinned genericity and
the weighted exception, rather than silently omitted.

The axial lemma itself works in any dimension. The probabilistic results
here are planar and consume IW at its stated scope. A higher-dimensional
version must name and consume a dimension-matched count theorem. No
uniformity as k tends to zero, marks grow, dimension or torus size
changes, and no evaluated constant or lifetime cutoff is asserted.

This comparison does not estimate two-witness collision strata, retire
`math.rn-region.witness-collision`, re-audit IW's continuum proof, or alter
scientific acceptance records. It supplies a proved event comparison with
explicit imported analytic inputs. Source hashes and engineering checks
establish their own limited facts, not mathematical acceptance.

**Attribution:** Dylan Roy — delegated AI work; actual analysis and coauthor
writing by OpenAI/Codex `/root/c47_math_frontier`, coordinated with the root
agent. Both are same-provider contributors with source exposure and zero
organizational-independence credit. This text is coauthor work, not a
nonauthor review. Dylan's personal reading is PENDING.
