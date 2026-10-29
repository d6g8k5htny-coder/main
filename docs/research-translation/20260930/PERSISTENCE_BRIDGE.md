# From a separating cap to an exactly counted persistence pair

This companion supplies the continuous planar argument behind manuscript M7–M9:
construct a cap, identify the **global** elder death, make that selector Borel,
and insert it into the marked Kac–Rice formula with the full determinant weight.
It expands existing CAP/P/E2 arguments; it does not establish a new global
asymptotic, a shrinking-collision estimate, or a numerical lifetime cutoff.
The last section checks a proposed collision shortcut against an existing
exact counterexample.

Read this between [Appendix B](APPENDICES.md#b-contact-regression-and-the-full-normalizer)
and Appendix C's weighted failure integral. The
[completion audit](COMPLETION_AUDIT.md) tracks the subsequent normalization,
pushforward and retained open obligations. Exact source identities and consumed
slices are in [BRIDGE_SOURCES.json](BRIDGE_SOURCES.json). CAP, P and E2 refer
to the marked cap proof, uniform matrix-cap parent and Borel replacement in
that file. Their bytes agree with the earlier manuscript cut. The C39 example
is [the preserved cap-assumption ablation](cap-assumption-ablation/PROOF.md).
This assembly and its reviews are AI work on Dylan Roy's behalf, not evidence
that he personally reviewed the mathematics or that an independent human did.

## 1. Fixed model, pair roles and the exact output

The ambient space is the fixed flat torus `X=ℝ²/(24ℤ²)`. The field is centered,
unit variance, with the exact covariance

```text
K(z) = Σ[n∈ℤ²] exp(−|z+24n|²/2) / Σ[n∈ℤ²] exp(−|24n|²/2).
```

Its Fourier coefficients are strictly positive and decay faster than any
power. It has a smooth version with finite moments of each finite derivative
supremum. The filtration is ordinary superlevel H₀; only finite bars are
counted. The global maximum carries the essential component and is excluded.
At a merging saddle the component with the *younger* maximum dies; the older
component survives. Some index-one saddles change topology without merging
components, so saddle type alone is not the death selector.

The ordered roles are `(M,S)` = (maximum, saddle). They are not an unordered
pair. Write `b=f(M)`, `s=f(S)=b−kr³`, where `k>0` and `r=dist(M,S)>0` in an
embedded local chart. At fixed sites and heights, `Q` is the canonical
Gaussian regression law on the two heights and the four zero gradient
coordinates. Define

```text
W = |det H_M det H_S| 1{H_M<0, index(H_S)=1},
Z = E_Q W,                    dQ^W = (W/Z)dQ,
e(f,M,S) = 1{S is the actual ordinary elder death of M},
p = E_Q[W e]/Z.
```

There is no pin density or coordinate Jacobian in `Z`. Below we prove the
following interface, for each fixed positive separation and distinct heights:

```text
G_cap ⊂ {e=1}                         Q-almost surely,
0 ≤ Z(1−p) = E_Q[W(1−e)] ≤ E_Q[W 1{G_cap fails}].      (1)
```

This interface is not an estimate of its right-hand side. Appendix C supplies
the compact-mark `O(r⁵)` weighted numerator and Appendix B the `Z≥z_*r²`
floor, with all their hypotheses. Only together do they give `1−p≤Cr³`.

## 2. The full planar cap construction

**Deterministic statement.** In orthonormal local coordinates let
`D=[−2r,2r]²`, `M=(−r/2,0)`, `S=(r/2,0)`. Suppose `f` is C⁴ on a
neighborhood of D, the two pins are critical, and their heights are b and
`s=b−kr³`. Let `M_j` be the maximum absolute coordinate partial of total
order j on D, and put `λ=−f_yy(M)`. Assume

```text
λ > [4/(3k)] r M₃²,                  r M₄ ≤ 3k/10.     (2)
```

If D embeds in the torus (the sufficient condition is `r<24/(4√2)`), there
is a cap `C=[−2r,r/2]×[−2r,2r]` with all three properties:

1. `f≤b` throughout C;
2. `f≤s` on its entire boundary;
3. a path from M through S ends at height greater than b and has minimum s.

The pins have the stated maximum and saddle types. For every global Morse
extension with distinct critical values, S is M's actual elder death.
The proof uses no information about the exterior of D.

### 2.1 Normalization and forced cubic derivative

Put `φ=f/(6k)`, `b₀=b/(6k)`, `s₀=s/(6k)=b₀−r³/6`, and denote its derivative
bounds by m,n and its pin curvature by λ₀. Equation (2) becomes
`λ₀>8rm²`, `rn≤1/20`. Set `a=−r/2`, `c=r/2`. Twice integrating by parts,
using `φ_x(a,0)=φ_x(c,0)=0`, gives

```text
φ(c,0)−φ(a,0)
  = −(1/2)∫[a,c](x−a)(c−x) φ_xxx(x,0) dx.            (3)
```

The kernel integral is `r³/6`. Its weighted average of `φ_xxx` is therefore
2, so `m≥2` and some `x₀∈[a,c]` has `φ_xxx(x₀,0)=2`. Each point of D is
at ℓ¹ coordinate distance less than `5r` from both `(x₀,0)` and M. The
derivative bounds along line segments in D imply

```text
φ_xxx ≥ 7/4,             φ_yy ≤ −δ,
δ := λ₀−5rm > rm(8m−5).                               (4)
```

### 2.2 A transverse maximizing ridge on the whole interval

Let `w(x)=φ_y(x,0)`. The pins give `w(a)=w(c)=0` and `|w''|≤m`. The
two-node interpolation remainder gives, also for x outside `[a,c]`,

```text
|w(x)| ≤ (m/2)|x²−r²/4| ≤ 15mr²/8 < 2mr².
```

For the outside case apply Rolle twice on the convex hull of a,c,x after
subtracting the quadratic matching the value at x and zeros at a,c.
Moreover `∫[a,c]w'(t)dt=0`, hence

```text
w'(x) = (1/r)∫[a,c](w'(x)−w'(t))dt,
|w'(x)| ≤ (m/r)∫[a,c]|x−t|dt ≤ 2mr.                  (5)
```

Write `q=8m−5≥11`. Strong transverse concavity in (4), together with the
bound on w, makes `φ_y(x,−2r)>0` and `φ_y(x,2r)<0`. There is a unique
interior transverse maximizer `y=h(x)`. The implicit function theorem gives
`h∈C³`, `h(a)=h(c)=0`, and

```text
|h| < 2r/q,
|h'| = |φ_xy/φ_yy| < 2/q+2/q² =: u ≤ 24/121,
m u ≤ 48/121.                                        (6)
```

Indeed `|φ_xy(x,h)|≤|w'(x)|+m|h|`, giving the second bound after division
by `δ>rmq`. For the last bound, write `m=(q+5)/8`; then
`mu=(q+5)(q+1)/(4q²)`, decreasing for `q≥11` and equal to `48/121` at 11.

### 2.3 The ridge derivative and its exact positive bound

Set `g(x)=φ(x,h(x))`, `F=g'=φ_x(x,h(x))`. Differentiating
`φ_y(x,h(x))=0` gives `φ_xy+φ_yy h'=0`. In the third derivative of g, the
terms involving h'' vanish by this identity and the term involving h'''
vanishes because `φ_y=0`. Thus

```text
F'' = φ_xxx + 3φ_xxy h' + 3φ_xyy(h')² + φ_yyy(h')³
    > 7/4 − (48/121)[3+3(24/121)+(24/121)²]
    = 2184415/7086244 > 1/4.                          (7)
```

This is a calculation on a maximizing ridge. It does not assert that the
ridge is an integral curve of the gradient.

### 2.4 Pin types, the interior bound and every boundary face

The critical pins give `F(a)=F(c)=0`. Strict convexity makes F negative
between its zeros, positive outside, and `F'(a)<0<F'(c)`. Every critical
point in D must lie on the ridge, so these are the only two. At a ridge
critical point the Hessian's transverse block is negative and its Schur
complement is `g''=F'`; consequently M is a maximum and S has index one.
Also g rises to `g(a)=b₀` on `[-2r,a]` and decreases to `g(c)=s₀` on
`[a,c]`. Strong concavity gives the load-bearing interior bound

```text
φ(x,y) ≤ g(x) ≤ b₀                      on C.         (8)
```

The function `F(x)−(x−a)(x−c)/8` is convex and zero at a,c, hence
nonnegative outside `[a,c]`. Integrating the comparison yields

```text
g(−2r) ≤ b₀−9r³/32 = s₀−11r³/96,
g(2r)  ≥ s₀+9r³/32 = b₀+11r³/96.                    (9)
```

For the cap's transverse faces, `−2r≤x≤c`, (6) gives
`|±2r−h(x)|>20r/11`, while `δ>22r`. Therefore

```text
φ(x,±2r) ≤ g(x)−(δ/2)|±2r−h(x)|²
         < b₀−(4400/121)r³ < s₀.                     (10)
```

The left face is below s₀ by (9). On the right face `h(c)=0` gives
`φ(c,y)≤s₀−δy²/2`, with equality only at S. This covers the corners too.
Along the ridge from a to 2r, g decreases to s₀ then increases to a value
strictly above b₀; its minimum is exactly s₀. Rescaling by `6k` proves
the three claimed properties and an older-endpoint margin at least
`(11/16)kr³`.

### 2.5 Why the conclusion is global persistence

For a maximum M of a compact Morse function with distinct critical values,
define

```text
d_f(M) = sup[γ(0)=M, f(γ(1))>f(M)] min[t∈[0,1]] f(γ(t)),
```

with value `−∞` if there is no older endpoint. At a regular level below
birth, the superlevel component containing M has met an older component
exactly when it contains a point higher than M. Such a component is path
connected. Its first merger with an older component consequently has level
`d_f(M)`; Morse theory and distinct critical values identify its unique
merging saddle. This is precisely ordinary elder death, irrespective of
the distance between the endpoints. The global maximum has no older
endpoint and accounts for the excluded essential bar.

By (8) every older endpoint for the constructed cap lies outside C. Any
path to it exits C and encounters height at most s. The ridge attains
minimum s on a path to an older endpoint. These two inequalities give
`d_f(M)=s`, proving the deterministic statement. Local branch adjacency
alone would not supply these inequalities; no such substitution is used.

## 3. Genericity for each pinned law and an explicit Borel selector

### 3.1 The finite-jet rank used here

For this exact Gaussian field, the variance of a finite linear combination
T of derivative evaluations is the positive Fourier sum
`Σ[n] a_n |T(exp(2πin·x/24))|²`. If it is zero, T annihilates every Fourier
mode. Fourier approximation of smooth periodic functions in each required
derivative norm implies that T is the zero distribution. At distinct sites,
bump test functions with prescribed finite Taylor jets show that the
coefficients of distinct derivative functionals must vanish. Thus these
finite jet covariances are positive definite. This is a qualitative
statement at distinct sites, not a quantitative floor as sites collide.

Conditioning on the six pins removes their linear span. The same argument
proves full rank for any appended independent functionals. In particular,
the pair of endpoint Hessians has a full conditional density, so each
singular-Hessian event has probability zero. On the type cones W is
positive and polynomially bounded; their conditional Gaussian probability
is positive. Consequently `0<Z<∞` for every fixed separated pinned law.

To handle unpinned critical points, exhaust compact parameter domains
away from the pins and, when two variable sites occur, away from their
mutual diagonal. The relevant Gaussian maps are:

| Bad event | Map | Parameter dimension | Output dimension |
|---|---|---:|---:|
| Singular critical Hessian | `(∇f(x),H_f(x)v)`, `v∈S¹` | 3 | 4 |
| Equal values of two critical points | `(∇f(x),∇f(y),f(x)−f(y))` | 4 | 5 |
| Unpinned critical value tied to a pin | `(∇f(x),f(x)−b)` or its s version | 2 | 3 |

Their conditional covariances have uniformly positive lower eigenvalues
on each such compact domain. In the first row `H↦Hv` is onto ℝ² for every
unit v; the derivative-distribution argument proves the rest. The maps
are C¹ and their pointwise densities have uniform bounds there.

Here is the zero-avoidance step. For a p-dimensional parameter domain and
q-dimensional output, a mesh of spacing ε has `O(ε^(−p))` points. On the
event that the derivative norm is at most J, any zero forces a mesh value
into a ball of radius `C J ε`. The union bound and bounded pointwise
densities bound its probability by `C_J ε^(q−p)`, tending to zero if q>p.
First let ε tend to zero, then let J tend to infinity, and finally take
the countable union of compact domains and coordinate charts. The smooth
sample paths have finite derivative norms on each compact. The endpoint
Hessian argument handles the pins themselves, whose two heights differ.
The pinned field is therefore almost surely Morse with distinct critical
values. Absolute continuity transfers this fact to Q^W. This is proved
**for each fixed regression law**, not on one common null-set complement
over all uncountably many pin parameters.

### 3.2 Countably many paths define the global mark

Let `E=C²(X)`. Choose the Borel representative of x in `[0,24)²` and all
finite lists of rational points in ℝ². For each list, join the representative
of x to those points by straight segments and project the path onto X.
This is a countable family of paths `γ_j(x)` with variable initial point x.
For each j set

```text
d_j(f,x) = min f(γ_j(x))  if f(γ_j(x)(1))>f(x),
           −∞           otherwise.
```

Each minimum is Borel in `(f,x)`: it is the infimum over rational path
times of Borel evaluations, and continuity in path time identifies that
infimum with the minimum. The endpoint test is Borel too. Every continuous
torus path lifts to ℝ²; uniform polygonal approximation followed by rational
vertex approximation preserves its strictly older endpoint and changes
its minimum by arbitrarily little, by uniform continuity of f. Thus

```text
d_f(x) = sup[j] d_j(f,x).                              (11)
```

The right-hand side is extended-real Borel. Hence equality with `f(y)`
is a Borel condition, with no continuity assertion for the equality
indicator. The locus of Morse functions with distinct critical values
is open in C²: finitely many nondegenerate critical points persist in
disjoint neighborhoods, the gradient is bounded away from zero outside,
and their finite collection of strict critical-value gaps persists.
Define the global mark on all of `X²×E` by requiring this locus, the
critical-point and type conditions, `f(x)>f(y)`, and `d_f(x)=f(y)`.
It is Borel, and on the generic locus it is exactly e of §1.

Every finite ordinary H₀ bar now corresponds to exactly one selected
ordered pair `(M,S)` and conversely. This identity uses the **global**
selector and makes no claim that every small bar is a short local contact.

## 4. The marked pair formula and the loss inequality

### 4.1 A continuous whole-field regression kernel

Let `D₀⊂X²\diag` be compact and `t=(x,y)`. Put
`G_t=(∇f(x),∇f(y))∈ℝ⁴`, `Σ_t=Cov(G_t)`, and `C_t=Cov(f,G_t)`, an
E-valued four-column covariance map. The kernel used throughout is

```text
K(t,z,·) = Law(f−C_tΣ_t^(−1)G_t+C_tΣ_t^(−1)z).         (12)
```

Gaussian orthogonality makes the residual independent of G_t; this holds
for the full field by a countable determining family of derivative
evaluations. It is therefore a regular conditional version for every z.
Smooth covariance, the compact covariance floor and a common smooth-field
coupling make the expression continuous in `(t,z)` as an E-valued random
variable. In particular the kernel is weakly continuous and Borel.

The external Gaussian Kac–Rice inputs are Theorems 2.2 and 7.1 of
[Armentano–Azaïs–León, version 3](https://arxiv.org/html/2304.07424v3).
We use the unweighted formula and its finite expectation on compact
separated domains, and the marked formula for bounded nonnegative
continuous cylinder weights. Here G is C¹ with nonsingular pointwise
covariance; (12) supplies the conditional mark law. To express C²
cylinders in the mark's continuous-field topology, take the joint
Gaussian continuous vector field consisting of f and all derivatives
through order two. These applications are the standard theorem inputs;
the Borel extension below is a separate argument.

### 4.2 Finite-measure extension, not semicontinuity of the elder mark

For Borel `A⊂D₀×E` define

```text
μ(A) = E Σ[t∈D₀:G_t=0] 1_A(t,f),
η(A) = ∫[D₀] p_Gt(0) E[Δ_t 1_A(t,f) | G_t=0] dt,
Δ_t = |det H_x det H_y|.                              (13)
```

The derivative of G with respect to `(x,y)` is block diagonal, so Δ is
its absolute determinant. The unweighted formula makes both measures
finite: the density is bounded on D₀ and conditional Hessian moments
are uniformly bounded there. Flat-torus coordinate charts followed by
a disjoint Borel partition give (13) intrinsically.

Bounded continuous cylinders in the location and finitely many derivative
evaluations form a unital algebra. Their nonnegative members satisfy
the marked formula, so their μ and η integrals agree; bounded signed
members follow by adding a constant. The evaluations through order two
on a countable dense set generate `B(E)`: they recover the C² norm by
countable suprema, and separability gives a countable base of norm balls.
Together with location coordinates they generate `B(D₀×E)`.

The class of bounded functions with equal μ and η integrals is a vector
space containing this algebra, and is closed under uniformly bounded
pointwise convergence, by dominated convergence for the two finite
measures. The functional monotone-class theorem proves equality for
every bounded Borel mark, including e. Nonnegative unbounded marks follow
by truncation and monotone convergence. No approximation of e by
continuous indicators and no lower-semicontinuity claim is needed.

### 4.3 Height disintegration retains exactly one determinant product

Let `q_t(b,s)` be the joint density of the two heights and four gradients
at `(b,s,0,0)`, in that order, and let `Q_(t,b,s)` be full-pin regression.
The six observations have nonsingular covariance at distinct sites.
For every nonnegative Borel test Ψ, the selected typed-pair identity is

```text
E Σ[(M,S)∈D₀, typed, b>s] e(f,M,S) Ψ(M,S,b,s)
 = ∫[D₀]∫[b>s] q_t(b,s) E_Q[W e] Ψ(t,b,s) db ds dt
 = ∫[D₀]∫[b>s] q_t(b,s) Z(t,b,s) p(t,b,s)
                                      Ψ(t,b,s) db ds dt. (14)
```

Indeed `p_Gt(0)p_(heights|G_t=0)(b,s)=q_t(b,s)`; multiplying the
determinant product in (13) by the type indicator gives exactly W.
Omitting e gives the candidate formula with `q_t Z`. There is no second
determinant product and no division by two. A density from disintegration
is identified almost everywhere; (12) and its six-observation analogue
specify the canonical versions used for pointwise pinned statements.

Increasing separated domains to all distinct pairs extends (14) by
nonnegative convergence, initially as an identity possibly taking value
infinity. This extension by itself proves no near-diagonal integrability.
The source estimates used in Appendix E provide that additional control.
Applying the lifetime map `ℓ=b−s` to the selected measure gives exactly
the expected finite-bar measure. Dividing by `24²` gives the manuscript's
per-area convention. Changing variables to r,k and obtaining a density
or its asymptotics are subsequent steps, not consequences of Borelness.

For the local geometry of §2, pinned genericity and the cap theorem give
`e=1` Q-almost surely on G_cap. Since `0≤e≤1` and `W≥0`, (1) follows
pointwise for the canonical pinned law. This is the exact interface
between deterministic topology and the Gaussian weighted failure bound.

## 5. A collision shortcut rejected by an actual window count

The C6 Palm source counts **all** additional critical points in the open
height window, excluding the two pins:

```text
N_r = #{x∉{M,S}: ∇f(x)=0, b−kr³<f(x)<b}.
```

It is not a count of already selected persistence competitors. For this
count the inclusion `{e=0}⊂{N_r≥2}` is false. The C39 construction proves
this for an exactly enumerated smooth Morse torus function, rather than
an abstract random integer.

Fix `0<ε<1/16` (C39 calls this bump parameter δ). Its eight original
critical values are

```text
0, −1, 10000, −10000, −1000, −1001, 9000, −11000.
```

The only new critical values are `ε²m∈(−1/2,0)` at a saddle T and
`ε²h∈(0,1)` at an older maximum O. The source's smaller cap proves the
exact death `d_f(M)=ε²m>−1`, while the proposed saddle S has height −1.
Thus S is rejected and the global open window `(−1,0)` contains exactly
T: `N=1` and `(N)₂=0`.

To put this on the present side-24 torus at arbitrarily small prescribed
r, set on the embedded square

```text
F_r(x,y) = b+kr³ f_ε(x/r,y/r).
```

Extend its separable one-dimensional pieces around the two side-24
circles by C39 §4's monotone-arc construction, with the same critical
values scaled by `kr³` and shifted by b in their sum. The complementary
arc lengths are positive for small r. Short endpoint collars and
sign-preserving derivative interpolations allow each prescribed finite
height difference without additional critical points; the scaled bump
extends by zero. The complete ten-point list and strict Morse types
are preserved. Hence

```text
d_Fr(M)=b+kr³ε²m > b−kr³,        N_r=1,        (N_r)₂=0. (15)
```

The sole additional window point is `T_r=(−r/2,rεt_−)`, at distance O(r)
from the midpoint. Every cutoff `a_r/r→∞` eventually contains it. The
corresponding inner second-factorial count is zero as well.

This counterexample also lies in the support of the **original six-pin
Gaussian law**, for each fixed small r,b,k. Here is the support argument;
it does not condition on C39's preserved infinite jets. Positive Fourier
coefficients give positive density for every finite trigonometric
polynomial's coefficient vector. The independent Fourier tail has
arbitrarily small expected C² norm after a sufficiently large cutoff,
so Markov's inequality gives a positive probability of a prescribed
small tail norm. Trigonometric polynomials are dense in C². These facts
prove full C² support of the unconditioned field.

Let L be the six-observation map, `Σ=Cov(Lf)`, and
`P₀=I−Cov(f,Lf)Σ^(−1)L`. This is a continuous projection onto `ker L`.
The pushforward of a full-support law by this projection has support
`ker L`; translating by the regression mean shows that Q has full
support on its affine space of exact pins. The constructed F_r belongs
to this space. A sufficiently small relative C² neighborhood preserves
all ten nondegenerate critical points, the strict nonpin height gaps to
the window endpoints, and the endpoint types. A fixed path to an older
point has minimum strictly greater than s, which persists in a smaller
C⁰ neighborhood. On a still smaller neighborhood W is bounded below
by a positive constant. Therefore, for each fixed such pinned law,

```text
Q^W{e=0 and N_r=1} > 0.                              (16)
```

No lower rate in r follows: the neighborhood, determinant bound and
probability can all shrink with r. Equation (16) does not contradict a
compact-mark `O(r³)` upper bound on selector failure. The example fails
the cap's whole-square fourth-derivative condition, as C39 computes.
Nor does it refute the separate regional factorial-moment theorems.

If the actual death lies strictly between s and b, its saddle does imply
`N_r≥1`; it does not force a second window point. If death is at or below
s, this reasoning gives no window-count implication. A collision-based
alternative proof must define a different topological witness, impose
an additional good event, or account for an explicit exceptional event
and prove the needed inclusion. Global or regional factorial bounds
for this raw count cannot replace that argument.

## 6. What this completes, and the remaining interface

This text supplies the full scalar cap construction, pinned genericity,
a countable global elder mark, the whole-field Borel Kac–Rice extension,
and the exact selected-pair measure. It makes the topological and
measurability steps readable in one place. The Gaussian Kac–Rice theorems
remain explicit standard inputs; the quantitative normalizer, failure
integral, all-mark integration, remainder and coefficient remain the
source-bound arguments of the other appendices.

The next analytic task must consume the **actual** event in (1), or
prove a new event comparison with it. Equations (15)–(16) prevent an
unjustified shortcut through the raw second factorial count. They do
not close the regional shrinking-collision node, evaluate its rate,
or change the accepted scope of the existing lifetime chain. No claim
of 100% mathematical completion, human review or formalization of this
analytic argument follows from this companion.
